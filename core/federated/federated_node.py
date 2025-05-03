import copy
import time
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import ray

from core.communication.message import Message
from core.communication.topology_manager import TopologyManager
from utils.exceptions import EndProcessException
from utils.queue import Queue
from validators.config_validator import ConfigValidator


class FederatedNode(object):

    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', federation_id: str = ""):
        """Creates a node in the federation. Must not be called directly or overridden.

        Args:
            node_id (str): The id of the node.
            role (str): The role of the node. It must be either a single role in
                ``["train", "eval", "test"]``, or a combination of them as a dash-separated
                string. For example, "train-eval" or "train-eval-test".
            federation_id (str): The id of the federation the node belongs to.
                Defaults to "".
            **build_args: Additional arguments to be passed to the build function.
        """
        # Node hyperparameters
        self._fed_id: str = federation_id
        self._id: str = node_id
        self._role: str = role

        # Communication interface
        self._tp_manager: TopologyManager = None
        self._message_queue: Queue = None

        # Node's version
        self._version: int = 0
        self._version_buffer: Queue = None
        self._node_metrics: Dict[str, Any] = {}

        # Buildup function
        self._node_config = config
        self.build(config)

    def build(self, config: 'ConfigValidator'):
        """
        Performs the setup of the node's environment when the node is added to
        a federation.

        The build method and has a twofold purpose.

        **Define object-level attributes**. This encloses attributes that are independent
        of whether the node is executing the training method or the test method (e.g.,
        choosing the optimizer, the loss function, etc.).

        **Perform all the resource-intensive operations in advance to avoid bottlenecks**.
        An example can be downloading the data from an external source, or instantiating
        a model with computationally-intensive techniques.

        Since it is called within the ``__init__`` method, the user can define additional
        class attributes.

        An example of build function can be the following:

        .. code-block:: python

            def build(self, dataset_name: str):
                self._dataset_name = dataset_name
                self._dataset = load_dataset(self._dataset_name)
        """
        pass

    def _setup_train(self):
        """Prepares the node's environment for the training process."""
        if self._tp_manager is None:
            self._tp_manager = ray.get_actor(
                "/".join([self._fed_id, "topology_manager"])
            )
        self._message_queue = Queue()
        self._version = 0
        self._version_buffer = Queue()
        return True

    def _train(self, **train_args):
        """Wrapper for the training function"""
        try:
            self.train(**train_args)
        except EndProcessException:
            print(f"Node {self.id} is exiting.")

        return self._node_metrics

    def train(self, **train_args) -> Dict:
        """Implements the core logic of a node within a training process. It is
        called by the federated when the training process starts.

        An example can be the client in the Federated Averaging algorithm:

        .. code-block:: python

            def train(self, **train_args):
                while True:
                    # Get the model
                    model = self.receive().body["model"]

                    # Get the data
                    data_fn = self.get_data()

                    # Train the model

                    model.train(self.dataset, self.optimizer, self.loss, self.metrics)

                    # Send the model to the server
                    self.send("model", model)
        """
        raise NotImplementedError

    def test(
        self, phase: Literal["train", "eval", "test"], **kwargs
    ) -> Tuple[float, int]:
        """Implements the core logic of a node within a test process. It is
        called by the federated when the test session starts.

        Args:
            phase (Literal["train", "eval", "test"]): The phase of the test process.
                It can be either "train", "eval" or "test".
            **kwargs: Additional arguments to be passed to the test function.

        Returns:
            Tuple(float, int): A tuple containing the average loss and the number of
                samples used for the test.
        """
        raise NotImplementedError

    def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
        """Sends a message to a specific node or to the neighbor nodes in the federated network.

        Args:
            header (str): The header of the message.
            body (Dict): The body of the message.
            to (Optional[Union[str, List[str]]], optional): The id of the node to which
                the message is sent. If None, the message is sent to the neighbor nodes.
                Defaults to None.
        """
        if isinstance(to, str):
            to = [to]

        msg = Message(header=header, sender_id=self._id, body=body)
        ray.get([self._tp_manager.forward.remote(msg, to)])

    def receive(self, timeout: Optional[float] = None) -> Message:
        """Receives a message from the message queue. If the timeout value is defined,
        it waits for a message for the specified amount of time. If no message is
        received within the timeout, it returns None. This allows to implement a node
        with an asynchronous behavior.

        Args:
            timeout (Optional[float], optional): The timeout value. Defaults to None.

        Returns:
            Message: The received message.

        Raises:
            EndProcessException: If the message received is a "STOP" message, it raises
                an EndProcessException to stop the process. This is handled under the
                hood by the training function.
        """
        try:
            msg = self._message_queue.get(timeout=timeout)
        except Queue.Empty:
            msg = None

        if msg is not None and msg.header == "STOP":
            raise EndProcessException
        return msg

    def update_version(self, **kwargs):
        """Updates the node's version. Whenever this function is called, the version is
        stored in an internal queue. The version is pulled from the queue whenever the
        federation calls the ``pull_version`` method.
        """
        to_save = {k: copy.deepcopy(v) for k, v in kwargs.items()}
        version_dict = {
            "id": self.id,
            "n_version": self.version,
            "timestamp": time.time(),
            "model": to_save,
        }
        self._version_buffer.put(version_dict)
        self._version += 1

    def stop(self):
        """Stops the node's processes."""
        self._message_queue.put(Message("STOP"), index=0)

    def enqueue(self, msg: ray.ObjectRef):
        """Enqueues a message in the node's message queue. This method is called by the
        topology manager when a message is sent from a neighbor.

        Args:
            msg (ray.ObjectRef): The message to be enqueued.

        Returns:
            bool: True, a dummy value for the federated.
        """
        self._message_queue.put(msg)
        return True

    def _invalidate_neighbors(self):
        """
        Invalidates the node's neighbors. This method is called by the topology manager
        when the topology changes. In future versions, this will be used to implement
        dynamic topologies.
        """
        # TODO: implement
        del self.neighbors

    def _pull_version(self):
        """
        Pulls the version from the version buffer. This method is called under the
        hood by the federated when the `pull_version` method is called.
        """
        return self._version_buffer.get(block=True)

    @property
    def id(self) -> str:
        """Returns the node's id."""
        return self._id

    @property
    def version(self) -> int:
        """Returns the node's current version."""
        return self._version

    @property
    def is_train_node(self) -> bool:
        """True if the node is a training node, False otherwise."""
        return "train" in self._role.split("-")

    @property
    def is_eval_node(self) -> bool:
        """True if the node is an evaluation node, False otherwise."""
        return "eval" in self._role.split("-")

    @property
    def is_test_node(self) -> bool:
        """True if the node is a test node, False otherwise."""
        return "test" in self._role.split("-")

    @cached_property
    def neighbors(self) -> List[str]:
        """Returns the list of the node's neighbor IDs."""
        return ray.get(self._tp_manager.get_neighbors.remote(self.id))
