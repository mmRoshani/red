import copy
import time
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import ray

from core.communication.queue import Queue
from utils.log import Log
from core.communication.message import Message
from core.communication.topology_manager import TopologyManager
from utils.exceptions import EndProcessException
from validators.config_validator import ConfigValidator


class FederatedNode(object):

    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', log=Log,  **kwargs):
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
        # RvQ; How is multi role possible

        self.config = config
        self.log = log
        # Node hyperparameters
        self._fed_id: str = self.config.FEDERATION_ID
        self._id: str = node_id
        self._role: str = role

        # Communication interface
        self._tp_manager: TopologyManager = None
        self._message_queue: Queue = None
        # RvQ: different Queue algorithms?

        # Node's version
        self._version: int = 0
        self._version_buffer: Queue = None
        self._node_metrics: Dict[str, Any] = {}
        # RvQ: What are node metrics

        # Buildup function
        self._node_config = kwargs
        self.build(**kwargs)
        # RvQ: what's a buildup function

    def build(self, **kwargs):
        """_summary_"""
        pass

    def _setup_train(self):
        """_summary_"""
        if self._tp_manager is None:
            self._tp_manager = ray.get_actor(f"{self._fed_id}/broker")
        self._message_queue = Queue()
        self._version = 0
        self._version_buffer = Queue()
        return True
        # RvQ: Why we need a broker?

    def _train(self, **train_args):
        try:
            self.train(**train_args)
        except EndProcessException:
            print(f"Node {self.id} is exiting.")

        return self._node_metrics

    def train(self, **train_args) -> Dict:
        """_summary_
        Raises:
            NotImplementedError: _description_
        Returns:
            Dict: _description_
        """
        raise NotImplementedError

    def test(self, phase: Literal["train", "eval", "test"], **kwargs):
        """_summary_
        Args:
            phase (Literal[&#39;train&#39;, &#39;eval&#39;, &#39;test&#39;]):
                _description_
        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
        """_summary_
        Args:
            header (str): _description_
            body (Dict): _description_
            to (Optional[Union[str, List[str]]], optional): _description_. Defaults to None.
        """
        if isinstance(to, str):
            to = [to]
            # RvQ: Like TF???

        msg = Message(header=header, sender_id=self._id, body=body)
        ray.get([self._tp_manager.publish.remote(msg, to)])

    def receive(self, block:bool = False, timeout: Optional[float] = None) -> Message | None:
        """_summary_
        Args:
            block:   bool: blocking until the required message be available.
            timeout: (Optional[float], optional): _description_. Defaults to None.
        Raises:
            EndProcessException: _description_
        Returns:
            Message: _description_
        """
        try:
            msg = self._message_queue.get(block=block, timeout=timeout)
        except Queue.Empty:
            msg = None

        if msg is not None and msg.header == "STOP":
            raise EndProcessException
        return msg

    def update_version(self, **kwargs):
        """_summary_
        Args:
            **kwargs: _description_
        Raises:
            NotImplementedError: _description_
        """
        to_save = {k: copy.deepcopy(v) for k, v in kwargs.items()}
        # RvQ: why deepcopy and why and what to update the version of?
        version_dict = {
            "id": self.id,
            "n_version": self.version,
            "timestamp": time.time(),
            "model": to_save,
        }
        self._version_buffer.put(version_dict)
        self._version += 1

    def stop(self):
        """_summary_"""
        self._message_queue.put(Message("STOP"), index=0)

    def enqueue(self, msg: ray.ObjectRef):
        """_summary_
        Args:
            msg (ray.ObjectRef): _description_
        Returns:
            _type_: _description_
        """
        self._message_queue.put(msg)
        return True

    def _invalidate_neighbors(self):
        """_summary_"""
        del self.neighbors

    def _pull_version(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self._version_buffer.get(block=True)

    @property
    def id(self) -> str:
        """_summary_
        Returns:
            str: _description_
        """
        return self._id

    @property
    def version(self) -> int:
        """_summary_
        Returns:
            int: _description_
        """
        return self._version

    @property
    def is_train_node(self) -> bool:
        """_summary_
        Returns:
            bool: _description_
        """
        return "train" in self._role.split("-")

    @property
    def is_eval_node(self) -> bool:
        """_summary_
        Returns:
            bool: _description_
        """
        return "eval" in self._role.split("-")

    @property
    def is_test_node(self) -> bool:
        """_summary_
        Returns:
            bool: _description_
        """
        return "test" in self._role.split("-")

    @cached_property
    def neighbors(self) -> List[str]:
        """_summary_
        Returns:
            List[str]: _description_
        """
        return ray.get(self._tp_manager.get_neighbors.remote(self.id))

