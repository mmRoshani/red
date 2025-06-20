import threading
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import ray
from ray.util.placement_group import PlacementGroup

from constants.topology_constants import TOPOLOGY_K_CONNECT
from core.federated.federated_base import FederatedBase
from core.federated.federated_node import FederatedNode
from core.federated.virtual_node import VirtualNode
from utils.client_ids_list import client_ids_list_generator
from validators.config_validator import ConfigValidator
from utils.log import Log
from core.communication.topology_manager import _get_or_create_broker

class KConnectFederatedLearningSchema(FederatedBase):
    """
    A KConnectFederatedLearningSchema implements a k-connected federated learning
    scheme where each client is connected to k neighbors, allowing for decentralized
    communication patterns. When k = n-1 (where n is the number of clients), 
    the topology becomes fully connected.
    """

    def __init__(
        self,
        client_template: Type[FederatedNode],
        roles: List[str],
        config: 'ConfigValidator',
        log: Log,
        server_id: str = "server",
        resources: Union[str, PlacementGroup] = "uniform",
        is_tune: bool = False,
        bundle_offset: int = 0,
    ) -> None:
        """Creates a new KConnectFederatedLearningSchema object.

        Args:
            client_template (Type[FederatedNode]): The template for the client nodes.
            roles (List[str]): A list of roles for the client nodes. The length of this
                list must be equal to the number of clients.
            config (ConfigValidator): The configuration validator object.
            log (Log): The logging object.
            server_id (str, optional): The ID of the server node. Defaults to "server".
            resources (Union[str, PlacementGroup], optional): The resources to be used
                for the nodes. Defaults to "uniform".
            is_tune (bool, optional): Whether the federation is used for a Ray Tune
                experiment. Defaults to False.
            bundle_offset (int, optional): The offset to be used for the bundle IDs.
                This is useful whenever we are allocating multiple federations in the
                same PlacementGroup. Defaults to 0.

        Raises:
            ValueError: If the number of clients does not match the number of roles.
        """
        self.log = log
        self.config = config
        n_clients_or_ids: Union[int, List[str]] = self.config.NUMBER_OF_CLIENTS
        if isinstance(n_clients_or_ids, int):
            c_ids = client_ids_list_generator(n_clients_or_ids, self.log)
        else:
            c_ids = n_clients_or_ids

        nodes = []
        for c_id, role in zip(c_ids, roles):
            nodes.append(
                VirtualNode(client_template, c_id, role, self.config, self.log)
            )

        super(KConnectFederatedLearningSchema, self).__init__(
            nodes=nodes, topology=TOPOLOGY_K_CONNECT, config=self.config, resources=resources, is_tune=is_tune, bundle_offset=bundle_offset
        )

    def train(self, client_args: Dict, blocking: bool = False) -> None:
        """
        Performs a training session in the federation. Before calling the train method
        of the nodes, the method instantiates the training nodes in the federation by
        calling the .build


        Args:
            client_args (Dict): The arguments to be passed to the train function of the
                client nodes.
            blocking (bool, optional): Whether to block the current thread until the
                training session is finished. Defaults to False.
        """
        if self._tp_manager is None:
            self._tp_manager = _get_or_create_broker(
                self._pg, self._fed_id, self._bundle_offset
            )
        print(f"================================> Self._tmp_manager is: {self._tp_manager}")
        train_nodes = []
        for i, node in enumerate(self._nodes, start=1 + self._bundle_offset):
            if "train" in node.role:
                if not node.built:
                    node.build(i, self._pg)
                train_nodes.append(node)
               
        ray.get(
            self._tp_manager.link_nodes.remote(
                [node.id for node in train_nodes], self._topology, self.config.CLIENT_K_NEIGHBORS
            )
        )
        ray.get([node.handle._setup_train.remote() for node in train_nodes])

        train_args = [
            client_args[i] if isinstance(client_args, List) else client_args
            for i, _ in enumerate(train_nodes)
        ]

        self._runtime_remotes = [
            node.handle._train.remote(**train_args[i])
            for i, node in enumerate(train_nodes)
        ]
        self._runtime = threading.Thread(
            target=ray.get, args=[self._runtime_remotes], daemon=True
        )
        self._runtime.start()
        if blocking:
            self._runtime.join()

    def test(
        self, phase: Literal["train", "eval", "test"], aggregate: bool = True, **kwargs
    ) -> Union[List[float], float]:
        """
        Performs a test session in the federation.

        Args:
            phase (Literal["train", "eval", "test"]): the role of the nodes on which
                the test should be performed.
            aggregate (bool, optional): Whether to aggregate the results weighted by the
                number of samples of the local datasets. If False, the results of each
                node are returned in a list. Defaults to True.
            **kwargs: The arguments to be passed to the test function of the nodes.

        Returns:
            Union[List[float], float]: The results of the test session. If aggregate is
                True, the results are averaged.
        """
        test_nodes = []
        for i, node in enumerate(self._nodes[1:], start=2 + self._bundle_offset):
            if phase in node.role:
                test_nodes.append(node)
                if node.handle is None:
                    node.build(i, self._pg)
        remotes = [node.handle.test.remote(phase, **kwargs) for node in test_nodes]

        results = ray.get(remotes)
        if not aggregate:
            return results

        values, weights = zip(*results)
        return np.average(values, weights=weights, axis=0)

    def pull_version(
        self,
        node_ids: Union[str, List[str]] = "server",
        timeout: Optional[float] = None,
    ) -> Dict:
        """
        Pulls the latest version of a model from in a federation. The default
        behavior is to pull the version from the server node.

        Args:
            node_ids (Union[str, List[str]], optional): The ID of the node(s) from which
                to pull the version. Defaults to "server".
            timeout (Optional[float], optional): The timeout for the pull operation.
                Defaults to None.

        Returns:
            Dict: The latest version of the model.
        """
        return super().pull_version(node_ids, timeout)

    @property
    def server(self):
        """Returns the handle of the server node."""
        return self._nodes[0].handle
