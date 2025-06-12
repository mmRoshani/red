import threading
from typing import Dict, List, Literal, Optional, Type, Union
import numpy as np
import ray
from ray.util.placement_group import PlacementGroup

from src.constants.topology_constants import TOPOLOGY_CUSTOM
from src.core.federated.federated_base import FederatedBase
from src.core.federated.federated_node import FederatedNode
from src.core.federated.virtual_node import VirtualNode
from src.utils.client_ids_list import client_ids_list_generator
from src.validators.config_validator import ConfigValidator
from src.utils.log import Log
from src.core.communication.topology_manager import _get_or_create_broker

class CustomFederatedLearningSchema(FederatedBase):

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

        super(CustomFederatedLearningSchema, self).__init__(
            nodes=nodes, topology=TOPOLOGY_CUSTOM, config=self.config, resources=resources, is_tune=is_tune, bundle_offset=bundle_offset
        )

    def train(self, client_args: Dict, blocking: bool = False) -> None:
        if self._tp_manager is None:
            self._tp_manager = _get_or_create_broker(
                self._pg, self._fed_id, self._bundle_offset
            )
        train_nodes = []
        for i, node in enumerate(self._nodes, start=1 + self._bundle_offset):
            if "train" in node.role:
                if not node.built:
                    node.build(i, self._pg)
                train_nodes.append(node)
               
        ray.get(
            self._tp_manager.link_nodes_with_adjacency_matrix.remote(
                [node.id for node in train_nodes], 
                self.config.ADJACENCY_MATRIX
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

        return super().pull_version(node_ids, timeout)

    @property
    def server(self):
        return self._nodes[0].handle