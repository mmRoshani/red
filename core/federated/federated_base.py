from core.communication.topology_manager import TopologyManager
from core.federated.virtual_node import VirtualNode
from utils.checker import device_checker
from utils.log import Log
import numpy as np
import ray
from ray.util.placement_group import PlacementGroup
import threading
from typing import Dict, List, Literal, Optional, Union
from core.communication.message import Message
from utils.resources import get_resources_split
from validators.config_validator import ConfigValidator


# def __init__(
#         self,
#         model,
#         config: 'ConfigValidator', log: 'Log'):
#     self.log = log
#     self.config = config
#     self.device = device_checker(self.config.DEVICE)
#     self.model = model.to(self.device)
#
#
# def evaluate(self):
#     _loss, _accuracy = model_evaluation(self.model, self.eval_loader)
#
#     if _loss < 1.0 and _accuracy > 0.6:
#         self.log.info(
#             f"testing done for client no {self.id} with accuracy of {_accuracy} and loss of {_loss} [GOOD]"
#         )
#     elif _loss < 2.0 and _accuracy > 0.4:
#         self.log.warn(
#             f"testing done for client no {self.id} with accuracy of {_accuracy} and loss of {_loss} [MODERATE]"
#         )
#     else:
#         self.log.warn(
#             f"testing done for client no {self.id} with accuracy of {_accuracy} and loss of {_loss} [POOR]"
#         )
#
#     return _accuracy



class FederatedBase(object):
    def __init__(
            self,
            nodes: List[VirtualNode],
            topology: Union[str, np.ndarray],
            config: 'ConfigValidator',
            resources: Union[str, PlacementGroup] = "uniform",
            is_tune: bool = False,
            bundle_offset: int = 0,
    ):
        """Creates a new Federation object.

        Args:
            nodes (List[VirtualNode]): A list of VirtualNode objects that are
                part of the federation.
            topology (Union[str, np.ndarray]): The topology of the network. Can
                be either a string, or a numpy array. If a string, it must be one
                of the following: "star". If a numpy array, it must be a square
                binary matrix of shape (N, N), where N is the number of nodes. The
                matrix must be symmetric, and the diagonal must be all zeros.
            resources (Union[str, PlacementGroup], optional): The resources to use.
                If a string, it must be one of the following: "uniform", "random".
                If a PlacementGroup, it must be a placement group that has been
                created with the same number of bundles as the number of nodes in
                the federation. If parameter `is_tune` is set to True, this argument
                is ignored. Defaults to "uniform".
            is_tune (bool, optional): Whether the federation has to be instantiated
                within a Tune experiment. Defaults to False.
            bundle_offset (int, optional): The bundle offset. This parameter is useful
                whenever multiple federations need to be allocated within the same
                PlacementGroup. Defaults to 0.
        # RvQ: What is a placementGroup in the context of resources
        # RvQ: TF is a bundle offset?

        Raises:
            ValueError: If the topology is not a valid topology.
            ValueError: If the resources are not a valid resource specification.
        """
        self._fed_id = config.FEDERATION_ID
        self._name = "supervisor"
        self._nodes: List[VirtualNode] = nodes
        self._topology: Union[str, np.ndarray] = topology
        self.config: 'ConfigValidator' = config
        # RvQ: TF is a federation id, and what's a supervisor

        if not is_tune:
            if isinstance(resources, str):
                self._pg = get_resources_split(
                    num_nodes=len(self._nodes), split_strategy=resources, num_gpus=1,
                )
            else:
                self._pg = resources
        else:
            self._pg = ray.util.get_current_placement_group()
        self._bundle_offset = 1 + bundle_offset if is_tune else bundle_offset

        self._tp_manager: TopologyManager = None
        self._state: Literal["IDLE", "RUNNING"] = "IDLE"
        self._runtime_remotes: List[ray.ObjectRef] = None
        self._runtime: threading.Thread = None

    def __getitem__(self, node_id: str):
        """Returns the handle of the node with the given ID."""
        for node in self._nodes:
            if node.id == node_id:
                return node.handle
        raise ValueError(f"Identifier {node_id} not found in process.")

    def train(self, blocking: bool = False, **train_args):
        """
        Trains the models in the federation.

        This method is responsible for dispatching the arguments of the training
        algorithm to the nodes. It then starts the training algorithm on the nodes,
        and returns the results of the training.
        """
        raise NotImplementedError

    def test(self, phase: Literal["train", "eval", "test"], **kwargs) -> List:
        """
        Tests the models in the federation.

        This method is responsible for dispatching the arguments of the testing
        algorithm to the nodes. It then starts the testing algorithm on the nodes,
        and returns the results of the testing.
        """
        raise NotImplementedError

    def pull_version(
            self, node_ids: Union[str, List[str]], timeout: Optional[float] = None
    ) -> Union[List, Dict]:
        """
        Pulls the version of the nodes with the given IDs.

        Args:
            node_ids (Union[str, List[str]]): The IDs of the nodes to pull the
                version from.
            timeout (Optional[float], optional): The timeout for the pull. If None,
                the pull is blocking. Defaults to None.
        Returns:
            Union[List, Dict]: The version of the nodes with the given IDs.
        """
        to_pull = [node_ids] if isinstance(node_ids, str) else node_ids
        to_pull = [
            node.handle._pull_version.remote()
            for node in self._nodes
            if node.id in to_pull
        ]
        # RvQ: TF is to? what to pull?
        if timeout is None:
            new_versions = ray.get(to_pull)
            return new_versions[0] if len(to_pull) == 1 else new_versions
        else:
            new_versions, _ = ray.wait(to_pull, timeout=timeout)
            if len(new_versions) == 0:
                return None
            else:
                return new_versions[0] if len(to_pull) == 1 else new_versions

    def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
        """
        Sends a message to the nodes with the given IDs.

        This method is useful whenever the user wishes to interact with the nodes in the
        federation during the training process. For example, the user can send a message
        to the nodes to change the learning rate of the models.

        Args:
            header (str): The header of the message.
            body (Dict): The body of the message.
            to (Optional[Union[str, List[str]]], optional): The IDs of the nodes to
                send the message to. If None, the message is sent to all nodes.
                Defaults to None.
        """
        if isinstance(to, str):
            to = [to]

        msg = Message(header=header, sender_id=self._name, body=body)
        ray.get([self._tp_manager.publish.remote(msg, to)])

    def stop(self) -> None:
        """Stops the federation."""
        ray.get(
            [
                node.handle.stop.remote()
                for node in self._nodes
                if node.built and "train" in node.role
            ]
        )
        self._runtime.join()
        self._state = "IDLE"

    # RvQ: Who is this property
    @property
    def running(self) -> bool:
        """Returns whether the federation is running a training process"""
        return (
                self._state == "RUNNING"
                and self._runtime is not None
                and self._runtime.is_alive()
        )

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the federation."""
        return len(self._nodes)

    @property
    def node_ids(self) -> List[str]:
        """Returns the IDs of the nodes in the federation."""
        return [node.id for node in self._nodes]

    @property
    def resources(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Returns the resources of the federation."""
        res_arr = self._pg.bundle_specs
        resources = {
            "all": {"CPU": res_arr[0]["CPU"], "GPU": 0},
            "tp_manager": {"CPU": res_arr[0]["CPU"]},
        }
        for i, node in enumerate(self._nodes, start=1):
            resources[node.id] = res_arr[i]
            resources["all"]["CPU"] += res_arr[i]["CPU"]
            if "GPU" in res_arr[i]:
                resources["all"]["GPU"] += res_arr[i]["GPU"]

        return resources
