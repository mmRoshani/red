import ray
import networkx as nx
import numpy as np

from utils.topology_generator import generate_k_connected_graph_from_nodes
from constants.framework import TOPOLOGY_MANAGER_CPU_RESOURCES
from constants.topology_constants import *
from .message import Message
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Union
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=TOPOLOGY_MANAGER_CPU_RESOURCES, max_concurrency=100)
class TopologyManager:
    def __init__(self, federation_id: str) -> None:
        self._fed_id = federation_id

        self._node_ids: List[str] = []
        self._nodes: Dict[str] = None
        self._topology = None
        self._graph: nx.Graph = None

    def publish(self, msg: Message, ids: Optional[Union[str, List[str]]] = None):
        if ids is None:
            ids = self.get_neighbors(msg.sender_id)
        else:
            neighbors = self.get_neighbors(msg.sender_id)
            for curr_id in ids:
                if not curr_id in neighbors:
                    raise ValueError(f"{curr_id} is not a neighbor of {msg.sender_id}")
        msg_ref = ray.put(msg)
        return ray.get([self._nodes[neigh].enqueue.remote(msg_ref) for neigh in ids])

    def get_neighbors(self, node_id: str):
        return [neigh for neigh in self._graph.neighbors(node_id)]

    # RvQ: Namana?
    def link_nodes(self, node_ids: List[str], topology: Union[str, np.ndarray], k: Optional[int] = None):
        if len(node_ids) < 2:
            raise ValueError("At least 2 nodes are required to setup the topology.")
        self._node_ids = node_ids
        self._nodes = {
            node_id: ray.get_actor(self._fed_id + "/" + node_id)
            for node_id in self._node_ids
        }

        self._topology = topology
        if isinstance(self._topology, str):
            if self._topology == TOPOLOGY_STAR:
                self._graph = nx.star_graph(self._node_ids)
            elif self._topology == TOPOLOGY_K_CONNECT:
                
                self._graph = generate_k_connected_graph_from_nodes(self._node_ids, k)
                # Relabel nodes to use actual node_ids instead of 0,1,2...
                mapping = {i: self._node_ids[i] for i in range(len(self._node_ids))}
                self._graph = nx.relabel_nodes(self._graph, mapping)
                
                nx.draw(self._graph, with_labels=True)
                plt.show()
            elif self._topology == TOPOLOGY_RING:
                print(f"the node_ids are {type(self._node_ids)} and the node itself is {type(self._node_ids[0])}")
                self._topology = nx.cycle_graph(self._node_ids)
                self._graph = nx.cycle_graph(self._node_ids)
                nx.draw(self._graph, with_labels=True)
                plt.show()
            elif self._topology == TOPOLOGY_CUSTOM:
                #self._graph = nx.from_numpy_array(np.array(self.adjacency_matrix))
                '''
                Some tips on how to make this more robust:
                Check for the attribute of adjacency matrix like this maybe:
                if hasattr(self, "adjacency_matrix"):
                    self._graph = nx.from_numpy_array(np.array(self.adjacency_matrix))
                else:
                    raise ValueError("Custom topology requires 'adjacency_matrix' attribute.")
                but here?
                
                Make sure adjacency_matrix is square and matches the number of nodes.

                If your nodes have IDs other than 0, 1, ..., n-1, you'll need to relabel the graph after creation
                
                       This goes here? |
                                       V
                '''
        elif isinstance(self._topology, np.ndarray):
            raise NotImplementedError


def  _get_or_create_broker(
    placement_group, federation_id: str, bundle_offset: int
) -> TopologyManager:
    return TopologyManager.options(
        name=federation_id + "/broker",
        num_cpus=TOPOLOGY_MANAGER_CPU_RESOURCES,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group, placement_group_bundle_index=0 + bundle_offset
        ),
    ).remote(federation_id=federation_id)