from src.core.client import Client
from typing import List


def update_clusters_based_on_indexes(clients: List[Client], cluster_indices: List[List[int]]) -> List[list[Client]]:
    client_clusters = []
    for cluster in cluster_indices:
        new_orientation = []
        for index in cluster:
            new_orientation.append(clients[index])
        client_clusters.append(new_orientation)
    return client_clusters