import networkx as nx
import random

def generate_k_connected_graph_from_nodes(node_ids: list[str], k: int):
    """Generates a k-connected graph using given string node IDs"""
    n = len(node_ids)
    assert n > k, "Number of nodes must be greater than k for a k-connected graph"
    k -= 1 # The number of edges for each node
    d = k + 1  # degree for random regular graph

    # Ensure n * d is even
    if (n * d) % 2 != 0:
        raise ValueError("Number of nodes times (k+1) must be even. Please adjust the list of node IDs.")

    while True:
        try:
            # Generate a random d-regular graph on integer nodes 0..n-1
            G_int = nx.random_regular_graph(d, n)
            if nx.node_connectivity(G_int) >= k:
                break
        except nx.NetworkXError:
            continue

    # Create a new graph with string node IDs
    G = nx.relabel_nodes(G_int, {i: node_ids[i] for i in range(n)})
    return G