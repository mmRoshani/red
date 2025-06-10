"""
Diagram generation utilities for Red federated learning framework
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Union
import numpy as np
from src.constants.topology_constants import TOPOLOGY_STAR, TOPOLOGY_RING, TOPOLOGY_CUSTOM, TOPOLOGY_K_CONNECT


def create_topology_graph(node_ids: List[str], topology: Union[str, np.ndarray]) -> nx.Graph:
    """
    Create a NetworkX graph based on topology specification
    
    Args:
        node_ids: List of node identifiers
        topology: Topology type or adjacency matrix
        
    Returns:
        NetworkX graph object
    """
    if isinstance(topology, str):
        if topology == TOPOLOGY_STAR:
            graph = nx.star_graph(len(node_ids) - 1)
            mapping = {i: node_ids[i] for i in range(len(node_ids))}
            return nx.relabel_nodes(graph, mapping)
            
        elif topology == TOPOLOGY_RING:
            graph = nx.cycle_graph(len(node_ids))
            mapping = {i: node_ids[i] for i in range(len(node_ids))}
            return nx.relabel_nodes(graph, mapping)
            
        elif topology == TOPOLOGY_K_CONNECT:
            #TODO: add k topology diagram
            raise ValueError("topology diagram needs to implemented")
            # graph = nx.complete_graph(node_ids)
            # return graph
            pass
        else:
            raise ValueError(f"Unknown topology type: {topology}")
            
    elif isinstance(topology, np.ndarray):
        # Create graph from adjacency matrix
        if topology.shape[0] != topology.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if len(node_ids) != topology.shape[0]:
            raise ValueError("Number of nodes must match adjacency matrix size")
            
        graph = nx.from_numpy_array(topology)
        mapping = {i: node_ids[i] for i in range(len(node_ids))}
        return nx.relabel_nodes(graph, mapping)
    
    else:
        raise ValueError("Topology must be string or numpy array")


def save_topology_diagram(graph: nx.Graph, output_path: str = "output.dot", 
                         format: str = "svg", layout: str = "spring"):
    """
    Save topology diagram to file
    
    Args:
        graph: NetworkX graph
        output_path: Output file path
        format: Output format ('svg', 'png', 'pdf', 'dot')
        layout: Layout algorithm ('spring', 'circular', 'shell', 'random')
    """
    if not graph.nodes():
        print("Warning: Graph is empty, creating placeholder diagram")
        # Create a simple placeholder graph
        graph = nx.Graph()
        graph.add_node("No topology configured")
    
    plt.figure(figsize=(10, 8))
    
    # Choose layout algorithm
    if layout == "spring":
        pos = nx.spring_layout(graph, k=2, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "shell":
        pos = nx.shell_layout(graph)
    else:
        pos = nx.random_layout(graph)
    
    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', 
                          node_size=1500, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, edge_color='gray', width=2, alpha=0.7)
    
    plt.title("Federation Network Topology", size=16, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the diagram
    base_path = os.path.splitext(output_path)[0]
    if format.lower() == 'svg':
        plt.savefig(f"{base_path}.svg", format='svg', dpi=300, bbox_inches='tight')
    elif format.lower() == 'png':
        plt.savefig(f"{base_path}.png", format='png', dpi=300, bbox_inches='tight')
    elif format.lower() == 'pdf':
        plt.savefig(f"{base_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    
    # Also save as DOT format for Graphviz
    if format.lower() == 'dot' or format.lower() == 'svg':
        try:
            nx.drawing.nx_pydot.write_dot(graph, f"{base_path}.dot")
        except ImportError:
            print("pydot not available, skipping DOT format")
    
    plt.close()
    print(f"Topology diagram saved as {base_path}.{format}")


def generate_example_topologies():
    """Generate example topology diagrams for documentation"""
    
    # Example 1: Star topology
    node_ids = ["server", "client1", "client2", "client3", "client4"]
    star_graph = create_topology_graph(node_ids, TOPOLOGY_STAR)
    save_topology_diagram(star_graph, "docs/assets/star_topology", "svg", "spring")
    
    # Example 2: Ring topology
    node_ids = ["node1", "node2", "node3", "node4", "node5"]
    ring_graph = create_topology_graph(node_ids, TOPOLOGY_RING)
    save_topology_diagram(ring_graph, "docs/assets/ring_topology", "svg", "circular")
    
    # Example 3: Mesh topology
    node_ids = ["nodeA", "nodeB", "nodeC", "nodeD"]
    mesh_graph = create_topology_graph(node_ids, TOPOLOGY_MESH)
    save_topology_diagram(mesh_graph, "docs/assets/mesh_topology", "svg", "spring")
    
    print("Example topology diagrams generated successfully!")


if __name__ == "__main__":
    # Generate example diagrams when run directly
    os.makedirs("docs/assets", exist_ok=True)
    generate_example_topologies() 