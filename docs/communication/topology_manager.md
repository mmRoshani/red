# TopologyManager

The TopologyManager component is tasked with the administration of the network topology within a federated learning system.
Its functions include the establishment of the network structure and the routing of messages to designated nodes based on defined neighbor relationships.

The concrete instantiation of the network topology is realized within the build_network method, invoked by the Federation object.
This method takes as input a collection of node identifiers and a specification of the desired topology.
Subsequently, the TopologyManager constructs a networkx graph, storing both the provided node identifiers and the generated graph structure.
This graph serves to delineate the neighborhood relationships between individual nodes, while the node identifiers facilitate the retrieval of corresponding nodes from the Federation's node registry.