### FederatedNode: Abstract Base Class for Federated Learning Participants

The `FederatedNode` class serves as an abstract base for defining individual participants within a federated learning system.

It encapsulates fundamental functionalities that enable a node to interact with the federation. **This class is intended for inheritance and extension by users to create custom node implementations tailored to specific roles and algorithms.** Direct instantiation of the `FederatedNode` class is not recommended.

When subclassing the `FederatedNode` class, users are expected to implement the `build`, `train`, and `test` methods to define the node's specific behavior.

The `train` method is invoked upon the commencement of the federated learning process. It governs the internal operations of the node during training, encompassing tasks such as local model updates for client nodes and aggregation procedures for server nodes, contingent upon the federated learning algorithm employed.

The `test` method is called to execute an evaluation round across the federation.

Inter-node communication is facilitated exclusively within the `train` method through the utilization of the `send` and `receive` methods. The `send` method enables the broadcasting of messages to neighboring nodes as defined by the network topology, or the direct transmission of messages to a specific node. Conversely, the `receive` method allows a node to retrieve or await incoming messages from its message queue. 
Notably, the `receive` method offers **optional** blocking behavior, thereby enabling the implementation of asynchronous node operations.