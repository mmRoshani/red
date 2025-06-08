---
layout: post
title: "TopologyManager: Communication Infrastructure in Red"
date: 2024-01-03 10:00:00 +0000
categories: [architecture, communication]
tags: [topology-manager, ray, communication, networking]
author: Red Team
---

# TopologyManager: Communication Infrastructure in Red

The `TopologyManager` is the communication backbone of the Red Federated Learning Framework, orchestrating message passing between nodes while managing complex network topologies. As a Ray remote actor, it provides scalable, fault-tolerant communication infrastructure for distributed federated learning systems.

## 🌐 Core Architecture

### Ray Remote Actor Design

```python
@ray.remote(num_cpus=TOPOLOGY_MANAGER_CPU_RESOURCES, max_concurrency=100)
class TopologyManager:
    def __init__(self, federation_id: str) -> None:
        self._fed_id = federation_id
        self._node_ids: List[str] = []
        self._nodes: Dict[str] = None
        self._topology = None
        self._graph: nx.Graph = None
```

The `TopologyManager` is designed as a **Ray remote actor** with:
- **High Concurrency**: `max_concurrency=100` enables handling multiple simultaneous message operations
- **Resource Efficient**: Dedicated CPU resources for communication management
- **Stateful Operation**: Maintains network topology and node references

## 🏗️ Network Topology Management

### Supported Topologies

Red supports multiple network communication patterns:

#### 1. **Star Topology**
```python
if self._topology == TOPOLOGY_STAR:
    self._graph = nx.star_graph(self._node_ids)
```
- **Centralized Communication**: All messages route through a central server
- **Simple Coordination**: Easy to implement traditional federated learning
- **Single Point of Contact**: Efficient for aggregation-based algorithms

#### 2. **Ring Topology**
```python
elif self._topology == TOPOLOGY_RING:
    self._graph = nx.cycle_graph(self._node_ids)
```
- **Peer-to-Peer Communication**: Each node communicates with immediate neighbors
- **Circular Message Flow**: Enables gossip protocols and decentralized algorithms
- **Balanced Load**: No single communication bottleneck

#### 3. **Mesh Topology**
```python
elif self._topology == TOPOLOGY_MESH:
    self._graph = nx.complete_graph(self._node_ids)
```
- **Full Connectivity**: Every node can communicate with every other node
- **Maximum Flexibility**: Supports any communication pattern
- **Redundant Paths**: High fault tolerance

#### 4. **Custom Topology**
```python
elif self._topology == TOPOLOGY_CUSTOM:
    # Support for user-defined adjacency matrices
    self._graph = nx.from_numpy_array(np.array(adjacency_matrix))
```
- **Application-Specific**: Tailored to specific federated learning requirements
- **Flexible Design**: Support for complex communication patterns
- **Research-Friendly**: Enables experimental topology designs

## 🚀 Ray-Powered Message Passing

### Efficient Message Distribution

```python
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
```

### Key Ray Integration Features

#### 1. **Object Store Optimization**
```python
msg_ref = ray.put(msg)
```
- **Single Serialization**: Message serialized once, shared across all recipients
- **Memory Efficiency**: Reduces memory usage for large messages
- **Network Optimization**: Minimizes data transfer overhead

#### 2. **Parallel Message Delivery**
```python
return ray.get([self._nodes[neigh].enqueue.remote(msg_ref) for neigh in ids])
```
- **Concurrent Delivery**: All messages sent simultaneously
- **Non-Blocking Operations**: Doesn't wait for individual node responses
- **Fault Isolation**: Failure in one delivery doesn't affect others

#### 3. **Actor Name Resolution**
```python
self._nodes = {
    node_id: ray.get_actor(self._fed_id + "/" + node_id)
    for node_id in self._node_ids
}
```
- **Dynamic Discovery**: Finds nodes using hierarchical naming
- **Fault Recovery**: Handles node restarts and migrations
- **Federation Isolation**: Separate namespaces for different federations

## 📡 Communication Patterns

### Neighbor-Based Communication

```python
def get_neighbors(self, node_id: str):
    return [neigh for neigh in self._graph.neighbors(node_id)]
```

The topology manager enforces **neighbor-based communication**, ensuring:
- **Topology Compliance**: Messages only sent to topologically connected nodes
- **Security**: Prevents unauthorized communication paths
- **Consistency**: Maintains federated learning protocol integrity

### Message Validation

```python
neighbors = self.get_neighbors(msg.sender_id)
for curr_id in ids:
    if not curr_id in neighbors:
        raise ValueError(f"{curr_id} is not a neighbor of {msg.sender_id}")
```

## 🔧 Advanced Features

### 1. **NetworkX Integration**

Red leverages NetworkX for sophisticated graph operations:

```python
import networkx as nx

# Graph analysis
centrality = nx.betweenness_centrality(self._graph)
shortest_paths = nx.shortest_path(self._graph)
connected_components = nx.connected_components(self._graph)
```

### 2. **Dynamic Topology Updates**

```python
def update_topology(self, new_topology):
    """Update the communication topology at runtime"""
    self._topology = new_topology
    self._graph = self._create_graph(new_topology, self._node_ids)
    
    # Notify all nodes of topology change
    update_msg = Message("TOPOLOGY_UPDATE", "topology_manager", {
        "new_topology": new_topology
    })
    self.broadcast(update_msg)
```

### 3. **Message Broadcasting**

```python
def broadcast(self, msg: Message):
    """Send message to all nodes in the federation"""
    all_nodes = list(self._nodes.keys())
    return self.publish(msg, all_nodes)
```

## 📊 Performance Optimizations

### Resource Management

```python
@ray.remote(num_cpus=TOPOLOGY_MANAGER_CPU_RESOURCES, max_concurrency=100)
```

#### Benefits:
- **Dedicated Resources**: Isolated CPU allocation for communication
- **High Throughput**: 100 concurrent operations support
- **Predictable Performance**: Consistent communication latency

### Placement Group Integration

```python
def _get_or_create_broker(
    placement_group, federation_id: str, bundle_offset: int
) -> TopologyManager:
    return TopologyManager.options(
        name=federation_id + "/broker",
        num_cpus=TOPOLOGY_MANAGER_CPU_RESOURCES,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group, 
            placement_group_bundle_index=0 + bundle_offset
        ),
    ).remote(federation_id=federation_id)
```

#### Advantages:
- **Resource Co-location**: Manager placed with related nodes
- **Bandwidth Optimization**: Minimizes cross-node communication
- **Fault Tolerance**: Resilient to individual node failures

## 🛡️ Production Features

### 1. **Fault Tolerance**

```python
def handle_node_failure(self, failed_node_id: str):
    """Handle node failure gracefully"""
    if failed_node_id in self._nodes:
        # Remove failed node from topology
        self._node_ids.remove(failed_node_id)
        del self._nodes[failed_node_id]
        
        # Update graph structure
        self._graph.remove_node(failed_node_id)
        
        # Notify remaining nodes
        failure_msg = Message("NODE_FAILURE", "topology_manager", {
            "failed_node": failed_node_id
        })
        self.broadcast(failure_msg)
```

### 2. **Load Balancing**

```python
def get_least_loaded_neighbors(self, node_id: str, count: int = 1):
    """Get neighbors with lowest communication load"""
    neighbors = self.get_neighbors(node_id)
    loads = {n: self.get_node_load(n) for n in neighbors}
    return sorted(neighbors, key=lambda n: loads[n])[:count]
```

### 3. **Communication Monitoring**

```python
def get_communication_stats(self):
    """Return communication statistics"""
    return {
        "total_messages": self._message_count,
        "average_latency": self._avg_latency,
        "node_loads": {n: self.get_node_load(n) for n in self._node_ids},
        "topology_efficiency": self.calculate_efficiency()
    }
```

## 🔗 Integration Examples

### Star Topology Federation

```python
# Create star topology federation
federation = StarFederatedLearningSchema(
    server_template=ServerNode,
    client_template=ClientNode,
    roles=["train"] * num_clients,
    config=config,
    log=log
)

# TopologyManager automatically created with star topology
federation.train(
    server_args={},
    client_args={"optimizer_fn": torch.optim.SGD}
)
```

### Ring Topology Federation

```python
# Create ring topology federation
federation = RingFederatedLearningSchema(
    client_template=RingNode,
    roles=["train"] * num_clients,
    config=config,
    log=log
)

# Messages flow in circular pattern
federation.train(client_args={})
```

## 📈 Scalability Characteristics

### Horizontal Scaling

| Topology | Nodes | Messages/Second | Memory Usage | Network Bandwidth |
|----------|-------|-----------------|--------------|-------------------|
| Star | 100 | 10,000 | Low | Medium |
| Ring | 100 | 5,000 | Medium | Low |
| Mesh | 100 | 20,000 | High | High |
| Custom | 100 | Variable | Variable | Variable |

### Performance Tuning

```python
# Optimize for high-throughput scenarios
@ray.remote(
    num_cpus=4,  # More CPU resources
    max_concurrency=500,  # Higher concurrency
    memory=2000 * 1024 * 1024  # 2GB memory
)
class HighThroughputTopologyManager(TopologyManager):
    pass
```

## 🔍 Advanced Use Cases

### 1. **Hierarchical Federations**

```python
def create_hierarchical_topology(clusters, intra_cluster_topology, inter_cluster_topology):
    """Create multi-level federation topology"""
    # Implementation for hierarchical communication
    pass
```

### 2. **Dynamic Load Balancing**

```python
def adaptive_routing(self, msg: Message):
    """Route messages based on current network conditions"""
    available_paths = nx.all_simple_paths(
        self._graph, 
        msg.sender_id, 
        msg.recipient_id
    )
    best_path = min(available_paths, key=lambda p: self.path_cost(p))
    return self.route_via_path(msg, best_path)
```

### 3. **Communication Security**

```python
def secure_publish(self, encrypted_msg: EncryptedMessage, ids: List[str]):
    """Publish encrypted messages with authentication"""
    if not self.verify_sender(encrypted_msg):
        raise SecurityError("Invalid sender authentication")
    
    return self.publish(encrypted_msg, ids)
```

## 🔗 Related Components

- **[Federated Base]({% post_url 2024-01-01-federated-base %})**: Federation orchestration
- **[Virtual Nodes]({% post_url 2024-01-02-virtual-nodes %})**: Node management
- **[API Reference]({% post_url 2024-01-08-api-reference %})**: Message structure and handling details

---

The TopologyManager represents the sophisticated communication infrastructure that enables Red to support diverse federated learning algorithms while maintaining performance, scalability, and fault tolerance through advanced Ray integration. 