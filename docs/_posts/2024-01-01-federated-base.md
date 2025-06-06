---
layout: post
title: "FederatedBase: The Foundation of Red's Architecture"
date: 2024-01-01 10:00:00 +0000
categories: [architecture, core-components]
tags: [federated-base, ray, distributed-computing]
author: Red Team
---

# FederatedBase: The Foundation of Red's Architecture

The `FederatedBase` class serves as the cornerstone of the Red Federated Learning Framework, providing the foundational infrastructure for all federated learning implementations. This abstract base class orchestrates the complex interplay between distributed nodes, resource management, and communication topologies.

## 🏗️ Core Architecture

### Class Overview

```python
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
```

The `FederatedBase` class manages:
- **Virtual Node Collection**: A list of `VirtualNode` objects representing federation participants
- **Network Topology**: Defines how nodes communicate (star, ring, mesh, or custom)
- **Resource Management**: Ray placement groups for optimal resource allocation
- **Federation Lifecycle**: Training, testing, and monitoring operations

## 🚀 Ray Integration

### Resource Management

```python
if not is_tune:
    if isinstance(resources, str):
        self._pg = get_resources_split(
            num_nodes=len(self._nodes), 
            split_strategy=resources, 
            num_gpus=1,
        )
    else:
        self._pg = resources
else:
    self._pg = ray.util.get_current_placement_group()
```

The class leverages Ray's **Placement Groups** to:
- **Optimize Resource Allocation**: Distribute CPU/GPU resources across nodes
- **Ensure Co-location**: Keep related processes on the same machines
- **Handle Heterogeneity**: Support different resource requirements per node

### Remote Execution Management

```python
def pull_version(
    self, node_ids: Union[str, List[str]], 
    timeout: Optional[float] = None
) -> Union[List, Dict]:
    to_pull = [
        node.handle._pull_version.remote()
        for node in self._nodes
        if node.id in to_pull
    ]
    
    if timeout is None:
        new_versions = ray.get(to_pull)
        return new_versions[0] if len(to_pull) == 1 else new_versions
```

## 🌐 Communication Architecture

### Topology Manager Integration

The `FederatedBase` class integrates with the `TopologyManager` to handle:

- **Message Routing**: Efficient message passing between nodes
- **Network Topology**: Implementation of different communication patterns
- **Fault Tolerance**: Handling node failures and network partitions

### Message Passing

```python
def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
    if isinstance(to, str):
        to = [to]

    msg = Message(header=header, sender_id=self._name, body=body)
    ray.get([self._tp_manager.publish.remote(msg, to)])
```

## 🔄 Federation Lifecycle

### Training Process

The abstract `train` method must be implemented by concrete classes:

```python
def train(self, blocking: bool = False, **train_args):
    """
    Trains the models in the federation.
    
    This method is responsible for dispatching the arguments of the training
    algorithm to the nodes. It then starts the training algorithm on the nodes,
    and returns the results of the training.
    """
    raise NotImplementedError
```

### Testing and Evaluation

```python
def test(self, phase: Literal["train", "eval", "test"], **kwargs) -> List:
    """
    Tests the models in the federation.
    
    This method is responsible for dispatching the arguments of the testing
    algorithm to the nodes. It then starts the testing algorithm on the nodes,
    and returns the results of the testing.
    """
    raise NotImplementedError
```

## 🎯 Key Features

### 1. **Lazy Node Initialization**
- Nodes are created as `VirtualNode` objects
- Actual Ray actors are spawned only when needed
- Efficient resource utilization

### 2. **Flexible Resource Management**
- Support for uniform and random resource distribution
- Integration with Ray Tune for hyperparameter optimization
- Custom placement group support

### 3. **State Management**
- Federation state tracking (`IDLE`, `RUNNING`)
- Thread-safe operations
- Graceful shutdown procedures

### 4. **Version Control**
- Model version tracking across nodes
- Asynchronous version pulling
- Timeout support for robustness

## 🔧 Implementation Examples

### Star Topology Federation

```python
from core.federated import FederatedBase
from core.federated.virtual_node import VirtualNode

# Create virtual nodes
nodes = [
    VirtualNode(ServerNode, "server", "train", config, log),
    VirtualNode(ClientNode, "client_1", "train", config, log),
    VirtualNode(ClientNode, "client_2", "train", config, log),
]

# Create federation with star topology
federation = FederatedBase(
    nodes=nodes,
    topology="star",
    config=config,
    resources="uniform"
)
```

## 📊 Performance Optimizations

### Resource Efficiency
- **Bundle Offset**: Enables multiple federations in the same placement group
- **CPU/GPU Allocation**: Intelligent resource distribution based on node roles
- **Memory Management**: Efficient handling of large model parameters

### Network Optimization
- **Message Batching**: Reduces communication overhead
- **Asynchronous Operations**: Non-blocking message passing
- **Fault Recovery**: Automatic handling of node failures

## 🔍 Advanced Features

### Ray Tune Integration
When `is_tune=True`, the federation integrates seamlessly with Ray Tune for:
- **Hyperparameter Optimization**: Automated parameter search
- **Resource Sharing**: Efficient resource utilization across trials
- **Distributed Evaluation**: Parallel evaluation of different configurations

### Custom Topologies
Support for user-defined adjacency matrices:
```python
# Custom topology as numpy array
custom_topology = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

federation = FederatedBase(
    nodes=nodes,
    topology=custom_topology,
    config=config
)
```

## 🛡️ Production Considerations

### Fault Tolerance
- **Node Failure Handling**: Automatic detection and recovery
- **Network Partitions**: Graceful degradation of service
- **Resource Exhaustion**: Intelligent resource reallocation

### Monitoring and Observability
- **Metrics Collection**: Performance and health monitoring
- **Logging Integration**: Comprehensive logging throughout the lifecycle
- **State Visualization**: Real-time federation status

## 🔗 Related Components

- **[Virtual Nodes]({% post_url 2024-01-02-virtual-nodes %})**: Node wrapper implementation
- **[Topology Manager]({% post_url 2024-01-03-topology-manager %})**: Communication infrastructure
- **[Ray Integration]({% post_url 2024-01-06-ray-integration-deep-dive %})**: Deep dive into Ray usage

---

The `FederatedBase` class represents the sophisticated foundation that makes Red a production-grade federated learning framework, seamlessly integrating Ray's distributed computing capabilities with advanced federated learning concepts. 