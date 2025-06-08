---
layout: post
title: "Red API Reference: Complete Developer Documentation"
date: 2024-01-08 10:00:00 +0000
categories: [reference, api]
tags: [api, documentation, reference, developers]
author: Red Team
---

# Red API Reference: Complete Developer Documentation

This comprehensive API reference provides detailed documentation for all Red components, classes, and methods. Use this as your go-to resource for building federated learning applications with Red.

## 🏗️ Core Components

### FederatedBase

The abstract base class for all federation implementations.

#### Constructor

```python
class FederatedBase(object):
    def __init__(
        self,
        nodes: List[VirtualNode],
        topology: Union[str, np.ndarray],
        config: ConfigValidator,
        resources: Union[str, PlacementGroup] = "uniform",
        is_tune: bool = False,
        bundle_offset: int = 0,
    )
```

**Parameters:**
- `nodes`: List of VirtualNode objects representing federation participants
- `topology`: Network topology ("star", "ring", "mesh", or custom adjacency matrix)
- `config`: Configuration validator instance
- `resources`: Resource allocation strategy or custom placement group
- `is_tune`: Whether federation is used within Ray Tune experiment
- `bundle_offset`: Bundle offset for multiple federations in same placement group

#### Methods

##### `train(blocking: bool = False, **train_args)`

Abstract method for training the federation.

**Parameters:**
- `blocking`: Whether to block until training completes
- `**train_args`: Training arguments passed to nodes

**Returns:** Training results (implementation-specific)

**Raises:** `NotImplementedError` (must be implemented by subclasses)

##### `test(phase: Literal["train", "eval", "test"], **kwargs) -> List`

Abstract method for testing the federation.

**Parameters:**
- `phase`: Testing phase identifier
- `**kwargs`: Testing arguments

**Returns:** List of test results

##### `pull_version(node_ids: Union[str, List[str]], timeout: Optional[float] = None)`

Pull model versions from specified nodes.

**Parameters:**
- `node_ids`: Node ID(s) to pull versions from
- `timeout`: Optional timeout for the operation

**Returns:** Model versions (single version if one node, list if multiple)

##### `send(header: str, body: Dict, to: Optional[Union[str, List[str]]] = None)`

Send message to federation nodes.

**Parameters:**
- `header`: Message header/type
- `body`: Message content dictionary
- `to`: Target node ID(s) (None for all nodes)

##### `stop() -> None`

Stop the federation and cleanup resources.

#### Properties

##### `running: bool`
Returns whether the federation is currently running a training process.

##### `num_nodes: int`
Returns the number of nodes in the federation.

##### `node_ids: List[str]`
Returns list of node IDs in the federation.

##### `resources: Dict[str, Dict[str, Union[int, float]]]`
Returns resource allocation information for the federation.

---

### VirtualNode

Wrapper class for lazy node initialization.

#### Constructor

```python
class VirtualNode(object):
    def __init__(
        self,
        template: Type[object],
        id: str,
        role: str,
        config: ConfigValidator,
        log: Log,
    ) -> None
```

**Parameters:**
- `template`: Node class template (e.g., FederatedNode subclass)
- `id`: Unique node identifier
- `role`: Node role ("train", "eval", "test", or combinations)
- `config`: Configuration validator instance
- `log`: Logging instance

#### Methods

##### `build(bundle_idx: int, placement_group: PlacementGroup)`

Create the actual Ray actor for this node.

**Parameters:**
- `bundle_idx`: Index of resource bundle in placement group
- `placement_group`: Ray placement group for resource allocation

#### Properties

##### `built: bool`
Returns whether the node has been built (Ray actor created).

---

### FederatedNode

Base class for individual federation participants.

#### Constructor

```python
class FederatedNode(object):
    def __init__(
        self, 
        node_id: str, 
        role: str, 
        config: ConfigValidator, 
        log: Log, 
        **kwargs
    )
```

**Parameters:**
- `node_id`: Unique identifier for the node
- `role`: Node role(s) (dash-separated for multiple roles)
- `config`: Configuration validator instance
- `log`: Logging instance
- `**kwargs`: Additional build arguments

#### Methods

##### `build(**kwargs)`

Abstract method for node initialization. Must be implemented by subclasses.

**Parameters:**
- `**kwargs`: Build-specific arguments

##### `train(**train_args) -> Dict`

Abstract training method. Must be implemented by subclasses.

**Parameters:**
- `**train_args`: Training arguments

**Returns:** Dictionary of training metrics

##### `test(phase: Literal["train", "eval", "test"], **kwargs)`

Abstract testing method. Must be implemented by subclasses.

**Parameters:**
- `phase`: Testing phase
- `**kwargs`: Testing arguments

##### `send(header: str, body: Dict, to: Optional[Union[str, List[str]]] = None)`

Send message to other nodes.

**Parameters:**
- `header`: Message header
- `body`: Message content
- `to`: Target node(s) (None for neighbors)

##### `receive(block: bool = False, timeout: Optional[float] = None) -> Optional[Message]`

Receive message from other nodes.

**Parameters:**
- `block`: Whether to block until message arrives
- `timeout`: Timeout for blocking reception

**Returns:** Message object or None

##### `update_version(**kwargs)`

Update node's model version.

**Parameters:**
- `**kwargs`: Model parameters and metadata

##### `stop()`

Gracefully stop the node.

##### `enqueue(msg: ray.ObjectRef)`

Add message to node's processing queue.

**Parameters:**
- `msg`: Ray object reference to message

**Returns:** `True` on success

#### Properties

##### `id: str`
Node identifier.

##### `version: int`
Current model version.

##### `is_train_node: bool`
Whether node has training role.

##### `is_eval_node: bool`
Whether node has evaluation role.

##### `is_test_node: bool`
Whether node has testing role.

##### `neighbors: List[str]`
List of neighboring nodes in topology (cached property).

---

### TopologyManager

Ray remote actor managing communication topology.

#### Constructor

```python
@ray.remote(num_cpus=TOPOLOGY_MANAGER_CPU_RESOURCES, max_concurrency=100)
class TopologyManager:
    def __init__(self, federation_id: str) -> None
```

**Parameters:**
- `federation_id`: Unique federation identifier

#### Methods

##### `publish(msg: Message, ids: Optional[Union[str, List[str]]] = None)`

Publish message to specified nodes.

**Parameters:**
- `msg`: Message object to publish
- `ids`: Target node IDs (None for sender's neighbors)

**Returns:** List of delivery confirmations

##### `get_neighbors(node_id: str)`

Get neighboring nodes for given node ID.

**Parameters:**
- `node_id`: Node to get neighbors for

**Returns:** List of neighbor node IDs

##### `link_nodes(node_ids: List[str], topology: Union[str, np.ndarray])`

Setup communication topology between nodes.

**Parameters:**
- `node_ids`: List of node identifiers
- `topology`: Topology specification

**Raises:** `ValueError` if less than 2 nodes provided

---

## 🌐 Federation Schemas

### StarFederatedLearningSchema

Centralized federation with server-client architecture.

#### Constructor

```python
class StarFederatedLearningSchema(FederatedBase):
    def __init__(
        self,
        server_template: Type[FederatedNode],
        client_template: Type[FederatedNode],
        roles: List[str],
        config: ConfigValidator,
        log: Log,
        server_id: str = "server",
        resources: Union[str, PlacementGroup] = "uniform",
        is_tune: bool = False,
        bundle_offset: int = 0,
    ) -> None
```

**Parameters:**
- `server_template`: Server node class
- `client_template`: Client node class
- `roles`: List of client roles
- `config`: Configuration validator
- `log`: Logging instance
- `server_id`: Server node identifier
- `resources`: Resource allocation
- `is_tune`: Ray Tune integration flag
- `bundle_offset`: Bundle offset for placement

#### Methods

##### `train(server_args: Dict, client_args: Dict, blocking: bool = False) -> None`

Train the star federation.

**Parameters:**
- `server_args`: Arguments for server training
- `client_args`: Arguments for client training
- `blocking`: Whether to block until completion

### RingFederatedLearningSchema

Decentralized federation with ring topology.

#### Constructor

```python
class RingFederatedLearningSchema(FederatedBase):
    def __init__(
        self,
        client_template: Type[FederatedNode],
        roles: List[str],
        config: ConfigValidator,
        log: Log,
        resources: Union[str, PlacementGroup] = "uniform",
        is_tune: bool = False,
        bundle_offset: int = 0,
    ) -> None
```

#### Methods

##### `train(client_args: Dict, blocking: bool = False) -> None`

Train the ring federation.

**Parameters:**
- `client_args`: Arguments for client training
- `blocking`: Whether to block until completion

---

## 🔧 Utility Components

### Message

Communication message structure.

#### Constructor

```python
class Message:
    def __init__(self, header: str, sender_id: str, body: Dict)
```

**Parameters:**
- `header`: Message type/header
- `sender_id`: ID of sending node
- `body`: Message content

### Queue

Asynchronous message queue.

#### Methods

##### `put(item, index: Optional[int] = None)`

Add item to queue.

**Parameters:**
- `item`: Item to add
- `index`: Optional position (0 for front)

##### `get(block: bool = True, timeout: Optional[float] = None)`

Get item from queue.

**Parameters:**
- `block`: Whether to block until item available
- `timeout`: Timeout for blocking get

**Returns:** Queue item

**Raises:** `Queue.Empty` if no item and not blocking

### ConfigValidator

Configuration validation and management.

#### Constructor

```python
class ConfigValidator(**config_dict)
```

**Parameters:**
- `**config_dict`: Configuration key-value pairs

#### Properties

All configuration values are accessible as properties:
- `FEDERATION_ID`: Federation identifier
- `NUMBER_OF_CLIENTS`: Number of clients
- `MODEL_TYPE`: Model architecture type
- `DATASET_TYPE`: Dataset specification
- `LEARNING_RATE`: Learning rate
- `FEDERATED_LEARNING_SCHEMA`: Federation type
- `FEDERATED_LEARNING_TOPOLOGY`: Network topology
- And many more...

---

## 🎛️ Resource Management

### get_resources_split

Create placement group for federation.

```python
def get_resources_split(
    num_nodes: int,
    num_cpus: int = None,
    num_gpus: int = None,
    split_strategy: Literal["random", "uniform"] = "uniform",
    placement_strategy: Literal["STRICT_PACK", "PACK", "STRICT_SPREAD", "SPREAD"] = "PACK",
    is_tune: bool = False,
    log: logging = logging
) -> Union[PlacementGroup, PlacementGroupFactory]
```

**Parameters:**
- `num_nodes`: Number of nodes in federation
- `num_cpus`: CPU allocation (None for auto)
- `num_gpus`: GPU allocation (None for auto)
- `split_strategy`: Resource distribution strategy
- `placement_strategy`: Node placement strategy
- `is_tune`: Ray Tune integration
- `log`: Logging instance

**Returns:** Placement group or factory for Ray Tune

---

## 🏃‍♂️ Execution Functions

### star_federated_learning_executor

Execute star topology federated learning.

```python
def star_federated_learning_executor(config: ConfigValidator, log: Log)
```

**Parameters:**
- `config`: Validated configuration
- `log`: Logging instance

### ring_federated_learning_executor

Execute ring topology federated learning.

```python
def ring_federated_learning_executor(config: ConfigValidator, log: Log)
```

**Parameters:**
- `config`: Validated configuration
- `log`: Logging instance

---

## 🔧 Decorators

### @remote

Enhanced Ray remote decorator.

```python
def remote(*args, **kwargs)
```

**Usage:**
```python
@remote
class MyFederatedNode(FederatedNode):
    # Automatically gets max_concurrency=100
    pass

@remote(max_concurrency=200, num_gpus=1)
class HighThroughputNode(FederatedNode):
    # Custom configuration
    pass
```

**Benefits:**
- Automatic max_concurrency setting
- Validation for federated learning requirements
- Consistent actor configuration

---

## 📊 Example Usage

### Basic Federation Setup

```python
from validators.config_validator import ConfigValidator
from utils.yaml_loader import load_objectified_yaml
from utils.log import Log
from schemas.star_federated_learning.star_federated_learning_executor import star_federated_learning_executor

# Load configuration
config_dict = load_objectified_yaml("config.yaml")
config = ConfigValidator(**config_dict)

# Initialize logging
log = Log("experiment", config.MODEL_TYPE, config.DISTANCE_METRIC)

# Run federation
star_federated_learning_executor(config, log)
```

### Custom Node Implementation

```python
from core.federated import FederatedNode
import torch

class CustomClientNode(FederatedNode):
    def build(self, **kwargs):
        self.model = self.create_model()
        self.data_loader = self.get_data()
        
    def train(self, optimizer_fn, loss_fn, **kwargs):
        # Custom training logic
        optimizer = optimizer_fn(self.model.parameters())
        criterion = loss_fn()
        
        for data, target in self.data_loader:
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Update version
        self.update_version(weights=self.model.state_dict())
        
        return {"loss": loss.item()}
```

### Custom Federation Schema

```python
from core.federated import FederatedBase
from core.federated.virtual_node import VirtualNode

class CustomFederation(FederatedBase):
    def __init__(self, node_template, num_nodes, config, log):
        nodes = [
            VirtualNode(node_template, f"node_{i}", "train", config, log)
            for i in range(num_nodes)
        ]
        super().__init__(nodes=nodes, topology="mesh", config=config)
        
    def train(self, **kwargs):
        # Custom training coordination
        pass
```

---

## 🔗 Integration APIs

### Ray Integration

Red seamlessly integrates with Ray ecosystem:

```python
import ray

# Ray cluster information
ray.cluster_resources()  # Available resources
ray.available_resources()  # Current availability
ray.nodes()  # Cluster nodes

# Actor management
ray.get_actor("federation_id/node_id")  # Get node actor
ray.kill(actor)  # Terminate actor
```

### Monitoring Integration

```python
# Federation monitoring
federation.running  # Check if running
federation.num_nodes  # Number of nodes
federation.pull_version(["client_1", "client_2"])  # Get versions

# Resource monitoring
from utils.resources import monitor_cluster_resources
resources = monitor_cluster_resources()
```

---

This API reference provides the foundation for building sophisticated federated learning applications with Red. For more detailed examples and tutorials, see the [Getting Started Guide]({% post_url 2024-01-05-getting-started %}) and [Architecture Overview]({% post_url 2024-01-07-architecture-overview %}). 