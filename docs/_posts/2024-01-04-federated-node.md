---
layout: post
title: "FederatedNode: Individual Participants in Red Federations"
date: 2024-01-04 10:00:00 +0000
categories: [architecture, core-components]
tags: [federated-node, ray, distributed-learning, actors]
author: Red Team
---

# FederatedNode: Individual Participants in Red Federations

The `FederatedNode` class represents individual participants in Red's federated learning ecosystem. Each node operates as an independent Ray actor, capable of training, evaluating, and communicating with other nodes while maintaining its own state and computational resources.

## 🏗️ Architecture Overview

### Core Design

```python
class FederatedNode(object):
    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', log=Log, **kwargs):
        # Node hyperparameters
        self._fed_id: str = self.config.FEDERATION_ID
        self._id: str = node_id
        self._role: str = role
        
        # Communication interface
        self._tp_manager: TopologyManager = None
        self._message_queue: Queue = None
        
        # Node's version and metrics
        self._version: int = 0
        self._version_buffer: Queue = None
        self._node_metrics: Dict[str, Any] = {}
```

### Multi-Role Architecture

Nodes support flexible role definitions:
- **Single Role**: `"train"`, `"eval"`, `"test"`
- **Multi-Role**: `"train-eval"`, `"train-eval-test"`
- **Custom Roles**: Application-specific role combinations

## 🚀 Ray Integration

### Remote Actor Execution

Each `FederatedNode` runs as a Ray remote actor, providing:

```python
@ray.remote
class ClientNode(FederatedNode):
    def __init__(self, node_id, role, config, log, **kwargs):
        super().__init__(node_id, role, config, log, **kwargs)
```

#### Key Benefits:
- **Isolation**: Each node runs in its own process
- **Fault Tolerance**: Node failures don't affect others
- **Resource Management**: Dedicated CPU/GPU allocation
- **Scalability**: Dynamic scaling across multiple machines

### Communication Infrastructure

```python
def _setup_train(self):
    """Setup training communication infrastructure"""
    if self._tp_manager is None:
        self._tp_manager = ray.get_actor(f"{self._fed_id}/broker")
    self._message_queue = Queue()
    self._version = 0
    self._version_buffer = Queue()
    return True
```

#### Communication Features:
- **Broker Discovery**: Automatic connection to TopologyManager
- **Message Queuing**: Asynchronous message handling
- **Version Control**: Model version tracking and synchronization

## 📡 Message Passing System

### Sending Messages

```python
def send(self, header: str, body: Dict, to: Optional[Union[str, List[str]]] = None):
    """Send message to other nodes"""
    if isinstance(to, str):
        to = [to]
        
    msg = Message(header=header, sender_id=self._id, body=body)
    ray.get([self._tp_manager.publish.remote(msg, to)])
```

### Receiving Messages

```python
def receive(self, block: bool = False, timeout: Optional[float] = None) -> Message | None:
    """Receive messages from other nodes"""
    try:
        msg = self._message_queue.get(block=block, timeout=timeout)
    except Queue.Empty:
        msg = None

    if msg is not None and msg.header == "STOP":
        raise EndProcessException
    return msg
```

#### Advanced Message Handling:
- **Non-Blocking Reception**: Configurable blocking behavior
- **Timeout Support**: Prevents indefinite waiting
- **Control Messages**: Special handling for system messages
- **Exception-Based Flow Control**: Clean process termination

## 🔄 Training Lifecycle

### Training Process

```python
def _train(self, **train_args):
    """Main training entry point"""
    try:
        self.train(**train_args)
    except EndProcessException:
        print(f"Node {self.id} is exiting.")
    
    return self._node_metrics

def train(self, **train_args) -> Dict:
    """Abstract training method - must be implemented by subclasses"""
    raise NotImplementedError
```

### Version Management

```python
def update_version(self, **kwargs):
    """Update node's model version"""
    to_save = {k: copy.deepcopy(v) for k, v in kwargs.items()}
    
    version_dict = {
        "id": self.id,
        "n_version": self.version,
        "timestamp": time.time(),
        "model": to_save,
    }
    self._version_buffer.put(version_dict)
    self._version += 1
```

#### Version Control Features:
- **Deep Copy Protection**: Prevents accidental model mutations
- **Timestamp Tracking**: Records when versions were created
- **Metadata Storage**: Comprehensive version information
- **Buffered Access**: Efficient version retrieval

## 🎯 Role-Based Functionality

### Role Detection

```python
@property
def is_train_node(self) -> bool:
    return "train" in self._role.split("-")

@property
def is_eval_node(self) -> bool:
    return "eval" in self._role.split("-")

@property
def is_test_node(self) -> bool:
    return "test" in self._role.split("-")
```

### Multi-Role Implementation Example

```python
class MultiRoleClientNode(FederatedNode):
    def train(self, **train_args):
        if self.is_train_node:
            # Perform training
            model_updates = self.local_training()
            self.update_version(weights=model_updates)
            
    def evaluate(self, **eval_args):
        if self.is_eval_node:
            # Perform evaluation
            accuracy = self.local_evaluation()
            return accuracy
            
    def test(self, phase, **test_args):
        if self.is_test_node:
            # Perform testing
            results = self.local_testing(phase)
            return results
```

## 🔧 Advanced Features

### 1. **Neighbor Discovery**

```python
@cached_property
def neighbors(self) -> List[str]:
    """Get neighboring nodes in the topology"""
    return ray.get(self._tp_manager.get_neighbors.remote(self.id))
```

#### Benefits:
- **Cached Results**: Efficient repeated access
- **Dynamic Updates**: Topology changes reflected automatically
- **Lazy Loading**: Computed only when needed

### 2. **Graceful Shutdown**

```python
def stop(self):
    """Gracefully stop the node"""
    self._message_queue.put(Message("STOP"), index=0)

def _invalidate_neighbors(self):
    """Force neighbor list refresh"""
    del self.neighbors
```

### 3. **Message Enqueueing**

```python
def enqueue(self, msg: ray.ObjectRef):
    """Add message to processing queue"""
    self._message_queue.put(msg)
    return True
```

## 🔄 Federated Learning Patterns

### 1. **Centralized Training (Star Topology)**

```python
class StarClientNode(FederatedNode):
    def train(self, optimizer_fn, loss_fn, **kwargs):
        # Local training
        local_updates = self.perform_local_training(optimizer_fn, loss_fn)
        
        # Send updates to server
        self.send("MODEL_UPDATE", {"weights": local_updates}, to="server")
        
        # Wait for global model
        global_model_msg = self.receive(block=True, timeout=30.0)
        if global_model_msg and global_model_msg.header == "GLOBAL_MODEL":
            self.update_model(global_model_msg.body["weights"])
```

### 2. **Decentralized Training (Ring Topology)**

```python
class RingClientNode(FederatedNode):
    def train(self, optimizer_fn, loss_fn, **kwargs):
        # Local training
        local_updates = self.perform_local_training(optimizer_fn, loss_fn)
        
        # Send to next neighbor in ring
        neighbors = self.neighbors
        next_neighbor = neighbors[0] if neighbors else None
        
        if next_neighbor:
            self.send("RING_UPDATE", {"weights": local_updates}, to=next_neighbor)
            
        # Receive from previous neighbor
        peer_msg = self.receive(block=True, timeout=30.0)
        if peer_msg and peer_msg.header == "RING_UPDATE":
            self.aggregate_with_peer(peer_msg.body["weights"])
```

### 3. **Mesh Training (Full Connectivity)**

```python
class MeshClientNode(FederatedNode):
    def train(self, optimizer_fn, loss_fn, **kwargs):
        # Local training
        local_updates = self.perform_local_training(optimizer_fn, loss_fn)
        
        # Broadcast to all neighbors
        self.send("MESH_UPDATE", {"weights": local_updates})
        
        # Collect updates from all neighbors
        neighbor_updates = []
        for _ in self.neighbors:
            msg = self.receive(block=True, timeout=10.0)
            if msg and msg.header == "MESH_UPDATE":
                neighbor_updates.append(msg.body["weights"])
                
        # Aggregate all updates
        aggregated = self.aggregate_all_updates(neighbor_updates)
        self.update_model(aggregated)
```

## 📊 Performance Optimizations

### 1. **Efficient Memory Management**

```python
def update_version(self, **kwargs):
    # Deep copy only what's necessary
    to_save = {k: copy.deepcopy(v) for k, v in kwargs.items()}
    
    # Compress large objects
    if self.should_compress(to_save):
        to_save = self.compress_data(to_save)
        
    version_dict = {
        "id": self.id,
        "n_version": self.version,
        "timestamp": time.time(),
        "model": to_save,
        "compressed": True if self.should_compress(to_save) else False
    }
    self._version_buffer.put(version_dict)
```

### 2. **Asynchronous Operations**

```python
async def async_train(self, **train_args):
    """Asynchronous training for better concurrency"""
    # Non-blocking local training
    training_task = asyncio.create_task(self.async_local_training())
    
    # Process messages while training
    while not training_task.done():
        msg = self.receive(block=False)
        if msg:
            await self.handle_message_async(msg)
        await asyncio.sleep(0.1)
    
    return await training_task
```

## 🛡️ Production Features

### 1. **Error Handling**

```python
def robust_train(self, **train_args):
    """Training with comprehensive error handling"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            return self.train(**train_args)
        except Exception as e:
            retry_count += 1
            self.log.error(f"Training failed (attempt {retry_count}): {e}")
            
            if retry_count >= max_retries:
                raise
                
            # Exponential backoff
            time.sleep(2 ** retry_count)
```

### 2. **Health Monitoring**

```python
def health_check(self):
    """Perform node health check"""
    checks = {
        "communication": self.test_communication(),
        "memory": self.check_memory_usage(),
        "model_state": self.validate_model_state(),
        "queue_size": len(self._message_queue)
    }
    
    return {
        "healthy": all(checks.values()),
        "details": checks,
        "timestamp": time.time()
    }
```

## 🔗 Integration Examples

### Basic Client Node

```python
from core.federated import FederatedNode
import torch

class SimpleClientNode(FederatedNode):
    def build(self, **kwargs):
        # Initialize model and data
        self.model = self.create_model()
        self.train_loader = self.get_train_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train(self, optimizer_fn, loss_fn, **kwargs):
        # Standard federated learning training
        self.model.train()
        optimizer = optimizer_fn(self.model.parameters(), lr=0.01)
        criterion = loss_fn()
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Update version with new weights
        self.update_version(weights=self.model.state_dict())
        
        return {"loss": loss.item(), "samples": len(self.train_loader)}
```

---

The `FederatedNode` class embodies the distributed, autonomous nature of Red's federated learning architecture, providing a robust foundation for building sophisticated federated learning applications with Ray's distributed computing capabilities. 