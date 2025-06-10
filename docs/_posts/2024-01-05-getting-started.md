---
layout: post
title: "Getting Started with Red Federated Learning"
date: 2024-01-05 10:00:00 +0000
categories: [tutorial, getting-started]
tags: [installation, quickstart, ray, federated-learning]
author: Red Team
---

# Getting Started with Red Federated Learning

This guide will help you set up and run your first federated learning experiment with Red, leveraging Ray's distributed computing capabilities for production-grade federated learning.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Ray 2.0+**
- **PyTorch 2.0+**
- **CUDA** (optional, for GPU support)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/red.git
cd red
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify Ray Installation**
```bash
python -c "import ray; ray.init(); print('Ray initialized successfully')"
```

## 🎯 Your First Federation

### Basic Configuration

Create a `config.yaml` file:

```yaml
# Basic Red Configuration
device: 'cpu'
federation_id: 'quickstart-federation'

# Federated Learning Schema
federated_learning_schema: 'StarFederatedLearning'
federated_learning_topology: 'star'

# Node Configuration
client_role: 'train'
number_of_clients: 4

# Model and Dataset
model_type: "cnn"
dataset_type: "fmnist"
learning_rate: 0.001

# Training Parameters
number_of_epochs: 1
train_batch_size: 32
test_batch_size: 32
federated_learning_rounds: 5

# Ray Configuration
# Ray will be automatically initialized
```

### Simple Star Topology Example

```python
# quickstart.py
import ray
from src.schemas.star_federated_learning.star_federated_learning_executor import star_federated_learning_executor
from src.validators.config_validator import ConfigValidator
from src.utils.yaml_loader import load_objectified_yaml
from src.utils.log import Log
import torch


def main():
    # Load configuration
    config_dict = load_objectified_yaml("./config.yaml")
    config = ConfigValidator(**config_dict)

    # Initialize logging
    log = Log("quickstart", config.MODEL_TYPE, config.DISTANCE_METRIC)
    log.info("Starting Red Federated Learning Quickstart")

    # Run federated learning
    star_federated_learning_executor(config, log)

    log.info("Federated learning completed successfully!")


if __name__ == "__main__":
    main()
```

### Run Your First Federation

```bash
python quickstart.py
```

## 🏗️ Understanding the Output

When you run Red, you'll see:

1. **Ray Initialization**
```
2024-01-05 10:00:00,000 INFO -- Ray initialized successfully
```

2. **Federation Setup**
```
----------    framework   setup   --------------------------------------------------
----------    datasets    distribution   --------------------------------------------------
----------    runtime configurations  --------------------------------------------------
----------    Schema  Factory --------------------------------------------------
```

3. **Training Progress**
```
Federation training started with 4 clients
Round 1/5: Training in progress...
Round 1/5: Model aggregation completed
Round 2/5: Training in progress...
```

## 🔧 Configuration Deep Dive

### Federation Schemas

Red supports multiple federated learning approaches:

#### Star Topology (Centralized)
```yaml
federated_learning_schema: 'StarFederatedLearning'
federated_learning_topology: 'star'
```
- **Use Case**: Traditional FedAvg algorithms
- **Communication**: All clients communicate through central server
- **Scalability**: Good for moderate number of clients

#### Ring Topology (Decentralized)
```yaml
federated_learning_schema: 'DecentralizedFederatedLearning'
federated_learning_topology: 'ring'
```
- **Use Case**: Peer-to-peer federated learning
- **Communication**: Circular message passing
- **Scalability**: Excellent for large number of clients

#### Mesh Topology (Fully Connected)
```yaml
federated_learning_schema: 'DecentralizedFederatedLearning'
federated_learning_topology: 'mesh'
```
- **Use Case**: Research and experimental setups
- **Communication**: All-to-all connectivity
- **Scalability**: Limited by network bandwidth

### Ray Configuration

Red automatically manages Ray initialization, but you can customize:

```python
# Advanced Ray setup
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "4"

# Custom Ray initialization
ray.init(
    num_cpus=8,
    num_gpus=2,
    dashboard_host="0.0.0.0",
    dashboard_port=8265
)
```

## 📊 Supported Datasets and Models

### Datasets
- **FMNIST**: Fashion-MNIST for image classification
- **CIFAR-10**: Color image classification
- **Custom**: Bring your own datasets

### Models
- **CNN**: Convolutional Neural Networks
- **ResNet**: Deep residual networks
- **Custom**: Define your own PyTorch models

## 🔧 Custom Implementation

### Creating Custom Nodes

```python
from src.core.federated import FederatedNode
import torch
import torch.nn as nn


class MyCustomClient(FederatedNode):
    def build(self, **kwargs):
        """Initialize the client with model and data"""
        self.model = self.create_model()
        self.train_loader = self.get_data_loader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def create_model(self):
        """Define your model architecture"""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def train(self, optimizer_fn, loss_fn, **kwargs):
        """Implement your training logic"""
        self.model.train()
        optimizer = optimizer_fn(self.model.parameters(), lr=0.01)
        criterion = loss_fn()

        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Update version with new model state
        self.update_version(weights=self.model.state_dict())

        return {
            "loss": total_loss / len(self.train_loader),
            "samples": len(self.train_loader.dataset)
        }
```

### Custom Federation Schema

```python
from src.core.federated import FederatedBase
from src.core.federated import VirtualNode
import ray
import threading


class MyCustomFederation(FederatedBase):
    def __init__(self, client_template, num_clients, config, log, **kwargs):
        # Create virtual nodes
        nodes = [
            VirtualNode(client_template, f"client_{i}", "train", config, log)
            for i in range(num_clients)
        ]

        super().__init__(
            nodes=nodes,
            topology="star",  # or your custom topology
            config=config,
            **kwargs
        )

    def train(self, client_args, blocking=False):
        """Implement your federated learning algorithm"""
        # Build and setup nodes
        for i, node in enumerate(self._nodes):
            if not node.built:
                node.build(i + 1, self._pg)

        # Initialize topology manager
        if self._tp_manager is None:
            self._tp_manager = self.create_topology_manager()

        # Setup communication
        ray.get([node.handle._setup_train.remote() for node in self._nodes])

        # Start training
        self._runtime_remotes = [
            node.handle._train.remote(**client_args)
            for node in self._nodes
        ]

        if blocking:
            results = ray.get(self._runtime_remotes)
            return results
```

## 🔍 Monitoring and Debugging

### Ray Dashboard

Access the Ray dashboard at `http://localhost:8265` to monitor:
- **Tasks**: Ray task execution
- **Actors**: Node status and resource usage
- **Cluster**: Overall cluster health

### Logging

Red provides comprehensive logging:

```python
from src.utils.log import Log

log = Log("my-experiment", "cnn", "coordinate")
log.info("Starting experiment")
log.error("Error occurred", extra={"details": error_details})
```

### Performance Monitoring

```python
# Check federation status
print(f"Federation running: {federation.running}")
print(f"Number of nodes: {federation.num_nodes}")
print(f"Node IDs: {federation.node_ids}")

# Pull model versions
versions = federation.pull_version(["client_1", "client_2"])
print(f"Current versions: {versions}")
```

## 🛠️ Advanced Configuration

### Resource Management

```yaml
# Custom resource allocation
resources:
  cpu_per_node: 2
  gpu_per_node: 0.5
  memory_per_node: 2048  # MB

# Ray placement strategy
placement_strategy: "PACK"  # or "SPREAD", "STRICT_PACK", "STRICT_SPREAD"
```

### Data Distribution

```yaml
# Control data heterogeneity
data_distribution_kind: "30"  # 30% label skew
dirichlet_beta: 0.5  # Controls non-IID distribution
```

### Aggregation Strategies

```yaml
# FedAvg or FedProx
aggregation_strategy: "FedAvg"
aggregation_sample_scaling: true  # Weight by number of samples
```

## 🔗 Next Steps

1. **[Architecture Overview]({% post_url 2024-01-07-architecture-overview %})**: Deep dive into Red's architecture
2. **[Ray Integration]({% post_url 2024-01-06-ray-integration-deep-dive %})**: Advanced Ray usage patterns
3. **[API Reference]({% post_url 2024-01-08-api-reference %})**: Complete API documentation

## 🐛 Troubleshooting

### Common Issues

**Ray Connection Error**
```bash
# Check Ray status
ray status

# Restart Ray
ray stop
ray start --head
```

**Memory Issues**
```yaml
# Reduce batch size
train_batch_size: 16
test_batch_size: 16

# Reduce number of clients
number_of_clients: 2
```

**GPU Not Detected**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

---

You're now ready to build production-grade federated learning systems with Red! Start with the simple examples and gradually explore the advanced features to meet your specific requirements. 