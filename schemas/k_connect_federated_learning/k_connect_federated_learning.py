from constants.framework import MODEL_UPDATE, SERVER_ID, MESSAGE_BODY_STATES
from core.communication.message import Message
from core.federated import FederatedNode
from decorators.remote import remote
from nets.network_factory import network_factory
import copy
from typing import Dict, List
import torch
from torch.nn import Module as NN
from utils.checker import device_checker
from utils.get_last_char_as_int import get_last_char_as_int
from validators.config_validator import ConfigValidator
from utils.log import Log

from core.aggregator.fed_avg_aggregator_base import FedAvgAggregator
from utils.client_ids_list import client_ids_list_generator
from typing import List
import random
import time


@remote(num_gpus=1, num_cpus=1)
class KConnectFederatedLearning(FederatedNode):
    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', log: Log) -> None:
        super().__init__(node_id=node_id, role=role, config=config, log=log)
        self.model = None
        self.federated_learning_rounds = None
        self.optimizer: torch.optim.SGD = None
        self.criterion: torch.nn.CrossEntropyLoss = None
        self.neighbors_models = {}  # Store received neighbor models
        self.local_epochs = None
        self.train_loader = None
        self.test_loader = None
        self.aggregator: FedAvgAggregator = None
        self.log = config.RUNTIME_COMFIG.log
        self.device = 'cpu'
        
        # DFL specific parameters
        self.local_round_counter = 0
        self.global_round_counter = 0
        self.local_data_size = 0  # d_i in the algorithm

    def build(self):
        """Initialize peer components"""
        self.device = device_checker(self.config.DEVICE)
        self.federated_learning_rounds = self.config.FEDERATED_LEARNING_ROUNDS
        self.local_epochs = self.config.NUMBER_OF_EPOCHS
        self.model = network_factory(
            model_type=self.config.MODEL_TYPE,
            number_of_classes=self.config.NUMBER_OF_CLASSES,
            pretrained=self.config.PRETRAINED_MODELS
        ).to(self.device)

        self.aggregator = FedAvgAggregator(config=self.config, log=self.log)

        # Initialize local dataset
        self.train_loader = self.config.RUNTIME_COMFIG.train_loaders[get_last_char_as_int(self.id)]
        self.test_loader = self.config.RUNTIME_COMFIG.test_loaders[get_last_char_as_int(self.id)]
        
        # Calculate local data size (d_i in the algorithm)
        self.local_data_size = len(self.train_loader.dataset)
        
        # Step 2: Each client i selects its own neighbors as a list of N
        # (This will be populated by the topology manager through self.neighbors)
        
        self.log.info(f'Client {self.id} initialized with local data size: {self.local_data_size}')

    def select_neighbors(self):
        if hasattr(self, 'neighbors') and self.neighbors:
            self.log.info(f'Client {self.id} has {len(self.neighbors)} neighbors: {self.neighbors}')
        else:
            self.log.warn(f'Client {self.id} has no neighbors assigned yet')

    def evaluate_model(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / num_batches
        
        self.model.train()  # Set back to training mode
        return accuracy, avg_loss

    def calculate_train_accuracy(self):
        train_accuracy, train_loss = self.evaluate_model(self.train_loader)
        self.log.info(f'Client {self.id}, Round {self.local_round_counter + 1} - Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.6f}')
        return train_accuracy, train_loss

    def calculate_test_accuracy(self):
        test_accuracy, test_loss = self.evaluate_model(self.test_loader)
        self.log.info(f'Client {self.id}, Round {self.local_round_counter + 1} - Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.6f}')
        return test_accuracy, test_loss

    def local_training_round(self):
        self.log.info(f'Client {self.id} starting local training round {self.local_round_counter + 1}')
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    self.log.info(f'Client {self.id}, Round {self.local_round_counter + 1}, Epoch {epoch + 1}/{self.local_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_epoch_loss = epoch_loss / num_batches
            self.log.info(f'Client {self.id}, Round {self.local_round_counter + 1}, Epoch {epoch + 1}/{self.local_epochs} completed with avg loss: {avg_epoch_loss:.6f}')

        self.log.info(f'Client {self.id} completed local training round {self.local_round_counter + 1}')

    def send_model_to_neighbors(self):
        message_body = {
            MESSAGE_BODY_STATES: self.model.to('cpu').state_dict(),  # Move to CPU for transmission
            'sender_id': self.id,
            'round': self.local_round_counter,
            'data_size': self.local_data_size
        }
        
        self.log.info(f"Client {self.id} sending model to {len(self.neighbors)} neighbors: {self.neighbors}")
        
        for neighbor_id in self.neighbors:
            self.send(header=MODEL_UPDATE, body=message_body, to=neighbor_id)

    def receive_models_from_neighbors(self):
        self.neighbors_models = {}
        received_count = 0
        expected_neighbors = len(self.neighbors)
        
        self.log.info(f'Client {self.id} waiting to receive models from {expected_neighbors} neighbors')
        
        while received_count < expected_neighbors:
            message = self.receive(block=True, timeout=300.0)
            
            if message is None:
                self.log.warn(f'Client {self.id} timed out waiting for neighbor models')
                continue
                
            if message.header == MODEL_UPDATE:
                sender_id = message.body.get('sender_id')
                received_state = message.body[MESSAGE_BODY_STATES]
                sender_data_size = message.body.get('data_size', 1)  # Default to 1 if not provided
                
                if sender_id in self.neighbors:
                    self.neighbors_models[sender_id] = {
                        'state_dict': received_state,
                        'data_size': sender_data_size
                    }
                    received_count += 1
                    self.log.info(f'Client {self.id} received model from neighbor {sender_id} ({received_count}/{expected_neighbors})')
                else:
                    self.log.warn(f'Client {self.id} received model from non-neighbor {sender_id}')
            else:
                self.log.warn(f'Client {self.id} received unexpected message: {message.header}')

    def aggregate_models(self):
        local_state = self.model.state_dict()
        aggregated_state = {}
        
        # Calculate denominator: d_i + sum(d_j for j in N)
        total_data_size = self.local_data_size + sum(
            neighbor_info['data_size'] for neighbor_info in self.neighbors_models.values()
        )
        
        self.log.info(f'Client {self.id} aggregating with local data size {self.local_data_size} and total data size {total_data_size}')
        
        # Initialize aggregated state with weighted local model
        for key in local_state.keys():
            # Start with weighted local model: d_i * w_i^r
            aggregated_state[key] = (self.local_data_size * local_state[key]).float()
            
            # Add weighted neighbor models: sum(d_j * w_j^r for j in N)
            for neighbor_id, neighbor_info in self.neighbors_models.items():
                neighbor_state = neighbor_info['state_dict']
                neighbor_data_size = neighbor_info['data_size']
                aggregated_state[key] += (neighbor_data_size * neighbor_state[key]).float()
            
            # Divide by total data size
            aggregated_state[key] = aggregated_state[key] / total_data_size
        
        # Load aggregated model
        self.model.load_state_dict(aggregated_state)
        self.model = self.model.to(self.device)  # Move back to device
        
        self.log.info(f'Client {self.id} completed model aggregation for round {self.local_round_counter + 1}')

    def run_dfl_rounds(self):
        # Ensure neighbors are properly selected
        self.select_neighbors()
        
        # Calculate initial accuracy before training
        self.log.info(f"================= Client {self.id} - Initial Evaluation =================")
        initial_train_acc, initial_train_loss = self.calculate_train_accuracy()
        initial_test_acc, initial_test_loss = self.calculate_test_accuracy()
        
        for round_num in range(self.federated_learning_rounds):
            self.local_round_counter = round_num
            
            self.log.info(f"================= Client {self.id} - DFL Round {round_num + 1}/{self.federated_learning_rounds} =================")
            
            # Step 4: Train local model w_i^r on local data
            self.local_training_round()
            
            # Calculate accuracy after local training
            train_accuracy, train_loss = self.calculate_train_accuracy()
            test_accuracy, test_loss = self.calculate_test_accuracy()
            
            # Step 6: Send w_i^r to each neighbor j in N
            self.send_model_to_neighbors()
            
            # Step 7: Receive w_j^r from each neighbor j
            self.receive_models_from_neighbors()
            
            # Step 9: Aggregate models using DFL formula
            self.aggregate_models()
            
            # Calculate accuracy after aggregation
            post_agg_train_acc, post_agg_train_loss = self.calculate_train_accuracy()
            post_agg_test_acc, post_agg_test_loss = self.calculate_test_accuracy()
            
            self.log.info(f"Client {self.id} Round {round_num + 1} Summary:")
            self.log.info(f"  After Local Training  - Train Acc: {train_accuracy:.2f}%, Train Loss: {train_loss:.6f}, Test Acc: {test_accuracy:.2f}%, Test Loss: {test_loss:.6f}")
            self.log.info(f"  After Aggregation     - Train Acc: {post_agg_train_acc:.2f}%, Train Loss: {post_agg_train_loss:.6f}, Test Acc: {post_agg_test_acc:.2f}%, Test Loss: {post_agg_test_loss:.6f}")
            
            self.log.info(f"Client {self.id} completed DFL round {round_num + 1}")

    def run(self):
        """Main execution method"""
        self.log.info(f"Client {self.id} starting DFL with {len(self.neighbors)} neighbors: {self.neighbors}")
        self.run_dfl_rounds()
        self.log.info(f"Client {self.id} completed all DFL rounds")

    def test(self, phase: str = "test"):
        if phase == "test":
            test_accuracy, test_loss = self.calculate_test_accuracy()
            return test_accuracy, len(self.test_loader.dataset)
        elif phase == "train":
            train_accuracy, train_loss = self.calculate_train_accuracy()
            return train_accuracy, len(self.train_loader.dataset)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def train(self, optimizer_fn, loss_fn):
        # Initialize components
        self.build()
        # Setup optimizer and loss function
        self.optimizer = optimizer_fn(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = loss_fn()
        
        self.log.info(f"Client {self.id} has {len(self.neighbors)} neighbors: {self.neighbors}")
        self.run()