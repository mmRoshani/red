from constants.framework import MODEL_UPDATE, SERVER_ID, MESSAGE_BODY_STATES
from core.communication.message import Message
from core.federated import FederatedNode
from decorators.remote import remote
from nets.network_factory import network_factory
import copy
from typing import Dict
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
class RingFederatedLearning(FederatedNode):
    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', log: Log) -> None:
        super().__init__(node_id=node_id, role=role, config=config, log=log)
        self.model = None
        self.federated_learning_rounds = None
        self.optimizer: torch.optim.SGD = None
        self.criterion: torch.nn.CrossEntropyLoss = None
        self.number_of_neighbors = None
        self.neighbors_id_list = []
        self.local_epochs = None
        self.train_loader = None
        self.test_loader = None
        self.aggregator: FedAvgAggregator = None
        self.log = config.RUNTIME_COMFIG.log
        self.device = 'cpu'
        
        # RDFL specific parameters
        self.synchronizing_interval = config.CLUSTERING_PERIOD  # K in the algorithm
        self.local_round_counter = 0
        self.global_round_counter = 0

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

    def local_training_round(self, round_num: int):
        """ Perform one round of local training """
        self.log.info(f'Client {self.id} starting local training round {round_num}')
        
        # Local training for specified epochs
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
                    self.log.info(f'Client {self.id} - Round {round_num}, Epoch {epoch + 1}/{self.local_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            avg_epoch_loss = epoch_loss / num_batches
            self.log.info(f'Client {self.id} - Round {round_num}, Epoch {epoch + 1}/{self.local_epochs} completed with avg loss: {avg_epoch_loss:.6f}')

        self.log.info(f'Client {self.id} completed local training round {round_num}')

    def send_model_through_ring_clock_wise(self):
        """ Send model parameters through the ring topology """
        message_body = {
            MESSAGE_BODY_STATES: self.model.to('cpu').state_dict(),
            'sender_id': self.id,
            'round': self.global_round_counter
        }
        
        self.log.info(f"Client {self.id} sending model parameters through ring to neighbor 
                      the right: {self.neighbors[1]}")
        self.send(header=MODEL_UPDATE, body=message_body, to=self.neighbors[1])

    def receive_and_aggregate_models(self):
        """ Receive models from ring and perform weighted aggregation """
        self.log.info(f'Client {self.id} starting model aggregation phase')
        
        # Setup aggregator for trusted nodes (all neighbors for now - malicious detection can be added later)
        trusted_nodes = self.neighbors[0].copy()
        self.aggregator.setup(trusted_nodes)
        
        received_models = {}
        received_count = 0
        expected_count = len(trusted_nodes)
        
        self.log.info(f'Client {self.id} waiting for {expected_count} models from trusted nodes: {trusted_nodes}')
        
        # Collect models from all trusted nodes in the ring
        while received_count < expected_count:
            neighbor_message = self.receive(block=True, timeout=60.0)
            
            if neighbor_message is None:
                self.log.warn(f'Client {self.id} timed out waiting for neighbor models')
                continue
                
            if neighbor_message.header == MODEL_UPDATE:
                sender_id = neighbor_message.sender_id
                if sender_id in trusted_nodes and sender_id not in received_models:
                    self.log.info(f'Client {self.id} received model from trusted node {sender_id}')
                    received_models[sender_id] = neighbor_message.body[MESSAGE_BODY_STATES]
                    self.aggregator.update(neighbor_message)
                    received_count += 1
                    self.log.info(f'Client {self.id} processed model from {sender_id} ({received_count}/{expected_count})')
                else:
                    self.log.warn(f'Client {self.id} received duplicate or untrusted model from {sender_id}')
            else:
                self.log.warn(f'Client {self.id} received unexpected message with header {neighbor_message.header}')

        # Perform weighted aggregation (equal weights for now - can be modified for trust scores)
        self.log.info(f'Client {self.id} performing weighted aggregation of {len(received_models)} models')
        aggregated_state = self.aggregator.compute()
        
        # Update local model with aggregated parameters (Step 9 in Algorithm 1)
        self.model.load_state_dict(aggregated_state)
        self.log.info(f'Client {self.id} updated local model with aggregated parameters')

    def run(self):
        """ Main RDFL execution """
        self.log.info(f"Client {self.id} initializing ring topology")
        self.log.info(f"Topology manager: {self._tp_manager}")
        self.neighbors_id_list = self.neighbors
        self.log.info(f"Client {self.id} neighbors: {self.neighbors_id_list}")
        self.log.info(f"Synchronizing interval K = {self.synchronizing_interval}")

        # Main federated learning loop
        for t in range(1, self.federated_learning_rounds + 1):
            self.global_round_counter = t
            self.log.info(f'=' * 80)
            self.log.info(f'Client {self.id} - GLOBAL ROUND {t}/{self.federated_learning_rounds}')
            self.log.info(f'=' * 80)
            
            # Local training round
            self.local_training_round(t)
            
            # Check if synchronization is needed (t mod K = 0)
            if t % self.synchronizing_interval == 0:
                self.log.info(f'Client {self.id} - SYNCHRONIZATION triggered at round {t} (t mod K = {t % self.synchronizing_interval})')
                
                # Step 5: Malicious node detection (placeholder for now)
                self.log.info(f'Client {self.id} - Malicious node detection (currently skipped)')
                
                # Add small delay based on client ID to prevent message collision
                client_num = get_last_char_as_int(self.id)
                delay = client_num * 0.2  # 0.2 second stagger
                self.log.info(f'Client {self.id} waiting {delay}s before sending (collision prevention)')
                time.sleep(delay)
                
                # Step 6: Send model through ring
                self.send_model_through_ring()
                
                # Steps 7-9: Receive and aggregate models
                self.receive_and_aggregate_models()
                
                self.log.info(f'Client {self.id} - SYNCHRONIZATION completed for round {t}')
            else:
                self.log.info(f'Client {self.id} - No synchronization needed for round {t} (next sync at round {((t // self.synchronizing_interval) + 1) * self.synchronizing_interval})')
                
        self.log.info(f'Client {self.id} completed all RDFL rounds!')

    def train(self, optimizer_fn, loss_fn):
        """Initialize and start RDFL training"""
        self.log.info(f"Node {self.id} initializing RDFL training")
        
        # Initialize components
        self.build()
        
        # Setup optimizer and loss function
        self.optimizer = optimizer_fn(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = loss_fn()
        
        self.log.info(f"Node {self.id} starting RDFL execution")
        
        # Start RDFL algorithm
        self.run()
        
        self.log.info(f"Node {self.id} completed RDFL training")

    def _send_model_to_neighbors(self, neighbor_sample: List[str]):
        """Send current global model to selected neighbors with necessary parameters"""
        message_body = {
            "state": self.model.to('cpu').state_dict(),
        }
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {self.id} is sending local model to the selected neighbors, {neighbor_sample}")
        self.send(header=MODEL_UPDATE, body=message_body, to=neighbor_sample)

    def sample_neighbors(self, neighbor_sampling_rate: float, random_seed: int = 42) -> List[str]:
        random.seed(random_seed)

        """Sample participating neighbors for the current round"""
        num_neighbors = self.number_of_neighbors
        num_to_sample = int(neighbor_sampling_rate * num_neighbors)

        if num_to_sample < 0 or num_to_sample > num_neighbors:
            raise ValueError("Invalid number of neighbors to sample")

        return random.sample(self.neighbors_id_list, num_to_sample)
