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
import utils.similarities.pairwise_cosine_similarity
from validators.config_validator import ConfigValidator
from utils.log import Log

from core.aggregator.fed_avg_aggregator_base import FedAvgAggregator
from utils.client_ids_list import client_ids_list_generator
from utils.similarities.pairwise_cosine_similarity import pairwise_cosine_similarity
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
        self.device = None
        
        self.synchronizing_interval = config.CLUSTERING_PERIOD  
        self.local_round_counter = 0
        self.global_round_counter = 0
        self.is_ring_initiator = False  

        distributed_hash_table = True #-------------------------#

    def build(self):
        self.device = device_checker(self.config.DEVICE)
        self.federated_learning_rounds = self.config.FEDERATED_LEARNING_ROUNDS
        self.local_epochs = self.config.NUMBER_OF_EPOCHS
        self.model = network_factory(
            model_type=self.config.MODEL_TYPE,
            number_of_classes=self.config.NUMBER_OF_CLASSES,
            pretrained=self.config.PRETRAINED_MODELS
        ).to(self.device)

        self.aggregator = FedAvgAggregator(config=self.config, log=self.log)

        self.train_loader = self.config.RUNTIME_COMFIG.train_loaders[get_last_char_as_int(self.id)]
        self.test_loader = self.config.RUNTIME_COMFIG.test_loaders[get_last_char_as_int(self.id)]
        
        self.is_ring_initiator = self.id == 'client_0'
        
        self.log.info(f'Client {self.id} initialized local model (Ring Initiator: {self.is_ring_initiator})')

    def local_training_round(self):
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
                    self.log.info(f'Client {self.id}, - Epoch {epoch + 1}/{self.local_epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            print(self.simsim())
            avg_epoch_loss = epoch_loss / num_batches
            self.log.info(f'Client {self.id}, - Epoch {epoch + 1}/{self.local_epochs} completed with avg loss: {avg_epoch_loss:.6f}')


        self.log.info(f'Client {self.id} completed local training.')

    def send_model_to_next_node(self):
        # next_node = self.neighbors[0]  # Right neighbor (clockwise)   
        next_node = self.get_next_node_in_ring()
        message_body = {
            MESSAGE_BODY_STATES: self.model.to(self.device).state_dict(),
            'sender_id': self.id,
            'ring_position': get_last_char_as_int(self.id)
        }
        
        self.log.info(f"Client {self.id} sending trained model to next node in ring: {next_node}")
        self.send(header=MODEL_UPDATE, body=message_body, to=next_node)

    def receive_and_aggregate_model(self):
        
        while True:
            message = self.receive(block=True, timeout=1000.0)
            
            if message is None:
                self.log.warn(f'Client {self.id} timed out waiting for ring model')
                continue
                
            if message.header == MODEL_UPDATE:
                sender_id = message.body.get('sender_id')
                received_state = message.body[MESSAGE_BODY_STATES]
                self.aggregate_with_received_model(received_state)
                
                self.log.info(f'Client {self.id} aggregated model from {sender_id} with local model')
                break
            else:
                self.log.warn(f'Client {self.id} received unexpected message: {message.header}')

    def aggregate_with_received_model(self, received_state):
        local_state = self.model.state_dict()
        aggregated_state = {}
        
        # Simple averaging (equal weights)
        for key in local_state.keys():
            aggregated_state[key] = (local_state[key] + received_state[key]) / 2.0
        self.model.load_state_dict(aggregated_state)

    def run_ring_cycle(self):
        """Execute one complete ring cycle where model passes through all nodes sequentially"""
        cycle_counter = 0
        while True:
            if self.is_ring_initiator:
                self.local_training_round()
                self.send_model_to_next_node()
                print(f"--------------------------------")
                self.receive_and_aggregate_model()
                cycle_counter += 1
                self.log.info(f"================= Ring Federated Learning Round {cycle_counter} completed =================")
            else:
                self.receive_and_aggregate_model()
                self.local_training_round()
                print(f"--------------------------------")
                self.send_model_to_next_node()
            if cycle_counter == self.federated_learning_rounds:
                break
            

    def run(self):
        self.run_ring_cycle()

    def train(self, optimizer_fn, loss_fn):
        """Initialize and start RDFL training"""
        # Initialize components
        self.build()
        # Setup optimizer and loss function
        self.optimizer = optimizer_fn(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = loss_fn()
        print(f"node {self.id} has the neighbors {self.neighbors}")
        self.run()

    def get_next_node_in_ring(self) -> str:
        """0→1→2→3→0"""
        current_id = get_last_char_as_int(self.id)  # Extract number from client_X
        total_clients = self.config.NUMBER_OF_CLIENTS
        next_id = (current_id + 1) % total_clients  # Wrap around using modulo
        return f"client_{next_id}"
    
    def get_previous_node_in_ring(self) -> str:
        """0←1←2←3←0"""  
        current_id = get_last_char_as_int(self.id)
        total_clients = self.config.NUMBER_OF_CLIENTS
        prev_id = (current_id - 1) % total_clients
        return f"client_{prev_id}"
