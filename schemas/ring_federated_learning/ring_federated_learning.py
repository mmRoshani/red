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

    def send_local_model_to_neighbors(self, local_model: NN):
        message_body = {
            MESSAGE_BODY_STATES: local_model.to('cpu').state_dict(),
        }
        self.send(header=MODEL_UPDATE, body=message_body, to=self.neighbors)

    def _send_model_to_neighbors(self, neighbor_sample: List[str]):
        """Send current global model to selected neighbors with necessary parameters"""
        message_body = {
            "state": self.model.to('cpu').state_dict(),
        }
        self.send(header=MODEL_UPDATE, body=message_body, to=neighbor_sample)

    def sample_neighbors(self, neighbor_sampling_rate: float, random_seed: int = 42) -> List[str]:
        random.seed(random_seed)

        """Sample participating neighbors for the current round"""
        num_neighbors = self.number_of_neighbors
        num_to_sample = int(neighbor_sampling_rate * num_neighbors)

        if num_to_sample < 0 or num_to_sample > num_neighbors:
            raise ValueError("Invalid number of neighbors to sample")

        return random.sample(self.neighbors_id_list, num_to_sample)

    def run(self):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {self._tp_manager}")
        self.number_of_neighbors = len(self._tp_manager.get_neighbors(self.id))

        """Main federated learning execution loop"""
        for fl_round in range(self.federated_learning_rounds):

            # Prepare and send global model to clients
            if fl_round != 0:
                self.log.info(f'This is {self.id} starting decentralized federated round {fl_round}')
                self.aggregator.setup(self.neighbors)

                # Collect neighbors updates
                while not self.aggregator.ready:
                    neighbor_message = self.receive(block=True)
                    if (neighbor_message is not None) and (neighbor_message.header == MODEL_UPDATE):
                        self.aggregator.update(neighbor_message)

                    else:
                        self.log.warn(
                            f'{self.id} received message with id of {neighbor_message.sender_id} and header of '
                            f'{neighbor_message.header}')

                self.log.info(f"{self.id} is ready for aggregation in fl round {fl_round}")

                # Update global model with aggregated parameters
                aggregated_state = self.aggregator.compute()
                self.model.load_state_dict(aggregated_state)

                self._send_model_to_neighbors(self.neighbors_id_list)
                self.log.info(f"{self.id} send aggregated model to clients")



    def epoch_trainer(self, local_model: NN) -> NN:
        for epoch in range(self.local_epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = local_model(data)
                loss = self.criterion(outputs, labels)


                loss.backward()
                self.optimizer.step()
            self.log.info(f'loss for client {self.id} is {loss}')

        return local_model

    def train(self, optimizer_fn, loss_fn):
        """Initialize client components"""
        self.build()
        self.run()

        """Learning configurations"""
        local_model: NN = copy.deepcopy(self.model)
        self.optimizer = optimizer_fn(local_model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = loss_fn()

        local_model.to(self.device)
        self.log.info(f"Node {self.id} initialized its model locally")

        if self.neighbors is not None:
            while True:

                local_model = self.epoch_trainer(local_model=local_model)
                self.send_local_model_to_neighbors(local_model)

                aggregated_message: Message = self.receive(block=True)
                local_model.load_state_dict(aggregated_message.body['state'])
                local_model.to(self.device)
        # else:
        #     self.log.warn(f'received message with id of {server_init_message.sender_id} and header of
        #     {server_init_message.header}')
