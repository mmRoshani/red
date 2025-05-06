import torch

from constants.framework import MODEL_UPDATE
from core.aggregator.fed_avg_aggregator_base import FedAvgAggregator
from core.communication.message import Message
from core.federated import FederatedNode
from decorators.remote import remote
from nets.network_factory import network_factory
from utils.checker import device_checker
from utils.client_ids_list import client_ids_list_generator
from validators.config_validator import ConfigValidator
from typing import List
import random
from utils.log import Log

@remote(num_gpus=1, num_cpus=1)
class TraditionalFederatedLearningServer(FederatedNode):
    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', log: 'Log') -> None:
        super().__init__(node_id=node_id, role=role, config=config, log=log)
        self.federated_learning_rounds = None
        self.model = None
        self.number_of_clients = None
        self.clients_id_list = []
        self.device = 'cpu'
        self.server_aggregator: FedAvgAggregator = None

    def train(self, verbose=False, **kwargs):
        if verbose:
            print("Starting federated training")
        self.build()
        self.run()

    def build(self):
        """Initialize server components"""
        self.device = device_checker(self.config.DEVICE)
        self.federated_learning_rounds = self.config.FEDERATED_LEARNING_ROUNDS
        self.model = network_factory(
            model_type=self.config.MODEL_TYPE,
            number_of_classes=self.config.NUMBER_OF_CLASSES,
            pretrained=self.config.PRETRAINED_MODELS).to(self.device)
        self.number_of_clients = self.config.NUMBER_OF_CLIENTS
        self.clients_id_list = self.config.RUNTIME_COMFIG.clients_id_list
        self.server_aggregator = FedAvgAggregator(config=self.config, log=self.log)

    def run(self):
        """Main federated learning execution loop"""
        for fl_round in range(self.federated_learning_rounds):

            # Prepare and send global model to clients
            # self.server_aggregator.set_iteration(client_sample)
            if fl_round == 0:
                self._send_model_to_clients(self.clients_id_list)
                self.log.info(f"send initial model to clients")
            else:
                self.log.info(f'this is the server side beginning og federated round number {fl_round}')
                self.server_aggregator.setup(self.neighbors)

                # Collect client updates
                while not self.server_aggregator.ready:
                    client_message = self.receive(block=True)
                    if (client_message is not None) and (client_message.header == MODEL_UPDATE):
                        self.server_aggregator.update(client_message)

                    else:
                        self.log.warn(
                            f'received message with id of {client_message.sender_id} and header of {client_message.header}')

                self.log.info(f">>>>>>>>>>>>>>>>>>>>ready for aggregation")

                # Update global model with aggregated parameters
                aggregated_state = self.server_aggregator.compute()
                self.model.load_state_dict(aggregated_state)

                self._send_model_to_clients(self.clients_id_list)
                self.log.info(f"send aggregated model to clients")

    def _send_model_to_clients(self, client_sample: List[str]):
        """Send current global model to selected clients with necessary parameters"""
        message_body = {
            "state": self.model.to('cpu').state_dict(),
        }
        self.send(header=MODEL_UPDATE, body=message_body, to=client_sample)

    def sample_clients(self, client_sampling_rate: float, random_seed: int = 42) -> List[str]:
        random.seed(random_seed)

        """Sample participating clients for the current round"""
        num_clients = self.number_of_clients
        num_to_sample = int(client_sampling_rate * num_clients)

        if num_to_sample < 0 or num_to_sample > num_clients:
            raise ValueError("Invalid number of clients to sample")

        return random.sample(self.clients_id_list, num_to_sample)