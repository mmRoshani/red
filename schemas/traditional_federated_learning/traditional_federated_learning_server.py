from core.federated import FederatedNode
from decorators.remote import remote
from nets.network_factory import network_factory
from schemas.traditional_federated_learning.traditional_federated_learning_aggregator import \
    TraditionalFederatedLearningAggregator
from utils.checker import device_checker
from validators.config_validator import ConfigValidator
from typing import List
import random

@remote
class TraditionalFederatedLearningServer(FederatedNode):
    def __init__(self, node_id: str, role: str, federation_id: str, config: 'ConfigValidator') -> None:
        super().__init__(node_id=node_id, role=role, config=config, federation_id=federation_id)
        self.federated_learning_rounds = None
        self.model = None
        self.number_of_clients = None
        self.clients_id_list = []
        self.device = 'cpu'

        self.server_aggregator = TraditionalFederatedLearningAggregator(config)

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

    def run(self):
        """Main federated learning execution loop"""
        for _ in range(self.federated_learning_rounds):
            # Sample clients for this round
            client_sample = self.sample_clients(
                client_sampling_rate=self.config.CLIENT_SAMPLING_RATE
            )

            # Prepare and send global model to clients
            self.server_aggregator.set_iteration(client_sample)
            self._send_model_to_clients(client_sample)

            # Collect client updates
            while not self.server_aggregator.ready:
                message = self.get_message(block=True)
                self.on_client_receive(message)

            # Update global model with aggregated parameters
            aggregated_state = self.server_aggregator.aggregate()
            self.model.load_state_dict(aggregated_state["state"])

    def train(self, verbose=False, **kwargs):
        if verbose:
            print("Starting federated training")
        self.build()
        self.run()

    def _send_model_to_clients(self, client_sample: List[str]):
        """Send current global model to selected clients with necessary parameters"""
        message_body = {
            "state": self.model.state_dict(),
            "hyperparams": {
                "mu": 0.1  # just relevant for FedProx, TODO: read from config
            }
        }
        self.send(header="model_update", body=message_body, to=client_sample)

    def on_client_receive(self, message):
        """Handle incoming client updates"""
        client_id = message.sender
        client_state = message.body["state"]
        n_samples = message.body["n_samples"]

        self.server_aggregator.update(
            client_id=client_id,
            client_dict={
                "state": client_state,
                "n_samples": n_samples
            }
        )

    def sample_clients(self, client_sampling_rate: float, random_seed: int = 42) -> List[str]:
        random.seed(random_seed)

        """Sample participating clients for the current round"""
        num_clients = self.number_of_clients
        num_to_sample = int(client_sampling_rate * num_clients)

        if num_to_sample < 0 or num_to_sample > num_clients:
            raise ValueError("Invalid number of clients to sample")

        return random.sample(self.clients_id_list, num_to_sample)
