from core.federated import FederatedNode
from decorators.remote import remote
from nets.network_factory import network_factory
import copy
from typing import Dict
import torch
from schemas.traditional_federated_learning.traditional_federated_learning_aggregator import \
    TraditionalFederatedLearningAggregator
from utils.checker import device_checker
from utils.get_last_char_as_int import get_last_char_as_int
from validators.config_validator import ConfigValidator
from typing import List
import random


@remote
class TraditionalFederatedLearningClient(FederatedNode):
    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', federation_id: str,) -> None:
        super().__init__(node_id=node_id, role=role, config=config, federation_id=federation_id)
        self.model = None
        self.local_epochs = None
        self.train_loader = None
        self.test_loader = None
        self.log = config.RUNTIME_COMFIG.log
        self.device = 'cpu'

    def build(self):
        """Initialize client components"""
        self.device = device_checker(self.config.DEVICE)
        self.local_epochs = self.config.NUMBER_OF_EPOCHS
        self.model = network_factory(
            model_type=self.config.MODEL_TYPE,
            number_of_classes=self.config.NUMBER_OF_CLASSES,
            pretrained=self.config.PRETRAINED_MODELS
        ).to(self.device)

        # Initialize local dataset
        self.train_loader = self.config.RUNTIME_COMFIG.train_loaders[get_last_char_as_int(self.id)]
        self.test_loader = self.config.RUNTIME_COMFIG.test_loaders[get_last_char_as_int(self.id)]

    def run(self):
        """Main client training loop"""
        while True:
            # Wait for server message
            message = self.receive()

            if message.header == "model_update":
                global_state = message.body["state"]

                # Load global model
                self.model.load_state_dict(global_state)

                # Local training
                local_state = self.train()

                # Send update back to server
                self.send(
                    header="client_update",
                    body={
                        "state": local_state,
                        "n_samples": len(self.train_loader.dataset)
                    },
                    to=message.sender
                )

    def train(self, optimizer_fn, loss_fn) -> Dict:
        self.build()
        """Perform local training with optional FedProx regularization"""
        local_model = copy.deepcopy(self.model)
        optimizer = optimizer_fn(local_model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = loss_fn()

        # FedProx specific logic (disabled) # TODO: Read from config
        mu = 0.0
        global_params = [param.detach().clone() for param in self.model.parameters()]

        for _ in range(self.local_epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = local_model(data)
                loss = criterion(outputs, labels)

                # Add FedProx regularization if mu > 0
                if mu > 0:
                    for param, global_param in zip(local_model.parameters(), global_params):
                        loss += mu / 2 * torch.norm(param - global_param) ** 2

                self.log.info(f'loss for client {self.id} is {loss}')

                loss.backward()
                optimizer.step()

        return local_model.state_dict()
