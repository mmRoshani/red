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


@remote(num_gpus=1, num_cpus=1)
class TraditionalFederatedLearningClient(FederatedNode):
    def __init__(self, node_id: str, role: str, config: 'ConfigValidator', log: Log) -> None:
        super().__init__(node_id=node_id, role=role, config=config, log=log)
        self.model = None
        self.optimizer: torch.optim.SGD = None
        self.criterion: torch.nn.CrossEntropyLoss = None
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

    def send_local_model_to_server(self, local_model: NN):
        message_body = {
            MESSAGE_BODY_STATES: local_model.to('cpu').state_dict(),
        }
        self.send(header=MODEL_UPDATE, body=message_body, to=SERVER_ID)

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
        """Learning configurations"""
        local_model: NN = copy.deepcopy(self.model)
        self.optimizer = optimizer_fn(local_model.parameters(), lr=self.config.LEARNING_RATE)
        self.criterion = loss_fn()

        """Waiting for server given initial model"""
        server_init_message: Message = self.receive(block=True)
        if (server_init_message is not None) and (server_init_message.header == MODEL_UPDATE):
            local_model.load_state_dict(server_init_message.body['state'])
            local_model.to(self.device)
            # self.model = copy.deepcopy(local_model)
            self.log.info(f'received initial model from server for client {self.id}')
            self.log.info(f'start training loop')

            while True:

                local_model = self.epoch_trainer(local_model=local_model)
                self.send_local_model_to_server(local_model)

                aggregated_message: Message = self.receive(block=True)
                local_model.load_state_dict(aggregated_message.body['state'])
                local_model.to(self.device)

        else:
            self.log.warn(f'received message with id of {server_init_message.sender_id} and header of {server_init_message.header}')