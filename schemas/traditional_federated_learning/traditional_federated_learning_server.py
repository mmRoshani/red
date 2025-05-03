from core.federated import FederatedNode
from nets.network_factory import network_factory
from validators.config_validator import ConfigValidator


class TraditionalFederatedLearningServer(FederatedNode):
    def __init__(self, node_id: str, role: str, federation_id: str, config: 'ConfigValidator') -> None:
        super().__init__(node_id=node_id, role=role, config=config, federation_id=federation_id)

    def build(self, config):
        self.federated_learning_rounds = config.FEDERATED_LEARNING_ROUNDS
        self.local_epochs = config.NUMBER_OF_EPOCHS
        self.model = network_factory(model_type=config.MODEL_TYPE, number_of_classes=config.NUMBER_OF_CLASSES, pretrained=config.PRETRAINED_MODELS)

    def run(self, config: 'ConfigValidator'):
        for _ in range(self.global_epochs):
            client_sample = self.sample_clients(client_sampling_rate=config.CLIENT_SAMPLING_RATE)

            self.send(msg_type="model", body=self.model.state_dict(), ids=client_sample)
            self.set_iteration(client_ids=client_sample)

            while not self.server_aggregator.ready:
                self.on_client_receive(self.get_message(block=True))
                self.aggregate()

            self.aggregate()

    def sample_clients(self, client_sampling_rate: float):
        raise NotImplementedError

    def on_client_receive(self, message):
        raise NotImplementedError

    def aggregate(self):
        raise NotImplementedError