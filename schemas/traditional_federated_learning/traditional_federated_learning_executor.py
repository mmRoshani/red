import ray
import time
from schemas.traditional_federated_learning.traditional_federated_learning_client import \
    TraditionalFederatedLearningClient
from schemas.traditional_federated_learning.traditional_federated_learning_schema import TraditionalFederatedLearningSchema
from schemas.traditional_federated_learning.traditional_federated_learning_server import \
    TraditionalFederatedLearningServer
from validators.config_validator import ConfigValidator
import torch

def traditional_federated_learning_executor(config: ConfigValidator):
    ray.init(num_gpus=1)

    federation = TraditionalFederatedLearningSchema(
        server_template=TraditionalFederatedLearningServer,
        client_template=TraditionalFederatedLearningClient,
        roles=[config.CLIENT_ROLE for _ in range(config.NUMBER_OF_CLIENTS)],
        config=config
    )

    federation.train(
        server_args={},
        client_args={"optimizer_fn": torch.optim.SGD, "loss_fn": torch.nn.CrossEntropyLoss,},
        blocking=True
    )

    for _ in range(4):
        version = federation.pull_version()
        print(version)
    time.sleep(3)

    federation.stop()