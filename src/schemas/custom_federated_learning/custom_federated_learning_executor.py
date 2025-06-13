import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "4"

import ray
import time

from src.core.federated import FederatedNode
from src.decorators.remote import remote
from src.schemas.custom_federated_learning.custom_federated_learning import \
    CustomFederatedLearning
from src.schemas.custom_federated_learning.custom_federated_learning_schema import CustomFederatedLearningSchema

from src.validators.config_validator import ConfigValidator
from src.utils.log import Log
import torch

def custom_federated_learning_executor(config: ConfigValidator, log: Log):
    # TODO: read from .env files
    runtime_env = {}
    ray.init(runtime_env=runtime_env) 

    federation = CustomFederatedLearningSchema(
        client_template=CustomFederatedLearning,
        roles=[config.CLIENT_ROLE for _ in range(config.NUMBER_OF_CLIENTS)],
        config=config,
        log=log,
        resources="uniform",
    )

    federation.train(
        client_args={"optimizer_fn": torch.optim.SGD, "loss_fn": torch.nn.CrossEntropyLoss,},
        blocking=True,
    )

    # for _ in range(config.NUMBER_OF_CLIENTS):
    #     version = federation.pull_version()
    #     print(version)
    time.sleep(3)

    federation.stop()