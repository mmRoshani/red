import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "4"

import ray
import time

from core.federated import FederatedNode
from decorators.remote import remote
from schemas.k_connect_federated_learning.k_connect_federated_learning import \
    KConnectFederatedLearning
from schemas.k_connect_federated_learning.k_connect_federated_learning_schema import KConnectFederatedLearningSchema

from validators.config_validator import ConfigValidator
from utils.log import Log
import torch

def k_connect_federated_learning_executor(config: ConfigValidator, log: Log):

    ray.init()

    federation = KConnectFederatedLearningSchema(
        client_template=KConnectFederatedLearning,
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