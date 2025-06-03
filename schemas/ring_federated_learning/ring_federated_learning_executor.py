import os

from constants.framework import SERVER_ID

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "4"

import ray
import time

from core.federated import FederatedNode
from decorators.remote import remote
from schemas.ring_federated_learning.ring_federated_learning import \
    RingFederatedLearning
from schemas.ring_federated_learning.ring_federated_learning_schema import RingFederatedLearningSchema

from validators.config_validator import ConfigValidator
from utils.log import Log
import torch

def ring_federated_learning_executor(config: ConfigValidator, log: Log):

    ray.init()

    federation = RingFederatedLearningSchema(
        client_template=RingFederatedLearning,
        roles=[config.CLIENT_ROLE for _ in range(config.NUMBER_OF_CLIENTS)],
        config=config,
        log=log,
        resources="uniform",
    )

    federation.train(
        client_args={"optimizer_fn": torch.optim.SGD, "loss_fn": torch.nn.CrossEntropyLoss,},
        blocking=False,
    )

    for _ in range(config.NUMBER_OF_CLIENTS):
        version = federation.pull_version()
        print(version)
    time.sleep(3)

    federation.stop()