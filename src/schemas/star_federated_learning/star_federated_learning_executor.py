import os

from src.constants import SERVER_ID

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "4"

import ray
import time

from src.core.federated import FederatedNode
from src.decorators.remote import remote
from src.schemas.star_federated_learning.star_federated_learning_client import \
    StarFederatedLearningClient
from src.schemas.star_federated_learning.star_federated_learning_schema import StarFederatedLearningSchema
from src.schemas.star_federated_learning.star_federated_learning_server import \
    StarFederatedLearningServer
from src.validators.config_validator import ConfigValidator
from src.utils.log import Log
import torch

def star_federated_learning_executor(config: ConfigValidator, log: Log):
    # TODO: read from .env files
    ray.init(runtime_env={"working_dir": "/home/amir/red", 'excludes': ['/home/amir/red/.git/objects/ff/2f5a96367a3c1656f09c93649f07143ff8b11e']}) 

    federation = StarFederatedLearningSchema(
        server_template=StarFederatedLearningServer,
        client_template=StarFederatedLearningClient,
        roles=[config.CLIENT_ROLE for _ in range(config.NUMBER_OF_CLIENTS)],
        config=config,
        log=log,
        resources="uniform",
        server_id=SERVER_ID,
    )

    federation.train(
        server_args={},
        client_args={"optimizer_fn": torch.optim.SGD, "loss_fn": torch.nn.CrossEntropyLoss,},
        blocking=False,
    )

    for _ in range(config.NUMBER_OF_CLIENTS):
        version = federation.pull_version()
        print(version)
    time.sleep(3)

    federation.stop()