import ray
import time
from typing import List
from schemas.traditional_federated_learning.templates import MessagingServer, MessagingClient
from schemas.traditional_federated_learning.traditional_federated_learning_schema import TraditionalFederatedLearningSchema
from validators.config_validator import ConfigValidator


def run_traditional_federated_learning(config: ConfigValidator):
    ray.init()

    federation = TraditionalFederatedLearningSchema(
        server_template=MessagingServer,
        client_template=MessagingClient,
        n_clients_or_ids=config.NUMBER_OF_CLIENTS,
        roles=[config.CLIENT_ROLE for _ in range(config.NUMBER_OF_CLIENTS)],
    )

    report = federation.train(
        server_args={"out_msg": lambda: "Hello from server!"},
        client_args={"out_msg": lambda: "Hello from client!"},
    )

    print(report)

    for _ in range(4):
        version = federation.pull_version()
        print(version)
    time.sleep(3)

    federation.stop()