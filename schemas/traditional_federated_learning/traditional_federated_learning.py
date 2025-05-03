import ray
import time
from typing import List
from schemas.traditional_federated_learning.templates import MessagingServer, MessagingClient
from schemas.traditional_federated_learning.traditional_federated_learning_schema import TraditionalFederatedLearningSchema


def run_traditional_federated_learning(number_clients_or_ids: int,roles: List[str]):
    ray.init()

    federation = TraditionalFederatedLearningSchema(
        server_template=MessagingServer,
        client_template=MessagingClient,
        n_clients_or_ids=number_clients_or_ids,
        roles=roles,
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