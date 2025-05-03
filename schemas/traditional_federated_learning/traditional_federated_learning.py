import ray
import time

from schemas.traditional_federated_learning.communication import MessagingServer, MessagingClient
from schemas.traditional_federated_learning.traditional_federated_learning_schema import TraditionalFederatedLearningSchema


def run_traditional_federated_learning():
    ray.init()
    federation = TraditionalFederatedLearningSchema(
        server_template=MessagingServer,
        client_template=MessagingClient,
        n_clients_or_ids=4,
        roles=["train" for _ in range(4)],
    )
    report = federation.train(
        server_args={"out_msg": lambda: "Hello from server!"},
        client_args={"out_msg": lambda: "Hello from client!"},
    )
    for _ in range(4):
        version = federation.pull_version()
        print(version)
    time.sleep(3)
    federation.stop()