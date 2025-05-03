from core.federated.federated_node import FederatedNode
from decorators.remote import remote
import time

@remote
class MessagingClient(FederatedNode):
    def train(self, out_msg: str) -> None:
        while True:
            self.send("exchange", {"msg": out_msg()})
            msg = self.receive()
            print(
                f"{self.id} received {msg.body['msg']} from {msg.sender_id}",
                msg.timestamp,
            )
            time.sleep(3)