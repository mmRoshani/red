from core.federated.federated_node import FederatedNode
from decorators.remote import remote


@remote
class MessagingServer(FederatedNode):
    def train(self, out_msg: str):
        n_exchanges = 0
        while True:
            msg = self.receive()
            print(
                f"{self.id} received {msg.body['msg']} from {msg.sender_id} at",
                {msg.timestamp},
            )
            self.send("exchange", {"msg": out_msg()}, to=msg.sender_id)
            self.update_version(n_exchanges=n_exchanges)