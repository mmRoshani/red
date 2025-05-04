from abc import ABC
from typing import Dict, List
from collections import defaultdict
from copy import deepcopy
from utils.log import Log

from validators.config_validator import ConfigValidator


class AggregatorBase(ABC):
    def __init__(self, config: 'ConfigValidator', log: 'Log'):
        self.config = config
        self.log = log
        self.expected_clients = []
        self.received_clients = []
        self.residual_ids = []
        self.state = defaultdict(lambda: 0)

    def setup(self, client_ids: List[str]):
        self.expected_clients = deepcopy(client_ids)
        self.received_clients = []
        self.state = defaultdict(lambda: 0)

    def __call__(self, msg):
        if msg.sender_id not in self.expected_clients:
            raise ValueError(
                f"Message received from client {msg.sender_id}, not included in the expected clients."
            )
        self.received_clients.append(msg.sender_id)
        self.update(msg.body)

    def update(self, client_dict: Dict):
        raise NotImplementedError

    def compute(self):
        return self.state

    @property
    def ready(self):
        return all(
            [(expected in self.received_clients) for expected in self.expected_clients]
        )
