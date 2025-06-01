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
        self.states = defaultdict(lambda: 0)

    def setup(self, client_ids: List[str]):
        self.expected_clients = deepcopy(client_ids)
        self.received_clients = [False] * len(client_ids)

        for client_id in self.expected_clients:
            self.states.update({client_id: None})

    def update(self, client_dict: Dict):
        raise NotImplementedError

    def compute(self) -> defaultdict:
        raise NotImplementedError
    # RvQ: Used nowhere, why?

    @property
    def ready(self):
        return all(
            [(expected in self.received_clients) for expected in self.expected_clients]
        )
