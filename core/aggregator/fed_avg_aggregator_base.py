from typing import Dict, Optional, List
import torch
from constants.framework import MODEL_UPDATE, MESSAGE_BODY_STATES
from core.communication.message import Message
from utils.log import Log
from core.aggregator.aggregator_base import AggregatorBase
from validators.config_validator import ConfigValidator
from collections import defaultdict


class FedAvgAggregator(AggregatorBase):
    def __init__(self, config: 'ConfigValidator', log: 'Log'):
        super().__init__(config, log)
        self.client_ids: Dict[str, bool] = {}
        self.state_dict: Optional[Dict] = None

    def update(self,msg: Message):
        sender_id = msg.sender_id
        if sender_id not in self.expected_clients:
            raise ValueError(
                f"Message received from client {sender_id}, not included in the expected clients."
            )
        self.received_clients.append(sender_id)
        self.states[sender_id] = msg.body[MESSAGE_BODY_STATES]
        self.log.info(f'received message from client {sender_id} for aggregation')

    def compute(self) -> Dict[str, torch.Tensor]:
        """Averages the state dictionaries received from clients."""

        valid_states = [state for state in self.states.values() if state is not None]
        num_valid_clients = len(valid_states)

        if num_valid_clients == 0:
            self.log.info("number of received client updates is 0")
            return {}

        aggregated_state = defaultdict(
            lambda: torch.zeros_like(list(valid_states[0].values())[0]))  # Initialize with zeros

        for client_state in valid_states:
            for key, value in client_state.items():
                aggregated_state[key] += value

        for key in aggregated_state:
            aggregated_state[key] = aggregated_state[key] / num_valid_clients

        return dict(aggregated_state)

    @property
    def ready(self):
        _client_readiness_list = [(expected in self.received_clients) for expected in self.expected_clients]
        self.log.info(f'client readiness list is {_client_readiness_list}')
        return all(_client_readiness_list)
