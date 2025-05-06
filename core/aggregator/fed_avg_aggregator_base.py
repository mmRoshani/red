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
        valid_states = [s for s in self.states.values() if s is not None]
        n = len(valid_states)
        if n == 0:
            self.log.info("no client updates")
            return {}

        aggregated_state: Dict[str, torch.Tensor] = {}

        for client_state in valid_states:
            for k, v in client_state.items():
                if k not in aggregated_state:
                    aggregated_state[k] = torch.zeros_like(v)
                aggregated_state[k] += v

        for k in aggregated_state:
            aggregated_state[k] /= n

        return aggregated_state

    @property
    def ready(self):
        _client_readiness_list = [(expected in self.received_clients) for expected in self.expected_clients]
        self.log.info(f'client readiness list is {_client_readiness_list}')

        return all(_client_readiness_list)
