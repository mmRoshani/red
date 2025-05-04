from typing import Dict, Optional, List
from utils.log import Log
from core.aggregator.aggregator_base import AggregatorBase
from validators.config_validator import ConfigValidator


class FedAvgAggregator(AggregatorBase):
    def __init__(self, config: 'ConfigValidator', log: 'Log', use_sample_scaling: bool = False, n_samples: int = 0):
        super().__init__(config, log)
        self.use_sample_scaling = use_sample_scaling
        self.client_ids: Dict[str, bool] = {}
        self.state_dict: Optional[Dict] = None
        self.n_samples: int = 0

        if not self.use_sample_scaling or n_samples == 0:
            self.n_samples = self.config.NUMBER_OF_CLIENTS
        else:
            self.n_samples = n_samples

        self.log.info(f'number of clients for aggregation: {self.n_samples}')

    def update(self, client_dict: Dict):
        n_samples = client_dict.pop("n_samples")
        self.state["n_samples"] += n_samples
        for k in client_dict:
            self.state[k] += client_dict[k] * n_samples

    def compute(self):
        n_samples = self.state["n_samples"]
        return {k: self.state[k] / n_samples for k in self.state}

    @property
    def ready(self):
        return all(
            [(expected in self.received_clients) for expected in self.expected_clients]
        )
