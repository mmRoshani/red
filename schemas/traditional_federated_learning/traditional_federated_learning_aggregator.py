from constants.aggregation_strategy_constants import AGGREGATION_STRATEGY_FED_PROX, AGGREGATION_STRATEGY_FED_AVG
from core.aggregator.fed_avg_aggregator_base import FedAvgBase
from core.aggregator.fed_prox_aggregator_base import FedProxBase
from validators.config_validator import ConfigValidator
from typing import List, Dict


class TraditionalFederatedLearningAggregator:
    def __init__(self, config: 'ConfigValidator'):
        self.config = config
        self.log = config.RUNTIME_COMFIG.log

        if self.config.AGGREGATION_STRATEGY == AGGREGATION_STRATEGY_FED_AVG:
            self.strategy = FedAvgBase(use_sample_scaling=self.config.AGGREGATION_SAMPLE_SCALING)
        elif self.config.AGGREGATION_STRATEGY == AGGREGATION_STRATEGY_FED_PROX:
            self.strategy = FedProxBase(
                mu= 0.1, # TODO: read from config
                use_sample_scaling=self.config.AGGREGATION_SAMPLE_SCALING
            )
        else:
            raise ValueError(f"Unsupported aggregation method: {self.config.AGGREGATION_STRATEGY}")

        self.log.info(f'initializing the TraditionalFederatedLearningAggregator with {self.config.AGGREGATION_STRATEGY} strategy')

    def set_iteration(self, client_ids: List[str]) -> None:
        self.strategy.set_iteration(client_ids)

    def update(self, client_id: str, client_dict: Dict) -> None:
        self.strategy.update(client_id, client_dict)

    def aggregate(self) -> Dict:
        return self.strategy.aggregate()

    @property
    def ready(self) -> bool:
        return self.strategy.ready