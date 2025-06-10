from src.core.aggregator.fed_avg_aggregator_base import FedAvgAggregator


class FedProxBase(FedAvgAggregator):
    """Inherits FedAvg aggregation but adds client-side regularization capability"""
    # RvQ: The fuck this means?
    def __init__(self, mu: float = 0.1, use_sample_scaling: bool = True):
        super().__init__(use_sample_scaling=use_sample_scaling)
        self.mu = mu
