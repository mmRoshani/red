from core.aggregator.fed_avg_aggregator_base import FedAvgBase


class FedProxBase(FedAvgBase):
    """Inherits FedAvg aggregation but adds client-side regularization capability"""

    def __init__(self, mu: float = 0.1, use_sample_scaling: bool = True):
        super().__init__(use_sample_scaling=use_sample_scaling)
        self.mu = mu
