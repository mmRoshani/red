from typing import Dict, List, Optional

from core.aggregator.aggregator_base import AggregatorBase


class FedAvgBase(AggregatorBase):
    def __init__(self, use_sample_scaling: bool = False):
        self.use_sample_scaling = use_sample_scaling  # Configurable scaling
        self.client_ids: Dict[str, bool] = {}
        self.state_dict: Optional[Dict] = None
        self.n_samples: int = 0  # Only used if use_sample_scaling=True

    def set_iteration(self, client_ids: List[str]) -> None:
        self.client_ids = {c_id: False for c_id in client_ids}
        self.state_dict = None
        self.n_samples = 0

    def update(self, client_id: str, client_dict: Dict) -> None:
        local_state = client_dict["state"]

        if self.use_sample_scaling:
            local_n = client_dict["n_samples"]
            scaled_state = {k: v * local_n for k, v in local_state.items()}
            self.n_samples += local_n
        else:
            scaled_state = local_state  # No scaling

        if self.state_dict is None:
            self.state_dict = {k: v.clone() for k, v in scaled_state.items()}
        else:
            for key in scaled_state:
                self.state_dict[key] += scaled_state[key]

        self.client_ids[client_id] = True

    def aggregate(self) -> Dict:
        if self.state_dict is None:
            return {"state": {}}

        if self.use_sample_scaling:
            divisor = self.n_samples if self.n_samples > 0 else 1
        else:
            divisor = len(self.client_ids)  # Simple average

        return {"state": {k: v / divisor for k, v in self.state_dict.items()}}

    @property
    def ready(self) -> bool:
        return all(self.client_ids.values())