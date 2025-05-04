from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class AggregatorBase(ABC):
    """Base class for federated aggregation strategies."""

    @abstractmethod
    def set_iteration(self, client_ids: List[str]) -> None:
        """Initialize/reset the aggregator for a new round."""
        pass

    @abstractmethod
    def update(self, client_id: str, client_dict: Dict) -> None:
        """Process updates from a client."""
        pass

    @abstractmethod
    def aggregate(self) -> Dict:
        """Compute the aggregated global model."""
        pass

    @property
    @abstractmethod
    def ready(self) -> bool:
        """Check if all clients have reported."""
        pass