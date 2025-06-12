from typing import Union, List
import ray
from ray.util.placement_group import PlacementGroup

from src.core.federated.federated_base import FederatedBase
from src.core.federated.federated_node import FederatedNode
from src.core.federated.virtual_node import VirtualNode
from src.utils.client_ids_list import client_ids_list_generator
from src.validators.config_validator import ConfigValidator
from src.utils.log import Log
from .custom_federated_learning_schema import CustomFederatedLearningSchema
from .custom_federated_learning import CustomFederatedLearning

def custom_federated_learning_executor(
    config: ConfigValidator,
    log: Log,
    resources: Union[str, PlacementGroup] = "uniform",
    is_tune: bool = False,
    bundle_offset: int = 0,
) -> FederatedBase:

    n_clients_or_ids: Union[int, List[str]] = config.NUMBER_OF_CLIENTS
    if isinstance(n_clients_or_ids, int):
        c_ids = client_ids_list_generator(n_clients_or_ids, log)
    else:
        c_ids = n_clients_or_ids

    roles = ["train"] * len(c_ids)
    
    return CustomFederatedLearningSchema(
        client_template=CustomFederatedLearning,
        roles=roles,
        config=config,
        log=log,
        resources=resources,
        is_tune=is_tune,
        bundle_offset=bundle_offset,
    )