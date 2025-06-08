from constants.schema_constants import CLUSTER_FEDERATED_LEARNING, DECENTRALIZED_FEDERATED_LEARNING, \
    TRADITIONAL_FEDERATED_LEARNING
from constants.topology_constants import *
from schemas import star_federated_learning_executor
from schemas.ring_federated_learning.ring_federated_learning_executor import ring_federated_learning_executor
from schemas.k_connect_federated_learning.k_connect_federated_learning_executor import k_connect_federated_learning_executor
from utils.log import Log


# Should it be renamed now that i give it topology too?
def schema_factory(schema: str, topology: str, log: 'Log'):
    if schema == TRADITIONAL_FEDERATED_LEARNING:
        function = star_federated_learning_executor
        log.info(f"returning {function.__name__}")
        return function
    if schema == DECENTRALIZED_FEDERATED_LEARNING:
        if topology == TOPOLOGY_RING:
            function = ring_federated_learning_executor
            log.info(f"returning {function.__name__}")
            return function
        elif topology == TOPOLOGY_K_CONNECT:
            function = k_connect_federated_learning_executor
            log.info(f"returning {function.__name__}")
            return function
        elif topology == TOPOLOGY_CUSTOM:
            return NotImplementedError
        return None
    elif schema == CLUSTER_FEDERATED_LEARNING:
        raise NotImplementedError()
    elif schema == DECENTRALIZED_FEDERATED_LEARNING:
        raise NotImplementedError()
    else:
        raise ValueError(f"unknown schema type: {schema}")
