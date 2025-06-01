from constants.federated_learning_schema_constants import CLUSTER_FEDERATED_LEARNING, DECENTRALIZED_FEDERATED_LEARNING, \
    TRADITIONAL_FEDERATED_LEARNING
from constants.federated_learning_topology_constants import STAR_TOPOLOGY, RING_TOPOLOGY, MESH_TOPOLOGY, CUSTOM_TOPOLOGY
from schemas import star_federated_learning_executor
from schemas.ring_federated_learning.ring_federated_learning_executor import ring_federated_learning_executor
from utils.log import Log


# Should it be renamed now that i give it topology too?
def schema_factory(schema: str, topology: str, log: 'Log'):
    if schema == TRADITIONAL_FEDERATED_LEARNING:
        function = star_federated_learning_executor
        log.info(f"returning {function.__name__}")
        return function
    if schema == DECENTRALIZED_FEDERATED_LEARNING:
        if topology == RING_TOPOLOGY:
            function = ring_federated_learning_executor
            log.info(f"returning {function.__name__}")
            return function
        elif topology == MESH_TOPOLOGY:
            return NotImplementedError
        elif topology == CUSTOM_TOPOLOGY:
            return NotImplementedError
        return None
    elif schema == CLUSTER_FEDERATED_LEARNING:
        raise NotImplementedError()
    elif schema == DECENTRALIZED_FEDERATED_LEARNING:
        raise NotImplementedError()
    else:
        raise ValueError(f"unknown schema type: {schema}")
