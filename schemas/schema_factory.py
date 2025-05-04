from constants.federated_learning_schema_constants import CLUSTER_FEDERATED_LEARNING, DECENTRALIZED_FEDERATED_LEARNING, \
    TRADITIONAL_FEDERATED_LEARNING
from schemas import traditional_federated_learning_executor
from utils.log import Log


def schema_factory(schema: str, log: 'Log'):
    if schema == TRADITIONAL_FEDERATED_LEARNING:
        function = traditional_federated_learning_executor
        log.info(f"returning {function.__name__}")
        return function
    elif schema == CLUSTER_FEDERATED_LEARNING:
        raise NotImplementedError()
    elif schema == DECENTRALIZED_FEDERATED_LEARNING:
        raise NotImplementedError()
    else:
        raise ValueError(f"unknown schema type: {schema}")
