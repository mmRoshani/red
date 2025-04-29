from utils.log import Log
from utils.vectorise_model_parameters import vectorise_model_parameters
import numpy as np

def pairwise_euclidean_similarity(clients, consider_parameters: bool, log: 'Log'):
    comparing_vectors = None
    if consider_parameters:
        log.info('running euclidean similarity on parameters')
        comparing_vectors = [
            vectorise_model_parameters(client.model).detach().cpu().numpy() for client in clients
        ]
    else:
        log.info('running euclidean similarity on gradients')
        comparing_vectors = [
            np.array(list(client.gradients.values())) for client in clients
        ]
        log.info("the length of gradients for each model is {len(comparing_vectors[0])}")

    n = len(clients)
    similarities = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarities[i][j] = np.linalg.norm(comparing_vectors[i] - comparing_vectors[j])

    similarity_matrix = -similarities

    return similarity_matrix