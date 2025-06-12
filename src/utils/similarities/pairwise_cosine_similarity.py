<<<<<<< HEAD:src/utils/similarities/pairwise_cosine_similarity.py
import numpy as np
from utils.vectorise_model_parameters import vectorise_model_parameters

def pairwise_cosine_similarity(clients, consider_parameters: bool, log):
    comparing_vectors = None
    if consider_parameters:
        log.info(
            f'running cosine similarity on parameters'
        )
        comparing_vectors = [
            vectorise_model_parameters(client.model).detach().cpu().numpy() for client in clients
        ]
    else:
        log.info(
            f'running cosine similarity on gradients'
        )
        comparing_vectors = [
            np.array(list(client.gradients.values())) for client in clients
        ]
        log.info(
            f"the length of gradients for each model is {len(comparing_vectors[0])}"
        )

    n = len(clients)
    similarities = np.zeros((n, n))

    for i in range(n):
        vi = comparing_vectors[i]
        norm_i = np.linalg.norm(vi)

        for j in range(n):
            vj = comparing_vectors[j]
            norm_j = np.linalg.norm(vj)
            if norm_i == 0 or norm_j == 0:
                similarities[i][j] = 0.0
            else:
                similarities[i][j] = np.dot(vi, vj) / (norm_i * norm_j)
    print(similarities)
    np.fill_diagonal(similarities, 1)

=======
import numpy as np
from utils.vectorise_model_parameters import vectorise_model_parameters

def pairwise_cosine_similarity(clients, consider_parameters: bool, log):
    comparing_vectors = None
    if consider_parameters:
        log.info(
            f'running cosine similarity on parameters'
        )
        comparing_vectors = [
            vectorise_model_parameters(client.model).detach().cpu().numpy() for client in clients
        ]
    else:
        log.info(
            f'running cosine similarity on gradients'
        )
        comparing_vectors = [
            np.array(list(client.gradients.values())) for client in clients
        ]
        log.info(
            f"the length of gradients for each model is {len(comparing_vectors[0])}"
        )

    n = len(clients)
    similarities = np.zeros((n, n))

    for i in range(n):
        vi = comparing_vectors[i]
        norm_i = np.linalg.norm(vi)

        for j in range(n):
            vj = comparing_vectors[j]
            norm_j = np.linalg.norm(vj)
            if norm_i == 0 or norm_j == 0:
                similarities[i][j] = 0.0
            else:
                similarities[i][j] = np.dot(vi, vj) / (norm_i * norm_j)
    print(similarities)
    np.fill_diagonal(similarities, 1)

>>>>>>> dev:utils/similarities/pairwise_cosine_similarity.py
    return similarities