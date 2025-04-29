import numpy as np
from heapq import nlargest
from utils.log import Log
from collections import Counter
from itertools import combinations

def pairwise_coordinate_similarity(clients, remove_common_ids: bool, log: 'Log'):
    sensitivity_percentage: float = clients[0].sensitivity_percentage

    _top_gradients_count = int(
        np.ceil(
            sensitivity_percentage * len(clients[0].gradients) / 100
        )
    )

    _top_sensitive_gradients = []
    for client in clients:
        grads = client.gradients.items()
        top_keys = [
            k for k, _ in nlargest(_top_gradients_count, grads, key=lambda x: x[1])
        ]

        log.info(
            f"top sensitive computed with {len(top_keys)} entries. and all are {len(top_keys) == len(set(top_keys))}ly unique."
        )
        _top_sensitive_gradients.append(set(top_keys))

    if remove_common_ids:
        all_ids = [id_ for ids in _top_sensitive_gradients for id_ in ids]
        id_counts = Counter(all_ids)
        common_ids = {id_ for id_, count in id_counts.items() if count == len(clients)}

        _top_sensitive_gradients = [
            ids - common_ids for ids in _top_sensitive_gradients
        ]

        for _top_g in _top_sensitive_gradients:
            log.info(
                f"top sensitive computed (removed common ids) with {len(_top_g)} entries. and all are {len(_top_g) == len(set(_top_g))}ly unique."
            )

    n_clients = len(clients)
    similarities = np.zeros((n_clients, n_clients), dtype=float)

    for i, j in combinations(range(n_clients), 2):
        set_i = _top_sensitive_gradients[i]
        set_j = _top_sensitive_gradients[j]
        intersection = len(set_i & set_j)
        similarities[i, j] = similarities[j, i] = intersection

    np.fill_diagonal(similarities, _top_gradients_count)
    similarities = similarities /  _top_gradients_count
    return similarities
