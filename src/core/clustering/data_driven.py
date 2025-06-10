import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.log import Log
from src.validators.config_validator import ConfigValidator


def calculate_label_distribution(
    dataloader, loader_name: str, config: ConfigValidator, log: "Log"
):
    label_counts = np.zeros(config.NUMBER_OF_CLASSES)
    for _, labels in dataloader:
        for label in labels.numpy():
            label_counts[label] += 1

    log.info(f"client {loader_name} label distribution is: {label_counts}")
    return label_counts


def compute_similarity_matrix(distributions):
    similarity_matrix = cosine_similarity(distributions)
    return similarity_matrix


def cluster_clients(similarity_matrix):
    clustering = AffinityPropagation(affinity="precomputed", random_state=42)
    clustering.fit(similarity_matrix)
    return clustering.labels_
