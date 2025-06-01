from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

from data.image.label_distribution import calculate_label_distribution
from utils.log import Log
from validators.config_validator import ConfigValidator


def compute_data_driven_clustering(loader, config: "ConfigValidator", log: "Log"):
    label_distributions = []
    label_distributions_hot = []
    for idx, loader in enumerate(loader):
        _label_distribution, _label_distribution_hot = calculate_label_distribution(
            loader, f"client_{idx}", config.NUMBER_OF_CLASSES, log
        )

        label_distributions.append(_label_distribution)
        label_distributions_hot.append(_label_distribution_hot)

    similarity_matrix = cosine_similarity(label_distributions)

    clustering = AffinityPropagation(affinity="precomputed", random_state=42)
    clustering.fit(similarity_matrix)
    labels = clustering.labels_

    clusters = {}
    for client_id, cluster_id in enumerate(labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(client_id)

    log.info(clusters)

    return clusters, label_distributions, label_distributions_hot
