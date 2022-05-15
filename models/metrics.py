import numpy as np
from functools import partial
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from model_logging import logger


def wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Sum of squared distance from points to centroids of clusters
    to which these points belong
    Args:
        X (np.ndarray)
        labels (np.ndarray)

    Returns:
        float
    """
    unique_labels = np.unique(labels)
    grouped_vectors = list(
        map(partial(_get_vectors_given_label, X=X, labels=labels), unique_labels)
    )
    score = 0
    for group in grouped_vectors:
        centroid = np.mean(group, axis=0)
        group -= centroid
        score += np.sum(np.square(group))
    return score


def min_interclust_dist(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Minimal distance between points of different clusters
    Args:
        X (np.ndarray)
        labels (np.ndarray)

    Returns:
        float
    """
    clusters = np.unique(labels)
    min_dist = np.inf
    for cluster1_label in clusters:
        cluster1 = _get_vectors_given_label(cluster1_label, X, labels)
        for cluster2_label in clusters:
            if cluster1_label != cluster2_label:
                cluster2 = _get_vectors_given_label(cluster2_label, X, labels)
                interclust_min_dist = np.min(distance.cdist(cluster1, cluster2))
                min_dist = np.min([min_dist, interclust_min_dist])
    return min_dist


def mean_inclust_dist(X: np.ndarray, labels: np.ndarray) -> float:
    inclust_dist_list = _inclust_mean_dists(X, labels)
    return np.mean(inclust_dist_list)


def _inclust_mean_dists(X: np.ndarray, labels: np.ndarray) -> list:
    """
    Mean distances from centroid for each cluster
    Args:
        X (np.ndarray)
        labels (np.ndarray)

    Returns:
        list[float]: mean distances of points from centroids for each cluster
    """
    clusters = set(labels)
    inclust_dist_list = []
    for cluster_label in clusters:
        cluster = _get_vectors_given_label(cluster_label, X, labels)
        inclust_dist = np.mean(distance.pdist(cluster))
        inclust_dist_list.append(inclust_dist)
    return inclust_dist_list


def _get_vectors_given_label(
    label: int, X: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    Gets rows with indices where corresponding label
    is equal to `label` parameter

    Args:
        label (int):_
        X (np.ndarray):
        labels (np.ndarray):

    Returns:
        np.ndarray:
    """
    return X[labels == label, :]


# Metrics. All are callable with argument X - matrix where each row corresponds
# to entry and labels - cluster assigned to entries from X
metrics_labels = [
    "WCSS",
    "Silhouette",
    "Min intercluster dist.",
    "Mean intercluster dist.",
]
metrics = [wcss, silhouette_score, min_interclust_dist, mean_inclust_dist]
logger.info("Loaded metrics: " + str([metric.__name__ for metric in metrics]))

# Testing code if metrics work
# from sklearn.datasets import load_iris
# dataset = load_iris()['data']
# from sklearn.cluster import KMeans
# model = KMeans(n_clusters=2)
# model.fit(dataset)
# for metric in metrics:
#     print(metric(dataset, model.predict(dataset)))
