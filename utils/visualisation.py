import sys
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import silhouette_score

sys.path.append("models")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from typing import List, Callable, Tuple
from metrics import wcss, min_interclust_dist, mean_inclust_dist

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.figsize"] = [16, 12]

metric_labels = {
    wcss: "WCSS",
    min_interclust_dist: "Minimun intercluster dist.",
    mean_inclust_dist: "Mean in-cluster dist.",
    silhouette_score: "Silhouette",
}


def get_metrics_of_model(
    model: BaseEstimator,
    metrics: List[Callable[[np.ndarray, np.ndarray], float]],
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> pd.DataFrame:
    """
    Returns values of metrics specified in input list for train and test
    samples of given model
    Args:
        model (BaseEstimator)
        metrics (List[Callable[[np.ndarray, np.ndarray], float]])
        X_train (np.ndarray)
        X_test (np.ndarray)

    Returns:
        pd.DataFrame: Table with metrics
    """
    train_clusters, test_clusters = _get_clusters(model, X_train, X_test)
    output_df = pd.DataFrame(columns=["metric", "sample", "value"])
    for metric in metrics:
        train_value = metric(X_train, train_clusters)
        train_metric_row = pd.DataFrame(
            {
                "metric": metric_labels.get(metric, metric.__name__),
                "sample": "train",
                "value": train_value,
            },
            index=[0],
        )
        output_df = pd.concat([output_df, train_metric_row]).reset_index(drop=True)
        test_value = metric(X_test, test_clusters)
        test_metric_row = pd.DataFrame(
            {
                "metric": metric_labels.get(metric, metric.__name__),
                "sample": "test",
                "value": test_value,
            },
            index=[0],
        )
        output_df = pd.concat([output_df, test_metric_row]).reset_index(drop=True)
    return pd.pivot(output_df, index=["metric"], columns=["sample"])


def plot_clustering(
    model: BaseEstimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> None:
    """
    Plots clusters generated by model for train and test sample.
    Projected to plane using truncated SVD. Plots are next to each other.

    Args:
        model (BaseEstimator)
        X_train (np.ndarray)
        X_test (np.ndarray)
    """
    train_clusters, test_clusters = _get_clusters(model, X_train, X_test)
    X_train, X_test = _project_to_plane(X_train, X_test)
    fig, axs = plt.subplots(ncols=2)
    axs[0].set_title("Train sample")
    axs[1].set_title("Test sample")
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], c=train_clusters, ax=axs[0])
    sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], c=test_clusters, ax=axs[1])


def _get_clusters(
    model: BaseEstimator, X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns cluster generated by model. Trains model on train sample, and
    returns clusters for both samples. If fit method isn't supported e.g.
    Agglomerative clustering, then train_predict is invoked.

    Args:
        model (BaseEstimator)
        X_train (np.ndarray)
        X_test (np.ndarray)

    Returns:
        Tuple[np.ndarray, np.ndarray]: clusters for both samples
    """
    try:
        model.fit(X_train)
        train_clusters = model.predict(X_train)
        test_clusters = model.predict(X_test)
    except:
        train_clusters = model.fit_predict(X_train)
        test_clusters = model.fit_predict(X_test)
    return train_clusters, test_clusters


def _project_to_plane(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For train and test samples creates and fits truncated SVD estimator.
    Returns both samples projected to 2D plane.
    Args:
        X_train (np.ndarray)
        X_test (np.ndarray)

    Returns:
        Tuple[np.ndarray, np.ndarray]: projected train and test samples
    """
    svd = TruncatedSVD(n_components=2)
    svd.fit(X_train)
    return svd.transform(X_train), svd.transform(X_test)
