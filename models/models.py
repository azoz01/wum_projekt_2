from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from model_logging import logger


class RandomModel(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X=None, **kwargs):
        out = np.random.random(size=(X.shape[0])) * 5
        return out.astype(int)


# Models - anything with fit and predict method
# (unfitted, but parameters can be set)
models_labels = ["Agglomerative", "Random model", "KMeans"]
models = [AgglomerativeClustering(n_clusters=5), RandomModel(), KMeans(n_clusters=5)]
logger.info("Loaded models: " + str(models_labels))

# Testing code if all models work (execute)
# from sklearn.datasets import load_iris
# dataset = load_iris()['data']
# for model in models:
#     model.fit(dataset)
#     print(model.predict(dataset)[:10])
