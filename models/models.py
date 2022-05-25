from sklearn.base import BaseEstimator
import numpy as np


class RandomModel(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X=None, **kwargs):
        out = np.random.random(size=(X.shape[0])) * 5
        return out.astype(int)
