from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import numpy as np
import random


class MockEncoding1(TransformerMixin):
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return np.ones((10, 1))


class MockEncoding2(TransformerMixin):
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return np.ones((10, 1))


class MockModel1(BaseEstimator):
    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X=None, **kwargs):
        out = np.random.random(size=(X.shape[0])) * 5
        return out.astype(int)


class MockModel2(BaseEstimator):
    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X=None, **kwargs):
        out = np.random.random(size=(X.shape[0])) * 5
        return out.astype(int)


def mock_metric(X, labels):
    return random.random()
