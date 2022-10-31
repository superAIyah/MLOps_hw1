from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


class CustomTransformer(BaseEstimator, TransformerMixin):
    trans_dict = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "PolynomialFeatures": PolynomialFeatures()
    }

    def __init__(self, names):
        self.transformers = [self.trans_dict[name] for name in names]

    def fit(self, X, y=None):
        for transformer in self.transformers:
            transformer.fit(X)
            X = transformer.transform(X)

    def transform(self, X, y=None):
        X_ = X.copy()
        for transformer in self.transformers:
            X_ = transformer.transform(X_)
        return X_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)
