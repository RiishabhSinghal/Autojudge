import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack

KEYWORDS = [
    "graph", "tree", "dp", "dynamic programming",
    "recursion", "greedy", "binary search",
    "matrix", "modulo", "shortest path",
    "dfs", "bfs", "segment tree"
]

class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = pd.DataFrame()
        data["text_length"] = X.apply(len)
        data["math_symbol_count"] = X.apply(
            lambda x: len(re.findall(r"[+\-*/%=<>^]", x))
        )

        for kw in KEYWORDS:
            data[f"kw_{kw.replace(' ', '_')}"] = X.str.count(kw)

        return data.values


class FeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, tfidf, handcrafted):
        self.tfidf = tfidf
        self.handcrafted = handcrafted

    def fit(self, X, y=None):
        self.tfidf.fit(X)
        self.handcrafted.fit(X)
        return self

    def transform(self, X):
        X1 = self.tfidf.transform(X)
        X2 = self.handcrafted.transform(X)
        return hstack([X1, X2])
