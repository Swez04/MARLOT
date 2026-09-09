"""
Isolation Forest baseline. Unsupervised anomaly detector — trained only
on features, no fraud labels used at fit time (labels are only used
later, for evaluation).
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from simulator.transaction import Transaction

CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "subscription", "jewelry"]


def featurize(transactions: list[Transaction]) -> np.ndarray:
    """Turn transactions into a simple numeric feature matrix.
    Keep this in sync with autoencoder.py's featurize for now — later
    you can build a shared feature module."""
    rows = []
    for t in transactions:
        cat_onehot = [1.0 if t.merchant_category == c else 0.0 for c in CATEGORIES]
        rows.append([t.amount] + cat_onehot)
    return np.array(rows)


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self._fitted = False

    def fit(self, transactions: list[Transaction]):
        X = featurize(transactions)
        self.model.fit(X)
        self._fitted = True
        return self

    def score_batch(self, transactions: list[Transaction]) -> list[int]:
        assert self._fitted, "call fit() before scoring"
        X = featurize(transactions)
        preds = self.model.predict(X)  # -1 = anomaly, 1 = normal
        return [1 if p == -1 else 0 for p in preds]
