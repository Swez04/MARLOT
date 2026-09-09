"""
Autoencoder baseline.

Uses sklearn's MLPRegressor as a bottleneck autoencoder (input -> small
hidden layer -> output, trained to reconstruct its own input). This
keeps the dependency list light for Phase 2. Swap in a real PyTorch
autoencoder later if you want more control / GPU speed.

High reconstruction error on unseen data = anomaly = likely fraud.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor

from detectors.isolation_forest import featurize  # reuse same features


class AutoencoderDetector:
    def __init__(self, hidden_size: int = 4, threshold_percentile: float = 95, random_state: int = 42):
        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_size,),
            max_iter=500,
            random_state=random_state,
        )
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None
        self._fitted = False

    def fit(self, transactions):
        X = featurize(transactions)
        self.model.fit(X, X)  # learn to reconstruct input
        recon = self.model.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1)
        self.threshold_ = np.percentile(errors, self.threshold_percentile)
        self._fitted = True
        return self

    def score_batch(self, transactions) -> list:
        assert self._fitted, "call fit() before scoring"
        X = featurize(transactions)
        recon = self.model.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1)
        return [1 if e > self.threshold_ else 0 for e in errors]
