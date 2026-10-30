"""Isolation Forest baseline for MARLOT fraud detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsolationForestConfig:
    """Configuration for the Isolation Forest detector."""

    n_estimators: int = 100
    max_samples: int | float | str = "auto"
    contamination: float | str = "auto"
    max_features: float = 1.0
    random_state: int = 42
    n_jobs: int | None = -1

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer")

        if self.contamination != "auto":
            if (
                isinstance(self.contamination, bool)
                or not isinstance(self.contamination, (int, float))
                or not 0.0 < self.contamination <= 0.5
            ):
                raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")

        if (
            isinstance(self.max_features, bool)
            or not isinstance(self.max_features, (int, float))
            or not 0.0 < self.max_features <= 1.0
        ):
            raise ValueError("max_features must be in (0, 1]")

        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise ValueError("random_state must be an integer")


def _validate_features(X: Any, name: str) -> Any:
    """Validate a dense or sparse numerical feature matrix."""

    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D matrix")

        values = X.data
    else:
        try:
            X = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numerical features") from exc

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D matrix")

        values = X

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return X


class IsolationForestDetector:
    """Fit an Isolation Forest and expose higher-is-riskier scores."""

    ARTIFACT_VERSION = 1

    def __init__(
        self,
        config: IsolationForestConfig | None = None,
    ) -> None:
        self.config = config or IsolationForestConfig()
        self.model: IsolationForest | None = None
        self.threshold: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X_train: Any) -> "IsolationForestDetector":
        """Fit the detector without accessing target labels."""

        X_train = _validate_features(X_train, "X_train")

        if X_train.shape[0] < 2:
            raise ValueError("At least two training rows are required")

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        self.model.fit(X_train)
        self.n_features_in_ = X_train.shape[1]

        return self

    def score_samples(self, X: Any) -> np.ndarray:
        """Return one anomaly score per row; higher means more anomalous."""

        if self.model is None:
            raise RuntimeError("The detector has not been fitted")

        X = _validate_features(X, "X")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        scores = -self.model.score_samples(X)

        if not np.isfinite(scores).all():
            raise RuntimeError("The model produced non-finite anomaly scores")

        return scores

    def set_threshold(self, threshold: float) -> None:
        """Set the operational threshold selected on validation data."""

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
        ):
            raise ValueError("threshold must be a finite number")

        self.threshold = float(threshold)

    def predict(
        self,
        X: Any,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return 1 for flagged transactions, 0 otherwise."""

        active_threshold = threshold if threshold is not None else self.threshold

        if active_threshold is None:
            raise RuntimeError(
                "No threshold configured. Select one on validation data first."
            )

        if not np.isfinite(active_threshold):
            raise ValueError("threshold must be finite")

        scores = self.score_samples(X)
        return (scores >= active_threshold).astype(np.int8)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model and threshold."""

        if self.model is None:
            raise RuntimeError("Cannot save an unfitted detector")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "artifact_version": self.ARTIFACT_VERSION,
                "config": asdict(self.config),
                "model": self.model,
                "threshold": self.threshold,
                "n_features_in": self.n_features_in_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestDetector":
        """Load an artifact previously saved by this class."""

        # Only load artifacts from trusted sources: joblib files are
        # pickle-based and can execute code during deserialization.
        artifact = joblib.load(path)

        if artifact.get("artifact_version") != cls.ARTIFACT_VERSION:
            raise ValueError("Unsupported Isolation Forest artifact version")

        detector = cls(IsolationForestConfig(**artifact["config"]))
        detector.model = artifact["model"]
        detector.threshold = artifact["threshold"]
        detector.n_features_in_ = artifact["n_features_in"]

        return detector
