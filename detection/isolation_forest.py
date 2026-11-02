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

    n_estimators: int = 300
    max_samples: int | float | str = 256
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
            raise ValueError("n_estimators must be a positive integer.")

        if isinstance(self.max_samples, str):
            if self.max_samples != "auto":
                raise ValueError(
                    "max_samples must be 'auto', a positive int, or a float in (0, 1]."
                )
        elif isinstance(self.max_samples, bool):
            raise ValueError("max_samples must not be a boolean.")
        elif isinstance(self.max_samples, int):
            if self.max_samples < 1:
                raise ValueError("Integer max_samples must be positive.")
        elif isinstance(self.max_samples, float):
            if not np.isfinite(self.max_samples) or not 0.0 < self.max_samples <= 1.0:
                raise ValueError("Float max_samples must be in (0, 1].")
        else:
            raise ValueError("Invalid max_samples type.")

        if self.contamination != "auto":
            if (
                isinstance(self.contamination, bool)
                or not isinstance(self.contamination, (int, float))
                or not np.isfinite(self.contamination)
                or not 0.0 < self.contamination <= 0.5
            ):
                raise ValueError("contamination must be 'auto' or a float in (0, 0.5].")

        if (
            isinstance(self.max_features, bool)
            or not isinstance(self.max_features, (int, float))
            or not np.isfinite(self.max_features)
            or not 0.0 < self.max_features <= 1.0
        ):
            raise ValueError("max_features must be in (0, 1].")

        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise ValueError("random_state must be an integer.")

        if self.n_jobs is not None:
            if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int):
                raise ValueError("n_jobs must be an integer or None.")


def _validate_features(X: Any, name: str) -> Any:
    """Validate a dense or sparse numerical feature matrix."""
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D matrix.")

        values = X.data
    else:
        try:
            X = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numerical features.") from exc

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 2D matrix.")

        values = X

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return X


class IsolationForestDetector:
    """Isolation Forest wrapper operating on preprocessed numerical features.

    Feature engineering and preprocessing are intentionally handled outside
    this class. This keeps the detector reusable across feature sets and
    compatible with the future Autoencoder detector interface.
    """

    ARTIFACT_VERSION = 1

    def __init__(
        self,
        config: IsolationForestConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(config, IsolationForestConfig):
            raise TypeError("config must be an IsolationForestConfig instance.")

        self.config = config or IsolationForestConfig()
        self.model: IsolationForest | None = None
        self.threshold: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X_train: Any) -> IsolationForestDetector:
        """Fit the detector on a non-empty matrix of training transactions."""
        X_train = _validate_features(X_train, "X_train")

        if X_train.shape[0] < 2:
            raise ValueError("At least two training rows are required.")

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(X_train)
        self.n_features_in_ = int(self.model.n_features_in_)

        return self

    def score_samples(self, X: Any) -> np.ndarray:
        """Return anomaly scores where larger values indicate greater risk."""
        if self.model is None or self.n_features_in_ is None:
            raise RuntimeError("The detector has not been fitted.")

        X = _validate_features(X, "X")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        # scikit-learn assigns lower scores to more anomalous observations.
        # Negation makes larger scores correspond to greater anomaly risk.
        scores = -self.model.score_samples(X)

        if not np.isfinite(scores).all():
            raise RuntimeError("The model produced non-finite anomaly scores.")

        return np.asarray(scores, dtype=np.float64)

    @staticmethod
    def _validate_threshold(threshold: Any) -> float:
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)
        ):
            raise ValueError("threshold must be a finite number.")

        return float(threshold)

    def set_threshold(self, threshold: float) -> None:
        """Set the binary classification threshold selected on validation data."""
        self.threshold = self._validate_threshold(threshold)

    def predict(
        self,
        X: Any,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return 1 for flagged transactions and 0 otherwise."""
        active_threshold = self.threshold if threshold is None else threshold

        if active_threshold is None:
            raise RuntimeError(
                "No threshold configured. Select one on validation data first."
            )

        active_threshold = self._validate_threshold(active_threshold)
        scores = self.score_samples(X)

        return (scores >= active_threshold).astype(np.int8)

    def save(self, path: str | Path) -> None:
        """Save the fitted detector (model, detector config, and threshold).

        This is a low-level detector artifact. The end-to-end inference bundle
        additionally needs its fitted feature preprocessor and feature columns;
        those are packaged by the training pipeline, not by this method.
        """
        if self.model is None or self.n_features_in_ is None:
            raise RuntimeError("Cannot save an unfitted detector.")

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
    def load(cls, path: str | Path) -> IsolationForestDetector:
        """Load a low-level detector artifact produced by ``save``."""
        artifact = joblib.load(path)

        if not isinstance(artifact, dict):
            raise TypeError("Isolation Forest artifact must be a dictionary.")

        required = {
            "artifact_version",
            "config",
            "model",
            "threshold",
            "n_features_in",
        }
        missing = required - set(artifact)
        if missing:
            raise ValueError(
                f"Isolation Forest artifact is missing keys: {sorted(missing)}"
            )

        if artifact["artifact_version"] != cls.ARTIFACT_VERSION:
            raise ValueError("Unsupported Isolation Forest artifact version.")

        model = artifact["model"]
        if not isinstance(model, IsolationForest):
            raise TypeError("Artifact does not contain a scikit-learn IsolationForest.")

        model_feature_count = getattr(model, "n_features_in_", None)
        if model_feature_count is None:
            raise ValueError("Saved Isolation Forest model is not fitted.")

        stored_feature_count = artifact["n_features_in"]
        if int(stored_feature_count) != int(model_feature_count):
            raise ValueError("Saved feature count does not match the fitted model.")

        detector = cls(IsolationForestConfig(**artifact["config"]))
        detector.model = model
        detector.n_features_in_ = int(model_feature_count)

        if artifact["threshold"] is not None:
            detector.set_threshold(artifact["threshold"])

        return detector
