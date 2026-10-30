"""Evaluation utilities for MARLOT anomaly detectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_binary_labels(y: Any, name: str) -> np.ndarray:
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if y.size == 0:
        raise ValueError(f"{name} must not be empty")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary labels 0 and 1")

    return y.astype(np.int8)


def _validate_scores(scores: Any, expected_size: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1 or scores.size != expected_size:
        raise ValueError("Scores must be 1D and aligned with the labels")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values")

    return scores


def select_f1_threshold(
    y_validation: Any,
    validation_scores: Any,
) -> tuple[float, dict[str, float]]:
    """Choose the threshold maximizing validation F1.

    Ties are resolved in favor of the highest threshold, which avoids
    choosing a more permissive threshold when F1 is identical.
    """

    y = _validate_binary_labels(y_validation, "y_validation")
    scores = _validate_scores(validation_scores, len(y))

    if np.unique(y).size != 2:
        raise ValueError(
            "Threshold selection requires both classes in validation data"
        )

    precision, recall, thresholds = precision_recall_curve(y, scores)

    # precision and recall each have one extra terminal element.
    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall
    f1_values = np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_f1 = float(f1_values.max())
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = int(candidates[-1])

    threshold = float(thresholds[best_index])

    return threshold, {
        "validation_f1": best_f1,
        "validation_precision": float(precision[best_index]),
        "validation_recall": float(recall[best_index]),
    }


def evaluate_scores(
    y_true: Any,
    scores: Any,
    threshold: float,
) -> dict[str, Any]:
    """Calculate ranking metrics and fixed-threshold classification metrics."""

    y = _validate_binary_labels(y_true, "y_true")
    risk_scores = _validate_scores(scores, len(y))

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    predictions = (risk_scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    has_both_classes = np.unique(y).size == 2

    return {
        "count": int(len(y)),
        "fraud_prevalence": float(y.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "threshold": float(threshold),
        "roc_auc": (
            float(roc_auc_score(y, risk_scores))
            if has_both_classes
            else None
        ),
        "average_precision": (
            float(average_precision_score(y, risk_scores))
            if has_both_classes
            else None
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_by_fraud_type(
    y_true: Any,
    fraud_types: Any,
    predictions: Any,
) -> dict[str, dict[str, float | int]]:
    """Calculate recall independently for each ground-truth fraud type."""

    y = _validate_binary_labels(y_true, "y_true")
    types = np.asarray(fraud_types, dtype=str)
    preds = np.asarray(predictions)

    if types.ndim != 1 or len(types) != len(y):
        raise ValueError("fraud_types must align with labels")

    if preds.ndim != 1 or len(preds) != len(y):
        raise ValueError("predictions must align with labels")

    if not np.isin(preds, [0, 1]).all():
        raise ValueError("predictions must contain only 0 and 1")

    results: dict[str, dict[str, float | int]] = {}

    # Fraud type is metadata for evaluation, never a model input.
    fraud_mask = y == 1

    for fraud_type in sorted(set(types[fraud_mask])):
        if not fraud_type:
            continue

        type_mask = fraud_mask & (types == fraud_type)
        support = int(type_mask.sum())

        results[str(fraud_type)] = {
            "support": support,
            "recall": float(preds[type_mask].mean()),
        }

    return results
