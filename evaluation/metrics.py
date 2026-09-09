"""
Evaluation metrics for detector comparison.
"""

import time
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score


def evaluate(y_true: list[int], y_pred: list[int], y_score: list[float] | None = None) -> dict:
    """Compute standard fraud-detection metrics.

    y_true:  ground truth fraud labels (0/1)
    y_pred:  detector's binary decisions (0/1)
    y_score: optional continuous risk score, needed for PR-AUC
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_score)

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return metrics


def time_detector(detector, transactions, fit_first: bool = False) -> dict:
    """Measure latency/throughput of a detector's score_batch call.
    If fit_first, fits the detector on the same data first (not timed)."""
    if fit_first and hasattr(detector, "fit"):
        detector.fit(transactions)

    start = time.perf_counter()
    preds = detector.score_batch(transactions)
    elapsed = time.perf_counter() - start

    n = len(transactions)
    return {
        "total_seconds": elapsed,
        "avg_latency_ms": (elapsed / n) * 1000 if n else 0.0,
        "throughput_tps": n / elapsed if elapsed > 0 else float("inf"),
        "predictions": preds,
    }
