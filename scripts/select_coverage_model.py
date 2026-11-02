"""Select one Isolation Forest using coverage-oriented validation criteria."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TUNING_DIR = Path("artifacts/detection/isolation_forest_tuning")

OUTPUT_DIR = Path("artifacts/detection/coverage_selection")

METRICS_PATH = TUNING_DIR / "metrics.json"
PREDICTIONS_PATH = TUNING_DIR / "validation_predictions.csv"

MAX_FALSE_POSITIVE_RATE = 0.15
MIN_FRAUD_TYPE_RECALL = 0.50

EXPECTED_FRAUD_TYPES = (
    "amount_spike",
    "collusion",
    "geo_hop",
    "velocity",
)


# ---------------------------------------------------------
# Load and validate inputs
# ---------------------------------------------------------


def load_inputs() -> tuple[dict, pd.DataFrame]:
    """Load validation metrics and prediction scores."""

    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(PREDICTIONS_PATH)

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if report.get("evaluation_split") != "validation":
        raise ValueError("Expected metrics generated from the validation split.")

    if report.get("test_set_used") is not False:
        raise ValueError(
            "The metrics report does not confirm that the test set " "was excluded."
        )

    if not isinstance(report.get("experiments"), dict):
        raise ValueError("Missing experiments in metrics.json.")

    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "transaction_id",
        "fraud_label",
        "fraud_type",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Validation predictions are missing columns: {sorted(missing)}"
        )

    if predictions["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs in validation predictions.")

    y = predictions["fraud_label"].to_numpy()

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Fraud labels must be binary.")

    fraud_types = predictions["fraud_type"].fillna("").astype(str)

    if not np.array_equal(y == 1, fraud_types.to_numpy() != ""):
        raise ValueError("Fraud labels and fraud types are inconsistent.")

    observed_types = set(fraud_types[y == 1])

    if observed_types != set(EXPECTED_FRAUD_TYPES):
        raise ValueError(
            "Unexpected fraud types. "
            f"Expected {EXPECTED_FRAUD_TYPES}, got {sorted(observed_types)}"
        )

    return report, predictions


# ---------------------------------------------------------
# Efficient threshold search
# ---------------------------------------------------------


def evaluate_thresholds(
    y_true: np.ndarray,
    fraud_types: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """
    Find the threshold maximizing macro fraud recall
    subject to the maximum FPR constraint.

    Each unique anomaly score is evaluated as a threshold.

    Higher scores indicate greater anomaly risk.
    """

    if not (len(y_true) == len(fraud_types) == len(scores)):
        raise ValueError("Labels, fraud types, and scores must align.")

    if not np.isfinite(scores).all():
        raise ValueError("Risk scores contain non-finite values.")

    n = len(y_true)

    positive_count = int((y_true == 1).sum())
    negative_count = int((y_true == 0).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both validation classes are required.")

    # Sort scores in descending order.
    order = np.argsort(-scores, kind="mergesort")

    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    sorted_types = fraud_types[order]

    # Evaluate only at unique score boundaries.
    # All transactions with equal scores enter together.
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        n - 1,
    ]

    thresholds = sorted_scores[group_ends]

    # Cumulative true positives and false positives.
    cumulative_tp = np.cumsum(sorted_labels == 1)[group_ends]
    cumulative_fp = np.cumsum(sorted_labels == 0)[group_ends]

    overall_recall = cumulative_tp / positive_count
    false_positive_rate = cumulative_fp / negative_count

    # Calculate individual fraud-type recall at every threshold.
    type_recalls = {}

    for fraud_type in EXPECTED_FRAUD_TYPES:

        type_support = int(np.sum((y_true == 1) & (fraud_types == fraud_type)))

        if type_support == 0:
            raise ValueError(f"No validation transactions for {fraud_type}.")

        cumulative_type_tp = np.cumsum(
            (sorted_labels == 1) & (sorted_types == fraud_type)
        )[group_ends]

        type_recalls[fraud_type] = cumulative_type_tp / type_support

    # Equal importance for every fraud category.
    macro_recall = np.mean(
        np.column_stack(list(type_recalls.values())),
        axis=1,
    )

    minimum_type_recall = np.min(
        np.column_stack(list(type_recalls.values())),
        axis=1,
    )

    precision = np.divide(
        cumulative_tp,
        cumulative_tp + cumulative_fp,
        out=np.zeros(len(thresholds), dtype=float),
        where=(cumulative_tp + cumulative_fp) > 0,
    )

    # First enforce the false-positive constraint.
    fpr_eligible = np.flatnonzero(
        false_positive_rate <= MAX_FALSE_POSITIVE_RATE + 1e-12
    )

    if len(fpr_eligible) == 0:
        raise RuntimeError("No threshold satisfies the FPR constraint.")

    # Then maximize macro recall.
    # Tie-breakers:
    # 1. Higher overall recall.
    # 2. Lower false-positive rate.
    # 3. Higher threshold (more conservative).
    ranking_order = np.lexsort(
        (
            -thresholds[fpr_eligible],
            false_positive_rate[fpr_eligible],
            -overall_recall[fpr_eligible],
            -macro_recall[fpr_eligible],
        )
    )

    best_index = int(fpr_eligible[ranking_order[0]])

    threshold = float(thresholds[best_index])

    selected_tp = int(cumulative_tp[best_index])
    selected_fp = int(cumulative_fp[best_index])

    selected_fn = positive_count - selected_tp
    selected_tn = negative_count - selected_fp

    selected_precision = float(precision[best_index])
    selected_recall = float(overall_recall[best_index])

    f1 = (
        2
        * selected_precision
        * selected_recall
        / (selected_precision + selected_recall)
        if selected_precision + selected_recall > 0
        else 0.0
    )

    selected_type_recalls = {
        fraud_type: float(values[best_index])
        for fraud_type, values in type_recalls.items()
    }

    selected_fpr = float(false_positive_rate[best_index])

    selected_macro_recall = float(macro_recall[best_index])

    selected_min_recall = float(minimum_type_recall[best_index])

    feasible = (
        selected_fpr <= MAX_FALSE_POSITIVE_RATE + 1e-12
        and selected_min_recall >= MIN_FRAUD_TYPE_RECALL
    )

    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "macro_recall": selected_macro_recall,
        "minimum_fraud_type_recall": selected_min_recall,
        "overall_recall": selected_recall,
        "precision": selected_precision,
        "f1": float(f1),
        "false_positive_rate": selected_fpr,
        "predicted_positive_rate": float((selected_tp + selected_fp) / n),
        "confusion_matrix": {
            "tn": selected_tn,
            "fp": selected_fp,
            "fn": selected_fn,
            "tp": selected_tp,
        },
        "fraud_type_recall": selected_type_recalls,
        "feasible": bool(feasible),
    }


# ---------------------------------------------------------
# Evaluate every experiment
# ---------------------------------------------------------


def evaluate_all_experiments(
    report: dict,
    predictions: pd.DataFrame,
) -> list[dict]:
    """Evaluate all saved experiments under the coverage policy."""

    y_true = predictions["fraud_label"].to_numpy(dtype=np.int8)

    fraud_types = predictions["fraud_type"].fillna("").astype(str).to_numpy()

    experiments = report["experiments"]

    results = []

    for experiment_name, experiment_metrics in experiments.items():

        score_column = f"{experiment_name}_risk_score"

        if score_column not in predictions.columns:
            raise ValueError(f"Missing risk score column: {score_column}")

        scores = predictions[score_column].to_numpy(dtype=float)

        metrics = evaluate_thresholds(
            y_true,
            fraud_types,
            scores,
        )

        config = experiment_metrics["configuration"]

        result = {
            "experiment": experiment_name,
            "feature_set": experiment_metrics["feature_set"],
            "configuration": config,
            "raw_feature_count": experiment_metrics["raw_feature_count"],
            "processed_feature_count": experiment_metrics["processed_feature_count"],
            **metrics,
        }

        results.append(result)

        print(
            f"{experiment_name}: "
            f"macro_recall={metrics['macro_recall']:.4f}, "
            f"overall_recall={metrics['overall_recall']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}, "
            f"minimum_type_recall="
            f"{metrics['minimum_fraud_type_recall']:.4f}, "
            f"feasible={metrics['feasible']}"
        )

    return results


# ---------------------------------------------------------
# Rank candidates
# ---------------------------------------------------------


def rank_candidates(results: list[dict]) -> list[dict]:
    """Rank feasible configurations ahead of infeasible ones."""

    return sorted(
        results,
        key=lambda result: (
            result["feasible"],
            result["macro_recall"],
            result["overall_recall"],
            result["average_precision"],
            -result["false_positive_rate"],
        ),
        reverse=True,
    )


# ---------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------


def save_results(
    ranked_results: list[dict],
) -> None:
    """Save the candidate table and selected model artifact."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary CSV.
    rows = []

    for result in ranked_results:

        row = {
            "experiment": result["experiment"],
            "feature_set": result["feature_set"],
            "n_estimators": result["configuration"]["n_estimators"],
            "max_samples": result["configuration"]["max_samples"],
            "max_features": result["configuration"]["max_features"],
            "threshold": result["threshold"],
            "roc_auc": result["roc_auc"],
            "average_precision": result["average_precision"],
            "macro_recall": result["macro_recall"],
            "minimum_fraud_type_recall": (result["minimum_fraud_type_recall"]),
            "overall_recall": result["overall_recall"],
            "precision": result["precision"],
            "f1": result["f1"],
            "false_positive_rate": result["false_positive_rate"],
            "feasible": result["feasible"],
        }

        for fraud_type, recall in result["fraud_type_recall"].items():
            row[f"recall_{fraud_type}"] = recall

        rows.append(row)

    candidates_df = pd.DataFrame(rows)

    candidates_path = OUTPUT_DIR / "coverage_candidates.csv"
    candidates_df.to_csv(candidates_path, index=False)

    feasible_results = [result for result in ranked_results if result["feasible"]]

    if not feasible_results:
        report = {
            "selection_status": "no_feasible_candidate",
            "constraints": {
                "maximum_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
                "minimum_fraud_type_recall": MIN_FRAUD_TYPE_RECALL,
            },
            "candidates_evaluated": len(ranked_results),
        }

        selection_path = OUTPUT_DIR / "coverage_selection.json"

        with selection_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)

        print("\nNo configuration satisfies all coverage constraints.")
        print("See:", candidates_path)
        return

    selected = feasible_results[0]

    # Preserve the original tuned artifact.
    source_artifact_path = TUNING_DIR / f"{selected['experiment']}.joblib"

    if not source_artifact_path.exists():
        raise FileNotFoundError(source_artifact_path)

    artifact = joblib.load(source_artifact_path)

    # Ensure that the selected model matches the selected configuration.
    if artifact.get("feature_columns") != selected.get(
        "feature_columns", artifact.get("feature_columns")
    ):
        raise ValueError("Selected artifact feature columns are inconsistent.")

    # Apply the coverage-oriented threshold to a copy of the artifact.
    artifact["threshold"] = selected["threshold"]

    artifact["threshold_selection"] = {
        "method": "maximum_macro_recall_subject_to_constraints",
        "threshold": selected["threshold"],
        "maximum_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        "minimum_fraud_type_recall": MIN_FRAUD_TYPE_RECALL,
    }

    artifact["validation_metrics"] = {
        key: selected[key]
        for key in (
            "roc_auc",
            "average_precision",
            "macro_recall",
            "minimum_fraud_type_recall",
            "overall_recall",
            "precision",
            "f1",
            "false_positive_rate",
            "predicted_positive_rate",
            "confusion_matrix",
            "fraud_type_recall",
        )
    }

    selected_artifact_path = OUTPUT_DIR / "selected_coverage_isolation_forest.joblib"

    joblib.dump(artifact, selected_artifact_path)

    # JSON selection report.
    selection_report = {
        "selection_status": "selected",
        "selection_objective": "maximize_macro_fraud_recall",
        "constraints": {
            "maximum_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
            "minimum_fraud_type_recall": MIN_FRAUD_TYPE_RECALL,
        },
        "selected_experiment": selected["experiment"],
        "feature_set": selected["feature_set"],
        "configuration": selected["configuration"],
        "threshold": selected["threshold"],
        "validation_metrics": artifact["validation_metrics"],
        "selected_artifact": str(selected_artifact_path),
        "candidates_evaluated": len(ranked_results),
        "feasible_candidates": len(feasible_results),
    }

    selection_path = OUTPUT_DIR / "coverage_selection.json"

    with selection_path.open("w", encoding="utf-8") as file:
        json.dump(selection_report, file, indent=2, allow_nan=False)

    # Console summary.
    print("\n" + "=" * 65)
    print("COVERAGE-ORIENTED MODEL SELECTION")
    print("=" * 65)

    print("Selected experiment:", selected["experiment"])
    print("Feature set:", selected["feature_set"])
    print("Configuration:", selected["configuration"])
    print("Threshold:", f"{selected['threshold']:.6f}")

    print("\nValidation results")
    print("  ROC-AUC:", f"{selected['roc_auc']:.4f}")
    print("  Average precision:", f"{selected['average_precision']:.4f}")
    print("  Macro recall:", f"{selected['macro_recall']:.4f}")
    print("  Overall recall:", f"{selected['overall_recall']:.4f}")
    print("  Precision:", f"{selected['precision']:.4f}")
    print("  F1:", f"{selected['f1']:.4f}")
    print("  False-positive rate:", f"{selected['false_positive_rate']:.4f}")

    print("\nFraud-type recall")

    for fraud_type, recall in selected["fraud_type_recall"].items():
        print(f"  {fraud_type:15s} {recall:.4f}")

    print("\nSaved outputs:")
    print("  Candidate ranking:", candidates_path)
    print("  Selection report:", selection_path)
    print("  Selected model:", selected_artifact_path)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:

    print("Loading validation predictions and experiment metrics...")

    report, predictions = load_inputs()

    print("Experiments found:", len(report["experiments"]))
    print("Validation rows:", len(predictions))

    print("\nEvaluating coverage-oriented thresholds...")

    results = evaluate_all_experiments(
        report,
        predictions,
    )

    ranked_results = rank_candidates(results)

    save_results(ranked_results)


if __name__ == "__main__":
    main()
