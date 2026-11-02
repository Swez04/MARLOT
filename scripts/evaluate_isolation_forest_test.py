from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from detection.evaluation import (
    evaluate_by_fraud_type,
    evaluate_scores,
)
from detection.features import (
    HistoricalFeatureBuilder,
    extract_labels,
    extract_metadata,
    transform_features,
)

DEFAULT_TEST_DATA = Path("data/detection/test.jsonl")
DEFAULT_MODEL = Path(
    "artifacts/detection/isolation_forest/selected_model.joblib"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved Isolation Forest bundle once on the independent "
            "raw test JSONL split. The threshold is loaded from the artifact."
        )
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=DEFAULT_TEST_DATA,
        help="Path to the raw test.jsonl file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to selected_model.joblib produced by training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the model artifact directory.",
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}."
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}."
                )
            records.append(record)

    if not records:
        raise ValueError(f"Test dataset is empty: {path}")
    return records


from detection.features import E_RATIO_CATEGORY_FEATURES


def load_and_validate_artifact(path):
    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise TypeError("Expected a dictionary-based model artifact.")

    required = {
        "model",
        "preprocessor",
        "feature_columns",
        "threshold",
        "configuration",
        "threshold_selection",
        "validation_metrics",
    }

    missing = required - set(artifact)

    if missing:
        raise ValueError(f"Model artifact is missing required keys: {sorted(missing)}")

    # Backward compatibility for older experiment artifacts.
    if "feature_set" not in artifact:
        if tuple(artifact["feature_columns"]) == tuple(E_RATIO_CATEGORY_FEATURES):
            artifact["feature_set"] = "E_ratio_category"
        else:
            artifact["feature_set"] = "custom"

    # Validate selected feature columns against the fitted preprocessor.
    feature_columns = tuple(artifact["feature_columns"])

    preprocessor_columns = tuple(
        str(name) for name in artifact["preprocessor"].feature_names_in_
    )

    if feature_columns != preprocessor_columns:
        raise ValueError(
            "Artifact feature_columns do not match "
            "the saved preprocessor input columns."
        )

    if artifact["model"].n_features_in_ != len(
        artifact["preprocessor"].get_feature_names_out()
    ):
        raise ValueError("Model input dimension does not match preprocessor output.")

    return artifact


def prepare_test_split(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    raw_features = HistoricalFeatureBuilder().transform(records)
    labels = extract_labels(records)
    metadata = extract_metadata(records)

    transaction_ids = metadata["transaction_id"].fillna("").astype(str).to_numpy()
    fraud_types = metadata["fraud_type"].fillna("").astype(str).to_numpy()

    row_count = len(raw_features)
    if not (len(labels) == len(transaction_ids) == len(fraud_types) == row_count):
        raise ValueError("Test features, labels, and metadata are not aligned.")
    if np.any(transaction_ids == ""):
        raise ValueError("Test set contains an empty transaction ID.")
    if len(set(transaction_ids.tolist())) != row_count:
        raise ValueError("Test set contains duplicate transaction IDs.")
    if np.any((labels == 0) & (fraud_types != "")):
        raise ValueError("Legitimate test records have a non-empty fraud_type.")
    if np.any((labels == 1) & (fraud_types == "")):
        raise ValueError("Fraudulent test records are missing fraud_type.")

    return {
        "X_raw": raw_features,
        "y": labels,
        "transaction_id": transaction_ids,
        "fraud_type": fraud_types,
    }


def add_classification_metrics(
    metrics: dict[str, Any],
    labels: np.ndarray,
    fraud_types: np.ndarray,
    predictions: np.ndarray,
) -> None:
    confusion = metrics["confusion_matrix"]
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0

    per_type = evaluate_by_fraud_type(labels, fraud_types, predictions)
    metrics["fraud_type_metrics"] = per_type
    metrics["macro_recall"] = (
        float(np.mean([values["recall"] for values in per_type.values()]))
        if per_type
        else 0.0
    )


def write_predictions(
    path: Path,
    transaction_ids: np.ndarray,
    labels: np.ndarray,
    fraud_types: np.ndarray,
    risk_scores: np.ndarray,
    predictions: np.ndarray,
) -> None:
    fieldnames = [
        "transaction_id",
        "fraud_label",
        "fraud_type",
        "risk_score",
        "predicted_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(labels)):
            writer.writerow(
                {
                    "transaction_id": str(transaction_ids[index]),
                    "fraud_label": int(labels[index]),
                    "fraud_type": str(fraud_types[index]),
                    "risk_score": float(risk_scores[index]),
                    "predicted_label": int(predictions[index]),
                }
            )


def evaluate_test(args: argparse.Namespace) -> dict[str, Any]:
    print("Loading saved Isolation Forest bundle...")
    artifact = load_and_validate_artifact(args.model)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_columns = tuple(artifact["feature_columns"])
    threshold = float(artifact["threshold"])

    print(f"Feature set: {artifact['feature_set']}")
    print(f"Selected raw features: {len(feature_columns)}")
    print(f"Model input features: {model.n_features_in_}")
    print(f"Saved threshold: {threshold:.12f}")

    print("\nLoading and preparing raw test transactions...")
    test = prepare_test_split(args.test_data)

    X_test = transform_features(
        test["X_raw"],
        preprocessor,
        feature_columns=feature_columns,
    )

    if X_test.shape[1] != model.n_features_in_:
        raise ValueError(
            "Transformed test feature count does not match the model: "
            f"{X_test.shape[1]} != {model.n_features_in_}."
        )

    print(f"Test transactions: {len(test['y']):,}")
    print(f"Fraud transactions: {int(test['y'].sum()):,}")
    print(f"Fraud prevalence: {test['y'].mean():.4%}")
    print(f"Transformed test matrix: {X_test.shape}")

    # scikit-learn's score_samples assigns lower values to anomalies.
    # Negation keeps the project's convention: higher score = higher risk.
    risk_scores = -model.score_samples(X_test)
    if not np.isfinite(risk_scores).all():
        raise RuntimeError("Isolation Forest produced non-finite risk scores.")

    predictions = (risk_scores >= threshold).astype(np.int8)

    metrics = evaluate_scores(
        test["y"],
        risk_scores,
        threshold,
    )
    add_classification_metrics(
        metrics,
        test["y"],
        test["fraud_type"],
        predictions,
    )

    report = {
        "model": "IsolationForest",
        "feature_set": artifact["feature_set"],
        "configuration": artifact["configuration"],
        "threshold": threshold,
        "feature_columns": list(feature_columns),
        "raw_feature_count": len(feature_columns),
        "processed_feature_count": int(X_test.shape[1]),
        "test_dataset": {
            "path": str(args.test_data),
            "count": int(len(test["y"])),
            "fraud_count": int(test["y"].sum()),
            "fraud_prevalence": float(test["y"].mean()),
        },
        "test_metrics": metrics,
    }

    output_dir = args.output_dir or args.model.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"

    metrics_path.write_text(
        json.dumps(report, indent=4, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_predictions(
        predictions_path,
        test["transaction_id"],
        test["y"],
        test["fraud_type"],
        risk_scores,
        predictions,
    )

    print("\n=== ISOLATION FOREST: FINAL TEST EVALUATION ===")
    for name in (
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "macro_recall",
    ):
        value = metrics.get(name)
        print(f"{name:25s}: {'N/A' if value is None else f'{value:.4f}'}")

    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print("\nFraud-type recall:")
    for fraud_type, values in metrics["fraud_type_metrics"].items():
        print(
            f"  {fraud_type:20s}: {values['recall']:.4f} "
            f"({values['support']} transactions)"
        )

    print(f"\nMetrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    return report


def main() -> None:
    args = parse_args()
    evaluate_test(args)


if __name__ == "__main__":
    main()
