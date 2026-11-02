from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from detection.evaluation import (
    evaluate_by_fraud_type,
    evaluate_scores,
    select_coverage_threshold,
)
from detection.features import (
    E_RATIO_CATEGORY_FEATURES,
    HistoricalFeatureBuilder,
    extract_labels,
    extract_metadata,
    fit_preprocessor,
    transform_features,
)
from detection.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)

DEFAULT_DATA_DIR = Path("data/detection")
DEFAULT_OUTPUT_DIR = Path("artifacts/detection/isolation_forest")

FEATURE_SET_NAME = "E_ratio_category"
MAXIMUM_FALSE_POSITIVE_RATE = 0.15
MINIMUM_FRAUD_TYPE_RECALL = 0.50

# Reference from the previously selected coverage-oriented experiment.
# This is a reproduction diagnostic only; it does not force the threshold.
REFERENCE_THRESHOLD = 0.5102980710521986


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected coverage-oriented Isolation Forest using "
            "raw train and validation JSONL files. The test split is never read."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.jsonl and validation.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which the candidate artifact and report are saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output files in a non-empty output directory.",
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL transaction split and validate its record structure."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

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
        raise ValueError(f"Dataset is empty: {path}")
    return records


def prepare_split(path: Path) -> dict[str, Any]:
    """Build chronological historical features, labels, and metadata."""
    records = load_jsonl(path)
    raw_features = HistoricalFeatureBuilder().transform(records)
    labels = extract_labels(records)
    metadata = extract_metadata(records)

    transaction_ids = metadata["transaction_id"].fillna("").astype(str).to_numpy()
    fraud_types = metadata["fraud_type"].fillna("").astype(str).to_numpy()

    row_count = len(raw_features)
    if not (len(labels) == len(transaction_ids) == len(fraud_types) == row_count):
        raise ValueError(f"Features, labels, and metadata are not aligned: {path}")
    if np.any(transaction_ids == ""):
        raise ValueError(f"Empty transaction ID found: {path}")
    if len(set(transaction_ids.tolist())) != row_count:
        raise ValueError(f"Duplicate transaction IDs found: {path}")
    if np.any((labels == 0) & (fraud_types != "")):
        raise ValueError(f"Legitimate records have a fraud_type: {path}")
    if np.any((labels == 1) & (fraud_types == "")):
        raise ValueError(f"Fraudulent records are missing fraud_type: {path}")

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
    """Add FPR and per-fraud-type recall to evaluate_scores output."""
    confusion = metrics["confusion_matrix"]
    tn = int(confusion["tn"])
    fp = int(confusion["fp"])
    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0

    per_type = evaluate_by_fraud_type(labels, fraud_types, predictions)
    if not per_type:
        raise ValueError("No fraud types were available for macro recall.")

    metrics["fraud_type_metrics"] = per_type
    metrics["macro_recall"] = float(
        np.mean([values["recall"] for values in per_type.values()])
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=4, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Train on legitimate-only train data and select threshold on validation."""
    train = prepare_split(args.data_dir / "train.jsonl")
    validation = prepare_split(args.data_dir / "validation.jsonl")

    X_train_raw = train["X_raw"]
    y_train = train["y"]
    X_val_raw = validation["X_raw"]
    y_val = validation["y"]

    if np.any(y_train != 0):
        raise ValueError(
            "Training split contains fraud labels. This baseline expects "
            "legitimate-only training data."
        )
    if np.unique(y_val).size != 2:
        raise ValueError(
            "Validation split must contain both legitimate and fraud rows."
        )

    # The selected feature list and its ordering come from features.py.
    feature_columns = tuple(E_RATIO_CATEGORY_FEATURES)

    print("\nFitting selected preprocessor on training data only...")
    preprocessor = fit_preprocessor(
        X_train_raw,
        feature_columns=feature_columns,
    )
    X_train = transform_features(
        X_train_raw,
        preprocessor,
        feature_columns=feature_columns,
    )
    X_val = transform_features(
        X_val_raw,
        preprocessor,
        feature_columns=feature_columns,
    )

    if X_train.shape[1] != 30 or X_val.shape[1] != 30:
        raise ValueError(
            "The selected feature pipeline should produce 30 processed "
            f"features; got train={X_train.shape[1]}, validation={X_val.shape[1]}."
        )

    config = IsolationForestConfig(
        n_estimators=300,
        max_samples=256,
        max_features=1.0,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining Isolation Forest...")
    detector = IsolationForestDetector(config)
    detector.fit(X_train)

    print("\nScoring validation data and selecting threshold...")
    validation_scores = detector.score_samples(X_val)
    threshold, threshold_diagnostics = select_coverage_threshold(
        y_validation=y_val,
        validation_scores=validation_scores,
        fraud_types=validation["fraud_type"],
        maximum_false_positive_rate=MAXIMUM_FALSE_POSITIVE_RATE,
        minimum_fraud_type_recall=MINIMUM_FRAUD_TYPE_RECALL,
    )
    detector.set_threshold(threshold)
    validation_predictions = detector.predict(X_val)

    validation_metrics = evaluate_scores(
        y_val,
        validation_scores,
        threshold,
    )
    add_classification_metrics(
        validation_metrics,
        y_val,
        validation["fraud_type"],
        validation_predictions,
    )

    artifact = {
        "artifact_type": "marlot_isolation_forest_bundle",
        "artifact_version": 1,
        "feature_set": FEATURE_SET_NAME,
        "model": detector.model,
        "preprocessor": preprocessor,
        "feature_columns": list(feature_columns),
        "threshold": float(threshold),
        "configuration": asdict(config),
        "threshold_selection": {
            **threshold_diagnostics,
            "threshold": float(threshold),
        },
        "validation_metrics": validation_metrics,
    }

    training_report = {
        "model": "IsolationForest",
        "feature_set": FEATURE_SET_NAME,
        "configuration": asdict(config),
        "feature_columns": list(feature_columns),
        "raw_feature_count": len(feature_columns),
        "processed_feature_count": int(X_train.shape[1]),
        "datasets": {
            "train": {
                "count": int(len(y_train)),
                "fraud_count": int(y_train.sum()),
                "fraud_prevalence": float(y_train.mean()),
            },
            "validation": {
                "count": int(len(y_val)),
                "fraud_count": int(y_val.sum()),
                "fraud_prevalence": float(y_val.mean()),
            },
        },
        "threshold_selection": artifact["threshold_selection"],
        "validation_metrics": validation_metrics,
        "environment": {
            "scikit_learn": sklearn.__version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    # This script deliberately does not load, inspect, or evaluate test.jsonl.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "selected_model.joblib"
    report_path = args.output_dir / "training_metrics.json"

    joblib.dump(artifact, model_path)
    write_json(report_path, training_report)

    threshold_difference = abs(threshold - REFERENCE_THRESHOLD)
    print("\nTraining complete.")
    print(f"Train matrix:      {X_train.shape}")
    print(f"Validation matrix: {X_val.shape}")
    print(f"Selected threshold: {threshold:.12f}")
    print(f"Reference threshold: {REFERENCE_THRESHOLD:.12f}")
    print(f"Absolute difference: {threshold_difference:.12g}")
    print(f"Validation macro recall: {validation_metrics['macro_recall']:.4f}")
    print(f"Validation FPR: {validation_metrics['false_positive_rate']:.4f}")
    print(f"\nModel artifact: {model_path}")
    print(f"Training report: {report_path}")

    return training_report


def main() -> None:
    args = parse_args()

    output_files = [
        args.output_dir / "selected_model.joblib",
        args.output_dir / "training_metrics.json",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        existing_text = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Training output already exists; refusing to overwrite:\n"
            f"{existing_text}\nPass --overwrite to replace it."
        )

    run_training(args)


if __name__ == "__main__":
    main()
