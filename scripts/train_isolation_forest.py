"""Train and evaluate MARLOT's Isolation Forest baseline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn

from detection.evaluation import (
    evaluate_by_fraud_type,
    evaluate_scores,
    select_f1_threshold,
)
from detection.isolation_forest import (
    IsolationForestConfig,
    IsolationForestDetector,
)

DEFAULT_PROCESSED_DIR = Path("data/detection/processed")
DEFAULT_OUTPUT_DIR = Path("artifacts/detection/isolation_forest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    return parser.parse_args(argv)


def load_split(path: Path) -> dict[str, np.ndarray]:
    """Load and validate one processed dataset split."""

    if not path.is_file():
        raise FileNotFoundError(f"Processed split not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "transaction_id", "fraud_type"}
        missing = required - set(data.files)

        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")

        result = {name: data[name].copy() for name in required}

    X = result["X"]
    y = result["y"]
    ids = result["transaction_id"]
    fraud_types = result["fraud_type"]

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"{path}: X must be a non-empty 2D matrix")

    n = X.shape[0]

    if not all(len(result[key]) == n for key in ("y", "transaction_id", "fraud_type")):
        raise ValueError(f"{path}: arrays are not row-aligned")

    if not np.isfinite(X).all():
        raise ValueError(f"{path}: feature matrix contains non-finite values")

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")

    if len(set(ids.tolist())) != n:
        raise ValueError(f"{path}: duplicate transaction IDs")

    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> dict:
    processed_dir: Path = args.processed_dir
    output_dir: Path = args.output_dir

    train = load_split(processed_dir / "train.npz")
    validation = load_split(processed_dir / "validation.npz")
    test = load_split(processed_dir / "test.npz")

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = validation["X"], validation["y"]
    X_test, y_test = test["X"], test["y"]

    # This is a one-class, normal-only training experiment.
    if np.any(y_train != 0):
        raise ValueError(
            "Training split contains fraud labels. "
            "This baseline expects legitimate-only training data."
        )

    feature_counts = {
        X_train.shape[1],
        X_val.shape[1],
        X_test.shape[1],
    }
    if len(feature_counts) != 1:
        raise ValueError("Feature counts differ between dataset splits")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = IsolationForestConfig(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        contamination="auto",
    )

    detector = IsolationForestDetector(config)
    detector.fit(X_train)

    # Validation data selects the threshold.
    val_scores = detector.score_samples(X_val)
    threshold, threshold_diagnostics = select_f1_threshold(
        y_val,
        val_scores,
    )
    detector.set_threshold(threshold)

    validation_metrics = evaluate_scores(
        y_val,
        val_scores,
        threshold,
    )

    # The test set is evaluated once using the frozen threshold.
    test_scores = detector.score_samples(X_test)
    test_predictions = detector.predict(X_test)

    test_metrics = evaluate_scores(
        y_test,
        test_scores,
        threshold,
    )
    test_metrics["fraud_type_metrics"] = evaluate_by_fraud_type(
        y_test,
        test["fraud_type"],
        test_predictions,
    )

    detector.save(output_dir / "model.joblib")

    preprocessor_path = processed_dir / "preprocessor.joblib"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(
            "Fitted Step 12 preprocessor not found: " f"{preprocessor_path}"
        )

    # Copy the exact fitted preprocessor used for these matrices.
    shutil.copy2(
        preprocessor_path,
        output_dir / "preprocessor.joblib",
    )

    metrics = {
        "model": "IsolationForest",
        "configuration": {
            "n_estimators": args.n_estimators,
            "random_state": args.seed,
            "n_jobs": args.n_jobs,
            "contamination": "auto",
        },
        "feature_count": int(X_train.shape[1]),
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
            "test": {
                "count": int(len(y_test)),
                "fraud_count": int(y_test.sum()),
                "fraud_prevalence": float(y_test.mean()),
            },
        },
        "threshold_selection": {
            "method": "maximum_validation_f1",
            "threshold": threshold,
            **threshold_diagnostics,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "environment": {
            "scikit_learn": sklearn.__version__,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    write_json(output_dir / "metrics.json", metrics)

    predictions_path = output_dir / "predictions_test.csv"
    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "transaction_id",
                "fraud_label",
                "fraud_type",
                "risk_score",
                "predicted_label",
            ],
        )
        writer.writeheader()

        for i in range(len(y_test)):
            writer.writerow(
                {
                    "transaction_id": str(test["transaction_id"][i]),
                    "fraud_label": int(y_test[i]),
                    "fraud_type": str(test["fraud_type"][i]),
                    "risk_score": float(test_scores[i]),
                    "predicted_label": int(test_predictions[i]),
                }
            )

    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_experiment(args)

    print("\n=== Isolation Forest Baseline ===")
    print(f"Features: {metrics['feature_count']}")
    print(f"Threshold: {metrics['threshold_selection']['threshold']:.6f}")

    for split in ("validation_metrics", "test_metrics"):
        report = metrics[split]
        print(f"\n{split.replace('_', ' ').title()}")
        for name in (
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1",
        ):
            value = report[name]
            display = "N/A" if value is None else f"{value:.4f}"
            print(f"  {name:20s} {display}")

        print("  Confusion matrix:", report["confusion_matrix"])

    print(f"\nArtifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
