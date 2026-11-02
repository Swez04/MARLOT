"""Compare Isolation Forest performance across feature ablation experiments."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from detection.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    prepare_detection_dataset,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_DIR = Path("data/detection")

OUTPUT_DIR = Path("artifacts/detection/isolation_forest_tuning")

TRAIN_PATH = DATA_DIR / "train.jsonl"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"

# Parameters that remain fixed across all experiments.
MODEL_BASE_CONFIG = {
    "contamination": "auto",
    "random_state": 42,
    "n_jobs": -1,
}

# Parameters to tune.
HYPERPARAMETER_GRID = {
    "n_estimators": [100, 300, 500],
    "max_samples": [256, 512, 1024],
    "max_features": [0.75, 1.0],
}


# ---------------------------------------------------------
# Feature groups
# ---------------------------------------------------------

# Original 17 raw features from before the feature improvements.
BASE_FEATURES = (
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "account_txn_count_60s",
    "account_txn_count_5m",
    "seconds_since_prev_account_txn",
    "has_prev_account_txn",
    "log_amount_deviation",
    "has_account_amount_history",
    "is_new_device_for_account",
    "location_changed",
    "merchant_txn_count_5m",
    "merchant_distinct_accounts_5m",
    "merchant_category",
    "location",
)

# Five new amount-related features.
AMOUNT_FEATURES = (
    "account_amount_ratio",
    "account_amount_robust_zscore",
    "has_reliable_account_amount_history",
    "category_amount_deviation",
    "has_category_amount_history",
)

# Four new geographical features.
GEOGRAPHY_FEATURES = (
    "geo_distance_km",
    "geo_travel_speed_kmph",
    "impossible_travel_flag",
    "has_geo_coordinates",
)

# Four controlled experiments.

FEATURE_SETS = {
    "A_original": BASE_FEATURES,
    "E_ratio_category": (
        BASE_FEATURES
        + (
            "account_amount_ratio",
            "category_amount_deviation",
        )
    ),
}


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load transaction records from a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc

    if not records:
        raise ValueError(f"Dataset is empty: {path}")

    return records


def validate_feature_sets() -> None:
    """Check that all experiment columns exist and are unique."""

    available = set(FEATURE_COLUMNS)

    for experiment_name, columns in FEATURE_SETS.items():

        if len(columns) != len(set(columns)):
            raise ValueError(f"{experiment_name} contains duplicate feature columns.")

        missing = set(columns) - available

        if missing:
            raise ValueError(
                f"{experiment_name} references unknown features: " f"{sorted(missing)}"
            )


# ---------------------------------------------------------
# Experiment-specific preprocessing
# ---------------------------------------------------------


def create_subset_preprocessor(
    feature_columns: tuple[str, ...],
) -> ColumnTransformer:
    """Create a fresh preprocessor for an experiment's feature subset."""

    numeric_columns = [
        column for column in feature_columns if column in NUMERIC_FEATURES
    ]

    categorical_columns = [
        column for column in feature_columns if column in CATEGORICAL_FEATURES
    ]

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_columns,
            )
        )

    if categorical_columns:

        # Compatibility with older and newer scikit-learn versions.
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        transformers.append(
            (
                "categorical",
                encoder,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_transform_subset(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Fit a new preprocessor on training features only.

    The fitted preprocessor is reused for validation features.
    """

    train_subset = X_train.loc[:, feature_columns]
    validation_subset = X_validation.loc[:, feature_columns]

    preprocessor = create_subset_preprocessor(feature_columns)

    preprocessor.fit(train_subset)

    X_train_processed = preprocessor.transform(train_subset)
    X_validation_processed = preprocessor.transform(validation_subset)

    if sparse.issparse(X_train_processed):
        X_train_processed = X_train_processed.toarray()

    if sparse.issparse(X_validation_processed):
        X_validation_processed = X_validation_processed.toarray()

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_validation_processed = np.asarray(
        X_validation_processed,
        dtype=np.float32,
    )

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Training features contain non-finite values.")

    if not np.isfinite(X_validation_processed).all():
        raise ValueError("Validation features contain non-finite values.")

    return (
        X_train_processed,
        X_validation_processed,
        preprocessor,
    )


# ---------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------


def select_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """
    Select the threshold that maximizes validation F1.

    Higher scores represent greater anomaly risk.

    In case of an F1 tie, choose the highest threshold.
    """

    if len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires both validation classes.")

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    if thresholds.size == 0:
        raise ValueError("No classification thresholds were generated.")

    precision = precision[:-1]
    recall = recall[:-1]

    denominators = precision + recall

    f1_values = np.divide(
        2.0 * precision * recall,
        denominators,
        out=np.zeros_like(precision),
        where=denominators > 0,
    )

    best_f1 = float(np.max(f1_values))

    best_indices = np.flatnonzero(np.isclose(f1_values, best_f1))

    # Choose the most conservative threshold among F1 ties.
    best_index = best_indices[np.argmax(thresholds[best_indices])]

    return float(thresholds[best_index]), best_f1


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------


def evaluate_fraud_types(
    y_true: np.ndarray,
    fraud_types: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate anomaly detection separately for each fraud type."""

    predictions = (scores >= threshold).astype(np.int8)

    results = {}

    fraud_names = sorted(name for name in np.unique(fraud_types) if name)

    for fraud_type in fraud_names:

        positive_mask = fraud_types == fraud_type

        # Treat this fraud type as positive and legitimate transactions
        # as negative. Exclude other fraud types from this ranking metric.
        ranking_mask = (y_true == 0) | positive_mask

        type_y = positive_mask[ranking_mask].astype(np.int8)
        type_scores = scores[ranking_mask]

        type_recall = recall_score(
            positive_mask.astype(np.int8),
            predictions,
            zero_division=0,
        )

        results[fraud_type] = {
            "support": int(positive_mask.sum()),
            "recall_at_global_threshold": float(type_recall),
            "roc_auc": float(roc_auc_score(type_y, type_scores)),
            "average_precision": float(average_precision_score(type_y, type_scores)),
        }

    return results


def evaluate_validation(
    y_true: np.ndarray,
    fraud_types: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate overall and fraud-type validation metrics."""

    predictions = (scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "count": int(len(y_true)),
        "fraud_prevalence": float(y_true.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) else 0.0),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "fraud_type_metrics": evaluate_fraud_types(
            y_true,
            fraud_types,
            scores,
            threshold,
        ),
    }


# ---------------------------------------------------------
# Train one experiment
# ---------------------------------------------------------


def run_experiment(
    experiment_name: str,
    feature_set_name: str,
    feature_columns: tuple[str, ...],
    model_config: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    validation_fraud_types: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Train and evaluate one Isolation Forest configuration."""

    print(f"\n{'=' * 65}")
    print(f"Experiment: {experiment_name}")
    print(f"Raw feature count: {len(feature_columns)}")
    print(f"{'=' * 65}")

    # Fit preprocessing independently for this feature set.
    (
        X_train_processed,
        X_validation_processed,
        preprocessor,
    ) = fit_transform_subset(
        X_train,
        X_validation,
        feature_columns,
    )

    # The training split should contain only legitimate transactions.
    # Still explicitly select legitimate rows for safety.
    normal_mask = y_train == 0
    X_normal = X_train_processed[normal_mask]

    if len(X_normal) < 2:
        raise ValueError(f"{experiment_name}: insufficient legitimate training rows.")

    print("Processed feature count:", X_train_processed.shape[1])
    print("Legitimate training rows:", len(X_normal))

    # Identical model configuration for every experiment.
    model = IsolationForest(**model_config)
    model.fit(X_normal)

    # Negate sklearn's score_samples output so that higher = riskier.
    validation_scores = -model.score_samples(X_validation_processed)

    # Select threshold on validation data only.
    threshold, selected_f1 = select_f1_threshold(
        y_validation,
        validation_scores,
    )

    metrics = evaluate_validation(
        y_validation,
        validation_fraud_types,
        validation_scores,
        threshold,
    )

    metrics.update(
        {
            "feature_set": feature_set_name,
            "raw_feature_count": len(feature_columns),
            "processed_feature_count": X_train_processed.shape[1],
            "training_count": len(X_train),
            "legitimate_training_count": len(X_normal),
            "threshold_selection": {
                "method": "maximum_validation_f1",
                "f1": selected_f1,
            },
            "feature_columns": list(feature_columns),
            "configuration": model_config.copy(),
        }
    )

    # Save model and its exact fitted preprocessor together.
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(feature_columns),
        "threshold": threshold,
        "configuration": model_config.copy(),
    }

    joblib.dump(
        artifact,
        OUTPUT_DIR / f"{experiment_name}.joblib",
    )

    print("\nValidation metrics")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Average precision: {metrics['average_precision']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1:                {metrics['f1']:.4f}")
    print(f"  FPR:               {metrics['false_positive_rate']:.4f}")
    print(f"  Threshold:         {threshold:.6f}")

    print("\nFraud-type recall")
    for fraud_type, type_metrics in metrics["fraud_type_metrics"].items():
        print(
            f"  {fraud_type:15s} " f"{type_metrics['recall_at_global_threshold']:.4f}"
        )

    return metrics, validation_scores, (validation_scores >= threshold).astype(np.int8)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validate_feature_sets()

    print("Loading current train and validation datasets...")

    train_records = load_jsonl(TRAIN_PATH)
    validation_records = load_jsonl(VALIDATION_PATH)

    # Each split receives independent historical feature state.
    X_train, y_train, _ = prepare_detection_dataset(train_records)

    X_validation, y_validation, validation_metadata = prepare_detection_dataset(
        validation_records
    )

    validation_fraud_types = (
        validation_metadata["fraud_type"].fillna("").astype(str).to_numpy()
    )

    if not np.isin(y_train, [0, 1]).all():
        raise ValueError("Training labels must be binary.")

    if not np.isin(y_validation, [0, 1]).all():
        raise ValueError("Validation labels must be binary.")

    if not np.all(y_train == 0):
        print(
            "Warning: training data contains fraud. "
            "The model will still fit only on legitimate transactions."
        )

    if len(np.unique(y_validation)) != 2:
        raise ValueError("Validation data must contain legitimate and fraudulent rows.")

    if not np.array_equal(
        (y_validation == 1),
        (validation_fraud_types != ""),
    ):
        raise ValueError("Validation fraud labels and fraud types are inconsistent.")

    print("\nDataset information")
    print("  Training rows:", f"{len(y_train):,}")
    print("  Validation rows:", f"{len(y_validation):,}")
    print("  Training fraud prevalence:", round(float(y_train.mean()), 4))
    print("  Validation fraud prevalence:", round(float(y_validation.mean()), 4))

    print("\nRunning 36 hyperparameter experiments...")

    all_metrics = {}
    all_scores = {}
    all_predictions = {}

    total_experiments = (
        len(FEATURE_SETS)
        * len(HYPERPARAMETER_GRID["n_estimators"])
        * len(HYPERPARAMETER_GRID["max_samples"])
        * len(HYPERPARAMETER_GRID["max_features"])
    )

    experiment_number = 0

    for feature_set_name, feature_columns in FEATURE_SETS.items():

        for n_estimators, max_samples, max_features in product(
            HYPERPARAMETER_GRID["n_estimators"],
            HYPERPARAMETER_GRID["max_samples"],
            HYPERPARAMETER_GRID["max_features"],
        ):

            experiment_number += 1

            # Build the configuration for this run.
            model_config = {
                **MODEL_BASE_CONFIG,
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "max_features": max_features,
            }

            # Generate a unique name so no experiment overwrites another.
            feature_suffix = str(max_features).replace(".", "p")

            experiment_name = (
                f"{feature_set_name}"
                f"__trees-{n_estimators}"
                f"__samples-{max_samples}"
                f"__features-{feature_suffix}"
            )

            print(f"\nExperiment {experiment_number}/{total_experiments}")

            metrics, scores, predictions = run_experiment(
                experiment_name=experiment_name,
                feature_set_name=feature_set_name,
                feature_columns=feature_columns,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                validation_fraud_types=validation_fraud_types,
            )

            all_metrics[experiment_name] = metrics
            all_scores[experiment_name] = scores
            all_predictions[experiment_name] = predictions

    print(f"\nCompleted {experiment_number} experiments.")

    # -----------------------------------------------------
    # Save combined metrics
    # -----------------------------------------------------

    report = {
        "experiment": "Isolation Forest Feature Ablation",
        "evaluation_split": "validation",
        "test_set_used": False,
        "training_split": str(TRAIN_PATH),
        "validation_split": str(VALIDATION_PATH),
        "fixed_model_configuration": MODEL_BASE_CONFIG,
        "hyperparameter_grid": HYPERPARAMETER_GRID,
        "experiments": all_metrics,
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    # -----------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------

    summary_rows = []

    for experiment_name, metrics in all_metrics.items():
        config = metrics["configuration"]

        row = {
            "experiment": experiment_name,
            "feature_set": metrics["feature_set"],
            "raw_features": metrics["raw_feature_count"],
            "processed_features": metrics["processed_feature_count"],
            "n_estimators": config["n_estimators"],
            "max_samples": config["max_samples"],
            "max_features": config["max_features"],
            "threshold": metrics["threshold"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
        }

        for fraud_type, type_metrics in metrics["fraud_type_metrics"].items():
            row[f"recall_{fraud_type}"] = type_metrics["recall_at_global_threshold"]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        by="average_precision",
        ascending=False,
    ).reset_index(drop=True)

    summary_path = OUTPUT_DIR / "comparison.csv"
    summary.to_csv(summary_path, index=False)

    # -----------------------------------------------------
    # Save validation predictions
    # -----------------------------------------------------

    predictions = pd.DataFrame(
        {
            "transaction_id": (
                validation_metadata["transaction_id"].astype(str).to_numpy()
            ),
            "fraud_label": y_validation,
            "fraud_type": validation_fraud_types,
        }
    )

    for experiment_name in all_scores:

        predictions[f"{experiment_name}_risk_score"] = all_scores[experiment_name]

        predictions[f"{experiment_name}_predicted_label"] = all_predictions[
            experiment_name
        ]

    predictions_path = OUTPUT_DIR / "validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print(f"\n{'=' * 90}")
    print("FEATURE ABLATION SUMMARY")
    print(f"{'=' * 90}")

    display_columns = [
        "experiment",
        "raw_features",
        "processed_features",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nFraud-type recall comparison")

    recall_columns = [
        column for column in summary.columns if column.startswith("recall_")
    ]

    print(
        summary[["experiment"] + recall_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    print("  Metrics:", metrics_path)
    print("  Summary:", summary_path)
    print("  Predictions:", predictions_path)
    print("  Model artifacts:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
