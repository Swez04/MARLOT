
"""Inspect the new MARLOT amount and geographical features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from detection.features import prepare_detection_dataset

DATA_DIR = Path("data/detection")
VALIDATION_PATH = DATA_DIR / "validation.jsonl"

# Features expected to be useful for each fraud type.
FEATURES_BY_TYPE = {
    "amount_spike": [
        "account_amount_ratio",
        "account_amount_robust_zscore",
        "log_amount_deviation",
        "category_amount_deviation",
    ],
    "geo_hop": [
        "geo_distance_km",
        "geo_travel_speed_kmph",
        "impossible_travel_flag",
        "location_changed",
    ],
    "velocity": [
        "account_txn_count_60s",
        "account_txn_count_5m",
        "merchant_txn_count_5m",
    ],
    "collusion": [
        "merchant_txn_count_5m",
        "merchant_distinct_accounts_5m",
    ],
}

AVAILABILITY_FEATURES = [
    "has_account_amount_history",
    "has_reliable_account_amount_history",
    "has_category_amount_history",
    "has_geo_coordinates",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    records = load_jsonl(VALIDATION_PATH)

    X, y, metadata = prepare_detection_dataset(records)

    fraud_types = metadata["fraud_type"].fillna("").astype(str).to_numpy()

    print("\n=== Dataset checks ===")
    print("Rows:", len(X))
    print("Raw feature count:", X.shape[1])
    print("Fraud prevalence:", round(float(y.mean()), 4))
    print("Feature columns:", list(X.columns))

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    if not np.isfinite(X.select_dtypes(include=np.number).to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values.")

    # Create diagnostic groups.
    cohorts = {
        "legitimate": y == 0,
        "amount_spike": fraud_types == "amount_spike",
        "geo_hop": fraud_types == "geo_hop",
        "velocity": fraud_types == "velocity",
        "collusion": fraud_types == "collusion",
    }

    # -----------------------------------------------------
    # 1. Feature distributions by transaction type
    # -----------------------------------------------------

    print("\n=== Feature distribution by fraud type ===")

    for fraud_type, features in FEATURES_BY_TYPE.items():

        print(f"\n--- {fraud_type.upper()} ---")

        for feature in features + AVAILABILITY_FEATURES:

            if feature not in X.columns:
                raise ValueError(f"Missing expected feature: {feature}")

            print(f"\n{feature}")

            for cohort_name, mask in cohorts.items():

                values = X.loc[mask, feature].astype(float)

                if values.empty:
                    continue

                print(
                    f"  {cohort_name:14s}"
                    f" n={len(values):5,}"
                    f" median={values.median():10.4f}"
                    f" p90={values.quantile(0.90):10.4f}"
                    f" mean={values.mean():10.4f}"
                )

    # -----------------------------------------------------
    # 2. Individual feature discrimination
    # -----------------------------------------------------

    print("\n=== Individual feature ROC-AUC ===")
    print("Each fraud type is compared against legitimate transactions.")

    for fraud_type, features in FEATURES_BY_TYPE.items():

        # Exclude other fraud categories from the negative class.
        mask = (y == 0) | (fraud_types == fraud_type)

        target = (
            fraud_types[mask] == fraud_type
        ).astype(np.int8)

        print(f"\n--- {fraud_type.upper()} ---")

        for feature in features:

            scores = X.loc[mask, feature].to_numpy(dtype=float)

            if np.unique(scores).size < 2:
                auc = 0.5
            else:
                auc = roc_auc_score(target, scores)

            print(f"  {feature:40s} AUC={auc:.4f}")

    print("\nDiagnostics completed.")


if __name__ == "__main__":
    main()
