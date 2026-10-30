"""Generate and export processed features for MARLOT Phase 2."""

import json
from pathlib import Path

import joblib
import numpy as np

from detection.features import (
    prepare_detection_dataset,
    fit_preprocessor,
    transform_features,
    save_processed_split,
)

DATA_DIR = Path("data/detection")
PROCESSED_DIR = DATA_DIR / "processed"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSON Lines transaction records."""
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def metadata_column(metadata, name: str) -> np.ndarray:
    """Extract an aligned metadata column from a DataFrame or record collection."""

    if hasattr(metadata, "columns"):
        if name not in metadata.columns:
            raise ValueError(f"Metadata is missing required column: {name}")
        values = metadata[name].to_numpy()

    elif isinstance(metadata, dict):
        if name not in metadata:
            raise ValueError(f"Metadata is missing required column: {name}")
        values = np.asarray(metadata[name])

    else:
        try:
            values = np.asarray([record[name] for record in metadata])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Could not extract {name!r} from evaluation metadata"
            ) from exc

    return values.astype(str)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the original datasets.
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    validation_records = load_jsonl(DATA_DIR / "validation.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")

    # 2. Build raw features independently for each dataset.
    #    Each call starts with independent historical feature state.
    X_train, y_train, train_metadata = prepare_detection_dataset(train_records)
    X_val, y_val, val_metadata = prepare_detection_dataset(validation_records)
    X_test, y_test, test_metadata = prepare_detection_dataset(test_records)

    # 3. Fit preprocessing ONLY on training features.
    preprocessor = fit_preprocessor(X_train)

    # 4. Apply the same fitted preprocessing to all three datasets.
    X_train_processed = transform_features(X_train, preprocessor)
    X_val_processed = transform_features(X_val, preprocessor)
    X_test_processed = transform_features(X_test, preprocessor)

    # 5. Export processed matrices and aligned evaluation metadata.
    splits = [
        (
            "train",
            X_train_processed,
            y_train,
            train_metadata,
        ),
        (
            "validation",
            X_val_processed,
            y_val,
            val_metadata,
        ),
        (
            "test",
            X_test_processed,
            y_test,
            test_metadata,
        ),
    ]

    for split_name, X, y, metadata in splits:
        transaction_ids = metadata_column(metadata, "transaction_id")
        fraud_types = metadata_column(metadata, "fraud_type")

        save_processed_split(
            PROCESSED_DIR / f"{split_name}.npz",
            X,
            y,
            transaction_ids,
            fraud_types,
        )

        print(
            f"{split_name:10s} "
            f"raw={len(y):>7,} x {X_train.shape[1]:<3} "
            f"processed={X.shape}"
        )

    # 6. Save the exact fitted preprocessor for inference.
    joblib.dump(
        preprocessor,
        PROCESSED_DIR / "preprocessor.joblib",
    )

    # 7. Display dataset statistics.
    print("\nFraud prevalence:")
    print(f"Training:   {np.mean(y_train):.3%}")
    print(f"Validation: {np.mean(y_val):.3%}")
    print(f"Test:       {np.mean(y_test):.3%}")

    print(f"\nProcessed data saved to: {PROCESSED_DIR.resolve()}")
    print("Saved fitted preprocessor.")


if __name__ == "__main__":
    main()
