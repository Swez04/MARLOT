from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from detection.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    HistoricalFeatureBuilder,
    build_features,
    create_preprocessor,
    extract_labels,
    extract_metadata,
    fit_preprocessor,
    prepare_detection_dataset,
    transform_features,
)


BASE_TIME = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)


def make_record(
    seconds: int,
    *,
    account_id: str = "acct_0001",
    merchant_id: str = "merch_001",
    device_id: str = "device_0001",
    amount: float = 100.0,
    location: str = "US-CA",
    merchant_category: str = "grocery",
    fraud_label: int = 0,
    fraud_type: str = "",
) -> dict:
    timestamp = BASE_TIME + timedelta(seconds=seconds)

    return {
        "transaction_id": f"txn_{seconds:06d}_{account_id}",
        "account_id": account_id,
        "merchant_id": merchant_id,
        "device_id": device_id,
        "amount": amount,
        "timestamp": timestamp.isoformat(),
        "location": location,
        "merchant_category": merchant_category,
        "fraud_label": fraud_label,
        "fraud_type": fraud_type,
    }


def test_feature_schema_excludes_labels_and_identifiers():
    records = [
        make_record(0),
        make_record(10, amount=200.0),
    ]

    X = build_features(records)

    assert len(X) == 2
    assert tuple(X.columns) == FEATURE_COLUMNS

    assert not (set(X.columns) & FORBIDDEN_FEATURE_COLUMNS)
    assert set(X.columns) == set(FEATURE_COLUMNS)


def test_first_transaction_uses_documented_history_defaults():
    X = build_features([make_record(0)])

    row = X.iloc[0]

    assert row["account_txn_count_60s"] == 0
    assert row["account_txn_count_5m"] == 0
    assert row["seconds_since_prev_account_txn"] == 0.0

    assert row["has_prev_account_txn"] == 0
    assert row["has_account_amount_history"] == 0
    assert row["log_amount_deviation"] == 0.0

    assert row["is_new_device_for_account"] == 1
    assert row["location_changed"] == 0

    assert row["merchant_txn_count_5m"] == 0
    assert row["merchant_distinct_accounts_5m"] == 0


def test_account_rolling_counts_and_amount_deviation():
    records = [
        make_record(0, amount=100.0),
        make_record(30, amount=200.0),
        make_record(61, amount=300.0),
    ]

    X = build_features(records)

    first = X.iloc[0]
    second = X.iloc[1]
    third = X.iloc[2]

    # The current transaction must never count itself.
    assert first["account_txn_count_60s"] == 0
    assert first["account_txn_count_5m"] == 0

    assert second["account_txn_count_60s"] == 1
    assert second["account_txn_count_5m"] == 1
    assert second["seconds_since_prev_account_txn"] == 30.0

    # At t=61, the t=0 transaction is outside the 60-second window.
    assert third["account_txn_count_60s"] == 1
    assert third["account_txn_count_5m"] == 2
    assert third["seconds_since_prev_account_txn"] == 31.0

    expected_deviation = (
        np.log1p(200.0) - np.log1p(100.0)
    )

    assert second["log_amount_deviation"] == pytest.approx(
        expected_deviation
    )


def test_sixty_second_window_includes_its_lower_boundary():
    records = [
        make_record(0),
        make_record(60, amount=150.0),
    ]

    X = build_features(records)

    # Window is [t - 60 seconds, t): the first transaction is included.
    assert X.iloc[1]["account_txn_count_60s"] == 1


def test_device_novelty_and_location_change_use_prior_history():
    records = [
        make_record(
            0,
            device_id="device_A",
            location="US-CA",
        ),
        make_record(
            10,
            device_id="device_A",
            location="US-CA",
        ),
        make_record(
            20,
            device_id="device_B",
            location="DE-BER",
        ),
    ]

    X = build_features(records)

    assert X.iloc[0]["is_new_device_for_account"] == 1
    assert X.iloc[1]["is_new_device_for_account"] == 0
    assert X.iloc[2]["is_new_device_for_account"] == 1

    assert X.iloc[0]["location_changed"] == 0
    assert X.iloc[1]["location_changed"] == 0
    assert X.iloc[2]["location_changed"] == 1


def test_merchant_rolling_statistics_track_distinct_accounts():
    records = [
        make_record(
            0,
            account_id="acct_A",
            merchant_id="merch_shared",
        ),
        make_record(
            10,
            account_id="acct_B",
            merchant_id="merch_shared",
        ),
        make_record(
            20,
            account_id="acct_C",
            merchant_id="merch_shared",
        ),
    ]

    X = build_features(records)

    assert X.iloc[0]["merchant_txn_count_5m"] == 0
    assert X.iloc[1]["merchant_txn_count_5m"] == 1
    assert X.iloc[2]["merchant_txn_count_5m"] == 2

    assert X.iloc[2]["merchant_distinct_accounts_5m"] == 2


def test_merchant_statistics_expire_old_transactions():
    records = [
        make_record(
            0,
            account_id="acct_A",
            merchant_id="merch_shared",
        ),
        make_record(
            301,
            account_id="acct_B",
            merchant_id="merch_shared",
        ),
    ]

    X = build_features(records)

    assert X.iloc[1]["merchant_txn_count_5m"] == 0
    assert X.iloc[1]["merchant_distinct_accounts_5m"] == 0


def test_builder_preserves_history_across_calls():
    builder = HistoricalFeatureBuilder()

    first = builder.transform_one(make_record(0))
    second = builder.transform_one(make_record(10, amount=150.0))

    assert first["account_txn_count_5m"] == 0
    assert second["account_txn_count_5m"] == 1
    assert second["seconds_since_prev_account_txn"] == 10.0


def test_reset_clears_historical_state():
    builder = HistoricalFeatureBuilder()

    builder.transform_one(make_record(100))
    builder.reset()

    # An earlier timestamp is accepted after resetting the builder.
    result = builder.transform_one(make_record(0))

    assert result["account_txn_count_5m"] == 0
    assert result["merchant_txn_count_5m"] == 0
    assert result["has_prev_account_txn"] == 0


def test_out_of_order_transactions_raise_value_error():
    records = [
        make_record(10),
        make_record(9),
    ]

    with pytest.raises(ValueError, match="strictly increasing"):
        build_features(records)


def test_missing_required_fields_raise_value_error():
    record = make_record(0)
    del record["account_id"]

    with pytest.raises(ValueError, match="required fields"):
        build_features([record])


def test_invalid_amount_is_rejected():
    record = make_record(0, amount=float("nan"))

    with pytest.raises(ValueError, match="finite and non-negative"):
        build_features([record])


def test_prepare_dataset_keeps_labels_and_metadata_aligned():
    records = [
        make_record(0, fraud_label=0),
        make_record(
            10,
            fraud_label=1,
            fraud_type="velocity",
        ),
    ]

    X, y, metadata = prepare_detection_dataset(records)

    assert len(X) == len(y) == len(metadata) == 2

    assert y.tolist() == [0, 1]
    assert metadata["transaction_id"].tolist() == [
        record["transaction_id"] for record in records
    ]
    assert metadata["fraud_type"].tolist() == [
        "",
        "velocity",
    ]

    assert "fraud_label" not in X.columns
    assert "fraud_type" not in X.columns


def test_extract_labels_rejects_invalid_values():
    records = [
        make_record(0, fraud_label=2),
    ]

    with pytest.raises(ValueError, match="Invalid binary fraud label"):
        extract_labels(records)


def test_extract_metadata_does_not_modify_records():
    records = [make_record(0)]
    original = dict(records[0])

    metadata = extract_metadata(records)

    assert len(metadata) == 1
    assert records[0] == original


def test_preprocessor_handles_unseen_categories():
    train_records = [
        make_record(
            0,
            merchant_category="grocery",
            location="US-CA",
        ),
        make_record(
            10,
            account_id="acct_B",
            merchant_category="electronics",
            location="US-NY",
        ),
    ]

    validation_records = [
        make_record(
            0,
            account_id="acct_NEW",
            merchant_category="travel",
            location="DE-BER",
        ),
    ]

    X_train = build_features(train_records)
    X_validation = build_features(validation_records)

    preprocessor = fit_preprocessor(X_train)

    X_train_processed = transform_features(X_train, preprocessor)
    X_validation_processed = transform_features(
        X_validation, preprocessor
    )

    assert X_train_processed.shape[0] == 2
    assert X_validation_processed.shape[0] == 1

    assert (
        X_train_processed.shape[1]
        == X_validation_processed.shape[1]
    )

    assert np.isfinite(X_train_processed).all()
    assert np.isfinite(X_validation_processed).all()


def test_preprocessor_rejects_label_leakage_columns():
    X = build_features([make_record(0)])
    X["fraud_label"] = [0]

    with pytest.raises(ValueError, match="Unexpected columns"):
        fit_preprocessor(X)


def test_preprocessor_rejects_missing_feature_columns():
    X = build_features([make_record(0)])
    X = X.drop(columns=["log_amount"])

    with pytest.raises(ValueError, match="Missing required"):
        fit_preprocessor(X)


def test_feature_extraction_is_reproducible():
    records = [
        make_record(0),
        make_record(10, amount=125.0),
        make_record(35, amount=250.0),
    ]

    first = build_features(records)
    second = build_features(records)

    pd.testing.assert_frame_equal(first, second)


def test_missing_categories_get_consistent_placeholder():
    record = make_record(0)
    record["merchant_category"] = None
    record["location"] = ""

    X = build_features([record])

    assert X.iloc[0]["merchant_category"] == "__MISSING__"
    assert X.iloc[0]["location"] == "__MISSING__"


def test_processed_matrix_contains_only_finite_values():
    records = [
        make_record(0),
        make_record(10, amount=200.0),
        make_record(
            20,
            account_id="acct_B",
            merchant_category="electronics",
            location="US-NY",
        ),
    ]

    X = build_features(records)
    preprocessor = create_preprocessor()
    preprocessor.fit(X)

    matrix = transform_features(X, preprocessor)

    assert matrix.ndim == 2
    assert matrix.shape[0] == len(records)
    assert np.isfinite(matrix).all()
