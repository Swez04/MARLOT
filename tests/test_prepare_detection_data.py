from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scripts import prepare_detection_data as data_prep
from simulator.transaction import Transaction

START = datetime.fromisoformat("2025-01-01T00:00:00")


def make_record(
    transaction_id: str = "tx-1",
    timestamp: datetime = START,
    fraud_label: int = 0,
    fraud_type: str = "",
) -> dict:
    return {
        "transaction_id": transaction_id,
        "account_id": "acct_0001",
        "merchant_id": "merchant_0001",
        "amount": 42.5,
        "timestamp": timestamp.isoformat(),
        "location": "US-CA",
        "merchant_category": "grocery",
        "device_id": "device_0001",
        "fraud_label": fraud_label,
        "fraud_type": fraud_type,
    }


def test_scenario_allocations_sum_to_requested_count() -> None:
    for split, weights in data_prep.SCENARIO_WEIGHTS.items():
        allocation = data_prep.allocate_scenarios(137, weights)
        assert sum(allocation.values()) == 137
        assert list(allocation) == list(weights)


def test_train_allocation_is_normal_only() -> None:
    assert data_prep.allocate_scenarios(23, data_prep.SCENARIO_WEIGHTS["train"]) == {
        "normal": 23
    }


def test_allocation_handles_zero_rows() -> None:
    assert (
        sum(
            data_prep.allocate_scenarios(0, data_prep.SCENARIO_WEIGHTS["test"]).values()
        )
        == 0
    )


def test_allocation_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        data_prep.allocate_scenarios(-1, {"normal": 1.0})


def test_record_validation_accepts_legitimate_and_fraud_rows() -> None:
    records = [
        make_record("train-000000001", START),
        make_record("train-000000002", START + timedelta(seconds=2), 1, "velocity"),
    ]
    data_prep.validate_records(records, expected_count=2)


def test_training_validation_rejects_fraud() -> None:
    records = [make_record(fraud_label=1, fraud_type="velocity")]
    with pytest.raises(ValueError, match="not clean"):
        data_prep.validate_records(records, 1, require_clean=True)


def test_validation_rejects_duplicate_ids() -> None:
    records = [
        make_record("duplicate", START),
        make_record("duplicate", START + timedelta(seconds=1)),
    ]
    with pytest.raises(ValueError, match="Duplicate transaction_id"):
        data_prep.validate_records(records, 2)


def test_validation_rejects_non_increasing_timestamps() -> None:
    records = [
        make_record("tx-1", START),
        make_record("tx-2", START),
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        data_prep.validate_records(records, 2)


def test_validation_rejects_inconsistent_labels() -> None:
    with pytest.raises(ValueError, match="empty fraud_type"):
        data_prep.validate_records(
            [make_record(fraud_label=0, fraud_type="velocity")], 1
        )
    with pytest.raises(ValueError, match="unsupported fraud_type"):
        data_prep.validate_records(
            [make_record(fraud_label=1, fraud_type="unknown")], 1
        )


def test_transaction_export_uses_deterministic_split_ids() -> None:
    tx = Transaction(
        transaction_id="some-random-internal-id",
        account_id="acct_0001",
        merchant_id="merchant_0001",
        amount=10.0,
        timestamp=START,
        location="US-CA",
        merchant_category="grocery",
        device_id="device_0001",
        fraud_label=0,
        fraud_type="",
    )
    record = data_prep.transaction_to_record(tx, "test", 7)
    assert record["transaction_id"] == "test-000000007"
    assert record["timestamp"] == START.isoformat()


def test_split_start_times_are_chronologically_separated() -> None:
    counts = {"train": 100, "validation": 50, "test": 40}
    allocation_counts = {
        split: data_prep.allocate_scenarios(
            counts[split], data_prep.SCENARIO_WEIGHTS[split]
        )
        for split in counts
    }
    starts = data_prep.split_start_times(START, counts, allocation_counts)
    assert starts["train"] == START
    assert starts["train"] < starts["validation"] < starts["test"]


def test_small_dataset_end_to_end_and_manifest(tmp_path: Path) -> None:
    # Uses the real simulator API, but deliberately keeps the test dataset small.
    args = data_prep.parse_args(
        [
            "--train-count",
            "40",
            "--validation-count",
            "20",
            "--test-count",
            "20",
            "--seed",
            "1122",
            "--start-time",
            "2025-01-01T00:00:00",
            "--output-dir",
            str(tmp_path),
        ]
    )
    manifest = data_prep.build_dataset(args)

    expected_counts = {"train": 40, "validation": 20, "test": 20}
    for split, expected in expected_counts.items():
        path = tmp_path / f"{split}.jsonl"
        assert path.exists()
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == expected
        assert len({row["transaction_id"] for row in rows}) == expected
        assert manifest["splits"][split]["summary"]["transaction_count"] == expected
        data_prep.validate_records(rows, expected, require_clean=(split == "train"))

    train_rows = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["fraud_label"] == 0 for row in train_rows)
    saved_manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["step"] == 11
    assert saved_manifest["splits"]["test"]["scenario_transaction_counts"]["mixed"] > 0


def test_same_seed_and_start_time_reproduce_records() -> None:
    first, first_allocation = data_prep.generate_split("test", 40, 777, START)
    second, second_allocation = data_prep.generate_split("test", 40, 777, START)
    assert first_allocation == second_allocation
    assert first == second


def test_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("existing\n", encoding="utf-8")
    args = data_prep.parse_args(
        [
            "--train-count",
            "1",
            "--validation-count",
            "1",
            "--test-count",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    with pytest.raises(FileExistsError, match="--overwrite"):
        data_prep.build_dataset(args)
