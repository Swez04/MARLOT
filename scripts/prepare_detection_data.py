"""Prepare reproducible train/validation/test datasets for MARLOT Phase 2.

The training split contains only legitimate transactions. Validation and test
splits contain a mixture of legitimate and fraudulent transactions generated
from the scenarios configured in simulator/scenarios.py.

Usage from the repository root:
    python scripts/prepare_detection_data.py
    python scripts/prepare_detection_data.py --train-count 5000 --validation-count 2000 --test-count 2000
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from simulator.generator import TransactionGenerator
from simulator.scenarios import get_scenario

DEFAULT_START_TIME = datetime.fromisoformat("2025-01-01T00:00:00")
DEFAULT_OUTPUT_DIR = Path("data/detection")

# These are fractions of TRANSACTIONS assigned to each scenario, not promised
# fraud-label proportions. Actual label counts are measured in the manifest.
SCENARIO_WEIGHTS: dict[str, dict[str, float]] = {
    "train": {"normal": 1.0},
    "validation": {
        "normal": 0.50,
        "amount_spike_attack": 0.125,
        "velocity_attack": 0.125,
        "geo_attack": 0.125,
        "collusion_attack": 0.125,
    },
    "test": {
        "normal": 0.25,
        "amount_spike_attack": 0.10,
        "velocity_attack": 0.10,
        "geo_attack": 0.10,
        "collusion_attack": 0.10,
        "mixed": 0.35,
    },
}

FRAUD_TYPES = frozenset({"amount_spike", "velocity", "geo_hop", "collusion"})
REQUIRED_FIELDS = frozenset(
    {
        "transaction_id",
        "account_id",
        "merchant_id",
        "amount",
        "timestamp",
        "location",
        "merchant_category",
        "device_id",
        "fraud_label",
        "fraud_type",
    }
)

# Transaction timestamps can advance within events and between events. This
# conservative allowance keeps the independently generated split time windows
# apart for the simulator's default 1-5 second event gaps.
MAX_SECONDS_PER_TRANSACTION = 10
SPLIT_GAP_SECONDS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=50_000)
    parser.add_argument("--validation-count", type=int, default=20_000)
    parser.add_argument("--test-count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--start-time",
        type=datetime.fromisoformat,
        default=DEFAULT_START_TIME,
        help="ISO-8601 timestamp, e.g. 2025-01-01T00:00:00",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace dataset files if they already exist",
    )
    return parser.parse_args(argv)


def validate_counts(args: argparse.Namespace) -> None:
    for name in ("train_count", "validation_count", "test_count"):
        count = getattr(args, name)
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"--{name.replace('_', '-')} must be an integer")
        if count < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not isinstance(args.seed, int) or isinstance(args.seed, bool):
        raise TypeError("--seed must be an integer")
    if not isinstance(args.start_time, datetime):
        raise TypeError("--start-time must be a valid ISO-8601 datetime")


def allocate_scenarios(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate exactly *total* rows using the largest-remainder method.

    The insertion order of ``weights`` is preserved, including for tie breaks.
    """
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("total must be an integer")
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        if total == 0:
            return {}
        raise ValueError("weights must not be empty when total is positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("scenario weights must be finite and non-negative")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("at least one scenario weight must be positive")

    normalized = {name: weight / weight_sum for name, weight in weights.items()}
    exact = {name: total * weight for name, weight in normalized.items()}
    allocation = {name: math.floor(value) for name, value in exact.items()}
    remainder = total - sum(allocation.values())
    order = list(weights)
    ranked = sorted(
        order, key=lambda name: exact[name] - allocation[name], reverse=True
    )
    for name in ranked[:remainder]:
        allocation[name] += 1
    return allocation


def transaction_to_record(
    transaction: Any, split_name: str, index: int
) -> dict[str, Any]:
    """Convert a Transaction to JSON-safe data with a stable exported ID.

    The simulator's in-memory transaction ID is intentionally replaced only in
    the exported dataset. A stable split-prefixed ID makes the resulting files
    byte-for-byte reproducible even if a simulator event internally uses UUID4.
    """
    record = transaction.to_dict()
    if not isinstance(record, dict):
        raise TypeError("Transaction.to_dict() must return a dictionary")
    timestamp = record.get("timestamp")
    if isinstance(timestamp, datetime):
        record["timestamp"] = timestamp.isoformat()
    if not isinstance(record.get("timestamp"), str):
        raise ValueError("Transaction timestamp must serialize to an ISO-8601 string")
    record["transaction_id"] = f"{split_name}-{index:09d}"
    return record


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    fraud_type_counts = Counter(
        record["fraud_type"] for record in records if record["fraud_label"] == 1
    )
    total = len(records)
    fraud_count = sum(record["fraud_label"] == 1 for record in records)
    return {
        "transaction_count": total,
        "legitimate_count": total - fraud_count,
        "fraud_count": fraud_count,
        "fraud_prevalence": fraud_count / total if total else 0.0,
        "fraud_type_counts": dict(sorted(fraud_type_counts.items())),
        "unique_accounts": len({record["account_id"] for record in records}),
        "unique_merchants": len({record["merchant_id"] for record in records}),
    }


def validate_records(
    records: list[dict[str, Any]], expected_count: int, *, require_clean: bool = False
) -> None:
    """Validate the dataset contract before any file is finalized."""
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} records, got {len(records)}")

    seen_ids: set[str] = set()
    previous_timestamp: datetime | None = None

    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")

        transaction_id = record["transaction_id"]
        if not isinstance(transaction_id, str) or not transaction_id:
            raise ValueError(f"Record {index} has an invalid transaction_id")
        if transaction_id in seen_ids:
            raise ValueError(f"Duplicate transaction_id: {transaction_id}")
        seen_ids.add(transaction_id)

        for field in (
            "account_id",
            "merchant_id",
            "device_id",
            "location",
            "merchant_category",
        ):
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"Record {index} has an invalid {field}")

        amount = record["amount"]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError(f"Record {index} has a non-numeric amount")
        if not math.isfinite(float(amount)) or amount < 0:
            raise ValueError(f"Record {index} has an invalid amount")

        label = record["fraud_label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"Record {index} fraud_label must be integer 0 or 1")
        fraud_type = record["fraud_type"]
        if not isinstance(fraud_type, str):
            raise ValueError(f"Record {index} fraud_type must be a string")

        if label == 0 and fraud_type != "":
            raise ValueError(f"Legitimate record {index} must have an empty fraud_type")
        if label == 1 and fraud_type not in FRAUD_TYPES:
            raise ValueError(
                f"Fraudulent record {index} has unsupported fraud_type {fraud_type!r}"
            )
        if require_clean and label != 0:
            raise ValueError(
                "Training split is not clean: a fraud-labelled record was found"
            )

        try:
            timestamp = datetime.fromisoformat(record["timestamp"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record {index} has an invalid ISO-8601 timestamp"
            ) from exc
        if previous_timestamp is not None:
            try:
                is_increasing = timestamp > previous_timestamp
            except TypeError as exc:
                raise ValueError(
                    "Timestamps mix timezone-aware and timezone-naive values"
                ) from exc
            if not is_increasing:
                raise ValueError(
                    "Timestamps must be strictly increasing; "
                    f"record {index} is {timestamp.isoformat()} after "
                    f"{previous_timestamp.isoformat()}"
                )
        previous_timestamp = timestamp


def split_start_times(
    initial: datetime,
    counts: dict[str, int],
    allocation_counts: dict[str, dict[str, int]],
) -> dict[str, datetime]:
    """Choose non-overlapping start times for independently initialized splits."""
    starts: dict[str, datetime] = {"train": initial}
    current = initial
    for split_name in ("train", "validation"):
        segment_count = sum(
            1 for count in allocation_counts[split_name].values() if count > 0
        )
        span_seconds = (
            counts[split_name] + segment_count + 1
        ) * MAX_SECONDS_PER_TRANSACTION + SPLIT_GAP_SECONDS
        current = current + timedelta(seconds=span_seconds)
        next_name = "validation" if split_name == "train" else "test"
        starts[next_name] = current
    return starts


def generate_split(
    split_name: str,
    count: int,
    seed: int,
    start_time: datetime,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    allocations = allocate_scenarios(count, SCENARIO_WEIGHTS[split_name])
    segments = [
        (scenario_name, scenario_count)
        for scenario_name, scenario_count in allocations.items()
    ]

    # Fail fast if scenario definitions in this script drift from scenarios.py.
    for scenario_name, _ in segments:
        get_scenario(scenario_name)

    generator = TransactionGenerator(seed=seed)
    transactions = generator.generate_scenario_sequence(segments, start_time=start_time)
    records = [
        transaction_to_record(transaction, split_name, index)
        for index, transaction in enumerate(transactions, start=1)
    ]
    validate_records(records, count, require_clean=(split_name == "train"))
    return records, allocations


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            file.write("\n")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    validate_counts(args)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    manifest_path = output_dir / "manifest.json"
    final_paths = [output_dir / filename for filename in filenames.values()] + [
        manifest_path
    ]
    existing = [path for path in final_paths if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Dataset output already exists ({joined}). Pass --overwrite to replace it."
        )

    counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    allocations_by_split = {
        split: allocate_scenarios(counts[split], SCENARIO_WEIGHTS[split])
        for split in ("train", "validation", "test")
    }
    starts = split_start_times(args.start_time, counts, allocations_by_split)
    seed_offsets = {"train": 0, "validation": 1_000_003, "test": 2_000_003}

    summaries: dict[str, Any] = {}
    temp_paths: list[Path] = []
    temp_to_final: list[tuple[Path, Path]] = []
    try:
        for split_name in ("train", "validation", "test"):
            records, allocations = generate_split(
                split_name=split_name,
                count=counts[split_name],
                seed=args.seed + seed_offsets[split_name],
                start_time=starts[split_name],
            )
            temp_path = output_dir / f".{filenames[split_name]}.tmp"
            write_jsonl(temp_path, records)
            temp_paths.append(temp_path)
            temp_to_final.append((temp_path, output_dir / filenames[split_name]))
            summaries[split_name] = {
                "file": filenames[split_name],
                "seed": args.seed + seed_offsets[split_name],
                "start_time": starts[split_name].isoformat(),
                "scenario_transaction_counts": allocations,
                "summary": summarize_records(records),
            }

        manifest = {
            "project": "MARLOT",
            "phase": 2,
            "step": 11,
            "dataset_format": "JSONL; one transaction object per line",
            "base_seed": args.seed,
            "initial_start_time": args.start_time.isoformat(),
            "scenario_allocation_note": (
                "Scenario transaction counts describe which generation scenario was used; "
                "they are not guaranteed fraud-label counts. Actual label counts are in each summary."
            ),
            "mixed_scenario_note": (
                "The test split's 'mixed' scenario is held out at the scenario-configuration level. "
                "It combines event types also present in validation; it is not an unseen fraud mechanism."
            ),
            "exported_transaction_id_policy": (
                "Exported IDs are deterministic split-prefixed row IDs; simulator-internal IDs are not preserved."
            ),
            "model_input_safety": {
                "labels_retained_for_evaluation": ["fraud_label", "fraud_type"],
                "must_not_be_model_input_features": [
                    "fraud_label",
                    "fraud_type",
                    "transaction_id",
                ],
                "note": "Feature engineering should derive behavioral features from past-only history and fit preprocessing on train only.",
            },
            "splits": summaries,
        }
        manifest_temp = output_dir / ".manifest.json.tmp"
        with manifest_temp.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        temp_paths.append(manifest_temp)
        temp_to_final.append((manifest_temp, manifest_path))

        # Finalize files only after every split passes validation and is written.
        for temp_path, final_path in temp_to_final:
            temp_path.replace(final_path)
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_dataset(args)
    print(f"Dataset written to: {args.output_dir}")
    for split_name in ("train", "validation", "test"):
        summary = manifest["splits"][split_name]["summary"]
        print(
            f"{split_name:10s} rows={summary['transaction_count']:>7,} "
            f"fraud={summary['fraud_count']:>7,} "
            f"prevalence={summary['fraud_prevalence']:.3%}"
        )
    print(f"Manifest: {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
