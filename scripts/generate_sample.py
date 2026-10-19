"""CLI for generating and inspecting synthetic payment transactions."""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from simulator.generator import TransactionGenerator
from simulator.scenarios import get_scenario


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer"
        ) from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "must be greater than or equal to zero"
        )

    return parsed


def parse_start_time(value: str) -> datetime:
    """Parse an ISO 8601 datetime."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a valid ISO 8601 datetime, "
            "for example 2026-10-20T10:00:00"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic payment transactions."
    )

    parser.add_argument(
        "--scenario",
        default="normal",
        help="Scenario name defined in simulator/scenarios.py",
    )

    parser.add_argument(
        "--count",
        type=non_negative_int,
        default=1000,
        help="Number of transactions to generate (default: 1000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulation",
    )

    parser.add_argument(
        "--start-time",
        type=parse_start_time,
        default=None,
        help="Simulation start time in ISO 8601 format",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSONL transactions",
    )

    return parser   


def is_fraudulent(txn) -> bool:
    return txn.fraud_label == 1 or txn.fraud_label is True


def format_timestamp(timestamp) -> str:
    if timestamp is None:
        return "N/A"

    if isinstance(timestamp, datetime):
        return timestamp.isoformat()

    return str(timestamp)


def print_summary(
    transactions: list,
    scenario: str,
    requested_count: int,
    seed: int | None,
) -> None:
    fraudulent = [
        txn for txn in transactions if is_fraudulent(txn)
    ]

    legitimate_count = len(transactions) - len(fraudulent)

    unique_accounts = {
        txn.account_id for txn in transactions
    }

    unique_merchants = {
        txn.merchant_id for txn in transactions
    }

    fraud_breakdown = Counter(
        txn.fraud_type
        for txn in fraudulent
        if txn.fraud_type
    )

    fraud_rate = (
        len(fraudulent) / len(transactions) * 100
        if transactions
        else 0.0
    )

    print("\n" + "=" * 45)
    print("          TRANSACTION SIMULATION")
    print("=" * 45)
    print(f"Scenario:               {scenario}")
    print(f"Requested transactions: {requested_count}")
    print(f"Generated transactions: {len(transactions)}")
    print(f"Seed:                   {seed if seed is not None else 'Not set'}")
    print(f"Unique accounts:        {len(unique_accounts)}")
    print(f"Unique merchants:       {len(unique_merchants)}")
    print(f"Legitimate transactions: {legitimate_count}")
    print(f"Fraudulent transactions: {len(fraudulent)}")
    print(f"Fraud rate:              {fraud_rate:.2f}%")

    print("\nFraud type breakdown:")
    if fraud_breakdown:
        for fraud_type, count in sorted(fraud_breakdown.items()):
            print(f"  {fraud_type}: {count}")
    else:
        print("  No fraudulent transactions")

    if transactions:
        print(
            "\nFirst timestamp: "
            f"{format_timestamp(transactions[0].timestamp)}"
        )
        print(
            "Last timestamp:  "
            f"{format_timestamp(transactions[-1].timestamp)}"
        )
    else:
        print("\nNo transactions generated.")

    print("=" * 45)


def json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def write_jsonl(
    transactions: list,
    output_path: Path,
) -> None:
    """Write one transaction per JSONL line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for txn in transactions:
            file.write(
                json.dumps(
                    txn.to_dict(),
                    default=json_default,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate the scenario before generating anything.
    try:
        scenario_config = get_scenario(args.scenario)
        if scenario_config is None:
            raise ValueError(
                f"Unknown scenario: {args.scenario}"
            )
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))

    generator = TransactionGenerator(seed=args.seed)

    try:
        transactions = generator.generate_stream(
            n=args.count,
            scenario_name=args.scenario,
            start_time=args.start_time,
        )
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))

    if len(transactions) != args.count:
        print(
            f"Error: requested {args.count} transactions, "
            f"but generated {len(transactions)}.",
            file=sys.stderr,
        )
        return 1

    print_summary(
        transactions=transactions,
        scenario=args.scenario,
        requested_count=args.count,
        seed=args.seed,
    )

    if args.output is not None:
        try:
            write_jsonl(transactions, args.output)
        except (OSError, TypeError, ValueError) as exc:
            print(
                f"Error writing output: {exc}",
                file=sys.stderr,
            )
            return 1

        print(f"\nTransactions saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
