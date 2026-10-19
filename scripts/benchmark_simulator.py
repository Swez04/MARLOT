"""Benchmark the synthetic transaction simulator.

Measures:
- Streaming and batch throughput
- Peak traced Python memory
- Transaction-count correctness
- Transaction ID uniqueness
- Timestamp ordering
- Merchant-state consistency

Run from the repository root:

    python -m scripts.benchmark_simulator \
        --scenario normal \
        --count 100000 \
        --seed 42 \
        --repeats 3 \
        --mode both
"""

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc

from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from simulator.generator import TransactionGenerator
from simulator.scenarios import get_scenario

DEFAULT_START_TIME = "2026-01-01T00:00:00"


def positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")

    return parsed


def parse_start_time(value: str) -> datetime:
    """Parse an ISO-8601 simulation start time."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a valid ISO-8601 datetime, " "e.g. 2026-01-01T00:00:00"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic transaction generation."
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        help="Scenario name defined in simulator/scenarios.py.",
    )

    parser.add_argument(
        "--count",
        type=positive_int,
        default=100_000,
        help="Transactions per run. Default: 100000.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--start-time",
        type=parse_start_time,
        default=parse_start_time(DEFAULT_START_TIME),
        help=f"ISO-8601 starting time. Default: {DEFAULT_START_TIME}.",
    )

    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        help="Number of repetitions per measurement. Default: 3.",
    )

    parser.add_argument(
        "--mode",
        choices=("stream", "batch", "both"),
        default="both",
        help="Generation mode to benchmark.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a JSON results report.",
    )

    return parser.parse_args()


def create_generator(seed: int) -> TransactionGenerator:
    """Create a fresh generator for an independent run."""
    return TransactionGenerator(seed=seed)


def close_iterator(iterator: Iterator[Any]) -> None:
    """Close a generator iterator when supported."""
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def total_merchant_transactions(
    generator: TransactionGenerator,
) -> int:
    """Return the total number of transactions recorded by merchants."""
    return sum(
        merchant.transaction_count for merchant in generator.state.merchants.values()
    )


def run_stream_throughput(
    scenario: str,
    count: int,
    seed: int,
    start_time: datetime,
) -> dict[str, Any]:
    """Measure lazy consumption without retaining all transactions."""
    generator = create_generator(seed)
    iterator = generator.stream(
        scenario_name=scenario,
        start_time=start_time,
    )

    generated = 0
    started = time.perf_counter()

    try:
        for _ in range(count):
            next(iterator)
            generated += 1
    except StopIteration as exc:
        raise RuntimeError(
            f"Stream ended after {generated} of {count} transactions."
        ) from exc
    finally:
        elapsed = time.perf_counter() - started
        close_iterator(iterator)

    if generated != count:
        raise RuntimeError(f"Expected {count} transactions, generated {generated}.")

    state_count = total_merchant_transactions(generator)

    if state_count != count:
        raise RuntimeError(
            "Merchant state is inconsistent: "
            f"expected {count} recorded transactions, got {state_count}."
        )

    return {
        "mode": "stream",
        "count": generated,
        "elapsed_seconds": elapsed,
        "transactions_per_second": generated / elapsed,
        "state_transaction_count": state_count,
    }


def run_batch_throughput(
    scenario: str,
    count: int,
    seed: int,
    start_time: datetime,
) -> dict[str, Any]:
    """Measure finite batch generation and collection."""
    generator = create_generator(seed)

    started = time.perf_counter()

    transactions = generator.generate_stream(
        n=count,
        scenario_name=scenario,
        start_time=start_time,
    )

    elapsed = time.perf_counter() - started

    if len(transactions) != count:
        raise RuntimeError(
            f"Expected {count} transactions, generated {len(transactions)}."
        )

    state_count = total_merchant_transactions(generator)

    if state_count != count:
        raise RuntimeError(
            "Merchant state is inconsistent: "
            f"expected {count} recorded transactions, got {state_count}."
        )

    result = {
        "mode": "batch",
        "count": len(transactions),
        "elapsed_seconds": elapsed,
        "transactions_per_second": len(transactions) / elapsed,
        "state_transaction_count": state_count,
    }

    del transactions
    return result


def measure_peak_memory(
    mode: str,
    scenario: str,
    count: int,
    seed: int,
    start_time: datetime,
) -> int:
    """Measure peak traced memory in a separate run."""
    gc.collect()
    generator = create_generator(seed)

    tracemalloc.start()

    try:
        if mode == "stream":
            iterator = generator.stream(
                scenario_name=scenario,
                start_time=start_time,
            )

            try:
                for _ in range(count):
                    next(iterator)
            finally:
                close_iterator(iterator)

        elif mode == "batch":
            transactions = generator.generate_stream(
                n=count,
                scenario_name=scenario,
                start_time=start_time,
            )

            if len(transactions) != count:
                raise RuntimeError(
                    f"Expected {count} transactions, " f"generated {len(transactions)}."
                )

        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")

        _, peak_bytes = tracemalloc.get_traced_memory()
        return peak_bytes

    finally:
        tracemalloc.stop()


def validate_simulation(
    scenario: str,
    count: int,
    seed: int,
    start_time: datetime,
) -> dict[str, Any]:
    """Run separate correctness checks without affecting performance results."""
    generator = create_generator(seed)
    iterator = generator.stream(
        scenario_name=scenario,
        start_time=start_time,
    )

    seen_ids: set[str] = set()
    previous_timestamp: datetime | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    generated = 0

    try:
        for _ in range(count):
            try:
                transaction = next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Stream ended after {generated} of {count} transactions."
                ) from exc

            generated += 1

            if transaction.transaction_id in seen_ids:
                raise RuntimeError(
                    f"Duplicate transaction ID: " f"{transaction.transaction_id}"
                )

            seen_ids.add(transaction.transaction_id)

            timestamp = transaction.timestamp

            if previous_timestamp is not None:
                if timestamp <= previous_timestamp:
                    raise RuntimeError(
                        "Transaction timestamps are not strictly increasing."
                    )

            if first_timestamp is None:
                first_timestamp = timestamp

            previous_timestamp = timestamp
            last_timestamp = timestamp

    finally:
        close_iterator(iterator)

    state_count = total_merchant_transactions(generator)

    if generated != count:
        raise RuntimeError(f"Expected {count} transactions, generated {generated}.")

    if state_count != count:
        raise RuntimeError(
            f"State records {state_count} transactions, expected {count}."
        )

    return {
        "transaction_count": generated,
        "unique_transaction_ids": len(seen_ids),
        "timestamps_strictly_increasing": True,
        "merchant_state_consistent": True,
        "first_timestamp": (
            first_timestamp.isoformat() if first_timestamp is not None else None
        ),
        "last_timestamp": (
            last_timestamp.isoformat() if last_timestamp is not None else None
        ),
    }


def benchmark_mode(
    mode: str,
    scenario: str,
    count: int,
    seed: int,
    start_time: datetime,
    repeats: int,
) -> dict[str, Any]:
    """Run throughput and memory measurements for one mode."""
    runner = run_stream_throughput if mode == "stream" else run_batch_throughput

    # Small, separate warm-up run; its result is excluded.
    warmup_count = min(1_000, count)
    runner(
        scenario=scenario,
        count=warmup_count,
        seed=seed,
        start_time=start_time,
    )

    throughput_runs = []

    for _ in range(repeats):
        throughput_runs.append(
            runner(
                scenario=scenario,
                count=count,
                seed=seed,
                start_time=start_time,
            )
        )

    memory_runs = []

    for _ in range(repeats):
        memory_runs.append(
            measure_peak_memory(
                mode=mode,
                scenario=scenario,
                count=count,
                seed=seed,
                start_time=start_time,
            )
        )

    throughputs = [run["transactions_per_second"] for run in throughput_runs]

    elapsed_times = [run["elapsed_seconds"] for run in throughput_runs]

    return {
        "mode": mode,
        "count": count,
        "throughput_runs": throughput_runs,
        "median_elapsed_seconds": statistics.median(elapsed_times),
        "median_transactions_per_second": statistics.median(throughputs),
        "peak_memory_bytes_runs": memory_runs,
        "median_peak_memory_bytes": statistics.median(memory_runs),
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a readable benchmark summary."""
    print()
    print("=" * 68)
    print("TRANSACTION SIMULATOR BENCHMARK")
    print("=" * 68)
    print(f"Scenario:       {report['scenario']}")
    print(f"Count per run:  {report['count']:,}")
    print(f"Seed:           {report['seed']}")
    print(f"Start time:     {report['start_time']}")
    print(f"Repetitions:    {report['repeats']}")
    print(f"Python:         {report['environment']['python']}")
    print(f"Platform:       {report['environment']['platform']}")
    print("-" * 68)

    for mode, result in report["results"].items():
        print(f"\nMode: {mode}")
        print(
            "Median elapsed time:       "
            f"{result['median_elapsed_seconds']:.4f} seconds"
        )
        print(
            "Median throughput:         "
            f"{result['median_transactions_per_second']:,.0f} txn/sec"
        )
        print(
            "Median peak traced memory: "
            f"{result['median_peak_memory_bytes'] / (1024 ** 2):.2f} MiB"
        )

        print("Throughput by repetition:")
        for index, run in enumerate(result["throughput_runs"], start=1):
            print(
                f"  Run {index}: "
                f"{run['transactions_per_second']:,.0f} txn/sec "
                f"({run['elapsed_seconds']:.4f} seconds)"
            )

        print("Peak memory by repetition:")
        for index, peak_bytes in enumerate(
            result["peak_memory_bytes_runs"],
            start=1,
        ):
            print(f"  Run {index}: " f"{peak_bytes / (1024 ** 2):.2f} MiB")

    validation = report["validation"]

    print("\nCorrectness validation")
    print(f"  Transaction count:       {validation['transaction_count']:,}")
    print(f"  Unique transaction IDs:  {validation['unique_transaction_ids']:,}")
    print(
        "  Strict timestamp order: " f"{validation['timestamps_strictly_increasing']}"
    )
    print("  Merchant state correct: " f"{validation['merchant_state_consistent']}")

    if report["results"]:
        best_throughput = max(
            result["median_transactions_per_second"]
            for result in report["results"].values()
        )
        print(f"\nBest measured throughput: {best_throughput:,.0f} txn/sec")
        print(
            "10,000 txn/sec target:   "
            f"{'REACHED' if best_throughput >= 10_000 else 'NOT REACHED'}"
        )

    print("=" * 68)


def main() -> int:
    args = parse_args()

    try:
        # Validate the scenario before starting any benchmark run.
        get_scenario(args.scenario)
    except (ValueError, KeyError) as exc:
        print(f"Error: invalid scenario {args.scenario!r}: {exc}", file=sys.stderr)
        return 2

    results = {}

    modes = ("stream", "batch") if args.mode == "both" else (args.mode,)

    try:
        # Correctness validation is a separate run so that its ID set
        # does not contaminate the streaming memory measurement.
        validation = validate_simulation(
            scenario=args.scenario,
            count=args.count,
            seed=args.seed,
            start_time=args.start_time,
        )

        for mode in modes:
            results[mode] = benchmark_mode(
                mode=mode,
                scenario=args.scenario,
                count=args.count,
                seed=args.seed,
                start_time=args.start_time,
                repeats=args.repeats,
            )

    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    report = {
        "scenario": args.scenario,
        "count": args.count,
        "seed": args.seed,
        "start_time": args.start_time.isoformat(),
        "repeats": args.repeats,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "results": results,
        "validation": validation,
    }

    print_report(report)

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            print(f"\nJSON results saved to: {args.output}")

        except OSError as exc:
            print(
                f"Could not save benchmark report: {exc}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
