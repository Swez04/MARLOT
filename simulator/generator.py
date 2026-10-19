import random
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import islice
from typing import Iterator

from simulator.scenarios import get_scenario
from simulator.transaction import Transaction
from simulator.state import SimulatorState
from simulator.events import EVENT_TYPES


class TransactionGenerator:
    def __init__(self, seed: int | None = None, state: SimulatorState | None = None):
        self.rng = random.Random(seed)
        self.state = state if state is not None else SimulatorState()
        self._last_timestamp: datetime | None = None

    def _get_validated_scenario(self, scenario_name: str):
        """validates scenario names and ensures every configured event has a corresponding class in EVENT_TYPES"""
        try:
            scenario = get_scenario(scenario_name)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown or invalid scenario: {scenario_name!r}") from exc

        if not scenario.event_weights:
            raise ValueError(f"Scenario {scenario_name!r} has no event weights")

        unknown_events = set(scenario.event_weights) - set(EVENT_TYPES)
        if unknown_events:
            raise ValueError(
                f"Scenario {scenario_name!r} references "
                f"unknown events: {sorted(unknown_events)}"
            )

        return scenario

    def stream(
        self,
        scenario_name: str = "normal",
        start_time: datetime | None = None,
    ) -> Iterator[Transaction]:
        """Lazily yield transactions under a scenario, forever.

        The stream never ends on its own: when an event finishes, a new one is
        chosen from the scenario weights. It stops only when the consumer stops
        asking for transactions.
        """

        start_time = start_time or datetime.now()
        scenario = self._get_validated_scenario(scenario_name)
        current_time = start_time

        while True:

            # Choose which type of event occurs next
            event_name = self.rng.choices(
                list(scenario.event_weights.keys()),
                weights=list(scenario.event_weights.values()),
                k=1,
            )[0]

            event_class = EVENT_TYPES[event_name]

            event = event_class(
                start_time=current_time,
                rng=self.rng,
                state=self.state,
            )

            # Generate transactions belonging to this event, one per next()
            while not event.is_complete:
                txn = event.next_transaction()

                # Enforce strictly increasing timestamps across event boundaries.
                if (
                    self._last_timestamp is not None
                    and txn.timestamp <= self._last_timestamp
                ):
                    txn.timestamp = self._last_timestamp + timedelta(microseconds=1)

                self.state.update(txn)  # state first, so the consumer sees it
                current_time = txn.timestamp
                self._last_timestamp = txn.timestamp

                yield txn  # pause here until the consumer asks for the next one

    def generate_stream(
        self,
        n: int,
        scenario_name: str = "normal",
        start_time: datetime | None = None,
    ) -> list[Transaction]:
        """Collect exactly n transactions from the lazy stream."""

        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")

        if n < 0:
            raise ValueError("n must be non-negative")

        # Validate before generating or mutating simulator state.
        self._get_validated_scenario(scenario_name)

        if n == 0:
            return []

        transaction_stream = self.stream(
            scenario_name=scenario_name,
            start_time=start_time,
        )

        try:
            return list(islice(transaction_stream, n))
        finally:
            transaction_stream.close()

    def generate_scenario_sequence(
        self,
        segments: Sequence[tuple[str, int]],
        start_time: datetime | None = None,
    ) -> list[Transaction]:
        """Generate finite scenario segments in sequence.

        Each segment is a (scenario_name, transaction_count) pair.

        Simulator state and RNG state persist across segments. Each segment
        starts a fresh event stream, so unfinished events do not carry over.
        """

        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise TypeError(
                "segments must be a sequence of "
                "(scenario_name, transaction_count) pairs"
            )

        if start_time is not None and not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime or None")

        validated_segments: list[tuple[str, int]] = []

        # Validate the entire request before generating anything.
        for index, segment in enumerate(segments):
            if not isinstance(segment, (tuple, list)) or len(segment) != 2:
                raise ValueError(
                    f"Segment {index} must contain "
                    "(scenario_name, transaction_count)"
                )

            scenario_name, count = segment

            if not isinstance(scenario_name, str) or not scenario_name.strip():
                raise ValueError(f"Segment {index} has an invalid scenario name")

            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(
                    f"Transaction count for segment {index} must be an integer"
                )

            if count < 0:
                raise ValueError(
                    f"Transaction count for segment {index} " "must be non-negative"
                )

            # Validate even zero-length segments.
            self._get_validated_scenario(scenario_name)

            validated_segments.append((scenario_name, count))

        # Empty requests should not generate transactions or alter state.
        if not validated_segments or not any(
            count > 0 for _, count in validated_segments
        ):
            return []

        current_start = start_time if start_time is not None else datetime.now()
        all_transactions: list[Transaction] = []

        for scenario_name, count in validated_segments:
            if count == 0:
                continue

            batch = self.generate_stream(
                n=count,
                scenario_name=scenario_name,
                start_time=current_start,
            )

            all_transactions.extend(batch)

            # Start the next segment strictly after the last transaction
            # in this segment.
            current_start = batch[-1].timestamp + timedelta(microseconds=1)

        return all_transactions


if __name__ == "__main__":
    gen = TransactionGenerator(seed=42)

    txns = gen.generate_scenario_sequence(
        [
            ("mixed", 50),
            ("normal", 50),
            ("velocity_attack", 50),
        ],
        start_time=datetime(2026, 1, 1, 0, 0, 0),
    )

    print(f"Generated {len(txns)} transactions")
    assert len(txns) == 150

    timestamps = [txn.timestamp for txn in txns]
    assert timestamps == sorted(timestamps)
    print("Timestamps are chronological")

    transaction_ids = [txn.transaction_id for txn in txns]
    assert len(transaction_ids) == len(set(transaction_ids))
    print("Transaction IDs are unique")

    valid_fraud_types = {
        "",
        "amount_spike",
        "velocity",
        "geo_hop",
        "collusion",
    }

    assert all(txn.fraud_type in valid_fraud_types for txn in txns)
    assert all(txn.fraud_label in {0, 1} for txn in txns)

    assert all(
        (txn.fraud_label == 0 and txn.fraud_type == "")
        or (txn.fraud_label == 1 and txn.fraud_type != "")
        for txn in txns
    )

    print("Fraud labels/types are valid")
