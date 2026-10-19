"""Orchestrate stateful synthetic transaction generation."""

import math
import random
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import islice
from typing import Iterator

from simulator.clock import SimulationClock
from simulator.events import EVENT_TYPES
from simulator.scenarios import get_scenario
from simulator.state import SimulatorState
from simulator.transaction import Transaction


class TransactionGenerator:
    def __init__(
        self,
        seed: int | None = None,
        state: SimulatorState | None = None,
        inter_event_gap_seconds: tuple[float, float] = (1.0, 5.0),
    ) -> None:
        self.rng = random.Random(seed)
        self.state = state if state is not None else SimulatorState()

        self.clock: SimulationClock | None = None
        self._last_timestamp: datetime | None = None
        self._initial_start_time: datetime | None = None

        if (
            not isinstance(inter_event_gap_seconds, (tuple, list))
            or len(inter_event_gap_seconds) != 2
        ):
            raise ValueError(
                "inter_event_gap_seconds must contain "
                "(minimum_seconds, maximum_seconds)"
            )

        minimum, maximum = inter_event_gap_seconds

        for value in (minimum, maximum):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Inter-event gap values must be finite numbers")

        if minimum <= 0 or maximum < minimum:
            raise ValueError(
                "Inter-event gaps must be positive and " "maximum must be >= minimum"
            )

        self.inter_event_gap_seconds = (
            float(minimum),
            float(maximum),
        )

    def _get_validated_scenario(self, scenario_name: str):
        """Validate a scenario and its event references."""
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

    def _advance_between_events(self) -> datetime:
        """Advance simulation time by the configured inter-event gap."""
        if self.clock is None:
            raise RuntimeError("Simulation clock has not been initialized")

        minimum, maximum = self.inter_event_gap_seconds
        gap_seconds = self.rng.uniform(minimum, maximum)

        return self.clock.advance_by(timedelta(seconds=gap_seconds))

    def _ensure_clock(
        self,
        start_time: datetime | None = None,
    ) -> SimulationClock:
        """Initialize the clock once without resetting simulation progress."""
        if start_time is not None and not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime or None")

        if self.clock is None:
            initial_time = start_time if start_time is not None else datetime.now()

            self.clock = SimulationClock(initial_time)
            self._initial_start_time = initial_time

        elif (
            start_time is not None
            and start_time != self._initial_start_time
            and start_time != self._last_timestamp
        ):
            raise ValueError(
                "start_time must match the original simulation start "
                "time or the last emitted transaction timestamp"
            )

        return self.clock

    def _stream_validated_scenario(
        self,
        scenario,
        *,
        advance_on_start: bool,
    ) -> Iterator[Transaction]:
        """Generate transactions using an already validated scenario.

        This is the shared generation loop used by the public streaming,
        batch, and scenario-sequence APIs.
        """
        if self.clock is None:
            raise RuntimeError("Simulation clock has not been initialized")

        clock = self.clock

        # A new iterator following an earlier batch or closed iterator
        # starts a fresh event after an inter-event gap.
        if advance_on_start:
            self._advance_between_events()

        while True:
            event_name = self.rng.choices(
                list(scenario.event_weights.keys()),
                weights=list(scenario.event_weights.values()),
                k=1,
            )[0]

            event_class = EVENT_TYPES[event_name]

            event = event_class(
                clock=clock,
                rng=self.rng,
                state=self.state,
            )

            while not event.is_complete:
                txn = event.next_transaction()

                if not isinstance(txn, Transaction):
                    raise TypeError(
                        f"Event {event_name!r} did not return a Transaction"
                    )

                if not isinstance(txn.timestamp, datetime):
                    raise TypeError(
                        f"Event {event_name!r} produced a transaction "
                        "with a non-datetime timestamp"
                    )

                if txn.timestamp != clock.current_time:
                    raise RuntimeError(
                        f"Event {event_name!r} produced timestamp "
                        f"{txn.timestamp!r}, but the simulation clock "
                        f"is at {clock.current_time!r}. Events must "
                        "use the shared clock."
                    )

                if (
                    self._last_timestamp is not None
                    and txn.timestamp <= self._last_timestamp
                ):
                    raise RuntimeError(
                        "Simulation timestamps must be strictly "
                        "increasing: "
                        f"{txn.timestamp!r} <= "
                        f"{self._last_timestamp!r}"
                    )

                # Record every transaction exactly once before yielding it.
                self.state.update(txn)
                self._last_timestamp = txn.timestamp

                yield txn

            # This executes when iteration resumes after an event's final
            # transaction. If the iterator is closed while paused at a yield,
            # the next iterator applies the inter-event gap instead.
            self._advance_between_events()

    def stream(
        self,
        scenario_name: str = "normal",
        start_time: datetime | None = None,
    ) -> Iterator[Transaction]:
        """Lazily yield transactions under a scenario, indefinitely.

        The current event continues while this iterator is paused.
        Closing the iterator discards an unfinished event, while the
        simulation clock, RNG, and simulator state persist.
        """
        scenario = self._get_validated_scenario(scenario_name)
        self._ensure_clock(start_time)

        yield from self._stream_validated_scenario(
            scenario,
            advance_on_start=self._last_timestamp is not None,
        )

    def _collect_batch(
        self,
        n: int,
        scenario,
    ) -> list[Transaction]:
        """Collect exactly n transactions using the shared lazy stream."""
        if n == 0:
            return []

        transaction_stream = self._stream_validated_scenario(
            scenario,
            advance_on_start=self._last_timestamp is not None,
        )

        try:
            return list(islice(transaction_stream, n))
        finally:
            transaction_stream.close()

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

        # Validate before initializing the clock or mutating state.
        scenario = self._get_validated_scenario(scenario_name)

        if start_time is not None and not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime or None")

        if n == 0:
            return []

        self._ensure_clock(start_time)

        return self._collect_batch(n, scenario)

    def generate_scenario_sequence(
        self,
        segments: Sequence[tuple[str, int]],
        start_time: datetime | None = None,
    ) -> list[Transaction]:
        """Generate scenario segments while preserving the simulation clock.

        Each segment is a (scenario_name, transaction_count) pair.
        Unfinished events are discarded at scenario boundaries, while
        simulator state, RNG state, and simulation time persist.

        The entire request is validated before generation begins.
        """
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
            raise TypeError(
                "segments must be a sequence of "
                "(scenario_name, transaction_count) pairs"
            )

        if start_time is not None and not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime or None")

        # Keep validated scenario objects so they do not need to be
        # looked up and validated again during each segment.
        validated_segments: list[tuple[str, int, object]] = []

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
                    f"Transaction count for segment {index} " "must be an integer"
                )

            if count < 0:
                raise ValueError(
                    f"Transaction count for segment {index} " "must be non-negative"
                )

            scenario = self._get_validated_scenario(scenario_name)

            validated_segments.append((scenario_name, count, scenario))

        # Empty requests must not initialize the clock or consume RNG.
        if not validated_segments or not any(
            count > 0 for _, count, _ in validated_segments
        ):
            return []

        self._ensure_clock(start_time)

        all_transactions: list[Transaction] = []

        for _, count, scenario in validated_segments:
            if count == 0:
                continue

            batch = self._collect_batch(count, scenario)
            all_transactions.extend(batch)

        return all_transactions
