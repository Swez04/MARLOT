from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import numpy as np
import random

from simulator.transaction import Transaction
from simulator.state import SimulatorState
from simulator.clock import SimulationClock

LOCATIONS = ["US-CA", "US-NY", "US-TX", "GB-LON", "DE-BER", "IN-BLR", "SG-SIN"]
CATEGORIES = [
    "grocery",
    "electronics",
    "travel",
    "restaurant",
    "subscription",
    "jewelry",
]

GEO_HOPS = {
    "US-CA": ["GB-LON", "DE-BER", "IN-BLR", "SG-SIN"],
    "US-NY": ["GB-LON", "DE-BER", "IN-BLR", "SG-SIN"],
    "US-TX": ["GB-LON", "DE-BER", "IN-BLR", "SG-SIN"],
    "GB-LON": ["US-CA", "US-NY", "IN-BLR", "SG-SIN"],
    "DE-BER": ["US-CA", "US-NY", "IN-BLR", "SG-SIN"],
    "IN-BLR": ["US-CA", "US-NY", "GB-LON", "DE-BER"],
    "SG-SIN": ["US-CA", "US-NY", "GB-LON", "DE-BER"],
}


class ClockedEventMixin:
    """Shared timing behavior for transaction events."""

    def _init_timing(
        self,
        clock: SimulationClock,
        rng,
        state,
    ) -> None:
        self.clock = clock
        self.rng = rng
        self.state = state
        self._timestamps_issued = 0

    def _next_timestamp(
        self,
        gap_seconds: tuple[float, float] | None = None,
    ):
        """Get this event's next timestamp from the shared clock.

        The first transaction uses the clock's current time.
        Later transactions advance time using the event's own gap range.
        """
        if self._timestamps_issued > 0:
            if gap_seconds is None:
                raise RuntimeError("This event has no configured inter-transaction gap")

            minimum, maximum = gap_seconds

            if minimum <= 0 or maximum < minimum:
                raise ValueError("Invalid inter-transaction gap range")

            gap = self.rng.uniform(minimum, maximum)

            self.clock.advance_by(timedelta(seconds=gap))

        self._timestamps_issued += 1
        return self.clock.current_time


class Event(ABC):
    """Base class for a single simulator event."""

    def __init__(self, clock: datetime, rng: random.Random, state: SimulatorState):
        self.clock = clock
        self.rng = rng
        self.state = state

        self.start_time = clock.current_time
        self.current_time = clock.current_time

        self._timestamps_issued = 0

        self.transaction_count = 0

        self.transactions_generated = 0

    @property
    def is_complete(self) -> bool:
        return self.transactions_generated >= self.transaction_count

    def _next_timestamp(
        self,
        gap_seconds: tuple[float, float] | None = None,
    ):
        """Return a timestamp from the shared simulation clock."""

        if self._timestamps_issued > 0:
            if gap_seconds is None:
                raise RuntimeError("A multi-transaction event requires an internal gap")

            minimum, maximum = gap_seconds

            if minimum <= 0 or maximum < minimum:
                raise ValueError("Invalid inter-transaction gap range")

            gap = self.rng.uniform(minimum, maximum)
            self.clock.advance_by(timedelta(seconds=gap))

        self._timestamps_issued += 1
        self.current_time = self.clock.current_time

        return self.clock.current_time

    def next_transaction(self) -> Transaction:
        """Generate the next transaction belonging to this event,
        and update this event's own completion bookkeeping."""
        if self.is_complete:
            raise RuntimeError("Cannot generate a transaction from a completed event.")

        txn = self._generate_transaction()
        self.transactions_generated += 1
        return txn

    @abstractmethod
    def _generate_transaction(self) -> Transaction:
        """Subclasses implement the event-specific transaction logic here."""
        pass

    def _random_transaction_id(self) -> str:
        return f"txn_{self.rng.getrandbits(64):016x}"

    def _random_account(self) -> str:
        return f"acct_{self.rng.randint(1, 500):04d}"

    def _random_merchant(self) -> str:
        return f"merch_{self.rng.randint(1, 200):03d}"

    def _random_device(self) -> str:
        return f"device_{self.rng.randint(1, 1000):04d}"


class LegitimateEvent(Event):
    def __init__(self, clock, rng, state):
        super().__init__(clock, rng, state)
        self.transaction_count = 1

    def _generate_transaction(self) -> Transaction:
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self._random_account(),
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self._next_timestamp(),
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=0,
            fraud_type="",
        )


class AmountSpikeEvent(Event):
    def __init__(self, clock, rng, state):
        super().__init__(clock, rng, state)
        self.transaction_count = 1

    def _generate_transaction(self) -> Transaction:
        if self.state.accounts:
            account = self.rng.choice(list(self.state.accounts.keys()))
            txns = self.state.recent_transactions(account)
        else:
            account = self._random_account()
            txns = []

        amount = round(self.rng.lognormvariate(3.2, 0.6), 2)

        if len(txns) >= 25:
            amounts = [previous.amount for previous in txns]
            baseline_amount = np.percentile(amounts, 75)
            amount = round(baseline_amount * self.rng.uniform(10, 50), 2)
        else:
            amount = round(amount * self.rng.uniform(10, 50), 2)

        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=account,
            merchant_id=self._random_merchant(),
            amount=amount,
            timestamp=self._next_timestamp(),
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="amount_spike",
        )


class VelocityEvent(Event):
    INTER_TRANSACTION_GAP_SECONDS = (1.0, 30.0)

    def __init__(self, clock, rng, state):
        super().__init__(clock, rng, state)

        self.transaction_count = self.rng.randint(5, 12)

        if self.state.accounts:
            self.account = self.rng.choice(list(self.state.accounts.keys()))
        else:
            self.account = self._random_account()

    def _generate_transaction(self) -> Transaction:
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self.account,
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self._next_timestamp(self.INTER_TRANSACTION_GAP_SECONDS),
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="velocity",
        )


class GeoHopEvent(Event):
    INTER_TRANSACTION_GAP_SECONDS = (30.0, 120.0)

    def __init__(self, clock, rng, state):
        super().__init__(clock, rng, state)

        self.transaction_count = self.rng.randint(3, 6)

        if self.state.accounts:
            self.account = self.rng.choice(list(self.state.accounts.keys()))
        else:
            self.account = self._random_account()

        self.previous_location = None

    def _generate_transaction(self) -> Transaction:
        if self.previous_location is None:
            location = self.rng.choice(LOCATIONS)
        else:
            location = self.rng.choice(GEO_HOPS[self.previous_location])

        self.previous_location = location

        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self.account,
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self._next_timestamp(self.INTER_TRANSACTION_GAP_SECONDS),
            location=location,
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="geo_hop",
        )


class CollusionEvent(Event):
    INTER_TRANSACTION_GAP_SECONDS = (10.0, 60.0)

    def __init__(self, clock, rng, state):
        super().__init__(clock, rng, state)

        self.transaction_count = self.rng.randint(5, 10)

        if self.state.merchants:
            self.merchant = self.rng.choice(list(self.state.merchants.keys()))
        else:
            self.merchant = self._random_merchant()

        if self.state.accounts:
            accounts = list(self.state.accounts.keys())
            num_accounts = min(
                self.rng.randint(2, 5),
                len(accounts),
            )
            self.accounts = self.rng.sample(accounts, num_accounts)
        else:
            self.accounts = [
                self._random_account() for _ in range(self.rng.randint(2, 5))
            ]

    def _generate_transaction(self) -> Transaction:
        account = self.rng.choice(self.accounts)

        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=account,
            merchant_id=self.merchant,
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self._next_timestamp(self.INTER_TRANSACTION_GAP_SECONDS),
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="collusion",
        )


EVENT_TYPES = {
    "legitimate": LegitimateEvent,
    "amount_spike": AmountSpikeEvent,
    "velocity": VelocityEvent,
    "geo_hop": GeoHopEvent,
    "collusion": CollusionEvent,
}
