"""
Synthetic transaction generator.

Produces a stream of mostly-normal transactions with injected fraud
patterns. Supports "regimes" so you can later test how detectors react
when the dominant fraud type changes over time (concept drift).
"""

import random
from uuid import uuid4
from datetime import datetime, timedelta

from simulator.regimes import REGIMES
from simulator.transaction import Transaction
from simulator.state import SimulatorState

LOCATIONS = ["US-CA", "US-NY", "US-TX", "GB-LON", "DE-BER", "IN-BLR", "SG-SIN"]
CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "subscription", "jewelry"]


class TransactionGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.state = SimulatorState()

    def _random_transaction_id(self) -> str:
        return f"txn_{self.rng.getrandbits(64):016x}"

    def _random_account(self) -> str:
        return f"acct_{self.rng.randint(1, 500):04d}"

    def _random_merchant(self) -> str:
        return f"merch_{self.rng.randint(1, 200):03d}"

    def _random_device(self) -> str:
        return f"device_{self.rng.randint(1, 1000):04d}"

    def _legit_transaction(self, ts: datetime) -> Transaction:
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self._random_account(),
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=ts,
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=0,
            fraud_type="",
        )

    def _fraud_transaction(self, ts: datetime, fraud_type: str) -> Transaction:
        txn = self._legit_transaction(ts)
        txn.fraud_label = 1
        txn.fraud_type = fraud_type

        if fraud_type == "amount_spike":
            txn.amount = round(self.rng.uniform(2000, 9000), 2)

        elif fraud_type == "velocity":
            txn.amount = round(self.rng.uniform(20, 200), 2)

        elif fraud_type == "geo_hop":
            known_locations = self.state.get_account(txn.account_id).locations
            choices = [loc for loc in LOCATIONS if loc not in known_locations] or LOCATIONS
            txn.location = self.rng.choice(choices)
            txn.amount = round(self.rng.uniform(50, 500), 2)

        elif fraud_type == "collusion":
            txn.merchant_id = "merch_666"
            txn.amount = round(self.rng.uniform(100, 800), 2)

        return txn

    def generate_stream(
        self,
        n: int,
        regime: str = "normal",
        start_time: datetime | None = None,
        seconds_between: float = 1.0,
    ) -> list[Transaction]:
        start_time = start_time or datetime.now()
        fraud_mix = REGIMES.get(regime, {})
        transactions = []

        for i in range(n):
            ts = start_time + timedelta(seconds=i * seconds_between)
            roll = self.rng.random()
            cumulative = 0.0
            chosen_fraud = None
            for fraud_type, rate in fraud_mix.items():
                cumulative += rate
                if roll < cumulative:
                    chosen_fraud = fraud_type
                    break

            txn = self._fraud_transaction(ts, chosen_fraud) if chosen_fraud else self._legit_transaction(ts)
            self.state.update(txn)
            transactions.append(txn)

        return transactions
    
    def generate_regime_sequence(
        self,
        regimes: list[str],
        n_per_regime: int = 1000,
    ) -> list[Transaction]:
        """Concatenate several regimes back to back — useful for testing
        whether a detector adapts as fraud patterns shift over time."""
        all_txns = []
        t = datetime.now()
        for regime in regimes:
            batch = self.generate_stream(n_per_regime, regime=regime, start_time=t)
            all_txns.extend(batch)
            t = batch[-1].timestamp + timedelta(seconds=1)
        return all_txns
    

if __name__ == "__main__":
    gen = TransactionGenerator(seed=42)
    txns = gen.generate_regime_sequence(
        ["normal", "amount_spike", "velocity", "geo_hop"],
        n_per_regime=50,
    )
    print(f"Generated {len(txns)} transactions across regimes")
    fraud_by_type = {}
    for t in txns:
        if t.fraud_type:
            fraud_by_type[t.fraud_type] = fraud_by_type.get(t.fraud_type, 0) + 1
    print("Fraud counts by type:", fraud_by_type)
