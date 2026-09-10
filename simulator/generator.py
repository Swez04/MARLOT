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

LOCATIONS = ["US-CA", "US-NY", "US-TX", "GB-LON", "DE-BER", "IN-BLR", "SG-SIN"]
CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "subscription", "jewelry"]


def _random_transaction() -> str:
    return f"txn_{uuid4().hex}"

def _random_account() -> str:
    return f"acct_{random.randint(1, 500):04d}"


def _random_merchant() -> str:
    return f"merch_{random.randint(1, 200):03d}"


def _random_device() -> str:
    return f"device_{random.randint(1, 1000):04d}"


def _legit_transaction(ts: datetime) -> Transaction:
    return Transaction(
        transaction_id=_random_transaction(),
        account_id=_random_account(),
        merchant_id=_random_merchant(),
        amount=round(random.lognormvariate(3.2, 0.6), 2),  # small everyday purchases
        timestamp=ts,
        location=random.choice(LOCATIONS),
        merchant_category=random.choice(CATEGORIES),
        device_id=_random_device(),
        fraud_label=0,
        fraud_type="",
    )


def _fraud_transaction(ts: datetime, fraud_type: str) -> Transaction:
    txn = _legit_transaction(ts)
    txn.fraud_label = 1
    txn.fraud_type = fraud_type

    if fraud_type == "amount_spike":
        txn.amount = round(random.uniform(2000, 9000), 2)

    elif fraud_type == "velocity":
        # normal amount, but caller is expected to burst many of these
        # for the same account_id in a tight time window
        txn.amount = round(random.uniform(20, 200), 2)

    elif fraud_type == "geo_hop":
        # amount looks normal; the "signal" is a location far from the
        # account's usual location, which you'd detect via account history
        txn.location = random.choice(LOCATIONS)
        txn.amount = round(random.uniform(50, 500), 2)

    elif fraud_type == "collusion":
        # same merchant repeatedly hit by different accounts
        txn.merchant_id = "merch_666"
        txn.amount = round(random.uniform(100, 800), 2)

    return txn


def generate_stream(
    n: int,
    regime: str = "normal",
    start_time: datetime | None = None,
    seconds_between: float = 1.0,
    seed: int | None = None,
) -> list[Transaction]:
    """Generate n transactions under a given regime.

    regime: key into REGIMES, controls which fraud types appear and at
            what rate. "normal" = no fraud injected.
    """
    if seed is not None:
        random.seed(seed)

    start_time = start_time or datetime.now()
    fraud_mix = REGIMES.get(regime, {})
    transactions = []

    for i in range(n):
        ts = start_time + timedelta(seconds=i * seconds_between)
        roll = random.random()
        cumulative = 0.0
        chosen_fraud = None
        for fraud_type, rate in fraud_mix.items():
            cumulative += rate
            if roll < cumulative:
                chosen_fraud = fraud_type
                break

        if chosen_fraud:
            transactions.append(_fraud_transaction(ts, chosen_fraud))
        else:
            transactions.append(_legit_transaction(ts))

    return transactions


def generate_regime_sequence(
    regimes: list[str],
    n_per_regime: int = 1000,
    seed: int | None = None,
) -> list[Transaction]:
    """Concatenate several regimes back to back - useful for testing
    whether a detector adapts as fraud patterns shift over time."""
    all_txns = []
    t = datetime.now()
    for regime in regimes:
        batch = generate_stream(n_per_regime, regime=regime, start_time=t, seed=seed)
        all_txns.extend(batch)
        t = batch[-1].timestamp + timedelta(seconds=1)
    return all_txns


if __name__ == "__main__":
    # quick test
    txns = generate_stream(10, regime="amount_spike", seed=42)
    for t in txns:
        print(t)
