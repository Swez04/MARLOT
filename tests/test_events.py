import random

import numpy as np
import pytest
from datetime import datetime

from simulator.events import (
    AmountSpikeEvent,
    CollusionEvent,
    GeoHopEvent,
    LegitimateEvent,
    VelocityEvent,
)
from simulator.state import SimulatorState


def test_legitimate_event_generates_one_transaction():
    state = SimulatorState()
    rng = random.Random(42)

    event = LegitimateEvent(
        start_time=datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    txn = event.next_transaction()

    assert txn is not None
    assert txn.fraud_label == 0
    assert txn.fraud_type == ""
    assert event.is_complete


def test_amount_spike_event_uses_existing_account_history(make_transaction):
    state = SimulatorState()
    rng = random.Random(42)

    amounts = [
        100.0,
        120.0,
        90.0,
        110.0,
        130.0,
        105.0,
        95.0,
        125.0,
        115.0,
        108.0,
        140.0,
        102.0,
        98.0,
        135.0,
        118.0,
        111.0,
        109.0,
        121.0,
        97.0,
        132.0,
        104.0,
        116.0,
        101.0,
        128.0,
        113.0,
    ]

    for i, amount in enumerate(amounts):
        state.update(
            make_transaction(
                transaction_id=f"normal_{i}",
                account_id="acct_0001",
                amount=amount,
            )
        )

    event = AmountSpikeEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    txn = event.next_transaction()

    baseline = np.percentile(amounts, 75)

    assert txn.account_id == "acct_0001"
    assert txn.fraud_label == 1
    assert txn.fraud_type == "amount_spike"

    # Event uses a 10x-50x multiplier.
    assert txn.amount >= round(baseline * 10, 2)


def test_amount_spike_event_handles_insufficient_history():
    state = SimulatorState()
    rng = random.Random(42)

    event = AmountSpikeEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    txn = event.next_transaction()

    assert txn is not None
    assert txn.fraud_label == 1
    assert txn.fraud_type == "amount_spike"
    assert txn.amount > 0


def test_velocity_event_generates_multiple_transactions():
    state = SimulatorState()
    rng = random.Random(42)

    event = VelocityEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    assert len(transactions) >= 5
    assert len(transactions) <= 12
    assert event.is_complete


def test_velocity_event_uses_same_account():
    state = SimulatorState()
    rng = random.Random(42)

    event = VelocityEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    account_ids = {txn.account_id for txn in transactions}

    assert len(account_ids) == 1


def test_velocity_event_timestamps_are_increasing_and_close():
    state = SimulatorState()
    rng = random.Random(42)

    event = VelocityEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    timestamps = [txn.timestamp for txn in transactions]

    assert timestamps == sorted(timestamps)

    gaps = [(b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:])]

    assert all(1 <= gap <= 30 for gap in gaps)


def test_velocity_event_is_persistent_across_calls():
    state = SimulatorState()
    rng = random.Random(42)

    event = VelocityEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    first = event.next_transaction()
    second = event.next_transaction()

    assert first.account_id == second.account_id
    assert second.timestamp > first.timestamp
    assert event.transactions_generated == 2


def test_geo_hop_event_generates_multiple_transactions():
    state = SimulatorState()
    rng = random.Random(42)

    event = GeoHopEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    assert len(transactions) >= 3
    assert len(transactions) <= 6
    assert event.is_complete


def test_geo_hop_uses_same_account():
    state = SimulatorState()
    rng = random.Random(42)

    event = GeoHopEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    account_ids = {txn.account_id for txn in transactions}

    assert len(account_ids) == 1


def test_geo_hop_changes_location():
    state = SimulatorState()
    rng = random.Random(42)

    event = GeoHopEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    locations = [txn.location for txn in transactions]

    assert len(set(locations)) >= 2


def test_geo_hop_timestamps_are_increasing_and_close():
    state = SimulatorState()
    rng = random.Random(42)

    event = GeoHopEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    timestamps = [txn.timestamp for txn in transactions]

    assert timestamps == sorted(timestamps)

    gaps = [(b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:])]

    assert all(30 <= gap <= 120 for gap in gaps)


def test_collusion_event_generates_multiple_transactions():
    state = SimulatorState()
    rng = random.Random(42)

    event = CollusionEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    assert len(transactions) >= 5
    assert len(transactions) <= 10
    assert event.is_complete


def test_collusion_uses_same_merchant():
    state = SimulatorState()
    rng = random.Random(42)

    event = CollusionEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    merchant_ids = {txn.merchant_id for txn in transactions}

    assert len(merchant_ids) == 1


def test_collusion_uses_multiple_accounts():
    state = SimulatorState()
    rng = random.Random(42)

    event = CollusionEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    transactions = []

    while not event.is_complete:
        transactions.append(event.next_transaction())

    account_ids = {txn.account_id for txn in transactions}

    assert len(account_ids) >= 2


@pytest.mark.parametrize(
    "event_class",
    [
        LegitimateEvent,
        AmountSpikeEvent,
        VelocityEvent,
        GeoHopEvent,
        CollusionEvent,
    ],
)
def test_every_event_eventually_completes(event_class):
    state = SimulatorState()
    rng = random.Random(42)

    event = event_class(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    max_steps = 100

    for _ in range(max_steps):
        if event.is_complete:
            break

        event.next_transaction()

    assert event.is_complete


def test_completed_event_cannot_generate_another_transaction():
    state = SimulatorState()
    rng = random.Random(42)

    event = LegitimateEvent(
        start_time=__import__("datetime").datetime(2026, 1, 1),
        rng=rng,
        state=state,
    )

    event.next_transaction()

    assert event.is_complete

    with pytest.raises(RuntimeError):
        event.next_transaction()
