from datetime import datetime
from itertools import islice

import pytest

from simulator.generator import TransactionGenerator

FIXED_START_TIME = datetime(2026, 1, 1, 12, 0, 0)


def transaction_signature(txn):
    """Return reproducible simulation fields, excluding the UUID transaction ID."""
    return (
        txn.account_id,
        txn.merchant_id,
        txn.amount,
        txn.timestamp,
        txn.location,
        txn.merchant_category,
        txn.device_id,
        txn.fraud_label,
        txn.fraud_type,
    )


@pytest.mark.parametrize("n", [1, 2, 5, 10, 50, 100])
def test_generate_stream_returns_exactly_n_transactions(n):
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=n,
        scenario_name="normal",
        start_time=FIXED_START_TIME,
    )

    assert len(transactions) == n


def test_generate_stream_zero_transactions():
    generator = TransactionGenerator(seed=42)
    rng_before = generator.rng.getstate()

    transactions = generator.generate_stream(n=0)

    assert transactions == []
    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0
    assert generator.rng.getstate() == rng_before


def test_generate_stream_rejects_negative_n():
    generator = TransactionGenerator(seed=42)

    with pytest.raises(ValueError, match="non-negative"):
        generator.generate_stream(n=-1)

    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0


@pytest.mark.parametrize("invalid_n", [1.5, "3", True, None])
def test_generate_stream_rejects_non_integer_n(invalid_n):
    generator = TransactionGenerator(seed=42)

    with pytest.raises(TypeError):
        generator.generate_stream(n=invalid_n)

    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0


def test_batch_and_stream_produce_equivalent_transactions():
    seed = 42
    n = 50

    batch_generator = TransactionGenerator(seed=seed)
    stream_generator = TransactionGenerator(seed=seed)

    batch = batch_generator.generate_stream(
        n=n,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )

    stream = stream_generator.stream(
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )
    try:
        streamed = list(islice(stream, n))
    finally:
        stream.close()

    assert len(batch) == n
    assert len(streamed) == n
    assert [transaction_signature(txn) for txn in batch] == [
        transaction_signature(txn) for txn in streamed
    ]


def test_generate_one_transaction_does_not_overshoot():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=1,
        scenario_name="velocity_attack",
        start_time=FIXED_START_TIME,
    )

    assert len(transactions) == 1


def test_normal_scenario_generates_only_legitimate_transactions():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="normal",
        start_time=FIXED_START_TIME,
    )

    assert len(transactions) == 100
    assert all(txn.fraud_label == 0 for txn in transactions)
    assert all(txn.fraud_type == "" for txn in transactions)


@pytest.mark.parametrize(
    "scenario_name",
    [
        "normal",
        "amount_spike_attack",
        "velocity_attack",
        "geo_attack",
        "collusion_attack",
        "mixed",
    ],
)
def test_generator_can_execute_all_known_scenarios(scenario_name):
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name=scenario_name,
        start_time=FIXED_START_TIME,
    )

    assert len(transactions) == 100


def test_generator_updates_account_state():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=20,
        scenario_name="normal",
        start_time=FIXED_START_TIME,
    )

    for txn in transactions:
        history = generator.state.recent_transactions(txn.account_id)
        assert txn in history


def test_generator_updates_merchant_state():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=20,
        scenario_name="normal",
        start_time=FIXED_START_TIME,
    )

    for txn in transactions:
        merchant = generator.state.get_merchant(txn.merchant_id)
        assert txn.account_id in merchant.accounts


def test_state_persists_across_generate_stream_calls():
    generator = TransactionGenerator(seed=42)

    first_batch = generator.generate_stream(
        n=10,
        scenario_name="normal",
        start_time=FIXED_START_TIME,
    )

    state_size_after_first = sum(
        len(account.transactions) for account in generator.state.accounts.values()
    )

    second_batch = generator.generate_stream(
        n=10,
        scenario_name="normal",
        start_time=first_batch[-1].timestamp,
    )

    state_size_after_second = sum(
        len(account.transactions) for account in generator.state.accounts.values()
    )

    assert len(first_batch) == 10
    assert len(second_batch) == 10
    assert state_size_after_second == state_size_after_first + 10
    assert second_batch[0].timestamp >= first_batch[-1].timestamp


def test_generated_transaction_ids_are_unique():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=1000,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )

    ids = [txn.transaction_id for txn in transactions]
    assert len(ids) == len(set(ids))


def test_generated_timestamps_are_chronological():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_scenario_sequence(
        [
            ("mixed", 50),
            ("normal", 50),
            ("velocity_attack", 50),
        ],
        start_time=FIXED_START_TIME,
    )

    timestamps = [txn.timestamp for txn in transactions]

    assert timestamps == sorted(timestamps)
    assert all(a < b for a, b in zip(timestamps, timestamps[1:]))


def test_same_seed_reproduces_simulation():
    g1 = TransactionGenerator(seed=42)
    g2 = TransactionGenerator(seed=42)

    txns1 = g1.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )
    txns2 = g2.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )

    assert [transaction_signature(txn) for txn in txns1] == [
        transaction_signature(txn) for txn in txns2
    ]


def test_different_seeds_produce_different_simulations():
    g1 = TransactionGenerator(seed=42)
    g2 = TransactionGenerator(seed=123)

    txns1 = g1.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )
    txns2 = g2.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )

    assert [transaction_signature(txn) for txn in txns1] != [
        transaction_signature(txn) for txn in txns2
    ]


def test_generator_state_contains_generated_devices_merchants_and_locations():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=50,
        scenario_name="mixed",
        start_time=FIXED_START_TIME,
    )

    for txn in transactions:
        account = generator.state.get_account(txn.account_id)
        assert txn.device_id in account.devices
        assert txn.merchant_id in account.merchants
        assert txn.location in account.locations


def test_generator_state_persists_across_scenarios():
    generator = TransactionGenerator(seed=42)

    first_batch = generator.generate_stream(
        n=20,
        scenario_name="normal",
        start_time=FIXED_START_TIME,
    )

    accounts_before = set(generator.state.accounts.keys())
    merchants_before = set(generator.state.merchants.keys())

    generator.generate_stream(
        n=20,
        scenario_name="velocity_attack",
        start_time=first_batch[-1].timestamp,
    )

    assert accounts_before.issubset(generator.state.accounts.keys())
    assert merchants_before.issubset(generator.state.merchants.keys())


def test_velocity_scenario_produces_velocity_behavior():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="velocity_attack",
        start_time=FIXED_START_TIME,
    )

    velocity_transactions = [
        txn for txn in transactions if txn.fraud_type == "velocity"
    ]
    assert velocity_transactions

    counts = {}
    for txn in velocity_transactions:
        counts[txn.account_id] = counts.get(txn.account_id, 0) + 1

    assert max(counts.values()) >= 2


def test_collusion_scenario_produces_shared_merchant_behavior():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="collusion_attack",
        start_time=FIXED_START_TIME,
    )

    collusion_transactions = [
        txn for txn in transactions if txn.fraud_type == "collusion"
    ]
    assert collusion_transactions

    merchant_accounts = {}
    for txn in collusion_transactions:
        merchant_accounts.setdefault(txn.merchant_id, set()).add(txn.account_id)

    assert any(len(accounts) >= 2 for accounts in merchant_accounts.values())


def test_geo_attack_produces_repeated_account_with_multiple_locations():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="geo_attack",
        start_time=FIXED_START_TIME,
    )

    geo_transactions = [txn for txn in transactions if txn.fraud_type == "geo_hop"]
    assert geo_transactions

    account_locations = {}
    for txn in geo_transactions:
        account_locations.setdefault(txn.account_id, set()).add(txn.location)

    assert any(len(locations) >= 2 for locations in account_locations.values())


def test_stream_produces_one_transaction_at_a_time():
    gen = TransactionGenerator(seed=42)
    stream = gen.stream("velocity_attack", start_time=FIXED_START_TIME)

    try:
        txn = next(stream)

        total_in_state = sum(
            len(account.transactions) for account in gen.state.accounts.values()
        )
        assert total_in_state == 1
        assert gen.state.recent_transactions(txn.account_id)[-1] is txn
    finally:
        stream.close()


def test_stream_continues_chronologically_and_accumulates_state():
    gen = TransactionGenerator(seed=42)
    stream = gen.stream("velocity_attack", start_time=FIXED_START_TIME)

    try:
        txns = [next(stream) for _ in range(30)]
    finally:
        stream.close()

    timestamps = [txn.timestamp for txn in txns]
    assert timestamps == sorted(timestamps)

    total_in_state = sum(
        len(account.transactions) for account in gen.state.accounts.values()
    )
    assert total_in_state == 30


def test_stream_crosses_event_boundaries():
    gen = TransactionGenerator(seed=42)
    stream = gen.stream("velocity_attack", start_time=FIXED_START_TIME)

    try:
        txns = [next(stream) for _ in range(300)]
    finally:
        stream.close()

    labels = [txn.fraud_label for txn in txns]
    label_changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert label_changes >= 4

    timestamps = [txn.timestamp for txn in txns]
    assert timestamps == sorted(timestamps)

    total_in_state = sum(
        len(account.transactions) for account in gen.state.accounts.values()
    )
    assert total_in_state == 300


# ---------------------------------------------------------------------------
# Step 7: batch and scenario-sequence API tests
# ---------------------------------------------------------------------------


def test_scenario_sequence_with_unequal_segment_sizes():
    generator = TransactionGenerator(seed=42)

    txns = generator.generate_scenario_sequence(
        [
            ("normal", 20),
            ("velocity_attack", 15),
            ("mixed", 30),
        ],
        start_time=FIXED_START_TIME,
    )

    assert len(txns) == 65


def test_scenario_sequence_timestamps_are_strictly_increasing():
    generator = TransactionGenerator(seed=42)

    txns = generator.generate_scenario_sequence(
        [
            ("normal", 20),
            ("velocity_attack", 15),
            ("mixed", 30),
        ],
        start_time=FIXED_START_TIME,
    )

    timestamps = [txn.timestamp for txn in txns]
    assert timestamps == sorted(timestamps)
    assert all(a < b for a, b in zip(timestamps, timestamps[1:]))


def test_scenario_sequence_preserves_state_across_segments():
    generator = TransactionGenerator(seed=42)

    txns = generator.generate_scenario_sequence(
        [
            ("normal", 20),
            ("velocity_attack", 15),
            ("mixed", 30),
        ],
        start_time=FIXED_START_TIME,
    )

    generated_ids = {txn.transaction_id for txn in txns}
    recorded_ids = {
        txn.transaction_id
        for account in generator.state.accounts.values()
        for txn in account.transactions
    }

    # The total sequence contains fewer than 100 transactions, so no per-account
    # history can have evicted a generated transaction from its bounded deque.
    assert generated_ids == recorded_ids
    assert len(recorded_ids) == len(txns)

    merchant_count = sum(
        merchant.transaction_count for merchant in generator.state.merchants.values()
    )
    assert merchant_count == len(txns)

    merchant_total = sum(
        merchant.total_amount for merchant in generator.state.merchants.values()
    )
    assert merchant_total == pytest.approx(sum(txn.amount for txn in txns))


def test_scenario_sequence_does_not_carry_unfinished_event_to_next_scenario():
    generator = TransactionGenerator(seed=42)

    txns = generator.generate_scenario_sequence(
        [
            ("velocity_attack", 1),
            ("normal", 5),
        ],
        start_time=FIXED_START_TIME,
    )

    assert len(txns) == 6
    normal_segment = txns[1:]
    assert all(txn.fraud_label == 0 for txn in normal_segment)
    assert all(txn.fraud_type == "" for txn in normal_segment)


def test_scenario_sequence_is_reproducible():
    segments = [
        ("normal", 20),
        ("velocity_attack", 15),
        ("mixed", 30),
    ]

    generator_a = TransactionGenerator(seed=42)
    generator_b = TransactionGenerator(seed=42)

    txns_a = generator_a.generate_scenario_sequence(
        segments,
        start_time=FIXED_START_TIME,
    )
    txns_b = generator_b.generate_scenario_sequence(
        segments,
        start_time=FIXED_START_TIME,
    )

    assert [transaction_signature(txn) for txn in txns_a] == [
        transaction_signature(txn) for txn in txns_b
    ]


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [("normal", 0)],
        [("normal", 0), ("velocity_attack", 0)],
    ],
)
def test_empty_or_zero_length_sequence_does_not_mutate_state_or_rng(segments):
    generator = TransactionGenerator(seed=42)
    rng_before = generator.rng.getstate()

    txns = generator.generate_scenario_sequence(
        segments,
        start_time=FIXED_START_TIME,
    )

    assert txns == []
    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0
    assert generator.rng.getstate() == rng_before


def test_invalid_scenario_sequence_fails_before_generation(monkeypatch):
    generator = TransactionGenerator(seed=42)
    rng_before = generator.rng.getstate()

    def unexpected_generation(*args, **kwargs):
        pytest.fail("Generation began before sequence validation completed")

    monkeypatch.setattr(generator, "generate_stream", unexpected_generation)

    with pytest.raises(ValueError):
        generator.generate_scenario_sequence(
            [
                ("normal", 10),
                ("velocity_attack", 5),
                ("invalid_scenario", 20),
            ],
            start_time=FIXED_START_TIME,
        )

    assert generator.rng.getstate() == rng_before
    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0


@pytest.mark.parametrize(
    ("segments", "expected_exception"),
    [
        ([("normal", -1)], ValueError),
        ([("normal", 1.5)], TypeError),
        ([("normal", "3")], TypeError),
        ([("normal", True)], TypeError),
        ([("normal",)], ValueError),
        ([("normal", 10), ("invalid_scenario", 5)], ValueError),
    ],
)
def test_scenario_sequence_rejects_invalid_segments(segments, expected_exception):
    generator = TransactionGenerator(seed=42)
    rng_before = generator.rng.getstate()

    with pytest.raises(expected_exception):
        generator.generate_scenario_sequence(
            segments,
            start_time=FIXED_START_TIME,
        )

    assert generator.rng.getstate() == rng_before
    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0


def test_scenario_sequence_rejects_string_as_segments():
    generator = TransactionGenerator(seed=42)

    with pytest.raises(TypeError):
        generator.generate_scenario_sequence("normal")


def test_scenario_sequence_rejects_invalid_start_time():
    generator = TransactionGenerator(seed=42)

    with pytest.raises(TypeError, match="start_time"):
        generator.generate_scenario_sequence(
            [("normal", 1)],
            start_time="2026-01-01",
        )

    assert len(generator.state.accounts) == 0
    assert len(generator.state.merchants) == 0
