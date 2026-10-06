from datetime import datetime
import pytest

from simulator.generator import TransactionGenerator


def transaction_signature(txn):
    return (
        txn.transaction_id,
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
    )

    assert len(transactions) == n


def test_generate_one_transaction_does_not_overshoot():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=1,
        scenario_name="velocity_attack",
    )

    assert len(transactions) == 1


def test_normal_scenario_generates_only_legitimate_transactions():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="normal",
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
    )

    assert len(transactions) == 100


def test_generator_updates_account_state():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=20,
        scenario_name="normal",
    )

    for txn in transactions:
        history = generator.state.recent_transactions(txn.account_id)

        assert txn in history


def test_generator_updates_merchant_state():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=20,
        scenario_name="normal",
    )

    for txn in transactions:
        merchant = generator.state.get_merchant(txn.merchant_id)

        assert txn.account_id in merchant.accounts


def test_state_persists_across_generate_stream_calls():
    generator = TransactionGenerator(seed=42)

    first_batch = generator.generate_stream(
        n=10,
        scenario_name="normal",
    )

    state_size_after_first = sum(
        len(account.transactions) for account in generator.state.accounts.values()
    )

    second_batch = generator.generate_stream(
        n=10,
        scenario_name="normal",
    )

    state_size_after_second = sum(
        len(account.transactions) for account in generator.state.accounts.values()
    )

    assert len(first_batch) == 10
    assert len(second_batch) == 10

    # No history has exceeded the 100-transaction account limit here,
    # so the total should increase by exactly 10.
    assert state_size_after_second == state_size_after_first + 10


def test_generated_transaction_ids_are_unique():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=1000,
        scenario_name="mixed",
    )

    ids = [txn.transaction_id for txn in transactions]

    assert len(ids) == len(set(ids))


def test_generated_timestamps_are_chronological():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_scenario_sequence(
        [
            "mixed",
            "normal",
            "velocity_attack",
        ],
        n_per_scenario=50,
    )

    timestamps = [txn.timestamp for txn in transactions]

    assert timestamps == sorted(timestamps)


def test_same_seed_reproduces_simulation():
    g1 = TransactionGenerator(seed=42)
    g2 = TransactionGenerator(seed=42)

    start_time = datetime(2026, 1, 1, 12, 0, 0)

    txns1 = g1.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=start_time,
    )

    txns2 = g2.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=start_time,
    )

    signatures1 = [transaction_signature(txn) for txn in txns1]
    signatures2 = [transaction_signature(txn) for txn in txns2]

    assert signatures1 == signatures2


def test_different_seeds_produce_different_simulations():
    g1 = TransactionGenerator(seed=42)
    g2 = TransactionGenerator(seed=123)

    start_time = datetime(2026, 1, 1, 12, 0, 0)

    txns1 = g1.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=start_time,
    )

    txns2 = g2.generate_stream(
        n=100,
        scenario_name="mixed",
        start_time=start_time,
    )

    signatures1 = [transaction_signature(txn) for txn in txns1]
    signatures2 = [transaction_signature(txn) for txn in txns2]

    assert signatures1 != signatures2


def test_different_seeds_produce_different_simulations():
    g1 = TransactionGenerator(seed=42)
    g2 = TransactionGenerator(seed=123)

    txns1 = g1.generate_stream(
        n=100,
        scenario_name="mixed",
    )

    txns2 = g2.generate_stream(
        n=100,
        scenario_name="mixed",
    )

    signatures1 = [transaction_signature(txn) for txn in txns1]
    signatures2 = [transaction_signature(txn) for txn in txns2]

    assert signatures1 != signatures2


def test_generator_state_contains_generated_devices_merchants_and_locations():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=50,
        scenario_name="mixed",
    )

    for txn in transactions:
        account = generator.state.get_account(txn.account_id)

        assert txn.device_id in account.devices
        assert txn.merchant_id in account.merchants
        assert txn.location in account.locations


def test_generator_scenario_sequence_preserves_state():
    generator = TransactionGenerator(seed=42)

    generator.generate_stream(
        n=20,
        scenario_name="normal",
    )

    accounts_before = set(generator.state.accounts.keys())
    merchants_before = set(generator.state.merchants.keys())

    generator.generate_stream(
        n=20,
        scenario_name="velocity_attack",
    )

    # Existing state must not have been reset.
    assert accounts_before.issubset(generator.state.accounts.keys())
    assert merchants_before.issubset(generator.state.merchants.keys())


def test_velocity_scenario_produces_velocity_behavior():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="velocity_attack",
    )

    velocity_transactions = [
        txn for txn in transactions if txn.fraud_type == "velocity"
    ]

    assert velocity_transactions

    # At least one account should have multiple velocity transactions.
    counts = {}

    for txn in velocity_transactions:
        counts[txn.account_id] = counts.get(txn.account_id, 0) + 1

    assert max(counts.values()) >= 2


def test_collusion_scenario_produces_shared_merchant_behavior():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="collusion_attack",
    )

    collusion_transactions = [
        txn for txn in transactions if txn.fraud_type == "collusion"
    ]

    assert collusion_transactions

    merchant_accounts = {}

    for txn in collusion_transactions:
        merchant_accounts.setdefault(
            txn.merchant_id,
            set(),
        ).add(txn.account_id)

    assert any(len(accounts) >= 2 for accounts in merchant_accounts.values())


def test_geo_attack_produces_repeated_account_with_multiple_locations():
    generator = TransactionGenerator(seed=42)

    transactions = generator.generate_stream(
        n=100,
        scenario_name="geo_attack",
    )

    geo_transactions = [txn for txn in transactions if txn.fraud_type == "geo_hop"]

    assert geo_transactions

    account_locations = {}

    for txn in geo_transactions:
        account_locations.setdefault(
            txn.account_id,
            set(),
        ).add(txn.location)

    assert any(len(locations) >= 2 for locations in account_locations.values())
