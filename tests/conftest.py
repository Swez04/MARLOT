from datetime import datetime

import pytest

from simulator.generator import TransactionGenerator
from simulator.state import SimulatorState
from simulator.transaction import Transaction


@pytest.fixture
def state():
    return SimulatorState()


@pytest.fixture
def generator():
    return TransactionGenerator(seed=42)


@pytest.fixture
def make_transaction():
    def _make_transaction(
        transaction_id="txn_test",
        account_id="acct_0001",
        merchant_id="merch_001",
        amount=100.0,
        timestamp=None,
        location="IN-BLR",
        merchant_category="grocery",
        device_id="device_0001",
        fraud_label=0,
        fraud_type="",
    ):
        return Transaction(
            transaction_id=transaction_id,
            account_id=account_id,
            merchant_id=merchant_id,
            amount=amount,
            timestamp=timestamp or datetime(2026, 1, 1),
            location=location,
            merchant_category=merchant_category,
            device_id=device_id,
            fraud_label=fraud_label,
            fraud_type=fraud_type,
        )

    return _make_transaction
