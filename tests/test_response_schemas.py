from datetime import datetime

import pytest
from pydantic import ValidationError

from simulator.api.schemas import (
    SimulationResponse,
    SimulationSummary,
    TransactionResponse,
)


@pytest.fixture
def transaction_data():
    return {
        "transaction_id": "txn_abc123",
        "account_id": "acct_0001",
        "merchant_id": "merch_001",
        "amount": 250.50,
        "timestamp": "2026-10-20T10:00:00",
        "location": "IN-BLR",
        "merchant_category": "grocery",
        "device_id": "device_0001",
        "fraud_label": 0,
        "fraud_type": "",
    }


@pytest.fixture
def summary_data():
    return {
        "unique_accounts": 1,
        "unique_merchants": 1,
        "legitimate_count": 1,
        "fraudulent_count": 0,
        "fraud_type_counts": {},
        "first_timestamp": datetime(2026, 10, 20, 10, 0, 0),
        "last_timestamp": datetime(2026, 10, 20, 10, 0, 0),
    }


def test_transaction_response(transaction_data):
    transaction = TransactionResponse(**transaction_data)

    assert transaction.transaction_id == "txn_abc123"
    assert transaction.amount == 250.50
    assert isinstance(transaction.timestamp, datetime)


def test_transaction_response_rejects_missing_field(transaction_data):
    transaction_data.pop("account_id")

    with pytest.raises(ValidationError):
        TransactionResponse(**transaction_data)


def test_simulation_summary(summary_data):
    summary = SimulationSummary(**summary_data)

    assert summary.unique_accounts == 1
    assert summary.legitimate_count == 1
    assert summary.fraudulent_count == 0


def test_empty_simulation_summary():
    summary = SimulationSummary(
        unique_accounts=0,
        unique_merchants=0,
        legitimate_count=0,
        fraudulent_count=0,
        first_timestamp=None,
        last_timestamp=None,
    )

    assert summary.first_timestamp is None
    assert summary.last_timestamp is None
    assert summary.fraud_type_counts == {}


def test_simulation_response(
    transaction_data,
    summary_data,
):
    response = SimulationResponse(
        scenario="normal",
        requested_count=1,
        generated_count=1,
        seed=42,
        summary=SimulationSummary(**summary_data),
        transactions=[TransactionResponse(**transaction_data)],
    )

    assert response.generated_count == 1
    assert len(response.transactions) == 1
    assert response.transactions[0].account_id == "acct_0001"


def test_transactions_can_be_omitted():
    response = SimulationResponse(
        scenario="normal",
        requested_count=0,
        generated_count=0,
        seed=None,
        summary=SimulationSummary(
            unique_accounts=0,
            unique_merchants=0,
            legitimate_count=0,
            fraudulent_count=0,
            fraud_type_counts={},
            first_timestamp=None,
            last_timestamp=None,
        ),
    )

    assert response.transactions is None
