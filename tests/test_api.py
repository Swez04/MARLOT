"""Integration tests for the transaction simulator REST API."""

from datetime import datetime
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from simulator.api.app import app

client = TestClient(app)

START_TIME = "2026-10-20T10:00:00"


def make_request(**overrides):
    """Build a valid simulation request with optional overrides."""
    payload = {
        "scenario": "normal",
        "count": 5,
        "seed": 42,
        "start_time": START_TIME,
        "include_transactions": True,
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ------------------------------------------------------------------
# Successful simulation requests
# ------------------------------------------------------------------


def test_simulate_returns_exact_requested_count():
    response = client.post(
        "/simulate",
        json=make_request(count=10),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["requested_count"] == 10
    assert body["generated_count"] == 10
    assert len(body["transactions"]) == 10


def test_simulate_returns_expected_response_structure():
    response = client.post(
        "/simulate",
        json=make_request(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["scenario"] == "normal"
    assert body["requested_count"] == 5
    assert body["generated_count"] == 5
    assert body["seed"] == 42

    assert "summary" in body
    assert "unique_accounts" in body["summary"]
    assert "unique_merchants" in body["summary"]
    assert "legitimate_count" in body["summary"]
    assert "fraudulent_count" in body["summary"]
    assert "fraud_type_counts" in body["summary"]


def test_transaction_payload_can_be_omitted():
    response = client.post(
        "/simulate",
        json=make_request(include_transactions=False),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["generated_count"] == 5
    assert body["transactions"] is None


def test_transaction_payload_contains_required_fields():
    response = client.post(
        "/simulate",
        json=make_request(count=2, include_transactions=True),
    )

    assert response.status_code == 200

    transactions = response.json()["transactions"]

    assert len(transactions) == 2

    expected_fields = {
        "transaction_id",
        "account_id",
        "merchant_id",
        "amount",
        "timestamp",
        "location",
        "merchant_category",
        "device_id",
        "fraud_label",
        "fraud_type",
    }

    for txn in transactions:
        assert expected_fields.issubset(txn.keys())


# ------------------------------------------------------------------
# Summary consistency
# ------------------------------------------------------------------


def test_summary_matches_generated_transactions():
    response = client.post(
        "/simulate",
        json=make_request(
            scenario="mixed",
            count=100,
            include_transactions=True,
        ),
    )

    assert response.status_code == 200

    body = response.json()
    transactions = body["transactions"]
    summary = body["summary"]

    assert len(transactions) == body["generated_count"]

    assert summary["unique_accounts"] == len(
        {txn["account_id"] for txn in transactions}
    )

    assert summary["unique_merchants"] == len(
        {txn["merchant_id"] for txn in transactions}
    )

    assert summary["legitimate_count"] == sum(
        txn["fraud_label"] == 0 for txn in transactions
    )

    assert summary["fraudulent_count"] == sum(
        txn["fraud_label"] == 1 for txn in transactions
    )

    expected_fraud_counts = Counter(
        txn["fraud_type"]
        for txn in transactions
        if txn["fraud_label"] == 1 and txn["fraud_type"]
    )

    assert summary["fraud_type_counts"] == dict(expected_fraud_counts)

    timestamps = [txn["timestamp"] for txn in transactions]

    assert summary["first_timestamp"] == timestamps[0]
    assert summary["last_timestamp"] == timestamps[-1]


def test_normal_scenario_contains_only_legitimate_transactions():
    response = client.post(
        "/simulate",
        json=make_request(scenario="normal", count=20),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["summary"]["legitimate_count"] == 20
    assert body["summary"]["fraudulent_count"] == 0
    assert body["summary"]["fraud_type_counts"] == {}


# ------------------------------------------------------------------
# Empty simulation
# ------------------------------------------------------------------


def test_zero_transaction_request():
    response = client.post(
        "/simulate",
        json=make_request(count=0),
    )

    assert response.status_code == 200

    body = response.json()
    summary = body["summary"]

    assert body["requested_count"] == 0
    assert body["generated_count"] == 0

    assert summary["unique_accounts"] == 0
    assert summary["unique_merchants"] == 0
    assert summary["legitimate_count"] == 0
    assert summary["fraudulent_count"] == 0
    assert summary["fraud_type_counts"] == {}

    assert summary["first_timestamp"] is None
    assert summary["last_timestamp"] is None
    assert body["transactions"] == []


# ------------------------------------------------------------------
# Request validation
# ------------------------------------------------------------------


@pytest.mark.parametrize("count", [-1, 10_001])
def test_invalid_transaction_count_is_rejected(count):
    response = client.post(
        "/simulate",
        json=make_request(count=count),
    )

    assert response.status_code == 422


def test_unknown_scenario_is_rejected():
    response = client.post(
        "/simulate",
        json=make_request(scenario="unknown_scenario"),
    )

    assert response.status_code == 422


def test_invalid_datetime_is_rejected():
    response = client.post(
        "/simulate",
        json=make_request(start_time="not-a-date"),
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected():
    response = client.post(
        "/simulate",
        json={
            "scenario": "normal",
        },
    )

    assert response.status_code == 422


def test_unexpected_request_field_is_rejected():
    response = client.post(
        "/simulate",
        json=make_request(scenaro="normal"),
    )

    assert response.status_code == 422


# ------------------------------------------------------------------
# Reproducibility and request isolation
# ------------------------------------------------------------------


def test_same_seed_reproduces_seed_controlled_values():
    payload = make_request(
        scenario="mixed",
        count=30,
        seed=42,
    )

    first = client.post("/simulate", json=payload)
    second = client.post("/simulate", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    first_body = first.json()
    second_body = second.json()

    assert first_body["summary"] == second_body["summary"]

    # UUID transaction IDs may intentionally differ between requests.
    # Compare the simulation values controlled by the seeded RNG.
    fields_to_compare = (
        "account_id",
        "merchant_id",
        "amount",
        "timestamp",
        "location",
        "merchant_category",
        "device_id",
        "fraud_label",
        "fraud_type",
    )

    first_transactions = [
        {key: txn[key] for key in fields_to_compare}
        for txn in first_body["transactions"]
    ]

    second_transactions = [
        {key: txn[key] for key in fields_to_compare}
        for txn in second_body["transactions"]
    ]

    assert first_transactions == second_transactions


def test_separate_requests_are_independent():
    payload = make_request(count=20)

    first = client.post("/simulate", json=payload)
    second = client.post("/simulate", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    first_body = first.json()
    second_body = second.json()

    # Each request generates exactly the requested number.
    assert first_body["generated_count"] == 20
    assert second_body["generated_count"] == 20

    # Identical seeded requests reproduce the simulation.
    assert first_body["summary"] == second_body["summary"]

    # Transaction IDs must be unique within each batch.
    first_ids = [
        txn["transaction_id"]
        for txn in first_body["transactions"]
    ]
    second_ids = [
        txn["transaction_id"]
        for txn in second_body["transactions"]
    ]

    assert len(first_ids) == len(set(first_ids))
    assert len(second_ids) == len(set(second_ids))
