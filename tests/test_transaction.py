def test_transaction_can_be_created(make_transaction):
    txn = make_transaction()

    assert txn.transaction_id == "txn_test"
    assert txn.account_id == "acct_0001"
    assert txn.merchant_id == "merch_001"
    assert txn.amount == 100.0
    assert txn.location == "IN-BLR"
    assert txn.merchant_category == "grocery"
    assert txn.device_id == "device_0001"


def test_transaction_can_represent_legitimate_transaction(make_transaction):
    txn = make_transaction(
        fraud_label=0,
        fraud_type="",
    )

    assert txn.fraud_label == 0
    assert txn.fraud_type == ""


def test_transaction_can_represent_fraudulent_transaction(make_transaction):
    txn = make_transaction(
        fraud_label=1,
        fraud_type="velocity",
    )

    assert txn.fraud_label == 1
    assert txn.fraud_type == "velocity"


def test_to_dict_serializes_timestamp(make_transaction):
    txn = make_transaction()

    result = txn.to_dict()

    assert isinstance(result, dict)
    assert result["transaction_id"] == "txn_test"
    assert result["timestamp"] == "2026-01-01T00:00:00"
