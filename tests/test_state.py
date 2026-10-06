from .conftest import make_transaction


def test_update_adds_transaction_to_account_history(state, make_transaction):
    txn = make_transaction()

    state.update(txn)

    history = state.recent_transactions(txn.account_id)

    assert len(history) == 1
    assert history[0] == txn


def test_update_records_device_for_account(state, make_transaction):
    txn = make_transaction(
        device_id="device_0077",
    )

    state.update(txn)

    account = state.get_account(txn.account_id)

    assert account.devices == {"device_0077"}


def test_update_records_merchant_for_account(state, make_transaction):
    txn = make_transaction(
        merchant_id="merch_123",
    )

    state.update(txn)

    merchants = state.known_merchants(txn.account_id)

    assert merchants == {"merch_123"}


def test_update_records_location_for_account(state, make_transaction):
    txn = make_transaction(
        location="US-NY",
    )

    state.update(txn)

    account = state.get_account(txn.account_id)

    assert list(account.locations) == ["US-NY"]


def test_update_updates_merchant_statistics(state, make_transaction):
    txn1 = make_transaction(
        transaction_id="txn_001",
        account_id="acct_001",
        merchant_id="merch_001",
        amount=100.0,
    )

    txn2 = make_transaction(
        transaction_id="txn_002",
        account_id="acct_002",
        merchant_id="merch_001",
        amount=250.0,
    )

    state.update(txn1)
    state.update(txn2)

    merchant = state.get_merchant("merch_001")

    assert merchant.transaction_count == 2
    assert merchant.total_amount == 350.0
    assert merchant.accounts == {"acct_001", "acct_002"}


def test_merchant_accounts_are_unique(state, make_transaction):
    txn1 = make_transaction(
        transaction_id="txn_001",
        account_id="acct_001",
        merchant_id="merch_001",
    )

    txn2 = make_transaction(
        transaction_id="txn_002",
        account_id="acct_001",
        merchant_id="merch_001",
    )

    state.update(txn1)
    state.update(txn2)

    merchant = state.get_merchant("merch_001")

    assert merchant.accounts == {"acct_001"}
    assert merchant.transaction_count == 2


def test_account_transaction_history_is_bounded(state, make_transaction):
    for i in range(105):
        txn = make_transaction(
            transaction_id=f"txn_{i:03d}",
        )
        state.update(txn)

    history = state.recent_transactions("acct_0001")

    assert len(history) == 100
    assert history[0].transaction_id == "txn_005"
    assert history[-1].transaction_id == "txn_104"


def test_account_transaction_history_preserves_order(state, make_transaction):
    for i in range(5):
        txn = make_transaction(
            transaction_id=f"txn_{i}",
        )
        state.update(txn)

    history = state.recent_transactions("acct_0001")

    ids = [txn.transaction_id for txn in history]

    assert ids == [
        "txn_0",
        "txn_1",
        "txn_2",
        "txn_3",
        "txn_4",
    ]


def test_location_history_is_bounded(state, make_transaction):
    for i in range(25):
        txn = make_transaction(
            transaction_id=f"txn_{i:03d}",
            location=f"LOC_{i}",
        )
        state.update(txn)

    account = state.get_account("acct_0001")

    assert len(account.locations) == 20
    assert account.locations[0] == "LOC_5"
    assert account.locations[-1] == "LOC_24"


def test_update_links_account_and_merchant(state, make_transaction):
    txn = make_transaction(
        account_id="acct_0042",
        merchant_id="merch_101",
    )

    state.update(txn)

    account = state.get_account("acct_0042")
    merchant = state.get_merchant("merch_101")

    assert "merch_101" in account.merchants
    assert "acct_0042" in merchant.accounts
