"""
Synthetic transaction generator.

Produces a stream of mostly-normal transactions with injected fraud patterns.
"""

import random
from datetime import datetime

from simulator.scenarios import get_scenario
from simulator.transaction import Transaction
from simulator.state import SimulatorState
from simulator.events import (
    LegitimateEvent,
    AmountSpikeEvent,
    VelocityEvent,
    GeoHopEvent,
    CollusionEvent,
)

EVENT_TYPES = {
    "legitimate": LegitimateEvent,
    "amount_spike": AmountSpikeEvent,
    "velocity": VelocityEvent,
    "geo_hop": GeoHopEvent,
    "collusion": CollusionEvent,
}


class TransactionGenerator:
    def __init__(self, seed: int | None = None, state: SimulatorState | None = None):
        self.rng = random.Random(seed)
        self.state = state if state is not None else SimulatorState()

    def generate_stream(
        self,
        n: int,
        scenario_name: str = "normal",
        start_time: datetime | None = None,
    ) -> list[Transaction]:

        start_time = start_time or datetime.now()
        scenario = get_scenario(scenario_name)

        transactions = []
        current_time = start_time

        while len(transactions) < n:

            # Choose which type of event occurs next
            event_name = self.rng.choices(
                list(scenario.event_weights.keys()),
                weights=list(scenario.event_weights.values()),
                k=1,
            )[0]

            event_class = EVENT_TYPES[event_name]

            event = event_class(
                start_time=current_time,
                rng=self.rng,
                state=self.state,
            )

            # Generate transactions belonging to this event
            while not event.is_complete and len(transactions) < n:
                txn = event.next_transaction()

                self.state.update(txn)
                event.transactions_generated += 1

                transactions.append(txn)
                current_time = txn.timestamp

        return transactions

    def generate_scenario_sequence(
        self,
        scenarios: list[str],
        n_per_scenario: int = 1000,
    ) -> list[Transaction]:
        """Concatenate several scenarios back to back — useful for testing
        whether a detector adapts as fraud patterns shift over time."""
        all_txns = []
        t = datetime.now()
        for scenario in scenarios:
            batch = self.generate_stream(
                n_per_scenario, scenario_name=scenario, start_time=t
            )
            all_txns.extend(batch)
            t = batch[-1].timestamp
        return all_txns


if __name__ == "__main__":
    gen = TransactionGenerator(seed=42)

    txns = gen.generate_scenario_sequence(
        ["mixed", "normal", "velocity_attack"],
        n_per_scenario=50,
    )

    print(f"Generated {len(txns)} transactions")

    # 1. Basic count check
    assert len(txns) == 150
    print("Correct number of transactions")

    # 2. Check timestamps are chronological
    timestamps = [txn.timestamp for txn in txns]
    assert timestamps == sorted(timestamps)
    print("Timestamps are chronological")

    # 3. Check transaction IDs are unique
    transaction_ids = [txn.transaction_id for txn in txns]
    assert len(transaction_ids) == len(set(transaction_ids))
    print("Transaction IDs are unique")

    # 4. Check every transaction has a valid fraud label/type
    valid_fraud_types = {
        "",
        "amount_spike",
        "velocity",
        "geo_hop",
        "collusion",
    }

    assert all(txn.fraud_type in valid_fraud_types for txn in txns)
    assert all(txn.fraud_label in {0, 1} for txn in txns)
    print("Fraud labels/types are valid")

    # 5. Check fraud label matches fraud type
    assert all(
        (txn.fraud_label == 0 and txn.fraud_type == "")
        or (txn.fraud_label == 1 and txn.fraud_type != "")
        for txn in txns
    )
    print("Fraud labels match fraud types")

    # 6. Print distribution
    fraud_by_type = {}

    for txn in txns:
        fraud_type = txn.fraud_type or "legitimate"
        fraud_by_type[fraud_type] = fraud_by_type.get(fraud_type, 0) + 1

    print("\nTransaction distribution:")
    for fraud_type, count in fraud_by_type.items():
        print(f"  {fraud_type}: {count}")
