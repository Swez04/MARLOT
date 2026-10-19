"""FastAPI application for the synthetic transaction simulator."""

from collections import Counter

from fastapi import FastAPI, HTTPException

from simulator.generator import TransactionGenerator
from simulator.scenarios import get_scenario
from simulator.transaction import Transaction

from .schemas import (
    SimulationRequest,
    SimulationResponse,
    SimulationSummary,
    TransactionResponse,
)

app = FastAPI(
    title="Synthetic Transaction Simulator API",
    description="HTTP interface for synthetic transaction simulation.",
    version="0.1.0",
)


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------


@app.get(
    "/health",
    tags=["health"],
    summary="Check API health",
)
def health_check() -> dict[str, str]:
    """Return the API process health status."""
    return {"status": "ok"}


# ------------------------------------------------------------------
# Summary helper
# ------------------------------------------------------------------


def _build_summary(
    transactions: list[Transaction],
) -> SimulationSummary:
    """Calculate summary statistics from generated transactions."""

    legitimate_count = sum(txn.fraud_label == 0 for txn in transactions)
    fraudulent_count = sum(txn.fraud_label == 1 for txn in transactions)

    fraud_type_counts = Counter(
        txn.fraud_type
        for txn in transactions
        if txn.fraud_label == 1 and txn.fraud_type
    )

    return SimulationSummary(
        unique_accounts=len({txn.account_id for txn in transactions}),
        unique_merchants=len({txn.merchant_id for txn in transactions}),
        legitimate_count=legitimate_count,
        fraudulent_count=fraudulent_count,
        fraud_type_counts=dict(fraud_type_counts),
        first_timestamp=(transactions[0].timestamp if transactions else None),
        last_timestamp=(transactions[-1].timestamp if transactions else None),
    )


# ------------------------------------------------------------------
# Simulation endpoint
# ------------------------------------------------------------------


@app.post(
    "/simulate",
    response_model=SimulationResponse,
    tags=["simulation"],
    summary="Generate synthetic transactions",
)
def simulate(
    request: SimulationRequest,
) -> SimulationResponse:
    """Generate a finite batch of synthetic transactions."""

    # 1. Validate the scenario before generating anything.
    try:
        get_scenario(request.scenario)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scenario: {request.scenario}",
        ) from exc

    # 2. Create an independent simulator for this request.
    generator = TransactionGenerator(seed=request.seed)

    # 3. Generate exactly the requested number of transactions.
    transactions = generator.generate_stream(
        n=request.count,
        scenario_name=request.scenario,
        start_time=request.start_time,
    )

    # 4. Calculate statistics from the actual output.
    summary = _build_summary(transactions)

    # 5. Include transaction details only when requested.
    transaction_responses = None

    if request.include_transactions:
        transaction_responses = [
            TransactionResponse.model_validate(txn.to_dict()) for txn in transactions
        ]

    # 6. Return the documented response model.
    return SimulationResponse(
        scenario=request.scenario,
        requested_count=request.count,
        generated_count=len(transactions),
        seed=request.seed,
        summary=summary,
        transactions=transaction_responses,
    )
