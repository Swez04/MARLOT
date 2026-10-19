from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    scenario: str = Field(min_length=1, max_length=64)
    count: int = Field(ge=0, le=10000)
    seed: int | None = None
    start_time: datetime | None = None
    include_transactions: bool = False


class TransactionResponse(BaseModel):
    """Public representation of a simulated transaction."""

    transaction_id: str
    account_id: str
    merchant_id: str
    amount: float
    timestamp: datetime
    location: str
    merchant_category: str
    device_id: str
    fraud_label: int
    fraud_type: str


class SimulationSummary(BaseModel):
    """Aggregated statistics for a simulation run."""

    unique_accounts: int
    unique_merchants: int
    legitimate_count: int
    fraudulent_count: int
    fraud_type_counts: dict[str, int] = Field(default_factory=dict)
    first_timestamp: datetime | None
    last_timestamp: datetime | None


class SimulationResponse(BaseModel):
    """Response returned after a simulation completes."""

    scenario: str
    requested_count: int
    generated_count: int
    seed: int | None
    summary: SimulationSummary
    transactions: list[TransactionResponse] | None = None
