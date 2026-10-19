from datetime import datetime

import pytest
from pydantic import ValidationError

from simulator.api.schemas import SimulationRequest


def test_valid_request():
    request = SimulationRequest(
        scenario="normal",
        count=1000,
        seed=42,
        start_time="2026-10-20T10:00:00",
        include_transactions=True,
    )

    assert request.scenario == "normal"
    assert request.count == 1000
    assert request.seed == 42
    assert request.start_time == datetime(2026, 10, 20, 10, 0, 0)
    assert request.include_transactions is True


def test_optional_fields_use_defaults():
    request = SimulationRequest(
        scenario="normal",
        count=10,
    )

    assert request.seed is None
    assert request.start_time is None
    assert request.include_transactions is False


@pytest.mark.parametrize("count", [-1, 10_001])
def test_invalid_count_is_rejected(count):
    with pytest.raises(ValidationError):
        SimulationRequest(
            scenario="normal",
            count=count,
        )


def test_missing_required_field_is_rejected():
    with pytest.raises(ValidationError):
        SimulationRequest(scenario="normal")


def test_invalid_datetime_is_rejected():
    with pytest.raises(ValidationError):
        SimulationRequest(
            scenario="normal",
            count=10,
            start_time="not-a-date",
        )


def test_empty_scenario_is_rejected():
    with pytest.raises(ValidationError):
        SimulationRequest(
            scenario="",
            count=10,
        )


def test_unexpected_field_is_rejected():
    with pytest.raises(ValidationError):
        SimulationRequest(
            scenario="normal",
            count=10,
            scenaro="normal",
        )
