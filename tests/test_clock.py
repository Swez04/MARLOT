from datetime import datetime, timedelta

import pytest

from simulator.clock import SimulationClock


def test_clock_starts_at_given_time():
    start = datetime(2026, 1, 1, 10, 0, 0)
    clock = SimulationClock(start)

    assert clock.current_time == start


def test_clock_advances_by_positive_duration():
    start = datetime(2026, 1, 1, 10, 0, 0)
    clock = SimulationClock(start)

    result = clock.advance_by(timedelta(seconds=2.5))

    assert result == start + timedelta(seconds=2.5)
    assert clock.current_time == result


@pytest.mark.parametrize(
    "delta",
    [
        timedelta(0),
        timedelta(seconds=-1),
    ],
)
def test_clock_rejects_non_positive_duration(delta):
    clock = SimulationClock(datetime(2026, 1, 1))

    with pytest.raises(ValueError):
        clock.advance_by(delta)


def test_clock_rejects_non_timedelta():
    clock = SimulationClock(datetime(2026, 1, 1))

    with pytest.raises(TypeError):
        clock.advance_by(5)


def test_clock_rejects_non_datetime_start():
    with pytest.raises(TypeError):
        SimulationClock("2026-01-01")
