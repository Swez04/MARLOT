"""Central clock for the synthetic transaction simulator."""

from datetime import datetime, timedelta


class SimulationClock:
    """Maintain the current time in a simulated world."""

    def __init__(self, start_time: datetime) -> None:
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime")

        self._current_time = start_time

    @property
    def current_time(self) -> datetime:
        """Return the current simulation timestamp."""
        return self._current_time

    def advance_by(self, delta: timedelta) -> datetime:
        """Advance simulation time by a strictly positive duration."""
        if not isinstance(delta, timedelta):
            raise TypeError("delta must be a timedelta")

        if delta <= timedelta(0):
            raise ValueError("delta must be strictly positive")

        self._current_time += delta
        return self._current_time
