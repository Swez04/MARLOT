"""Event-selection configurations for the transaction simulator."""

from dataclasses import dataclass


VALID_EVENTS = frozenset({
    "legitimate",
    "amount_spike",
    "velocity",
    "geo_hop",
    "collusion",
})


@dataclass(frozen=True)
class Scenario:
    name: str
    event_weights: dict[str, float]

    def __post_init__(self) -> None:
        for event_type, weight in self.event_weights.items():

            if event_type not in VALID_EVENTS:
                raise ValueError(
                    f"Unknown event type {event_type!r}. "
                    f"Expected one of: {VALID_EVENTS}"
                )

            if weight < 0:
                raise ValueError(
                    f"Event weight for {event_type!r} must be non-negative"
                )

        if not self.event_weights:
            raise ValueError("Scenario must contain at least one event type")

        if sum(self.event_weights.values()) <= 0:
            raise ValueError(
                "At least one event weight must be greater than 0"
            )


SCENARIOS = {
    "normal": Scenario(
        name="normal",
        event_weights={
            "legitimate": 1.0,
        },
    ),

    "mixed": Scenario(
        name="mixed",
        event_weights={
            "legitimate": 0.95,
            "amount_spike": 0.01,
            "velocity": 0.01,
            "geo_hop": 0.01,
            "collusion": 0.02,
        },
    ),

    "velocity_attack": Scenario(
        name="velocity_attack",
        event_weights={
            "legitimate": 0.90,
            "velocity": 0.10,
        },
    ),

    "geo_attack": Scenario(
        name="geo_attack",
        event_weights={
            "legitimate": 0.90,
            "geo_hop": 0.10,
        },
    ),

    "collusion_attack": Scenario(
        name="collusion_attack",
        event_weights={
            "legitimate": 0.90,
            "collusion": 0.10,
        },
    ),

    "amount_spike_attack": Scenario(
        name="amount_spike_attack",
        event_weights={
            "legitimate": 0.90,
            "amount_spike": 0.10,
        },
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        available = ", ".join(sorted(SCENARIOS))
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {available}"
        ) from None
        

if __name__ == "__main__":
    scenario = get_scenario("mixed")

    print(scenario.name)
    print(scenario.event_weights)
