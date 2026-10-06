import pytest

from simulator.scenarios import get_scenario


KNOWN_SCENARIOS = [
    "normal",
    "amount_spike_attack",
    "velocity_attack",
    "geo_attack",
    "collusion_attack",
    "mixed",
]


@pytest.mark.parametrize("scenario_name", KNOWN_SCENARIOS)
def test_known_scenario_can_be_loaded(scenario_name):
    scenario = get_scenario(scenario_name)

    assert scenario is not None
    assert scenario.name == scenario_name


def test_unknown_scenario_raises_value_error():
    with pytest.raises(ValueError):
        get_scenario("does_not_exist")


@pytest.mark.parametrize("scenario_name", KNOWN_SCENARIOS)
def test_scenario_has_event_weights(scenario_name):
    scenario = get_scenario(scenario_name)

    assert scenario.event_weights
    assert isinstance(scenario.event_weights, dict)


@pytest.mark.parametrize("scenario_name", KNOWN_SCENARIOS)
def test_scenario_weights_are_non_negative(scenario_name):
    scenario = get_scenario(scenario_name)

    assert all(
        weight >= 0
        for weight in scenario.event_weights.values()
    )


@pytest.mark.parametrize("scenario_name", KNOWN_SCENARIOS)
def test_scenario_weights_sum_to_valid_probability_mass(scenario_name):
    scenario = get_scenario(scenario_name)

    assert sum(scenario.event_weights.values()) > 0
    assert sum(scenario.event_weights.values()) <= 1.0


def test_normal_scenario_has_only_legitimate_event():
    scenario = get_scenario("normal")

    assert set(scenario.event_weights.keys()) == {"legitimate"}


def test_velocity_attack_scenario_contains_velocity_event():
    scenario = get_scenario("velocity_attack")

    assert "velocity" in scenario.event_weights


def test_geo_attack_scenario_contains_geo_hop_event():
    scenario = get_scenario("geo_attack")

    assert "geo_hop" in scenario.event_weights


def test_collusion_attack_scenario_contains_collusion_event():
    scenario = get_scenario("collusion_attack")

    assert "collusion" in scenario.event_weights
