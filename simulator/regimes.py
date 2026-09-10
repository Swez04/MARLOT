
"""Each regime defines which fraud types are "active" and roughly how much
of the traffic they should account for."""

REGIMES = {
    "normal": {},
    "amount_spike": {"amount_spike": 0.05},
    "velocity": {"velocity": 0.05},
    "geo_hop": {"geo_hop": 0.05},
    "collusion": {"collusion": 0.03, "amount_spike": 0.02},
    "mixed_unseen": {"velocity": 0.03, "geo_hop": 0.03, "collusion": 0.02},
}
