
"""Each regime defines which fraud types are "active" and roughly how much
of the traffic they should account for."""

from dataclasses import dataclass

@dataclass
class RegimeConfig:
    name: str
    fraud_mix: dict[str, float]
    
    def __post_init__(self) -> None:
        
        valid_frauds = frozenset({'amount_spike', 'velocity', 'geo_hop', 'collusion'})
        
        for fraud_type, rate in self.fraud_mix.items():
            if fraud_type not in valid_frauds:
                raise ValueError(f"Unknown fraud type {fraud_type}. "
                    f"Expected one of: {valid_frauds}")
                
            if rate < 0:
                raise ValueError(f"Fraud rate for {fraud_type} must be non-negative")
            
            if rate > 1:
                raise ValueError(f"Fraud rate for {fraud_type} cannot exceed 1")
            
        if sum(self.fraud_mix.values()) > 1:
            raise ValueError("Total fraud mix must not exceed 1")
        
    
REGIMES = {
    "normal": RegimeConfig('normal',{}),
    "amount_spike": RegimeConfig('amount_spike', {"amount_spike": 0.05}),
    "velocity": RegimeConfig('velocity', {"velocity": 0.05}),
    "geo_hop": RegimeConfig('geo_hop', {"geo_hop": 0.05}),
    "collusion": RegimeConfig('collusion', {"collusion": 0.03, "amount_spike": 0.02}),
    "mixed_unseen": RegimeConfig('mixed_unseen', {"velocity": 0.03, "geo_hop": 0.03, "collusion": 0.02}),
}
    
def get_regime(name: str) -> RegimeConfig:
    try:
        return REGIMES[name]
    except KeyError:
        available = ", ".join(sorted(REGIMES))
        raise ValueError(f"Unknown regime {name!r}. Available regimes: {available}"
        ) from None
    

if __name__ == "__main__":
    regime = RegimeConfig('mixed_unseen', {"velocity": 0.03, "geo_hop": 0.03, "collusion": 0.02})
    print(regime.name, regime.fraud_mix)
