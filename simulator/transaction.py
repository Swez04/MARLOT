from dataclasses import dataclass
from datetime import datetime


@dataclass
class Transaction:
    transaction_id: str
    account_id: str
    merchant_id: str
    amount: float
    timestamp: datetime
    location: str            # e.g. "US-CA", "US-NY", "GB-LON"
    merchant_category: str   # e.g. "grocery", "electronics", "travel"
    device_id: str
    fraud_label: int         # 0 = legit, 1 = fraud
    fraud_type: str          # "" if legit, else "amount_spike" / "velocity" / "geo_hop" / "collusion"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["timestamp"] = self.timestamp.isoformat()
        return d
