from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import numpy as np
import random

from simulator.transaction import Transaction
from simulator.state import SimulatorState


LOCATIONS = ["US-CA", "US-NY", "US-TX", "GB-LON", "DE-BER", "IN-BLR", "SG-SIN"]
CATEGORIES = ["grocery", "electronics", "travel", "restaurant", "subscription", "jewelry"]

GEO_HOPS = {
    "US-CA": ["GB-LON", "DE-BER", "IN-BLR", "SG-SIN"],
    "US-NY": ["GB-LON", "DE-BER", "IN-BLR", "SG-SIN"],
    "US-TX": ["GB-LON", "DE-BER", "IN-BLR", "SG-SIN"],
    "GB-LON": ["US-CA", "US-NY", "IN-BLR", "SG-SIN"],
    "DE-BER": ["US-CA", "US-NY", "IN-BLR", "SG-SIN"],
    "IN-BLR": ["US-CA", "US-NY", "GB-LON", "DE-BER"],
    "SG-SIN": ["US-CA", "US-NY", "GB-LON", "DE-BER"],
}

class Event(ABC):
    """ Base class for a single simulator event. """
    
    def __init__(self, start_time: datetime, transaction_count: int, rng: random.Random, state: SimulatorState):
        self.rng = rng
        self.state = state
        self.start_time = start_time
        self.current_time = start_time
        self.transaction_count = transaction_count
        
        self.transactions_generated = 0
        
    @property
    def is_complete(self) -> bool:
        return self.transactions_generated >= self.transaction_count
    
    @abstractmethod
    def next_transaction(self) -> Transaction:
        """ Generate the next transaction belonging to this event.
            Subclasses must implement the event-specific behavior. """
        pass
    
    def _random_transaction_id(self) -> str:
        return f"txn_{self.rng.getrandbits(64):016x}"
        
    def _random_account(self) -> str:
        return f"acct_{self.rng.randint(1, 500):04d}"

    def _random_merchant(self) -> str:
        return f"merch_{self.rng.randint(1, 200):03d}"

    def _random_device(self) -> str:
        return f"device_{self.rng.randint(1, 1000):04d}"

        
    
class LegitimateEvent(Event):
    
    def next_transaction(self) -> Transaction:
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self._random_account(),
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self.start_time,
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=0,
            fraud_type="",
        )


class AmountSpikeEvent(Event):
    
    def next_transaction(self) -> Transaction:
        if self.state.accounts:
            account = self.rng.choice(list(self.state.accounts.keys()))
            txns = self.state.recent_transactions(account)
        else:
            account = self._random_account()
            txns = []
            
        
        txn = Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=account,
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self.current_time,
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="amount_spike",
        )
        
        if len(txns) >= 25:                                  # ensure sufficient account history for historical baseline
            amounts = [txn.amount for txn in txns]
            baseline_amount = np.percentile(amounts, 75)        # set the baseline amount to 75th percentile
            txn.amount = round(baseline_amount * self.rng.uniform(10, 50), 2)       # 10-50x amount spike
        else:
            txn.amount = round(txn.amount * self.rng.uniform(10, 50), 2)
        
        self.current_time += timedelta(minutes=2)
        return txn


class VelocityEvent(Event):
    def __init__(self, start_time, transaction_count, rng, state):
        super().__init__(start_time, transaction_count, rng, state)
                
        if self.state.accounts:
            self.account = self.rng.choice(list(self.state.accounts.keys()))
        else:
            self.account = self._random_account()
        
    
    def next_transaction(self) -> Transaction:
        self.current_time += timedelta(seconds=self.rng.randint(1, 30))
        
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self.account,
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self.current_time,
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="velocity",
        )

        
class GeoHopEvent(Event):
    def __init__(self, start_time, transaction_count, rng, state):
        super().__init__(start_time, transaction_count, rng, state)
                
        if self.state.accounts:
            self.account = self.rng.choice(list(self.state.accounts.keys()))
        else:
            self.account = self._random_account()
        
        self.previous_location = None        
        
    def next_transaction(self) -> Transaction:
        if self.previous_location is None:
            location = self.rng.choice(LOCATIONS)

        else:
            location = self.rng.choice(GEO_HOPS[self.previous_location])
            
        self.previous_location = location
        
        self.current_time += timedelta(seconds=self.rng.randint(30, 120))
        
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=self.account,
            merchant_id=self._random_merchant(),
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self.current_time,
            location=location,
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="geo_hop",
        )
        

class CollusionEvent(Event):
    def __init__(self, start_time, transaction_count, rng, state):
        super().__init__(start_time, transaction_count, rng, state)
        
        if self.state.merchants:
            self.merchant = self.rng.choice(list(self.state.merchants.keys()))
        else:
            self.merchant = self._random_merchant()

        if self.state.accounts:
            accounts = list(self.state.accounts.keys())
            num_accounts = min(self.rng.randint(2, 5), len(accounts))
            self.accounts = self.rng.sample(accounts, num_accounts)
        else:
            self.accounts = [self._random_account() for _ in range(self.rng.randint(2, 5))]

    
    def next_transaction(self) -> Transaction:
        account = self.rng.choice(self.accounts)
        
        self.current_time += timedelta(seconds=self.rng.randint(10, 60))
        
        return Transaction(
            transaction_id=self._random_transaction_id(),
            account_id=account,
            merchant_id=self.merchant,
            amount=round(self.rng.lognormvariate(3.2, 0.6), 2),
            timestamp=self.current_time,
            location=self.rng.choice(LOCATIONS),
            merchant_category=self.rng.choice(CATEGORIES),
            device_id=self._random_device(),
            fraud_label=1,
            fraud_type="collusion",
        )
        
