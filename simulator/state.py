from collections import defaultdict, deque
from dataclasses import dataclass, field

from simulator.transaction import Transaction

@dataclass
class AccountState:
    transactions: deque = field(default_factory=lambda: deque(maxlen=100))
    devices: set[str] = field(default_factory=set)
    merchants: set[str] = field(default_factory=set)
    locations: deque = field(default_factory=lambda: deque(maxlen=20))

@dataclass
class MerchantState:
    accounts: set[str] = field(default_factory=set)
    transaction_count: int = 0
    total_amount: float = 0.0


class SimulatorState:
    def __init__(self):
        self.accounts: dict[str, AccountState] = defaultdict(AccountState)
        self.merchants: dict[str, MerchantState] = defaultdict(MerchantState)


    def update(self, txn: Transaction) -> None:
        account = self.accounts[txn.account_id]
        merchant = self.merchants[txn.merchant_id]
        
        account.transactions.append(txn)
        account.devices.add(txn.device_id)
        account.merchants.add(txn.merchant_id)
        account.locations.append(txn.location)
        
        merchant.accounts.add(txn.account_id)
        merchant.transaction_count += 1
        merchant.total_amount += txn.amount
        
    
    def get_account(self, account_id: str) -> AccountState:
        return self.accounts[account_id]
    
    def get_merchant(self, merchant_id: str) -> MerchantState:
        return self.merchants[merchant_id]
    
    def recent_transactions(self, account_id: str) -> deque:
        return self.accounts[account_id].transactions
    
    def known_merchants(self, account_id: str) -> set[str]:
        return self.accounts[account_id].merchants
    
    def merchant_accounts(self, merchant_id: str) -> set[str]:
        return self.merchants[merchant_id].accounts
    
    
