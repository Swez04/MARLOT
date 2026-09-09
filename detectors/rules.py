"""
Simple rule-based / static-threshold detector.

This is the dumbest possible baseline: fixed thresholds on amount and
transaction velocity (count per account in a rolling window). Everything
else (Isolation Forest, autoencoder, RL) should beat this.
"""

from collections import defaultdict, deque
from simulator.transaction import Transaction

AMOUNT_THRESHOLD = 1500.0
VELOCITY_WINDOW_SECONDS = 60
VELOCITY_THRESHOLD = 4  # more than N txns per account in the window = flag


class RuleBasedDetector:
    def __init__(self, amount_threshold: float = AMOUNT_THRESHOLD,
                 velocity_window: int = VELOCITY_WINDOW_SECONDS,
                 velocity_threshold: int = VELOCITY_THRESHOLD):
        self.amount_threshold = amount_threshold
        self.velocity_window = velocity_window
        self.velocity_threshold = velocity_threshold
        self._history = defaultdict(deque)  # account_id -> deque[timestamp]

    def score(self, txn: Transaction) -> int:
        """Returns 1 (flag) or 0 (allow)."""
        flagged = 0

        # rule 1: large amount
        if txn.amount >= self.amount_threshold:
            flagged = 1

        # rule 2: velocity — too many transactions for this account recently
        hist = self._history[txn.account_id]
        hist.append(txn.timestamp)
        while hist and (txn.timestamp - hist[0]).total_seconds() > self.velocity_window:
            hist.popleft()
        if len(hist) > self.velocity_threshold:
            flagged = 1

        return flagged

    def score_batch(self, transactions: list[Transaction]) -> list[int]:
        return [self.score(t) for t in transactions]
