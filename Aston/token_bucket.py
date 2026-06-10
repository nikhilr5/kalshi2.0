"""Local mirror of Kalshi's write-token bucket.

Kalshi gives no remaining-tokens header — the only way to know the
balance is to model it: capacity 600, refill 300/s (Advanced tier),
debit per write at the real endpoint costs (create=100, cancel=20).

Single-owner design, NO lock: all spend() calls must come from one
thread (the OSM worker — _send_place/_send_cancel run there).  Reads
via remaining() from other threads (UI display) are safe — a single
float attribute read, worst case microscopically stale.

drain() snaps the mirror to empty; call it on a real 429 so the model
re-syncs to reality instead of accumulating drift.
"""

import time


class TokenBucket:

    def __init__(self, capacity: float = 600.0, refill_rate: float = 300.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._balance = capacity
        self._ts = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        self._balance = min(self.capacity,
                            self._balance + (now - self._ts) * self.refill_rate)
        self._ts = now

    def spend(self, cost: float) -> bool:
        """Refill, then debit `cost` if affordable.  Returns False (and
        debits nothing) when the balance can't cover it.  Owner thread only."""
        self._refill()
        if self._balance < cost:
            return False
        self._balance -= cost
        return True

    def remaining(self) -> float:
        """Current balance estimate.  Read-only — mutates nothing, so it
        is safe to call from any thread (e.g. the UI) without breaking
        the single-owner rule on spend()/drain()."""
        now = time.monotonic()
        return min(self.capacity,
                   self._balance + (now - self._ts) * self.refill_rate)

    def drain(self):
        """Snap to empty — call when Kalshi returns a real 429 so the
        mirror re-syncs to the truth."""
        self._balance = 0.0
        self._ts = time.monotonic()
