"""OrderGateway — exchange-interaction policy between OSM and KalshiAPI.

    Strategy2 -> OSM -> OrderGateway -> KalshiAPI -> Kalshi
    (pricing)   (state)  (policy)        (transport)

Owns the write-token budget (TokenBucket) and response normalization.
Creates are gated by the budget — a create the bucket can't afford is
skipped locally (returns None) instead of earning a 429 and a blackout.
Cancels debit the budget but are never blocked: pulling a quote is risk
management and always goes out.

Threading: place()/cancel()/on_rate_limited() must be called from a
single thread (the OSM worker) — the TokenBucket is lock-free by the
single-owner rule.  Callbacks fire on api executor threads and must not
touch the bucket (they just enqueue to OSM, same as before).

Lives for the whole process (app.py owns it), so the budget survives
the per-market OSM teardown/rebuild at every 15-minute roll.
"""

from kalshi_api import _CREATE_TOKEN_COST, _CANCEL_TOKEN_COST
from token_bucket import TokenBucket


class OrderGateway:

    def __init__(self, api):
        self.api = api
        self.budget = TokenBucket()

    # ------------------------------------------------------------------
    # Response normalization — one envelope shape no matter what happened
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(future) -> dict:
        """Future result -> envelope.  Network exceptions become
        {"status_code": 0, "message": ...} so consumers branch on one
        shape."""
        try:
            return future.result()
        except Exception as e:
            return {"status_code": 0,
                    "message": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def place(self, callback, **order_kwargs):
        """Budget-gated create.  Returns the Future if sent, None if the
        bucket can't afford it (caller skips; next tick retries).
        `callback` receives the normalized envelope when the response
        lands."""
        if not self.budget.spend(_CREATE_TOKEN_COST):
            return None
        f = self.api.create_order_async(**order_kwargs)
        f.add_done_callback(lambda fut: callback(self._normalize(fut)))
        return f

    def cancel(self, order_id, callback):
        """Cancel — debits the budget but always sends."""
        self.budget.spend(_CANCEL_TOKEN_COST)
        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(lambda fut: callback(self._normalize(fut)))
        return f

    # ------------------------------------------------------------------
    # Feedback + observability
    # ------------------------------------------------------------------
    def on_rate_limited(self):
        """A real 429 means the local model overestimated the balance —
        snap it to empty so it re-syncs with reality."""
        self.budget.drain()

    def tokens_remaining(self) -> float:
        return self.budget.remaining()
