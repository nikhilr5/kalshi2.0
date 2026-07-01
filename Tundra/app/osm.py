"""OSM — order state manager for the weather-floor app.

Strategy marks a bucket DEAD (provably worth $0) and calls want_short(). OSM's
job: get us short `size` of that bucket by HITTING THE BID (sell YES into the
best yes bid, IOC, priced 1c through the bid so it's marketable). No resting
offers, no chase -- guaranteed fills, simplest to validate (this is the chosen
mechanic; the passive-chase variant is intentionally not built).

Single source of truth for position = websocket fills (on_fill). An in-flight
guard + per-ticker cooldown prevent double-sends while a fill round-trips.
Thread-safe: state mutated under a lock; orders sent via the api's async pool.
"""
import threading
import time

_MIN_PX = 0.01            # weather markets trade in whole cents


class OSM:
    def __init__(self, api, log=print):
        self.api = api
        self.log = log
        self.armed = False
        self.size = 1                       # target short per dead bucket (configurable)
        self.max_position = 10              # global cap on total contracts short
        self.cooldown_sec = 2.0             # min seconds between sends on the same ticker
        self._lock = threading.Lock()
        self.pos: dict[str, int] = {}       # ticker -> net YES position (negative = short)
        self._pending: dict[str, int] = {}  # ticker -> reserved in-flight count (also the in-flight guard)
        self._cooldown: dict[str, float] = {}   # ticker -> monotonic ts of last send
        self.fills_log: list[dict] = []     # (ts, ticker, price, count) for the UI

    # ---- config ----
    def set_armed(self, on: bool):
        with self._lock:
            self.armed = bool(on)
        self.log(f"[OSM] {'ARMED' if on else 'disarmed'}")

    def set_size(self, n: int):
        with self._lock:
            self.size = max(1, int(n))

    def set_max_position(self, n: int):
        with self._lock:
            self.max_position = max(0, int(n))

    def set_cooldown(self, secs: float):
        with self._lock:
            self.cooldown_sec = max(0.0, float(secs))

    def _total_short_locked(self) -> int:
        return sum(-p for p in self.pos.values() if p < 0) + sum(self._pending.values())

    def total_short(self) -> int:
        with self._lock:
            return self._total_short_locked()

    def seed_positions(self, positions: list):
        """Seed net positions from REST (so a restart doesn't re-sell)."""
        with self._lock:
            for p in positions or []:
                tk = p.get("ticker")
                if tk:
                    self.pos[tk] = int(p.get("position", 0))

    def position(self, ticker: str) -> int:
        with self._lock:
            return self.pos.get(ticker, 0)

    # ---- fills (from websocket; the truth) ----
    def on_fill(self, ticker, action, side, price, count):
        if not ticker or count <= 0:
            return
        signed = count if side == "yes" else -count
        if action == "sell":
            signed = -signed
        with self._lock:
            self.pos[ticker] = self.pos.get(ticker, 0) + signed
            self.fills_log.append({"ts": time.time(), "ticker": ticker,
                                   "action": action, "side": side,
                                   "price": price, "count": count,
                                   "pos": self.pos[ticker]})
        self.log(f"[FILL] {ticker} {action} {side} x{count} @ {price:.2f} -> pos {self.pos[ticker]}")

    # ---- the trade decision ----
    def want_short(self, ticker: str, yes_bid: float, bid_size: int):
        """Dead bucket: if armed and not yet short `size`, hit the bid for the
        remaining count -- capped by the global max-position headroom. Called
        frequently (on book updates) -- cheap + guarded."""
        with self._lock:
            if not self.armed or yes_bid <= 0 or ticker in self._pending:
                return
            if time.monotonic() - self._cooldown.get(ticker, 0.0) < self.cooldown_sec:
                return
            need = self.size + self.pos.get(ticker, 0)      # want pos == -size
            if need <= 0:
                return
            headroom = self.max_position - self._total_short_locked()    # global risk cap
            if headroom <= 0:
                return
            count = min(need, headroom)
            if bid_size > 0:
                count = min(count, bid_size)
            if count <= 0:
                return
            self._pending[ticker] = count                   # reserve so concurrent fires don't overshoot
            self._cooldown[ticker] = time.monotonic()
        price = max(_MIN_PX, round(yes_bid - 0.01, 2))       # 1c through the bid -> marketable
        self.log(f"[SELL] {ticker} x{count} @ {price:.2f} (bid {yes_bid:.2f})")
        fut = self.api.create_order_async(
            ticker=ticker, side="yes", action="sell",
            price_dollars=f"{price:.2f}", count=count,
            time_in_force="immediate_or_cancel", tag="floor")
        fut.add_done_callback(lambda f, tk=ticker: self._on_response(tk, f))

    def _on_response(self, ticker: str, fut):
        try:
            resp = fut.result()
            sc = resp.get("status_code")
            if sc != 201:
                self.log(f"[SELL REJECT] {ticker} status={sc} {resp.get('message', '')}")
        except Exception as e:
            self.log(f"[SELL ERROR] {ticker} {e}")
        finally:
            with self._lock:
                self._pending.pop(ticker, None)   # release reservation; position updates via on_fill
