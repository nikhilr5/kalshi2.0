"""
Live CF Benchmarks index via Kalshi's `cfbenchmarks_value` WS channel.

This is the index Kalshi actually settles its 15-min crypto contracts on
(e.g. ETHUSD_RTI for ETH), NOT an exchange spot price.  Pricing theo off
this removes the Coinbase-vs-index basis (±$1-3, sign-flipping) that
mis-prices near-ATM contracts and hands you false-conviction trades.

Drop-in for CryptoPriceFeed: same `on_price(price, bid, ask)` callback and
start()/stop() lifecycle.  Two differences:
  * needs a KalshiAPI for the WS-handshake auth, and
  * keyed off the same Coinbase `product_id` (mapped series->product->index),
    so existing call sites only add the `api` arg.

The index has no order book, so bid == ask == price.  The raw per-second
index value is emitted as `price` (most current); the trailing 60s average
(what settlement actually uses) is kept on `last_avg60` for later
settlement-exact modelling.

Usage:
    feed = CFBenchmarksFeed(on_price, "ETH-USD", api)
    feed.start(); ...; feed.stop()
"""

import asyncio
import json
import threading
import time
import websockets
from collections.abc import Callable


# Coinbase product id -> CF Benchmarks real-time index id.
# (Series -> product lives in recorder.SERIES_TO_PRODUCT; this is the last hop.)
PRODUCT_TO_INDEX = {
    "ETH-USD": "ETHUSD_RTI",
    "BTC-USD": "BRTI",
}


class CFBenchmarksFeed:

    WS_URL  = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
    WS_PATH = "/trade-api/ws/v2"          # path used for the auth signature
    STALE_THRESHOLD = 15.0                # seconds with no value -> on_stale

    def __init__(self, on_price: Callable, product_id: str, api,
                 on_stale: Callable | None = None):
        self.on_price = on_price
        self.on_stale = on_stale
        self.api = api
        self.product_id = product_id
        self.index_id = PRODUCT_TO_INDEX.get(product_id)
        if not self.index_id:
            raise ValueError(
                f"No CF index mapped for product {product_id!r}; "
                f"known: {list(PRODUCT_TO_INDEX)}")
        self.ws = None
        self._thread = None
        self._loop = None
        self._running = False
        self.last_price: float = 0.0
        self.last_avg60: float = 0.0
        # The quarter-hour close average — the exact settlement value.
        # Only sent in the final minute before :00/:15/:30/:45, so we keep a
        # timestamp; `close_avg_if_fresh()` returns it only while it's live.
        self.last_close_avg: float | None = None
        self.last_close_avg_ts: float = 0.0
        self.last_update_ts: float = 0.0
        self._stale_fired = False

    # ------------------------------------------------------------------
    # Lifecycle (mirrors CryptoPriceFeed)
    # ------------------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"cf-{self.index_id}")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._schedule_close)

    def _schedule_close(self):
        if self.ws:
            asyncio.ensure_future(self._safe_close(), loop=self._loop)

    async def _safe_close(self):
        try:
            await self.ws.close()
        except Exception:
            pass

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        delay = 2
        while self._running:
            try:
                self._loop.run_until_complete(self._connect_and_listen())
            except Exception as e:
                print(f"[CF] {self.index_id} error: {e}")
            if not self._running:
                break
            print(f"[CF] {self.index_id} disconnected, reconnecting in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)

    # ------------------------------------------------------------------
    # Connection + message handling
    # ------------------------------------------------------------------
    async def _connect_and_listen(self):
        headers = self.api.ws_auth_headers(self.WS_PATH)
        async with websockets.connect(self.WS_URL,
                                      additional_headers=headers) as ws:
            self.ws = ws
            await ws.send(json.dumps({
                "id": 1, "cmd": "subscribe",
                "params": {"channels": ["cfbenchmarks_value"],
                           "index_ids": [self.index_id]},
            }))
            stale_task = asyncio.ensure_future(self._stale_monitor())
            try:
                async for raw in ws:
                    if not self._running:
                        break
                    try:
                        self._handle(raw)
                    except Exception as e:
                        print(f"[CF] {self.index_id} parse error: {e}")
            finally:
                stale_task.cancel()

    def _handle(self, raw: str):
        d = json.loads(raw)
        t = d.get("type")
        if t == "error":
            print(f"[CF] {self.index_id} server error: {d.get('msg')}")
            return
        if t != "cfbenchmarks_value":
            return                                  # subscribed / control frames
        m = d.get("msg", {})
        if m.get("index_id") != self.index_id:
            return

        # raw per-second value is in the nested CF frame (`data`, a JSON string)
        val = None
        data = m.get("data")
        if isinstance(data, str):
            try:
                val = float(json.loads(data).get("value"))
            except Exception:
                val = None

        a60 = (m.get("avg_60s_data") or {}).get("value")
        if a60 is not None:
            try:
                self.last_avg60 = float(a60)
            except (TypeError, ValueError):
                pass

        # Quarter-hour close average — present only in the final minute.
        # Field may be a scalar or an object with a `value`.
        w15 = m.get("last_60s_windowed_average_15min")
        if w15 is not None:
            raw15 = w15.get("value") if isinstance(w15, dict) else w15
            try:
                self.last_close_avg = float(raw15)
                self.last_close_avg_ts = time.time()
            except (TypeError, ValueError):
                pass

        if val is None:                              # fall back to the 60s avg
            val = self.last_avg60 or self.last_price
        if not val:
            return

        self.last_price = val
        self.last_update_ts = time.time()
        self._stale_fired = False
        try:
            self.on_price(val, val, val)             # index has no bid/ask
        except Exception as e:
            print(f"[CF] {self.index_id} callback error: {e}")

    def close_avg_if_fresh(self, max_age: float = 2.5) -> float | None:
        """The settlement close-average, but only while it's actually being
        sent (the final minute).  None otherwise.  Display-only."""
        if (self.last_close_avg is not None
                and time.time() - self.last_close_avg_ts < max_age):
            return self.last_close_avg
        return None

    async def _stale_monitor(self):
        while self._running:
            await asyncio.sleep(5)
            if (self.last_update_ts
                    and time.time() - self.last_update_ts > self.STALE_THRESHOLD
                    and not self._stale_fired):
                self._stale_fired = True
                print(f"[CF] {self.index_id} STALE — no value for "
                      f"{time.time() - self.last_update_ts:.0f}s")
                if self.on_stale:
                    try:
                        self.on_stale()
                    except Exception:
                        pass
