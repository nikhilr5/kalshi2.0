"""
Live crypto top-of-book via Coinbase websocket level2 channel.

Free, no API key required. Subscribes to the level2 orderbook channel
which fires on every book change (50-100+ updates/sec for BTC-USD).
Maintains a local book and fires a callback with best bid/ask on each update.

Also subscribes to ticker for last trade price.

Runs on a background daemon thread with its own asyncio event loop.

Usage:
    feed = CryptoPriceFeed(on_price_callback, product_id="BTC-USD")
    feed.start()
    # callback fires as: on_price(price, bid, ask)
    feed.stop()
"""

import asyncio
import json
import threading
import websockets
from collections.abc import Callable
from sortedcontainers import SortedDict


class CryptoPriceFeed:

    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, on_price: Callable, product_id: str = "BTC-USD",
                 on_stale: Callable | None = None):
        """
        Args:
            on_price: callback(price, bid, ask) fired on every top-of-book change
            product_id: Coinbase product ID, e.g. "BTC-USD", "ETH-USD", "SOL-USD"
            on_stale: optional callback() fired when no data for STALE_THRESHOLD
                seconds.  Trading apps use this as a kill switch.
        """
        self.on_price = on_price
        self.on_stale = on_stale
        self.product_id = product_id
        self.ws = None
        self._thread = None
        self._loop = None
        self._running = False
        self.last_price = 0.0
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_update_ts: float = 0.0  # timestamp of last data received

        # Local orderbook — SortedDict for fast best bid/ask lookup
        # bids: price -> size (sorted ascending, best bid = last key)
        # asks: price -> size (sorted ascending, best ask = first key)
        self._bids = SortedDict()
        self._asks = SortedDict()

    async def _connect_and_listen(self):
        """Connect to Coinbase and stream level2 + ticker."""
        while self._running:
            try:
                self.ws = await websockets.connect(
                    self.WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                )

                subscribe = {
                    "type": "subscribe",
                    "product_ids": [self.product_id],
                    "channels": ["level2", "ticker"],
                }
                await self.ws.send(json.dumps(subscribe))
                print(f"[Price] Connected, subscribing to level2+ticker for {self.product_id}")

                # Start staleness monitor
                stale_task = asyncio.ensure_future(self._stale_monitor())

                async for message in self.ws:
                    if not self._running:
                        break

                    import time as _time
                    self.last_update_ts = _time.time()
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "snapshot":
                        self._handle_snapshot(data)
                    elif msg_type == "l2update":
                        self._handle_l2update(data)
                    elif msg_type == "ticker" and "price" in data:
                        self.last_price = float(data["price"])
                        # Also fire from ticker if L2 hasn't provided book yet
                        if not self._bids:
                            bid = float(data.get("best_bid", 0) or 0)
                            ask = float(data.get("best_ask", 0) or 0)
                            if bid > 0:
                                self.last_bid = bid
                            if ask > 0:
                                self.last_ask = ask
                            try:
                                self.on_price(self.last_price, self.last_bid, self.last_ask)
                            except Exception as e:
                                print(f"[Price] Ticker callback error: {e}")

            except websockets.ConnectionClosed:
                pass
            except Exception as e:
                print(f"[Price:{self.product_id}] Error: {e}")
            finally:
                try:
                    stale_task.cancel()
                except Exception:
                    pass
                try:
                    if self.ws:
                        await self.ws.close()
                except Exception:
                    pass

            # Reset book on reconnect
            self._bids.clear()
            self._asks.clear()

            if self._running:
                await asyncio.sleep(3)

    def _handle_snapshot(self, data: dict):
        """Process the initial level2 snapshot — full orderbook."""
        self._bids.clear()
        self._asks.clear()
        for price_str, size_str in data.get("bids", []):
            price = float(price_str)
            size = float(size_str)
            if size > 0:
                self._bids[price] = size
        for price_str, size_str in data.get("asks", []):
            price = float(price_str)
            size = float(size_str)
            if size > 0:
                self._asks[price] = size
        print(f"[Price] L2 snapshot: {len(self._bids)} bids, {len(self._asks)} asks")
        self._fire_update()

    def _handle_l2update(self, data: dict):
        """Process an incremental level2 update."""
        for side, price_str, size_str in data.get("changes", []):
            price = float(price_str)
            size = float(size_str)
            book = self._bids if side == "buy" else self._asks
            if size <= 0:
                book.pop(price, None)
            else:
                book[price] = size
        self._fire_update()

    def _fire_update(self):
        """Extract best bid/ask — only fire callback if top-of-book changed."""
        bid = self._bids.keys()[-1] if self._bids else 0.0
        ask = self._asks.keys()[0] if self._asks else 0.0

        # Only fire if best bid or ask actually changed
        if bid == self.last_bid and ask == self.last_ask:
            return

        if bid > 0:
            self.last_bid = bid
        if ask > 0:
            self.last_ask = ask
        # Use midpoint as price if no last trade yet
        price = self.last_price
        if price <= 0 and bid > 0 and ask > 0:
            price = (bid + ask) / 2.0
        self.last_price = price
        try:
            self.on_price(price, self.last_bid, self.last_ask)
        except Exception as e:
            print(f"[Price] Callback error: {e}")

    async def _stale_monitor(self):
        """Force reconnect if no data for 15s (Coinbase normally updates many times/sec)."""
        import time
        _STALE_THRESHOLD = 15
        while self._running:
            await asyncio.sleep(5)
            if self.last_update_ts > 0 and (time.time() - self.last_update_ts) > _STALE_THRESHOLD:
                print(f"[Price] STALE — no data for {time.time() - self.last_update_ts:.0f}s, forcing reconnect")
                # Notify the app BEFORE we drop the socket so it can
                # cancel resting orders while still online.
                if self.on_stale:
                    try:
                        self.on_stale()
                    except Exception as e:
                        print(f"[Price] on_stale callback error: {e}")
                self.last_update_ts = 0.0
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                break

    def start(self):
        """Start the price feed on a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Gracefully stop the feed."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._schedule_close)

    def _schedule_close(self):
        """Schedule websocket close from the event loop thread."""
        if self.ws:
            asyncio.ensure_future(self._safe_close(), loop=self._loop)

    async def _safe_close(self):
        """Close websocket, catching any errors."""
        try:
            await self.ws.close()
        except Exception:
            pass

    def _run_loop(self):
        """Background thread entry point."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except RuntimeError:
            pass
        except Exception:
            pass
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
