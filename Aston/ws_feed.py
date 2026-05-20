"""
Kalshi websocket orderbook feed.

Runs on a background daemon thread with its own asyncio event loop.
Subscribes to orderbook_delta channel for all tickers in one connection.
Maintains local orderbook state and fires a callback with YES bid/ask
on every snapshot or delta update.

Usage:
    feed = KalshiWsFeed(api, on_update_callback)
    feed.start(["KXBTC-26APR1017-B72000", ...])
    # callback fires as: on_update(ticker, yes_bid, yes_ask)
    feed.stop()  # graceful shutdown
"""

import asyncio
import json
import threading
import websockets
from collections.abc import Callable
from kalshi_api import KalshiAPI


# user_orders is hosted on a separate Kalshi WS host (external-api-ws.kalshi.com)
# but uses the SAME path as the main endpoint (/trade-api/ws/v2).  Earlier
# docs that named the path "/user_orders" were misleading — that's the
# channel name, not a URL path.  Probed empirically.
USER_ORDERS_WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


class KalshiWsFeed:

    def __init__(self, api: KalshiAPI, on_update: Callable,
                 on_fill: Callable | None = None,
                 on_fill_raw: Callable | None = None,
                 on_trade: Callable | None = None,
                 on_order_event: Callable | None = None,
                 on_book_change: Callable | None = None,
                 on_stale: Callable | None = None):
        """
        Args:
            api: KalshiAPI instance (used for auth headers and WS URL)
            on_update: callback(ticker, yes_bid, yes_ask, bid_size, ask_size)
                       fired on every TOB change (computed from raw deltas)
            on_fill: optional callback(ticker, action, side, price, count)
                     fired on every fill for this account
            on_fill_raw: optional callback(msg_dict) — same fills as
                         `on_fill` but passes the raw message so the
                         caller can pull `ts_ms`, `trade_id`, etc.  Used
                         by the recorder for authoritative match-time
                         timestamps.
            on_trade: optional callback(msg_dict) — public trade tape
            on_order_event: optional callback(msg_dict) — every state change of
                            this account's orders (placed/resting/canceled/executed)
            on_book_change: optional callback(ticker, side, price, delta) fired
                            on every raw level change (before the BBO fold).
                            Useful for forensic recording where you want every
                            depth change, not just TOB.
        """
        self.api = api
        self.on_update = on_update
        self.on_fill = on_fill
        self.on_fill_raw = on_fill_raw
        self.on_trade = on_trade
        self.on_order_event = on_order_event
        self.on_book_change = on_book_change
        # Fired when staleness monitor trips (30s of no data).  Trading
        # apps wire this to a kill-switch that cancels resting orders.
        self.on_stale = on_stale
        self.ws = None
        self.msg_id = 1
        self._thread = None
        self._loop = None
        self._running = False
        self.last_update_ts: float = 0.0  # timestamp of last data received

        # Local orderbook state per ticker:
        # { ticker: { "yes_levels": {price: qty}, "no_levels": {price: qty} } }
        self.books: dict[str, dict] = {}
        self._tickers: list[str] = []

    def start(self, tickers: list[str]):
        """Start websocket connection on a background daemon thread."""
        self._tickers = list(tickers)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def subscribe_tickers(self, new_tickers: list[str]):
        """Subscribe to additional tickers on the existing connection."""
        if not new_tickers or not self._running or not self._loop:
            return
        added = []
        for ticker in new_tickers:
            if ticker not in self.books:
                self.books[ticker] = {"yes_levels": {}, "no_levels": {}}
                self._tickers.append(ticker)
                added.append(ticker)
        if added:
            self._loop.call_soon_threadsafe(
                lambda t=added: asyncio.ensure_future(self._send_subscribe(t))
            )

    async def _send_subscribe(self, tickers: list[str]):
        """Send subscribe command for additional tickers."""
        if not self.ws:
            return
        msg = {
            "id": self.msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }
        self.msg_id += 1
        try:
            await self.ws.send(json.dumps(msg))
        except Exception:
            pass

    def stop(self):
        """Gracefully stop the websocket. Safe to call from any thread.
        Sets running flag to False so message loop exits, then closes the socket."""
        self._running = False
        # Schedule websocket close from the event loop's thread
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._schedule_close)

    def _schedule_close(self):
        """Schedule websocket close on the event loop. Called via call_soon_threadsafe."""
        if self.ws:
            asyncio.ensure_future(self._safe_close(), loop=self._loop)

    async def _safe_close(self):
        """Close websocket, catching any errors."""
        try:
            await self.ws.close()
        except Exception:
            pass

    def _run_loop(self):
        """Entry point for the background thread. Creates its own event loop.
        Reconnects on disconnect with exponential backoff."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        reconnect_delay = 2
        while self._running:
            try:
                self._loop.run_until_complete(self._connect_and_listen())
            except RuntimeError:
                # "Event loop stopped before Future completed" — safe to ignore on shutdown
                break
            except Exception as e:
                print(f"[WS] Loop error: {e}")
            if not self._running:
                break
            print(f"[WS] Disconnected, reconnecting in {reconnect_delay}s...")
            import time
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
        try:
            self._loop.close()
        except Exception:
            pass

    async def _connect_and_listen(self):
        """Connect to Kalshi WS, subscribe to all tickers, and process messages."""

        # --- Connect ---
        headers = self.api.ws_auth_headers()
        try:
            self.ws = await websockets.connect(
                self.api.WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
        except Exception as e:
            print(f"[WS] Connect error: {e}")
            return

        # --- Subscribe to market-scoped channels (book/fill/trade) ---
        # NOTE: `user_orders` does NOT live on this endpoint despite
        # appearing in some docs — it has its own URL
        # (USER_ORDERS_WS_URL) handled by a separate connection below.
        channels = ["orderbook_delta"]
        if self.on_fill or self.on_fill_raw:
            channels.append("fill")
        if self.on_trade:
            channels.append("trade")
        subscribe_msg = {
            "id": self.msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": channels,
                "market_tickers": self._tickers,
            },
        }
        self.msg_id += 1
        await self.ws.send(json.dumps(subscribe_msg))

        # --- Initialize empty books ---
        for ticker in self._tickers:
            self.books[ticker] = {"yes_levels": {}, "no_levels": {}}

        # --- Start staleness monitor ---
        stale_task = asyncio.ensure_future(self._stale_monitor())

        # --- Message loop ---
        try:
            async for message in self.ws:
                if not self._running:
                    break
                self._handle_message(json.loads(message))
        except websockets.ConnectionClosed:
            # Normal disconnection (stop() was called, or server closed)
            pass
        except Exception:
            pass
        finally:
            stale_task.cancel()
            # Always clean up — use try/except since ws might already be closed
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _stale_monitor(self):
        """Monitor for staleness — if no data for 30s, force reconnect."""
        import time
        _STALE_THRESHOLD = 30
        while self._running:
            await asyncio.sleep(5)
            if self.last_update_ts > 0 and (time.time() - self.last_update_ts) > _STALE_THRESHOLD:
                print(f"[WS] STALE — no data for {time.time() - self.last_update_ts:.0f}s, forcing reconnect")
                # Fire the kill-switch callback BEFORE closing the socket
                # so the app can issue cancels while we still have connectivity.
                if self.on_stale:
                    try:
                        self.on_stale()
                    except Exception as e:
                        print(f"[WS] on_stale callback error: {e}")
                self.last_update_ts = 0.0
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                break

    def _handle_message(self, data: dict):
        """Route incoming WS message to the appropriate handler."""
        import time
        self.last_update_ts = time.time()
        msg_type = data.get("type")

        if msg_type == "orderbook_snapshot":
            self._process_snapshot(data["msg"])
        elif msg_type == "orderbook_delta":
            self._process_delta(data["msg"])
        elif msg_type == "fill" and (self.on_fill or self.on_fill_raw):
            self._process_fill(data["msg"])
        elif msg_type == "trade" and self.on_trade:
            try:
                self.on_trade(data.get("msg", {}))
            except Exception as e:
                print(f"[WS] on_trade error: {e}")
        elif msg_type == "user_order" and self.on_order_event:
            try:
                self.on_order_event(data.get("msg", {}))
            except Exception as e:
                print(f"[WS] on_order_event error: {e}")
        elif msg_type == "error":
            # Surface subscription errors from Kalshi so we can diagnose
            # silently-rejected channels (e.g. wrong scope params).
            print(f"[WS] Server error: {data}")
        elif msg_type == "subscribed":
            # Confirmation — useful to see which channels actually took
            print(f"[WS] Subscribed: {data.get('msg', data)}")

    def _process_snapshot(self, msg: dict):
        """Process a full orderbook snapshot. Replaces all levels for this ticker."""
        ticker = msg["market_ticker"]
        if ticker not in self.books:
            self.books[ticker] = {"yes_levels": {}, "no_levels": {}}

        # Rebuild yes levels: list of [price_str, qty_str] pairs
        self.books[ticker]["yes_levels"] = {
            float(price): float(qty)
            for price, qty in msg.get("yes_dollars_fp", [])
        }
        # Rebuild no levels
        self.books[ticker]["no_levels"] = {
            float(price): float(qty)
            for price, qty in msg.get("no_dollars_fp", [])
        }

        self._fire_update(ticker)

    def _process_delta(self, msg: dict):
        """Process an incremental orderbook update. Adjusts one price level."""
        ticker = msg["market_ticker"]
        if ticker not in self.books:
            return

        price = float(msg["price_dollars"])
        delta = float(msg["delta_fp"])        # positive = added, negative = removed
        side = msg["side"]                     # "yes" or "no"

        # Pick the correct side of the book
        key = "yes_levels" if side == "yes" else "no_levels"
        levels = self.books[ticker][key]

        # Apply the delta to the existing quantity
        current = levels.get(price, 0.0)
        new_qty = current + delta

        if new_qty <= 0:
            levels.pop(price, None)
        else:
            levels[price] = new_qty

        # Fire raw-change hook before the TOB fold — useful for forensic
        # recording (every depth change captured, not just TOB transitions).
        if self.on_book_change:
            try:
                self.on_book_change(ticker, side, price, delta)
            except Exception:
                pass

        self._fire_update(ticker)

    def _fire_update(self, ticker: str):
        """Compute best YES bid and best YES ask, then fire the callback.

        YES bid = highest price someone is willing to pay for YES
        YES ask = 1 - highest NO bid (selling NO at X = buying YES at 1-X)
        """
        book = self.books.get(ticker, {})
        yes_levels = book.get("yes_levels", {})
        no_levels = book.get("no_levels", {})

        # Best YES bid: highest yes price with quantity > 0
        yes_bid = max(yes_levels.keys()) if yes_levels else 0.0
        bid_size = int(yes_levels.get(yes_bid, 0)) if yes_bid > 0 else 0

        # Best YES ask: derived from best NO bid.  Round to 3 decimals
        # to preserve sub-cent precision on markets that quote in
        # tenths-of-cent (e.g. KXBTC15M); markets that trade in whole
        # cents land on the same value either way.
        if no_levels:
            best_no_bid = max(no_levels.keys())
            yes_ask = round(1.0 - best_no_bid, 3)
            ask_size = int(no_levels.get(best_no_bid, 0))
        else:
            yes_ask = 0.0
            ask_size = 0

        # Crossed book = stale state — clear and re-subscribe for fresh snapshot
        if yes_bid > 0 and yes_ask > 0 and yes_bid >= yes_ask:
            self.books[ticker] = {"yes_levels": {}, "no_levels": {}}
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(
                    lambda t=ticker: asyncio.ensure_future(
                        self._send_subscribe([t])
                    )
                )
            return

        # Fire callback to GUI (runs on WS thread, GUI picks up via dirty flag)
        try:
            self.on_update(ticker, yes_bid, yes_ask, bid_size, ask_size)
        except Exception:
            pass

    def _process_fill(self, msg: dict):
        """Process a fill notification. Fires on_fill callback."""
        # Log raw message so we can verify field names
        print(f"[WS Fill RAW] {msg}")

        ticker = msg.get("market_ticker", "")
        action = msg.get("action", "")       # "buy" or "sell"
        side = msg.get("side", "")           # "yes" or "no"
        count = int(float(msg.get("count_fp", msg.get("count", 0))))

        # Try multiple possible field names for price
        price = 0.0
        for key in ("yes_price", "yes_price_dollars", "price",
                     "no_price", "no_price_dollars"):
            val = msg.get(key)
            if val:
                price = float(val)
                break

        print(f"[WS Fill] {ticker} {action} {side} x{count} @ ${price:.2f}")
        if self.on_fill:
            try:
                self.on_fill(ticker, action, side, price, count)
            except Exception:
                pass
        if self.on_fill_raw:
            try:
                self.on_fill_raw(msg)
            except Exception as e:
                print(f"[WS] on_fill_raw error: {e}")


# =============================================================================
# Dedicated user_orders feed
#
# `user_orders` lives at a separate URL from the rest of the WS surface
# (`wss://external-api-ws.kalshi.com/user_orders`).  Subscribing to it on
# the standard endpoint is silently ignored, which is why we need a
# second connection.  Account-scoped — no market_tickers parameter.
# =============================================================================

class UserOrdersWsFeed:

    def __init__(self, api: KalshiAPI, on_order_event: Callable):
        """
        Args:
            api: KalshiAPI instance for auth header signing.
            on_order_event: callback(msg_dict) fired on every state change
                of one of this account's orders (resting, canceled, executed).
        """
        self.api = api
        self.on_order_event = on_order_event
        self.ws = None
        self.msg_id = 1
        self._thread = None
        self._loop = None
        self._running = False

    def start(self):
        """Start the user_orders WS in a background daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True,
            name="kalshi-user-orders",
        )
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

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        delay = 2
        while self._running:
            try:
                self._loop.run_until_complete(self._connect_and_listen())
            except RuntimeError:
                break
            except Exception as e:
                print(f"[UserOrders WS] Loop error: {e}")
            if not self._running:
                break
            print(f"[UserOrders WS] Disconnected, reconnecting in {delay}s...")
            import time as _time
            _time.sleep(delay)
            delay = min(delay * 2, 30)
        try:
            self._loop.close()
        except Exception:
            pass

    async def _connect_and_listen(self):
        # Same path as the main WS endpoint — external-api-ws is a
        # separate host but accepts identical handshake/signing.
        headers = self.api.ws_auth_headers(path="/trade-api/ws/v2")
        try:
            self.ws = await websockets.connect(
                USER_ORDERS_WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            print("[UserOrders WS] Connected")
        except Exception as e:
            print(f"[UserOrders WS] Connect error: {e}")
            return

        # Subscribe — account-scoped, no market_tickers.
        msg = {
            "id": self.msg_id,
            "cmd": "subscribe",
            "params": {"channels": ["user_orders"]},
        }
        self.msg_id += 1
        await self.ws.send(json.dumps(msg))
        print("[UserOrders WS] Connected, subscribed")

        try:
            async for raw in self.ws:
                if not self._running:
                    break
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                t = data.get("type")
                if t == "user_order":
                    try:
                        self.on_order_event(data.get("msg", {}))
                    except Exception as e:
                        print(f"[UserOrders WS] callback error: {e}")
                elif t == "error":
                    print(f"[UserOrders WS] Server error: {data}")
                elif t == "subscribed":
                    print(f"[UserOrders WS] Subscribed: {data.get('msg', data)}")
        except websockets.ConnectionClosed:
            pass
        except Exception:
            pass
        finally:
            try:
                await self.ws.close()
            except Exception:
                pass