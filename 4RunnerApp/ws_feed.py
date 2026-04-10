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


class KalshiWsFeed:

    def __init__(self, api: KalshiAPI, on_update: Callable):
        """
        Args:
            api: KalshiAPI instance (used for auth headers and WS URL)
            on_update: callback(ticker, yes_bid, yes_ask) fired on every book change
        """
        self.api = api
        self.on_update = on_update
        self.ws = None
        self.msg_id = 1
        self._thread = None
        self._loop = None
        self._running = False

        # Local orderbook state per ticker:
        # { ticker: { "yes_levels": {price: qty}, "no_levels": {price: qty} } }
        self.books: dict[str, dict] = {}
        self._tickers: list[str] = []

    def start(self, tickers: list[str]):
        """Start websocket connection on a background daemon thread."""
        self._tickers = tickers
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

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
        """Entry point for the background thread. Creates its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except RuntimeError:
            # "Event loop stopped before Future completed" — safe to ignore on shutdown
            pass
        except Exception:
            pass
        finally:
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

        # --- Subscribe to all tickers in one command ---
        subscribe_msg = {
            "id": self.msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": self._tickers,
            },
        }
        self.msg_id += 1
        await self.ws.send(json.dumps(subscribe_msg))

        # --- Initialize empty books ---
        for ticker in self._tickers:
            self.books[ticker] = {"yes_levels": {}, "no_levels": {}}

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
            # Always clean up — use try/except since ws might already be closed
            try:
                await self.ws.close()
            except Exception:
                pass

    def _handle_message(self, data: dict):
        """Route incoming WS message to the appropriate handler."""
        msg_type = data.get("type")

        if msg_type == "orderbook_snapshot":
            # Full book replacement — sent once per ticker after subscribing
            self._process_snapshot(data["msg"])
        elif msg_type == "orderbook_delta":
            # Incremental update — one price level changed
            self._process_delta(data["msg"])

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

        # Best YES ask: derived from best NO bid
        if no_levels:
            best_no_bid = max(no_levels.keys())
            yes_ask = round(1.0 - best_no_bid, 2)
        else:
            yes_ask = 0.0

        # Fire callback to GUI (runs on WS thread, GUI picks up via dirty flag)
        try:
            self.on_update(ticker, yes_bid, yes_ask)
        except Exception:
            pass