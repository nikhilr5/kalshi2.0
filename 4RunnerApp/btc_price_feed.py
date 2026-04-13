"""
Live crypto spot price via Coinbase websocket.

Free, no API key required. Connects to Coinbase WS feed
and fires a callback with the latest price on every update (~1-2 seconds).

Runs on a background daemon thread with its own asyncio event loop.

Usage:
    feed = CryptoPriceFeed(on_price_callback, product_id="BTC-USD")
    feed.start()
    # callback fires as: on_price(price_float)
    feed.stop()
"""

import asyncio
import json
import threading
import websockets
from collections.abc import Callable


class CryptoPriceFeed:

    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, on_price: Callable, product_id: str = "BTC-USD"):
        """
        Args:
            on_price: callback(price: float) fired on every price update
            product_id: Coinbase product ID, e.g. "BTC-USD", "ETH-USD", "SOL-USD"
        """
        self.on_price = on_price
        self.product_id = product_id
        self.ws = None
        self._thread = None
        self._loop = None
        self._running = False
        self.last_price = 0.0

    async def _connect_and_listen(self):
        """Connect to Coinbase and stream prices."""
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
                    "channels": ["ticker"]
                }
                await self.ws.send(json.dumps(subscribe))

                async for message in self.ws:
                    if not self._running:
                        break

                    data = json.loads(message)
                    if data.get("type") == "ticker" and "price" in data:
                        price = float(data["price"])
                        self.last_price = price
                        try:
                            self.on_price(price)
                        except Exception:
                            pass

            except websockets.ConnectionClosed:
                pass
            except Exception as e:
                print(f"[Price:{self.product_id}] Error: {e}")
            finally:
                try:
                    if self.ws:
                        await self.ws.close()
                except Exception:
                    pass

            if self._running:
                await asyncio.sleep(3)

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