import asyncio
import json
import time
import base64
import websockets
import requests
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from collections.abc import Callable


class KalshiMd:

    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    REST_URL = "https://api.elections.kalshi.com/trade-api/v2"
    ACCESS_KEY = "2bc651e6-3882-4206-b539-93540910df06"
    PRIVATE_KEY_PATH = Path.home() / "private_key.pem"

    def __init__(self, callback: Callable):
        self.callback = callback
        self.ws = None
        self.msg_id = 1
        self.markets: list[dict] = []

        # books[date][ticker] = { best_yes_bid, best_no_bid, yes_levels, no_levels }
        self.books: dict[str, dict[str, dict]] = {}

        with open(self.PRIVATE_KEY_PATH, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: int, method: str, path: str) -> str:
        message = f"{timestamp_ms}{method}{path}"
        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict:
        timestamp_ms = int(time.time() * 1000)
        sig = self._sign(timestamp_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.ACCESS_KEY,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

    def _parse_date(self, ticker: str) -> str:
        """Extract date portion from ticker like KXGOLDD-26APR0717-T3050"""
        parts = ticker.split("-")
        return parts[1] if len(parts) >= 2 else "unknown"

    def _get_or_create_book(self, ticker: str) -> dict:
        date = self._parse_date(ticker)
        if date not in self.books:
            self.books[date] = {}
        if ticker not in self.books[date]:
            self.books[date][ticker] = {
                "best_yes_bid": None,
                "best_no_bid": None,
                "yes_levels": {},
                "no_levels": {},
            }
        return self.books[date][ticker]

    def fetch_markets(self, series_ticker: str) -> list[dict]:
        """Fetch all open markets for a given series via REST API with pagination."""
        all_markets = []
        cursor = None

        while True:
            path = "/trade-api/v2/markets"
            headers = self._auth_headers("GET", path)
            params = {
                "series_ticker": series_ticker,
                "status": "open",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(f"{self.REST_URL}/markets", headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            markets = data.get("markets", [])
            all_markets.extend(markets)

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

        self.markets = all_markets
        return all_markets

    async def connect(self):
        headers = self._auth_headers("GET", "/trade-api/ws/v2")
        self.ws = await websockets.connect(self.WS_URL, additional_headers=headers)

    async def subscribe(self, market_ticker: str):
        self._get_or_create_book(market_ticker)
        msg = {
            "id": self.msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": market_ticker,
            },
        }
        self.msg_id += 1
        await self.ws.send(json.dumps(msg))

    async def subscribe_all(self, tickers: list[str]):
        for ticker in tickers:
            await self.subscribe(ticker)

    def _process_snapshot(self, data: dict):
        ticker = data["market_ticker"]
        book = self._get_or_create_book(ticker)

        yes_levels = data.get("yes_dollars_fp", [])
        no_levels = data.get("no_dollars_fp", [])

        best_yes = None
        for price, qty in reversed(yes_levels):
            if float(qty) > 0:
                best_yes = (float(price), float(qty))
                break

        best_no = None
        for price, qty in reversed(no_levels):
            if float(qty) > 0:
                best_no = (float(price), float(qty))
                break

        book["best_yes_bid"] = best_yes
        book["best_no_bid"] = best_no
        book["yes_levels"] = {float(p): float(q) for p, q in yes_levels}
        book["no_levels"] = {float(p): float(q) for p, q in no_levels}

    def _process_delta(self, data: dict):
        ticker = data["market_ticker"]
        book = self._get_or_create_book(ticker)

        price = float(data["price_dollars"])
        delta = float(data["delta_fp"])
        side = data["side"]

        levels_key = "yes_levels" if side == "yes" else "no_levels"
        levels = book[levels_key]

        current_qty = levels.get(price, 0.0)
        new_qty = current_qty + delta
        if new_qty <= 0:
            levels.pop(price, None)
        else:
            levels[price] = new_qty

        best_key = "best_yes_bid" if side == "yes" else "best_no_bid"
        if levels:
            best_price = max(levels.keys())
            book[best_key] = (best_price, levels[best_price])
        else:
            book[best_key] = None

    def get_top(self, ticker: str) -> dict | None:
        date = self._parse_date(ticker)
        if date not in self.books or ticker not in self.books[date]:
            return None
        book = self.books[date][ticker]
        return {
            "best_yes_bid": book["best_yes_bid"],
            "best_no_bid": book["best_no_bid"],
        }

    async def listen(self):
        async for message in self.ws:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "orderbook_snapshot":
                self._process_snapshot(data["msg"])
                ticker = data["msg"]["market_ticker"]
                self.callback(ticker, self.get_top(ticker))
            elif msg_type == "orderbook_delta":
                self._process_delta(data["msg"])
                ticker = data["msg"]["market_ticker"]
                self.callback(ticker, self.get_top(ticker))

    async def run(self):
        await self.connect()
        await self.listen()

    async def disconnect(self):
        if self.ws:
            await self.ws.close()