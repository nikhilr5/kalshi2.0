"""Kalshi REST API — auth, orders, markets, positions.

Trimmed port of Aston/kalshi_api.py for the weather-floor app. Same RSA-PSS
signing (over the FULL path /trade-api/v2/...), same persistent httpx HTTP/2
session, same async order wrappers. Dropped: the order-attempt JSONL logger
and RTT telemetry (not needed at this app's order cadence).
"""
import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiAPI:
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    ACCESS_KEY = "73e2b386-6ca6-4ed8-beaf-f58404c6bba0"
    PRIVATE_KEY_PATH = Path.home() / "private_key.pem"

    def __init__(self):
        with open(self.PRIVATE_KEY_PATH, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = httpx.Client(
            http2=True,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            timeout=10.0,
        )
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kalshi-rest")

    def shutdown(self):
        self._executor.shutdown(wait=False)
        try:
            self.session.close()
        except Exception:
            pass

    # --- auth ---
    def _sign(self, ts_ms: int, method: str, path: str) -> str:
        msg = f"{ts_ms}{method}{path}".encode("utf-8")
        sig = self.private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _headers(self, method: str, full_path: str) -> dict:
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.ACCESS_KEY,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, full_path),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "Content-Type": "application/json",
        }

    def ws_auth_headers(self, path: str = "/trade-api/ws/v2") -> dict:
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.ACCESS_KEY,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, "GET", path),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
        }

    # --- transport ---
    def _get(self, path: str, params: dict = None) -> dict:
        headers = self._headers("GET", f"/trade-api/v2{path}")
        resp = self.session.get(f"{self.BASE_URL}{path}", headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        headers = self._headers("POST", f"/trade-api/v2{path}")
        resp = self.session.post(f"{self.BASE_URL}{path}", headers=headers, json=body)
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}
        return {"status_code": resp.status_code, **payload}

    def _delete(self, path: str, body: dict = None) -> dict:
        headers = self._headers("DELETE", f"/trade-api/v2{path}")
        resp = self.session.request("DELETE", f"{self.BASE_URL}{path}", headers=headers, json=body)
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}
        return {"status_code": resp.status_code, **payload}

    # --- markets ---
    def get_orderbook(self, ticker: str, depth: int = 1) -> dict:
        data = self._get(f"/markets/{ticker}/orderbook", {"depth": depth})
        book = data.get("orderbook_fp", data.get("orderbook", {}))
        parse = lambda raw: [(float(p), float(q)) for p, q in (raw or [])]
        return {
            "yes": parse(book.get("yes_dollars", book.get("yes_dollars_fp", []))),
            "no": parse(book.get("no_dollars", book.get("no_dollars_fp", []))),
        }

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_markets(self, series_ticker: str = None, status: str = None) -> list:
        markets, cursor = [], None
        while True:
            params = {"limit": 200}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params)
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break
        return markets

    # --- orders ---
    def create_order(self, ticker: str, side: str, action: str, price_dollars: str,
                     count: int, time_in_force: str = "immediate_or_cancel",
                     tag: str = "") -> dict:
        """Limit order. side='yes', action='sell' -> sell YES at price_dollars.
        Default IOC: hit whatever's resting at/through the price, cancel the rest."""
        body = {
            "ticker": ticker, "side": side, "action": action,
            "yes_price_dollars": price_dollars, "count": count,
            "client_order_id": f"{tag + '_' if tag else ''}{uuid.uuid4()}",
            "type": "limit", "time_in_force": time_in_force,
        }
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_orders(self, status: str = "resting") -> list:
        orders, cursor = [], None
        while True:
            params = {"status": status, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/portfolio/orders", params)
            orders.extend(data.get("orders", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("orders"):
                break
        return orders

    def get_positions(self) -> list:
        positions, cursor = [], None
        while True:
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/portfolio/positions", params)
            positions.extend(data.get("market_positions", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("market_positions"):
                break
        return positions

    def get_balance(self) -> dict:
        return self._get("/portfolio/balance")

    # --- async wrappers (return Future; run on the pool) ---
    def create_order_async(self, **kw):
        return self._executor.submit(self.create_order, **kw)

    def get_positions_async(self):
        return self._executor.submit(self.get_positions)
