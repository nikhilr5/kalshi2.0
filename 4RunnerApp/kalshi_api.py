"""
Kalshi REST API — auth, orders, positions, market discovery.

IMPORTANT: Signing uses the full path /trade-api/v2/... not just /...
This is required for authenticated endpoints (orders, portfolio).
"""

import time
import base64
import uuid
import requests
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiAPI:
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    ACCESS_KEY = "2bc651e6-3882-4206-b539-93540910df06"
    PRIVATE_KEY_PATH = Path.home() / "private_key.pem"

    def __init__(self):
        with open(self.PRIVATE_KEY_PATH, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, timestamp_ms: int, method: str, path: str) -> str:
        """RSA-PSS signature of {timestamp}{method}{path}."""
        message = f"{timestamp_ms}{method}{path}"
        signature = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        """Generate auth headers. path must be the FULL path e.g. /trade-api/v2/..."""
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.ACCESS_KEY,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, path),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "Content-Type": "application/json",
        }

    def ws_auth_headers(self) -> dict:
        """Auth headers for websocket handshake."""
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.ACCESS_KEY,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, "GET", "/trade-api/ws/v2"),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
        }

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request. path is relative e.g. /portfolio/balance."""
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("GET", full_path)
        resp = requests.get(f"{self.BASE_URL}{path}", headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("POST", full_path)
        print(f"[API POST] {path} body={body}")
        resp = requests.post(f"{self.BASE_URL}{path}", headers=headers, json=body, timeout=10)
        if resp.status_code != 201:
            print(f"[API ERROR] {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, body: dict = None) -> dict:
        """DELETE request with full-path signing."""
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("DELETE", full_path)
        resp = requests.delete(f"{self.BASE_URL}{path}", headers=headers, json=body, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # --- Market Discovery ---

    def get_markets_for_event(self, event_ticker: str) -> list:
        params = {"event_ticker": event_ticker, "limit": 200}
        return self._get("/markets", params).get("markets", [])

    # --- Orders ---

    def create_order(self, ticker: str, side: str, action: str,
                     price_dollars: str, count: int) -> dict:
        """Place a limit order.
        side='yes', action='sell' → sell YES at yes_price_dollars.
        """
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "yes_price_dollars": price_dollars,
            "count": count,
            "client_order_id": str(uuid.uuid4()),
            "type": "limit",
            "time_in_force": "good_till_canceled",
        }
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f"/portfolio/orders/{order_id}")

    def get_orders(self, status: str = "resting") -> list:
        orders = []
        cursor = None
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

    # --- Portfolio ---

    def get_positions(self) -> list:
        positions = []
        cursor = None
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