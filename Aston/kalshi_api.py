"""
Kalshi REST API — auth, orders, positions, market discovery.

IMPORTANT: Signing uses the full path /trade-api/v2/... not just /...
This is required for authenticated endpoints (orders, portfolio).
"""

import time
import base64
import uuid
import requests
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiAPI:
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    ACCESS_KEY = "73e2b386-6ca6-4ed8-beaf-f58404c6bba0"
    PRIVATE_KEY_PATH = Path.home() / "private_key.pem"

    def __init__(self):
        with open(self.PRIVATE_KEY_PATH, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.on_rate_limit = None  # callback(remaining, limit, reset_ts)
        self.rate_limited_until = 0  # timestamp when rate limit expires
        # Worker pool for non-blocking order writes (cancel/place).
        # Sized for concurrent reprices across strikes — one round-trip per
        # call, mostly idle on network IO so threads are cheap.
        self._executor = ThreadPoolExecutor(
            max_workers=20, thread_name_prefix="kalshi-rest"
        )
        # Rolling log of write-token consumption — used to compute
        # tokens/sec for display.  Each entry: (monotonic_ts, tokens).
        self._write_log: deque = deque()

    def shutdown(self):
        """Wait for in-flight async writes to complete, then stop the pool."""
        self._executor.shutdown(wait=True)

    # --- Write throughput tracking ---

    def _log_write(self, tokens: int):
        """Record a write call's token cost for tokens/sec computation."""
        now = time.monotonic()
        self._write_log.append((now, tokens))
        # Keep at most ~5s of history — plenty for 1s windows
        while self._write_log and now - self._write_log[0][0] > 5.0:
            self._write_log.popleft()

    def write_tokens_per_sec(self, window: float = 1.0) -> tuple:
        """Return (tokens, calls) used in the last `window` seconds."""
        now = time.monotonic()
        cutoff = now - window
        tokens = 0
        calls = 0
        for ts, t in self._write_log:
            if ts >= cutoff:
                tokens += t
                calls += 1
        return tokens, calls

    # --- Async wrappers — return a Future immediately, work runs on the pool ---

    def create_order_async(self, **kwargs):
        return self._executor.submit(self.create_order, **kwargs)

    def cancel_order_async(self, order_id: str):
        return self._executor.submit(self.cancel_order, order_id)

    def cancel_orders_batched_async(self, order_ids: list):
        return self._executor.submit(self.cancel_orders_batched, order_ids)

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

    def ws_auth_headers(self, path: str = "/trade-api/ws/v2") -> dict:
        """Auth headers for a websocket handshake.  Pass `path` for any
        non-standard endpoint (e.g. `/user_orders` on the dedicated host)."""
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.ACCESS_KEY,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, "GET", path),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
        }

    def is_rate_limited(self) -> bool:
        """Check if we're currently in a rate limit cooldown."""
        return time.time() < self.rate_limited_until

    def _check_rate_limit(self, resp):
        """Check response for rate limit headers and 429 status."""
        # Check headers (may or may not be present depending on Kalshi's API)
        remaining = resp.headers.get("X-Ratelimit-Remaining") or resp.headers.get("Ratelimit-Remaining")
        limit = resp.headers.get("X-Ratelimit-Limit") or resp.headers.get("Ratelimit-Limit")
        reset = resp.headers.get("X-Ratelimit-Reset") or resp.headers.get("Ratelimit-Reset")
        if remaining is not None and self.on_rate_limit:
            try:
                self.on_rate_limit(int(remaining), int(limit or 0), int(reset or 0))
            except (ValueError, TypeError):
                pass
        if resp.status_code == 429:
            # Back off for 10 seconds on rate limit
            self.rate_limited_until = time.time() + 10
            endpoint = f"{resp.request.method} {resp.request.path_url}"
            if self.on_rate_limit:
                self.on_rate_limit(0, int(limit or 0), int(reset or 0), endpoint=endpoint)
            print(f"[API] RATE LIMITED on {endpoint} — backing off 10s")

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request. path is relative e.g. /portfolio/balance."""
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("GET", full_path)
        resp = requests.get(f"{self.BASE_URL}{path}", headers=headers, params=params, timeout=10)
        self._check_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("POST", full_path)
        print(f"[API POST] {path} body={body}")
        resp = requests.post(f"{self.BASE_URL}{path}", headers=headers, json=body, timeout=10)
        self._check_rate_limit(resp)
        if resp.status_code != 201:
            print(f"[API ERROR] {resp.status_code}: {resp.text}")
            # Include the response body in the raised exception so the
            # cause is visible in logs without inspecting `resp.text`
            # separately.  Default raise_for_status() drops the body
            # and leaves only the status line.
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {full_path}: {resp.text}",
                response=resp,
            )
        return resp.json()

    def _delete(self, path: str, body: dict = None) -> dict:
        """DELETE request with full-path signing."""
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("DELETE", full_path)
        resp = requests.delete(f"{self.BASE_URL}{path}", headers=headers, json=body, timeout=10)
        self._check_rate_limit(resp)
        resp.raise_for_status()
        return resp.json()

    # --- Orderbook ---

    def get_orderbook(self, ticker: str, depth: int = 1) -> dict:
        """Fetch the orderbook for a single market.

        Returns {"yes": [(price, qty), ...], "no": [(price, qty), ...]}
        with prices as floats in dollar terms.
        """
        data = self._get(f"/markets/{ticker}/orderbook", {"depth": depth})
        book = data.get("orderbook_fp", data.get("orderbook", {}))

        def parse_levels(raw):
            return [(float(p), float(q)) for p, q in (raw or [])]

        return {
            "yes": parse_levels(book.get("yes_dollars", book.get("yes_dollars_fp", []))),
            "no": parse_levels(book.get("no_dollars", book.get("no_dollars_fp", []))),
        }

    # --- Market Discovery ---

    def get_markets_for_event(self, event_ticker: str) -> list:
        params = {"event_ticker": event_ticker, "limit": 200}
        return self._get("/markets", params).get("markets", [])

    def get_markets(self, series_ticker: str = None, status: str = None) -> list:
        """Get markets with optional filters. Paginates automatically."""
        markets = []
        cursor = None
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

    # --- Orders ---

    def create_order(self, ticker: str, side: str, action: str,
                     price_dollars: str, count: int,
                     time_in_force: str = "good_till_canceled",
                     tag: str = "",
                     post_only: bool = False) -> dict:
        """Place a limit order.
        side='yes', action='sell' → sell YES at yes_price_dollars.
        time_in_force: 'good_till_canceled', 'immediate_or_cancel', or 'fill_or_kill'.
        tag: optional prefix for client_order_id (e.g. 'init' or 'flat').
        post_only: if True, exchange rejects the order if it would cross.
        """
        prefix = f"{tag}_" if tag else ""
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "yes_price_dollars": price_dollars,
            "count": count,
            "client_order_id": f"{prefix}{uuid.uuid4()}",
            "type": "limit",
            "time_in_force": time_in_force,
        }
        if post_only:
            body["post_only"] = True
        self._log_write(10)  # create_order = default 10 tokens
        return self._post("/portfolio/orders", body)

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order by ID."""
        return self._get(f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> dict:
        self._log_write(2)  # cancel_order = 2 tokens
        return self._delete(f"/portfolio/orders/{order_id}")

    def cancel_orders_batched(self, order_ids: list) -> dict:
        """Batch cancel — DELETE /portfolio/orders/batched.

        One round-trip cancels N orders (vs N round-trips with cancel_order).
        Still billed 2 rate-limit tokens per order in the batch, so the
        cost saving is network latency, not API budget.

        Body: {"orders": [{"order_id": "..."}, ...]}
        Returns: {"orders": [{order_id, order, reduced_by_fp, error}, ...]}
        Each item has its own `error` field — partial failures don't fail
        the request.  Callers should inspect per-order results.
        """
        ids = [oid for oid in (order_ids or []) if oid]
        if not ids:
            return {"orders": []}
        body = {"orders": [{"order_id": oid} for oid in ids]}
        self._log_write(2 * len(ids))  # 2 tokens per order in the batch
        return self._delete("/portfolio/orders/batched", body)

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

    def get_fills(self, ticker: str = None) -> list:
        """Fetch fill history. Optionally filter by ticker."""
        fills = []
        cursor = None
        while True:
            params = {"limit": 200}
            if ticker:
                params["ticker"] = ticker
            if cursor:
                params["cursor"] = cursor
            data = self._get("/portfolio/fills", params)
            fills.extend(data.get("fills", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("fills"):
                break
        return fills

    def get_trades(self, ticker: str, limit: int = 1000) -> list:
        """Fetch public trade history for a market ticker."""
        trades = []
        cursor = None
        while True:
            params = {"ticker": ticker, "limit": min(limit - len(trades), 200)}
            if cursor:
                params["cursor"] = cursor
            data = self._get(f"/markets/trades", params)
            trades.extend(data.get("trades", []))
            cursor = data.get("cursor")
            if not cursor or not data.get("trades") or len(trades) >= limit:
                break
        return trades

    def get_balance(self) -> dict:
        return self._get("/portfolio/balance")