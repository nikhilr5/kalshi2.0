"""
Kalshi REST API — auth, orders, positions, market discovery.

IMPORTANT: Signing uses the full path /trade-api/v2/... not just /...
This is required for authenticated endpoints (orders, portfolio).
"""

import json
import queue
import threading
import time
import base64
import uuid
import httpx
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding


_MONTHS_UC = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_CANCEL_TOKEN_COST = 20
_CREATE_TOKEN_COST = 100


def _attempt_log_path(base_dir: Path) -> Path:
    """`order_attempts-{YYMONDD}.jsonl` in the recorder's data dir."""
    d = datetime.now(tz=timezone.utc)
    suffix = f"{d.year % 100:02d}{_MONTHS_UC[d.month - 1]}{d.day:02d}"
    return base_dir / f"order_attempts-{suffix}.jsonl"


def _attempt_log_drain(q: queue.Queue, base_dir: Path):
    """Background daemon thread: pulls events off the queue and writes
    them to today's order_attempts JSONL.  Reopens the file when UTC
    date rolls over.  Long-running app sessions stay correct without
    a restart.  None on the queue is the shutdown sentinel."""
    fh = None
    current_path: Path | None = None
    while True:
        event = q.get()
        if event is None:
            break
        try:
            path = _attempt_log_path(base_dir)
            if fh is None or path != current_path:
                if fh is not None:
                    try:
                        fh.close()
                    except Exception:
                        pass
                path.parent.mkdir(parents=True, exist_ok=True)
                fh = open(path, "a", buffering=1)   # line-buffered
                current_path = path
            fh.write(json.dumps(event) + "\n")
        except Exception as e:
            # Never raise from the logger.  Print and continue so the
            # producer side keeps draining.
            print(f"[attempt-log] write failed: {e}")
    # Shutdown: flush + close.
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


class KalshiAPI:
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    ACCESS_KEY = "73e2b386-6ca6-4ed8-beaf-f58404c6bba0"
    PRIVATE_KEY_PATH = Path.home() / "private_key.pem"

    def __init__(self, attempt_log_dir: Path | None = None):
        with open(self.PRIVATE_KEY_PATH, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        # Persistent Session — reuses TCP + TLS across calls.  Saves
        # ~22ms per request vs a fresh client (which opens a new TCP+TLS
        # connection each time).  At 2-4 REST calls per reprice cycle,
        # that's 40-100ms saved per cycle.  httpx.Client with http2=True
        # also multiplexes concurrent requests over a single connection,
        # cutting head-of-line blocking vs HTTP/1.1.
        # Connection limits bound how many sockets a burst can pressure —
        # errno 35 (EAGAIN) orphans came from bursts saturating the
        # connection's TCP buffers.
        self.session = httpx.Client(
            http2=True,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            timeout=10.0,
        )
        # One-shot flag — prints negotiated HTTP version on the first
        # response so we can confirm HTTP/2 is actually in use rather
        # than silently falling back to HTTP/1.1.
        self._http_version_logged = False
        # Worker pool for non-blocking order writes (cancel/place).
        # Legitimate concurrency is ~3-5 requests (OSM in-flight guard
        # allows ~2 order ops + sweep read + occasional position fetch);
        # 8 leaves headroom while capping how hard a burst can hit the
        # socket (20 workers let 429-resume herds trigger EAGAIN).
        self._executor = ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="kalshi-rest"
        )
        # Rolling RTT log (per-call wall time in ms) for latency analysis.
        # Each entry: (monotonic_ts, method, path, rtt_ms).
        self._rtt_log: deque = deque(maxlen=10000)

        # ---- Async order-attempt logger (off the hot path) ----
        # Producer: create_order / cancel_order put_nowait an event dict.
        # Consumer: a single daemon thread drains the queue and writes
        # one JSON line per event to a daily-rotated JSONL in the
        # recorder's data dir.  Recorder tails this file and ingests
        # into the per-day SQLite's order_attempts table.
        self._attempt_log_queue: queue.Queue | None = None
        self._attempt_log_writer: threading.Thread | None = None
        if attempt_log_dir is not None:
            self._attempt_log_queue = queue.Queue()
            self._attempt_log_writer = threading.Thread(
                target=_attempt_log_drain,
                args=(self._attempt_log_queue, Path(attempt_log_dir)),
                daemon=True, name="attempt-log-writer",
            )
            self._attempt_log_writer.start()

    def _enqueue_attempt(self, event: dict):
        """Producer-side: push event onto the logger queue.  Returns
        instantly (~microseconds); never blocks the trading path."""
        if self._attempt_log_queue is not None:
            try:
                self._attempt_log_queue.put_nowait(event)
            except Exception:
                pass  # never let logging break trading

    def shutdown(self):
        """Wait for in-flight async writes to complete, then stop the pool."""
        self._executor.shutdown(wait=True)
        # Drain attempt-log queue cleanly before exit.
        if self._attempt_log_queue is not None:
            try:
                self._attempt_log_queue.put_nowait(None)  # sentinel
                if self._attempt_log_writer is not None:
                    self._attempt_log_writer.join(timeout=5.0)
            except Exception:
                pass
        try:
            self.session.close()
        except Exception:
            pass

    # --- Async wrappers — return a Future immediately, work runs on the pool ---

    def create_order_async(self, **kwargs):
        return self._executor.submit(self.create_order, **kwargs)

    def cancel_order_async(self, order_id: str):
        return self._executor.submit(self.cancel_order, order_id)

    def cancel_orders_batched_async(self, order_ids: list):
        return self._executor.submit(self.cancel_orders_batched, order_ids)

    def get_orders_async(self, status: str = "resting"):
        return self._executor.submit(self.get_orders, status)

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

    def _log_http_version(self, resp):
        """One-shot print of the negotiated HTTP version.  Smoke test
        that http2=True is actually being honored end-to-end (Kalshi's
        edge could ALPN-downgrade to HTTP/1.1)."""
        if not self._http_version_logged:
            self._http_version_logged = True
            print(f"[API] negotiated {resp.http_version}")

    def _log_rtt(self, method: str, path: str, t0: float):
        """Record wall-time RTT for one REST call.

        Every 100 calls, also print a summary of the most recent 100 to
        stdout — visible in the tee'd aston.log.  Useful for confirming
        Session reuse is keeping the connection warm (p50 should be
        6–12ms; ~50ms would mean the Session isn't reusing connections).
        """
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        self._rtt_log.append((time.monotonic(), method, path, rtt_ms))
        if len(self._rtt_log) % 100 == 0:
            recent = [r[3] for r in list(self._rtt_log)[-100:]]
            recent.sort()
            n = len(recent)
            p50  = recent[n // 2]
            p95  = recent[int(n * 0.95) - 1]
            mx   = recent[-1]
            print(f"[API RTT] last {n} calls: "
                  f"p50={p50:.1f}ms  p95={p95:.1f}ms  max={mx:.1f}ms  "
                  f"(total {len(self._rtt_log)})")

    def _get(self, path: str, params: dict = None) -> dict:
        """GET request. path is relative e.g. /portfolio/balance."""
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("GET", full_path)
        t0 = time.perf_counter()
        resp = self.session.get(f"{self.BASE_URL}{path}", headers=headers, params=params)
        self._log_rtt("GET", path, t0)
        self._log_http_version(resp)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("POST", full_path)
        ts = datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
        print(f"[API POST] {ts} {path} body={body}")
        t0 = time.perf_counter()
        resp = self.session.post(f"{self.BASE_URL}{path}", headers=headers, json=body)
        self._log_rtt("POST", path, t0)
        self._log_http_version(resp)
        if resp.status_code != 201:
            print(f"[API ERROR] {resp.status_code}: {resp.text}")
        try:
            payload = resp.json()
        except Exception:
            # HTML 400s from the CDN, or any non-JSON response body.
            payload = {"raw": resp.text}
        return {"status_code": resp.status_code, **payload}

    def _delete(self, path: str, body: dict = None) -> dict:
        """DELETE request with full-path signing."""
        full_path = f"/trade-api/v2{path}"
        headers = self._headers("DELETE", full_path)
        t0 = time.perf_counter()
        # httpx.Client.delete() does not accept json=; use request() to
        # send a body on DELETE (Kalshi's batch-cancel endpoint needs it).
        resp = self.session.request(
            "DELETE", f"{self.BASE_URL}{path}", headers=headers, json=body,
        )
        self._log_rtt("DELETE", path, t0)
        self._log_http_version(resp)
        if resp.status_code >= 400:
            print(f"[API ERROR] DELETE {resp.status_code}: {resp.text}")
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}
        return {"status_code": resp.status_code, **payload}

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

    def get_market(self, ticker: str) -> dict:
        """Single-market details.  Post-settlement the response has a
        `result` field of `"yes"` or `"no"`."""
        return self._get(f"/markets/{ticker}").get("market", {})

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
        client_order_id = f"{prefix}{uuid.uuid4()}"
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "yes_price_dollars": price_dollars,
            "count": count,
            "client_order_id": client_order_id,
            "type": "limit",
            "time_in_force": time_in_force,
        }
        if post_only:
            body["post_only"] = True
        t0 = time.perf_counter()
        ts_request = datetime.now(tz=timezone.utc).isoformat()
        http_status = 0
        success = 0
        error_code: str | None = None
        error_msg: str | None = None
        server_order_id: str | None = None
        try:
            resp = self._post("/portfolio/orders", body)
            http_status = int(resp['status_code'])

            #look at status code to determine if success placed post
            if http_status != 201:
                error_code = http_status
                error_msg = json.dumps(resp or {"code": resp.get("code"), "message": resp.get("message")})[:500]
                success = 0
            else: 
                success = 1
                if isinstance(resp, dict):
                    server_order_id = (resp.get("order") or {}).get("order_id")
            return resp
        except Exception as e:
            error_code = 0
            error_msg = str(e)[:500]
            raise
        finally:
            self._enqueue_attempt({
                "ts_request": ts_request,
                "ts_response": datetime.now(tz=timezone.utc).isoformat(),
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "client_order_id": client_order_id,
                "server_order_id": server_order_id,
                "ticker": ticker,
                "action": action,
                "side": side,
                "price": float(price_dollars) if price_dollars else None,
                "count": count,
                "request_type": "create",
                "http_status": http_status,
                "success": success,
                "error_code": error_code,
                "error_msg": error_msg,
            })

    def get_order(self, order_id: str) -> dict:
        """Fetch a single order by ID."""
        return self._get(f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> dict:
        t0 = time.perf_counter()
        ts_request = datetime.now(tz=timezone.utc).isoformat()
        http_status = 0
        success = 0
        error_code: str | None = None
        error_msg: str | None = None
        try:
            resp = self._delete(f"/portfolio/orders/{order_id}")
            http_status = resp['status_code']
            if http_status != 200:
                error_code = http_status
                error_msg = json.dumps(resp or {"code": resp.get("code"), "message": resp.get("message")})[:500]
                success = 0
            else: 
                success = 1
            return resp
        except Exception as e:
            error_code = 0
            error_msg = str(e)[:500]
            raise
        finally:
            self._enqueue_attempt({
                "ts_request": ts_request,
                "ts_response": datetime.now(tz=timezone.utc).isoformat(),
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "client_order_id": None,         # cancel works on server_order_id
                "server_order_id": order_id,
                "ticker": None,                   # not in the cancel call signature
                "action": None,
                "side": None,
                "price": None,
                "count": None,
                "request_type": "cancel",
                "http_status": http_status,
                "success": success,
                "error_code": error_code,
                "error_msg": error_msg,
            })

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