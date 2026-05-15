"""Standalone state recorder.

Captures everything needed to reconstruct an Aston trading session:

    fills        — executed trades (via user_orders WS)
    order_events — terminal state transitions (placed / cancelled / filled)
    spot_ticks   — every Coinbase WS tick for the configured product
    kalshi_book  — every Kalshi orderbook_delta on the active market
    theo_state   — theo, sigma, per-horizon RVs on every recompute

Theo and HAR-RV are computed *inside* the recorder (not borrowed from a
live Aston process) so this works headless.  Same HARRVEstimator and
compute_theo the app uses, so numbers match by construction.

File layout: one SQLite per (series, UTC day) in
analysis/backtesting/data/, e.g. `KXETH15M-26MAY14.db`.  Each file
holds all five tables for that day.

Default series: KXETH15M (ETH 15-min up/down).  Override with --series.

Usage:
    python3 recorder.py                  # ETH, default
    python3 recorder.py --series KXBTC15M
"""

import argparse
import re
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

DEFAULT_DATA_DIR = (Path(__file__).resolve().parent.parent
                    / "analysis" / "backtesting" / "data")

_MONTHS = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
           7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}

# series → Coinbase product mapping.  Keep aligned with SERIES_15M in
# market_discovery.py — that's the source of truth for the app.
SERIES_TO_PRODUCT = {
    "KXBTC15M": "BTC-USD",
    "KXETH15M": "ETH-USD",
    "KXSOL15M": "SOL-USD",
    "KXXRP15M": "XRP-USD",
}


def db_path_for(series_ticker: str, day_utc: datetime,
                base_dir: Path = DEFAULT_DATA_DIR) -> Path:
    yy = day_utc.year % 100
    suffix = f"{yy:02d}{_MONTHS[day_utc.month]}{day_utc.day:02d}"
    return base_dir / f"{series_ticker}-{suffix}.db"


# =============================================================================
# Multi-table SQLite writer
# =============================================================================

class StateRecorder:
    """Thread-safe SQLite writer keyed by file path.  Holds one open
    connection per file (WAL mode) so per-row writes don't pay the
    open/close tax.  All five tables in the same DB so a single JOIN
    reconstructs the full state at any timestamp.
    """

    def __init__(self, base_dir: Path = DEFAULT_DATA_DIR):
        self.base_dir = base_dir
        self._conns: dict[Path, sqlite3.Connection] = {}
        self._lock = threading.Lock()

    def _conn_for(self, path: Path) -> sqlite3.Connection:
        if path in self._conns:
            return self._conns[path]
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        # Existing fills table — schema kept identical to legacy
        # 4Runner recorder + PositionManager.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ticker TEXT NOT NULL,
                event_ticker TEXT,
                action TEXT,
                side TEXT,
                count REAL,
                price REAL,
                strike REAL,
                spot_bid REAL,
                spot_ask REAL,
                kalshi_yes_bid REAL,
                kalshi_yes_ask REAL,
                client_order_id TEXT,
                fee REAL DEFAULT 0,
                is_taker INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills(ticker)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                order_id TEXT NOT NULL,
                ticker TEXT,
                event_type TEXT NOT NULL,   -- placed | cancelled | filled
                side TEXT,
                action TEXT,
                price REAL,
                count REAL,
                remaining_count REAL,
                status TEXT,
                client_order_id TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_oe_ts ON order_events(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_oe_oid ON order_events(order_id)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spot_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                product TEXT NOT NULL,
                price REAL,
                bid REAL,
                ask REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_st_ts ON spot_ticks(ts)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS kalshi_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ticker TEXT NOT NULL,
                yes_bid REAL,
                yes_ask REAL,
                bid_size INTEGER,
                ask_size INTEGER
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_ts ON kalshi_book(ts)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS theo_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ticker TEXT NOT NULL,
                spot REAL,
                strike REAL,
                seconds_to_expiry REAL,
                sigma REAL,
                theo REAL,
                rv_15m REAL,
                rv_30m REAL,
                rv_4h REAL,
                rv_24h REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_ts ON theo_state(ts)")

        conn.commit()
        self._conns[path] = conn
        return conn

    def _conn_now(self, series_ticker: str) -> sqlite3.Connection:
        """Conn keyed by today's UTC day — file naturally rotates at
        midnight UTC because db_path_for() returns a new filename.

        Also closes any conn whose path is no longer the current
        write target.  That way the daily_rotate job can safely
        upload + remove yesterday's file without racing an open
        SQLite handle in this process."""
        path = db_path_for(series_ticker, datetime.now(tz=timezone.utc),
                           self.base_dir)
        for stale_path in [p for p in self._conns if p != path]:
            try:
                self._conns[stale_path].close()
            except Exception:
                pass
            del self._conns[stale_path]
        return self._conn_for(path)

    # ---- write methods (each obtains the conn, executes, commits) ----

    def write_fill(self, series_ticker: str, ticker: str, action: str,
                   side: str, count: float, price: float, strike: float,
                   spot: float, kalshi_bid: float, kalshi_ask: float,
                   client_order_id: str = "init", is_taker: int = 0):
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            conn = self._conn_now(series_ticker)
            conn.execute("""
                INSERT INTO fills (
                    ts, ticker, event_ticker, action, side, count, price, strike,
                    spot_bid, spot_ask, kalshi_yes_bid, kalshi_yes_ask,
                    client_order_id, fee, is_taker
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (now, ticker, ticker, action, side, count, price, strike,
                  spot, spot, kalshi_bid, kalshi_ask,
                  client_order_id, 0.0, is_taker))
            conn.commit()

    def write_order_event(self, series_ticker: str, order_id: str,
                          ticker: str, event_type: str, side: str,
                          action: str, price: float, count: float,
                          remaining_count: float, status: str,
                          client_order_id: str):
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            conn = self._conn_now(series_ticker)
            conn.execute("""
                INSERT INTO order_events (
                    ts, order_id, ticker, event_type, side, action,
                    price, count, remaining_count, status, client_order_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (now, order_id, ticker, event_type, side, action,
                  price, count, remaining_count, status, client_order_id))
            conn.commit()

    def write_spot_tick(self, series_ticker: str, product: str,
                        price: float, bid: float, ask: float):
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            conn = self._conn_now(series_ticker)
            conn.execute("""
                INSERT INTO spot_ticks (ts, product, price, bid, ask)
                VALUES (?,?,?,?,?)
            """, (now, product, price, bid, ask))
            conn.commit()

    def write_kalshi_book(self, series_ticker: str, ticker: str,
                          yes_bid: float, yes_ask: float,
                          bid_size: int, ask_size: int):
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            conn = self._conn_now(series_ticker)
            conn.execute("""
                INSERT INTO kalshi_book
                (ts, ticker, yes_bid, yes_ask, bid_size, ask_size)
                VALUES (?,?,?,?,?,?)
            """, (now, ticker, yes_bid, yes_ask, bid_size, ask_size))
            conn.commit()

    def write_theo_state(self, series_ticker: str, ticker: str,
                         spot: float, strike: float,
                         seconds_to_expiry: float,
                         sigma: float | None, theo: float | None,
                         rv_15m: float | None, rv_30m: float | None,
                         rv_4h: float | None, rv_24h: float | None):
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            conn = self._conn_now(series_ticker)
            conn.execute("""
                INSERT INTO theo_state
                (ts, ticker, spot, strike, seconds_to_expiry,
                 sigma, theo, rv_15m, rv_30m, rv_4h, rv_24h)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (now, ticker, spot, strike, seconds_to_expiry,
                  sigma, theo, rv_15m, rv_30m, rv_4h, rv_24h))
            conn.commit()

    def close(self):
        with self._lock:
            for c in self._conns.values():
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()


# Backwards-compat alias for code that imported the legacy fills-only
# class.  No remaining call sites in this tree but kept for safety.
FillRecorder = StateRecorder


# =============================================================================
# Standalone recorder process
# =============================================================================

_TICKER_SERIES_RE = re.compile(r"^(KX[A-Z]+15M)-")


def _series_from_ticker(ticker: str) -> str | None:
    m = _TICKER_SERIES_RE.match(ticker or "")
    return m.group(1) if m else None


def _seed_har_from_coinbase(estimator, product_id: str,
                            hours: int = 25) -> int:
    """Pull last `hours` of 1-min Coinbase candles to seed HAR-RV.
    Mirrors HarSeedWorker in app.py but synchronous (recorder startup
    blocks ~5 seconds — fine for a long-running daemon)."""
    url = (f"https://api.exchange.coinbase.com/products/"
           f"{product_id}/candles")
    end = datetime.now(tz=timezone.utc)
    start_overall = end - timedelta(hours=hours)
    cursor = end
    rows = []
    while cursor > start_overall:
        batch_start = max(cursor - timedelta(minutes=300), start_overall)
        r = requests.get(url, params={
            "granularity": 60,
            "start": batch_start.isoformat(),
            "end": cursor.isoformat(),
        }, timeout=10)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch_start
    now_minute = int(time.time() // 60)
    # Drop the in-progress current minute so live aggregation owns it
    # exclusively (matches HarSeedWorker behavior in app.py).
    # Coinbase row order: [time, low, high, open, close, volume]
    # HAR-RV (Parkinson) wants (minute, high, low).
    candles = [(int(r[0]) // 60, float(r[2]), float(r[1]))
               for r in rows
               if (int(r[0]) // 60) < now_minute]
    estimator.seed_from_candles(candles)
    return len(candles)


class StandaloneRecorder:
    """Orchestrates the three WS feeds + market discovery + HAR-RV,
    writes everything through StateRecorder.  One process per series.

    Threading model:
      - Coinbase WS thread → _on_spot_tick → DB writes
      - Kalshi market WS thread → _on_book → DB writes
      - Kalshi user_orders WS thread → _on_order_event → DB writes
      - Discovery thread → rolls the market WS subscription every 60s
    SQLite serializes writes via StateRecorder's internal lock.
    """

    def __init__(self, series_ticker: str):
        # Imports inside __init__ so the library imports
        # (`from recorder import StateRecorder`) stay free of GUI / WS
        # dependencies.
        from kalshi_api import KalshiAPI
        from ws_feed import KalshiWsFeed, UserOrdersWsFeed
        from crypto_feed import CryptoPriceFeed
        from har_rv import HARRVEstimator
        from market_discovery import get_active_market, parse_strike

        self.series = series_ticker
        self.product = SERIES_TO_PRODUCT.get(series_ticker)
        if not self.product:
            raise SystemExit(
                f"Unknown series {series_ticker!r} — "
                f"expected one of {list(SERIES_TO_PRODUCT)}")

        self.api = KalshiAPI()
        self.state = StateRecorder()

        # In-memory market state, refreshed by feeds.
        self.current_ticker: str | None = None
        self.current_market: dict | None = None
        self.strike: float = 0.0
        self.spot: float = 0.0
        self.yes_bid: float = 0.0
        self.yes_ask: float = 0.0
        self.bid_size: int = 0
        self.ask_size: int = 0

        # Theo + vol — same classes the app uses.
        self.har_est = HARRVEstimator(
            coef_path=Path(__file__).resolve().parent / "har_coefficients.json"
        )

        # Refs to live feeds (book WS rolls per 15-min cycle).
        self.book_ws = None
        self.orders_ws = None
        self.spot_ws = None
        self._KalshiWsFeed = KalshiWsFeed
        self._UserOrdersWsFeed = UserOrdersWsFeed
        self._CryptoPriceFeed = CryptoPriceFeed
        self._get_active_market = get_active_market
        self._parse_strike = parse_strike

        # user_orders state tracking — keyed by order_id.
        # We emit terminal events only: placed, cancelled, filled.
        # `_seen_placed` prevents re-emitting placed on every msg for
        # the same resting order; `_terminated` is the final-state
        # latch so we never re-emit cancelled/filled twice.
        self._seen_placed: set[str] = set()
        self._terminated: set[str] = set()
        # Per-order incremental fill tracking (already in legacy recorder).
        self._order_fill_count: dict[str, float] = {}
        self._order_taker_cost: dict[str, float] = {}

        # seconds_to_close lives in market_discovery — pull lazily.
        from market_discovery import seconds_to_close
        self._secs_to_close = seconds_to_close

        self._stop_event = threading.Event()
        self._discovery_thread: threading.Thread | None = None

    # ---- startup / shutdown ----

    def start(self):
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        print(f"[Recorder] series={self.series} "
              f"product={self.product}")
        print(f"[Recorder] seeding HAR-RV from Coinbase…")
        try:
            n = _seed_har_from_coinbase(self.har_est, self.product)
            print(f"[Recorder] seeded {n} candles "
                  f"(buffer={self.har_est.sample_count()})")
        except Exception as e:
            print(f"[Recorder] HAR seed failed: {e} — prior coefs only")

        # Initial market discovery — sets self.current_ticker
        self._roll_market()

        # Coinbase WS
        self.spot_ws = self._CryptoPriceFeed(
            self._on_spot_tick, self.product)
        self.spot_ws.start()

        # user_orders WS (account-scoped, no ticker filter)
        self.orders_ws = self._UserOrdersWsFeed(
            self.api, self._on_order_event)
        self.orders_ws.start()

        # Discovery / market-roll thread
        self._discovery_thread = threading.Thread(
            target=self._discovery_loop, daemon=True,
            name="recorder-discovery")
        self._discovery_thread.start()

        print("[Recorder] Running.  Ctrl-C to stop.")
        while not self._stop_event.is_set():
            time.sleep(1)
        self._shutdown_real()

    def _shutdown(self):
        # signal handler — set flag, let main loop exit cleanly
        self._stop_event.set()

    def _shutdown_real(self):
        print("\n[Recorder] Shutting down…")
        for ws in (self.spot_ws, self.book_ws, self.orders_ws):
            try:
                if ws:
                    ws.stop()
            except Exception:
                pass
        try:
            self.state.close()
        except Exception:
            pass
        print("[Recorder] Done")

    # ---- market discovery & WS rolls ----

    def _discovery_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(60)
            if self._stop_event.is_set():
                break
            try:
                self._roll_market()
            except Exception as e:
                print(f"[Recorder] discovery error: {e}")

    def _roll_market(self):
        """Idempotent: if the active market hasn't changed, do nothing.
        If it has, tear down the old book WS and stand up a new one for
        the new ticker."""
        try:
            market = self._get_active_market(self.api, self.series)
        except Exception as e:
            print(f"[Recorder] get_active_market failed: {e}")
            return
        if not market:
            return
        new_ticker = market.get("ticker", "")
        if new_ticker == self.current_ticker:
            self.current_market = market
            return
        # Roll.
        self.current_market = market
        self.current_ticker = new_ticker
        self.strike = self._parse_strike(market)
        # Seed BBO from the market dict in case the WS takes a beat
        # to push the first update.
        try:
            self.yes_bid = float(market.get("yes_bid", 0) or 0) / 100.0
        except Exception:
            self.yes_bid = 0.0
        try:
            self.yes_ask = float(market.get("yes_ask", 0) or 0) / 100.0
        except Exception:
            self.yes_ask = 0.0
        # Restart book WS on the new ticker.
        if self.book_ws:
            try:
                self.book_ws.stop()
            except Exception:
                pass
        self.book_ws = self._KalshiWsFeed(self.api, on_update=self._on_book)
        self.book_ws.start([new_ticker])
        print(f"[Recorder] tracking {new_ticker}  strike={self.strike}")

    # ---- WS callbacks ----

    def _on_spot_tick(self, price: float, bid: float = 0.0, ask: float = 0.0):
        if price <= 0:
            return
        self.spot = price
        self.har_est.on_price(price)
        try:
            self.state.write_spot_tick(
                self.series, self.product, price, bid, ask)
        except Exception as e:
            print(f"[Recorder] spot_tick write failed: {e}")
        self._record_theo_state()

    def _on_book(self, ticker: str, yes_bid: float, yes_ask: float,
                 bid_size: int = 0, ask_size: int = 0):
        if ticker != self.current_ticker:
            return
        self.yes_bid = yes_bid
        self.yes_ask = yes_ask
        self.bid_size = bid_size
        self.ask_size = ask_size
        try:
            self.state.write_kalshi_book(
                self.series, ticker, yes_bid, yes_ask, bid_size, ask_size)
        except Exception as e:
            print(f"[Recorder] kalshi_book write failed: {e}")
        self._record_theo_state()

    def _record_theo_state(self):
        """Recompute theo from current spot/strike/sigma/T, write one
        row per call.  Skipped if any input isn't ready yet."""
        if not self.current_market or self.strike <= 0 or self.spot <= 0:
            return
        from theo_engine import compute_theo
        sigma = self.har_est.get_annualized_vol()
        secs = self._secs_to_close(self.current_market)
        theo = compute_theo(self.spot, self.strike, sigma or 0.0, secs) \
               if sigma is not None else None
        b = self.har_est.horizon_breakdown()
        try:
            self.state.write_theo_state(
                self.series, self.current_ticker,
                self.spot, self.strike, secs,
                sigma, theo,
                b.get("rv_15m"), b.get("rv_30m"),
                b.get("rv_4h"),  b.get("rv_24h"))
        except Exception as e:
            print(f"[Recorder] theo_state write failed: {e}")

    def _on_order_event(self, msg: dict):
        """Captures both fills and terminal order-lifecycle events
        from user_orders WS.  Filters to our series."""
        order_id = msg.get("order_id", "")
        ticker = msg.get("ticker") or msg.get("market_ticker", "")
        if not order_id or not ticker:
            return
        series = _series_from_ticker(ticker)
        if series != self.series:
            # Different series (or non-15M) — let another recorder
            # handle it.  One process per series.
            return

        status = (msg.get("status") or "").lower()
        is_yes = bool(msg.get("is_yes"))
        action = msg.get("action") or ("buy" if is_yes else "sell")
        side = "yes"  # standalone: yes-only, matches legacy 4Runner
        price = float(msg.get("yes_price_dollars", 0) or 0)
        count = float(msg.get("count_fp", msg.get("count", 0)) or 0)
        remaining = float(msg.get("remaining_count_fp",
                                   msg.get("remaining_count", 0)) or 0)
        client_order_id = msg.get("client_order_id", "")

        # ---- Fills first (deltas of fill_count) ----
        new_fill = float(msg.get("fill_count_fp",
                                  msg.get("fill_count", 0)) or 0)
        prev_fill = self._order_fill_count.get(order_id, 0.0)
        delta = new_fill - prev_fill
        if delta > 0:
            new_taker = float(msg.get("taker_fill_cost_dollars", 0) or 0)
            prev_taker = self._order_taker_cost.get(order_id, 0.0)
            is_taker = 1 if (new_taker - prev_taker) > 1e-9 else 0
            self._order_fill_count[order_id] = new_fill
            self._order_taker_cost[order_id] = new_taker
            try:
                self.state.write_fill(
                    series_ticker=series, ticker=ticker, action=action,
                    side=side, count=delta, price=price,
                    strike=self.strike, spot=self.spot,
                    kalshi_bid=self.yes_bid, kalshi_ask=self.yes_ask,
                    client_order_id=client_order_id, is_taker=is_taker)
                print(f"[Recorder] FILL {ticker} {action.upper()} "
                      f"x{delta:g} @ {price*100:.1f}¢")
            except Exception as e:
                print(f"[Recorder] fill write failed for {ticker}: {e}")

        # ---- Terminal order-event detection ----
        # placed: first time we see this order in 'resting' state
        # cancelled: status == 'cancelled'/'canceled'
        # filled: remaining == 0 and we've seen at least one fill
        if order_id in self._terminated:
            return

        event_type: str | None = None
        if status in ("cancelled", "canceled"):
            event_type = "cancelled"
            self._terminated.add(order_id)
        elif (new_fill > 0 and remaining == 0
                and order_id not in self._terminated):
            event_type = "filled"
            self._terminated.add(order_id)
        elif (order_id not in self._seen_placed
                and status in ("resting", "open")):
            event_type = "placed"
            self._seen_placed.add(order_id)

        if event_type:
            try:
                self.state.write_order_event(
                    series_ticker=series, order_id=order_id, ticker=ticker,
                    event_type=event_type, side=side, action=action,
                    price=price, count=count, remaining_count=remaining,
                    status=status, client_order_id=client_order_id)
                print(f"[Recorder] {event_type.upper()} {ticker} "
                      f"{action.upper()} @ {price*100:.1f}¢ "
                      f"({order_id[:12]})")
            except Exception as e:
                print(f"[Recorder] order_event write failed: {e}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXETH15M",
                    choices=list(SERIES_TO_PRODUCT.keys()),
                    help="Which 15-min series to record (default: KXETH15M)")
    args = ap.parse_args()
    StandaloneRecorder(args.series).start()
