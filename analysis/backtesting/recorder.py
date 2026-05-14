"""
Event-driven market data recorder for KXBTCD daily/weekly markets.

Two layers of recording:

    1. `events` table — every WS message captured with native timestamp.
       Includes spot ticks, Kalshi book changes, public trades, and our own
       order state changes.  This is the firehose for microstructure
       forensics (e.g. reconstructing what happened around a phase 3 fire).

    2. `snapshots` table — periodic 30s reconciliation rows with computed
       IVs.  Useful for trends, smile fits, and analytical queries that
       don't need every tick.

    3. `fills` table — kept for back-compat with the position manager.
       Populated from the user_orders WS stream rather than REST polling.

Storage: per-event SQLite files at analysis/backtesting/data/<event>.db.

Usage:
    python recorder.py                     # start recording
    python recorder.py --export 2026-05-04 # export a day to CSV
"""

import os
import re
import subprocess
import sys
import json
import math
import signal
import socket
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

_norm = NormalDist()

# Add 4RunnerApp2.0 to path for shared modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "4RunnerApp2.0"))

from kalshi_api import KalshiAPI
from market_discovery import discover_events_for_series, parse_strike, display_strike
from btc_price_feed import CryptoPriceFeed
from ws_feed import KalshiWsFeed, UserOrdersWsFeed

# =============================================================================
# Config
# =============================================================================

SERIES = "KXBTCD"               # above/below (daily + weekly)
COINBASE_PRODUCT = "BTC-USD"
OTM_FILTER_PCT = 8.0
SNAPSHOT_INTERVAL = 5           # seconds — needed for cross-strike theo
                                # smile fitting (events table is per-ticker
                                # so it can't drive a single-moment fit)
RISK_FREE_RATE = 0.043
DB_PATH = Path(__file__).parent / "data" / "recorder.db"  # legacy
DATA_DIR = Path(__file__).parent / "data"

# UDP port the app publishes theo events on.  The recorder binds here
# and writes received messages to the per-event SQLite events table.
THEO_PUB_PORT = 9871

# Auto-archival to S3 — events that closed more than ARCHIVE_AFTER_DAYS
# ago are uploaded to S3 and deleted locally to keep disk in check.
# Set the bucket via the KALSHI_ARCHIVE_BUCKET env var.  No-op when unset.
ARCHIVE_BUCKET_ENV = "KALSHI_ARCHIVE_BUCKET"
ARCHIVE_PREFIX = "kalshi-events/"
ARCHIVE_AFTER_DAYS = 2.0
ARCHIVE_INTERVAL_SEC = 3600  # check once per hour


def event_db_path(event_ticker: str) -> Path:
    """Per-event SQLite file path."""
    return DATA_DIR / f"{event_ticker}.db"

# =============================================================================
# IV computation
# =============================================================================

def implied_vol(price: float, spot: float, strike: float,
                T: float, r: float = RISK_FREE_RATE) -> float:
    """Closed-form quadratic IV for a binary above option.

    Given P(above K) = N(d2), invert for sigma.
    Returns IV as a decimal (e.g. 0.65 for 65%), or 0 if unsolvable.
    """
    if price <= 0.01 or price >= 0.99 or spot <= 0 or strike <= 0 or T <= 0:
        return 0.0
    try:
        x = _norm.inv_cdf(price)
        m = math.log(spot / strike) + r * T
        disc = x * x + 2 * m
        if disc < 0:
            return 0.0
        sqrt_disc = math.sqrt(disc)
        u1 = -x + sqrt_disc
        u2 = -x - sqrt_disc
        candidates = [u for u in (u1, u2) if u > 0]
        if not candidates:
            return 0.0
        return min(candidates) / math.sqrt(T)
    except Exception:
        return 0.0

# =============================================================================
# Database
# =============================================================================

def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_ticker TEXT,
            strike REAL,
            close_time TEXT,
            T REAL,
            kalshi_yes_bid REAL,
            kalshi_yes_ask REAL,
            bid_size INTEGER,
            ask_size INTEGER,
            spot_bid REAL,
            spot_ask REAL,
            spot_mid REAL,
            mid_iv REAL,
            bid_iv REAL,
            ask_iv REAL
        )
    """)
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
    # Schema migration: add is_taker column to fills if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fills)").fetchall()]
    if "is_taker" not in cols:
        try:
            conn.execute("ALTER TABLE fills ADD COLUMN is_taker INTEGER DEFAULT 0")
        except Exception:
            pass

    # Events table — every WS message captured for microstructure forensics.
    # `event_type` discriminator: spot | book | trade | order
    # `payload` is the raw message JSON for type-specific fields.
    # `ts_us` is unix microseconds (high-res so we can order events that
    # arrive in the same wall-clock millisecond).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_us INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            ticker TEXT,
            payload TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_ts
        ON snapshots (ts)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
        ON snapshots (ticker)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_fills_ts
        ON fills (ts)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_ts
        ON events (ts_us)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_type
        ON events (event_type, ts_us)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_ticker
        ON events (ticker, ts_us)
    """)
    # Composite index — matches the dashboard's chart loaders
    # (event_type + ticker + ts_us range).  Without this, queries
    # filter by event_type then table-scan the rest, costing seconds.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_type_ticker_ts
        ON events (event_type, ticker, ts_us)
    """)
    conn.commit()
    return conn

# =============================================================================
# Recorder
# =============================================================================

class Recorder:

    def __init__(self):
        self.api = KalshiAPI()
        # One connection per event_ticker
        self.conns: dict[str, sqlite3.Connection] = {}
        # Serialize all writes — recorder has multiple writer threads
        # (Coinbase WS, Kalshi WS, UDP listener, main loop).  SQLite
        # connections aren't safe for concurrent writes from multiple
        # Python threads even with check_same_thread=False — we get
        # "bad parameter or other API misuse" / "no transaction active"
        # errors.  One lock around every execute+commit fixes it.
        self._db_lock = threading.Lock()

        self.running = False
        self.spot_price = 0.0
        self.spot_bid = 0.0
        self.spot_ask = 0.0

        self.price_feed = None
        self.ws_feed = None
        # Dedicated WS connection for user_orders (separate Kalshi endpoint)
        self.user_orders_feed: UserOrdersWsFeed | None = None

        # {ticker: {display_strike, event_ticker, close_time}}
        self.tracked = {}
        # {ticker: {yes_bid, yes_ask, bid_size, ask_size}}
        self.book = {}
        # Fill dedup — keyed by trade_id (REST seed) and (order_id, fill_count)
        # for WS-derived fills so we can detect new partial fills.
        self._seen_fill_ids = set()
        # Track last-seen fill_count per order_id so we can detect partial
        # fills as user_orders events stream in.
        self._order_fill_count: dict[str, float] = {}
        # Track cumulative taker fill cost per order_id so we can detect
        # whether a NEW fill_count delta arrived as taker vs maker —
        # user_orders msgs report `taker_fill_cost_dollars` (cumulative)
        # but no per-fill is_taker flag.
        self._order_taker_cost: dict[str, float] = {}

        # UDP listener for theo events published by the app
        self._theo_sock: socket.socket | None = None
        self._theo_thread: threading.Thread | None = None

    def start(self):
        self.running = True
        signal.signal(signal.SIGINT, lambda *_: self._shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self._shutdown())

        # Discover KXBTCD events — daily 5pm ET + nearest Friday weekly
        print(f"[Recorder] Discovering {SERIES} events...")
        events = discover_events_for_series(self.api, SERIES)
        if not events:
            print("[Recorder] No events found")
            return

        # Filter: keep daily 5pm ET (close hour = 21 UTC) and nearest Friday weekly
        # Drop hourly events (close at other hours)
        now_utc = datetime.now(tz=timezone.utc)
        selected_events = []
        for event in events:
            close_str = event.get("close_time", "")
            if not close_str:
                continue
            try:
                close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                if close_utc <= now_utc:
                    continue
                # Daily/weekly = closes at 21:00 UTC (5pm ET)
                if close_utc.hour == 21:
                    selected_events.append(event)
            except Exception:
                continue

        if not selected_events:
            print("[Recorder] No upcoming 5pm ET events found")
            return

        for ev in selected_events:
            print(f"[Recorder] Tracking: {ev['event_ticker']} "
                  f"(close={ev.get('close_time', '')})")

        all_markets = {}
        for event in selected_events:
            et = event["event_ticker"]
            close = event.get("close_time", "")
            for m in event["markets"]:
                ticker = m["ticker"]
                raw = parse_strike(ticker)
                if raw > 0 and ticker not in all_markets:
                    all_markets[ticker] = {
                        "display_strike": display_strike(raw),
                        "event_ticker": et,
                        "close_time": close,
                    }
            print(f"[Recorder]   {et}: {len(event['markets'])} markets, close={close}")

        # Start Coinbase feed
        print(f"[Recorder] Starting Coinbase feed for {COINBASE_PRODUCT}...")
        self.price_feed = CryptoPriceFeed(self._on_price, COINBASE_PRODUCT)
        self.price_feed.start()

        # Wait for spot
        print("[Recorder] Waiting for spot price...")
        for _ in range(100):
            if self.spot_price > 0:
                break
            time.sleep(0.1)

        if self.spot_price <= 0:
            print("[Recorder] No spot price — tracking all markets")
            self.tracked = all_markets
        else:
            for ticker, info in all_markets.items():
                disp = info["display_strike"]
                otm = abs((disp - self.spot_price) / self.spot_price * 100)
                if otm <= OTM_FILTER_PCT:
                    self.tracked[ticker] = info

        print(f"[Recorder] Tracking {len(self.tracked)} markets "
              f"(within {OTM_FILTER_PCT}% OTM of ${self.spot_price:,.0f})")

        # Start Kalshi market WS — orderbook deltas, fills, public trades.
        tickers = list(self.tracked.keys())
        if tickers:
            self.ws_feed = KalshiWsFeed(
                self.api,
                on_update=self._on_ws_update,
                on_fill=self._on_ws_fill,
                on_trade=self._on_ws_trade,
                on_book_change=self._on_ws_book_change,
            )
            self.ws_feed.start(tickers)
            print(f"[Recorder] Kalshi market WS started for {len(tickers)} tickers "
                  "(orderbook + fill + trade)")

        # Start dedicated user_orders WS on its own host (every order
        # state change for this account, not market-scoped).
        self.user_orders_feed = UserOrdersWsFeed(
            self.api, on_order_event=self._on_ws_order_event,
        )
        self.user_orders_feed.start()

        # Seed known fills so we don't re-record old ones
        self._seed_fills()

        # Start UDP listener for theo events from the app
        self._start_theo_listener()

        # Main loop
        self._run_loop()

    def _conn_for(self, event_ticker: str) -> sqlite3.Connection:
        """Get or create per-event SQLite connection."""
        c = self.conns.get(event_ticker)
        if c is None:
            c = init_db(event_db_path(event_ticker))
            self.conns[event_ticker] = c
        return c

    def _seed_fills(self):
        """Load existing fills from API. Record any not already in DB, then mark all as seen."""
        try:
            # Build set of already-recorded client_order_ids across all event DBs
            existing_coids = set()
            for et in set(info["event_ticker"] for info in self.tracked.values()):
                if not et:
                    continue
                try:
                    cursor = self._conn_for(et).execute(
                        "SELECT DISTINCT client_order_id FROM fills WHERE client_order_id != ''"
                    )
                    for r in cursor.fetchall():
                        existing_coids.add(r[0])
                except Exception:
                    pass

            fills = self.api.get_fills()
            recorded = 0
            for f in fills:
                fid = f.get("trade_id") or f.get("fill_id") or f.get("id", "")
                if not fid:
                    continue
                fid_str = str(fid)
                self._seen_fill_ids.add(fid_str)

                # Record if not already in DB
                ticker = f.get("ticker", "")
                info = self.tracked.get(ticker, {})
                if not info:
                    continue
                event_ticker = info.get("event_ticker", "")
                if not event_ticker:
                    continue

                action = f.get("action", "")
                side = f.get("side", "yes")
                price = float(f.get("yes_price_dollars", 0) or 0)
                count = float(f.get("count_fp", f.get("count", 0)))
                if side == "no":
                    side = "yes"
                fee = float(f.get("fee_cost", 0) or 0)
                is_taker = 1 if f.get("is_taker") else 0

                order_id = f.get("order_id", "")
                client_order_id = ""
                if order_id:
                    try:
                        order = self.api.get_order(order_id)
                        client_order_id = order.get("order", order).get("client_order_id", "")
                    except Exception:
                        pass

                # Skip if already recorded
                if client_order_id and client_order_id in existing_coids:
                    continue

                ts = f.get("created_time", "")
                if not ts:
                    ts = datetime.now(tz=timezone.utc).isoformat()

                conn = self._conn_for(event_ticker)
                with self._db_lock:
                    conn.execute("""
                        INSERT INTO fills (
                            ts, ticker, event_ticker, action, side, count, price, strike,
                            spot_bid, spot_ask, kalshi_yes_bid, kalshi_yes_ask,
                            client_order_id, fee, is_taker
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        ts, ticker, event_ticker, action, side,
                        count, price, info.get("display_strike", 0.0),
                        self.spot_bid or self.spot_price, self.spot_ask or self.spot_price,
                        0, 0, client_order_id, fee, is_taker,
                    ))
                    conn.commit()
                recorded += 1

            print(f"[Recorder] Seeded {len(self._seen_fill_ids)} fills, recorded {recorded} new")
        except Exception as e:
            print(f"[Recorder] Seed fills failed: {e}")

    def _poll_fills(self):
        """Poll REST API for new fills and record any we haven't seen."""
        try:
            fills = self.api.get_fills()
        except Exception as e:
            print(f"[Recorder] Fill poll failed: {e}")
            return

        spot_b = self.spot_bid if self.spot_bid > 0 else self.spot_price
        spot_a = self.spot_ask if self.spot_ask > 0 else self.spot_price

        for f in fills:
            fid = str(f.get("trade_id") or f.get("fill_id") or f.get("id", ""))
            if not fid or fid in self._seen_fill_ids:
                continue
            self._seen_fill_ids.add(fid)

            ticker = f.get("ticker", "")
            action = f.get("action", "")
            side = f.get("side", "yes")
            count = float(f.get("count_fp", f.get("count", 0)))
            price = float(f.get("yes_price_dollars", 0) or f.get("yes_price", 0) or 0)

            # Normalize to yes terms
            # API: sell no = Kalshi UI "Sold Yes" = sell yes
            # API: buy no = Kalshi UI "Bought Yes" (via no side) = buy yes
            # The yes_price_dollars is always the yes-equivalent price
            if side == "no":
                side = "yes"
                # action stays the same — sell no = sell yes, buy no = buy yes
            fee = float(f.get("fee_cost", 0) or 0)
            is_taker = 1 if f.get("is_taker") else 0

            info = self.tracked.get(ticker, {})
            strike = info.get("display_strike", 0.0)
            event_ticker = info.get("event_ticker", "")

            # Look up order to get client_order_id (fill doesn't have it)
            client_order_id = ""
            order_id = f.get("order_id", "")
            if order_id:
                try:
                    order = self.api.get_order(order_id)
                    client_order_id = order.get("order", order).get("client_order_id", "")
                except Exception:
                    pass

            bk = self.book.get(ticker, {})

            if not event_ticker:
                continue
            conn = self._conn_for(event_ticker)
            with self._db_lock:
                conn.execute("""
                    INSERT INTO fills (
                        ts, ticker, event_ticker, action, side, count, price, strike,
                        spot_bid, spot_ask, kalshi_yes_bid, kalshi_yes_ask,
                        client_order_id, fee, is_taker
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    datetime.now(tz=timezone.utc).isoformat(),
                    ticker, event_ticker, action, side, count, price, strike,
                    spot_b, spot_a,
                    bk.get("yes_bid", 0), bk.get("yes_ask", 0),
                    client_order_id, fee, is_taker,
                ))
                conn.commit()

            tag = "init" if client_order_id.startswith("init_") else \
                  "phase3" if client_order_id.startswith("phase3_") else \
                  "flat" if client_order_id.startswith("flat_") else "?"
            role = "TAKER" if is_taker else "MAKER"
            print(f"[Recorder] FILL: {ticker} {action} {side} x{count} "
                  f"@ ${price:.2f} [{tag}|{role}] fee=${fee:.2f}")

    # ---- Auto-archival to S3 ----------------------------------------------
    # Event tickers look like KXBTCD-26MAY0517 → year/month/day/hour.
    _EVENT_RE = re.compile(r"^KXBTCD-(\d{2})([A-Z]{3})(\d{2})(\d{2})$")
    _MONTH_MAP = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

    def _parse_event_close(self, event_ticker: str):
        m = self._EVENT_RE.match(event_ticker)
        if not m:
            return None
        yy, mon, dd, hh = m.groups()
        month = self._MONTH_MAP.get(mon)
        if not month:
            return None
        try:
            return datetime(2000 + int(yy), month, int(dd), int(hh), 0, 0,
                            tzinfo=timezone.utc)
        except Exception:
            return None

    def _archive_expired_events(self):
        """Find DBs whose close is more than ARCHIVE_AFTER_DAYS in the past,
        upload to s3://<bucket>/<prefix><yyyy>/<mm>/<event>.db, and delete
        locally on verified upload.  No-op if KALSHI_ARCHIVE_BUCKET unset.
        """
        bucket = os.environ.get(ARCHIVE_BUCKET_ENV, "").strip()
        if not bucket:
            return
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=ARCHIVE_AFTER_DAYS)
        # Don't touch DBs we currently have a connection open to —
        # close those first if they qualify.
        for path in sorted(DATA_DIR.glob("*.db")):
            if path.name == "recorder.db":
                continue
            event_ticker = path.stem
            close = self._parse_event_close(event_ticker)
            if close is None or close >= cutoff:
                continue
            # Close any cached connection for this event before uploading
            conn = self.conns.pop(event_ticker, None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            key = f"{ARCHIVE_PREFIX}{close:%Y/%m}/{event_ticker}.db"
            s3_uri = f"s3://{bucket}/{key}"
            try:
                size = path.stat().st_size
            except OSError:
                continue
            print(f"[Archive] {path.name} ({size/1024/1024:.1f}MB) → {s3_uri}")
            try:
                cp = subprocess.run(
                    ["aws", "s3", "cp", str(path), s3_uri,
                     "--storage-class", "STANDARD_IA"],
                    capture_output=True, text=True, timeout=600,
                )
                if cp.returncode != 0:
                    print(f"[Archive] upload failed: {cp.stderr.strip()}")
                    continue
                verify = subprocess.run(
                    ["aws", "s3api", "head-object",
                     "--bucket", bucket, "--key", key],
                    capture_output=True, text=True, timeout=30,
                )
                if verify.returncode != 0:
                    print(f"[Archive] verify failed — leaving local file intact")
                    continue
                path.unlink()
                print(f"[Archive] freed {size/1024/1024:.1f}MB locally")
            except subprocess.TimeoutExpired:
                print(f"[Archive] aws CLI timed out for {path.name}")
            except FileNotFoundError:
                print("[Archive] aws CLI not found — install or skip archival")
                return
            except Exception as e:
                print(f"[Archive] error: {e}")

    def _run_loop(self):
        last_snapshot = 0
        last_refilter = 0
        last_archive = 0
        # Fills + orders now arrive via WS — no REST polling needed.
        # Snapshots run on a 30s reconciliation cadence; the events table
        # captures everything in between with native timestamps.
        # On startup, archive immediately — handles the case where the
        # process was down and DBs accumulated past the cutoff.
        try:
            self._archive_expired_events()
        except Exception as e:
            print(f"[Archive] startup archival failed: {e}")
        last_archive = time.time()

        while self.running:
            time.sleep(0.5)
            now = time.time()

            if now - last_refilter >= 60:
                last_refilter = now
                try:
                    self._refilter()
                except Exception as e:
                    print(f"[Recorder] refilter failed: {e}")

            if now - last_snapshot >= SNAPSHOT_INTERVAL:
                last_snapshot = now
                try:
                    self._take_snapshot()
                except Exception as e:
                    print(f"[Recorder] take_snapshot failed: {e}")

            if now - last_archive >= ARCHIVE_INTERVAL_SEC:
                last_archive = now
                try:
                    self._archive_expired_events()
                except Exception as e:
                    print(f"[Archive] periodic archival failed: {e}")

    def _refilter(self):
        """Pick up new markets, drop expired ones."""
        if self.spot_price <= 0:
            return
        try:
            now = datetime.now(tz=timezone.utc)
            new_tickers = []

            events = discover_events_for_series(self.api, SERIES)

            # Keep only daily/weekly 5pm ET events (close hour = 21 UTC)
            selected = []
            for event in events:
                cs = event.get("close_time", "")
                if not cs:
                    continue
                try:
                    cu = datetime.fromisoformat(cs.replace("Z", "+00:00"))
                    if cu > now and cu.hour == 21:
                        selected.append(event)
                except Exception:
                    continue

            if not selected:
                return

            for event in selected:
                et = event["event_ticker"]
                close_str = event.get("close_time", "")

                for m in event["markets"]:
                    ticker = m["ticker"]
                    if ticker in self.tracked:
                        continue
                    raw = parse_strike(ticker)
                    if raw <= 0:
                        continue
                    disp = display_strike(raw)
                    otm = abs((disp - self.spot_price) / self.spot_price * 100)
                    if otm <= OTM_FILTER_PCT:
                        self.tracked[ticker] = {
                            "display_strike": disp,
                            "event_ticker": et,
                            "close_time": close_str,
                        }
                        new_tickers.append(ticker)

            # Drop expired
            expired = []
            for ticker, info in self.tracked.items():
                close_str = info.get("close_time", "")
                if close_str:
                    try:
                        close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        if (now - close_utc).total_seconds() > 600:
                            expired.append(ticker)
                    except Exception:
                        pass
            for ticker in expired:
                del self.tracked[ticker]

            if expired:
                print(f"[Recorder] Dropped {len(expired)} expired markets")
            if new_tickers and self.ws_feed:
                self.ws_feed.subscribe_tickers(new_tickers)
                print(f"[Recorder] Added {len(new_tickers)} new markets "
                      f"(total: {len(self.tracked)})")

        except Exception as e:
            print(f"[Recorder] Refilter error: {e}")

    def _take_snapshot(self):
        if not self.tracked:
            return

        spot_b = self.spot_bid if self.spot_bid > 0 else self.spot_price
        spot_a = self.spot_ask if self.spot_ask > 0 else self.spot_price
        spot_mid = (spot_b + spot_a) / 2 if spot_b > 0 and spot_a > 0 else self.spot_price
        if spot_mid <= 0:
            return

        ts = datetime.now(tz=timezone.utc).isoformat()
        now_epoch_ms = datetime.now(tz=timezone.utc).timestamp() * 1000
        # Group rows by event_ticker so we can write to per-event DBs
        rows_by_event: dict[str, list] = {}

        for ticker, info in self.tracked.items():
            disp = info["display_strike"]
            close = info.get("close_time", "")
            event_ticker = info.get("event_ticker", "")
            if not event_ticker:
                continue

            # Compute T
            T = 0.0
            if close:
                try:
                    close_utc = datetime.fromisoformat(close.replace("Z", "+00:00"))
                    T = max((close_utc.timestamp() * 1000 - now_epoch_ms)
                            / 1000.0 / (365.25 * 24 * 3600), 0.0)
                except Exception:
                    pass

            # Kalshi book
            bk = self.book.get(ticker, {})
            yes_bid = bk.get("yes_bid", 0.0)
            yes_ask = bk.get("yes_ask", 0.0)
            bid_size = bk.get("bid_size", 0)
            ask_size = bk.get("ask_size", 0)

            # Compute IVs
            mid_price = (yes_bid + yes_ask) / 2.0 if yes_bid > 0 and yes_ask > 0 else 0.0
            mid_iv_val = implied_vol(mid_price, spot_mid, disp, T) if mid_price > 0 else 0.0
            bid_iv_val = implied_vol(yes_bid, spot_mid, disp, T) if yes_bid > 0 else 0.0
            ask_iv_val = implied_vol(yes_ask, spot_mid, disp, T) if yes_ask > 0 else 0.0

            rows_by_event.setdefault(event_ticker, []).append((
                ts, ticker, event_ticker, disp, close, T,
                yes_bid, yes_ask, bid_size, ask_size,
                spot_b, spot_a, spot_mid,
                mid_iv_val, bid_iv_val, ask_iv_val,
            ))

        total_rows = 0
        for event_ticker, rows in rows_by_event.items():
            if not rows:
                continue
            try:
                conn = self._conn_for(event_ticker)
                with self._db_lock:
                    conn.executemany("""
                        INSERT INTO snapshots (
                            ts, ticker, event_ticker, strike, close_time, T,
                            kalshi_yes_bid, kalshi_yes_ask, bid_size, ask_size,
                            spot_bid, spot_ask, spot_mid,
                            mid_iv, bid_iv, ask_iv
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, rows)
                    conn.commit()
                total_rows += len(rows)
            except Exception as e:
                # Don't let a transient DB error crash the recorder —
                # next snapshot tick will retry.  Drop the cached
                # connection so we reopen fresh.
                print(f"[Recorder] snapshot write failed for {event_ticker}: {e}")
                try:
                    if event_ticker in self.conns:
                        self.conns[event_ticker].close()
                except Exception:
                    pass
                self.conns.pop(event_ticker, None)

        if total_rows:
            sample_T = next(iter(rows_by_event.values()))[0][5]
            print(f"[Recorder] Snapshot: {total_rows} markets across "
                  f"{len(rows_by_event)} events, spot=${spot_mid:,.2f}, T={sample_T:.6f}y")

    # --- Theo event listener (UDP from app) ---

    def _start_theo_listener(self):
        """Bind a UDP socket on THEO_PUB_PORT and run a background thread
        that consumes theo events from the app and writes them to the
        events table.  No-op (with a warning) if the port is already
        bound by another process."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", THEO_PUB_PORT))
            sock.settimeout(1.0)  # so the loop can check `self.running`
        except Exception as e:
            print(f"[Recorder] Could not bind theo listener: {e}")
            return
        self._theo_sock = sock
        self._theo_thread = threading.Thread(
            target=self._theo_listener_loop, daemon=True,
            name="theo-listener",
        )
        self._theo_thread.start()
        print(f"[Recorder] Theo listener bound on UDP {THEO_PUB_PORT}")

    def _theo_listener_loop(self):
        """Receive UDP packets, parse JSON, write to events table."""
        sock = self._theo_sock
        if sock is None:
            return
        recv_count = 0
        while self.running:
            try:
                data, _addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed during shutdown
                return
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            event_ticker = msg.get("event_ticker", "")
            ticker = msg.get("ticker", "")
            event_type = msg.get("event_type", "theo")
            payload = msg.get("payload", {})
            ts_us = int(msg.get("ts_us") or (time.time() * 1_000_000))
            if not event_ticker:
                continue
            try:
                conn = self._conn_for(event_ticker)
                with self._db_lock:
                    conn.execute(
                        "INSERT INTO events (ts_us, event_type, ticker, payload) "
                        "VALUES (?,?,?,?)",
                        (ts_us, event_type, ticker or None,
                         json.dumps(payload, separators=(",", ":"))),
                    )
                    conn.commit()
                recv_count += 1
                if recv_count % 200 == 0:
                    print(f"[Recorder] Theo events received: {recv_count}")
            except Exception as e:
                print(f"[Recorder] Theo write failed: {e}")

    # --- Event recording ---

    def _record_event(self, event_type: str, ticker: str, payload: dict):
        """Append a single event row to the appropriate per-event DB.

        For market-scoped events (book/trade/order), look up the event_ticker
        from `self.tracked`.  For account-wide events (spot), write to ALL
        currently-tracked event DBs so each per-event file is self-contained.
        """
        ts_us = int(time.time() * 1_000_000)
        payload_json = json.dumps(payload, separators=(",", ":"))

        if ticker:
            # Market-scoped — write to one DB
            info = self.tracked.get(ticker)
            if not info:
                return
            event_ticker = info.get("event_ticker", "")
            if not event_ticker:
                return
            try:
                conn = self._conn_for(event_ticker)
                with self._db_lock:
                    conn.execute(
                        "INSERT INTO events (ts_us, event_type, ticker, payload) "
                        "VALUES (?,?,?,?)",
                        (ts_us, event_type, ticker, payload_json),
                    )
                    conn.commit()
            except Exception as e:
                print(f"[Recorder] events write failed: {e}")
        else:
            # Account-wide (e.g. spot) — write to every active event DB so
            # each file holds the spot context for its window.
            seen = set()
            for info in self.tracked.values():
                et = info.get("event_ticker", "")
                if not et or et in seen:
                    continue
                seen.add(et)
                try:
                    conn = self._conn_for(et)
                    with self._db_lock:
                        conn.execute(
                            "INSERT INTO events (ts_us, event_type, ticker, payload) "
                            "VALUES (?,?,?,?)",
                            (ts_us, event_type, None, payload_json),
                        )
                        conn.commit()
                except Exception:
                    pass

    # --- Callbacks ---

    def _on_price(self, price: float, bid: float = 0.0, ask: float = 0.0):
        self.spot_price = price
        if bid > 0:
            self.spot_bid = bid
        if ask > 0:
            self.spot_ask = ask
        # Log every spot tick — Coinbase fires only on TOB change, so this
        # is event-rate, not poll-rate.
        self._record_event("spot", "", {
            "price": price,
            "bid": self.spot_bid,
            "ask": self.spot_ask,
        })

    def _on_ws_update(self, ticker: str, yes_bid: float, yes_ask: float,
                      bid_size: int = 0, ask_size: int = 0):
        self.book[ticker] = {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }
        # Folded TOB record — useful for queries that just want the
        # current best bid/ask without scanning every depth change.
        self._record_event("book_tob", ticker, {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spot_bid": self.spot_bid,
            "spot_ask": self.spot_ask,
        })

    def _on_ws_book_change(self, ticker: str, side: str,
                           price: float, delta: float):
        """Raw level-by-level book delta — every depth change captured."""
        self._record_event("book_delta", ticker, {
            "side": side,           # "yes" or "no"
            "price": price,
            "delta": delta,
        })

    def _on_ws_trade(self, msg: dict):
        """Public trade tape — anyone hitting any level on a market we track."""
        ticker = msg.get("market_ticker", "")
        if not ticker or ticker not in self.tracked:
            return
        self._record_event("trade", ticker, msg)

    def _on_ws_order_event(self, msg: dict):
        """Every state change on our own orders.  Detects new partial fills
        by tracking each order's running fill_count and emitting them to
        the legacy `fills` table for back-compat with the position manager.

        user_order msgs use `ticker` (not `market_ticker` like the book
        channel) — try both for robustness.
        """
        ticker = msg.get("ticker") or msg.get("market_ticker", "")
        order_id = msg.get("order_id", "")

        # Always log the raw event — even if the order is on a market
        # we don't currently track (e.g. expired), it's useful forensics
        if ticker and ticker in self.tracked:
            self._record_event("order", ticker, msg)

        # Detect new fills by looking at fill_count delta.  Also detect
        # whether the new portion was taker by looking at the cumulative
        # taker_fill_cost delta — user_orders msgs don't have a per-fill
        # is_taker flag, just running totals.
        if not order_id:
            return
        new_fill_count = float(msg.get("fill_count_fp", msg.get("fill_count", 0)) or 0)
        prev_fill_count = self._order_fill_count.get(order_id, 0.0)
        delta = new_fill_count - prev_fill_count
        if delta > 0:
            new_taker_cost = float(msg.get("taker_fill_cost_dollars", 0) or 0)
            prev_taker_cost = self._order_taker_cost.get(order_id, 0.0)
            taker_cost_delta = new_taker_cost - prev_taker_cost
            # If this fill increment included any taker portion, treat
            # the whole increment as taker.  In practice phase 3 orders
            # cross immediately and partials are rare; init/flat are
            # post_only so taker_cost stays 0 the entire life of the order.
            is_taker_fill = taker_cost_delta > 1e-9
            self._order_fill_count[order_id] = new_fill_count
            self._order_taker_cost[order_id] = new_taker_cost
            self._record_fill_from_order(msg, delta, is_taker_fill)

    def _record_fill_from_order(self, order_msg: dict, fill_delta: float,
                                is_taker_fill: bool = False):
        """Insert a row into the legacy `fills` table from a user_orders
        event.  `fill_delta` is the new contracts filled since the last
        seen state for this order_id.  `is_taker_fill` is derived by the
        caller from the per-event taker_fill_cost delta — user_orders
        msgs don't carry a per-fill is_taker flag."""
        ticker = order_msg.get("ticker") or order_msg.get("market_ticker", "")
        info = self.tracked.get(ticker, {})
        if not info:
            return
        event_ticker = info.get("event_ticker", "")
        if not event_ticker:
            return

        order_id = order_msg.get("order_id", "")
        # Use (order_id, fill_count) as a dedup key
        dedup_key = f"{order_id}:{order_msg.get('fill_count_fp', order_msg.get('fill_count', 0))}"
        if dedup_key in self._seen_fill_ids:
            return
        self._seen_fill_ids.add(dedup_key)

        # user_orders msgs don't include `action` (buy/sell) directly.
        # We derive it from `is_yes`: buying YES means is_yes=True, selling
        # YES means is_yes=False (the strategy never trades the no side).
        # For Kalshi's binary contracts:
        #   bid yes  → is_yes=True   → action="buy"
        #   ask yes  → is_yes=False  → action="sell"
        is_yes = bool(order_msg.get("is_yes"))
        action = order_msg.get("action") or ("buy" if is_yes else "sell")
        side = order_msg.get("side", "yes")
        if side == "no":
            side = "yes"
        price = float(order_msg.get("yes_price_dollars", 0) or 0)
        is_taker = 1 if is_taker_fill else 0
        # Per-fill fee isn't directly on user_orders msgs; use 0 as a
        # placeholder — REST `get_fills` reconciliation can backfill later.
        fee = 0.0

        bk = self.book.get(ticker, {})
        spot_b = self.spot_bid or self.spot_price
        spot_a = self.spot_ask or self.spot_price

        # client_order_id IS on user_orders msgs (one of the few places it
        # comes through directly without a get_order lookup)
        client_order_id = order_msg.get("client_order_id", "")

        try:
            conn = self._conn_for(event_ticker)
            with self._db_lock:
                conn.execute("""
                    INSERT INTO fills (
                        ts, ticker, event_ticker, action, side, count, price, strike,
                        spot_bid, spot_ask, kalshi_yes_bid, kalshi_yes_ask,
                        client_order_id, fee, is_taker
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    datetime.now(tz=timezone.utc).isoformat(),
                    ticker, event_ticker, action, side,
                    fill_delta, price, info.get("display_strike", 0.0),
                    spot_b, spot_a,
                    bk.get("yes_bid", 0), bk.get("yes_ask", 0),
                    client_order_id, fee, is_taker,
                ))
                conn.commit()
        except Exception as e:
            print(f"[Recorder] fill write failed: {e}")
            return

        tag = ("init" if client_order_id.startswith("init_") else
               "phase3t" if client_order_id.startswith("phase3t_") else
               "phase3d" if client_order_id.startswith("phase3d_") else
               "phase3" if client_order_id.startswith("phase3_") else
               "flat" if client_order_id.startswith("flat_") else "?")
        role = "TAKER" if is_taker else "MAKER"
        print(f"[Recorder] FILL: {ticker} {action} {side} x{fill_delta} "
              f"@ ${price:.2f} [{tag}|{role}]")

    def _on_ws_fill(self, ticker: str, action: str, side: str,
                    price: float, count: int):
        """Legacy fill callback — the user_orders stream already handles
        fill detection via fill_count deltas, so this is now redundant.
        Kept as a safety-net in case user_orders disconnects briefly."""
        # No-op: order events are the source of truth for fills now.
        # If we wanted belt-and-suspenders dedup, we could record here too,
        # but in practice user_orders fires on every fill state change.
        pass

    def _shutdown(self):
        print("\n[Recorder] Shutting down...")
        self.running = False
        if self.price_feed:
            self.price_feed.stop()
        if self.ws_feed:
            self.ws_feed.stop()
        if self.user_orders_feed:
            self.user_orders_feed.stop()
        if self._theo_sock:
            try:
                self._theo_sock.close()
            except Exception:
                pass
        for conn in self.conns.values():
            try:
                conn.close()
            except Exception:
                pass
        print("[Recorder] Done")


# =============================================================================
# Export
# =============================================================================

def export_day(date_str: str):
    """Export a day's snapshots to CSV (across all per-event DBs)."""
    import pandas as pd
    dfs = []
    for db_file in DATA_DIR.glob("*.db"):
        if db_file.name == "recorder.db":
            continue
        try:
            conn = sqlite3.connect(str(db_file))
            df = pd.read_sql(
                "SELECT * FROM snapshots WHERE ts LIKE ?",
                conn, params=[f"{date_str}%"],
            )
            conn.close()
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        print(f"No snapshots for {date_str}")
        return
    df = pd.concat(dfs, ignore_index=True)
    out = DATA_DIR / f"snapshots_{date_str}.csv"
    df.to_csv(out, index=False)
    print(f"Exported {len(df)} rows to {out}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weekly BTC market recorder")
    parser.add_argument("--export", metavar="DATE",
                        help="Export a day to CSV (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.export:
        export_day(args.export)
    else:
        recorder = Recorder()
        recorder.start()
