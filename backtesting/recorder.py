"""
Kalshi Market Data Recorder

Records live orderbook snapshots and BTC spot price to SQLite databases.
Creates a new database file each day: market_data_2026-04-12.db

Features:
    - Multi-series support: record KXBTC, KXGOLDMON, etc. simultaneously
    - Auto-rediscovery: finds new weekly events every N hours
    - Daily file rotation: new DB file at midnight, no restart needed
    - Skip empty books: only records tickers with a bid or ask
    - BTC spot price: recorded every 1 second
    - Orderbook snapshots: configurable interval (default 5s)

Usage:
    python3 recorder.py
    python3 recorder.py --series KXBTC KXGOLDMON --dir data/
    python3 recorder.py --interval 10 --rediscover 2 --weeks 3

Ctrl+C to stop gracefully.
"""

import argparse
import os
import signal
import sqlite3
import time
import threading
from datetime import datetime, date
import sys
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/4RunnerApp")

from kalshi_api import KalshiAPI
from ws_feed import KalshiWsFeed
from btc_price_feed import BtcPriceFeed
from market_discovery import discover_weekly_events, parse_strike


# =============================================================================
# Database Manager (with daily rotation)
# =============================================================================

class MarketDatabase:
    """SQLite database with automatic daily file rotation."""

    def __init__(self, db_dir: str = "."):
        self.db_dir = db_dir
        self._lock = threading.Lock()
        self.conn = None
        self.current_date = None

        os.makedirs(db_dir, exist_ok=True)
        self._rotate_if_needed()

    def _db_path_for_date(self, d: date) -> str:
        return os.path.join(self.db_dir, f"market_data_{d.isoformat()}.db")

    def _rotate_if_needed(self):
        """Check if we need a new database file for today."""
        today = date.today()
        if today == self.current_date:
            return

        with self._lock:
            # Close old connection
            if self.conn:
                self.conn.close()
                print(f"[DB] Closed {self._db_path_for_date(self.current_date)}")

            # Open new connection for today
            self.current_date = today
            db_path = self._db_path_for_date(today)
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self._create_tables()
            print(f"[DB] Opened {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS orderbook (
                timestamp TEXT NOT NULL,
                series TEXT NOT NULL,
                ticker TEXT NOT NULL,
                event_ticker TEXT NOT NULL,
                yes_sub_title TEXT,
                yes_bid REAL,
                yes_ask REAL,
                btc_price REAL,
                strike REAL
            );

            CREATE TABLE IF NOT EXISTS btc_price (
                timestamp TEXT NOT NULL,
                price REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ob_ticker_time
                ON orderbook(ticker, timestamp);

            CREATE INDEX IF NOT EXISTS idx_ob_event_time
                ON orderbook(event_ticker, timestamp);

            CREATE INDEX IF NOT EXISTS idx_ob_series_time
                ON orderbook(series, timestamp);

            CREATE INDEX IF NOT EXISTS idx_btc_time
                ON btc_price(timestamp);
        """)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    def insert_orderbook_batch(self, rows: list):
        """Insert batch: (timestamp, series, ticker, event, subtitle, bid, ask, btc, strike)"""
        if not rows:
            return
        self._rotate_if_needed()
        with self._lock:
            self.conn.executemany(
                "INSERT INTO orderbook VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
            self.conn.commit()

    def insert_btc_price(self, timestamp: str, price: float):
        self._rotate_if_needed()
        with self._lock:
            self.conn.execute(
                "INSERT INTO btc_price VALUES (?,?)", (timestamp, price)
            )
            self.conn.commit()

    def get_row_counts(self) -> dict:
        self._rotate_if_needed()
        with self._lock:
            ob = self.conn.execute("SELECT COUNT(*) FROM orderbook").fetchone()[0]
            btc = self.conn.execute("SELECT COUNT(*) FROM btc_price").fetchone()[0]
        return {"orderbook": ob, "btc_price": btc}

    def get_series_counts(self) -> dict:
        """Row counts broken down by series."""
        self._rotate_if_needed()
        with self._lock:
            rows = self.conn.execute(
                "SELECT series, COUNT(*) FROM orderbook GROUP BY series"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def close(self):
        with self._lock:
            if self.conn:
                self.conn.close()
        print("[DB] Closed")


# =============================================================================
# Recorder
# =============================================================================

class MarketRecorder:
    """Coordinates websocket feeds, event discovery, and database recording."""

    def __init__(self, series_list: list, db_dir: str = ".",
                 snapshot_interval: int = 5, weeks_ahead: int = 2,
                 rediscover_hours: float = 4):
        """
        Args:
            series_list: list of series tickers e.g. ["KXBTC", "KXGOLDMON"]
            db_dir: directory for database files
            snapshot_interval: seconds between orderbook snapshots
            weeks_ahead: how many weeks of events to discover
            rediscover_hours: hours between event rediscovery checks
        """
        self.api = KalshiAPI()
        self.db = MarketDatabase(db_dir)
        self.series_list = series_list
        self.snapshot_interval = snapshot_interval
        self.weeks_ahead = weeks_ahead
        self.rediscover_hours = rediscover_hours

        # Live data state
        self.btc_price = 0.0
        self.orderbooks: dict[str, dict] = {}   # ticker -> {yes_bid, yes_ask}
        self.market_info: dict[str, dict] = {}   # ticker -> {series, event, subtitle, strike}

        # Feeds — keep references to prevent GC
        self.ws_feeds: list[KalshiWsFeed] = []
        self.btc_feed = None

        # BTC 1-second throttle
        self._last_btc_write = 0.0

        # Thread lock for orderbook/market_info modifications
        self._data_lock = threading.Lock()

        # Control
        self._running = False

    def start(self):
        """Discover events, start feeds, begin recording."""
        self._running = True

        # --- Initial discovery across all series ---
        self._discover_all_series()

        if not self.orderbooks:
            print("[RECORDER] No tickers found across any series, exiting")
            return

        # --- Start BTC price feed ---
        self.btc_feed = BtcPriceFeed(self._on_btc_price)
        self.btc_feed.start()

        # --- Start snapshot thread ---
        snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
        snapshot_thread.start()

        # --- Start rediscovery thread ---
        rediscover_thread = threading.Thread(target=self._rediscover_loop, daemon=True)
        rediscover_thread.start()

        print(f"\n[RECORDER] Orderbook snapshots every {self.snapshot_interval}s (active tickers only)")
        print(f"[RECORDER] BTC price every 1s")
        print(f"[RECORDER] Rediscovery every {self.rediscover_hours}h")
        print("[RECORDER] Press Ctrl+C to stop\n")

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def stop(self):
        """Stop all feeds and close database."""
        print("\n[RECORDER] Stopping...")
        self._running = False

        # Stop all websocket feeds
        for feed in self.ws_feeds:
            feed.stop()
        if self.btc_feed:
            self.btc_feed.stop()

        # Final stats
        counts = self.db.get_row_counts()
        series_counts = self.db.get_series_counts()
        print(f"[RECORDER] Today's totals: {counts['orderbook']:,} orderbook / "
              f"{counts['btc_price']:,} BTC price")
        for series, count in series_counts.items():
            print(f"  {series}: {count:,} rows")

        self.db.close()
        print("[RECORDER] Done")

    # --- Discovery ---

    def _discover_all_series(self):
        """Discover events for all series and start WS feeds."""
        for series in self.series_list:
            self._discover_series(series)

    def _discover_series(self, series: str) -> list:
        """Discover events for one series, register tickers, start WS feed.
        Returns list of new tickers found."""
        print(f"[RECORDER] Discovering {series}...")

        try:
            events = discover_weekly_events(self.api, series, self.weeks_ahead)
        except Exception as e:
            print(f"[RECORDER] Error discovering {series}: {e}")
            return []

        if not events:
            print(f"[RECORDER] No events for {series}")
            return []

        # Find new tickers (not already tracked)
        new_tickers = []
        with self._data_lock:
            for event in events:
                event_ticker = event["event_ticker"]
                brackets = [m for m in event["markets"] if "-B" in m["ticker"]]

                for m in brackets:
                    ticker = m["ticker"]
                    if ticker not in self.orderbooks:
                        self.market_info[ticker] = {
                            "series": series,
                            "event_ticker": event_ticker,
                            "yes_sub_title": m.get("yes_sub_title", ""),
                            "strike": parse_strike(ticker),
                        }
                        self.orderbooks[ticker] = {"yes_bid": 0.0, "yes_ask": 0.0}
                        new_tickers.append(ticker)

                if brackets:
                    print(f"  {event_ticker}: {len(brackets)} brackets")

        # Start a WS feed for new tickers
        if new_tickers:
            print(f"[RECORDER] Subscribing to {len(new_tickers)} new {series} tickers")
            ws_feed = KalshiWsFeed(self.api, self._on_orderbook_update)
            ws_feed.start(new_tickers)
            self.ws_feeds.append(ws_feed)
        else:
            print(f"[RECORDER] No new tickers for {series}")

        return new_tickers

    def _rediscover_loop(self):
        """Periodically check for new events across all series."""
        while self._running:
            # Sleep for rediscover_hours, checking _running every second
            sleep_seconds = int(self.rediscover_hours * 3600)
            for _ in range(sleep_seconds):
                if not self._running:
                    return
                time.sleep(1)

            if not self._running:
                return

            print(f"\n[RECORDER] Periodic rediscovery ({self.rediscover_hours}h)...")
            total_new = 0

            for series in self.series_list:
                new_tickers = self._discover_series(series)
                total_new += len(new_tickers)

            if total_new > 0:
                print(f"[RECORDER] Rediscovery found {total_new} new tickers total")
            else:
                print(f"[RECORDER] Rediscovery: no new tickers")

            # Clean up stopped feeds
            self.ws_feeds = [f for f in self.ws_feeds if f._running]

    # --- Callbacks ---

    def _on_orderbook_update(self, ticker: str, yes_bid: float, yes_ask: float):
        """Called from Kalshi WS thread on every orderbook delta."""
        with self._data_lock:
            if ticker in self.orderbooks:
                self.orderbooks[ticker]["yes_bid"] = yes_bid
                self.orderbooks[ticker]["yes_ask"] = yes_ask

    def _on_btc_price(self, price: float):
        """Record BTC price at most once per second."""
        self.btc_price = price
        now = time.time()
        if now - self._last_btc_write >= 1.0:
            self._last_btc_write = now
            self.db.insert_btc_price(datetime.utcnow().isoformat(), price)

    # --- Snapshot Loop ---

    def _snapshot_loop(self):
        """Periodically snapshot all active orderbooks to database."""
        last_status = time.time()

        while self._running:
            time.sleep(self.snapshot_interval)
            if not self._running:
                break

            now = datetime.utcnow().isoformat()

            # Build batch — only tickers with a bid or ask
            rows = []
            with self._data_lock:
                for ticker, book in self.orderbooks.items():
                    # Skip empty books
                    if book["yes_bid"] <= 0 and book["yes_ask"] <= 0:
                        continue

                    info = self.market_info.get(ticker, {})
                    rows.append((
                        now,
                        info.get("series", ""),
                        ticker,
                        info.get("event_ticker", ""),
                        info.get("yes_sub_title", ""),
                        book["yes_bid"],
                        book["yes_ask"],
                        self.btc_price,
                        info.get("strike", 0.0),
                    ))

            if rows:
                self.db.insert_orderbook_batch(rows)

            # Status every 60 seconds
            current_time = time.time()
            if current_time - last_status >= 60:
                counts = self.db.get_row_counts()
                total_tickers = len(self.orderbooks)
                with self._data_lock:
                    active = sum(1 for b in self.orderbooks.values()
                                 if b["yes_bid"] > 0 or b["yes_ask"] > 0)

                db_path = self.db._db_path_for_date(self.db.current_date)
                try:
                    db_size = os.path.getsize(db_path) / (1024 * 1024)
                except OSError:
                    db_size = 0

                series_str = ", ".join(self.series_list)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"BTC: ${self.btc_price:,.2f} | "
                    f"Active: {active}/{total_tickers} | "
                    f"Series: {series_str} | "
                    f"Rows: {counts['orderbook']:,} ob / {counts['btc_price']:,} btc | "
                    f"Size: {db_size:.1f}MB"
                )
                last_status = current_time


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kalshi Market Data Recorder")
    parser.add_argument(
        "--series", type=str, nargs="+", default=["KXBTC"],
        help="Series tickers to record (default: KXBTC). Example: --series KXBTC KXGOLDMON"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Seconds between orderbook snapshots (default: 5)"
    )
    parser.add_argument(
        "--dir", type=str, default="marketdata/",
        help="Directory for database files (default: marketdata/)"
    )
    parser.add_argument(
        "--weeks", type=int, default=2,
        help="Weeks ahead to discover events (default: 2)"
    )
    parser.add_argument(
        "--rediscover", type=float, default=4,
        help="Hours between event rediscovery (default: 4)"
    )
    args = parser.parse_args()

    print(f"[RECORDER] Series: {args.series}")
    print(f"[RECORDER] Snapshot interval: {args.interval}s")
    print(f"[RECORDER] Rediscovery interval: {args.rediscover}h")
    print(f"[RECORDER] DB directory: {args.dir}")
    print()

    recorder = MarketRecorder(
        series_list=args.series,
        db_dir=args.dir,
        snapshot_interval=args.interval,
        weeks_ahead=args.weeks,
        rediscover_hours=args.rediscover,
    )

    # Graceful Ctrl+C
    def sigint_handler(*_):
        recorder.stop()
        exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    recorder.start()


if __name__ == "__main__":
    main()
