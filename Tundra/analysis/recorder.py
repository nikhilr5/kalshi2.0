"""Standalone weather-market recorder.

Captures everything needed to forward-validate a Kalshi daily-temperature
forecasting model against the live market:

    market_book   — top-of-book for every open weather market, written
                    EVENT-DRIVEN off the Kalshi orderbook_delta WS: one
                    row per top-of-book change (bid/ask/sizes + the parsed
                    bucket bounds).  This replaces the old ~5-min REST
                    snapshot poll — we now capture every book change.
    market_stats  — low-frequency (every STATS_INTERVAL_S) poll of
                    get_markets for the fields the WS book channel does
                    NOT carry: volume / open_interest / last_price.
    forecasts     — NWS deterministic daily-high (F) per tracked city/
                    station, on the forecast loop.
    settlements   — when a market resolves, its result + (where
                    available) the observed high.

Market data is acquired over ONE Kalshi WS connection (reused from
Aston/feeds/ws_feed.py — KalshiWsFeed).  The forecast loop, settlement
loop, and S3 rotation are unchanged and remain REST/NWS-driven.

THREAD SAFETY: the WS `on_update` callback fires on the feed's background
thread, concurrent with the forecast/settlement/stats loops.  All DB
writes go through WeatherStore's single lock + per-file connection
(check_same_thread=False), so callback writes are safe.

File layout: one SQLite per UTC day in Tundra/analysis/data/, e.g.
`WEATHER-26JUN18.db`, holding all four tables for that day.

Structure, S3 rotation, and robustness mirror the crypto recorder at
Aston/tools/recorder.py.  Forecast fetch + sliver filter + bucket parse
are reused verbatim from analysis/Aston/weather/weather_lib.py.

Usage:
    python3 recorder.py                 # all confirmed weather series
    python3 recorder.py --once          # one stats/forecast cycle, then exit
    python3 recorder.py --no-rotate     # skip the S3 rotation thread
"""

import argparse
import importlib.util
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Aston root holds kalshi_api.py; Aston/feeds holds the WS feed we reuse.
ASTON = Path(__file__).resolve().parents[2] / "Aston"
if str(ASTON) not in sys.path:
    sys.path.insert(0, str(ASTON))
_FEEDS = ASTON / "feeds"
if str(_FEEDS) not in sys.path:
    sys.path.insert(0, str(_FEEDS))
from kalshi_api import KalshiAPI  # noqa: E402
from ws_feed import KalshiWsFeed  # noqa: E402

# Reuse weather_lib by file location (its own sys.path math only works
# from its home dir, so we load it as a module from its absolute path
# rather than importing by name).
_WL_PATH = Path(__file__).resolve().parent / "weather_lib.py"
_spec = importlib.util.spec_from_file_location("weather_lib", _WL_PATH)
wl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wl)

_MONTHS = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
           7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}


# city -> config.  `series` is the Kalshi series_ticker; lat/lon is the
# NWS gridpoint to forecast; station/station_name is the ASOS the
# Climatological Report (Daily) reads the high from (the settlement
# site).  `kind` is 'high' or 'precip'.  Coordinates are the official
# NWS climate-site locations; verify the station mapping against each
# series' rules_primary (see REPORT.md settlement-station table).
CITIES = {
    # detailed series — rules name the exact site
    "NYC":   dict(series="KXHIGHNY",   lat=40.78,  lon=-73.97,  station="KNYC", station_name="Central Park, NY",            kind="high"),
    "LAX":   dict(series="KXHIGHLAX",  lat=33.94,  lon=-118.41, station="KLAX", station_name="Los Angeles Intl",           kind="high"),
    "CHI":   dict(series="KXHIGHCHI",  lat=41.79,  lon=-87.75,  station="KMDW", station_name="Chicago Midway",             kind="high"),
    "MIA":   dict(series="KXHIGHMIA",  lat=25.79,  lon=-80.32,  station="KMIA", station_name="Miami Intl",                 kind="high"),
    "PHIL":  dict(series="KXHIGHPHIL", lat=39.87,  lon=-75.23,  station="KPHL", station_name="Philadelphia Intl",          kind="high"),
    "DEN":   dict(series="KXHIGHDEN",  lat=39.85,  lon=-104.66, station="KDEN", station_name="Denver Intl",                kind="high"),
    # terse series — rules name only the city; station is best-guess (verify)
    "OKC":   dict(series="KXHIGHTOKC", lat=35.39,  lon=-97.60,  station="KOKC", station_name="Will Rogers, OKC",           kind="high"),
    "BOS":   dict(series="KXHIGHTBOS", lat=42.36,  lon=-71.01,  station="KBOS", station_name="Boston Logan",               kind="high"),
    "DAL":   dict(series="KXHIGHTDAL", lat=32.85,  lon=-96.85,  station="KDAL", station_name="Dallas Love Field",          kind="high"),
    "DC":    dict(series="KXHIGHTDC",  lat=38.85,  lon=-77.03,  station="KDCA", station_name="Reagan National (DCA)",      kind="high"),
    "HOU":   dict(series="KXHIGHTHOU", lat=29.98,  lon=-95.36,  station="KIAH", station_name="Houston Intercontinental",   kind="high"),
    "PHX":   dict(series="KXHIGHTPHX", lat=33.43,  lon=-112.01, station="KPHX", station_name="Phoenix Sky Harbor",         kind="high"),
    "SEA":   dict(series="KXHIGHTSEA", lat=47.45,  lon=-122.31, station="KSEA", station_name="Seattle-Tacoma",             kind="high"),
    "ATL":   dict(series="KXHIGHTATL", lat=33.64,  lon=-84.43,  station="KATL", station_name="Atlanta Hartsfield",         kind="high"),
    # precip
    "NYC_RAIN": dict(series="KXRAINNYC", lat=40.78, lon=-73.97, station="KNYC", station_name="Central Park, NY",           kind="precip"),
}


def db_path_for(day_utc, base_dir=DATA_DIR):
    yy = day_utc.year % 100
    suffix = f"{yy:02d}{_MONTHS[day_utc.month]}{day_utc.day:02d}"
    return base_dir / f"WEATHER-{suffix}.db"


def _f(v, default=None):
    """Parse a Kalshi *_fp / *_dollars string field to float; None-safe."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# =============================================================================
# SQLite writer — one open connection per per-day file (WAL mode)
# =============================================================================

class WeatherStore:
    def __init__(self, base_dir=DATA_DIR):
        self.base_dir = base_dir
        self._conns = {}
        self._lock = threading.Lock()

    def _conn_for(self, path):
        if path in self._conns:
            return self._conns[path]
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Event-driven top-of-book, one row per WS top-of-book change.
        # vol/OI/last are NOT in WS book messages — see market_stats.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ticker TEXT NOT NULL,
                city TEXT,
                series TEXT,
                event_day TEXT,
                kind TEXT,
                bucket_sub TEXT,
                bucket_kind TEXT,
                bucket_lo REAL,
                bucket_hi REAL,
                yes_bid REAL,
                yes_ask REAL,
                bid_size REAL,
                ask_size REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_book_ts ON market_book(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_book_ticker ON market_book(ticker)")
        # Low-frequency stats poll: the fields the WS book channel omits.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ticker TEXT NOT NULL,
                city TEXT,
                series TEXT,
                event_day TEXT,
                volume REAL,
                open_interest REAL,
                last_price REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_ts ON market_stats(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_stats_ticker ON market_stats(ticker)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                city TEXT,
                station TEXT,
                event_day TEXT,
                forecast_high REAL,
                source TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_ts ON forecasts(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fc_city ON forecasts(city)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS settlements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                ticker TEXT NOT NULL UNIQUE,
                city TEXT,
                series TEXT,
                event_day TEXT,
                result TEXT,
                observed_high REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_settle_ticker ON settlements(ticker)")
        conn.commit()
        self._conns[path] = conn
        return conn

    def _today_path(self):
        return db_path_for(datetime.now(tz=timezone.utc), self.base_dir)

    def write_book(self, row):
        """One top-of-book row.  Called from the WS feed thread on every
        TOB change — keep it cheap (single insert + commit under lock)."""
        path = self._today_path()
        with self._lock:
            conn = self._conn_for(path)
            conn.execute("""
                INSERT INTO market_book
                    (ts, ticker, city, series, event_day, kind, bucket_sub,
                     bucket_kind, bucket_lo, bucket_hi, yes_bid, yes_ask,
                     bid_size, ask_size)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, row)
            conn.commit()
        return 1

    def write_stats(self, rows):
        if not rows:
            return 0
        path = self._today_path()
        with self._lock:
            conn = self._conn_for(path)
            conn.executemany("""
                INSERT INTO market_stats
                    (ts, ticker, city, series, event_day,
                     volume, open_interest, last_price)
                VALUES (?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
        return len(rows)

    def write_forecasts(self, rows):
        if not rows:
            return 0
        path = self._today_path()
        with self._lock:
            conn = self._conn_for(path)
            conn.executemany("""
                INSERT INTO forecasts
                    (ts, city, station, event_day, forecast_high, source)
                VALUES (?,?,?,?,?,?)
            """, rows)
            conn.commit()
        return len(rows)

    def write_settlement(self, row):
        path = self._today_path()
        with self._lock:
            conn = self._conn_for(path)
            # UNIQUE(ticker): first observed settlement wins, ignore replays.
            cur = conn.execute("""
                INSERT OR IGNORE INTO settlements
                    (ts, ticker, city, series, event_day, result, observed_high)
                VALUES (?,?,?,?,?,?,?)
            """, row)
            conn.commit()
            return cur.rowcount   # 1 if inserted, 0 if a dup was ignored

    def close_conn(self, path):
        """Release the handle to one file (used before rotation deletes it)."""
        with self._lock:
            c = self._conns.pop(path, None)
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass

    def close_all(self):
        with self._lock:
            for c in self._conns.values():
                try:
                    c.close()
                except Exception:
                    pass
            self._conns.clear()


# =============================================================================
# Recorder
# =============================================================================

class WeatherRecorder:
    CYCLE_INTERVAL_S = 60           # housekeeping cadence (forecast/settle/discovery wake-ups)
    FORECAST_REFRESH_S = 1800       # re-fetch NWS forecasts every 30 min
    SETTLE_CHECK_S = 1800           # scan settled archive every 30 min
    DISCOVERY_INTERVAL_S = 1800     # re-query series for NEW tickers every 30 min
    STATS_INTERVAL_S = 1800         # low-freq vol/OI/last poll every 30 min

    S3_ROTATION_BUCKET = "s3://kalshibtc/weather-archive"
    ROTATION_AGE_S = 24 * 3600
    ROTATION_INTERVAL_S = 3600      # hourly scan

    def __init__(self, base_dir=DATA_DIR, do_rotate=True):
        self.api = KalshiAPI()
        self.store = WeatherStore(base_dir)
        self.do_rotate = do_rotate
        self._stop_event = threading.Event()
        self._rotation_thread = None
        # forecast cache: city -> (fetched_monotonic, {day_iso: high_F})
        self._fc_cache = {}
        # settlements already recorded this process, to skip re-checking.
        self._settled = set()
        self._last_settle_check = 0.0   # monotonic; 0 forces a first scan
        self._last_discovery = 0.0
        self._last_stats = 0.0

        # WS feed (one connection, all weather tickers).
        self.feed = KalshiWsFeed(self.api, on_update=self._on_book_update)
        self._subscribed = set()        # tickers handed to the feed
        # ticker -> resolved metadata for the WS callback.  series -> city
        # is static; bucket is parsed lazily from the subtitle on discovery.
        self._series_to_city = {cfg["series"]: city
                                for city, cfg in CITIES.items()}
        self._ticker_meta = {}          # ticker -> dict(city, series, event_day, kind, sub, bkind, blo, bhi)
        self._meta_lock = threading.Lock()

    # ---- forecasts ----

    def _forecast_for_city(self, city, cfg):
        """Cached NWS forecast highs for a city.  Refreshes every
        FORECAST_REFRESH_S.  Returns {day_iso: high_F} or {} (never raises)."""
        now = time.monotonic()
        cached = self._fc_cache.get(city)
        if cached and (now - cached[0]) < self.FORECAST_REFRESH_S:
            return cached[1]
        highs = None
        try:
            highs = wl.nws_forecast_highs(cfg["lat"], cfg["lon"])
        except Exception as e:
            print(f"[forecast] {city} fetch raised: {e}")
        if highs is None:
            # keep the stale cache if we have one; better than nothing
            return cached[1] if cached else {}
        self._fc_cache[city] = (now, highs)
        return highs

    def _collect_forecasts(self, ts):
        """One forecast row per (city, future event_day) from the NWS grid.
        Precip cities are skipped (no high to forecast)."""
        rows = []
        for city, cfg in CITIES.items():
            if cfg["kind"] != "high":
                continue
            highs = self._forecast_for_city(city, cfg)
            for day_iso, high_f in sorted(highs.items()):
                rows.append((ts, city, cfg["station"], day_iso,
                             round(high_f, 2), "NWS"))
        return rows

    # ---- market data: WS book (event-driven) + low-freq stats poll ----

    def _on_book_update(self, ticker, yes_bid, yes_ask, bid_size, ask_size):
        """WS callback — fires on the feed's background thread on every
        top-of-book change.  Resolve cached metadata and write one row.
        Never raises (the feed swallows callback exceptions, but we guard
        anyway so a parse miss doesn't drop the write)."""
        if self._stop_event.is_set():
            return
        with self._meta_lock:
            meta = self._ticker_meta.get(ticker)
        if meta is None:
            # Unknown ticker (e.g. delta arrived before discovery cached
            # its metadata).  Record TOB with nulls rather than dropping.
            meta = dict(city=None, series=None, event_day=wl.ticker_day(ticker),
                        kind=None, sub=None, bkind=None, blo=None, bhi=None)
        ts = datetime.now(tz=timezone.utc).isoformat()
        try:
            self.store.write_book((
                ts, ticker, meta["city"], meta["series"], meta["event_day"],
                meta["kind"], meta["sub"], meta["bkind"], meta["blo"],
                meta["bhi"], yes_bid, yes_ask, bid_size, ask_size,
            ))
        except Exception as e:
            print(f"[book] write failed {ticker}: {e}")

    def _discover(self, ts, want_stats):
        """Re-query each weather series' open markets.  Cache metadata for
        the WS callback, subscribe any NEW tickers on the existing feed,
        and (when want_stats) emit vol/OI/last rows from the same payload.
        Returns (n_new_tickers, n_stats_rows)."""
        new_tickers = []
        stats_rows = []
        for city, cfg in CITIES.items():
            if self._stop_event.is_set():
                break
            try:
                markets = self.api.get_markets(series_ticker=cfg["series"],
                                               status="open")
            except Exception as e:
                print(f"[discover] {cfg['series']} get_markets failed: {e} — skip")
                continue
            for m in markets:
                ticker = m.get("ticker")
                if not ticker:
                    continue
                if ticker not in self._subscribed:
                    sub = m.get("yes_sub_title") or m.get("subtitle")
                    bk = wl.parse_bucket(sub) if cfg["kind"] == "high" else None
                    bkind, blo, bhi = (bk if bk else (None, None, None))
                    with self._meta_lock:
                        self._ticker_meta[ticker] = dict(
                            city=city, series=cfg["series"],
                            event_day=wl.ticker_day(ticker), kind=cfg["kind"],
                            sub=sub, bkind=bkind, blo=blo, bhi=bhi)
                    new_tickers.append(ticker)
                if want_stats:
                    stats_rows.append((
                        ts, ticker, city, cfg["series"], wl.ticker_day(ticker),
                        _f(m.get("volume_fp")),
                        _f(m.get("open_interest_fp")),
                        _f(m.get("last_price_dollars")),
                    ))
        if new_tickers:
            self._subscribed.update(new_tickers)
            if self.feed._running:
                self.feed.subscribe_tickers(new_tickers)
            else:
                self.feed.start(new_tickers)
            print(f"[discover] +{len(new_tickers)} tickers subscribed "
                  f"(total {len(self._subscribed)})")
        n_stats = self.store.write_stats(stats_rows)
        return len(new_tickers), n_stats

    # ---- settlements ----

    def _check_settlements(self, ts):
        """Detect markets that flipped to settled and record result +
        observed high.  We scan settled markets per series and read
        `result`; the observed high is parsed from the city's recorded
        forecast day if the rules expose it, else left NULL (IEM backfill
        offline).  Best-effort, never raises."""
        n = 0
        for city, cfg in CITIES.items():
            try:
                settled = self.api.get_markets(series_ticker=cfg["series"],
                                                status="settled")
            except Exception as e:
                print(f"[settle] {cfg['series']} fetch failed: {e} — skip")
                continue
            for m in settled or []:
                ticker = m.get("ticker")
                if not ticker or ticker in self._settled:
                    continue
                result = m.get("result")
                if not result:
                    continue
                obs = _f(m.get("expiration_value"))
                changed = self.store.write_settlement(
                    (ts, ticker, city, cfg["series"],
                     wl.ticker_day(ticker), result, obs))
                self._settled.add(ticker)
                if changed:
                    n += 1
        return n

    # ---- main cycle ----

    def _cycle(self):
        """Housekeeping pass (not book data — that lands event-driven on
        the WS thread).  Runs every CYCLE_INTERVAL_S; each sub-task gates
        itself on its own interval.  Book updates are written separately
        by _on_book_update."""
        ts = datetime.now(tz=timezone.utc).isoformat()
        now = time.monotonic()

        n_fc = 0
        try:
            n_fc = self.store.write_forecasts(self._collect_forecasts(ts))
        except Exception as e:
            print(f"[cycle] forecast collection failed: {e}")

        n_new = n_stats = 0
        want_stats = (now - self._last_stats) >= self.STATS_INTERVAL_S
        if (now - self._last_discovery) >= self.DISCOVERY_INTERVAL_S or want_stats:
            try:
                n_new, n_stats = self._discover(ts, want_stats)
                self._last_discovery = now
                if want_stats:
                    self._last_stats = now
            except Exception as e:
                print(f"[cycle] discovery/stats failed: {e}")

        n_settle = 0
        if now - self._last_settle_check >= self.SETTLE_CHECK_S:
            try:
                n_settle = self._check_settlements(ts)
                self._last_settle_check = now
            except Exception as e:
                print(f"[cycle] settlement check failed: {e}")

        print(f"[cycle] {ts} forecasts={n_fc} new_tickers={n_new} "
              f"stats={n_stats} settlements={n_settle} "
              f"subscribed={len(self._subscribed)}")
        return n_new, n_fc, n_settle

    def run_once(self):
        """One housekeeping cycle (forecast + discovery + stats + settle),
        then stop the feed.  Forces stats so the smoke test exercises the
        get_markets path.  Book data needs the WS to run for a while, so
        --once is not a book test — use a brief full run for that."""
        self._last_stats = -1e9      # force a stats poll
        self._last_discovery = -1e9
        try:
            return self._cycle()
        finally:
            self.feed.stop()

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: self._stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop_event.set())
        print(f"[Recorder] weather recorder up — {len(CITIES)} cities, "
              f"cycle={self.CYCLE_INTERVAL_S}s, "
              f"rotate={'on' if self.do_rotate else 'off'}")
        if self.do_rotate:
            self._rotation_thread = threading.Thread(
                target=self._rotation_loop, daemon=True,
                name="weather-rotation")
            self._rotation_thread.start()

        # Initial discovery: caches metadata, starts the WS feed with the
        # current open tickers, and seeds the first stats row.  _discover
        # calls feed.start() on the first batch (feed not yet running).
        try:
            self._discover(datetime.now(tz=timezone.utc).isoformat(),
                           want_stats=True)
            self._last_discovery = self._last_stats = time.monotonic()
        except Exception as e:
            print(f"[Recorder] initial discovery failed: {e}")

        print("[Recorder] Running.  Ctrl-C to stop.")
        while not self._stop_event.is_set():
            try:
                self._cycle()
            except Exception as e:
                print(f"[cycle] unhandled: {e}")
            self._stop_event.wait(self.CYCLE_INTERVAL_S)

        self.feed.stop()
        self.store.close_all()
        print("[Recorder] stopped.")

    # ---- daily-rotation to S3 (mirrors Aston/tools/recorder.py) ----

    def _rotation_loop(self):
        self._stop_event.wait(60)
        while not self._stop_event.is_set():
            try:
                self._do_rotation_pass()
            except Exception as e:
                print(f"[rotate] pass failed: {e}")
            self._stop_event.wait(self.ROTATION_INTERVAL_S)

    def _do_rotation_pass(self):
        now = time.time()
        today_name = db_path_for(datetime.now(tz=timezone.utc),
                                 self.store.base_dir).name

        eligible = []
        for f in self.store.base_dir.glob("WEATHER-*.db"):
            if f.name == today_name:
                continue
            if (now - f.stat().st_mtime) < self.ROTATION_AGE_S:
                continue
            eligible.append(f)
        if not eligible:
            return

        print(f"[rotate] {len(eligible)} stale DB(s) eligible")
        for db in sorted(eligible):
            # Release any idle handle so the file can be deleted.
            self.store.close_conn(db)

            siblings = [db]
            for ext in (".db-wal", ".db-shm"):
                sib = db.with_suffix(ext)
                if sib.exists():
                    siblings.append(sib)

            ok = True
            for f in siblings:
                remote = f"{self.S3_ROTATION_BUCKET}/{f.name}"
                print(f"[rotate] up {f.name}")
                try:
                    cp = subprocess.run(
                        ["aws", "s3", "cp", str(f), remote,
                         "--only-show-errors"],
                        capture_output=True, text=True, timeout=1800)
                except Exception as e:
                    print(f"[rotate] cp raised {f.name}: {e}")
                    ok = False
                    break
                if cp.returncode != 0:
                    print(f"[rotate] FAIL {f.name}: "
                          f"{cp.stderr.strip() or cp.stdout.strip()}")
                    ok = False
                    break
                try:
                    verify = subprocess.run(
                        ["aws", "s3", "ls", remote],
                        capture_output=True, text=True, timeout=30)
                except Exception as e:
                    print(f"[rotate] verify raised {f.name}: {e}")
                    ok = False
                    break
                if verify.returncode != 0 or not verify.stdout.strip():
                    print(f"[rotate] FAIL verify {f.name}")
                    ok = False
                    break
            if ok:
                for f in siblings:
                    try:
                        f.unlink()
                        print(f"[rotate] rm {f.name}")
                    except Exception as e:
                        print(f"[rotate] could not rm {f.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="run one cycle then exit (smoke test)")
    ap.add_argument("--no-rotate", action="store_true",
                    help="skip the S3 rotation thread")
    args = ap.parse_args()

    rec = WeatherRecorder(do_rotate=not args.no_rotate)
    if args.once:
        rec.run_once()
    else:
        rec.run()


if __name__ == "__main__":
    main()
