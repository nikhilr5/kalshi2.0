"""
SQLite + Parquet storage for market data recording.

Tables:
    fills           — every trade executed on Kalshi
    market_snapshots — periodic snapshot of all tracked markets + theos
    sessions        — recording session metadata
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "marketdata"
DB_PATH = DB_DIR / "recorder.db"
PARQUET_DIR = DB_DIR / "parquet"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RecorderDB:

    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS fills (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            event_ticker    TEXT,
            action          TEXT NOT NULL,
            side            TEXT NOT NULL,
            count           INTEGER NOT NULL,
            price           REAL NOT NULL,
            strike          REAL,
            spot_bid        REAL,
            spot_ask        REAL,
            theo_bid        REAL,
            theo_ask        REAL,
            kalshi_yes_bid  REAL,
            kalshi_yes_ask  REAL,
            deribit_bid_iv  REAL,
            deribit_ask_iv  REAL,
            client_order_id TEXT,
            theo_bid_weekly     REAL,
            theo_ask_weekly     REAL,
            deribit_bid_iv_weekly REAL,
            deribit_ask_iv_weekly REAL
        );
        CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts);
        CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills(ticker);

        CREATE TABLE IF NOT EXISTS market_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            event_ticker    TEXT,
            strike          REAL NOT NULL,
            close_time      TEXT,
            kalshi_yes_bid  REAL,
            kalshi_yes_ask  REAL,
            kalshi_bid_size INTEGER,
            kalshi_ask_size INTEGER,
            spot_bid        REAL,
            spot_ask        REAL,
            spot_mid        REAL,
            theo_bid        REAL,
            theo_ask        REAL,
            deribit_bid_iv  REAL,
            deribit_ask_iv  REAL,
            deribit_index   REAL,
            otm_pct         REAL,
            edge_bid        REAL,
            edge_ask        REAL,
            theo_bid_weekly     REAL,
            theo_ask_weekly     REAL,
            deribit_bid_iv_weekly REAL,
            deribit_ask_iv_weekly REAL
        );
        CREATE INDEX IF NOT EXISTS idx_snap_ts ON market_snapshots(ts);
        CREATE INDEX IF NOT EXISTS idx_snap_ticker ON market_snapshots(ticker);

        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            start_ts        TEXT NOT NULL,
            end_ts          TEXT,
            series_ticker   TEXT NOT NULL,
            event_ticker    TEXT,
            num_markets     INTEGER,
            snapshot_interval_sec REAL,
            otm_filter_pct  REAL
        );
        """)
        self.conn.commit()
        # Migrate: add client_order_id to existing fills tables
        try:
            self.conn.execute("ALTER TABLE fills ADD COLUMN client_order_id TEXT")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migrate: add weekly theo columns
        for table in ("fills", "market_snapshots"):
            for col in ("theo_bid_weekly REAL", "theo_ask_weekly REAL",
                        "deribit_bid_iv_weekly REAL", "deribit_ask_iv_weekly REAL"):
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
                    self.conn.commit()
                except sqlite3.OperationalError:
                    pass
        # Migrate: add fee column to fills
        try:
            self.conn.execute("ALTER TABLE fills ADD COLUMN fee REAL DEFAULT 0")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def insert_fill(self, ticker: str, action: str, side: str,
                    count: float, price: float, strike: float = 0,
                    event_ticker: str = "",
                    spot_bid: float = 0, spot_ask: float = 0,
                    theo_bid: float = 0, theo_ask: float = 0,
                    kalshi_yes_bid: float = 0, kalshi_yes_ask: float = 0,
                    deribit_bid_iv: float = 0, deribit_ask_iv: float = 0,
                    client_order_id: str = "",
                    theo_bid_weekly: float = 0, theo_ask_weekly: float = 0,
                    deribit_bid_iv_weekly: float = 0, deribit_ask_iv_weekly: float = 0,
                    fee: float = 0):
        self.conn.execute(
            """INSERT INTO fills (ts, ticker, event_ticker, action, side,
               count, price, strike, spot_bid, spot_ask, theo_bid, theo_ask,
               kalshi_yes_bid, kalshi_yes_ask, deribit_bid_iv, deribit_ask_iv,
               client_order_id,
               theo_bid_weekly, theo_ask_weekly,
               deribit_bid_iv_weekly, deribit_ask_iv_weekly, fee)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (_now_iso(), ticker, event_ticker, action, side,
             count, price, strike, spot_bid, spot_ask, theo_bid, theo_ask,
             kalshi_yes_bid, kalshi_yes_ask, deribit_bid_iv, deribit_ask_iv,
             client_order_id,
             theo_bid_weekly, theo_ask_weekly,
             deribit_bid_iv_weekly, deribit_ask_iv_weekly, fee),
        )
        self.conn.commit()

    def insert_snapshots(self, rows: list[dict]):
        if not rows:
            return
        ts = _now_iso()
        self.conn.executemany(
            """INSERT INTO market_snapshots (ts, ticker, event_ticker, strike,
               close_time, kalshi_yes_bid, kalshi_yes_ask, kalshi_bid_size,
               kalshi_ask_size, spot_bid, spot_ask, spot_mid, theo_bid, theo_ask,
               deribit_bid_iv, deribit_ask_iv, deribit_index, otm_pct,
               edge_bid, edge_ask,
               theo_bid_weekly, theo_ask_weekly,
               deribit_bid_iv_weekly, deribit_ask_iv_weekly)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(ts, r["ticker"], r.get("event_ticker", ""), r["strike"],
              r.get("close_time", ""),
              r.get("kalshi_yes_bid", 0), r.get("kalshi_yes_ask", 0),
              r.get("kalshi_bid_size", 0), r.get("kalshi_ask_size", 0),
              r.get("spot_bid", 0), r.get("spot_ask", 0), r.get("spot_mid", 0),
              r.get("theo_bid", 0), r.get("theo_ask", 0),
              r.get("deribit_bid_iv", 0), r.get("deribit_ask_iv", 0),
              r.get("deribit_index", 0), r.get("otm_pct", 0),
              r.get("edge_bid", 0), r.get("edge_ask", 0),
              r.get("theo_bid_weekly", 0), r.get("theo_ask_weekly", 0),
              r.get("deribit_bid_iv_weekly", 0), r.get("deribit_ask_iv_weekly", 0))
             for r in rows],
        )
        self.conn.commit()

    def insert_session(self, series_ticker: str, event_ticker: str,
                       num_markets: int, snapshot_interval: float,
                       otm_filter_pct: float) -> int:
        cur = self.conn.execute(
            """INSERT INTO sessions (start_ts, series_ticker, event_ticker,
               num_markets, snapshot_interval_sec, otm_filter_pct)
               VALUES (?,?,?,?,?,?)""",
            (_now_iso(), series_ticker, event_ticker, num_markets,
             snapshot_interval, otm_filter_pct),
        )
        self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int):
        self.conn.execute(
            "UPDATE sessions SET end_ts = ? WHERE id = ?",
            (_now_iso(), session_id),
        )
        self.conn.commit()

    def export_parquet(self, table: str, date_str: str):
        """Export one day of data to Parquet. Requires pandas + pyarrow."""
        import pandas as pd
        next_day = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        df = pd.read_sql(
            f"SELECT * FROM {table} WHERE ts >= ? AND ts < ?",
            self.conn,
            params=[f"{date_str}T00:00:00", f"{next_day}T00:00:00"],
        )
        if df.empty:
            print(f"[DB] No {table} data for {date_str}")
            return
        out = PARQUET_DIR / table / f"{date_str}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[DB] Exported {len(df)} {table} rows to {out}")

    def close(self):
        self.conn.close()
