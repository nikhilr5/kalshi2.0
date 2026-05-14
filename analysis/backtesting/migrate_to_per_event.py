"""
One-time migration: split recorder.db into per-event SQLite files.

Each event gets its own file: data/<event_ticker>.db
Snapshots and fills for that event are copied over.
After running, the original recorder.db can be archived/removed.

Usage:
    python migrate_to_per_event.py
"""

import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SOURCE_DB = DATA_DIR / "recorder.db"


def init_event_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
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
            fee REAL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON snapshots(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts)")
    conn.commit()
    return conn


def main():
    if not SOURCE_DB.exists():
        print(f"Source DB not found: {SOURCE_DB}")
        sys.exit(1)

    src = sqlite3.connect(str(SOURCE_DB))
    src.row_factory = sqlite3.Row
    cur = src.cursor()

    # Discover events
    cur.execute("SELECT DISTINCT event_ticker FROM snapshots WHERE event_ticker IS NOT NULL")
    events = [r["event_ticker"] for r in cur.fetchall()]
    print(f"Found {len(events)} events")

    for event in events:
        out_path = DATA_DIR / f"{event}.db"
        if out_path.exists():
            print(f"  {event}: already exists, skipping")
            continue

        print(f"  {event}: migrating...")
        dst = init_event_db(out_path)

        # Copy snapshots
        cur.execute("SELECT * FROM snapshots WHERE event_ticker = ?", (event,))
        rows = cur.fetchall()
        if rows:
            cols = rows[0].keys()
            placeholders = ",".join(["?"] * len(cols))
            col_list = ",".join(cols)
            dst.executemany(
                f"INSERT INTO snapshots ({col_list}) VALUES ({placeholders})",
                [tuple(r) for r in rows]
            )
            print(f"    snapshots: {len(rows):,}")

        # Copy fills
        cur.execute("SELECT * FROM fills WHERE event_ticker = ?", (event,))
        rows = cur.fetchall()
        if rows:
            cols = rows[0].keys()
            placeholders = ",".join(["?"] * len(cols))
            col_list = ",".join(cols)
            dst.executemany(
                f"INSERT INTO fills ({col_list}) VALUES ({placeholders})",
                [tuple(r) for r in rows]
            )
            print(f"    fills: {len(rows):,}")

        dst.commit()
        dst.close()

    src.close()
    print(f"\nDone. Per-event files written to {DATA_DIR}")
    print(f"You can now archive/delete {SOURCE_DB}")


if __name__ == "__main__":
    main()
