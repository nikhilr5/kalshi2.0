"""
Backfill weekly theo columns for existing daily snapshots and fills.

Strategy:
  For each daily snapshot, find the closest-in-time weekly snapshot at
  a similar strike to get the weekly Deribit IV, then recompute theo
  using Black-Scholes with the daily's close_time.

Usage:
    python backfill_weekly_theo.py
"""

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "marketdata" / "recorder.db"
RISK_FREE_RATE = 0.043


def bs_prob_above(S: float, K: float, sigma: float, T: float, r: float) -> float:
    """Black-Scholes P(S > K) = N(d2)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return max(min(0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))), 1.0), 0.0)


def parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp to unix seconds."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def T_years(snapshot_ts: float, close_time_str: str) -> float:
    """Time to expiry in years from snapshot timestamp to close_time."""
    try:
        close_utc = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        return max((close_utc.timestamp() - snapshot_ts) / (365.25 * 24 * 3600), 0.0)
    except Exception:
        return 0.0


def backfill_snapshots(conn):
    """Backfill market_snapshots with weekly theo from weekly event IVs."""
    # Step 1: Build a time-indexed IV lookup from weekly event snapshots
    # Weekly events are KXBTC (not KXBTCD)
    print("[Backfill] Loading weekly event snapshots for IV lookup...")
    weekly_rows = conn.execute("""
        SELECT ts, strike, deribit_bid_iv, deribit_ask_iv
        FROM market_snapshots
        WHERE event_ticker LIKE 'KXBTC-%%'
          AND event_ticker NOT LIKE 'KXBTCD-%%'
          AND deribit_bid_iv > 0
        ORDER BY ts
    """).fetchall()

    if not weekly_rows:
        print("[Backfill] No weekly event IV data found. Cannot backfill.")
        return

    print(f"[Backfill] Loaded {len(weekly_rows)} weekly IV snapshots")

    # Build lookup: {rounded_ts_10s: {strike: (bid_iv, ask_iv)}}
    # Round timestamps to 10s buckets for matching
    weekly_iv = {}
    for ts_str, strike, bid_iv, ask_iv in weekly_rows:
        ts = int(parse_ts(ts_str) / 10) * 10  # round to 10s
        if ts not in weekly_iv:
            weekly_iv[ts] = {}
        weekly_iv[ts][strike] = (bid_iv, ask_iv)

    weekly_times = sorted(weekly_iv.keys())
    print(f"[Backfill] {len(weekly_times)} unique time buckets")

    # Step 2: Load daily snapshots that need backfilling
    print("[Backfill] Loading daily snapshots to backfill...")
    daily_rows = conn.execute("""
        SELECT id, ts, strike, close_time, spot_bid, spot_ask
        FROM market_snapshots
        WHERE event_ticker LIKE 'KXBTCD-%%'
          AND (theo_bid_weekly IS NULL OR theo_bid_weekly = 0)
        ORDER BY ts
    """).fetchall()

    if not daily_rows:
        print("[Backfill] No daily snapshots need backfilling.")
        return

    print(f"[Backfill] {len(daily_rows)} daily snapshots to backfill")

    # Step 3: For each daily snapshot, find closest weekly IV and compute theo
    import bisect
    updates = []
    matched = 0
    for row_id, ts_str, strike, close_time, spot_bid, spot_ask in daily_rows:
        ts = int(parse_ts(ts_str) / 10) * 10
        # Find closest weekly time bucket
        idx = bisect.bisect_left(weekly_times, ts)
        best_ts = None
        best_diff = float("inf")
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(weekly_times):
                diff = abs(weekly_times[candidate_idx] - ts)
                if diff < best_diff:
                    best_diff = diff
                    best_ts = weekly_times[candidate_idx]

        if best_ts is None or best_diff > 30:  # within 30s
            continue

        iv_map = weekly_iv[best_ts]
        # Find closest strike in weekly data
        best_strike = None
        best_sdist = float("inf")
        for wk_strike in iv_map:
            dist = abs(wk_strike - strike)
            if dist < best_sdist:
                best_sdist = dist
                best_strike = wk_strike

        if best_strike is None:
            continue

        bid_iv_w, ask_iv_w = iv_map[best_strike]
        snap_ts = parse_ts(ts_str)
        T = T_years(snap_ts, close_time)

        spot_b = spot_bid if spot_bid > 0 else 0
        spot_a = spot_ask if spot_ask > 0 else 0
        if spot_b <= 0 or spot_a <= 0:
            continue

        theo_bid_w = bs_prob_above(spot_b, strike, bid_iv_w, T, RISK_FREE_RATE) if bid_iv_w > 0 else 0
        theo_ask_w = bs_prob_above(spot_a, strike, ask_iv_w, T, RISK_FREE_RATE) if ask_iv_w > 0 else 0

        updates.append((theo_bid_w, theo_ask_w, bid_iv_w, ask_iv_w, row_id))
        matched += 1

    print(f"[Backfill] Matched {matched}/{len(daily_rows)} daily snapshots with weekly IVs")

    # Step 4: Batch update
    if updates:
        conn.executemany("""
            UPDATE market_snapshots
            SET theo_bid_weekly = ?, theo_ask_weekly = ?,
                deribit_bid_iv_weekly = ?, deribit_ask_iv_weekly = ?
            WHERE id = ?
        """, updates)
        conn.commit()
        print(f"[Backfill] Updated {len(updates)} snapshot rows")


def backfill_fills(conn):
    """Backfill fills with weekly theo from weekly event IVs."""
    print("[Backfill] Loading weekly event snapshots for fill IV lookup...")
    weekly_rows = conn.execute("""
        SELECT ts, strike, deribit_bid_iv, deribit_ask_iv
        FROM market_snapshots
        WHERE event_ticker LIKE 'KXBTC-%%'
          AND event_ticker NOT LIKE 'KXBTCD-%%'
          AND deribit_bid_iv > 0
        ORDER BY ts
    """).fetchall()

    if not weekly_rows:
        print("[Backfill] No weekly event IV data found for fills.")
        return

    # Build time-indexed lookup
    weekly_iv = {}
    for ts_str, strike, bid_iv, ask_iv in weekly_rows:
        ts = int(parse_ts(ts_str) / 10) * 10
        if ts not in weekly_iv:
            weekly_iv[ts] = {}
        weekly_iv[ts][strike] = (bid_iv, ask_iv)

    weekly_times = sorted(weekly_iv.keys())

    # Load daily fills needing backfill
    # Fills don't have close_time, so we need to look it up from snapshots
    print("[Backfill] Loading close_time mapping from snapshots...")
    close_map = {}
    rows = conn.execute("""
        SELECT DISTINCT event_ticker, close_time
        FROM market_snapshots
        WHERE event_ticker LIKE 'KXBTCD-%%' AND close_time IS NOT NULL AND close_time != ''
    """).fetchall()
    for et, ct in rows:
        close_map[et] = ct

    daily_fills = conn.execute("""
        SELECT id, ts, strike, event_ticker, spot_bid, spot_ask
        FROM fills
        WHERE (event_ticker LIKE 'KXBTCD-%%' OR ticker LIKE 'KXBTCD-%%')
          AND (theo_bid_weekly IS NULL OR theo_bid_weekly = 0)
        ORDER BY ts
    """).fetchall()

    if not daily_fills:
        print("[Backfill] No daily fills need backfilling.")
        return

    print(f"[Backfill] {len(daily_fills)} daily fills to backfill")

    import bisect
    updates = []
    for row_id, ts_str, strike, event_ticker, spot_bid, spot_ask in daily_fills:
        close_time = close_map.get(event_ticker, "")
        if not close_time or strike <= 0:
            continue

        ts = int(parse_ts(ts_str) / 10) * 10
        idx = bisect.bisect_left(weekly_times, ts)
        best_ts = None
        best_diff = float("inf")
        for candidate_idx in (idx - 1, idx):
            if 0 <= candidate_idx < len(weekly_times):
                diff = abs(weekly_times[candidate_idx] - ts)
                if diff < best_diff:
                    best_diff = diff
                    best_ts = weekly_times[candidate_idx]

        if best_ts is None or best_diff > 30:
            continue

        iv_map = weekly_iv[best_ts]
        best_strike = min(iv_map.keys(), key=lambda s: abs(s - strike), default=None)
        if best_strike is None:
            continue

        bid_iv_w, ask_iv_w = iv_map[best_strike]
        snap_ts = parse_ts(ts_str)
        T = T_years(snap_ts, close_time)

        spot_b = spot_bid if spot_bid > 0 else 0
        spot_a = spot_ask if spot_ask > 0 else 0
        if spot_b <= 0 or spot_a <= 0:
            continue

        theo_bid_w = bs_prob_above(spot_b, strike, bid_iv_w, T, RISK_FREE_RATE) if bid_iv_w > 0 else 0
        theo_ask_w = bs_prob_above(spot_a, strike, ask_iv_w, T, RISK_FREE_RATE) if ask_iv_w > 0 else 0

        updates.append((theo_bid_w, theo_ask_w, bid_iv_w, ask_iv_w, row_id))

    print(f"[Backfill] Matched {len(updates)}/{len(daily_fills)} daily fills with weekly IVs")

    if updates:
        conn.executemany("""
            UPDATE fills
            SET theo_bid_weekly = ?, theo_ask_weekly = ?,
                deribit_bid_iv_weekly = ?, deribit_ask_iv_weekly = ?
            WHERE id = ?
        """, updates)
        conn.commit()
        print(f"[Backfill] Updated {len(updates)} fill rows")


def main():
    print(f"[Backfill] Opening {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # Ensure columns exist
    for table in ("fills", "market_snapshots"):
        for col in ("theo_bid_weekly REAL", "theo_ask_weekly REAL",
                    "deribit_bid_iv_weekly REAL", "deribit_ask_iv_weekly REAL"):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
                conn.commit()
            except sqlite3.OperationalError:
                pass

    backfill_snapshots(conn)
    backfill_fills(conn)
    conn.close()
    print("[Backfill] Done!")


if __name__ == "__main__":
    main()
