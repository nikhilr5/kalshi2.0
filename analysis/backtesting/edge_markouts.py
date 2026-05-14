"""
Hypothetical-edge markout analysis for a single event.

For each strike + candidate edge, simulate what would have filled if we
quoted at theo±edge, then compute markouts at 30s/5m/15m/30m.

A "fill" is triggered by a real public trade on the contract:
    - yes-taker (someone bought yes at P) hits our hypothetical sell at
      theo+E if theo+E <= P  →  fill at price (theo+E)
    - no-taker (someone sold yes at P) hits our hypothetical buy at
      theo-E if theo-E >= P  →  fill at price (theo-E)

Theo at trade time = midpoint of Kalshi BBO at the nearest preceding
snapshot.  Markout exit = BBO mid at +N seconds.

Usage:
    python edge_markouts.py KXBTCD-26MAY0817
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
EDGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
MARKOUT_INTERVALS = [(30, "30s"), (300, "5m"), (900, "15m"), (1800, "30m")]


def load_snapshots(conn):
    """ts, ticker, mid (UTC ts as datetime, mid = (bid+ask)/2)."""
    df = pd.read_sql(
        "SELECT ts, ticker, kalshi_yes_bid AS bid, kalshi_yes_ask AS ask "
        "FROM snapshots WHERE kalshi_yes_bid > 0 AND kalshi_yes_ask > 0 "
        "ORDER BY ticker, ts",
        conn,
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df


def load_trades(conn):
    """ts, ticker, price, taker_side from events table."""
    df = pd.read_sql(
        "SELECT ts_us, payload FROM events WHERE event_type='trade'",
        conn,
    )
    if df.empty:
        return df
    parsed = df["payload"].apply(json.loads)
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["ts_us"], unit="us", utc=True),
        "ticker": parsed.apply(lambda p: p.get("market_ticker", "")),
        "price": parsed.apply(lambda p: float(p.get("yes_price_dollars", 0) or 0)),
        "taker_side": parsed.apply(lambda p: p.get("taker_side", "")),
        "size": parsed.apply(lambda p: float(p.get("count_fp", p.get("count", 0)) or 0)),
    })
    return out.sort_values(["ticker", "ts"]).reset_index(drop=True)


def analyze_strike(strike: float, ticker: str,
                   snaps: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per-edge fill counts + markouts for one ticker.  Returns rows of
    (edge, n_fills, mean_markout_30s, ..., total_markout_30s, ...)."""
    tk_snaps = snaps[snaps["ticker"] == ticker].sort_values("ts").reset_index(drop=True)
    tk_trades = trades[trades["ticker"] == ticker].sort_values("ts").reset_index(drop=True)
    if tk_snaps.empty or tk_trades.empty:
        return pd.DataFrame()

    # Pre-compute "theo at trade time" via merge_asof (nearest preceding snapshot)
    merged = pd.merge_asof(
        tk_trades, tk_snaps[["ts", "mid", "bid", "ask"]],
        on="ts", direction="backward", suffixes=("", "_snap"),
    )
    merged = merged.dropna(subset=["mid"])

    # For markout lookups: snapshot ts as monotonic numpy array
    snap_ts = tk_snaps["ts"].values.astype("datetime64[ns]")
    snap_mid = tk_snaps["mid"].values

    results = []
    for E in EDGES:
        for direction, mask, target_fn in (
            ("sell",
             merged["taker_side"] == "yes",
             lambda r: r["mid"] + E),
            ("buy",
             merged["taker_side"] == "no",
             lambda r: r["mid"] - E),
        ):
            sub = merged[mask].copy()
            if sub.empty:
                continue
            target = sub.apply(target_fn, axis=1)
            # Sell fill iff trade price >= our sell target (taker willing
            # to pay our high price).  Buy fill iff trade price <= our buy
            # target.  Fill price = the target (we set it).
            if direction == "sell":
                fills = sub[sub["price"] >= target]
                fill_target = fills["mid"] + E
            else:
                fills = sub[sub["price"] <= target]
                fill_target = fills["mid"] - E
            if fills.empty:
                continue

            # Markouts: exit price = mid at fill_ts + interval
            row_acc = {"edge": E, "side": direction, "n_fills": len(fills)}
            for sec, label in MARKOUT_INTERVALS:
                exit_ts = fills["ts"] + pd.Timedelta(seconds=sec)
                idx = np.searchsorted(snap_ts, exit_ts.values.astype("datetime64[ns]"))
                idx = np.clip(idx, 0, len(snap_mid) - 1)
                exit_mid = snap_mid[idx]
                if direction == "sell":
                    # We sold at fill_target; profit if exit < fill
                    markout = (fill_target.values - exit_mid) * 100
                else:
                    markout = (exit_mid - fill_target.values) * 100
                row_acc[f"mean_{label}"] = float(np.mean(markout))
                row_acc[f"total_{label}"] = float(np.sum(markout))
            results.append(row_acc)

    return pd.DataFrame(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python edge_markouts.py <event_ticker>")
        sys.exit(1)
    event_ticker = sys.argv[1]
    db_path = DATA_DIR / f"{event_ticker}.db"
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    print(f"[edge_markouts] Loading {db_path}...")
    conn = sqlite3.connect(str(db_path))
    snaps = load_snapshots(conn)
    trades = load_trades(conn)
    conn.close()
    print(f"  snapshots: {len(snaps):,}, trades: {len(trades):,}")

    # Discover strike->ticker mapping
    strike_ticker = (snaps[["ticker"]].drop_duplicates()
                     .merge(snaps.groupby("ticker")["ts"].count().rename("n"),
                            on="ticker"))
    # Pull strike from snapshots table directly
    strike_lookup = pd.read_sql(
        "SELECT DISTINCT ticker, strike FROM snapshots",
        sqlite3.connect(str(db_path))
    )

    all_results = []
    for _, row in strike_lookup.sort_values("strike").iterrows():
        strike = row["strike"]
        ticker = row["ticker"]
        df = analyze_strike(strike, ticker, snaps, trades)
        if df.empty:
            continue
        df["strike"] = strike
        df["ticker"] = ticker
        all_results.append(df)

    if not all_results:
        print("No data to analyze.")
        return

    res = pd.concat(all_results, ignore_index=True)

    # For each strike, find the edge with the best total markout at each
    # interval (combining buy + sell sides).
    for label_sec, label in MARKOUT_INTERVALS:
        col = f"total_{label}"
        agg = (res.groupby(["strike", "edge"])[col].sum()
                  .reset_index())
        # Best edge per strike
        idx = agg.groupby("strike")[col].idxmax()
        best = agg.loc[idx].sort_values("strike")
        print(f"\n=== Best edge by total markout @ {label} ===")
        print(best.to_string(index=False))

    # Also print n_fills summary
    print("\n=== Total fills per (strike, edge) ===")
    fills_pivot = (res.groupby(["strike", "edge"])["n_fills"].sum()
                       .unstack(fill_value=0))
    print(fills_pivot.to_string())


if __name__ == "__main__":
    main()
