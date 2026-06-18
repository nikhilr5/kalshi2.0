"""Favorite-longshot calibration of market-wide KXETH15M trades.

Two analyses:
  1. Moneyness calibration in 0.05 yes_price steps (contract-weighted).
  2. Same, cross-tabbed by time-to-expiration.

Market-wide flow (not capturable fills): this is the opportunity/calibration
shape, not realized maker edge. Run from analysis/Aston/.

    python3 moneyness_calibration.py            # uses cache if present
    python3 moneyness_calibration.py --refresh   # re-hit Kalshi API
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utility as U

START = "2026-05-16"
CACHE = Path(__file__).resolve().parent / f"trades_cache_{START}.csv"
PRICE_BINS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PRICE_LABELS = [f"{PRICE_BINS[i]:.2f}-{PRICE_BINS[i+1]:.2f}"
                for i in range(len(PRICE_BINS) - 1)]
# time-to-expiry buckets (seconds). final 90s = TWAP settlement window.
TIME_EDGES = [0, 90, 300, 600, 900, np.inf]
TIME_LABELS = ["<90s (TWAP)", "90s-5m", "5-10m", "10-15m", ">15m"]


def get_trades(refresh=False):
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE)
    df = U.load_trades(START)
    df.to_csv(CACHE, index=False)
    return df


def _wavg(g, col, w="count"):
    return np.average(g[col], weights=g[w])


def calib_table(df, by_price="price_bin"):
    rows = []
    for lbl, g in df.groupby(by_price, observed=True):
        n = len(g)
        contracts = g["count"].sum()
        avg_p = _wavg(g, "yes_price")
        settle = _wavg(g, "outcome")
        rows.append({
            "bucket": lbl, "trades": n, "contracts": round(contracts, 1),
            "avg_price": round(avg_p, 4), "settle_yes": round(settle, 4),
            "sell_edge": round(avg_p - settle, 4),
        })
    out = pd.DataFrame(rows).set_index("bucket").reindex(PRICE_LABELS).dropna(how="all")
    return out


def crosstab_sell_edge(df):
    piv_edge = pd.DataFrame(index=PRICE_LABELS, columns=TIME_LABELS, dtype=float)
    piv_n = pd.DataFrame(index=PRICE_LABELS, columns=TIME_LABELS, dtype=float)
    for tlbl, tg in df.groupby("time_bin", observed=True):
        ct = calib_table(tg)
        piv_edge[tlbl] = ct["sell_edge"]
        piv_n[tlbl] = ct["contracts"]
    return piv_edge[TIME_LABELS], piv_n[TIME_LABELS]


def main():
    refresh = "--refresh" in sys.argv
    df = get_trades(refresh)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["secs_to_exp"] = U.secs_to_expiry(df["ticker"], df["ts"])

    # drop trades after close (negative) — keep only pre-close flow
    df = df[df["secs_to_exp"] >= 0].copy()

    df["price_bin"] = pd.cut(df["yes_price"], bins=PRICE_BINS,
                             labels=PRICE_LABELS, include_lowest=True)
    df["time_bin"] = pd.cut(df["secs_to_exp"], bins=TIME_EDGES,
                            labels=TIME_LABELS, right=False)

    # ETH up/down day split via per-market outcome base rate is not it;
    # use trade-date ETH direction proxy: settle-YES rate by UTC date.
    df["date"] = df["ts"].dt.tz_convert("UTC").dt.date

    pd.set_option("display.width", 160, "display.max_columns", 20)

    print("=" * 78)
    print(f"MARKET-WIDE KXETH15M CALIBRATION  (since {START})")
    print(f"trades={len(df):,}  contracts={df['count'].sum():,.0f}  "
          f"markets={df['ticker'].nunique():,}  "
          f"overall settle-YES={_wavg(df,'outcome'):.4f}")
    print("=" * 78)

    print("\n--- ANALYSIS 1: moneyness calibration (0.05 steps, contract-weighted) ---")
    print(calib_table(df).to_string())

    # exclude TWAP window for the clean version
    clean = df[df["secs_to_exp"] >= 90]
    print("\n--- ANALYSIS 1b: same, EXCLUDING final 90s (look-ahead removed) ---")
    print(calib_table(clean).to_string())

    print("\n--- ANALYSIS 2: sell-edge cross-tab (price bucket x time-to-expiry) ---")
    edge, ncon = crosstab_sell_edge(df)
    print("\nSELL-EDGE (avg_price - settle_yes), contract-weighted:")
    print(edge.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\nCONTRACTS per cell (sample size):")
    print(ncon.to_string(float_format=lambda x: f"{x:,.0f}"))

    # taker-side split (fade the YES-buying taker)
    print("\n--- ANALYSIS 3: by taker_side (taker bought YES = retail lottery) ---")
    for side in ["yes", "no"]:
        sub = clean[clean["taker_side"] == side]
        if sub.empty:
            continue
        print(f"\ntaker_side={side}  (n={len(sub):,}, contracts={sub['count'].sum():,.0f})  "
              f"[final 90s excluded]")
        print(calib_table(sub).to_string())

    # ETH up/down split: per-UTC-day net direction proxied by that day's
    # contract-weighted settle-YES rate. >0.5 = net-up day, else net-down.
    # Separates favorite-longshot SHAPE from the down-market LEVEL shift.
    print("\n--- ANALYSIS 4: up vs down days (separate shape from down-market) ---")
    day_dir = clean.groupby("date").apply(
        lambda g: _wavg(g, "outcome"), include_groups=False)
    up_days = set(day_dir[day_dir >= 0.5].index)
    clean = clean.copy()
    clean["regime"] = np.where(clean["date"].isin(up_days), "up", "down")
    print(f"up-days={len(up_days)} (settle>=0.5)  down-days={day_dir.size-len(up_days)}  "
          f"[final 90s excluded]")
    for reg in ["up", "down"]:
        sub = clean[clean["regime"] == reg]
        print(f"\nregime={reg}  (days={ (day_dir>=0.5).sum() if reg=='up' else (day_dir<0.5).sum() }, "
              f"contracts={sub['count'].sum():,.0f}, settle-YES={_wavg(sub,'outcome'):.4f})")
        print(calib_table(sub)[["avg_price", "settle_yes", "sell_edge", "contracts"]].to_string())


if __name__ == "__main__":
    main()
