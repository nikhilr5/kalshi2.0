"""Build the master fills frame: every fill enriched with strike, close_time,
TWAP outcome, IV at fill, theo at fill, time-to-close, moneyness, BBO at
fill, plus markouts at multiple horizons.  This is the dataframe every
investigation script keys off."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _loader import load, attach_market_meta, compute_settlements_twap, pnl_held_to_close

sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
from utility import implied_sigma, theo_vec, theo_vec_twap, calculate_markouts, SECONDS_PER_YEAR

OUT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache/master_fills.pkl")
SETTLEMENTS_OUT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache/settlements.pkl")
MARKOUTS_OUT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache/markouts.pkl")


def main():
    t0 = time.time()
    d = load()
    theo, book, spot, fills, events = d["theo"], d["book"], d["spot"], d["fills"], d["events"]
    print(f"loaded {len(fills):,} fills  ({time.time()-t0:.1f}s)")

    # ---- Settlements (TWAP 60s) ----
    setts = compute_settlements_twap(theo, spot, window_s=60)
    print(f"settlements: {len(setts)} / {theo['ticker'].nunique()} tickers")
    setts.to_pickle(SETTLEMENTS_OUT)

    # ---- Attach metadata ----
    fills = fills.sort_values("ts").reset_index(drop=True)
    fills = attach_market_meta(fills, theo)
    fills = fills.merge(setts[["ticker", "outcome", "twap"]], on="ticker", how="left")
    print(f"fills with settlement: {fills['outcome'].notna().sum():,}/{len(fills):,}")

    # ---- Time-to-close, day, hour ----
    fills["secs_to_close"] = (fills["close_time"] - fills["ts"]).dt.total_seconds()
    fills["mins_to_close"] = fills["secs_to_close"] / 60.0
    fills["date"] = fills["ts"].dt.tz_convert("America/Chicago").dt.date
    fills["hour_ct"] = fills["ts"].dt.tz_convert("America/Chicago").dt.hour
    fills["dow"] = fills["ts"].dt.tz_convert("America/Chicago").dt.day_name()

    # ---- Per-fill theo & IV: merge_asof against theo_state ----
    theo_lite = theo[["ts", "ticker", "spot", "sigma", "theo", "seconds_to_expiry"]].sort_values("ts")
    fills = pd.merge_asof(
        fills.sort_values("ts"),
        theo_lite,
        by="ticker", on="ts", direction="backward",
        tolerance=pd.Timedelta("10s"),
        suffixes=("", "_theo"),
    )
    print(f"fills with theo: {fills['theo'].notna().sum():,}")

    # ---- Per-fill BBO from kalshi_book ----
    book_lite = book[["ts", "ticker", "yes_bid", "yes_ask", "bid_size", "ask_size"]].rename(
        columns={"yes_bid": "book_bid", "yes_ask": "book_ask"}).sort_values("ts")
    fills = pd.merge_asof(
        fills.sort_values("ts"),
        book_lite,
        by="ticker", on="ts", direction="backward",
        tolerance=pd.Timedelta("2s"),
    )
    fills["mid_at_fill"] = (fills["book_bid"] + fills["book_ask"]) / 2

    # ---- Implied sigma from market mid at fill ----
    fills["iv_mid"] = implied_sigma(
        fills["mid_at_fill"].values, fills["spot"].values,
        fills["strike"].values, fills["seconds_to_expiry"].values,
    )

    # ---- Held-to-close P&L ----
    payoff = np.where(fills["action"] == "buy",
                      (fills["outcome"] - fills["price"]) * 100,
                      (fills["price"] - fills["outcome"]) * 100)
    fills["pnl_settle_c"] = payoff * fills["count"] - fills["fee"].fillna(0) * 100

    # ---- Edge at fill (vs theo) ----
    fills["edge_c"] = np.where(
        fills["action"] == "buy",
        (fills["theo"] - fills["price"]) * 100,  # buy below theo = +edge
        (fills["price"] - fills["theo"]) * 100,  # sell above theo = +edge
    )

    # ---- Moneyness at fill: z = log(spot/strike) / (sigma * sqrt(T)) ----
    T = fills["seconds_to_expiry"] / SECONDS_PER_YEAR
    fills["z"] = np.log(fills["spot"] / fills["strike"]) / (fills["sigma"] * np.sqrt(T))

    print(f"\nfinal master fills: {len(fills):,}")
    print(f"  with theo:    {fills['theo'].notna().sum():,}")
    print(f"  with mid:     {fills['mid_at_fill'].notna().sum():,}")
    print(f"  with outcome: {fills['outcome'].notna().sum():,}")
    print(f"  with IV:      {fills['iv_mid'].notna().sum():,}")

    fills.to_pickle(OUT)
    print(f"\nwrote {OUT}")

    # ---- Markouts — DIY vectorized via merge_asof per horizon ----
    print("\ncomputing markouts...")
    t1 = time.time()
    book_lookup = book[["ts", "ticker", "yes_bid", "yes_ask"]].sort_values("ts").copy()
    book_lookup["mid"] = (book_lookup["yes_bid"] + book_lookup["yes_ask"]) / 2

    mo = fills[["ts", "ticker", "action", "price", "count", "seconds_to_expiry"]].copy()
    mo["fid"] = np.arange(len(mo))
    for h in [1, 5, 30, 60, 120]:
        target = mo.copy()
        target["t_target"] = target["ts"] + pd.Timedelta(seconds=h)
        # merge_asof needs sorted-by-time
        target = target.sort_values("t_target")
        merged = pd.merge_asof(
            target,
            book_lookup,
            by="ticker", left_on="t_target", right_on="ts",
            direction="backward", suffixes=("", "_b"),
            tolerance=pd.Timedelta("60s"),
        )
        # mask out markouts beyond expiry
        out_mid = np.where(merged["seconds_to_expiry"] >= h, merged["mid"], np.nan)
        # for buy: pnl = (mid_later - fill_price) * 100 ; sell: (fill - mid) * 100
        pnl = np.where(merged["action"] == "buy",
                       (out_mid - merged["price"]) * 100,
                       (merged["price"] - out_mid) * 100)
        # preserve fid ordering
        m = pd.Series(pnl, index=merged["fid"].values)
        mo[f"markout_{h}s"] = m.reindex(mo["fid"].values).values
    print(f"markouts done ({time.time()-t1:.1f}s)")
    mo[["ts", "ticker", "fid"] + [c for c in mo.columns if c.startswith("markout_")]].to_pickle(MARKOUTS_OUT)
    print(f"total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
