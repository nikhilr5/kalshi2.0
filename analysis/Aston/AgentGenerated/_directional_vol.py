"""Alternative strategy: directional vol arb.
At each snapshot, compute HAR theo vs mid. If they differ by > X cents
and we take the trade (against the mid), what's the expected EV?
This is the 'taker on signal' alternative to passive MM.
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load
from utility import SECONDS_PER_YEAR, bootstrap_ci

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
setts = pd.read_pickle(ROOT / "settlements.pkl")

d = load()
theo = d["theo"]; book = d["book"]
g = theo.groupby("ticker").agg(
    last_ts=("ts","max"), last_secs=("seconds_to_expiry","min"),
    strike=("strike","first")).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(setts[["ticker","outcome"]], on="ticker", how="inner")
theo = theo.merge(g[["ticker","close_time","outcome"]], on="ticker", how="inner")
theo["secs_to_close"] = (theo["close_time"] - theo["ts"]).dt.total_seconds()

# Take T-12m, T-8m, T-5m, T-3m snapshots
RESULTS = []
for off_min in [12, 8, 5, 3, 2]:
    off = off_min * 60
    snaps = theo[theo["secs_to_close"].between(off-1, off+1)]
    snaps = snaps.sort_values(["ticker","secs_to_close"]).drop_duplicates("ticker", keep="first").copy()
    bk = book[["ts","ticker","yes_bid","yes_ask"]].sort_values("ts")
    snaps = snaps.sort_values("ts")
    snaps = pd.merge_asof(snaps, bk, by="ticker", on="ts", direction="backward",
                          tolerance=pd.Timedelta("3s"))
    snaps["mid"] = (snaps["yes_bid"] + snaps["yes_ask"]) / 2
    snaps = snaps.dropna(subset=["mid","theo","outcome","yes_bid","yes_ask"])
    snaps["spread"] = (snaps["yes_ask"] - snaps["yes_bid"]) * 100
    T = snaps["seconds_to_expiry"] / SECONDS_PER_YEAR
    snaps["z"] = np.log(snaps["spot"] / snaps["strike"]) / (snaps["sigma"] * np.sqrt(T))

    print(f"\n=== T-{off_min}m: n={len(snaps):,}  median spread={snaps['spread'].median():.1f}c ===")

    # Take a trade if |theo - mid| > thresh, in the direction of theo.
    # Buy at ask, sell at bid. Held-to-close.
    for thresh_c in [1, 2, 3, 5, 8, 12]:
        thresh = thresh_c / 100
        signal = snaps["theo"] - snaps["mid"]
        # Buy when theo > mid + thresh
        buy = snaps[signal > thresh].copy()
        buy["entry"] = buy["yes_ask"]
        buy["pnl_c"] = (buy["outcome"] - buy["entry"]) * 100

        # Sell when theo < mid - thresh
        sell = snaps[signal < -thresh].copy()
        sell["entry"] = sell["yes_bid"]
        sell["pnl_c"] = (sell["entry"] - sell["outcome"]) * 100

        all_trades = pd.concat([buy, sell])
        if len(all_trades) == 0:
            continue
        total = all_trades["pnl_c"].sum() / 100
        per_trade = all_trades["pnl_c"].mean()
        n = len(all_trades)
        RESULTS.append(dict(off_min=off_min, thresh_c=thresh_c, n=n,
                            n_buy=len(buy), n_sell=len(sell),
                            total=total, per_trade=per_trade,
                            buy_mean=buy["pnl_c"].mean() if len(buy)>0 else np.nan,
                            sell_mean=sell["pnl_c"].mean() if len(sell)>0 else np.nan,
                           ))
        print(f"  thresh={thresh_c:>2}c  n={n:>4} (b={len(buy):>3} s={len(sell):>3})  "
              f"total=${total:>+6.2f}  per_trade={per_trade:+.2f}c  "
              f"buy={buy['pnl_c'].mean() if len(buy)>0 else np.nan:+.2f}c  "
              f"sell={sell['pnl_c'].mean() if len(sell)>0 else np.nan:+.2f}c")

print("\n=== SUMMARY ===")
R = pd.DataFrame(RESULTS)
# Best per-trade EV
print("\nTop 10 by per-trade P&L (held to close):")
print(R.sort_values("per_trade", ascending=False).head(10).to_string(index=False))
print("\nTop 10 by total P&L (held to close):")
print(R.sort_values("total", ascending=False).head(10).to_string(index=False))

# Per-day equivalent (11 days)
print("\n--- Per-day rate (top configs) ---")
top = R.sort_values("total", ascending=False).head(5)
for _, row in top.iterrows():
    print(f"  T-{row['off_min']}m thresh={row['thresh_c']}c:  ${row['total']/11:+.2f}/day  "
          f"({row['n']/11:.1f} trades/day)")

print("\nNote: this is held-to-close, no costs, taker side (pays spread).")
print("Median spread = ~2-3c, so the threshold needs to clear that easily.")
