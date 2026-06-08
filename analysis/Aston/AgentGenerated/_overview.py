"""Top-level dashboard of every fill: daily P&L, side breakdown, edge
distribution, hold-to-close vs short-horizon markouts, distribution
of edges and counts.  This is the 'is the strategy working at all'
sanity check."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
from utility import bootstrap_ci

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
fills = pd.read_pickle(ROOT / "master_fills.pkl")
mo = pd.read_pickle(ROOT / "markouts.pkl")
f = fills.merge(mo[["fid", "markout_1s", "markout_5s", "markout_30s",
                    "markout_60s", "markout_120s"]],
                left_index=True, right_on="fid", how="left")

# strip incomplete partial day
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]

print("=" * 78)
print("AGGREGATE OVERVIEW — Aston ETH 15M, 2026-05-15 to 2026-05-25")
print("=" * 78)
print(f"\nTotal fills:           {len(f):,}")
print(f"Total contracts:       {f['count'].sum():,.0f}")
print(f"Buys / Sells:          {(f['action']=='buy').sum():,} / {(f['action']=='sell').sum():,}")
print(f"Tickers traded:        {f['ticker'].nunique()}")

f_ok = f.dropna(subset=["pnl_settle_c"])
print(f"\n--- HELD-TO-CLOSE P&L (per fill, in cents) ---")
print(f"Total realized:        ${f_ok['pnl_settle_c'].sum()/100:,.2f}")
print(f"Mean per fill:         {f_ok['pnl_settle_c'].mean():+.2f}c")
print(f"Mean per contract:     {(f_ok['pnl_settle_c'].sum()/f_ok['count'].sum()):+.2f}c")
lo, hi = bootstrap_ci(f_ok['pnl_settle_c'].values, B=2000)
print(f"  bootstrap CI/fill:   [{lo:+.2f}, {hi:+.2f}]c")
print(f"Win rate:              {(f_ok['pnl_settle_c']>0).mean():.1%}")

print(f"\n--- DAILY P&L (held-to-close, $) ---")
daily = f_ok.groupby("date").agg(
    fills=("ts", "count"),
    contracts=("count", "sum"),
    pnl_settle_d=("pnl_settle_c", lambda x: x.sum() / 100),
    mean_edge_c=("edge_c", "mean"),
).reset_index()
daily["pnl_settle_d"] = daily["pnl_settle_d"].round(2)
daily["mean_edge_c"] = daily["mean_edge_c"].round(2)
print(daily.to_string(index=False))

print(f"\nDaily mean: ${daily['pnl_settle_d'].mean():+.2f}  median: ${daily['pnl_settle_d'].median():+.2f}")
print(f"Daily std:  ${daily['pnl_settle_d'].std():.2f}")
n = len(daily)
sharpe_daily = daily['pnl_settle_d'].mean() / max(daily['pnl_settle_d'].std(), 1e-9)
sharpe_ann = sharpe_daily * np.sqrt(365.25)  # crypto trades 24/7
print(f"Sharpe (naive, daily): {sharpe_daily:.2f}   annualized: {sharpe_ann:.2f}")
print(f"  (n={n} days, treat with skepticism)")

print(f"\n--- BY SIDE ---")
for action in ["buy", "sell"]:
    s = f_ok[f_ok["action"] == action]
    print(f"  {action}:  n={len(s):>5}  pnl=${s['pnl_settle_c'].sum()/100:+8.2f}  "
          f"mean={s['pnl_settle_c'].mean():+.2f}c  edge={s['edge_c'].mean():+.2f}c  "
          f"per-contract={s['pnl_settle_c'].sum()/s['count'].sum():+.2f}c")

print(f"\n--- MARKOUTS (mean cents, mid-mark) ---")
for h in [1, 5, 30, 60, 120]:
    c = f[f"markout_{h}s"].dropna()
    if len(c) == 0: continue
    lo, hi = bootstrap_ci(c.values, B=2000)
    print(f"  T+{h:>4}s:  mean={c.mean():+.2f}c  [{lo:+.2f},{hi:+.2f}]  n={len(c):,}")
print("  (negative at 1-5s = adverse selection — classic MM signature)")

print(f"\n--- MARKOUTS BY SIDE ---")
for action in ["buy", "sell"]:
    print(f"  {action}:")
    for h in [1, 5, 30, 60, 120]:
        c = f[(f["action"] == action)][f"markout_{h}s"].dropna()
        if len(c) == 0: continue
        print(f"    T+{h:>4}s:  mean={c.mean():+.2f}c  n={len(c):,}")

print(f"\n--- EDGE DISTRIBUTION (cents at fill) ---")
print(f"  Mean edge:         {f['edge_c'].mean():+.2f}c")
print(f"  Median edge:       {f['edge_c'].median():+.2f}c")
print(f"  Std:               {f['edge_c'].std():.2f}c")
print(f"  Pct negative edge: {(f['edge_c'] < 0).mean():.1%}  "
      f"(filled WORSE than theo — adverse, or theo moved post-fill)")
for q in [5, 25, 50, 75, 95]:
    print(f"  Q{q}:  {f['edge_c'].quantile(q/100):+.2f}c")

print(f"\n--- PNL VS EDGE BUCKET ---")
f['edge_bucket'] = pd.cut(f['edge_c'], bins=[-100, -2, 0, 2, 4, 6, 8, 10, 12, 100],
                            include_lowest=True)
g = f.groupby('edge_bucket', observed=True).agg(
    n=('pnl_settle_c', 'count'),
    pnl_total=('pnl_settle_c', lambda x: x.sum()/100),
    pnl_mean_c=('pnl_settle_c', 'mean'),
    mo_60s=('markout_60s', 'mean'),
).reset_index()
print(g.to_string(index=False))
