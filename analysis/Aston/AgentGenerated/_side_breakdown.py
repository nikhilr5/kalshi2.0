"""Why do buys lose money while sells make money?
Hypothesis: buy fills cluster where theo is wrong (lows / regime breaks).
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
from utility import bootstrap_ci

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
mo = pd.read_pickle(ROOT / "markouts.pkl")
f = f.merge(mo[["fid", "markout_60s", "markout_120s"]], left_index=True, right_on="fid")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]
f = f.dropna(subset=["pnl_settle_c"])

print("=" * 78)
print("BUY VS SELL DEEP DIVE")
print("=" * 78)

# By time to close
print("\n--- BY MINUTES TO CLOSE ---")
f["bucket_mins"] = pd.cut(f["mins_to_close"],
                          bins=[0, 0.5, 1, 2, 3, 5, 8, 12, 16],
                          include_lowest=True)
for action in ["buy", "sell"]:
    print(f"\n  {action}:")
    g = f[f["action"] == action].groupby("bucket_mins", observed=True).agg(
        n=("pnl_settle_c", "count"),
        pnl_settle=("pnl_settle_c", lambda x: x.sum()/100),
        per_fill=("pnl_settle_c", "mean"),
        edge=("edge_c", "mean"),
        mo_60s=("markout_60s", "mean"),
    ).reset_index()
    print(g.to_string(index=False))

# By moneyness at fill (z-score)
print("\n--- BY MONEYNESS (z = log(S/K) / sigma*sqrt(T)) ---")
f["z_bucket"] = pd.cut(f["z"],
                       bins=[-100, -2, -1, -0.5, 0, 0.5, 1, 2, 100])
for action in ["buy", "sell"]:
    print(f"\n  {action}:")
    g = f[f["action"] == action].groupby("z_bucket", observed=True).agg(
        n=("pnl_settle_c", "count"),
        pnl_total=("pnl_settle_c", lambda x: x.sum()/100),
        per_fill=("pnl_settle_c", "mean"),
        edge=("edge_c", "mean"),
        mo_60s=("markout_60s", "mean"),
        win_rate=("pnl_settle_c", lambda x: (x > 0).mean()),
    ).reset_index()
    print(g.to_string(index=False))

# By IV at fill (regime)
print("\n--- BY IV REGIME (market-implied vol at fill, annualized) ---")
f["iv_bucket"] = pd.cut(f["iv_mid"], bins=[0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 5.0])
for action in ["buy", "sell"]:
    print(f"\n  {action}:")
    g = f[f["action"] == action].groupby("iv_bucket", observed=True).agg(
        n=("pnl_settle_c", "count"),
        pnl_total=("pnl_settle_c", lambda x: x.sum()/100),
        per_fill=("pnl_settle_c", "mean"),
        mo_60s=("markout_60s", "mean"),
    ).reset_index()
    print(g.to_string(index=False))

# Are buys filling in spikes? Look at spot move 1m before fill
print("\n--- Where does buy bleed concentrate? ---")
# Compute decile of edge_c for buys vs sells P&L
for action in ["buy", "sell"]:
    sub = f[f["action"] == action].copy()
    sub["price_bucket"] = pd.cut(sub["price"],
                                  bins=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
    print(f"\n  {action} by fill price:")
    g = sub.groupby("price_bucket", observed=True).agg(
        n=("pnl_settle_c", "count"),
        pnl=("pnl_settle_c", lambda x: x.sum()/100),
        per_fill=("pnl_settle_c", "mean"),
        edge=("edge_c", "mean"),
    ).reset_index()
    print(g.to_string(index=False))

# Bias check — when we buy, was theo higher than market? then we should win on hold-to-close
# IF theo is unbiased
print("\n--- THEO BIAS CHECK ---")
print("If we BUY @ price P, theo says fair=T, T>P. Settlement is 0 or 1.")
print("Mean realization vs theo, by side:")
for action in ["buy", "sell"]:
    sub = f[f["action"] == action].dropna(subset=["theo", "outcome"])
    print(f"  {action}:  mean(theo) = {sub['theo'].mean():.4f}  "
          f"mean(outcome) = {sub['outcome'].mean():.4f}  "
          f"diff = {sub['outcome'].mean() - sub['theo'].mean():+.4f}  "
          f"n = {len(sub):,}")

# Mean P&L when theo>0.5 vs theo<0.5
print("\n--- THEO REGIME ---")
print("Theo > 0.5 (model says 'yes') vs theo < 0.5 ('no')")
for action in ["buy", "sell"]:
    for hi in [True, False]:
        sub = f[(f["action"] == action) & ((f["theo"] > 0.5) == hi)]
        if len(sub) == 0: continue
        side = "theo_yes" if hi else "theo_no"
        print(f"  {action:>4}/{side}:  n={len(sub):>5}  pnl=${sub['pnl_settle_c'].sum()/100:+7.2f}  "
              f"per_fill={sub['pnl_settle_c'].mean():+.2f}  "
              f"win={(sub['pnl_settle_c']>0).mean():.1%}")
