"""Time-of-day, day-of-week, vol-regime patterns.
Where is edge concentrated? Where would gating help?"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
from utility import bootstrap_ci

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
mo = pd.read_pickle(ROOT / "markouts.pkl")
f = f.merge(mo[["fid", "markout_60s"]], left_index=True, right_on="fid")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]
f = f.dropna(subset=["pnl_settle_c"])

print("=" * 78)
print("BY HOUR OF DAY (CT) — total + per side")
print("=" * 78)
print(f"{'hour':>4} {'fills':>6} {'pnl_$':>8} {'per_fill':>9} {'buy_pnl':>9} {'sell_pnl':>9}")
for h in range(24):
    sub = f[f["hour_ct"] == h]
    if len(sub) == 0: continue
    p = sub["pnl_settle_c"].sum() / 100
    pf = sub["pnl_settle_c"].mean()
    bp = sub[sub["action"]=="buy"]["pnl_settle_c"].sum() / 100
    sp = sub[sub["action"]=="sell"]["pnl_settle_c"].sum() / 100
    print(f"  {h:>2}  {len(sub):>6}  {p:>+7.2f}  {pf:>+8.2f}c  {bp:>+8.2f}  {sp:>+8.2f}")

print("\n" + "=" * 78)
print("BY DAY OF WEEK")
print("=" * 78)
g = f.groupby("dow", observed=True).agg(
    fills=("ts", "count"),
    pnl_d=("pnl_settle_c", lambda x: x.sum()/100),
    per_fill=("pnl_settle_c", "mean"),
    buy_per_fill=("pnl_settle_c", lambda x: x[f.loc[x.index, "action"]=="buy"].mean()),
    sell_per_fill=("pnl_settle_c", lambda x: x[f.loc[x.index, "action"]=="sell"].mean()),
).reset_index()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
g["dow"] = pd.Categorical(g["dow"], categories=order, ordered=True)
print(g.sort_values("dow").to_string(index=False))

print("\n" + "=" * 78)
print("BY IV REGIME (σ_implied) — all-fills")
print("=" * 78)
f["iv_bucket"] = pd.cut(f["iv_mid"], bins=[0, 0.15, 0.25, 0.35, 0.5, 0.75, 5.0])
g = f.groupby("iv_bucket", observed=True).agg(
    n=("ts","count"),
    pnl_d=("pnl_settle_c", lambda x: x.sum()/100),
    per_fill=("pnl_settle_c", "mean"),
    mo_60s=("markout_60s", "mean"),
    buy_n=("action", lambda x: (x=="buy").sum()),
    sell_n=("action", lambda x: (x=="sell").sum()),
).reset_index()
print(g.to_string(index=False))

print("\n" + "=" * 78)
print("VOL REGIME × SIDE")
print("=" * 78)
for action in ["buy", "sell"]:
    print(f"\n  {action}:")
    g = f[f["action"]==action].groupby("iv_bucket", observed=True).agg(
        n=("ts","count"),
        pnl_d=("pnl_settle_c", lambda x: x.sum()/100),
        per_fill=("pnl_settle_c", "mean"),
        mo_60s=("markout_60s", "mean"),
        edge=("edge_c", "mean"),
    ).reset_index()
    print(g.to_string(index=False))

# σ from the recorder
print("\n" + "=" * 78)
print("HOURLY VOLATILITY (mean HAR-σ at fill, by hour)")
print("=" * 78)
g = f.groupby("hour_ct").agg(
    n=("ts","count"),
    sigma=("sigma","mean"),
    iv=("iv_mid","mean"),
    pnl_per_fill=("pnl_settle_c","mean"),
).reset_index()
print(g.to_string(index=False))

print("\n" + "=" * 78)
print("WEEKLY ROLLING P&L (smooth out daily noise)")
print("=" * 78)
daily = f.groupby("date").agg(pnl=("pnl_settle_c", lambda x: x.sum()/100)).reset_index()
daily["pnl_cum"] = daily["pnl"].cumsum()
daily["pnl_3d"] = daily["pnl"].rolling(3).mean()
daily["pnl_5d"] = daily["pnl"].rolling(5).mean()
print(daily.to_string(index=False))
