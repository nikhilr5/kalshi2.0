"""Late-market dynamics: where does the edge live in the final 5 minutes?
And: what would happen if we changed the 90s auto-off threshold?"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
mo = pd.read_pickle(ROOT / "markouts.pkl")
f = f.merge(mo[["fid", "markout_60s", "markout_120s"]], left_index=True, right_on="fid")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]
f = f.dropna(subset=["pnl_settle_c"])
n_days = f["date"].nunique()

print("=" * 78)
print("LATE-MARKET FINE GRID — minutes to close, P&L by side")
print("=" * 78)
print(f"\n{'mins_band':>14} {'n':>5} {'pnl_d':>9} {'per_fill':>9} {'buy_pnl_d':>10} {'sell_pnl_d':>11} {'edge':>7}")
for lo, hi in [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 16)]:
    s = f[(f["mins_to_close"] >= lo) & (f["mins_to_close"] < hi)]
    if len(s) == 0: continue
    bp = s[s["action"]=="buy"]["pnl_settle_c"].sum() / 100
    sp = s[s["action"]=="sell"]["pnl_settle_c"].sum() / 100
    print(f"  [{lo:>4.1f},{hi:>4.1f})  {len(s):>5}  ${s['pnl_settle_c'].sum()/100:>+7.2f}  "
          f"{s['pnl_settle_c'].mean():>+7.2f}c  ${bp:>+8.2f}  ${sp:>+9.2f}  {s['edge_c'].mean():>+6.2f}")

print("\n" + "=" * 78)
print("AUTO-OFF THRESHOLDS: P&L if we stopped trading at different cutoffs")
print("=" * 78)
# Currently auto-off at 90s. Test 30s, 60s, 90s, 120s, 180s, 240s
for cutoff in [30, 60, 90, 120, 180, 240, 300]:
    inside = f[f["secs_to_close"] >= cutoff]  # kept
    outside = f[f["secs_to_close"] < cutoff]  # dropped
    kept_pnl = inside["pnl_settle_c"].sum() / 100
    dropped_pnl = outside["pnl_settle_c"].sum() / 100
    print(f"  cutoff={cutoff:>3}s:  drop n={len(outside):>4}  "
          f"dropped_pnl=${dropped_pnl:>+6.2f}  kept_pnl=${kept_pnl:>+7.2f}  "
          f"per_day_kept=${kept_pnl/n_days:>+5.2f}")

print("\n--- AUTO-OFF SENSITIVITY BY SIDE ---")
for cutoff in [60, 90, 120, 180, 240]:
    print(f"\n  cutoff={cutoff}s — dropped fills breakdown:")
    dropped = f[f["secs_to_close"] < cutoff]
    for action in ["buy", "sell"]:
        s = dropped[dropped["action"] == action]
        if len(s) == 0: continue
        print(f"    {action}:  n={len(s):>4}  pnl=${s['pnl_settle_c'].sum()/100:+.2f}  "
              f"per_fill={s['pnl_settle_c'].mean():+.2f}c")

print("\n" + "=" * 78)
print("ARE THE LATE-MARKET FILLS BETTER OR WORSE?")
print("=" * 78)
late = f[f["secs_to_close"] < 180]
early = f[f["secs_to_close"] >= 180]
print(f"  T<3m fills:   n={len(late):>5}  pnl=${late['pnl_settle_c'].sum()/100:+.2f}  per_fill={late['pnl_settle_c'].mean():+.2f}c")
print(f"  T>=3m fills:  n={len(early):>5}  pnl=${early['pnl_settle_c'].sum()/100:+.2f}  per_fill={early['pnl_settle_c'].mean():+.2f}c")

# Just for the data: do late-market fills have higher edge_c?
print(f"\n  T<3m edge:   {late['edge_c'].mean():+.2f}c   median={late['edge_c'].median():+.2f}c")
print(f"  T>=3m edge:  {early['edge_c'].mean():+.2f}c   median={early['edge_c'].median():+.2f}c")
