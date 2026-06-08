"""Test: should we vary edge by moneyness (z) since theo is biased there?
Theo at z=-0.5 to 0 over-predicts YES (means our BUY there is bad).
Solution: don't buy at z < 0 (or use much wider edge there)."""
import sys, pandas as pd, numpy as np
from pathlib import Path

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]
f = f.dropna(subset=["pnl_settle_c", "edge_c", "theo", "outcome", "z"])
n_days = f["date"].nunique()

# Bin by z, action
f["z_bucket"] = pd.cut(f["z"], bins=[-100, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 100])

print("=" * 78)
print("WHERE THE EDGE LIVES — by side × moneyness")
print("=" * 78)
print(f"{'z_bucket':>15} {'side':>6} {'n':>5} {'pnl_d':>9} {'per_fill':>9} {'per_contract':>12} {'edge':>7}")
for action in ["buy", "sell"]:
    for zb, gg in f[f["action"]==action].groupby("z_bucket", observed=True):
        c = gg["count"].sum()
        if c == 0: continue
        pc = gg['pnl_settle_c'].sum()/c
        per_fill = gg['pnl_settle_c'].mean()
        print(f"  {str(zb):>15} {action:>6} {len(gg):>5}  {gg['pnl_settle_c'].sum()/100:>+8.2f}  "
              f"{per_fill:>+8.2f}c {pc:>+11.2f}c  {gg['edge_c'].mean():>+6.2f}c")

# Now: directional edge by moneyness — proposed regimes
print("\n" + "=" * 78)
print("PROPOSED RULES: gate by sign of theo - 0.5 (where edge concentrates)")
print("=" * 78)

# Rule A: only sell when theo>0.5, only buy when theo<0.5 (vol-selling near 50)
print("\nRule A: side requires theo on same side of 50 (only sell when theo>0.5)")
keep = ((f["action"] == "buy") & (f["theo"] < 0.5)) | ((f["action"] == "sell") & (f["theo"] > 0.5))
print(f"  Drops: {(~keep).sum():,} fills.  Keeps: {keep.sum():,}.")
kept = f[keep]
dropped = f[~keep]
print(f"  Kept P&L:    ${kept['pnl_settle_c'].sum()/100:+.2f}  per_day=${kept['pnl_settle_c'].sum()/100/n_days:.2f}")
print(f"  Dropped P&L: ${dropped['pnl_settle_c'].sum()/100:+.2f}")
print(f"  → Net improvement: ${-dropped['pnl_settle_c'].sum()/100/n_days:+.2f}/day")

# Rule B: opposite — only sell when theo<0.5, only buy when theo>0.5
print("\nRule B (mirror): side requires theo on OPPOSITE side of 50 (passive MM lean)")
keep = ((f["action"] == "buy") & (f["theo"] > 0.5)) | ((f["action"] == "sell") & (f["theo"] < 0.5))
print(f"  Drops: {(~keep).sum():,} fills.  Keeps: {keep.sum():,}.")
kept = f[keep]
dropped = f[~keep]
print(f"  Kept P&L:    ${kept['pnl_settle_c'].sum()/100:+.2f}  per_day=${kept['pnl_settle_c'].sum()/100/n_days:.2f}")
print(f"  Dropped P&L: ${dropped['pnl_settle_c'].sum()/100:+.2f}")
print(f"  → Net improvement: ${-dropped['pnl_settle_c'].sum()/100/n_days:+.2f}/day")

# Rule C: drop only the buy/theo_no fills (the worst bucket)
print("\nRule C: drop buy/theo<0.5 fills")
mask = (f["action"] == "buy") & (f["theo"] < 0.5)
print(f"  Drops: {mask.sum():,}  pnl=${f[mask]['pnl_settle_c'].sum()/100:+.2f}")
print(f"  Improvement: ${-f[mask]['pnl_settle_c'].sum()/100/n_days:+.2f}/day")

# Rule D: drop buy when -1<z<0 (the worst moneyness bucket)
print("\nRule D: drop buy fills at z in [-1, 0]")
mask = (f["action"] == "buy") & (f["z"] >= -1) & (f["z"] < 0)
print(f"  Drops: {mask.sum():,}  pnl=${f[mask]['pnl_settle_c'].sum()/100:+.2f}")
print(f"  Improvement: ${-f[mask]['pnl_settle_c'].sum()/100/n_days:+.2f}/day")

# Rule E: drop buy at z<0 entirely
print("\nRule E: drop buy fills at z<0")
mask = (f["action"] == "buy") & (f["z"] < 0)
print(f"  Drops: {mask.sum():,}  pnl=${f[mask]['pnl_settle_c'].sum()/100:+.2f}")
print(f"  Improvement: ${-f[mask]['pnl_settle_c'].sum()/100/n_days:+.2f}/day")

# Compute: total P&L with the best simple rule (combination)
print("\n" + "=" * 78)
print("COMBINED: drop buy when (z<0 OR theo<0.5), keep sells as-is")
print("=" * 78)
mask = (f["action"] == "buy") & ((f["z"] < 0) | (f["theo"] < 0.5))
dropped = f[mask]
kept = f[~mask]
print(f"  Drops {mask.sum():,} fills  ({100*mask.mean():.1f}%)")
print(f"  Dropped P&L:  ${dropped['pnl_settle_c'].sum()/100:+.2f}  "
      f"per_day=${dropped['pnl_settle_c'].sum()/100/n_days:+.2f}")
print(f"  Kept P&L:     ${kept['pnl_settle_c'].sum()/100:+.2f}  "
      f"per_day=${kept['pnl_settle_c'].sum()/100/n_days:+.2f}")
print(f"  Net improvement: +${-dropped['pnl_settle_c'].sum()/100/n_days:+.2f}/day")

# A nicer chart: P&L per-fill by (z, action)
print("\n" + "=" * 78)
print("DENSER GRID: P&L per fill by (action, z bucket)")
print("=" * 78)
print("(- = drop / don't quote here)")
pivot = f.groupby(["action", "z_bucket"], observed=True).agg(
    per_fill=("pnl_settle_c", "mean"),
    n=("pnl_settle_c", "count"),
).reset_index()
print(pivot.to_string(index=False))
