"""Hold-to-close vs early exit comparison.
For each fill: what if we exited at T-90s mid, T-30s mid, T-10s mid?
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load

d = load()
ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()].copy()
f = f.dropna(subset=["pnl_settle_c", "close_time", "outcome"])

book = d["book"]
book["ts"] = pd.to_datetime(book["ts"], utc=True)

# Compute exit_mid at various offsets before close (T-90s, T-30s, T-10s)
for offset_s in [90, 30, 10]:
    f[f"exit_ts_{offset_s}"] = f["close_time"] - pd.Timedelta(seconds=offset_s)

# Build a lookup
bk = book[["ts","ticker","yes_bid","yes_ask"]].sort_values("ts").copy()
bk["mid"] = (bk["yes_bid"] + bk["yes_ask"]) / 2

for offset_s in [90, 30, 10]:
    col = f"exit_ts_{offset_s}"
    f = f.sort_values(col)
    merged = pd.merge_asof(
        f[[col, "ticker"]].rename(columns={col: "ts"}),
        bk, by="ticker", on="ts", direction="backward",
        tolerance=pd.Timedelta("30s"))
    f[f"exit_mid_{offset_s}"] = merged["mid"].values

# P&L if we exit at T-X
for offset_s in [90, 30, 10]:
    em = f[f"exit_mid_{offset_s}"]
    # buy: pnl = exit_mid - fill_price; sell: pnl = fill_price - exit_mid
    pnl = np.where(f["action"] == "buy",
                   (em - f["price"]) * 100,
                   (f["price"] - em) * 100)
    f[f"pnl_exit_{offset_s}_c"] = pnl * f["count"]

# Comparison
print("=" * 78)
print("EARLY EXIT vs HOLD-TO-CLOSE — total P&L")
print("=" * 78)
mask = f[["pnl_settle_c"] + [f"pnl_exit_{o}_c" for o in [90, 30, 10]]].notna().all(axis=1)
g = f[mask]
print(f"  n with all values: {len(g):,}")
print(f"  Hold-to-close:     ${g['pnl_settle_c'].sum()/100:+.2f}")
for offset_s in [90, 30, 10]:
    print(f"  Exit at T-{offset_s:>3}s:   ${g[f'pnl_exit_{offset_s}_c'].sum()/100:+.2f}  "
          f"per_fill={g[f'pnl_exit_{offset_s}_c'].mean():+.2f}c  "
          f"(hold pnl mean {g['pnl_settle_c'].mean():+.2f}c)")

# Side breakdown
print("\n--- BY SIDE: hold vs exit @ T-30s ---")
for action in ["buy", "sell"]:
    s = g[g["action"] == action]
    print(f"  {action}:")
    print(f"    Hold-to-close: ${s['pnl_settle_c'].sum()/100:+.2f}  per_fill={s['pnl_settle_c'].mean():+.2f}c")
    print(f"    Exit T-30s:    ${s['pnl_exit_30_c'].sum()/100:+.2f}  per_fill={s['pnl_exit_30_c'].mean():+.2f}c")

# What if we exit any negative-edge buys early at T-3m mid?
print("\n--- 'Dump bad buys at T-3m mid' ---")
# For buy fills where theo<0.5 (the killer), exit at T-180s if it's in the future
# else hold to close
exit_ts = f["close_time"] - pd.Timedelta(seconds=180)
f = f.sort_values("close_time")
merged = pd.merge_asof(
    f[["close_time", "ticker"]].rename(columns={"close_time": "ts"}).assign(ts=exit_ts),
    bk, by="ticker", on="ts", direction="backward", tolerance=pd.Timedelta("30s"))
f["exit_mid_180"] = merged["mid"].values

# rule: if buy and theo<0.5, use exit_mid_180; else hold
# but exit only valid if fill happened BEFORE exit_ts
use_exit = ((f["action"] == "buy") & (f["theo"] < 0.5) & (f["ts"] < (f["close_time"] - pd.Timedelta(seconds=180))))
pnl_buy_exit = np.where(use_exit,
                        (f["exit_mid_180"] - f["price"]) * 100 * f["count"],
                        f["pnl_settle_c"])
total_exit = pnl_buy_exit[~np.isnan(pnl_buy_exit)].sum() / 100
n_days = f["date"].nunique()
print(f"  Total if we early-exit buys/theo<0.5 at T-3m: ${total_exit:+.2f}  per_day=${total_exit/n_days:+.2f}")
print(f"  Total hold-to-close:                          ${f['pnl_settle_c'].sum()/100:+.2f}  per_day=${f['pnl_settle_c'].sum()/100/n_days:+.2f}")
print(f"  Affected: n={use_exit.sum():,}")
