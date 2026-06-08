"""Final summary metrics to nail down the headline numbers:
- Baseline P&L: realized, fee-adjusted, per day, weekly cumulative
- Sharpe (with autocorrelation adjustment if possible)
- Per-side: how much of P&L is buy vs sell
- Per-day standard error
- Counterfactual best simple changes
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
from utility import bootstrap_ci

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]
f = f.dropna(subset=["pnl_settle_c"])
n_days = f["date"].nunique()

# ---- Baseline ----
print("=" * 78)
print("BASELINE METRICS (held-to-close, includes Kalshi fees)")
print("=" * 78)
daily = f.groupby("date")["pnl_settle_c"].sum() / 100
print(f"  Days:               {n_days}")
print(f"  Total realized P&L: ${daily.sum():+.2f}")
print(f"  Daily mean:         ${daily.mean():+.2f}")
print(f"  Daily std:          ${daily.std():.2f}")
print(f"  Daily SE of mean:   ${daily.std()/np.sqrt(n_days):.2f}")
print(f"  95% CI on per-day:  [${daily.mean() - 1.96*daily.std()/np.sqrt(n_days):+.2f}, ${daily.mean() + 1.96*daily.std()/np.sqrt(n_days):+.2f}]")
print(f"  Daily Sharpe:       {daily.mean()/daily.std():.2f}")
print(f"  Annualized Sharpe:  {daily.mean()/daily.std() * np.sqrt(365.25):.2f}")
print(f"  N positive days:    {(daily>0).sum()} / {n_days}")

# Ticker-clustered bootstrap
print("\n--- Bootstrap on PER-TICKER mean P&L ---")
ticker_pnl = f.groupby("ticker")["pnl_settle_c"].sum().values
print(f"  N tickers:          {len(ticker_pnl):,}")
print(f"  Per-ticker mean:    {ticker_pnl.mean():+.2f}c")
print(f"  Per-ticker std:     {ticker_pnl.std():.2f}c")
lo, hi = bootstrap_ci(ticker_pnl, B=5000)
print(f"  95% CI per-ticker:  [{lo:+.2f}, {hi:+.2f}]c")
print(f"  Annualized at 95/day: ${ticker_pnl.mean() / 100 * 95 * 365.25:.0f}/year (if pattern holds)")

# ---- Side decomposition with CIs ----
print("\n" + "=" * 78)
print("SIDE DECOMPOSITION")
print("=" * 78)
for action in ["buy", "sell"]:
    s = f[f["action"] == action]
    daily_s = s.groupby("date")["pnl_settle_c"].sum() / 100
    ts = s.groupby("ticker")["pnl_settle_c"].sum().values
    print(f"\n  {action}:")
    print(f"    n fills:    {len(s):,}  ({100*len(s)/len(f):.0f}%)")
    print(f"    Total P&L:  ${daily_s.sum():+.2f}")
    print(f"    Per day:    ${daily_s.mean():+.2f}  ± ${daily_s.std()/np.sqrt(n_days):.2f}")
    lo, hi = bootstrap_ci(ts, B=5000)
    print(f"    Per ticker: {ts.mean():+.2f}c [CI {lo:+.2f}, {hi:+.2f}]")

# ---- Quick scenario summary ----
print("\n" + "=" * 78)
print("SCENARIO IMPROVEMENT TABLE")
print("=" * 78)
print(f"\nBaseline:                           ${daily.sum()/n_days:+.2f}/day")
# Drop buys at theo<0.5
mask = (f["action"] == "buy") & (f["theo"] < 0.5)
modified = f[~mask]
print(f"Drop buy fills when theo<0.5:       ${modified['pnl_settle_c'].sum()/100/n_days:+.2f}/day  "
      f"(+${(modified['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:.2f})")
# Drop buys at z<0
mask = (f["action"] == "buy") & (f["z"] < 0)
modified = f[~mask]
print(f"Drop buy fills when z<0:            ${modified['pnl_settle_c'].sum()/100/n_days:+.2f}/day  "
      f"(+${(modified['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:.2f})")
# Drop all neg-edge fills
mask = f["edge_c"] < 0
modified = f[~mask]
print(f"Drop fills with edge<0 (stale):     ${modified['pnl_settle_c'].sum()/100/n_days:+.2f}/day  "
      f"(+${(modified['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:.2f})")
# Push auto-off to 180s
mask = f["secs_to_close"] < 180
modified = f[~mask]
print(f"Auto-off at T-3m (was T-90s):       ${modified['pnl_settle_c'].sum()/100/n_days:+.2f}/day  "
      f"(+${(modified['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:.2f})")
# Combine: drop buy<theo0.5 + auto-off 180s
mask = ((f["action"] == "buy") & (f["theo"] < 0.5)) | (f["secs_to_close"] < 180)
modified = f[~mask]
print(f"COMBINED (theo<0.5 gate + 180s off):${modified['pnl_settle_c'].sum()/100/n_days:+.2f}/day  "
      f"(+${(modified['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:.2f})")
# All neg-edge dropped
mask = (f["edge_c"] < 0) | ((f["action"] == "buy") & (f["theo"] < 0.5))
modified = f[~mask]
print(f"COMBINED (theo<0.5 + edge>=0):      ${modified['pnl_settle_c'].sum()/100/n_days:+.2f}/day  "
      f"(+${(modified['pnl_settle_c'].sum()-f['pnl_settle_c'].sum())/100/n_days:.2f})")
