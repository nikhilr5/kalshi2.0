"""Where am I making money and where am I losing it?

Run me with:
    python3 _pnl_breakdown.py [--days N] [--rebuild]

Default: latest 7 days from the slim fills cache.
Add --rebuild to refresh the cache from per-day DBs first.

Outputs:
- Daily P&L roll-up (cumulative + per-day)
- Side x theo bin (where the edge concentrates)
- Side x time-to-close
- Side x quote_age (stale quote bleed)
- Side x adverse 60s spot momentum
- Hour-of-day
- Top 5 best / worst markets

No CIs. This is for daily monitoring, not statistical claims.
"""
import argparse, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0")
CACHE = ROOT / "analysis/Aston/AgentGenerated/_cache/fills_full_5may15_to_5may30.pkl"
BUILDER = ROOT / "analysis/Aston/AgentGenerated/_full_window_gate.py"

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=7, help="last N days to analyze")
ap.add_argument("--rebuild", action="store_true", help="rerun the cache builder first")
ap.add_argument("--all", action="store_true", help="analyze the entire cached window")
args = ap.parse_args()

if args.rebuild or not CACHE.exists():
    print(f"[rebuild] running {BUILDER.name} ...")
    subprocess.run([sys.executable, str(BUILDER)], check=True)

f = pd.read_pickle(CACHE)
f = f.dropna(subset=["pnl_settle_c", "theo"]).copy()
all_dates = sorted(f["date"].unique())

if args.all:
    pass
else:
    keep_dates = set(all_dates[-args.days:])
    f = f[f["date"].isin(keep_dates)]

dates = sorted(f["date"].unique())
N = len(f)
N_DAYS = len(dates)
TOTAL = f["pnl_settle_c"].sum() / 100

print("=" * 90)
print(f"P&L breakdown — {N:,} fills over {N_DAYS} days ({dates[0]} -> {dates[-1]})")
print(f"Headline: ${TOTAL:+.2f} total = ${TOTAL/N_DAYS:+.2f}/day  "
      f"per-fill {f['pnl_settle_c'].mean():+.2f}c  hit {(f['pnl_settle_c']>0).mean():.3f}")
print("=" * 90)


def pivot(df, by_cols, sort_col="total_d", ascending=False):
    g = (df.groupby(by_cols, observed=True)
           .agg(n=("pnl_settle_c", "size"),
                c_per_fill=("pnl_settle_c", "mean"),
                total_d=("pnl_settle_c", lambda x: x.sum() / 100),
                hit=("pnl_settle_c", lambda x: (x > 0).mean()))
           .reset_index())
    return g.sort_values(sort_col, ascending=ascending)


print("\n--- Daily P&L ---")
daily = (f.groupby("date")
           .agg(n=("pnl_settle_c", "size"),
                buys=("action", lambda x: (x == "buy").sum()),
                sells=("action", lambda x: (x == "sell").sum()),
                total_d=("pnl_settle_c", lambda x: x.sum() / 100))
           .reset_index())
daily["cum_d"] = daily["total_d"].cumsum()
print(daily.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


print("\n--- Side baseline ---")
print(pivot(f, ["action"]).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


print("\n--- Side x theo bin (where does the edge come from?) ---")
f["theo_bin"] = pd.cut(f["theo"], bins=np.arange(0, 1.01, 0.1),
                       labels=[f"{int(x*100):02d}-{int(x*100)+10:02d}"
                               for x in np.arange(0, 1, 0.1)],
                       include_lowest=True)
print(pivot(f, ["action", "theo_bin"]).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


print("\n--- Side x time-to-close ---")
f["ttc_bin"] = pd.cut(f["mins_to_close"],
                      bins=[-0.1, 1, 2, 5, 10, 16],
                      labels=["<1m", "1-2m", "2-5m", "5-10m", "10-15m"])
print(pivot(f, ["action", "ttc_bin"]).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if f["qage_s"].notna().any():
    print("\n--- Quote age (stale-quote bleed check) ---")
    f["qa_bin"] = pd.cut(f["qage_s"],
                         bins=[-0.1, 1, 5, 15, 30, 60, 300, 1e9],
                         labels=["<1s", "1-5s", "5-15s", "15-30s",
                                 "30-60s", "1-5m", ">5m"])
    print(pivot(f, ["action", "qa_bin"]).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if f["adv60_bp"].notna().any():
    print("\n--- Side x adverse pre-fill 60s spot momentum ---")
    f["adv_bin"] = pd.cut(f["adv60_bp"],
                          bins=[-1e9, -5, -2, 2, 5, 10, 1e9],
                          labels=["<-5bp", "-5..-2", "-2..2", "2..5",
                                  "5..10", ">10bp"])
    print(pivot(f, ["action", "adv_bin"]).to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


print("\n--- Hour-of-day (CT) ---")
f["hour_ct"] = f["ts"].dt.tz_convert("America/Chicago").dt.hour
print(pivot(f, ["hour_ct"], sort_col="hour_ct", ascending=True).to_string(
    index=False, float_format=lambda x: f"{x:+.2f}"))


print("\n--- Top 5 best markets ---")
mkt = pivot(f, ["ticker"]).head(5)
print(mkt.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n--- Top 5 worst markets ---")
mkt = pivot(f, ["ticker"], ascending=True).head(5)
print(mkt.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
