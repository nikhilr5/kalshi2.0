"""Calibration of HAR theo vs market mid against actual settlement.
Where does theo over/under-shoot?
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
from utility import bootstrap_ci, brier_score, implied_sigma

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(ROOT / "master_fills.pkl")
setts = pd.read_pickle(ROOT / "settlements.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()]

# ---- We're going to score theo / mid / IV at the FILL TIME against the actual outcome
# But better — score every theo_state snapshot, not just at fills.
# Load theo + book and grid at T-30s, -2m, -5m, -10m

import pickle, sqlite3
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load
d = load()
theo, book, spot = d["theo"], d["book"], d["spot"]

# Build per-ticker close_time
g = theo.groupby("ticker").agg(
    last_ts=("ts", "max"),
    last_secs=("seconds_to_expiry", "min"),
    strike=("strike", "first"),
).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(setts[["ticker", "outcome"]], on="ticker", how="inner")
print(f"settled tickers: {len(g)}")

# Pick snapshot offsets
OFFSETS = [30, 120, 300, 600, 900]  # seconds before close

# For each ticker, find the theo snapshot closest to each offset
theo = theo.merge(g[["ticker", "close_time", "outcome"]], on="ticker", how="inner")
theo["secs_to_close"] = (theo["close_time"] - theo["ts"]).dt.total_seconds()

snaps = []
for off in OFFSETS:
    # closest snapshot with secs_to_close in [off-1, off+1]
    near = theo[(theo["secs_to_close"] >= off-1) & (theo["secs_to_close"] <= off+1)]
    # one per ticker
    near = near.sort_values(["ticker", "secs_to_close"]).drop_duplicates("ticker", keep="first")
    near = near.copy()
    near["offset"] = off
    snaps.append(near)
snaps = pd.concat(snaps, ignore_index=True)
print(f"snapshots: {len(snaps):,}  (across {snaps['ticker'].nunique()} tickers)")

# Attach market mid via merge_asof
book_lite = book[["ts", "ticker", "yes_bid", "yes_ask"]].sort_values("ts").copy()
snaps = snaps.sort_values("ts")
snaps = pd.merge_asof(snaps, book_lite, by="ticker", on="ts",
                     direction="backward", tolerance=pd.Timedelta("3s"),
                     suffixes=("", "_b"))
snaps["mid"] = (snaps["yes_bid"] + snaps["yes_ask"]) / 2
print(f"snapshots with mid: {snaps['mid'].notna().sum():,}")

# Moneyness
from utility import SECONDS_PER_YEAR
T = snaps["seconds_to_expiry"] / SECONDS_PER_YEAR
snaps["z"] = np.log(snaps["spot"] / snaps["strike"]) / (snaps["sigma"] * np.sqrt(T))
snaps["z_abs"] = snaps["z"].abs()

# ---- Brier overall ----
print("\n" + "=" * 78)
print("BRIER BY OFFSET (all snapshots)")
print("=" * 78)
print(f"{'offset':>8} {'n':>7} {'B_theo':>9} {'B_mid':>9} {'gap':>9} {'gap CI':>20}")
for off in OFFSETS:
    s = snaps[snaps["offset"] == off].dropna(subset=["theo", "mid", "outcome"])
    bt = ((s["theo"] - s["outcome"])**2).mean()
    bm = ((s["mid"]  - s["outcome"])**2).mean()
    diff_arr = ((s["theo"]-s["outcome"])**2 - (s["mid"]-s["outcome"])**2).values
    lo, hi = bootstrap_ci(diff_arr, B=3000)
    print(f"  T-{off:>4}s {len(s):>7,}  {bt:>9.4f} {bm:>9.4f} {bt-bm:>+9.4f} [{lo:+.4f},{hi:+.4f}]")

# ---- Brier by moneyness ----
print("\n" + "=" * 78)
print("BRIER BY MONEYNESS (T-30s snapshot only — settlement closest)")
print("=" * 78)
s30 = snaps[snaps["offset"] == 30].dropna(subset=["theo", "mid", "outcome"])
s30["z_b"] = pd.cut(s30["z"], bins=[-100, -2, -1, -0.5, 0, 0.5, 1, 2, 100])
print(f"{'z_bucket':>16} {'n':>6} {'B_theo':>9} {'B_mid':>9} {'gap':>9} {'CI':>20}")
for zb, gg in s30.groupby("z_b", observed=True):
    bt = ((gg["theo"] - gg["outcome"])**2).mean()
    bm = ((gg["mid"]  - gg["outcome"])**2).mean()
    diff = ((gg["theo"]-gg["outcome"])**2 - (gg["mid"]-gg["outcome"])**2).values
    lo, hi = bootstrap_ci(diff, B=2000)
    print(f"  {str(zb):>16} {len(gg):>6} {bt:>9.4f} {bm:>9.4f} {bt-bm:>+9.4f} [{lo:+.4f},{hi:+.4f}]")

# ---- Calibration plot data ----
print("\n" + "=" * 78)
print("THEO CALIBRATION (T-30s): in pred bucket, fraction actually settled YES")
print("=" * 78)
print(f"{'pred_bin':>14} {'n':>6} {'theo_mean':>10} {'mid_mean':>10} {'actual':>10}")
s30["theo_bin"] = pd.cut(s30["theo"], bins=np.linspace(0, 1, 11))
for tb, gg in s30.groupby("theo_bin", observed=True):
    print(f"  {str(tb):>14} {len(gg):>6} {gg['theo'].mean():>10.4f} {gg['mid'].mean():>10.4f} {gg['outcome'].mean():>10.4f}")

# Same for buys: condition on |theo| > X (where we'd be quoting)
print("\n" + "=" * 78)
print("CALIBRATION BY THEO BIN, AT T-3m (typical fill time)")
print("=" * 78)
s180 = snaps[snaps["offset"] == 300].dropna(subset=["theo", "mid", "outcome"]) if 300 in OFFSETS else None
# use 300s
s_use = snaps[snaps["offset"] == 300].dropna(subset=["theo", "mid", "outcome"])
s_use["theo_bin"] = pd.cut(s_use["theo"], bins=np.linspace(0, 1, 11))
print(f"{'pred_bin':>14} {'n':>6} {'theo_mean':>10} {'mid_mean':>10} {'actual':>10} {'theo-actual':>12}")
for tb, gg in s_use.groupby("theo_bin", observed=True):
    a = gg["outcome"].mean()
    print(f"  {str(tb):>14} {len(gg):>6} {gg['theo'].mean():>10.4f} {gg['mid'].mean():>10.4f} {a:>10.4f} {gg['theo'].mean()-a:>+12.4f}")
