"""Where does the theo bias come from?
Test: rebuild theo using market-implied sigma instead of HAR sigma, holding
everything else (model structure = N(d2)) constant. If the calibration gap
collapses, σ is the problem. If it persists, the model structure is wrong.
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from utility import bootstrap_ci, implied_sigma, theo_vec, theo_vec_twap, realized_sigma_forward, SECONDS_PER_YEAR, ANN_MIN

from _loader import load
ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
setts = pd.read_pickle(ROOT / "settlements.pkl")
d = load()
theo, book, spot = d["theo"], d["book"], d["spot"]

g = theo.groupby("ticker").agg(
    last_ts=("ts", "max"),
    last_secs=("seconds_to_expiry", "min"),
    strike=("strike", "first"),
).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(setts[["ticker", "outcome", "twap"]], on="ticker", how="inner")

theo = theo.merge(g[["ticker", "close_time", "outcome"]], on="ticker", how="inner")
theo["secs_to_close"] = (theo["close_time"] - theo["ts"]).dt.total_seconds()

OFFSETS = [30, 120, 300, 600]
snaps = []
for off in OFFSETS:
    near = theo[(theo["secs_to_close"] >= off-1) & (theo["secs_to_close"] <= off+1)]
    near = near.sort_values(["ticker", "secs_to_close"]).drop_duplicates("ticker", keep="first").copy()
    near["offset"] = off
    snaps.append(near)
snaps = pd.concat(snaps, ignore_index=True)

# Mid via merge_asof
book_lite = book[["ts", "ticker", "yes_bid", "yes_ask"]].sort_values("ts")
snaps = snaps.sort_values("ts")
snaps = pd.merge_asof(snaps, book_lite, by="ticker", on="ts", direction="backward",
                     tolerance=pd.Timedelta("3s"), suffixes=("", "_b"))
snaps["mid"] = (snaps["yes_bid"] + snaps["yes_ask"]) / 2

# IV from market
snaps["iv_mid"] = implied_sigma(
    snaps["mid"].values, snaps["spot"].values,
    snaps["strike"].values, snaps["seconds_to_expiry"].values)

# Realized σ forward — minute Parkinson over the forward period
# horizon = (snap_ts → close_ts) in minutes
snaps["horizon_min"] = (snaps["close_time"] - snaps["ts"]).dt.total_seconds() / 60.0
# Build a forward σ lookup
print("computing realized forward σ (1m, 2m, 5m, 10m)...")
rv = {}
for h in [1, 2, 5, 10]:
    rv[h] = realized_sigma_forward(spot, horizon_minutes=h)
    rv[h] = rv[h][["minute", f"realized_{h}m"]].rename(columns={f"realized_{h}m": f"rsig_{h}m"})

# Map snap_ts → floor(1min) → look up forward σ
snaps["minute"] = snaps["ts"].dt.floor("1min")
for h in [1, 2, 5, 10]:
    snaps = snaps.merge(rv[h], on="minute", how="left")

# Recompute theo using market IV at fill (with same structure)
T_yrs = snaps["seconds_to_expiry"] / SECONDS_PER_YEAR
snaps["theo_with_iv"] = theo_vec(snaps["spot"], snaps["strike"], snaps["iv_mid"], snaps["seconds_to_expiry"])
# TWAP version
snaps["theo_twap"] = theo_vec_twap(snaps["spot"], snaps["strike"], snaps["sigma"], snaps["seconds_to_expiry"])
# Theo with realized forward σ
snaps["theo_with_realized"] = theo_vec(snaps["spot"], snaps["strike"], snaps["rsig_10m"], snaps["seconds_to_expiry"])

# Brier vs outcome
print("\n" + "=" * 78)
print("BRIER BY PREDICTOR (settled tickers only)")
print("=" * 78)
print(f"{'predictor':>22}  {'B@30s':>9} {'B@2m':>9} {'B@5m':>9} {'B@10m':>9}")
for name, col in [
    ("HAR theo (current)", "theo"),
    ("theo w/ market IV", "theo_with_iv"),
    ("theo w/ TWAP adj",  "theo_twap"),
    ("theo w/ realized σ", "theo_with_realized"),
    ("Market mid", "mid"),
]:
    row = [f"{name:>22}"]
    for off in [30, 120, 300, 600]:
        s = snaps[snaps["offset"] == off].dropna(subset=[col, "outcome"])
        b = ((s[col] - s["outcome"])**2).mean()
        row.append(f"{b:>9.4f}")
    print("  ".join(row))

# Look at the σ bias separately
print("\n" + "=" * 78)
print("σ FORECAST ACCURACY: HAR σ vs realized σ over forward window")
print("=" * 78)
for off in [120, 300, 600]:
    s = snaps[snaps["offset"] == off].dropna(subset=["sigma", "rsig_5m", "iv_mid"])
    if len(s) < 50: continue
    bias_h = (s["sigma"] - s["rsig_5m"]).mean()
    bias_m = (s["iv_mid"] - s["rsig_5m"]).mean()
    corr_h = np.corrcoef(s["sigma"], s["rsig_5m"])[0,1]
    corr_m = np.corrcoef(s["iv_mid"], s["rsig_5m"])[0,1]
    print(f"  T-{off:>4}s  n={len(s):>4}  "
          f"HAR bias={bias_h*100:+.2f}%, corr={corr_h:+.3f}  |  "
          f"Mkt bias={bias_m*100:+.2f}%, corr={corr_m:+.3f}")

# Bias of theo by theo bin
print("\n" + "=" * 78)
print("BIAS COMPARED ACROSS THREE THEO VARIANTS @ T-5m")
print("=" * 78)
s = snaps[snaps["offset"] == 300].dropna(subset=["theo", "theo_with_iv", "theo_with_realized", "outcome"])
s["b"] = pd.cut(s["theo"], bins=np.linspace(0, 1, 11))
print(f"{'theo_bin':>14} {'n':>4} {'actual':>8} {'HAR theo':>10} {'+w/IV':>10} {'+w/real':>10}")
for tb, gg in s.groupby("b", observed=True):
    print(f"  {str(tb):>14} {len(gg):>4} {gg['outcome'].mean():>8.3f} "
          f"{gg['theo'].mean():>10.3f} {gg['theo_with_iv'].mean():>10.3f} "
          f"{gg['theo_with_realized'].mean():>10.3f}")

# Drift check: is there a systematic post-snapshot drift toward/away from strike?
print("\n" + "=" * 78)
print("DRIFT: spot move from snap → close, by snap moneyness")
print("=" * 78)
# Need close-time spot via TWAP
snaps_close = snaps.merge(g[["ticker", "twap"]].rename(columns={"twap": "spot_close"}), on="ticker", how="left")
snaps_close["spot_ret_pct"] = (snaps_close["spot_close"] - snaps_close["spot"]) / snaps_close["spot"] * 100
snaps_close["log_sk"] = np.log(snaps_close["spot"] / snaps_close["strike"])
snaps_close["z"] = np.log(snaps_close["spot"]/snaps_close["strike"]) / (snaps_close["sigma"] * np.sqrt(T_yrs))
snaps_close["z_bin"] = pd.cut(snaps_close["z"], bins=[-100, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 100])
for off in [120, 300, 600]:
    print(f"\n  T-{off}s:")
    g_off = snaps_close[snaps_close["offset"] == off]
    summ = g_off.groupby("z_bin", observed=True).agg(
        n=("spot_ret_pct", "count"),
        mean_ret_pct=("spot_ret_pct", "mean"),
        med_ret_pct=("spot_ret_pct", "median"),
        std_ret_pct=("spot_ret_pct", "std"),
    ).reset_index()
    print(summ.to_string(index=False))
