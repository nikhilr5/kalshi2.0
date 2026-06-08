"""HAR refit test: using the live data we have (May 15-25), refit HAR
coefficients and compare to the current ones. Has the relationship changed?
"""
import sys, pandas as pd, numpy as np, json
from pathlib import Path
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis")))
sys.path.insert(0, str(Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated")))
from _loader import load
from utility import ANN_MIN

d = load()
spot = d["spot"].sort_values("ts").reset_index(drop=True)

# Per-minute high/low for Parkinson
spot["minute"] = spot["ts"].dt.floor("1min")
bars = spot.groupby("minute").agg(
    high=("price", "max"), low=("price", "min"), close=("price", "last")
).reset_index().sort_values("minute").reset_index(drop=True)

# Parkinson per-minute variance
hl_ratio = bars["high"] / bars["low"]
bars["pk_var"] = np.where((bars["high"] > 0) & (bars["low"] > 0) & (bars["high"] > bars["low"]),
                         np.log(hl_ratio)**2 / (4 * np.log(2)), 0.0)

# Trailing realized σ over different horizons (annualized)
def trailing_sigma(var, n):
    return np.sqrt(var.rolling(n, min_periods=n).sum() * (ANN_MIN / n))

bars["sig_15"] = trailing_sigma(bars["pk_var"], 15)
bars["sig_30"] = trailing_sigma(bars["pk_var"], 30)
bars["sig_4h"] = trailing_sigma(bars["pk_var"], 240)
bars["sig_24h"] = trailing_sigma(bars["pk_var"], 1440)
bars["sig_fwd_15"] = trailing_sigma(bars["pk_var"], 15).shift(-15)  # forward 15-min

# Refit: forward σ_15 ~ β_0 + β_15·σ_15 + β_30·σ_30 + β_4h·σ_4h + β_24h·σ_24h
df = bars.dropna(subset=["sig_15", "sig_30", "sig_4h", "sig_24h", "sig_fwd_15"]).copy()
X = df[["sig_15", "sig_30", "sig_4h", "sig_24h"]].values
y = df["sig_fwd_15"].values

# 70/30 train/test, time-ordered
split = int(len(df) * 0.7)
Xt, Xv = X[:split], X[split:]
yt, yv = y[:split], y[split:]

# OLS with intercept
def fit_ols(X, y):
    Xa = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return beta
beta = fit_ols(Xt, yt)
def predict(X, b):
    return b[0] + X @ b[1:]

yp_t = predict(Xt, beta)
yp_v = predict(Xv, beta)
ss_t = 1 - ((yp_t - yt)**2).sum() / ((yt - yt.mean())**2).sum()
ss_v = 1 - ((yp_v - yv)**2).sum() / ((yv - yv.mean())**2).sum()

print("=" * 78)
print("HAR REFIT — on live May 15-25 ETH 1m data")
print("=" * 78)
print(f"  n train: {len(Xt):,}  n test: {len(Xv):,}")
print(f"  Coefficients (new):")
print(f"    β_0  = {beta[0]:+.4f}   (current: +0.0314)")
print(f"    β_15 = {beta[1]:+.4f}   (current: +0.4485)")
print(f"    β_30 = {beta[2]:+.4f}   (current: +0.1293)")
print(f"    β_4h = {beta[3]:+.4f}   (current: +0.1843)")
print(f"    β_24h= {beta[4]:+.4f}   (current: +0.1149)")
print(f"  R² train: {ss_t:.3f}   (current: 0.474)")
print(f"  R² test:  {ss_v:.3f}   (current: 0.566)")

# Compare prediction quality on test set
print("\n--- σ forecast accuracy on test set ---")
err_new = yv - yp_v
print(f"  New coefs: bias={err_new.mean()*100:+.2f}%  MAE={np.abs(err_new).mean()*100:.2f}%  "
      f"RMSE={np.sqrt((err_new**2).mean())*100:.2f}%  corr={np.corrcoef(yv, yp_v)[0,1]:+.3f}")

current = json.loads((Path("/Users/nikhilr5/Desktop/Kalshi2.0/Aston/har_coefficients.json")).read_text())
b_cur = np.array([current["beta0"], current["beta_15"], current["beta_30"],
                  current["beta_4h"], current["beta_24h"]])
yp_v_cur = predict(Xv, b_cur)
err_cur = yv - yp_v_cur
print(f"  Cur coefs: bias={err_cur.mean()*100:+.2f}%  MAE={np.abs(err_cur).mean()*100:.2f}%  "
      f"RMSE={np.sqrt((err_cur**2).mean())*100:.2f}%  corr={np.corrcoef(yv, yp_v_cur)[0,1]:+.3f}")

# Average σ over time periods
print("\n--- σ_15 by week (annualized) ---")
bars["wk"] = bars["minute"].dt.isocalendar().week
wk = bars.groupby("wk").agg(
    n=("sig_15","count"),
    sig_15=("sig_15","mean"),
).reset_index()
print(wk.to_string(index=False))
