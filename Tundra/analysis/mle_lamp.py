"""Same-day (lead-0) sigma fit on LAMP forecast error.

Identical heteroskedastic MLE to mle_heteroskedastic.py, but the forecast that
defines the error is LAMP (model LAV) instead of NBS -- the hourly, obs-aware
guidance used as the same-day mean. LAMP only reaches ~25h, so this is lead-0
only. Saves cache/sigma_coeffs_lamp.json (the day-ahead NBS fit is untouched).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel
from util import intraday_peak_frame

HERE = Path(__file__).resolve().parent
KEYS = ["location", "date", "days_out", "predict_hour"]

# ── build same-day frames: LAMP (mean/error) + NBS,GFS (for the spread feature) ──
lamp = intraday_peak_frame(model="LAV", days_out=(0,))[
    KEYS + ["season", "hours_to_peak_exp", "forecast_high", "actual_peak"]]
nbs = intraday_peak_frame(model="NBS", days_out=(0,))[KEYS + ["forecast_high"]]
gfs = intraday_peak_frame(model="GFS", days_out=(0,))[KEYS + ["forecast_high"]]

df = (lamp.merge(nbs, on=KEYS, suffixes=("_lamp", "_nbs"))
          .merge(gfs.rename(columns={"forecast_high": "forecast_high_gfs"}), on=KEYS))
df["model_spread"] = (df["forecast_high_nbs"] - df["forecast_high_gfs"]).abs()
df["error"] = df["forecast_high_lamp"] - df["actual_peak"]           # LAMP forecast error
df = df.dropna(subset=["error", "model_spread", "hours_to_peak_exp"]).copy()

# ── standardize numerics (saving mean/std for the raw back-transform) ──
num = ["hours_to_peak_exp", "model_spread"]
mean = {c: df[c].mean() for c in num}
std = {c: df[c].std() for c in num}
for c in num:
    df[c + "_z"] = (df[c] - mean[c]) / std[c]

X = pd.get_dummies(df[[c + "_z" for c in num] + ["season", "location"]],
                   columns=["season", "location"], drop_first=True).astype(float)
X.insert(0, "const", 1.0)
names = list(X.columns)
Xmat, y = X.to_numpy(), df["error"].to_numpy()
groups = (df["location"] + df["date"]).to_numpy()
good = np.isfinite(Xmat).all(axis=1) & np.isfinite(y)
Xmat, y, groups = Xmat[good], y[good], groups[good]
k = Xmat.shape[1]
print("LAMP same-day fit | rows:", good.sum())


class HetNormal(GenericLikelihoodModel):
    def loglikeobs(self, p):
        return norm.logpdf(self.endog, self.exog @ p[:k], np.exp(self.exog @ p[k:]))
    def loglike(self, p):
        return self.loglikeobs(p).sum()


start = np.zeros(2 * k); start[k] = np.log(y.std())
res = HetNormal(y, Xmat).fit(start_params=start, method="bfgs", maxiter=3000, disp=0,
                             cov_type="cluster", cov_kwds={"groups": groups})

p = res.params
print("\nfeature            σ-coef   ×σ     p(clustered)")
for i, nm in enumerate(names):
    print(f"  {nm:17s} {p[k+i]:+.3f}  {np.exp(p[k+i]):.2f}   {res.pvalues[k+i]:.2e}")

# ── back-transform to RAW units + save ──
sig = dict(zip(names, p[k:]))
raw = {"const": sig["const"]}
for c in num:
    raw[c] = sig[c + "_z"] / std[c]
    raw["const"] -= sig[c + "_z"] * mean[c] / std[c]
for nm in names:
    if nm.startswith(("season_", "location_")):
        raw[nm] = sig[nm]

json.dump({"raw": raw, "features": num, "n": int(good.sum()),
           "error_def": "LAMP(LAV) forecast_high - actual_peak | lead-0 only"},
          open(HERE / "cache" / "sigma_coeffs_lamp.json", "w"), indent=2)
print("\nsaved cache/sigma_coeffs_lamp.json")


def _sig(coef, st, season, h2p, spread):
    return np.exp(coef["const"] + coef["hours_to_peak_exp"] * h2p
                  + coef["model_spread"] * spread
                  + coef.get(f"season_{season}", 0.0) + coef.get(f"location_{st}", 0.0))


nbs_raw = json.load(open(HERE / "cache" / "sigma_coeffs.json"))["raw"]
print("\nσ comparison (NYC JJA, spread 0):")
for h in (10, 6, 2):
    print(f"  hours_to_peak={h:2d}:  NBS σ={_sig(nbs_raw,'KNYC','JJA',h,0):.2f}   "
          f"LAMP σ={_sig(raw,'KNYC','JJA',h,0):.2f}")
