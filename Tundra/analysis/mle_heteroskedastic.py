"""Heteroskedastic MLE for the weather forecast-error sigma.

sigma is NOT one number -- it depends on features (time-to-peak, model
disagreement, season, station). We model:  error ~ Normal(mu(x), sigma(x)),
sigma(x) = exp(X @ beta), and find the coefficients by maximum likelihood.

High level:
  1. Score:  given candidate coefficients, compute the total likelihood of the
     observed errors ("how probable are these errors if the coeffs were true").
  2. Search: an optimizer (BFGS) hill-climbs the coefficients to raise that score.
  3. Stop:   the coefficients at the peak are the maximum-likelihood estimates.

We FIT standardized (numerically stable) then BACK-TRANSFORM the coefficients to
raw units, so you can plug raw values straight into the equation.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel
from util import intraday_peak_frame

# ── 1. LOAD + MERGE ───────────────────────────────────────────────────────────
# Two independent forecast models so we can measure their disagreement. Merge key
# MUST include days_out (same date+hour appears for both 0- and 1-day-out rows).
nbs_df = intraday_peak_frame(model="NBS")[
    ["location","season","date","predict_hour","days_out",
     "hours_to_peak_exp","forecast_high","running_max","actual_peak"]]
gfs_df = intraday_peak_frame(model="GFS")[
    ["location","date","predict_hour","days_out","forecast_high"]]
df = nbs_df.merge(gfs_df, on=["location","date","days_out","predict_hour"],
                  suffixes=("_nbs","_gfs"))

# ── 2. BUILD TARGET + FEATURES ────────────────────────────────────────────────
df["model_spread"] = (df["forecast_high_nbs"] - df["forecast_high_gfs"]).abs()  # disagreement
df["error"] = df["forecast_high_nbs"] - df["actual_peak"]                        # what sigma describes

# ── 3. CLEAN + STANDARDIZE ────────────────────────────────────────────────────
# drop rows missing target/features; standardize numerics for a stable optimizer.
# Save mean/std so we can convert the fitted coeffs back to raw units later.
df = df.dropna(subset=["error","model_spread","hours_to_peak_exp"]).copy()
num = ["hours_to_peak_exp", "model_spread"]
mean = {c: df[c].mean() for c in num}
std  = {c: df[c].std()  for c in num}
for c in num:
    df[c+"_z"] = (df[c] - mean[c]) / std[c]

# ── 4. DESIGN MATRIX ──────────────────────────────────────────────────────────
# numeric (standardized) + one-hot categoricals; const column = the intercept.
X = pd.get_dummies(
        df[["hours_to_peak_exp_z","model_spread_z","season","location"]],
        columns=["season","location"], drop_first=True).astype(float)
X.insert(0, "const", 1.0)
names = list(X.columns)

# ── 5. ALIGN + MASK ───────────────────────────────────────────────────────────
# X, y, and the cluster groups must be the SAME rows in the SAME order; mask any
# inf/nan out of all three together (misalignment here = garbage coefficients).
Xmat   = X.to_numpy()
y      = df["error"].to_numpy()
groups = (df["location"] + df["date"]).to_numpy()
good = np.isfinite(Xmat).all(axis=1) & np.isfinite(y)
Xmat, y, groups = Xmat[good], y[good], groups[good]
k = Xmat.shape[1]
print("dropped", (~good).sum(), "bad rows; fitting on", good.sum())

# ── 6. HETEROSKEDASTIC MLE ────────────────────────────────────────────────────
# first k params = mu (bias) coeffs, next k = sigma coeffs (log-scale, exp keeps >0).
class HetNormal(GenericLikelihoodModel):
    def loglikeobs(self, p):                                  # per-row (for cluster SEs)
        mu  = self.exog @ p[:k]
        sig = np.exp(self.exog @ p[k:])
        return norm.logpdf(self.endog, mu, sig)
    def loglike(self, p):                                     # scalar (for the optimizer)
        return self.loglikeobs(p).sum()

start = np.zeros(2*k)
start[k] = np.log(y.std())                                    # sigma intercept guess
res = HetNormal(y, Xmat).fit(start_params=start, method="bfgs", maxiter=3000, disp=0,
                             cov_type="cluster",              # day-clustered SEs (honest p-values)
                             cov_kwds={"groups": groups})

# ── 7. RESULTS (standardized) ─────────────────────────────────────────────────
p, pv = res.params, res.pvalues
print("converged:", res.mle_retvals["converged"], "| n =", len(y))
print("\nfeature            σ-coef   ×σ     p(clustered)")
for i, nm in enumerate(names):
    print(f"  {nm:17s} {p[k+i]:+.3f}  {np.exp(p[k+i]):.2f}   {pv[k+i]:.2e}")

# ── 8. BACK-TRANSFORM to RAW units ────────────────────────────────────────────
# convert the standardized sigma coeffs so the equation takes RAW inputs:
#   raw_slope = z_slope / std ;  raw_const folds in the means ;  dummies unchanged.
sig = dict(zip(names, p[k:]))
raw = {"const": sig["const"]}
for c in num:
    raw[c] = sig[c+"_z"] / std[c]
    raw["const"] -= sig[c+"_z"] * mean[c] / std[c]
for nm in names:
    if nm.startswith(("season_", "location_")):
        raw[nm] = sig[nm]

print("\nRAW-unit σ equation:  σ = exp( const + Σ coef·feature )   [plug raw values in]")
for key in ["const"] + num + [n for n in names if n.startswith(("season_","location_"))]:
    print(f"  {key:18s} {raw[key]:+.4f}")

# persist the raw coefficients so pricing can load them without refitting
import json
from pathlib import Path
out = Path(__file__).resolve().parent / "cache" / "sigma_coeffs.json"
json.dump({"raw": raw, "features": num, "n": int(len(y)),
           "error_def": "forecast_high_nbs - actual_peak (Option 1: floor applied at pricing)"},
          open(out, "w"), indent=2)
print(f"\nsaved coefficients -> {out}")

# ── 9. USE IT (raw inputs in, sigma out) ──────────────────────────────────────
def sigma_raw(location, season, hours_to_peak, model_spread):
    z = (raw["const"]
         + raw["hours_to_peak_exp"] * hours_to_peak
         + raw["model_spread"] * model_spread
         + raw.get(f"season_{season}", 0.0)        # baseline DJF -> 0
         + raw.get(f"location_{location}", 0.0))   # baseline KBOS -> 0
    return float(np.exp(z))

print("\nsanity:")
print("  NYC day-ahead (h2p=30, spread=1.5):", round(sigma_raw("KNYC","JJA",30,1.5), 2))
print("  NYC near peak (h2p=2,  spread=1.5):", round(sigma_raw("KNYC","JJA", 2,1.5), 2))
