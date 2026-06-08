"""Platt refit on ETH data through 2026-05-29.
Compares to prior (a=-0.025, b=1.150) fit from 2026-05-28."""
import sys, sqlite3, glob
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit, logit
# sklearn not in venv — roll our own random KFold
def kfold_splits(n, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield tr, te

sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/analysis")
from utility import fetch_settlements_from_api  # noqa

# ---- 1. Load theo from every available KXETH15M DB through MAY29 (skip MAY30 incomplete) ----
LOCAL = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data")
S3C   = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/_s3_cache")
paths = {}
for p in list(LOCAL.glob("KXETH15M-*.db")) + list(S3C.glob("KXETH15M-*.db")):
    if p.name.endswith(".db") and "MAY30" not in p.name:  # exclude incomplete day
        # Keep local copy preference if duplicated
        paths.setdefault(p.name, p)
files = sorted(paths.values(), key=lambda x: x.name)
print(f"[load] {len(files)} ETH DB files (through MAY29)")

theo_l = []
for f in files:
    conn = sqlite3.connect(str(f))
    try:
        df = pd.read_sql(
            "SELECT ts, ticker, theo, seconds_to_expiry FROM theo_state", conn)
        theo_l.append(df)
    except Exception as e:
        print(f"  [warn] {f.name}: {e}")
    finally:
        conn.close()
theo = pd.concat(theo_l, ignore_index=True)
theo["ts"] = pd.to_datetime(theo["ts"], utc=True, format="ISO8601")
print(f"[load] {len(theo):,} theo snapshots, {theo['ticker'].nunique()} tickers")

# ---- 2. Load settlements ----
import json
SETT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/.settlements_cache.json")
sett_map = {k: int(v) for k, v in json.load(open(SETT)).items()}
tickers = theo["ticker"].unique()
missing = [t for t in tickers if t not in sett_map]
print(f"[sett] {len(tickers) - len(missing)}/{len(tickers)} cached, {len(missing)} missing")
# We'll just drop missing — likely MAY30 spillover or markets still open
sett_df = pd.DataFrame([(t, o) for t, o in sett_map.items() if t in set(tickers)],
                       columns=["ticker", "outcome"])

# ---- 3. Build training set: same downsampling as _fit_calibrator.py ----
g = theo.groupby("ticker").agg(
    last_ts=("ts", "max"), last_secs=("seconds_to_expiry", "min")
).reset_index()
g["close_time"] = g["last_ts"] + pd.to_timedelta(g["last_secs"], unit="s")
g = g.merge(sett_df, on="ticker", how="inner")
print(f"[sett] settled tickers w/ theo: {len(g):,}")

t = theo.merge(g[["ticker", "close_time", "outcome"]], on="ticker", how="inner")
t["secs_to_close"] = (t["close_time"] - t["ts"]).dt.total_seconds()
t = t[(t["secs_to_close"] >= 30) & (t["secs_to_close"] <= 720)]
t["bucket"] = (t["secs_to_close"] // 5).astype(int)
t = (t.sort_values(["ticker", "bucket", "secs_to_close"])
       .drop_duplicates(["ticker", "bucket"], keep="first"))
print(f"[fit] training pairs: {len(t):,}")
print(f"      date range: {t['ts'].min():%Y-%m-%d %H:%M} → {t['ts'].max():%Y-%m-%d %H:%M}")

X = t["theo"].clip(1e-4, 1 - 1e-4).values
y = t["outcome"].astype(float).values

# ---- 4. Platt fit: minimize negative log-likelihood of p = sigmoid(a + b * logit(theo)) ----
def platt_fit(theo_v, y_v):
    z = logit(np.clip(theo_v, 1e-4, 1 - 1e-4))
    def nll(params):
        a, b = params
        p = expit(a + b * z)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y_v * np.log(p) + (1 - y_v) * np.log(1 - p)).mean()
    res = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000})
    return float(res.x[0]), float(res.x[1])

a, b = platt_fit(X, y)
print("\n" + "=" * 60)
print("PLATT FIT (full data)")
print("=" * 60)
print(f"  a = {a:+.4f}   (prior: -0.025,  Δ = {a - (-0.025):+.4f})")
print(f"  b = {b:+.4f}   (prior: +1.150,  Δ = {b - 1.150:+.4f})")
# Crossover (where calibrated == raw): solve a + b*z == z  ->  z = a/(1-b) ; then theo = expit(z)
if abs(1 - b) > 1e-6:
    z_cross = a / (1 - b)
    p_cross = float(expit(z_cross))
    print(f"  crossover theo (recal == raw): {p_cross:.3f}  (prior: 0.54)")

# ---- 5. 5-fold CV Brier: raw vs recal ----
print("\n" + "=" * 60)
print("5-FOLD CV BRIER  (ticker-stratified would be ideal; using random KFold like prior)")
print("=" * 60)
b_raw_folds, b_recal_folds = [], []
splits = list(kfold_splits(len(X), n_splits=5, seed=42))
for fold, (tr_i, te_i) in enumerate(splits):
    a_f, b_f = platt_fit(X[tr_i], y[tr_i])
    z_te = logit(np.clip(X[te_i], 1e-4, 1 - 1e-4))
    p_recal = expit(a_f + b_f * z_te)
    b_raw = ((X[te_i] - y[te_i]) ** 2).mean()
    b_rec = ((p_recal - y[te_i]) ** 2).mean()
    b_raw_folds.append(b_raw)
    b_recal_folds.append(b_rec)
    print(f"  fold {fold+1}: a={a_f:+.4f} b={b_f:+.4f}  "
          f"B_raw={b_raw:.4f}  B_recal={b_rec:.4f}  Δ={b_raw - b_rec:+.5f}")

b_raw_m  = float(np.mean(b_raw_folds))
b_rec_m  = float(np.mean(b_recal_folds))
b_raw_se = float(np.std(b_raw_folds, ddof=1) / np.sqrt(5))
b_rec_se = float(np.std(b_recal_folds, ddof=1) / np.sqrt(5))

print(f"\n  CV Brier raw:   {b_raw_m:.4f}  (SE across folds: {b_raw_se:.4f})")
print(f"  CV Brier recal: {b_rec_m:.4f}  (SE across folds: {b_rec_se:.4f})")
print(f"  Δ = {b_raw_m - b_rec_m:+.5f}  ({100*(b_raw_m - b_rec_m)/b_raw_m:+.2f}%)")
print(f"  prior delta:  +0.0008  (~0.6%)")

# ---- 6. Per-fold parameter stability ----
fold_params = []
for tr_i, _ in splits:
    fold_params.append(platt_fit(X[tr_i], y[tr_i]))
fold_params = np.array(fold_params)
print(f"\n  Fold a: mean={fold_params[:,0].mean():+.4f}  "
      f"sd={fold_params[:,0].std(ddof=1):.4f}  "
      f"range=[{fold_params[:,0].min():+.4f}, {fold_params[:,0].max():+.4f}]")
print(f"  Fold b: mean={fold_params[:,1].mean():+.4f}  "
      f"sd={fold_params[:,1].std(ddof=1):.4f}  "
      f"range=[{fold_params[:,1].min():+.4f}, {fold_params[:,1].max():+.4f}]")
