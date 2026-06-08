"""Two questions:
1. Is Parkinson upward-biased vs close-to-close (C2C) realized vol on ETH 1-min?
2. If so, does refitting HAR with a C2C target (not just scaling) change the coefficients meaningfully?

Note: just scaling the label by 0.92 is mathematically equivalent to scaling β by 0.92 (same R², same shape).
What's actually informative is using a DIFFERENT realized-vol estimator as the target."""

import math, sys, time
from datetime import datetime, timedelta, timezone
import numpy as np, requests

sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/Aston")
from har_fit import (fetch_all_candles, parkinson_per_minute,
                     annualize, H_15, H_30, H_4H, H_24H)

# --- C2C per-minute variance ---
def c2c_per_minute(closes: np.ndarray) -> np.ndarray:
    # Log returns squared. Index aligns with the close-of-minute t.
    out = np.zeros_like(closes)
    r = np.diff(np.log(closes))
    out[1:] = r * r
    return out

def build_table(highs, lows, closes, target='parkinson'):
    park = parkinson_per_minute(highs, lows)
    c2c  = c2c_per_minute(closes)
    tgt_arr = park if target == 'parkinson' else c2c
    n = len(park)
    X, y = [], []
    for t in range(H_24H, n - H_15, H_15):
        X.append([
            1.0,
            annualize(float(park[t - H_15:t].sum()),  H_15),
            annualize(float(park[t - H_30:t].sum()),  H_30),
            annualize(float(park[t - H_4H:t].sum()),  H_4H),
            annualize(float(park[t - H_24H:t].sum()), H_24H),
        ])
        y.append(annualize(float(tgt_arr[t:t + H_15].sum()), H_15))
    return np.array(X), np.array(y)

print("Pulling 30 days ETH-USD candles...")
rows = fetch_all_candles("ETH-USD", 30)
print(f"  got {len(rows)} candles")
rows = sorted(rows, key=lambda r: r[0])
lows  = np.array([float(r[1]) for r in rows])
highs = np.array([float(r[2]) for r in rows])
closes = np.array([float(r[4]) for r in rows])

# --- Q1: Parkinson vs C2C per 15-min window ---
park_min = parkinson_per_minute(highs, lows)
c2c_min  = c2c_per_minute(closes)
park_15 = []
c2c_15  = []
for t in range(0, len(closes) - H_15, H_15):
    park_15.append(annualize(float(park_min[t:t + H_15].sum()), H_15))
    c2c_15.append(annualize(float(c2c_min[t:t + H_15].sum()), H_15))
park_15 = np.array(park_15)
c2c_15  = np.array(c2c_15)
mask = (park_15 > 0) & (c2c_15 > 0)
park_15 = park_15[mask]
c2c_15  = c2c_15[mask]

print()
print("Q1: Parkinson vs C2C realized σ (15-min windows, annualized)")
print("=" * 60)
print(f"  N windows: {len(park_15)}")
print(f"  Parkinson:  mean {park_15.mean():.4f}  median {np.median(park_15):.4f}")
print(f"  C2C:        mean {c2c_15.mean():.4f}  median {np.median(c2c_15):.4f}")
print(f"  Parkinson / C2C ratio (mean):   {park_15.mean() / c2c_15.mean():.3f}")
print(f"  Parkinson / C2C ratio (median): {np.median(park_15) / np.median(c2c_15):.3f}")
print(f"  Per-window ratio mean: {np.mean(park_15 / c2c_15):.3f}")
print(f"  Per-window ratio median: {np.median(park_15 / c2c_15):.3f}")
print(f"  Theoretical Parkinson efficiency: ~5x more efficient than C2C, but UNBIASED if returns are pure Brownian.")
print(f"  Any persistent ratio != 1 suggests microstructure noise (Parkinson HIGH-biased) or jumps (C2C bias varies).")

# --- Q2: Refit HAR with C2C target ---
print()
print("Q2: Refit HAR with target = forward C2C-realized (not Parkinson)")
print("=" * 60)
for target in ['parkinson', 'c2c']:
    X, y = build_table(highs, lows, closes, target=target)
    split = int(0.8 * len(y))
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    pred = Xte @ beta
    r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    print(f"\n  target = {target.upper()}  (n_train={len(ytr)})")
    print(f"    β0    = {beta[0]:+.4f}")
    print(f"    β_15  = {beta[1]:+.4f}")
    print(f"    β_30  = {beta[2]:+.4f}")
    print(f"    β_4h  = {beta[3]:+.4f}")
    print(f"    β_24h = {beta[4]:+.4f}")
    print(f"    R² OOS = {r2:.3f}")
    print(f"    mean forecast σ (OOS): {pred.mean():.4f}")
    print(f"    mean realized σ (OOS): {yte.mean():.4f}")
    print(f"    bias = pred - realized: {pred.mean() - yte.mean():+.4f}")
