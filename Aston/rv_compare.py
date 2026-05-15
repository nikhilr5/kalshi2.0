"""Realized-vol estimator bake-off.

Pulls 30 days of 1-minute Coinbase OHLC, computes four different vol
estimators over a trailing 4-hour window at each minute, and measures
how well each forecasts the NEXT 15 minutes of close-to-close RV.

Estimators compared
-------------------
  cc_1min     : Σ r²  on close-to-close 1-min returns (current HAR input)
  parkinson   : Σ ln(H/L)² / (4·ln2)              — high/low range
  garman_klass: Σ [0.5·ln(H/L)² − (2·ln2−1)·ln(C/O)²]  — OHLC
  cc_5min     : Σ r²  on 5-min close-to-close returns (same 4-h window)

Each is annualized assuming 24/7 trading (525,960 min/year).  Non-
overlapping 15-min stride for the comparison so the rows are roughly
independent.

Output columns
--------------
  mean_σ : mean of σ_estimator (level)
  bias   : mean(σ_estimator − σ_future) — positive = estimator too high
  corr   : corr(σ_estimator, σ_future)  — predictive strength
  RMSE   : √mean((σ_estimator − σ_future)²)
  MAPE   : mean(|σ_estimator − σ_future| / σ_future)

Pick the estimator with high `corr` and bias near zero.  RMSE and MAPE
break ties.

Usage:
    python3 rv_compare.py
    python3 rv_compare.py --product ETH-USD --days 14 --window 240
"""

import argparse
import math
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

CANDLE_URL_FMT = "https://api.exchange.coinbase.com/products/{}/candles"
CANDLES_PER_CALL = 300
ANN_MIN = 365.25 * 24 * 60  # 525,960
FOUR_LN2 = 4.0 * math.log(2.0)
GK_K = 2.0 * math.log(2.0) - 1.0


def fetch_candles(product: str, days_back: int) -> np.ndarray:
    """Page backward through /candles, return np.ndarray columns
    [time, low, high, open, close, volume] sorted by time, deduped."""
    url = CANDLE_URL_FMT.format(product)
    end = datetime.now(tz=timezone.utc)
    start_overall = end - timedelta(days=days_back)
    cursor = end
    rows = []
    while cursor > start_overall:
        batch_start = max(cursor - timedelta(minutes=CANDLES_PER_CALL),
                          start_overall)
        r = requests.get(url, params={
            "granularity": 60,
            "start": batch_start.isoformat(),
            "end": cursor.isoformat(),
        }, timeout=10)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch_start
        time.sleep(0.25)
    arr = np.array(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    # Dedupe on time.
    _, idx = np.unique(arr[:, 0], return_index=True)
    return arr[np.sort(idx)]


def rolling_sum(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum of `window` elements ending at each index.

    Output length = len(a) - window + 1.  Output[i] = sum(a[i:i+window]).
    """
    cum = np.concatenate([[0.0], np.cumsum(a)])
    return cum[window:] - cum[:-window]


def annualize_var(var_sum: np.ndarray, period_min: int) -> np.ndarray:
    """var_sum is variance accrued over `period_min` minutes.  Scale
    to annual variance, return σ."""
    var_sum = np.maximum(var_sum, 0.0)
    return np.sqrt(var_sum * (ANN_MIN / period_min))


def per_min_increments(O, H, L, C):
    """Per-minute variance contributions for the three OHLC-based
    estimators.  cc_5min is handled separately because it operates on
    a downsampled series."""
    # Use prepend=O[0] for first row so r[0] is well-defined and
    # equal to ln(C[0]/O[0]) — matches the per-minute idea that
    # this candle's return is its own open→close, not zero.
    log_c = np.log(C)
    r1 = np.empty_like(C)
    r1[0] = math.log(C[0] / O[0]) if O[0] > 0 else 0.0
    r1[1:] = np.diff(log_c)

    park = np.log(H / L) ** 2 / FOUR_LN2
    gk = 0.5 * np.log(H / L) ** 2 - GK_K * np.log(C / O) ** 2
    gk = np.maximum(gk, 0.0)  # numerical floor — GK can dip negative
    return {
        "cc_1min":      r1 ** 2,
        "parkinson":    park,
        "garman_klass": gk,
    }


def cc_5min_series(C: np.ndarray, window_min: int) -> np.ndarray:
    """σ over a `window_min`-min trailing window using 5-min sampling.

    Returns an array length N (1-min indexed).  NaN where not defined
    (need ≥ window_min/5 prior 5-min returns; also NaN at non-5-min
    boundaries — caller compares only at boundary timestamps)."""
    N = len(C)
    # Down-sample closes to 5-min boundaries (indices 0, 5, 10, ...).
    C_5 = C[::5]
    n_returns_needed = window_min // 5  # 48 for 4h
    log_c5 = np.log(C_5)
    r5 = np.empty_like(C_5)
    r5[0] = 0.0
    r5[1:] = np.diff(log_c5)
    r5_sq = r5 ** 2
    rs = rolling_sum(r5_sq, n_returns_needed)
    sigma_5 = annualize_var(rs, window_min)
    # rs[i] corresponds to 5-min index (i + n_returns_needed - 1), which
    # is 1-min index 5 * (i + n_returns_needed - 1).
    out = np.full(N, np.nan)
    for i, s in enumerate(sigma_5):
        t1 = 5 * (i + n_returns_needed - 1)
        if t1 < N:
            out[t1] = s
    return out


def future_15m_sigma(r1_sq: np.ndarray, forecast_min: int) -> np.ndarray:
    """At minute t, σ over [t+1, t+forecast_min].  Output length =
    N - forecast_min.  NaN-padded at the tail by the caller."""
    cum = np.concatenate([[0.0], np.cumsum(r1_sq)])
    N = len(r1_sq)
    # var[t] = sum r²[t+1..t+forecast_min]
    var = cum[forecast_min + 1: N + 1] - cum[1: N - forecast_min + 1]
    return annualize_var(var, forecast_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--window", type=int, default=240,
                    help="trailing window minutes (default 240 = 4h)")
    ap.add_argument("--forecast", type=int, default=15,
                    help="forecast horizon minutes (default 15)")
    ap.add_argument("--stride", type=int, default=15,
                    help="sample stride minutes (default 15 = "
                         "non-overlapping label windows)")
    ap.add_argument("--csv", default=None,
                    help="optional output path for the per-sample CSV")
    args = ap.parse_args()

    print(f"Pulling {args.days} days of {args.product} 1-min candles…")
    candles = fetch_candles(args.product, args.days)
    print(f"  got {len(candles)} candles")
    if len(candles) < args.window + args.forecast + 10:
        sys.exit("Not enough data.")

    O = candles[:, 3]
    H = candles[:, 2]
    L = candles[:, 1]
    C = candles[:, 4]
    N = len(C)

    incs = per_min_increments(O, H, L, C)
    # Rolling 4h σ for the three per-minute estimators
    sigmas: dict[str, np.ndarray] = {}
    for name, inc in incs.items():
        rs = rolling_sum(inc, args.window)
        full = np.full(N, np.nan)
        # rs[i] = sum inc[i : i+window].  Convention: σ AT minute t
        # uses the trailing window ending AT t, i.e. inc[t-window+1 : t+1].
        # So rs[i] corresponds to t = i + window - 1.
        full[args.window - 1:] = annualize_var(rs, args.window)
        sigmas[name] = full

    # cc_5min in a separate path (uses 5-min subsampling)
    sigmas["cc_5min"] = cc_5min_series(C, args.window)

    # Label
    future = future_15m_sigma(incs["cc_1min"], args.forecast)
    future_full = np.full(N, np.nan)
    # future[t] is defined for t in [0, N - forecast_min - 1] (indexes
    # into future).  Map back to 1-min t.
    future_full[: N - args.forecast] = future

    # Sample at every `stride` minutes starting at the first index
    # where every estimator is defined.  Stride 15 + start at window
    # boundary keeps cc_5min defined too (window=240 → start at 240,
    # 240 % 5 == 0).
    start = args.window
    sample_idx = np.arange(start, N - args.forecast, args.stride)
    label = future_full[sample_idx]

    print()
    header = f"{'Estimator':<14} {'mean_σ':>8} {'bias':>9} {'bias%':>7} {'corr':>6} {'RMSE':>8} {'MAPE':>7}"
    print(header)
    print("-" * len(header))
    estimator_samples: dict[str, np.ndarray] = {}
    for name in ("cc_1min", "parkinson", "garman_klass", "cc_5min"):
        sig = sigmas[name][sample_idx]
        valid = ~np.isnan(sig) & ~np.isnan(label) & (label > 0)
        s = sig[valid]
        y = label[valid]
        if len(s) < 10:
            print(f"{name:<14}  (insufficient samples: {len(s)})")
            continue
        bias = (s - y).mean()
        bias_pct = bias / y.mean() * 100
        corr = float(np.corrcoef(s, y)[0, 1])
        rmse = math.sqrt(((s - y) ** 2).mean())
        mape = (np.abs(s - y) / y).mean()
        mean_s = s.mean()
        estimator_samples[name] = sig
        print(f"{name:<14} {mean_s:>8.3f} {bias:>+9.3f} {bias_pct:>+6.1f}% "
              f"{corr:>6.3f} {rmse:>8.3f} {mape:>6.1%}")
    print("-" * len(header))
    valid_lbl = ~np.isnan(label) & (label > 0)
    print(f"{'(target σ)':<14} {label[valid_lbl].mean():>8.3f}"
          f"   n = {valid_lbl.sum()} samples")

    if args.csv:
        # Save tidy table for optional plotting.
        out = np.full((len(sample_idx), 6), np.nan)
        out[:, 0] = candles[sample_idx, 0]  # unix ts
        out[:, 1] = label
        for i, name in enumerate(("cc_1min", "parkinson",
                                   "garman_klass", "cc_5min")):
            out[:, 2 + i] = sigmas[name][sample_idx]
        header_csv = "ts,future_15m," + ",".join(
            ("cc_1min", "parkinson", "garman_klass", "cc_5min"))
        np.savetxt(args.csv, out, delimiter=",", header=header_csv,
                   comments="", fmt="%.6f")
        print(f"\nSaved per-sample CSV → {args.csv}")


if __name__ == "__main__":
    main()
