"""HAR-RV fitter on Parkinson (high-low) RV.

One-time setup script.  Pulls 30 days of 1-minute Coinbase candles,
computes per-minute Parkinson variance from H/L, builds 4 trailing-
window RVs (15m / 30m / 4h / 24h) plus the next-15m Parkinson RV
label, fits OLS, and writes coefficients to har_coefficients.json.

Re-run weekly on a rolling window to keep the betas current.

Usage:
    python3 har_fit.py
    python3 har_fit.py --product ETH-USD --days 21
"""

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

CANDLE_URL_FMT = "https://api.exchange.coinbase.com/products/{}/candles"
GRANULARITY = 60
CANDLES_PER_CALL = 300
ANN_MINUTES = 365.25 * 24 * 60  # 525,960 — minutes per year, 24/7
FOUR_LN2 = 4.0 * math.log(2.0)

# Coinbase Exchange public limit: 10 req/s.  0.6s pacing → ~1.7 req/s,
# plenty of headroom and avoids tripping any per-second burst rules.
REQ_PACING_S = 0.6
# Tolerate occasional empty batches (Coinbase glitch) without ending the
# whole fetch.  Stop only after this many CONSECUTIVE empty windows —
# enough to recognize "no more historical data" without giving up early.
EMPTY_STREAK_STOP = 5

H_15, H_30, H_4H, H_24H = 15, 30, 240, 1440


def _fetch_window(url: str, start, end,
                  max_attempts: int = 5) -> list:
    """Fetch one candle window with exponential-backoff retry on 429,
    5xx, network errors, and empty responses.  Returns the list of
    rows or [] if every attempt failed."""
    for attempt in range(max_attempts):
        try:
            r = requests.get(url, params={
                "granularity": GRANULARITY,
                "start": start.isoformat(),
                "end":   end.isoformat(),
            }, timeout=15)
            if r.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"  [429] backing off {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = min(2 ** attempt, 30)
                print(f"  [{r.status_code}] backing off {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            batch = r.json()
            if batch:
                return batch
            # Empty response — could be a transient glitch.  Retry once
            # before letting the caller treat it as legitimately empty.
            if attempt < max_attempts - 1:
                time.sleep(1.0)
                continue
            return []
        except requests.exceptions.RequestException as e:
            wait = min(2 ** attempt, 30)
            print(f"  [err {e}] backing off {wait}s")
            time.sleep(wait)
    return []


def fetch_all_candles(product: str, days_back: int) -> list:
    """Page backward through /candles in 300-minute chunks with
    retry-on-failure semantics.  Tolerates transient empties; only
    stops when EMPTY_STREAK_STOP consecutive windows are empty,
    indicating we've hit the end of Coinbase's history."""
    url = CANDLE_URL_FMT.format(product)
    end = datetime.now(timezone.utc)
    start_overall = end - timedelta(days=days_back)
    cursor = end
    rows = []
    batches = 0
    empty_streak = 0
    expected_batches = (days_back * 1440 + CANDLES_PER_CALL - 1) // CANDLES_PER_CALL

    while cursor > start_overall:
        batch_start = max(cursor - timedelta(minutes=CANDLES_PER_CALL),
                          start_overall)
        batch = _fetch_window(url, batch_start, cursor)
        batches += 1

        if batch:
            empty_streak = 0
            rows.extend(batch)
            if batches % 20 == 0:
                print(f"  [{batches}/{expected_batches}] "
                      f"{len(rows)} candles so far  "
                      f"({batch_start.isoformat()[:16]} ← {cursor.isoformat()[:16]})")
        else:
            empty_streak += 1
            print(f"  [empty {empty_streak}/{EMPTY_STREAK_STOP}] "
                  f"window {batch_start.isoformat()[:16]} ← "
                  f"{cursor.isoformat()[:16]}")
            if empty_streak >= EMPTY_STREAK_STOP:
                print(f"  [stop] {EMPTY_STREAK_STOP} consecutive empty "
                      f"batches — assuming no older data available")
                break

        cursor = batch_start
        time.sleep(REQ_PACING_S)

    pct = len(rows) / (days_back * 1440) * 100
    print(f"  fetched {batches} batches, {len(rows)} candles "
          f"({pct:.1f}% of {days_back*1440} expected)")

    seen = set()
    cleaned = []
    for row in sorted(rows, key=lambda r: r[0]):
        if row[0] in seen:
            continue
        seen.add(row[0])
        cleaned.append(row)
    return cleaned


def annualize(sq_sum: float, window_minutes: int) -> float:
    if sq_sum <= 0:
        return 0.0
    return math.sqrt(sq_sum * (ANN_MINUTES / window_minutes))


def parkinson_per_minute(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """Per-minute variance estimate: ln(H/L)² / (4·ln 2).  Negative /
    degenerate inputs clamp to 0."""
    safe = (highs > 0) & (lows > 0) & (highs > lows)
    out = np.zeros_like(highs)
    out[safe] = np.log(highs[safe] / lows[safe]) ** 2 / FOUR_LN2
    return out


def build_training_table(highs: np.ndarray,
                          lows: np.ndarray) -> tuple:
    """Walk in non-overlapping 15-min steps.  Each row = predictors
    (Parkinson RV at 4 horizons) observed at t plus the label
    (Parkinson RV over [t, t+15min])."""
    var = parkinson_per_minute(highs, lows)
    n = len(var)
    X, y = [], []
    for t in range(H_24H, n - H_15, H_15):
        X.append([
            1.0,
            annualize(float(var[t - H_15:t].sum()),  H_15),
            annualize(float(var[t - H_30:t].sum()),  H_30),
            annualize(float(var[t - H_4H:t].sum()),  H_4H),
            annualize(float(var[t - H_24H:t].sum()), H_24H),
        ])
        y.append(annualize(float(var[t:t + H_15].sum()), H_15))
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true",
                    help="Fit and print coefficients but do NOT overwrite "
                         "settings/har_coefficients.json (read-only research).")
    args = ap.parse_args()

    print(f"Pulling {args.days} days of {args.product} 1-minute candles…")
    rows = fetch_all_candles(args.product, args.days)
    print(f"  got {len(rows)} candles "
          f"(expected ~{args.days * 1440})")
    if len(rows) < H_24H + H_15 + 1:
        raise RuntimeError("Not enough candles to fit HAR.")

    # Coinbase row order: [time, low, high, open, close, volume]
    lows  = np.array([float(r[1]) for r in rows])
    highs = np.array([float(r[2]) for r in rows])

    print("Building training table (Parkinson H/L)…")
    X, y = build_training_table(highs, lows)
    print(f"  {len(y)} labeled rows")

    split = int(0.8 * len(y))
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    def r2(X_, y_):
        pred = X_ @ beta
        ss_res = ((y_ - pred) ** 2).sum()
        ss_tot = ((y_ - y_.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    out = {
        "product":   args.product,
        "estimator": "parkinson",
        "beta0":     float(beta[0]),
        "beta_15":   float(beta[1]),
        "beta_30":   float(beta[2]),
        "beta_4h":   float(beta[3]),
        "beta_24h":  float(beta[4]),
        "r2_train":  r2(X_train, y_train),
        "r2_test":   r2(X_test,  y_test),
        "n_train":   int(len(y_train)),
        "n_test":    int(len(y_test)),
        "days_back": args.days,
        "fit_at":    datetime.now(timezone.utc).isoformat(),
    }
    # Coefficients live in Aston/settings/ (next to aston_settings.json) —
    # app.py and recorder.py load them from there.
    path = Path(__file__).resolve().parents[1] / "settings" / "har_coefficients.json"
    if args.dry_run:
        print()
        print("DRY RUN — not written (live coefficients untouched)")
    else:
        with path.open("w") as f:
            json.dump(out, f, indent=2)
        print()
        print(f"Wrote {path}")
    print(f"  β0    = {out['beta0']:+.4f}")
    print(f"  β_15  = {out['beta_15']:+.4f}")
    print(f"  β_30  = {out['beta_30']:+.4f}")
    print(f"  β_4h  = {out['beta_4h']:+.4f}")
    print(f"  β_24h = {out['beta_24h']:+.4f}")
    print(f"  R² in-sample     = {out['r2_train']:.3f}")
    print(f"  R² out-of-sample = {out['r2_test']:.3f}")


if __name__ == "__main__":
    main()
