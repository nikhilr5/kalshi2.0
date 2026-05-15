"""Parkinson RV verifier.

Pulls the last N minutes of Coinbase 1-min candles and computes the
Parkinson realized vol the same way HARRVEstimator does live.  Print
every per-minute contribution so each step can be eyeballed against
the runtime tooltip.

Usage:
    python3 verify_parkinson.py                       # BTC 15m
    python3 verify_parkinson.py --product ETH-USD
    python3 verify_parkinson.py --horizon 30          # 30-min window
    python3 verify_parkinson.py --horizon 240         # 4h
"""

import argparse
import math
from datetime import datetime, timedelta, timezone

import requests

CANDLE_URL_FMT = "https://api.exchange.coinbase.com/products/{}/candles"
ANN_MIN = 365.25 * 24 * 60  # 525,960 — minutes per year (24/7)
FOUR_LN2 = 4.0 * math.log(2.0)


def fetch_candles(product: str, minutes: int) -> list:
    """Pull a window covering at least `minutes` minutes plus buffer."""
    url = CANDLE_URL_FMT.format(product)
    end = datetime.now(tz=timezone.utc)
    # Pull `minutes + 10` so we have headroom for the in-progress minute
    # and minor clock skew.  Coinbase caps at 300 candles per call, so
    # this is one round-trip for any horizon ≤ 290 minutes.
    start = end - timedelta(minutes=minutes + 10)
    r = requests.get(url, params={
        "granularity": 60,
        "start": start.isoformat(),
        "end":   end.isoformat(),
    }, timeout=10)
    r.raise_for_status()
    rows = r.json()
    rows.sort(key=lambda x: x[0])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--horizon", type=int, default=15,
                    help="window in minutes (default 15)")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-minute table")
    args = ap.parse_args()

    if args.horizon > 290:
        print("Horizons > 290 require pagination; use har_fit-style fetch.")
        return

    rows = fetch_candles(args.product, args.horizon)
    print(f"Pulled {len(rows)} candles from Coinbase")

    # Match the runtime: drop the current in-progress minute so live
    # aggregation owns it (HARRVEstimator on_price logic).
    now_minute = int(datetime.now(tz=timezone.utc).timestamp() // 60)
    closed = [r for r in rows if (int(r[0]) // 60) < now_minute]
    window = closed[-args.horizon:]

    if len(window) < args.horizon:
        print(f"  warning: only {len(window)} closed minutes available "
              f"(wanted {args.horizon})")

    earliest = datetime.fromtimestamp(window[0][0], tz=timezone.utc)
    latest   = datetime.fromtimestamp(window[-1][0], tz=timezone.utc)
    print(f"Window: {len(window)} minutes  "
          f"{earliest.strftime('%H:%M:%S')} → {latest.strftime('%H:%M:%S')} UTC")
    print()

    if args.verbose or args.horizon <= 30:
        print(f"  {'time':<10} {'high':>11} {'low':>11} "
              f"{'ln(H/L)²/(4·ln2)':>22}")

    total = 0.0
    for row in window:
        ts, low, high = row[0], row[1], row[2]
        if high > 0 and low > 0 and high > low:
            v = math.log(high / low) ** 2 / FOUR_LN2
        else:
            v = 0.0
        total += v
        if args.verbose or args.horizon <= 30:
            t = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")
            print(f"  {t:<10} {high:>11.2f} {low:>11.2f} {v:>22.4e}")

    print()
    print(f"Σ per-minute variances    = {total:.6e}")
    annual_var = total * (ANN_MIN / args.horizon)
    sigma = math.sqrt(annual_var) if annual_var > 0 else 0.0
    print(f"× (ANN_MIN / horizon)     = {annual_var:.6e}    (annual variance)")
    print(f"√annual variance          = {sigma:.6f}")
    print()
    print(f"σ (annualized Parkinson)  = {sigma*100:.2f}%")
    print()
    print("Cross-check against runtime tooltip immediately after this run.")


if __name__ == "__main__":
    main()
