"""
Backtest: IV Theo accuracy on settled KXBTCD above/below markets.

For each settled event:
1. Determine the settlement price (highest strike that settled Yes)
2. For each OTM "above" contract (strike > settlement):
   - Compute what IV Theo would have predicted at 1h, 2h, 4h before close
   - Compare to last traded price on Kalshi
   - Check the actual result (should be "no" since OTM)

Also produces a calibration analysis: group all contracts by their
distance from settlement, and check if market prices were fair.

Uses historical BTC hourly prices from Coinbase for spot at each lookback.
"""

import math
import sys
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path

# Add 4RunnerApp2.0 to path for kalshi_api import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "4RunnerApp2.0"))
from kalshi_api import KalshiAPI


# ============================================================================
# Helpers
# ============================================================================

def parse_strike(ticker: str) -> float:
    if "-T" in ticker:
        try:
            return float(ticker.split("-T")[1])
        except ValueError:
            return 0.0
    return 0.0


def display_strike(raw: float) -> float:
    return math.ceil(raw)


def get_coinbase_candles(product: str, start_iso: str, end_iso: str) -> list:
    """Fetch hourly candles from Coinbase for a time range."""
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {
        "start": start_iso,
        "end": end_iso,
        "granularity": 3600,  # 1 hour
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    # Returns [[timestamp, low, high, open, close, volume], ...]
    return resp.json()


def spot_at_time(candles: list, target_ts: float) -> float:
    """Find the closest candle close price to a target timestamp."""
    best = None
    best_dist = float("inf")
    for c in candles:
        dist = abs(c[0] - target_ts)
        if dist < best_dist:
            best_dist = dist
            best = c[4]  # close price
    return best or 0.0


def bs_prob_above(S: float, K: float, sigma: float, T_years: float) -> float:
    """Black-Scholes P(S > K) = N(d2)."""
    if T_years <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T_years)
    d2 = (math.log(S / K) + (-0.5 * sigma * sigma) * T_years) / (sigma * sqrt_T)
    return max(min(0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0))), 1.0), 0.0)


# ============================================================================
# Main
# ============================================================================

def main():
    api = KalshiAPI()

    print("Fetching settled KXBTCD markets...")
    markets = api.get_markets(series_ticker="KXBTCD", status="settled")
    print(f"Got {len(markets)} settled markets")

    # Group by event
    events = {}
    for m in markets:
        et = m.get("event_ticker", "")
        if et not in events:
            events[et] = []
        events[et].append(m)

    print(f"{len(events)} events\n")

    # IV levels to test
    iv_levels = [0.40, 0.50, 0.60, 0.70, 0.80]

    # Lookback hours before close
    lookbacks_h = [1, 2, 4, 6]

    # Collect all results for calibration
    # Key: (iv, lookback_h, otm_bucket) -> {"count": N, "settled_yes": N, "theo_sum": f, "market_price_sum": f}
    calibration = defaultdict(lambda: {"count": 0, "settled_yes": 0, "theo_sum": 0.0, "market_sum": 0.0})

    # Per-event analysis
    for et in sorted(events):
        ms = events[et]
        close_str = ms[0].get("close_time", "")
        if not close_str:
            continue

        close_utc = datetime.fromisoformat(close_str.replace("Z", "+00:00"))

        # Settlement price = highest display strike that settled "yes"
        yes_strikes = []
        for m in ms:
            if m.get("result") == "yes":
                raw = parse_strike(m["ticker"])
                if raw > 0:
                    yes_strikes.append(display_strike(raw))
        if not yes_strikes:
            continue
        settle_price = max(yes_strikes)

        # Get historical spot prices from Coinbase
        start_t = close_utc - timedelta(hours=max(lookbacks_h) + 1)
        try:
            candles = get_coinbase_candles(
                "BTC-USD",
                start_t.isoformat(),
                close_utc.isoformat(),
            )
        except Exception as e:
            print(f"  Coinbase error for {et}: {e}")
            continue

        print(f"{'='*70}")
        print(f"Event: {et}")
        print(f"Close: {close_str}")
        print(f"Settlement: ${settle_price:,.0f}")
        print(f"Markets: {len(ms)} ({len(yes_strikes)} yes, {len(ms)-len(yes_strikes)} no)")

        # Analyze OTM "above" contracts (strike > settlement = settled No)
        otm_above = []
        for m in ms:
            if m.get("result") != "no":
                continue
            raw = parse_strike(m["ticker"])
            if raw <= 0:
                continue
            disp = display_strike(raw)
            if disp <= settle_price:
                continue  # this was ITM or ATM

            last_price = float(m.get("last_price_dollars", 0) or 0)
            otm_above.append({
                "ticker": m["ticker"],
                "strike": disp,
                "otm_pct": (disp - settle_price) / settle_price * 100,
                "last_price": last_price,
                "result": "no",
            })

        otm_above.sort(key=lambda x: x["strike"])

        if not otm_above:
            print("  No OTM above contracts")
            continue

        # For each lookback, compute theo at various IVs
        print(f"\n  OTM Above contracts (strike > ${settle_price:,.0f}):")
        print(f"  {'Strike':>10} {'OTM%':>6} {'LastPx':>7} | ", end="")
        for lb in lookbacks_h:
            print(f"  Theo@{lb}h(IV=60%)", end="")
        print()

        for contract in otm_above[:15]:  # show top 15 closest to ATM
            K = contract["strike"]
            otm_pct = contract["otm_pct"]
            last_px = contract["last_price"]

            line = f"  ${K:>8,.0f} {otm_pct:>5.1f}% ${last_px:>5.2f} | "

            for lb in lookbacks_h:
                target_ts = (close_utc - timedelta(hours=lb)).timestamp()
                spot = spot_at_time(candles, target_ts)
                T_years = lb / (365.25 * 24)
                theo = bs_prob_above(spot, K, 0.60, T_years)
                line += f"  ${theo:>6.4f}          "

                # OTM bucket: 0-1%, 1-2%, 2-3%, 3-5%, 5%+
                if otm_pct < 1:
                    bucket = "0-1%"
                elif otm_pct < 2:
                    bucket = "1-2%"
                elif otm_pct < 3:
                    bucket = "2-3%"
                elif otm_pct < 5:
                    bucket = "3-5%"
                else:
                    bucket = "5%+"

                for iv in iv_levels:
                    theo_iv = bs_prob_above(spot, K, iv, T_years)
                    key = (iv, lb, bucket)
                    calibration[key]["count"] += 1
                    calibration[key]["settled_yes"] += 0  # all these are OTM, settled no
                    calibration[key]["theo_sum"] += theo_iv
                    calibration[key]["market_sum"] += last_px

            print(line)

    # =========================================================================
    # Calibration Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("CALIBRATION SUMMARY — OTM Above Contracts")
    print("For each IV & lookback: avg theo vs avg market price vs actual settle rate")
    print("All these contracts settled NO (worth $0). Market price = what someone paid.\n")

    for lb in lookbacks_h:
        print(f"\n--- {lb}h before close ---")
        print(f"  {'OTM Bucket':>10} {'Count':>6} | ", end="")
        for iv in iv_levels:
            print(f"  IV={iv:.0%}  ", end="")
        print("| MktPx  | Actual")

        for bucket in ["0-1%", "1-2%", "2-3%", "3-5%", "5%+"]:
            line = f"  {bucket:>10} "
            count = 0
            for iv in iv_levels:
                key = (iv, lb, bucket)
                d = calibration[key]
                count = d["count"]
                if count > 0:
                    avg_theo = d["theo_sum"] / count
                    line_start = f"{count:>6} | " if iv == iv_levels[0] else ""
                    line += f"{line_start} ${avg_theo:.4f} "
                else:
                    if iv == iv_levels[0]:
                        line += f"{'0':>6} | "
                    line += f"    --   "

            # Market price (same across IVs)
            key0 = (iv_levels[0], lb, bucket)
            d0 = calibration[key0]
            if d0["count"] > 0:
                avg_mkt = d0["market_sum"] / d0["count"]
                line += f"| ${avg_mkt:.4f} | $0.00"
            else:
                line += "|   --   | --"

            print(line)

    # =========================================================================
    # Key Question: Is the market overpriced?
    # =========================================================================
    print(f"\n{'='*70}")
    print("KEY INSIGHT: Overpayment on OTM tails")
    print("If avg market price > avg theo → market is overpriced → selling edge exists\n")

    for lb in [2, 4]:
        print(f"  {lb}h before close, IV=60%:")
        for bucket in ["1-2%", "2-3%", "3-5%", "5%+"]:
            key = (0.60, lb, bucket)
            d = calibration[key]
            if d["count"] > 0:
                avg_theo = d["theo_sum"] / d["count"]
                avg_mkt = d["market_sum"] / d["count"]
                edge = avg_mkt - avg_theo
                pct = (edge / avg_theo * 100) if avg_theo > 0.001 else float("inf")
                symbol = "✓ EDGE" if edge > 0.01 else "  thin" if edge > 0 else "  none"
                print(f"    {bucket:>5} OTM: theo=${avg_theo:.4f}  mkt=${avg_mkt:.4f}  "
                      f"edge=${edge:+.4f} ({pct:+.0f}%)  {symbol}")


if __name__ == "__main__":
    main()
