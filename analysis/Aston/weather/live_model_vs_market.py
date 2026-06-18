"""Core deliverable: LIVE weather-model vs Kalshi-market comparison.

For each city x forecast-day x bucket: model prob (Normal CDF around the NWS
forecast high) vs market mid. Flags buckets where |model - market| exceeds the
spread (a potential taker edge), ranked by edge-beyond-spread.

Run:  python3 live_model_vs_market.py
Writes: cache/live_compare_<UTCstamp>.csv  and prints a ranked table.

Robust: one bad city/market/orderbook is logged and skipped, never crashes.
"""

import sys
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

# Station local tz (for deciding which markets are "today" vs day-ahead).
STATION_TZ = {"NYC": "America/New_York", "BOS": "America/New_York",
              "LAX": "America/Los_Angeles", "OKC": "America/Chicago",
              "DAL": "America/Chicago"}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weather_lib import (CITIES, CACHE_DIR, nws_forecast_highs, parse_bucket,
                         bucket_prob, book_mid_spread, ticker_day)

# Per-station sigma (F) — MEASURED day-ahead forecast error std, from IEM GFS+NAM
# MOS (12Z run -> next-afternoon n_x) vs IEM observed daily highs, n=78 aligned
# pairs/station over 2026-04-01..06-18. Residual std after the forecast.
SIGMA_F = {"NYC": 3.1, "LAX": 1.9, "OKC": 3.3, "BOS": 4.0, "DAL": 3.0}

# Measured GFS+NAM MOS forecast bias (F) = mean(forecast - observed), n=78.
# NOT applied: we measured bias on MOS but forecast off the NWS gridpoint — a
# different product — so applying the MOS bias to the NWS mean is unjustified and
# empirically degraded BOS/OKC agreement (pushed the mode into a tail artifact).
# Kept here only to document that a real per-product bias calibration is future
# work. Model runs bias-free (BIAS_F all 0) until NWS-grid-vs-observed is measured.
MEASURED_MOS_BIAS_F = {"NYC": -0.65, "LAX": -0.87, "OKC": -1.83, "BOS": -2.14, "DAL": 0.33}
BIAS_F = {c: 0.0 for c in CITIES}

SIGMA_SOURCE = ("MEASURED (magnitude) — per-station day-ahead forecast-error std "
                "from IEM GFS+NAM MOS vs observed highs, n=78/station (Apr-Jun "
                "2026): NYC 3.1, LAX 1.9, OKC 3.3, BOS 4.0, DAL 3.0 F. Applied to "
                "the NWS gridpoint forecast as a proxy. Bias correction NOT applied "
                "(measured on MOS, not NWS grid). Spring window; summer may widen.")


def _suffix_to_iso(suffix):
    """'26JUN19' -> '2026-06-19' (local forecast day)."""
    try:
        return datetime.strptime(suffix, "%y%b%d").date().isoformat()
    except Exception:
        return None


def build(skip_same_day=True):
    """skip_same_day: drop markets whose forecast day is the station's local
    'today' — those are hours from settlement and the market already prices
    intraday obs the day-ahead grid forecast can't see. The clean test is
    day-ahead (and beyond) only."""
    api = _api()
    rows = []
    for city, cfg in CITIES.items():
        local_today = datetime.now(ZoneInfo(STATION_TZ[city])).date().isoformat()
        try:
            highs = nws_forecast_highs(cfg["lat"], cfg["lon"])
        except Exception as e:
            print(f"[{city}] forecast fetch crashed: {e} — skip")
            continue
        if not highs:
            print(f"[{city}] no NWS forecast — skip")
            continue
        try:
            markets = api.get_markets(series_ticker=cfg["series"], status="open")
        except Exception as e:
            print(f"[{city}] get_markets failed: {e} — skip")
            continue
        print(f"[{city}] {len(markets)} open markets; forecast days "
              f"{sorted(highs)[:4]}")
        sigma = SIGMA_F[city]
        for mk in markets:
            tkr = mk["ticker"]
            sub = mk.get("yes_sub_title")
            parsed = parse_bucket(sub)
            if parsed is None:
                print(f"   [{tkr}] unparseable bucket {sub!r} — skip")
                continue
            kind, lo, hi = parsed
            day_iso = _suffix_to_iso(ticker_day(tkr))
            if skip_same_day and day_iso is not None and day_iso <= local_today:
                continue   # same-local-day (or past) market — contaminated test
            fhigh = highs.get(day_iso)
            if fhigh is None:
                # no forecast for this market's day (too far out) — skip quietly
                continue
            mean = fhigh - BIAS_F.get(city, 0.0)   # bias-correct toward observed
            model_p = bucket_prob(lo, hi, mean, sigma)
            try:
                ob = api.get_orderbook(tkr, depth=1)
            except Exception as e:
                print(f"   [{tkr}] orderbook failed: {e} — skip")
                continue
            mid, spread, by, bn = book_mid_spread(ob)
            rows.append(dict(
                city=city, station=cfg["station"], day=day_iso, ticker=tkr,
                bucket=sub, kind=kind, lo=lo, hi=hi,
                forecast_high=round(fhigh, 1), model_mean=round(mean, 1), sigma=sigma,
                model_prob=round(model_p, 4),
                market_mid=None if mid is None else round(mid, 4),
                spread=None if spread is None else round(spread, 4),
                yes_bid=by, no_bid=bn,
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows built — every city failed.")
        return df
    df["disagreement"] = (df["model_prob"] - df["market_mid"]).abs()
    # edge beyond spread: how far the model is from market, net of the half-spread
    # you'd pay to take. Conservative: require the FULL spread to be crossed.
    df["edge_beyond_spread"] = df["disagreement"] - df["spread"].fillna(1.0)
    df["actionable"] = df["edge_beyond_spread"] > 0
    df = df.sort_values("edge_beyond_spread", ascending=False).reset_index(drop=True)
    return df


def _api():
    from kalshi_api import KalshiAPI
    return KalshiAPI()


def main():
    print(f"sigma source: {SIGMA_SOURCE}")
    df = build()
    if df.empty:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = CACHE_DIR / f"live_compare_{stamp}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")

    show = ["city", "day", "bucket", "forecast_high", "model_prob",
            "market_mid", "spread", "disagreement", "edge_beyond_spread"]
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("\n=== Ranked by edge-beyond-spread (taker view) ===")
    print(df[show].to_string(index=False))
    n_act = int(df["actionable"].sum())
    print(f"\nactionable buckets (edge>spread): {n_act} / {len(df)}")


if __name__ == "__main__":
    main()
