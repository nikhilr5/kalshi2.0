"""Forward harness — run daily to accrue a REAL model-vs-market backtest.

Each run appends one row per (city, day-ahead bucket) with: forecast high,
model bucket prob, market mid/spread, and — once available — the settled
observed high and the realized bucket outcome. Over weeks this CSV becomes the
only honest validation dataset (there is no historical Kalshi weather price
archive, so the backtest must be built forward).

Run daily (cron / launchd):  python3 log_weather.py
Appends to: weather_log.csv

Two things happen each run:
  1) LOG today's day-ahead snapshot (forecast + model probs + live market).
  2) BACKFILL settled outcomes for prior rows whose observed high is now known
     (pulled from IEM ASOS daily max_temp_f).

Idempotent-ish: a (run_date, ticker) row is logged once per run_date; re-running
the same day overwrites that day's snapshot rather than duplicating.
"""

import io
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weather_lib import (CITIES, UA, nws_forecast_highs, parse_bucket,
                         bucket_prob, book_mid_spread, ticker_day)
from live_model_vs_market import SIGMA_F, BIAS_F, STATION_TZ, _suffix_to_iso

LOG = Path(__file__).resolve().parent / "weather_log.csv"

# IEM network per station (state ASOS network; station id = code without 'K').
IEM_NET = {"KNYC": "NY_ASOS", "KLAX": "CA_ASOS", "KOKC": "OK_ASOS",
           "KBOS": "MA_ASOS", "KDAL": "TX_ASOS"}


def observed_high(station, day_iso):
    """Observed daily max_temp_f for one station+day from IEM, or None."""
    net = IEM_NET.get(station)
    sid = station[1:] if station.startswith("K") else station
    if not net:
        return None
    y, m, d = day_iso.split("-")
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?"
           f"network={net}&stations={sid}&year1={y}&month1={int(m)}&day1={int(d)}"
           f"&year2={y}&month2={int(m)}&day2={int(d)}&format=comma")
    for _ in range(2):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty or "max_temp_f" not in df.columns:
                return None
            v = pd.to_numeric(df["max_temp_f"], errors="coerce").dropna()
            return float(v.iloc[0]) if len(v) else None
        except Exception:
            continue
    return None


def snapshot():
    """Build today's day-ahead log rows across all cities. Never crashes."""
    from kalshi_api import KalshiAPI
    api = KalshiAPI()
    run_date = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for city, cfg in CITIES.items():
        try:
            highs = nws_forecast_highs(cfg["lat"], cfg["lon"])
            if not highs:
                print(f"[{city}] no forecast — skip"); continue
            markets = api.get_markets(series_ticker=cfg["series"], status="open")
        except Exception as e:
            print(f"[{city}] fetch failed: {e} — skip"); continue
        local_today = datetime.now(ZoneInfo(STATION_TZ[city])).date().isoformat()
        sigma = SIGMA_F[city]
        for mk in markets:
            tkr = mk["ticker"]
            parsed = parse_bucket(mk.get("yes_sub_title"))
            if parsed is None:
                continue
            day_iso = _suffix_to_iso(ticker_day(tkr))
            if day_iso is None or day_iso <= local_today:
                continue   # log day-ahead+ only
            kind, lo, hi = parsed
            fhigh = highs.get(day_iso)
            if fhigh is None:
                continue
            mean = fhigh - BIAS_F.get(city, 0.0)
            try:
                mid, spread, by, bn = book_mid_spread(api.get_orderbook(tkr, depth=1))
            except Exception:
                mid = spread = by = bn = None
            rows.append(dict(
                run_date=run_date, city=city, station=cfg["station"], day=day_iso,
                ticker=tkr, bucket=mk.get("yes_sub_title"), kind=kind, lo=lo, hi=hi,
                forecast_high=round(fhigh, 1), model_mean=round(mean, 1), sigma=sigma,
                model_prob=round(bucket_prob(lo, hi, mean, sigma), 4),
                market_mid=mid, spread=spread, yes_bid=by, no_bid=bn,
                observed_high="", outcome="",
            ))
    return pd.DataFrame(rows)


def backfill(df):
    """Fill observed_high + outcome for rows whose day is now in the past and
    not yet scored. Caches per (station, day) so we hit IEM once each."""
    if df.empty:
        return df
    today = datetime.now(timezone.utc).date().isoformat()
    need = df[(df["observed_high"].astype(str).str.len() == 0)
              & (df["day"] < today)]
    cache = {}
    for (station, day), _ in need.groupby(["station", "day"]):
        cache[(station, day)] = observed_high(station, day)
    for i in need.index:
        oh = cache.get((df.at[i, "station"], df.at[i, "day"]))
        if oh is None:
            continue
        df.at[i, "observed_high"] = oh
        lo, hi = float(df.at[i, "lo"]), float(df.at[i, "hi"])
        df.at[i, "outcome"] = int(lo <= round(oh) <= hi) if hi != float("inf") \
            else int(round(oh) >= lo) if lo != float("-inf") else int(round(oh) <= hi)
    return df


def main():
    new = snapshot()
    if LOG.exists():
        old = pd.read_csv(LOG, dtype={"observed_high": str, "outcome": str},
                          keep_default_na=False)
        # drop any existing rows for this run_date+ticker, then append fresh
        if not new.empty:
            key = set(zip(new["run_date"], new["ticker"]))
            old = old[~old.apply(lambda r: (r["run_date"], r["ticker"]) in key, axis=1)]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = backfill(df)
    df.to_csv(LOG, index=False)
    n_scored = int((df["outcome"].astype(str).str.len() > 0).sum())
    print(f"weather_log.csv now {len(df)} rows ({n_scored} scored). "
          f"Logged {len(new)} snapshot rows for {datetime.now(timezone.utc).date()}.")


if __name__ == "__main__":
    main()
