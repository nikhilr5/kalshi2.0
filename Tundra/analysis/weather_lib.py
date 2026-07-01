"""Shared weather-model helpers: NWS forecast fetch, Kalshi bucket parsing,
and the Normal-CDF probability model. Every external call is retry-once-then-skip;
nothing here raises on a bad city/source — callers get None / empty and a logged warning.
"""

import re
import sys
import time
import math
from pathlib import Path

import requests
from scipy.stats import norm

# repo root is analysis/Aston/weather -> up 3 = Kalshi2.0; Aston app lives there.
ASTON = str(Path(__file__).resolve().parents[3] / "Aston")
if ASTON not in sys.path:
    sys.path.insert(0, ASTON)

UA = {"User-Agent": "kalshi-weather-research (nikhil.richard84@gmail.com)"}
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# city -> (Kalshi series, forecast gridpoint lat/lon, settlement station, station name)
# Gridpoints are the official NWS climate sites Kalshi settles each market on.
# Station IDs are the ASOS the high is read from. Flagged-unsure noted in STATION_NOTES.
CITIES = {
    "NYC":    dict(series="KXHIGHNY",   lat=40.78, lon=-73.97, station="KNYC",  station_name="Central Park, NY"),
    "LAX":    dict(series="KXHIGHLAX",  lat=33.94, lon=-118.41, station="KLAX",  station_name="LA Intl"),
    "OKC":    dict(series="KXHIGHTOKC", lat=35.39, lon=-97.60, station="KOKC",  station_name="Will Rogers, OKC"),
    "BOS":    dict(series="KXHIGHTBOS", lat=42.36, lon=-71.01, station="KBOS",  station_name="Boston Logan"),
    "DAL":    dict(series="KXHIGHTDAL", lat=32.85, lon=-96.85, station="KDAL",  station_name="Dallas Love Field"),
}

# Settlement-station confidence. Kalshi NYC settles on KNYC (Central Park) — confirmed.
# Others mapped from the airport the series is named for; flag any not field-verified.
STATION_NOTES = {
    "NYC": "HIGH confidence — Kalshi KXHIGHNY settles on KNYC Central Park.",
    "LAX": "MED — assumed KLAX downtown/airport; Kalshi 'LA' series. Verify before sizing.",
    "OKC": "MED — KOKC Will Rogers assumed.",
    "BOS": "MED — KBOS Logan assumed.",
    "DAL": "MED — KDAL Love Field assumed (NOT KDFW). Dallas series ambiguity — verify.",
}


def _get(url, tries=2, timeout=20):
    """GET with one retry; returns parsed JSON or None (never raises)."""
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i + 1 >= tries:
                print(f"[nws] GET failed {url}: {e}")
                return None
            time.sleep(1.0)
    return None


def nws_forecast_highs(lat, lon):
    """Return {date_iso: high_F} from the NWS deterministic grid forecast.
    maxTemperature.values are °C keyed by an ISO interval; we take the local
    date of the interval start as the forecast day. None on failure."""
    pts = _get(f"https://api.weather.gov/points/{lat},{lon}")
    if not pts:
        return None
    grid_url = pts.get("properties", {}).get("forecastGridData")
    if not grid_url:
        print(f"[nws] no forecastGridData for {lat},{lon}")
        return None
    g = _get(grid_url)
    if not g:
        return None
    vals = g.get("properties", {}).get("maxTemperature", {}).get("values", [])
    out = {}
    for v in vals:
        c = v.get("value")
        vt = v.get("validTime", "")
        if c is None or not vt:
            continue
        # validTime is 'ISO/PnDTnH'. A day's TRUE daily-high entry is the long
        # daytime interval; NWS also emits short overnight slivers (e.g.
        # 00:00Z/PT2H) that carry the PREVIOUS afternoon's heat and would
        # corrupt the high. Keep only intervals that start in the CONUS
        # daytime band (10:00-20:00 UTC) AND span >=6h. This drops the
        # spurious slivers. (Caught live: OKC 00:00Z/PT2H = 98F sliver vs the
        # real 13:00Z/PT13H = 82F daily high.)
        start = vt.split("/")[0]
        try:
            hh = int(start[11:13])
        except Exception:
            continue
        dur_h = _iso_dur_hours(vt)
        if not (10 <= hh <= 20 and dur_h >= 6):
            continue
        day = start.split("T")[0]
        f = c * 9 / 5 + 32
        out[day] = max(out.get(day, -999), f)
    return out or None


def _iso_dur_hours(valid_time):
    """Hours in an NWS validTime '<iso>/P[nD]T[nH]' duration. 0 if unparseable."""
    dur = valid_time.split("/")[-1]
    m = re.search(r"P(?:(\d+)D)?T?(?:(\d+)H)?", dur)
    if not m:
        return 0
    days = int(m.group(1) or 0)
    hrs = int(m.group(2) or 0)
    return days * 24 + hrs


_SUB_RANGE = re.compile(r"^(\d+)°?\s*to\s*(\d+)°?$", re.I)
_SUB_ABOVE = re.compile(r"^(\d+)°?\s*or above$", re.I)
_SUB_BELOW = re.compile(r"^(\d+)°?\s*or below$", re.I)


def parse_bucket(sub_title):
    """Parse Kalshi yes_sub_title into (kind, lo_F, hi_F) on the CONTINUOUS scale,
    where buckets cover integer observed highs and we widen by 0.5° each side so
    the segments tile the real line.

      '83° to 84°'   -> ('range', 82.5, 84.5)
      '89° or above' -> ('above', 88.5, +inf)
      '80° or below' -> ('below', -inf, 80.5)
    Returns None if unparseable.
    """
    s = (sub_title or "").strip()
    m = _SUB_RANGE.match(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return ("range", a - 0.5, b + 0.5)
    m = _SUB_ABOVE.match(s)
    if m:
        return ("above", int(m.group(1)) - 0.5, math.inf)
    m = _SUB_BELOW.match(s)
    if m:
        return ("below", -math.inf, int(m.group(1)) + 0.5)
    return None


def bucket_prob(lo, hi, mean, sigma):
    """P(observed high in [lo, hi]) under Normal(mean, sigma)."""
    plo = 0.0 if lo == -math.inf else norm.cdf(lo, mean, sigma)
    phi = 1.0 if hi == math.inf else norm.cdf(hi, mean, sigma)
    return float(phi - plo)


def floored_bucket_prob(lo, hi, mean, sigma, running_max=None):
    """P(high in [lo, hi]) under Normal(mean, sigma) with the running-max FLOOR:
    the day's high cannot be below what's already been observed, so we condition
    on high >= running_max (left-truncate at running_max and renormalize).

    running_max=None -> no floor (same as bucket_prob). Use mean = the raw
    forecast here, NOT max(running_max, forecast): the floor is applied by this
    truncation, so flooring the mean too would double-count it."""
    if running_max is None:
        return bucket_prob(lo, hi, mean, sigma)
    lo = max(lo, running_max)                       # dead mass below the floor
    if lo >= hi:                                     # bucket entirely below floor
        return 0.0
    denom = 1.0 - norm.cdf(running_max, mean, sigma)  # P(high >= floor)
    if denom <= 1e-9:                               # floor already past the forecast
        return 0.0
    return bucket_prob(lo, hi, mean, sigma) / denom


def price_buckets(buckets, mean, sigma, running_max=None):
    """Floor-adjusted fair prices for a set of buckets.
    buckets = list of (lo, hi) on the continuous scale (from parse_bucket).
    Returns a list of probs that sum to ~1 across buckets tiling [floor, inf)."""
    return [floored_bucket_prob(lo, hi, mean, sigma, running_max) for lo, hi in buckets]


def book_mid_spread(ob):
    """From get_orderbook(depth=1) dict -> (yes_mid, spread, best_yes_bid, best_no_bid).
    yes_mid = (best_yes_bid + (1 - best_no_bid))/2; spread = (1-best_no_bid) - best_yes_bid.
    Returns (None, None, by, bn) if a side is missing."""
    yes = ob.get("yes") or []
    no = ob.get("no") or []
    by = yes[0][0] if yes else None
    bn = no[0][0] if no else None
    if by is None and bn is None:
        return (None, None, by, bn)
    yes_ask = (1 - bn) if bn is not None else None   # best ask to BUY yes = 1 - best no bid
    yes_bid = by                                      # best bid to SELL yes
    if yes_bid is None or yes_ask is None:
        # one-sided book: mid is the single quote, spread undefined
        single = yes_bid if yes_bid is not None else yes_ask
        return (single, None, by, bn)
    mid = (yes_bid + yes_ask) / 2
    spread = yes_ask - yes_bid
    return (mid, spread, by, bn)


def ticker_day(ticker):
    """KXHIGHNY-26JUN19-T88 -> '26JUN19' (the forecast/event day suffix)."""
    parts = ticker.split("-")
    return parts[1] if len(parts) >= 2 else None
