"""Sigma lookups for the weather model.

  daily_sigma(station, lead, season=None)  -> (sigma, bias)   future-day trades
  intraday_sigma(station, local_hour)      -> (sigma, bias)   settlement-day decay
  settlement_day_params(station, local_hour, forecast, high_so_far)
                                           -> (mean, sigma)    ready for bucket_prob

Tables come from the historical builds:
  cache/sigma_model.json    (daily, by station x lead x season)
  cache/intraday_sigma.csv  (intraday, by station x local hour)
The intraday curve is linearly interpolated between measured hours and flat-held
outside the measured 5..21 local-hour band.
"""
import json
from pathlib import Path
from bisect import bisect_left

import pandas as pd

_HERE = Path(__file__).resolve().parent
_DAILY = json.load(open(_HERE / "cache" / "sigma_model.json"))["stations"]

_it = pd.read_csv(_HERE / "cache" / "intraday_sigma.csv")
# {station: ([hours], [sigmas], [biases])} sorted by hour
_INTRA = {}
for st, g in _it.sort_values("hour").groupby("station"):
    _INTRA[st] = (g["hour"].tolist(), g["sigma_F"].tolist(), g["bias_F"].tolist())


def daily_sigma(station, lead, season=None):
    s = _DAILY.get(station)
    if not s:
        return None
    lead = str(min(max(int(lead), 0), 7))
    if season:
        byls = s["by_lead_season"].get(lead, {})
        if season in byls and byls[season].get("sigma"):
            return byls[season]["sigma"], byls[season]["bias"]
    bl = s["by_lead"].get(lead)
    return (bl["sigma"], bl["bias"]) if bl else None


def _interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    if xs[i] == x:
        return ys[i]
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def intraday_sigma(station, local_hour):
    """(sigma, bias) at a fractional local hour on the settlement day."""
    t = _INTRA.get(station)
    if not t:
        return None
    hours, sig, bias = t
    return _interp(hours, sig, local_hour), _interp(hours, bias, local_hour)


def settlement_day_params(station, local_hour, forecast, high_so_far=None):
    """Ready-to-use (mean, sigma) for the settlement day. Applies the running-max
    floor and the hour-matched bias. Pass high_so_far = max temp observed today."""
    sb = intraday_sigma(station, local_hour)
    if not sb:
        return None
    sigma, bias = sb
    anchor = forecast if high_so_far is None else max(high_so_far, forecast)
    return anchor - bias, sigma
