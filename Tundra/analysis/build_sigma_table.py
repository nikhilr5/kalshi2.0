"""Historical forecast-error sigma table for the weather model.

For each settlement station, pull a multi-year MOS forecast archive (IEM) and
the matching ASOS observed daily highs (IEM), align them by local calendar day,
and fit forecast-error bias + sigma by station x model x season x lead.

  error  = forecast_high - observed_high   (per local day)
  sigma  = std(error)   (the model's distribution width)
  bias   = mean(error)  (systematic warm/cold offset to subtract)

Forecast high  = max of hourly `tmp` over the local calendar day, taken from the
                 12Z model run only (one independent forecast per issue-day).
Observed high  = IEM ASOS daily max_temp_f  (METAR-derived; ~1F vs CLI settlement
                 truth -- fine for sigma, noted in the report).

Raw pulls are cached under cache/sigma_raw/ so re-runs are cheap. No args; run:
    python3 build_sigma_table.py
"""

import io
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "cache" / "sigma_raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "kalshi-weather-research (nikhil.richard84@gmail.com)"}

# station -> (ASOS id, ASOS network, IANA tz). The five modeled settlement sites.
STATIONS = {
    "KNYC": dict(asos="NYC", net="NY_ASOS", tz="America/New_York"),
    "KLAX": dict(asos="LAX", net="CA_ASOS", tz="America/Los_Angeles"),
    "KOKC": dict(asos="OKC", net="OK_ASOS", tz="America/Chicago"),
    "KBOS": dict(asos="BOS", net="MA_ASOS", tz="America/New_York"),
    "KDAL": dict(asos="DAL", net="TX_ASOS", tz="America/Chicago"),
}

# MOS models -> the daily issue cycle (UTC) to sample (one run/day => independent
# daily samples). NBS=NBM short (live proxy, 01/07/13/19Z, leads 0-3); NBE=NBM
# extended (live proxy long range, 00/12Z, leads to ~7); GFS=GFS-MOS short (MAV,
# 00/06/12/18Z) and MEX=GFS extended MOS (00/12Z) = independent cross-check.
MODELS = {"NBS": 13, "NBE": 12, "GFS": 12, "MEX": 12}

# Two-year window for seasonal + lead coverage.
START = "2024-06-19"
END = "2026-06-19"

MAX_LEAD = 7                 # days
# A local day's daily high lives in any of these columns depending on model:
# tmp = hourly temp (NBS/GFS), n_x / txn = the MOS max/min line (MEX/NBE extended).
# Taking the per-day MAX across all three recovers the true daytime high (the
# overnight-min entries are smaller and never win).
HIGH_COLS = ["tmp", "n_x", "txn"]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def _get_text(url, tries=3, timeout=240):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if i + 1 >= tries:
                print(f"  ! GET failed {url[:90]}...: {e}")
                return None
            time.sleep(3.0)
    return None


def fetch_mos(station, model):
    """Bulk MOS CSV for station/model over [START,END], cached. -> DataFrame or None."""
    cache = RAW / f"mos_{station}_{model}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        return pd.read_csv(cache, dtype={"tmp": "float"}, low_memory=False)
    url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?"
           f"station={station}&model={model}"
           f"&sts={START}T00:00:00Z&ets={END}T00:00:00Z&format=csv")
    print(f"  fetching MOS {station}/{model} ...", flush=True)
    txt = _get_text(url)
    if not txt or "runtime" not in txt[:200]:
        print(f"  ! no MOS data {station}/{model}")
        return None
    cache.write_text(txt)
    return pd.read_csv(io.StringIO(txt), dtype={"tmp": "float"}, low_memory=False)


def fetch_obs(station):
    """ASOS daily max_temp_f over [START,END], cached. -> dict {local_date: high_F}."""
    cache = RAW / f"obs_{station}.csv"
    s = STATIONS[station]
    if cache.exists() and cache.stat().st_size > 100:
        txt = cache.read_text()
    else:
        sy, sm, sd = START.split("-")
        ey, em, ed = END.split("-")
        url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?"
               f"network={s['net']}&stations={s['asos']}"
               f"&year1={sy}&month1={sm}&day1={sd}"
               f"&year2={ey}&month2={em}&day2={ed}"
               f"&var=max_temp_f&format=onlycomma")
        print(f"  fetching OBS {station} ...", flush=True)
        txt = _get_text(url)
        if not txt:
            return {}
        cache.write_text(txt)
    df = pd.read_csv(io.StringIO(txt))
    df = df[pd.to_numeric(df["max_temp_f"], errors="coerce").notna()]
    return {d: float(v) for d, v in zip(df["day"], df["max_temp_f"])}


def forecast_highs(mos, tz, issue_hour):
    """From a MOS CSV df -> rows (run_date, target_date, lead_days, fc_high).
    Daily high = max over the local calendar day of any temp/max-line column.
    Keep only the issue_hour run/day, and only target days with afternoon
    coverage (an ftime at local hour 15-23) so truncated horizon-edge days
    (which see only cold morning hours -> spurious cold bias) are dropped."""
    z = ZoneInfo(tz)
    rt = pd.to_datetime(mos["runtime"], utc=True, errors="coerce")
    ft = pd.to_datetime(mos["ftime"], utc=True, errors="coerce")
    high = None
    for c in HIGH_COLS:
        if c in mos.columns:
            v = pd.to_numeric(mos[c], errors="coerce")
            high = v if high is None else np.fmax(high, v)
    m = pd.DataFrame({"rt": rt, "ft": ft, "high": high})
    m = m.dropna(subset=["rt", "ft", "high"])
    m = m[m["rt"].dt.hour == issue_hour]
    if m.empty:
        return pd.DataFrame()
    loc = m["ft"].dt.tz_convert(z)
    m["target_date"] = loc.dt.date
    m["loc_hour"] = loc.dt.hour
    m["run_date"] = m["rt"].dt.tz_convert(z).dt.date
    m["run_id"] = m["rt"]
    grp = m.groupby(["run_id", "run_date", "target_date"])
    g = grp.agg(fc_high=("high", "max"), max_loc_hour=("loc_hour", "max"),
                has_pm=("loc_hour", lambda h: ((h >= 15) & (h <= 23)).any())).reset_index()
    g = g[g["has_pm"]]                       # drop truncated (morning-only) days
    g["lead_days"] = (pd.to_datetime(g["target_date"]) - pd.to_datetime(g["run_date"])).dt.days
    g = g[(g["lead_days"] >= 0) & (g["lead_days"] <= MAX_LEAD)]
    return g[["run_date", "target_date", "lead_days", "fc_high"]]


def build():
    rows = []
    for station in STATIONS:
        tz = STATIONS[station]["tz"]
        obs = fetch_obs(station)
        if not obs:
            print(f"  ! no OBS for {station}, skipping")
            continue
        for model, issue_hour in MODELS.items():
            mos = fetch_mos(station, model)
            if mos is None or mos.empty:
                continue
            fh = forecast_highs(mos, tz, issue_hour)
            if fh.empty:
                print(f"  ! no 12Z forecasts {station}/{model}")
                continue
            fh["obs_high"] = fh["target_date"].astype(str).map(obs)
            fh = fh.dropna(subset=["obs_high"])
            fh["error"] = fh["fc_high"] - fh["obs_high"]
            fh["station"] = station
            fh["model"] = model
            fh["season"] = pd.to_datetime(fh["target_date"]).dt.month.map(SEASONS)
            rows.append(fh)
            print(f"  {station}/{model}: {len(fh)} matched day-forecasts")
            time.sleep(1.0)
    if not rows:
        print("NO DATA — aborting")
        return None
    err = pd.concat(rows, ignore_index=True)
    err = err[["station", "model", "season", "lead_days", "run_date",
               "target_date", "fc_high", "obs_high", "error"]]
    err.to_csv(HERE / "cache" / "sigma_history_errors.csv", index=False)
    return err


def aggregate(err):
    def agg(g):
        e = g["error"].to_numpy()
        return pd.Series({"n": len(e), "bias_F": round(float(e.mean()), 2),
                          "sigma_F": round(float(e.std(ddof=1)), 2) if len(e) > 1 else np.nan,
                          "rmse_F": round(float(np.sqrt((e ** 2).mean())), 2),
                          "mae_F": round(float(np.abs(e).mean()), 2)})
    full = err.groupby(["station", "model", "season", "lead_days"]).apply(agg, include_groups=False).reset_index()
    full.to_csv(HERE / "cache" / "sigma_table.csv", index=False)
    # headline: NBS (live proxy), pooled across seasons, by station x lead
    nbs = err[err["model"] == "NBS"]
    head = nbs.groupby(["station", "lead_days"]).apply(agg, include_groups=False).reset_index()
    head.to_csv(HERE / "cache" / "sigma_recommended_nbs.csv", index=False)
    return full, head


if __name__ == "__main__":
    print("=== building historical sigma table ===")
    err = build()
    if err is None:
        sys.exit(1)
    full, head = aggregate(err)
    print(f"\nTotal matched day-forecasts: {len(err)}")
    print("\n=== NBS (live proxy) sigma by station x lead, pooled seasons ===")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(head.to_string(index=False))
    print("\nWrote: cache/sigma_history_errors.csv, cache/sigma_table.csv, "
          "cache/sigma_recommended_nbs.csv")
