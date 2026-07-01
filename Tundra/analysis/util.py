"""Tundra weather analysis helpers.

intraday_peak_frame() -> the time-aware training frame for the settlement-day
sigma model: one row per (location, day, prediction-time), pairing the forecast
*as of that moment* (reconstructed from the timestamped MOS run archive, so it
reflects intraday forecast updates) with that day's realized peak.

Reuses the raw caches the sigma builds already pulled:
  cache/intraday_raw/asos_<ICAO>.csv   hourly observed temps
  cache/sigma_raw/mos_<ICAO>_<model>.csv  archived MOS runs
"""
import sys
from pathlib import Path
from datetime import timedelta
from bisect import bisect_right

import numpy as np
import pandas as pd

from build_sigma_table import STATIONS, HIGH_COLS, fetch_mos, fetch_obs  # noqa
from build_intraday_sigma import fetch_hourly

HERE = Path(__file__).resolve().parent
_ET = "America/New_York"        # all swing/obs times reported in Eastern
_ASTON = str(Path(__file__).resolve().parents[2] / "Aston")
if _ASTON not in sys.path:
    sys.path.insert(0, _ASTON)


def _kalshi_api():
    from kalshi_api import KalshiAPI
    return KalshiAPI()
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def _run_forecasts(mos, tz):
    """{target_date_str: sorted [(run_local_ts, fc_high)]} from every MOS run.
    Daily high per (run, target local day) = max of temp/max-min cols, keeping
    only runs that SPAN the peak for that day -- ftimes both before noon AND in
    the afternoon. This drops runs truncated at either end: a late-issued run
    (no morning, only declining evening temps -> too cold) and a horizon-edge
    run (no afternoon, only cool morning -> too cold)."""
    rt = pd.to_datetime(mos["runtime"], utc=True, errors="coerce")
    ft = pd.to_datetime(mos["ftime"], utc=True, errors="coerce")
    high = None
    for c in HIGH_COLS:
        if c in mos.columns:
            v = pd.to_numeric(mos[c], errors="coerce")
            high = v if high is None else np.fmax(high, v)
    loc = ft.dt.tz_convert(tz)
    d = pd.DataFrame({"run": rt.dt.tz_convert(tz), "tdate": loc.dt.date,
                      "lhour": loc.dt.hour, "high": high}).dropna(subset=["run", "tdate", "high"])
    spans = lambda h: (h <= 12).any() and ((h >= 15) & (h <= 23)).any()  # morning AND afternoon
    g = d.groupby(["run", "tdate"]).agg(fc_high=("high", "max"),
                                        ok=("lhour", spans)).reset_index()
    g = g[g["ok"]]
    out = {}
    for td, sub in g.groupby("tdate"):
        out[str(td)] = sorted(zip(sub["run"], sub["fc_high"]))
    return out


def _forecast_asof(runs_for_day, predict_ts):
    """Latest (run, fc_high) with run <= predict_ts -> (fc_high, run_ts) or (nan, None)."""
    if not runs_for_day:
        return np.nan, None
    times = [r for r, _ in runs_for_day]
    i = bisect_right(times, predict_ts)
    if i == 0:
        return np.nan, None
    return runs_for_day[i - 1][1], runs_for_day[i - 1][0]


def intraday_peak_frame(stations=None, model="NBS", grid_hours=range(5, 22),
                        days_out=(0, 1), fetch_missing=True):
    """Time-aware frame. One row per (station, target day, days_out, predict hour).

    days_out = how many days BEFORE the target day the prediction is made:
      0 = same-day (settlement day, the running-max floor is live)
      1 = day-ahead (target day not yet observed, so running_max is NaN, no floor)
    Pass days_out=(0,1,2,...) for a longer horizon.

    Columns:
      location          settlement station ICAO (e.g. KNYC)
      season            DJF/MAM/JJA/SON of the TARGET day
      date              target (settlement) local calendar day
      days_out          days between predict day and target day (0,1,...)
      predict_time      local timestamp the prediction is made
      predict_hour      local hour of predict_time (numeric, 0-23)
      hours_to_peak     predict_time -> when the actual peak occurred (future info)
      hours_to_peak_exp predict_time -> the target day's EXPECTED peak (15:00 local);
                        the unified, leakage-free time feature across days_out
      hours_to_eod      predict_time -> target day's end (local midnight)
      running_max       max temp observed so far at predict_time (NaN if days_out>0)
      forecast_high     forecast's predicted high available as of predict_time
      forecast_age_h    hours since that forecast run was issued (staleness)
      predicted_peak    the model's call = max(running_max, forecast_high)
      actual_peak       the day's realized high
      actual_peak_time  local time the high first occurred
      error             predicted_peak - actual_peak  (the sigma target)
    """
    stations = stations or list(STATIONS)
    rows = []
    for st in stations:
        tz = STATIONS[st]["tz"]
        # observations
        obs = pd.read_csv(HERE / "cache" / "intraday_raw" / f"asos_{st}.csv") \
            if (HERE / "cache" / "intraday_raw" / f"asos_{st}.csv").exists() \
            else (fetch_hourly(st) if fetch_missing else None)
        if obs is None or obs.empty:
            continue
        obs = obs.copy()
        obs["tmpf"] = pd.to_numeric(obs["tmpf"], errors="coerce")
        obs = obs.dropna(subset=["tmpf"])
        lt = pd.to_datetime(obs["valid"], utc=True).dt.tz_convert(tz)
        obs["ltime"], obs["date"] = lt, lt.dt.date
        # forecasts
        mcache = HERE / "cache" / "sigma_raw" / f"mos_{st}_{model}.csv"
        mos = pd.read_csv(mcache, low_memory=False) if mcache.exists() \
            else (fetch_mos(st, model) if fetch_missing else None)
        if mos is None or mos.empty:
            continue
        runs = _run_forecasts(mos, tz)

        for day, gd in obs.groupby("date"):
            gd = gd.sort_values("ltime")
            actual_peak = float(gd["tmpf"].max())
            peak_time = gd.loc[gd["tmpf"].idxmax(), "ltime"]   # first occurrence
            midnight = pd.Timestamp(day, tz=tz)
            eod = midnight + timedelta(days=1)
            exp_peak = midnight + timedelta(hours=15)          # expected ~3pm peak
            season = SEASON[pd.Timestamp(day).month]           # of the TARGET day
            runs_day = runs.get(str(day), [])                  # all runs predicting it
            tt = gd["ltime"].astype("int64").to_numpy()        # UTC epoch ns, tz-safe
            tv = gd["tmpf"].to_numpy()
            for d_out in days_out:
                base = midnight - timedelta(days=int(d_out))   # the predict day
                for h in grid_hours:
                    ptime = base + timedelta(hours=int(h))
                    seen = tv[tt <= ptime.value]               # target-day obs so far
                    run_max = float(seen.max()) if seen.size else np.nan
                    fc, frun = _forecast_asof(runs_day, ptime)
                    if not np.isfinite(fc) and not np.isfinite(run_max):
                        continue
                    pred = np.nanmax([run_max, fc])
                    rows.append(dict(
                        location=st, season=season, date=str(day), days_out=int(d_out),
                        predict_time=ptime, predict_hour=float(h),
                        hours_to_peak=round((peak_time - ptime).total_seconds() / 3600, 2),
                        hours_to_peak_exp=round((exp_peak - ptime).total_seconds() / 3600, 2),
                        hours_to_eod=round((eod - ptime).total_seconds() / 3600, 2),
                        running_max=run_max, forecast_high=fc,
                        forecast_age_h=round((ptime - frun).total_seconds() / 3600, 2) if frun is not None else np.nan,
                        predicted_peak=pred, actual_peak=actual_peak,
                        actual_peak_time=peak_time, error=round(pred - actual_peak, 2)))
    return pd.DataFrame(rows)


def load_trades_days(start, end, series="KXHIGHNY", kalshi_api=None, cache=True):
    """All trades for a series' buckets whose event-day is in [start,end].
    start/end are 'YYYY-MM-DD'. Returns df[ticker, event_day, ts, yes_price,
    count, taker_side]. Cached per event-day under cache/trades/ (parquet)."""
    api = kalshi_api or _kalshi_api()
    cdir = HERE / "cache" / "trades"
    cdir.mkdir(parents=True, exist_ok=True)
    # one-time market enumeration across statuses -> event_day -> tickers
    by_day = {}
    for status in ("settled", "closed", "open"):
        try:
            for m in api.get_markets(series_ticker=series, status=status):
                tk = m.get("ticker")
                if tk and "-" in tk:
                    by_day.setdefault(tk.split("-")[1], set()).add(tk)
        except Exception as e:
            print(f"[trades] get_markets {status} failed: {e}")
    frames = []
    for d in pd.date_range(start, end, freq="D"):
        ev_day = d.strftime("%y%b%d").upper()
        cf = cdir / f"{series}_{ev_day}.parquet"
        if cache and cf.exists():
            frames.append(pd.read_parquet(cf))
            continue
        rows = []
        for tk in sorted(by_day.get(ev_day, [])):
            try:
                for t in api.get_trades(tk, limit=100000):
                    rows.append((tk, ev_day, t.get("created_time"),
                                 float(t.get("yes_price_dollars") or 0),
                                 float(t.get("count_fp") or 0),
                                 t.get("taker_side")))
            except Exception as e:
                print(f"[trades] {tk} failed: {e}")
        df = pd.DataFrame(rows, columns=["ticker", "event_day", "ts",
                                         "yes_price", "count", "taker_side"])
        if cache:
            df.to_parquet(cf)
        print(f"  {ev_day}: {len(df)} trades, {df.ticker.nunique() if len(df) else 0} buckets")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _smoothed_px(g, smooth_min):
    """trades for one bucket -> smooth_min-minute trailing VWAP on a 1-min grid."""
    s = g.sort_values("ts").set_index("ts")
    val = (s["yes_price"] * s["count"]).resample("1min").sum()
    cnt = s["count"].resample("1min").sum()
    px = (val / cnt).ffill()                         # per-minute VWAP, gaps carried
    return px.rolling(f"{smooth_min}min").mean().dropna()


def _find_swings(px, move, window_min, hold_move, hold_min):
    """Swing = smoothed price moves >= `move` within `window_min` AND then holds
    within `hold_move` for `hold_min`. Returns list of dicts (onset/peak/magnitude)."""
    v, idx = px.to_numpy(), px.index
    n, i, out = len(v), 0, []
    while i < n - 1:
        hit = next((j for j in range(i + 1, min(n, i + window_min + 1))
                    if abs(v[j] - v[i]) >= move), None)
        if hit is None:
            i += 1
            continue
        h_end = min(n, hit + hold_min + 1)
        held = (h_end - hit) >= max(2, int(hold_min * 0.7)) and \
            all(abs(v[k] - v[hit]) <= hold_move for k in range(hit, h_end))
        if held:
            out.append(dict(onset=idx[i], peak=idx[hit],
                            start_px=round(float(v[i]), 3), end_px=round(float(v[hit]), 3),
                            magnitude=round(float(v[hit] - v[i]), 3),
                            move_mins=hit - i))
            i = hit                                  # advance past the move
        else:
            i += 1
    return out


def _implied_high_series(day_tr, smooth_min, min_buckets=3):
    """Market's implied expected high (deg F) over time for one event-day:
    expected_high(t) = sum(bucket_level * price) / sum(price) across the RANGE (B)
    buckets, each priced by its smoothed VWAP. NaN where < min_buckets are priced."""
    cols = {}
    for tk, g in day_tr.groupby("ticker"):
        lvl = _bucket_strike(tk)
        if not np.isfinite(lvl) or len(g) < 3:
            continue
        px = _smoothed_px(g, smooth_min)
        if len(px):
            cols[lvl] = px
    if len(cols) < min_buckets:
        return pd.Series(dtype=float)
    wide = pd.DataFrame(cols).resample("1min").mean().ffill()    # bucket prices aligned
    lvl = np.array(wide.columns, dtype=float)
    pr = wide.to_numpy()
    npr = np.where(np.isnan(pr), 0.0, pr)
    den = npr.sum(axis=1)
    eh = (npr * lvl).sum(axis=1) / np.where(den > 0, den, np.nan)
    eh = pd.Series(eh, index=wide.index)
    eh[(~np.isnan(pr)).sum(axis=1) < min_buckets] = np.nan       # need enough live buckets
    return eh.dropna()


def market_swings(start, end, series="KXHIGHNY", kalshi_api=None, cache=True,
                  move=1.0, window_min=120, hold_move=0.6, hold_min=20, smooth_min=10):
    """Detect swings in the market's IMPLIED EXPECTED HIGH (deg F), collapsed across
    all range (B) buckets -- NOT a single ticker.

    A swing = the smoothed implied expected high moves >= `move` (default 1.0 F)
    within `window_min` (default 120 min) and then HOLDS within `hold_move`
    (default 0.6 F) for `hold_min` (default 20 min). All tunable. NOTE: the
    expected high is much smoother than a single bucket's price -- it drifts
    gradually -- so use a lower threshold over a WIDER window than for bucket-level.

    Returns df (one row per swing event): event_day, onset, peak, start_high,
    end_high, magnitude (deg F), direction, move_mins -- sorted by onset.
    """
    tr = load_trades_days(start, end, series, kalshi_api, cache)
    if tr.empty:
        return pd.DataFrame()
    tr = tr[tr["ticker"].str.contains("-B")]                     # range buckets only
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601").dt.tz_convert(_ET)
    rows = []
    for ed, day_tr in tr.groupby("event_day"):
        eh = _implied_high_series(day_tr, smooth_min)
        if len(eh) < window_min + hold_min:
            continue
        for sw in _find_swings(eh, move, window_min, hold_move, hold_min):
            rows.append(dict(event_day=ed, onset=sw["onset"], peak=sw["peak"],
                             start_high=sw["start_px"], end_high=sw["end_px"],
                             magnitude=sw["magnitude"], move_mins=sw["move_mins"],
                             direction="up" if sw["magnitude"] > 0 else "down"))
    cols = ["event_day", "onset", "peak", "start_high", "end_high",
            "magnitude", "direction", "move_mins"]
    return (pd.DataFrame(rows)[cols].sort_values("onset").reset_index(drop=True)
            if rows else pd.DataFrame(columns=cols))


# cloud cover -> numeric fraction (oktas-ish), so "increase" is measurable
_CLOUD = {"CLR": 0.0, "SKC": 0.0, "FEW": 0.1, "SCT": 0.4, "BKN": 0.75, "OVC": 1.0}


def obs_history(start, end, asos="NYC", tries=5):
    """Historical ASOS obs over [start,end] (YYYY-MM-DD), UTC. Returns df[ts, tmpf,
    skyc1, cloud (0-1), skyl1, dwpf, sknt, drct, wxcodes]. Retries on IEM 429s."""
    import io
    import time
    import requests
    sy, sm, sd = start.split("-")
    ey, em, ed = end.split("-")
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
           f"station={asos}&data=tmpf&data=skyc1&data=skyl1&data=dwpf&data=sknt"
           "&data=drct&data=wxcodes&tz=Etc/UTC&format=onlycomma&missing=M&trace=T"
           f"&year1={sy}&month1={int(sm)}&day1={int(sd)}"
           f"&year2={ey}&month2={int(em)}&day2={int(ed)}")
    txt = ""
    for i in range(tries):
        txt = requests.get(url, headers={"User-Agent": "kalshi-weather"}, timeout=120).text
        if "valid" in txt[:50]:
            break
        time.sleep(4 * (i + 1))                          # back off on 429 / slow-down
    if "valid" not in txt[:50]:
        raise RuntimeError(f"obs_history failed for {asos} {start}..{end}: {txt[:80]}")
    df = pd.read_csv(io.StringIO(txt))
    df["ts"] = pd.to_datetime(df["valid"], utc=True, errors="coerce").dt.tz_convert(_ET)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df["cloud"] = df["skyc1"].map(_CLOUD)
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def swings_with_features(swings, asos="NYC", lookback_hours=3):
    """For each expected-high swing onset, attach the ASOS state in
    [onset-lookback, onset]:
      cloud_at      cloud fraction at the last ob before onset (0=CLR .. 1=OVC)
      cloud_chg     change in cloud fraction across the window (+ = clouding up)
      cloud_jump    clouds INCREASED >=0.5 over the window (directional)
      run_max       highest temp observed in the window
      n_obs         obs available in the window
    Extend with forecast / wind / dewpoint features as needed."""
    if swings.empty:
        return swings
    onsets = pd.to_datetime(swings["onset"], utc=True)
    lo = (onsets.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    hi = (onsets.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    obs = obs_history(lo, hi, asos)
    feats = []
    for r in swings.reset_index(drop=True).itertuples():
        on = pd.Timestamp(r.onset)
        if on.tzinfo is None:
            on = on.tz_localize(_ET)
        win = obs[(obs["ts"] >= on - pd.Timedelta(hours=lookback_hours)) & (obs["ts"] <= on)]
        cl = win["cloud"].dropna()
        dw = pd.to_numeric(win.get("dwpf"), errors="coerce").dropna() if "dwpf" in win else pd.Series(dtype=float)
        wd = pd.to_numeric(win.get("drct"), errors="coerce").dropna() if "drct" in win else pd.Series(dtype=float)
        feats.append(dict(
            cloud_at=round(float(cl.iloc[-1]), 2) if len(cl) else np.nan,
            cloud_chg=round(float(cl.iloc[-1] - cl.iloc[0]), 2) if len(cl) else np.nan,
            cloud_jump=bool(len(cl) and (cl.iloc[-1] - cl.iloc[0]) >= 0.5),
            run_max=float(win["tmpf"].max()) if win["tmpf"].notna().any() else np.nan,
            dwpf_at=float(dw.iloc[-1]) if len(dw) else np.nan,
            wind_dir=float(wd.iloc[-1]) if len(wd) else np.nan,
            n_obs=len(win)))
    return pd.concat([swings.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def obs_1min(start, end, asos="NYC", cache=True, tries=5):
    """1-MINUTE ASOS temps over [start,end] (YYYY-MM-DD) -> df[ts(ET), tmpf].
    IEM's 1-min archive lags ~2 days, so this is for BACKTESTING at the resolution
    a live 1-min feed would give (catches between-hour crosses the :51 METAR misses).
    Cached to cache/asos1min_*.parquet."""
    import io
    import time
    import requests
    cf = HERE / "cache" / f"asos1min_{asos}_{start}_{end}.parquet"
    if cache and cf.exists() and cf.stat().st_size > 100:
        return pd.read_parquet(cf)
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?"
           f"station={asos}&vars=tmpf&sts={start}T00:00Z&ets={end}T23:59Z"
           "&sample=1min&what=download&tz=UTC&gis=no")
    txt = ""
    for i in range(tries):
        txt = requests.get(url, headers={"User-Agent": "kalshi-weather"}, timeout=240).text
        if "valid" in txt[:60]:
            break
        time.sleep(4 * (i + 1))
    if "valid" not in txt[:60]:
        raise RuntimeError(f"obs_1min failed {asos} {start}..{end}: {txt[:80]}")
    df = pd.read_csv(io.StringIO(txt))
    vcol = [c for c in df.columns if "valid" in c.lower()][0]
    df["ts"] = pd.to_datetime(df[vcol], utc=True, errors="coerce").dt.tz_convert(_ET)
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    df = df.dropna(subset=["ts", "tmpf"])[["ts", "tmpf"]].sort_values("ts").reset_index(drop=True)
    if cache:
        df.to_parquet(cf)
    return df


def _bucket_strike(ticker):
    """Bucket temperature level from a KXHIGH ticker, e.g. ...-B76.5 -> 76.5, ...-T80 -> 80."""
    import re
    m = re.search(r"-[BT](\d+\.?\d*)$", ticker or "")
    return float(m.group(1)) if m else np.nan


def base_rate_summary(swf, signal="cloud_jump"):
    """Confusion matrix + precision / recall / base-rate / Fisher's p for a boolean
    `signal` predicting a DOWN swing in the expected high. signal=True is treated
    as 'predict down'; the outcome is direction=='down'. Returns a dict and prints.

    The number that matters: precision = P(down | signal) vs the base rate P(down).
    A signal only helps if precision is meaningfully ABOVE the base rate."""
    from scipy.stats import fisher_exact
    d = swf.dropna(subset=[signal]).copy()
    d["down"] = d["direction"] == "down"
    sig = d[signal].astype(bool)
    TP = int((sig & d["down"]).sum())
    FP = int((sig & ~d["down"]).sum())
    FN = int((~sig & d["down"]).sum())
    TN = int((~sig & ~d["down"]).sum())
    n, base = len(d), d["down"].mean()
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    _, p = fisher_exact([[TP, FP], [FN, TN]])
    print(f"n={n} swing events | base rate P(down)={base:.0%}")
    print(f"confusion (signal = {signal}):")
    print(f"            down   up")
    print(f"  {signal}=T   {TP:4d} {FP:4d}")
    print(f"  {signal}=F   {FN:4d} {TN:4d}")
    print(f"precision P(down | {signal}) = {prec:.0%}   "
          f"(base {base:.0%}, lift {prec / base:.2f}x)" if base else "")
    print(f"recall    P({signal} | down) = {rec:.0%}")
    print(f"Fisher exact p = {p:.3f}   ({'significant' if p < 0.05 else 'NOT significant'} at 0.05)")
    return dict(n=n, TP=TP, FP=FP, FN=FN, TN=TN, base_rate=base,
                precision=prec, recall=rec, fisher_p=p)


if __name__ == "__main__":
    df = intraday_peak_frame(stations=["KNYC"])
    print("shape:", df.shape)
    print(df[["location", "date", "predict_hour", "hours_to_peak", "hours_to_eod",
              "running_max", "forecast_high", "predicted_peak", "actual_peak",
              "error"]].head(10).to_string(index=False))
