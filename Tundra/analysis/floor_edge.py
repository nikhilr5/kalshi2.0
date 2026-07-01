"""Floor-edge optimizer (the one real signal).

Sell a bucket once a 1-MINUTE temperature reading proves it's dead (running max
exceeds the bucket), filtering false kills (1-min noise / CLI rounding) with a
MARGIN and a SUSTAIN requirement. Validate against actual Kalshi settlements and
compute net EV per contract.

  bucket B<m> covers highs {m-0.5 .. m+0.5} (e.g. B72.5 = 72-73). It's dead once
  the high >= m+1.5.  kill threshold = m + 1.5 + margin.  Require the temp to sit
  >= threshold for `sustain` consecutive minutes (kills 1-tick spikes).

Strategy: at the kill, SELL yes at the median traded price over the next 10 min.
PnL = sell_px - settle(0/1) - fee.   EV>0 net = edge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Aston"))
from util import load_trades_days, obs_1min, _bucket_strike     # noqa: E402
from kalshi_api import KalshiAPI                                  # noqa: E402

START, END = "2026-05-01", "2026-06-20"        # 1-min lags ~2d, so cap END
_FEE = lambda p: 0.07 * p * (1 - p)
_SETTLE_CACHE = {}


def _result(api, tk):
    if tk not in _SETTLE_CACHE:
        try:
            _SETTLE_CACHE[tk] = api.get_market(tk).get("result")
        except Exception:
            _SETTLE_CACHE[tk] = None
    return _SETTLE_CACHE[tk]


def kills(series, asos, margin, sustain, sell_window=10, res_min=1, coarse_c=False,
          latency=0, resamp="last", low=False, lst_off=None, before_win=60):
    tr = load_trades_days(START, END, series)
    tr = tr[tr["ticker"].str.contains("-B")].copy()
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    tr["strike"] = tr["ticker"].map(_bucket_strike)
    obs = obs_1min(START, END, asos=asos)
    if res_min > 1:                              # simulate a coarser live feed (e.g. 5-min)
        # a real 5-min ASOS ob is a spot reading, not a 5-min average -- use last/max,
        # NOT mean (mean smooths the peak DOWN and hides/delays the kill).
        agg = {"last": "last", "max": "max", "mean": "mean"}[resamp]
        obs = (obs.set_index("ts").resample(f"{res_min}min")["tmpf"].agg(agg)
               .dropna().reset_index())
    if coarse_c:                                 # quantize to whole deg C (Synoptic HF-ASOS precision)
        obs["tmpf"] = (((obs["tmpf"] - 32) * 5 / 9).round() * 9 / 5 + 32)
    rows = []
    for ed, day in tr.groupby("event_day"):
        # LST climate-day window matches settlement; matters for LOWS (occur near
        # the midnight boundary). lst_off<0 => 00:00 LST = -lst_off hours UTC.
        if lst_off is not None:
            d0 = pd.Timestamp(pd.to_datetime(ed, format="%y%b%d").date(), tz="UTC") - pd.Timedelta(hours=lst_off)
        else:
            d0 = pd.Timestamp(pd.to_datetime(ed, format="%y%b%d").date(), tz="America/New_York")
        do = obs[(obs["ts"] >= d0) & (obs["ts"] < d0 + pd.Timedelta(days=1))].sort_values("ts")
        if do.empty:
            continue
        ts_arr, t_arr = do["ts"].to_numpy(), do["tmpf"].to_numpy()
        for tk, g in day.groupby("ticker"):
            strike = g["strike"].iloc[0]
            if not np.isfinite(strike):
                continue
            # HIGH: dead once running max >= strike+1.5.  LOW (inverted): dead once
            # running min <= strike-1.5 (the low already passed below the bucket).
            if low:
                thr = strike - 1.5 - margin
                cond = t_arr <= thr
            else:
                thr = strike + 1.5 + margin
                cond = t_arr >= thr
            # first index where `sustain` consecutive readings satisfy the kill
            run, hit = 0, None
            for k, a in enumerate(cond):
                run = run + 1 if a else 0
                if run >= sustain:
                    hit = k - sustain + 1
                    break
            if hit is None:
                continue
            cross = pd.Timestamp(ts_arr[hit])
            act = cross + pd.Timedelta(minutes=latency)   # when you can actually sell (feed lag)
            g = g.sort_values("ts")
            before = g[(g["ts"] >= cross - pd.Timedelta(minutes=before_win)) & (g["ts"] < cross)]
            if before.empty or before["yes_price"].mean() < 0.05:     # bucket must have been LIVE
                continue
            after = g[(g["ts"] >= act) & (g["ts"] <= act + pd.Timedelta(minutes=sell_window))]
            if after.empty:
                continue
            yes_buy_vol = float(after.loc[after["taker_side"] == "yes", "count"].sum())
            rows.append(dict(event_day=ed, ticker=tk, cross=cross,
                             sell_px=float(after["yes_price"].median()),
                             sell_vol=yes_buy_vol,        # contracts you could sell into
                             price_before=float(before["yes_price"].iloc[-1])))
    return pd.DataFrame(rows)


def evaluate(df, api):
    if df.empty:
        return None
    df = df.copy()
    df["result"] = [_result(api, tk) for tk in df["ticker"]]
    df = df.dropna(subset=["result", "sell_px"])
    if df.empty:
        return None
    df["won"] = (df["result"] == "yes").astype(int)       # bucket won => our short loses
    df["pnl"] = df["sell_px"] - df["won"] - df["sell_px"].map(_FEE)
    n = len(df)
    ev = df["pnl"].mean()
    se = df["pnl"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, win_rate=1 - df["won"].mean(), sell_px=df["sell_px"].mean(),
                ev=ev, ev_c=ev * 100, se_c=se * 100, t=ev / se if se else np.nan,
                total=df["pnl"].sum())


def main(series="KXHIGHNY", asos="NYC"):
    api = KalshiAPI()
    print(f"\n########## FLOOR EDGE: {series} ({asos}) ##########")
    print(f"{'margin':>6} {'sustain':>7} | {'n':>4} {'win%':>5} {'sellpx':>6} "
          f"{'EV/ct':>7} {'±se':>6} {'t':>5} {'totalPnL':>9}")
    best = None
    for margin in (0.0, 0.5, 1.0, 1.5):
        for sustain in (1, 3, 5):
            df = kills(series, asos, margin, sustain)
            r = evaluate(df, api)
            if not r:
                continue
            print(f"{margin:6.1f} {sustain:7d} | {r['n']:4d} {r['win_rate']*100:5.0f} "
                  f"{r['sell_px']:6.2f} {r['ev_c']:+7.1f} {r['se_c']:6.1f} {r['t']:5.2f} {r['total']:+9.2f}")
            if r["n"] >= 8 and (best is None or r["ev_c"] > best[1]["ev_c"]):
                best = ((margin, sustain), r)
    if best:
        (m, s), r = best
        print(f"\nBEST (n>=8): margin={m}, sustain={s}min -> EV {r['ev_c']:+.1f}c/contract "
              f"net of fee, win {r['win_rate']*100:.0f}%, t={r['t']:.2f}, n={r['n']}")
    return best


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    a = sys.argv[2] if len(sys.argv) > 2 else "NYC"
    main(s, a)
