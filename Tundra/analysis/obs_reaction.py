"""Does the market UNDER-REACT to an observation and then drift?  (the right test)

Anchor on each ASOS obs at time T (not on market swings). Compute what the obs
changed (new running-max high, temp delta, cloud delta) and then measure the
market's implied-high move in [T, T+horizon]. If the obs change PREDICTS the
SUBSEQUENT move, the market under-reacted at T and drifts -> tradeable. If the
obs change predicts nothing forward, the market already priced it instantly.

Also splits the reaction into IMMEDIATE [T-15,T] vs DELAYED [T,T+h] so we can see
whether the move happens at the obs (efficient) or after it (sticky/edge).
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from util import load_trades_days, _implied_high_series, obs_history

START, END, HORIZON = "2026-05-01", "2026-06-22", 60


def study(series="KXHIGHNY", asos="NYC", horizon=HORIZON):
    tr = load_trades_days(START, END, series)
    tr = tr[tr["ticker"].str.contains("-B")].copy()
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    obs = obs_history(START, END, asos=asos)
    H = pd.Timedelta(minutes=horizon)
    rows = []
    for ed, day in tr.groupby("event_day"):
        eh = _implied_high_series(day, 10)                       # market implied high, 1-min
        if len(eh) < horizon + 30:
            continue
        lo, hi = eh.index[0], eh.index[-1]
        d0 = lo.normalize()
        do = obs[(obs["ts"] >= d0) & (obs["ts"] < d0 + pd.Timedelta(days=1))].sort_values("ts").copy()
        do["run_max"] = do["tmpf"].cummax()
        for i in range(1, len(do)):
            T = do["ts"].iloc[i]
            if T <= lo + pd.Timedelta(minutes=15) or T + H > hi:
                continue
            now, then = eh.asof(T), eh.asof(T + H)
            pre = eh.asof(T - pd.Timedelta(minutes=15))
            if pd.isna(now) or pd.isna(then) or pd.isna(pre):
                continue
            c_now, c_prev = do["cloud"].iloc[i], do["cloud"].iloc[i - 1]
            rows.append(dict(
                new_high=float(do["run_max"].iloc[i] - do["run_max"].iloc[i - 1]),
                dtemp=float(do["tmpf"].iloc[i] - do["tmpf"].iloc[i - 1]),
                dcloud=float(c_now - c_prev) if pd.notna(c_now) and pd.notna(c_prev) else np.nan,
                immediate=float(now - pre),                      # move AROUND the obs
                delayed=float(then - now)))                      # drift AFTER the obs
    df = pd.DataFrame(rows)
    print(f"\n=== {series} ({asos}) === {len(df)} obs-events | "
          f"mean |delayed move| over {horizon}min = {df['delayed'].abs().mean():.2f}F\n")
    print("Does the obs predict the SUBSEQUENT (delayed) market move?  -> under-reaction = edge")
    for f in ["new_high", "dtemp", "dcloud"]:
        d = df.dropna(subset=[f, "delayed"])
        r, p = pearsonr(d[f], d["delayed"])
        ri, pi = pearsonr(d[f], d.loc[d.index, "immediate"])
        print(f"  {f:9s}: corr w/ DELAYED move r={r:+.2f} p={p:.3f}   "
              f"(vs IMMEDIATE r={ri:+.2f} p={pi:.3f})   n={len(d)}")
    # when a NEW HIGH prints, does the market keep drifting up afterwards?
    nh = df["new_high"] > 0
    print(f"\n  new-high prints (n={nh.sum()}): mean delayed move = {df.loc[nh,'delayed'].mean():+.2f}F "
          f"(vs {df.loc[~nh,'delayed'].mean():+.2f}F when no new high)")
    cu = df["dcloud"] >= 0.3
    print(f"  clouds-up obs (n={int(cu.sum())}): mean delayed move = {df.loc[cu,'delayed'].mean():+.2f}F")
    return df


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    a = sys.argv[2] if len(sys.argv) > 2 else "NYC"
    study(s, a)
