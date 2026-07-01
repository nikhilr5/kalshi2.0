"""Low-temp floor: explore the TIMING + premise before computing EV.

Inverted floor: a low bucket B<m> (covers m-0.5..m+0.5) is dead once the running
MINIMUM drops to <= m-1.5 (the low already went below it; min is monotonic so the
kill is permanent). Unlike highs, the low locks near DAWN -- so the question is
whether dead low-buckets keep getting BOUGHT by retail during daytime hours (the
liquid window), hours after the kill. That's the premise; verify it before EV.

For each (day, B-bucket): find the kill time (running_min crosses), then report
  - kill hour-of-day (is it really dawn?)
  - price just before kill (was the bucket live?)
  - price + yes-buy volume in daytime windows AFTER the kill
"""
import sys
import numpy as np
import pandas as pd

from util import load_trades_days, obs_1min, _bucket_strike

START, END = "2026-05-01", "2026-06-20"


def med_vol(g, t0, a_min, b_min):
    w = g[(g["ts"] >= t0 + pd.Timedelta(minutes=a_min)) &
          (g["ts"] < t0 + pd.Timedelta(minutes=b_min))]
    yb = w.loc[w["taker_side"] == "yes", "count"].sum() if len(w) else 0
    return (w["yes_price"].median() if len(w) else np.nan, int(len(w)), float(yb))


def study(series, asos):
    tr = load_trades_days(START, END, series)
    tr = tr[tr["ticker"].str.contains("-B")].copy()
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    tr["strike"] = tr["ticker"].map(_bucket_strike)
    obs = obs_1min(START, END, asos=asos)
    rows = []
    for ed, day in tr.groupby("event_day"):
        d0 = pd.Timestamp(pd.to_datetime(ed, format="%y%b%d").date(), tz="America/New_York")
        do = obs[(obs["ts"] >= d0) & (obs["ts"] < d0 + pd.Timedelta(days=1))].sort_values("ts").copy()
        if do.empty:
            continue
        do["run_min"] = do["tmpf"].cummin()
        for tk, g in day.groupby("ticker"):
            strike = g["strike"].iloc[0]
            if not np.isfinite(strike):
                continue
            kill = strike - 1.5                       # run_min <= kill => low bucket dead
            crossed = do[do["run_min"] <= kill]
            if crossed.empty:
                continue
            t = crossed["ts"].iloc[0]
            g = g.sort_values("ts")
            before = g[(g["ts"] >= t - pd.Timedelta(minutes=120)) & (g["ts"] < t)]
            live_px = before["yes_price"].iloc[-1] if len(before) else np.nan
            # daytime windows AFTER the kill (kill is ~dawn; liquidity comes later)
            p_0_30, n0, v0 = med_vol(g, t, 0, 30)
            p_30_2h, n1, v1 = med_vol(g, t, 30, 120)
            p_2h_8h, n2, v2 = med_vol(g, t, 120, 480)
            p_8h_end, n3, v3 = med_vol(g, t, 480, 1200)
            rows.append(dict(event_day=ed, ticker=tk, kill_hr=t.hour,
                             live_px_before=live_px,
                             p_0_30=p_0_30, p_30_2h=p_30_2h, p_2h_8h=p_2h_8h, p_8h_end=p_8h_end,
                             ybuy_after=v0 + v1 + v2 + v3))
    return pd.DataFrame(rows)


def main(series, asos):
    r = study(series, asos)
    if r.empty:
        print(f"{series}: no kill events"); return
    print(f"\n=== {series} ({asos}) LOW-floor exploration === {len(r)} bucket-kills while/after live\n")
    print(f"  kill hour-of-day (ET):  median {int(r['kill_hr'].median())}h  "
          f"[{int(r['kill_hr'].quantile(.1))}h - {int(r['kill_hr'].quantile(.9))}h]")
    print(f"  price just before kill: {r['live_px_before'].median():.3f} (median; NaN={r['live_px_before'].isna().mean():.0%})")
    print(f"  dead-bucket price  0-30min after kill: {r['p_0_30'].median():.3f}")
    print(f"  dead-bucket price 30min-2h after kill: {r['p_30_2h'].median():.3f}")
    print(f"  dead-bucket price    2h-8h after kill: {r['p_2h_8h'].median():.3f}")
    print(f"  dead-bucket price   8h-end after kill: {r['p_8h_end'].median():.3f}")
    print(f"  total yes-BUY contracts into dead buckets after kill: {int(r['ybuy_after'].sum())}")
    print(f"  buckets still >=5c at 2h-8h window: {(r['p_2h_8h']>=0.05).mean():.0%}")
    return r


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "KXLOWTNYC"
    a = sys.argv[2] if len(sys.argv) > 2 else "NYC"
    main(s, a)
