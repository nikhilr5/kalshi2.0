"""Floor-latency test: when a new running-max high KILLS a bucket (running_max
exceeds the bucket -> the high already passed it -> bucket provably worth $0),
does the market crash that bucket to ~0 instantly, or does it linger at a price
you could sell into?

For each (day, bucket): find the obs time the running max first exceeds the
bucket (the 'kill'), require the bucket was LIVE before (traded >5c), then track
its trade price in windows after the kill. Lingering price = the latency edge
(you could sell the dead bucket / buy NO for ~free).
"""
import sys
import numpy as np
import pandas as pd

from util import load_trades_days, obs_history, obs_1min, _bucket_strike

START, END = "2026-05-01", "2026-06-22"


def med(g, t0, a, b):
    w = g[(g["ts"] >= t0 + pd.Timedelta(minutes=a)) & (g["ts"] < t0 + pd.Timedelta(minutes=b))]
    return (w["yes_price"].median(), len(w)) if len(w) else (np.nan, 0)


def study(series="KXHIGHNY", asos="NYC", use_1min=False):
    tr = load_trades_days(START, END, series)
    tr = tr[tr["ticker"].str.contains("-B")].copy()
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    tr["strike"] = tr["ticker"].map(_bucket_strike)
    obs = obs_1min(START, END, asos=asos) if use_1min else obs_history(START, END, asos=asos)
    res_kind = "1-MINUTE" if use_1min else "hourly METAR"
    rows = []
    for ed, day in tr.groupby("event_day"):
        # the SETTLEMENT day (from the ticker), NOT the earliest-trade day (markets
        # open the day before, so min-trade-date would be the wrong day's temps)
        d0 = pd.Timestamp(pd.to_datetime(ed, format="%y%b%d").date(), tz="America/New_York")
        do = obs[(obs["ts"] >= d0) & (obs["ts"] < d0 + pd.Timedelta(days=1))].sort_values("ts").copy()
        if do.empty:
            continue
        do["run_max"] = do["tmpf"].cummax()
        for tk, g in day.groupby("ticker"):
            strike = g["strike"].iloc[0]
            if not np.isfinite(strike):
                continue
            kill = strike + 1.5                          # run_max >= kill => bucket dead
            crossed = do[do["run_max"] >= kill]
            if crossed.empty:
                continue
            t = crossed["ts"].iloc[0]
            g = g.sort_values("ts")
            before = g[(g["ts"] >= t - pd.Timedelta(minutes=60)) & (g["ts"] < t)]
            if before.empty or before["yes_price"].mean() < 0.05:   # must have been LIVE
                continue
            p05, n05 = med(g, t, 0, 5)
            p515, n515 = med(g, t, 5, 15)
            p1530, n1530 = med(g, t, 15, 30)
            rows.append(dict(event_day=ed, ticker=tk, kill_temp=kill,
                             price_before=round(float(before["yes_price"].iloc[-1]), 3),
                             p_0_5=p05, n_0_5=n05, p_5_15=p515, p_15_30=p1530,
                             yes_buys_after=int((g[(g["ts"] >= t)]["taker_side"] == "yes").sum())))
    r = pd.DataFrame(rows)
    if r.empty:
        print(f"{series}: no kill events"); return r
    print(f"\n=== {series} ({asos}) floor-latency [{res_kind} kill detection] === "
          f"{len(r)} 'bucket killed while live' events\n")
    print(f"  price just BEFORE kill (live):      {r['price_before'].median():.2f}  (median)")
    print(f"  median dead-bucket price 0-5min after kill:  {r['p_0_5'].median():.3f}  "
          f"(n trades {int(r['n_0_5'].sum())})")
    print(f"  median dead-bucket price 5-15min after kill: {r['p_5_15'].median():.3f}")
    print(f"  median dead-bucket price 15-30min after kill:{r['p_15_30'].median():.3f}")
    lin = r[r["p_0_5"] >= 0.03]
    print(f"\n  events where dead bucket still >=3c in first 5 min: {len(lin)}/{len(r)} "
          f"({len(lin)/len(r):.0%})")
    print(f"  of those, median price still tradeable 0-5min: {lin['p_0_5'].median():.3f}")
    print(f"  total yes-BUY trades into dead buckets after kill: {int(r['yes_buys_after'].sum())}")
    print("\n  -> if prices crash to ~0 instantly = efficient (no edge);")
    print("     if they linger >0 for minutes = latency money (sell the dead bucket).")
    r.to_csv("cache/floor_latency.csv", index=False)
    return r


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    a = sys.argv[2] if len(sys.argv) > 2 else "NYC"
    use1 = "1min" in sys.argv
    study(s, a, use_1min=use1)
