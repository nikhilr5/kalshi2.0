"""Feature search v2 -- the 'sticky' hypothesis.

Tests whether the TEMPERATURE TRAJECTORY (a new running-max printing = climbing,
vs stalling) and cloud change PRECEDE the market's expected-high swing -- and
whether they LEAD it (tradeable) by re-running with a lead gap: features measured
in a window that ENDS `lead_gap` minutes BEFORE the swing onset.

If a signal still predicts direction with a 30-60 min lead gap, the obs led the
market (sticky/laggy market = edge). If it only works at gap=0, it's coincident.
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from util import market_swings, obs_history

START, END = "2026-05-01", "2026-06-22"


def feats_for(onset, obs, win_min, lead_gap):
    """obs-trajectory features in [onset-gap-win, onset-gap]. Running max uses the
    whole day up to each edge, so run_max_chg = did a NEW daily high print in window."""
    w_end = onset - pd.Timedelta(minutes=lead_gap)
    w_start = w_end - pd.Timedelta(minutes=win_min)
    day0 = onset.normalize()
    upto_end = obs[(obs["ts"] >= day0) & (obs["ts"] <= w_end)]
    upto_start = obs[(obs["ts"] >= day0) & (obs["ts"] <= w_start)]
    win = obs[(obs["ts"] >= w_start) & (obs["ts"] <= w_end)]
    t = win["tmpf"].dropna()
    cl = win["cloud"].dropna()
    rmax_e = upto_end["tmpf"].max()
    rmax_s = upto_start["tmpf"].max()
    return dict(
        run_max_chg=(rmax_e - rmax_s) if pd.notna(rmax_e) and pd.notna(rmax_s) else np.nan,
        temp_chg=(t.iloc[-1] - t.iloc[0]) if len(t) >= 2 else np.nan,
        cloud_chg=(cl.iloc[-1] - cl.iloc[0]) if len(cl) >= 2 else np.nan,
        hour=onset.hour + onset.minute / 60.0, n=len(win))


def test_signal(swf, mask, target, label):
    y = (swf["direction"] == target)
    m = mask.fillna(False)
    TP = int((m & y).sum()); FP = int((m & ~y).sum())
    FN = int((~m & y).sum()); TN = int((~m & ~y).sum())
    base = y.mean(); prec = TP / (TP + FP) if (TP + FP) else np.nan
    _, p = fisher_exact([[TP, FP], [FN, TN]])
    return dict(signal=label, predicts=target, fires=TP + FP, precision=prec,
                base=base, lift=prec / base if base else np.nan,
                recall=TP / (TP + FN) if (TP + FN) else np.nan, p=p)


def main(series="KXHIGHNY", asos="NYC"):
    swf0 = market_swings(START, END, series=series)      # onset/direction/event_day
    if swf0.empty:
        print(f"{series}: no swings (too thin?)"); return
    onsets = pd.to_datetime(swf0["onset"])
    obs = obs_history((onsets.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      (onsets.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), asos=asos)
    print(f"{series} ({asos}): n={len(swf0)} swings over {swf0.event_day.nunique()} days, "
          f"base P(down)={(swf0.direction=='down').mean():.0%}\n")

    for gap in (0, 30, 60):
        f = pd.DataFrame([feats_for(o, obs, win_min=60, lead_gap=gap) for o in onsets])
        d = pd.concat([swf0[["direction"]].reset_index(drop=True), f], axis=1)
        sig = {
            "new high printed (run_max_chg>0)": (d["run_max_chg"] > 0, "up"),
            "stalled (run_max_chg==0)":         (d["run_max_chg"] == 0, "down"),
            "temp rising (temp_chg>0)":         (d["temp_chg"] > 0, "up"),
            "temp falling (temp_chg<0)":        (d["temp_chg"] < 0, "down"),
            "clouds up (cloud_chg>=.3)":         (d["cloud_chg"] >= 0.3, "down"),
            "clouds clearing (cloud_chg<=-.3)":  (d["cloud_chg"] <= -0.3, "up"),
            "stalled & clouds up":               ((d["run_max_chg"] == 0) & (d["cloud_chg"] >= 0.3), "down"),
            "new high & clouds clearing":        ((d["run_max_chg"] > 0) & (d["cloud_chg"] <= -0.3), "up"),
        }
        rows = [test_signal(d, m, tgt, name) for name, (m, tgt) in sig.items()]
        res = pd.DataFrame(rows).sort_values("lift", ascending=False)
        print(f"===== LEAD GAP = {gap} min  (obs window ends {gap}m before onset) =====")
        print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print()


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    a = sys.argv[2] if len(sys.argv) > 2 else "NYC"
    main(s, a)
