"""Low-temp floor scan: the floor edge INVERTED onto the cold tail.

A low bucket B<m> is dead once the running MIN drops to <= m-1.5 (the daily low
already passed below it; min is monotonic -> permanent). Sell the dead bucket into
retail. Risks unique to lows vs highs: (1) the low occurs near the midnight climate
-day boundary, so we bin by Local STANDARD Time (lst_off) to match settlement;
(2) the dawn minimum is a noisy basin, so use sustain>=3.

Validates every kill against the actual Kalshi settlement (win = bucket settled NO,
i.e. our short won). win_rate = 1 - false-kill rate -- the headline robustness number.

  python3 floor_scan_low.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
import floor_edge as fe                    # noqa: E402
from kalshi_api import KalshiAPI          # noqa: E402

# series -> (IEM asos id, Local STANDARD Time offset).  Reuses the asos ids verified
# in the high scan; lst_off = standard (no-DST) UTC offset of the climate day.
LOW_CITIES = [
    ("KXLOWTNYC",  "NYC", -5), ("KXLOWTCHI",  "MDW", -6), ("KXLOWTPHIL", "PHL", -5),
    ("KXLOWTMIA",  "MIA", -5), ("KXLOWTAUS",  "AUS", -6), ("KXLOWTDEN",  "DEN", -7),
    ("KXLOWTOKC",  "OKC", -6), ("KXLOWTBOS",  "BOS", -5), ("KXLOWTDAL",  "DAL", -6),
    ("KXLOWTHOU",  "IAH", -6), ("KXLOWTPHX",  "PHX", -7), ("KXLOWTATL",  "ATL", -5),
    ("KXLOWTSEA",  "SEA", -8), ("KXLOWTLAX",  "LAX", -8),
]
CFG = dict(margin=0.0, sustain=3, sell_window=60, latency=5, before_win=180, low=True)
WEEKS = 7.0


def run_city(series, asos, lst_off, api, lst=True):
    df = fe.kills(series, asos, CFG["margin"], CFG["sustain"], sell_window=CFG["sell_window"],
                  latency=CFG["latency"], before_win=CFG["before_win"], low=True,
                  lst_off=(lst_off if lst else None))
    if df.empty:
        return None
    r = fe.evaluate(df, api)
    if r:
        df2 = df.copy()
        df2["result"] = [fe._result(api, tk) for tk in df2["ticker"]]
        df2 = df2.dropna(subset=["result", "sell_px"])
        r["vol_wk"] = float(df2["sell_vol"].sum()) / WEEKS
        r["ev_x_vol_wk"] = r["ev"] * float(df2["sell_vol"].sum()) / WEEKS
    return r


def main():
    api = KalshiAPI()
    print("\n### LOW-TEMP FLOOR (LST-binned, sustain=3, lat=5, sell_window=60min) ###")
    print(f"{'series':14} {'st':4} | {'n':>3} {'win%':>4} {'sellpx':>6} {'EV/ct':>6} "
          f"{'t':>5} {'vol/wk':>7} {'$EV/wk':>7}")
    rows = []
    for series, asos, lst_off in LOW_CITIES:
        try:
            r = run_city(series, asos, lst_off, api, lst=True)
        except Exception as e:
            print(f"{series:14} {asos:4} | ERR {str(e)[:45]}")
            continue
        if not r:
            print(f"{series:14} {asos:4} | no kills / no data")
            continue
        print(f"{series:14} {asos:4} | {r['n']:>3} {r['win_rate']*100:>4.0f} {r['sell_px']:>6.2f} "
              f"{r['ev_c']:>+6.1f} {r['t']:>5.2f} {r.get('vol_wk',0):>7.0f} {r.get('ev_x_vol_wk',0):>+7.1f}")
        rows.append((series, asos, r))

    rows.sort(key=lambda x: x[2].get("ev_x_vol_wk", -1e9), reverse=True)
    print("\n--- ranked by $EV/wk (significant = t>=2 & EV>0) ---")
    for series, asos, r in rows:
        flag = "" if (r["t"] >= 2 and r["ev_c"] > 0) else "  <- weak"
        print(f"  {series:14} {asos:4} ${r.get('ev_x_vol_wk',0):+7.1f}/wk  "
              f"(EV {r['ev_c']:+.1f}c, win {r['win_rate']*100:.0f}%, t={r['t']:.2f}, n={r['n']}){flag}")

    # ET-vs-LST check on the 3 biggest -- does the boundary fix reduce false kills?
    print("\n--- LST vs ET day-binning (false-kill check) on 3 high-volume lows ---")
    for series, asos, lst_off in [("KXLOWTNYC", "NYC", -5), ("KXLOWTDAL", "DAL", -6),
                                  ("KXLOWTPHX", "PHX", -7)]:
        for lab, lst in [("LST", True), ("ET ", False)]:
            try:
                r = run_city(series, asos, lst_off, api, lst=lst)
                if r:
                    print(f"  {series:12} {lab}: win {r['win_rate']*100:.0f}%  EV {r['ev_c']:+.1f}c  "
                          f"t={r['t']:.2f}  n={r['n']}")
            except Exception as e:
                print(f"  {series} {lab}: ERR {str(e)[:40]}")
    return rows


if __name__ == "__main__":
    main()
