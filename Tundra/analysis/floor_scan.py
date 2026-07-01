"""Scan the floor edge across ALL MADIS-covered Kalshi high-temp cities.

Uses the REALISTIC obtainable live config (the corrected one, not the 1-min
archive ideal): 5-min cadence, spot reading (resamp='last'), 5-min feed latency,
sustain=1. Ranks cities by EV/contract x sellable volume so we see where the
money actually is, not just where the t-stat is prettiest.

  python3 floor_scan.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
import floor_edge as fe                    # noqa: E402  (sets up util/Aston path)
from kalshi_api import KalshiAPI          # noqa: E402

# Kalshi high-temp series -> IEM asos id (settlement station, K-prefix dropped).
# NYC excluded: KNYC is hourly-only in MADIS, no live 5-min feed.
CITIES = [
    ("KXHIGHCHI",  "MDW"),   # already validated, kept as reference
    ("KXHIGHPHIL", "PHL"),
    ("KXHIGHLAX",  "LAX"),
    ("KXHIGHMIA",  "MIA"),
    ("KXHIGHAUS",  "AUS"),   # CAVEAT: confirm Kalshi settles KAUS not Camp Mabry KATT
    ("KXHIGHDEN",  "DEN"),
    ("KXHIGHTOKC", "OKC"),
    ("KXHIGHTBOS", "BOS"),
    ("KXHIGHTDAL", "DAL"),
    ("KXHIGHTHOU", "IAH"),   # CAVEAT: confirm KIAH vs KHOU
    ("KXHIGHTPHX", "PHX"),
    ("KXHIGHTATL", "ATL"),
    ("KXHIGHTSEA", "SEA"),
]

CFG = dict(margin=0.0, sustain=1, res_min=5, resamp="last", latency=5, coarse_c=False)
WEEKS = 7.0          # START..END span ~ May1-Jun20


def scan_city(series, asos, api):
    df = fe.kills(series, asos, CFG["margin"], CFG["sustain"], res_min=CFG["res_min"],
                  resamp=CFG["resamp"], latency=CFG["latency"], coarse_c=CFG["coarse_c"])
    if df.empty:
        return None
    r = fe.evaluate(df, api)
    if not r:
        return None
    # total contracts you could actually sell into, over the evaluated events
    df2 = df.copy()
    df2["result"] = [fe._result(api, tk) for tk in df2["ticker"]]
    df2 = df2.dropna(subset=["result", "sell_px"])
    vol = float(df2["sell_vol"].sum())
    r["vol"] = vol
    r["vol_wk"] = vol / WEEKS
    r["ev_x_vol_wk"] = r["ev"] * vol / WEEKS        # $/week upper-bound at median fill
    return r


def main():
    api = KalshiAPI()
    print(f"\n### FLOOR SCAN (5-min fine, spot, lat=5min, sustain=1) ###")
    print(f"{'series':14} {'st':4} | {'n':>3} {'win%':>4} {'sellpx':>6} {'EV/ct':>6} "
          f"{'t':>5} {'vol/wk':>7} {'$EV/wk':>7}")
    rows = []
    for series, asos in CITIES:
        try:
            r = scan_city(series, asos, api)
        except Exception as e:
            print(f"{series:14} {asos:4} | ERR {str(e)[:50]}")
            continue
        if not r:
            print(f"{series:14} {asos:4} | no kill events / no data")
            continue
        print(f"{series:14} {asos:4} | {r['n']:>3} {r['win_rate']*100:>4.0f} {r['sell_px']:>6.2f} "
              f"{r['ev_c']:>+6.1f} {r['t']:>5.2f} {r['vol_wk']:>7.0f} {r['ev_x_vol_wk']:>+7.1f}")
        rows.append((series, asos, r))
    rows.sort(key=lambda x: x[2]["ev_x_vol_wk"], reverse=True)
    print("\n--- ranked by $EV/week (EV/ct x sellable vol, upper bound at median fill) ---")
    for series, asos, r in rows:
        flag = "" if (r["t"] >= 2 and r["ev_c"] > 0) else "  <- weak (t<2 or EV<=0)"
        print(f"  {series:14} {asos:4}  ${r['ev_x_vol_wk']:+7.1f}/wk  "
              f"(EV {r['ev_c']:+.1f}c x {r['vol_wk']:.0f}/wk, t={r['t']:.2f}, n={r['n']}){flag}")
    return rows


if __name__ == "__main__":
    main()
