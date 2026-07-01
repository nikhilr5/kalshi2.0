"""Does retail's YES/hope side systematically lose in POLITICS behavioral markets?

Thesis: weather is dry (no emotion) -> efficient. Politics is emotional (partisans
bet hope) -> retail YES-overpricing -> selling YES / buying NO is +EV. Test it.

For every trade: yes_price, taker_side (yes = retail aggressively buying yes), settled.
  calibration:  at price p, settle-YES rate.  edge = p - settle (>0 = YES overpriced)
  yes-share:    fraction of taker volume that is buying YES (retail hope tilt)
  sell-YES EV:  selling to retail yes-buyers, PnL/ct = price - settle - fee
Clustered SE by (series, event). Compared head-to-head with weather.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
from util import load_trades_days                       # noqa: E402
from kalshi_api import KalshiAPI                          # noqa: E402

START, END = "2026-03-15", "2026-06-20"
FEE = lambda p: 0.07 * p * (1 - p)
POLI = ["KXTRUMPACT", "KXEOWEEK", "KXAPRPOTUS", "KXTRUTHSOCIAL"]
WEATHER = ["KXHIGHCHI", "KXHIGHNY"]      # dry-market baseline
api = KalshiAPI()


def meta(series):
    m = {}
    try:
        for mk in api.get_markets(series_ticker=series):
            tk = mk.get("ticker")
            if tk:
                m[tk] = mk.get("result")
    except Exception as e:
        print(f"  {series} meta err: {e}")
    return m


def load(series_list):
    frames = []
    for series in series_list:
        try:
            tr = load_trades_days(START, END, series)
        except Exception as e:
            print(f"  {series}: trades err {e}"); continue
        if tr.empty:
            print(f"  {series}: no trades"); continue
        md = meta(series)
        tr["result"] = tr["ticker"].map(md)
        tr = tr[tr["result"].isin(["yes", "no"])].copy()
        if tr.empty:
            print(f"  {series}: no settled"); continue
        tr["settle"] = (tr["result"] == "yes").astype(int)
        tr["series"] = series
        frames.append(tr[["series", "ticker", "event_day", "yes_price", "count", "taker_side", "settle"]])
        print(f"  {series}: {len(tr)} settled trades, {tr['count'].sum():.0f} ct")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calib(df, label):
    print(f"\n### CALIBRATION — {label} ###")
    print(f"{'price band':>12} {'contracts':>10} {'avg_px':>7} {'settleYES':>10} {'edge=px-settle':>15}")
    for lo, hi in [(0, .05), (.05, .15), (.15, .30), (.30, .50), (.50, .70), (.70, .90), (.90, 1.01)]:
        s = df[(df["yes_price"] >= lo) & (df["yes_price"] < hi)]
        if s.empty:
            continue
        w = s["count"].to_numpy()
        px = np.average(s["yes_price"], weights=w); sr = np.average(s["settle"], weights=w)
        print(f"{lo:.2f}-{hi:.2f}".rjust(12) + f" {w.sum():>10.0f} {px:>7.3f} {sr:>10.3f} {px-sr:>+15.3f}")


def sell_yes(df, label):
    """Edge from selling YES into retail yes-buyers (the hope-fade)."""
    s = df[df["taker_side"] == "yes"].copy()
    if s.empty:
        print(f"  {label:22} no yes-takers"); return
    s["pnl"] = s["yes_price"] - s["settle"] - FEE(s["yes_price"])
    w = s["count"].to_numpy()
    ev = np.average(s["pnl"], weights=w)
    day = s.groupby(["series", "event_day"]).apply(
        lambda g: np.average(g["pnl"], weights=g["count"]), include_groups=False)
    se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 1 else np.nan
    t = ev / se if se else np.nan
    yes_share = df[df["taker_side"] == "yes"]["count"].sum() / df["count"].sum()
    print(f"  {label:22} sellYES EV={ev*100:+6.2f}c/ct t={t:>6.2f} | yes-taker share={yes_share*100:.0f}% "
          f"| ct={w.sum():.0f} events={len(day)}")


def main():
    print("loading POLITICS...")
    pol = load(POLI)
    print("\nloading WEATHER baseline...")
    wx = load(WEATHER)
    if not pol.empty:
        calib(pol, "POLITICS (all)")
        print("\n### SELL-YES (hope-fade) EV ###")
        sell_yes(pol, "POLITICS pooled")
        for s in POLI:
            sub = pol[pol["series"] == s]
            if not sub.empty:
                sell_yes(sub, s)
    if not wx.empty:
        calib(wx, "WEATHER baseline (all)")
        sell_yes(wx, "WEATHER pooled")


if __name__ == "__main__":
    main()
