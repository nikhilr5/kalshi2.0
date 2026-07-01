"""Deep-dive on the LATE-SESSION underreaction edge found in taker_edge.py.

Two effects, both from price+time only (NO weather feed):
  SELL cheap late buckets (<0.15)  -- overpriced, small edge
  BUY  the neglected contender band -- underpriced late, large edge (the headline)

Rigor: fine calibration curve, clustered-by-day t-stats, per-city/side, OOS split,
time-to-close decay, and capacity ($/day). Goal: confirm or kill the buy edge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
from taker_edge import load, FEE                          # noqa: E402

_CACHE = Path(__file__).resolve().parent / "cache" / "taker_df.parquet"


def get_df():
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    df = load()
    _CACHE.parent.mkdir(exist_ok=True)
    df.to_parquet(_CACHE)
    return df


def fine_calib(df, lo=0.0, hi=0.50, step=0.05):
    print(f"\n### FINE CALIBRATION (LATE <=2h)  settle-rate vs price ###")
    print(f"{'band':>12} {'contracts':>10} {'avg_px':>7} {'settle':>7} {'edge':>8} {'buy_EV':>7} {'sell_EV':>7}")
    e = lo
    while e < hi:
        s = df[(df["yes_price"] >= e) & (df["yes_price"] < e + step)]
        if not s.empty:
            w = s["count"].to_numpy()
            px = np.average(s["yes_price"], weights=w); sr = np.average(s["settle"], weights=w)
            fee = FEE(px)
            print(f"{e:.2f}-{e+step:.2f}".rjust(12) + f" {w.sum():>10.0f} {px:>7.3f} {sr:>7.3f} "
                  f"{px-sr:>+8.3f} {(sr-px-fee)*100:>+6.1f}c {(px-sr-fee)*100:>+6.1f}c")
        e += step


def ev(df, side_buy, px_lo, px_hi, htc_max, taker=None):
    """Clustered EV. side_buy=True -> buy yes (PnL=settle-px-fee); else sell (px-settle-fee)."""
    s = df[(df["yes_price"] >= px_lo) & (df["yes_price"] < px_hi) & (df["htc"] <= htc_max)]
    if taker:
        s = s[s["taker_side"] == taker]
    if s.empty:
        return None
    fee = FEE(s["yes_price"])
    pnl = (s["settle"] - s["yes_price"] - fee) if side_buy else (s["yes_price"] - s["settle"] - fee)
    s = s.assign(pnl=pnl)
    w = s["count"].to_numpy()
    e = np.average(s["pnl"], weights=w)
    day = s.groupby(["series", "event_day"]).apply(
        lambda g: np.average(g["pnl"], weights=g["count"]), include_groups=False)
    se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 1 else np.nan
    return dict(ev_c=e * 100, t=(e / se if se else np.nan), n_ct=w.sum(), days=len(day))


def line(tag, r):
    if r:
        print(f"  {tag:24} EV={r['ev_c']:+6.2f}c/ct  t={r['t']:>6.2f}  contracts={r['n_ct']:>9.0f}  days={r['days']}")
    else:
        print(f"  {tag:24} (no data)")


def main():
    df = get_df()
    print(f"loaded {len(df)} trades, {df['count'].sum():.0f} contracts")
    df["d"] = pd.to_datetime(df["event_day"], format="%y%b%d")
    late = df[df["htc"] <= 2]

    fine_calib(late)

    # ---- BUY contender band (the headline) ----
    print("\n### BUY contender band [0.15,0.30), late ###")
    line("ALL", ev(df, True, 0.15, 0.30, 2))
    for s in ("high", "low"):
        line(f"side={s}", ev(df[df["side"] == s], True, 0.15, 0.30, 2))
    mid = df["d"].quantile(0.5)
    line("OOS first half", ev(df[df["d"] <= mid], True, 0.15, 0.30, 2))
    line("OOS second half", ev(df[df["d"] > mid], True, 0.15, 0.30, 2))
    print("  by time-to-close:")
    for a, b in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8)):
        line(f"htc {a}-{b}h", ev(df[df["htc"] > a], True, 0.15, 0.30, b))
    print("  per city:")
    for series in df["series"].unique():
        line(series, ev(df[df["series"] == series], True, 0.15, 0.30, 2))

    # ---- SELL cheap band (for comparison) ----
    print("\n### SELL cheap band [0.03,0.15), late (sell into yes-buyers) ###")
    line("ALL", ev(df, False, 0.03, 0.15, 2, taker="yes"))
    for s in ("high", "low"):
        line(f"side={s}", ev(df[df["side"] == s], False, 0.03, 0.15, 2, taker="yes"))

    # ---- capacity ----
    b = late[(late["yes_price"] >= 0.15) & (late["yes_price"] < 0.30)]
    ndays = df["d"].nunique()
    print(f"\nCAPACITY (buy band): {b['count'].sum():.0f} contracts over ~{ndays} calendar days "
          f"= {b['count'].sum()/ndays:.0f} ct/day across {df['series'].nunique()} cities")
    r = ev(df, True, 0.15, 0.30, 2)
    if r:
        print(f"  est gross $/day = {r['ev_c']/100 * b['count'].sum()/ndays:.0f}  (median-fill upper bound)")


if __name__ == "__main__":
    main()
