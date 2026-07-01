"""Does retail systematically OVERPAY for cheap weather buckets late in the session?

If yes, you can SELL them (sell YES into the buyers) and profit -- using ONLY price
and time-to-close, NO weather feed. That sidesteps the precision wall that kills the
live floor. Pure calibration edge from historical trades + settlements.

Test: for every weather-bucket TRADE, record (yes_price, hours_to_close, settled).
  calibration: at price p, what fraction of contracts settle YES?  edge if settle < p.
  strategy:    sell YES into retail yes-buys; PnL/ct = price - settled - fee.
  control:     compare LATE (<2h to close) vs EARLY (>6h) -- a real 'dead bucket bought
               late' effect should grow toward close, not be flat (= static longshot bias).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Aston"))
from util import load_trades_days                       # noqa: E402
from kalshi_api import KalshiAPI                          # noqa: E402

START, END = "2026-05-01", "2026-06-20"
FEE = lambda p: 0.07 * p * (1 - p)
CITIES = [
    ("KXHIGHCHI", "high"), ("KXHIGHNY", "high"), ("KXHIGHPHIL", "high"),
    ("KXHIGHMIA", "high"), ("KXHIGHDEN", "high"),
    ("KXLOWTCHI", "low"), ("KXLOWTNYC", "low"), ("KXLOWTPHIL", "low"),
    ("KXLOWTMIA", "low"), ("KXLOWTDEN", "low"),
]
api = KalshiAPI()


def meta(series):
    """ticker -> (close_ts, result) in one bulk pull (all statuses)."""
    m = {}
    try:
        for mk in api.get_markets(series_ticker=series):
            tk = mk.get("ticker")
            if tk:
                m[tk] = (mk.get("close_time") or mk.get("expiration_time"), mk.get("result"))
    except Exception as e:
        print(f"  {series} meta error: {e}")
    return m


def load():
    frames = []
    for series, side in CITIES:
        try:
            tr = load_trades_days(START, END, series)
        except Exception as e:
            print(f"  {series} trades error: {e}"); continue
        tr = tr[tr["ticker"].str.contains("-B")].copy()
        if tr.empty:
            continue
        md = meta(series)
        tr["close_ts"] = tr["ticker"].map(lambda t: md.get(t, (None, None))[0])
        tr["result"] = tr["ticker"].map(lambda t: md.get(t, (None, None))[1])
        tr = tr[tr["result"].isin(["yes", "no"])].copy()
        if tr.empty:
            continue
        tr["ts"] = pd.to_datetime(tr["ts"], utc=True, format="ISO8601")
        tr["close"] = pd.to_datetime(tr["close_ts"], utc=True, format="ISO8601", errors="coerce")
        tr = tr.dropna(subset=["close"])
        tr["htc"] = (tr["close"] - tr["ts"]).dt.total_seconds() / 3600.0
        tr["settle"] = (tr["result"] == "yes").astype(int)
        tr["series"], tr["side"] = series, side
        frames.append(tr[["series", "side", "ticker", "event_day", "yes_price",
                           "count", "taker_side", "htc", "settle"]])
        print(f"  {series}: {len(tr)} settled trades")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def calib(df, label):
    """Contract-weighted settle-rate vs price -> is cheap overpriced?"""
    print(f"\n### CALIBRATION — {label} ###")
    print(f"{'price band':>12} {'contracts':>10} {'avg_px':>7} {'settleYES':>10} {'edge(px-settle)':>15}")
    bands = [(0, .05), (.05, .10), (.10, .15), (.15, .25), (.25, .50), (.50, 1.01)]
    for lo, hi in bands:
        s = df[(df["yes_price"] >= lo) & (df["yes_price"] < hi)]
        if s.empty:
            continue
        w = s["count"].to_numpy()
        px = np.average(s["yes_price"], weights=w)
        sr = np.average(s["settle"], weights=w)
        vol = w.sum()
        print(f"{lo:.2f}-{hi:.2f}".rjust(12) + f" {vol:>10.0f} {px:>7.3f} {sr:>10.3f} {px-sr:>+15.3f}")


def strat(df, label, px_max=0.15, htc_max=2.0):
    """Sell YES into retail yes-buys that are cheap+late. PnL/ct = price - settle - fee."""
    s = df[(df["taker_side"] == "yes") & (df["yes_price"] <= px_max) & (df["htc"] <= htc_max)].copy()
    if s.empty:
        print(f"\n{label}: no trades"); return None
    s["pnl"] = s["yes_price"] - s["settle"] - FEE(s["yes_price"])
    w = s["count"].to_numpy()
    pnl = s["pnl"].to_numpy()
    n_ct = w.sum()
    ev = np.average(pnl, weights=w)
    # cluster SE by event_day (trades within a day are correlated)
    day = s.groupby(["series", "event_day"]).apply(
        lambda g: np.average(g["pnl"], weights=g["count"]), include_groups=False)
    se = day.std(ddof=1) / np.sqrt(len(day)) if len(day) > 1 else np.nan
    t = ev / se if se else np.nan
    print(f"\n{label}: sell cheap(<= {px_max}) late(<= {htc_max}h) | "
          f"contracts={n_ct:.0f} days={len(day)} EV={ev*100:+.2f}c/ct t={t:.2f} "
          f"win={(pnl<0).mean()*100:.0f}%  (win=settled NO)")
    return ev, t, n_ct, len(day)


def main():
    print("loading trades + settlements...")
    df = load()
    if df.empty:
        print("no data"); return
    print(f"\nTOTAL: {len(df)} settled trades, {df['count'].sum():.0f} contracts")

    # ---- calibration: late vs early (the control) ----
    calib(df[df["htc"] <= 2], "LATE  (<=2h to close)")
    calib(df[df["htc"] > 6], "EARLY (>6h to close)")

    # ---- strategy EV: late-cheap sell, with controls ----
    print("\n### STRATEGY: sell cheap-late buckets (price+time only, NO weather feed) ###")
    strat(df, "ALL")
    strat(df, "EARLY control (>6h)", htc_max=999)   # compare; expect weaker
    for side in ("high", "low"):
        strat(df[df["side"] == side], f"side={side}")
    print("\n  per city:")
    for series, _ in CITIES:
        strat(df[df["series"] == series], f"  {series}")

    # ---- OOS split (first half vs second half by date) ----
    df["d"] = pd.to_datetime(df["event_day"], format="%y%b%d")
    mid = df["d"].quantile(0.5)
    strat(df[df["d"] <= mid], "OOS: first half")
    strat(df[df["d"] > mid], "OOS: second half")


if __name__ == "__main__":
    main()
