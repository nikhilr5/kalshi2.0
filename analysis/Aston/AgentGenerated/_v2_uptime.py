"""Per-side market-presence (quote uptime) for the post-fix v2 engine.

For each day, reconstruct per-order live intervals (placed -> first
cancel/fill), union them per side, and measure:

  - plain uptime: % of quotable time at least one order of that side is live
  - price-quality uptime: same, but only counting time the resting quote
    is on the CORRECT side of theo and within a sane edge band (a quote
    that has drifted to the wrong side of theo, or way past its intended
    edge, is "present but toxic" -- the old BBO-clamp failure mode).

Quotable time = seconds with an active market (theo, 90s < sec_to_exp <=
900s); excludes auto-off window + inter-market gaps. Identical denom for
both sides and both metrics.

Config edges (KXETH15M v2): buy rests ~theo-7c, sell rests ~theo+5c.
"good" = correct side of theo and |price-theo| <= MAX_EDGE.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting")
# full post-fix v2 days; 6/10 only complete in the S3 cache copy
DAYS = {
    "26JUN10": BT / "_s3_cache/KXETH15M-26JUN10.db",
    "26JUN11": BT / "data/KXETH15M-26JUN11.db",
    "26JUN12": BT / "data/KXETH15M-26JUN12.db",   # partial (~12h)
}
MAX_EDGE = 0.15   # a resting quote >15c off theo on the correct side is stale/junk


def live_intervals(ev, side, day_end_ns):
    """List of (open_ns, close_ns, order_id, price) for one side."""
    sub = ev[ev.act == side]
    out = []
    for oid, g in sub.groupby("order_id", sort=False):
        g = g.sort_values("ts")
        p = g[g.event_type == "placed"]
        if p.empty:
            continue
        to = p["ts"].iloc[0]
        price = p["price"].iloc[0]
        term = g[(g.event_type != "placed") & (g.ts >= to)]
        tc = term["ts"].iloc[0] if not term.empty else pd.Timestamp(day_end_ns, tz="UTC")
        out.append((to.value, tc.value, oid, price))
    return out


def union_len(intervals):
    """Total covered seconds of a list of (open_ns, close_ns, ...)."""
    if not intervals:
        return 0.0
    iv = sorted((a, b) for a, b, *_ in intervals)
    cs, ce = iv[0]
    tot = 0.0
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    tot += ce - cs
    return tot / 1e9


def good_quote_intervals(intervals, theo_by_tkr, ev_placed, side):
    """Trim each live interval to the portion where the resting price is
    on the correct side of theo and within MAX_EDGE. We approximate theo
    over the order's life by its value at PLACEMENT (theo moves <~1c over
    the typical sub-minute order life; this is a conservative proxy)."""
    # map order_id -> (ticker, placement_ts)
    pl = ev_placed[ev_placed.act == side].set_index("order_id")
    keep = []
    for a, b, oid, price in intervals:
        row = pl.loc[oid] if oid in pl.index else None
        if row is None:
            continue
        tkr = row["ticker"]
        th = theo_by_tkr.get(tkr)
        if th is None or th.empty:
            continue
        # theo at placement (asof backward)
        pos = th["ts_ns"].searchsorted(a, side="right") - 1
        if pos < 0:
            continue
        theo = th["theo"].iat[pos]
        if side == "buy":
            ok = (price < theo) and (theo - price <= MAX_EDGE)
        else:  # sell
            ok = (price > theo) and (price - theo <= MAX_EDGE)
        if ok:
            keep.append((a, b, oid, price))
    return keep


def quotable_seconds(theo_df):
    th = theo_df[(theo_df.seconds_to_expiry > 90) & (theo_df.seconds_to_expiry <= 900)]
    return th["ts"].dt.floor("s").nunique()


rows = []
for day, db in DAYS.items():
    if not db.exists():
        print(f"[skip] {day}: missing"); continue
    c = sqlite3.connect(str(db))
    ev = pd.read_sql(
        "SELECT ts,order_id,ticker,action,price,event_type FROM order_events "
        "WHERE event_type IN ('placed','cancelled','filled')", c)
    th = pd.read_sql("SELECT ts,ticker,theo,seconds_to_expiry FROM theo_state", c)
    c.close()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True, format="ISO8601")
    th["ts"] = pd.to_datetime(th["ts"], utc=True, format="ISO8601")
    ev = ev.sort_values("ts")
    d0, d1 = ev["ts"].min(), ev["ts"].max()
    span = (d1 - d0).total_seconds()

    act = ev[ev.event_type == "placed"].groupby("order_id")["action"].first()
    ev["act"] = ev["order_id"].map(act)
    ev_placed = ev[ev.event_type == "placed"][["order_id", "ticker", "act", "price"]]

    qsec = quotable_seconds(th)

    th_sorted = th.sort_values("ts")
    th_sorted["ts_ns"] = th_sorted["ts"].astype("int64")
    theo_by_tkr = {t: g.reset_index(drop=True)
                   for t, g in th_sorted[["ts_ns", "theo"]].assign(
                       tkr=th_sorted["ticker"].values).groupby("tkr")}

    rec = {"day": day, "span_h": span / 3600, "quotable_h": qsec / 3600}
    for side in ("buy", "sell"):
        iv = live_intervals(ev, side, d1.value)
        plain = union_len(iv)
        good_iv = good_quote_intervals(iv, theo_by_tkr, ev_placed, side)
        good = union_len(good_iv)
        n_pl = int((ev_placed.act == side).sum())
        rec[f"{side}_plain_pct"] = 100 * plain / qsec
        rec[f"{side}_good_pct"] = 100 * good / qsec
        rec[f"{side}_n"] = n_pl
        rec[f"{side}_good_frac"] = good / plain if plain else np.nan
    rows.append(rec)
    print(f"[{day}] span {rec['span_h']:.2f}h  quotable {rec['quotable_h']:.2f}h  "
          f"placed buy={rec['buy_n']} sell={rec['sell_n']}")

df = pd.DataFrame(rows)
print("\n" + "=" * 78)
print("PER-SIDE QUOTE UPTIME (% of quotable time)  —  post-fix v2")
print("=" * 78)
print(f"  {'day':<9}{'BUY plain':>10}{'BUY good':>9}{'SELL plain':>11}{'SELL good':>10}")
for r in df.itertuples():
    print(f"  {r.day:<9}{r.buy_plain_pct:>9.1f}%{r.buy_good_pct:>8.1f}%"
          f"{r.sell_plain_pct:>10.1f}%{r.sell_good_pct:>9.1f}%")

# weight full days equally; flag the partial day
full = df[df["span_h"] > 20]
print(f"\n  full-day average (n={len(full)} days, excludes partial):")
for side in ("buy", "sell"):
    print(f"    {side:<5} plain {full[f'{side}_plain_pct'].mean():5.1f}%   "
          f"good {full[f'{side}_good_pct'].mean():5.1f}%   "
          f"(good/plain = {100*full[f'{side}_good_frac'].mean():.0f}% of present-time is on-theo)")
print(f"\n  MAX_EDGE for 'good' = {MAX_EDGE*100:.0f}c off theo, correct side. "
      f"6/12 is partial ({df.iloc[-1]['span_h']:.1f}h) — shown but excluded from avg.")
