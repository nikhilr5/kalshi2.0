"""MM markout decomposition for Aston fills: SPREAD CAPTURE vs ADVERSE SELECTION.

The honest test of whether Aston sits on a real market-making edge (vs the direction-
neutral lens, which conflated the making edge with unhedged inventory drift).

For each fill, with a fair value F (book mid; theo as fallback):
  spread_capture/ct = qsign * (F_at_fill - price)     # sold ABOVE fair / bought BELOW = +
  markout_h/ct      = qsign * (F_at_fill+h - F_at_fill) # fair drift AGAINST you after = adverse
  net_h/ct          = spread_capture + markout_h
  qsign = +1 if (buy & yes) or (sell & no), else -1   # signed YES direction

Read it: spread_capture > 0 => you provide liquidity (real gross edge). markout strongly
negative => informed flow picks you off (you're the toxic-fill taker). net_60s > 0 => a
REAL making edge before long-horizon inventory/direction. Split by side should show the
asymmetry: SELL fills (retail buys from you) net-positive, BUY fills net-negative.

Run in your normal terminal (this agent session is TCC-blocked from the Desktop):
    python3 analysis/markout.py analysis/backtesting/data/KXETH15M-26JUN21.db
"""
import sys
import sqlite3

import numpy as np
import pandas as pd

HORIZONS = [10, 30, 60, 300]      # seconds


def find(cols, *subs):
    for s in subs:
        for c in cols:
            if s in c.lower():
                return c
    return None


def to_ts(s):
    # handle ISO strings or epoch (s / ms)
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().median() > 1e12 else "s"
        return pd.to_datetime(s, unit=unit, utc=True)
    return pd.to_datetime(s, utc=True, errors="coerce")


def main(db):
    con = sqlite3.connect(db)
    tabs = pd.read_sql("select name from sqlite_master where type='table'", con)["name"].tolist()
    print("TABLES:", tabs)
    for t in tabs:
        c = pd.read_sql(f"select * from {t} limit 0", con).columns.tolist()
        print(f"  {t}: {c}")

    # ---- fills ----
    fills = pd.read_sql("select * from fills", con)
    fc = fills.columns.tolist()
    ts_c = find(fc, "ts", "time", "created"); px_c = find(fc, "price")
    ct_c = find(fc, "count", "qty", "size", "quantity"); sd_c = find(fc, "side")
    ac_c = find(fc, "action", "taker"); tk_c = find(fc, "ticker", "market")
    print(f"\nfills cols -> ts={ts_c} price={px_c} count={ct_c} side={sd_c} action={ac_c} ticker={tk_c}")
    fills["t"] = to_ts(fills[ts_c])
    fills["price"] = pd.to_numeric(fills[px_c], errors="coerce")
    fills["q"] = pd.to_numeric(fills[ct_c], errors="coerce").fillna(1)
    fills["ticker"] = fills[tk_c].astype(str)
    side = fills[sd_c].astype(str).str.lower() if sd_c else "yes"
    act = fills[ac_c].astype(str).str.lower() if ac_c else "buy"
    buy = act.str.contains("buy")
    yes = side.str.contains("yes") if sd_c else True
    fills["qsign"] = np.where(buy == yes, 1, -1)      # buy+yes or sell+no -> long yes
    fills = fills.dropna(subset=["t", "price"]).sort_values("t")

    # ---- fair value series: prefer book mid, else theo ----
    fair = None; src = None
    if "kalshi_book" in tabs:
        b = pd.read_sql("select * from kalshi_book", con)
        bc = b.columns.tolist()
        bid = find(bc, "yes_bid", "bid"); ask = find(bc, "yes_ask", "ask")
        bts = find(bc, "ts", "time"); btk = find(bc, "ticker", "market")
        if bid and ask and bts:
            b["t"] = to_ts(b[bts]); b["ticker"] = b[btk].astype(str)
            b["fair"] = (pd.to_numeric(b[bid], errors="coerce") + pd.to_numeric(b[ask], errors="coerce")) / 2
            fair = b.dropna(subset=["t", "fair"])[["ticker", "t", "fair"]]; src = "book mid"
    if fair is None and "theo_state" in tabs:
        th = pd.read_sql("select * from theo_state", con)
        thc = th.columns.tolist()
        thv = find(thc, "mid", "theo"); tts = find(thc, "ts", "time"); ttk = find(thc, "ticker", "market")
        th["t"] = to_ts(th[tts]); th["ticker"] = th[ttk].astype(str)
        th["fair"] = pd.to_numeric(th[thv], errors="coerce")
        fair = th.dropna(subset=["t", "fair"])[["ticker", "t", "fair"]]; src = f"theo ({thv})"
    print(f"fair source: {src},  {len(fair)} ticks,  {len(fills)} fills")
    fair = fair.sort_values("t")

    # ---- fair at fill time and at horizons (asof, by ticker) ----
    def fair_at(shift_s):
        q = fills[["ticker", "t"]].copy()
        q["t"] = q["t"] + pd.to_timedelta(shift_s, "s")
        q = q.sort_values("t")
        m = pd.merge_asof(q, fair, on="t", by="ticker", direction="backward",
                          tolerance=pd.Timedelta("20min"))
        return m["fair"].to_numpy()

    fills = fills.reset_index(drop=True)
    F0 = fair_at(0)
    fills["spread"] = fills["qsign"] * (F0 - fills["price"])
    for h in HORIZONS:
        Fh = fair_at(h)
        fills[f"mk{h}"] = fills["qsign"] * (Fh - F0)
        fills[f"net{h}"] = fills["spread"] + fills[f"mk{h}"]

    # ---- report (contract-weighted, in cents) ----
    def agg(df, label):
        w = df["q"].to_numpy(); n = w.sum()
        if n == 0:
            return
        sp = np.average(df["spread"], weights=w) * 100
        row = f"  {label:16} ct={n:>8.0f}  spread={sp:+6.2f}c | "
        for h in HORIZONS:
            mk = np.average(df[f"mk{h}"], weights=w) * 100
            nt = np.average(df[f"net{h}"], weights=w) * 100
            row += f"{h}s:adv={mk:+5.2f}/net={nt:+5.2f}  "
        print(row)

    print("\n### MARKOUT DECOMPOSITION (per contract, cents) ###")
    print("  spread>0 = you make markets;  adv = fair drift after fill (neg=picked off);  net = spread+adv")
    agg(fills, "ALL")
    agg(fills[fills["qsign"] > 0], "long-yes (buys)")
    agg(fills[fills["qsign"] < 0], "short-yes (sells)")
    print("\n  -> SELL-side net>0 & BUY-side net<0 confirms the counterparty asymmetry is a real "
          "making edge, and the directional bleed is an inventory-mgmt problem, not absence of edge.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 analysis/markout.py <recorder_db.db>")
        sys.exit(1)
    main(sys.argv[1])
