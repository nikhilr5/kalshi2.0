"""Post-fix v2 fill counts + markouts, split by buy/sell.

Markout = mid move in our favor after a fill, in cents per contract:
  buy : (mid_{t+h} - fill_price) * 100
  sell: (fill_price - mid_{t+h}) * 100
Positive = we got a good fill (mid moved toward/past us). Negative =
adverse selection (we were picked off, mid moved against us).

Guardrails:
  - mid from kalshi_book, drop crossed/empty books
  - merge_asof(backward) fill -> book at ts+h
  - MASK markouts whose horizon runs past the market's close (no stale
    post-expiry book scoring). seconds_to_expiry derived from the
    ticker's close time (parsed from the ticker) minus fill ts.
  - per-contract weighting via fill `count`
  - taker fills flagged separately (we pay spread on those; different
    population from passive MM fills)
"""

import re
import sqlite3
import zoneinfo
from pathlib import Path

import numpy as np
import pandas as pd

ET = zoneinfo.ZoneInfo("America/New_York")

BT = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting")
DAYS = {
    "26JUN10": BT / "_s3_cache/KXETH15M-26JUN10.db",
    "26JUN11": BT / "data/KXETH15M-26JUN11.db",
    "26JUN12": BT / "data/KXETH15M-26JUN12.db",   # partial
}
HORIZONS = [30, 60, 120]
_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}


def close_ts_from_ticker(tkr):
    # KXETH15M-26JUN102015-15 -> 20:15 ET on 6/10 = 00:15 UTC 6/11.
    # The ticker date/time are in ET (exchange local); convert to UTC.
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})-", tkr)
    if not m:
        return pd.NaT
    yy, mon, dd, hh, mm = m.groups()
    local = pd.Timestamp(2000+int(yy), _MON[mon], int(dd), int(hh), int(mm), tz=ET)
    return local.tz_convert("UTC")


def load_day(db):
    c = sqlite3.connect(str(db))
    f = pd.read_sql("SELECT ts,ticker,action,side,price,count,is_taker FROM fills", c)
    b = pd.read_sql("SELECT ts,ticker,yes_bid,yes_ask FROM kalshi_book", c)
    c.close()
    if f.empty or b.empty:
        return None, None
    for d in (f, b):
        d["ts"] = pd.to_datetime(d["ts"], utc=True, format="ISO8601")
    b = b[(b.yes_bid > 0) & (b.yes_ask > 0) & (b.yes_ask >= b.yes_bid)].copy()
    b["mid"] = (b.yes_bid + b.yes_ask) / 2
    f["close_ts"] = f["ticker"].map(close_ts_from_ticker)
    f["sec_to_exp"] = (f["close_ts"] - f["ts"]).dt.total_seconds()
    f = f.sort_values("ts").reset_index(drop=True)
    b = b.sort_values("ts")
    return f, b


def add_markouts(f, b):
    for h in HORIZONS:
        tgt = f[["ts", "ticker", "action", "price", "sec_to_exp"]].copy()
        tgt["mk_ts"] = tgt["ts"] + pd.Timedelta(seconds=h)
        tgt = tgt.sort_values("mk_ts")
        m = pd.merge_asof(tgt, b[["ts", "ticker", "mid"]],
                          left_on="mk_ts", right_on="ts", by="ticker",
                          direction="backward",
                          tolerance=pd.Timedelta(seconds=30))
        m = m.sort_index()
        mk = np.where(m["action"].values == "buy",
                      (m["mid"].values - m["price"].values) * 100,
                      (m["price"].values - m["mid"].values) * 100)
        # mask: horizon must not exceed remaining time to expiry
        mk = np.where(f["sec_to_exp"].values >= h, mk, np.nan)
        f[f"mk_{h}"] = mk
    return f


def wmean(v, w):
    mask = v.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


rows = []
for day, db in DAYS.items():
    if not db.exists():
        continue
    f, b = load_day(db)
    if f is None:
        continue
    f = add_markouts(f, b)
    span_h = (f["ts"].max() - f["ts"].min()).total_seconds() / 3600
    for side in ("buy", "sell"):
        s = f[f.action == side]
        rec = {"day": day, "side": side, "span_h": span_h,
               "n_fills": len(s), "n_contracts": s["count"].sum(),
               "taker_frac": s["is_taker"].mean()}
        for h in HORIZONS:
            rec[f"mk_{h}"] = wmean(s[f"mk_{h}"], s["count"])
            rec[f"n_{h}"] = int(s[f"mk_{h}"].notna().sum())
        rows.append(rec)

df = pd.DataFrame(rows)

print("=" * 88)
print("POST-FIX v2 — FILL COUNTS + MARKOUTS (cents/contract, count-weighted), by side")
print("=" * 88)
print(f"  {'day':<9}{'side':<6}{'fills':>7}{'ctrs':>7}{'tk%':>6}"
      f"{'mk30':>9}{'mk60':>9}{'mk120':>9}")
for r in df.itertuples():
    print(f"  {r.day:<9}{r.side:<6}{r.n_fills:>7}{int(r.n_contracts):>7}"
          f"{100*r.taker_frac:>5.0f}%{r.mk_30:>+9.2f}{r.mk_60:>+9.2f}{r.mk_120:>+9.2f}")

print("\n  Aggregate across ALL days (contract-weighted markouts):")
print(f"  {'side':<6}{'fills':>7}{'ctrs':>8}{'tk%':>6}{'mk30':>9}{'mk60':>9}{'mk120':>9}")
# recompute aggregate from the raw per-fill rows for correct weighting
allf = []
for day, db in DAYS.items():
    if not db.exists():
        continue
    f, b = load_day(db)
    f = add_markouts(f, b)
    f["day"] = day
    allf.append(f)
allf = pd.concat(allf, ignore_index=True)
for side in ("buy", "sell"):
    s = allf[allf.action == side]
    line = (f"  {side:<6}{len(s):>7}{int(s['count'].sum()):>8}"
            f"{100*s['is_taker'].mean():>5.0f}%")
    for h in HORIZONS:
        line += f"{wmean(s[f'mk_{h}'], s['count']):>+9.2f}"
    print(line)

print("\n  Full-day-only aggregate (excl. partial 6/12):")
fd = allf[allf["day"] != "26JUN12"]
for side in ("buy", "sell"):
    s = fd[fd.action == side]
    line = f"  {side:<6}{len(s):>7}{int(s['count'].sum()):>8}{100*s['is_taker'].mean():>5.0f}%"
    for h in HORIZONS:
        line += f"{wmean(s[f'mk_{h}'], s['count']):>+9.2f}"
    print(line)

# taker vs passive split on the 60s markout (taker fills pay spread)
print("\n  60s markout split by taker/passive (all days, contract-wtd):")
for side in ("buy", "sell"):
    for tk, lab in [(0, "passive"), (1, "taker")]:
        s = allf[(allf.action == side) & (allf.is_taker == tk)]
        if len(s) == 0:
            continue
        print(f"    {side:<5} {lab:<8} n={len(s):>5}  mk60={wmean(s['mk_60'], s['count']):+.2f}c")
