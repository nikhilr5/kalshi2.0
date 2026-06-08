"""Full 16-day window (2026-05-15 .. 2026-05-30) gate backtest.

Builds a slim per-fill dataframe from each per-day DB (no giant pickle).
For each fill computes:
  - spot @ fill, spot 60s back -> ret_60s_bp, adv_60s_bp
  - theo @ fill, theo 60s back -> theo_drift, adverse-direction theo_drift
  - quote_age_s
  - held-to-close P&L from TWAP settlement
Then evaluates a sweep of gates on the full window, plus a real OOS split.
"""

import sqlite3, sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0")
sys.path.insert(0, str(ROOT / "analysis"))
from utility import list_eligible_dbs, theo_vec

OUT = ROOT / "analysis/Aston/AgentGenerated/_cache/fills_full_5may15_to_5may30.pkl"

files = list_eligible_dbs("KXETH15M", "26MAY15")
print(f"loading {len(files)} dbs")

per_day = []
for path in files:
    t0 = time.time()
    conn = sqlite3.connect(str(path))
    try:
        fills = pd.read_sql("SELECT id, ts, ticker, action, side, count, price, "
                            "client_order_id, fee FROM fills ORDER BY ts", conn)
        theo  = pd.read_sql("SELECT ts, ticker, spot, sigma, theo, strike, "
                            "seconds_to_expiry FROM theo_state ORDER BY ts", conn)
        spot  = pd.read_sql("SELECT ts, price FROM spot_ticks ORDER BY ts", conn)
        ev    = pd.read_sql("SELECT ts, order_id, event_type, client_order_id "
                            "FROM order_events WHERE event_type='placed' ORDER BY ts", conn)
    finally:
        conn.close()

    if fills.empty:
        print(f"  {path.name}  no fills, skipping")
        continue
    for df in (fills, theo, spot, ev):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")

    # --- per-fill theo & spot at fill (backward asof, by ticker)
    theo_lite = theo.sort_values("ts")
    fills = pd.merge_asof(
        fills.sort_values("ts"), theo_lite,
        by="ticker", on="ts", direction="backward",
        tolerance=pd.Timedelta("10s"),
    )

    # --- spot 60s before fill
    fb = fills[["ts"]].copy()
    fb["back"] = fb["ts"] - pd.Timedelta(seconds=60)
    sp_back = pd.merge_asof(
        fb[["back"]].sort_values("back"),
        spot.rename(columns={"price": "sp_b"}).sort_values("ts"),
        left_on="back", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("30s"),
    ).drop(columns="ts")
    sp_back.index = fb.index
    fills["sp_b"] = sp_back["sp_b"].values
    fills["ret60_bp"] = (fills["spot"] / fills["sp_b"] - 1) * 1e4
    fills["adv60_bp"] = np.where(fills["action"] == "buy", -1.0, +1.0) * fills["ret60_bp"]

    # --- theo 60s before fill (per ticker, backward asof on theo_lite)
    fb2 = fills[["ticker", "ts"]].copy()
    fb2["back"] = fb2["ts"] - pd.Timedelta(seconds=60)
    theo_back = pd.merge_asof(
        fb2[["ticker", "back"]].sort_values("back"),
        theo_lite[["ts", "ticker", "theo"]].rename(columns={"theo": "theo_b"}).sort_values("ts"),
        by="ticker", left_on="back", right_on="ts",
        direction="backward", tolerance=pd.Timedelta("30s"),
    ).drop(columns="ts")
    theo_back.index = fb2.index
    fills["theo_b"] = theo_back["theo_b"].values
    fills["theo_drift"] = fills["theo"] - fills["theo_b"]
    # Adverse theo drift: BUY hurt by theo falling (drift<0); SELL hurt by theo rising
    fills["theo_drift_adv"] = np.where(
        fills["action"] == "buy", -fills["theo_drift"], +fills["theo_drift"])

    # --- quote age
    ev_lite = ev[["client_order_id", "ts", "order_id"]].rename(columns={"ts": "tp"})
    fills = fills.merge(ev_lite, on="client_order_id", how="left")
    fills["qage_s"] = (fills["ts"] - fills["tp"]).dt.total_seconds()

    # --- per-ticker close_time + TWAP outcome
    last_ts = theo.groupby("ticker").agg(
        last_ts=("ts", "max"),
        last_secs=("seconds_to_expiry", "min"),
        strike_meta=("strike", "first"),
    ).reset_index()
    last_ts["close_time"] = last_ts["last_ts"] + pd.to_timedelta(last_ts["last_secs"], unit="s")

    # TWAP over final 60s
    twap_rows = []
    for tk, g in theo.groupby("ticker"):
        ct = last_ts.loc[last_ts["ticker"] == tk, "close_time"].iloc[0]
        sk = last_ts.loc[last_ts["ticker"] == tk, "strike_meta"].iloc[0]
        w = spot[(spot["ts"] >= ct - pd.Timedelta(seconds=60)) & (spot["ts"] <= ct)]
        if len(w) < 5: continue
        twap = float(w["price"].mean())
        twap_rows.append((tk, twap, int(twap > sk), ct))
    setts = pd.DataFrame(twap_rows, columns=["ticker", "twap", "outcome", "close_time"])
    fills = fills.merge(setts, on="ticker", how="left")
    fills["secs_to_close"] = (fills["close_time"] - fills["ts"]).dt.total_seconds()
    fills["mins_to_close"] = fills["secs_to_close"] / 60.0

    # --- held-to-close P&L (cents)
    payoff = np.where(fills["action"] == "buy",
                      (fills["outcome"] - fills["price"]) * 100,
                      (fills["price"] - fills["outcome"]) * 100)
    fills["pnl_settle_c"] = payoff * fills["count"] - fills["fee"].fillna(0) * 100

    fills["date"] = fills["ts"].dt.tz_convert("America/Chicago").dt.date
    per_day.append(fills)
    print(f"  {path.name}  fills={len(fills):,}  with_outcome={fills['outcome'].notna().sum():,}  "
          f"({time.time()-t0:.1f}s)")
    del theo, spot, ev, theo_lite, sp_back, theo_back; gc.collect()

all_fills = pd.concat(per_day, ignore_index=True)
print(f"\ntotal fills: {len(all_fills):,}")
all_fills.to_pickle(OUT)
print(f"wrote {OUT}")
