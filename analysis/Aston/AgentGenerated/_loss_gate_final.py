"""Final gate proposals: rank the candidate single-condition gates and
test combinations for additive benefit. Also reports an OOS check
using days 2026-05-24..25 vs 15..23 as a poor-man's holdout."""

import sys, pickle, gc
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0")
CACHE = ROOT / "analysis/Aston/AgentGenerated/_cache"

f = pd.read_pickle(CACHE / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()].copy()
f = f.dropna(subset=["pnl_settle_c","outcome","theo"])

# Add momentum (60s pre-fill adverse bp)
with open(CACHE / "all_26MAY15.pkl", "rb") as fh:
    all_data = pickle.load(fh)
spot = all_data["spot"][["ts","price"]].sort_values("ts").reset_index(drop=True)
events = all_data["events"][["ts","order_id","ticker","event_type",
                              "client_order_id"]].copy()
del all_data; gc.collect()
f = f.sort_values("ts").reset_index(drop=True)

fb = f[["ts"]].copy()
fb["ts_back"] = fb["ts"] - pd.Timedelta(seconds=60)
fb_back = pd.merge_asof(
    fb[["ts_back"]].sort_values("ts_back"),
    spot.rename(columns={"price":"spot_back"}),
    left_on="ts_back", right_on="ts",
    direction="backward", tolerance=pd.Timedelta("30s"),
).drop(columns="ts")
fb_back.index = fb.index
f["spot_back"] = fb_back["spot_back"].values
f["ret_60s_bp"] = (f["spot"] / f["spot_back"] - 1) * 1e4
f["adv_60s_bp"] = np.where(f["action"]=="buy", -1.0, +1.0) * f["ret_60s_bp"]

# Add quote_age
places = events[events["event_type"]=="placed"][["ts","order_id"]].rename(
    columns={"ts":"ts_placed"})
co2oi = events[["client_order_id","order_id"]].dropna().drop_duplicates(subset="client_order_id")
f2 = f.merge(co2oi, on="client_order_id", how="left").merge(places, on="order_id", how="left")
f["quote_age_s"] = (f2["ts"] - f2["ts_placed"]).dt.total_seconds().values
del events, places, spot, fb_back; gc.collect()

dates = sorted(f["date"].unique())
N_DAYS = len(dates)
TOTAL_BASE = f["pnl_settle_c"].sum() / 100
PERDAY_BASE = TOTAL_BASE / N_DAYS
RNG = np.random.default_rng(42)
print(f"data: {len(f):,} fills, {N_DAYS} days "
      f"({dates[0]} -> {dates[-1]})")
print(f"baseline: ${PERDAY_BASE:+.2f}/day  (total ${TOTAL_BASE:+.2f})\n")


def boot_day(arr, B=5000):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(arr), size=(B, len(arr)))
    m = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def gate(label, mask, df=None):
    df = df if df is not None else f
    drp = df[mask]
    by_day = drp.groupby("date")["pnl_settle_c"].sum() / 100.0
    full = pd.Index(sorted(df["date"].unique()))
    by_day = by_day.reindex(full, fill_value=0.0)
    sav, lo, hi = boot_day(-by_day.values)
    n_drop = mask.sum()
    return dict(label=label, n=int(n_drop), pct=float(100*n_drop/len(df)),
                drop_d=float(drp["pnl_settle_c"].sum()/100),
                drop_c=float(drp["pnl_settle_c"].mean()) if n_drop else 0,
                drop_hit=float((drp["pnl_settle_c"]>0).mean()) if n_drop else 0,
                save=sav, lo=lo, hi=hi)


# ============================================================
# Single-condition gates ranked by savings
# ============================================================
print("="*78)
print("SINGLE-CONDITION GATES — ranked by median savings")
print("="*78)

candidates = [
    ("BUY when theo<0.30",          (f["action"]=="buy") & (f["theo"] < 0.30)),
    ("BUY when theo<0.40",          (f["action"]=="buy") & (f["theo"] < 0.40)),
    ("BUY when theo<0.45",          (f["action"]=="buy") & (f["theo"] < 0.45)),
    ("BUY when theo<0.50",          (f["action"]=="buy") & (f["theo"] < 0.50)),
    ("BUY when price<0.30 ",        (f["action"]=="buy") & (f["price"] < 0.30)),
    ("BUY when price<0.40 ",        (f["action"]=="buy") & (f["price"] < 0.40)),
    ("BUY when price<0.50 ",        (f["action"]=="buy") & (f["price"] < 0.50)),
    ("BUY when |z|<0.50",            (f["action"]=="buy") & (f["z"].abs() < 0.5)),
    ("BUY when |z|<0.75",            (f["action"]=="buy") & (f["z"].abs() < 0.75)),
    ("BUY when adv_60s_bp > 3",     (f["action"]=="buy") & (f["adv_60s_bp"] > 3)),
    ("BUY when adv_60s_bp > 5",     (f["action"]=="buy") & (f["adv_60s_bp"] > 5)),
    ("Any when adv_60s_bp > 5",     f["adv_60s_bp"] > 5),
    ("Any when quote_age > 30s",    f["quote_age_s"] > 30),
    ("Any when quote_age > 60s",    f["quote_age_s"] > 60),
    ("BUY when quote_age > 30s",    (f["action"]=="buy") & (f["quote_age_s"] > 30)),
    ("SELL when theo>0.70",         (f["action"]=="sell") & (f["theo"] > 0.70)),
    ("SELL when theo>0.85",         (f["action"]=="sell") & (f["theo"] > 0.85)),
]

results = [gate(lbl, m) for lbl, m in candidates]
df_res = pd.DataFrame(results).sort_values("save", ascending=False)
print(f"\n{'label':<35} {'n':>5} {'%':>5} {'drop$':>8} {'c/fill':>7} "
      f"{'hit':>5} {'$save/d':>9} {'95% CI':>20}")
for _, r in df_res.iterrows():
    print(f"{r['label']:<35} {r['n']:>5} {r['pct']:>5.1f} "
          f"{r['drop_d']:>+8.2f} {r['drop_c']:>+7.2f} {r['drop_hit']:>5.2f} "
          f"{r['save']:>+9.2f}  [{r['lo']:>+6.2f},{r['hi']:>+6.2f}]")


# ============================================================
# Stacked gates
# ============================================================
print("\n" + "="*78)
print("STACKED GATES (additive savings?)")
print("="*78)

stacked = [
    ("BUY<0.5 + Any adv>5",
     ((f["action"]=="buy") & (f["theo"]<0.5)) | (f["adv_60s_bp"] > 5)),
    ("BUY<0.5 + Any quote_age>30s",
     ((f["action"]=="buy") & (f["theo"]<0.5)) | (f["quote_age_s"] > 30)),
    ("BUY<0.5 + adv>5 + quote_age>30s",
     ((f["action"]=="buy") & (f["theo"]<0.5))
     | (f["adv_60s_bp"] > 5) | (f["quote_age_s"] > 30)),
    ("BUY<0.45 + adv>5 (any)",
     ((f["action"]=="buy") & (f["theo"]<0.45)) | (f["adv_60s_bp"] > 5)),
]
print(f"\n{'label':<45} {'n':>5} {'%':>5} {'drop$':>8} {'$save/d':>9} {'95% CI':>20}")
for lbl, m in stacked:
    r = gate(lbl, m)
    print(f"{r['label']:<45} {r['n']:>5} {r['pct']:>5.1f} "
          f"{r['drop_d']:>+8.2f} {r['save']:>+9.2f}  [{r['lo']:>+6.2f},{r['hi']:>+6.2f}]")


# ============================================================
# Pseudo-OOS: fit threshold on first N-3 days, evaluate on last 3
# ============================================================
print("\n" + "="*78)
print("PSEUDO-OOS — gate fit on first 8 days, evaluated on last 3 days")
print("="*78)

dates_arr = np.array(dates)
fit_dates = dates_arr[:-3]
oos_dates = dates_arr[-3:]
f_fit = f[f["date"].isin(fit_dates)]
f_oos = f[f["date"].isin(oos_dates)]
print(f"  fit days ({len(fit_dates)}): {fit_dates[0]} .. {fit_dates[-1]}")
print(f"  oos days ({len(oos_dates)}): {oos_dates[0]} .. {oos_dates[-1]}")
print(f"  fit baseline: ${f_fit['pnl_settle_c'].sum()/100/len(fit_dates):+.2f}/day")
print(f"  oos baseline: ${f_oos['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/day")

# Search best theo cutoff for BUY on fit, then apply to oos
best = None
for cut in np.arange(0.20, 0.55, 0.05):
    m = (f_fit["action"]=="buy") & (f_fit["theo"] < cut)
    sav = -f_fit[m]["pnl_settle_c"].sum() / 100 / len(fit_dates)
    if best is None or sav > best[1]: best = (cut, sav)
cut, fit_sav = best
print(f"\n  best fit cutoff: BUY when theo < {cut:.2f}  -> "
      f"fit savings ${fit_sav:+.2f}/day")

m_oos = (f_oos["action"]=="buy") & (f_oos["theo"] < cut)
oos_drop_d = f_oos[m_oos]["pnl_settle_c"].sum() / 100
oos_sav = -oos_drop_d / len(oos_dates)
print(f"  oos savings:    ${oos_sav:+.2f}/day  "
      f"(dropped {m_oos.sum()} fills, ${oos_drop_d:+.2f})")
print(f"  oos kept P&L:   ${f_oos[~m_oos]['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/day  "
      f"vs baseline ${f_oos['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/day")
