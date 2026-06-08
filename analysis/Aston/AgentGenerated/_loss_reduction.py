"""Where do per-fill losses concentrate, and what one gate cuts the
worst decile without nuking the rest of the book?"""

import sys, pickle, gc
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/nikhilr5/Desktop/Kalshi2.0")
CACHE = ROOT / "analysis/Aston/AgentGenerated/_cache"

f = pd.read_pickle(CACHE / "master_fills.pkl")
f = f[f["date"] >= pd.Timestamp("2026-05-15").date()].copy()
f = f.dropna(subset=["pnl_settle_c", "outcome", "theo"])
N_DAYS = f["date"].nunique()
N = len(f)
TOTAL_BASE = f["pnl_settle_c"].sum() / 100
PERDAY_BASE = TOTAL_BASE / N_DAYS
print(f"loaded {N:,} fills across {N_DAYS} days, "
      f"{f['ticker'].nunique()} unique markets")
print(f"baseline P&L: ${TOTAL_BASE:+.2f} total = ${PERDAY_BASE:+.2f}/day")
print(f"per-fill: mean {f['pnl_settle_c'].mean():+.2f}c  "
      f"std {f['pnl_settle_c'].std():.1f}c  "
      f"hit-rate {(f['pnl_settle_c'] > 0).mean():.3f}")

RNG = np.random.default_rng(42)


def boot_day(arr, B=5000):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(arr), size=(B, len(arr)))
    m = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def gate_eval(label, drop_mask, df=None):
    if df is None: df = f
    dropped = df[drop_mask]
    kept = df[~drop_mask]
    n_drop = drop_mask.sum()
    pct_drop = 100 * n_drop / len(df)
    # per-day savings = -dropped$/day
    by_day_d = dropped.groupby("date")["pnl_settle_c"].sum() / 100.0
    full = pd.Index(sorted(df["date"].unique()))
    by_day_d = by_day_d.reindex(full, fill_value=0.0)
    sav_d, lo, hi = boot_day(-by_day_d.values)
    print(f"\n  {label}")
    print(f"    drops {n_drop:,} ({pct_drop:.1f}%); "
          f"drop $: {dropped['pnl_settle_c'].sum()/100:+.2f}  "
          f"({dropped['pnl_settle_c'].mean():+.2f}c/fill, "
          f"hit {(dropped['pnl_settle_c']>0).mean():.3f})")
    print(f"    keep $: {kept['pnl_settle_c'].sum()/100:+.2f}  "
          f"=> ${kept['pnl_settle_c'].sum()/100/N_DAYS:+.2f}/day")
    print(f"    SAVINGS: ${sav_d:+.2f}/day  CI=[${lo:+.2f}, ${hi:+.2f}]")
    return sav_d, lo, hi


# =============================================================
# 1) Loss decomposition
# =============================================================
print("\n" + "="*78)
print("1.  LOSS DECOMPOSITION")
print("="*78)

print("\n(a) action x theo bin  [n, mean_c, total_$, hit]")
f["theo_bin"] = pd.cut(f["theo"], bins=np.arange(0, 1.01, 0.1),
                        labels=[f"{int(x*100):02d}-{int(x*100)+10:02d}" for x in np.arange(0, 1, 0.1)],
                        include_lowest=True)
pv = (f.groupby(["action","theo_bin"], observed=True)
        .agg(n=("pnl_settle_c","size"),
             mean_c=("pnl_settle_c","mean"),
             total_d=("pnl_settle_c", lambda x: x.sum()/100),
             hit=("pnl_settle_c", lambda x: (x>0).mean()))
        .reset_index())
print(pv.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n(b) action x time-to-close")
f["ttc_bin"] = pd.cut(f["mins_to_close"],
                       bins=[-0.1, 1, 2, 5, 10, 16],
                       labels=["<1m","1-2m","2-5m","5-10m","10-15m"])
pv = (f.groupby(["action","ttc_bin"], observed=True)
        .agg(n=("pnl_settle_c","size"),
             mean_c=("pnl_settle_c","mean"),
             total_d=("pnl_settle_c", lambda x: x.sum()/100))
        .reset_index())
print(pv.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n(c) hour-of-day (CT)")
pv = (f.groupby("hour_ct")
        .agg(n=("pnl_settle_c","size"),
             mean_c=("pnl_settle_c","mean"),
             total_d=("pnl_settle_c", lambda x: x.sum()/100))
        .reset_index())
print(pv.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

print("\n(d) action x |z| bucket")
f["abs_z"] = f["z"].abs()
f["abs_z_bin"] = pd.cut(f["abs_z"], bins=[0, 0.25, 0.5, 0.75, 1.0, 1.5, 5.0])
pv = (f.groupby(["action","abs_z_bin"], observed=True)
        .agg(n=("pnl_settle_c","size"),
             mean_c=("pnl_settle_c","mean"),
             total_d=("pnl_settle_c", lambda x: x.sum()/100))
        .reset_index())
print(pv.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


# =============================================================
# 2) Spot-momentum at fill
# =============================================================
print("\n" + "="*78)
print("2.  SPOT MOMENTUM AT FILL (adverse pre-fill move)")
print("="*78)

with open(CACHE / "all_26MAY15.pkl", "rb") as fh:
    all_data = pickle.load(fh)
spot = all_data["spot"][["ts","price"]].sort_values("ts").reset_index(drop=True)
events = all_data["events"][["ts","order_id","ticker","event_type","side","action","price","client_order_id"]].copy()
del all_data; gc.collect()
print(f"spot ticks: {len(spot):,}; order events: {len(events):,}")

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
# Adverse sign = direction that hurts our position
adverse_sign = np.where(f["action"] == "buy", -1.0, +1.0)
f["adv_60s_bp"] = adverse_sign * f["ret_60s_bp"]

print("\n  buckets by adv_60s_bp (positive = adverse pre-fill move)")
f["adv_bin"] = pd.cut(f["adv_60s_bp"],
                       bins=[-1e9, -10, -5, -2, 0, 2, 5, 10, 1e9],
                       labels=["<-10","-10..-5","-5..-2","-2..0",
                               "0..2","2..5","5..10",">10"])
pv = (f.groupby(["action","adv_bin"], observed=True)
        .agg(n=("pnl_settle_c","size"),
             mean_c=("pnl_settle_c","mean"),
             total_d=("pnl_settle_c", lambda x: x.sum()/100))
        .reset_index())
print(pv.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


# =============================================================
# 3) Quote-age at fill
# =============================================================
print("\n" + "="*78)
print("3.  QUOTE AGE AT FILL")
print("="*78)

places = events[events["event_type"] == "placed"][["ts","order_id"]].rename(
    columns={"ts": "ts_placed"})
# fills have 'id' column from fills table, but the order id is...?
# Master fills schema: 'id' (fill PK), 'ticker', no order_id directly.
# Look at original fills DB row:
print("  fills columns:", f.columns.tolist()[:15])
# Looking at schema — fills has 'client_order_id'. Match to order_id via events.
# events.client_order_id <-> events.order_id mapping
co2oi = events[["client_order_id","order_id"]].dropna().drop_duplicates(
    subset="client_order_id")
print(f"  client_order_id->order_id rows: {len(co2oi):,}")
f2 = f.merge(co2oi, on="client_order_id", how="left", suffixes=("","_evt"))
print(f"  fills with order_id matched: {f2['order_id'].notna().sum():,}/{len(f2):,}")
# Now join place ts
f2 = f2.merge(places, on="order_id", how="left")
f2["quote_age_s"] = (f2["ts"] - f2["ts_placed"]).dt.total_seconds()
print(f"  fills with quote_age: {f2['quote_age_s'].notna().sum():,}")
print(f"  quote_age summary: median={f2['quote_age_s'].median():.2f}s  "
      f"p95={f2['quote_age_s'].quantile(0.95):.2f}s  "
      f"max={f2['quote_age_s'].max():.0f}s")

f["quote_age_s"] = f2["quote_age_s"].values
f["qa_bin"] = pd.cut(f["quote_age_s"],
                      bins=[-0.1, 1, 3, 10, 30, 100, 1e4],
                      labels=["<1s","1-3s","3-10s","10-30s","30-100s",">100s"])
print("\n  P&L by quote_age (action × qa_bin)")
pv = (f.dropna(subset=["quote_age_s"]).groupby(["action","qa_bin"], observed=True)
        .agg(n=("pnl_settle_c","size"),
             mean_c=("pnl_settle_c","mean"),
             total_d=("pnl_settle_c", lambda x: x.sum()/100))
        .reset_index())
print(pv.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

del events, places, spot, fb_back, fb; gc.collect()


# =============================================================
# 4) Gate backtests
# =============================================================
print("\n" + "="*78)
print("4.  GATE BACKTESTS — per-day savings with CI")
print("="*78)

print("\nBASELINE: ${:+.2f}/day (n={})".format(PERDAY_BASE, N))

# Gate 1: drop buy fills at theo<0.5
gate_eval("G1: drop BUY when theo < 0.5",
          (f["action"]=="buy") & (f["theo"] < 0.5))

# Gate 2: drop buy at theo<0.4 (less aggressive)
gate_eval("G2: drop BUY when theo < 0.4",
          (f["action"]=="buy") & (f["theo"] < 0.4))

# Gate 3: drop sell fills at theo>0.5 (mirror — should NOT help based on prior moneyness)
gate_eval("G3: drop SELL when theo > 0.5  (mirror; should not help)",
          (f["action"]=="sell") & (f["theo"] > 0.5))

# Gate 4: drop ATM (|z|<0.5) — the dead zone
gate_eval("G4: drop both sides when |z| < 0.5",
          (f["abs_z"] < 0.5))

# Gate 5: drop adverse-momentum BUYs (spot fell hard right before)
gate_eval("G5: drop BUY when adv_60s_bp > 5",
          (f["action"]=="buy") & (f["adv_60s_bp"] > 5))

# Gate 6: drop adverse-momentum SELLs
gate_eval("G6: drop SELL when adv_60s_bp > 5",
          (f["action"]=="sell") & (f["adv_60s_bp"] > 5))

# Gate 7: drop BOTH adverse momentum
gate_eval("G7: drop any fill when adv_60s_bp > 5",
          f["adv_60s_bp"] > 5)

# Gate 8: drop stale-quote fills (quote_age > 30s, the cancel-race victims)
gate_eval("G8: drop fills with quote_age > 30s",
          f["quote_age_s"] > 30)

# Gate 9: drop late-window BUY only (T < 2m and buy)
gate_eval("G9: drop BUY when mins_to_close < 2",
          (f["action"]=="buy") & (f["mins_to_close"] < 2))

# Gate 10: combo — drop BUY when theo<0.5 AND adv>5
gate_eval("G10: drop BUY when theo<0.5 AND adv_60s_bp > 5",
          (f["action"]=="buy") & (f["theo"] < 0.5) & (f["adv_60s_bp"] > 5))

# Gate 11: drop BUY when theo<0.5  (cleanest single-condition winner expected)
gate_eval("G11: drop BUY when theo < 0.45 (slightly tighter cutoff)",
          (f["action"]=="buy") & (f["theo"] < 0.45))

# Gate 12: stale-quote BUYs only
gate_eval("G12: drop BUY when quote_age > 30s",
          (f["action"]=="buy") & (f["quote_age_s"] > 30))


# =============================================================
# 5) Best-gate ROBUSTNESS: day-by-day breakdown
# =============================================================
print("\n" + "="*78)
print("5.  TOP-CANDIDATE GATE: per-day breakdown (consistency check)")
print("="*78)

best_mask = (f["action"]=="buy") & (f["theo"] < 0.5)
daily = (f.assign(drop=best_mask)
           .groupby(["date","drop"])["pnl_settle_c"]
           .sum().unstack(fill_value=0) / 100.0)
daily.columns = ["kept_$", "dropped_$"]
daily["savings_$"] = -daily["dropped_$"]
daily["total_kept_$"] = daily["kept_$"]
print(daily.to_string(float_format=lambda x: f"{x:+.2f}"))
print(f"\n  days savings > 0: {(daily['savings_$'] > 0).sum()}/{len(daily)}")
print(f"  median daily savings: ${daily['savings_$'].median():+.2f}")
print(f"  mean daily savings:   ${daily['savings_$'].mean():+.2f}")


# =============================================================
# 6) Verify gate is symmetric (impact on hit-rate, not just $$)
# =============================================================
print("\n" + "="*78)
print("6.  TOP GATE — sanity checks")
print("="*78)
g = (f["action"]=="buy") & (f["theo"] < 0.5)
g_buy = (f["action"]=="buy")
print(f"  All BUY fills: n={g_buy.sum():,}  total=${f[g_buy]['pnl_settle_c'].sum()/100:+.2f}  "
      f"mean={f[g_buy]['pnl_settle_c'].mean():+.2f}c  hit={(f[g_buy]['pnl_settle_c']>0).mean():.3f}")
print(f"  BUY @ theo<0.5: n={g.sum():,}  total=${f[g]['pnl_settle_c'].sum()/100:+.2f}  "
      f"mean={f[g]['pnl_settle_c'].mean():+.2f}c  hit={(f[g]['pnl_settle_c']>0).mean():.3f}")
print(f"  BUY @ theo>=0.5: n={(g_buy & ~g).sum():,}  total=${f[g_buy & ~g]['pnl_settle_c'].sum()/100:+.2f}  "
      f"mean={f[g_buy & ~g]['pnl_settle_c'].mean():+.2f}c  hit={(f[g_buy & ~g]['pnl_settle_c']>0).mean():.3f}")

# What % of dropped fills are 'stale'? cancel-race style?
stale = (f["quote_age_s"] > 5)
print(f"\n  fills with quote_age>5s: {stale.sum():,}/{f['quote_age_s'].notna().sum():,} "
      f"({100*stale.sum()/f['quote_age_s'].notna().sum():.1f}%)")
overlap = (g & stale).sum()
print(f"  BUY@theo<0.5 that are ALSO stale: {overlap:,} ({100*overlap/max(g.sum(),1):.1f}% of the gate)")
