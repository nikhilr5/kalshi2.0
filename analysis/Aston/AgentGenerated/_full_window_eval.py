"""Full 16-day gate evaluation.

Sweeps thresholds for adv_60s_bp, theo_drift_adv, quote_age.
Compares symmetric vs side-aware gates.
Real OOS: fit on first 12 days, score on last 4.
Also re-runs the ETH momentum test on the full window.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("/Users/nikhilr5/Desktop/Kalshi2.0/analysis/Aston/AgentGenerated/_cache")
f = pd.read_pickle(CACHE / "fills_full_5may15_to_5may30.pkl")
print(f"loaded {len(f):,} fills, {f['ticker'].nunique()} tickers")

# Drop fills without outcome or theo
f = f.dropna(subset=["pnl_settle_c", "theo", "spot"]).copy()
dates = sorted(f["date"].unique())
N_DAYS = len(dates)
TOTAL = f["pnl_settle_c"].sum() / 100
PERDAY = TOTAL / N_DAYS
print(f"after dropna: {len(f):,} fills, {N_DAYS} days "
      f"({dates[0]} -> {dates[-1]})")
print(f"baseline: ${PERDAY:+.2f}/day  (total ${TOTAL:+.2f})")
print(f"hit-rate: {(f['pnl_settle_c']>0).mean():.3f}  "
      f"per-fill mean: {f['pnl_settle_c'].mean():+.2f}c  "
      f"std: {f['pnl_settle_c'].std():.1f}c")

# coverage of new features
print(f"\nfeature coverage:")
print(f"  adv60_bp:        {f['adv60_bp'].notna().sum():,} / {len(f):,}")
print(f"  theo_drift_adv:  {f['theo_drift_adv'].notna().sum():,} / {len(f):,}")
print(f"  qage_s:          {f['qage_s'].notna().sum():,} / {len(f):,}")

RNG = np.random.default_rng(42)


def gate(label, mask, df=None):
    df = df if df is not None else f
    drp = df[mask]
    by_day = drp.groupby("date")["pnl_settle_c"].sum() / 100.0
    full = pd.Index(sorted(df["date"].unique()))
    by_day = by_day.reindex(full, fill_value=0.0)
    arr = -by_day.values
    idx = RNG.integers(0, len(arr), size=(5000, len(arr)))
    boot = arr[idx].mean(axis=1)
    n = int(mask.sum())
    keep = ~mask
    return dict(
        label=label, n=n, pct=100*n/len(df),
        drop_d=float(drp["pnl_settle_c"].sum()/100),
        drop_c=float(drp["pnl_settle_c"].mean()) if n else 0.0,
        drop_hit=float((drp["pnl_settle_c"]>0).mean()) if n else 0.0,
        sav=float(arr.mean()),
        lo=float(np.quantile(boot, 0.025)),
        hi=float(np.quantile(boot, 0.975)),
        pos_days=int((arr > 0).sum()),
        n_days=len(arr),
        kept_d_perday=float(df[keep]["pnl_settle_c"].sum()/100/N_DAYS),
    )


def show(label, mask):
    r = gate(label, mask)
    print(f"  {r['label']:<48} {r['n']:>6} ({r['pct']:>4.1f}%) "
          f"drop$={r['drop_d']:>+7.1f}  drop_c={r['drop_c']:>+5.1f}  hit={r['drop_hit']:.2f}  "
          f"SAVE=${r['sav']:>+5.2f}/d  CI=[${r['lo']:>+5.2f},${r['hi']:>+5.2f}]  "
          f"{r['pos_days']}/{r['n_days']}d  kept=${r['kept_d_perday']:>+5.2f}/d")
    return r


# ============================================================
print("\n" + "="*120)
print("1.  THRESHOLD SWEEP — adv_60s_bp (signed: positive = adverse pre-fill move)")
print("="*120)
for thr in [3, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
    show(f"any side, adv_60s_bp > {thr}", f["adv60_bp"] > thr)


print("\n" + "="*120)
print("2.  THRESHOLD SWEEP — theo_drift_adv (cents, signed adverse drift in theo over 60s)")
print("="*120)
for thr in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
    show(f"any side, theo_drift_adv > {thr*100:.0f}c", f["theo_drift_adv"] > thr)


print("\n" + "="*120)
print("3.  THRESHOLD SWEEP — quote_age_s (any side)")
print("="*120)
for thr in [5, 10, 15, 20, 30, 45, 60, 90, 120]:
    show(f"any side, quote_age > {thr}s", f["qage_s"] > thr)


print("\n" + "="*120)
print("4.  COMBINED — best-of-three thresholds")
print("="*120)
combos = [
    ("adv60>8 OR qage>30",   (f["adv60_bp"] > 8)  | (f["qage_s"] > 30)),
    ("adv60>10 OR qage>30",  (f["adv60_bp"] > 10) | (f["qage_s"] > 30)),
    ("adv60>8 OR qage>60",   (f["adv60_bp"] > 8)  | (f["qage_s"] > 60)),
    ("drift>8c OR qage>30",  (f["theo_drift_adv"] > 0.08) | (f["qage_s"] > 30)),
    ("drift>10c OR qage>30", (f["theo_drift_adv"] > 0.10) | (f["qage_s"] > 30)),
    ("drift>8c OR adv60>8 OR qage>30",
        (f["theo_drift_adv"] > 0.08) | (f["adv60_bp"] > 8) | (f["qage_s"] > 30)),
    ("drift>10c OR adv60>10 OR qage>60",
        (f["theo_drift_adv"] > 0.10) | (f["adv60_bp"] > 10) | (f["qage_s"] > 60)),
]
for lbl, m in combos:
    show(lbl, m)


# ============================================================
# 5. REAL OOS: fit on first 12 days, score on last 4
# ============================================================
print("\n" + "="*120)
print("5.  REAL OOS — fit on first 12 days (5/15..5/26), score on last 4 (5/27..5/30)")
print("="*120)
fit_dates = dates[:12]
oos_dates = dates[12:]
f_fit = f[f["date"].isin(fit_dates)]
f_oos = f[f["date"].isin(oos_dates)]
print(f"  fit: {len(f_fit):,} fills, {len(fit_dates)} days, "
      f"baseline ${f_fit['pnl_settle_c'].sum()/100/len(fit_dates):+.2f}/day")
print(f"  oos: {len(f_oos):,} fills, {len(oos_dates)} days, "
      f"baseline ${f_oos['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/day")

# Search best threshold for adv_60s_bp on fit only
print(f"\n  --- single-feature OOS: adv_60s_bp ---")
best = None
for thr in [5, 6, 7, 8, 9, 10, 12, 15, 20]:
    m = f_fit["adv60_bp"] > thr
    sav = -f_fit[m]["pnl_settle_c"].sum() / 100 / len(fit_dates)
    if best is None or sav > best[1]:
        best = (thr, sav)
thr, fit_sav = best
print(f"  best fit threshold: adv60>{thr}bp  -> fit_save=${fit_sav:+.2f}/d")
m_oos = f_oos["adv60_bp"] > thr
oos_sav = -f_oos[m_oos]["pnl_settle_c"].sum() / 100 / len(oos_dates)
print(f"  OOS at same threshold: dropped {m_oos.sum()} fills, "
      f"oos_save=${oos_sav:+.2f}/d")
print(f"  OOS kept P&L: ${f_oos[~m_oos]['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/d "
      f"vs baseline ${f_oos['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/d")

# Same for theo_drift_adv
print(f"\n  --- single-feature OOS: theo_drift_adv ---")
best = None
for thr in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
    m = f_fit["theo_drift_adv"] > thr
    sav = -f_fit[m]["pnl_settle_c"].sum() / 100 / len(fit_dates)
    if best is None or sav > best[1]:
        best = (thr, sav)
thr, fit_sav = best
print(f"  best fit threshold: drift>{thr*100:.0f}c  -> fit_save=${fit_sav:+.2f}/d")
m_oos = f_oos["theo_drift_adv"] > thr
oos_sav = -f_oos[m_oos]["pnl_settle_c"].sum() / 100 / len(oos_dates)
print(f"  OOS at same threshold: dropped {m_oos.sum()} fills, "
      f"oos_save=${oos_sav:+.2f}/d")

# Stacked
print(f"\n  --- stacked OOS: adv60>8 OR qage>30 ---")
m_fit = (f_fit["adv60_bp"] > 8) | (f_fit["qage_s"] > 30)
m_oos = (f_oos["adv60_bp"] > 8) | (f_oos["qage_s"] > 30)
print(f"  fit:  drop {m_fit.sum()} ({100*m_fit.mean():.1f}%), save=${-f_fit[m_fit]['pnl_settle_c'].sum()/100/len(fit_dates):+.2f}/d")
print(f"  oos:  drop {m_oos.sum()} ({100*m_oos.mean():.1f}%), save=${-f_oos[m_oos]['pnl_settle_c'].sum()/100/len(oos_dates):+.2f}/d")


# ============================================================
# 6. ETH momentum test on the FULL window
# ============================================================
print("\n" + "="*120)
print("6.  ETH MOMENTUM (full window): does pre-fill 60s spot return predict")
print("    the 15-min forward return?  (=#1 from earlier, but with all data)")
print("="*120)

# Need forward spot return — fill["spot"] -> spot @ close_time
# All we have on the fill row is spot, ret60_bp, and the outcome.
# For the forward-spot test we'd need spot @ close_time. The TWAP (mean
# of last 60s) is a proxy. Use it.
f["fwd_ret_bp"] = (f["twap"] / f["spot"] - 1) * 1e4
m = f.dropna(subset=["ret60_bp", "fwd_ret_bp"])
m = m[(m["ret60_bp"].abs() < 100) & (m["fwd_ret_bp"].abs() < 200)]
print(f"  n: {len(m):,}")
print(f"  corr(ret60_bp, fwd_ret_bp): {m[['ret60_bp','fwd_ret_bp']].corr().iloc[0,1]:+.4f}")
m["ret60_bin"] = pd.cut(m["ret60_bp"], bins=[-1e6,-15,-10,-5,-2,2,5,10,15,1e6],
                         labels=["<-15","-15..-10","-10..-5","-5..-2","-2..2",
                                 "2..5","5..10","10..15",">15"])
g = m.groupby("ret60_bin", observed=True).agg(
    n=("fwd_ret_bp","size"),
    fwd_mean_bp=("fwd_ret_bp","mean"),
    fwd_median_bp=("fwd_ret_bp","median"),
).reset_index()
print(g.round(2).to_string(index=False))


# ============================================================
# 7. Per-day breakdown of recommended gate
# ============================================================
print("\n" + "="*120)
print("7.  RECOMMENDED GATE per-day P&L breakdown")
print("    gate = adv_60s_bp > 8 OR quote_age > 30s")
print("="*120)
mask = (f["adv60_bp"] > 8) | (f["qage_s"] > 30)
by_day = f.groupby(["date", mask.rename("dropped")])["pnl_settle_c"].sum().unstack(fill_value=0) / 100.0
by_day.columns = ["kept_$", "dropped_$"]
by_day["save_$"] = -by_day["dropped_$"]
by_day["new_$"] = by_day["kept_$"]
print(by_day.to_string(float_format=lambda x: f"{x:+.2f}"))
print(f"\n  pos save days: {(by_day['save_$']>0).sum()}/{len(by_day)}")
print(f"  mean daily save: ${by_day['save_$'].mean():+.2f}")
print(f"  median daily save: ${by_day['save_$'].median():+.2f}")


# ============================================================
# 8. Inventory neutrality check
# ============================================================
print("\n" + "="*120)
print("8.  INVENTORY NEUTRALITY (recommended gate)")
print("="*120)
keep = ~mask
inv_base = f.groupby("ticker").apply(
    lambda d: ((d["action"]=="buy").astype(int) - (d["action"]=="sell").astype(int)).sum(),
    include_groups=False)
inv_gate = f[keep].groupby("ticker").apply(
    lambda d: ((d["action"]=="buy").astype(int) - (d["action"]=="sell").astype(int)).sum(),
    include_groups=False)
print(f"  baseline buys/sells: {(f['action']=='buy').sum()}/{(f['action']=='sell').sum()}")
print(f"  post-gate buys/sells: {((f['action']=='buy') & keep).sum()}/{((f['action']=='sell') & keep).sum()}")
print(f"  net inv per market: baseline {inv_base.mean():+.2f}, gate {inv_gate.mean():+.2f}")


# ============================================================
# 9. Side-by-side: gate drops by side
# ============================================================
print("\n" + "="*120)
print("9.  SIDE-BY-SIDE: who gets filtered, who doesn't")
print("    (catching adverse-selection fairly on both sides?)")
print("="*120)
for action in ["buy", "sell"]:
    sub = f[f["action"] == action]
    sub_drop = sub[(sub["adv60_bp"] > 8) | (sub["qage_s"] > 30)]
    sub_keep = sub[~((sub["adv60_bp"] > 8) | (sub["qage_s"] > 30))]
    print(f"  {action}: total n={len(sub):,}, "
          f"drop n={len(sub_drop):,} ({100*len(sub_drop)/len(sub):.1f}%)  "
          f"drop_c/fill={sub_drop['pnl_settle_c'].mean():+.2f}  "
          f"keep_c/fill={sub_keep['pnl_settle_c'].mean():+.2f}")
