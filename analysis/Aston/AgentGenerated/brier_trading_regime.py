"""Brier comparison restricted to the regime Aston actually trades (TTE >= 90s).

Row-aligned per-tick: merge_asof mid onto each theo row, then bucket. No
implied-vol filter (symmetric: both HAR and Market are evaluated on the
EXACT same rows). Ticker-clustered bootstrap CI.
"""

import gc
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import fetch_settlements_from_api, list_eligible_dbs, theo_vec
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
TRADE_TTE_S   = 90              # Aston auto-off threshold
N_BOOT        = 5_000
RNG           = np.random.default_rng(42)

_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
def _key(t):
    d = t.split('-')[1][:7]
    return (int(d[:2]), _MON[d[2:5]], int(d[5:7]))
cutoff_key = _key(f"X-{CUTOFF_DAY}")


def load_one(path):
    conn = sqlite3.connect(str(path))
    try:
        th = pd.read_sql(
            "SELECT ts, ticker, spot, strike, seconds_to_expiry, "
            "rv_15m, rv_30m, rv_4h, rv_24h FROM theo_state ORDER BY ts", conn)
        bk = pd.read_sql(
            "SELECT ts, ticker, yes_bid, yes_ask FROM kalshi_book ORDER BY ts", conn)
    finally:
        conn.close()
    th['ts'] = pd.to_datetime(th['ts'], utc=True, format='ISO8601')
    bk['ts'] = pd.to_datetime(bk['ts'], utc=True, format='ISO8601')
    return th, bk


# ----- load + align all files -----
files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY)
print(f"[load] {len(files)} db file(s)")

all_tickers = set()
for p in files:
    conn = sqlite3.connect(str(p))
    try:
        all_tickers.update(pd.read_sql("SELECT DISTINCT ticker FROM theo_state", conn)['ticker'].dropna().tolist())
    finally:
        conn.close()
api = KalshiAPI()
cache = Path(__file__).resolve().parent.parent / ".settlements_cache.json"
settle = fetch_settlements_from_api(list(all_tickers), api, cache_path=cache)

aligned_parts = []
for p in files:
    th, bk = load_one(p)
    th = th[(th['seconds_to_expiry'] > 0) & (th['seconds_to_expiry'] <= 900)]
    th = th[th['ticker'].map(lambda t: _key(t) >= cutoff_key)].copy()
    bk = bk[bk['ticker'].map(lambda t: _key(t) >= cutoff_key)].copy()
    th['outcome'] = th['ticker'].map(settle)
    bk['outcome'] = bk['ticker'].map(settle)
    th = th.dropna(subset=['outcome'])
    bk = bk.dropna(subset=['outcome'])
    if th.empty or bk.empty:
        continue

    th['forecasted_vol'] = (0.0314 + 0.4485*th['rv_15m'] + 0.1293*th['rv_30m']
                            + 0.1843*th['rv_4h']  + 0.1149*th['rv_24h'])
    th['har_theo'] = theo_vec(th['spot'], th['strike'], th['forecasted_vol'], th['seconds_to_expiry'])
    bk['mid'] = (bk['yes_bid'] + bk['yes_ask']) / 2

    th = th.sort_values(['ticker','ts'])
    bk = bk[['ts','ticker','mid']].sort_values(['ticker','ts'])
    m = pd.merge_asof(
        th[['ts','ticker','seconds_to_expiry','har_theo','outcome']],
        bk, on='ts', by='ticker', direction='backward',
        tolerance=pd.Timedelta(seconds=30),
    ).dropna(subset=['mid'])
    m['e_har'] = (m['har_theo'] - m['outcome'])**2
    m['e_mkt'] = (m['mid']      - m['outcome'])**2
    m['gap']   = m['e_har'] - m['e_mkt']
    aligned_parts.append(m[['ts','ticker','seconds_to_expiry','e_har','e_mkt','gap']])
    print(f"  {p.name}  aligned rows={len(m):,}  tickers={m['ticker'].nunique()}")
    del th, bk, m; gc.collect()

df = pd.concat(aligned_parts, ignore_index=True)


def boot_ci(per_ticker_vals, n=N_BOOT, alpha=0.05):
    arr = np.asarray(per_ticker_vals)
    if len(arr) == 0:
        return (np.nan,)*4
    boot = np.empty(n)
    for i in range(n):
        boot[i] = RNG.choice(arr, size=len(arr), replace=True).mean()
    return arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr)), np.quantile(boot, alpha/2), np.quantile(boot, 1-alpha/2)


def report(sub, label):
    if sub.empty:
        print(f"  {label}: empty"); return
    per_t = sub.groupby('ticker')['gap'].mean().values
    m, se, lo, hi = boot_ci(per_t)
    sig = "YES" if (lo > 0 or hi < 0) else "no — within noise"
    print(f"  {label:<22} n_rows={len(sub):>7,d}  n_tkr={len(per_t):>4d}  "
          f"HAR={sub['e_har'].mean():.4f}  Mkt={sub['e_mkt'].mean():.4f}  "
          f"gap={m:+.4f}  SE={se:.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  {sig}")


# ----- 1. Headline: TTE >= 90s -----
print("\n" + "="*82)
print(f"1.  HEADLINE — row-aligned Brier, KXETH15M ≥ {CUTOFF_DAY}, TTE ≥ {TRADE_TTE_S}s")
print("="*82)
print(f"  n unique markets (full)  : {df['ticker'].nunique()}")
report(df, "FULL (TTE 0-900s)")
report(df[df['seconds_to_expiry'] >= TRADE_TTE_S], f"TRADE (TTE ≥ {TRADE_TTE_S}s)")
report(df[df['seconds_to_expiry'] <  TRADE_TTE_S], f"BELOW (TTE < {TRADE_TTE_S}s)")


# ----- 2. TTE buckets within the trade regime -----
print("\n" + "="*82)
print(f"2.  TTE buckets within TRADE regime (TTE ≥ {TRADE_TTE_S}s), row-aligned")
print("="*82)
tr = df[df['seconds_to_expiry'] >= TRADE_TTE_S].copy()
tr['tte_bin'] = pd.cut(tr['seconds_to_expiry'],
                       bins=[90, 180, 420, 900],
                       labels=['90s-3m','3m-7m','7m-15m'],
                       include_lowest=True)

g2 = (tr.groupby('tte_bin', observed=True)
        .agg(n=('e_har','count'),
             n_tkr=('ticker','nunique'),
             har=('e_har','mean'),
             mkt=('e_mkt','mean'),
             gap=('gap','mean')))
print(g2.round(4).to_string())
print("\n  Ticker-clustered 95% CI per bucket:")
for b in g2.index:
    sub = tr[tr['tte_bin'] == b]
    per_t = sub.groupby('ticker')['gap'].mean().values
    if len(per_t) < 5:
        continue
    mean, se, lo, hi = boot_ci(per_t)
    sig = "YES" if (lo>0 or hi<0) else "no"
    print(f"    {b:>10s}  n_tkr={len(per_t):>3d}  gap={mean:+.4f}  SE={se:.4f}  "
          f"95% CI=[{lo:+.4f}, {hi:+.4f}]  sig={sig}")


# ----- 3. Sensitivity: 15s TTE buckets 0..180s -----
print("\n" + "="*82)
print("3.  SENSITIVITY — gap by 15s TTE bucket, 0..180s")
print("    (where does HAR start losing?)")
print("="*82)
edges = np.arange(0, 181, 15)
labels = [f"{edges[i]:>3d}-{edges[i+1]:>3d}s" for i in range(len(edges)-1)]
df_lo = df[df['seconds_to_expiry'] <= 180].copy()
df_lo['fine'] = pd.cut(df_lo['seconds_to_expiry'], bins=edges, labels=labels, include_lowest=True)

rows = []
for b in labels:
    sub = df_lo[df_lo['fine'] == b]
    if sub.empty:
        continue
    per_t = sub.groupby('ticker')['gap'].mean().values
    mean, se, lo, hi = boot_ci(per_t) if len(per_t) >= 5 else (sub['gap'].mean(), np.nan, np.nan, np.nan)
    rows.append({
        'tte_bin': b, 'n': len(sub), 'n_tkr': len(per_t),
        'har': sub['e_har'].mean(), 'mkt': sub['e_mkt'].mean(),
        'gap': mean, 'ci_lo': lo, 'ci_hi': hi,
        'sig': 'YES' if (not np.isnan(lo) and (lo>0 or hi<0)) else '',
    })
sens = pd.DataFrame(rows)
with pd.option_context('display.float_format', '{:+.4f}'.format):
    print(sens.to_string(index=False))


# ----- 4. One-paragraph verdict -----
print("\n" + "="*82)
print("VERDICT")
print("="*82)
tr_per_t = tr.groupby('ticker')['gap'].mean().values
m, se, lo, hi = boot_ci(tr_per_t)
direction = "underperforms" if m > 0 else "outperforms"
sig_word = "distinguishable from zero" if (lo > 0 or hi < 0) else "indistinguishable from zero"
print(f"  TTE ≥ {TRADE_TTE_S}s (Aston's actual quoting regime):")
print(f"    n_tickers={len(tr_per_t)}  HAR-Market gap = {m:+.4f}  "
      f"95% CI=[{lo:+.4f}, {hi:+.4f}]  ({sig_word})")
print(f"    HAR {direction} market by {abs(m)*1e4:.1f} bp Brier on the rows where we quote.")

out_dir = Path(__file__).resolve().parent
df.to_parquet(out_dir / "brier_aligned_full.parquet") if False else None  # off by default
sens.to_csv(out_dir / "brier_sensitivity_0_180s.csv", index=False)
print(f"\n  [saved] {out_dir/'brier_sensitivity_0_180s.csv'}")
