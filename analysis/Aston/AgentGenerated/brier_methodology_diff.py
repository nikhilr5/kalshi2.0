"""Resolve per-tick vs 5-snapshot Brier disagreement.

Memory-conscious: process one DB at a time, accumulate just the
columns we need for per-tick row-aligned Brier bucketed by TTE.
"""

import gc
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    brier_score, fetch_settlements_from_api, implied_sigma,
    list_eligible_dbs, theo_vec,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

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


# ----- pass 1: enumerate tickers across all files to fetch settlements once -----
files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY)
print(f"[load] {len(files)} file(s)")
all_tickers = set()
for p in files:
    conn = sqlite3.connect(str(p))
    try:
        ts = pd.read_sql("SELECT DISTINCT ticker FROM theo_state", conn)['ticker']
        all_tickers.update(ts.dropna().tolist())
    finally:
        conn.close()

api = KalshiAPI()
cache = Path(__file__).resolve().parent.parent / ".settlements_cache.json"
settle = fetch_settlements_from_api(list(all_tickers), api, cache_path=cache)


# ----- pass 2: per-file accumulation -----
# We accumulate:
#   loose_a, strict_a — for vol_forecasting replication (row-disjoint Brier)
#     contains per-row e_har OR e_mkt with a `kind` label
#   loose_b, strict_b — row-aligned per-tick (mid asof onto theo)
loose_a_har, loose_a_mkt = [], []
strict_a_har, strict_a_mkt = [], []
loose_b, strict_b = [], []

theo_rows_loose = book_rows_loose = 0
theo_rows_strict = book_rows_strict = 0
theo_size_per_tkr_loose = []
book_size_per_tkr_loose = []
theo_size_per_tkr_strict = []
book_size_per_tkr_strict = []

for p in files:
    th, bk = load_one(p)
    th = th[th['seconds_to_expiry'] > 0]
    th['outcome'] = th['ticker'].map(settle)
    bk['outcome'] = bk['ticker'].map(settle)
    th = th.dropna(subset=['outcome'])
    bk = bk.dropna(subset=['outcome'])

    th['forecasted_vol'] = (0.0314 + 0.4485*th['rv_15m'] + 0.1293*th['rv_30m']
                            + 0.1843*th['rv_4h']  + 0.1149*th['rv_24h'])
    th['har_theo'] = theo_vec(th['spot'], th['strike'], th['forecasted_vol'], th['seconds_to_expiry'])
    bk['mid'] = (bk['yes_bid'] + bk['yes_ask']) / 2

    # strict ticker-date filter (parse ticker date)
    th_strict = th[th['ticker'].map(lambda t: _key(t) >= cutoff_key)].copy()
    bk_strict = bk[bk['ticker'].map(lambda t: _key(t) >= cutoff_key)].copy()

    # ---- A. row-disjoint (replicate vol_forecasting) ----
    # HAR Brier over theo rows, Market Brier over book rows (with implied σ filter)
    # for market mid, need to backward-asof spot/strike/secs from theo to compute IV
    def mk_market_for_iv(theo_df, book_df):
        bk2 = book_df.sort_values(['ticker','ts']).copy()
        thm = theo_df[['ts','ticker','spot','strike','seconds_to_expiry']].sort_values(['ticker','ts'])
        bk2 = pd.merge_asof(bk2, thm, on='ts', by='ticker', direction='backward')
        bk2['implied_vol'] = implied_sigma(bk2['mid'], bk2['spot'], bk2['strike'], bk2['seconds_to_expiry'])
        return bk2[bk2['implied_vol'].between(0.05, 3.0)]

    bk_loose_iv  = mk_market_for_iv(th,        bk)
    bk_strict_iv = mk_market_for_iv(th_strict, bk_strict)

    loose_a_har.append(th[['har_theo','outcome']])
    loose_a_mkt.append(bk_loose_iv[['mid','outcome']])
    strict_a_har.append(th_strict[['har_theo','outcome']])
    strict_a_mkt.append(bk_strict_iv[['mid','outcome']])

    # ---- B. row-aligned per-tick (asof mid onto theo) ----
    def aligned(theo_df, book_df):
        if theo_df.empty or book_df.empty:
            return pd.DataFrame()
        thx = theo_df[(theo_df['seconds_to_expiry'] <= 900)].sort_values(['ticker','ts']).copy()
        bkx = book_df[['ts','ticker','mid']].sort_values(['ticker','ts']).copy()
        m = pd.merge_asof(
            thx[['ts','ticker','seconds_to_expiry','har_theo','outcome']],
            bkx, on='ts', by='ticker', direction='backward',
            tolerance=pd.Timedelta(seconds=30),
        )
        m = m.dropna(subset=['mid'])
        m['e_har'] = (m['har_theo'] - m['outcome'])**2
        m['e_mkt'] = (m['mid']      - m['outcome'])**2
        m['gap']   = m['e_har'] - m['e_mkt']
        return m[['ts','ticker','seconds_to_expiry','e_har','e_mkt','gap']]

    loose_b.append(aligned(th, bk))
    strict_b.append(aligned(th_strict, bk_strict))

    theo_rows_loose  += len(th);         theo_rows_strict  += len(th_strict)
    book_rows_loose  += len(bk);         book_rows_strict  += len(bk_strict)
    theo_size_per_tkr_loose.extend(th.groupby('ticker').size().tolist())
    book_size_per_tkr_loose.extend(bk.groupby('ticker').size().tolist())
    theo_size_per_tkr_strict.extend(th_strict.groupby('ticker').size().tolist())
    book_size_per_tkr_strict.extend(bk_strict.groupby('ticker').size().tolist())

    print(f"  {p.name}  theo={len(th):,} (strict {len(th_strict):,})  "
          f"book={len(bk):,} (strict {len(bk_strict):,})")

    del th, bk, th_strict, bk_strict, bk_loose_iv, bk_strict_iv
    gc.collect()


# ----- A. headline (row-disjoint, replicates vol_forecasting) -----
print("\n" + "="*72)
print("A.  Row-disjoint Brier (vol_forecasting.py methodology)")
print("="*72)
def cat_brier(parts, p_col):
    df = pd.concat(parts, ignore_index=True)
    return brier_score(df[p_col], df['outcome']), len(df)
har_l_b, n_har_l = cat_brier(loose_a_har, 'har_theo')
mkt_l_b, n_mkt_l = cat_brier(loose_a_mkt, 'mid')
har_s_b, n_har_s = cat_brier(strict_a_har, 'har_theo')
mkt_s_b, n_mkt_s = cat_brier(strict_a_mkt, 'mid')
print(f"  LOOSE (no ticker-date filter):")
print(f"    HAR    n={n_har_l:>9,}  Brier={har_l_b:.4f}")
print(f"    Market n={n_mkt_l:>9,}  Brier={mkt_l_b:.4f}   gap={har_l_b-mkt_l_b:+.4f}")
print(f"  STRICT (ticker-date ≥ {CUTOFF_DAY}):")
print(f"    HAR    n={n_har_s:>9,}  Brier={har_s_b:.4f}")
print(f"    Market n={n_mkt_s:>9,}  Brier={mkt_s_b:.4f}   gap={har_s_b-mkt_s_b:+.4f}")
print("  NOTE: HAR and Market Brier are computed on DIFFERENT row sets.")


# ----- B. row-aligned per-tick — same rows for both, then bucket by TTE -----
print("\n" + "="*72)
print("B.  Row-aligned per-tick (mid asof'd onto each theo row)")
print("    HAR and Market evaluated on IDENTICAL rows. Buckets by TTE.")
print("="*72)

def report_aligned(parts, label):
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if m.empty:
        print(f"  {label}: empty")
        return
    m['tte_bin'] = pd.cut(m['seconds_to_expiry'],
                          bins=[0, 60, 180, 420, 900],
                          labels=['0-60s','1-3m','3-7m','7-15m'])
    print(f"\n  {label}: n_rows={len(m):,}  n_tickers={m['ticker'].nunique()}")
    print(f"  overall  HAR={m['e_har'].mean():.4f}  Market={m['e_mkt'].mean():.4f}  "
          f"gap={m['gap'].mean():+.4f}")
    g = (m.groupby('tte_bin', observed=True)
            .agg(n=('e_har','count'),
                 n_tkr=('ticker','nunique'),
                 har=('e_har','mean'),
                 mkt=('e_mkt','mean'),
                 gap=('gap','mean')))
    print(g.round(4).to_string())

    # ticker-clustered CIs overall + per bin
    rng = np.random.default_rng(42)
    per_t = m.groupby('ticker')['gap'].mean().values
    boot = np.array([rng.choice(per_t, size=len(per_t), replace=True).mean()
                     for _ in range(2000)])
    print(f"  overall ticker-clustered gap: n_tkr={len(per_t)}  "
          f"mean={per_t.mean():+.4f}  95% CI=[{np.quantile(boot,.025):+.4f}, {np.quantile(boot,.975):+.4f}]")

    print("  per-TTE-bin ticker-clustered gap CI:")
    for binname in ['0-60s','1-3m','3-7m','7-15m']:
        sub = m[m['tte_bin']==binname]
        if sub.empty:
            continue
        per_t_b = sub.groupby('ticker')['gap'].mean().values
        if len(per_t_b) < 5:
            continue
        boot_b = np.array([rng.choice(per_t_b, size=len(per_t_b), replace=True).mean()
                           for _ in range(2000)])
        sig = "YES" if (np.quantile(boot_b,.025)>0 or np.quantile(boot_b,.975)<0) else "no"
        print(f"    {binname:>8s}  n_tkr={len(per_t_b):>3d}  mean={per_t_b.mean():+.4f}  "
              f"95% CI=[{np.quantile(boot_b,.025):+.4f}, {np.quantile(boot_b,.975):+.4f}]  sig={sig}")

report_aligned(loose_b,  "LOOSE")
report_aligned(strict_b, "STRICT")


# ----- C. row-set asymmetry numbers -----
print("\n" + "="*72)
print("C.  Update-cadence asymmetry (why row-disjoint Brier is misleading)")
print("="*72)
def med_mean(x):
    a = np.array(x)
    return f"median={int(np.median(a)):>7,d}  mean={int(a.mean()):>7,d}"
print(f"  LOOSE   theo rows/tkr:  {med_mean(theo_size_per_tkr_loose)}")
print(f"  LOOSE   book rows/tkr:  {med_mean(book_size_per_tkr_loose)}")
print(f"  STRICT  theo rows/tkr:  {med_mean(theo_size_per_tkr_strict)}")
print(f"  STRICT  book rows/tkr:  {med_mean(book_size_per_tkr_strict)}")
print(f"  ratio (strict theo / book per ticker): "
      f"{np.mean(theo_size_per_tkr_strict)/max(1, np.mean(book_size_per_tkr_strict)):.1f}x")
