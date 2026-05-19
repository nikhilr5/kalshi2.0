"""Brier-gap root-cause diagnostic for HAR-RV vs market mid.

Loads the validation-window data, computes per-ticker Brier scores for
HAR / market / HAR-TWAP at a handful of fixed time-to-close offsets,
bootstraps the gap CI at the ticker level (markets are the independent
unit), and breaks the gap down by tenor / realized-σ quintile /
moneyness.  No calibration — pure structural diagnostic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    fetch_settlements_from_api,
    implied_sigma,
    load_all_data,
    realized_sigma_forward,
    theo_vec,
    theo_vec_twap,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
OFFSETS_S     = [840, 600, 300, 120, 30]    # T-14m, T-10m, T-5m, T-2m, T-30s
SNAP_TOL_S    = 5
N_BOOT        = 5_000
RNG           = np.random.default_rng(42)


# ----- load + clean -----
theo, book, spot, _, _ = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo = theo[(theo['seconds_to_expiry'] > 0) & (theo['seconds_to_expiry'] <= 900)].copy()

# Strict ticker-date filter: file-level rollover at UTC midnight lets a few
# late-UTC-May-14 markets leak into the 26MAY15 db. Parse the ticker's own
# date and require it ≥ CUTOFF_DAY.
_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}
def _ticker_date_key(t):
    d = t.split('-')[1][:7]   # e.g. '26MAY15'
    return (int(d[:2]), _MON[d[2:5]], int(d[5:7]))
cutoff_key = _ticker_date_key(f"X-{CUTOFF_DAY}")
theo = theo[theo['ticker'].map(lambda t: _ticker_date_key(t) >= cutoff_key)].copy()
book = book[book['ticker'].map(lambda t: _ticker_date_key(t) >= cutoff_key)].copy()

theo['forecasted_vol'] = (
    0.0314
    + 0.4485 * theo['rv_15m']
    + 0.1293 * theo['rv_30m']
    + 0.1843 * theo['rv_4h']
    + 0.1149 * theo['rv_24h']
)
theo['har_theo'] = theo_vec(theo['spot'], theo['strike'],
                            theo['forecasted_vol'], theo['seconds_to_expiry'])
theo['har_theo_twap'] = theo_vec_twap(theo['spot'], theo['strike'],
                                       theo['forecasted_vol'], theo['seconds_to_expiry'])

book = book.sort_values(['ticker', 'ts']).copy()
book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2

# settlements (authoritative)
api = KalshiAPI()
cache = Path(__file__).resolve().parent.parent / ".settlements_cache.json"
settlements = fetch_settlements_from_api(theo['ticker'].unique().tolist(), api, cache_path=cache)
theo['outcome'] = theo['ticker'].map(settlements)
theo = theo[theo['outcome'].notna()].copy()

# market mid aligned onto each theo_state row by ticker+ts (backward)
theo = theo.sort_values(['ticker', 'ts'])
mid_aligned = pd.merge_asof(
    theo[['ts', 'ticker']], book[['ts', 'ticker', 'mid']],
    on='ts', by='ticker', direction='backward',
    tolerance=pd.Timedelta(seconds=30),
)
theo['mid'] = mid_aligned['mid'].values


# ----- snapshot each market at fixed seconds-to-close offsets -----
# pick the theo row whose seconds_to_expiry is closest to each target.
snaps = []
for offset in OFFSETS_S:
    theo['_dist'] = (theo['seconds_to_expiry'] - offset).abs()
    idx = theo.groupby('ticker')['_dist'].idxmin()
    s = theo.loc[idx, ['ticker', 'ts', 'spot', 'strike', 'seconds_to_expiry',
                       'forecasted_vol', 'har_theo', 'har_theo_twap',
                       'mid', 'outcome']].copy()
    s = s[s['_dist'] <= SNAP_TOL_S] if False else s   # keep all; we'll filter on dist next
    s['offset'] = offset
    s['snap_dist'] = theo.loc[idx, '_dist'].values
    snaps.append(s)
snap = pd.concat(snaps, ignore_index=True)
snap = snap[snap['snap_dist'] <= 30]            # within 30s of target
snap = snap[snap['mid'].notna()].copy()
theo.drop(columns=['_dist'], inplace=True)


# ----- realized σ over the *backward* 15m window for moneyness/regime buckets -----
# Use the actual realized σ of the market (close - 15m → close) by
# re-running the Parkinson computation forward from each market's open.
mb = realized_sigma_forward(spot, horizon_minutes=15)
# For each ticker, compute realized σ over the market's lifetime by
# aligning to the *open* minute (close - 900s).
snap['close_ts'] = snap['ts'] + pd.to_timedelta(snap['seconds_to_expiry'], unit='s')
snap['open_minute'] = (snap['close_ts'] - pd.Timedelta(seconds=900)).dt.floor('1min')
mb = mb.rename(columns={'minute': 'open_minute', 'realized_15m': 'realized_sigma'})
snap = snap.merge(mb[['open_minute', 'realized_sigma']], on='open_minute', how='left')

snap['log_mny'] = np.log(snap['spot'] / snap['strike'])
snap['z_mny']   = snap['log_mny'] / (snap['forecasted_vol']
                                      * np.sqrt(snap['seconds_to_expiry'] / (365.25*24*3600)))


# ----- per-(ticker,offset) Brier components -----
snap['e_har']  = (snap['har_theo']      - snap['outcome'])**2
snap['e_mkt']  = (snap['mid']           - snap['outcome'])**2
snap['e_twap'] = (snap['har_theo_twap'] - snap['outcome'])**2
snap['gap_har_mkt']  = snap['e_har']  - snap['e_mkt']
snap['gap_twap_mkt'] = snap['e_twap'] - snap['e_mkt']
snap['gap_twap_har'] = snap['e_twap'] - snap['e_har']


# ----- headline numbers (averaged across all offsets) -----
def boot_ci(values, n=N_BOOT, alpha=0.05):
    """Cluster-bootstrap by ticker.  `values` is keyed by ticker → mean(gap)."""
    arr = values.values
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    means = np.empty(n)
    for i in range(n):
        s = RNG.choice(arr, size=len(arr), replace=True)
        means[i] = s.mean()
    return arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr)), np.quantile(means, alpha/2), np.quantile(means, 1-alpha/2)


print("="*72)
print(f"BRIER GAP ROOT-CAUSE  —  {SERIES_PREFIX} ≥ {CUTOFF_DAY}")
print("="*72)
print(f"n unique settled markets : {snap['ticker'].nunique()}")
print(f"n snapshots              : {len(snap)}  (offsets={OFFSETS_S})")
print()

# Headline Brier — average across all offsets, equal-weighted by snapshot
print(f"{'metric':<28}{'HAR':>10}{'Market':>10}{'HAR-TWAP':>12}")
print(f"{'mean squared err (all)':<28}{snap['e_har'].mean():>10.4f}"
      f"{snap['e_mkt'].mean():>10.4f}{snap['e_twap'].mean():>12.4f}")

# Ticker-level gap CI (one observation per ticker — average across its offsets)
per_ticker_gap = snap.groupby('ticker')['gap_har_mkt'].mean()
m, se, lo, hi = boot_ci(per_ticker_gap)
print(f"\nHAR − Market Brier (ticker-clustered):")
print(f"  mean gap = {m:+.4f}   SE = {se:.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]")
print(f"  n_tickers = {len(per_ticker_gap)}   "
      f"sig = {'YES' if (lo>0 or hi<0) else 'NO — within sampling noise'}")

per_ticker_twap = snap.groupby('ticker')['gap_twap_mkt'].mean()
m2, se2, lo2, hi2 = boot_ci(per_ticker_twap)
print(f"\nHAR-TWAP − Market Brier (ticker-clustered):")
print(f"  mean gap = {m2:+.4f}   SE = {se2:.4f}   95% CI = [{lo2:+.4f}, {hi2:+.4f}]")


# ----- DECOMPOSITION 1: by time-to-close offset -----
print("\n" + "="*72)
print("1.  By time-to-close")
print("="*72)
g1 = (snap.groupby('offset')
          .agg(n=('ticker', 'count'),
               har=('e_har', 'mean'),
               mkt=('e_mkt', 'mean'),
               twap=('e_twap', 'mean'),
               gap=('gap_har_mkt', 'mean')))
g1['gap_pct'] = g1['gap'] / g1['mkt'] * 100
print(g1.round(4).to_string())


# ----- DECOMPOSITION 2: σ-forecast bias -----
print("\n" + "="*72)
print("2.  σ-forecast bias  (HAR vs market-implied vs realized)")
print("="*72)
snap['implied_sigma'] = implied_sigma(snap['mid'], snap['spot'],
                                        snap['strike'], snap['seconds_to_expiry'])
# implied σ goes pathological when mid is near 0/1 OR when |z| is large
# enough that the no-drift quadratic has no real root. Filter to a plausible
# crypto-σ range.
bias = snap[(snap['implied_sigma'].between(0.05, 3.0))
            & snap['realized_sigma'].notna()].copy()
print(f"  n with realized+implied   : {len(bias)}")
print(f"  mean realized σ           : {bias['realized_sigma'].mean()*100:6.2f}%")
print(f"  mean HAR forecast σ       : {bias['forecasted_vol'].mean()*100:6.2f}%   "
      f"bias = {(bias['forecasted_vol']-bias['realized_sigma']).mean()*100:+.2f}%")
print(f"  mean market implied σ     : {bias['implied_sigma'].mean()*100:6.2f}%   "
      f"bias = {(bias['implied_sigma']-bias['realized_sigma']).mean()*100:+.2f}%")


# ----- DECOMPOSITION 3: by realized-σ quintile -----
print("\n" + "="*72)
print("3.  By realized-σ quintile (per-ticker; uses the market's own realized)")
print("="*72)
per_tkr = (snap.groupby('ticker')
                .agg(realized=('realized_sigma', 'first'),
                     gap=('gap_har_mkt', 'mean'),
                     har_b=('e_har', 'mean'),
                     mkt_b=('e_mkt', 'mean'))
                .dropna())
per_tkr['q'] = pd.qcut(per_tkr['realized'], 5, labels=['Q1-low','Q2','Q3','Q4','Q5-high'])
g3 = per_tkr.groupby('q', observed=True).agg(n=('gap','count'),
                                              rlz=('realized','mean'),
                                              har=('har_b','mean'),
                                              mkt=('mkt_b','mean'),
                                              gap=('gap','mean'))
print(g3.round(4).to_string())


# ----- DECOMPOSITION 4: by moneyness (|z| under HAR σ at snapshot time) -----
print("\n" + "="*72)
print("4.  By moneyness  |z| = |log(S/K)| / (σ_HAR √T)")
print("="*72)
snap['abs_z'] = snap['z_mny'].abs()
mny = snap.dropna(subset=['abs_z'])
mny = mny.assign(zbin=pd.cut(mny['abs_z'], bins=[0,0.25,0.5,1.0,2.0,np.inf],
                              labels=['ATM<0.25','0.25-0.5','0.5-1.0','1.0-2.0','>2.0']))
g4 = mny.groupby('zbin', observed=True).agg(n=('ticker','count'),
                                             har=('e_har','mean'),
                                             mkt=('e_mkt','mean'),
                                             twap=('e_twap','mean'),
                                             gap=('gap_har_mkt','mean'))
print(g4.round(4).to_string())

# How much of the headline gap is the deep-OTM bucket alone?
no_otm = snap[snap['abs_z'] <= 2.0]
per_tkr_no_otm = no_otm.groupby('ticker')['gap_har_mkt'].mean()
m_no, _, lo_no, hi_no = boot_ci(per_tkr_no_otm)
print(f"\n  Excluding |z|>2.0 (deep-OTM resolved markets):")
print(f"    n_tickers = {len(per_tkr_no_otm)}   mean gap = {m_no:+.4f}   "
      f"95% CI = [{lo_no:+.4f}, {hi_no:+.4f}]   "
      f"sig = {'YES' if (lo_no>0 or hi_no<0) else 'NO'}")


# ----- DECOMPOSITION 5: TWAP diagnostic vs offset -----
print("\n" + "="*72)
print("5.  TWAP-aware theo: does it close the gap?")
print("="*72)
g5 = snap.groupby('offset').agg(n=('ticker','count'),
                                  har=('e_har','mean'),
                                  twap=('e_twap','mean'),
                                  mkt=('e_mkt','mean'))
g5['twap_minus_har'] = g5['twap'] - g5['har']
g5['twap_minus_mkt'] = g5['twap'] - g5['mkt']
print(g5.round(4).to_string())


# ----- save tidy snapshot for further inspection -----
out_dir = Path(__file__).resolve().parent
snap.to_csv(out_dir / "brier_snapshots.csv", index=False)
print(f"\n[saved] {out_dir/'brier_snapshots.csv'}  ({len(snap):,} rows)")
