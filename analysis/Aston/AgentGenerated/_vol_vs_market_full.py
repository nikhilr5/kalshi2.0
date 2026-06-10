"""Full-window study: is HAR-RV σ a better vol forecast than the market?

Scales up brier_root_cause.py to the entire data window (26MAY15 →
present, excluding empty / corrupt days). Same methodology:

  - snapshot each market at T-14m/-10m/-5m/-2m/-30s (within 30s of target)
  - score HAR theo and market mid on IDENTICAL rows
  - ticker-clustered bootstrap CIs (the market is the independent unit)
  - forward realized σ (Parkinson) from snapshot's market open to close
  - vol-forecast accuracy: HAR σ / implied σ vs forward realized σ

Memory: the full theo_state is tens of GB. We process ONE day-DB at a
time, reduce to the slim per-(ticker,offset) snapshot frame, and only
accumulate those. Settlements come from the authoritative Kalshi cache.

Adds over the root-cause script:
  - data inventory (rows / tickers / book density per day, gap flagging)
  - weekly stability of the Brier gap with per-block bootstrap CI + plot
  - vol-forecast accuracy table at snapshot level

Rerunnable as more days land. Writes a stability plot to this dir.
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_S3_CACHE_DIR,
    fetch_settlements_from_api,
    implied_sigma,
    list_eligible_dbs,
    realized_sigma_forward,
    theo_vec,
    theo_vec_twap,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
EXCLUDE_DAYS  = {"26JUN09"}          # corrupt b-trees per project note
OFFSETS_S     = [840, 600, 300, 120, 30]   # T-14m, -10m, -5m, -2m, -30s
SNAP_TOL_S    = 30                   # keep a snapshot within 30s of target
N_BOOT        = 5_000
RNG           = np.random.default_rng(42)

HAR_B0, HAR_15, HAR_30, HAR_4H, HAR_24H = 0.0314, 0.4485, 0.1293, 0.1843, 0.1149

_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}


def _ticker_date_key(t):
    d = t.split('-')[1][:7]
    return (int(d[:2]), _MON[d[2:5]], int(d[5:7]))


def _day_suffix(path):
    return path.stem.rsplit("-", 1)[-1]


def _snapshot_day(path):
    """Load one day-DB, snapshot every market at the fixed offsets, align
    market mid, compute forward realized σ, return the slim snap frame.
    Returns (snap_df, inventory_dict). Empty/unreadable tables -> None."""
    suffix = _day_suffix(path)
    inv = {"day": suffix, "theo_rows": 0, "book_rows": 0,
           "tickers": 0, "snaps": 0, "note": ""}
    try:
        conn = sqlite3.connect(str(path))
    except Exception as e:
        inv["note"] = f"open failed: {e}"
        return None, inv
    try:
        theo = pd.read_sql(
            "SELECT ts,ticker,spot,strike,sigma,theo,seconds_to_expiry,"
            "rv_15m,rv_30m,rv_4h,rv_24h FROM theo_state", conn)
        book = pd.read_sql(
            "SELECT ts,ticker,yes_bid,yes_ask FROM kalshi_book", conn)
        spot = pd.read_sql("SELECT ts,price FROM spot_ticks", conn)
    except Exception as e:
        inv["note"] = f"read failed: {e}"
        return None, inv
    finally:
        conn.close()

    if theo.empty or book.empty or spot.empty:
        inv["note"] = "empty table(s)"
        return None, inv

    for df in (theo, book, spot):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")

    theo = theo[(theo['seconds_to_expiry'] > 0)
                & (theo['seconds_to_expiry'] <= 900)].copy()
    # strict ticker-date filter (file rollover leaks late prior-UTC-day mkts)
    cutoff_key = _ticker_date_key(f"X-{CUTOFF_DAY}")
    keep = theo['ticker'].map(lambda t: _ticker_date_key(t) >= cutoff_key)
    theo = theo[keep].copy()
    book = book[book['ticker'].map(lambda t: _ticker_date_key(t) >= cutoff_key)].copy()

    inv["theo_rows"] = len(theo)
    inv["book_rows"] = len(book)
    inv["tickers"] = theo['ticker'].nunique()
    if theo.empty or book.empty:
        inv["note"] = "no in-window tickers"
        return None, inv

    theo['fvol'] = (HAR_B0
                    + HAR_15 * theo['rv_15m'] + HAR_30 * theo['rv_30m']
                    + HAR_4H * theo['rv_4h']  + HAR_24H * theo['rv_24h'])
    theo['har_theo'] = theo_vec(theo['spot'], theo['strike'],
                                theo['fvol'], theo['seconds_to_expiry'])
    theo['har_theo_twap'] = theo_vec_twap(theo['spot'], theo['strike'],
                                          theo['fvol'], theo['seconds_to_expiry'])

    book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2
    book_s = book.sort_values('ts')

    # merge_asof requires the `on` key globally sorted on both sides
    theo = theo.sort_values('ts').reset_index(drop=True)
    mid_aligned = pd.merge_asof(
        theo[['ts', 'ticker']], book_s[['ts', 'ticker', 'mid']],
        on='ts', by='ticker', direction='backward',
        tolerance=pd.Timedelta(seconds=30))
    theo['mid'] = mid_aligned['mid'].values

    # snapshot at each offset: the theo row closest to target sec-to-close
    snaps = []
    cols = ['ticker', 'ts', 'spot', 'strike', 'seconds_to_expiry',
            'fvol', 'har_theo', 'har_theo_twap', 'mid']
    for offset in OFFSETS_S:
        d = (theo['seconds_to_expiry'] - offset).abs()
        idx = d.groupby(theo['ticker']).idxmin()
        s = theo.loc[idx, cols].copy()
        s['offset'] = offset
        s['snap_dist'] = d.loc[idx].values
        snaps.append(s)
    snap = pd.concat(snaps, ignore_index=True)
    snap = snap[(snap['snap_dist'] <= SNAP_TOL_S) & snap['mid'].notna()].copy()
    if snap.empty:
        inv["note"] = "no snaps"
        return None, inv

    # forward realized σ over the market's own 15m window (open -> close)
    mb = realized_sigma_forward(spot, horizon_minutes=15)
    snap['close_ts'] = snap['ts'] + pd.to_timedelta(snap['seconds_to_expiry'], unit='s')
    snap['open_minute'] = (snap['close_ts'] - pd.Timedelta(seconds=900)).dt.floor('1min')
    mb = mb.rename(columns={'minute': 'open_minute', 'realized_15m': 'realized_sigma'})
    snap = snap.merge(mb[['open_minute', 'realized_sigma']],
                      on='open_minute', how='left')

    snap['implied_sigma'] = implied_sigma(snap['mid'], snap['spot'],
                                          snap['strike'], snap['seconds_to_expiry'])
    snap['log_mny'] = np.log(snap['spot'] / snap['strike'])
    T_yr = snap['seconds_to_expiry'] / (365.25 * 24 * 3600)
    snap['abs_z'] = (snap['log_mny'] / (snap['fvol'] * np.sqrt(T_yr))).abs()
    snap['day'] = suffix
    inv["snaps"] = len(snap)
    return snap, inv


def boot_ci(per_ticker, n=N_BOOT, alpha=0.05):
    arr = per_ticker.values
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    means = np.empty(n)
    for i in range(n):
        means[i] = RNG.choice(arr, size=len(arr), replace=True).mean()
    return (arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr)),
            np.quantile(means, alpha/2), np.quantile(means, 1-alpha/2), len(arr))


# ----- load + snapshot every eligible day -----
files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY,
                          DEFAULT_LOCAL_DIR, DEFAULT_S3_CACHE_DIR)
files = [f for f in files if _day_suffix(f) not in EXCLUDE_DAYS]

snap_parts, inventory = [], []
for f in files:
    s, inv = _snapshot_day(f)
    inventory.append(inv)
    if s is not None:
        snap_parts.append(s)
    print(f"   {inv['day']}: theo={inv['theo_rows']:>9,}  book={inv['book_rows']:>9,}  "
          f"tkr={inv['tickers']:>3}  snaps={inv['snaps']:>4}  {inv['note']}")

snap = pd.concat(snap_parts, ignore_index=True)

# ----- settlements (authoritative, cached) -----
api = KalshiAPI()
cache = Path(__file__).resolve().parent.parent / ".settlements_cache.json"
settlements = fetch_settlements_from_api(snap['ticker'].unique().tolist(),
                                         api, cache_path=cache)
snap['outcome'] = snap['ticker'].map(settlements)
snap = snap[snap['outcome'].notna()].copy()

# per-snapshot squared errors + gaps
snap['e_har']  = (snap['har_theo']      - snap['outcome'])**2
snap['e_mkt']  = (snap['mid']           - snap['outcome'])**2
snap['e_twap'] = (snap['har_theo_twap'] - snap['outcome'])**2
snap['gap']    = snap['e_har'] - snap['e_mkt']           # neg => HAR better

snap['week'] = pd.to_datetime(snap['close_ts']).dt.isocalendar().week


# =====================================================================
# 1. DATA INVENTORY
# =====================================================================
inv_df = pd.DataFrame(inventory)
print("\n" + "="*78)
print("1.  DATA INVENTORY")
print("="*78)
usable = inv_df[inv_df['snaps'] > 0]
print(inv_df[['day', 'theo_rows', 'book_rows', 'tickers', 'snaps', 'note']]
      .to_string(index=False))
print(f"\n  usable days        : {len(usable)} / {len(inv_df)}")
print(f"  settled markets    : {snap['ticker'].nunique():,}")
print(f"  total snapshots    : {len(snap):,}")
# row-density sanity: theo rows per ticker (flag sparse recorder days)
inv_df['rows_per_tkr'] = (inv_df['theo_rows'] / inv_df['tickers']).round(0)
med = inv_df.loc[inv_df['snaps'] > 0, 'rows_per_tkr'].median()
sparse = inv_df[(inv_df['snaps'] > 0) & (inv_df['rows_per_tkr'] < 0.4 * med)]
if len(sparse):
    print(f"  SPARSE days (<40% median {med:.0f} rows/tkr): "
          f"{', '.join(sparse['day'])}")
else:
    print(f"  no sparse days (median {med:.0f} theo rows/ticker)")


# =====================================================================
# 2. VOL-FORECAST ACCURACY  (HAR σ / implied σ vs forward realized σ)
# =====================================================================
print("\n" + "="*78)
print("2.  VOL-FORECAST ACCURACY vs forward realized σ  (snapshot level)")
print("="*78)
va = snap.dropna(subset=['realized_sigma']).copy()
va_imp = va[va['implied_sigma'].between(0.05, 3.0)]

def vstats(pred, actual):
    err = pred - actual
    return dict(n=len(pred), corr=np.corrcoef(pred, actual)[0, 1],
                bias=err.mean()*100, mae=err.abs().mean()*100,
                rmse=np.sqrt((err**2).mean())*100, mean=pred.mean()*100)

rlz = va['realized_sigma'].mean() * 100
h = vstats(va['fvol'], va['realized_sigma'])
m = vstats(va_imp['implied_sigma'], va_imp['realized_sigma'])
print(f"  realized σ (mean)  : {rlz:6.2f}%   n={len(va):,}")
print(f"  {'series':<10}{'n':>9}{'corr':>8}{'mean':>8}{'bias':>8}{'MAE':>8}{'RMSE':>8}")
print(f"  {'HAR':<10}{h['n']:>9,}{h['corr']:>8.3f}{h['mean']:>7.2f}%"
      f"{h['bias']:>+7.2f}%{h['mae']:>7.2f}%{h['rmse']:>7.2f}%")
print(f"  {'implied':<10}{m['n']:>9,}{m['corr']:>8.3f}{m['mean']:>7.2f}%"
      f"{m['bias']:>+7.2f}%{m['mae']:>7.2f}%{m['rmse']:>7.2f}%")


# =====================================================================
# 3. BRIER  (overall + by offset + by moneyness)
# =====================================================================
print("\n" + "="*78)
print("3.  BRIER  —  HAR vs Market mid (ticker-clustered bootstrap)")
print("="*78)
print(f"  {'metric':<26}{'HAR':>10}{'Market':>10}{'HAR-TWAP':>11}")
print(f"  {'mean sq err (all)':<26}{snap['e_har'].mean():>10.4f}"
      f"{snap['e_mkt'].mean():>10.4f}{snap['e_twap'].mean():>11.4f}")
pt = snap.groupby('ticker')['gap'].mean()
gm, se, lo, hi, nt = boot_ci(pt)
print(f"\n  HAR − Market gap (overall): {gm:+.4f}  SE={se:.4f}  "
      f"95% CI [{lo:+.4f}, {hi:+.4f}]  n_tkr={nt}  "
      f"sig={'YES' if (lo>0 or hi<0) else 'NO'}")

print("\n  By time-to-close offset:")
g_off = (snap.groupby('offset')
         .agg(n=('ticker', 'count'), har=('e_har', 'mean'),
              mkt=('e_mkt', 'mean'), gap=('gap', 'mean')))
for off in OFFSETS_S:
    sub = snap[snap['offset'] == off]
    pto = sub.groupby('ticker')['gap'].mean()
    _, _, l, hh, _ = boot_ci(pto)
    lbl = {840:'T-14m', 600:'T-10m', 300:'T-5m', 120:'T-2m', 30:'T-30s'}[off]
    r = g_off.loc[off]
    print(f"    {lbl:<7} n={int(r['n']):>5}  HAR={r['har']:.4f}  Mkt={r['mkt']:.4f}  "
          f"gap={r['gap']:+.4f}  CI[{l:+.4f},{hh:+.4f}]"
          f"{'  *' if (l>0 or hh<0) else ''}")

print("\n  By moneyness |z| = |log(S/K)| / (σ_HAR √T):")
mny = snap.dropna(subset=['abs_z']).copy()
mny['zbin'] = pd.cut(mny['abs_z'], [0, 0.25, 0.5, 1.0, 2.0, np.inf],
                     labels=['ATM<0.25', '0.25-0.5', '0.5-1.0', '1.0-2.0', '>2.0'])
for zb in ['ATM<0.25', '0.25-0.5', '0.5-1.0', '1.0-2.0', '>2.0']:
    sub = mny[mny['zbin'] == zb]
    if sub.empty:
        continue
    ptz = sub.groupby('ticker')['gap'].mean()
    g, _, l, hh, ntz = boot_ci(ptz)
    print(f"    {zb:<9} n={len(sub):>5}  HAR={sub['e_har'].mean():.4f}  "
          f"Mkt={sub['e_mkt'].mean():.4f}  gap={g:+.4f}  "
          f"CI[{l:+.4f},{hh:+.4f}]{'  *' if (l>0 or hh<0) else ''}")

no_otm = snap[snap['abs_z'] <= 2.0]
pto = no_otm.groupby('ticker')['gap'].mean()
g, _, l, hh, ntn = boot_ci(pto)
print(f"\n  Excluding |z|>2.0 (the regime you actually quote in):")
print(f"    gap={g:+.4f}  CI[{l:+.4f},{hh:+.4f}]  n_tkr={ntn}  "
      f"sig={'YES' if (l>0 or hh<0) else 'NO'}")


# =====================================================================
# 4. STABILITY OVER TIME  (weekly Brier gap with CI)
# =====================================================================
print("\n" + "="*78)
print("4.  STABILITY — weekly Brier gap (HAR − Market), ticker-clustered")
print("="*78)
rows = []
for wk, sub in snap.groupby('week'):
    ptw = sub.groupby('ticker')['gap'].mean()
    gw, sew, lw, hw, ntw = boot_ci(ptw)
    d0 = pd.to_datetime(sub['close_ts']).min().strftime('%m-%d')
    d1 = pd.to_datetime(sub['close_ts']).max().strftime('%m-%d')
    rows.append(dict(week=int(wk), span=f"{d0}→{d1}", n_tkr=ntw,
                     gap=gw, lo=lw, hi=hw, sig=(lw > 0 or hw < 0)))
    print(f"    wk{int(wk)} {d0}→{d1}  n_tkr={ntw:>4}  gap={gw:+.4f}  "
          f"CI[{lw:+.4f},{hw:+.4f}]{'  *sig' if (lw>0 or hw<0) else ''}")
wk_df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(wk_df))
ax.errorbar(x, wk_df['gap'],
            yerr=[wk_df['gap'] - wk_df['lo'], wk_df['hi'] - wk_df['gap']],
            fmt='o', capsize=4, color='#a78bfa', ecolor='#5a6270',
            markersize=8, label='HAR − Market Brier gap')
ax.axhline(0, color='#888', lw=1, ls='--')
ax.set_xticks(list(x))
ax.set_xticklabels([f"wk{r.week}\n{r.span}" for r in wk_df.itertuples()], fontsize=8)
ax.set_ylabel('Brier gap (HAR − Market)  ·  negative = HAR better')
ax.set_title(f'Weekly Brier-gap stability — {SERIES_PREFIX}  '
             f'({snap["ticker"].nunique():,} settled markets)')
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
out_png = Path(__file__).resolve().parent / "vol_vs_market_stability.png"
fig.savefig(out_png, dpi=120)
print(f"\n[saved] {out_png}")

snap_out = Path(__file__).resolve().parent / "vol_vs_market_snapshots.csv"
snap.drop(columns=['open_minute'], errors='ignore').to_csv(snap_out, index=False)
print(f"[saved] {snap_out}  ({len(snap):,} rows)")
