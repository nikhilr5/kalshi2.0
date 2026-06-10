"""Deep study: is the HAR-RV N(d2) theo better than the market mid at
predicting Kalshi 15-min binary settlements — and is it monetizable?

Builds on _vol_vs_market_full.py (cached snapshots in
vol_vs_market_snapshots.csv, May15->Jun8, n=2127 markets). This script:

  - extends the snapshot frame with NEW days (Jun 10; Jun 9 corrupt ->
    excluded) using the identical snapshot/align/forward-rv logic
  - the CENTERPIECE: disagreement-conditional test. When |theo-mid| is
    large, who is right? Per-side Brier, direction-hit-rate, by offset
    and by sign of disagreement, ticker-clustered CI on the gap.
  - reliability curves (calibration diagrams) for theo, mid, and
    Platt-corrected theo on identical rows.
  - weekly Brier-gap stability incl. the new June regime.
  - economic translation: counterfactual $/trade of fading mid->theo at
    |theo-mid|>=5c, T-5m, paying half-spread.

Reuses the cached CSV for the established window; only snapshots the new
day(s). Identical rows for both predictors throughout. Ticker-clustered
bootstrap (settlements within a ticker share one outcome).
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from utility import (
    fetch_settlements_from_api, implied_sigma, realized_sigma_forward,
    theo_vec, theo_vec_twap,
)
sys.path.insert(0, str(HERE.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

DATA_DIR   = Path("~/Desktop/Kalshi2.0/analysis/backtesting/data").expanduser()
CACHED_CSV = HERE / "vol_vs_market_snapshots.csv"
NEW_DAYS   = ["26JUN10"]          # Jun 9 corrupt; Jun8 & earlier in cache
CUTOFF_DAY = "26MAY15"
OFFSETS_S  = [840, 600, 300, 120, 30]
SNAP_TOL_S = 30
N_BOOT     = 5_000
RNG        = np.random.default_rng(42)
PLATT_B    = 1.116                 # prior-fit slope; a=0
HALF_SPREAD = 0.02                 # 2c half-spread cost assumption for econ

HAR_B0, HAR_15, HAR_30, HAR_4H, HAR_24H = 0.0314, 0.4485, 0.1293, 0.1843, 0.1149
_MON = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
        'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}


def _tk_key(t):
    d = t.split('-')[1][:7]
    return (int(d[:2]), _MON[d[2:5]], int(d[5:7]))


def snapshot_day(path, suffix):
    """Snapshot one day-DB at the fixed offsets; same logic as the full
    pipeline. Returns slim snap frame (no outcome/gap yet) or None."""
    conn = sqlite3.connect(str(path))
    try:
        theo = pd.read_sql(
            "SELECT ts,ticker,spot,strike,sigma,theo,seconds_to_expiry,"
            "rv_15m,rv_30m,rv_4h,rv_24h FROM theo_state", conn)
        book = pd.read_sql(
            "SELECT ts,ticker,yes_bid,yes_ask FROM kalshi_book", conn)
        spot = pd.read_sql("SELECT ts,price FROM spot_ticks", conn)
    finally:
        conn.close()
    if theo.empty or book.empty or spot.empty:
        return None
    for df in (theo, book, spot):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")

    theo = theo[(theo['seconds_to_expiry'] > 0)
                & (theo['seconds_to_expiry'] <= 900)].copy()
    ck = _tk_key(f"X-{CUTOFF_DAY}")
    theo = theo[theo['ticker'].map(lambda t: _tk_key(t) >= ck)].copy()
    book = book[book['ticker'].map(lambda t: _tk_key(t) >= ck)].copy()
    if theo.empty or book.empty:
        return None

    theo['fvol'] = (HAR_B0 + HAR_15*theo['rv_15m'] + HAR_30*theo['rv_30m']
                    + HAR_4H*theo['rv_4h'] + HAR_24H*theo['rv_24h'])
    theo['har_theo'] = theo_vec(theo['spot'], theo['strike'],
                                theo['fvol'], theo['seconds_to_expiry'])
    theo['har_theo_twap'] = theo_vec_twap(theo['spot'], theo['strike'],
                                          theo['fvol'], theo['seconds_to_expiry'])

    # drop crossed / empty books before forming mid
    book = book[(book['yes_bid'] > 0) & (book['yes_ask'] > 0)
                & (book['yes_ask'] >= book['yes_bid'])].copy()
    book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2
    book_s = book.sort_values('ts')

    theo = theo.sort_values('ts').reset_index(drop=True)
    m = pd.merge_asof(theo[['ts', 'ticker']],
                      book_s[['ts', 'ticker', 'mid']],
                      on='ts', by='ticker', direction='backward',
                      tolerance=pd.Timedelta(seconds=30))
    theo['mid'] = m['mid'].values

    cols = ['ticker', 'ts', 'spot', 'strike', 'seconds_to_expiry',
            'fvol', 'har_theo', 'har_theo_twap', 'mid']
    snaps = []
    for off in OFFSETS_S:
        d = (theo['seconds_to_expiry'] - off).abs()
        idx = d.groupby(theo['ticker']).idxmin()
        s = theo.loc[idx, cols].copy()
        s['offset'] = off
        s['snap_dist'] = d.loc[idx].values
        snaps.append(s)
    snap = pd.concat(snaps, ignore_index=True)
    snap = snap[(snap['snap_dist'] <= SNAP_TOL_S) & snap['mid'].notna()].copy()
    if snap.empty:
        return None

    mb = realized_sigma_forward(spot, horizon_minutes=15)
    snap['close_ts'] = snap['ts'] + pd.to_timedelta(snap['seconds_to_expiry'], unit='s')
    snap['open_minute'] = (snap['close_ts'] - pd.Timedelta(seconds=900)).dt.floor('1min')
    mb = mb.rename(columns={'minute': 'open_minute', 'realized_15m': 'realized_sigma'})
    snap = snap.merge(mb[['open_minute', 'realized_sigma']], on='open_minute', how='left')
    snap['implied_sigma'] = implied_sigma(snap['mid'], snap['spot'],
                                          snap['strike'], snap['seconds_to_expiry'])
    snap['log_mny'] = np.log(snap['spot'] / snap['strike'])
    T_yr = snap['seconds_to_expiry'] / (365.25 * 24 * 3600)
    snap['abs_z'] = (snap['log_mny'] / (snap['fvol'] * np.sqrt(T_yr))).abs()
    snap['day'] = suffix
    return snap.drop(columns=['open_minute', 'snap_dist'], errors='ignore')


def boot_ci(per_ticker, n=N_BOOT, alpha=0.05):
    arr = np.asarray(per_ticker, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    means = np.array([RNG.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n)])
    return (arr.mean(), arr.std(ddof=1)/np.sqrt(len(arr)),
            np.quantile(means, alpha/2), np.quantile(means, 1-alpha/2), len(arr))


# ===================================================================
# LOAD: cached snapshots + new-day snapshots
# ===================================================================
cached = pd.read_csv(CACHED_CSV)
for c in ('ts', 'close_ts'):
    cached[c] = pd.to_datetime(cached[c], utc=True, format='ISO8601')
print(f"[cache] {len(cached):,} rows, {cached['ticker'].nunique()} tickers, "
      f"days {cached['day'].min()}..{cached['day'].max()}")

new_parts = []
for d in NEW_DAYS:
    p = DATA_DIR / f"KXETH15M-{d}.db"
    if not p.exists():
        print(f"[new] {d}: missing -> skip"); continue
    s = snapshot_day(p, d)
    if s is None:
        print(f"[new] {d}: no snaps"); continue
    print(f"[new] {d}: {len(s)} snaps, {s['ticker'].nunique()} tickers")
    new_parts.append(s)

if new_parts:
    new = pd.concat(new_parts, ignore_index=True)
    keep = [c for c in cached.columns
            if c in new.columns or c in ('outcome','e_har','e_mkt','e_twap','gap','week')]
    snap = pd.concat([cached, new], ignore_index=True)
else:
    snap = cached.copy()

# settlements for any ticker lacking an outcome (new days)
api = KalshiAPI()
cache_p = HERE.parent / ".settlements_cache.json"
need = snap[snap.get('outcome').isna() if 'outcome' in snap else slice(None)]['ticker'].unique().tolist() \
       if 'outcome' in snap else snap['ticker'].unique().tolist()
settle = fetch_settlements_from_api(need, api, cache_path=cache_p)
if 'outcome' not in snap.columns:
    snap['outcome'] = np.nan
fill = snap['outcome'].isna()
snap.loc[fill, 'outcome'] = snap.loc[fill, 'ticker'].map(settle)
snap = snap[snap['outcome'].notna()].copy()
snap['outcome'] = snap['outcome'].astype(int)

# recompute derived error/gap columns on the FULL frame (consistent)
snap['e_har']  = (snap['har_theo'] - snap['outcome'])**2
snap['e_mkt']  = (snap['mid'] - snap['outcome'])**2
snap['e_twap'] = (snap['har_theo_twap'] - snap['outcome'])**2
snap['gap']    = snap['e_har'] - snap['e_mkt']
snap['week']   = snap['close_ts'].dt.isocalendar().week.astype(int)
snap['disagree'] = snap['har_theo'] - snap['mid']   # +ve: theo>mid

LBL = {840:'T-14m', 600:'T-10m', 300:'T-5m', 120:'T-2m', 30:'T-30s'}
print(f"\n[final] {len(snap):,} snapshots, {snap['ticker'].nunique():,} settled markets, "
      f"days {snap['day'].min()}..{snap['day'].max()}\n")


# ===================================================================
# 1. OVERALL BRIER (anchor — confirm vs prior)
# ===================================================================
print("="*82)
print("1.  OVERALL BRIER (anchor)")
print("="*82)
pt = snap.groupby('ticker')['gap'].mean()
gm, se, lo, hi, nt = boot_ci(pt)
print(f"  HAR Brier {snap['e_har'].mean():.4f}   Mkt Brier {snap['e_mkt'].mean():.4f}")
print(f"  gap (HAR-Mkt) {gm:+.4f}  SE {se:.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
      f"n_tkr={nt}  {'SIG' if (lo>0 or hi<0) else 'ns'}")
noz = snap[snap['abs_z'] <= 2.0]
g2, _, l2, h2, n2 = boot_ci(noz.groupby('ticker')['gap'].mean())
print(f"  excl |z|>2 : gap {g2:+.4f}  CI[{l2:+.4f},{h2:+.4f}]  "
      f"n_tkr={n2}  {'SIG' if (l2>0 or h2<0) else 'ns'}")


# ===================================================================
# 2. DISAGREEMENT-CONDITIONAL TEST  (centerpiece)
# ===================================================================
print("\n" + "="*82)
print("2.  DISAGREEMENT-CONDITIONAL — when theo & mid disagree, who's right?")
print("="*82)

def disagree_block(df, thr):
    sub = df[df['disagree'].abs() >= thr].copy()
    if sub.empty:
        return None
    bh, bm = sub['e_har'].mean(), sub['e_mkt'].mean()
    g, se_, l, h, n = boot_ci(sub.groupby('ticker')['gap'].mean())
    # direction hit-rate: when theo says higher than mid, does YES settle?
    up = sub[sub['disagree'] > 0]      # theo > mid -> theo bets YES more
    dn = sub[sub['disagree'] < 0]      # theo < mid -> theo bets NO more
    up_hit = up['outcome'].mean() if len(up) else np.nan   # P(YES) when theo>mid
    dn_hit = dn['outcome'].mean() if len(dn) else np.nan
    return dict(thr=thr, n_snap=len(sub), n_tkr=sub['ticker'].nunique(),
                brier_har=bh, brier_mkt=bm, gap=g, lo=l, hi=h,
                n_up=len(up), up_pYES=up_hit, up_mid=up['mid'].mean() if len(up) else np.nan,
                n_dn=len(dn), dn_pYES=dn_hit, dn_mid=dn['mid'].mean() if len(dn) else np.nan,
                sig=(l > 0 or h < 0))

print(f"\n  {'thr':>5}{'n_snap':>8}{'n_tkr':>7}{'Brier_HAR':>11}{'Brier_Mkt':>11}"
      f"{'gap':>9}{'95% CI':>20}  sig")
rows = []
for thr in (0.02, 0.05, 0.10):
    r = disagree_block(snap, thr)
    if r is None: continue
    rows.append(r)
    print(f"  {thr*100:>4.0f}c{r['n_snap']:>8}{r['n_tkr']:>7}{r['brier_har']:>11.4f}"
          f"{r['brier_mkt']:>11.4f}{r['gap']:>+9.4f}"
          f"   [{r['lo']:+.4f},{r['hi']:+.4f}]  {'SIG' if r['sig'] else 'ns'}")

print("\n  Direction hit-rate (does the side theo leans toward actually settle?):")
print(f"  {'thr':>5}  theo>mid: n / P(YES) / mid    |  theo<mid: n / P(YES) / mid")
for r in rows:
    print(f"  {r['thr']*100:>4.0f}c  {r['n_up']:>5} / {r['up_pYES']:.3f} / {r['up_mid']:.3f}"
          f"          |  {r['n_dn']:>5} / {r['dn_pYES']:.3f} / {r['dn_mid']:.3f}")
print("  (theo>mid is right if P(YES) > mid; theo<mid is right if P(YES) < mid)")

print("\n  Disagreement gap (>=5c) by time-to-close offset:")
sub5 = snap[snap['disagree'].abs() >= 0.05]
for off in OFFSETS_S:
    o = sub5[sub5['offset'] == off]
    if o.empty: continue
    g, _, l, h, n = boot_ci(o.groupby('ticker')['gap'].mean())
    print(f"    {LBL[off]:<7} n={len(o):>5}  HAR={o['e_har'].mean():.4f}  "
          f"Mkt={o['e_mkt'].mean():.4f}  gap={g:+.4f}  CI[{l:+.4f},{h:+.4f}]"
          f"{'  *' if (l>0 or h<0) else ''}")

print("\n  Disagreement gap (>=5c) split by SIGN, and by |z|<=2 (quoting regime):")
for name, mask in [('theo>mid', sub5['disagree'] > 0),
                   ('theo<mid', sub5['disagree'] < 0)]:
    o = sub5[mask]
    g, _, l, h, n = boot_ci(o.groupby('ticker')['gap'].mean())
    print(f"    {name:<9} n={len(o):>5}  gap={g:+.4f}  CI[{l:+.4f},{h:+.4f}]"
          f"{'  *' if (l>0 or h<0) else ''}")
o = sub5[sub5['abs_z'] <= 2.0]
g, _, l, h, n = boot_ci(o.groupby('ticker')['gap'].mean())
print(f"    |z|<=2    n={len(o):>5}  gap={g:+.4f}  CI[{l:+.4f},{h:+.4f}]"
      f"{'  *' if (l>0 or h<0) else ''}")


# ===================================================================
# 3. RELIABILITY CURVES  (theo, mid, Platt-theo) identical rows
# ===================================================================
print("\n" + "="*82)
print("3.  RELIABILITY (5c bins) — predicted vs observed YES freq")
print("="*82)
snap['platt'] = expit(PLATT_B * logit(snap['har_theo'].clip(1e-6, 1-1e-6)))
bins = np.arange(0, 1.0001, 0.05)
ctr = (bins[:-1] + bins[1:]) / 2

def reliab(pred_col):
    b = pd.cut(snap[pred_col], bins, include_lowest=True)
    g = snap.groupby(b, observed=False).agg(
        pred=(pred_col, 'mean'), obs=('outcome', 'mean'), n=('outcome', 'size'))
    return g

rel_har, rel_mid, rel_pl = reliab('har_theo'), reliab('mid'), reliab('platt')
print(f"  {'bin':>10}{'n_HAR':>8}{'HAR_pred':>10}{'HAR_obs':>9}"
      f"{'n_mid':>8}{'mid_pred':>10}{'mid_obs':>9}")
for i in range(len(bins)-1):
    lab = f"{bins[i]:.2f}-{bins[i+1]:.2f}"
    rh, rm = rel_har.iloc[i], rel_mid.iloc[i]
    nh = int(rh['n']); nm = int(rm['n'])
    print(f"  {lab:>10}{nh:>8}{rh['pred']:>10.3f}{rh['obs']:>9.3f}"
          f"{nm:>8}{rm['pred']:>10.3f}{rm['obs']:>9.3f}")

def ece(rel):
    w = rel['n'] / rel['n'].sum()
    return float((w * (rel['pred'] - rel['obs']).abs()).fillna(0).sum())
print(f"\n  ECE (weighted |pred-obs|), ALL rows: HAR {ece(rel_har):.4f}  "
      f"mid {ece(rel_mid):.4f}  Platt {ece(rel_pl):.4f}")

# clean quoting regime |z|<=2 — strip the deep-OTM artifact (mid stuck
# ~0.49 while market resolves; that's where mid's miscalibration lives)
clean = snap[snap['abs_z'] <= 2.0]
def reliab_c(col):
    bb = pd.cut(clean[col], bins, include_lowest=True)
    return clean.groupby(bb, observed=False).agg(
        pred=(col, 'mean'), obs=('outcome', 'mean'), n=('outcome', 'size'))
print(f"  ECE (|z|<=2, the regime you quote): HAR {ece(reliab_c('har_theo')):.4f}  "
      f"mid {ece(reliab_c('mid')):.4f}  Platt {ece(reliab_c('platt')):.4f}  "
      f"(n={len(clean):,})")

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], '--', color='#888', lw=1, label='perfect')
for rel, lab, col in [(rel_har, 'HAR theo', '#a78bfa'),
                      (rel_mid, 'Market mid', '#60a5fa'),
                      (rel_pl, 'Platt theo (b=1.116)', '#34d399')]:
    m = rel['n'] > 0
    ax.plot(rel['pred'][m], rel['obs'][m], 'o-', color=col, label=lab, ms=5)
ax.set_xlabel('Predicted P(YES)'); ax.set_ylabel('Observed YES freq')
ax.set_title(f'Reliability — identical rows ({len(snap):,} snaps, '
             f'{snap["ticker"].nunique():,} markets)')
ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(HERE / "theo_vs_market_reliability.png", dpi=120)
print(f"[saved] {HERE/'theo_vs_market_reliability.png'}")


# ===================================================================
# 4. WEEKLY STABILITY  (incl. June regime)
# ===================================================================
print("\n" + "="*82)
print("4.  WEEKLY BRIER-GAP STABILITY")
print("="*82)
wrows = []
for wk, sub in snap.groupby('week'):
    g, _, l, h, n = boot_ci(sub.groupby('ticker')['gap'].mean())
    d0 = sub['close_ts'].min().strftime('%m-%d')
    d1 = sub['close_ts'].max().strftime('%m-%d')
    wrows.append(dict(week=wk, span=f"{d0}->{d1}", n_tkr=n, gap=g, lo=l, hi=h))
    print(f"    wk{wk} {d0}->{d1}  n_tkr={n:>4}  gap={g:+.4f}  "
          f"CI[{l:+.4f},{h:+.4f}]{'  *sig' if (l>0 or h<0) else ''}")
wk_df = pd.DataFrame(wrows)

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(wk_df))
ax.errorbar(x, wk_df['gap'],
            yerr=[wk_df['gap']-wk_df['lo'], wk_df['hi']-wk_df['gap']],
            fmt='o', capsize=4, color='#a78bfa', ecolor='#5a6270', ms=8)
ax.axhline(0, color='#888', lw=1, ls='--')
ax.set_xticks(list(x))
ax.set_xticklabels([f"wk{r.week}\n{r.span}" for r in wk_df.itertuples()], fontsize=8)
ax.set_ylabel('Brier gap (HAR-Mkt)  ·  negative = HAR better')
ax.set_title('Weekly Brier-gap stability — KXETH15M')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(HERE / "theo_vs_market_weekly_gap.png", dpi=120)
print(f"[saved] {HERE/'theo_vs_market_weekly_gap.png'}")


# ===================================================================
# 5. ECONOMIC TRANSLATION  (counterfactual, not a backtest)
# ===================================================================
print("\n" + "="*82)
print("5.  ECONOMIC TRANSLATION — fade mid->theo at |theo-mid|>=5c, T-5m")
print("="*82)
# At T-5m, when theo>mid+5c we BUY yes at the ask (~mid+half_spread); payoff
# = outcome - entry. When theo<mid-5c we SELL yes at the bid (~mid-half_spread);
# payoff = entry - outcome. Pay half-spread either way (cross to take liquidity).
t5 = snap[(snap['offset'] == 300) & (snap['disagree'].abs() >= 0.05)].copy()
t5['side'] = np.where(t5['disagree'] > 0, 1, -1)   # +1 buy yes, -1 sell yes
t5['entry'] = t5['mid'] + t5['side'] * HALF_SPREAD
t5['pnl'] = np.where(t5['side'] == 1,
                     t5['outcome'] - t5['entry'],
                     t5['entry'] - t5['outcome'])      # dollars per 1-lot
# trades/day: one trade per market that disagrees at T-5m; ~96 markets/day
n_days = snap['day'].nunique()
n_mkt_total = snap['ticker'].nunique()
markets_per_day = n_mkt_total / n_days
trades = t5.groupby('ticker')['pnl'].mean()   # one per market (cluster)
g, se_, l, h, n = boot_ci(trades)
trd_per_day = len(t5) / n_days
print(f"  qualifying trades (T-5m, |theo-mid|>=5c): {len(t5)}  over {n_days} days "
      f"= {trd_per_day:.1f}/day")
print(f"  mean $/trade (after {HALF_SPREAD*100:.0f}c half-spread): "
      f"${g:+.4f}  CI[${l:+.4f}, ${h:+.4f}]  {'SIG' if (l>0 or h<0) else 'ns'}")
print(f"  implied $/day (1-lot): ${g*trd_per_day:+.2f}  "
      f"CI[${l*trd_per_day:+.2f}, ${h*trd_per_day:+.2f}]")
# sensitivity: zero-cost (pure forecast edge) and 10c threshold
for thr, hs, lab in [(0.05, 0.0, '5c thr, 0c cost (pure edge)'),
                     (0.10, 0.02, '10c thr, 2c cost')]:
    z = snap[(snap['offset'] == 300) & (snap['disagree'].abs() >= thr)].copy()
    z['side'] = np.where(z['disagree'] > 0, 1, -1)
    z['entry'] = z['mid'] + z['side'] * hs
    z['pnl'] = np.where(z['side'] == 1, z['outcome']-z['entry'], z['entry']-z['outcome'])
    gg, _, ll, hh, _ = boot_ci(z.groupby('ticker')['pnl'].mean())
    tpd = len(z) / n_days
    print(f"  [{lab}] $/trade ${gg:+.4f} CI[${ll:+.4f},${hh:+.4f}]  "
          f"{tpd:.1f}/day  -> ${gg*tpd:+.2f}/day")

# persist extended snapshot frame for reuse
out_csv = HERE / "theo_vs_market_deep_snapshots.csv"
snap.to_csv(out_csv, index=False)
print(f"\n[saved] {out_csv}  ({len(snap):,} rows)")
