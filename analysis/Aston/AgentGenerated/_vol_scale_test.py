"""Test: does sigma * 0.88 give the same correction as Platt b=1.116?
If yes: bias is pure vol overestimate.
If no (esp. in tails): bias is distributional / fat tails."""
import sys, sqlite3, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

ANN_MIN = 525960  # crypto trading minutes/year

def theo_nd2(spot, strike, sigma_ann, secs):
    """N(d2) probability of finishing above strike."""
    T = secs / 60.0 / ANN_MIN
    sigma_T = sigma_ann * np.sqrt(T)
    sigma_T = np.where(sigma_T < 1e-6, 1e-6, sigma_T)
    d2 = (np.log(spot / strike) - 0.5 * sigma_T**2) / sigma_T
    return norm.cdf(d2)

LOCAL = Path('analysis/backtesting/data')
S3C   = Path('analysis/backtesting/_s3_cache')
paths = {}
for p in list(LOCAL.glob('KXETH15M-*.db')) + list(S3C.glob('KXETH15M-*.db')):
    if 'MAY30' not in p.name:
        paths.setdefault(p.name, p)
files = sorted(paths.values(), key=lambda x: x.name)
print(f'[load] {len(files)} files')

dfs = []
for f in files:
    c = sqlite3.connect(str(f))
    try:
        d = pd.read_sql('SELECT ts, ticker, spot, strike, sigma, theo, seconds_to_expiry FROM theo_state', c)
        dfs.append(d)
    finally:
        c.close()
d = pd.concat(dfs, ignore_index=True)
d['ts'] = pd.to_datetime(d['ts'], utc=True, format='ISO8601')
print(f'[load] {len(d):,} rows')

sett = {k: int(v) for k, v in json.load(open('analysis/Aston/.settlements_cache.json')).items()}
d['outcome'] = d['ticker'].map(sett)
d = d.dropna(subset=['outcome', 'spot', 'strike', 'sigma'])
d = d[(d['sigma'] > 0) & (d['spot'] > 0) & (d['strike'] > 0)]

# Bucket: per-ticker, per-5sec-bucket (same downsampling as refit)
g = d.groupby('ticker').agg(close_ts=('ts','max'), min_secs=('seconds_to_expiry','min')).reset_index()
g['close_time'] = g['close_ts'] + pd.to_timedelta(g['min_secs'], unit='s')
d = d.merge(g[['ticker','close_time']], on='ticker')
d['secs_to_close'] = (d['close_time'] - d['ts']).dt.total_seconds()
d = d[(d['secs_to_close'] >= 30) & (d['secs_to_close'] <= 720)]
d['bucket'] = (d['secs_to_close'] // 5).astype(int)
d = d.sort_values(['ticker','bucket','secs_to_close']).drop_duplicates(['ticker','bucket'], keep='first')
print(f'[fit] {len(d):,} downsampled pairs')

# ---- Compute three theos ----
d['theo_raw']  = theo_nd2(d['spot'], d['strike'], d['sigma'],        d['seconds_to_expiry'])
d['theo_v88']  = theo_nd2(d['spot'], d['strike'], d['sigma'] * 0.88, d['seconds_to_expiry'])

a, b = 0.107, 1.116
z = logit(d['theo_raw'].clip(1e-4, 1-1e-4))
d['theo_platt'] = expit(a + b * z)

# Sanity: stored theo vs recomputed raw
diff = (d['theo'] - d['theo_raw']).abs()
print(f'[check] stored vs recomputed theo: max |diff|={diff.max():.4f}, mean={diff.mean():.4f}')

# ---- Moneyness bins (using raw theo as the bucket key) ----
bins = [0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]
labels = ['0-10 (deep OTM)', '10-25', '25-40', '40-50', '50-60', '60-75', '75-90', '90-100 (deep ITM)']
d['theo_bin'] = pd.cut(d['theo_raw'], bins=bins, labels=labels, include_lowest=True)

y = d['outcome'].astype(float)
rows = []
for label, grp in d.groupby('theo_bin', observed=True):
    yv = grp['outcome'].astype(float).values
    n = len(grp)
    if n < 50:
        continue
    b_raw   = ((grp['theo_raw']   - yv)**2).mean()
    b_v88   = ((grp['theo_v88']   - yv)**2).mean()
    b_platt = ((grp['theo_platt'] - yv)**2).mean()
    rows.append((label, n,
                 grp['theo_raw'].mean(), yv.mean(),
                 b_raw, b_v88, b_platt,
                 b_raw - b_v88, b_raw - b_platt))

r = pd.DataFrame(rows, columns=['bin','n','mean_theo','yes_rate','B_raw','B_v88','B_platt','Δv88','Δplatt'])
print()
print('Brier by theo bin')
print('=' * 100)
print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)))
print()
print('Overall Brier:')
print(f'  raw:        {((d["theo_raw"]   - y)**2).mean():.4f}')
print(f'  σ × 0.88:   {((d["theo_v88"]   - y)**2).mean():.4f}')
print(f'  Platt:      {((d["theo_platt"] - y)**2).mean():.4f}')
