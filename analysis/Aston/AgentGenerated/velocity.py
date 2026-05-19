"""Theo velocity + spot velocity at fill time vs realized PnL.

For each fill, compute:
  • theo_vel_Ns_cps   = (theo_at_fill - theo_at_fill_minus_N) / N   [cents/sec]
  • spot_vel_Ns_bps_s = (spot_at_fill - spot_at_fill_minus_N) / spot / N * 10000
                                                                    [basis-points/sec]
  • adverse versions: −sgn * velocity   (positive = moving against us)

Bucket fills by adverse velocity, look at realized PnL.  If high
velocity predicts bleed, Aston has a real-time quoting gate.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    bootstrap_ci,
    fetch_settlements_from_api,
    load_all_data,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
LAGS_S        = [1, 5, 30]


# =============================================================================
# Load + theo at fill time
# =============================================================================
theo, _book, spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo  = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
spot  = spot.sort_values('ts').reset_index(drop=True)
fills = fills.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

theo_lookup = theo[['ts', 'ticker', 'theo', 'seconds_to_expiry']].sort_values('ts')
fills = pd.merge_asof(fills, theo_lookup, on='ts', by='ticker', direction='backward')
fills['_row'] = np.arange(len(fills))


# =============================================================================
# Theo + spot velocity lookbacks
# =============================================================================
def lag_lookup(ts_offset_s: int, lookup: pd.DataFrame,
               by: str | None = None) -> pd.DataFrame:
    back = fills[['_row', 'ts'] + ([by] if by else [])].copy()
    back['ts'] = back['ts'] - pd.Timedelta(seconds=ts_offset_s)
    back = back.sort_values('ts')
    if by:
        return pd.merge_asof(back, lookup, on='ts', by=by,
                              direction='backward').sort_values('_row')
    return pd.merge_asof(back, lookup, on='ts',
                          direction='backward').sort_values('_row')


for d in LAGS_S:
    j = lag_lookup(d, theo_lookup[['ts', 'ticker', 'theo']], by='ticker')
    fills[f'theo_lag_{d}'] = j['theo'].values
    fills[f'theo_vel_{d}_cps'] = (fills['theo'] - fills[f'theo_lag_{d}']) / d * 100

spot_lookup = spot[['ts', 'price']].sort_values('ts')
m = lag_lookup(0, spot_lookup)
fills['spot_now'] = m['price'].values
for d in LAGS_S:
    j = lag_lookup(d, spot_lookup)
    fills[f'spot_lag_{d}'] = j['price'].values
    fills[f'spot_vel_{d}_bps_s'] = (
        (fills['spot_now'] - fills[f'spot_lag_{d}'])
        / fills[f'spot_lag_{d}'] * 10000 / d
    )

fills['sgn'] = np.where(fills['action'] == 'buy', +1, -1)
for d in LAGS_S:
    fills[f'adv_theo_vel_{d}'] = -fills['sgn'] * fills[f'theo_vel_{d}_cps']
    fills[f'adv_spot_vel_{d}'] = -fills['sgn'] * fills[f'spot_vel_{d}_bps_s']


# =============================================================================
# Settlements + realized
# =============================================================================
api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100

fills['ttc_bucket'] = pd.cut(
    fills['seconds_to_expiry'],
    bins=[-1, 60, 300, 600, 1e9], labels=['<1m', '1-5m', '5-10m', '>10m'],
)
fills['theo_vel_5_bucket'] = pd.cut(
    fills['adv_theo_vel_5'],
    bins=[-1e6, -0.5, -0.1, 0.1, 0.5, 1e6],
    labels=['for us fast', 'for us mild', 'flat', 'against mild', 'against fast'],
)
fills['spot_vel_5_bucket'] = pd.cut(
    fills['adv_spot_vel_5'],
    bins=[-1e6, -2, -0.5, 0.5, 2, 1e6],
    labels=['for us fast', 'for us mild', 'flat', 'against mild', 'against fast'],
)


# =============================================================================
# Summary
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'theo_vel_5_med': g['adv_theo_vel_5'].median(),
           'spot_vel_5_med': g['adv_spot_vel_5'].median()}
    v = g['realized_c'].dropna().values
    if len(v) >= 2:
        lo, hi = bootstrap_ci(v, B=2000)
        out['realized'] = float(v.mean())
        out['lo'] = lo
        out['hi'] = hi
    else:
        out['realized'] = out['lo'] = out['hi'] = np.nan
    return pd.Series(out)


pd.set_option('display.float_format', '{:+.2f}'.format)
pd.set_option('display.width', 200)

print(f"\n=== Theo velocity @ 5s (adverse, cents/sec) ===")
print(fills.groupby('theo_vel_5_bucket', observed=True)
            .apply(summary, include_groups=False))

print(f"\n=== Spot velocity @ 5s (adverse, bps/sec) ===")
print(fills.groupby('spot_vel_5_bucket', observed=True)
            .apply(summary, include_groups=False))

print(f"\n=== Theo velocity × Spot velocity ===")
print(fills.groupby(['theo_vel_5_bucket', 'spot_vel_5_bucket'], observed=True)
            .apply(summary, include_groups=False))

print(f"\n=== Theo velocity × ttc_bucket ===")
print(fills.groupby(['ttc_bucket', 'theo_vel_5_bucket'], observed=True)
            .apply(summary, include_groups=False))

print(f"\n=== Correlations with realized PnL ===")
cols = (['adv_theo_vel_1', 'adv_theo_vel_5', 'adv_theo_vel_30',
         'adv_spot_vel_1', 'adv_spot_vel_5', 'adv_spot_vel_30'])
for c in cols:
    df = fills[[c, 'realized_c']].dropna()
    if len(df) >= 10:
        print(f"  ρ({c:>20s}, realized) = {df[c].corr(df['realized_c']):+.4f}  (n={len(df)})")


# =============================================================================
# Viz
# =============================================================================
theo_df = (fills.groupby('theo_vel_5_bucket', observed=True)
                 .apply(summary, include_groups=False).reset_index())
spot_df = (fills.groupby('spot_vel_5_bucket', observed=True)
                 .apply(summary, include_groups=False).reset_index())

cmb = (fills.groupby(['theo_vel_5_bucket', 'spot_vel_5_bucket'], observed=True)
              ['realized_c'].agg(['mean', 'count']).reset_index())
heat = cmb.pivot(index='theo_vel_5_bucket', columns='spot_vel_5_bucket', values='mean')
heat_n = cmb.pivot(index='theo_vel_5_bucket', columns='spot_vel_5_bucket', values='count')

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Realized P&L by theo velocity (5s lookback)",
        "Realized P&L by spot velocity (5s lookback)",
        "Realized P&L heatmap — theo vel × spot vel (cents/fill)",
        "Sample counts per (theo vel × spot vel) cell",
    ),
    horizontal_spacing=0.12, vertical_spacing=0.18,
)
colors = ['#22c55e', '#84cc16', '#facc15', '#f97316', '#dc2626']

def add_bar(df, row, col):
    err_lo = (df['realized'] - df['lo']).clip(lower=0)
    err_hi = (df['hi'] - df['realized']).clip(lower=0)
    bucket_col = [c for c in df.columns if c.endswith('bucket')][0]
    labels = [f"{b}<br>(n={int(n)})" for b, n in zip(df[bucket_col], df['n'])]
    fig.add_trace(go.Bar(
        x=labels, y=df['realized'], marker_color=colors[:len(df)],
        error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                      color='#333', thickness=1.5),
        showlegend=False,
    ), row=row, col=col)
    fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'),
                  row=row, col=col)
    fig.update_yaxes(title_text='realized P&L (cents)', row=row, col=col)

add_bar(theo_df, 1, 1)
add_bar(spot_df, 1, 2)

fig.add_trace(go.Heatmap(
    z=heat.values, x=[str(c) for c in heat.columns],
    y=[str(r) for r in heat.index],
    colorscale='RdYlGn', zmid=0,
    text=np.round(heat.values, 1), texttemplate='%{text}',
    colorbar=dict(title='¢', x=0.46, len=0.4, y=0.22),
), row=2, col=1)
fig.add_trace(go.Heatmap(
    z=heat_n.values, x=[str(c) for c in heat_n.columns],
    y=[str(r) for r in heat_n.index],
    colorscale='Blues',
    text=heat_n.values.astype(int), texttemplate='%{text}',
    colorbar=dict(title='n', x=1.0, len=0.4, y=0.22),
), row=2, col=2)
fig.update_xaxes(title_text='spot velocity bucket', row=2, col=1)
fig.update_xaxes(title_text='spot velocity bucket', row=2, col=2)
fig.update_yaxes(title_text='theo velocity bucket', row=2, col=1)
fig.update_yaxes(title_text='theo velocity bucket', row=2, col=2)

fig.update_layout(
    title=f"Theo + spot velocity at fill time — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills",
    template='plotly_dark', height=1000,
)
fig.show()
