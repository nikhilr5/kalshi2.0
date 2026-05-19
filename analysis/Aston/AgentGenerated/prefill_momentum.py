"""Pre-fill spot momentum — was the market already moving against us
before our fills landed?

For each fill, compute the ETH spot move over the previous 30/60/120/300s
using spot_ticks (higher frequency than theo_state).  Define:

    adverse_pre_{N}_bps = -sgn * (spot_now - spot_{N}_ago) / spot * 10000

Positive value = spot was moving in the direction that hurts our new
position before the fill happened.  Filled-buys preceded by spot drops
and filled-sells preceded by spot rises are the smoking gun for
informed flow (or for stale quoting if theo lags spot).
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
LAGS_S        = [30, 60, 120, 300]


# =============================================================================
# Load + identify spot at fill time (from spot_ticks, higher freq than theo_state)
# =============================================================================
theo, _book, spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
spot  = spot.sort_values('ts').reset_index(drop=True)
fills = fills.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

fills['_row'] = np.arange(len(fills))
fills_keys = fills[['ts', '_row']].sort_values('ts')

spot_now = pd.merge_asof(
    fills_keys, spot[['ts', 'price']],
    on='ts', direction='backward',
    tolerance=pd.Timedelta(seconds=10),
).sort_values('_row')
fills['spot_now'] = spot_now['price'].values

for d in LAGS_S:
    back = fills[['ts', '_row']].copy()
    back['ts'] = back['ts'] - pd.Timedelta(seconds=d)
    back = back.sort_values('ts')
    j = pd.merge_asof(
        back, spot[['ts', 'price']], on='ts', direction='backward',
    ).sort_values('_row')
    fills[f'spot_lag_{d}'] = j['price'].values
    fills[f'spot_move_{d}_bps'] = (
        (fills['spot_now'] - fills[f'spot_lag_{d}'])
        / fills[f'spot_lag_{d}'] * 10000
    )


# =============================================================================
# Adverse direction + ttc bucket + realized PnL
# =============================================================================
fills['sgn'] = np.where(fills['action'] == 'buy', +1, -1)
for d in LAGS_S:
    fills[f'adverse_pre_{d}_bps'] = -fills['sgn'] * fills[f'spot_move_{d}_bps']

theo = theo.sort_values('ts')
fills = pd.merge_asof(
    fills, theo[['ts', 'ticker', 'seconds_to_expiry']],
    on='ts', by='ticker', direction='backward',
)
fills['ttc_bucket'] = pd.cut(
    fills['seconds_to_expiry'],
    bins=[-1, 60, 300, 600, 1e9], labels=['<1m', '1-5m', '5-10m', '>10m'],
)

api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome'] = fills['ticker'].map(settlements)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100


# =============================================================================
# Summary tables with bootstrap CIs
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g)}
    for d in LAGS_S:
        v = g[f'adverse_pre_{d}_bps'].dropna().values
        if len(v) < 2:
            out[f'adv_{d}'] = np.nan
            out[f'adv_{d}_lo'] = np.nan
            out[f'adv_{d}_hi'] = np.nan
            continue
        lo, hi = bootstrap_ci(v)
        out[f'adv_{d}'] = float(v.mean())
        out[f'adv_{d}_lo'] = lo
        out[f'adv_{d}_hi'] = hi
    return pd.Series(out)


pd.set_option('display.float_format', '{:+.2f}'.format)
pd.set_option('display.width', 200)

print("\n=== Adverse pre-fill spot move (bps) by action ===")
print("(positive = spot moved against us in the seconds before our fill)")
print(fills.groupby('action').apply(summary, include_groups=False))

print("\n=== By action × ttc_bucket ===")
print(fills.groupby(['action', 'ttc_bucket'], observed=True)
            .apply(summary, include_groups=False))

print("\n=== Correlation: pre-fill adverse bps vs realized PnL (cents) ===")
for d in LAGS_S:
    df = fills[['action', f'adverse_pre_{d}_bps', 'realized_c']].dropna()
    rows = []
    for a, g in df.groupby('action'):
        if len(g) < 10:
            continue
        rows.append((a, len(g), g[f'adverse_pre_{d}_bps'].corr(g['realized_c'])))
    print(f"  {d:>3}s:  " + "   ".join(f"{a}: ρ={c:+.3f} (n={n})"
                                        for a, n, c in rows))


# =============================================================================
# Viz — bar chart of mean adverse move by action × ttc, plus scatter
# =============================================================================
bar_df = (fills.groupby(['action', 'ttc_bucket'], observed=True)
                .apply(summary, include_groups=False).reset_index())

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Mean adverse pre-fill spot move (bps) — positive = market moved against us",
        "Pre-fill spot move @60s (bps) vs realized PnL per fill (cents)",
    ),
    row_heights=[0.45, 0.55], vertical_spacing=0.15,
)

colors = {30: '#facc15', 60: '#f97316', 120: '#f87171', 300: '#dc2626'}
for d, c in colors.items():
    for action in ['buy', 'sell']:
        sub = bar_df[bar_df['action'] == action]
        x = [f"{action}/{t}" for t in sub['ttc_bucket']]
        mid = sub[f'adv_{d}']
        err_lo = (mid - sub[f'adv_{d}_lo']).clip(lower=0)
        err_hi = (sub[f'adv_{d}_hi'] - mid).clip(lower=0)
        fig.add_trace(go.Bar(
            x=x, y=mid, name=f'{d}s', marker_color=c, opacity=0.85,
            error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                          color='#333', thickness=1),
            legendgroup=f'{d}s', showlegend=(action == 'buy'),
        ), row=1, col=1)

for action, c in [('buy', '#22c55e'), ('sell', '#a78bfa')]:
    sub = fills[fills['action'] == action].dropna(
        subset=['adverse_pre_60_bps', 'realized_c'])
    fig.add_trace(go.Scatter(
        x=sub['adverse_pre_60_bps'], y=sub['realized_c'],
        mode='markers', name=f'{action} (n={len(sub)})',
        marker=dict(size=3, color=c, opacity=0.4),
    ), row=2, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)
fig.add_vline(x=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)

fig.update_yaxes(title_text='bps', row=1, col=1)
fig.update_xaxes(title_text='adverse pre-fill spot move @60s (bps)', row=2, col=1)
fig.update_yaxes(title_text='realized PnL (cents/fill)', row=2, col=1)
fig.update_layout(
    title=f"Pre-fill spot momentum — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills",
    template='plotly_dark', height=950, barmode='group',
)
fig.show()
