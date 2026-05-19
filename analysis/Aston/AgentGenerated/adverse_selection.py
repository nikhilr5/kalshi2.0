"""Adverse-selection measurement for Aston KXETH15M fills.

For every recorded fill compute, in cents per contract:
    spread_capture       = (theo_at_fill - price) * sgn
    adv_sel_{N}s         = (theo_at_fill - theo_at_fill+N) * sgn   (positive = picked off)
    realized_to_close    = (settlement_outcome - price) * sgn      (full per-fill PnL)
    total_after_fill     = realized_to_close - spread_capture      (adv_sel + settle variance)

Bucket by (action × ttc × moneyness) with bootstrap 95% CIs and plot.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    SECONDS_PER_YEAR,
    bootstrap_ci,
    fetch_settlements_from_api,
    load_all_data,
    theo_vec_twap,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
HORIZONS_S    = [30, 60, 120, 300]


# =============================================================================
# Load + merge theo_state at fill time
# =============================================================================
theo, _book, _spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo  = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
fills = fills.sort_values('ts').reset_index(drop=True)

if 'side' in fills.columns and (fills['side'] != 'yes').any():
    n_other = int((fills['side'] != 'yes').sum())
    print(f"[warn] dropping {n_other} non-yes fills — analysis assumes yes-side only")
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

theo_cols = ['ts', 'ticker', 'spot', 'sigma', 'theo', 'seconds_to_expiry']
fills = pd.merge_asof(
    fills, theo[theo_cols].sort_values('ts'),
    on='ts', by='ticker', direction='backward',
)

# Theo at fill + δ for each horizon (used for adverse-selection cost).
theo_lookup = theo[['ts', 'ticker', 'theo']].sort_values('ts')
for d in HORIZONS_S:
    fwd = fills[['ts', 'ticker']].copy()
    fwd['ts'] = fwd['ts'] + pd.Timedelta(seconds=d)
    fwd = fwd.sort_values('ts')
    j = pd.merge_asof(
        fwd, theo_lookup, on='ts', by='ticker', direction='backward',
    )
    fills[f'theo_p{d}'] = j['theo'].values


# =============================================================================
# Settlements + per-fill economics
# =============================================================================
api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome'] = fills['ticker'].map(settlements)

fills['sgn'] = np.where(fills['action'] == 'buy', +1, -1)
fills['spread_capture_c']    = (fills['theo'] - fills['price'])   * fills['sgn'] * 100
fills['realized_to_close_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100
fills['total_after_fill_c']  = fills['realized_to_close_c'] - fills['spread_capture_c']
for d in HORIZONS_S:
    fills[f'adv_sel_c_{d}'] = (fills['theo'] - fills[f'theo_p{d}']) * fills['sgn'] * 100

# TWAP-Asian binary theo at fill time — same inputs but with effective T
# reduced for settlement-on-average.  Defaults to 60s TWAP window.
fills['theo_twap'] = theo_vec_twap(
    fills['spot'], fills['strike'], fills['sigma'], fills['seconds_to_expiry'],
    twap_window_s=60,
)
fills['spread_capture_twap_c']   = (fills['theo_twap'] - fills['price']) * fills['sgn'] * 100
fills['total_after_fill_twap_c'] = fills['realized_to_close_c'] - fills['spread_capture_twap_c']
fills['theo_shift_c']            = (fills['theo_twap'] - fills['theo']) * 100


# =============================================================================
# Buckets — time-to-close and moneyness in σ-units
# =============================================================================
fills['ttc_bucket'] = pd.cut(
    fills['seconds_to_expiry'],
    bins=[-1, 60, 300, 600, 1e9],
    labels=['<1m', '1-5m', '5-10m', '>10m'],
)
T_years = fills['seconds_to_expiry'].clip(lower=1) / SECONDS_PER_YEAR
fills['mny_z'] = (fills['spot'] - fills['strike']).abs() / (
    fills['spot'] * fills['sigma'] * np.sqrt(T_years)
)
fills['mny_bucket'] = pd.cut(
    fills['mny_z'],
    bins=[-0.01, 0.25, 0.75, 1.5, 999],
    labels=['atm', 'near', 'far', 'tail'],
)


# =============================================================================
# Aggregation with bootstrap CIs
# =============================================================================
METRIC_COLS = (['spread_capture_c']
               + [f'adv_sel_c_{d}' for d in HORIZONS_S]
               + ['realized_to_close_c', 'total_after_fill_c'])


def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g)}
    for c in METRIC_COLS:
        v = g[c].dropna().values
        if len(v) < 2:
            out[c] = np.nan
            out[f'{c}_lo'] = np.nan
            out[f'{c}_hi'] = np.nan
            continue
        lo, hi = bootstrap_ci(v)
        out[c] = float(np.mean(v))
        out[f'{c}_lo'] = lo
        out[f'{c}_hi'] = hi
    return pd.Series(out)


pd.set_option('display.float_format', '{:+.2f}'.format)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

print("\n=== Headline: by action ===")
print(fills.groupby('action').apply(summary, include_groups=False))

print("\n=== By action × ttc_bucket ===")
print(fills.groupby(['action', 'ttc_bucket'], observed=True)
            .apply(summary, include_groups=False))

print("\n=== By action × ttc_bucket × mny_bucket ===")
print(fills.groupby(['action', 'ttc_bucket', 'mny_bucket'], observed=True)
            .apply(summary, include_groups=False))


# =============================================================================
# TWAP-Asian test — does using TWAP theo reduce the gap between paper edge
# and realized PnL?  Spread-capture under TWAP vs European, same outcomes.
# =============================================================================
print("\n=== TWAP vs European theo — overall ===")
cmp_overall = (fills.groupby('action')
                    [['theo_shift_c',
                      'spread_capture_c', 'spread_capture_twap_c',
                      'total_after_fill_c', 'total_after_fill_twap_c',
                      'realized_to_close_c']]
                    .mean())
cmp_overall['gap_eur'] = (cmp_overall['realized_to_close_c']
                          - cmp_overall['spread_capture_c'])
cmp_overall['gap_twap'] = (cmp_overall['realized_to_close_c']
                           - cmp_overall['spread_capture_twap_c'])
print(cmp_overall)

print("\n=== TWAP vs European — by action × ttc × mny ===")
cmp_bucket = (fills.groupby(['action', 'ttc_bucket', 'mny_bucket'], observed=True)
                    [['theo_shift_c',
                      'spread_capture_c', 'spread_capture_twap_c',
                      'total_after_fill_c', 'total_after_fill_twap_c',
                      'realized_to_close_c']]
                    .mean())
cmp_bucket['gap_eur']  = cmp_bucket['realized_to_close_c']  - cmp_bucket['spread_capture_c']
cmp_bucket['gap_twap'] = cmp_bucket['realized_to_close_c']  - cmp_bucket['spread_capture_twap_c']
cmp_bucket['n']        = (fills.groupby(['action', 'ttc_bucket', 'mny_bucket'],
                                         observed=True).size())
print(cmp_bucket.to_string())


# =============================================================================
# Viz — bar chart with CI bars + heatmap of realized PnL by ttc × mny
# =============================================================================
bar_df = (fills.groupby(['action', 'ttc_bucket'], observed=True)
                .apply(summary, include_groups=False)
                .reset_index())

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Mean per-fill economics by action × time-to-close (95% bootstrap CI)",
        "Realized P&L to close — heatmap (cents/fill, action ⊕ ttc × moneyness)",
    ),
    row_heights=[0.55, 0.45], vertical_spacing=0.12,
)

bar_metrics = [('spread_capture_c',    '#22c55e', 'spread capture'),
               ('adv_sel_c_300',       '#facc15', 'adv sel @ +300s'),
               ('total_after_fill_c',  '#f87171', 'total after fill'),
               ('realized_to_close_c', '#a78bfa', 'realized to close')]

for col, color, name in bar_metrics:
    for action in ['buy', 'sell']:
        d = bar_df[bar_df['action'] == action]
        x = [f"{action}/{t}" for t in d['ttc_bucket']]
        err_lo = (d[col] - d[f'{col}_lo']).clip(lower=0)
        err_hi = (d[f'{col}_hi'] - d[col]).clip(lower=0)
        fig.add_trace(go.Bar(
            x=x, y=d[col], name=name if action == 'buy' else None,
            marker_color=color, opacity=0.85,
            error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                          color='#333', thickness=1),
            legendgroup=name, showlegend=(action == 'buy'),
        ), row=1, col=1)

heat = (fills.groupby(['action', 'ttc_bucket', 'mny_bucket'], observed=True)
              ['realized_to_close_c'].mean().reset_index())
heat['row'] = heat['action'].astype(str) + ' / ' + heat['ttc_bucket'].astype(str)
pivot = heat.pivot(index='row', columns='mny_bucket', values='realized_to_close_c')
fig.add_trace(go.Heatmap(
    z=pivot.values, x=[str(c) for c in pivot.columns],
    y=[str(r) for r in pivot.index],
    colorscale='RdYlGn', zmid=0,
    text=np.round(pivot.values, 1), texttemplate='%{text}',
    colorbar=dict(title='¢/fill'),
), row=2, col=1)

fig.update_yaxes(title_text='cents per fill', row=1, col=1)
fig.update_layout(
    title=f"Adverse selection — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills · {len(settlements)} settled markets",
    template='plotly_dark', height=1000, barmode='group', bargap=0.15,
)
fig.show()
