"""Quote age + theo drift during quote lifetime — quantifying the
tolerance-gate bleed.

For each fill, reconstruct from order_events 'placed' rows:
  • quote_age_s   = fill_ts − placed_ts        (how long was order resting)
  • theo_drift_c  = (theo_at_fill − theo_at_placed) * sgn * 100
        positive = theo moved in our favor while resting
        negative = theo moved against us (we got picked off)
  • sub-tolerance flag = |theo_drift_c| < TOLERANCE_CENTS
        these fills wouldn't have triggered a reprice under current logic

Bucket fills, compare realized P&L.  Hypothesis: against-us drift fills
bleed; favor-us drift fills profit; the gap is what asymmetric tolerance
would have recovered.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
TOLERANCE_C   = 1.0  # cents — current Aston reprice tolerance


# =============================================================================
# Load
# =============================================================================
theo, _book, _spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo   = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
fills  = fills.sort_values('ts').reset_index(drop=True)
events = events.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

placed = (events[events['event_type'] == 'placed']
          [['ts', 'client_order_id', 'price']]
          .rename(columns={'ts': 'placed_ts', 'price': 'placed_price'})
          .sort_values('placed_ts')
          .drop_duplicates('client_order_id', keep='first'))

fills = fills.merge(placed, on='client_order_id', how='left')
coverage = fills['placed_ts'].notna().mean()
print(f"\n[coverage] {coverage:.1%} of fills matched to a placed event "
      f"({fills['placed_ts'].notna().sum()}/{len(fills)})")

fills = fills.dropna(subset=['placed_ts']).reset_index(drop=True)
fills['quote_age_s'] = (fills['ts'] - fills['placed_ts']).dt.total_seconds()
fills['_row'] = np.arange(len(fills))


# =============================================================================
# Look up theo at placement and at fill
# =============================================================================
theo_lookup = theo[['ts', 'ticker', 'theo', 'seconds_to_expiry']].sort_values('ts')


def merge_theo_at(ts_col: str) -> pd.DataFrame:
    df = (fills[['_row', ts_col, 'ticker']]
          .rename(columns={ts_col: 'ts'})
          .sort_values('ts'))
    return (pd.merge_asof(df, theo_lookup, on='ts', by='ticker',
                          direction='backward')
              .sort_values('_row'))


m_p = merge_theo_at('placed_ts')
fills['theo_at_placed'] = m_p['theo'].values
m_f = merge_theo_at('ts')
fills['theo_at_fill']       = m_f['theo'].values
fills['seconds_to_expiry']  = m_f['seconds_to_expiry'].values

fills = fills.dropna(subset=['theo_at_placed', 'theo_at_fill']).reset_index(drop=True)


# =============================================================================
# Drift, settlements, realized
# =============================================================================
fills['sgn']          = np.where(fills['action'] == 'buy', +1, -1)
fills['theo_drift_c'] = (fills['theo_at_fill'] - fills['theo_at_placed']) * fills['sgn'] * 100

api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100


# =============================================================================
# Buckets
# =============================================================================
fills['quote_age_bucket'] = pd.cut(
    fills['quote_age_s'],
    bins=[-1, 1, 5, 30, 120, 1e9],
    labels=['<1s', '1-5s', '5-30s', '30-120s', '>2min'],
)
fills['drift_bucket'] = pd.cut(
    fills['theo_drift_c'],
    bins=[-1e9, -TOLERANCE_C, -0.1, 0.1, TOLERANCE_C, 1e9],
    labels=[f'against >{TOLERANCE_C}¢',
            f'against 0.1-{TOLERANCE_C}¢ (sub-tol)',
            'flat (±0.1¢)',
            f'favor 0.1-{TOLERANCE_C}¢ (sub-tol)',
            f'favor >{TOLERANCE_C}¢'],
)


# =============================================================================
# Summary
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'drift_c': g['theo_drift_c'].mean(),
           'age_med_s': g['quote_age_s'].median()}
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
pd.set_option('display.max_columns', None)

print(f"\n=== After all filters: {len(fills):,} fills ===")

print("\n=== By quote_age bucket ===")
print(fills.groupby('quote_age_bucket', observed=True)
            .apply(summary, include_groups=False))

print("\n=== By theo_drift bucket (THE KEY TABLE) ===")
print(fills.groupby('drift_bucket', observed=True)
            .apply(summary, include_groups=False))

print("\n=== By quote_age × drift_bucket ===")
print(fills.groupby(['quote_age_bucket', 'drift_bucket'], observed=True)
            .apply(summary, include_groups=False))


# =============================================================================
# Tolerance-gate impact estimate
# =============================================================================
print(f"\n=== Tolerance-gate impact (current setting: {TOLERANCE_C}¢) ===")
against     = fills[fills['theo_drift_c'] < -0.1]
sub_against = fills[(fills['theo_drift_c'] < -0.1)
                    & (fills['theo_drift_c'] > -TOLERANCE_C)]
favor       = fills[fills['theo_drift_c'] > 0.1]

n_days = 4
print(f"  Against-us drift fills (any):        n={len(against):>5}, "
      f"realized {against['realized_c'].mean():+.2f}¢/fill")
print(f"  Sub-tolerance against-us (<{TOLERANCE_C}¢):  n={len(sub_against):>5}, "
      f"realized {sub_against['realized_c'].mean():+.2f}¢/fill  "
      f"<-- the bucket asymmetric tolerance would recover")
print(f"  Favor-us drift fills (any):          n={len(favor):>5}, "
      f"realized {favor['realized_c'].mean():+.2f}¢/fill")

if len(sub_against) > 0:
    delta_per_fill = favor['realized_c'].mean() - sub_against['realized_c'].mean()
    total_recoverable = delta_per_fill * len(sub_against) / 100  # to dollars
    print(f"\n  Estimated PnL recovery from asymmetric tolerance:")
    print(f"    if sub-tol against-us fills had performed like favor fills: "
          f"${total_recoverable:.2f} over {n_days} days = ${total_recoverable/n_days:.2f}/day")


# =============================================================================
# Viz
# =============================================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

bar_df = (fills.groupby('drift_bucket', observed=True)
                .apply(summary, include_groups=False).reset_index())
age_df = (fills.groupby('quote_age_bucket', observed=True)
                .apply(summary, include_groups=False).reset_index())

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Realized P&L by theo drift during quote lifetime (the tolerance-gate test)",
        "Realized P&L by quote age",
    ),
    row_heights=[0.5, 0.5], vertical_spacing=0.15,
)

drift_colors = ['#dc2626', '#f97316', '#facc15', '#84cc16', '#22c55e']
err_lo = (bar_df['realized'] - bar_df['lo']).clip(lower=0)
err_hi = (bar_df['hi'] - bar_df['realized']).clip(lower=0)
labels = [f"{b}<br>(n={int(n)})" for b, n in zip(bar_df['drift_bucket'], bar_df['n'])]
fig.add_trace(go.Bar(
    x=labels, y=bar_df['realized'],
    marker_color=drift_colors[:len(bar_df)],
    error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                  color='#333', thickness=1.5),
    showlegend=False,
), row=1, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=1, col=1)

err_lo = (age_df['realized'] - age_df['lo']).clip(lower=0)
err_hi = (age_df['hi'] - age_df['realized']).clip(lower=0)
age_labels = [f"{b}<br>(n={int(n)})" for b, n in zip(age_df['quote_age_bucket'], age_df['n'])]
fig.add_trace(go.Bar(
    x=age_labels, y=age_df['realized'], marker_color='#a78bfa',
    error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                  color='#333', thickness=1.5),
    showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)

fig.update_yaxes(title_text='realized P&L (cents)', row=1, col=1)
fig.update_yaxes(title_text='realized P&L (cents)', row=2, col=1)
fig.update_layout(
    title=f"Quote age and theo drift — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills, {coverage:.0%} coverage",
    template='plotly_dark', height=900,
)
fig.show()
