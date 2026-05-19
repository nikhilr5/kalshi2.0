"""Optimal-edge analysis — E[$/placement] vs intended edge.

For each `placed` event in order_events:
  • edge_at_placement = (theo_at_placement - placement_price) * sgn * 100
  • outcome: did the order fill or cancel?
  • if filled: what was realized P&L?

Bucket placements by edge_c and compute:
  fill_rate              = n_filled / n_placed
  mean_realized_c        = realized P&L per fill in this bucket
  EV_per_placement_c     = fill_rate * mean_realized_c
                         = the optimization target

Peak of EV_per_placement_c per side (buy/sell) is the optimal edge.

Caveat: n_placed includes cancel-replace churn — most placements are
replacements that get re-replaced before filling, so absolute fill rates
look low.  Relative comparisons across buckets are still valid.
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


# =============================================================================
# Load + filter
# =============================================================================
theo, _book, _spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo   = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
fills  = fills.sort_values('ts').reset_index(drop=True)
events = events.sort_values('ts').reset_index(drop=True)

if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

placed = (events[events['event_type'] == 'placed']
          .sort_values('ts')
          .drop_duplicates('client_order_id', keep='first')
          .reset_index(drop=True))
placed = placed[placed['side'] == 'yes'].reset_index(drop=True)
placed['sgn'] = np.where(placed['action'] == 'buy', +1, -1)


# =============================================================================
# Theo at placement → edge_c
# =============================================================================
theo_lookup = theo[['ts', 'ticker', 'theo']].sort_values('ts')
placed = pd.merge_asof(
    placed.sort_values('ts'), theo_lookup,
    on='ts', by='ticker', direction='backward',
)
placed['theo_at_placed'] = placed['theo']
placed['edge_c'] = (placed['theo_at_placed'] - placed['price']) * placed['sgn'] * 100


# =============================================================================
# Fill outcome + realized P&L per order
# =============================================================================
api = KalshiAPI()
all_tickers = list(set(placed['ticker'].dropna().unique()) |
                   set(fills['ticker'].dropna().unique()))
settlements = fetch_settlements_from_api(
    all_tickers, api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome'] = fills['ticker'].map(settlements)
fills['fsgn'] = np.where(fills['action'] == 'buy', +1, -1)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['fsgn'] * 100

realized_per_order = (fills.dropna(subset=['client_order_id', 'realized_c'])
                            [['client_order_id', 'realized_c']]
                            .drop_duplicates('client_order_id', keep='first'))
placed = placed.merge(realized_per_order, on='client_order_id', how='left')
placed['filled'] = placed['realized_c'].notna().astype(int)


# =============================================================================
# Edge buckets + summary
# =============================================================================
edge_bins   = [-100, 0, 2, 4, 6, 8, 10, 15, 100]
edge_labels = ['<0¢', '0-2¢', '2-4¢', '4-6¢', '6-8¢', '8-10¢', '10-15¢', '>15¢']
placed['edge_bucket'] = pd.cut(
    placed['edge_c'], bins=edge_bins, labels=edge_labels)


def summary(g: pd.DataFrame) -> pd.Series:
    n_placed = len(g)
    filled = g[g['filled'] == 1].dropna(subset=['realized_c'])
    n_filled = len(filled)
    fill_rate = n_filled / n_placed if n_placed > 0 else 0
    if n_filled >= 2:
        mean_r = filled['realized_c'].mean()
        lo, hi = bootstrap_ci(filled['realized_c'].values, B=2000)
    else:
        mean_r = lo = hi = np.nan
    ev = fill_rate * (mean_r if mean_r == mean_r else 0)
    return pd.Series({
        'n_placed':    n_placed,
        'n_filled':    n_filled,
        'fill_pct':    fill_rate * 100,
        'realized_c':  mean_r,
        'realized_lo': lo,
        'realized_hi': hi,
        'EV_c':        ev,
    })


pd.set_option('display.float_format', '{:+.3f}'.format)
pd.set_option('display.width', 200)

print(f"\n=== Overall — by edge bucket ({len(placed):,} placements) ===")
print(placed.groupby('edge_bucket', observed=True)
            .apply(summary, include_groups=False))

print(f"\n=== By action × edge bucket ===")
by_action = (placed.groupby(['action', 'edge_bucket'], observed=True)
                    .apply(summary, include_groups=False))
print(by_action)


# =============================================================================
# Find optimum per side
# =============================================================================
print(f"\n=== Optimal edge by E[¢/placement] ===")
for action in ['buy', 'sell']:
    sub = placed[placed['action'] == action]
    bucket_ev = (sub.groupby('edge_bucket', observed=True)
                     .apply(summary, include_groups=False))
    # Find peak (excluding negative-edge buckets where EV is meaningless)
    valid = bucket_ev[bucket_ev.index.astype(str) != '<0¢']
    if len(valid) == 0:
        continue
    best = valid['EV_c'].idxmax()
    print(f"   {action.upper():>4s}:  best edge = {best:>8s}  "
          f"EV = {valid.loc[best, 'EV_c']:+.4f}¢/placement  "
          f"(fill_rate={valid.loc[best, 'fill_pct']:.1f}%, "
          f"realized={valid.loc[best, 'realized_c']:+.2f}¢/fill)")


# =============================================================================
# Viz
# =============================================================================
def collect_curve(action):
    sub = placed[placed['action'] == action]
    b = (sub.groupby('edge_bucket', observed=True)
            .apply(summary, include_groups=False).reset_index())
    return b


fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Fill rate by edge",
        "Mean realized P&L per fill (95% CI)",
        "E[¢/placement] = fill_rate × realized — the optimization target",
        "Per-bucket sample sizes",
    ),
    horizontal_spacing=0.10, vertical_spacing=0.15,
)

for action, color in [('buy', '#22c55e'), ('sell', '#a78bfa')]:
    d = collect_curve(action)
    labels = [str(b) for b in d['edge_bucket']]
    fig.add_trace(go.Scatter(
        x=labels, y=d['fill_pct'], name=f'{action} fill %',
        mode='lines+markers', line=dict(color=color, width=2),
        marker=dict(size=8), legendgroup=action,
    ), row=1, col=1)

    err_lo = (d['realized_c'] - d['realized_lo']).clip(lower=0)
    err_hi = (d['realized_hi'] - d['realized_c']).clip(lower=0)
    fig.add_trace(go.Scatter(
        x=labels, y=d['realized_c'], name=f'{action} realized ¢/fill',
        mode='lines+markers', line=dict(color=color, width=2, dash='dash'),
        marker=dict(size=8), legendgroup=action, showlegend=False,
        error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                      color=color, thickness=1),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=labels, y=d['EV_c'], name=f'{action} EV/placement',
        mode='lines+markers', line=dict(color=color, width=3),
        marker=dict(size=10), legendgroup=action, showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=d['n_placed'], name=f'{action} n_placed',
        marker_color=color, opacity=0.6,
        legendgroup=action, showlegend=False,
    ), row=2, col=2)

for r, c in [(1,1), (1,2), (2,1), (2,2)]:
    fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=r, col=c)

fig.update_xaxes(title_text='edge bucket', row=2, col=1)
fig.update_xaxes(title_text='edge bucket', row=2, col=2)
fig.update_yaxes(title_text='fill %', row=1, col=1)
fig.update_yaxes(title_text='cents/fill', row=1, col=2)
fig.update_yaxes(title_text='cents/placement', row=2, col=1)
fig.update_yaxes(title_text='count', row=2, col=2)
fig.update_layout(
    title=f"Optimal-edge analysis — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(placed):,} placements, {placed['filled'].sum():,} fills",
    template='plotly_dark', height=950,
)
fig.show()
