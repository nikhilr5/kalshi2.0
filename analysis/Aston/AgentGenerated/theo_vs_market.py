"""Theo vs market disagreement at fill time vs realized PnL.

Hypothesis: large gaps between our theo and market mid (which trigger
the fill via our edge condition) systematically lose money — meaning
the market was closer to truth than our theo.  This is the statistical
adverse-selection mechanism baked into the quoting rule itself.

For each fill compute:
    disagree_c = (theo - mid) * sgn * 100   in cents
    realized_c = (outcome - price) * sgn * 100   in cents

Bin by disagree_c.  If theo is right → realized should rise with
disagreement.  If market is right → realized should plateau or fall
in the big-disagreement bins.
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
# Load + merge theo at fill time
# =============================================================================
theo, _book, _spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo  = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
fills = fills.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

fills = pd.merge_asof(
    fills,
    theo[['ts', 'ticker', 'theo', 'seconds_to_expiry']].sort_values('ts'),
    on='ts', by='ticker', direction='backward',
)

# Mid from the kalshi book snapshot recorded with each fill
fills['mid'] = (fills['kalshi_yes_bid'] + fills['kalshi_yes_ask']) / 2
fills = fills[(fills['kalshi_yes_bid'] > 0)
              & (fills['kalshi_yes_ask'] > fills['kalshi_yes_bid'])
              ].reset_index(drop=True)

api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome'] = fills['ticker'].map(settlements)


# =============================================================================
# Disagreement + realized
# =============================================================================
fills['sgn']            = np.where(fills['action'] == 'buy', +1, -1)
fills['disagree_c']     = (fills['theo']    - fills['mid'])   * fills['sgn'] * 100
fills['spread_cap_c']   = (fills['theo']    - fills['price']) * fills['sgn'] * 100
fills['realized_c']     = (fills['outcome'] - fills['price']) * fills['sgn'] * 100

fills['ttc_bucket'] = pd.cut(
    fills['seconds_to_expiry'],
    bins=[-1, 60, 300, 600, 1e9], labels=['<1m', '1-5m', '5-10m', '>10m'],
)
fills['disagree_bucket'] = pd.cut(
    fills['disagree_c'],
    bins=[-100, 0, 2, 5, 10, 20, 100],
    labels=['<0', '0-2', '2-5', '5-10', '10-20', '>20'],
)


# =============================================================================
# Summary tables
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'disagree_mean': g['disagree_c'].mean(),
           'spread_cap_mean': g['spread_cap_c'].mean()}
    v = g['realized_c'].dropna().values
    if len(v) >= 2:
        lo, hi = bootstrap_ci(v)
        out['realized_mean'] = float(v.mean())
        out['realized_lo'] = lo
        out['realized_hi'] = hi
    else:
        out['realized_mean'] = out['realized_lo'] = out['realized_hi'] = np.nan
    return pd.Series(out)


pd.set_option('display.float_format', '{:+.2f}'.format)
pd.set_option('display.width', 200)

print("\n=== Realized PnL by theo-vs-market disagreement bucket ===")
print("If theo is right: realized rises with disagreement.")
print("If market is right: realized plateaus or falls in big-disagreement bins.\n")
print(fills.groupby('disagree_bucket', observed=True)
            .apply(summary, include_groups=False))

print("\n=== By action × disagreement bucket ===")
print(fills.groupby(['action', 'disagree_bucket'], observed=True)
            .apply(summary, include_groups=False))

print("\n=== By disagreement × ttc — does the 5-10m ATM bleed sit in big-disagreement bins? ===")
print(fills.groupby(['ttc_bucket', 'disagree_bucket'], observed=True)
            .apply(summary, include_groups=False))

df = fills[['disagree_c', 'realized_c', 'spread_cap_c']].dropna()
print(f"\n=== ρ(disagree, realized)     = {df['disagree_c'].corr(df['realized_c']):+.3f}")
print(f"=== ρ(spread_cap, realized)   = {df['spread_cap_c'].corr(df['realized_c']):+.3f}")


# =============================================================================
# Viz — binned-mean curve + scatter
# =============================================================================
bucket_df = (fills.groupby('disagree_bucket', observed=True)
                   .apply(summary, include_groups=False).reset_index())

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Mean realized P&L per fill by theo-vs-market disagreement (95% CI). "
        "Compare to spread_capture (the paper edge we expected to earn).",
        "Per-fill scatter — does big disagreement predict the bleed?",
    ),
    row_heights=[0.5, 0.5], vertical_spacing=0.15,
)

# Top: realized + spread_capture bars side by side
err_lo = (bucket_df['realized_mean'] - bucket_df['realized_lo']).clip(lower=0)
err_hi = (bucket_df['realized_hi'] - bucket_df['realized_mean']).clip(lower=0)
labels = [f"{b}¢\n(n={int(n)})" for b, n in zip(bucket_df['disagree_bucket'],
                                                  bucket_df['n'])]
fig.add_trace(go.Bar(
    x=labels, y=bucket_df['realized_mean'],
    name='realized P&L', marker_color='#a78bfa',
    error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                  color='#333', thickness=1.5),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=labels, y=bucket_df['spread_cap_mean'],
    mode='lines+markers', name='spread capture (paper edge)',
    line=dict(color='#facc15', width=2),
    marker=dict(size=8),
), row=1, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=1, col=1)

# Bottom: per-fill scatter
for action, color in [('buy', '#22c55e'), ('sell', '#a78bfa')]:
    sub = fills[fills['action'] == action].dropna(subset=['disagree_c', 'realized_c'])
    fig.add_trace(go.Scatter(
        x=sub['disagree_c'], y=sub['realized_c'],
        mode='markers', name=f'{action} (n={len(sub)})',
        marker=dict(size=3, color=color, opacity=0.35),
    ), row=2, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)
fig.add_vline(x=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)

fig.update_yaxes(title_text='cents per fill', row=1, col=1)
fig.update_xaxes(title_text='disagreement bucket (theo − mid, cents, our-favor signed)', row=1, col=1)
fig.update_xaxes(title_text='disagreement (cents)', row=2, col=1)
fig.update_yaxes(title_text='realized P&L (cents)', row=2, col=1)
fig.update_layout(
    title=f"Theo vs market disagreement — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills",
    template='plotly_dark', height=950,
)
fig.show()
