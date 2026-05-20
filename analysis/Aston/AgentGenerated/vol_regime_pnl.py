"""Realized P&L by vol regime — does the strategy work the same in
calm vs volatile markets?

For each fill, attach:
  • sigma_at_fill   = HAR forecast at fill time (forecasted vol regime)
  • sigma_realized  = Parkinson σ over the 15min AFTER the fill (truth)
  • realized_c      = (outcome - price) * sgn * 100  (per-contract cents)
  • forecast_error  = sigma_at_fill - sigma_realized   (HAR's mistake)

Bucket fills by sigma_at_fill (the regime visible at decision time)
and by sigma_realized (what actually happened).  Compute mean realized
PnL per bucket.  Plot fill-level scatter and bucket means.

Output answers:
  1. Does P&L change across vol regimes?
  2. Is the strategy regime-agnostic, regime-specific, or fragile at extremes?
  3. Does HAR's forecast error correlate with P&L?
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
    realized_sigma_forward,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"


# =============================================================================
# Load + merge sigma at fill time
# =============================================================================
theo, _book, spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo  = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
spot  = spot.sort_values('ts').reset_index(drop=True)
fills = fills.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

fills = pd.merge_asof(
    fills,
    theo[['ts', 'ticker', 'sigma', 'theo', 'seconds_to_expiry']]
        .rename(columns={'sigma': 'sigma_at_fill'})
        .sort_values('ts'),
    on='ts', by='ticker', direction='backward',
)
fills = fills.dropna(subset=['sigma_at_fill']).reset_index(drop=True)


# =============================================================================
# Realized sigma over the 15 min after fill (Parkinson)
# =============================================================================
realized = realized_sigma_forward(spot, horizon_minutes=15)
fills['minute'] = fills['ts'].dt.floor('1min')
fills = fills.merge(
    realized[['minute', 'realized_15m']].rename(columns={'realized_15m': 'sigma_realized'}),
    on='minute', how='left',
)
fills['sigma_forecast_error'] = fills['sigma_at_fill'] - fills['sigma_realized']


# =============================================================================
# Settlements + realized P&L
# =============================================================================
api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['sgn']        = np.where(fills['action'] == 'buy', +1, -1)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * fills['count'] * 100
fills = fills.dropna(subset=['realized_c']).reset_index(drop=True)


# =============================================================================
# Bucket by vol regime
# =============================================================================
vol_bins   = [0, 0.5, 1.0, 1.5, 2.0, 99]
vol_labels = ['< 0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '> 2.0']
fills['vol_bucket_forecast'] = pd.cut(fills['sigma_at_fill'], bins=vol_bins, labels=vol_labels)
fills['vol_bucket_realized'] = pd.cut(fills['sigma_realized'], bins=vol_bins, labels=vol_labels)


# =============================================================================
# Summary
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'mean_sigma': g['sigma_at_fill'].mean()}
    v = g['realized_c'].dropna().values
    if len(v) >= 2:
        lo, hi = bootstrap_ci(v, B=2000)
        out['realized_mean'] = float(v.mean())
        out['realized_lo'] = lo
        out['realized_hi'] = hi
        out['total_$'] = float(v.sum() / 100)
    else:
        out['realized_mean'] = out['realized_lo'] = out['realized_hi'] = np.nan
        out['total_$'] = np.nan
    return pd.Series(out)


pd.set_option('display.float_format', '{:+.3f}'.format)
pd.set_option('display.width', 200)

print(f"\n=== Realized P&L by FORECASTED vol bucket (HAR σ at fill) ===")
print(f"(σ in annualized units — 1.0 = 100% annualized vol)")
print(fills.groupby('vol_bucket_forecast', observed=True)
            .apply(summary, include_groups=False))

print(f"\n=== Realized P&L by REALIZED vol bucket (Parkinson σ next 15min) ===")
print(fills.groupby('vol_bucket_realized', observed=True)
            .apply(summary, include_groups=False))

# Cross-tab: forecast bucket × realized bucket
print(f"\n=== Cross-tab — fill counts by (forecast σ × realized σ) ===")
ct = pd.crosstab(fills['vol_bucket_forecast'], fills['vol_bucket_realized'],
                  margins=True, margins_name='all')
print(ct.to_string())

print(f"\n=== HAR forecast error by vol bucket (sigma_at_fill − sigma_realized) ===")
err_summary = (fills.groupby('vol_bucket_forecast', observed=True)
                     ['sigma_forecast_error'].agg(['count', 'mean', 'std']))
print(err_summary.round(3))


# =============================================================================
# Viz — bucket bars + fill-level scatter
# =============================================================================
fbar = (fills.groupby('vol_bucket_forecast', observed=True)
              .apply(summary, include_groups=False).reset_index())
rbar = (fills.groupby('vol_bucket_realized', observed=True)
              .apply(summary, include_groups=False).reset_index())

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Realized P&L by FORECASTED vol (HAR at fill)",
        "Realized P&L by REALIZED vol (next 15min)",
        "Fill-level: realized P&L vs σ at fill",
        "Fill-level: realized P&L vs HAR forecast error",
    ),
    horizontal_spacing=0.10, vertical_spacing=0.18,
)

# Top-left: forecast bucket bars
err_lo = (fbar['realized_mean'] - fbar['realized_lo']).clip(lower=0)
err_hi = (fbar['realized_hi'] - fbar['realized_mean']).clip(lower=0)
labels = [f"{b}<br>(n={int(n)})" for b, n in zip(fbar['vol_bucket_forecast'], fbar['n'])]
fig.add_trace(go.Bar(
    x=labels, y=fbar['realized_mean'], marker_color='#22c55e',
    error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                  color='#333', thickness=1.5),
    showlegend=False,
), row=1, col=1)

# Top-right: realized bucket bars
err_lo = (rbar['realized_mean'] - rbar['realized_lo']).clip(lower=0)
err_hi = (rbar['realized_hi'] - rbar['realized_mean']).clip(lower=0)
labels = [f"{b}<br>(n={int(n)})" for b, n in zip(rbar['vol_bucket_realized'], rbar['n'])]
fig.add_trace(go.Bar(
    x=labels, y=rbar['realized_mean'], marker_color='#a78bfa',
    error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                  color='#333', thickness=1.5),
    showlegend=False,
), row=1, col=2)

# Bottom-left: scatter realized P&L vs σ at fill
sub = fills.dropna(subset=['sigma_at_fill', 'realized_c'])
fig.add_trace(go.Scatter(
    x=sub['sigma_at_fill'], y=sub['realized_c'],
    mode='markers', marker=dict(size=3, color='#22c55e', opacity=0.4),
    showlegend=False,
), row=2, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)

# Bottom-right: scatter realized P&L vs forecast error
sub2 = fills.dropna(subset=['sigma_forecast_error', 'realized_c'])
fig.add_trace(go.Scatter(
    x=sub2['sigma_forecast_error'], y=sub2['realized_c'],
    mode='markers', marker=dict(size=3, color='#facc15', opacity=0.4),
    showlegend=False,
), row=2, col=2)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=2)
fig.add_vline(x=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=2)

fig.update_yaxes(title_text='cents/fill', row=1, col=1)
fig.update_yaxes(title_text='cents/fill', row=1, col=2)
fig.update_xaxes(title_text='σ at fill (HAR forecast)', row=2, col=1)
fig.update_yaxes(title_text='realized P&L (cents)', row=2, col=1)
fig.update_xaxes(title_text='σ forecast error (HAR − realized)', row=2, col=2)
fig.update_yaxes(title_text='realized P&L (cents)', row=2, col=2)

for r, c in [(1,1), (1,2)]:
    fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=r, col=c)

fig.update_layout(
    title=f"Realized P&L by vol regime — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills",
    template='plotly_dark', height=950,
)
fig.show()
