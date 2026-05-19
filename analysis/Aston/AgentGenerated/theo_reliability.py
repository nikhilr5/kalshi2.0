"""Theo reliability curve — is theo biased on average, or only when
conditioned on a fill?

Bin theo into 20 buckets (width 0.05).  Compute mean settlement outcome
per bin.  Compare:
  • Unconditional — every theo_state row (the model itself)
  • Fill-conditioned — theo at the moment of fills (the model on the
    rows where we actually trade)

A well-calibrated theo sits on y=x.  Below y=x means theo overstates
yes; above means it understates.  Divergence between the two curves
indicates selection bias (we fill at moments where theo is more
biased than average).  Faceted by time-to-close.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    fetch_settlements_from_api,
    load_all_data,
)


def proportion_ci(outcomes: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion.  Closed-form, instant."""
    n = len(outcomes)
    if n < 2:
        return (float('nan'), float('nan'))
    p = float(np.mean(outcomes))
    z = 1.959963984540054  # 1 - alpha/2 quantile for alpha=0.05
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"
THEO_BINS = np.linspace(0, 1, 21)  # 20 equal-width bins


# =============================================================================
# Load + attach settlement outcomes
# =============================================================================
theo, _book, _spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo = theo[theo['seconds_to_expiry'] > 0].reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

api = KalshiAPI()
all_tickers = list(set(theo['ticker'].unique()) | set(fills['ticker'].unique()))
settlements = fetch_settlements_from_api(
    all_tickers, api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
theo['outcome']  = theo['ticker'].map(settlements)
fills['outcome'] = fills['ticker'].map(settlements)
theo  = theo[theo['outcome'].notna()].reset_index(drop=True)
fills = fills[fills['outcome'].notna()].reset_index(drop=True)

fills = pd.merge_asof(
    fills.sort_values('ts'),
    theo[['ts', 'ticker', 'theo', 'seconds_to_expiry']].sort_values('ts'),
    on='ts', by='ticker', direction='backward',
    suffixes=('_orig', ''),
)
fills = fills.dropna(subset=['theo', 'seconds_to_expiry']).reset_index(drop=True)

ttc_bins   = [-1, 60, 300, 600, 1e9]
ttc_labels = ['<1m', '1-5m', '5-10m', '>10m']
theo['ttc_bucket']  = pd.cut(theo['seconds_to_expiry'],  bins=ttc_bins, labels=ttc_labels)
fills['ttc_bucket'] = pd.cut(fills['seconds_to_expiry'], bins=ttc_bins, labels=ttc_labels)


# =============================================================================
# Reliability curve helper
# =============================================================================
def reliability_curve(theo_values: pd.Series, outcomes: pd.Series) -> pd.DataFrame:
    """Bin theo, return per-bin mean outcome + 95% bootstrap CI."""
    df = pd.DataFrame({'theo': theo_values.values,
                       'outcome': outcomes.values}).dropna()
    df['bin'] = pd.cut(df['theo'], bins=THEO_BINS, include_lowest=True)
    rows = []
    for b, g in df.groupby('bin', observed=True):
        v = g['outcome'].values
        if len(v) < 2:
            continue
        lo, hi = proportion_ci(v)
        rows.append({
            'bin_mid':      float(b.mid),
            'n':            len(v),
            'mean_outcome': float(v.mean()),
            'lo':           lo, 'hi': hi,
            'mean_theo':    float(g['theo'].mean()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Print summary
# =============================================================================
pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.width', 120)

print(f"\n=== Overall — unconditional ({len(theo):,} theo_state rows) ===")
print(reliability_curve(theo['theo'], theo['outcome']).to_string(index=False))

print(f"\n=== Overall — fill-conditioned ({len(fills):,} fills) ===")
print(reliability_curve(fills['theo'], fills['outcome']).to_string(index=False))


# =============================================================================
# Viz — 2x2 panel by ttc, with unconditional + fill-conditioned curves
# =============================================================================
by_ttc = {}
for bucket in ttc_labels:
    t = theo[theo['ttc_bucket']  == bucket]
    f = fills[fills['ttc_bucket'] == bucket]
    by_ttc[bucket] = dict(
        unc = reliability_curve(t['theo'], t['outcome']),
        fc  = reliability_curve(f['theo'], f['outcome']),
        n_unc = len(t), n_fc = len(f),
    )

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        f">10m to close (n_theo={by_ttc['>10m']['n_unc']:,}, n_fills={by_ttc['>10m']['n_fc']:,})",
        f"5-10m to close (n_theo={by_ttc['5-10m']['n_unc']:,}, n_fills={by_ttc['5-10m']['n_fc']:,})",
        f"1-5m to close (n_theo={by_ttc['1-5m']['n_unc']:,}, n_fills={by_ttc['1-5m']['n_fc']:,})",
        f"<1m to close (n_theo={by_ttc['<1m']['n_unc']:,}, n_fills={by_ttc['<1m']['n_fc']:,})",
    ],
    horizontal_spacing=0.10, vertical_spacing=0.12,
)

positions = {'>10m': (1, 1), '5-10m': (1, 2), '1-5m': (2, 1), '<1m': (2, 2)}
for bucket, (r, c) in positions.items():
    unc = by_ttc[bucket]['unc']
    fc  = by_ttc[bucket]['fc']
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#666', dash='dash', width=1),
        showlegend=False, hoverinfo='skip',
    ), row=r, col=c)
    if len(unc) > 0:
        unc_lo = (unc['mean_outcome'] - unc['lo']).clip(lower=0)
        unc_hi = (unc['hi'] - unc['mean_outcome']).clip(lower=0)
        fig.add_trace(go.Scatter(
            x=unc['mean_theo'], y=unc['mean_outcome'],
            mode='lines+markers', name='unconditional (theo_state)',
            line=dict(color='#facc15', width=2), marker=dict(size=6),
            error_y=dict(type='data', array=unc_hi, arrayminus=unc_lo,
                          color='#facc15', thickness=1),
            legendgroup='unc', showlegend=(r == 1 and c == 1),
        ), row=r, col=c)
    if len(fc) > 0:
        fc_lo = (fc['mean_outcome'] - fc['lo']).clip(lower=0)
        fc_hi = (fc['hi'] - fc['mean_outcome']).clip(lower=0)
        fig.add_trace(go.Scatter(
            x=fc['mean_theo'], y=fc['mean_outcome'],
            mode='lines+markers', name='fill-conditioned (theo at fill)',
            line=dict(color='#a78bfa', width=2),
            marker=dict(size=8, symbol='diamond'),
            error_y=dict(type='data', array=fc_hi, arrayminus=fc_lo,
                          color='#a78bfa', thickness=1),
            legendgroup='fc', showlegend=(r == 1 and c == 1),
        ), row=r, col=c)
    fig.update_xaxes(title_text='theo (predicted yes prob)', range=[0, 1], row=r, col=c)
    fig.update_yaxes(title_text='mean realized outcome',       range=[0, 1], row=r, col=c)

fig.update_layout(
    title=(f"Theo reliability — {SERIES_PREFIX} ≥ {CUTOFF_DAY}<br>"
           f"<sub>y=x = perfect calibration · "
           f"point BELOW line = theo overstates yes · "
           f"point ABOVE = theo understates yes</sub>"),
    template='plotly_dark', height=950,
)
fig.show()
