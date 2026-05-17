"""Vol-forecast accuracy: HAR vs market-implied σ vs realized 15-min σ.

Loads recorded data from local + S3 (filtered by cutoff day), computes
the three vol series on a 1-min grid, and renders two plotly charts:

    1. σ-over-time overlay (HAR / implied / realized)
    2. Calibration scatter — each prediction vs realized, with summary
       stats (corr / bias / MAE / RMSE) annotated.

All heavy lifting is in `utility` — this file is just the analysis
recipe.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utility import (
    brier_score,
    fetch_settlements_from_api,
    forecast_error_stats,
    implied_sigma,
    load_all_data,
    realized_sigma_forward,
    theo_vec,
    theo_vec_twap
)

# Aston package is a sibling — let Python find the KalshiAPI module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI


# =============================================================================
# Config
# =============================================================================
SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"   # inclusive lower bound, YYMONDD


# =============================================================================
# Load + clean
# =============================================================================
theo, book, spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo = theo[theo['seconds_to_expiry'] > 0]


# =============================================================================
# HAR forecast σ — fitted coefficients applied to recorded per-horizon RVs
# =============================================================================
theo['forecasted_vol'] = (
    0.0314
    + 0.4485 * theo['rv_15m']
    + 0.1293 * theo['rv_30m']
    + 0.1843 * theo['rv_4h']
    + 0.1149 * theo['rv_24h']
)


# =============================================================================
# Market-implied σ — invert N(d2) using spot/strike/T from theo_state
# =============================================================================
book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2

theo_for_merge = (
    theo[['ts', 'ticker', 'spot', 'strike', 'seconds_to_expiry']]
        .sort_values(['ticker', 'ts'])
)
book = book.sort_values(['ticker', 'ts'])
book = pd.merge_asof(
    book, theo_for_merge,
    on='ts', by='ticker', direction='backward',
)
book['implied_vol'] = implied_sigma(
    book['mid'], book['spot'], book['strike'], book['seconds_to_expiry'],
)
book = book[book['implied_vol'].between(0.05, 3.0)]


# =============================================================================
# Settlement outcomes — authoritative `result` from Kalshi API.
# Cached to .settlements_cache.json so re-runs don't re-hit the API for
# tickers already known to be settled.
# =============================================================================
_api = KalshiAPI()
unique_tickers = list(theo['ticker'].dropna().unique())
settlements = fetch_settlements_from_api(
    unique_tickers,
    _api,
    cache_path=Path(__file__).resolve().parent / ".settlements_cache.json",
)
n_settled = len(settlements)

theo['outcome'] = theo['ticker'].map(settlements)
book['outcome'] = book['ticker'].map(settlements)

theo['calculated_theo'] = theo_vec(theo['spot'], theo['strike'], theo['forecasted_vol'], theo['seconds_to_expiry'])
har_brier = brier_score(theo['calculated_theo'], theo['outcome'])
mkt_brier = brier_score(book['mid'], book['outcome'])

theo['calculated_theo_twap'] = theo_vec_twap(
    theo['spot'], theo['strike'],
    theo['forecasted_vol'], theo['seconds_to_expiry']
)

har_brier_twap = brier_score(theo['calculated_theo_twap'], theo['outcome'])
har_brier_until_90s = brier_score(theo[theo['seconds_to_expiry'] > 90]['calculated_theo'], theo['outcome'])
mkt_brier_until_90s = brier_score(book[book['seconds_to_expiry'] > 90]['mid'], book['outcome'])



# =============================================================================
# Realized σ over the next 15 minutes — Parkinson per minute
# =============================================================================
minute_bars = realized_sigma_forward(spot, horizon_minutes=15)


# =============================================================================
# Resample forecast + implied to a 1-min grid, merge with realized
# =============================================================================
# Use resample(on='ts') instead of set_index('ts').  set_index copies
# the whole frame; with multi-day loads + ~10 columns the copy itself
# can dominate runtime.  resample(on=...) just builds the bucketing
# index over the slim 2-col view.
theo_minute = (
    theo[['ts', 'forecasted_vol']]
        .resample('1min', on='ts')['forecasted_vol']
        .mean()
        .reset_index()
        .rename(columns={'ts': 'minute'})
)
book_minute = (
    book[['ts', 'implied_vol']]
        .resample('1min', on='ts')['implied_vol']
        .mean()
        .reset_index()
        .rename(columns={'ts': 'minute'})
)
combined = (
    minute_bars[['minute', 'realized_15m']]
        .merge(theo_minute, on='minute', how='outer')
        .merge(book_minute, on='minute', how='outer')
        .sort_values('minute')
        .reset_index(drop=True)
)


# =============================================================================
# Combined dashboard — time series on top, two calibration scatters below.
# =============================================================================
scatter_df = combined.dropna(subset=['realized_15m'])
df_f = scatter_df.dropna(subset=['forecasted_vol'])
df_i = scatter_df.dropna(subset=['implied_vol'])

fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"colspan": 2}, None],          # row 1: time series spans both cols
        [{},             {}            ],  # row 2: two scatters
    ],
    row_heights=[0.55, 0.45],
    subplot_titles=(
        "σ over time — Forecast vs Implied vs Realized",
        "HAR forecast vs realized",
        "Market implied vs realized",
    ),
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# --- Row 1: time-series overlay ---
fig.add_trace(go.Scatter(
    x=combined['minute'], y=combined['forecasted_vol'] * 100,
    name='HAR forecast', line=dict(color='#a78bfa', width=1.5),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=combined['minute'], y=combined['implied_vol'] * 100,
    name='Market implied', line=dict(color='#facc15', width=1.5),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=combined['minute'], y=combined['realized_15m'] * 100,
    name='Realized (next 15m)', line=dict(color='#22c55e', width=1.5),
), row=1, col=1)

# --- Row 2: calibration scatters ---
fig.add_trace(go.Scatter(
    x=df_f['forecasted_vol'] * 100, y=df_f['realized_15m'] * 100,
    mode='markers',
    marker=dict(size=3, color='#a78bfa', opacity=0.4),
    showlegend=False,
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=df_i['implied_vol'] * 100, y=df_i['realized_15m'] * 100,
    mode='markers',
    marker=dict(size=3, color='#facc15', opacity=0.4),
    showlegend=False,
), row=2, col=2)

# y=x diagonals on each scatter, scaled to data extent.
max_val = float(max(
    df_f[['forecasted_vol', 'realized_15m']].max().max() if not df_f.empty else 0,
    df_i[['implied_vol',    'realized_15m']].max().max() if not df_i.empty else 0,
) * 100)
for col in (1, 2):
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', line=dict(color='#5a6270', dash='dash', width=1),
        showlegend=False, hoverinfo='skip',
    ), row=2, col=col)

# Stats annotations on each scatter panel.  With the spanning row-1
# subplot, plotly assigns the time series to axes 1 and the two row-2
# scatters to axes 2 and 3 respectively.
panel_briers = (har_brier, mkt_brier)
panel_briers_90s = (har_brier_until_90s, mkt_brier_until_90s)
scatter_axes = (('x2', 'y2'), ('x3', 'y3'))
for i, (pred, actual) in enumerate([
        (df_f['forecasted_vol'], df_f['realized_15m']),
        (df_i['implied_vol'],    df_i['realized_15m']),
]):
    sigma_block = forecast_error_stats(pred, actual, sep='<br>')
    brier_val = panel_briers[i]
    brier_val_90s = panel_briers_90s[i]
    if brier_val is not None and n_settled > 0:
        brier_block = (f"<br><br>Brier ={brier_val:.4f}"
                       f"<br>Brier (Up until 90s until expiry)={brier_val_90s:.4f}"
                       f"<br>Brier TWAP Theo={har_brier_twap:.4f}"
                       f"<br>markets={n_settled}")
    else:
        brier_block = "<br><br>Brier =--"
    xref, yref = scatter_axes[i]
    fig.add_annotation(
        text=sigma_block + brier_block,
        xref=xref, yref=yref,
        x=max_val * 0.04, y=max_val * 0.96,
        xanchor='left', yanchor='top',
        showarrow=False, align='left',
        font=dict(family='monospace', color='#c8cdd5', size=11),
        bgcolor='rgba(20,25,35,0.85)',
        bordercolor='#1e2736', borderwidth=1,
    )

fig.update_xaxes(title_text='Time (UTC)',          row=1, col=1)
fig.update_yaxes(title_text='σ (% annualized)',     row=1, col=1)
fig.update_xaxes(title_text='HAR forecast σ (%)',   row=2, col=1)
fig.update_yaxes(title_text='Realized σ next 15m (%)', row=2, col=1)
fig.update_xaxes(title_text='Market implied σ (%)', row=2, col=2)
fig.update_yaxes(title_text='Realized σ next 15m (%)', row=2, col=2)

fig.update_layout(
    title=f"vol-forecast dashboard — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{n_settled} settled markets",
    template='plotly_dark',
    height=1100,
    hovermode='closest',
)
fig.show()
