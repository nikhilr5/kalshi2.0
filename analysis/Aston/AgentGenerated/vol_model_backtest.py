"""Vol-model backtest — compares Brier scores across candidate σ models.

Rerun-friendly: change CUTOFF_DAY to widen window as data accumulates.

For each candidate σ model:
  1. Compute forecasted σ per minute (from 1-min Coinbase bars)
  2. For each theo_state row, recompute counterfactual theo = N(d2) with model σ
  3. Brier score vs Kalshi settlement
  4. Optional slices: by ttc bucket, by moneyness

Candidate models:
  • HAR-RV (current Aston model, baseline)
  • Parkinson rolling — 15m / 30m / 60m windows
  • Garman-Klass rolling 30m (uses OHLC; more efficient at same data)
  • Market-implied σ (invert from Kalshi mid)
  • HAR + Market 50/50 blend

Output: per-model Brier overall + by ttc bucket + by moneyness, plus
a forecast-accuracy panel (corr, MAE, RMSE vs realized 15m σ).
"""

import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    ANN_MIN,
    SECONDS_PER_YEAR,
    FOUR_LN2,
    brier_score,
    fetch_settlements_from_api,
    implied_sigma,
    load_all_data,
    realized_sigma_forward,
    theo_vec,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX = "KXETH15M"
CUTOFF_DAY    = "26MAY15"

# HAR-RV coefficients from CLAUDE.md
HAR_COEFFS = dict(b0=0.0314, b15=0.4485, b30=0.1293, b4h=0.1843, b24h=0.1149)


# =============================================================================
# Load
# =============================================================================
theo, book, spot, _fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo = theo[theo['seconds_to_expiry'] > 0].reset_index(drop=True)
spot = spot.sort_values('ts').reset_index(drop=True)
book = book.sort_values(['ticker', 'ts']).reset_index(drop=True)


# =============================================================================
# Build per-minute bars from spot ticks (OHLC)
# =============================================================================
spot['minute'] = spot['ts'].dt.floor('1min')
minute_bars = (spot.groupby('minute')['price']
                    .agg(open='first', high='max', low='min', close='last')
                    .reset_index()
                    .sort_values('minute')
                    .reset_index(drop=True))


# =============================================================================
# Per-minute σ estimators (annualized)
# =============================================================================
hl_ratio = minute_bars['high'] / minute_bars['low']
minute_bars['park_var_1m'] = np.where(
    (minute_bars['high'] > 0) & (minute_bars['low'] > 0)
    & (minute_bars['high'] > minute_bars['low']),
    np.log(hl_ratio) ** 2 / FOUR_LN2, 0.0,
)
# Garman-Klass per-minute variance — uses OHLC, more efficient than Parkinson
log_hl = np.log(minute_bars['high'] / minute_bars['low'].replace(0, np.nan))
log_co = np.log(minute_bars['close'] / minute_bars['open'].replace(0, np.nan))
minute_bars['gk_var_1m'] = (
    0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
).fillna(0.0).clip(lower=0)


def rolling_sigma(per_min_var: pd.Series, window: int) -> pd.Series:
    """Annualized σ from a per-minute variance series, rolling window."""
    rolled = per_min_var.rolling(window=window, min_periods=window // 2).mean()
    return np.sqrt(rolled * ANN_MIN)


minute_bars['sigma_park_15m'] = rolling_sigma(minute_bars['park_var_1m'], 15)
minute_bars['sigma_park_30m'] = rolling_sigma(minute_bars['park_var_1m'], 30)
minute_bars['sigma_park_60m'] = rolling_sigma(minute_bars['park_var_1m'], 60)
minute_bars['sigma_gk_30m']   = rolling_sigma(minute_bars['gk_var_1m'],   30)

# HAR-RV — apply coefficients to multi-horizon Parkinson RVs
def har_forecast(park_var_1m, m, h_15=15, h_30=30, h_4h=240, h_24h=1440):
    """Compute HAR forecast using Aston's actual mixture."""
    cum = park_var_1m.cumsum().shift(1).fillna(0)
    def lookback(h):
        idx = m.index
        back = idx - h
        back = back.clip(lower=0)
        return np.sqrt((cum.values[idx] - cum.values[back]) / h * ANN_MIN)
    rv_15  = lookback(h_15)
    rv_30  = lookback(h_30)
    rv_4h  = lookback(min(h_4h, len(park_var_1m)))
    rv_24h = lookback(min(h_24h, len(park_var_1m)))
    return (HAR_COEFFS['b0']
            + HAR_COEFFS['b15'] * rv_15
            + HAR_COEFFS['b30'] * rv_30
            + HAR_COEFFS['b4h'] * rv_4h
            + HAR_COEFFS['b24h'] * rv_24h)


# Recompute HAR ourselves for parity with recorded theo
minute_bars = minute_bars.reset_index(drop=True)
m_idx = minute_bars.index.to_series()
park_var = minute_bars['park_var_1m']
cum = park_var.cumsum()

def lookback_rv(h):
    cum_now = cum.values
    cum_back = np.r_[np.zeros(h), cum_now[:-h]] if h <= len(cum_now) else np.zeros_like(cum_now)
    eff = cum_now - cum_back
    return np.sqrt(np.maximum(eff, 0) / h * ANN_MIN)

minute_bars['rv_15m']  = lookback_rv(15)
minute_bars['rv_30m']  = lookback_rv(30)
minute_bars['rv_4h']   = lookback_rv(min(240, len(minute_bars)))
minute_bars['rv_24h']  = lookback_rv(min(1440, len(minute_bars)))
minute_bars['sigma_har'] = (
    HAR_COEFFS['b0']
    + HAR_COEFFS['b15'] * minute_bars['rv_15m']
    + HAR_COEFFS['b30'] * minute_bars['rv_30m']
    + HAR_COEFFS['b4h'] * minute_bars['rv_4h']
    + HAR_COEFFS['b24h'] * minute_bars['rv_24h']
).clip(lower=0.01)


# =============================================================================
# Merge σ candidates into theo_state on minute boundaries
# =============================================================================
theo['minute'] = theo['ts'].dt.floor('1min')
sigma_cols = ['sigma_har', 'sigma_park_15m', 'sigma_park_30m',
              'sigma_park_60m', 'sigma_gk_30m']
theo = theo.merge(
    minute_bars[['minute'] + sigma_cols], on='minute', how='left')


# =============================================================================
# Market-implied σ — from book mid
# =============================================================================
book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2
theo_for_book = (theo[['ts', 'ticker', 'spot', 'strike', 'seconds_to_expiry']]
                 .sort_values(['ticker', 'ts']))
book = book.sort_values(['ticker', 'ts'])
book = pd.merge_asof(
    book, theo_for_book, on='ts', by='ticker', direction='backward',
)
book['sigma_implied'] = implied_sigma(
    book['mid'], book['spot'], book['strike'], book['seconds_to_expiry'],
)
book = book.dropna(subset=['sigma_implied'])
book = book[(book['sigma_implied'] > 0.05) & (book['sigma_implied'] < 3.0)]
# Snap implied σ to minute and merge into theo_state rows for that ticker
book['minute'] = book['ts'].dt.floor('1min')
implied_minute = (book.groupby(['ticker', 'minute'])['sigma_implied']
                       .mean().reset_index())
theo = theo.merge(implied_minute, on=['ticker', 'minute'], how='left')

# Blend HAR + implied 50/50
theo['sigma_blend'] = (theo['sigma_har'] + theo['sigma_implied']) / 2


# =============================================================================
# Settlements + compute counterfactual theo per model
# =============================================================================
api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(theo['ticker'].dropna().unique()), api,
    cache_path=Path(__file__).resolve().parent.parent
                / ".settlements_cache.json",
)
theo['outcome'] = theo['ticker'].map(settlements)
theo = theo.dropna(subset=['outcome']).reset_index(drop=True)

MODELS = [
    'sigma_har',
    'sigma_park_15m',
    'sigma_park_30m',
    'sigma_park_60m',
    'sigma_gk_30m',
    'sigma_implied',
    'sigma_blend',
]
for col in MODELS:
    sub = theo.dropna(subset=[col])
    theo_pred = theo_vec(sub['spot'], sub['strike'],
                          sub[col].clip(lower=0.01), sub['seconds_to_expiry'])
    theo.loc[sub.index, f'theo_{col}'] = theo_pred


# =============================================================================
# Brier per model — overall + by ttc + by moneyness
# =============================================================================
ttc_bins   = [-1, 60, 300, 600, 1e9]
ttc_labels = ['<1m', '1-5m', '5-10m', '>10m']
theo['ttc_bucket'] = pd.cut(theo['seconds_to_expiry'], bins=ttc_bins, labels=ttc_labels)

# Moneyness via existing theo column (closest to truth)
theo['mny_bucket'] = pd.cut(
    theo['theo'].clip(0, 1),
    bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
    labels=['deep_OTM', 'OTM', 'ATM', 'ITM', 'deep_ITM'],
)


def briers(df: pd.DataFrame) -> dict:
    out = {'n': len(df)}
    for col in MODELS:
        c = f'theo_{col}'
        out[col] = brier_score(df[c], df['outcome'])
    return out


print(f"\n=== Brier overall ({len(theo):,} theo_state rows) ===")
overall = briers(theo)
print(f"   n = {overall['n']:,}")
for col in MODELS:
    val = overall[col]
    print(f"   {col:>20s}: {val:.4f}" if val is not None else f"   {col:>20s}:  --")

print(f"\n=== Brier by ttc_bucket ===")
ttc_rows = []
for b in ttc_labels:
    sub = theo[theo['ttc_bucket'] == b]
    if len(sub) < 100:
        continue
    row = {'bucket': b}
    row.update(briers(sub))
    ttc_rows.append(row)
print(pd.DataFrame(ttc_rows).to_string(index=False))

print(f"\n=== Brier by moneyness bucket ===")
mny_rows = []
for b in ['deep_OTM', 'OTM', 'ATM', 'ITM', 'deep_ITM']:
    sub = theo[theo['mny_bucket'] == b]
    if len(sub) < 100:
        continue
    row = {'bucket': b}
    row.update(briers(sub))
    mny_rows.append(row)
print(pd.DataFrame(mny_rows).to_string(index=False))


# =============================================================================
# σ forecast accuracy vs realized 15m
# =============================================================================
forward_15m = realized_sigma_forward(spot, horizon_minutes=15)
forward_15m = forward_15m[['minute', 'realized_15m']]
theo_min = (theo.groupby('minute')[MODELS]
                 .mean().reset_index())
acc = forward_15m.merge(theo_min, on='minute', how='inner').dropna()
print(f"\n=== σ forecast accuracy vs realized 15m σ ===")
for col in MODELS:
    valid = acc[[col, 'realized_15m']].dropna()
    if len(valid) < 30:
        continue
    corr = valid[col].corr(valid['realized_15m'])
    mae  = (valid[col] - valid['realized_15m']).abs().mean()
    rmse = np.sqrt(((valid[col] - valid['realized_15m']) ** 2).mean())
    print(f"   {col:>20s}:  corr={corr:+.3f}  MAE={mae*100:.2f}%  RMSE={rmse*100:.2f}%  n={len(valid):,}")


# =============================================================================
# Viz — Brier bar chart per model, faceted
# =============================================================================
all_briers = pd.DataFrame([{'bucket': 'overall', **overall}] + ttc_rows + mny_rows)
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Brier by model — overall + ttc buckets", "Brier by moneyness"),
    row_heights=[0.5, 0.5], vertical_spacing=0.18,
)
palette = ['#a78bfa', '#22c55e', '#84cc16', '#facc15', '#f97316', '#dc2626', '#0ea5e9']
for i, col in enumerate(MODELS):
    sub_top = all_briers[~all_briers['bucket'].isin(['deep_OTM','OTM','ATM','ITM','deep_ITM'])]
    fig.add_trace(go.Bar(
        x=sub_top['bucket'], y=sub_top[col], name=col,
        marker_color=palette[i % len(palette)], showlegend=True,
    ), row=1, col=1)
    sub_bot = all_briers[all_briers['bucket'].isin(['deep_OTM','OTM','ATM','ITM','deep_ITM'])]
    fig.add_trace(go.Bar(
        x=sub_bot['bucket'], y=sub_bot[col], name=col,
        marker_color=palette[i % len(palette)], showlegend=False,
    ), row=2, col=1)

fig.update_yaxes(title_text='Brier', row=1, col=1)
fig.update_yaxes(title_text='Brier', row=2, col=1)
fig.update_layout(
    title=f"Vol-model Brier backtest — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(theo):,} theo_state rows",
    template='plotly_dark', height=900, barmode='group',
)
fig.show()
