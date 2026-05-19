"""Asymmetric-tolerance backtest — quantify the recoverable PnL from
shipping `tolerance_worsening = 0.1¢` post-2026-06-05.

For each historical fill, classify by adverse drift since placement:
  • favor or flat (≤0.1¢ adverse) — keep, fill would still happen
  • sub-tolerance against (0.1¢ < drift ≤ 1¢) — under tight tolerance,
    Aston would have cancelled before fill — PREVENT in simulation
  • super-tolerance against (drift > 1¢) — Aston tried to cancel under
    current 1¢ tolerance too, but the cancel race was lost.  Two
    interpretations bracket the truth:
        Aggressive: all prevented (cancel race wins at smaller drift)
        Conservative: all still happen (race lost at any drift)

Bootstrap the per-day P&L delta for each scenario.  Output: defensible
range of recoverable dollars for the 2026-06-05 decision memo.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX        = "KXETH15M"
CUTOFF_DAY           = "26MAY15"
CURRENT_TOLERANCE_C  = 1.0   # current Aston setting
PROPOSED_TOLERANCE_C = 0.1   # post-2026-06-05 target
N_DAYS               = 4
B_BOOTSTRAP          = 2000


# =============================================================================
# Load + compute max adverse drift per fill (mirror cancel_race.py logic)
# =============================================================================
theo, _book, _spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo   = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
fills  = fills.sort_values('ts').reset_index(drop=True)
events = events.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

placed = (events[events['event_type'] == 'placed']
          [['ts', 'client_order_id']]
          .rename(columns={'ts': 'placed_ts'})
          .sort_values('placed_ts')
          .drop_duplicates('client_order_id', keep='first'))
fills = fills.merge(placed, on='client_order_id', how='left')
fills = fills.dropna(subset=['placed_ts']).reset_index(drop=True)
fills['sgn'] = np.where(fills['action'] == 'buy', +1, -1)

theo_lookup = theo[['ts', 'ticker', 'theo']].sort_values('ts')
fills['_row'] = np.arange(len(fills))
keys_p = (fills[['_row', 'placed_ts', 'ticker']]
          .rename(columns={'placed_ts': 'ts'}).sort_values('ts'))
m_p = (pd.merge_asof(keys_p, theo_lookup, on='ts', by='ticker',
                      direction='backward').sort_values('_row'))
fills['theo_at_placed'] = m_p['theo'].values

theo_by_ticker = {}
for tk, g in theo.groupby('ticker'):
    ts_ns = g['ts'].values.astype('datetime64[ns]').astype('int64')
    theo_vals = g['theo'].values.astype(float)
    theo_by_ticker[tk] = (ts_ns, theo_vals)

max_drift_c = []
for row in fills.itertuples():
    tk = row.ticker
    if pd.isna(row.theo_at_placed) or tk not in theo_by_ticker:
        max_drift_c.append(0.0)
        continue
    p_ts = row.placed_ts.value
    f_ts = row.ts.value
    ts_ns, theo_vals = theo_by_ticker[tk]
    mask = (ts_ns > p_ts) & (ts_ns <= f_ts)
    if not mask.any():
        max_drift_c.append(0.0)
        continue
    adverse_c = (row.theo_at_placed - theo_vals[mask]) * row.sgn * 100
    max_drift_c.append(float(adverse_c.max()))
fills['max_adverse_drift_c'] = max_drift_c


# =============================================================================
# Settlements + realized PnL
# =============================================================================
api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100
fills = fills.dropna(subset=['realized_c']).reset_index(drop=True)


# =============================================================================
# Counterfactual classification
# =============================================================================
def classify(drift_c: float) -> str:
    if drift_c <= PROPOSED_TOLERANCE_C:
        return 'keep'
    if drift_c <= CURRENT_TOLERANCE_C:
        return 'sub_tol_against'
    return 'super_tol_against'


fills['cf_class'] = fills['max_adverse_drift_c'].map(classify)

n_keep = (fills['cf_class'] == 'keep').sum()
n_sub  = (fills['cf_class'] == 'sub_tol_against').sum()
n_sup  = (fills['cf_class'] == 'super_tol_against').sum()
print(f"\n=== Counterfactual classification ===")
print(f"   keep (drift ≤ {PROPOSED_TOLERANCE_C}¢):              {n_keep:>5,}")
print(f"   sub-tol against ({PROPOSED_TOLERANCE_C}-{CURRENT_TOLERANCE_C}¢):  {n_sub:>5,}  ← prevented in ALL scenarios")
print(f"   super-tol against (>{CURRENT_TOLERANCE_C}¢):         {n_sup:>5,}  ← prevented in aggressive scenario only")


# =============================================================================
# Bootstrap the per-day P&L delta
# =============================================================================
realized = fills['realized_c'].values
is_sub   = (fills['cf_class'] == 'sub_tol_against').values
is_sup   = (fills['cf_class'] == 'super_tol_against').values

rng = np.random.default_rng(seed=0)
n = len(fills)
deltas_aggressive = np.zeros(B_BOOTSTRAP)
deltas_conservative = np.zeros(B_BOOTSTRAP)
baselines = np.zeros(B_BOOTSTRAP)

for b in range(B_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)
    r_s   = realized[idx]
    sub_s = is_sub[idx]
    sup_s = is_sup[idx]
    baseline = r_s.sum()
    # Aggressive: zero out both sub_tol and super_tol against fills
    aggressive = r_s.copy()
    aggressive[sub_s | sup_s] = 0
    # Conservative: only zero out sub_tol against fills (super-tol still fill via race)
    conservative = r_s.copy()
    conservative[sub_s] = 0
    baselines[b] = baseline / 100 / N_DAYS  # cents → dollars → per day
    deltas_aggressive[b]   = (aggressive.sum()   - baseline) / 100 / N_DAYS
    deltas_conservative[b] = (conservative.sum() - baseline) / 100 / N_DAYS


def fmt_ci(arr):
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return f"${arr.mean():+.2f}/day  [{lo:+.2f}, {hi:+.2f}]"


print(f"\n=== Bootstrap results ({B_BOOTSTRAP} resamples, 95% CI) ===")
print(f"   Baseline (current behavior):           {fmt_ci(baselines)}")
print(f"   Δ Conservative (only sub-tol cancel):  {fmt_ci(deltas_conservative)}")
print(f"   Δ Aggressive (all against prevented):  {fmt_ci(deltas_aggressive)}")
print(f"   Counterfactual aggressive total:       {fmt_ci(baselines + deltas_aggressive)}")
print(f"   Counterfactual conservative total:     {fmt_ci(baselines + deltas_conservative)}")


# =============================================================================
# Per-bucket attribution — where does the recovery come from?
# =============================================================================
print(f"\n=== Recovery attribution (point estimates, ¢/fill avg) ===")
for cls in ['sub_tol_against', 'super_tol_against']:
    sub = fills[fills['cf_class'] == cls]
    if len(sub) == 0:
        continue
    total_c = sub['realized_c'].sum()
    total_d = total_c / 100
    print(f"   {cls:>20s}:  n={len(sub):>5,}  "
          f"mean={sub['realized_c'].mean():+.2f}¢/fill  "
          f"total={total_d:+.2f}$ = {total_d/N_DAYS:+.2f}$/day")


# =============================================================================
# Viz
# =============================================================================
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Per-day P&L distribution — baseline vs counterfactuals",
        "Recovery $ by source bucket",
    ),
    column_widths=[0.55, 0.45], horizontal_spacing=0.12,
)

for arr, name, color in [
    (baselines,                                  'Baseline',          '#a78bfa'),
    (baselines + deltas_conservative,            'Conservative fix',  '#facc15'),
    (baselines + deltas_aggressive,              'Aggressive fix',    '#22c55e'),
]:
    fig.add_trace(go.Histogram(
        x=arr, name=name, marker_color=color, opacity=0.6, nbinsx=40,
    ), row=1, col=1)

sub_total = fills[fills['cf_class'] == 'sub_tol_against']['realized_c'].sum() / 100
sup_total = fills[fills['cf_class'] == 'super_tol_against']['realized_c'].sum() / 100
fig.add_trace(go.Bar(
    x=['sub-tol against<br>(both scenarios)', 'super-tol against<br>(aggressive only)'],
    y=[-sub_total / N_DAYS, -sup_total / N_DAYS],
    marker_color=['#facc15', '#22c55e'],
    text=[f"+${-sub_total/N_DAYS:.2f}/day", f"+${-sup_total/N_DAYS:.2f}/day"],
    textposition='outside',
    showlegend=False,
), row=1, col=2)

fig.update_xaxes(title_text='$/day P&L', row=1, col=1)
fig.update_xaxes(title_text='bucket', row=1, col=2)
fig.update_yaxes(title_text='bootstrap density', row=1, col=1)
fig.update_yaxes(title_text='$/day recovered', row=1, col=2)
fig.update_layout(
    title=f"Asymmetric-tolerance backtest — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills · proposed tolerance: {PROPOSED_TOLERANCE_C}¢ vs current {CURRENT_TOLERANCE_C}¢",
    template='plotly_dark', height=600, barmode='overlay',
)
fig.show()
