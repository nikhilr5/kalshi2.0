"""Cancel-race-lost fills — fills that landed after tolerance triggered
but before the cancel could complete.

For each filled order:
  1. Get placed_ts (from order_events) and theo_at_placed
  2. Walk theo_state between placed_ts and fill_ts
  3. Find FIRST ts where adverse drift crossed tolerance — that's when
     Aston should have issued a cancel
  4. If such a ts exists, race_window = fill_ts − trigger_ts
  5. Realized P&L on those fills = the cost of the race

Cleanly separates:
  • Fills where tolerance NEVER crossed (these aren't race-lost)
  • Fills where tolerance crossed before fill (race-lost)
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
TOLERANCE_C   = 1.0


# =============================================================================
# Load + join placed events
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


# =============================================================================
# Theo at placement + walk theo_state in [placed_ts, fill_ts] per fill
# =============================================================================
theo_lookup = theo[['ts', 'ticker', 'theo']].sort_values('ts')
fills['_row'] = np.arange(len(fills))

keys_p = (fills[['_row', 'placed_ts', 'ticker']]
          .rename(columns={'placed_ts': 'ts'})
          .sort_values('ts'))
m_p = (pd.merge_asof(keys_p, theo_lookup, on='ts', by='ticker',
                      direction='backward')
         .sort_values('_row'))
fills['theo_at_placed'] = m_p['theo'].values

# Pre-group theo_state by ticker, storing ts as int64 ns for fast comparison
theo_by_ticker = {}
for tk, g in theo.groupby('ticker'):
    ts_ns = g['ts'].values.astype('datetime64[ns]').astype('int64')
    theo_vals = g['theo'].values.astype(float)
    theo_by_ticker[tk] = (ts_ns, theo_vals)

trigger_ts_list  = []
max_drift_c_list = []
for row in fills.itertuples():
    tk = row.ticker
    p_ts = row.placed_ts.value  # nanoseconds since epoch
    f_ts = row.ts.value
    sgn = row.sgn
    theo_p = row.theo_at_placed
    if pd.isna(theo_p) or tk not in theo_by_ticker:
        trigger_ts_list.append(pd.NaT)
        max_drift_c_list.append(np.nan)
        continue
    ts_ns, theo_vals = theo_by_ticker[tk]
    mask = (ts_ns > p_ts) & (ts_ns <= f_ts)
    if not mask.any():
        trigger_ts_list.append(pd.NaT)
        max_drift_c_list.append(0.0)
        continue
    sub_ts = ts_ns[mask]
    sub_theo = theo_vals[mask]
    adverse_c = (theo_p - sub_theo) * sgn * 100
    max_drift_c_list.append(float(adverse_c.max()))
    crossed = adverse_c >= TOLERANCE_C
    if crossed.any():
        first_idx = int(np.argmax(crossed))
        trigger_ts_list.append(pd.Timestamp(int(sub_ts[first_idx]), tz='UTC'))
    else:
        trigger_ts_list.append(pd.NaT)

fills['tolerance_trigger_ts'] = trigger_ts_list
fills['max_adverse_drift_c']  = max_drift_c_list
fills['race_window_ms'] = (
    (fills['ts'] - fills['tolerance_trigger_ts']).dt.total_seconds() * 1000
)


# =============================================================================
# Settlements + realized
# =============================================================================
api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100

race_lost = fills[fills['tolerance_trigger_ts'].notna()]
race_won  = fills[fills['tolerance_trigger_ts'].isna()]


# =============================================================================
# Headlines
# =============================================================================
pd.set_option('display.float_format', '{:+.2f}'.format)
pd.set_option('display.width', 200)

n_days = 4
print(f"\n=== Total fills: {len(fills):,} (over {n_days} days) ===")
print(f"   Tolerance NEVER crossed (max drift < {TOLERANCE_C}¢):  "
      f"{len(race_won):>5,}  ({len(race_won)/len(fills):.0%})")
print(f"   Tolerance crossed before fill (race-lost):       "
      f"{len(race_lost):>5,}  ({len(race_lost)/len(fills):.0%})")


def report_realized(label, df):
    v = df['realized_c'].dropna().values
    if len(v) < 2:
        print(f"   [{label}] insufficient data")
        return
    lo, hi = bootstrap_ci(v, B=2000)
    total = v.sum() / 100
    print(f"   [{label}] n={len(v):,}  mean={v.mean():+.2f}¢/fill  "
          f"CI [{lo:+.2f}, {hi:+.2f}]  total ${total:+.2f} "
          f"= ${total/n_days:+.2f}/day")


print(f"\n=== Realized P&L ===")
report_realized('tolerance never crossed', race_won)
report_realized('cancel race lost      ',  race_lost)


# =============================================================================
# By race window bucket
# =============================================================================
race_lost = race_lost.copy()
race_lost['race_bucket'] = pd.cut(
    race_lost['race_window_ms'],
    bins=[-1, 100, 500, 1000, 5000, 1e9],
    labels=['<100ms', '100-500ms', '500ms-1s', '1-5s', '>5s'],
)


def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'race_med_ms': g['race_window_ms'].median(),
           'max_drift_med_c': g['max_adverse_drift_c'].median()}
    v = g['realized_c'].dropna().values
    if len(v) >= 2:
        lo, hi = bootstrap_ci(v, B=2000)
        out['realized'] = float(v.mean())
        out['lo'] = lo
        out['hi'] = hi
    else:
        out['realized'] = out['lo'] = out['hi'] = np.nan
    return pd.Series(out)


print(f"\n=== Cancel-race-lost — by race window bucket ===")
print(race_lost.groupby('race_bucket', observed=True)
                .apply(summary, include_groups=False))


# =============================================================================
# Distribution of race windows
# =============================================================================
print(f"\n=== Race window distribution (cancel-race-lost fills) ===")
qs = race_lost['race_window_ms'].quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
for q, v in qs.items():
    print(f"   p{int(q*100):>2}: {v:>8.0f} ms")


# =============================================================================
# Viz
# =============================================================================
bar_df = (race_lost.groupby('race_bucket', observed=True)
                    .apply(summary, include_groups=False).reset_index())

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        f"Cancel-race-lost fills: realized P&L by race window (n={len(race_lost):,})",
        f"Race window distribution (log-scale, ms)",
    ),
    row_heights=[0.5, 0.5], vertical_spacing=0.15,
)

err_lo = (bar_df['realized'] - bar_df['lo']).clip(lower=0)
err_hi = (bar_df['hi'] - bar_df['realized']).clip(lower=0)
labels = [f"{b}<br>(n={int(n)})" for b, n in zip(bar_df['race_bucket'], bar_df['n'])]
fig.add_trace(go.Bar(
    x=labels, y=bar_df['realized'],
    marker_color=['#facc15', '#f97316', '#dc2626', '#7f1d1d', '#450a0a'][:len(bar_df)],
    error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                  color='#333', thickness=1.5),
    showlegend=False,
), row=1, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=1, col=1)

windows = race_lost['race_window_ms'].dropna().values
windows = windows[windows > 0]
fig.add_trace(go.Histogram(
    x=np.log10(windows), nbinsx=40,
    marker_color='#a78bfa', showlegend=False,
), row=2, col=1)

fig.update_yaxes(title_text='realized P&L (cents/fill)', row=1, col=1)
fig.update_xaxes(title_text='log₁₀(race window ms)', row=2, col=1)
fig.update_yaxes(title_text='count', row=2, col=1)
fig.update_layout(
    title=f"Cancel race analysis — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"tolerance={TOLERANCE_C}¢",
    template='plotly_dark', height=900,
)
fig.show()
