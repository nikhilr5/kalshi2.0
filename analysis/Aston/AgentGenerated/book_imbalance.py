"""Book imbalance at fill time vs realized PnL.

For each fill, look up the kalshi_book snapshot just before the fill,
compute imbalance = (bid_size - ask_size) / total_size, and sign it
relative to our position:

    adverse_imb = -sgn * imbalance   (positive = book was hostile)

Bucket fills by adverse_imb level and look at mean realized PnL.
Also slice on non-stale fills (disagree_c >= 0) to test whether the
signal predicts the structural bleed, not just the stale-quote bleed.
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
# Load + merge theo (for disagree) + book sizes (for imbalance)
# =============================================================================
theo, book, _spot, fills, _events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
theo  = theo.sort_values(['ticker', 'ts']).reset_index(drop=True)
book  = book.sort_values(['ticker', 'ts']).reset_index(drop=True)
fills = fills.sort_values('ts').reset_index(drop=True)
if 'side' in fills.columns:
    fills = fills[fills['side'] == 'yes'].reset_index(drop=True)

fills = pd.merge_asof(
    fills,
    theo[['ts', 'ticker', 'theo', 'seconds_to_expiry']].sort_values('ts'),
    on='ts', by='ticker', direction='backward',
)
fills = pd.merge_asof(
    fills.sort_values('ts'),
    book[['ts', 'ticker', 'bid_size', 'ask_size']].sort_values('ts'),
    on='ts', by='ticker', direction='backward',
    tolerance=pd.Timedelta(seconds=5),
)


# =============================================================================
# Imbalance + sign + disagree + realized
# =============================================================================
total_size = fills['bid_size'].fillna(0) + fills['ask_size'].fillna(0)
fills['imbalance'] = np.where(
    total_size > 0,
    (fills['bid_size'] - fills['ask_size']) / total_size,
    np.nan,
)
fills['sgn']         = np.where(fills['action'] == 'buy', +1, -1)
fills['adverse_imb'] = -fills['sgn'] * fills['imbalance']

api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * 100

fills['mid'] = (fills['kalshi_yes_bid'] + fills['kalshi_yes_ask']) / 2
fills['disagree_c'] = (fills['theo'] - fills['mid']) * fills['sgn'] * 100

fills['imb_bucket'] = pd.cut(
    fills['adverse_imb'],
    bins=[-1.01, -0.5, -0.1, 0.1, 0.5, 1.01],
    labels=['strong favorable', 'mild favorable', 'neutral',
            'mild hostile', 'strong hostile'],
)


# =============================================================================
# Summary tables
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'imb_mean': g['adverse_imb'].mean(),
           'disagree_mean_c': g['disagree_c'].mean()}
    v = g['realized_c'].dropna().values
    if len(v) >= 2:
        lo, hi = bootstrap_ci(v, B=2000)
        out['realized_mean'] = float(v.mean())
        out['realized_lo'] = lo
        out['realized_hi'] = hi
    else:
        out['realized_mean'] = out['realized_lo'] = out['realized_hi'] = np.nan
    return pd.Series(out)


pd.set_option('display.float_format', '{:+.2f}'.format)
pd.set_option('display.width', 200)

print("\n=== ALL FILLS — realized PnL by adverse_imb bucket ===")
print(fills.groupby('imb_bucket', observed=True)
            .apply(summary, include_groups=False))

print("\n=== ALL FILLS — by action × imb_bucket ===")
print(fills.groupby(['action', 'imb_bucket'], observed=True)
            .apply(summary, include_groups=False))

nonstale = fills[fills['disagree_c'] >= 0]
print(f"\n=== NON-STALE FILLS ({len(nonstale):,}) — by imb_bucket ===")
print("(filter: disagree_c >= 0 — excludes stale-quote bleed)")
print(nonstale.groupby('imb_bucket', observed=True)
              .apply(summary, include_groups=False))

df = fills.dropna(subset=['adverse_imb', 'realized_c'])
print(f"\n=== ρ(adverse_imb, realized) = {df['adverse_imb'].corr(df['realized_c']):+.3f} "
      f"(all fills, n={len(df)}) ===")
df_ns = nonstale.dropna(subset=['adverse_imb', 'realized_c'])
print(f"=== ρ(adverse_imb, realized) = {df_ns['adverse_imb'].corr(df_ns['realized_c']):+.3f} "
      f"(non-stale only, n={len(df_ns)}) ===")


# =============================================================================
# Viz — bar chart of mean realized by imb_bucket for all-fills and non-stale
# =============================================================================
bucket_all = (fills.groupby('imb_bucket', observed=True)
                    .apply(summary, include_groups=False).reset_index())
bucket_ns  = (nonstale.groupby('imb_bucket', observed=True)
                       .apply(summary, include_groups=False).reset_index())

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        f"All fills ({len(fills.dropna(subset=['adverse_imb'])):,}) — "
        "mean realized P&L by book imbalance bucket",
        f"Non-stale fills only ({len(nonstale.dropna(subset=['adverse_imb'])):,}) — "
        "same view, excludes stale-quote bleed",
    ),
    row_heights=[0.5, 0.5], vertical_spacing=0.15,
)
bar_colors = ['#22c55e', '#84cc16', '#facc15', '#f97316', '#dc2626']
for row, d in [(1, bucket_all), (2, bucket_ns)]:
    err_lo = (d['realized_mean'] - d['realized_lo']).clip(lower=0)
    err_hi = (d['realized_hi'] - d['realized_mean']).clip(lower=0)
    labels = [f"{b}<br>(n={int(n)})" for b, n in zip(d['imb_bucket'], d['n'])]
    fig.add_trace(go.Bar(
        x=labels, y=d['realized_mean'],
        marker_color=bar_colors[:len(d)],
        error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                      color='#333', thickness=1.5),
        showlegend=False,
    ), row=row, col=1)
    fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=row, col=1)
    fig.update_yaxes(title_text='realized P&L (cents)', row=row, col=1)

fig.update_layout(
    title=f"Book imbalance vs realized P&L — {SERIES_PREFIX} ≥ {CUTOFF_DAY}",
    template='plotly_dark', height=900,
)
fig.show()
