"""Realized P&L by ITM / ATM / OTM moneyness.

Uses theo at fill time as the moneyness proxy:
  • theo > 0.5  → ITM yes (yes favored, spot > strike)
  • theo < 0.5  → OTM yes
  • theo ≈ 0.5  → ATM

Bucket fills into 5 moneyness regions and show realized P&L per
(action × moneyness).  Tests whether bleed concentrates in ATM
(settlement variance) or has directional ITM/OTM bias.
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
    brier_score,
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
fills = fills.dropna(subset=['theo']).reset_index(drop=True)

api = KalshiAPI()
settlements = fetch_settlements_from_api(
    list(fills['ticker'].unique()), api,
    cache_path=Path(__file__).resolve().parent.parent / ".settlements_cache.json",
)
fills['outcome']    = fills['ticker'].map(settlements)
fills['sgn']        = np.where(fills['action'] == 'buy', +1, -1)
fills['realized_c'] = (fills['outcome']   - fills['price']) * fills['sgn'] * 100
fills['spread_c']   = (fills['theo']      - fills['price']) * fills['sgn'] * 100

fills['mid'] = (fills['kalshi_yes_bid'] + fills['kalshi_yes_ask']) / 2
fills['mny_bucket'] = pd.cut(
    fills['theo'],
    bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
    labels=['deep_OTM (<0.2)', 'OTM (0.2-0.4)', 'ATM (0.4-0.6)',
            'ITM (0.6-0.8)', 'deep_ITM (>0.8)'],
)


# =============================================================================
# Summary
# =============================================================================
def summary(g: pd.DataFrame) -> pd.Series:
    out = {'n': len(g),
           'mean_theo': g['theo'].mean(),
           'spread_mean': g['spread_c'].mean()}
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

print("\n=== Realized P&L by moneyness (overall) ===")
print(fills.groupby('mny_bucket', observed=True)
            .apply(summary, include_groups=False))

print("\n=== Realized P&L by action × moneyness ===")
by_action = (fills.groupby(['action', 'mny_bucket'], observed=True)
                   .apply(summary, include_groups=False))
print(by_action)

print("\n=== Theo as predictor by moneyness — calibration vs Kalshi mid ===")
predictor_rows = []
for b in fills['mny_bucket'].cat.categories:
    sub = fills[fills['mny_bucket'] == b].dropna(subset=['theo', 'outcome'])
    if len(sub) < 5:
        continue
    sub_mid = sub.dropna(subset=['mid'])
    sub_mid = sub_mid[(sub_mid['kalshi_yes_bid'] > 0)
                      & (sub_mid['kalshi_yes_ask'] > sub_mid['kalshi_yes_bid'])]
    theo_brier = brier_score(sub['theo'], sub['outcome'])
    mid_brier  = brier_score(sub_mid['mid'], sub_mid['outcome'])
    theo_mae   = (sub['theo']     - sub['outcome']).abs().mean()
    mid_mae    = (sub_mid['mid']  - sub_mid['outcome']).abs().mean()
    theo_bias  = (sub['theo']     - sub['outcome']).mean()
    mid_bias   = (sub_mid['mid']  - sub_mid['outcome']).mean()
    predictor_rows.append({
        'bucket':     str(b),
        'n':          len(sub),
        'theo_mean':  sub['theo'].mean(),
        'outcome':    sub['outcome'].mean(),
        'theo_bias':  theo_bias,
        'theo_MAE':   theo_mae,
        'theo_Brier': theo_brier,
        'mid_Brier':  mid_brier,
        'winner':     'theo' if (theo_brier or 1) < (mid_brier or 1) else 'mid',
    })
predictor_df = pd.DataFrame(predictor_rows)
print(predictor_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

# Overall Brier for reference
overall_theo = brier_score(fills['theo'], fills['outcome'])
overall_mid  = brier_score(fills.dropna(subset=['mid'])['mid'],
                            fills.dropna(subset=['mid'])['outcome'])
print(f"\n  Overall:  theo_Brier={overall_theo:.4f}  mid_Brier={overall_mid:.4f}")

print("\n=== Dollar impact per bucket (over 4 days) ===")
agg = (fills.groupby(['action', 'mny_bucket'], observed=True)
              ['realized_c'].agg(['count', 'sum', 'mean']))
agg['total_$'] = agg['sum'] / 100
agg['$/day'] = agg['total_$'] / 4
print(agg.round(2))


# =============================================================================
# Viz — bar chart with CI bars, side-by-side by action
# =============================================================================
labels = ['deep_OTM (<0.2)', 'OTM (0.2-0.4)', 'ATM (0.4-0.6)',
          'ITM (0.6-0.8)', 'deep_ITM (>0.8)']

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Realized P&L per fill by moneyness — buys vs sells (CI bars)",
        "Total dollar P&L per bucket over 4 days (count × mean)",
    ),
    row_heights=[0.55, 0.45], vertical_spacing=0.15,
)

for action, color in [('buy', '#22c55e'), ('sell', '#a78bfa')]:
    rows = []
    for b in labels:
        sub = fills[(fills['action'] == action) & (fills['mny_bucket'] == b)]
        v = sub['realized_c'].dropna().values
        if len(v) >= 2:
            lo, hi = bootstrap_ci(v, B=2000)
            rows.append((b, len(v), float(v.mean()), lo, hi))
        else:
            rows.append((b, len(v), np.nan, np.nan, np.nan))
    n = [r[1] for r in rows]
    m = [r[2] for r in rows]
    lo = [r[3] for r in rows]
    hi = [r[4] for r in rows]
    err_lo = [(mi - li) if mi == mi and li == li else 0 for mi, li in zip(m, lo)]
    err_hi = [(hi_ - mi) if mi == mi and hi_ == hi_ else 0 for mi, hi_ in zip(m, hi)]
    err_lo = [max(0, x) for x in err_lo]
    err_hi = [max(0, x) for x in err_hi]
    xlbl = [f"{b}<br>(n={ni})" for b, ni in zip(labels, n)]
    fig.add_trace(go.Bar(
        x=xlbl, y=m, name=action, marker_color=color, opacity=0.85,
        error_y=dict(type='data', array=err_hi, arrayminus=err_lo,
                      color='#333', thickness=1.5),
    ), row=1, col=1)

# Bottom — dollar totals stacked by action
totals_buy = []
totals_sell = []
for b in labels:
    bv = fills[(fills['action'] == 'buy') & (fills['mny_bucket'] == b)]['realized_c'].sum() / 100
    sv = fills[(fills['action'] == 'sell') & (fills['mny_bucket'] == b)]['realized_c'].sum() / 100
    totals_buy.append(bv)
    totals_sell.append(sv)

fig.add_trace(go.Bar(x=labels, y=totals_buy,  name='buy total $',
                     marker_color='#22c55e', opacity=0.85, showlegend=False),
              row=2, col=1)
fig.add_trace(go.Bar(x=labels, y=totals_sell, name='sell total $',
                     marker_color='#a78bfa', opacity=0.85, showlegend=False),
              row=2, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=1, col=1)
fig.add_hline(y=0, line=dict(color='#666', width=1, dash='dot'), row=2, col=1)

fig.update_yaxes(title_text='realized P&L (cents/fill)', row=1, col=1)
fig.update_yaxes(title_text='total P&L ($)', row=2, col=1)
fig.update_xaxes(title_text='theo at fill (moneyness bucket)', row=2, col=1)
fig.update_layout(
    title=f"Realized P&L by moneyness — {SERIES_PREFIX} ≥ {CUTOFF_DAY} · "
          f"{len(fills):,} fills",
    template='plotly_dark', height=900, barmode='group',
)
fig.show()
