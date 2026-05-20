"""Daily adverse-selection dashboard for Aston KXETH15M fills.

Usage:
    python daily_dashboard.py                 # yesterday in CT
    python daily_dashboard.py --date 2026-05-19

Writes a single self-contained HTML to ./dashboards/{YYYY-MM-DD}.html.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'Aston'))
from utility import (
    SECONDS_PER_YEAR, fetch_settlements_from_api, load_all_data,
)
from kalshi_api import KalshiAPI

SERIES_PREFIX = 'KXETH15M'
CUTOFF_DAY    = '26MAY14'
MARKOUT_GRID  = [5, 10, 30, 60, 120, 300]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--date', default=None,
                   help='YYYY-MM-DD in local CT (default: yesterday)')
    return p.parse_args()


def resolve_day(arg: str | None) -> dt.date:
    if arg:
        return dt.date.fromisoformat(arg)
    return (dt.datetime.now(dt.timezone.utc)
            .astimezone(dt.timezone(dt.timedelta(hours=-5)))  # CT, no DST math
            .date() - dt.timedelta(days=1))


# =============================================================================
# Load and slice to the requested CT day
# =============================================================================
def load_day(day_ct: dt.date):
    theo, book, _spot, fills, events = load_all_data(SERIES_PREFIX, CUTOFF_DAY)
    fills = fills[fills['side'] == 'yes'].copy()

    for df in (theo, book, fills, events):
        df['day_ct'] = df['ts'].dt.tz_convert('America/Chicago').dt.date

    day_mask = lambda df: df['day_ct'] == day_ct
    return (theo[day_mask(theo)].drop(columns='day_ct').reset_index(drop=True),
            book[day_mask(book)].drop(columns='day_ct').reset_index(drop=True),
            fills[day_mask(fills)].drop(columns='day_ct').reset_index(drop=True),
            events[day_mask(events)].drop(columns='day_ct').reset_index(drop=True))


# =============================================================================
# Enrich fills with theo, mid, forward-mid markouts, settlement, moneyness
# =============================================================================
def enrich_fills(fills, theo, book) -> pd.DataFrame:
    theo = theo.sort_values(['ticker', 'ts'])
    fills = fills.sort_values('ts').reset_index(drop=True)

    fills = pd.merge_asof(
        fills,
        theo[['ts', 'ticker', 'spot', 'sigma', 'theo', 'seconds_to_expiry']].sort_values('ts'),
        on='ts', by='ticker', direction='backward',
    )

    book = book.sort_values(['ticker', 'ts']).copy()
    book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2
    mid_lookup = book[['ts', 'ticker', 'mid']].sort_values('ts')
    theo_lookup = theo[['ts', 'ticker', 'theo']].sort_values('ts')

    for h in MARKOUT_GRID:
        fwd = fills[['ts', 'ticker']].copy()
        fwd['ts'] = fwd['ts'] + pd.Timedelta(seconds=h)
        fwd = fwd.sort_values('ts')
        jm = pd.merge_asof(fwd, mid_lookup, on='ts', by='ticker', direction='backward')
        jt = pd.merge_asof(fwd, theo_lookup, on='ts', by='ticker', direction='backward')
        # mask rows where horizon would cross expiry — `mid` post-expiry is stale
        valid = fills['seconds_to_expiry'] >= h
        fills[f'mid_p{h}']  = np.where(valid, jm['mid'].values, np.nan)
        fills[f'theo_p{h}'] = np.where(valid, jt['theo'].values, np.nan)

    fills['sgn'] = np.where(fills['action'] == 'buy', +1, -1)
    fills['edge_at_fill_c']  = (fills['theo'] - fills['price']) * fills['sgn'] * 100
    fills['posted_edge_c']   = np.where(fills['action'] == 'buy', 7.0, 5.0)  # config snapshot
    for h in MARKOUT_GRID:
        fills[f'mkt_mid_{h}']  = (fills[f'mid_p{h}']  - fills['price']) * fills['sgn'] * 100
        fills[f'mkt_theo_{h}'] = (fills[f'theo_p{h}'] - fills['price']) * fills['sgn'] * 100

    # Moneyness in σ-units
    T = fills['seconds_to_expiry'].clip(lower=1) / SECONDS_PER_YEAR
    fills['mny_z'] = ((fills['spot'] - fills['strike']).abs()
                      / (fills['spot'] * fills['sigma'] * np.sqrt(T)))
    fills['mny_bucket'] = pd.cut(
        fills['mny_z'], bins=[-0.01, 0.25, 0.75, 1.5, 999],
        labels=['atm', 'near', 'far', 'tail'],
    )
    return fills


def attach_settlements(fills, api):
    settle = fetch_settlements_from_api(
        list(fills['ticker'].unique()), api,
        cache_path=ROOT / 'analysis/Aston/.settlements_cache.json',
    )
    fills['outcome'] = fills['ticker'].map(settle)
    fills['pnl_c'] = ((fills['outcome'] - fills['price']) * fills['sgn'] * 100
                      - fills['fee'].fillna(0) * 100)
    return fills


# =============================================================================
# Quote lifetime — proxy for "tolerance-crossed-to-fill" latency
# =============================================================================
def quote_lifetimes(fills, events) -> pd.DataFrame:
    """Per-fill seconds between order placement and fill (proxy for stale-quote latency).

    The cancel-race memory frames the real metric as "time between
    tolerance crossing and reprice/fill", which requires the strategy's
    desired-quote stream.  In its absence, fill-quote lifetime is a
    coarse lower bound: any latency above ~1s is a candidate stale-fill.
    """
    placed = events[events['event_type'] == 'placed'][['ts', 'client_order_id']].rename(columns={'ts': 'placed_ts'})
    return fills.merge(placed, on='client_order_id', how='left').assign(
        quote_life_s=lambda d: (d['ts'] - d['placed_ts']).dt.total_seconds(),
    )


def theo_drift_during_quote(fills_with_life, theo) -> pd.Series:
    """For each fill, compute (theo_at_fill - theo_at_place) * sgn * 100.

    Positive = theo moved in our favor while quote rested (i.e. we got
    filled at a price that was even more attractive by the time the
    counterparty took us — adverse-selection signal)."""
    placed_ts = fills_with_life[['placed_ts', 'ticker']].rename(columns={'placed_ts': 'ts'})
    placed_ts['_idx'] = np.arange(len(placed_ts))
    placed_ts = placed_ts.dropna(subset=['ts']).sort_values('ts')
    theo_l = theo[['ts', 'ticker', 'theo']].sort_values('ts')
    j = pd.merge_asof(placed_ts, theo_l, on='ts', by='ticker', direction='backward')
    j = j.set_index('_idx').reindex(np.arange(len(fills_with_life)))
    return (fills_with_life['theo'] - j['theo']) * fills_with_life['sgn'] * 100


# =============================================================================
# Figure builder
# =============================================================================
def build_figure(day_ct, fills, fills_q):
    n = len(fills)
    pnl = fills['pnl_c'].sum() / 100 if 'pnl_c' in fills else np.nan
    edge = fills['edge_at_fill_c'].mean()
    mk30 = fills['mkt_mid_30'].mean()
    n_markets = fills['ticker'].nunique()
    summary = (f"<b>{day_ct.isoformat()}</b>  n_fills={n}  "
               f"markets={n_markets}  Pnl=${pnl:+.2f}  "
               f"edge_at_fill={edge:+.2f}c  markout@30s(mid)={mk30:+.2f}c")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Markout curve (mid primary, theo overlay)',
            'Edge-at-fill vs posted edge',
            'Quote-lifetime CDF (placed→fill, sec)',
            'Quote-life theo drift vs 30s mid markout',
            'P&L heatmap — action × moneyness (¢/fill)',
            'Per-market settled P&L ($), worst first',
        ),
        vertical_spacing=0.10, horizontal_spacing=0.09,
    )

    # (1,1) markout curve — mean across fills, by action
    for action, color in (('buy', '#22c55e'), ('sell', '#f87171')):
        sub = fills[fills['action'] == action]
        mid_means = [sub[f'mkt_mid_{h}'].mean() for h in MARKOUT_GRID]
        theo_means = [sub[f'mkt_theo_{h}'].mean() for h in MARKOUT_GRID]
        fig.add_trace(go.Scatter(
            x=MARKOUT_GRID, y=mid_means, mode='lines+markers',
            name=f'{action} mid', line=dict(color=color, width=2),
            legendgroup=action,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=MARKOUT_GRID, y=theo_means, mode='lines+markers',
            name=f'{action} theo', line=dict(color=color, dash='dot'),
            legendgroup=action,
        ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#888', width=1), row=1, col=1)

    # (1,2) edge-at-fill histogram with posted-edge marker
    for action, color in (('buy', '#22c55e'), ('sell', '#f87171')):
        v = fills.loc[fills['action'] == action, 'edge_at_fill_c'].values
        fig.add_trace(go.Histogram(
            x=v, name=f'{action}', marker_color=color, opacity=0.55, nbinsx=40,
            legendgroup=f'edge_{action}',
        ), row=1, col=2)
    fig.add_vline(x=7, line=dict(color='#22c55e', dash='dash'),
                  annotation_text='buy posted 7c', row=1, col=2)
    fig.add_vline(x=5, line=dict(color='#f87171', dash='dash'),
                  annotation_text='sell posted 5c', row=1, col=2)

    # (2,1) quote-lifetime CDF
    life = fills_q['quote_life_s'].dropna().sort_values().values
    if len(life):
        ys = np.arange(1, len(life) + 1) / len(life)
        fig.add_trace(go.Scatter(
            x=life, y=ys, mode='lines', line=dict(color='#a78bfa'),
            name='quote life', showlegend=False,
        ), row=2, col=1)
        for q in (0.5, 0.9, 0.99):
            v = np.quantile(life, q)
            fig.add_vline(x=v, line=dict(color='#888', dash='dot'),
                          annotation_text=f'p{int(q*100)}={v:.1f}s', row=2, col=1)
        fig.update_xaxes(type='log', row=2, col=1)

    # (2,2) quote-life theo drift vs 30s markout
    valid = fills_q[['theo_drift_c', 'mkt_mid_30']].dropna()
    if len(valid):
        fig.add_trace(go.Scatter(
            x=valid['theo_drift_c'], y=valid['mkt_mid_30'], mode='markers',
            marker=dict(size=4, color='#facc15', opacity=0.55), showlegend=False,
        ), row=2, col=2)
        if len(valid) >= 5:
            corr = valid['theo_drift_c'].corr(valid['mkt_mid_30'])
            fig.add_annotation(text=f'r={corr:+.2f}  n={len(valid)}',
                               xref='x4 domain', yref='y4 domain',
                               x=0.02, y=0.95, showarrow=False,
                               font=dict(color='#fff'), row=2, col=2)
        fig.add_hline(y=0, line=dict(color='#888', width=1), row=2, col=2)
        fig.add_vline(x=0, line=dict(color='#888', width=1), row=2, col=2)

    # (3,1) heatmap action × moneyness
    if 'pnl_c' in fills.columns:
        heat = (fills.groupby(['action', 'mny_bucket'], observed=True)['pnl_c']
                      .mean().reset_index())
        pivot = heat.pivot(index='action', columns='mny_bucket', values='pnl_c')
        fig.add_trace(go.Heatmap(
            z=pivot.values, x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale='RdYlGn', zmid=0,
            text=np.round(pivot.values, 1), texttemplate='%{text}',
            colorbar=dict(title='c/fill', x=0.46, len=0.28, y=0.16),
        ), row=3, col=1)

    # (3,2) per-market P&L bar
    if 'pnl_c' in fills.columns:
        per_mkt = (fills.dropna(subset=['outcome'])
                         .groupby('ticker')['pnl_c'].sum()
                         .div(100).sort_values())
        colors = ['#f87171' if v < 0 else '#22c55e' for v in per_mkt.values]
        fig.add_trace(go.Bar(
            x=per_mkt.values, y=per_mkt.index.astype(str), orientation='h',
            marker_color=colors, showlegend=False,
        ), row=3, col=2)

    fig.update_xaxes(title_text='horizon (s)', row=1, col=1)
    fig.update_yaxes(title_text='c/fill', row=1, col=1)
    fig.update_xaxes(title_text='edge_at_fill (c)', row=1, col=2)
    fig.update_xaxes(title_text='seconds (log)', row=2, col=1)
    fig.update_yaxes(title_text='cdf', row=2, col=1)
    fig.update_xaxes(title_text='theo drift while resting (c)', row=2, col=2)
    fig.update_yaxes(title_text='markout@30s (c)', row=2, col=2)
    fig.update_xaxes(title_text='$ per market', row=3, col=2)
    fig.update_yaxes(showticklabels=(n_markets < 50), row=3, col=2)

    fig.update_layout(
        title=summary, template='plotly_dark',
        height=1300, barmode='overlay', bargap=0.05,
        margin=dict(l=80, r=40, t=120, b=60),
    )
    return fig


def main():
    args = parse_args()
    day_ct = resolve_day(args.date)
    print(f'[dash] day_ct={day_ct.isoformat()}')

    theo, book, fills, events = load_day(day_ct)
    if fills.empty:
        print(f'[dash] no fills for {day_ct.isoformat()} — abort')
        return

    fills = enrich_fills(fills, theo, book)
    api = KalshiAPI()
    fills = attach_settlements(fills, api)
    fills_q = quote_lifetimes(fills, events)
    fills_q['theo_drift_c'] = theo_drift_during_quote(fills_q, theo)

    out_dir = Path(__file__).resolve().parent / 'dashboards'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'{day_ct.isoformat()}.html'
    fig = build_figure(day_ct, fills, fills_q)
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'[dash] wrote {out_path}')


if __name__ == '__main__':
    main()
