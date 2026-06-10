"""Aston Live Dashboard — browser, two tabs, auto-refresh.

  Tab 1 — P&L:                cumulative + daily + fill count
  Tab 2 — Adverse Selection:  markout@30s, late-cancel detector,
                              theo drift while resting

Read-only SQLite over the recorder DBs (no contention with the live
writer).  Refresh on a single timer; both tabs update together.

Run:    python3 live_dashboard.py
Open:   http://localhost:8050
"""

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'Aston'))
from utility import (
    bootstrap_ci, fetch_settlements_from_api, list_eligible_dbs,
    parse_day_suffix,
)
from kalshi_api import KalshiAPI

SERIES_PREFIX     = 'KXETH15M'
CUTOFF_DAY        = '26MAY15'
CUTOFF_DATE       = parse_day_suffix(CUTOFF_DAY)
MARKOUT_HORIZON   = 30
REFRESH_SECONDS   = 900  # 15 min — aligned with the 15-min market roll cadence
PORT              = 8050
CACHE_PATH        = ROOT / 'analysis/Aston/.settlements_cache.json'
BOOTSTRAP_B       = 500   # was 2000 — 4x faster, CIs still tight at n>=200
TOLERANCE_C       = 1.0   # configured strategy tolerance (cents)

api = KalshiAPI()
_SETTLE_CACHE: dict = {}  # ticker → outcome, populated once per refresh


# =============================================================================
# Read-only SQLite loaders
# =============================================================================
def _read_table_ro(path: Path, table: str) -> pd.DataFrame:
    for uri in (f"file:{path}?mode=ro", f"file:{path}?immutable=1"):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5)
            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY ts", conn)
            conn.close()
            return df
        except Exception:
            continue
    return pd.DataFrame()


def load_readonly(tables: tuple) -> dict:
    """Returns {table_name: concatenated DataFrame} across eligible DBs."""
    files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY)
    parts: dict[str, list] = {t: [] for t in tables}
    for path in files:
        for t in tables:
            df = _read_table_ro(path, t)
            if not df.empty:
                parts[t].append(df)
    out = {}
    for t, ps in parts.items():
        df = pd.concat(ps, ignore_index=True) if ps else pd.DataFrame()
        if not df.empty and 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'], utc=True, format='ISO8601')
        out[t] = df
    return out


# =============================================================================
# P&L tab
# =============================================================================
def compute_pnl(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    fills = fills[fills['side'] == 'yes'].copy()
    fills['outcome'] = fills['ticker'].map(_SETTLE_CACHE)
    fills['sgn']     = np.where(fills['action'] == 'buy', +1, -1)
    fills['realized_c'] = ((fills['outcome'] - fills['price'])
                            * fills['sgn'] * fills['count'] * 100)
    fills = fills.dropna(subset=['realized_c']).copy()
    fills['date'] = pd.to_datetime(
        fills['ticker'].str.split('-').str[1].str[:7],
        format='%y%b%d', errors='coerce').dt.date
    fills = fills.dropna(subset=['date'])
    fills = fills[fills['date'] >= CUTOFF_DATE]
    return fills.sort_values('ts').reset_index(drop=True)


def pnl_stats(fills):
    if fills.empty:
        return html.Div("No fills yet.", style={"color": "#888"})
    daily = fills.groupby('date')['realized_c'].sum() / 100
    cumulative = daily.cumsum()
    total = float(cumulative.iloc[-1])
    n_days = len(daily)
    mean = float(daily.mean())
    std  = float(daily.std(ddof=1)) if n_days > 1 else 0.0
    sharpe = (mean / std * (365 ** 0.5)) if std > 0 else float('nan')
    tstat = ((mean / (std / (n_days ** 0.5)))
             if (std > 0 and n_days > 1) else float('nan'))
    drawdown = cumulative - cumulative.cummax()
    max_dd = float(-drawdown.min())
    today = fills[fills['date'] == fills['date'].max()]
    today_pnl   = float(today['realized_c'].sum() / 100)
    today_fills = len(today)
    last_7 = daily.tail(7).sum()
    color_total = '#22c55e' if total >= 0 else '#dc2626'
    color_today = '#22c55e' if today_pnl >= 0 else '#dc2626'
    return html.Div([
        html.H2(f"${total:+.2f}", style={"color": color_total, "margin": 0}),
        html.Div("Cumulative net P&L", style={"color": "#888"}),
        html.Hr(),
        html.Div([
            html.Div([html.B("Today: "),
                      html.Span(f"${today_pnl:+.2f}",
                                style={"color": color_today}),
                      html.Span(f"  ({today_fills} fills)",
                                style={"color": "#888"})]),
            html.Div([html.B("Last 7 days: "),
                      html.Span(f"${last_7:+.2f}")]),
            html.Div([html.B("Days traded: "), html.Span(f"{n_days}")]),
            html.Div([html.B("Total fills: "), html.Span(f"{len(fills):,}")]),
        ]),
        html.Hr(),
        html.Div([
            html.Div([html.B("Mean daily: "),
                      html.Span(f"${mean:+.2f}/day")]),
            html.Div([html.B("Daily SD: "), html.Span(f"${std:.2f}")]),
            html.Div([html.B("Sharpe (annualized): "),
                      html.Span(f"{sharpe:+.2f}")]),
            html.Div([html.B("t-stat (daily PnL > 0): "),
                      html.Span(f"{tstat:+.2f}",
                                style={"color": ("#22c55e" if tstat > 2
                                                  else "#facc15" if tstat > 1
                                                  else "#dc2626")}
                                if tstat == tstat else {}),
                      html.Span(f"  (n={n_days} days)",
                                style={"color": "#888"})]),
            html.Div([html.B("Max drawdown: "), html.Span(f"${max_dd:.2f}")]),
        ]),
        html.Hr(),
        html.Div(f"Refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
                 style={"color": "#666", "fontSize": "0.8em"}),
    ], style={"fontFamily": "monospace", "fontSize": "1.1em",
              "lineHeight": "1.7"})


# 2026-06-09: v2 strategy (OSM rewrite) + 3-lot orders + cap 30 went live
# (edges stayed at the 7c/5c baseline).  Data left of this line is the
# old config: v1 engine, 1-lot, cap 8-10.
CONFIG_CHANGE_DATE = "2026-06-09"


def pnl_figure(fills):
    if fills.empty:
        return go.Figure().update_layout(
            template='plotly_dark', height=700,
            annotations=[dict(text="no fills yet", x=0.5, y=0.5, showarrow=False)])
    daily = fills.groupby('date')['realized_c'].sum() / 100
    cumulative = daily.cumsum()
    fills_per_day = fills.groupby('date').size()
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Cumulative realized P&L ($)", "Daily P&L ($)",
                         "Fills per day"),
        row_heights=[0.5, 0.3, 0.2], shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values, mode='lines+markers',
        line=dict(color='#22c55e', width=3), marker=dict(size=8)), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'), row=1, col=1)
    bar_colors = ['#22c55e' if v >= 0 else '#dc2626' for v in daily.values]
    fig.add_trace(go.Bar(
        x=daily.index, y=daily.values, marker_color=bar_colors,
        text=[f"${v:+.2f}" for v in daily.values], textposition='outside'),
        row=2, col=1)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'), row=2, col=1)
    fig.add_trace(go.Bar(
        x=fills_per_day.index, y=fills_per_day.values,
        marker_color='#a78bfa'), row=3, col=1)
    for row in (1, 2, 3):
        fig.add_vline(x=CONFIG_CHANGE_DATE, line=dict(color='#facc15', width=2),
                      row=row, col=1)
    fig.add_annotation(x=CONFIG_CHANGE_DATE, y=1, yref='paper',
                       text="v2 + 3-lot", showarrow=False,
                       font=dict(color='#facc15', size=11),
                       xanchor='left', yanchor='top')
    fig.update_yaxes(title_text='$', row=1, col=1)
    fig.update_yaxes(title_text='$', row=2, col=1)
    fig.update_yaxes(title_text='fills', row=3, col=1)
    fig.update_layout(template='plotly_dark', height=850, showlegend=False)
    return fig


# =============================================================================
# Adverse-selection tab
# =============================================================================
def enrich_fills(fills, theo, book, events) -> pd.DataFrame:
    fills = fills[fills['side'] == 'yes'].sort_values('ts').reset_index(drop=True)
    fills = pd.merge_asof(
        fills,
        theo[['ts', 'ticker', 'theo', 'seconds_to_expiry']].sort_values('ts'),
        on='ts', by='ticker', direction='backward')
    fills['sgn'] = np.where(fills['action'] == 'buy', +1, -1)
    book = book.sort_values(['ticker', 'ts']).copy()
    book['mid'] = (book['yes_bid'] + book['yes_ask']) / 2
    fwd = fills[['ts', 'ticker']].copy()
    fwd['ts'] = fwd['ts'] + pd.Timedelta(seconds=MARKOUT_HORIZON)
    fwd = fwd.sort_values('ts')
    j = pd.merge_asof(
        fwd, book[['ts', 'ticker', 'mid']].sort_values('ts'),
        on='ts', by='ticker', direction='backward')
    valid = fills['seconds_to_expiry'] >= MARKOUT_HORIZON
    fills['mid_p30'] = np.where(valid, j['mid'].values, np.nan)
    fills['mkt_mid_30'] = (fills['mid_p30'] - fills['price']) * fills['sgn'] * 100
    # Pure adverse-selection: mid-vs-mid drift, independent of edge captured.
    fills['mid_at_fill'] = (fills['kalshi_yes_bid'] + fills['kalshi_yes_ask']) / 2
    fills['mid_drift_30'] = (fills['mid_p30'] - fills['mid_at_fill']) * fills['sgn'] * 100

    placed = events[events['event_type'] == 'placed'][
        ['ts', 'ticker', 'client_order_id']
    ].rename(columns={'ts': 'placed_ts'}).drop_duplicates('client_order_id')
    fills = fills.merge(placed, on=['client_order_id', 'ticker'], how='left')
    pa = fills[['placed_ts', 'ticker']].copy()
    pa['_idx'] = np.arange(len(pa))
    pa = pa.dropna(subset=['placed_ts']).rename(
        columns={'placed_ts': 'ts'}).sort_values('ts')
    j2 = pd.merge_asof(
        pa, theo[['ts', 'ticker', 'theo']]
            .rename(columns={'theo': 'theo_at_place'}).sort_values('ts'),
        on='ts', by='ticker', direction='backward')
    theo_at_place = pd.Series(np.nan, index=np.arange(len(fills)))
    theo_at_place.loc[j2['_idx'].values] = j2['theo_at_place'].values
    fills['theo_at_place'] = theo_at_place.values
    fills['theo_drift_c'] = (
        (fills['theo'] - fills['theo_at_place']) * fills['sgn'] * 100)

    fills['day'] = fills['ts'].dt.tz_convert('America/Chicago').dt.date
    return fills


def attach_settlements_adv(fills) -> pd.DataFrame:
    fills['outcome'] = fills['ticker'].map(_SETTLE_CACHE)
    fills['realized_c'] = (
        (fills['outcome'] - fills['price']) * fills['sgn'] * 100)
    return fills


def cancel_race(fills, events) -> pd.DataFrame:
    f = fills[fills['kalshi_ts'].notna()].copy()
    cancels = events[(events['event_type'] == 'cancelled')
                     & (events['kalshi_ts'].notna())][
        ['client_order_id', 'kalshi_ts']
    ].rename(columns={'kalshi_ts': 'kalshi_ts_cancel'})
    if f.empty or cancels.empty:
        return f.iloc[0:0].assign(cancel_behind_ms=np.nan)
    merged = f.merge(cancels, on='client_order_id', how='inner')
    merged['kalshi_ts_fill'] = pd.to_datetime(
        merged['kalshi_ts'], utc=True, format='ISO8601')
    merged['kalshi_ts_cancel'] = pd.to_datetime(
        merged['kalshi_ts_cancel'], utc=True, format='ISO8601')
    merged['cancel_behind_ms'] = (
        (merged['kalshi_ts_cancel'] - merged['kalshi_ts_fill'])
        .dt.total_seconds() * 1000)
    return merged[merged['cancel_behind_ms'] > 0].reset_index(drop=True)


def _ci(g, col):
    v = g[col].dropna().values
    if len(v) < 2:
        return pd.Series({'mean': np.nan, 'lo': np.nan, 'hi': np.nan, 'n': len(v)})
    lo, hi = bootstrap_ci(v, B=BOOTSTRAP_B)
    return pd.Series({'mean': float(np.mean(v)), 'lo': lo, 'hi': hi, 'n': len(v)})


def adv_daily(fills, late):
    by_day = fills.groupby('day')
    mk = by_day.apply(_ci, col='mkt_mid_30', include_groups=False)
    td = by_day.apply(_ci, col='theo_drift_c', include_groups=False)
    pnl = by_day['realized_c'].sum().div(100).rename('pnl_$')
    n   = by_day.size().rename('n_fills')
    if not late.empty:
        ln = late.groupby('day').size().rename('late_n')
        lc = late.groupby('day')['realized_c'].sum().div(100).rename('late_$')
    else:
        ln = pd.Series(dtype=int, name='late_n')
        lc = pd.Series(dtype=float, name='late_$')
    out = pd.concat([
        n, pnl,
        mk.rename(columns={'mean': 'mkt30_mean', 'lo': 'mkt30_lo',
                            'hi': 'mkt30_hi', 'n': 'mkt30_n'}),
        td.rename(columns={'mean': 'drift_mean', 'lo': 'drift_lo',
                            'hi': 'drift_hi', 'n': 'drift_n'}),
        ln, lc,
    ], axis=1).reset_index()
    out['late_n'] = out['late_n'].fillna(0).astype(int)
    out['late_$'] = out['late_$'].fillna(0.0)
    return out.sort_values('day').reset_index(drop=True)


def _ci_band(fig, x, mean, lo, hi, name, color, row):
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(hi) + list(lo)[::-1],
        fill='toself', fillcolor=color.replace('1.0)', '0.2)'),
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
        showlegend=False), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode='lines+markers', name=name,
        line=dict(color=color, width=2), marker=dict(size=8)),
        row=row, col=1)


def adv_figure(daily):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=(
            'Markout @ 30s (mid) — daily mean ¢/fill, 95% CI',
            'Late-cancel detector — daily count (bars) + realized cost (line)',
            'Theo drift while resting — daily mean ¢/fill, 95% CI'),
        vertical_spacing=0.07,
        specs=[[{}], [{'secondary_y': True}], [{}]])
    if daily.empty:
        fig.update_layout(template='plotly_dark', height=950)
        return fig
    x = daily['day']
    _ci_band(fig, x, daily['mkt30_mean'], daily['mkt30_lo'], daily['mkt30_hi'],
             'markout@30s', 'rgba(34,197,94,1.0)', row=1)
    fig.add_hline(y=0, line=dict(color='#888', width=1, dash='dot'), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x, y=daily['late_n'], name='late-cancel n',
        marker_color='#a78bfa', opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=daily['late_$'], name='late-cancel $',
        mode='lines+markers', line=dict(color='#f87171', width=2)),
        row=2, col=1, secondary_y=True)
    _ci_band(fig, x, daily['drift_mean'], daily['drift_lo'], daily['drift_hi'],
             'theo drift while resting', 'rgba(250,204,21,1.0)', row=3)
    fig.add_hline(y=0, line=dict(color='#888', width=1, dash='dot'), row=3, col=1)
    fig.update_yaxes(title_text='¢/fill', row=1, col=1)
    fig.update_yaxes(title_text='# fills', row=2, col=1)
    fig.update_yaxes(title_text='$ cost', row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text='¢/fill', row=3, col=1)
    fig.update_xaxes(title_text='day (CT)', row=3, col=1)
    fig.update_layout(
        template='plotly_dark', height=950, hovermode='x unified',
        margin=dict(l=80, r=80, t=80, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1))
    return fig


def adv_stats(daily):
    if daily.empty:
        return html.Div("No fills yet.", style={"color": "#888"})
    total_fills = int(daily['n_fills'].sum())
    total_pnl   = float(daily['pnl_$'].sum())
    total_late  = int(daily['late_n'].sum())
    total_late_cost = float(daily['late_$'].sum())
    today       = daily.iloc[-1]

    def row(label, value, color="#e0e0e0"):
        return html.Div([
            html.Span(label, style={"color": "#888", "fontSize": "12px"}),
            html.Div(value, style={"color": color, "fontSize": "18px",
                                    "fontWeight": "bold"}),
        ], style={"marginBottom": "16px"})

    pnl_color = "#22c55e" if total_pnl >= 0 else "#dc2626"
    today_color = "#22c55e" if today['pnl_$'] >= 0 else "#dc2626"
    return html.Div([
        html.H3("Window", style={"color": "#facc15", "marginBottom": "12px"}),
        row("days", f"{len(daily)}"),
        row("total fills", f"{total_fills:,}"),
        row("total P&L", f"${total_pnl:+.2f}", pnl_color),
        row("late-cancel n", f"{total_late:,}"),
        row("late-cancel $", f"${total_late_cost:+.2f}",
            "#dc2626" if total_late_cost < 0 else "#888"),
        html.Hr(style={"borderColor": "#1e2736"}),
        html.H3("Today", style={"color": "#facc15", "marginBottom": "12px"}),
        row("date", today['day'].isoformat()),
        row("n fills", f"{int(today['n_fills']):,}"),
        row("P&L", f"${today['pnl_$']:+.2f}", today_color),
        row("mkt@30s", f"{today['mkt30_mean']:+.2f}c"
            if pd.notna(today['mkt30_mean']) else "—"),
        row("theo drift", f"{today['drift_mean']:+.2f}c"
            if pd.notna(today['drift_mean']) else "—"),
        row("late-cancel", f"{int(today['late_n'])} fills · "
                            f"${today['late_$']:+.2f}"),
    ], style={"fontFamily": "monospace"})


# =============================================================================
# Bleed-decomposition tab — conditional cross-section view
# =============================================================================
DRIFT_BINS   = [-100, -3, -1, -0.5, 0.5, 1, 3, 100]
DRIFT_LABELS = ['<<-3 (huge adverse)', '-3 to -1', '-1 to -0.5',
                'within ±0.5', '+0.5 to +1', '+1 to +3', '>+3 (huge favor)']


def reconstruct_breach_duration(adv: pd.DataFrame,
                                  theo: pd.DataFrame) -> pd.Series:
    """For each fill, walk back through theo_state for the ticker and find
    the most recent moment where |theo - fill_price| <= TOLERANCE_C.
    Returns seconds between that moment and the fill.  If the quote was
    out-of-tolerance for its whole life, returns (fill_ts - placed_ts).
    """
    out = pd.Series(np.nan, index=adv.index, dtype=float)
    theo_by = {t: g.sort_values('ts').reset_index(drop=True)
               for t, g in theo[['ts', 'ticker', 'theo']].groupby('ticker')}
    tol_dollars = TOLERANCE_C / 100.0
    for idx, row in adv.iterrows():
        g = theo_by.get(row['ticker'])
        if g is None or g.empty or pd.isna(row['placed_ts']):
            continue
        ts_arr = g['ts'].to_numpy()
        theo_arr = g['theo'].to_numpy()
        lo = np.searchsorted(ts_arr, row['placed_ts'], side='left')
        hi = np.searchsorted(ts_arr, row['ts'], side='right')
        if hi <= lo:
            continue
        in_tol = np.abs(theo_arr[lo:hi] - row['price']) <= tol_dollars
        if not in_tol.any():
            # Quote was breached from birth — duration = full quote life
            out.loc[idx] = (row['ts'] - row['placed_ts']).total_seconds()
            continue
        # Index of last in-tolerance moment (within the [lo:hi] slice)
        last_in_tol_offset = np.where(in_tol)[0][-1]
        last_in_tol_ts = ts_arr[lo + last_in_tol_offset]
        out.loc[idx] = (row['ts'] - pd.Timestamp(last_in_tol_ts)
                        ).total_seconds()
    return out


def bleed_compute(adv: pd.DataFrame) -> tuple:
    """Returns (cohort_df, bucket_df) for the bleed-decomposition tab."""
    m = adv.dropna(subset=['theo_drift_c', 'mkt_mid_30',
                            'mid_drift_30', 'realized_c']).copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Cohort split: theo against (drift<0) vs stable/favor (drift>=0).
    m['cohort'] = np.where(m['theo_drift_c'] < 0,
                            'Theo against quote', 'Theo stable/favor')
    cohort = m.groupby('cohort').agg(
        n=('theo_drift_c', 'size'),
        mkt_mid_30=('mkt_mid_30', 'mean'),
        mid_drift_30=('mid_drift_30', 'mean'),
        realized_c=('realized_c', 'mean'),
        drift_c=('theo_drift_c', 'mean'),
    ).reset_index()
    cohort['pct'] = cohort['n'] / len(m) * 100

    # Magnitude buckets.
    m['drift_bin'] = pd.cut(m['theo_drift_c'],
                             bins=DRIFT_BINS, labels=DRIFT_LABELS)
    bucket = m.groupby('drift_bin', observed=True).agg(
        n=('theo_drift_c', 'size'),
        drift_c=('theo_drift_c', 'mean'),
        mkt_mid_30=('mkt_mid_30', 'mean'),
        mid_drift_30=('mid_drift_30', 'mean'),
        realized_c=('realized_c', 'mean'),
    ).reset_index()
    bucket['total_$'] = bucket['n'] * bucket['realized_c'] / 100
    return cohort, bucket


def bleed_figure(cohort, bucket, breach):
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Cohort split — markouts and realized P&L (¢/fill)',
            'By |theo drift| bucket — ¢/fill and total $ bleed',
            f'Tolerance-breach reconstruction — seconds quote was '
            f'out-of-tol (>{TOLERANCE_C:.1f}¢) before fill'),
        row_heights=[0.30, 0.35, 0.35], vertical_spacing=0.10,
        specs=[[{}], [{'secondary_y': True}], [{'secondary_y': True}]])

    if cohort.empty or bucket.empty:
        fig.update_layout(template='plotly_dark', height=1100)
        return fig

    # --- Row 1: cohort split grouped bars ---
    metrics = [('mkt_mid_30',  'mkt@30s vs fill', '#22c55e'),
               ('mid_drift_30', 'mid drift (vs mid)', '#facc15'),
               ('realized_c',   'realized @ settle', '#a78bfa')]
    for col, name, color in metrics:
        fig.add_trace(go.Bar(
            x=cohort['cohort'],
            y=cohort[col],
            name=name, marker_color=color,
            text=[f"{v:+.2f}c" for v in cohort[col]],
            textposition='outside',
        ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'), row=1, col=1)

    # --- Row 2: drift buckets — realized $ + per-fill metrics ---
    bar_colors = ['#dc2626' if v < 0 else '#22c55e' for v in bucket['realized_c']]
    fig.add_trace(go.Bar(
        x=bucket['drift_bin'].astype(str),
        y=bucket['total_$'],
        name='total $ realized', marker_color=bar_colors,
        text=[f"${v:+.0f} (n={n})" for v, n in zip(bucket['total_$'], bucket['n'])],
        textposition='outside',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bucket['drift_bin'].astype(str),
        y=bucket['mid_drift_30'],
        mode='lines+markers', name='mid drift (¢/fill)',
        line=dict(color='#facc15', width=2),
        marker=dict(size=10),
    ), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=bucket['drift_bin'].astype(str),
        y=bucket['realized_c'],
        mode='lines+markers', name='realized ¢/fill',
        line=dict(color='#a78bfa', dash='dash', width=2),
        marker=dict(size=10),
    ), row=2, col=1, secondary_y=True)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'),
                  row=2, col=1, secondary_y=True)

    # --- Row 3: breach-duration buckets ---
    if breach is not None and not breach.empty:
        breach_bins = [-0.01, 0.5, 2, 5, 30, 120, 99999]
        breach_labels = ['<0.5s (no breach)', '0.5-2s', '2-5s',
                         '5-30s', '30-120s', '>120s']
        b = breach.copy()
        b['bucket'] = pd.cut(b['breach_duration_s'],
                              bins=breach_bins, labels=breach_labels)
        agg = b.groupby('bucket', observed=True).agg(
            n=('breach_duration_s', 'size'),
            realized=('realized_c', 'mean'),
            mid_drift=('mid_drift_30', 'mean'),
            total=('realized_c', lambda x: x.sum() / 100)).reset_index()
        bar_colors = ['#dc2626' if v < 0 else '#22c55e' for v in agg['total']]
        fig.add_trace(go.Bar(
            x=agg['bucket'].astype(str),
            y=agg['total'],
            name='total $ realized', marker_color=bar_colors,
            text=[f"${v:+.0f}<br>(n={n})" for v, n in zip(agg['total'], agg['n'])],
            textposition='outside', showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=agg['bucket'].astype(str), y=agg['realized'],
            mode='lines+markers', name='realized ¢/fill',
            line=dict(color='#a78bfa', dash='dash', width=2),
            marker=dict(size=10), showlegend=False,
        ), row=3, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(
            x=agg['bucket'].astype(str), y=agg['mid_drift'],
            mode='lines+markers', name='mid drift ¢/fill',
            line=dict(color='#facc15', width=2),
            marker=dict(size=10), showlegend=False,
        ), row=3, col=1, secondary_y=True)
        fig.add_hline(y=0, line=dict(color='#666', dash='dot'),
                      row=3, col=1, secondary_y=True)

    fig.update_yaxes(title_text='¢/fill', row=1, col=1)
    fig.update_yaxes(title_text='total $ (window)', row=2, col=1)
    fig.update_yaxes(title_text='¢/fill', row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text='theo drift bucket (cents)', row=2, col=1)
    fig.update_yaxes(title_text='total $ (window)', row=3, col=1)
    fig.update_yaxes(title_text='¢/fill', row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text='breach-duration bucket', row=3, col=1)
    fig.update_layout(
        template='plotly_dark', height=1300, barmode='group',
        margin=dict(l=80, r=80, t=80, b=60),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1))
    return fig


def bleed_stats(cohort, bucket):
    if cohort.empty or bucket.empty:
        return html.Div("No fills yet.", style={"color": "#888"})

    def row(label, value, color="#e0e0e0"):
        return html.Div([
            html.Span(label, style={"color": "#888", "fontSize": "12px"}),
            html.Div(value, style={"color": color, "fontSize": "16px",
                                    "fontWeight": "bold"}),
        ], style={"marginBottom": "14px"})

    adverse = cohort[cohort['cohort'] == 'Theo against quote'].iloc[0]
    favor   = cohort[cohort['cohort'] == 'Theo stable/favor'].iloc[0]
    worst_b = bucket.iloc[bucket['total_$'].idxmin()]

    return html.Div([
        html.H3("Cohorts", style={"color": "#facc15", "marginBottom": "12px"}),
        row("adverse fills",
            f"{int(adverse['n']):,} ({adverse['pct']:.0f}%)",
            "#dc2626"),
        row("adverse realized",
            f"{adverse['realized_c']:+.2f}c/fill", "#dc2626"),
        row("favor fills",
            f"{int(favor['n']):,} ({favor['pct']:.0f}%)",
            "#22c55e"),
        row("favor realized",
            f"{favor['realized_c']:+.2f}c/fill", "#22c55e"),
        html.Hr(style={"borderColor": "#1e2736"}),
        html.H3("Worst bucket", style={"color": "#facc15", "marginBottom": "12px"}),
        row("bucket", str(worst_b['drift_bin'])),
        row("n_fills", f"{int(worst_b['n']):,}"),
        row("realized/fill",
            f"{worst_b['realized_c']:+.2f}c", "#dc2626"),
        row("total $ bleed",
            f"${worst_b['total_$']:+.2f}", "#dc2626"),
        html.Hr(style={"borderColor": "#1e2736"}),
        html.Div(
            "Cohort = sign of theo movement while quote was resting.  "
            "Buckets = magnitude.  Mid-drift is pure adverse selection; "
            "realized is per-fill P&L vs settlement.",
            style={"color": "#666", "fontSize": "11px",
                   "lineHeight": "1.4", "marginTop": "10px"})
    ], style={"fontFamily": "monospace"})


# =============================================================================
# Single compute, both tabs
# =============================================================================
def _bleed_views(adv: pd.DataFrame) -> tuple:
    """Given a (possibly date-filtered) adv frame, returns
    (cohort, bucket, breach_df) for the bleed tab."""
    if adv.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    cohort, bucket = bleed_compute(adv)
    breach = adv[['ts', 'ticker', 'breach_duration_s', 'theo_drift_c',
                   'mid_drift_30', 'realized_c']].dropna(
                   subset=['breach_duration_s'])
    return cohort, bucket, breach


def compute_all():
    """PnL-only mode — loads `fills` across all days and skips the heavy
    `kalshi_book`/`theo_state`/`order_events` tables to keep memory bounded."""
    global _SETTLE_CACHE
    tables = load_readonly(('fills',))
    fills = tables['fills']

    if fills.empty:
        return pd.DataFrame()

    _SETTLE_CACHE = fetch_settlements_from_api(
        list(fills['ticker'].unique()), api, cache_path=CACHE_PATH)

    return compute_pnl(fills)


# =============================================================================
# Dash app
# =============================================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Aston Live Dashboard"

TAB_STYLE = {"backgroundColor": "#0a0e1a", "color": "#888",
             "border": "1px solid #1e2736", "padding": "10px"}
TAB_SELECTED_STYLE = {"backgroundColor": "#1a212e", "color": "#facc15",
                       "border": "1px solid #facc15", "padding": "10px",
                       "fontWeight": "bold"}

app.layout = html.Div([
    html.Div([
        html.H1("Aston Live Dashboard", style={"margin": 0}),
        html.Div(f"{SERIES_PREFIX} · refresh every {REFRESH_SECONDS}s",
                 style={"color": "#888"}),
    ], style={"padding": "10px 20px"}),
    html.Div(id='tab-content', style={"padding": "10px 0"}),
    dcc.Interval(id='interval',
                 interval=REFRESH_SECONDS * 1000, n_intervals=0),
    dcc.Store(id='store'),
], style={"backgroundColor": "#0a0e1a", "color": "#e0e0e0",
          "minHeight": "100vh", "fontFamily": "Arial"})


@app.callback(Output('store', 'data'), Input('interval', 'n_intervals'))
def refresh_store(_):
    global _PNL_FILLS
    _PNL_FILLS = compute_all()
    return datetime.now(timezone.utc).isoformat()


_PNL_FILLS: pd.DataFrame = pd.DataFrame()


@app.callback(
    Output('tab-content', 'children'),
    Input('store', 'data'),
)
def render_tab(_store_ts):
    return html.Div([
        html.Div(pnl_stats(_PNL_FILLS),
                 style={"width": "300px", "padding": "20px"}),
        html.Div([dcc.Graph(figure=pnl_figure(_PNL_FILLS))],
                 style={"flex": 1}),
    ], style={"display": "flex"})


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT} — refresh every {REFRESH_SECONDS}s")
    app.run(debug=False, port=PORT, host="127.0.0.1")
