"""Live P&L dashboard — browser-based, auto-refreshes from recorder DBs.

Run:    python3 pnl_dashboard.py
Open:   http://localhost:8050

Refreshes every 60s.  As recorder.py writes new fills + settlements
arrive from Kalshi, the graph updates.  Independent of Aston — runs as
a separate process, reads from the same SQLite files.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utility import (
    fetch_settlements_from_api,
    list_eligible_dbs,
    parse_day_suffix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "Aston"))
from kalshi_api import KalshiAPI

SERIES_PREFIX     = "KXETH15M"
CUTOFF_DAY        = "26MAY15"
CUTOFF_DATE       = parse_day_suffix(CUTOFF_DAY)
REFRESH_SECONDS   = 60
CACHE_PATH        = (Path(__file__).resolve().parent.parent
                     / ".settlements_cache.json")

api = KalshiAPI()


# =============================================================================
# Data refresh — called every interval
# =============================================================================
def _read_fills_readonly() -> pd.DataFrame:
    """Read fills from every eligible DB in SQLite read-only mode.
    Avoids WAL/lock contention with the live recorder writer."""
    files = list_eligible_dbs(SERIES_PREFIX, CUTOFF_DAY)
    parts = []
    for path in files:
        # Try mode=ro first (sees WAL writes); fall back to immutable=1
        # (ignores WAL — slightly stale, but never blocks).
        for uri in (f"file:{path}?mode=ro",
                     f"file:{path}?immutable=1"):
            try:
                conn = sqlite3.connect(uri, uri=True, timeout=5)
                df = pd.read_sql("SELECT * FROM fills ORDER BY ts", conn)
                conn.close()
                parts.append(df)
                print(f"   {path.name} ({len(df)})")
                break
            except Exception:
                continue
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out['ts'] = pd.to_datetime(out['ts'], utc=True, format='ISO8601')
    return out


def compute_pnl() -> pd.DataFrame:
    """Reload all fills + settlements, return per-fill realized P&L frame."""
    fills = _read_fills_readonly()
    if fills.empty:
        return pd.DataFrame()
    if 'side' in fills.columns:
        fills = fills[fills['side'] == 'yes'].copy()
    settlements = fetch_settlements_from_api(
        list(fills['ticker'].unique()), api,
        cache_path=CACHE_PATH,
    )
    fills['outcome'] = fills['ticker'].map(settlements)
    fills['sgn']     = np.where(fills['action'] == 'buy', +1, -1)
    # Multiply by count — ~20 fills are partial-fills (count < 1) where
    # a 1-lot order matched against multiple smaller bids/asks; treating
    # each row as 1-lot would double-count.
    fills['realized_c'] = (fills['outcome'] - fills['price']) * fills['sgn'] * fills['count'] * 100
    fills = fills.dropna(subset=['realized_c']).copy()
    # Bucket by the contract's UTC expiry date (extracted from ticker)
    # so dates match pnl.py and the data isn't pulled into a phantom
    # previous CT day from early-UTC-morning fills.
    fills['date'] = pd.to_datetime(
        fills['ticker'].str.split('-').str[1].str[:7],
        format='%y%b%d', errors='coerce',
    ).dt.date
    fills = fills.dropna(subset=['date'])
    # Exclude tickers from before the cutoff (e.g., late WS fills for
    # markets that expired before the validation window).  Matches pnl.py.
    fills = fills[fills['date'] >= CUTOFF_DATE]
    return fills.sort_values('ts').reset_index(drop=True)


def build_stats(fills: pd.DataFrame):
    """Right-side stats panel."""
    if fills.empty:
        return html.Div("No fills yet.", style={"color": "#888"})

    daily = fills.groupby('date')['realized_c'].sum() / 100
    cumulative = daily.cumsum()
    total = float(cumulative.iloc[-1])
    n_days = len(daily)
    mean = float(daily.mean())
    std  = float(daily.std(ddof=1)) if n_days > 1 else 0.0
    sharpe = (mean / std * (365 ** 0.5)) if std > 0 else float('nan')
    # t-stat: mean_daily / SE(mean_daily), where SE = std / sqrt(n)
    tstat = (mean / (std / (n_days ** 0.5))) if (std > 0 and n_days > 1) else float('nan')

    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
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
            html.Div([html.B("Days traded: "),
                      html.Span(f"{n_days}")]),
            html.Div([html.B("Total fills: "),
                      html.Span(f"{len(fills):,}")]),
        ]),
        html.Hr(),
        html.Div([
            html.Div([html.B("Mean daily: "),
                      html.Span(f"${mean:+.2f}/day")]),
            html.Div([html.B("Daily SD: "),
                      html.Span(f"${std:.2f}")]),
            html.Div([html.B("Sharpe (annualized): "),
                      html.Span(f"{sharpe:+.2f}")]),
            html.Div([html.B("t-stat (daily PnL > 0): "),
                      html.Span(
                          f"{tstat:+.2f}",
                          style={"color": ("#22c55e" if tstat > 2
                                            else "#facc15" if tstat > 1
                                            else "#dc2626")}
                          if tstat == tstat else {},
                      ),
                      html.Span(f"  (n={n_days} days)",
                                style={"color": "#888"})]),
            html.Div([html.B("Max drawdown: "),
                      html.Span(f"${max_dd:.2f}")]),
        ]),
        html.Hr(),
        html.Div(f"Refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
                 style={"color": "#666", "fontSize": "0.8em"}),
    ], style={"fontFamily": "monospace", "fontSize": "1.1em",
              "lineHeight": "1.7"})


def build_figure(fills: pd.DataFrame):
    if fills.empty:
        return go.Figure().update_layout(
            template='plotly_dark', height=700,
            annotations=[dict(text="no fills yet",
                              x=0.5, y=0.5, showarrow=False)])

    daily = fills.groupby('date')['realized_c'].sum() / 100
    cumulative = daily.cumsum()
    fills_per_day = fills.groupby('date').size()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Cumulative realized P&L ($)",
            "Daily P&L ($)",
            "Fills per day",
        ),
        row_heights=[0.5, 0.3, 0.2],
        shared_xaxes=True, vertical_spacing=0.06,
    )

    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values,
        mode='lines+markers', line=dict(color='#22c55e', width=3),
        marker=dict(size=8), name='Cumulative',
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'), row=1, col=1)

    bar_colors = ['#22c55e' if v >= 0 else '#dc2626' for v in daily.values]
    fig.add_trace(go.Bar(
        x=daily.index, y=daily.values, marker_color=bar_colors,
        text=[f"${v:+.2f}" for v in daily.values],
        textposition='outside', name='Daily',
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'), row=2, col=1)

    fig.add_trace(go.Bar(
        x=fills_per_day.index, y=fills_per_day.values,
        marker_color='#a78bfa', name='Fill count',
    ), row=3, col=1)

    fig.update_yaxes(title_text='$', row=1, col=1)
    fig.update_yaxes(title_text='$', row=2, col=1)
    fig.update_yaxes(title_text='fills', row=3, col=1)
    fig.update_layout(template='plotly_dark', height=850, showlegend=False)
    return fig


# =============================================================================
# Dash app
# =============================================================================
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1("Aston Live P&L", style={"margin": 0}),
        html.Div(f"{SERIES_PREFIX} · refresh every {REFRESH_SECONDS}s",
                 style={"color": "#888"}),
    ], style={"padding": "10px 20px"}),
    html.Div([
        html.Div(id='stats', style={"width": "300px", "padding": "20px"}),
        html.Div([dcc.Graph(id='pnl-graph')],
                 style={"flex": 1}),
    ], style={"display": "flex"}),
    dcc.Interval(id='interval', interval=REFRESH_SECONDS * 1000, n_intervals=0),
], style={"backgroundColor": "#0a0e1a", "color": "#e0e0e0",
          "minHeight": "100vh", "fontFamily": "Arial"})


@app.callback(
    Output('stats',     'children'),
    Output('pnl-graph', 'figure'),
    Input('interval',   'n_intervals'),
)
def refresh(_):
    fills = compute_pnl()
    return build_stats(fills), build_figure(fills)


if __name__ == "__main__":
    print(f"Open http://localhost:8050 — refresh every {REFRESH_SECONDS}s")
    app.run(debug=False, port=8050, host="127.0.0.1")
