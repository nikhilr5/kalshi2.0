"""Aston Live Dashboard — browser, two tabs, auto-refresh.

  Tab 1 — P&L:         cumulative + daily + fill count
  Tab 2 — Moneyness:   realized fill P&L per side, bucketed by entry
                       price (OTM -> ITM)

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
    CF_INDEX_CUTOVER, bootstrap_ci, day_range, day_to_suffix,
    fetch_settlements_from_api, list_eligible_dbs, load_book, load_daily_spot,
    load_fills, parse_day_suffix,
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
    # Current-config window t-stat — measured from the CF-index switch (the
    # latest change).  The all-days t-stat above blends every config.
    pc = (fills[fills['date'] >= CF_CHANGE_D]
          if 'date' in fills.columns else fills.iloc[0:0])
    if not pc.empty:
        pc_daily = pc.groupby('date')['realized_c'].sum() / 100
        pc_n = len(pc_daily)
        pc_mean = float(pc_daily.mean())
        pc_std = float(pc_daily.std(ddof=1)) if pc_n > 1 else 0.0
        pc_t = ((pc_mean / (pc_std / (pc_n ** 0.5)))
                if (pc_std > 0 and pc_n > 1) else float('nan'))
        pc_total = float(pc_daily.sum())
    else:
        pc_n, pc_mean, pc_t, pc_total = 0, 0.0, float('nan'), 0.0
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
            html.Div([html.B("t-stat (all days): "),
                      html.Span(f"{tstat:+.2f}",
                                style={"color": ("#22c55e" if tstat > 2
                                                  else "#facc15" if tstat > 1
                                                  else "#dc2626")}
                                if tstat == tstat else {}),
                      html.Span(f"  (n={n_days} days)",
                                style={"color": "#888"})]),
            html.Div([html.B(f"t-stat ({CF_CHANGE_LABEL}): "),
                      html.Span(f"{pc_t:+.2f}",
                                style={"color": ("#22c55e" if pc_t > 2
                                                  else "#facc15" if pc_t > 1
                                                  else "#dc2626")}
                                if pc_t == pc_t else {"color": "#888"}),
                      html.Span(f"  (${pc_total:+.0f}, n={pc_n} days)",
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
# old config: v1 engine, 1-lot, cap 8-10.  Moneyness + Markout tabs scope
# to this window (current config); P&L tab marks it with the yellow line.
CONFIG_CHANGE_DATE = "2026-06-09"          # string for the P&L vline
CONFIG_CHANGE_DAY  = "26JUN09"             # suffix for day_range
CONFIG_CHANGE_D    = parse_day_suffix(CONFIG_CHANGE_DAY)   # date for filtering
CONFIG_CHANGE_LABEL = "v2 + 3-lot"

# 2026-06-15: theo switched from Coinbase spot to the CF Benchmarks index
# (ETHUSD_RTI) — the reference Kalshi actually settles on.  Second boundary;
# the post-change t-stat is measured from here (the current config).
# CF-index cutover — single source of truth is utility.CF_INDEX_CUTOVER
# (a UTC instant; the live app's Coinbase->CF restart).  Derive the
# day-granularity date used by the dashboard's filters and vline from it
# so the analysis boundary and the visual marker can never disagree.
CF_CHANGE_D     = CF_INDEX_CUTOVER.date()
CF_CHANGE_DAY   = day_to_suffix(CF_CHANGE_D)
CF_CHANGE_DATE  = CF_CHANGE_D.isoformat()
CF_CHANGE_LABEL = "CF index"

# Sell-only longshot-fade config: quote the ask only when the price sits in
# ~0.10-0.35, fading retail's overpriced cheap YES.  Moneyness + Markout tabs
# scope to fills on/after this point; the P&L tab marks it with the green line.
# UPDATE the date to the actual restart instant when you flip the live app
# into the longshot config (it's a placeholder until then).
# Single source of truth: the exact UTC INSTANT you flipped the live app into
# the longshot config.  Moneyness + Markout tabs filter fills on `ts >= this`
# (precise time, not whole days), so today's pre-flip fills are excluded.
# SET THIS to the real flip time.
LONGSHOT_CHANGE_TS    = datetime(2026, 6, 17, 16, 0, tzinfo=timezone.utc)  # placeholder
LONGSHOT_CHANGE_D     = LONGSHOT_CHANGE_TS.date()
LONGSHOT_CHANGE_DAY   = day_to_suffix(LONGSHOT_CHANGE_D)
LONGSHOT_CHANGE_DATE  = LONGSHOT_CHANGE_D.isoformat()
LONGSHOT_CHANGE_LABEL = "Longshot fade · sell 0.10–0.35"


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
        subplot_titles=("Cumulative realized P&L ($)  ·  ETH spot overlay",
                         "Daily P&L ($)", "Fills per day"),
        row_heights=[0.5, 0.3, 0.2], shared_xaxes=True, vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{}], [{}]])
    # ETH spot (faint, secondary axis) — drawn first so P&L sits on top.
    spot = load_daily_spot(cumulative.index.min(), until=cumulative.index.max())
    if not spot.empty:
        fig.add_trace(go.Scatter(
            x=spot['date'], y=spot['spot'], mode='lines', name='ETH spot',
            line=dict(color='#94a3b8', width=1.5), opacity=0.35,
            hovertemplate='ETH %{y:.0f}<extra></extra>'),
            row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values, mode='lines+markers',
        line=dict(color='#22c55e', width=3), marker=dict(size=8)),
        row=1, col=1, secondary_y=False)
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
        fig.add_vline(x=CF_CHANGE_DATE, line=dict(color='#38bdf8', width=2),
                      row=row, col=1)
        fig.add_vline(x=LONGSHOT_CHANGE_DATE, line=dict(color='#4ade80', width=2),
                      row=row, col=1)
    fig.add_annotation(x=CONFIG_CHANGE_DATE, y=1, yref='paper',
                       text=CONFIG_CHANGE_LABEL, showarrow=False,
                       font=dict(color='#facc15', size=11),
                       xanchor='left', yanchor='top')
    fig.add_annotation(x=CF_CHANGE_DATE, y=1, yref='paper',
                       text=CF_CHANGE_LABEL, showarrow=False,
                       font=dict(color='#38bdf8', size=11),
                       xanchor='left', yanchor='top')
    fig.add_annotation(x=LONGSHOT_CHANGE_DATE, y=1, yref='paper',
                       text=LONGSHOT_CHANGE_LABEL, showarrow=False,
                       font=dict(color='#4ade80', size=11),
                       xanchor='left', yanchor='top')
    fig.update_yaxes(title_text='$', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='ETH spot', row=1, col=1, secondary_y=True,
                     showgrid=False, color='#94a3b8')
    fig.update_yaxes(title_text='$', row=2, col=1)
    fig.update_yaxes(title_text='fills', row=3, col=1)
    fig.update_layout(template='plotly_dark', height=850, showlegend=False)
    return fig


# =============================================================================
# Moneyness tab — fill P&L by side, bucketed by entry price (OTM -> ITM)
# =============================================================================
# 0.05-wide buckets across the longshot band (0.10-0.35); fills outside
# the band fall to NaN and drop out (the strategy only quotes here).
PX_EDGES  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
PX_LABELS = ['10-15', '15-20', '20-25', '25-30', '30-35']
SIDE_COLOR = {'buy': '#22c55e', 'sell': '#ef4444'}   # match app LIVE ORDERS


def _post_change(fills: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the current-config window (>= CONFIG_CHANGE_D)."""
    if fills.empty or 'date' not in fills.columns:
        return fills
    return fills[fills['date'] >= CONFIG_CHANGE_D]


def _post_longshot(fills: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the post-longshot-fade window (ts >= LONGSHOT_CHANGE_TS).
    Filters on the precise flip INSTANT, not whole days, so this morning's
    pre-flip fills are excluded.  Moneyness and markouts scope here so they
    reflect only the sell-only 0.10-0.35 config."""
    if fills.empty or 'ts' not in fills.columns:
        return fills
    return fills[fills['ts'] >= LONGSHOT_CHANGE_TS]


def moneyness_breakdown(fills: pd.DataFrame) -> pd.DataFrame:
    """Per (price-bucket, side): mean c/fill, total $, fill count.  The YES
    fill price is the moneyness axis — low=OTM, high=ITM.  Scoped to the
    post-CF-index window so theo (the moneyness reference) is the correct
    CF index, not stale Coinbase-spot theo."""
    fills = _post_longshot(fills)
    if fills.empty:
        return pd.DataFrame()
    f = fills.copy()
    f['px_bin'] = pd.cut(f['price'], bins=PX_EDGES, labels=PX_LABELS,
                         include_lowest=True)
    g = (f.groupby(['px_bin', 'action'], observed=True)
           .agg(realized_sum=('realized_c', 'sum'),
                n=('count', 'sum'))                  # n = total CONTRACTS, not fills
           .reset_index())
    g['cpf'] = g['realized_sum'] / g['n']            # cents per CONTRACT (size-weighted)
    g['total'] = g['realized_sum'] / 100             # total $
    return g[['px_bin', 'action', 'cpf', 'total', 'n']]


def moneyness_figure(fills: pd.DataFrame):
    g = moneyness_breakdown(fills)
    if g.empty:
        return go.Figure().update_layout(
            template='plotly_dark', height=700,
            annotations=[dict(text="no fills yet", x=0.5, y=0.5,
                              showarrow=False)])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Realized edge per contract (c) by entry price & side",
                        "Contract count (thin buckets are noisy)"))
    for side in ('buy', 'sell'):
        s = g[g['action'] == side]
        fig.add_trace(go.Bar(
            x=s['px_bin'], y=s['cpf'], name=side.upper(),
            marker_color=SIDE_COLOR[side],
            text=[f"{v:+.1f}" for v in s['cpf']], textposition='outside'),
            row=1, col=1)
        fig.add_trace(go.Bar(
            x=s['px_bin'], y=s['n'], marker_color=SIDE_COLOR[side],
            showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'), row=1, col=1)
    fig.update_yaxes(title_text='c/contract', row=1, col=1)
    fig.update_yaxes(title_text='contracts', row=2, col=1)
    fig.update_xaxes(title_text='YES fill price (c) — OTM → ITM', row=2, col=1)
    cf = _post_longshot(fills)
    n = int(g['n'].sum())
    sub = _cf_subtitle(n, cf['date'].max() if not cf.empty else None)
    fig.update_layout(template='plotly_dark', height=770, barmode='group',
                      margin=dict(t=90), legend=dict(orientation='h', y=1.06),
                      title=dict(text=sub, x=0.5, y=0.98, font=dict(size=13,
                                 color='#38bdf8')))
    return fig


def _window_label(fills: pd.DataFrame) -> str:
    f = _post_change(fills)
    if f.empty:
        return f"{CONFIG_CHANGE_DATE} → (no fills yet)"
    return (f"{CONFIG_CHANGE_DATE} → {f['date'].max()}  "
            f"({CONFIG_CHANGE_LABEL})")


def _cf_subtitle(n: int, last_date=None) -> str:
    """Scope line for the moneyness/markout tabs (post-longshot-fade window).
    n is the fill count in the filtered frame the caller already built."""
    if not n:
        return f"longshot fade ({LONGSHOT_CHANGE_DATE}): no fills yet"
    return f"longshot fade: n={n:,} fills, {LONGSHOT_CHANGE_DATE} → {last_date}"


def moneyness_panel(fills: pd.DataFrame):
    g = moneyness_breakdown(fills)
    if g.empty:
        return html.Div("No fills yet in window.", style={"color": "#888"})

    cf = _post_longshot(fills)
    window = html.Div(
        _cf_subtitle(int(g['n'].sum()),
                     cf['date'].max() if not cf.empty else None),
        style={"color": "#38bdf8", "marginBottom": "10px",
               "fontSize": "0.85em"})
    summary = []
    for side in ('buy', 'sell'):
        s = g[g['action'] == side]
        tot = float(s['total'].sum())
        n = int(s['n'].sum())
        cpf = (tot * 100 / n) if n else 0.0
        summary.append(html.Div([
            html.B(side.upper(), style={"color": SIDE_COLOR[side]}),
            html.Span(f"  ${tot:+.2f}  ·  {cpf:+.2f}c/contract  ·  {n} contracts"),
        ]))

    header = html.Tr([html.Th(h, style={"textAlign": "right",
                                        "padding": "2px 8px"})
                      for h in ("Price", "Side", "c/contract", "Total $", "contracts")])
    rows = [header]
    for _, r in g.iterrows():
        color = SIDE_COLOR[r['action']]
        cells = [r['px_bin'], r['action'].upper(),
                 f"{r['cpf']:+.2f}", f"${r['total']:+.2f}", int(r['n'])]
        rows.append(html.Tr([
            html.Td(c, style={"textAlign": "right", "padding": "2px 8px",
                              "color": color if i == 1 else "#e0e0e0"})
            for i, c in enumerate(cells)]))

    return html.Div([
        window,
        html.H3("By side", style={"marginBottom": "6px"}),
        *summary,
        html.Hr(),
        html.Table(rows, style={"borderCollapse": "collapse"}),
        html.Div("Bucket = YES fill price; low = OTM, high = ITM. "
                 "c/fill is realized edge net of nothing (maker, ~0 fee).",
                 style={"color": "#666", "fontSize": "0.8em",
                        "marginTop": "10px", "maxWidth": "340px"}),
    ], style={"fontFamily": "monospace", "fontSize": "0.95em",
              "lineHeight": "1.6"})


# =============================================================================
# Markout tab — short-horizon adverse selection, built lean:
#   * post-change window only (a few days, not the full ~4 weeks)
#   * entry mid comes FROM the fills table (book@fill already recorded),
#     so only the *later* mid needs book
#   * book loaded one day at a time, reduced to a few floats per fill
#   * computed lazily (only when the tab is viewed) + cached per refresh
# =============================================================================
MARKOUT_HORIZONS = (30, 60, 120)


def _markouts_for_day(day: str) -> pd.DataFrame:
    fills = load_fills(day, series_prefix=SERIES_PREFIX)
    if fills.empty:
        return pd.DataFrame()
    fills = fills[fills['side'] == 'yes'].copy()
    fills = fills[(fills['kalshi_yes_bid'] > 0) & (fills['kalshi_yes_ask'] > 0)]
    if fills.empty:
        return pd.DataFrame()
    book = load_book(day, series_prefix=SERIES_PREFIX)
    if book.empty:
        return pd.DataFrame()
    book = book[['ts', 'ticker', 'mid']].dropna().sort_values('ts')
    fills = fills.reset_index(drop=True)
    fills['fid'] = np.arange(len(fills))
    fills['entry_mid'] = (fills['kalshi_yes_bid'] + fills['kalshi_yes_ask']) / 2
    fills['sgn'] = np.where(fills['action'] == 'buy', 1, -1)
    res = fills[['fid', 'ts', 'ticker', 'action', 'sgn',
                 'entry_mid', 'price']].copy()
    for h in MARKOUT_HORIZONS:
        tgt = res[['fid', 'ticker', 'entry_mid', 'sgn']].copy()
        tgt['ts_h'] = res['ts'] + pd.Timedelta(seconds=h)
        tgt = tgt.sort_values('ts_h')
        j = pd.merge_asof(
            tgt, book.rename(columns={'ts': 'bts', 'mid': 'mid_h'}),
            left_on='ts_h', right_on='bts', by='ticker',
            direction='backward', tolerance=pd.Timedelta(seconds=30))
        j[f'mk{h}'] = j['sgn'] * (j['mid_h'] - j['entry_mid']) * 100
        res = res.merge(j[['fid', f'mk{h}']], on='fid', how='left')
    res['date'] = parse_day_suffix(day)
    return res[['date', 'ts', 'action', 'price'] + [f'mk{h}' for h in MARKOUT_HORIZONS]]


# Completed days never change, so their markouts are cached two ways:
#   in-memory (hot, within a session) + on-disk (survives dashboard restarts).
# Bump MARKOUT_VER if the markout computation ever changes — old files are
# then ignored (stale by version) and can be deleted.
MARKOUT_VER = "v2"   # per-day frame now keeps `ts` (for the precise longshot cutover)
MARKOUT_CACHE_DIR = Path(__file__).resolve().parent / "_markout_cache"
_MARKOUT_DAY_CACHE: dict = {}


def _day_cache_path(day: str) -> Path:
    return MARKOUT_CACHE_DIR / f"markout-{MARKOUT_VER}-{day}.pkl"


def compute_markouts() -> pd.DataFrame:
    # Post-longshot-fade only: scope markouts to the sell-only 0.10-0.35
    # config so they reflect that strategy.  Start the day-range at the
    # longshot cutover day.
    days = day_range(LONGSHOT_CHANGE_DAY, "today")
    today = days[-1] if days else None        # partial — never cached
    frames = []
    for day in days:
        if day in _MARKOUT_DAY_CACHE:                       # hot (in-memory)
            frames.append(_MARKOUT_DAY_CACHE[day])
            continue
        path = _day_cache_path(day)
        if day != today and path.exists():                 # warm (on-disk)
            try:
                d = pd.read_pickle(path)
                _MARKOUT_DAY_CACHE[day] = d
                frames.append(d)
                continue
            except Exception:
                pass                                        # unreadable → recompute
        d = _markouts_for_day(day)                          # cold (compute)
        if day != today and not d.empty:                   # cache completed days
            _MARKOUT_DAY_CACHE[day] = d
            try:
                MARKOUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                d.to_pickle(path)
            except Exception:
                pass                                        # cache is best-effort
        if not d.empty:
            frames.append(d)
    mk = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Drop pre-flip fills on the cutover day — scope to the exact flip instant.
    if not mk.empty and 'ts' in mk.columns:
        mk = mk[mk['ts'] >= LONGSHOT_CHANGE_TS]
    return mk


def markout_figure(mk: pd.DataFrame):
    if mk.empty:
        return go.Figure().update_layout(
            template='plotly_dark', height=600,
            annotations=[dict(text="no markout data in window", x=.5, y=.5,
                              showarrow=False)])
    xs = [f'{h}s' for h in MARKOUT_HORIZONS]
    fig = go.Figure()
    for side in ('buy', 'sell'):
        s = mk[mk['action'] == side]
        ys = [s[f'mk{h}'].mean() for h in MARKOUT_HORIZONS]
        fig.add_trace(go.Bar(x=xs, y=ys, name=side.upper(),
                             marker_color=SIDE_COLOR[side],
                             text=[f'{v:+.2f}' for v in ys],
                             textposition='outside'))
    fig.add_hline(y=0, line=dict(color='#666', dash='dot'))
    fig.update_yaxes(title_text='mean markout (c/contract)')
    fig.update_xaxes(title_text='horizon after fill')
    n = int(mk[f'mk{MARKOUT_HORIZONS[0]}'].notna().sum())
    sub = _cf_subtitle(n, mk['date'].max())
    fig.update_layout(
        template='plotly_dark', height=620, barmode='group', margin=dict(t=90),
        title=dict(text='Markout by side — positive = mid moved your way'
                        f'<br><span style="font-size:12px;color:#38bdf8">{sub}'
                        '</span>', y=0.97),
        legend=dict(orientation='h', y=1.12))
    return fig


def markout_panel(mk: pd.DataFrame):
    if mk.empty:
        return html.Div("No markout data in window.", style={"color": "#888"})
    n = int(mk[f'mk{MARKOUT_HORIZONS[0]}'].notna().sum())
    window = html.Div(
        _cf_subtitle(n, mk['date'].max()),
        style={"color": "#38bdf8", "marginBottom": "10px", "fontSize": "0.85em"})
    head = html.Tr([html.Th(h, style={"textAlign": "right", "padding": "2px 8px"})
                    for h in ('Side', '30s', '60s', '120s', 'n')])
    rows = [head]
    for side in ('buy', 'sell'):
        s = mk[mk['action'] == side]
        cells = ([side.upper()]
                 + [f"{s[f'mk{h}'].mean():+.2f}" for h in MARKOUT_HORIZONS]
                 + [int(s[f'mk{MARKOUT_HORIZONS[0]}'].notna().sum())])
        color = SIDE_COLOR[side]
        rows.append(html.Tr([
            html.Td(c, style={"textAlign": "right", "padding": "2px 8px",
                              "color": color if i == 0 else "#e0e0e0"})
            for i, c in enumerate(cells)]))
    return html.Div([
        window,
        html.H3("Markout (c/contract)", style={"marginBottom": "6px"}),
        html.Table(rows, style={"borderCollapse": "collapse"}),
        html.Div("Positive = market moved in your favor after the fill. "
                 "Entry mid is the book at fill time (from the fills row); "
                 "later mid is asof fill+horizon. Buy bleed / sell-clean is "
                 "the structural adverse-selection signature.",
                 style={"color": "#666", "fontSize": "0.8em",
                        "marginTop": "10px", "maxWidth": "340px"}),
    ], style={"fontFamily": "monospace", "fontSize": "0.95em",
              "lineHeight": "1.6"})


_MARKOUT_CACHE = {'ts': None, 'df': pd.DataFrame()}


def get_markouts(store_ts):
    """Compute markouts at most once per refresh — the heavy book load only
    runs when the Markout tab is actually viewed."""
    if _MARKOUT_CACHE['ts'] != store_ts:
        _MARKOUT_CACHE['df'] = compute_markouts()
        _MARKOUT_CACHE['ts'] = store_ts
    return _MARKOUT_CACHE['df']


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
    dcc.Tabs(id='tabs', value='pnl', children=[
        dcc.Tab(label='P&L', value='pnl',
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Moneyness', value='moneyness',
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
        dcc.Tab(label='Markout', value='markout',
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
    ]),
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
    Input('tabs', 'value'),
)
def render_tab(_store_ts, tab):
    if tab == 'moneyness':
        return html.Div([
            html.Div(moneyness_panel(_PNL_FILLS),
                     style={"width": "380px", "padding": "20px"}),
            html.Div([dcc.Graph(figure=moneyness_figure(_PNL_FILLS))],
                     style={"flex": 1}),
        ], style={"display": "flex"})
    if tab == 'markout':
        mk = get_markouts(_store_ts)
        return html.Div([
            html.Div(markout_panel(mk),
                     style={"width": "380px", "padding": "20px"}),
            html.Div([dcc.Graph(figure=markout_figure(mk))],
                     style={"flex": 1}),
        ], style={"display": "flex"})
    return html.Div([
        html.Div(pnl_stats(_PNL_FILLS),
                 style={"width": "300px", "padding": "20px"}),
        html.Div([dcc.Graph(figure=pnl_figure(_PNL_FILLS))],
                 style={"flex": 1}),
    ], style={"display": "flex"})


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT} — refresh every {REFRESH_SECONDS}s")
    app.run(debug=False, port=PORT, host="127.0.0.1")
