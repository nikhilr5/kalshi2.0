"""
Interactive Deribit Theo vs Kalshi Market — with Edge & PnL

Dash web app that plots recorded Deribit theos against Kalshi bid/ask.
Enter an edge amount to create Deribit bid (theo - edge) and Deribit ask
(theo + edge). Highlights where:
    - Deribit bid > market ask  → BUY signal  (green triangle)
    - Deribit ask <= market bid → SELL signal  (red triangle)

Computes cumulative PnL from trading every signal (1 contract each).

Usage:
    python3 graph_deribit_recorded.py
    python3 graph_deribit_recorded.py --event KXBTC-26APR1717
    python3 graph_deribit_recorded.py --dbs marketdata/market_data_2026-04-13.db marketdata/market_data_2026-04-14.db
"""

import argparse
import sqlite3
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State


# =============================================================================
# Data loading
# =============================================================================

def load_data(db_paths: list[str], event_ticker: str | None = None) -> dict:
    """Load orderbook + deribit_theo from SQLite databases."""
    all_rows = []

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(orderbook)").fetchall()]
        if "deribit_theo" not in cols:
            print(f"  {db_path}: no deribit_theo column, skipping")
            conn.close()
            continue

        has_smooth = "deribit_theo_smooth" in cols
        smooth_col = ", deribit_theo_smooth" if has_smooth else ", NULL"

        query = f"""
            SELECT timestamp, ticker, event_ticker, yes_sub_title, strike,
                   yes_bid, yes_ask, btc_price, deribit_theo{smooth_col}
            FROM orderbook
            WHERE deribit_theo IS NOT NULL
        """
        params = ()
        if event_ticker:
            query += " AND event_ticker = ?"
            params = (event_ticker,)
        query += " ORDER BY timestamp, strike"

        rows = conn.execute(query, params).fetchall()
        all_rows.extend(rows)
        print(f"  {db_path}: {len(rows):,} rows with deribit_theo")
        conn.close()

    if not all_rows:
        return {}

    all_rows.sort(key=lambda r: r[0])

    # Discover events and tickers
    events = {}
    for row in all_rows:
        ev = row[2]
        ticker = row[1]
        subtitle = row[3]
        strike = row[4]
        events.setdefault(ev, {})
        events[ev][ticker] = {"subtitle": subtitle, "strike": strike}

    # Build time series per ticker
    series_data = {}
    for row in all_rows:
        ts_str, ticker, ev, subtitle, strike, bid, ask, btc, theo = row[:9]
        theo_smooth = row[9] if len(row) > 9 else None
        ts = datetime.fromisoformat(ts_str)

        key = (ev, ticker)
        if key not in series_data:
            series_data[key] = {
                "timestamps": [], "bids": [], "asks": [],
                "theos": [], "theos_smooth": [], "btc_prices": [],
                "subtitle": subtitle,
            }

        d = series_data[key]
        d["timestamps"].append(ts)
        d["bids"].append(bid if bid > 0 else None)
        d["asks"].append(ask if ask > 0 else None)
        d["theos"].append(theo)
        d["theos_smooth"].append(theo_smooth)
        d["btc_prices"].append(btc)

    return {"events": events, "series_data": series_data}


# =============================================================================
# Chart builder
# =============================================================================

def build_figure(data: dict, event_ticker: str, ticker: str, edge: float) -> go.Figure:
    """Build the full chart for a given bracket and edge level."""
    key = (event_ticker, ticker)
    d = data["series_data"].get(key)
    if not d:
        return go.Figure()

    timestamps = d["timestamps"]
    bids = d["bids"]
    asks = d["asks"]
    theos_raw = d["theos"]
    theos_smooth = d["theos_smooth"]
    btc_prices = d["btc_prices"]
    subtitle = d["subtitle"]

    # Use smooth theo for edge/signals when available, fall back to raw
    theos = [s if s is not None else r for s, r in zip(theos_smooth, theos_raw)]

    # Deribit bid/ask = smooth_theo -/+ edge
    deribit_bids = [t - edge if t is not None else None for t in theos]
    deribit_asks = [t + edge if t is not None else None for t in theos]

    # Find buy/sell signals
    buy_ts, buy_prices = [], []       # deribit_bid > market_ask → BUY at ask
    sell_ts, sell_prices = [], []     # deribit_ask <= market_bid → SELL at bid

    # PnL tracking
    # Buy at market_ask, mark-to-market at smooth theo. PnL = theo - ask.
    # Sell at market_bid, mark-to-market at smooth theo. PnL = bid - theo.
    pnl_ts = []
    cumulative_pnl = []
    running_pnl = 0.0
    trade_count = 0

    for i in range(len(timestamps)):
        theo = theos[i]
        bid = bids[i]
        ask = asks[i]
        db = deribit_bids[i]
        da = deribit_asks[i]

        if theo is None:
            continue

        # BUY signal: our deribit bid exceeds market ask
        if db is not None and ask is not None and db > ask:
            buy_ts.append(timestamps[i])
            buy_prices.append(ask)
            running_pnl += (theo - ask)
            trade_count += 1
            pnl_ts.append(timestamps[i])
            cumulative_pnl.append(running_pnl)

        # SELL signal: our deribit ask is at or below market bid
        if da is not None and bid is not None and da <= bid:
            sell_ts.append(timestamps[i])
            sell_prices.append(bid)
            running_pnl += (bid - theo)
            trade_count += 1
            pnl_ts.append(timestamps[i])
            cumulative_pnl.append(running_pnl)

    # Build figure
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.05,
        subplot_titles=[
            f"{subtitle}  |  edge = ${edge:.3f}  |  {trade_count} trades",
            f"Cumulative PnL: ${running_pnl:.2f}  ({trade_count} trades)",
            "BTC Spot",
        ],
    )

    # --- Row 1: Prices ---

    # Kalshi bid (blue dots)
    ts_bid = [ts for ts, b in zip(timestamps, bids) if b is not None]
    vals_bid = [b for b in bids if b is not None]
    fig.add_trace(go.Scatter(
        x=ts_bid, y=vals_bid, mode="markers",
        marker=dict(size=3, color="steelblue", opacity=0.4),
        name="Kalshi Bid",
    ), row=1, col=1)

    # Kalshi ask (orange dots)
    ts_ask = [ts for ts, a in zip(timestamps, asks) if a is not None]
    vals_ask = [a for a in asks if a is not None]
    fig.add_trace(go.Scatter(
        x=ts_ask, y=vals_ask, mode="markers",
        marker=dict(size=3, color="#f97316", opacity=0.4),
        name="Kalshi Ask",
    ), row=1, col=1)

    # Deribit raw theo (dim green, thin)
    fig.add_trace(go.Scatter(
        x=timestamps, y=theos_raw, mode="lines",
        line=dict(color="#22c55e", width=1, dash="dot"),
        name="Deribit Raw",
        opacity=0.3,
    ), row=1, col=1)

    # Deribit smooth theo (bright green, thick)
    fig.add_trace(go.Scatter(
        x=timestamps, y=theos, mode="lines",
        line=dict(color="#22c55e", width=2),
        name="Deribit Smooth",
    ), row=1, col=1)

    # Deribit bid (theo - edge, dashed green)
    fig.add_trace(go.Scatter(
        x=timestamps, y=deribit_bids, mode="lines",
        line=dict(color="#22c55e", width=1, dash="dash"),
        name=f"Deribit Bid (theo-{edge:.3f})",
    ), row=1, col=1)

    # Deribit ask (theo + edge, dashed red)
    fig.add_trace(go.Scatter(
        x=timestamps, y=deribit_asks, mode="lines",
        line=dict(color="#ef4444", width=1, dash="dash"),
        name=f"Deribit Ask (theo+{edge:.3f})",
    ), row=1, col=1)

    # BUY signals (green triangles at market ask)
    if buy_ts:
        fig.add_trace(go.Scatter(
            x=buy_ts, y=buy_prices, mode="markers",
            marker=dict(symbol="triangle-up", size=9, color="#22c55e",
                        line=dict(width=1, color="white")),
            name=f"BUY ({len(buy_ts)})",
        ), row=1, col=1)

    # SELL signals (red triangles at market bid)
    if sell_ts:
        fig.add_trace(go.Scatter(
            x=sell_ts, y=sell_prices, mode="markers",
            marker=dict(symbol="triangle-down", size=9, color="#ef4444",
                        line=dict(width=1, color="white")),
            name=f"SELL ({len(sell_ts)})",
        ), row=1, col=1)

    # --- Row 2: Cumulative PnL ---
    if pnl_ts:
        pnl_colors = ["#22c55e" if p >= 0 else "#ef4444" for p in cumulative_pnl]
        fig.add_trace(go.Scatter(
            x=pnl_ts, y=cumulative_pnl, mode="lines+markers",
            line=dict(color="#22c55e", width=2),
            marker=dict(size=3, color=pnl_colors),
            name="Cumulative PnL",
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.1)",
        ), row=2, col=1)

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray",
                      opacity=0.5, row=2, col=1)

    # --- Row 3: BTC ---
    fig.add_trace(go.Scatter(
        x=timestamps, y=btc_prices, mode="lines",
        line=dict(color="#facc15", width=1),
        name="BTC Spot",
    ), row=3, col=1)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=900,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        margin=dict(t=80),
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    fig.update_yaxes(title_text="BTC ($)", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)

    return fig


# =============================================================================
# Dash app
# =============================================================================

def create_app(data: dict, event_ticker: str) -> Dash:
    """Create a Dash app with edge input and bracket dropdown."""
    events = data["events"]
    tickers_sorted = sorted(
        events[event_ticker].items(), key=lambda x: x[1]["strike"]
    )

    # Dropdown options
    options = []
    for ticker, info in tickers_sorted:
        key = (event_ticker, ticker)
        if key in data["series_data"]:
            options.append({
                "label": info["subtitle"],
                "value": ticker,
            })

    default_ticker = options[0]["value"] if options else ""

    app = Dash(__name__)

    app.layout = html.Div(style={"backgroundColor": "#111827", "minHeight": "100vh",
                                  "padding": "20px", "fontFamily": "monospace"}, children=[
        html.H2(f"Deribit Theo vs Kalshi — {event_ticker}",
                style={"color": "white", "marginBottom": "15px"}),

        html.Div(style={"display": "flex", "gap": "30px", "alignItems": "center",
                         "marginBottom": "15px"}, children=[
            # Bracket dropdown
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
                html.Label("Bracket:", style={"color": "white", "fontSize": "14px"}),
                dcc.Dropdown(
                    id="bracket-dropdown",
                    options=options,
                    value=default_ticker,
                    style={"width": "280px", "backgroundColor": "#1f2937",
                           "color": "white"},
                ),
            ]),

            # Edge input
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
                html.Label("Edge ($):", style={"color": "white", "fontSize": "14px"}),
                dcc.Input(
                    id="edge-input",
                    type="number",
                    value=0.02,
                    step=0.005,
                    min=0,
                    max=0.5,
                    style={"width": "100px", "backgroundColor": "#1f2937",
                           "color": "white", "border": "1px solid #374151",
                           "borderRadius": "4px", "padding": "8px",
                           "fontSize": "14px"},
                ),
                html.Button("Update", id="update-btn", n_clicks=0,
                           style={"backgroundColor": "#22c55e", "color": "black",
                                  "border": "none", "borderRadius": "4px",
                                  "padding": "8px 16px", "cursor": "pointer",
                                  "fontWeight": "bold", "fontSize": "14px"}),
            ]),

            # Stats output
            html.Div(id="stats-output", style={"color": "#9ca3af", "fontSize": "13px"}),
        ]),

        # Chart
        dcc.Graph(id="main-chart", style={"height": "900px"}),
    ])

    @app.callback(
        [Output("main-chart", "figure"),
         Output("stats-output", "children")],
        [Input("update-btn", "n_clicks"),
         Input("bracket-dropdown", "value")],
        [State("edge-input", "value")],
    )
    def update_chart(n_clicks, ticker, edge):
        if not ticker:
            return go.Figure(), ""

        edge = float(edge or 0)
        fig = build_figure(data, event_ticker, ticker, edge)

        # Compute stats
        key = (event_ticker, ticker)
        d = data["series_data"].get(key, {})
        theos_raw = d.get("theos", [])
        theos_smooth = d.get("theos_smooth", [])
        bids = d.get("bids", [])
        asks = d.get("asks", [])

        # Use smooth when available
        theos = [s if s is not None else r
                 for s, r in zip(theos_smooth, theos_raw)]

        buy_count = 0
        sell_count = 0
        total_pnl = 0.0
        for theo, bid, ask in zip(theos, bids, asks):
            if theo is None:
                continue
            db = theo - edge
            da = theo + edge
            if ask is not None and db > ask:
                buy_count += 1
                total_pnl += (theo - ask)
            if bid is not None and da <= bid:
                sell_count += 1
                total_pnl += (bid - theo)

        n_points = len(theos)
        stats = (f"Snapshots: {n_points:,}  |  "
                 f"Buys: {buy_count}  |  Sells: {sell_count}  |  "
                 f"PnL: ${total_pnl:.2f}  |  "
                 f"PnL/trade: ${total_pnl / (buy_count + sell_count):.4f}"
                 if (buy_count + sell_count) > 0
                 else f"Snapshots: {n_points:,}  |  No trades at edge=${edge:.3f}")

        return fig, stats

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Deribit theo vs Kalshi with edge & PnL"
    )
    parser.add_argument(
        "--dbs", type=str, nargs="+",
        default=[
            "marketdata/market_data_2026-04-13.db",
            "marketdata/market_data_2026-04-14.db",
        ],
        help="Database files to load",
    )
    parser.add_argument("--event", type=str, default=None, help="Event ticker")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.dbs, args.event)

    if not data:
        print("No data with deribit_theo found")
        return

    events = data["events"]
    print(f"\nEvents with Deribit theos:")
    for ev, tickers in events.items():
        print(f"  {ev}: {len(tickers)} brackets")

    # Pick event
    if args.event:
        event_ticker = args.event
    else:
        event_list = sorted(events.keys())
        if len(event_list) == 1:
            event_ticker = event_list[0]
        else:
            for i, ev in enumerate(event_list):
                print(f"  [{i}] {ev}")
            idx = int(input("\nSelect event number: "))
            event_ticker = event_list[idx]

    print(f"\nStarting Dash app for {event_ticker}...")
    print(f"Open http://localhost:{args.port} in your browser\n")

    app = create_app(data, event_ticker)
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
