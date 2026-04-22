"""
Interactive live fill + market + PnL + markout graph for Kalshi expirations.

Select an expiration from the dropdown — data loads on demand.
Auto-refreshes every hour.

Usage:
    python graph_fills.py
    # Opens browser at http://127.0.0.1:8050
"""

import sqlite3
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State

DB_PATH = Path(__file__).resolve().parent.parent / "marketdata" / "recorder.db"
REFRESH_MS = 60 * 60 * 1000  # 1 hour
MARKOUT_INTERVALS = [5, 30, 60, 120, 300, 600]  # seconds

LEAD_LAG_INTERVALS = [5, 30, 60, 120, 300, 600]  # seconds

# Vertical annotation lines: (ct_datetime, label, color)
VLINES = []


def compute_lead_lag(market_by_ticker_data, side="bid", intervals=LEAD_LAG_INTERVALS,
                     theo_bid_idx=4, theo_ask_idx=5):
    """Compute lead-lag correlation between theo-market signal and future market moves.

    Args:
        side: "bid" compares theo_bid vs market_bid,
              "ask" compares theo_ask vs market_ask.

    Returns {interval: (correlation, n_samples)}.
    """
    import bisect

    # Build arrays of (utc_s, market_price, theo_price)
    points = []
    for m in market_by_ticker_data:
        mkt_bid, mkt_ask = m[2], m[3]
        tb, ta = m[theo_bid_idx], m[theo_ask_idx]
        if side == "bid":
            mkt_price, theo_price = mkt_bid, tb
        else:
            mkt_price, theo_price = mkt_ask, ta
        if mkt_price > 0 and theo_price > 0:
            points.append((m[0], mkt_price, theo_price))

    if len(points) < 10:
        return {iv: (0.0, 0) for iv in intervals}

    utc_times = [p[0] for p in points]
    result = {}

    for iv in intervals:
        signals = []
        moves = []
        for i, (t, mp, tp) in enumerate(points):
            target = t + iv
            j = bisect.bisect_left(utc_times, target)

            # Find closest snapshot to target
            best_j = None
            best_dist = float("inf")
            for candidate in [j - 1, j]:
                if 0 <= candidate < len(points) and candidate != i:
                    dist = abs(utc_times[candidate] - target)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = candidate

            if best_j is None or best_dist > iv + 10:
                continue

            signal = tp - mp  # theo - market
            future_move = points[best_j][1] - mp  # market(t+Δ) - market(t)
            signals.append(signal)
            moves.append(future_move)

        n = len(signals)
        if n < 5:
            result[iv] = (0.0, n)
            continue

        # Pearson correlation
        mean_s = sum(signals) / n
        mean_m = sum(moves) / n
        cov = sum((s - mean_s) * (m - mean_m) for s, m in zip(signals, moves)) / n
        std_s = (sum((s - mean_s) ** 2 for s in signals) / n) ** 0.5
        std_m = (sum((m - mean_m) ** 2 for m in moves) / n) ** 0.5

        if std_s > 0 and std_m > 0:
            corr = cov / (std_s * std_m)
        else:
            corr = 0.0

        result[iv] = (corr, n)

    return result


def compute_edge_magnitude(market_by_ticker_data, side="bid", intervals=LEAD_LAG_INTERVALS,
                           theo_bid_idx=4, theo_ask_idx=5):
    """Compute average future market move conditional on spread direction.

    Returns {interval: {
        "pos_move": avg move when theo > market (in cents),
        "neg_move": avg move when theo < market (in cents),
        "edge": difference between the two (in cents),
        "n_pos": sample count for positive spread,
        "n_neg": sample count for negative spread,
    }}
    """
    import bisect

    points = []
    for m in market_by_ticker_data:
        mkt_bid, mkt_ask = m[2], m[3]
        tb, ta = m[theo_bid_idx], m[theo_ask_idx]
        if side == "bid":
            mkt_price, theo_price = mkt_bid, tb
        else:
            mkt_price, theo_price = mkt_ask, ta
        if mkt_price > 0 and theo_price > 0:
            points.append((m[0], mkt_price, theo_price))

    if len(points) < 10:
        return {iv: {"pos_move": 0, "neg_move": 0, "edge": 0,
                     "n_pos": 0, "n_neg": 0} for iv in intervals}

    utc_times = [p[0] for p in points]
    result = {}

    for iv in intervals:
        pos_moves = []  # future moves when theo > market
        neg_moves = []  # future moves when theo < market

        for i, (t, mp, tp) in enumerate(points):
            target = t + iv
            j = bisect.bisect_left(utc_times, target)

            best_j = None
            best_dist = float("inf")
            for candidate in [j - 1, j]:
                if 0 <= candidate < len(points) and candidate != i:
                    dist = abs(utc_times[candidate] - target)
                    if dist < best_dist:
                        best_dist = dist
                        best_j = candidate

            if best_j is None or best_dist > iv + 10:
                continue

            spread = tp - mp
            future_move = (points[best_j][1] - mp) * 100  # in cents

            if spread > 0:
                pos_moves.append(future_move)
            elif spread < 0:
                neg_moves.append(future_move)

        avg_pos = sum(pos_moves) / len(pos_moves) if pos_moves else 0
        avg_neg = sum(neg_moves) / len(neg_moves) if neg_moves else 0

        result[iv] = {
            "pos_move": avg_pos,
            "neg_move": avg_neg,
            "edge": avg_pos - avg_neg,
            "n_pos": len(pos_moves),
            "n_neg": len(neg_moves),
        }

    return result


def get_events():
    conn = sqlite3.connect(str(DB_PATH))
    fill_events = conn.execute("""
        SELECT DISTINCT event_ticker FROM fills WHERE event_ticker != ''
    """).fetchall()
    ticker_events = conn.execute("""
        SELECT DISTINCT substr(ticker, 1, instr(ticker, '-T') - 1)
        FROM fills WHERE ticker LIKE '%-T%'
    """).fetchall()
    snap_events = conn.execute("""
        SELECT DISTINCT substr(ticker, 1, instr(ticker, '-T') - 1)
        FROM market_snapshots WHERE ticker LIKE '%-T%'
    """).fetchall()
    conn.close()
    events = set()
    for row in fill_events + ticker_events + snap_events:
        if row[0]:
            events.add(row[0])
    return sorted(events)


def load_event_data(event):
    conn = sqlite3.connect(str(DB_PATH))

    fills = conn.execute("""
        SELECT ts, ticker, action, side, count, price, client_order_id,
               COALESCE(fee, 0) as fee
        FROM fills
        WHERE event_ticker LIKE ? OR ticker LIKE ?
        ORDER BY id
    """, (f"%{event.split('-', 1)[1]}%", f"{event}%")).fetchall()

    snapshots = conn.execute("""
        SELECT ts, spot_mid
        FROM market_snapshots
        WHERE ticker LIKE ?
        GROUP BY ts ORDER BY ts
    """, (f"{event}%",)).fetchall()

    market = conn.execute("""
        SELECT ts, ticker, kalshi_yes_bid, kalshi_yes_ask, theo_bid, theo_ask,
               deribit_bid_iv, deribit_ask_iv,
               theo_bid_weekly, theo_ask_weekly,
               deribit_bid_iv_weekly, deribit_ask_iv_weekly
        FROM market_snapshots
        WHERE ticker LIKE ?
        ORDER BY ts
    """, (f"{event}%",)).fetchall()

    conn.close()
    return fills, snapshots, market


def parse_ts(ts_str):
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def to_ct(dt):
    import zoneinfo
    ct = zoneinfo.ZoneInfo("America/Chicago")
    return dt.astimezone(ct).replace(tzinfo=None)


def compute_pnl_series(fill_list, strike_filter=None):
    state = {}
    pnl_points = {}

    for row in fill_list:
        ts_str, ticker, action, side, count, price = row[:6]
        fee = float(row[7]) if len(row) > 7 else 0.0
        if strike_filter and not ticker.endswith(f"-T{strike_filter}"):
            continue

        strike = ticker.split("-T")[1] if "-T" in ticker else ticker
        dt = to_ct(parse_ts(ts_str))

        if count == 0:
            count = 1

        if ticker not in state:
            state[ticker] = {"position": 0, "queue": deque(), "realized_pnl": 0.0}
        s = state[ticker]

        # Subtract fee from PnL
        s["realized_pnl"] -= fee

        if (action == "buy" and side == "yes") or (action == "sell" and side == "no"):
            direction = 1
        else:
            direction = -1

        remaining = count
        while remaining > 0 and s["queue"] and (
            (direction == 1 and s["position"] < 0) or
            (direction == -1 and s["position"] > 0)
        ):
            open_price, open_count = s["queue"][0]
            close_count = min(remaining, open_count)
            if s["position"] > 0:
                s["realized_pnl"] += (price - open_price) * close_count
            else:
                s["realized_pnl"] += (open_price - price) * close_count
            remaining -= close_count
            if close_count == open_count:
                s["queue"].popleft()
            else:
                s["queue"][0] = (open_price, open_count - close_count)
            s["position"] += direction * close_count

        if remaining > 0:
            s["queue"].append((price, remaining))
            s["position"] += direction * remaining

        if strike not in pnl_points:
            pnl_points[strike] = []
        pnl_points[strike].append((dt, round(s["realized_pnl"], 2)))

    return pnl_points


def build_market_by_ticker(market_data):
    """Build {ticker: [(utc_ts_seconds, ct_dt, bid, ask, theo_bid, theo_ask,
                        deribit_bid_iv, deribit_ask_iv,
                        theo_bid_weekly, theo_ask_weekly,
                        deribit_bid_iv_weekly, deribit_ask_iv_weekly), ...]}."""
    result = {}
    for row in market_data:
        ts_str, ticker = row[0], row[1]
        bid, ask = row[2] or 0, row[3] or 0
        theo_bid, theo_ask = row[4] or 0, row[5] or 0
        db_bid_iv, db_ask_iv = row[6] or 0, row[7] or 0
        theo_bid_w = row[8] or 0 if len(row) > 8 else 0
        theo_ask_w = row[9] or 0 if len(row) > 9 else 0
        db_bid_iv_w = row[10] or 0 if len(row) > 10 else 0
        db_ask_iv_w = row[11] or 0 if len(row) > 11 else 0
        dt_utc = parse_ts(ts_str)
        dt_ct = to_ct(dt_utc)
        utc_s = dt_utc.timestamp()
        if ticker not in result:
            result[ticker] = []
        result[ticker].append((utc_s, dt_ct, bid, ask, theo_bid, theo_ask,
                               db_bid_iv, db_ask_iv,
                               theo_bid_w, theo_ask_w, db_bid_iv_w, db_ask_iv_w))
    return result


def compute_markouts(fills, market_by_ticker, event, strike_filter,
                     intervals=MARKOUT_INTERVALS):
    """Compute markouts for each fill at each interval.

    Markout = (exit_price - fill_price) * direction
    For buys: exit_price = bid (what you'd sell at to exit)
    For sells: exit_price = ask (what you'd buy at to exit)
    direction = +1 for buys, -1 for sells

    Returns {interval: [(ct_time, markout_cents, hover_text), ...]}.
    """
    import bisect

    filtered = [f for f in fills if f[1].endswith(f"-T{strike_filter}")]
    ticker_key = f"{event}-T{strike_filter}"
    mkt = market_by_ticker.get(ticker_key, [])

    if not mkt or not filtered:
        return {iv: [] for iv in intervals}

    # Pre-sort market data by UTC timestamp for binary search
    mkt_times = [m[0] for m in mkt]

    result = {iv: [] for iv in intervals}

    for row in filtered:
        ts_str, ticker, action, side, count, price = row[:6]
        dt_utc = parse_ts(ts_str)
        fill_utc_s = dt_utc.timestamp()
        fill_ct = to_ct(dt_utc)

        # Direction: +1 if buying (want price up), -1 if selling (want price down)
        if (action == "buy" and side == "yes") or (action == "sell" and side == "no"):
            direction = 1
        else:
            direction = -1

        action_label = "BUY" if direction == 1 else "SELL"

        for iv in intervals:
            target_s = fill_utc_s + iv
            # Find nearest snapshot to target time
            idx = bisect.bisect_left(mkt_times, target_s)

            # Pick closest of idx-1 and idx
            best_idx = None
            best_dist = float("inf")
            for candidate in [idx - 1, idx]:
                if 0 <= candidate < len(mkt_times):
                    dist = abs(mkt_times[candidate] - target_s)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = candidate

            if best_idx is None or best_dist > iv + 10:
                # No snapshot close enough
                continue

            snap_bid = mkt[best_idx][2]
            snap_ask = mkt[best_idx][3]
            if snap_bid <= 0 or snap_ask <= 0:
                continue
            # Use exit price: bid for buys (what you'd sell at), ask for sells (what you'd buy at)
            exit_price = snap_bid if direction == 1 else snap_ask

            # Markout in cents
            markout = (exit_price - price) * direction * 100

            hover = (f"{action_label} @ ${price:.2f}<br>"
                     f"Exit +{iv}s: ${exit_price:.3f}<br>"
                     f"Markout: {markout:+.1f}¢<br>"
                     f"{fill_ct:%H:%M:%S}")

            result[iv].append((fill_ct, markout, hover))

    return result


def get_strike_trade_counts(fills):
    counts = {}
    for row in fills:
        _, ticker = row[0], row[1]
        strike = ticker.split("-T")[1] if "-T" in ticker else ""
        if strike:
            counts[strike] = counts.get(strike, 0) + 1
    return counts


def tag_init_flatten(fills):
    """Tag each fill as 'init' or 'flatten'.

    Uses client_order_id prefix ('init_' or 'flat_') when available.
    Falls back to position tracking: increases absolute position = 'init',
    decreases = 'flatten'.
    Returns list of tags parallel to fills.
    """
    position = {}  # ticker -> net position
    tags = []
    for row in fills:
        ts_str, ticker, action, side, count, price = row[:6]
        client_order_id = row[6] if len(row) > 6 else None

        # Use client_order_id tag if available
        if client_order_id and client_order_id.startswith("init_"):
            tag = "init"
        elif client_order_id and client_order_id.startswith("flat_"):
            tag = "flatten"
        else:
            # Fallback: infer from position change
            if count == 0:
                count = 1
            if (action == "buy" and side == "yes") or (action == "sell" and side == "no"):
                direction = 1
            else:
                direction = -1
            prev_pos = position.get(ticker, 0)
            new_pos = prev_pos + direction * count
            tag = "init" if abs(new_pos) > abs(prev_pos) else "flatten"

        # Track position regardless of tagging method
        if count == 0:
            count = 1
        if (action == "buy" and side == "yes") or (action == "sell" and side == "no"):
            direction = 1
        else:
            direction = -1
        position[ticker] = position.get(ticker, 0) + direction * count

        tags.append(tag)
    return tags


STRIKE_COLORS = [
    "#f59e0b", "#3b82f6", "#a855f7", "#ec4899", "#14b8a6", "#f97316",
    "#6366f1", "#84cc16",
]

MARKOUT_COLORS = {
    5: "#3b82f6",   # blue
    30: "#ec4899",  # pink
    60: "#f59e0b",  # amber
    120: "#14b8a6", # teal
    300: "#22c55e", # green
    600: "#a855f7", # purple
}

app = Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#0d1117", "padding": "20px", "minHeight": "100vh"},
    children=[
        html.H2("Fills Dashboard",
                 style={"color": "#c8cdd5", "fontFamily": "monospace"}),
        html.Div([
            html.Label("Expiration:", style={"color": "#c8cdd5", "marginRight": "10px"}),
            dcc.Dropdown(
                id="event-dropdown", options=[], value=None,
                placeholder="Select expiration...",
                style={"width": "300px", "backgroundColor": "#141923", "color": "#000"},
            ),
            html.Span("", style={"width": "30px"}),
            html.Label("Strike:", style={"color": "#c8cdd5", "marginRight": "10px"}),
            dcc.Dropdown(
                id="strike-dropdown", value="ALL",
                style={"width": "250px", "backgroundColor": "#141923", "color": "#000"},
            ),
            html.Span("", style={"width": "30px"}),
            dcc.Checklist(
                id="init-only-check",
                options=[{"label": " Init trades only", "value": "init"}],
                value=[],
                style={"color": "#c8cdd5", "fontSize": "13px"},
                inputStyle={"marginRight": "5px"},
            ),
            html.Span("", style={"width": "20px"}),
            html.Label("Theo:", style={"color": "#c8cdd5", "marginRight": "6px", "fontSize": "13px"}),
            dcc.RadioItems(
                id="theo-source",
                options=[
                    {"label": " Daily IV", "value": "daily"},
                    {"label": " Weekly IV", "value": "weekly"},
                ],
                value="daily",
                style={"color": "#c8cdd5", "fontSize": "13px", "display": "flex", "gap": "12px"},
                inputStyle={"marginRight": "4px"},
            ),
            html.Span(id="last-refresh",
                       style={"color": "#5a6270", "fontSize": "11px",
                              "marginLeft": "20px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
        dcc.Graph(id="fill-graph"),
        html.Div(id="window-stats",
                 style={"color": "#c8cdd5", "fontFamily": "monospace",
                         "fontSize": "13px", "padding": "10px 0"}),
        dcc.Graph(id="ll-strike-graph", style={"display": "none"}),
        dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="data-store"),
    ],
)


@app.callback(
    Output("event-dropdown", "options"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_events(_n):
    events = get_events()
    # Query close_time and earliest snapshot per event in one query
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("""
        SELECT substr(ticker, 1, instr(ticker, '-T') - 1) AS evt,
               close_time, MIN(ts)
        FROM market_snapshots
        WHERE ticker LIKE '%-T%' AND close_time != ''
        GROUP BY evt
    """).fetchall()
    conn.close()
    event_info = {r[0]: (r[1], r[2]) for r in rows if r[0]}

    import zoneinfo
    _et = zoneinfo.ZoneInfo("America/New_York")

    def label(e):
        try:
            info = event_info.get(e)
            if not info:
                return f"{e} (?)"
            close_str, first_snap = info
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            close_et = close_dt.astimezone(_et)
            if close_et.hour != 17 or close_et.minute != 0:
                kind = "Hourly"
            else:
                first_dt = datetime.fromisoformat(first_snap.replace("Z", "+00:00"))
                hours = (close_dt - first_dt).total_seconds() / 3600
                kind = "Weekly" if hours > 48 else "Daily"
        except Exception:
            kind = "?"
        return f"{e} ({kind})"
    return [{"label": label(e), "value": e} for e in events]


@app.callback(
    [Output("data-store", "data"),
     Output("strike-dropdown", "options"),
     Output("strike-dropdown", "value"),
     Output("last-refresh", "children")],
    [Input("event-dropdown", "value"),
     Input("refresh-interval", "n_intervals")],
)
def load_event(event, _n):
    if not event:
        return None, [], "ALL", ""

    fills, snapshots, market = load_event_data(event)
    trade_counts = get_strike_trade_counts(fills)

    # Collect strikes from both fills and market snapshots
    all_strikes = set()
    for f in fills:
        if "-T" in f[1]:
            all_strikes.add(f[1].split("-T")[1])
    for m in market:
        if "-T" in m[1]:
            all_strikes.add(m[1].split("-T")[1])
    all_strikes.discard("")

    strikes = sorted(all_strikes, key=lambda s: float(s))

    total_trades = sum(trade_counts.values())
    options = [{"label": f"All Strikes ({total_trades})", "value": "ALL"}]
    for s in strikes:
        cnt = trade_counts.get(s, 0)
        options.append({"label": f"${float(s):,.0f} ({cnt})", "value": s})

    now_ct = to_ct(datetime.now(tz=timezone.utc))
    refresh_text = f"Last refresh: {now_ct:%H:%M CT}"

    data = {
        "event": event,
        "fills": fills,
        "snapshots": snapshots,
        "market": market,
    }
    return data, options, "ALL", refresh_text


@app.callback(
    [Output("fill-graph", "figure"),
     Output("fill-graph", "style")],
    [Input("strike-dropdown", "value"),
     Input("data-store", "data"),
     Input("init-only-check", "value"),
     Input("theo-source", "value")],
)
def update_graph(selected_strike, data, init_only_check, theo_source):
    if not data:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#141923",
            annotations=[dict(
                text="Select an expiration to view data",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#5a6270", size=18),
            )],
        )
        return fig, {"height": "950px"}

    event = data["event"]
    fills = data["fills"]
    snapshots = data["snapshots"]
    market_data = data["market"]
    market_by_ticker = build_market_by_ticker(market_data)

    # Theo column indices based on source selection
    use_weekly = (theo_source == "weekly")
    _tb = 8 if use_weekly else 4   # theo_bid index
    _ta = 9 if use_weekly else 5   # theo_ask index
    _ib = 10 if use_weekly else 6  # deribit_bid_iv index
    _ia = 11 if use_weekly else 7  # deribit_ask_iv index

    init_only = "init" in (init_only_check or [])

    # Tag fills as init/flatten and filter if checkbox is checked
    tags = tag_init_flatten(fills)
    if init_only:
        fills = [f for f, t in zip(fills, tags) if t == "init"]

    spot_times = [to_ct(parse_ts(s[0])) for s in snapshots]
    spot_prices = [s[1] for s in snapshots]

    single_strike = selected_strike and selected_strike != "ALL"

    if single_strike:
        # 13 rows: market+fills, PnL, 6x markouts, market vs theo, deribit IV, theo-bid spread, theo-ask spread, spot
        n_markouts = len(MARKOUT_INTERVALS)
        markout_titles = []
        for iv in MARKOUT_INTERVALS:
            markout_titles.append(f"Markout {iv}s" if iv < 60 else f"Markout {iv // 60}m")
        n_rows = 7 + n_markouts  # fills, pnl, markouts, mkt vs theo, deribit IV, bid spread, ask spread, spot
        markout_heights = [0.04] * n_markouts
        row_heights = [0.12, 0.04] + markout_heights + [0.12, 0.10, 0.10, 0.10, 0.12]
        # Normalize
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]
        iv_label = "Weekly" if use_weekly else "Daily"
        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=[
                f"Market & Fills — ${float(selected_strike):,.0f}",
                "Cumulative PnL",
                *markout_titles,
                f"Market vs Theo ({iv_label} IV) — ${float(selected_strike):,.0f}",
                f"Deribit {iv_label} IV — ${float(selected_strike):,.0f}",
                f"Theo Bid − Market Bid (¢) ({iv_label}) — ${float(selected_strike):,.0f}",
                f"Theo Ask − Market Ask (¢) ({iv_label}) — ${float(selected_strike):,.0f}",
                "Spot (BTC)",
            ],
        )
        graph_height = "3400px"
    else:
        n_rows = 2
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=["Fills", "Cumulative PnL"],
        )
        graph_height = "950px"

    # --- Fills ---
    filtered = fills
    strike_filter = None
    if single_strike:
        filtered = [f for f in fills if f[1].endswith(f"-T{selected_strike}")]
        strike_filter = selected_strike

    buy_times, buy_prices, buy_texts = [], [], []
    sell_times, sell_prices, sell_texts = [], [], []

    for row in filtered:
        ts_str, ticker, action, side, count, price = row[:6]
        dt = to_ct(parse_ts(ts_str))
        strike = ticker.split("-T")[1] if "-T" in ticker else ""
        label = f"${float(strike):,.0f}" if strike else ticker
        hover = f"{action} {side} x{count} @ ${price:.2f}<br>{label}<br>{dt:%H:%M:%S}"
        if action == "buy":
            buy_times.append(dt)
            buy_prices.append(price)
            buy_texts.append(hover)
        else:
            sell_times.append(dt)
            sell_prices.append(price)
            sell_texts.append(hover)

    fig.add_trace(go.Scatter(
        x=buy_times, y=buy_prices, mode="markers", name="Buy",
        marker=dict(color="#22c55e", size=12, symbol="triangle-up",
                    line=dict(width=1, color="#ffffff")),
        text=buy_texts, hoverinfo="text",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_times, y=sell_prices, mode="markers", name="Sell",
        marker=dict(color="#ef4444", size=12, symbol="triangle-down",
                    line=dict(width=1, color="#ffffff")),
        text=sell_texts, hoverinfo="text",
    ), row=1, col=1)

    if single_strike:
        # Market bid/ask on row 1
        ticker_key = f"{event}-T{selected_strike}"
        mkt = market_by_ticker.get(ticker_key, [])
        if mkt:
            fig.add_trace(go.Scatter(
                x=[m[1] for m in mkt], y=[m[2] for m in mkt],
                mode="lines", name="Market Bid",
                line=dict(color="#2e7d32", width=1), opacity=0.4,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[m[1] for m in mkt], y=[m[3] for m in mkt],
                mode="lines", name="Market Ask",
                line=dict(color="#c62828", width=1), opacity=0.4,
            ), row=1, col=1)

        # PnL on row 2
        pnl_series = compute_pnl_series(fills, strike_filter)
        pts = pnl_series.get(selected_strike, [])
        if pts:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                mode="lines+markers",
                name=f"PnL ${float(selected_strike):,.0f}",
                line=dict(color="#f59e0b", width=2), marker=dict(size=5),
            ), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_dash="dash",
                      line_color="#5a6270", line_width=0.5)

        # Markouts on rows 3-6
        markouts = compute_markouts(fills, market_by_ticker, event,
                                     selected_strike)
        for i, iv in enumerate(MARKOUT_INTERVALS):
            row = 3 + i
            pts = markouts[iv]
            color = MARKOUT_COLORS[iv]

            if pts:
                times = [p[0] for p in pts]
                values = [p[1] for p in pts]
                hovers = [p[2] for p in pts]

                # Color markers by sign: green positive, red negative
                colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]

                iv_label = f"{iv}s" if iv < 60 else f"{iv // 60}m"
                fig.add_trace(go.Bar(
                    x=times, y=values,
                    name=f"{iv_label} Markout",
                    marker=dict(color=colors, opacity=1.0,
                                line=dict(width=1, color=colors)),
                    text=hovers, hoverinfo="text",
                    showlegend=True,
                    width=60000,  # bar width in ms
                ), row=row, col=1)

                # Mark zero-value trades with a dot
                zero_times = [t for t, v in zip(times, values) if v == 0]
                zero_hovers = [h for h, v in zip(hovers, values) if v == 0]
                if zero_times:
                    fig.add_trace(go.Scatter(
                        x=zero_times, y=[0] * len(zero_times),
                        mode="markers", name=f"{iv}s Zero",
                        marker=dict(color="#ffffff", size=6,
                                    line=dict(width=1, color="#5a6270")),
                        text=zero_hovers, hoverinfo="text",
                        showlegend=False,
                    ), row=row, col=1)

                # Average markout line
                avg = sum(values) / len(values)
                fig.add_hline(
                    y=avg, row=row, col=1,
                    line_dash="dot", line_color=color, line_width=1.5,
                    annotation_text=f"avg: {avg:+.1f}¢",
                    annotation_font_color=color,
                    annotation_font_size=10,
                )

            fig.add_hline(y=0, row=row, col=1, line_dash="dash",
                          line_color="#5a6270", line_width=0.5)
            fig.update_yaxes(title_text="¢", gridcolor="#1e2736", row=row, col=1)

        # Dynamic row offsets after markouts
        _r_mkt_theo = 3 + n_markouts      # market vs theo
        _r_iv = _r_mkt_theo + 1            # deribit IV
        _r_bid_spread = _r_iv + 1          # theo bid - mkt bid
        _r_ask_spread = _r_bid_spread + 1  # theo ask - mkt ask
        _r_spot = _r_ask_spread + 1        # spot

        # Market vs Theo
        ticker_key_theo = f"{event}-T{selected_strike}"
        mkt_theo = market_by_ticker.get(ticker_key_theo, [])
        if mkt_theo:
            mkt_times_ct = [m[1] for m in mkt_theo]
            mkt_bids = [m[2] for m in mkt_theo]
            mkt_asks = [m[3] for m in mkt_theo]
            theo_bids = [m[_tb] for m in mkt_theo]
            theo_asks = [m[_ta] for m in mkt_theo]

            fig.add_trace(go.Scatter(
                x=mkt_times_ct, y=mkt_bids,
                mode="lines", name="Market Bid",
                line=dict(color="#2e7d32", width=1.5), opacity=0.6,
                legendgroup="mkt_theo",
            ), row=_r_mkt_theo, col=1)
            fig.add_trace(go.Scatter(
                x=mkt_times_ct, y=mkt_asks,
                mode="lines", name="Market Ask",
                line=dict(color="#c62828", width=1.5), opacity=0.6,
                legendgroup="mkt_theo",
            ), row=_r_mkt_theo, col=1)

            theo_bid_t = [(t, b) for t, b in zip(mkt_times_ct, theo_bids) if b > 0]
            theo_ask_t = [(t, a) for t, a in zip(mkt_times_ct, theo_asks) if a > 0]

            if theo_bid_t:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in theo_bid_t],
                    y=[p[1] for p in theo_bid_t],
                    mode="lines", name="Theo Bid",
                    line=dict(color="#66bb6a", width=2, dash="dash"),
                    legendgroup="mkt_theo",
                ), row=_r_mkt_theo, col=1)
            if theo_ask_t:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in theo_ask_t],
                    y=[p[1] for p in theo_ask_t],
                    mode="lines", name="Theo Ask",
                    line=dict(color="#ef5350", width=2, dash="dash"),
                    legendgroup="mkt_theo",
                ), row=_r_mkt_theo, col=1)

        # Fills on row 7
        if buy_times:
            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode="markers", name="Buy",
                marker=dict(color="#22c55e", size=12, symbol="triangle-up",
                            line=dict(width=1, color="#ffffff")),
                text=buy_texts, hoverinfo="text",
                showlegend=False, legendgroup="mkt_theo",
            ), row=_r_mkt_theo, col=1)
        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode="markers", name="Sell",
                marker=dict(color="#ef4444", size=12, symbol="triangle-down",
                            line=dict(width=1, color="#ffffff")),
                text=sell_texts, hoverinfo="text",
                showlegend=False, legendgroup="mkt_theo",
            ), row=_r_mkt_theo, col=1)

        fig.update_yaxes(title_text="Price ($)", gridcolor="#1e2736", row=_r_mkt_theo, col=1)

        # Deribit IV on row 8
        ticker_key_iv = f"{event}-T{selected_strike}"
        mkt_iv = market_by_ticker.get(ticker_key_iv, [])
        if mkt_iv:
            iv_times = [m[1] for m in mkt_iv]
            bid_ivs = [m[_ib] * 100 for m in mkt_iv]  # convert to percentage
            ask_ivs = [m[_ia] * 100 for m in mkt_iv]

            # Filter out zero values for cleaner plot
            bid_iv_pts = [(t, v) for t, v in zip(iv_times, bid_ivs) if v > 0]
            ask_iv_pts = [(t, v) for t, v in zip(iv_times, ask_ivs) if v > 0]

            if bid_iv_pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in bid_iv_pts],
                    y=[p[1] for p in bid_iv_pts],
                    mode="lines", name="Deribit Bid IV",
                    line=dict(color="#22c55e", width=1.5),
                ), row=_r_iv, col=1)
            if ask_iv_pts:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in ask_iv_pts],
                    y=[p[1] for p in ask_iv_pts],
                    mode="lines", name="Deribit Ask IV",
                    line=dict(color="#ef4444", width=1.5),
                ), row=_r_iv, col=1)

        fig.update_yaxes(title_text="IV (%)", gridcolor="#1e2736", row=_r_iv, col=1)

        # Theo Bid − Market Bid on row 9
        ticker_key_spread = f"{event}-T{selected_strike}"
        mkt_spread = market_by_ticker.get(ticker_key_spread, [])
        if mkt_spread:
            bid_spread_times, bid_spread_vals = [], []
            ask_spread_times, ask_spread_vals = [], []
            for m in mkt_spread:
                mkt_bid, mkt_ask = m[2], m[3]
                tb, ta = m[_tb], m[_ta]
                if mkt_bid > 0 and tb > 0:
                    bid_spread_times.append(m[1])
                    bid_spread_vals.append((tb - mkt_bid) * 100)
                if mkt_ask > 0 and ta > 0:
                    ask_spread_times.append(m[1])
                    ask_spread_vals.append((ta - mkt_ask) * 100)

            if bid_spread_times:
                fig.add_trace(go.Scatter(
                    x=bid_spread_times, y=bid_spread_vals,
                    mode="lines", name="Theo Bid − Mkt Bid",
                    line=dict(color="#22c55e", width=1.5),
                ), row=_r_bid_spread, col=1)

            # Theo Ask − Market Ask on row 10
            if ask_spread_times:
                fig.add_trace(go.Scatter(
                    x=ask_spread_times, y=ask_spread_vals,
                    mode="lines", name="Theo Ask − Mkt Ask",
                    line=dict(color="#ef4444", width=1.5),
                ), row=_r_ask_spread, col=1)

        fig.add_hline(y=0, row=_r_bid_spread, col=1, line_dash="dash",
                      line_color="#5a6270", line_width=0.5)
        fig.update_yaxes(title_text="¢", gridcolor="#1e2736", row=_r_bid_spread, col=1)
        fig.add_hline(y=0, row=_r_ask_spread, col=1, line_dash="dash",
                      line_color="#5a6270", line_width=0.5)
        fig.update_yaxes(title_text="¢", gridcolor="#1e2736", row=_r_ask_spread, col=1)

        # Spot on row 11
        if spot_times:
            fig.add_trace(go.Scatter(
                x=spot_times, y=spot_prices,
                mode="lines", name="Spot (BTC)",
                line=dict(color="#5a6270", width=1),
            ), row=_r_spot, col=1)

        fig.update_yaxes(title_text="Price ($)", gridcolor="#1e2736", row=1, col=1)
        fig.update_yaxes(title_text="PnL ($)", gridcolor="#1e2736", row=2, col=1)
        fig.update_yaxes(title_text="Spot ($)", gridcolor="#1e2736", row=_r_spot, col=1)
        fig.update_xaxes(title_text="Time (CT)", gridcolor="#1e2736", row=_r_spot, col=1)

    else:
        # All mode
        if spot_times:
            fig.add_trace(go.Scatter(
                x=spot_times, y=spot_prices,
                mode="lines", name="Spot (BTC)",
                line=dict(color="#5a6270", width=1),
            ), row=1, col=1, secondary_y=True)

        pnl_series = compute_pnl_series(fills, None)
        strike_keys = sorted(pnl_series.keys())

        for i, strike in enumerate(strike_keys):
            pts = pnl_series[strike]
            color = STRIKE_COLORS[i % len(STRIKE_COLORS)]
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts],
                mode="lines+markers",
                name=f"PnL ${float(strike):,.0f}",
                line=dict(color=color, width=1.5), marker=dict(size=4),
            ), row=2, col=1)

        all_times_pnl = []
        running = {}
        for row in fills:
            ts_str, ticker, action, side, count, price = row[:6]
            strike = ticker.split("-T")[1] if "-T" in ticker else ticker
            dt = to_ct(parse_ts(ts_str))
            if strike in pnl_series:
                for pt in pnl_series[strike]:
                    if pt[0] == dt:
                        running[strike] = pt[1]
                        break
            all_times_pnl.append((dt, sum(running.values())))

        if all_times_pnl:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in all_times_pnl],
                y=[p[1] for p in all_times_pnl],
                mode="lines", name="PnL Total",
                line=dict(color="#ffffff", width=2.5, dash="dot"),
            ), row=2, col=1)

        fig.add_hline(y=0, row=2, col=1, line_dash="dash",
                      line_color="#5a6270", line_width=0.5)

        fig.update_yaxes(title_text="Fill Price ($)", gridcolor="#1e2736", row=1, col=1)
        fig.update_yaxes(title_text="Spot ($)", gridcolor="#1e2736", showgrid=False,
                         row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="PnL ($)", gridcolor="#1e2736", row=2, col=1)
        fig.update_xaxes(title_text="Time (CT)", gridcolor="#1e2736", row=2, col=1)

    title_extra = f" — ${float(selected_strike):,.0f}" if single_strike else ""
    fig.update_layout(
        title=dict(text=f"{event}{title_extra}", font=dict(color="#c8cdd5")),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#141923",
        legend=dict(bgcolor="#141923"),
        hovermode="closest",
    )

    for ann in fig.layout.annotations:
        ann.font.color = "#c8cdd5"

    # Add vertical annotation lines across all subplots
    for vline_dt, vline_label, vline_color in VLINES:
        for row_i in range(1, n_rows + 1):
            # xref/yref use "x"/"y" for row 1, "x2"/"y2" for row 2, etc.
            xref = "x" if row_i == 1 else f"x{row_i}"
            yref = "y" if row_i == 1 else f"y{row_i}"
            fig.add_shape(
                type="line",
                x0=vline_dt, x1=vline_dt, y0=0, y1=1,
                xref=xref, yref=f"{yref} domain",
                line=dict(color=vline_color, width=1, dash="dash"),
            )
        # Add label annotation on the first row only
        fig.add_annotation(
            x=vline_dt, y=1, yref="y domain", xref="x",
            text=vline_label, showarrow=False,
            font=dict(color=vline_color, size=10),
            textangle=-90, yanchor="bottom", xanchor="left",
        )

    return fig, {"height": graph_height}


@app.callback(
    Output("window-stats", "children"),
    [Input("fill-graph", "relayoutData")],
    [State("data-store", "data"),
     State("strike-dropdown", "value"),
     State("theo-source", "value")],
)
def update_window_stats(relayout_data, data, selected_strike, theo_source):
    if not data or not selected_strike or selected_strike == "ALL":
        return ""

    use_weekly = (theo_source == "weekly")
    _tb = 8 if use_weekly else 4
    _ta = 9 if use_weekly else 5

    event = data["event"]
    market_data = data["market"]
    market_by_ticker = build_market_by_ticker(market_data)
    ticker_key = f"{event}-T{selected_strike}"
    mkt = market_by_ticker.get(ticker_key, [])
    if not mkt:
        return ""

    # Parse visible x-axis range from relayoutData
    x_min = None
    x_max = None
    if relayout_data:
        # Shared x-axis: keys are "xaxis.range[0]" / "xaxis.range[1]"
        # or "xaxis10.range[0]" etc depending on which subplot was zoomed
        for key, val in relayout_data.items():
            if "range[0]" in key and "xaxis" in key:
                try:
                    x_min = parse_ts(val) if isinstance(val, str) else datetime.fromtimestamp(val / 1000)
                    x_min = to_ct(x_min) if x_min.tzinfo else x_min
                except Exception:
                    x_min = None
            if "range[1]" in key and "xaxis" in key:
                try:
                    x_max = parse_ts(val) if isinstance(val, str) else datetime.fromtimestamp(val / 1000)
                    x_max = to_ct(x_max) if x_max.tzinfo else x_max
                except Exception:
                    x_max = None

        # Check for autorange (double-click reset)
        if any("autorange" in k for k in relayout_data):
            x_min = None
            x_max = None

    # Filter market data to visible window
    if x_min or x_max:
        filtered_mkt = []
        for m in mkt:
            ct = m[1]  # ct_dt
            if x_min and ct < x_min:
                continue
            if x_max and ct > x_max:
                continue
            filtered_mkt.append(m)
    else:
        filtered_mkt = mkt

    if not filtered_mkt:
        return ""

    # Compute avg spreads per side
    bid_spread_vals = []
    ask_spread_vals = []
    for m in filtered_mkt:
        mkt_bid, mkt_ask = m[2], m[3]
        tb, ta = m[_tb], m[_ta]
        if mkt_bid > 0 and tb > 0:
            bid_spread_vals.append((tb - mkt_bid) * 100)
        if mkt_ask > 0 and ta > 0:
            ask_spread_vals.append((ta - mkt_ask) * 100)

    avg_bid_spread = sum(bid_spread_vals) / len(bid_spread_vals) if bid_spread_vals else 0
    avg_ask_spread = sum(ask_spread_vals) / len(ask_spread_vals) if ask_spread_vals else 0

    # Compute lead-lag correlation per side
    ll_bid = compute_lead_lag(filtered_mkt, side="bid", theo_bid_idx=_tb, theo_ask_idx=_ta)
    ll_ask = compute_lead_lag(filtered_mkt, side="ask", theo_bid_idx=_tb, theo_ask_idx=_ta)

    # Build display
    def corr_color(c):
        if c >= 0.05:
            return "#22c55e"
        elif c >= 0.02:
            return "#86efac"
        elif c > -0.02:
            return "#c8cdd5"
        else:
            return "#ef4444"

    def corr_label(c):
        if c >= 0.05:
            return "strong"
        elif c >= 0.02:
            return "weak"
        elif c > -0.02:
            return "neutral"
        else:
            return "adverse"

    window_label = "visible window" if (x_min or x_max) else "all data"

    stats_children = [
        html.Span(f"Theo vs Market Stats ({window_label})",
                  style={"fontWeight": "bold"}),
        html.Br(), html.Br(),

        # Bid side
        html.Span("BID SIDE   ",
                  style={"fontWeight": "bold", "color": "#22c55e", "marginRight": "10px"}),
        html.Span(f"Avg: {avg_bid_spread:+.1f}¢",
                  style={"marginRight": "25px",
                         "color": "#22c55e" if avg_bid_spread >= 0 else "#ef4444"}),
    ]
    for iv, (corr, n) in sorted(ll_bid.items()):
        label = f"{iv}s" if iv < 60 else f"{iv // 60}m"
        stats_children.append(
            html.Span(
                f"{label}: {corr:+.3f} ({corr_label(corr)}, n={n})",
                style={"marginRight": "18px", "color": corr_color(corr)},
            )
        )

    stats_children.append(html.Br())

    # Ask side
    stats_children.append(
        html.Span("ASK SIDE   ",
                  style={"fontWeight": "bold", "color": "#ef4444", "marginRight": "10px"})
    )
    stats_children.append(
        html.Span(f"Avg: {avg_ask_spread:+.1f}¢",
                  style={"marginRight": "25px",
                         "color": "#22c55e" if avg_ask_spread >= 0 else "#ef4444"})
    )
    for iv, (corr, n) in sorted(ll_ask.items()):
        label = f"{iv}s" if iv < 60 else f"{iv // 60}m"
        stats_children.append(
            html.Span(
                f"{label}: {corr:+.3f} ({corr_label(corr)}, n={n})",
                style={"marginRight": "18px", "color": corr_color(corr)},
            )
        )

    stats_children.append(html.Br())
    stats_children.append(
        html.Span(
            "Lead-lag legend:  ≥+0.05 strong (theo leads market)  |  "
            "+0.02 to +0.05 weak  |  −0.02 to +0.02 neutral  |  "
            "≤−0.02 adverse (market leads theo)",
            style={"color": "#5a6270", "fontSize": "11px"},
        )
    )

    # Edge magnitude
    mag_bid = compute_edge_magnitude(filtered_mkt, side="bid", theo_bid_idx=_tb, theo_ask_idx=_ta)
    mag_ask = compute_edge_magnitude(filtered_mkt, side="ask", theo_bid_idx=_tb, theo_ask_idx=_ta)

    stats_children.append(html.Br())
    stats_children.append(html.Br())
    stats_children.append(
        html.Span("Edge Magnitude (avg future market move by spread direction)",
                  style={"fontWeight": "bold"})
    )

    for side_label, mag, color in [("BID SIDE", mag_bid, "#22c55e"),
                                    ("ASK SIDE", mag_ask, "#ef4444")]:
        stats_children.append(html.Br())
        stats_children.append(
            html.Span(f"{side_label}   ",
                      style={"fontWeight": "bold", "color": color, "marginRight": "10px"})
        )
        for iv in sorted(mag.keys()):
            m = mag[iv]
            label = f"{iv}s" if iv < 60 else f"{iv // 60}m"
            edge_color = "#22c55e" if m["edge"] > 0.1 else "#c8cdd5" if m["edge"] > 0 else "#ef4444"
            stats_children.append(
                html.Span(
                    f"{label}: theo>mkt {m['pos_move']:+.2f}¢ (n={m['n_pos']}) | "
                    f"theo<mkt {m['neg_move']:+.2f}¢ (n={m['n_neg']}) | "
                    f"edge={m['edge']:.2f}¢",
                    style={"marginRight": "18px", "color": edge_color, "fontSize": "12px"},
                )
            )

    stats_children.append(html.Br())
    stats_children.append(
        html.Span(
            "Edge magnitude: when theo>mkt, market should move up (positive); "
            "when theo<mkt, market should move down (negative). "
            "Edge = difference between the two (bigger = stronger signal in cents).",
            style={"color": "#5a6270", "fontSize": "11px"},
        )
    )

    return stats_children


@app.callback(
    [Output("ll-strike-graph", "figure"),
     Output("ll-strike-graph", "style")],
    [Input("fill-graph", "relayoutData"),
     Input("strike-dropdown", "value"),
     Input("data-store", "data"),
     Input("theo-source", "value")],
)
def update_ll_strike_graph(relayout_data, selected_strike, data, theo_source):
    hide = {"display": "none"}
    empty_fig = go.Figure()

    if not data or not selected_strike or selected_strike != "ALL":
        return empty_fig, hide

    use_weekly = (theo_source == "weekly")
    _tb = 8 if use_weekly else 4
    _ta = 9 if use_weekly else 5

    market_data = data["market"]
    market_by_ticker = build_market_by_ticker(market_data)

    # Parse visible x-axis range
    x_min = None
    x_max = None
    if relayout_data:
        for key, val in relayout_data.items():
            if "range[0]" in key and "xaxis" in key:
                try:
                    x_min = parse_ts(val) if isinstance(val, str) else datetime.fromtimestamp(val / 1000)
                    x_min = to_ct(x_min) if x_min.tzinfo else x_min
                except Exception:
                    x_min = None
            if "range[1]" in key and "xaxis" in key:
                try:
                    x_max = parse_ts(val) if isinstance(val, str) else datetime.fromtimestamp(val / 1000)
                    x_max = to_ct(x_max) if x_max.tzinfo else x_max
                except Exception:
                    x_max = None
        if any("autorange" in k for k in relayout_data):
            x_min = None
            x_max = None

    # Compute lead-lag per strike
    strikes = []
    bid_corrs = {iv: [] for iv in LEAD_LAG_INTERVALS}
    ask_corrs = {iv: [] for iv in LEAD_LAG_INTERVALS}

    for ticker_key, mkt in sorted(market_by_ticker.items()):
        if "-T" not in ticker_key:
            continue
        strike_str = ticker_key.split("-T")[1]
        try:
            strike_val = float(strike_str)
        except ValueError:
            continue

        # Filter to visible window
        if x_min or x_max:
            filtered = [m for m in mkt
                        if (not x_min or m[1] >= x_min) and (not x_max or m[1] <= x_max)]
        else:
            filtered = mkt

        if len(filtered) < 20:
            continue

        ll_bid = compute_lead_lag(filtered, side="bid", theo_bid_idx=_tb, theo_ask_idx=_ta)
        ll_ask = compute_lead_lag(filtered, side="ask", theo_bid_idx=_tb, theo_ask_idx=_ta)

        strikes.append(strike_val)
        for iv in LEAD_LAG_INTERVALS:
            bid_corrs[iv].append(ll_bid[iv][0])
            ask_corrs[iv].append(ll_ask[iv][0])

    if not strikes:
        return empty_fig, hide

    # Build figure: one row per interval
    n_intervals = len(LEAD_LAG_INTERVALS)
    titles = [f"Lead-Lag {iv}s" if iv < 60 else f"Lead-Lag {iv // 60}m"
              for iv in LEAD_LAG_INTERVALS]
    fig = make_subplots(
        rows=n_intervals, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=titles,
    )

    strike_labels = [f"{s/1000:.1f}k" for s in strikes]

    for i, iv in enumerate(LEAD_LAG_INTERVALS):
        row = i + 1
        fig.add_trace(go.Bar(
            x=strike_labels, y=bid_corrs[iv],
            name=f"Bid {iv}s" if iv < 60 else f"Bid {iv // 60}m",
            marker=dict(color="#22c55e", opacity=0.8),
            showlegend=(i == 0),
            legendgroup="bid",
        ), row=row, col=1)
        fig.add_trace(go.Bar(
            x=strike_labels, y=ask_corrs[iv],
            name=f"Ask {iv}s" if iv < 60 else f"Ask {iv // 60}m",
            marker=dict(color="#ef4444", opacity=0.8),
            showlegend=(i == 0),
            legendgroup="ask",
        ), row=row, col=1)
        fig.add_hline(y=0, row=row, col=1, line_dash="dash",
                      line_color="#5a6270", line_width=0.5)
        fig.add_hline(y=0.05, row=row, col=1, line_dash="dot",
                      line_color="#f59e0b", line_width=0.5)
        fig.add_hline(y=-0.02, row=row, col=1, line_dash="dot",
                      line_color="#f59e0b", line_width=0.5)
        fig.update_yaxes(title_text="Corr", gridcolor="#1e2736", row=row, col=1)

    window_label = "visible window" if (x_min or x_max) else "all data"
    fig.update_layout(
        title=dict(text=f"Lead-Lag by Strike ({window_label})",
                   font=dict(color="#c8cdd5")),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#141923",
        legend=dict(bgcolor="#141923"),
        barmode="group",
        height=250 * n_intervals,
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#c8cdd5"

    for row in range(1, n_intervals + 1):
        fig.update_xaxes(showticklabels=True, gridcolor="#1e2736", row=row, col=1)
    fig.update_xaxes(title_text="Strike", row=n_intervals, col=1)

    return fig, {"display": "block"}


if __name__ == "__main__":
    events = get_events()
    print(f"Available expirations: {events}")
    app.run(debug=True, port=8050, host="0.0.0.0")
