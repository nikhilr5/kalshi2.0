"""
Backtest: Vol Smile vs Market Prices

Reads recorded orderbook data from SQLite, calibrates the vol smile at each
timestamp, computes theo for ALL brackets, then displays an interactive Plotly
chart with a dropdown to switch between brackets.

Usage:
    python3 backtest_smile.py
    python3 backtest_smile.py --db marketdata/market_data_2026-04-12.db
    python3 backtest_smile.py --db marketdata/market_data_2026-04-12.db --event KXBTC-26APR1717
"""

import argparse
import sqlite3
import sys
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/4RunnerApp")

from vol_smile import VolSmile, _bracket_prob

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# =============================================================================
# Data loading
# =============================================================================

def load_events(conn: sqlite3.Connection) -> list[str]:
    """Return all distinct event tickers in the database."""
    rows = conn.execute(
        "SELECT DISTINCT event_ticker FROM orderbook ORDER BY event_ticker"
    ).fetchall()
    return [r[0] for r in rows]


def load_tickers(conn: sqlite3.Connection, event_ticker: str) -> list[dict]:
    """Return all tickers for an event, sorted by strike."""
    rows = conn.execute(
        """SELECT DISTINCT ticker, yes_sub_title, strike
           FROM orderbook
           WHERE event_ticker = ?
           ORDER BY strike""",
        (event_ticker,),
    ).fetchall()
    return [{"ticker": r[0], "subtitle": r[1], "strike": r[2]} for r in rows]


def load_snapshots(conn: sqlite3.Connection, event_ticker: str) -> list[dict]:
    """Load all orderbook snapshots for an event, grouped by timestamp.

    Returns list of snapshots, each containing:
        - timestamp: datetime
        - btc_price: float
        - books: {ticker: {"bid": float, "ask": float}}
    """
    rows = conn.execute(
        """SELECT timestamp, ticker, yes_bid, yes_ask, btc_price
           FROM orderbook
           WHERE event_ticker = ?
           ORDER BY timestamp, strike""",
        (event_ticker,),
    ).fetchall()

    snapshots = []
    current_ts = None
    current_snap = None

    for ts_str, ticker, bid, ask, btc in rows:
        if ts_str != current_ts:
            if current_snap:
                snapshots.append(current_snap)
            current_ts = ts_str
            current_snap = {
                "timestamp": datetime.fromisoformat(ts_str),
                "btc_price": btc,
                "books": {},
            }
        current_snap["books"][ticker] = {"bid": bid, "ask": ask}

    if current_snap:
        snapshots.append(current_snap)

    return snapshots


# =============================================================================
# Theo calculation — all brackets at once
# =============================================================================

def compute_all_theos(
    snapshots: list[dict],
    tickers: list[dict],
    expiry: datetime,
) -> dict:
    """Compute theo and market bid/ask for ALL brackets at each timestamp.

    The smile is calibrated once per timestamp and reused across brackets.

    Returns:
        {
            "timestamps": [datetime, ...],
            "btc_prices": [float, ...],
            "brackets": {
                ticker: {
                    "subtitle": str,
                    "theos": [float, ...],
                    "bids": [float|None, ...],
                    "asks": [float|None, ...],
                    "vols": [float, ...],
                },
                ...
            }
        }
    """
    strikes = [t["strike"] for t in tickers]
    ticker_list = [t["ticker"] for t in tickers]

    # Pre-compute bracket bounds for each ticker
    bounds = []
    for i, t in enumerate(tickers):
        k_low = strikes[i]
        k_high = strikes[i + 1] if i + 1 < len(strikes) else None
        mid_k = (k_low + k_high) / 2.0 if (k_high and k_high > 0) else k_low
        bounds.append((k_low, k_high, mid_k))

    # Initialize result containers
    timestamps = []
    btc_prices = []
    bracket_data = {}
    for t in tickers:
        bracket_data[t["ticker"]] = {
            "subtitle": t["subtitle"],
            "theos": [],
            "bids": [],
            "asks": [],
            "vols": [],
        }

    smile = VolSmile()

    for snap_idx, snap in enumerate(snapshots):
        ts = snap["timestamp"]
        spot = snap["btc_price"]
        books = snap["books"]

        if spot <= 0:
            continue

        T = (expiry - ts).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            continue

        # Build mid-prices for smile calibration (only both bid and ask)
        mid_prices = []
        for t in tickers:
            book = books.get(t["ticker"])
            if book and book["bid"] > 0 and book["ask"] > 0:
                mid_prices.append((book["bid"] + book["ask"]) / 2.0)
            else:
                mid_prices.append(0.0)

        # Calibrate smile once for this timestamp
        ok = smile.calibrate(spot, strikes, mid_prices, T)

        timestamps.append(ts)
        btc_prices.append(spot)

        # Compute theo for every bracket using the same smile
        for i, t in enumerate(tickers):
            k_low, k_high, mid_k = bounds[i]

            if ok:
                sigma = smile.vol_at(mid_k)
            else:
                sigma = 0.50

            theo = _bracket_prob(spot, k_low, k_high, T, sigma)

            # Market bid/ask
            book = books.get(t["ticker"])
            bid = book["bid"] if (book and book["bid"] > 0) else None
            ask = book["ask"] if (book and book["ask"] > 0) else None

            bracket_data[t["ticker"]]["theos"].append(theo)
            bracket_data[t["ticker"]]["bids"].append(bid)
            bracket_data[t["ticker"]]["asks"].append(ask)
            bracket_data[t["ticker"]]["vols"].append(sigma)

        # Progress
        if (snap_idx + 1) % 500 == 0:
            print(f"  {snap_idx + 1}/{len(snapshots)} snapshots processed...")

    return {
        "timestamps": timestamps,
        "btc_prices": btc_prices,
        "brackets": bracket_data,
    }


# =============================================================================
# Plotting — dropdown to select bracket
# =============================================================================

def plot_all_brackets(result: dict, tickers: list[dict]):
    """Interactive Plotly chart with a dropdown to switch between brackets."""
    timestamps = result["timestamps"]
    btc_prices = result["btc_prices"]
    brackets = result["brackets"]
    n_brackets = len(tickers)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        vertical_spacing=0.04,
        subplot_titles=["Theo vs Market", "Theo − Market", "Vol (%)", "BTC Spot"],
    )

    # For each bracket, add 4 traces (market mid, theo, spread, vol)
    # All hidden by default except the first bracket
    trace_sets = []  # list of (start_idx, count) per bracket

    for i, t in enumerate(tickers):
        ticker = t["ticker"]
        data = brackets[ticker]
        theos = data["theos"]
        bids = data["bids"]
        asks = data["asks"]
        subtitle = data["subtitle"]
        visible = (i == 0)

        start_idx = len(fig.data)

        # Market bid (scatter)
        ts_bid = [ts for ts, b in zip(timestamps, bids) if b is not None]
        vals_bid = [b for b in bids if b is not None]
        fig.add_trace(
            go.Scatter(
                x=ts_bid, y=vals_bid, mode="markers",
                marker=dict(size=3, color="steelblue", opacity=0.5),
                name="Bid",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Market ask (scatter)
        ts_ask = [ts for ts, a in zip(timestamps, asks) if a is not None]
        vals_ask = [a for a in asks if a is not None]
        fig.add_trace(
            go.Scatter(
                x=ts_ask, y=vals_ask, mode="markers",
                marker=dict(size=3, color="#f97316", opacity=0.5),
                name="Ask",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Theo (line)
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=theos, mode="lines",
                line=dict(color="#22c55e", width=1.5),
                name="Theo (Smile)",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Spread vs mid (bar) — theo minus midpoint of bid/ask
        ts_spread = []
        spreads = []
        colors = []
        for ts, theo, bid, ask in zip(timestamps, theos, bids, asks):
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
                ts_spread.append(ts)
                s = theo - mid
                spreads.append(s)
                colors.append("rgba(0,200,80,0.6)" if s >= 0 else "rgba(240,60,60,0.6)")

        fig.add_trace(
            go.Bar(
                x=ts_spread, y=spreads,
                marker_color=colors,
                name="Spread",
                visible=visible,
                showlegend=False,
            ),
            row=2, col=1,
        )

        # Vol (line) — show annualised vol as percentage
        vols = data["vols"]
        vols_pct = [v * 100.0 for v in vols]
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=vols_pct, mode="lines",
                line=dict(color="#e879f9", width=1.2),
                name="Vol (%)",
                visible=visible,
                showlegend=True,
            ),
            row=3, col=1,
        )

        trace_sets.append((start_idx, 5))  # 5 traces per bracket

    # BTC spot price (always visible, one trace at the end)
    btc_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=btc_prices, mode="lines",
            line=dict(color="#facc15", width=1),
            name="BTC Spot",
            showlegend=True,
        ),
        row=4, col=1,
    )
    total_traces = len(fig.data)

    # Build dropdown buttons — one per bracket
    buttons = []
    for i, t in enumerate(tickers):
        subtitle = brackets[t["ticker"]]["subtitle"]

        # Compute stats for the label (RMSE vs bid/ask midpoint)
        theos = brackets[t["ticker"]]["theos"]
        bids = brackets[t["ticker"]]["bids"]
        asks = brackets[t["ticker"]]["asks"]
        paired = [(th, (b + a) / 2.0) for th, b, a in zip(theos, bids, asks)
                  if b is not None and a is not None]
        rmse = ""
        if paired:
            rmse_val = (sum((th - m) ** 2 for th, m in paired) / len(paired)) ** 0.5
            rmse = f"  RMSE={rmse_val:.4f}"

        # Visibility: hide all bracket traces, show only this bracket's 5 + BTC
        vis = [False] * total_traces
        start, count = trace_sets[i]
        for j in range(start, start + count):
            vis[j] = True
        vis[btc_trace_idx] = True  # BTC always visible

        buttons.append(dict(
            label=f"{subtitle}{rmse}",
            method="update",
            args=[
                {"visible": vis},
                {"title": dict(
                    text=f"Theo vs Market — {subtitle}",
                    font_size=16,
                )},
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"Theo vs Market — {tickers[0]['subtitle'] if tickers else ''}",
            font_size=16,
        ),
        template="plotly_dark",
        height=950,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0,
            xanchor="left",
            y=1.18,
            yanchor="top",
            buttons=buttons,
            bgcolor="#1e2736",
            font=dict(color="white", size=11),
            active=0,
        )],
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Vol (%)", row=3, col=1)
    fig.update_yaxes(title_text="BTC ($)", row=4, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=4, col=1)

    fig.show()


# =============================================================================
# Main
# =============================================================================

def parse_expiry_from_event(event_ticker: str) -> datetime:
    """Parse expiry datetime from event ticker like KXBTC-26APR1717.

    Format: YYMMMDDHR where HR is the hour in ET (Eastern Time).
    The 17 at the end means 5pm ET = 21:00 UTC (during EDT).
    """
    parts = event_ticker.split("-")
    if len(parts) < 2:
        raise ValueError(f"Can't parse expiry from {event_ticker}")

    date_part = parts[1]  # e.g. "26APR1717"
    year = 2000 + int(date_part[:2])
    month_str = date_part[2:5].upper()
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    month = months.get(month_str, 1)
    day = int(date_part[5:7])
    hour_et = int(date_part[7:9])

    # Convert ET to UTC (EDT = UTC-4)
    hour_utc = hour_et + 4
    return datetime(year, month, day, hour_utc, 0, 0)


def main():
    parser = argparse.ArgumentParser(description="Backtest vol smile vs market")
    parser.add_argument(
        "--db", type=str, default="marketdata/market_data_2026-04-12.db",
        help="Path to SQLite database",
    )
    parser.add_argument("--event", type=str, default=None, help="Event ticker")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # --- Pick event ---
    events = load_events(conn)
    if not events:
        print("No events found in database")
        return

    if args.event:
        event_ticker = args.event
    else:
        print("\nAvailable events:")
        for i, ev in enumerate(events):
            print(f"  [{i}] {ev}")
        idx = int(input("\nSelect event number: "))
        event_ticker = events[idx]

    print(f"\nEvent: {event_ticker}")

    expiry = parse_expiry_from_event(event_ticker)
    print(f"Expiry: {expiry} UTC")

    tickers = load_tickers(conn, event_ticker)
    if not tickers:
        print("No tickers found for this event")
        return
    print(f"Brackets: {len(tickers)}")

    # --- Load data ---
    print("Loading snapshots...")
    snapshots = load_snapshots(conn, event_ticker)
    print(f"Loaded {len(snapshots)} snapshots")
    conn.close()

    # --- Compute all brackets ---
    print("Computing theo for all brackets (this may take a moment)...")
    result = compute_all_theos(snapshots, tickers, expiry)
    print(f"Done — {len(result['timestamps'])} timestamps x {len(tickers)} brackets")

    # --- Print summary stats ---
    print("\nPer-bracket RMSE:")
    for t in tickers:
        data = result["brackets"][t["ticker"]]
        paired = [(th, (b + a) / 2.0) for th, b, a in zip(data["theos"], data["bids"], data["asks"])
                  if b is not None and a is not None]
        if paired:
            rmse = (sum((th - m) ** 2 for th, m in paired) / len(paired)) ** 0.5
            avg = sum(th - m for th, m in paired) / len(paired)
            print(f"  {t['subtitle']:>25s}  RMSE={rmse:.4f}  avg={avg:+.4f}  ({len(paired)} pts)")

    # --- Plot ---
    print("\nOpening interactive chart...")
    plot_all_brackets(result, tickers)


if __name__ == "__main__":
    main()
