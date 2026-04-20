"""
Backtest: Deribit-Implied Theo vs Kalshi Market Prices

Fetches the CURRENT Deribit option chain, builds the risk-neutral
density via Breeden-Litzenberger, then computes theo for each Kalshi
bracket. Plots these theos against historical Kalshi bid/ask from
the recorded SQLite data.

Since Deribit's public API doesn't provide historical mark prices
per option, we use the CURRENT density as a static reference. This
shows how the Deribit-implied theos compare to where Kalshi was
actually trading. The density shifts over time as spot moves, so
this is an approximation — but it reveals the structural relationship
between the two markets.

For a true historical backtest, you'd need to record Deribit option
snapshots alongside Kalshi data (future enhancement).

Usage:
    python3 backtest_deribit.py
    python3 backtest_deribit.py --db marketdata/market_data_2026-04-12.db
    python3 backtest_deribit.py --db marketdata/market_data_2026-04-13.db --event KXBTC-26APR1717
"""

import argparse
import sqlite3
import sys
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/4RunnerApp")

from deribit_vol import DeribitBracketPricer, find_deribit_expiry
from market_discovery import parse_strike

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# =============================================================================
# Data loading (same as other backtest scripts)
# =============================================================================

def load_events(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT event_ticker FROM orderbook ORDER BY event_ticker"
    ).fetchall()
    return [r[0] for r in rows]


def load_tickers(conn: sqlite3.Connection, event_ticker: str) -> list[dict]:
    rows = conn.execute(
        """SELECT DISTINCT ticker, yes_sub_title, strike
           FROM orderbook
           WHERE event_ticker = ?
           ORDER BY strike""",
        (event_ticker,),
    ).fetchall()
    return [{"ticker": r[0], "subtitle": r[1], "strike": r[2]} for r in rows]


def load_snapshots(conn: sqlite3.Connection, event_ticker: str) -> list[dict]:
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
# Compute Deribit theos for each snapshot
# =============================================================================

def compute_deribit_theos(
    snapshots: list[dict],
    tickers: list[dict],
    pricer: DeribitBracketPricer,
) -> dict:
    """Compute Deribit-implied theo for each bracket at each timestamp.

    Since we only have the CURRENT density, the theo for each bracket
    is static (doesn't change with time). But we still plot it as a
    horizontal line against the moving bid/ask to show the relationship.

    Returns same structure as backtest_smile for compatibility.
    """
    strikes = [t["strike"] for t in tickers]

    # Pre-compute bracket bounds
    bounds = []
    for i in range(len(tickers)):
        k_low = strikes[i]
        k_high = strikes[i + 1] if i + 1 < len(strikes) else None
        bounds.append((k_low, k_high))

    # Compute static Deribit theo for each bracket
    deribit_theos = {}
    for i, t in enumerate(tickers):
        k_low, k_high = bounds[i]
        theo = pricer.bracket_theo(k_low, k_high)
        deribit_theos[t["ticker"]] = theo

    # Build time series
    timestamps = []
    btc_prices = []
    bracket_data = {}
    for t in tickers:
        bracket_data[t["ticker"]] = {
            "subtitle": t["subtitle"],
            "theos": [],
            "bids": [],
            "asks": [],
        }

    for snap in snapshots:
        ts = snap["timestamp"]
        spot = snap["btc_price"]
        books = snap["books"]

        if spot <= 0:
            continue

        timestamps.append(ts)
        btc_prices.append(spot)

        for t in tickers:
            ticker = t["ticker"]
            book = books.get(ticker)

            bid = book["bid"] if (book and book["bid"] > 0) else None
            ask = book["ask"] if (book and book["ask"] > 0) else None

            bracket_data[ticker]["theos"].append(deribit_theos[ticker])
            bracket_data[ticker]["bids"].append(bid)
            bracket_data[ticker]["asks"].append(ask)

    return {
        "timestamps": timestamps,
        "btc_prices": btc_prices,
        "brackets": bracket_data,
        "deribit_theos": deribit_theos,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_deribit_vs_market(result: dict, tickers: list[dict], pricer: DeribitBracketPricer):
    """Interactive Plotly chart: Deribit theo vs Kalshi bid/ask per bracket."""
    timestamps = result["timestamps"]
    btc_prices = result["btc_prices"]
    brackets = result["brackets"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.04,
        subplot_titles=[
            "Deribit Theo vs Kalshi Market",
            "Theo − Mid",
            "BTC Spot",
        ],
    )

    # Per-bracket traces: bid, ask, theo, spread (4 per bracket)
    trace_sets = []

    for i, t in enumerate(tickers):
        ticker = t["ticker"]
        data = brackets[ticker]
        theos = data["theos"]
        bids = data["bids"]
        asks = data["asks"]
        subtitle = data["subtitle"]
        visible = (i == 0)

        start_idx = len(fig.data)

        # Kalshi bid
        ts_bid = [ts for ts, b in zip(timestamps, bids) if b is not None]
        vals_bid = [b for b in bids if b is not None]
        fig.add_trace(
            go.Scatter(
                x=ts_bid, y=vals_bid, mode="markers",
                marker=dict(size=3, color="steelblue", opacity=0.5),
                name="Kalshi Bid",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Kalshi ask
        ts_ask = [ts for ts, a in zip(timestamps, asks) if a is not None]
        vals_ask = [a for a in asks if a is not None]
        fig.add_trace(
            go.Scatter(
                x=ts_ask, y=vals_ask, mode="markers",
                marker=dict(size=3, color="#f97316", opacity=0.5),
                name="Kalshi Ask",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Deribit theo (line — static value since we use current density)
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=theos, mode="lines",
                line=dict(color="#22c55e", width=2),
                name="Deribit Theo",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Spread: theo vs mid (bar)
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

        trace_sets.append((start_idx, 4))  # 4 traces per bracket

    # BTC spot (always visible)
    btc_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=btc_prices, mode="lines",
            line=dict(color="#facc15", width=1),
            name="BTC Spot",
            showlegend=True,
        ),
        row=3, col=1,
    )
    total_traces = len(fig.data)

    # Dropdown buttons
    buttons = []
    for i, t in enumerate(tickers):
        ticker = t["ticker"]
        subtitle = brackets[ticker]["subtitle"]
        deribit_theo = result["deribit_theos"][ticker]

        # RMSE vs mid
        bids = brackets[ticker]["bids"]
        asks = brackets[ticker]["asks"]
        theos = brackets[ticker]["theos"]
        paired = [(th, (b + a) / 2.0) for th, b, a in zip(theos, bids, asks)
                  if b is not None and a is not None]
        rmse_str = ""
        if paired:
            rmse_val = (sum((th - m) ** 2 for th, m in paired) / len(paired)) ** 0.5
            rmse_str = f"  RMSE={rmse_val:.4f}"

        vis = [False] * total_traces
        start, count = trace_sets[i]
        for j in range(start, start + count):
            vis[j] = True
        vis[btc_trace_idx] = True

        buttons.append(dict(
            label=f"{subtitle}  theo=${deribit_theo:.3f}{rmse_str}",
            method="update",
            args=[
                {"visible": vis},
                {"title": dict(
                    text=f"Deribit Theo vs Kalshi — {subtitle} (theo=${deribit_theo:.3f})",
                    font_size=14,
                )},
            ],
        ))

    lo, hi = pricer.strike_range
    fig.update_layout(
        title=dict(
            text=(f"Deribit Theo vs Kalshi — {tickers[0]['subtitle'] if tickers else ''} | "
                  f"{pricer.n_options} options, density [{lo/1000:.0f}k-{hi/1000:.0f}k]"),
            font_size=14,
        ),
        template="plotly_dark",
        height=850,
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
    fig.update_yaxes(title_text="BTC ($)", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)

    fig.show()


# =============================================================================
# Expiry parser
# =============================================================================

def parse_expiry_from_event(event_ticker: str) -> str:
    """Convert Kalshi event ticker to a fake close_time for Deribit matching.

    KXBTC-26APR1717 → "2026-04-17T21:00:00Z" (5pm ET = 21:00 UTC in EDT)
    """
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return ""

    date_part = parts[1]  # "26APR1717"
    year = 2000 + int(date_part[:2])
    month_str = date_part[2:5].upper()
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    month = months.get(month_str, 1)
    day = int(date_part[5:7])
    hour_et = int(date_part[7:9])
    hour_utc = hour_et + 4  # EDT

    return f"{year}-{month:02d}-{day:02d}T{hour_utc:02d}:00:00Z"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest Deribit theo vs Kalshi")
    parser.add_argument(
        "--db", type=str, default="marketdata/market_data_2026-04-12.db",
        help="Path to SQLite database",
    )
    parser.add_argument("--event", type=str, default=None, help="Event ticker")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # Pick event
    events = load_events(conn)
    if not events:
        print("No events found")
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

    # Find matching Deribit expiry
    close_time = parse_expiry_from_event(event_ticker)
    deribit_expiry = find_deribit_expiry(close_time)
    if not deribit_expiry:
        print("No matching Deribit expiry found")
        return
    print(f"Deribit expiry: {deribit_expiry}")

    # Fetch Deribit option chain and build density
    print("Fetching Deribit option chain (this takes ~30s)...")
    pricer = DeribitBracketPricer()
    ok = pricer.fetch_options(deribit_expiry)
    if not ok:
        print("Failed to build Deribit density")
        return

    lo, hi = pricer.strike_range
    print(f"Density: {pricer.n_density_points} points, "
          f"range=[${lo:,.0f} - ${hi:,.0f}]")

    # Load Kalshi data
    tickers = load_tickers(conn, event_ticker)
    if not tickers:
        print("No tickers found")
        return
    print(f"Kalshi brackets: {len(tickers)}")

    print("Loading Kalshi snapshots...")
    snapshots = load_snapshots(conn, event_ticker)
    print(f"Loaded {len(snapshots)} snapshots")
    conn.close()

    # Compute theos
    print("Computing Deribit theos...")
    result = compute_deribit_theos(snapshots, tickers, pricer)
    print(f"Done — {len(result['timestamps'])} timestamps")

    # Print per-bracket summary
    print(f"\nDeribit theo per bracket:")
    for t in tickers:
        ticker = t["ticker"]
        theo = result["deribit_theos"][ticker]
        bids = result["brackets"][ticker]["bids"]
        asks = result["brackets"][ticker]["asks"]

        valid_bids = [b for b in bids if b is not None]
        valid_asks = [a for a in asks if a is not None]
        avg_bid = sum(valid_bids) / len(valid_bids) if valid_bids else 0
        avg_ask = sum(valid_asks) / len(valid_asks) if valid_asks else 0

        in_spread = "IN" if (avg_bid <= theo <= avg_ask and avg_bid > 0) else "  "
        print(f"  {t['subtitle']:>28s}  theo=${theo:.3f}  "
              f"avg_bid=${avg_bid:.3f}  avg_ask=${avg_ask:.3f}  {in_spread}")

    # Plot
    print("\nOpening interactive chart...")
    plot_deribit_vs_market(result, tickers, pricer)


if __name__ == "__main__":
    main()
