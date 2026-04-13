"""
Backtest: Arb Scanner on Recorded Data

Reads recorded orderbook snapshots from SQLite and checks at each timestamp
whether a sell-all or buy-all arbitrage existed across all brackets.

Note: recorded data only has top-of-book (best bid/ask), so this checks
arb at size=1. Real depth would be needed for larger sizes.

Displays an interactive Plotly chart showing:
    1. Sell-all premium over time (sum of YES bids) vs $1.00 line
    2. Buy-all cost over time (sum of YES asks) vs $1.00 line
    3. BTC spot price

Usage:
    python3 backtest_arb.py
    python3 backtest_arb.py --db marketdata/market_data_2026-04-12.db
    python3 backtest_arb.py --db marketdata/market_data_2026-04-12.db --event KXBTC-26APR1717
"""

import argparse
import sqlite3
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Data loading
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
    """Load snapshots grouped by timestamp.

    Returns list of:
        {"timestamp": datetime, "btc_price": float,
         "books": {ticker: {"bid": float, "ask": float}}}
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
# Arb computation
# =============================================================================

def compute_arb_series(
    snapshots: list[dict],
    tickers: list[dict],
) -> dict:
    """Compute sell-all and buy-all premium/cost at each timestamp.

    For sell-all: sum of YES bids across all brackets (what you'd collect).
    For buy-all: sum of YES asks across all brackets (what you'd pay).

    Also tracks per-bracket contribution and coverage (how many brackets
    have a bid or ask).

    Returns:
        {
            "timestamps": [datetime, ...],
            "btc_prices": [float, ...],
            "sell_premium": [float, ...],      # sum of bids
            "buy_cost": [float, ...],          # sum of asks
            "sell_edge": [float, ...],         # sell_premium - 1.0
            "buy_edge": [float, ...],          # 1.0 - buy_cost
            "bid_coverage": [int, ...],        # brackets with bid > 0
            "ask_coverage": [int, ...],        # brackets with ask > 0
            "both_coverage": [int, ...],       # brackets with both bid and ask
            "n_brackets": int,
            "per_bracket_bids": {ticker: [float, ...]},
            "per_bracket_asks": {ticker: [float, ...]},
        }
    """
    ticker_list = [t["ticker"] for t in tickers]
    n_brackets = len(tickers)

    timestamps = []
    btc_prices = []
    sell_premium = []
    buy_cost = []
    sell_edge = []
    buy_edge = []
    bid_coverage = []
    ask_coverage = []
    both_coverage = []

    per_bracket_bids = {t["ticker"]: [] for t in tickers}
    per_bracket_asks = {t["ticker"]: [] for t in tickers}

    for snap in snapshots:
        ts = snap["timestamp"]
        btc = snap["btc_price"]
        books = snap["books"]

        total_bid = 0.0
        total_ask = 0.0
        n_bid = 0
        n_ask = 0
        n_both = 0

        for t in tickers:
            ticker = t["ticker"]
            book = books.get(ticker)

            bid = book["bid"] if (book and book["bid"] > 0) else 0.0
            ask = book["ask"] if (book and book["ask"] > 0) else 0.0

            # For buy-all: if no ask, assume $1.00 (worst case)
            # For sell-all: if no bid, assume $0.00 (worst case)
            total_bid += bid
            total_ask += ask if ask > 0 else 1.0

            if bid > 0:
                n_bid += 1
            if ask > 0:
                n_ask += 1
            if bid > 0 and ask > 0:
                n_both += 1

            per_bracket_bids[ticker].append(bid)
            per_bracket_asks[ticker].append(ask if ask > 0 else 1.0)

        timestamps.append(ts)
        btc_prices.append(btc)
        sell_premium.append(total_bid)
        buy_cost.append(total_ask)
        sell_edge.append(total_bid - 1.0)
        buy_edge.append(1.0 - total_ask)
        bid_coverage.append(n_bid)
        ask_coverage.append(n_ask)
        both_coverage.append(n_both)

    return {
        "timestamps": timestamps,
        "btc_prices": btc_prices,
        "sell_premium": sell_premium,
        "buy_cost": buy_cost,
        "sell_edge": sell_edge,
        "buy_edge": buy_edge,
        "bid_coverage": bid_coverage,
        "ask_coverage": ask_coverage,
        "both_coverage": both_coverage,
        "n_brackets": n_brackets,
        "per_bracket_bids": per_bracket_bids,
        "per_bracket_asks": per_bracket_asks,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_arb(result: dict, tickers: list[dict]):
    """Interactive Plotly chart showing arb opportunities over time."""
    ts = result["timestamps"]
    n = result["n_brackets"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.30, 0.30, 0.20, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            "Sell-All Premium (sum of YES bids)",
            "Buy-All Cost (sum of YES asks)",
            "Bracket Coverage",
            "BTC Spot",
        ],
    )

    # --- Row 1: Sell-all premium ---
    # Color points by whether arb exists
    sell_colors = ["rgba(0,200,80,0.7)" if e > 0 else "rgba(100,100,200,0.4)"
                   for e in result["sell_edge"]]
    fig.add_trace(
        go.Scatter(
            x=ts, y=result["sell_premium"], mode="markers",
            marker=dict(size=3, color=sell_colors),
            name="Sell Premium",
            showlegend=True,
        ),
        row=1, col=1,
    )
    # $1.00 reference line
    fig.add_hline(y=1.0, line_dash="dash", line_color="white",
                  opacity=0.5, row=1, col=1)

    # --- Row 2: Buy-all cost ---
    buy_colors = ["rgba(0,200,80,0.7)" if e > 0 else "rgba(240,100,60,0.4)"
                  for e in result["buy_edge"]]
    fig.add_trace(
        go.Scatter(
            x=ts, y=result["buy_cost"], mode="markers",
            marker=dict(size=3, color=buy_colors),
            name="Buy Cost",
            showlegend=True,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="white",
                  opacity=0.5, row=2, col=1)

    # --- Row 3: Coverage ---
    fig.add_trace(
        go.Scatter(
            x=ts, y=result["bid_coverage"], mode="lines",
            line=dict(color="steelblue", width=1),
            name=f"Brackets w/ bid (of {n})",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts, y=result["ask_coverage"], mode="lines",
            line=dict(color="#f97316", width=1),
            name=f"Brackets w/ ask (of {n})",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts, y=result["both_coverage"], mode="lines",
            line=dict(color="#22c55e", width=1),
            name=f"Brackets w/ both (of {n})",
        ),
        row=3, col=1,
    )

    # --- Row 4: BTC ---
    fig.add_trace(
        go.Scatter(
            x=ts, y=result["btc_prices"], mode="lines",
            line=dict(color="#facc15", width=1),
            name="BTC Spot",
        ),
        row=4, col=1,
    )

    # --- Stats annotation ---
    sell_arb_count = sum(1 for e in result["sell_edge"] if e > 0)
    buy_arb_count = sum(1 for e in result["buy_edge"] if e > 0)
    total = len(ts)

    sell_arb_edges = [e for e in result["sell_edge"] if e > 0]
    buy_arb_edges = [e for e in result["buy_edge"] if e > 0]

    avg_sell_edge = sum(sell_arb_edges) / len(sell_arb_edges) if sell_arb_edges else 0
    avg_buy_edge = sum(buy_arb_edges) / len(buy_arb_edges) if buy_arb_edges else 0
    max_sell_edge = max(sell_arb_edges) if sell_arb_edges else 0
    max_buy_edge = max(buy_arb_edges) if buy_arb_edges else 0

    fig.update_layout(
        title=dict(
            text=(f"Arb Scanner — {result['n_brackets']} brackets, "
                  f"{total} snapshots | "
                  f"Sell arb: {sell_arb_count}/{total} "
                  f"({100*sell_arb_count/total:.1f}%) "
                  f"avg=${avg_sell_edge:.4f} max=${max_sell_edge:.4f} | "
                  f"Buy arb: {buy_arb_count}/{total} "
                  f"({100*buy_arb_count/total:.1f}%) "
                  f"avg=${avg_buy_edge:.4f} max=${max_buy_edge:.4f}"),
            font_size=13,
        ),
        template="plotly_dark",
        height=900,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
    )

    fig.update_yaxes(title_text="Premium ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=2, col=1)
    fig.update_yaxes(title_text="# Brackets", row=3, col=1)
    fig.update_yaxes(title_text="BTC ($)", row=4, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=4, col=1)

    fig.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest arb scanner")
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

    tickers = load_tickers(conn, event_ticker)
    print(f"Brackets: {len(tickers)}")

    # Load data
    print("Loading snapshots...")
    snapshots = load_snapshots(conn, event_ticker)
    print(f"Loaded {len(snapshots)} snapshots")
    conn.close()

    # Compute
    print("Scanning for arb opportunities...")
    result = compute_arb_series(snapshots, tickers)

    # Summary stats
    sell_arb = sum(1 for e in result["sell_edge"] if e > 0)
    buy_arb = sum(1 for e in result["buy_edge"] if e > 0)
    total = len(result["timestamps"])

    print(f"\nResults ({total} snapshots):")
    print(f"  Sell-all arb: {sell_arb}/{total} snapshots "
          f"({100*sell_arb/total:.1f}%)")
    if sell_arb > 0:
        edges = [e for e in result["sell_edge"] if e > 0]
        print(f"    avg edge: ${sum(edges)/len(edges):.4f}  "
              f"max: ${max(edges):.4f}")

    print(f"  Buy-all arb:  {buy_arb}/{total} snapshots "
          f"({100*buy_arb/total:.1f}%)")
    if buy_arb > 0:
        edges = [e for e in result["buy_edge"] if e > 0]
        print(f"    avg edge: ${sum(edges)/len(edges):.4f}  "
              f"max: ${max(edges):.4f}")

    # Coverage stats
    avg_bid = sum(result["bid_coverage"]) / total
    avg_ask = sum(result["ask_coverage"]) / total
    avg_both = sum(result["both_coverage"]) / total
    print(f"\n  Avg bracket coverage:")
    print(f"    With bid:  {avg_bid:.1f}/{len(tickers)}")
    print(f"    With ask:  {avg_ask:.1f}/{len(tickers)}")
    print(f"    With both: {avg_both:.1f}/{len(tickers)}")

    # Plot
    print("\nOpening interactive chart...")
    plot_arb(result, tickers)


if __name__ == "__main__":
    main()
