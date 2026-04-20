"""
Historical Vol: Realized BTC Vol vs Implied Vol per Bracket

Computes:
    1. Realized vol — rolling standard deviation of BTC log returns,
       annualised.  Same for every bracket (it's the underlying).
    2. Implied vol — per bracket, extracted from the market bid/ask
       mid-price using bisection (same method as vol_smile.py).

Displays an interactive Plotly chart with a dropdown to switch between
brackets.  Each view shows realized vol and that bracket's implied vol
so you can see whether the market is pricing more or less vol than BTC
is actually delivering.

Usage:
    python3 historical_vol.py
    python3 historical_vol.py --db marketdata/market_data_2026-04-12.db
    python3 historical_vol.py --db marketdata/market_data_2026-04-12.db --event KXBTC-26APR1717
    python3 historical_vol.py --window 60   # rolling window in minutes
"""

import argparse
import math
import sqlite3
import sys
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/4RunnerApp")

from vol_smile import _implied_vol_bracket

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


def load_btc_prices(conn: sqlite3.Connection) -> list[tuple[datetime, float]]:
    """Load all BTC prices from the btc_price table, sorted by time."""
    rows = conn.execute(
        "SELECT timestamp, price FROM btc_price ORDER BY timestamp"
    ).fetchall()
    return [(datetime.fromisoformat(r[0]), float(r[1])) for r in rows]


def load_orderbook_snapshots(
    conn: sqlite3.Connection, event_ticker: str
) -> list[dict]:
    """Load orderbook snapshots grouped by timestamp.

    Returns list of snapshots:
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
# Realized vol — rolling window on BTC log returns
# =============================================================================

def compute_realized_vol(
    btc_data: list[tuple[datetime, float]],
    window_minutes: int = 30,
) -> tuple[list[datetime], list[float]]:
    """Compute annualised realized vol from BTC price history.

    Uses a rolling window of `window_minutes` to compute the standard
    deviation of log returns, then annualises it.

    Returns (timestamps, realized_vols) as percentages.
    """
    if len(btc_data) < 2:
        return [], []

    # Compute log returns between consecutive observations
    times = []
    log_returns = []
    for i in range(1, len(btc_data)):
        t_prev, p_prev = btc_data[i - 1]
        t_curr, p_curr = btc_data[i]
        if p_prev > 0 and p_curr > 0:
            dt_seconds = (t_curr - t_prev).total_seconds()
            if 0 < dt_seconds < 300:  # skip gaps > 5 min
                lr = math.log(p_curr / p_prev)
                times.append(t_curr)
                log_returns.append((lr, dt_seconds))

    if not log_returns:
        return [], []

    window_seconds = window_minutes * 60
    timestamps_out = []
    vols_out = []

    # Sliding window
    left = 0
    for right in range(len(log_returns)):
        # Move left pointer to maintain window
        while left < right:
            gap = (times[right] - times[left]).total_seconds()
            if gap <= window_seconds:
                break
            left += 1

        n = right - left + 1
        if n < 10:  # need enough samples
            continue

        # Compute variance of log returns in this window
        returns_in_window = [log_returns[j][0] for j in range(left, right + 1)]
        mean_r = sum(returns_in_window) / n
        var_r = sum((r - mean_r) ** 2 for r in returns_in_window) / (n - 1)

        # Average dt between observations in this window
        avg_dt = sum(log_returns[j][1] for j in range(left, right + 1)) / n

        # Annualise: vol_annual = sqrt(var_per_obs / avg_dt * seconds_per_year)
        seconds_per_year = 365.25 * 24 * 3600
        if avg_dt > 0:
            annual_var = var_r / avg_dt * seconds_per_year
            annual_vol = math.sqrt(max(annual_var, 0))
        else:
            annual_vol = 0.0

        timestamps_out.append(times[right])
        vols_out.append(annual_vol * 100.0)  # as percentage

    return timestamps_out, vols_out


# =============================================================================
# Implied vol per bracket over time
# =============================================================================

def compute_implied_vols(
    snapshots: list[dict],
    tickers: list[dict],
    expiry: datetime,
) -> dict:
    """Extract implied vol for each bracket at each timestamp.

    Returns:
        {
            "timestamps": [datetime, ...],
            "btc_prices": [float, ...],
            "brackets": {
                ticker: {
                    "subtitle": str,
                    "ivs": [float|None, ...],   # implied vol % or None
                },
                ...
            }
        }
    """
    strikes = [t["strike"] for t in tickers]

    # Pre-compute bracket bounds
    bounds = {}
    for i, t in enumerate(tickers):
        k_low = strikes[i]
        k_high = strikes[i + 1] if i + 1 < len(strikes) else None
        bounds[t["ticker"]] = (k_low, k_high)

    timestamps = []
    btc_prices = []
    bracket_data = {}
    for t in tickers:
        bracket_data[t["ticker"]] = {
            "subtitle": t["subtitle"],
            "ivs": [],
        }

    for snap_idx, snap in enumerate(snapshots):
        ts = snap["timestamp"]
        spot = snap["btc_price"]
        books = snap["books"]

        if spot <= 0:
            continue

        T = (expiry - ts).total_seconds() / (365.25 * 24 * 3600)
        if T <= 0:
            continue

        timestamps.append(ts)
        btc_prices.append(spot)

        for t in tickers:
            ticker = t["ticker"]
            k_low, k_high = bounds[ticker]
            book = books.get(ticker)

            iv = None
            if book and book["bid"] > 0 and book["ask"] > 0:
                mid = (book["bid"] + book["ask"]) / 2.0
                raw_iv = _implied_vol_bracket(spot, k_low, k_high, T, mid)
                if raw_iv is not None:
                    iv = raw_iv * 100.0  # as percentage

            bracket_data[ticker]["ivs"].append(iv)

        if (snap_idx + 1) % 500 == 0:
            print(f"  {snap_idx + 1}/{len(snapshots)} snapshots processed...")

    return {
        "timestamps": timestamps,
        "btc_prices": btc_prices,
        "brackets": bracket_data,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_vols(
    rv_timestamps: list[datetime],
    rv_vols: list[float],
    iv_result: dict,
    tickers: list[dict],
):
    """Interactive Plotly chart: realized vol vs implied vol per bracket."""
    iv_timestamps = iv_result["timestamps"]
    btc_prices = iv_result["btc_prices"]
    brackets = iv_result["brackets"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.05,
        subplot_titles=["Realized Vol vs Implied Vol", "BTC Spot"],
    )

    # Per-bracket traces: implied vol (toggled by dropdown)
    trace_sets = []  # (start_idx, count) per bracket

    for i, t in enumerate(tickers):
        ticker = t["ticker"]
        data = brackets[ticker]
        ivs = data["ivs"]
        visible = (i == 0)

        start_idx = len(fig.data)

        # Implied vol (scatter — only where we have data)
        ts_iv = [ts for ts, iv in zip(iv_timestamps, ivs) if iv is not None]
        vals_iv = [iv for iv in ivs if iv is not None]
        fig.add_trace(
            go.Scatter(
                x=ts_iv, y=vals_iv, mode="markers",
                marker=dict(size=4, color="#f97316", opacity=0.6),
                name="Implied Vol",
                visible=visible,
                showlegend=True,
            ),
            row=1, col=1,
        )

        trace_sets.append((start_idx, 1))  # 1 trace per bracket

    # Realized vol — always visible
    rv_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=rv_timestamps, y=rv_vols, mode="lines",
            line=dict(color="#22c55e", width=1.5),
            name="Realized Vol",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # BTC spot — always visible
    btc_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=iv_timestamps, y=btc_prices, mode="lines",
            line=dict(color="#facc15", width=1),
            name="BTC Spot",
            showlegend=True,
        ),
        row=2, col=1,
    )

    total_traces = len(fig.data)

    # Dropdown buttons
    buttons = []
    for i, t in enumerate(tickers):
        subtitle = brackets[t["ticker"]]["subtitle"]
        ivs = brackets[t["ticker"]]["ivs"]

        # Compute average IV for label
        valid_ivs = [iv for iv in ivs if iv is not None]
        avg_iv = f"  avg={sum(valid_ivs)/len(valid_ivs):.1f}%" if valid_ivs else ""

        vis = [False] * total_traces
        start, count = trace_sets[i]
        for j in range(start, start + count):
            vis[j] = True
        vis[rv_trace_idx] = True   # realized vol always visible
        vis[btc_trace_idx] = True  # BTC always visible

        buttons.append(dict(
            label=f"{subtitle}{avg_iv}",
            method="update",
            args=[
                {"visible": vis},
                {"title": dict(
                    text=f"Vol — {subtitle}",
                    font_size=16,
                )},
            ],
        ))

    fig.update_layout(
        title=dict(
            text=f"Vol — {tickers[0]['subtitle'] if tickers else ''}",
            font_size=16,
        ),
        template="plotly_dark",
        height=700,
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

    fig.update_yaxes(title_text="Vol (%)", row=1, col=1)
    fig.update_yaxes(title_text="BTC ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    fig.show()


# =============================================================================
# Expiry parser (same as backtest_smile.py)
# =============================================================================

def parse_expiry_from_event(event_ticker: str) -> datetime:
    """Parse expiry datetime from event ticker like KXBTC-26APR1717."""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        raise ValueError(f"Can't parse expiry from {event_ticker}")

    date_part = parts[1]
    year = 2000 + int(date_part[:2])
    month_str = date_part[2:5].upper()
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    month = months.get(month_str, 1)
    day = int(date_part[5:7])
    hour_et = int(date_part[7:9])
    hour_utc = hour_et + 4
    return datetime(year, month, day, hour_utc, 0, 0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Historical vol analysis")
    parser.add_argument(
        "--db", type=str, default="marketdata/market_data_2026-04-12.db",
        help="Path to SQLite database",
    )
    parser.add_argument("--event", type=str, default=None, help="Event ticker")
    parser.add_argument(
        "--window", type=int, default=30,
        help="Rolling window for realized vol in minutes (default: 30)",
    )
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

    # --- Realized vol from BTC prices ---
    print(f"Loading BTC prices (window={args.window}min)...")
    btc_data = load_btc_prices(conn)
    print(f"  {len(btc_data)} BTC price observations")
    rv_timestamps, rv_vols = compute_realized_vol(btc_data, args.window)
    print(f"  {len(rv_timestamps)} realized vol data points")

    if rv_vols:
        avg_rv = sum(rv_vols) / len(rv_vols)
        print(f"  Average realized vol: {avg_rv:.1f}%")

    # --- Implied vol per bracket ---
    print("Loading orderbook snapshots...")
    snapshots = load_orderbook_snapshots(conn, event_ticker)
    print(f"  {len(snapshots)} snapshots")
    conn.close()

    print("Extracting implied vols per bracket...")
    iv_result = compute_implied_vols(snapshots, tickers, expiry)
    print(f"  {len(iv_result['timestamps'])} timestamps")

    # --- Summary ---
    print("\nPer-bracket implied vol summary:")
    for t in tickers:
        ivs = iv_result["brackets"][t["ticker"]]["ivs"]
        valid = [v for v in ivs if v is not None]
        if valid:
            avg = sum(valid) / len(valid)
            lo = min(valid)
            hi = max(valid)
            print(f"  {t['subtitle']:>25s}  avg={avg:6.1f}%  "
                  f"range=[{lo:.1f}%, {hi:.1f}%]  ({len(valid)} pts)")
        else:
            print(f"  {t['subtitle']:>25s}  no data")

    # --- Plot ---
    print("\nOpening interactive chart...")
    plot_vols(rv_timestamps, rv_vols, iv_result, tickers)


if __name__ == "__main__":
    main()
