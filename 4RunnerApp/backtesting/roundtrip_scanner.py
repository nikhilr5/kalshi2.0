"""
Round-Trip Trade Scanner Using Deribit Theo

Scans recorded data for brackets where you could have:
    1. BOUGHT YES when market ask < theo - buy_edge  (market is cheap)
    2. SOLD YES when market bid > entry + min_profit  (take profit)
       or market bid > theo + sell_edge               (market is rich)

All positions must be closed before a configurable cutoff time before
expiry. No position at expiration.

Ranks brackets by total PnL and shows the best opportunities.

Usage:
    python3 roundtrip_scanner.py
    python3 roundtrip_scanner.py --buy-edge 0.02 --sell-edge 0.02
    python3 roundtrip_scanner.py --buy-edge 0.03 --sell-edge 0.03 --min-profit 0.01
    python3 roundtrip_scanner.py --event KXBTC-26APR1417
"""

import argparse
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Trade:
    """A single round-trip trade."""
    ticker: str
    subtitle: str
    buy_time: datetime
    buy_price: float      # bought at market ask
    buy_theo: float       # theo at time of buy
    sell_time: datetime = None
    sell_price: float = 0.0    # sold at market bid
    sell_theo: float = 0.0     # theo at time of sell
    pnl: float = 0.0
    hold_seconds: float = 0.0
    exit_reason: str = ""      # "profit", "theo_rich", "cutoff"


@dataclass
class BracketResult:
    """Results for one bracket."""
    ticker: str
    subtitle: str
    strike: float
    trades: list = field(default_factory=list)
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    avg_hold_seconds: float = 0.0


# =============================================================================
# Data loading
# =============================================================================

def load_all_data(db_paths: list[str], event_ticker: str | None = None) -> dict:
    """Load data from multiple DBs, grouped by event and ticker."""
    all_rows = []

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(orderbook)").fetchall()]
        if "deribit_theo" not in cols:
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
        print(f"  {db_path}: {len(rows):,} rows")
        conn.close()

    if not all_rows:
        return {}

    all_rows.sort(key=lambda r: r[0])

    # Group by event -> ticker -> time series
    data = {}  # event -> ticker -> {subtitle, strike, snapshots: [{ts, bid, ask, theo, btc}]}
    for row in all_rows:
        ts_str, ticker, ev, subtitle, strike, bid, ask, btc, theo_raw = row[:9]
        theo_smooth = row[9] if len(row) > 9 else None
        # Use smooth when available, fall back to raw
        theo = theo_smooth if theo_smooth is not None else theo_raw
        ts = datetime.fromisoformat(ts_str)

        if ev not in data:
            data[ev] = {}
        if ticker not in data[ev]:
            data[ev][ticker] = {
                "subtitle": subtitle,
                "strike": strike,
                "snapshots": [],
            }

        data[ev][ticker]["snapshots"].append({
            "ts": ts,
            "bid": bid if bid > 0 else None,
            "ask": ask if ask > 0 else None,
            "theo": theo,
            "btc": btc,
        })

    return data


# =============================================================================
# Round-trip simulator
# =============================================================================

def simulate_roundtrips(
    snapshots: list[dict],
    ticker: str,
    subtitle: str,
    buy_edge: float,
    sell_edge: float,
    min_profit: float,
    cutoff_time: datetime | None = None,
    max_position: int = 1,
) -> list[Trade]:
    """Simulate round-trip trades on one bracket.

    Entry: buy when ask < theo - buy_edge (market is cheap vs Deribit)
    Exit:  sell when bid > entry + min_profit (take profit)
           or bid > theo + sell_edge (market is rich vs Deribit)
    Force close: at cutoff_time if still holding

    Only holds max_position contracts at a time (default 1).
    """
    trades = []
    open_trades = []  # multiple positions allowed

    for snap in snapshots:
        ts = snap["ts"]
        bid = snap["bid"]
        ask = snap["ask"]
        theo = snap["theo"]

        if theo is None:
            continue

        # Try to close open positions
        if bid is not None:
            still_open = []
            for ot in open_trades:
                past_cutoff = cutoff_time and ts >= cutoff_time
                profit_target_hit = bid >= ot.buy_price + min_profit
                theo_rich = bid >= theo + sell_edge
                theo_fair = bid >= theo

                if past_cutoff or profit_target_hit or theo_rich or theo_fair:
                    ot.sell_time = ts
                    ot.sell_price = bid
                    ot.sell_theo = theo
                    ot.pnl = bid - ot.buy_price
                    ot.hold_seconds = (ts - ot.buy_time).total_seconds()
                    ot.exit_reason = ("cutoff" if past_cutoff
                                      else "profit" if profit_target_hit
                                      else "theo_rich" if theo_rich
                                      else "fair_value")
                    trades.append(ot)
                else:
                    still_open.append(ot)
            open_trades = still_open

        # Enter new position when ask is cheap vs theo
        # Only enter if we don't already have an open position at this price
        # (avoids stacking hundreds of trades during one cheap period)
        if ask is not None:
            if cutoff_time and ts >= cutoff_time:
                continue

            if ask < theo - buy_edge:
                # Skip if we already have an open trade at the same ask price
                already_in = any(ot.buy_price == ask for ot in open_trades)
                if not already_in:
                    open_trades.append(Trade(
                        ticker=ticker,
                        subtitle=subtitle,
                        buy_time=ts,
                        buy_price=ask,
                        buy_theo=theo,
                    ))

    # Force close anything still open at end of data
    for ot in open_trades:
        for snap in reversed(snapshots):
            if snap["bid"] is not None:
                ot.sell_time = snap["ts"]
                ot.sell_price = snap["bid"]
                ot.sell_theo = snap["theo"] or 0
                ot.pnl = snap["bid"] - ot.buy_price
                ot.hold_seconds = (snap["ts"] - ot.buy_time).total_seconds()
                ot.exit_reason = "end_of_data"
                trades.append(ot)
                break

    return trades


# =============================================================================
# Scanner
# =============================================================================

def parse_expiry_from_event(event_ticker: str) -> datetime | None:
    """Parse expiry datetime from event ticker.

    KXBTC-26APR1417 → 2026-04-14 17:00 ET → 21:00 UTC (EDT)
    """
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None

    date_part = parts[1]  # "26APR1417"
    try:
        year = 2000 + int(date_part[:2])
        month_str = date_part[2:5].upper()
        months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                  "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
        month = months.get(month_str, 1)
        day = int(date_part[5:7])
        hour_et = int(date_part[7:9])
        hour_utc = hour_et + 4  # EDT offset
        from datetime import timezone
        return datetime(year, month, day, hour_utc, 0, 0, tzinfo=timezone.utc)
    except Exception:
        return None


def scan_event(
    event_data: dict,
    event_ticker: str,
    buy_edge: float,
    sell_edge: float,
    min_profit: float,
    cutoff_hours: float,
) -> list[BracketResult]:
    """Scan all brackets in an event for round-trip opportunities."""

    # Parse actual expiry from event ticker
    expiry = parse_expiry_from_event(event_ticker)
    if expiry:
        cutoff_time = expiry - timedelta(hours=cutoff_hours)
        print(f"  Expiry: {expiry.strftime('%Y-%m-%d %H:%M')} UTC  "
              f"Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M')} UTC")
    else:
        # Fallback to end of data if can't parse
        all_timestamps = []
        for ticker, tdata in event_data.items():
            for snap in tdata["snapshots"]:
                all_timestamps.append(snap["ts"])
        if not all_timestamps:
            return []
        cutoff_time = max(all_timestamps) - timedelta(hours=cutoff_hours)

    results = []
    for ticker in sorted(event_data.keys(), key=lambda t: event_data[t]["strike"]):
        tdata = event_data[ticker]
        subtitle = tdata["subtitle"]
        strike = tdata["strike"]
        snapshots = tdata["snapshots"]

        if len(snapshots) < 10:
            continue

        trades = simulate_roundtrips(
            snapshots, ticker, subtitle,
            buy_edge, sell_edge, min_profit,
            cutoff_time=cutoff_time,
        )

        if not trades:
            continue

        result = BracketResult(
            ticker=ticker,
            subtitle=subtitle,
            strike=strike,
            trades=trades,
            total_pnl=sum(t.pnl for t in trades),
            win_count=sum(1 for t in trades if t.pnl > 0),
            loss_count=sum(1 for t in trades if t.pnl <= 0),
            avg_hold_seconds=(sum(t.hold_seconds for t in trades) / len(trades)
                              if trades else 0),
        )
        results.append(result)

    # Sort by total PnL descending
    results.sort(key=lambda r: r.total_pnl, reverse=True)
    return results


# =============================================================================
# Print results
# =============================================================================

def print_results(results: list[BracketResult], buy_edge: float, sell_edge: float,
                  min_profit: float):
    """Print a summary table of all brackets with trades."""
    total_pnl = sum(r.total_pnl for r in results)
    total_trades = sum(len(r.trades) for r in results)
    total_wins = sum(r.win_count for r in results)
    total_losses = sum(r.loss_count for r in results)

    print(f"\n{'='*90}")
    print(f"ROUND-TRIP SCANNER RESULTS")
    print(f"  buy_edge=${buy_edge:.3f}  sell_edge=${sell_edge:.3f}  min_profit=${min_profit:.3f}")
    print(f"  {total_trades} trades across {len(results)} brackets")
    print(f"  Total PnL: ${total_pnl:.2f}  |  Wins: {total_wins}  Losses: {total_losses}  "
          f"Win rate: {total_wins/(total_wins+total_losses)*100:.0f}%"
          if (total_wins + total_losses) > 0 else "")
    print(f"{'='*90}")

    print(f"\n{'Bracket':>30s}  {'Trades':>6s}  {'Wins':>4s}  {'Loss':>4s}  "
          f"{'PnL':>8s}  {'PnL/Trade':>10s}  {'AvgHold':>10s}")
    print("-" * 90)

    for r in results:
        avg_pnl = r.total_pnl / len(r.trades) if r.trades else 0
        hold_min = r.avg_hold_seconds / 60
        pnl_color = "" if r.total_pnl >= 0 else ""
        print(f"  {r.subtitle:>28s}  {len(r.trades):>6d}  {r.win_count:>4d}  "
              f"{r.loss_count:>4d}  ${r.total_pnl:>7.2f}  ${avg_pnl:>9.4f}  "
              f"{hold_min:>8.1f}m")

    print("-" * 90)
    print(f"  {'TOTAL':>28s}  {total_trades:>6d}  {total_wins:>4d}  "
          f"{total_losses:>4d}  ${total_pnl:>7.2f}")

    # Top 5 individual trades
    all_trades = []
    for r in results:
        all_trades.extend(r.trades)
    all_trades.sort(key=lambda t: t.pnl, reverse=True)

    if all_trades:
        print(f"\n  Top 5 trades:")
        for t in all_trades[:5]:
            hold = t.hold_seconds / 60
            print(f"    {t.subtitle:>28s}  buy=${t.buy_price:.3f} → sell=${t.sell_price:.3f}  "
                  f"pnl=${t.pnl:+.3f}  hold={hold:.0f}m  exit={t.exit_reason}")

        print(f"\n  Worst 5 trades:")
        for t in all_trades[-5:]:
            hold = t.hold_seconds / 60
            print(f"    {t.subtitle:>28s}  buy=${t.buy_price:.3f} → sell=${t.sell_price:.3f}  "
                  f"pnl=${t.pnl:+.3f}  hold={hold:.0f}m  exit={t.exit_reason}")


# =============================================================================
# Plot best brackets
# =============================================================================

def plot_best_brackets(event_data: dict, results: list[BracketResult],
                       buy_edge: float, sell_edge: float, top_n: int = 10):
    """Interactive chart showing the best brackets with trade markers."""
    if not results:
        print("No results to plot")
        return

    show_results = results[:top_n]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=["Market vs Deribit Theo (with trades)", "Cumulative PnL"],
    )

    trace_sets = []

    for i, r in enumerate(show_results):
        tdata = event_data[r.ticker]
        snapshots = tdata["snapshots"]
        visible = (i == 0)

        timestamps = [s["ts"] for s in snapshots]
        bids = [s["bid"] for s in snapshots]
        asks = [s["ask"] for s in snapshots]
        theos = [s["theo"] for s in snapshots]
        theo_lo = [t - buy_edge if t else None for t in theos]
        theo_hi = [t + sell_edge if t else None for t in theos]

        start_idx = len(fig.data)

        # Kalshi bid
        ts_b = [t for t, b in zip(timestamps, bids) if b is not None]
        v_b = [b for b in bids if b is not None]
        fig.add_trace(go.Scatter(
            x=ts_b, y=v_b, mode="markers",
            marker=dict(size=3, color="steelblue", opacity=0.4),
            name="Kalshi Bid", visible=visible,
        ), row=1, col=1)

        # Kalshi ask
        ts_a = [t for t, a in zip(timestamps, asks) if a is not None]
        v_a = [a for a in asks if a is not None]
        fig.add_trace(go.Scatter(
            x=ts_a, y=v_a, mode="markers",
            marker=dict(size=3, color="#f97316", opacity=0.4),
            name="Kalshi Ask", visible=visible,
        ), row=1, col=1)

        # Deribit theo
        fig.add_trace(go.Scatter(
            x=timestamps, y=theos, mode="lines",
            line=dict(color="#22c55e", width=2),
            name="Deribit Theo", visible=visible,
        ), row=1, col=1)

        # Theo - buy_edge (buy zone)
        fig.add_trace(go.Scatter(
            x=timestamps, y=theo_lo, mode="lines",
            line=dict(color="#22c55e", width=1, dash="dash"),
            name=f"Buy zone (theo-{buy_edge:.3f})", visible=visible,
        ), row=1, col=1)

        # Theo + sell_edge (sell zone)
        fig.add_trace(go.Scatter(
            x=timestamps, y=theo_hi, mode="lines",
            line=dict(color="#ef4444", width=1, dash="dash"),
            name=f"Sell zone (theo+{sell_edge:.3f})", visible=visible,
        ), row=1, col=1)

        # Buy markers
        buy_ts = [t.buy_time for t in r.trades]
        buy_px = [t.buy_price for t in r.trades]
        fig.add_trace(go.Scatter(
            x=buy_ts, y=buy_px, mode="markers",
            marker=dict(symbol="triangle-up", size=11, color="#22c55e",
                        line=dict(width=1, color="white")),
            name=f"BUY ({len(buy_ts)})", visible=visible,
        ), row=1, col=1)

        # Sell markers
        sell_ts = [t.sell_time for t in r.trades if t.sell_time]
        sell_px = [t.sell_price for t in r.trades if t.sell_time]
        fig.add_trace(go.Scatter(
            x=sell_ts, y=sell_px, mode="markers",
            marker=dict(symbol="triangle-down", size=11, color="#ef4444",
                        line=dict(width=1, color="white")),
            name=f"SELL ({len(sell_ts)})", visible=visible,
        ), row=1, col=1)

        # Cumulative PnL
        pnl_ts = [t.sell_time for t in r.trades if t.sell_time]
        cum_pnl = []
        running = 0.0
        for t in r.trades:
            if t.sell_time:
                running += t.pnl
                cum_pnl.append(running)

        fig.add_trace(go.Scatter(
            x=pnl_ts, y=cum_pnl, mode="lines+markers",
            line=dict(color="#22c55e", width=2),
            marker=dict(size=4),
            name="Cumulative PnL", visible=visible,
            fill="tozeroy", fillcolor="rgba(34,197,94,0.1)",
        ), row=2, col=1)

        traces_per = 8
        trace_sets.append((start_idx, traces_per))

    # Zero line on PnL
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)

    total_traces = len(fig.data)

    # Dropdown buttons
    buttons = []
    for i, r in enumerate(show_results):
        vis = [False] * total_traces
        start, count = trace_sets[i]
        for j in range(start, start + count):
            vis[j] = True

        avg_pnl = r.total_pnl / len(r.trades) if r.trades else 0
        buttons.append(dict(
            label=f"{r.subtitle}  PnL=${r.total_pnl:.2f} ({len(r.trades)}t)",
            method="update",
            args=[
                {"visible": vis},
                {"title": dict(
                    text=(f"{r.subtitle} | {len(r.trades)} trades | "
                          f"PnL=${r.total_pnl:.2f} | "
                          f"W/L={r.win_count}/{r.loss_count} | "
                          f"avg=${avg_pnl:.4f}/trade"),
                    font_size=13,
                )},
            ],
        ))

    top = show_results[0]
    fig.update_layout(
        title=dict(
            text=(f"{top.subtitle} | {len(top.trades)} trades | "
                  f"PnL=${top.total_pnl:.2f} | "
                  f"W/L={top.win_count}/{top.loss_count}"),
            font_size=13,
        ),
        template="plotly_dark",
        height=850,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.0, xanchor="left", y=1.15, yanchor="top",
            buttons=buttons,
            bgcolor="#1e2736",
            font=dict(color="white", size=11),
            active=0,
        )],
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    fig.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Round-trip trade scanner")
    parser.add_argument(
        "--dbs", type=str, nargs="+",
        default=[
            "marketdata/market_data_2026-04-13.db",
            "marketdata/market_data_2026-04-14.db",
        ],
    )
    parser.add_argument("--event", type=str, default=None)
    parser.add_argument("--buy-edge", type=float, default=0.02,
                        help="Buy when ask < theo - buy_edge (default: 0.02)")
    parser.add_argument("--sell-edge", type=float, default=0.02,
                        help="Sell when bid > theo + sell_edge (default: 0.02)")
    parser.add_argument("--min-profit", type=float, default=0.01,
                        help="Min profit per contract to take profit (default: 0.01)")
    parser.add_argument("--cutoff-hours", type=float, default=2.0,
                        help="Hours before end of data to stop trading (default: 2)")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of best brackets to plot (default: 10)")
    args = parser.parse_args()

    print("Loading data...")
    data = load_all_data(args.dbs, args.event)
    if not data:
        print("No data found")
        return

    print(f"\nEvents:")
    for ev in sorted(data.keys()):
        print(f"  {ev}: {len(data[ev])} brackets")

    # Pick event
    if args.event:
        event_ticker = args.event
    else:
        event_list = sorted(data.keys())
        if len(event_list) == 1:
            event_ticker = event_list[0]
        else:
            for i, ev in enumerate(event_list):
                print(f"  [{i}] {ev}")
            idx = int(input("\nSelect event number: "))
            event_ticker = event_list[idx]

    print(f"\nScanning {event_ticker}...")
    print(f"  buy_edge=${args.buy_edge:.3f}  sell_edge=${args.sell_edge:.3f}  "
          f"min_profit=${args.min_profit:.3f}  cutoff={args.cutoff_hours}h")

    results = scan_event(
        data[event_ticker], event_ticker,
        args.buy_edge, args.sell_edge, args.min_profit,
        args.cutoff_hours,
    )

    if not results:
        print("No round-trip opportunities found. Try lowering the edges.")
        return

    print_results(results, args.buy_edge, args.sell_edge, args.min_profit)

    print(f"\nPlotting top {args.top} brackets...")
    plot_best_brackets(data[event_ticker], results,
                       args.buy_edge, args.sell_edge, args.top)


if __name__ == "__main__":
    main()
