"""
Arbitrage Scanner for Kalshi Bracket Markets

Scans all brackets in an event for two types of guaranteed-profit opportunities:

1. SELL-ALL arb: sum of YES bids across all brackets > $1.00
   → Sell YES on every bracket, one pays out $1, you collected more.

2. BUY-ALL arb: sum of YES asks across all brackets < $1.00
   → Buy YES on every bracket, one pays $1, you paid less.

Handles varying orderbook depth — finds the maximum profitable size
by walking the book. Also considers existing positions to show what
additional trades would complete the hedge.

Usage:
    python3 arb_scanner.py
    python3 arb_scanner.py --series KXBTC
    python3 arb_scanner.py --event KXBTC-26APR1717
    python3 arb_scanner.py --event KXBTC-26APR1717 --depth 20
"""

import argparse
import sys
import time
sys.path.insert(0, "/Users/nikhilr5/Desktop/Kalshi2.0/4RunnerApp")
from kalshi_api import KalshiAPI


# =============================================================================
# Orderbook helpers
# =============================================================================

def compute_avg_fill_price(levels: list[tuple[float, float]], size: int) -> float | None:
    """Compute the average fill price for a given size from orderbook levels.

    levels: [(price, qty), ...] sorted best-to-worst.
    Returns average price per contract, or None if not enough liquidity.
    """
    if size <= 0:
        return None

    remaining = size
    total_cost = 0.0

    for price, qty in levels:
        fill = min(remaining, qty)
        total_cost += fill * price
        remaining -= fill
        if remaining <= 0:
            break

    if remaining > 0:
        return None  # not enough liquidity

    return total_cost / size


def total_available_size(levels: list[tuple[float, float]]) -> int:
    """Total quantity available across all levels."""
    return int(sum(qty for _, qty in levels))


# =============================================================================
# Arb analysis
# =============================================================================

def analyze_event(
    api: KalshiAPI,
    event_ticker: str,
    depth: int = 20,
    positions: dict | None = None,
) -> dict:
    """Analyze an event for arbitrage opportunities.

    Args:
        api:            KalshiAPI instance
        event_ticker:   event to scan
        depth:          orderbook depth to fetch
        positions:      {ticker: qty} existing positions (negative = short YES)

    Returns dict with full analysis results.
    """
    positions = positions or {}

    # Fetch all markets for this event
    markets = api.get_markets_for_event(event_ticker)
    markets = [m for m in markets if m.get("status") == "active"]
    markets.sort(key=lambda m: m.get("strike_value", 0) or 0)

    if not markets:
        return {"error": "No active markets found", "event": event_ticker}

    print(f"\n{'='*70}")
    print(f"Event: {event_ticker}  ({len(markets)} active brackets)")
    print(f"{'='*70}")

    # Fetch orderbooks for all brackets
    brackets = []
    for m in markets:
        ticker = m["ticker"]
        subtitle = m.get("yes_sub_title", ticker)
        try:
            book = api.get_orderbook(ticker, depth=depth)
            time.sleep(0.05)  # rate limit
        except Exception as e:
            print(f"  [WARN] Failed to fetch {ticker}: {e}")
            book = {"yes": [], "no": []}

        # YES bids = what you can sell YES into
        # NO bids at price P = YES ask at (1 - P)
        yes_bids = book["yes"]   # [(price, qty), ...] best first
        no_bids = book["no"]     # [(price, qty), ...]

        # Convert NO bids to YES asks: price = 1.00 - no_bid_price
        yes_asks = [(round(1.0 - p, 2), q) for p, q in no_bids]
        yes_asks.sort(key=lambda x: x[0])  # sort by price ascending (cheapest first)

        pos_qty = positions.get(ticker, 0)

        brackets.append({
            "ticker": ticker,
            "subtitle": subtitle,
            "yes_bids": yes_bids,
            "yes_asks": yes_asks,
            "best_bid": yes_bids[0][0] if yes_bids else 0.0,
            "best_ask": yes_asks[0][0] if yes_asks else 1.0,
            "bid_depth": total_available_size(yes_bids),
            "ask_depth": total_available_size(yes_asks),
            "position": pos_qty,
        })

    # --- Print current market state ---
    print(f"\n{'Bracket':>30s}  {'Bid':>7s}  {'Ask':>7s}  "
          f"{'BidQty':>6s}  {'AskQty':>6s}  {'Pos':>5s}")
    print("-" * 85)

    total_best_bid = 0.0
    total_best_ask = 0.0
    for b in brackets:
        total_best_bid += b["best_bid"]
        total_best_ask += b["best_ask"]
        pos_str = str(b["position"]) if b["position"] != 0 else ""
        print(f"  {b['subtitle']:>28s}  ${b['best_bid']:.2f}   ${b['best_ask']:.2f}   "
              f"{b['bid_depth']:>5d}   {b['ask_depth']:>5d}   {pos_str:>5s}")

    print("-" * 85)
    print(f"  {'TOTAL':>28s}  ${total_best_bid:.2f}   ${total_best_ask:.2f}")

    # --- Quick check at top-of-book ---
    sell_all_edge = total_best_bid - 1.0
    buy_all_edge = 1.0 - total_best_ask

    print(f"\n  Sell-all (top of book): ${total_best_bid:.4f}  "
          f"{'✓ ARB' if sell_all_edge > 0 else '✗ no arb'}  "
          f"edge=${sell_all_edge:+.4f}")
    print(f"  Buy-all  (top of book): ${total_best_ask:.4f}  "
          f"{'✓ ARB' if buy_all_edge > 0 else '✗ no arb'}  "
          f"edge=${buy_all_edge:+.4f}")

    # --- Size analysis: find max profitable size ---
    print(f"\n  Size analysis (sell-all):")
    print(f"  {'Size':>6s}  {'TotalPremium':>13s}  {'Edge/contract':>14s}  "
          f"{'TotalProfit':>12s}  {'Status':>8s}")

    sell_results = []
    for size in [1, 5, 10, 25, 50, 100, 200, 500]:
        total_premium = 0.0
        feasible = True

        for b in brackets:
            avg = compute_avg_fill_price(b["yes_bids"], size)
            if avg is None:
                feasible = False
                break
            total_premium += avg

        if feasible:
            edge = total_premium - 1.0
            profit = edge * size
            status = "✓ ARB" if edge > 0 else "✗"
            print(f"  {size:>6d}  ${total_premium:>11.4f}  ${edge:>12.4f}  "
                  f"${profit:>10.2f}  {status:>8s}")
            sell_results.append({"size": size, "premium": total_premium,
                                 "edge": edge, "profit": profit})
        else:
            print(f"  {size:>6d}  {'insufficient liquidity':>40s}")
            break

    print(f"\n  Size analysis (buy-all):")
    print(f"  {'Size':>6s}  {'TotalCost':>13s}  {'Edge/contract':>14s}  "
          f"{'TotalProfit':>12s}  {'Status':>8s}")

    buy_results = []
    for size in [1, 5, 10, 25, 50, 100, 200, 500]:
        total_cost = 0.0
        feasible = True

        for b in brackets:
            avg = compute_avg_fill_price(b["yes_asks"], size)
            if avg is None:
                feasible = False
                break
            total_cost += avg

        if feasible:
            edge = 1.0 - total_cost
            profit = edge * size
            status = "✓ ARB" if edge > 0 else "✗"
            print(f"  {size:>6d}  ${total_cost:>11.4f}  ${edge:>12.4f}  "
                  f"${profit:>10.2f}  {status:>8s}")
            buy_results.append({"size": size, "cost": total_cost,
                                "edge": edge, "profit": profit})
        else:
            print(f"  {size:>6d}  {'insufficient liquidity':>40s}")
            break

    # --- Position-aware analysis ---
    if any(b["position"] != 0 for b in brackets):
        print(f"\n  Position-aware analysis:")
        _analyze_with_positions(brackets)

    return {
        "event": event_ticker,
        "brackets": brackets,
        "sell_all_edge_tob": sell_all_edge,
        "buy_all_edge_tob": buy_all_edge,
        "sell_results": sell_results,
        "buy_results": buy_results,
    }


def _analyze_with_positions(brackets: list[dict]):
    """Analyze what trades are needed to complete the hedge given positions.

    If you're already short YES on some brackets, shows what you need
    to sell on the remaining brackets (and at what sizes) to have a
    fully hedged portfolio.
    """
    # Find the minimum short YES size across brackets with positions
    short_sizes = {}
    for b in brackets:
        if b["position"] < 0:
            short_sizes[b["ticker"]] = abs(b["position"])

    if not short_sizes:
        print("    No existing short YES positions.")
        return

    min_short = min(short_sizes.values())
    max_short = max(short_sizes.values())

    print(f"    Existing shorts: {len(short_sizes)}/{len(brackets)} brackets")
    print(f"    Short sizes: min={min_short}, max={max_short}")

    # For the minimum size, compute what you'd need on the missing brackets
    target_size = min_short
    missing = [b for b in brackets if b["ticker"] not in short_sizes]
    uneven = [b for b in brackets
              if b["ticker"] in short_sizes and short_sizes[b["ticker"]] > target_size]

    print(f"\n    To complete hedge at size {target_size}:")
    total_additional_premium = 0.0
    total_existing_premium = 0.0
    all_feasible = True

    # Premium from existing positions (estimate using current best bid)
    for b in brackets:
        if b["ticker"] in short_sizes:
            # Already short — use the price they were sold at (unknown, use current bid as estimate)
            total_existing_premium += b["best_bid"]

    # What we need to sell on missing brackets
    if missing:
        print(f"    Need to SELL YES on {len(missing)} brackets:")
        for b in missing:
            avg = compute_avg_fill_price(b["yes_bids"], target_size)
            if avg is not None:
                total_additional_premium += avg
                print(f"      {b['subtitle']:>28s}  sell {target_size} @ ~${avg:.2f}  "
                      f"(depth: {b['bid_depth']})")
            else:
                all_feasible = False
                print(f"      {b['subtitle']:>28s}  ✗ insufficient liquidity "
                      f"(need {target_size}, have {b['bid_depth']})")

    total = total_existing_premium + total_additional_premium
    if all_feasible:
        edge = total - 1.0
        print(f"\n    Estimated total premium: ${total:.4f} "
              f"(existing ~${total_existing_premium:.4f} + "
              f"new ~${total_additional_premium:.4f})")
        print(f"    Edge: ${edge:+.4f} per contract  "
              f"{'✓ PROFITABLE' if edge > 0 else '✗ NOT PROFITABLE'}")
    else:
        print(f"\n    ✗ Cannot complete hedge — insufficient liquidity on some brackets")

    # Also show opportunities to equalize uneven positions
    if uneven:
        excess = sum(short_sizes[b["ticker"]] - target_size for b in uneven)
        print(f"\n    Note: {len(uneven)} brackets have excess short "
              f"({excess} total contracts above min size)")
        print(f"    These are unhedged and carry directional risk.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Kalshi bracket arb scanner")
    parser.add_argument("--series", type=str, default="KXBTC",
                        help="Series ticker to scan (default: KXBTC)")
    parser.add_argument("--event", type=str, default=None,
                        help="Specific event ticker (skips event selection)")
    parser.add_argument("--depth", type=int, default=20,
                        help="Orderbook depth to fetch (default: 20)")
    parser.add_argument("--with-positions", action="store_true",
                        help="Include current portfolio positions in analysis")
    args = parser.parse_args()

    api = KalshiAPI()

    # Get existing positions if requested
    positions = {}
    if args.with_positions:
        print("Fetching positions...")
        try:
            pos_list = api.get_positions()
            for p in pos_list:
                ticker = p.get("ticker", "")
                qty = int(float(p.get("position_fp", 0)))
                if qty != 0:
                    positions[ticker] = qty
            print(f"  {len(positions)} positions loaded")
        except Exception as e:
            print(f"  [WARN] Could not fetch positions: {e}")

    if args.event:
        # Scan a specific event
        analyze_event(api, args.event, depth=args.depth, positions=positions)
    else:
        # Find all active events for the series
        print(f"Scanning {args.series} for active events...")
        markets = api.get_markets(series_ticker=args.series, status="active")

        # Group by event
        events = {}
        for m in markets:
            ev = m.get("event_ticker", "")
            if ev:
                events.setdefault(ev, []).append(m)

        if not events:
            print("No active events found")
            return

        print(f"Found {len(events)} active events:")
        for ev, mkt_list in sorted(events.items()):
            print(f"  {ev} ({len(mkt_list)} brackets)")

        # Analyze each event
        results = []
        for ev in sorted(events.keys()):
            result = analyze_event(api, ev, depth=args.depth, positions=positions)
            results.append(result)

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for r in results:
            if "error" in r:
                continue
            ev = r["event"]
            sell_edge = r["sell_all_edge_tob"]
            buy_edge = r["buy_all_edge_tob"]
            sell_flag = " ← ARB!" if sell_edge > 0 else ""
            buy_flag = " ← ARB!" if buy_edge > 0 else ""
            print(f"  {ev}  sell-edge=${sell_edge:+.4f}{sell_flag}  "
                  f"buy-edge=${buy_edge:+.4f}{buy_flag}")


if __name__ == "__main__":
    main()
