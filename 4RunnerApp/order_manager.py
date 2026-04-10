"""
Order manager for Kalshi bracket trading.

Tracks contract state (bid/ask from websocket, positions/orders from REST).
Handles order placement (sell YES), cancellation, balance tracking.
Detects fills and triggers email notifications.

Pricing logic:
    sell_price = yes_bid + edge   (post above the bid to rest in the book)
    If no yes bid exists, sell_price = edge (just the edge as the price)
"""

from dataclasses import dataclass
from kalshi_api import KalshiAPI
from notifier import send_fill_notification


@dataclass
class ContractState:
    """Tracks all state for a single bracket contract."""
    ticker: str
    yes_sub_title: str = ""         # e.g. "$71,500 to 71,999.99"
    best_bid: float = 0.0           # best YES bid (from websocket)
    best_ask: float = 0.0           # best YES ask (from websocket)
    position_dollars: float = 0.0   # current position value in $
    open_quantity_dollars: float = 0.0  # resting order value in $
    open_level: float = 0.0         # yes price level of resting sell order
    order_id: str = ""              # order ID of resting order
    # Track previous remaining count to detect fills
    prev_remaining: float = 0.0


class OrderManager:

    def __init__(self, api: KalshiAPI, default_quantity: int = 50):
        self.api = api
        self.default_quantity = default_quantity
        self.contracts: dict[str, ContractState] = {}

        # Balance tracking (updated via REST every 10s)
        self.balance_dollars = 0.0
        self.portfolio_value_dollars = 0.0

    def add_contract(self, ticker: str, yes_sub_title: str = ""):
        """Register a contract to track."""
        if ticker not in self.contracts:
            self.contracts[ticker] = ContractState(
                ticker=ticker, yes_sub_title=yes_sub_title
            )

    def update_book(self, ticker: str, yes_bid: float, yes_ask: float):
        """Called by websocket feed on every orderbook update."""
        state = self.contracts.get(ticker)
        if state:
            state.best_bid = yes_bid
            state.best_ask = yes_ask

    # --- Order Placement ---

    def sell_yes(self, ticker: str, edge: float) -> dict:
        """Sell YES at best_yes_bid + edge.

        Posts as side='yes', action='sell' with yes_price_dollars.
        """
        state = self.contracts.get(ticker)
        if not state:
            return {"error": "Contract not tracked"}

        # Calculate sell price: bid + edge, or just edge if no bid
        if state.best_bid > 0:
            yes_price = round(state.best_bid + edge, 2)
        else:
            yes_price = round(edge, 2)

        if yes_price <= 0 or yes_price >= 1.0:
            return {"error": f"Invalid price: {yes_price}"}

        try:
            result = self.api.create_order(
                ticker=ticker,
                side="yes",
                action="sell",
                price_dollars=f"{yes_price:.2f}",
                count=self.default_quantity,
            )
            order = result.get("order", {})
            state.order_id = order.get("order_id", "")
            state.open_level = yes_price
            state.open_quantity_dollars = yes_price * self.default_quantity
            state.prev_remaining = float(self.default_quantity)
            return result
        except Exception as e:
            return {"error": str(e)}

    # --- Order Cancellation ---

    def cancel_all_orders(self) -> int:
        """Cancel all resting orders."""
        cancelled = 0
        for state in self.contracts.values():
            if state.order_id:
                try:
                    self.api.cancel_order(state.order_id)
                    cancelled += 1
                except Exception:
                    pass
                state.order_id = ""
                state.open_level = 0.0
                state.open_quantity_dollars = 0.0
                state.prev_remaining = 0.0
        return cancelled

    # --- REST Refreshes ---

    def refresh_orders(self):
        """Fetch resting orders and detect fills."""
        try:
            orders = self.api.get_orders(status="resting")

            # Build lookup: ticker -> order
            order_map = {}
            for o in orders:
                t = o.get("ticker", "")
                if t in self.contracts and t not in order_map:
                    order_map[t] = o

            for ticker, state in self.contracts.items():
                if ticker in order_map:
                    o = order_map[ticker]
                    state.order_id = o.get("order_id", "")
                    yes_price = float(o.get("yes_price_dollars", 0))
                    remaining = float(o.get("remaining_count_fp", 0))
                    initial = float(o.get("initial_count_fp", 0))

                    # Detect partial or full fills
                    if state.prev_remaining > 0 and remaining < state.prev_remaining:
                        filled_qty = int(state.prev_remaining - remaining)
                        if filled_qty > 0:
                            send_fill_notification(
                                ticker=ticker,
                                side="yes",
                                action="sell",
                                price=yes_price if yes_price > 0 else state.open_level,
                                quantity=filled_qty,
                                order_id=state.order_id,
                            )

                    state.prev_remaining = remaining
                    state.open_level = yes_price
                    state.open_quantity_dollars = yes_price * remaining
                else:
                    # No resting order — check if it was fully filled
                    if state.order_id and state.prev_remaining > 0:
                        send_fill_notification(
                            ticker=ticker,
                            side="yes",
                            action="sell",
                            price=state.open_level,
                            quantity=int(state.prev_remaining),
                            order_id=state.order_id,
                        )

                    state.order_id = ""
                    state.open_level = 0.0
                    state.open_quantity_dollars = 0.0
                    state.prev_remaining = 0.0
        except Exception:
            pass

    def refresh_positions(self):
        """Fetch positions from REST API."""
        try:
            positions = self.api.get_positions()
            pos_map = {p.get("ticker", ""): p for p in positions}

            for ticker, state in self.contracts.items():
                if ticker in pos_map:
                    p = pos_map[ticker]
                    yes_pos = float(p.get("yes_position_fp", p.get("yes_position", 0)))
                    no_pos = float(p.get("no_position_fp", p.get("no_position", 0)))
                    state.position_dollars = (
                        yes_pos * state.best_bid
                        - no_pos * (1.0 - state.best_bid)
                    )
                else:
                    state.position_dollars = 0.0
        except Exception:
            pass

    def refresh_balance(self):
        """Fetch account balance from REST API."""
        try:
            data = self.api.get_balance()
            # API returns cents, convert to dollars
            self.balance_dollars = float(data.get("balance", 0)) / 100.0
            self.portfolio_value_dollars = float(data.get("portfolio_value", 0)) / 100.0
        except Exception:
            pass