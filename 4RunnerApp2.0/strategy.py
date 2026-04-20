"""
Two-sided market making strategy with flatten-at-theo.

Quoting modes depending on position:

  position == 0 (flat):
      Bid @ bid_theo - edge_bid
      Ask @ ask_theo + edge_ask

  position < 0 (short, need to buy to flatten):
      Buy (flatten) @ bid_theo  (no edge — aggressive)
      Buy size = abs(position)                 — full size to get flat
      Sell (continue) @ ask_theo + edge_ask    — keep earning spread

  position > 0 (long, need to sell to flatten):
      Sell (flatten) @ ask_theo  (no edge — aggressive)
      Sell size = position                     — full size to get flat
      Buy (continue) @ bid_theo - edge_bid     — keep earning spread

P&L per round trip ≈ edge (captured on open, gave up 0 on close).

Order protection (non-flatten orders only):

  1. Post-only / no-cross: if the computed price would cross the
     opposite side of the Kalshi book (buy >= ask, sell <= bid), the
     order is skipped entirely to avoid taking liquidity.

  2. Price improvement cap: if the computed price is more aggressive
     than the current BBO (buy > best bid, sell < best ask), the price
     is capped at the BBO.  We join the best level rather than
     improving it, avoiding unnecessary price improvement that gives
     away edge.

  3. Lonely-at-BBO pull-back: if our resting order sits at the BBO and
     the total size at that level is <= our resting size (meaning we
     are the only liquidity there), the order is cancelled.  Being the
     sole order at the inside exposes us to adverse selection — we only
     want to post when there is other liquidity providing cover at the
     same level.  The order will be re-placed on the next tick if other
     participants rejoin.

  Flatten orders bypass the post-only check only.  Price improvement
  cap and lonely-BBO pull-back apply to ALL orders including flattens:
  we never post a better price than the current BBO, and we never sit
  alone at the inside.  If the BBO moves away, we follow it back.

Flatten walk (configurable per strike):

  When enabled (flatten_walk_interval > 0), flatten orders start at
  theo and walk toward the market over time:

    Sell (long flatten): starts at ask_theo, walks DOWN by
        flatten_walk_step every flatten_walk_interval seconds,
        floored at the kalshi_bid.

    Buy (short flatten): starts at bid_theo, walks UP by
        flatten_walk_step every flatten_walk_interval seconds,
        capped at the kalshi_ask.

  This gives the market a chance to come to you before you give up
  edge.  If nobody fills you, you converge to the BBO after
  (theo_distance / step) * interval seconds.

  The walk timer resets when position returns to zero.

Usage:
    strat = Strategy(ticker="KXBTCD-26APR1517-T83799.99",
                     strike=83800.0,
                     edge_bid=0.03, edge_ask=0.03,
                     size_bid=10, size_ask=10,
                     max_position=50, api=api)
    strat.start()
    strat.update_theo(bid_theo=0.05, ask_theo=0.06)
    strat.stop()
"""

import math
import time
import threading
from kalshi_api import KalshiAPI


class Strategy:

    def __init__(self, ticker: str, strike: float,
                 edge_bid: float, edge_ask: float,
                 size_bid: int, size_ask: int,
                 max_position: int, api: KalshiAPI,
                 tolerance: float = 0.01, on_max_position=None):
        self.ticker = ticker
        self.strike = strike
        self.edge_bid = edge_bid
        self.edge_ask = edge_ask
        self.size_bid = size_bid
        self.size_ask = size_ask
        self.max_position = max_position
        self.tolerance = tolerance
        self.api = api
        self.on_max_position = on_max_position
        self._lock = threading.Lock()

        self.active = False
        self.bid_active = False   # bid side enabled
        self.ask_active = False   # ask side enabled

        # Sell side (ask) state
        self.resting_sell_id: str | None = None
        self.current_sell_price: float | None = None

        # Buy side (bid) state
        self.resting_buy_id: str | None = None
        self.current_buy_price: float | None = None

        self.position: int = 0
        self.exposure: float = 0.0
        self.realized_pnl: float = 0.0
        self._at_max_logged: bool = False

        # Kalshi book — updated by app before each update_theo call.
        # Used for post-only checks, price-improvement caps, and
        # lonely-BBO detection (see module docstring).
        self.kalshi_bid: float = 0.0
        self.kalshi_ask: float = 0.0
        self.kalshi_bid_size: int = 0   # total size resting at best bid
        self.kalshi_ask_size: int = 0   # total size resting at best ask

        # Track our resting order sizes for lonely-BBO comparison.
        # Set when placing, cleared when cancelling.
        self.resting_buy_count: int = 0
        self.resting_sell_count: int = 0

        # Flatten walk — when enabled, flatten orders start at theo and
        # walk toward the market by flatten_walk_step every
        # flatten_walk_interval seconds until filled or reaching the BBO.
        # Set interval to 0 to disable (default: sit at theo forever).
        self.flatten_walk_interval: float = 0.0   # seconds between steps
        self.flatten_walk_step: float = 0.01       # price step per walk
        self._flatten_sell_start: float | None = None  # monotonic time
        self._flatten_buy_start: float | None = None
        self._flatten_sell_base: float = 0.0  # locked starting price for sell walk
        self._flatten_buy_base: float = 0.0   # locked starting price for buy walk

        # Average entry price of current position — set by app from fills.
        # Used to determine if a flatten order can walk past the BBO
        # while still being profitable.
        self.avg_entry: float = 0.0

    # --- Convenience properties for backward compat ---
    @property
    def edge(self):
        return self.edge_ask

    @property
    def size(self):
        return self.size_ask

    def start(self, bid: bool = True, ask: bool = True):
        """Activate one or both sides. Orders placed on next update_theo call."""
        self.bid_active = bid
        self.ask_active = ask
        self.active = bid or ask

    def stop(self, bid: bool = True, ask: bool = True):
        """Deactivate one or both sides and cancel their resting orders."""
        with self._lock:
            if bid:
                self.bid_active = False
                self._cancel_buy()
            if ask:
                self.ask_active = False
                self._cancel_sell()
            self.active = self.bid_active or self.ask_active

    def on_fill(self):
        """Called on any fill. Position is already updated by caller.

        Do NOT clear order IDs here — the order may be partially filled
        and still resting. Let update_theo handle repricing: it will
        cancel the real order (by ID) and replace if needed.
        """
        pass

    def update_params(self, edge_bid: float, edge_ask: float,
                      size_bid: int, size_ask: int,
                      max_position: int, tolerance: float = 0.01,
                      flatten_walk_interval: float = 0.0,
                      flatten_walk_step: float = 0.01):
        """Update tunable parameters. Next update_theo will reprice if needed."""
        self.edge_bid = edge_bid
        self.edge_ask = edge_ask
        self.size_bid = size_bid
        self.size_ask = size_ask
        self.max_position = max_position
        self.tolerance = tolerance
        self.flatten_walk_interval = flatten_walk_interval
        self.flatten_walk_step = flatten_walk_step

    def update_theo(self, bid_theo: float, ask_theo: float):
        """Core logic — thread-safe. Called from Coinbase and Deribit WS threads.

        bid_theo: computed from Deribit bid IV + spot bid
        ask_theo: computed from Deribit ask IV + spot ask

        Three modes based on position:
          flat (0):   normal two-sided MM with edges
          short (<0): flatten buy at theo, continue sell with edge
          long (>0):  flatten sell at theo, continue buy with edge
        """
        if not self.active:
            return
        if not self._lock.acquire(blocking=False):
            return  # another thread is already updating, skip this tick

        try:
            self._update_theo_locked(bid_theo, ask_theo)
        finally:
            self._lock.release()

    def _update_theo_locked(self, bid_theo: float, ask_theo: float):
        """Actual update logic, called under lock."""
        pos = self.position
        at_max = abs(pos) >= self.max_position
        remaining = max(self.max_position - abs(pos), 0)

        if at_max and not self._at_max_logged:
            self._at_max_logged = True
            side = "LONG" if pos > 0 else "SHORT"
            print(f"[Strategy] {self.strike:,.0f} MAX POSITION hit "
                  f"({pos}), pulling {side} side")
            if self.on_max_position:
                self.on_max_position(self.ticker)
        elif not at_max and self._at_max_logged:
            self._at_max_logged = False
            print(f"[Strategy] {self.strike:,.0f} back under max position "
                  f"({pos}), resuming both sides")

        if pos == 0:
            # --- FLAT: reset flatten walk timers ---
            self._flatten_sell_start = None
            self._flatten_buy_start = None
            self._flatten_sell_base = 0.0
            self._flatten_buy_base = 0.0
            # --- FLAT: normal two-sided MM ---
            if self.ask_active:
                self._quote_sell(ask_theo, self.edge_ask, min(self.size_ask, remaining))
            else:
                self._cancel_sell()
            if self.bid_active:
                self._quote_buy(bid_theo, self.edge_bid, min(self.size_bid, remaining))
            else:
                self._cancel_buy()

        elif pos < 0:
            # --- SHORT: flatten buy, pull sell if at max ---
            if self.bid_active:
                self._quote_buy(bid_theo, 0.0, abs(pos), flatten=True)
            else:
                self._cancel_buy()
            if at_max:
                self._cancel_sell()
            elif self.ask_active:
                self._quote_sell(ask_theo, self.edge_ask, min(self.size_ask, remaining))
            else:
                self._cancel_sell()

        else:
            # --- LONG: flatten sell, pull buy if at max ---
            if self.ask_active:
                self._quote_sell(ask_theo, 0.0, pos, flatten=True)
            else:
                self._cancel_sell()
            if at_max:
                self._cancel_buy()
            elif self.bid_active:
                self._quote_buy(bid_theo, self.edge_bid, min(self.size_bid, remaining))
            else:
                self._cancel_buy()

    def _quote_sell(self, theo: float, edge: float, size: int, flatten: bool = False):
        """Compute sell price and place/reprice if needed.

        Price improvement cap applies to ALL orders (flatten and normal):
        never improve the ask — join it at most.  If the BBO moves away
        (ask rises), follow it back so we're never better than best.

        Additional non-flatten checks:
          1. Post-only: skip if price would cross the bid.
          2. Lonely BBO: cancel if we are the sole order at the inside ask.

        For flatten orders with walk enabled, the price starts at theo
        and walks down toward the ask BBO by flatten_walk_step every
        flatten_walk_interval seconds, floored at the ask.
        """
        if size <= 0 or theo <= 0:
            self._cancel_sell()
            return
        new_sell = math.ceil((theo + edge) * 100) / 100  # round up (outward)

        # Flatten walk: lock in starting price, walk toward the ask BBO over time.
        # The base price is frozen when the walk begins so that theo
        # fluctuations don't cause the walked price to jitter.
        if flatten and self.flatten_walk_interval > 0:
            now = time.monotonic()
            if self._flatten_sell_start is None:
                self._flatten_sell_start = now
                self._flatten_sell_base = new_sell  # lock in starting price
            new_sell = self._flatten_sell_base
            elapsed = now - self._flatten_sell_start
            steps = int(elapsed / self.flatten_walk_interval)
            if steps > 0:
                new_sell = round(new_sell - steps * self.flatten_walk_step, 2)

        new_sell = max(0.01, min(0.99, new_sell))

        # Price improvement cap:
        # - Normally: never improve the ask, join it.
        # - Flatten + profitable (>2c after fees): allow improving the ask,
        #   but floor at one tick above the bid (don't cross the book).
        #   The 2c threshold covers the taker fee so we don't cross the
        #   spread only to give the profit away in fees.
        if self.kalshi_ask > 0 and new_sell < self.kalshi_ask:
            take_price = max(new_sell, self.kalshi_bid) if self.kalshi_bid > 0 else new_sell
            pnl_per = take_price - self.avg_entry if self.avg_entry > 0 else 0
            if (flatten and pnl_per > 0.02
                    and self.kalshi_bid > 0):
                # Profitable flatten after fees — allow taking the bid
                floor = self.kalshi_bid
                new_sell = max(new_sell, floor)
                print(f"[Strategy] {self.strike:,.0f} flatten sell TAKE: "
                      f"${new_sell:.2f} (entry=${self.avg_entry:.2f}, "
                      f"pnl/ct=${pnl_per:.2f}, bid=${floor:.2f})")
            else:
                new_sell = self.kalshi_ask

        # Lonely BBO (ALL orders): if we're the only one at the ask, pull back.
        # Skip this check if we're intentionally improving the ask (below it).
        if (self.resting_sell_id is not None
                and self.current_sell_price is not None
                and self.kalshi_ask > 0
                and abs(self.current_sell_price - self.kalshi_ask) < 0.001
                and self.kalshi_ask_size <= self.resting_sell_count):
            print(f"[Strategy] {self.strike:,.0f} alone at ask "
                  f"${self.kalshi_ask:.2f}, pulling sell")
            self._cancel_sell()
            return

        if not flatten:
            # Post-only: don't cross the book (strict: equal is joining, not crossing)
            if self.kalshi_bid > 0 and new_sell < self.kalshi_bid:
                print(f"[Strategy] {self.strike:,.0f} sell ${new_sell:.2f} would cross "
                      f"bid ${self.kalshi_bid:.2f}, skipping")
                return

        if self._should_reprice_sell(new_sell):
            if self._cancel_sell():
                self._place_sell(new_sell, size)

    def _quote_buy(self, theo: float, edge: float, size: int, flatten: bool = False):
        """Compute buy price and place/reprice if needed.

        Price improvement cap applies to ALL orders (flatten and normal):
        never improve the bid — join it at most.  If the BBO moves away
        (bid drops), follow it back so we're never better than best.

        Additional non-flatten checks:
          1. Post-only: skip if price would cross the ask.
          2. Lonely BBO: cancel if we are the sole order at the inside bid.

        For flatten orders with walk enabled, the price starts at theo
        and walks up toward the bid BBO by flatten_walk_step every
        flatten_walk_interval seconds, capped at the bid.
        """
        if size <= 0 or theo <= 0:
            self._cancel_buy()
            return
        new_buy = math.floor((theo - edge) * 100) / 100  # round down (outward)

        # Flatten walk: lock in starting price, walk toward the bid BBO over time.
        # The base price is frozen when the walk begins so that theo
        # fluctuations don't cause the walked price to jitter.
        if flatten and self.flatten_walk_interval > 0:
            now = time.monotonic()
            if self._flatten_buy_start is None:
                self._flatten_buy_start = now
                self._flatten_buy_base = new_buy  # lock in starting price
            new_buy = self._flatten_buy_base
            elapsed = now - self._flatten_buy_start
            steps = int(elapsed / self.flatten_walk_interval)
            if steps > 0:
                new_buy = round(new_buy + steps * self.flatten_walk_step, 2)

        new_buy = max(0.01, min(0.99, new_buy))

        # Price improvement cap:
        # - Normally: never improve the bid, join it.
        # - Flatten + profitable (>2c after fees): allow improving the bid,
        #   but ceiling at one tick below the ask (don't cross the book).
        #   The 2c threshold covers the taker fee so we don't cross the
        #   spread only to give the profit away in fees.
        if self.kalshi_bid > 0 and new_buy > self.kalshi_bid:
            take_price = min(new_buy, self.kalshi_ask) if self.kalshi_ask > 0 else new_buy
            pnl_per = self.avg_entry - take_price if self.avg_entry > 0 else 0
            if (flatten and pnl_per > 0.02
                    and self.kalshi_ask > 0):
                # Profitable flatten after fees — allow taking the ask
                ceiling = self.kalshi_ask
                new_buy = min(new_buy, ceiling)
                print(f"[Strategy] {self.strike:,.0f} flatten buy TAKE: "
                      f"${new_buy:.2f} (entry=${self.avg_entry:.2f}, "
                      f"pnl/ct=${pnl_per:.2f}, ask=${ceiling:.2f})")
            else:
                new_buy = self.kalshi_bid

        # Lonely BBO (ALL orders): if we're the only one at the bid, pull back.
        # Skip this check if we're intentionally improving the bid (above it).
        if (self.resting_buy_id is not None
                and self.current_buy_price is not None
                and self.kalshi_bid > 0
                and abs(self.current_buy_price - self.kalshi_bid) < 0.001
                and self.kalshi_bid_size <= self.resting_buy_count):
            print(f"[Strategy] {self.strike:,.0f} alone at bid "
                  f"${self.kalshi_bid:.2f}, pulling buy")
            self._cancel_buy()
            return

        if not flatten:
            # Post-only: don't cross the book (strict: equal is joining, not crossing)
            if self.kalshi_ask > 0 and new_buy > self.kalshi_ask:
                print(f"[Strategy] {self.strike:,.0f} buy ${new_buy:.2f} would cross "
                      f"ask ${self.kalshi_ask:.2f}, skipping")
                return

        if self._should_reprice_buy(new_buy):
            if self._cancel_buy():
                self._place_buy(new_buy, size)

    def _should_reprice_sell(self, new_price: float) -> bool:
        """Check if sell order needs repricing."""
        if self.current_sell_price is None:
            return True
        diff = new_price - self.current_sell_price
        if diff < 0:
            # Improving (lowering sell) — always allow if >= 1 tick
            return abs(diff) >= 0.01
        else:
            # Worsening (raising sell) — require tolerance
            return diff >= self.tolerance

    def _should_reprice_buy(self, new_price: float) -> bool:
        """Check if buy order needs repricing."""
        if self.current_buy_price is None:
            return True
        diff = new_price - self.current_buy_price
        if diff > 0:
            # Improving (raising buy) — always allow if >= 1 tick
            return abs(diff) >= 0.01
        else:
            # Worsening (lowering buy) — require tolerance
            return abs(diff) >= self.tolerance

    def _place_sell(self, price: float, count: int):
        """Place a Sell Yes limit order."""
        if self.resting_sell_id is not None:
            # Safety: don't place if we still think an order is out there
            return
        try:
            resp = self.api.create_order(
                ticker=self.ticker,
                side="yes",
                action="sell",
                price_dollars=f"{price:.2f}",
                count=count,
            )
            order = resp.get("order", {})
            self.resting_sell_id = order.get("order_id")
            self.current_sell_price = price
            self.resting_sell_count = count
            print(f"[Strategy] {self.strike:,.0f} SELL YES @ ${price:.2f} "
                  f"x{count}  pos={self.position}  order={self.resting_sell_id}")
        except Exception as e:
            # Place may have succeeded on Kalshi despite error — set price
            # so _should_reprice blocks duplicates. _cancel_sell will clean up.
            print(f"[Strategy] {self.strike:,.0f} sell order failed: {e}")
            self.current_sell_price = price

    def _place_buy(self, price: float, count: int):
        """Place a Buy Yes limit order."""
        if self.resting_buy_id is not None:
            # Safety: don't place if we still think an order is out there
            return
        try:
            resp = self.api.create_order(
                ticker=self.ticker,
                side="yes",
                action="buy",
                price_dollars=f"{price:.2f}",
                count=count,
            )
            order = resp.get("order", {})
            self.resting_buy_id = order.get("order_id")
            self.current_buy_price = price
            self.resting_buy_count = count
            print(f"[Strategy] {self.strike:,.0f} BUY YES @ ${price:.2f} "
                  f"x{count}  pos={self.position}  order={self.resting_buy_id}")
        except Exception as e:
            # Place may have succeeded on Kalshi despite error — set price
            # so _should_reprice blocks duplicates. _cancel_sell will clean up.
            print(f"[Strategy] {self.strike:,.0f} buy order failed: {e}")
            self.current_buy_price = price

    def _cancel_sell(self) -> bool:
        """Cancel the resting sell order. Returns True if cancelled or no order."""
        if self.resting_sell_id is None:
            return True
        try:
            self.api.cancel_order(self.resting_sell_id)
            print(f"[Strategy] {self.strike:,.0f} cancelled sell {self.resting_sell_id}")
        except Exception as e:
            err = str(e)
            print(f"[Strategy] {self.strike:,.0f} cancel sell failed: {e}")
            if "404" in err or "400" in err or "not_found" in err.lower():
                # Order already filled/cancelled on Kalshi — safe to clear
                pass
            else:
                return False
        self.resting_sell_id = None
        self.current_sell_price = None
        self.resting_sell_count = 0
        return True

    def _cancel_buy(self) -> bool:
        """Cancel the resting buy order. Returns True if cancelled or no order."""
        if self.resting_buy_id is None:
            return True
        try:
            self.api.cancel_order(self.resting_buy_id)
            print(f"[Strategy] {self.strike:,.0f} cancelled buy {self.resting_buy_id}")
        except Exception as e:
            err = str(e)
            print(f"[Strategy] {self.strike:,.0f} cancel buy failed: {e}")
            if "404" in err or "400" in err or "not_found" in err.lower():
                # Order already filled/cancelled on Kalshi — safe to clear
                pass
            else:
                return False
        self.resting_buy_id = None
        self.current_buy_price = None
        self.resting_buy_count = 0
        return True
