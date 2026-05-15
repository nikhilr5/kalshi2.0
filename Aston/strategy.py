"""Two-sided market making for a single 15-min up/down market.

Hold-to-close — there is NO flatten logic.  Positions accumulated
during the window settle at expiry against the TWAP.  Edge math runs
per fill (avg edge × fill count over enough trades), not per round
trip, because we don't intend to round-trip during the window.

Quoting rules (per tick):

  Bid @ theo - edge_bid   (buy yes if filled cheaper than fair)
  Ask @ theo + edge_ask   (sell yes if filled richer than fair)

Both sides are skipped when at max_position on that side.  No
auto-flatten — the position rides to settlement.

Order protections (apply to both sides):

  1. Post-only: if the computed price would cross the opposite book,
     skip placement entirely.  Kalshi's `post_only` flag enforces this
     server-side too as a backstop.

  2. Price-improvement cap: never improve past the BBO.  If our
     computed bid > best bid, cap at best bid; mirror for asks.  We
     join, never improve.

  3. Lonely-at-BBO pull-back: if our resting order is the sole size
     at the inside, cancel.  Without cover, we're a free pickoff for
     informed flow.

Async order management — REST writes go through
`api.{create,cancel}_order_async`.  Flags `_pending_*` guard against
double-issue while a request is on the wire.

Usage:
    strat = Strategy(ticker="KXBTC15M-26MAY131315",
                     strike=79064.20,
                     edge_bid=0.03, edge_ask=0.03,
                     size_bid=10, size_ask=10,
                     max_position=50, api=api)
    strat.start()
    strat.update_theo(theo=0.55)
    ...
    strat.stop()
"""

import math
import threading
from kalshi_api import KalshiAPI


class Strategy:

    def __init__(self, ticker: str, strike: float,
                 edge_bid: float, edge_ask: float,
                 size_bid: int, size_ask: int,
                 max_position: int, api: KalshiAPI,
                 tolerance: float = 0.01):
        self.ticker = ticker
        self.strike = strike
        self.edge_bid = edge_bid
        self.edge_ask = edge_ask
        self.size_bid = size_bid
        self.size_ask = size_ask
        self.max_position = max_position
        self.tolerance = tolerance
        self.api = api
        self._lock = threading.Lock()

        self.active = False
        self.bid_active = False
        self.ask_active = False

        # Sell side state
        self.resting_sell_id: str | None = None
        self.current_sell_price: float | None = None
        self.current_sell_fair: float | None = None

        # Buy side state
        self.resting_buy_id: str | None = None
        self.current_buy_price: float | None = None
        self.current_buy_fair: float | None = None

        # In-flight guards — REST writes are async; these prevent
        # double cancel/place while the previous one is on the wire.
        self._pending_cancel_sell: bool = False
        self._pending_cancel_buy: bool = False
        self._pending_place_sell: bool = False
        self._pending_place_buy: bool = False

        # Position state — net YES contracts (positive = long).
        self.position: int = 0
        self.pending_buy_size: int = 0
        self.pending_sell_size: int = 0
        self.resting_buy_count: int = 0
        self.resting_sell_count: int = 0

        # Kalshi book — refreshed by caller (app) before each
        # update_theo call.  Drives post-only, improve-cap, lonely-BBO.
        self.kalshi_bid: float = 0.0
        self.kalshi_ask: float = 0.0
        self.kalshi_bid_size: int = 0
        self.kalshi_ask_size: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, bid: bool = True, ask: bool = True):
        """Activate one or both sides.  Orders post on next update_theo."""
        self.bid_active = bid
        self.ask_active = ask
        self.active = bid or ask

    def stop(self, bid: bool = True, ask: bool = True):
        """Deactivate side(s) and cancel resting orders."""
        with self._lock:
            if bid:
                self.bid_active = False
                self._cancel_buy()
            if ask:
                self.ask_active = False
                self._cancel_sell()
            self.active = self.bid_active or self.ask_active

    def update_params(self, edge_bid: float, edge_ask: float,
                      size_bid: int, size_ask: int,
                      max_position: int, tolerance: float = 0.01):
        """Update tunables.  Next tick will reprice if needed."""
        self.edge_bid = edge_bid
        self.edge_ask = edge_ask
        self.size_bid = size_bid
        self.size_ask = size_ask
        self.max_position = max_position
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def on_fill(self, action: str = "", price: float = 0.0,
                count: float = 0, side: str = "yes"):
        """Called when a fill arrives.  Position is already updated by
        the caller — we don't clear order IDs here since the resting
        order may be partially filled and still alive.  Next quote tick
        will reprice as needed.
        """
        # Currently nothing to do — kept as a hook for future state
        # (e.g. avg_entry tracking if a flatten layer is added later).
        pass

    # ------------------------------------------------------------------
    # Cancel helpers exposed to the app (for kill-switch / shutdown).
    # ------------------------------------------------------------------

    def cancel_all_orders_local(self) -> list:
        """Drop local state for all resting orders, return their ids.
        Caller batch-cancels on Kalshi so an in-flight place callback
        won't write resurrection state."""
        ids = []
        if self.resting_buy_id:
            ids.append(self.resting_buy_id)
            self._clear_buy_state()
        if self.resting_sell_id:
            ids.append(self.resting_sell_id)
            self._clear_sell_state()
        return ids

    # ------------------------------------------------------------------
    # Core quote loop
    # ------------------------------------------------------------------

    def update_theo(self, theo: float):
        """Reprice both sides around `theo`.  Thread-safe.  Caller
        passes a single fair-value estimate (N(d2)); we apply edges
        symmetrically.
        """
        if not self.active:
            return
        if not self._lock.acquire(blocking=False):
            return  # another tick already in flight
        try:
            self._update_theo_locked(theo)
        finally:
            self._lock.release()

    def _update_theo_locked(self, theo: float):
        pos = self.position

        # Tail-market guard — asymmetric.  When the inside bid is at
        # 98¢ or more, a sell at theo+edge has no room left, so pull
        # the sell.  But theo-edge may still be a profitable bid, so
        # leave the buy side to its own logic.  Mirror at the other
        # corner: inside ask ≤ 2¢ kills the buy only.  The per-side
        # range guards inside _quote_{buy,sell} (fair_sell ≥ 1 /
        # fair_buy ≤ 0) handle the same cases from the theo side.
        if self.kalshi_bid >= 0.98 and self.kalshi_bid > 0:
            self._cancel_sell()
            tail_kills_sell = True
        else:
            tail_kills_sell = False
        if self.kalshi_ask > 0 and self.kalshi_ask <= 0.02:
            self._cancel_buy()
            tail_kills_buy = True
        else:
            tail_kills_buy = False

        # Effective exposure includes pending + resting.  Cap independently
        # per side so a long position doesn't block sells (sells reduce
        # toward zero) and vice versa — but both bounded by max_position
        # in their respective directions.
        effective_long = max(pos, 0) + self.pending_buy_size + self.resting_buy_count
        effective_short = max(-pos, 0) + self.pending_sell_size + self.resting_sell_count

        bid_remaining = max(self.max_position - effective_long, 0)
        ask_remaining = max(self.max_position - effective_short, 0)

        if self.bid_active and bid_remaining > 0 and not tail_kills_buy:
            self._quote_buy(theo, self.edge_bid, min(self.size_bid, bid_remaining))
        elif not tail_kills_buy:
            # Tail-kill already cancelled — don't re-cancel.
            self._cancel_buy()

        if self.ask_active and ask_remaining > 0 and not tail_kills_sell:
            self._quote_sell(theo, self.edge_ask, min(self.size_ask, ask_remaining))
        elif not tail_kills_sell:
            self._cancel_sell()

    # ------------------------------------------------------------------
    # Per-side quote logic
    # ------------------------------------------------------------------

    @staticmethod
    def _round_to_tick(price: float, side: str) -> float:
        """Round `price` outward to a valid Kalshi tick.

        Kalshi binary markets allow 0.1¢ ticks only in the wings of the
        price range — strictly below 10¢ or strictly above 90¢.  Whole
        cents elsewhere.  Outward = away from theo+edge (worse for us)
        so the post-only constraint holds.

          side='sell' → round UP   (ceil)
          side='buy'  → round DOWN (floor)
        """
        # Pick the grid based on where the unrounded fair sits.  The
        # post-rounding price can land exactly on the 0.10/0.90
        # boundary, which is a valid whole-cent price either way.
        sub_cent = price < 0.10 or price > 0.90
        grid = 1000.0 if sub_cent else 100.0
        if side == "sell":
            return math.ceil(price * grid) / grid
        return math.floor(price * grid) / grid

    def _quote_sell(self, theo: float, edge: float, size: int):
        if size <= 0 or theo <= 0:
            self._cancel_sell()
            return
        fair_sell = theo + edge
        # Range guard: a binary can never settle above $1, so if our
        # fair sell price is at-or-above $1 there's no meaningful ask
        # to post.  Pull any resting sell rather than placing at a
        # capped value that has no economic content.
        if fair_sell >= 1.0:
            self._cancel_sell()
            return
        new_sell = self._round_to_tick(fair_sell, "sell")
        new_sell = max(0.001, min(0.999, new_sell))

        # Improvement cap: never quote inside the best ask.  If the book
        # has moved up (best ask > our computed price), join the ask
        # rather than improving past it.
        if self.kalshi_ask > 0 and new_sell < self.kalshi_ask:
            new_sell = self.kalshi_ask

        # Lonely-at-BBO: pull if we're the sole size at the inside ask.
        # Epsilon must be smaller than one tick (0.001) so we don't treat
        # adjacent levels as the same.
        if (self.resting_sell_id is not None
                and self.current_sell_price is not None
                and self.kalshi_ask > 0
                and abs(self.current_sell_price - self.kalshi_ask) < 1e-6
                and self.kalshi_ask_size <= self.resting_sell_count):
            print(f"[Strategy] {self.ticker} alone at ask "
                  f"${self.kalshi_ask:.2f}, pulling sell")
            self._cancel_sell()
            return

        # Post-only: skip if price would cross the bid.
        if self.kalshi_bid > 0 and new_sell <= self.kalshi_bid:
            return

        if self._should_reprice_sell(new_sell, new_fair=fair_sell):
            self._reprice_sell(new_sell, size, fair=fair_sell)

    def _quote_buy(self, theo: float, edge: float, size: int):
        if size <= 0 or theo <= 0:
            self._cancel_buy()
            return
        fair_buy = theo - edge
        # Range guard: a binary can never settle below $0, so if our
        # fair bid is at-or-below $0 there's no meaningful bid to post.
        # Pull any resting buy.
        if fair_buy <= 0.0:
            self._cancel_buy()
            return
        new_buy = self._round_to_tick(fair_buy, "buy")
        new_buy = max(0.001, min(0.999, new_buy))

        # Improvement cap: never quote inside the best bid.
        if self.kalshi_bid > 0 and new_buy > self.kalshi_bid:
            new_buy = self.kalshi_bid

        # Lonely-at-BBO: pull if we're the sole size at the inside bid.
        if (self.resting_buy_id is not None
                and self.current_buy_price is not None
                and self.kalshi_bid > 0
                and abs(self.current_buy_price - self.kalshi_bid) < 1e-6
                and self.kalshi_bid_size <= self.resting_buy_count):
            print(f"[Strategy] {self.ticker} alone at bid "
                  f"${self.kalshi_bid:.2f}, pulling buy")
            self._cancel_buy()
            return

        # Post-only: skip if price would cross the ask.
        if self.kalshi_ask > 0 and new_buy >= self.kalshi_ask:
            return

        if self._should_reprice_buy(new_buy, new_fair=fair_buy):
            self._reprice_buy(new_buy, size, fair=fair_buy)

    # ------------------------------------------------------------------
    # Reprice decision — compare against stored fair (not rounded price)
    # so a sub-tick theo flicker that crosses a 1c boundary doesn't
    # bounce us.
    # ------------------------------------------------------------------

    def _should_reprice_sell(self, new_price: float,
                             new_fair: float | None = None) -> bool:
        if self.current_sell_price is None:
            return True
        # BBO-improvement override: if our resting price is currently
        # tighter than the best ask (BBO moved up away from us), we are
        # improving the BBO and must back off — bypass the tolerance
        # gate.  Only fires when the new_price actually raises us, so
        # we don't loop on identical prices.
        if (self.kalshi_ask > 0
                and self.current_sell_price < self.kalshi_ask
                and new_price > self.current_sell_price):
            return True
        diff = new_price - self.current_sell_price
        if diff < 0:
            # Improving (lowering sell) — always allow if >= 1 tick (0.1¢)
            return abs(diff) >= 0.001
        # Worsening (raising sell) — gate on fair-side move
        if new_fair is not None and self.current_sell_fair is not None:
            return (new_fair - self.current_sell_fair) >= self.tolerance
        return diff >= self.tolerance

    def _should_reprice_buy(self, new_price: float,
                            new_fair: float | None = None) -> bool:
        if self.current_buy_price is None:
            return True
        # Mirror of sell: if our resting buy is currently tighter than
        # the best bid (BBO moved down away from us), back off to BBO.
        if (self.kalshi_bid > 0
                and self.current_buy_price > self.kalshi_bid
                and new_price < self.current_buy_price):
            return True
        diff = new_price - self.current_buy_price
        if diff > 0:
            return abs(diff) >= 0.001  # 1 tick (0.1¢)
        if new_fair is not None and self.current_buy_fair is not None:
            return (self.current_buy_fair - new_fair) >= self.tolerance
        return abs(diff) >= self.tolerance

    # ------------------------------------------------------------------
    # Async order primitives
    # ------------------------------------------------------------------

    def _clear_sell_state(self):
        self.resting_sell_id = None
        self.current_sell_price = None
        self.current_sell_fair = None
        self.resting_sell_count = 0

    def _clear_buy_state(self):
        self.resting_buy_id = None
        self.current_buy_price = None
        self.current_buy_fair = None
        self.resting_buy_count = 0

    def _place_sell(self, price: float, count: int,
                    fair: float | None = None):
        if self.resting_sell_id is not None or self._pending_place_sell:
            return
        # Hard cap — never let net short exceed max_position.
        pos = self.position
        current_short = max(-pos, 0) + self.pending_sell_size + self.resting_sell_count
        cap = max(self.max_position - current_short, 0)
        if cap <= 0:
            return
        count = min(count, cap)
        if count <= 0:
            return

        self._pending_place_sell = True
        self.pending_sell_size = count

        def on_done(future):
            self._pending_place_sell = False
            self.pending_sell_size = 0
            try:
                resp = future.result()
            except Exception as e:
                print(f"[Strategy] {self.ticker} sell place failed: {e}")
                return
            order = resp.get("order", {})
            self.resting_sell_id = order.get("order_id")
            self.current_sell_price = price
            self.current_sell_fair = fair
            self.resting_sell_count = count
            print(f"[Strategy] {self.ticker} SELL YES @ {price*100:.1f}¢ "
                  f"x{count}  pos={self.position}  id={self.resting_sell_id}")

        # 3-decimal price covers the 15-min tenth-of-cent tick.
        f = self.api.create_order_async(
            ticker=self.ticker, side="yes", action="sell",
            price_dollars=f"{price:.3f}", count=count,
            tag="init", post_only=True,
        )
        f.add_done_callback(on_done)

    def _place_buy(self, price: float, count: int,
                   fair: float | None = None):
        if self.resting_buy_id is not None or self._pending_place_buy:
            return
        pos = self.position
        current_long = max(pos, 0) + self.pending_buy_size + self.resting_buy_count
        cap = max(self.max_position - current_long, 0)
        if cap <= 0:
            return
        count = min(count, cap)
        if count <= 0:
            return

        self._pending_place_buy = True
        self.pending_buy_size = count

        def on_done(future):
            self._pending_place_buy = False
            self.pending_buy_size = 0
            try:
                resp = future.result()
            except Exception as e:
                print(f"[Strategy] {self.ticker} buy place failed: {e}")
                return
            order = resp.get("order", {})
            self.resting_buy_id = order.get("order_id")
            self.current_buy_price = price
            self.current_buy_fair = fair
            self.resting_buy_count = count
            print(f"[Strategy] {self.ticker} BUY YES @ {price*100:.1f}¢ "
                  f"x{count}  pos={self.position}  id={self.resting_buy_id}")

        f = self.api.create_order_async(
            ticker=self.ticker, side="yes", action="buy",
            price_dollars=f"{price:.3f}", count=count,
            tag="init", post_only=True,
        )
        f.add_done_callback(on_done)

    def _cancel_sell(self):
        if self.resting_sell_id is None or self._pending_cancel_sell:
            return
        order_id = self.resting_sell_id
        self._pending_cancel_sell = True

        def on_done(future):
            self._pending_cancel_sell = False
            try:
                future.result()
                print(f"[Strategy] {self.ticker} cancelled sell {order_id}")
                self._clear_sell_state()
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.ticker} cancel sell failed: {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    self._clear_sell_state()

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_done)

    def _cancel_buy(self):
        if self.resting_buy_id is None or self._pending_cancel_buy:
            return
        order_id = self.resting_buy_id
        self._pending_cancel_buy = True

        def on_done(future):
            self._pending_cancel_buy = False
            try:
                future.result()
                print(f"[Strategy] {self.ticker} cancelled buy {order_id}")
                self._clear_buy_state()
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.ticker} cancel buy failed: {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    self._clear_buy_state()

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_done)

    def _reprice_sell(self, price: float, count: int,
                      fair: float | None = None):
        """Cancel then place — chained so Kalshi sees them in order and
        we never have two sells co-existing in the book."""
        if self._pending_cancel_sell or self._pending_place_sell:
            return
        if self.resting_sell_id is None:
            self._place_sell(price, count, fair=fair)
            return
        order_id = self.resting_sell_id
        self._pending_cancel_sell = True

        def on_cancel_done(future):
            self._pending_cancel_sell = False
            try:
                future.result()
                print(f"[Strategy] {self.ticker} cancelled sell {order_id} (chain)")
                self._clear_sell_state()
                self._place_sell(price, count, fair=fair)
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.ticker} cancel sell failed (chain): {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    self._clear_sell_state()
                    self._place_sell(price, count, fair=fair)

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_cancel_done)

    def _reprice_buy(self, price: float, count: int,
                     fair: float | None = None):
        if self._pending_cancel_buy or self._pending_place_buy:
            return
        if self.resting_buy_id is None:
            self._place_buy(price, count, fair=fair)
            return
        order_id = self.resting_buy_id
        self._pending_cancel_buy = True

        def on_cancel_done(future):
            self._pending_cancel_buy = False
            try:
                future.result()
                print(f"[Strategy] {self.ticker} cancelled buy {order_id} (chain)")
                self._clear_buy_state()
                self._place_buy(price, count, fair=fair)
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.ticker} cancel buy failed (chain): {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    self._clear_buy_state()
                    self._place_buy(price, count, fair=fair)

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_cancel_done)
