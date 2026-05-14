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
  theo and walk toward the contra BBO / entry price over time:

    Sell (long flatten): starts at ask_theo, walks DOWN by
        flatten_walk_step every flatten_walk_interval seconds.
        Floor = min(kalshi_bid, avg_entry) — whichever is more
        aggressive.  Walk can join the bid for an immediate fill;
        if avg_entry is below the bid, walk can keep going to entry.

    Buy (short flatten): starts at bid_theo, walks UP by
        flatten_walk_step every flatten_walk_interval seconds.
        Ceiling = max(kalshi_ask, avg_entry) — whichever is more
        aggressive.

  This lets the walk reach the contra BBO (where you'd actually
  get filled) instead of hovering at avg_entry where no one is
  trading.  If avg_entry is between theo and the contra BBO, the
  walk can still pass it — phase 3 (time/drift trigger) is the
  guardrail against riding losers indefinitely.

  Note: the walk is allowed to set the new BBO (improve past inside),
  unlike init quoting where lonely-BBO is bad.  For flattening,
  setting the BBO speeds up your fill.

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
        self.init_only = False    # when True, always quote with edge (no flatten)

        # Sell side (ask) state
        self.resting_sell_id: str | None = None
        self.current_sell_price: float | None = None
        # Unrounded fair (theo + edge) at the time we placed this resting
        # sell — used by the reprice-on-worsen check so a sub-tick theo
        # flicker that crosses a 1c rounding boundary doesn't bounce us.
        self.current_sell_fair: float | None = None
        self._sell_is_flatten: bool = False  # current resting sell is a flatten order

        # Buy side (bid) state
        self.resting_buy_id: str | None = None
        self.current_buy_price: float | None = None
        self.current_buy_fair: float | None = None
        self._buy_is_flatten: bool = False   # current resting buy is a flatten order

        # In-flight tracking — REST writes are async; these flags prevent
        # us from issuing duplicate cancel/place while the previous one is
        # still on the wire.  Cleared in the completion callback.
        self._pending_cancel_sell: bool = False
        self._pending_cancel_buy: bool = False
        self._pending_place_sell: bool = False
        self._pending_place_buy: bool = False

        # Velocity guard — when set by the manager, skip placing INIT quotes
        # until this monotonic timestamp.  Flatten/phase3 orders ignore the
        # cooldown (they're exiting an existing position).
        self._velocity_cooldown_until: float = 0.0

        self.position: int = 0
        self.pending_buy_size: int = 0    # orders sent but not yet acked/filled
        self.pending_sell_size: int = 0
        self.exposure: float = 0.0
        self.realized_pnl: float = 0.0
        # Once a WS fill arrives, the strategy owns avg_entry locally.
        # Until then the app may seed it from REST get_fills (used at
        # cold-start when pos > 0 from a prior session).  This avoids
        # the REST round-trip race that caused stale-avg phase 3 fires.
        self._avg_entry_locked: bool = False

        # Delta lean — added to edge by app based on portfolio delta
        self.lean_bid: float = 0.0   # extra edge added to buy side
        self.lean_ask: float = 0.0   # extra edge added to sell side
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
        # walk toward the entry price by flatten_walk_step every
        # flatten_walk_interval seconds, floored/capped at avg_entry
        # so we don't realize a loss on the spread itself.
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

        # Phase 3: when flatten walk has been running too long OR theo has
        # drifted too far against the position, cross the spread to exit.
        # Set phase3_after_sec=0 to disable (default disabled).
        self.phase3_after_sec: float = 0.0       # time stop (seconds)
        self.phase3_theo_drift_cents: float = 0.0  # theo drift stop (cents)
        # Grace window before the time-based phase 3 fires: relax the
        # avg_entry floor by 1c so we can post AT avg_entry (zero round-
        # trip profit but no loss) hoping for a maker fill before phase 3
        # crosses and pays taker fees.  Stays post-only.
        self.phase3_grace_sec: float = 30.0
        # Once phase 3 fires, init quoting on the opposite side is paused
        # until the position returns to zero (otherwise we keep re-opening
        # the bad position as fast as we flatten it).
        self._phase3_active: bool = False
        # Side that phase 3 fired on — "sell" (was long) or "buy" (was short).
        # Used to clear `_phase3_active` once the position is flat *or* has
        # crossed past zero (e.g. an aggressive cross overshoots into the
        # opposite side and we still want phase 3 available to flatten it).
        self._phase3_fired_side: str = ""
        # Position at the moment phase 3 fired.  Used by on_fill to detect
        # confirmed fills on the phase 3 order — once position changes
        # from this value, we re-arm so the next tick can fire phase 3
        # again on the remainder (handles partial fills / book moves).
        self._phase3_fire_pos: int = 0

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

    def on_fill(self, action: str = "", price: float = 0.0,
                count: float = 0, side: str = "yes"):
        """Called on any fill. Position is already updated by caller.

        Do NOT clear order IDs here — the order may be partially filled
        and still resting. Let update_theo handle repricing: it will
        cancel the real order (by ID) and replace if needed.

        Re-arm phase 3: if a fill has changed our position since phase 3
        last fired, clear the active flag so the next quote tick can fire
        again on the remainder.

        Update `avg_entry` locally from the fill itself — once we own
        it, the REST round-trip from `_refresh_positions()` no longer
        races with WS-driven position updates.  All four args optional
        for back-compat; without them avg_entry isn't recomputed.
        """
        if self._phase3_active and self.position != self._phase3_fire_pos:
            print(f"[Strategy] {self.strike:,.0f} phase 3 re-armed "
                  f"(fired at pos={self._phase3_fire_pos}, now {self.position})")
            self._phase3_active = False
            self._phase3_fired_side = ""

        # Update avg_entry from this WS fill — same accounting the app
        # uses for the position manager.  `self.position` was already
        # updated by the caller before this method, so we back out the
        # delta to recover the prior position.
        if action and price > 0 and count > 0:
            # Yes-equivalent signed delta:
            #   yes side  + buy  → +qty (acquiring yes)
            #   yes side  + sell → -qty (releasing yes)
            #   no  side  + buy  → -qty (long no = short yes)
            #   no  side  + sell → +qty
            if (side == "yes" and action == "buy") or \
               (side == "no" and action == "sell"):
                delta = float(count)
            else:
                delta = -float(count)
            new_pos = self.position
            prev_pos = new_pos - delta
            prev_avg = self.avg_entry
            if prev_pos == 0:
                self.avg_entry = price
            elif (prev_pos > 0 and delta > 0) or (prev_pos < 0 and delta < 0):
                # Adding to same direction — weighted avg
                self.avg_entry = (prev_avg * abs(prev_pos)
                                  + price * abs(delta)) / abs(new_pos)
            elif new_pos == 0:
                self.avg_entry = 0.0
            elif abs(delta) > abs(prev_pos):
                # Flipping past zero — new side opens at fill price
                self.avg_entry = price
            # else: reducing same direction — avg unchanged
            self._avg_entry_locked = True

    def seed_avg_entry_if_unlocked(self, value: float):
        """App calls this on every tick with REST-computed avg_entry.
        We accept it only until the first WS fill arrives — after that
        the strategy is authoritative and the REST value would race."""
        if not self._avg_entry_locked:
            self.avg_entry = value

    # --- Order-intent accessors ---
    # `resting_*_id` is a single field — only one order rests per side at a time.
    # The `_*_is_flatten` flag distinguishes intent so callers (e.g. the
    # velocity guard) can pull only init orders without touching flatten ones.

    @property
    def init_buy_id(self) -> str | None:
        return self.resting_buy_id if (self.resting_buy_id and not self._buy_is_flatten) else None

    @property
    def init_sell_id(self) -> str | None:
        return self.resting_sell_id if (self.resting_sell_id and not self._sell_is_flatten) else None

    @property
    def flatten_buy_id(self) -> str | None:
        return self.resting_buy_id if (self.resting_buy_id and self._buy_is_flatten) else None

    @property
    def flatten_sell_id(self) -> str | None:
        return self.resting_sell_id if (self.resting_sell_id and self._sell_is_flatten) else None

    def cancel_all_orders_local(self) -> list:
        """Clear local state for ALL resting orders (init + flatten + phase3)
        and return their ids.  Used by the app's stale-feed kill switch:
        caller batch-cancels everything on Kalshi while we drop local state
        so an in-flight place callback won't write resurrection state.
        """
        ids = []
        if self.resting_buy_id:
            ids.append(self.resting_buy_id)
            self._clear_buy_state()
        if self.resting_sell_id:
            ids.append(self.resting_sell_id)
            self._clear_sell_state()
        return ids

    def cancel_init_orders_local(self) -> list:
        """Clear local state for init orders ONLY and return their ids.

        Used by the velocity guard: caller batch-cancels the returned ids
        on Kalshi while we clear local state immediately so a fill arriving
        during the cancel doesn't double-track.  Flatten/phase3 orders
        are left untouched.

        In-flight place callbacks check `_velocity_cooldown_until` and
        cancel any order that lands during the cooldown, so it's safe to
        clear state here even if a place is currently on the wire.
        """
        ids = []
        if self.init_buy_id:
            ids.append(self.resting_buy_id)
            self._clear_buy_state()
        if self.init_sell_id:
            ids.append(self.resting_sell_id)
            self._clear_sell_state()
        return ids

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

        # Auto-disable if market is at the tails (no edge to capture).
        # Bid >= $0.98: market priced as near-certain YES.
        # Ask <= $0.02: market priced as near-certain NO.
        if ((self.kalshi_bid >= 0.98 and self.kalshi_bid > 0)
                or (self.kalshi_ask > 0 and self.kalshi_ask <= 0.02)):
            print(f"[Strategy] {self.strike:,.0f} TAIL MARKET "
                  f"(bid=${self.kalshi_bid:.2f}, ask=${self.kalshi_ask:.2f}) "
                  f"— auto-disabling strategy")
            self.stop()
            return

        # Include pending orders in exposure calculation
        effective_long = max(pos, 0) + self.pending_buy_size + self.resting_buy_count
        effective_short = max(-pos, 0) + self.pending_sell_size + self.resting_sell_count
        effective_exposure = max(effective_long, effective_short)
        at_max = effective_exposure >= self.max_position
        remaining = max(self.max_position - effective_exposure, 0)

        # Clear phase 3 lock once the cycle is done — flat OR position
        # crossed past zero into the opposite side (overshoot from an
        # aggressive cross).  Without the overshoot case, the flag could
        # stick forever if pos jumps from 1 directly to -7, blocking the
        # buy-side phase 3 that's now needed to flatten the new short.
        if self._phase3_active:
            done = (
                pos == 0
                or (self._phase3_fired_side == "sell" and pos < 0)
                or (self._phase3_fired_side == "buy" and pos > 0)
            )
            if done:
                self._phase3_active = False
                self._phase3_fired_side = ""
                print(f"[Strategy] {self.strike:,.0f} phase 3 cleared (pos={pos})")

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

        # Apply delta lean to edges
        eff_edge_bid = self.edge_bid + self.lean_bid
        eff_edge_ask = self.edge_ask + self.lean_ask

        if self.init_only or pos == 0:
            # --- INIT ONLY or FLAT: always quote both sides with edge ---
            if pos == 0:
                self._flatten_sell_start = None
                self._flatten_buy_start = None
                self._flatten_sell_base = 0.0
                self._flatten_buy_base = 0.0
            if at_max:
                # At max, only allow the side that reduces position
                if pos > 0:
                    self._cancel_buy()
                    if self.ask_active:
                        self._quote_sell(ask_theo, eff_edge_ask, min(self.size_ask, remaining))
                    else:
                        self._cancel_sell()
                elif pos < 0:
                    self._cancel_sell()
                    if self.bid_active:
                        self._quote_buy(bid_theo, eff_edge_bid, min(self.size_bid, remaining))
                    else:
                        self._cancel_buy()
                else:
                    self._cancel_buy()
                    self._cancel_sell()
            else:
                if self.ask_active:
                    self._quote_sell(ask_theo, eff_edge_ask, min(self.size_ask, remaining))
                else:
                    self._cancel_sell()
                if self.bid_active:
                    self._quote_buy(bid_theo, eff_edge_bid, min(self.size_bid, remaining))
                else:
                    self._cancel_buy()

        elif pos < 0:
            # --- SHORT: flatten buy, pull sell if at max ---
            if self.bid_active:
                self._quote_buy(bid_theo, 0.0, abs(pos), flatten=True)
            else:
                self._cancel_buy()
            # Cancel init sell if at max OR phase 3 is active (don't re-short)
            if at_max or self._phase3_active:
                self._cancel_sell()
            elif self.ask_active:
                self._quote_sell(ask_theo, eff_edge_ask, min(self.size_ask, remaining))
            else:
                self._cancel_sell()

        else:
            # --- LONG: flatten sell, pull buy if at max ---
            if self.ask_active:
                self._quote_sell(ask_theo, 0.0, pos, flatten=True)
            else:
                self._cancel_sell()
            # Cancel init buy if at max OR phase 3 is active (don't re-long)
            if at_max or self._phase3_active:
                self._cancel_buy()
            elif self.bid_active:
                self._quote_buy(bid_theo, eff_edge_bid, min(self.size_bid, remaining))
            else:
                self._cancel_buy()

    def _quote_sell(self, theo: float, edge: float, size: int, flatten: bool = False):
        """Compute sell price and place/reprice if needed.

        Init quoting: never improve the ask — join it at most.
        Flatten quoting: walk down from ask_theo toward avg_entry as a
        floor (won't realize a loss on the spread).  Allowed to set
        the new BBO (improve past inside) since fast fills are the goal
        when flattening.

        Additional non-flatten checks:
          1. Post-only: skip if price would cross the bid.
          2. Lonely BBO: cancel if we are the sole order at the inside ask.
        """
        if size <= 0 or theo <= 0:
            self._cancel_sell()
            return
        fair_sell = theo + edge                             # unrounded
        new_sell = math.ceil(fair_sell * 100) / 100         # round up (outward)

        phase3_cross = False  # True if phase 3 fired — cross the spread
        phase3_reason = ""    # "time" or "drift" when phase3 triggers

        # Flatten walk: lock in starting price, walk down over time.
        # The base price is frozen when the walk begins so that theo
        # fluctuations don't cause the walked price to jitter.
        if flatten and self.flatten_walk_interval > 0:
            now = time.monotonic()
            if self._flatten_sell_start is None:
                self._flatten_sell_start = now
                self._flatten_sell_base = new_sell  # lock in starting price
            elapsed = now - self._flatten_sell_start

            # Phase 3 trigger (sell side): time stop or downward theo drift.
            phase3_reason = self._is_phase3_triggered_sell(theo, elapsed)
            if phase3_reason:
                if self.kalshi_bid > 0:
                    new_sell = self.kalshi_bid  # cross the spread
                    phase3_cross = True
                    self._phase3_active = True
                    self._phase3_fired_side = "sell"
                    self._phase3_fire_pos = self.position
                    print(f"[Strategy] {self.strike:,.0f} PHASE 3 sell cross "
                          f"[{phase3_reason}] @${new_sell:.2f} "
                          f"(elapsed={elapsed:.0f}s, theo=${theo:.3f}, "
                          f"entry=${self.avg_entry:.2f}, pos={self.position})")

            if not phase3_cross:
                new_sell = self._flatten_sell_base
                steps = int(elapsed / self.flatten_walk_interval)
                if steps > 0:
                    new_sell = round(new_sell - steps * self.flatten_walk_step, 2)
                # Floor at MAX(bid + 1c, avg_entry + buffer) — stay strictly
                # above the bid (post-only, no taker fills) and above entry
                # by `entry_buffer`.  The buffer is normally 1c (preserve
                # profit on round-trip), but in the last `phase3_grace_sec`
                # before the time-based phase 3 fires we drop it to 0c so
                # the walk can post AT entry — last shot at a maker fill
                # before phase 3 crosses and pays taker fees.
                entry_buffer = 0.01
                if (self.phase3_after_sec > 0
                        and elapsed >= self.phase3_after_sec - self.phase3_grace_sec
                        and elapsed < self.phase3_after_sec):
                    entry_buffer = 0.0
                floors = []
                if self.avg_entry > 0:
                    floors.append(self.avg_entry + entry_buffer)
                if self.kalshi_bid > 0:
                    floors.append(self.kalshi_bid + 0.01)
                if floors:
                    new_sell = max(new_sell, max(floors))

        new_sell = max(0.01, min(0.99, new_sell))

        # Price improvement cap:
        # - Init: never improve the ask, join it at most.
        # - Flatten: allowed to improve past the ask (set the new BBO).
        #   The avg_entry floor already prevents losing exits.
        # - Phase 3 cross: bypasses all caps — we want to exit immediately.
        if (not flatten and not phase3_cross
                and self.kalshi_ask > 0 and new_sell < self.kalshi_ask):
            new_sell = self.kalshi_ask

        # Lonely BBO: if we're the only one at the ask, pull back.
        # Only applies to init quotes — flatten orders WANT to be at the BBO.
        if (not flatten
                and self.resting_sell_id is not None
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
            # Velocity guard: don't post fresh init quotes while in cooldown.
            # Existing init was already cancelled by the manager when the
            # guard fired; this prevents immediate re-quoting at stale prices.
            if time.monotonic() < self._velocity_cooldown_until:
                return

        if self._should_reprice_sell(new_sell, new_fair=fair_sell):
            self._reprice_sell(new_sell, size, flatten=flatten,
                               phase3=phase3_cross,
                               phase3_reason=phase3_reason,
                               fair=fair_sell)

    def _quote_buy(self, theo: float, edge: float, size: int, flatten: bool = False):
        """Compute buy price and place/reprice if needed.

        Price improvement cap applies to ALL orders (flatten and normal):
        never improve the bid — join it at most.  If the BBO moves away
        (bid drops), follow it back so we're never better than best.

        Additional non-flatten checks:
          1. Post-only: skip if price would cross the ask.
          2. Lonely BBO: cancel if we are the sole order at the inside bid.

        For flatten orders with walk enabled, the price starts at theo
        and walks up over time, capped at avg_entry (won't realize a
        loss on the spread).  Flatten orders are allowed to set the
        new BBO since fast fills are the goal when flattening.
        """
        if size <= 0 or theo <= 0:
            self._cancel_buy()
            return
        fair_buy = theo - edge                              # unrounded
        new_buy = math.floor(fair_buy * 100) / 100          # round down (outward)

        phase3_cross = False  # True if phase 3 fired — cross the spread
        phase3_reason = ""    # "time" or "drift" when phase3 triggers

        # Flatten walk: lock in starting price, walk up over time.
        # The base price is frozen when the walk begins so that theo
        # fluctuations don't cause the walked price to jitter.
        if flatten and self.flatten_walk_interval > 0:
            now = time.monotonic()
            if self._flatten_buy_start is None:
                self._flatten_buy_start = now
                self._flatten_buy_base = new_buy  # lock in starting price
            elapsed = now - self._flatten_buy_start

            # Phase 3 trigger (buy side): time stop or upward theo drift.
            phase3_reason = self._is_phase3_triggered_buy(theo, elapsed)
            if phase3_reason:
                if self.kalshi_ask > 0:
                    new_buy = self.kalshi_ask  # cross the spread
                    phase3_cross = True
                    self._phase3_active = True
                    self._phase3_fired_side = "buy"
                    self._phase3_fire_pos = self.position
                    print(f"[Strategy] {self.strike:,.0f} PHASE 3 buy cross "
                          f"[{phase3_reason}] @${new_buy:.2f} "
                          f"(elapsed={elapsed:.0f}s, theo=${theo:.3f}, "
                          f"entry=${self.avg_entry:.2f})")

            if not phase3_cross:
                new_buy = self._flatten_buy_base
                steps = int(elapsed / self.flatten_walk_interval)
                if steps > 0:
                    new_buy = round(new_buy + steps * self.flatten_walk_step, 2)
                # Ceiling at MIN(ask - 1c, avg_entry - buffer) — stay strictly
                # below the ask (post-only) and below entry by `entry_buffer`.
                # The buffer is normally 1c, but in the last `phase3_grace_sec`
                # before time-based phase 3 we drop it to 0c so the walk can
                # post AT entry — last shot at a maker fill before phase 3
                # crosses and pays taker fees.
                entry_buffer = 0.01
                if (self.phase3_after_sec > 0
                        and elapsed >= self.phase3_after_sec - self.phase3_grace_sec
                        and elapsed < self.phase3_after_sec):
                    entry_buffer = 0.0
                ceilings = []
                if self.avg_entry > 0:
                    ceilings.append(self.avg_entry - entry_buffer)
                if self.kalshi_ask > 0:
                    ceilings.append(self.kalshi_ask - 0.01)
                if ceilings:
                    new_buy = min(new_buy, min(ceilings))

        new_buy = max(0.01, min(0.99, new_buy))

        # Price improvement cap:
        # - Init: never improve the bid, join it at most.
        # - Flatten: allowed to improve past the bid (set the new BBO).
        #   The avg_entry cap already prevents losing exits.
        # - Phase 3 cross: bypasses all caps — we want to exit immediately.
        if (not flatten and not phase3_cross
                and self.kalshi_bid > 0 and new_buy > self.kalshi_bid):
            new_buy = self.kalshi_bid

        # Lonely BBO: if we're the only one at the bid, pull back.
        # Only applies to init quotes — flatten orders WANT to be at the BBO.
        if (not flatten
                and self.resting_buy_id is not None
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
            # Velocity guard: don't post fresh init quotes while in cooldown.
            if time.monotonic() < self._velocity_cooldown_until:
                return

        if self._should_reprice_buy(new_buy, new_fair=fair_buy):
            self._reprice_buy(new_buy, size, flatten=flatten,
                              phase3=phase3_cross,
                              phase3_reason=phase3_reason,
                              fair=fair_buy)

    def _is_phase3_triggered_sell(self, theo: float, elapsed: float) -> str | None:
        """Returns 'time', 'drift', or None for long flatten phase 3 trigger.

        Re-fire guard: once phase 3 has crossed for this flatten cycle
        (`_phase3_active = True`), don't fire again until pos returns to 0
        and the flag clears.  Prevents the cross from re-triggering on the
        next tick before fills propagate, which previously caused a single
        flatten cycle to flip the position past zero into a fresh short.
        """
        if self._phase3_active:
            return None
        if self.phase3_after_sec <= 0 and self.phase3_theo_drift_cents <= 0:
            return None
        # Time stop
        if self.phase3_after_sec > 0 and elapsed >= self.phase3_after_sec:
            return "time"
        # Theo drift: long flatten = bad if theo dropped below entry
        if self.phase3_theo_drift_cents > 0 and self.avg_entry > 0:
            drift = self.avg_entry - theo  # positive when theo < entry
            if drift >= self.phase3_theo_drift_cents / 100.0:
                return "drift"
        return None

    def _is_phase3_triggered_buy(self, theo: float, elapsed: float) -> str | None:
        """Returns 'time', 'drift', or None for short flatten phase 3 trigger.

        Mirror of `_is_phase3_triggered_sell` — same re-fire guard.
        """
        if self._phase3_active:
            return None
        if self.phase3_after_sec <= 0 and self.phase3_theo_drift_cents <= 0:
            return None
        if self.phase3_after_sec > 0 and elapsed >= self.phase3_after_sec:
            return "time"
        # Theo drift: short flatten = bad if theo rose above entry
        if self.phase3_theo_drift_cents > 0 and self.avg_entry > 0:
            drift = theo - self.avg_entry  # positive when theo > entry
            if drift >= self.phase3_theo_drift_cents / 100.0:
                return "drift"
        return None

    def _should_reprice_sell(self, new_price: float,
                             new_fair: float | None = None) -> bool:
        """Check if sell order needs repricing.

        `new_fair` is the unrounded `theo + edge` for the new computation.
        For WORSENING (raising the sell), we compare the underlying fair
        against the fair-at-place to gate on tolerance — comparing rounded
        prices would treat any 1c boundary cross as a full tick worsening
        (e.g. a 0.001 theo flicker rounds 0.51 → 0.52, which trivially
        beats any sub-tick tolerance).
        """
        if self.current_sell_price is None:
            return True
        diff = new_price - self.current_sell_price
        if diff < 0:
            # Improving (lowering sell) — always allow if >= 1 tick
            return abs(diff) >= 0.01
        # Worsening (raising sell) — require fair-side move >= tolerance.
        # Fall back to price-side check if we don't have a stored fair.
        if new_fair is not None and self.current_sell_fair is not None:
            return (new_fair - self.current_sell_fair) >= self.tolerance
        return diff >= self.tolerance

    def _should_reprice_buy(self, new_price: float,
                            new_fair: float | None = None) -> bool:
        """Check if buy order needs repricing.  Mirror of `_should_reprice_sell`."""
        if self.current_buy_price is None:
            return True
        diff = new_price - self.current_buy_price
        if diff > 0:
            # Improving (raising buy) — always allow if >= 1 tick
            return abs(diff) >= 0.01
        # Worsening (lowering buy) — require fair-side move >= tolerance
        if new_fair is not None and self.current_buy_fair is not None:
            return (self.current_buy_fair - new_fair) >= self.tolerance
        return abs(diff) >= self.tolerance

    # --------------------------------------------------------------------------
    # Async order management — see module docstring.
    #
    # All REST writes go through `api.{create,cancel}_order_async`, which
    # submits to a thread pool and returns a Future immediately.  The caller
    # registers a completion callback that updates local state when the
    # response arrives.  This keeps the calling thread (Coinbase WS) free to
    # process new spot ticks instead of blocking on per-call HTTP latency.
    #
    # Two flags per side guard against double-issue:
    #   _pending_cancel_*  — a cancel is on the wire, don't issue another
    #   _pending_place_*   — a place is on the wire, don't issue another
    # --------------------------------------------------------------------------

    def _clear_sell_state(self):
        self.resting_sell_id = None
        self.current_sell_price = None
        self.current_sell_fair = None
        self.resting_sell_count = 0
        self._sell_is_flatten = False

    def _clear_buy_state(self):
        self.resting_buy_id = None
        self.current_buy_price = None
        self.current_buy_fair = None
        self.resting_buy_count = 0
        self._buy_is_flatten = False

    def _place_sell(self, price: float, count: int, flatten: bool = False,
                    phase3: bool = False, phase3_reason: str = "",
                    fair: float | None = None):
        """Submit an async Sell Yes limit order.  Local state is updated when
        the API response arrives via the completion callback.

        `fair` is the unrounded theo+edge that produced `price`.  Stored
        on success so future tolerance checks compare fair-to-fair instead
        of rounded-price-to-price (the latter mis-fires on sub-tick flicker).
        """
        if self.resting_sell_id is not None or self._pending_place_sell:
            # Already resting or already submitting — don't double-up
            return
        # Hard cap: never let total short exposure exceed max_position
        pos = self.position
        if flatten and pos > 0:
            cap = pos  # can sell up to current long
        else:
            current_short = max(-pos, 0) + self.pending_sell_size + self.resting_sell_count
            cap = max(self.max_position - current_short, 0)
        if cap <= 0:
            print(f"[Strategy] {self.strike:,.0f} sell BLOCKED: cap={cap}, "
                  f"pos={pos}, pending={self.pending_sell_size}, "
                  f"resting={self.resting_sell_count}")
            return
        count = min(count, cap)
        if count <= 0:
            return

        # phase3t = time trigger, phase3d = drift trigger
        if phase3:
            tag = f"phase3{phase3_reason[0]}" if phase3_reason else "phase3"
        elif flatten:
            tag = "flat"
        else:
            tag = "init"

        self._pending_place_sell = True
        self.pending_sell_size = count

        def on_done(future):
            self._pending_place_sell = False
            self.pending_sell_size = 0
            try:
                resp = future.result()
            except Exception as e:
                print(f"[Strategy] {self.strike:,.0f} sell order failed: {e}")
                self.current_sell_price = price
                return
            order = resp.get("order", {})
            order_id = order.get("order_id")
            # If the velocity guard fired while this place was on the wire,
            # the order landed during a cooldown — kill it immediately
            # rather than letting it sit at a stale price.
            if order_id and time.monotonic() < self._velocity_cooldown_until:
                print(f"[Strategy] {self.strike:,.0f} place landed during "
                      f"velocity cooldown — cancelling {order_id}")
                self.api.cancel_order_async(order_id)
                return
            self.resting_sell_id = order_id
            self.current_sell_price = price
            self.current_sell_fair = fair  # for tolerance check on next tick
            self.resting_sell_count = count
            self._sell_is_flatten = flatten or phase3  # phase3 is a flatten path
            print(f"[Strategy] {self.strike:,.0f} SELL YES @ ${price:.2f} "
                  f"x{count}  pos={self.position}  order={self.resting_sell_id}")

        # Post-only for everything EXCEPT phase 3 (only path that takes).
        f = self.api.create_order_async(
            ticker=self.ticker,
            side="yes",
            action="sell",
            price_dollars=f"{price:.2f}",
            count=count,
            tag=tag,
            post_only=not phase3,
        )
        f.add_done_callback(on_done)

    def _place_buy(self, price: float, count: int, flatten: bool = False,
                   phase3: bool = False, phase3_reason: str = "",
                   fair: float | None = None):
        """Submit an async Buy Yes limit order.  Mirror of _place_sell."""
        if self.resting_buy_id is not None or self._pending_place_buy:
            return
        pos = self.position
        if flatten and pos < 0:
            cap = abs(pos)
        else:
            current_long = max(pos, 0) + self.pending_buy_size + self.resting_buy_count
            cap = max(self.max_position - current_long, 0)
        if cap <= 0:
            print(f"[Strategy] {self.strike:,.0f} buy BLOCKED: cap={cap}, "
                  f"pos={pos}, pending={self.pending_buy_size}, "
                  f"resting={self.resting_buy_count}")
            return
        count = min(count, cap)
        if count <= 0:
            return

        if phase3:
            tag = f"phase3{phase3_reason[0]}" if phase3_reason else "phase3"
        elif flatten:
            tag = "flat"
        else:
            tag = "init"

        self._pending_place_buy = True
        self.pending_buy_size = count

        def on_done(future):
            self._pending_place_buy = False
            self.pending_buy_size = 0
            try:
                resp = future.result()
            except Exception as e:
                print(f"[Strategy] {self.strike:,.0f} buy order failed: {e}")
                self.current_buy_price = price
                return
            order = resp.get("order", {})
            order_id = order.get("order_id")
            if order_id and time.monotonic() < self._velocity_cooldown_until:
                print(f"[Strategy] {self.strike:,.0f} place landed during "
                      f"velocity cooldown — cancelling {order_id}")
                self.api.cancel_order_async(order_id)
                return
            self.resting_buy_id = order_id
            self.current_buy_price = price
            self.current_buy_fair = fair  # for tolerance check on next tick
            self.resting_buy_count = count
            self._buy_is_flatten = flatten or phase3
            print(f"[Strategy] {self.strike:,.0f} BUY YES @ ${price:.2f} "
                  f"x{count}  pos={self.position}  order={self.resting_buy_id}")

        f = self.api.create_order_async(
            ticker=self.ticker,
            side="yes",
            action="buy",
            price_dollars=f"{price:.2f}",
            count=count,
            tag=tag,
            post_only=not phase3,
        )
        f.add_done_callback(on_done)

    def _cancel_sell(self):
        """Submit an async cancel for the resting sell.  No return value —
        local state is cleared in the completion callback."""
        if self.resting_sell_id is None or self._pending_cancel_sell:
            return
        order_id = self.resting_sell_id
        self._pending_cancel_sell = True

        def on_done(future):
            self._pending_cancel_sell = False
            try:
                future.result()
                print(f"[Strategy] {self.strike:,.0f} cancelled sell {order_id}")
                self._clear_sell_state()
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.strike:,.0f} cancel sell failed: {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    # Already gone (filled or already-cancelled) — safe to clear
                    self._clear_sell_state()

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_done)

    def _cancel_buy(self):
        """Submit an async cancel for the resting buy.  Mirror of _cancel_sell."""
        if self.resting_buy_id is None or self._pending_cancel_buy:
            return
        order_id = self.resting_buy_id
        self._pending_cancel_buy = True

        def on_done(future):
            self._pending_cancel_buy = False
            try:
                future.result()
                print(f"[Strategy] {self.strike:,.0f} cancelled buy {order_id}")
                self._clear_buy_state()
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.strike:,.0f} cancel buy failed: {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    self._clear_buy_state()

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_done)

    def _reprice_sell(self, price: float, count: int, flatten: bool = False,
                     phase3: bool = False, phase3_reason: str = "",
                     fair: float | None = None):
        """Cancel the current sell (if any) and place a new one.  When a
        cancel is needed, the place is chained off the cancel's success
        callback so Kalshi sees them in order — preventing two orders
        briefly co-existing in the book."""
        if self._pending_cancel_sell or self._pending_place_sell:
            return  # in-flight; let the previous cycle finish
        if self.resting_sell_id is None:
            self._place_sell(price, count, flatten=flatten,
                             phase3=phase3, phase3_reason=phase3_reason,
                             fair=fair)
            return
        order_id = self.resting_sell_id
        self._pending_cancel_sell = True

        def on_cancel_done(future):
            self._pending_cancel_sell = False
            try:
                future.result()
                print(f"[Strategy] {self.strike:,.0f} cancelled sell {order_id}")
                self._clear_sell_state()
                self._place_sell(price, count, flatten=flatten,
                                 phase3=phase3, phase3_reason=phase3_reason,
                                 fair=fair)
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.strike:,.0f} cancel sell failed (chain): {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    # Already gone — proceed with the place
                    self._clear_sell_state()
                    self._place_sell(price, count, flatten=flatten,
                                     phase3=phase3, phase3_reason=phase3_reason,
                                     fair=fair)
                # On other errors, don't place — the next quote tick will retry

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_cancel_done)

    def _reprice_buy(self, price: float, count: int, flatten: bool = False,
                    phase3: bool = False, phase3_reason: str = "",
                    fair: float | None = None):
        """Cancel the current buy (if any) and place a new one.  Mirror of
        _reprice_sell."""
        if self._pending_cancel_buy or self._pending_place_buy:
            return
        if self.resting_buy_id is None:
            self._place_buy(price, count, flatten=flatten,
                            phase3=phase3, phase3_reason=phase3_reason,
                            fair=fair)
            return
        order_id = self.resting_buy_id
        self._pending_cancel_buy = True

        def on_cancel_done(future):
            self._pending_cancel_buy = False
            try:
                future.result()
                print(f"[Strategy] {self.strike:,.0f} cancelled buy {order_id}")
                self._clear_buy_state()
                self._place_buy(price, count, flatten=flatten,
                                phase3=phase3, phase3_reason=phase3_reason,
                                fair=fair)
            except Exception as e:
                err = str(e)
                print(f"[Strategy] {self.strike:,.0f} cancel buy failed (chain): {e}")
                if "404" in err or "400" in err or "not_found" in err.lower():
                    self._clear_buy_state()
                    self._place_buy(price, count, flatten=flatten,
                                    phase3=phase3, phase3_reason=phase3_reason,
                                    fair=fair)

        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_cancel_done)
