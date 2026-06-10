import math
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field


@dataclass
class Quote:
    order_id: str
    price: float
    size: int


@dataclass
class PendingOp:
    request_id: str
    kind: str       # "place" | "cancel"
    side: str       # "bid" | "ask"
    price: float    # for "place"; not used for "cancel"
    # The httpx Future from the api executor.  Retained so that
    # `cancel_all_sync` can wait on any in-flight place at teardown
    # and then cancel the resulting order_id — otherwise a place
    # that lands AFTER the OSM worker exits leaves an orphan order
    # on Kalshi.
    future: Future | None = field(default=None, repr=False)


class OSM:

    SUCCESSFUL_CODES = frozenset({200, 201})

    #how often to check for forgetten orders
    SWEEP_INTERVAL_S = 60.0

    def __init__(self, ticker, tolerance, api, max_position,
                 position: int = 0, strategy_queue=None):
        self.ticker = ticker
        self.tolerance = tolerance
        self.api = api
        self.max_position = max_position
        self.queue = queue.Queue()
        self.running = False 

        # What Strategy wants
        self.desired_bid_price: float | None = None
        self.desired_ask_price: float | None = None
        self.desired_bid_size: int | None = None
        self.desired_ask_size: int | None = None

        # What's actually on Kalshi
        self.resting_bid: Quote | None = None
        self.resting_ask: Quote | None = None

        # In-flight API ops keyed by request_id
        self.pending_ops: dict[str, PendingOp] = {}

        # Net YES-equivalent position from fills OSM has observed.
        # Seeded from caller (e.g. REST position fetch on app restart) so
        # the max_position cap is correct from the first message, not
        # only after the first fill.  Dedupe by trade_id below ensures
        # WS replays don't double-count on top of the seed.
        self.position: int = int(position)

        # Dedupe — Kalshi can replay fills on WS reconnect.
        self._seen_trade_ids: set[str] = set()
        # Set by cancel_all_sync().  When True, ENSURE_BID/ASK handlers
        # become no-ops so a queued-but-not-yet-processed ENSURE_*
        # (e.g. from Strategy2's last pre-stop tick) can't place a
        # new order during teardown.
        self._stopping: bool = False
        # Orphan fills — fill WS arrived before the matching place
        # response.  Keyed by order_id; drained in _handle_api_success
        # when the place lands.
        self._orphan_fills: dict[str, list[dict]] = {}

        #forgotten orders last checked time
        self._last_sweep = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)

    def run(self):
        while self.running:
            try:
                header, payload = self.queue.get(timeout=0.1)
                # Commands from Strategy
                if   header == "ENSURE_BID":     self._handle_ensure_bid(payload)
                elif header == "ENSURE_ASK":     self._handle_ensure_ask(payload)
                elif header == "CANCEL_BID":     self._handle_cancel_bid()
                elif header == "CANCEL_ASK":     self._handle_cancel_ask()
                elif header == "CANCEL_ALL":     self._handle_cancel_all()
                elif header == "UPDATE_TOLERANCE":  self._handle_update_tolerance(payload)
                elif header == "UPDATE_MAX_POSITION": self._handle_update_max_position(payload)
                # State updates from external sources
                elif header == "FILL":           self._handle_fill(payload)
                elif header == "API_RESPONSE":   self._handle_api_response(payload)
                #pushed by self when the queue is sempty
                elif header == "SWEEP_RESULT":  self._handle_sweep_result(payload)
            except queue.Empty:
                if time.time() - self._last_sweep > self.SWEEP_INTERVAL_S:
                    try:
                        self._start_sweep()
                    except Exception as e:
                        print(f"[OSM] sweep error: {e}")
                continue
            except Exception as e:
              print(f"[OSM] handler error on {header}: {e}")

    # ------------------------------------------------------------------
    # Public commands (Strategy calls these)
    # ------------------------------------------------------------------
    def ensure_bid(self, price: float, size: int):
        self.queue.put(("ENSURE_BID", (price, size)))

    def ensure_ask(self, price: float, size: int):
        self.queue.put(("ENSURE_ASK", (price, size)))

    def cancel_bid(self):
        self.queue.put(("CANCEL_BID", None))

    def cancel_ask(self):
        self.queue.put(("CANCEL_ASK", None))

    def cancel_all(self):
        self.queue.put(("CANCEL_ALL", None))

    def cancel_all_sync(self, timeout: float = 2.0):
        """Synchronously drain in-flight places + cancel all resting
        orders and block until Kalshi responds (or `timeout` elapses).

        Three-phase teardown:
          1. **Set _stopping** so any new ENSURE_BID/ASK that arrives
             during teardown (e.g. queued by Strategy2 just before its
             worker exited) is ignored by the handlers.
          2. **Wait for in-flight places to land** (with half the
             timeout budget) and harvest their order_ids — these would
             otherwise be orphan orders on Kalshi if the worker exits
             before the place response is processed.
          3. **Cancel all known + newly-landed order_ids** (with the
             remaining budget) and clear local desired/resting state.

        Caller (`Strategy2.stop()`) is expected to have already flipped
        its own `running` flag and joined its worker thread, so no new
        ENSURE_* messages will be enqueued during this teardown beyond
        what was already in-flight at stop-time.
        """
        from concurrent.futures import wait
        self._stopping = True

        # --- Phase 1: snapshot in-flight place futures + resting ids ---
        # Snapshot under no lock — worker thread can still mutate, but
        # we'll be tolerant of partial state and let the timeout bound
        # the wait.
        place_futures = [
            op.future for op in list(self.pending_ops.values())
            if op.kind == "place" and op.future is not None
        ]
        ids_to_cancel = []
        if self.resting_bid is not None:
            ids_to_cancel.append(self.resting_bid.order_id)
        if self.resting_ask is not None:
            ids_to_cancel.append(self.resting_ask.order_id)

        # --- Phase 2: wait for in-flight places, harvest order_ids ---
        if place_futures:
            place_budget = max(0.1, timeout / 2.0)
            done, _ = wait(place_futures, timeout=place_budget)
            for fut in done:
                try:
                    resp = fut.result()
                    oid = (resp.get("order", {}) or {}).get("order_id")
                    if oid:
                        ids_to_cancel.append(oid)
                except Exception:
                    # Place errored — nothing to cancel for this one.
                    pass

        # --- Phase 3: cancel everything (resting + newly-landed) ---
        if ids_to_cancel:
            cancel_budget = max(0.1, timeout - (timeout / 2.0))
            cancel_futures = [
                self.api.cancel_order_async(oid) for oid in ids_to_cancel
            ]
            wait(cancel_futures, timeout=cancel_budget)

        # Clear local state.  Whatever didn't land in time is a
        # best-effort orphan — the next session's recorder + OSM
        self.resting_bid = None
        self.resting_ask = None
        self.desired_bid_price = None
        self.desired_ask_price = None
    
    def update_tolerance(self, tolerance: float):
      self.queue.put(("UPDATE_TOLERANCE", tolerance))

    def update_max_position(self, max_position: int):
      self.queue.put(("UPDATE_MAX_POSITION", int(max_position)))

    # ------------------------------------------------------------------
    # Public read-only state (Strategy reads these)
    # ------------------------------------------------------------------
    @property
    def has_bid(self) -> bool: return self.resting_bid is not None

    @property
    def has_ask(self) -> bool: return self.resting_ask is not None

    @property
    def bid(self) -> float | None:
        return self.resting_bid.price if self.resting_bid else None

    @property
    def ask(self) -> float | None:
        return self.resting_ask.price if self.resting_ask else None
    
    # ------------------------------------------------------------------
    # WS event entry points (app.py wires these from fill / user_orders WS)
    # ------------------------------------------------------------------
    def on_fill(self, msg: dict):
        self.queue.put(("FILL", msg))

    # ------------------------------------------------------------------
    # Command handlers — update desired state, then reconcile
    # ------------------------------------------------------------------
    def _handle_ensure_bid(self, tup: tuple):
        # Strategy sends its desired size; OSM clamps to remaining
        # capacity here, atomically against current position + resting
        # state.  Single-threaded worker means no race vs fills.
        if self._stopping:
            # Teardown in progress — reject new placements.
            return
        price, requested_size = tup
        sz = self._get_bid_size(requested_size)
        if sz <= 0:
            # No room left under max_position — drop any resting bid.
            self.desired_bid_price = None
            self.desired_bid_size = None
            self._reconcile_bid()
            return
        self.desired_bid_price = price
        self.desired_bid_size = sz
        self._reconcile_bid()

    def _handle_ensure_ask(self, tup: tuple):
        if self._stopping:
            return
        price, requested_size = tup
        sz = self._get_ask_size(requested_size)
        if sz <= 0:
            self.desired_ask_price = None
            self.desired_ask_size = None
            self._reconcile_ask()
            return
        self.desired_ask_price = price
        self.desired_ask_size = sz
        self._reconcile_ask()

    def _get_bid_size(self, requested_size) -> int:
        # effective_long = current long + resting bid (would add to long)
        resting = self.resting_bid.size if self.resting_bid else 0
        effective_long = max(self.position, 0) + resting
        available_capacity =  max(self.max_position - effective_long, 0)
        return min(int(requested_size), available_capacity)

    def _get_ask_size(self, requested_size) -> int:
        # effective_short = current short + resting ask (would add to short)
        resting = self.resting_ask.size if self.resting_ask else 0
        effective_short = max(-self.position, 0) + resting
        available_capacity =  max(self.max_position - effective_short, 0)
        return min(int(requested_size), available_capacity)

    def _handle_cancel_bid(self):
        self.desired_bid_price = None
        self._reconcile_bid()

    def _handle_cancel_ask(self):
        self.desired_ask_price = None
        self._reconcile_ask()

    def _handle_cancel_all(self):
        self.desired_bid_price = None
        self.desired_ask_price = None
        self._reconcile_both()

    def _clear_resting(self, side: str):
      if side == "bid":
          self.resting_bid = None
      else:
          self.resting_ask = None

    def _handle_update_tolerance(self, tolerance):
      self.tolerance = tolerance
      # Possibly reconcile in case the new tolerance makes a current
      # mismatch actionable that wasn't before
      self._reconcile_both()

    def _handle_update_max_position(self, max_position):
      self.max_position = int(max_position)
      # Next ENSURE_BID/ASK will pick up the new cap automatically.
      # No reconcile needed — resting sizes don't change retroactively.

    # ------------------------------------------------------------------
    # WS event handlers
    # ------------------------------------------------------------------

    def _handle_fill(self, msg):
        # 1. Dedupe by trade_id — WS reconnects can replay events.
        trade_id = msg.get("trade_id", "")
        if trade_id and trade_id in self._seen_trade_ids:
            print(f"[OSM], _handle_fill, Already seen trade id: {trade_id}")
            return
        if trade_id:
            self._seen_trade_ids.add(trade_id)

        # 2. Defensive ticker check.
        ticker = msg.get("market_ticker") or msg.get("ticker", "")
        if ticker != self.ticker:
            return

        # 3. Parse fill details.
        order_id = msg.get("order_id", "")
        try:
            count = float(msg.get("count_fp", 0) or 0)
        except (TypeError, ValueError):
            print(f"[OSM], _handle_fill, unable to parse count for order id: {order_id}")
            return
        if count <= 0:
            print(f"[OSM], _handle_fill, count less than 0 for order id: {order_id}, count={count}")
            return

        # 4. Match against resting bid/ask by order_id; otherwise buffer
        #    as an orphan (fill arrived before its place response).
        if self.resting_bid and self.resting_bid.order_id == order_id:
            self._apply_fill_to_side("bid", count)
        elif self.resting_ask and self.resting_ask.order_id == order_id:
            self._apply_fill_to_side("ask", count)
        else:
            self._orphan_fills.setdefault(order_id, []).append(msg)

        # 5. Update net position.
        action = (msg.get("action") or "").lower()
        if action == "buy":
            self.position += int(count)
        elif action == "sell":
            self.position -= int(count)

        self._reconcile_both()

    def _apply_fill_to_side(self, side: str, count: float):
        #Decrement resting size; clear the side if fully consumed
        quote = self.resting_bid if side == "bid" else self.resting_ask
        if quote is None:
            return
        quote.size -= int(count)
        if quote.size <= 0:
            if side == "bid":
                self.resting_bid = None
            else:
                self.resting_ask = None

    def _handle_api_response(self, payload):
        req_id, response_or_error = payload
        op = self.pending_ops.pop(req_id, None)
        if op is None:
            return
        
        if int(response_or_error['status_code']) in self.SUCCESSFUL_CODES:
            # Success path: state advanced (resting set on place, resting
            # cleared on cancel) — reconcile immediately so the next step
            # in the cancel→place sequence fires without waiting for a
            # BBO/theo tick.
            self._handle_api_success(op, response_or_error)
            self._reconcile_both()
        else:
            self._handle_api_error(op, response_or_error)

    def _handle_api_error(self, op: PendingOp, error_payload: dict):
        status_code = error_payload['status_code']
        # Kalshi wraps most errors as {"error": {"code": ..., "message": ...}}.
        # CDN/gateway responses and our network-exception envelope put the
        # fields at top level — fall through to that shape too.
        err = error_payload.get('error') or error_payload
        message = err.get('message', '') if isinstance(err, dict) else str(err)

        # 404 on cancel means Kalshi has no record of this order (already
        # filled or already cancelled).  Clear local resting state so we
        # don't ghost-cancel it on every BBO tick.
        if op.kind == "cancel" and status_code == 404:
            self._clear_resting(op.side)

        if status_code == 400:
            print(f'[OSM] BadRequestError on {op.kind} {op.side}: {message}')
        elif status_code == 401:
            print(f'[OSM] UnauthorizedError on {op.kind} {op.side}: {message}')
        elif status_code == 404:
            print(f'[OSM] NotFoundError on {op.kind} {op.side}: {message}')
        elif status_code == 409:
            print(f'[OSM] ConflictError on {op.kind} {op.side}: {message}')
        elif status_code == 429:
            # api.is_rate_limited() is set inside _check_rate_limit on the
            # 429 response.  Subsequent _send_place calls are gated until
            # cooldown elapses; nothing extra to do here.
            print(f'[OSM] RateLimitError on {op.kind} {op.side}: {message}')
        elif status_code == 500:
            print(f'[OSM] InternalServerError on {op.kind} {op.side}: {message}')
        elif status_code == 0:
            print(f'[OSM] network exception on {op.kind} {op.side}: {message}')
        else:
            print(f'[OSM] unexpected status {status_code} on {op.kind} {op.side}: {error_payload}')


    def _handle_api_success(self, op: PendingOp, response: dict):
        if op.kind == "place":
            order = response.get("order", {})
            order_id = order.get("order_id")
            remaining = int(float(order.get("remaining_count_fp", 0) or 0))
            if remaining > 0:
                q = Quote(order_id=order_id, price=op.price, size=remaining)
                if op.side == "bid":  self.resting_bid = q
                else:                 self.resting_ask = q
                # Drain any orphan fills buffered for this order_id (fill
                # WS arrived before this place response).  Apply them to
                # the just-written resting state; may clear it back to
                # None if fully consumed.
                for fill_msg in self._orphan_fills.pop(order_id, []):
                    try:
                        count = float(fill_msg.get("count_fp", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if count > 0:
                        self._apply_fill_to_side(op.side, count)
            else:
                # Order already fully filled at place-time — the response's
                # remaining=0 signal already accounts for any fills.  Drop
                # any orphan fills buffered for this order_id to keep the
                # buffer from growing.
                self._orphan_fills.pop(order_id, None)
        elif op.kind == "cancel":
            # Cancel succeeded — resting is gone.
            self._clear_resting(op.side)

    # ------------------------------------------------------------------
    # Reconcile — drive resting toward desired (level-triggered)
    # ------------------------------------------------------------------
    def _reconcile_both(self):
        self._reconcile_bid()
        self._reconcile_ask()

    def _reconcile_bid(self):
        # If any bid-side op is in flight, wait for it.
        if any(op.side == "bid" for op in self.pending_ops.values()):
            return

        d, r, s = self.desired_bid_price, self.resting_bid, self.desired_bid_size
        self._reconcile_action("bid", d, r, s)

    def _reconcile_ask(self):
        if any(op.side == 'ask' for op in self.pending_ops.values()):
            return #something still  in flight

        d, r, s = self.desired_ask_price, self.resting_ask, self.desired_ask_size
        self._reconcile_action("ask", d, r, s)

    def _reconcile_action(self, side: str, d: float, r: Quote, s: int):
        # No desired orders nor resting orders
        if d is None and r is None:
            return

        # Snap once so all comparisons (memo + price-equality with resting)
        # use the value Kalshi will actually see.
        snapped_d = self._round_to_tick(d, side) if d is not None else None
        if snapped_d is not None and r is not None and snapped_d == r.price:
            return
        
        #if the desired price does not differ from resting price by more than tolerance
        #do not change order
        if d is not None and r is not None and abs(d - r.price) < self.tolerance:
            return

        # Need to act
        if d is None and r is not None:
            #no order desired so cancel resting order
            self._send_cancel(side, r.order_id)
        elif d is not None and r is None:
            self._send_place(side, snapped_d, s)
        elif d is not None and r is not None:
            # Want to cancel the resting order and place a new order at the desired price
            self._send_cancel(side, r.order_id)

    # ------------------------------------------------------------------
    # API call primitives
    # ------------------------------------------------------------------
    def _send_place(self, side, price, size):
        # Global API rate-limit gate
        if self.api.is_rate_limited():
            return
        
        req_id = str(uuid.uuid4())

        # callback once we get a response
        # handle errors or successes approiately 
        def on_done(future):
            try:
                resp = future.result()
                self.queue.put(("API_RESPONSE", (req_id, resp)))
            except Exception as e:
                error = {}
                error['message'] = f"{type(e).__name__}: {e}"
                error['status_code'] = 0
                self.queue.put(("API_RESPONSE", (req_id, error)))

        f = self.api.create_order_async(
            ticker=self.ticker, side="yes",
            action="buy" if side == "bid" else "sell",
            price_dollars=f"{price:.3f}", count=size,
            tag="aston", post_only=True,
        )
        self.pending_ops[req_id] = PendingOp(
            request_id=req_id, kind="place", side=side, price=price, future=f)
        f.add_done_callback(on_done)

    def _send_cancel(self, side, order_id):
        # Cancels are not rate-gated.  Risk management (clearing stale
        # orders) takes priority over rate-limit hygiene; cancels also
        # cost only 2 tokens vs 10 for places.  A rejected cancel will
        # naturally re-fire on the next reconcile cycle.
        req_id = str(uuid.uuid4())

        def on_done(future):
            try:
                resp = future.result()
                self.queue.put(("API_RESPONSE", (req_id, resp)))
            except Exception as e:
                error = {}
                error['message'] = f"{type(e).__name__}: {e}"
                error['status_code'] = 0
                self.queue.put(("API_RESPONSE", (req_id, error)))

        f = self.api.cancel_order_async(order_id)
        self.pending_ops[req_id] = PendingOp(
            request_id=req_id, kind="cancel", side=side, price=0.0, future=f)
        f.add_done_callback(on_done)

    # ------------------------------------------------------------------
    # Detect forgetten orders 
    # ------------------------------------------------------------------

    def _start_sweep(self):
        #update time for last time we swept through and checked
        self._last_sweep = time.time()
        # An in-flight place is live on Kalshi but not yet in resting_* —
        # it would look like an orphan.  Skip this round; retry in 60s.
        if self.pending_ops:
            return

        def on_done(future):
            try:
                self.queue.put(("SWEEP_RESULT", future.result()))
            except Exception:
                pass  # best-effort; next interval retries
        print(f"[OSM] Starting Sweep...")
        f = self.api.get_orders_async(status="resting")
        f.add_done_callback(on_done)

    def _handle_sweep_result(self, orders):
      if self.pending_ops:
          return  # state moved while the fetch was in flight; skip
      known = {q.order_id for q in (self.resting_bid, self.resting_ask) if q}
      for order in orders:
          if order.get("ticker") != self.ticker:        # only OUR market
              continue
          oid = order.get("order_id")
          if oid and oid not in known:
              print(f"[OSM] sweep: cancelling orphan {oid}")
              self.api.cancel_order_async(oid)

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------

    @staticmethod
    def _round_to_tick(price: float, side: str) -> float:
        """Kalshi tick grid: 1¢ in the body [0.10, 0.90], 0.1¢ in the
        wings (< 0.10 or > 0.90).  Bid floors, ask ceils, so neither
        side accidentally crosses.  Clamped to (0.001, 0.999) so the
        result is always a valid post-only price.
        """
        grid = 1000.0 if (price < 0.10 or price > 0.90) else 100.0
        if side == "bid":
            return max(math.floor(price * grid) / grid, 0.001)
        return min(math.ceil(price * grid) / grid, 0.999)
