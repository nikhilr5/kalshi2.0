import queue
import threading
import uuid
from dataclasses import dataclass


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


class OSM:

    def __init__(self, ticker, tolerance, api, strategy_queue=None):
        self.ticker = ticker
        self.tolerance = tolerance
        self.api = api
        self.queue = queue.Queue(maxsize=1024)
        self.running = False
        # Strategy's queue — OSM forwards parsed fills here so Strategy
        # can update position without touching Kalshi WS schemas itself.
        self.strategy_queue = strategy_queue

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
        # Source of truth for Strategy's max_position cap — OSM is the
        # one process that already dedupes by trade_id, so position
        # derived here can't double-count on WS reconnect.
        self.position: int = 0

        # Dedupe — Kalshi can replay fills on WS reconnect.
        self._seen_trade_ids: set[str] = set()
        # Orphan fills — fill WS arrived before the matching place
        # response.  Keyed by order_id; drained in _handle_api_success
        # when the place lands.
        self._orphan_fills: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

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
                # State updates from external sources
                elif header == "FILL":           self._handle_fill(payload)
                elif header == "API_RESPONSE":   self._handle_api_response(payload)
                # Self-driven recovery
                elif header == "PROBE_RESULT":   self._handle_probe_result(payload)
                elif header == "RECONCILE":      self._reconcile_both()
            except queue.Empty:
                continue

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
        """Synchronously cancel all currently-resting orders and block
        until Kalshi responds (or `timeout` elapses).

        Bypasses the OSM command queue — instead of enqueueing
        CANCEL_ALL and hoping the worker drains it before `stop()`
        flips `running = False`, this submits cancel REST calls
        directly to the shared API executor and waits for the futures.

        Use from `Strategy.stop()` (and any other teardown path that
        needs cancels to definitively land) so resting orders aren't
        left alive when we move to a new market or shut down.

        Clears local resting/desired state at the end so subsequent
        reconciles don't re-fire.
        """
        from concurrent.futures import wait
        # Snapshot order_ids — OSM's worker can still mutate resting_*,
        # so capture into locals before submitting.
        ids = []
        if self.resting_bid is not None:
            ids.append(self.resting_bid.order_id)
        if self.resting_ask is not None:
            ids.append(self.resting_ask.order_id)
        if not ids:
            return
        futures = [self.api.cancel_order_async(oid) for oid in ids]
        wait(futures, timeout=timeout)
        # Whatever the outcome (success / 404 / timeout) we treat the
        # local state as cleared.  We're stopping; further reconciles
        # off stale resting refs would only cause harm.
        self.resting_bid = None
        self.resting_ask = None
        self.desired_bid_price = None
        self.desired_ask_price = None
    
    def update_tolerance(self, tolerance: float):
      self.queue.put(("UPDATE_TOLERANCE", tolerance))

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
        price, size = tup
        self.desired_bid_price = price
        self.desired_bid_size = size
        self._reconcile_bid()

    def _handle_ensure_ask(self, tup: tuple):
        price, size = tup
        self.desired_ask_price = price
        self.desired_ask_size = size
        self._reconcile_ask()

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

    # ------------------------------------------------------------------
    # WS event handlers
    # ------------------------------------------------------------------

    def _handle_fill(self, msg):
        # 1. Dedupe by trade_id — WS reconnects can replay events.
        trade_id = msg.get("trade_id", "")
        if trade_id and trade_id in self._seen_trade_ids:
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
            return
        if count <= 0:
            return

        # 4. Match against resting bid/ask by order_id; otherwise buffer
        #    as an orphan (fill arrived before its place response).
        if self.resting_bid and self.resting_bid.order_id == order_id:
            self._apply_fill_to_side("bid", count)
        elif self.resting_ask and self.resting_ask.order_id == order_id:
            self._apply_fill_to_side("ask", count)
        else:
            self._orphan_fills.setdefault(order_id, []).append(msg)

        # 5. Update net position.  Aston only trades the yes side, so
        #    BUY adds and SELL subtracts.  Dedupe at step 1 ensures this
        #    is increment-once-per-fill even on WS replays.
        action = (msg.get("action") or "").lower()
        if action == "buy":
            self.position += int(count)
        elif action == "sell":
            self.position -= int(count)

        # 6. Forward clean derived event to Strategy for any
        #    fill-driven logic (Strategy itself reads position from
        #    osm.position now; this hook is retained for diagnostics).
        if self.strategy_queue is not None:
            self.strategy_queue.put(("FILL", {
                "action": msg.get("action", ""),
                "count":  count,
                "price":  float(msg.get("yes_price_dollars", 0) or 0),
            }))

        self._reconcile_both()

    def _apply_fill_to_side(self, side: str, count: float):
        """Decrement resting size; clear the side if fully consumed."""
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
        req_id, ok, response_or_error = payload
        op = self.pending_ops.pop(req_id, None)
        if op is None:
            return

        if ok:
            self._handle_api_success(op, response_or_error)
        else:
            self._handle_api_failure(op, response_or_error)
        self._reconcile_both()

    def _handle_api_failure(self, op: PendingOp, error: str):
        err = error.lower()
        is_terminal = ("404" in err
                        or "400" in err
                        or "not_found" in err
                        or "already" in err)          # already filled/cancelled
        is_transient = ("timeout" in err
                        or "500" in err
                        or "502" in err
                        or "503" in err
                        or "connection" in err)

        if op.kind == "cancel":
            if is_terminal:
                # Kalshi confirms the order is gone (already filled or already
                # cancelled).  Clear resting state for this side.
                self._clear_resting(op.side)
            elif is_transient:
                # Ambiguous — Kalshi may have cancelled it.  Probe ground truth
                # asynchronously via get_order; until then, leave resting in
                # place.  Next reconcile will see resting still set and retry
                # the cancel if needed.
                self._schedule_probe(op.side)
            else:
                print(f"[OSM] unexpected cancel error: {error}")
                self._schedule_probe(op.side)

        elif op.kind == "place":
            if is_terminal:
                # Place rejected (invalid price, post-only would cross, etc.).
                # No order created.  Resting stays None; reconcile will retry
                # if desired is still set.
                pass
            elif is_transient:
                # Ambiguous — order may or may not have been placed.  Probe.
                # Otherwise we risk double-placing on retry.
                self._schedule_probe(op.side, expected_price=op.price)
            else:
                print(f"[OSM] unexpected place error: {error}")
                self._schedule_probe(op.side, expected_price=op.price)


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

    def _schedule_probe(self, side: str, expected_price: float | None = None):
        """Query Kalshi for ground truth on `side`.  Submits get_orders
        to the api's executor so it doesn't block the OSM worker; on
        completion enqueues PROBE_RESULT for the worker to apply."""
        def on_done(future):
            try:
                orders = future.result()
                self.queue.put(("PROBE_RESULT", (side, orders)))
            except Exception as e:
                print(f"[OSM] probe failed for {side}: {e}")

        f = self.api.get_orders_async("resting")
        f.add_done_callback(on_done)

    def _handle_probe_result(self, payload):
        """Reconcile local resting_*_id against what Kalshi actually has.
        If anything is in flight for this side, the probe is stale (the
        in-flight op will determine state) — drop it.
        """
        side, orders = payload

        if any(op.side == side for op in self.pending_ops.values()):
            print(f"[OSM] probe result for {side} stale (pending ops in flight)")
            return

        action = "buy" if side == "bid" else "sell"
        ours = [o for o in orders
                if o.get("ticker") == self.ticker
                and o.get("action") == action]

        if not ours:
            # Kalshi has no order for us on this side — clear stale state.
            if side == "bid" and self.resting_bid is not None:
                print(f"[OSM] probe: clearing stale resting_bid "
                      f"(was {self.resting_bid})")
                self.resting_bid = None
            elif side == "ask" and self.resting_ask is not None:
                print(f"[OSM] probe: clearing stale resting_ask "
                      f"(was {self.resting_ask})")
                self.resting_ask = None
        else:
            # Kalshi has an order — adopt it as the source of truth.
            o = ours[0]
            try:
                order_id = o.get("order_id")
                price = float(o.get("yes_price_dollars", 0) or 0)
                remaining = int(float(o.get("remaining_count_fp", 0) or 0))
            except (TypeError, ValueError):
                return
            q = Quote(order_id=order_id, price=price, size=remaining)
            if side == "bid":
                self.resting_bid = q
            else:
                self.resting_ask = q
            print(f"[OSM] probe: resting_{side} = {q}")

        self._reconcile_both()

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
        
        #if the desired price does not differ from resting price by more than tolerance
        #do not change order
        if d is not None and r is not None and abs(d - r.price) < self.tolerance:
            return

        # Need to act
        if d is None and r is not None:
            #no order desired so cancel resting order
            self._send_cancel(side, r.order_id)
        elif d is not None and r is None:
            #good to place new order since we have no resting orders
            self._send_place(side, d, s)
        elif d is not None and r is not None:
            # Want to cancel the resting order and place a new order at the desired price
            self._send_cancel(side, r.order_id)

    # ------------------------------------------------------------------
    # API call primitives
    # ------------------------------------------------------------------
    def _send_place(self, side, price, size):
        req_id = str(uuid.uuid4())
        self.pending_ops[req_id] = PendingOp(
            request_id=req_id, kind="place", side=side, price=price)

        def on_done(future):
            try:
                resp = future.result()
                self.queue.put(("API_RESPONSE", (req_id, True, resp)))
            except Exception as e:
                self.queue.put(("API_RESPONSE", (req_id, False, str(e))))

        f = self.api.create_order_async(
            ticker=self.ticker, side="yes",
            action="buy" if side == "bid" else "sell",
            price_dollars=f"{price:.3f}", count=size,
            tag="aston", post_only=True,
        )
        f.add_done_callback(on_done)

    def _send_cancel(self, side, order_id):
        req_id = str(uuid.uuid4())
        self.pending_ops[req_id] = PendingOp(
            request_id=req_id, kind="cancel", side=side, price=0.0)
        
        def on_done(future):
            try:
                resp = future.result()
                self.queue.put(("API_RESPONSE", (req_id, True, resp)))
            except Exception as e:
                self.queue.put(("API_RESPONSE", (req_id, False, str(e))))
        
        f = self.api.cancel_order_async(order_id)
        f.add_done_callback(on_done)
