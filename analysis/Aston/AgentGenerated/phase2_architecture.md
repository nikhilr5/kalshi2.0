# Aston Phase-2 architecture — queue + single-worker model

Post-2026-06-05 rewrite of the strategy's threading model.  Goal:
make the cancel-pipeline bugs identified in `strategy_fix_bundle_2026-06-05.md`
**structurally impossible** rather than relying on careful coordination
between 4+ threads.

## What this replaces

**Today:**

```
Coinbase WS thread      ─┐
Kalshi book WS thread   ─┼──► strategy methods (mutate shared state, coordinated by locks)
Kalshi user_orders WS   ─┤
ThreadPoolExecutor cbs  ─┘
```

Four threads mutating `resting_*_id`, `current_*_fair`, `_pending_*`
flags, and position state.  Coordination via `self._lock` (non-blocking)
and the `_pending_cancel_*` / `_pending_place_*` dead-zone flags.

**Failure modes this enables:**
- Tick drops (non-blocking lock)
- Dead-zone suppression (`_pending_*` flags)
- State divergence between callback orderings
- Race between WS fill notification and API place callback

## The replacement

```
Coinbase WS thread     ─┐    ┌─────────────────────────┐    ┌──────────────────┐
Kalshi book WS thread  ─┼──► │  Event queue (priority) │ ──►│ Strategy worker  │
Kalshi user_orders WS  ─┤    └─────────────────────────┘    │ (single thread)  │
API response callbacks ─┘                                   └────────┬─────────┘
                                                                     │
                              ┌──────────────────────────────────────┘
                              ▼
                       async cancel / place (results enqueued back)
```

WS callbacks become trivial — `queue.put(event)`.  All state lives on
the strategy worker.  Worker drains the queue one event at a time,
synchronously.  No locks.  No race conditions.  No dead-zones.

## Event types

Five categories.  Each is a dataclass with explicit fields — no
"msg dict" tagged-union magic.

```python
@dataclass(frozen=True, slots=True)
class SpotTick:
    ts_local: datetime          # local wall clock receipt
    ts_source: datetime         # exchange-side ts if available
    product: str                # "ETH-USD"
    price: float
    bid: float
    ask: float

@dataclass(frozen=True, slots=True)
class BookUpdate:
    ts_local: datetime
    ts_source: datetime
    ticker: str
    yes_bid: float
    yes_ask: float
    bid_size: int
    ask_size: int

@dataclass(frozen=True, slots=True)
class FillEvent:
    ts_local: datetime
    ts_source: datetime         # Kalshi match-engine ts
    trade_id: str
    order_id: str
    client_order_id: str
    ticker: str
    side: str
    action: str
    price: float
    count: float
    is_taker: bool
    remaining_count: float      # so worker can decide full vs partial

@dataclass(frozen=True, slots=True)
class OrderStateChange:
    ts_local: datetime
    ts_source: datetime
    order_id: str
    client_order_id: str
    ticker: str
    status: str                 # placed | cancelled | executed | rejected
    remaining_count: float
    price: float

@dataclass(frozen=True, slots=True)
class ApiResponse:
    request_id: str             # the strategy assigned this when issuing
    kind: str                   # "place" | "cancel"
    ticker: str
    success: bool
    payload: dict               # full response on success
    error: str | None           # error string on failure
```

Optional sixth:

```python
@dataclass(frozen=True, slots=True)
class TimerTick:
    ts_local: datetime
    purpose: str  # "reconcile_orders" | "auto_off_check" | etc.
```

## Queue

**Priority lanes — three tiers:**

```python
class PriorityEventQueue:
    """Lower priority value = higher priority.

    Lane 0: state transitions (FillEvent, OrderStateChange, ApiResponse).
            Always process first — these are facts, not opinions.

    Lane 1: timer events (TimerTick).  Periodic reconciliation,
            auto-off, kill-switch checks.

    Lane 2: market data (SpotTick, BookUpdate).  Most volume, lowest
            priority.  Theo updates are derived from these.
    """
```

**Coalescing at dequeue (lane 2 only):**

When pulling from lane 2, peek ahead — if the next SpotTick (or
BookUpdate for the same ticker) is also pending, drop the older one
and use the newer.  Keeps the worker reading fresh data without
processing every tick.  Theo updates are idempotent w.r.t. latest
inputs, so dropping older ticks loses nothing.

**Bounded size with sane drop policy:**

If queue size exceeds a high-water mark (say 10,000 events), drop
the OLDEST lane-2 events.  Never drop lane 0 (state transitions).
Log a warning.  In practice this should never fire if the worker
keeps up.

## Worker loop

```python
def worker_loop(self):
    while not self._stop.is_set():
        event = self.queue.get(timeout=0.1)
        if event is None:
            continue
        try:
            self._dispatch(event)
        except Exception as e:
            print(f"[Strategy] handler error on {type(event).__name__}: {e}")
            # never let one bad event kill the loop

def _dispatch(self, event):
    match event:
        case SpotTick():       self._on_spot(event)
        case BookUpdate():     self._on_book(event)
        case FillEvent():      self._on_fill(event)
        case OrderStateChange(): self._on_order_state(event)
        case ApiResponse():    self._on_api_response(event)
        case TimerTick():      self._on_timer(event)
```

The worker is the **only** thread that touches strategy state.  No
locks anywhere.  No `_pending_*` flags needed — the queue serializes
everything.

## State model on the worker

```python
@dataclass
class StrategyState:
    # Live model inputs
    spot: float = 0.0
    sigma: float = 0.0          # from HAR estimator (updated on SpotTick)
    theo: float = 0.0           # derived from spot + sigma + market params
    kalshi_bid: float = 0.0
    kalshi_ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0

    # Quote tracking — single source of truth for what's on the book
    resting_buy:  Quote | None = None
    resting_sell: Quote | None = None
    # In-flight API ops we've issued but not yet heard back about
    pending_ops: dict[str, PendingOp] = field(default_factory=dict)

    position: int = 0

@dataclass
class Quote:
    order_id: str
    client_order_id: str
    price: float                # where it's sitting on the book
    count: int
    placed_at: datetime
    # NOTE: no `current_fair` here — we always compute drift against
    # the current theo at evaluation time, never a stored historical fair.

@dataclass
class PendingOp:
    request_id: str
    kind: str                   # "place" | "cancel"
    issued_at: datetime
    side: str                   # "buy" | "sell"
    timeout_at: datetime        # if no response by then, force reconcile
```

**Key invariant:** every state mutation happens inside an event
handler.  The handler runs on the worker thread.  No other code path
mutates state.  This is the property that makes locks unnecessary.

## Handler logic (pseudo-code)

### `_on_spot(event: SpotTick)`

```python
self.state.spot = event.price
self.state.sigma = self.har.update(event)
self.state.theo = compute_theo(self.state.spot, ..., self.state.sigma)
self._maybe_reprice()  # evaluate both sides against the fresh theo
```

### `_on_book(event: BookUpdate)`

```python
self.state.kalshi_bid = event.yes_bid
self.state.kalshi_ask = event.yes_ask
self.state.bid_size = event.bid_size
self.state.ask_size = event.ask_size
self._maybe_reprice()  # BBO change can re-trigger the BBO-cap branch
```

### `_on_fill(event: FillEvent)`

```python
self.state.position += event.count if event.action == "buy" else -event.count
if event.action == "buy" and self.state.resting_buy is not None:
    self.state.resting_buy.count -= event.count
    if self.state.resting_buy.count <= 0:
        self.state.resting_buy = None
# mirror for sell
self._maybe_reprice()  # fill changed exposure — re-evaluate quoting
```

(`on_fill = pass` bug is structurally impossible here — the handler
*has* to touch state by definition.)

### `_on_order_state(event: OrderStateChange)`

```python
if event.status in ("cancelled", "executed", "rejected"):
    # whichever side this order is on, the quote is gone
    if self.state.resting_buy and self.state.resting_buy.order_id == event.order_id:
        self.state.resting_buy = None
    elif self.state.resting_sell and self.state.resting_sell.order_id == event.order_id:
        self.state.resting_sell = None
self._maybe_reprice()
```

### `_on_api_response(event: ApiResponse)`

```python
pending = self.state.pending_ops.pop(event.request_id, None)
if pending is None:
    return  # shouldn't happen, but tolerate replay-ish races
if event.success and pending.kind == "place":
    # write the new quote — values from the API response (canonical)
    ...
elif event.success and pending.kind == "cancel":
    # OrderStateChange will arrive separately and clear the quote
    pass
elif not event.success:
    # ANY non-success — terminal or transient — trigger a reconcile
    self._schedule_reconcile(pending.side, pending.order_id)
```

(Bug #4 — non-404 cancel errors — is handled by `_schedule_reconcile`
which queries Kalshi for ground truth and updates state to match.)

### `_maybe_reprice()`

```python
fresh_theo = self.state.theo
# Buy side
desired_buy_price = round_to_tick(fresh_theo - self.edge_bid, "buy")
if self.state.resting_buy is None:
    self._issue_place("buy", desired_buy_price)
else:
    posted_edge = fresh_theo - self.state.resting_buy.price
    intended_edge = self.edge_bid
    if (intended_edge - posted_edge) >= self.tolerance:
        self._issue_cancel_replace("buy", desired_buy_price)
# mirror for sell
```

(Bug #3 — stale `current_*_fair` — is structurally impossible.  Gate
compares against `fresh_theo` and the actual posted price, both of
which are real numbers about the present, not historical captures.)

### `_issue_cancel_replace(side, new_price)`

```python
quote = self.state.resting_buy if side == "buy" else self.state.resting_sell
# Mark intent — but no "_pending_cancel_*" dead-zone gate.
req_id = generate_request_id()
self.state.pending_ops[req_id] = PendingOp(
    request_id=req_id, kind="cancel", issued_at=now(),
    side=side, timeout_at=now() + timedelta(seconds=2),
)
self.api.cancel_order_async(quote.order_id, request_id=req_id)
# Place is issued when the OrderStateChange or ApiResponse confirms
# the cancel — keeps cancel/place strictly ordered without a dead-zone
# flag.  Alternative: issue both in parallel, accept brief overlap
# (post_only=True makes this safe; second order is rejected if first
# still resting).  Pick one explicitly.
```

(The dead-zone bug is gone.  `pending_ops` is a *list* of in-flight
operations, not a binary flag.  Multiple ops can be in flight; nothing
silently returns.)

## Timer events for self-healing

Two periodic checks on a slow timer (1-5 sec):

```python
def _on_timer(event: TimerTick):
    if event.purpose == "reconcile_orders":
        # Drift detection: ask Kalshi what orders WE think are resting,
        # check status, fix any divergence.
        for quote in [self.state.resting_buy, self.state.resting_sell]:
            if quote is None: continue
            status = self.api.get_order(quote.order_id).get("status")
            if status in ("cancelled", "executed", "rejected"):
                # We thought it was resting; Kalshi says otherwise.
                # Clear it; next _maybe_reprice will place fresh.
                ...
    elif event.purpose == "pending_ops_timeout":
        # Any pending_op past its timeout_at → force a reconcile.
        # Closes the "API call hung, no response, state stuck" class.
        ...
```

This is the *belt-and-suspenders* layer that makes the system
self-healing: even if a code bug or network blip causes state
divergence, the timer catches it within a few seconds.

## What this architecture makes structurally impossible

- ✅ Dropped theo ticks — coalescing at dequeue, but every tick *was*
  on the queue.
- ✅ Dead-zone suppression of reprices — single consumer, no
  `_pending_*` flags.
- ✅ `on_fill` doing nothing — handlers *are* the state mutations.
- ✅ Stale `current_*_fair` — there is no `current_*_fair`.  Gate
  compares posted price vs current theo, always.
- ✅ Lock contention — there is no lock.
- ✅ Ghost-id resurrection on shutdown — single thread; shutdown is a
  Command event processed in order.
- ✅ State divergence — reconcile timer pulls ground truth from
  Kalshi every few seconds.

## What this architecture does NOT fix

- ❌ Cancel-race losses to taker — fundamental, requires lower
  latency on the wire (colo, kernel bypass) or different quoting
  posture.
- ❌ Informed-flow adverse selection on fresh quotes — irreducible
  microstructure, requires not quoting in certain regimes.
- ❌ Bad edge config — orthogonal.
- ❌ Bugs in business logic that you'd write the same way regardless
  of architecture.

## Performance considerations

**Single-consumer concern: does one worker keep up?**

Back-of-envelope:
- Coinbase WS: ~10-20 ticks/sec/product → after coalescing, ~5/sec effective
- Kalshi book WS: ~30-60 deltas/sec across active strikes → coalescing per-ticker, ~10-20/sec effective
- Fills + order events: ~0.1-1/sec
- API responses: 1-1 with cancel/place issuance, so ~5-10/sec

Effective dequeue rate target: **~50 events/sec sustained, ~200 peak.**
Each handler is a few microseconds of pure Python (no IO).  Trivial
for a single thread.

If it doesn't keep up under burst, the coalescing on lane 2 absorbs
the spike automatically.

**Latency:** event arrives → queued (microseconds) → dequeued
(microseconds if worker free, milliseconds if not) → handler runs
(microseconds).  Total internal latency well under 1ms in typical
conditions, 5-10ms under burst.  Compare to ~35ms one-way to Kalshi
— internal latency is invisible.

## Migration path

Three phases, each independently shippable.

### Phase 2a: introduce the queue, keep callbacks as adapters

- Add `Event` dataclasses.
- Add `PriorityEventQueue`.
- Add `worker_loop` that just dispatches to existing strategy methods
  (no logic refactor yet — just bridging).
- Replace WS callbacks with thin adapters that build events and call
  `queue.put`.
- API executor callbacks similarly bridge.

End state: same behavior, but all state mutations happen on the
worker thread.  Locks can be removed; dead-zone flags still present
but redundant.

**Risk: low.**  Behavior-identical refactor.  Bugs caught are
existing bugs surfaced by serialization.

### Phase 2b: rewrite handlers using the new state model

- Replace `current_*_fair` / `current_*_price` with `Quote` objects.
- Replace `_pending_cancel_*` / `_pending_place_*` with
  `pending_ops` dict + timeout-based reconcile.
- Rewrite `_maybe_reprice` against current-theo/current-posted-price
  gate.

End state: the bugs from `strategy_fix_bundle_2026-06-05.md` are
structurally impossible.

**Risk: medium.**  Semantic changes.  Run in paper mode if available
before live cutover.

### Phase 2c: add timer events + reconcile loop

- Periodic `get_orders` reconcile.
- Pending-op timeout sweep.

End state: self-healing under any state divergence.

**Risk: low.**  Pure additions; existing logic continues to work.

## Estimated timeline

- Phase 2a: ~2-3 days of focused work
- Phase 2b: ~3-5 days, plus 1-2 days paper-mode validation
- Phase 2c: ~1 day

Total: **~2 weeks** of focused work to migrate cleanly.  Worth doing
once the 6-05 surgical bundle has demonstrated the fix-bundle EV is
real.
