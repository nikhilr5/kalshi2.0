# Strategy fix bundle — 2026-06-05 deploy

Bundle of 6 changes to `Aston/strategy.py` to deploy at the validation
decision point (2026-06-05).  All fixes target the cancel-pipeline
bleed identified in the live data:

- 1,436 fills had >5s tolerance breach before fill (only 1% had a
  successful cancel land)
- 297 fills demonstrably breached tolerance with zero cancel attempt
  ever recorded
- 162 of those 297 sat ≥2s past breach (strategy had time, didn't act)

The dead-zone in `_reprice_*` plus stale `current_*_fair` plus
`on_fill = pass` form a stack of small issues that compound: a fill
leaves `resting_*_id` stale, the next reprice burns its dead-zone
window cleaning up the ghost, and during that window every same-side
theo tick is silently dropped.

---

## Fix 1 — Tighten tolerance (already queued)

**Where:** wherever `Strategy` is constructed in `app.py`.

**Change:** `tolerance=0.01` → `tolerance=0.001` (1¢ → 0.1¢).

**Why:** moves the reprice trigger boundary much closer to theo.
Combined with the dead-zone fixes below, the reprice chain has more
time to fire before a drift becomes catastrophic.

**Smoke test after deploy:** check `live_dashboard.py` Bleed
Decomposition tab — the `<0.5s breach` bucket should grow (more
quotes flagged as "out of tolerance" earlier), the `>5s breach`
buckets should shrink.

---

## Fix 2 — Clear resting state on full fill (the keystone bug)

**Where:** `strategy.py:140-149` — currently `on_fill` is `pass`.

**Before:**

```python
def on_fill(self, action: str = "", price: float = 0.0,
            count: float = 0, side: str = "yes"):
    """Called when a fill arrives.  Position is already updated by
    the caller — we don't clear order IDs here since the resting
    order may be partially filled and still alive.  Next quote tick
    will reprice as needed.
    """
    pass
```

**After:**

```python
def on_fill(self, action: str = "", price: float = 0.0,
            count: float = 0, side: str = "yes"):
    """Decrement resting count for the matching side; clear state
    when the resting order is fully consumed.  Without this, a
    fully-filled order leaves resting_*_id stale and the next
    reprice burns its dead-zone window cleaning up a ghost."""
    with self._lock:
        if action == "buy" and self.resting_buy_id is not None:
            self.resting_buy_count = max(self.resting_buy_count - count, 0)
            if self.resting_buy_count <= 0:
                self._clear_buy_state()
        elif action == "sell" and self.resting_sell_id is not None:
            self.resting_sell_count = max(self.resting_sell_count - count, 0)
            if self.resting_sell_count <= 0:
                self._clear_sell_state()
```

**Why:** this is the root cause of most of the 297-case cohort.  When
a fill lands, the strategy thinks the order is still resting.  The
next theo tick fires `_reprice_*`, sees a stale `resting_*_id`,
issues a cancel for a non-existent order — that cancel takes a full
RTT to fail with 404 and then clear state.  During that window the
same-side `_pending_cancel_*` flag is True and every other reprice
attempt is silently dropped.

**Smoke test:** after a fill, the next log line should be a fresh
`SELL YES @ ...` or `BUY YES @ ...` from the place callback rather
than a `cancel ... failed: 404` ghost-chase.

---

## Fix 3 — Refresh `current_*_fair` against the latest theo on every reprice

**Where:** `strategy.py:336-373` — `_should_reprice_sell` and
`_should_reprice_buy`.

**Problem:** `current_sell_fair` is the fair value that was true when
`_place_sell` ran its `on_done` callback (line 419), which can be
30-60ms stale by the time the next reprice trigger evaluates.  The
tolerance gate compares `new_fair vs current_sell_fair` so a stale
baseline can suppress legitimate reprices.

**Approach:** after the place callback writes `current_*_fair`,
overwrite it from the *latest* theo observed when the next
`_update_theo_locked` runs.  Simplest concrete change: in
`_update_theo_locked`, before calling `_quote_sell`/`_quote_buy`,
refresh `self.current_sell_fair = theo + self.edge_ask` and
`self.current_buy_fair = theo - self.edge_bid` IF the resting price
hasn't changed since the last evaluation — but only if we want the
gate to compare against the *current* desired fair, not the
historical placed fair.

**Alternative (simpler, equivalent effect):** change the gate from
"how far has fair drifted since last *placement*" to "how far is my
posted price from the *current* fair."  In `_should_reprice_sell`:

**Before:**

```python
diff = new_price - self.current_sell_price
if diff < 0:
    return abs(diff) >= 0.001
if new_fair is not None and self.current_sell_fair is not None:
    return (new_fair - self.current_sell_fair) >= self.tolerance
return diff >= self.tolerance
```

**After:**

```python
diff = new_price - self.current_sell_price
if diff < 0:
    return abs(diff) >= 0.001        # always tighten aggressively
# Worsening: gate on how stale the posted price is vs CURRENT fair,
# not vs the fair we placed at.  Cheaper to compute and never stale.
if new_fair is not None:
    posted_edge = self.current_sell_price - new_fair  # current edge captured
    intended_edge = self.edge_ask
    return (intended_edge - posted_edge) >= self.tolerance
return diff >= self.tolerance
```

**Why:** the new gate asks "is my posted edge meaningfully smaller
than what I want to post?" rather than "has theo drifted since I last
placed?"  The first question is invariant of place-time staleness.
Mirror for `_should_reprice_buy`.

**Smoke test:** the daily-dashboard `theo drift while resting` panel
should compress — fewer fills with large negative drift.

---

## Fix 4 — Clear `resting_*_id` on ALL cancel failure branches

**Where:** `strategy.py:483-487, 504-508, 532-537, 559-564` — all
four `on_done` callbacks for cancel attempts.

**Current pattern (standalone `_cancel_sell`):**

```python
except Exception as e:
    err = str(e)
    print(f"[Strategy] {self.ticker} cancel sell failed: {e}")
    if "404" in err or "400" in err or "not_found" in err.lower():
        self._clear_sell_state()
    # else: do nothing — but Kalshi may have actually cancelled!
```

**After:**

```python
except Exception as e:
    err = str(e)
    err_lower = err.lower()
    print(f"[Strategy] {self.ticker} cancel sell failed: {e}")
    is_terminal = ("404" in err or "400" in err
                   or "not_found" in err_lower
                   or "already" in err_lower)
    is_transient = ("timeout" in err_lower
                    or "500" in err or "502" in err
                    or "503" in err or "connection" in err_lower)
    if is_terminal:
        self._clear_sell_state()
    elif is_transient:
        # Probe Kalshi for ground truth before assuming state.
        try:
            order = self.api.get_order(order_id)
            status = (order.get("status") or "").lower()
            if status in ("canceled", "cancelled", "executed"):
                self._clear_sell_state()
        except Exception:
            pass  # leave state; next tick will retry
```

Apply same pattern to all four callbacks (standalone + chained, buy +
sell).  In the chained `_reprice_*` callbacks, also re-issue the
intended `_place_*` after `_clear_*_state` so the reprice doesn't
strand the strategy with no resting quote on that side.

**Why:** today, a transient 500/timeout on a cancel that *actually
succeeded on Kalshi* leaves the strategy permanently wedged on that
order_id.  Every subsequent reprice cycles through the same failure.
A single `get_order` probe disambiguates.

**Smoke test:** count cancel-failed log lines per `order_id` over a
day.  Today some order_ids appear 5-10 times in the failure log.
Post-fix, no order_id should appear more than 2-3 times.

---

## Fix 5 — Cancel on post-only cross-skip

**Where:** `strategy.py:287-289` (sell) and `:323-325` (buy).

**Before:**

```python
# Post-only: skip if price would cross the bid.
if self.kalshi_bid > 0 and new_sell <= self.kalshi_bid:
    return
```

**After:**

```python
# Post-only: would cross the bid — can't safely post.  Cancel
# any resting order; its price is now economically wrong.
if self.kalshi_bid > 0 and new_sell <= self.kalshi_bid:
    if self.resting_sell_id is not None:
        self._cancel_sell()
    return
```

Mirror for buy.

**Why:** if our resting sell at 53¢ is now crossing a bid that moved
up to 54¢, the current logic silently leaves the stale 53¢ ask
resting — which is exactly the kind of mis-priced quote informed flow
takes.  At minimum, pull it.

**Smoke test:** any `[Strategy] ... alone at ask ...` or
`... alone at bid ...` log line should be followed by a cancel
within the same second.

---

## Fix 6 — Coalesce theo ticks instead of dropping them

**Where:** `strategy.py:172-184` — `update_theo`.

**Before:**

```python
def update_theo(self, theo: float):
    if not self.active:
        return
    if not self._lock.acquire(blocking=False):
        return  # another tick already in flight
    try:
        self._update_theo_locked(theo)
    finally:
        self._lock.release()
```

**After:**

```python
_latest_theo_pending: float | None = None  # class attr or __init__

def update_theo(self, theo: float):
    if not self.active:
        return
    # Always write the latest theo to a coalescing slot.  If the
    # lock is contended, the holder will pick up the latest value
    # in its next iteration; we don't drop ticks.
    self._latest_theo_pending = theo
    if not self._lock.acquire(blocking=False):
        return
    try:
        # Drain the slot — process the freshest theo, not the one
        # we were called with.
        while self._latest_theo_pending is not None:
            t = self._latest_theo_pending
            self._latest_theo_pending = None
            self._update_theo_locked(t)
    finally:
        self._lock.release()
```

**Why:** today a burst of theo ticks loses all but the first one to
the non-blocking lock.  The drained one might be the one whose fair
crossed tolerance.  Coalescing keeps the freshest value and processes
it as soon as the lock frees, with no extra work per tick.

**Smoke test:** during high-vol windows, the count of theo updates
processed per second should approach the count emitted (currently it
caps at a few/s due to dropped ticks; post-fix should track input
rate).

---

## Deploy order

1. **Pre-flight:** snapshot live performance for 24h before the
   deploy.  Note daily P&L, mean theo drift while resting, n fills
   in the >5s breach bucket.

2. **Stop Aston cleanly.**  Wait for any in-flight orders to
   resolve.

3. **Apply fixes in this order:**
   - Fix 2 (`on_fill` clears state) — by itself this should already
     reduce the 297-case bleed measurably.
   - Fix 4 (cancel-failure recovery) — closes the wedged-state class.
   - Fix 6 (tick coalescing) — closes the lock-drop class.
   - Fix 5 (post-only cross-skip cancels) — small but real.
   - Fix 3 (current_fair refresh) — the trickiest semantically; test
     in paper mode first if a paper mode is available.
   - Fix 1 (tolerance 1¢ → 0.1¢) — DEPLOY LAST.  This amplifies the
     reprice trigger rate; the prior fixes ensure the cancel pipeline
     keeps up.

4. **Restart Aston.**  Watch the log for the first 5 minutes —
   `[Strategy] CANCEL SENT` lines (assumes you add the optional
   visibility logging from the earlier proposal) should appear at a
   higher rate than today, and `cancel ... failed` lines should
   drop relative to attempts.

5. **24-48h verification window:** re-pull the same metrics from
   step 1.  Expected directional changes:
   - 297-case-equivalent should drop by >80%
   - >5s breach bucket should shrink by 30-50%
   - mid_drift per fill should improve by 1-2¢
   - daily P&L (if signal-to-noise allows) should improve

6. **If anything regresses unexpectedly, revert in reverse order.**
   Fix 1 is the lowest-risk to revert (config change).  Fix 3 is the
   highest-risk (semantic gate change).

---

## What's NOT in this bundle (deferred to post-deploy validation)

- Bug #6 (cancel_all_orders_local ghost-id resurrection) — only
  matters during shutdown; not a steady-state issue.
- Bug #7 (fill-mid-reprice resizes against pre-fill exposure) — the
  cap-check at `_place_*` is the safety net; works.
- Asymmetric tolerance (tight on adverse moves, loose on favorable)
  — the right v2 design, but the symmetric 0.1¢ captures most of the
  value and is one knob, not two.
- Trade-channel raw msg wired into recorder for authoritative fill
  ts_ms — separate effort, already scaffolded in `_on_fill` /
  `on_fill_raw` wiring.

---

## Estimated impact

Based on the per-fill economics observed:

- 297 cases × ~−5¢/fill avg = ~$15 per day at current 1-lot size on
  ETH alone.
- The remaining "stale quote got hit" tail (the 1,400+ fills with
  >5s breach but no acute drift) is harder to quantify but probably
  another $5-10/day in cents-per-fill cleanup.
- Total expected EV: **+$15-25/day post-fix, with the same
  validation phase config (1-lot, 5/7¢ edges).**
- Scales linearly with size.  At the planned 5-lot post-validation
  sizing, this is +$75-125/day from cancel-pipeline fixes alone,
  separate from any edge/strategy tuning.
