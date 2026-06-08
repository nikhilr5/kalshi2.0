# Stale fills — no cancel attempted: root cause

**Data:** `analysis/backtesting/data/KXETH15M-26MAY22.db`
**Scope:** 199 adverse-stale fills (rest ≥5s AND |theo drift| >1¢) on 2026-05-22.

## TL;DR

Two distinct bugs account for the no-cancel-attempted stale fills. **Mode A
(failed-create orphans) is dominant (~62% of cases)**. Mode B (tolerance
gate in price-space) is a separate, smaller contributor (~8%).

| Mode | Mechanism | Count | % |
|---|---|---|---|
| **A. Orphan from failed create** | local HTTP error (Errno 35 / EAGAIN), order actually landed on Kalshi, OSM never recorded `server_order_id` | ~123 | 62% |
| **B. Tolerance gate (price-space)** | `_reconcile_action` compares raw desired vs snapped resting with 1¢ tolerance; bid/ask snap to tick erases the signal | ~16 | 8% |
| (remainder) | likely Mode A pre-instrumentation, or true cancel races | ~60 | 30% |

The user's hypothesis #1 (tolerance gate) is real but minor. Hypothesis #2 (BBO clamp) is **not observed** in the data. Hypothesis #3 (OSM losing track of the order) is the **dominant root cause** but via a very specific mechanism: the `_send_place` retry on local error spawns a duplicate on Kalshi.

---

## Mode A: Orphan from failed create (DOMINANT)

### Mechanism

1. `osm._send_place` calls `api.create_order_async(...)`.
2. The HTTP send fails locally with **`[Errno 35] Resource temporarily unavailable`** — macOS's EAGAIN on a non-blocking socket write that couldn't be queued. **The actual POST was still transmitted to Kalshi and accepted.**
3. `on_done` callback (osm.py:580) catches the exception and enqueues `("API_RESPONSE", (req_id, False, "[Errno 35] ..."))`.
4. `_handle_api_failure` (osm.py:382) classifies the error:
   - terminal (404/400/`not_found`/`already`) → no
   - transient (`timeout`/500/502/503/`connection`) → **"Resource temporarily unavailable" matches none of these** → falls through to "unknown" branch, line 436: `print(...)` and returns.
5. `_handle_api_response` then calls `_reconcile_both()` (osm.py:380). Since `pending_ops` is now empty for this side and `resting_bid` is still `None`, `_reconcile_action` fires a **second `_send_place` at the same desired price**.
6. The second create succeeds. OSM sets `resting_bid` to the *second* `server_order_id`. **The first order is live on Kalshi but unknown to OSM.**
7. When the next reprice happens, OSM cancels the second order. The first orphan sits until it fills (often into adverse theo move).

### Code citations

- `osm.py:382-436` — `_handle_api_failure`. The error classifier is incomplete: `is_transient` does not match `"Resource temporarily unavailable"` / EAGAIN / EWOULDBLOCK.
- `osm.py:418-436` — place-side failure path: no record kept of the request, no probe, no retry-defer. Resting state stays as-is. Next reconcile fires a fresh place.
- `osm.py:573-595` — `_send_place`. `pending_ops[req_id] = PendingOp(...)` is registered AFTER `api.create_order_async`, but this is not the bug — the failure is the local error classification.

### Evidence

Of the 199 adverse-stale fills:
- **383 client_order_ids** had a same-price duplicate-placed twin within 2s where the first was never cancelled — the orphan signature.
- **104** of those orphans were also adverse-stale fills (Mode A).
- **123** of the 199 had a same-side `placed` event during the rest window (wider window orphan detection).

For the 5 orphan cases in the order_attempts window (post-17:15, where the new instrumentation captured local errors):

```
coid=aston_1323a320-...  err=[Errno 35] Resource temporarily unavailable  → filled 12 minutes later @ 47¢, theo had crashed to 30¢
coid=aston_25ba5111-...  err=[Errno 35] Resource temporarily unavailable  → filled 81s later
coid=aston_8790a386-...  err=[Errno 35] Resource temporarily unavailable  → filled 78s later @ 99.8¢
coid=aston_d265d915-...  err=[Errno 35] Resource temporarily unavailable  → filled 78s later @ 99.8¢
```

All five orphan cases in the instrumented window match: local create attempt success=0 with `Errno 35`, BUT order_events shows a `placed` event from the user_orders WS feed with a real `server_order_id`. Kalshi accepted the order; OSM was told it failed.

Five worst-drift Mode A examples (sorted by adverse drift):

```
side=ask rest=29s   drift=28.7¢  placed@31¢   raw_des=59¢  theo 26→54¢   (orphan twin: aston_51d8db76)
side=bid rest=715s  drift=24.4¢  placed@47¢   raw_des=23¢  theo 54→30¢   (orphan twin: aston_ecf1d4ae)
side=bid rest=6s    drift=19.4¢  placed@65¢   raw_des=46¢  theo 72→53¢   (orphan twin: aston_860eea8d)
side=bid rest=233s  drift=19.3¢  placed@85¢   raw_des=66¢  theo 92→73¢   (orphan twin: aston_d709d01d)
side=bid rest=23s   drift=16.9¢  placed@38¢   raw_des=20¢  theo 44→27¢   (orphan twin: aston_07f41e5f)
```

In each, OSM did cancel/reprice the twin (e.g. `aston_ecf1d4ae`) repeatedly while the orphan sat invisible. The orphan eventually filled when theo crashed through the orphan's price.

---

## Mode B: Tolerance gate in price-space

### Mechanism

`osm.py:556`:
```python
if d is not None and r is not None and abs(d - r.price) < self.tolerance:
    return
```

- `d` = `self.desired_bid_price`, stored at osm.py:242 from `strategy2._repost` as the **raw, unsnapped** price: `min(self.theo - self.edge_bid, self.best_bid)`.
- `r.price` = the **snapped-to-tick** price recorded when the place succeeded (osm.py:594 `op.price = snapped` → osm.py:445 `Quote(price=op.price)`).
- `self.tolerance` = 0.01 (1¢, set from app.py:1157 `self.tolerance_input.spin.value() / 100.0` with default 1.0¢).

With tolerance set to **exactly one tick**, the gate fires when `|raw_desired - snapped_resting| < 1¢`. After bid-floor or ask-ceil rounding, the raw can stand up to ~1¢ away from the resting tick while the gate still says "keep". In particular:

- For an ask resting @ 88¢ with raw_desired = 88.0¢ (clamped to BBO ask): diff = 0, gate fires, no cancel. (Case oid=986ca100, drift = 1.7¢ adverse.)
- For a bid resting @ 57¢ with raw_desired = 56.59¢: diff = 0.41¢, gate fires, no cancel. (Case oid=00a937d0, drift = 2.6¢ adverse.)

In both, raw fair theo moved >1¢ against the position but the gate's *price-space* comparison didn't react because of tick rounding plus the BBO clamp.

### Evidence

16 of the 199 adverse-stale fills match Mode B (no orphan twin, but `|raw_desired - resting_price| < 1¢` at the moment of fill). Worst examples:

```
side=ask rest=6s    drift=4.2¢  placed@68¢   raw_des=68.6¢  diff=0.6¢
side=bid rest=7s    drift=2.9¢  placed@53¢   raw_des=53.0¢  diff=0.0¢
side=ask rest=26s   drift=2.8¢  placed@45¢   raw_des=45.0¢  diff=0.0¢
side=bid rest=7s    drift=2.6¢  placed@57¢   raw_des=56.6¢  diff=0.4¢
side=ask rest=10s   drift=1.8¢  placed@91.4¢ raw_des=92.1¢  diff=0.7¢
```

These align with the moneyness-bleed pattern previously seen: the BBO clamp (strategy2.py:110-111 `min(theo - edge, best_bid)` / `max(theo + edge, best_ask)`) pins desired at the resting price's tick, hiding the underlying drift signal.

---

## Mode C: BBO clamp (NOT a primary cause)

Not observed as an independent failure mode. The BBO clamp does contribute to Mode B (clamping desired to a price that happens to equal resting), but in no case did the data show desired pinned to an unchanged BBO with no other failure.

---

## Hypothesis #3 was right (sort of)

The user's third hypothesis — "OSM losing track of the order" — was correct. The specific path is not `pending_ops` corruption or `_handle_api_success` race, but rather: a **local HTTP error misclassified as "no order created"** when in fact the order was created server-side. The order_id never enters OSM's state.

---

## Fix

### Primary (Mode A) — fix `_handle_api_failure` for place errors

The classifier in `osm.py:382-436` must treat all *ambiguous* local errors as "order may have landed; probe to find out". Two minimally-invasive options:

**Option 1 (surgical, recommended):** Add an explicit probe after any non-terminal place failure. Place errors fall into three buckets:
- terminal (Kalshi explicitly rejected) — safe to ignore.
- ambiguous (anything else — Errno 35, timeout, network) — **request landed unknown**. Probe Kalshi for our orders on this ticker+side, adopt any that match (price, side, action) as `resting_*`, then let the next reconcile cycle decide whether to cancel.

```python
elif op.kind == "place":
    if is_terminal:
        # Place rejected (invalid price, post-only would cross, etc.).
        pass
    else:
        # Ambiguous — request may have landed on Kalshi without OSM
        # recording the server_order_id.  Probe to adopt the actual
        # resting order before issuing any further places on this side.
        # NOTE: probe path was disabled (2026-05-21) but the failure
        # mode it was protecting against (clobbering resting_*) is
        # exactly what we need here; re-enable for the place-failure
        # case specifically.
        self._schedule_probe(op.side, expected_price=op.price)
```

The probe handler (`_handle_probe_result`, osm.py:483-525) already adopts a Kalshi-side order as the new `resting_*`. Re-enabling it only for the place-failure path avoids the original disabled-on-2026-05-21 problem (probes during normal reconciles racing fresh places), because we only probe when we *know* a place may have orphaned.

**Option 2 (defensive, complementary):** Widen `is_transient` to include EAGAIN/EWOULDBLOCK explicitly. This is just better classification — does not fix the orphan, only ensures we log it as transient. Useful alongside Option 1.

```python
is_transient = ("timeout" in err
                or "500" in err or "502" in err or "503" in err
                or "connection" in err
                or "errno 35" in err
                or "errno 11" in err               # EAGAIN linux
                or "temporarily unavailable" in err)
```

### Secondary (Mode B) — fix tolerance gate to compare in snapped space

Change `_reconcile_action` (osm.py:556) to compare the **snapped** desired vs resting, since resting is always on a tick:

```python
# Before:
if d is not None and r is not None and abs(d - r.price) < self.tolerance:
    return

# After: compare against the snapped desired so the gate sees what
# would actually be placed.  Avoids the 1¢-tick + 1¢-tolerance trap
# where raw drift up to 1¢ is hidden by tick rounding.
if d is not None and r is not None:
    snapped_d = self._round_to_tick(d, side)
    if snapped_d == r.price:
        return   # same tick → no need to act
```

This also lets the user lower `tolerance` back toward zero without losing the no-churn behavior. With `tolerance=0.005¢` (already validation-tested per memory `aston_adverse_selection_investigation`), the snapped-equality check is the cleaner correct behavior: only repost when the actual placeable tick changes.

### Tertiary — `_send_place` register-before-submit

Move `self.pending_ops[req_id] = PendingOp(...)` to *before* `api.create_order_async`. Not load-bearing for the current bug, but eliminates a real (if rare) window where `add_done_callback` could fire on an already-done future before the registration. Mechanical fix, low risk:

```python
def _send_place(self, side, price, size):
    snapped = self._round_to_tick(price, side)
    req_id = str(uuid.uuid4())

    # Register BEFORE submitting so a synchronously-completing future
    # (or a same-thread on_done) can never find an empty pending_ops.
    self.pending_ops[req_id] = PendingOp(
        request_id=req_id, kind="place", side=side, price=snapped, future=None)

    def on_done(future):
        try:
            resp = future.result()
            self.queue.put(("API_RESPONSE", (req_id, True, resp)))
        except Exception as e:
            self.queue.put(("API_RESPONSE", (req_id, False, str(e))))

    f = self.api.create_order_async(...)
    self.pending_ops[req_id].future = f
    f.add_done_callback(on_done)
```

---

## Confirmation metric

After deployment, the following should drop sharply:

1. **Stale-fill orphan rate**: number of fills where the matching `placed` event was followed by no `cancelled` event AND no `filled` event on the *paired* order_id within 2s. Today's count: 119. Target: <10/day.
2. **Failed-create-followed-by-WS-placed**: count rows in `order_attempts` with `request_type='create' AND success=0 AND error_msg LIKE '%Errno%' OR '%temporarily unavailable%'` that have a matching row in `order_events` with `event_type='placed'` and same `client_order_id`. Today's count: 8 (in the small instrumented window). Target: 0 after Option 1 probe is in place (probes either adopt or clear).
3. **Tolerance-gate slips**: count adverse-stale fills (rest≥5s, |theo drift|>1¢) where `|raw_desired_at_fill - placed_price| < 1¢` and no cancel attempt. Today's count: ~16. Target: <3/day after the snapped-equality fix.

Run `stale_no_cancel_rootcause.py` against tomorrow's recorder DB. The Mode A bucket should approach zero; Mode B and ?_OTHER should both drop.
