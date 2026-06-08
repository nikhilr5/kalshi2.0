# BBO Clamping as Root Cause of Missing-Cancel Stale Fills

**Verdict: NOT the cause.** Evidence ruled it out on the 2026-05-22 recorder DB.

## 1. The clamp, located

`Aston/strategy2.py:110-111`:
```python
desired_bid = min(self.theo - self.edge_bid, self.best_bid)
desired_ask = max(self.theo + self.edge_ask, self.best_ask)
```

This is the *only* place clamping happens. There is no second clamp inside OSM's `_send_place` — `osm.py:573-595` snaps to the tick grid but does not bound to BBO. The clamped value is what Strategy hands to `osm.ensure_*`, and is also what the reconciler stores as `desired_*_price` and compares against `resting.price` at `osm.py:556` (`abs(d - r.price) < tolerance` → "keep").

So the architectural pathology the hypothesis describes is real: the reconciler compares **post-clamp desired** vs resting. If the clamp were biting, drift would be invisible to it.

## 2. Why the clamp exists (microstructure)

It enforces a "never quote lonely-on-the-inside" rule. If `theo + edge_ask < best_ask`, joining at theo+edge would establish a new, tighter inside ask all by itself. That's bad for two reasons: (a) you're advertising tighter pricing than the rest of the market is willing to offer, inviting adverse selection from any participant whose information has already moved past your stale theo; (b) on a post-only contract, crossing the spread gets you rejected, and being the lonely best risks you getting picked off by the first informed taker. Joining the existing best is the standard MM safety pattern — you queue *behind* existing depth and let others act as the canary.

## 3. Data evidence (KXETH15M-26MAY22.db, n=343 stale ≥5s maker fills)

| Metric | Value |
|---|---|
| Stale fills with adverse theo drift ≥1¢ | 55 / 343 |
| Mean theo drift in that subset | **5.39¢** |
| Mean same-side BBO move in that subset | **10.88¢** |
| BBO moved in the same direction as theo (adverse) | 100% of the 55 |
| Stale fills placed AT same-side BBO (clamped at place) | 21.9% |
| Stale fills where clamp was still active at fill time | 49.9% |
| Stale fills where current (clamped) reconciler would return KEEP | 24.2% |
| Of those KEEPs, where unclamped desired would FIRE CANCEL | **12 / 83 (14.5%)** |
| Net stale fills the unclamp-the-reconciler fix would catch | **12 / 343 = 3.5%** |

The 55 worst stale fills all have BBO moving *faster* than theo (mean 10.9¢ vs 5.4¢). The market is not pinning ahead of us — the BBO is racing away while our order sits. So the clamp's input (best_bid/best_ask) is moving, the clamped desired is moving with it, and the reconciler's `abs(d - r.price)` is well above tolerance. The cancel is *not* being suppressed by stale clamp output.

For the remaining 96.5% of stale fills, the cancel decision should fire under current logic. The bug must live elsewhere — most likely in *whether the reconciler is being invoked at all* when the BBO updates. The clamp itself is fine.

## 4. Principled fix (recommended even though it isn't the primary bug)

Even at 3.5% incremental capture, the asymmetry is architecturally ugly: a defensive *placement* rule is leaking into the *cancel* decision. They are different MM primitives and should not share state.

**Cleanest design:** track two quantities separately.

- `desired_post_price` = clamped (what to send to Kalshi on `_send_place` — preserves the no-lonely-best, no-post-only-rejection invariant).
- `desired_fair_price` = unclamped `theo ± edge` (what reconciler compares to resting for the cancel-decision tolerance check).

Then `_reconcile_action` compares `abs(desired_fair_price - resting.price)` against tolerance. If it exceeds tolerance, cancel — regardless of whether the new placement would be clamped down again to the same BBO. The placement leg keeps the clamp; the cancel leg sees true fair drift.

Trade-off: occasionally you'll cancel + re-place at the same clamped price (churn). Mitigate by adding a second guard: only fire cancel if `abs(desired_fair - resting) ≥ tolerance` AND `abs(desired_post - resting) ≥ tolerance / 2`. That way you don't churn just to land on the same BBO tick.

## 5. Where to look next for the real bug

96.5% of stale fills land in regimes where the clamp is not the gate. Candidates:
1. **Reconcile not invoked on relevant events** — `_reconcile_both` is called on FILL, API_RESPONSE, RECONCILE, but `_handle_ensure_*` only fires when Strategy sends ENSURE_*. If Strategy's `_repost` isn't running on every BBO/THEO tick (e.g. dedupe at line 96–97 silently swallowing updates when `self.theo == theo` exactly), no ENSURE goes out, no reconcile triggers, the gap widens silently.
2. **Pending-ops gate at `osm.py:536-543`** — if any op on that side is in-flight, reconcile early-returns. A wedged pending-op (e.g. cancel that timed out and was logged but never cleared from `pending_ops`) would suppress all future reconciles on that side.

Both are higher-yield than the clamp.
