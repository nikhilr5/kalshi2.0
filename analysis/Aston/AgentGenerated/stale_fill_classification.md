# Stale-fill classification — 2026-05-22 (KXETH15M)

Script: `analysis/Aston/AgentGenerated/stale_fill_classification.py`
DB: `analysis/backtesting/data/KXETH15M-26MAY22.db`
Tolerance: 1.0¢ (`Aston/aston_settings.json:8`)

## Cohort

- 1,155 fills today; 1,151 matched to a `placed` event by `client_order_id`.
- Resting-duration percentiles across all matched fills: p50=1.87s, p90=20.94s, p99=326.6s.
- Stale cohort (5s ≤ rest ≤ 30s): **n=257**. The user's "~160" undercounts; the true number is ~60% higher.

## Bucket breakdown

| Bucket | n | % | Definition |
|---|---:|---:|---|
| 3 — no reprice needed | 129 | 50.2% | `|theo_drift| ≤ 1¢` between placement and fill. Tolerance gate at `osm.py:556` (`abs(d - r.price) < self.tolerance`) correctly held. **Not a bug.** |
| 2 — pending-cancel limbo | 116 | 45.1% | Signed adverse drift > 1¢ AND no cancel event ever recorded. Strategy wanted to reprice; the cancel never landed before the fill did. |
| 1 — queue backup (high churn) | 4 | 1.6% | Drift > 1¢ excluded (already in bucket 2 by priority), top-quartile book churn in 2s pre-fill (≥135 updates). |
| 4 — unknown | 8 | 3.1% | Mostly rows with missing `theo_at_placed` (theo_state hadn't written yet, n=6) or sign edge cases. |

Cancel-event lookup matched **0** of 257 stale orders — expected, since the order terminated via fill not cancel. Bucket 2's interpretation hinges on the WS cancel ack arriving after the fill (which would have evicted it from this cohort because the order is gone before the fill happens).

## Diagnosis

**Half the cohort isn't a bug.** Bucket 3 (50.2%) — these are orders OSM correctly held because rounded-desired matched resting price within tolerance. Quoting these as "stale" inflates the failure count. Real failures: ~120 orders, not 257.

**Bucket 2 dominates the real failures (90%+ of bucket 1+2+4).** Median adverse drift on those 116 orders is **+7.83¢** (cohort p90 of `theo_drift_signed`), with max +39.6¢. This is the strategy thinking "fair moved a nickel against me, get out" and not getting the cancel acked in time.

**Bucket 1 and 2 are correlated, not independent.** Case-2 orders have higher median book churn (98 vs 87 cohort-wide) and 31% sit in the top churn quartile vs the unconditional 25%. High BBO churn drives more `_repost` calls into OSM's queue (`strategy2.py:107-128`) AND more theo-drift events to react to. The priority ordering (3 → 2 → 1) puts these in bucket 2, so the queue-backup story is under-represented numerically but is plausibly the upstream cause for a meaningful fraction of bucket 2.

## Recommended fix order

1. **Dedup `_repost` at the strategy2 layer** (`strategy2.py:107-128`). Every BBO tick enqueues an ENSURE on both sides unconditionally. With 135+ book updates in 2s during churn (top quartile), that's hundreds of redundant ENSUREs queued behind a single cancel. Cheapest win: if `(desired_bid, desired_ask, size)` is unchanged from last `_repost`, skip the enqueue. This shrinks OSM's queue depth without changing semantics.

2. **Add order-attempt logging.** Bucket 2 (n=116) is "the strategy wanted to cancel but no cancel WS event ever appeared." Without timestamps of cancel API calls and their responses, we can't distinguish: (a) OSM enqueued the cancel but the worker hadn't drained yet, (b) the cancel request was sent and Kalshi 429'd/5xx'd, (c) the cancel was sent and acked AFTER the fill (race lost). All three feel the same in this analysis. Log `(client_order_id, request_id, kind, sent_ts, response_ts, status)` for every API call OSM makes.

3. **Cancel retry on no-ack timeout** is premature without (2) — we don't know whether retry would help or just make the duplicate-order risk worse.

## Caveats

- 6 of 257 rows have `theo_at_placed = NaN` (theo_state hadn't written yet at session start); they end up in bucket 1 or 4 by current logic. Negligible.
- "Book churn" is the count of `kalshi_book` rows in the 2s prior to the fill. This conflates own-quote updates with external book changes. A clean version requires filtering own-quote updates out, which needs `order_events` cross-referencing.
- Bucket 3's logic uses `|theo_drift| ≤ tolerance`, not "did the rounded-to-tick desired equal the resting tick." These usually coincide but not always (e.g., 0.51 → 0.515 rounds differently bid vs ask). A stricter version would replay `_round_to_tick` (`osm.py:597-607`) per row. The bucket-3 share would shift by a few percent.
