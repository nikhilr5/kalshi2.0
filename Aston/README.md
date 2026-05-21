# Aston

PyQt6 market-maker for Kalshi 15-minute crypto up/down binaries.
Quotes both sides of the contract around an HAR-RV–driven theo and
holds to TWAP settlement (no flatten layer).

## Architecture (Strategy2 + OSM)

```
                    ┌───────────────────────────────────────┐
                    │           app.py  (Qt main)           │
                    │  - market discovery & roll            │
                    │  - settings panel (edges/size/etc)    │
                    │  - REST position seed                 │
                    │  - WS book → BBO push                 │
                    │  - WS user_orders → fill push         │
                    └─────┬────────────┬────────────────┬───┘
                          │            │                │
                  BBO     │   THEO     │   SETTINGS     │   FILL
                  updates │  updates   │   updates      │  (WS)
                          ▼            ▼                │
                    ┌─────────────────────────┐         │
                    │       Strategy2         │         │
                    │  ┌───────────────────┐  │         │
                    │  │   queue (FIFO)    │  │         │
                    │  └───────────────────┘  │         │
                    │  single worker thread   │         │
                    │                         │         │
                    │  PURE PRICING:          │         │
                    │   • theo ± edge         │         │
                    │   • clamp at BBO        │         │
                    │   • tail / fair-range   │         │
                    │     guards              │         │
                    └─────┬───────────────────┘         │
                          │                             │
                ENSURE_*  │   UPDATE_TOLERANCE          │
                CANCEL_*  │   UPDATE_MAX_POSITION       │
                          ▼                             ▼
                    ┌───────────────────────────────────────┐
                    │             OSM                       │
                    │  ┌─────────────────────────────────┐  │
                    │  │      queue (FIFO, unbounded)    │  │
                    │  └─────────────────────────────────┘  │
                    │  single worker thread                 │
                    │                                       │
                    │  ORDER-STATE OWNER:                   │
                    │   • position (seeded from REST)       │
                    │   • resting_bid / resting_ask         │
                    │   • pending_ops + place/cancel        │
                    │     futures                           │
                    │   • capacity clamp (max_position)     │
                    │   • tolerance dedup                   │
                    │   • tick-rounding (outward)           │
                    │   • probe-based recovery (5xx)        │
                    │   • orphan-fill buffer                │
                    └─────┬─────────────────────────────────┘
                          │
                  REST    │  place_order / cancel_order
                          │  (httpx + thread-pool executor)
                          ▼
                    ┌─────────────────────┐
                    │       Kalshi        │
                    └─────────────────────┘
```

## Why two queues

The architecture has **exactly two contracts**:

- `app.py → Strategy2.queue` — market events Strategy reacts to
- `Strategy2 + WS + REST → OSM.queue` — order-state mutations OSM serializes

Both queues are single-consumer / single-worker. **No cross-thread
mutable-state reads** between Strategy2 and OSM — Strategy2 doesn't
look at OSM's resting orders or position. OSM clamps requested size
against current state on its own thread, atomically.

This eliminates the legacy race conditions where:

1. A fill callback (WS thread) and a theo recompute (price-feed thread)
   could both write to position simultaneously.
2. Strategy could send a place while an earlier cancel was in flight,
   leading to duplicate orders.
3. A reprice gate could be silently skipped if the strategy lock was
   held, leaving stale quotes resting past tolerance.

## Strategy2 queue messages

| Msg type    | Producer       | Effect                                       |
|-------------|----------------|----------------------------------------------|
| `BBO`       | app (book WS)  | Update `best_bid` / `best_ask`, repost       |
| `THEO`      | app (theo tick)| Update `theo`, repost                        |
| `SETTINGS`  | app (UI panel) | Update edges/sizes, forward tol+max to OSM   |

## OSM queue messages

| Msg type              | Producer            | Effect                                      |
|-----------------------|---------------------|---------------------------------------------|
| `ENSURE_BID/ASK`      | Strategy2           | Clamp size, set desired, reconcile          |
| `CANCEL_BID/ASK/ALL`  | Strategy2           | Drop desired, reconcile                     |
| `UPDATE_TOLERANCE`    | Strategy2           | New cancel/replace threshold                |
| `UPDATE_MAX_POSITION` | Strategy2           | New risk cap                                |
| `FILL`                | app (user_orders WS)| Match against resting or buffer as orphan   |
| `API_RESPONSE`        | REST executor       | Resolve a pending place/cancel              |
| `PROBE_RESULT`        | OSM (self)          | Reconcile after ambiguous API failure       |
| `RECONCILE`           | OSM (self)          | Drive resting toward desired                |

## Lifecycle invariants

- **Reprice never double-quotes.** OSM's reconcile is cancel-then-place:
  the new place fires only after a successful cancel response lands.
- **Stop drains in-flight orders.** `Strategy2.stop()` flips `running=False`,
  joins its worker, then calls `osm.cancel_all_sync()` which (a) sets
  `_stopping=True` to reject new ensure_* commands, (b) waits for any
  in-flight place to land and harvests its `order_id`, (c) cancels every
  resting + newly-landed order.
- **Position cap is checked atomically.** OSM's worker is single-threaded,
  so the capacity check and the place happen against the same state with
  no race vs incoming fills.

## Files

| File                     | Role                                                |
|--------------------------|-----------------------------------------------------|
| `app.py`                 | Qt UI, market discovery, WS wiring, settings, seed  |
| `strategy.py`            | Legacy direct-callback strategy (`-s 1`)            |
| `strategy2.py`           | Queue-based strategy (`-s 2`)                       |
| `osm.py`                 | Order-state manager + Kalshi API boundary           |
| `ws_feed.py`             | Kalshi book + user_orders WS clients                |
| `crypto_feed.py`         | Coinbase spot price feed                            |
| `kalshi_api.py`          | Kalshi REST client (httpx, thread pool, signing)    |
| `theo_engine.py`         | N(d2) probability of YES at expiry                  |
| `har_rv.py`              | HAR-RV vol forecaster (Parkinson, 1-min)            |
| `har_coefficients.json`  | Fitted HAR β coefficients                           |
| `recorder.py`            | Standalone process: writes per-day SQLite + S3 push |
| `market_discovery.py`    | Find active 15-min market for series                |

## Running

```bash
# Legacy strategy
python3 Aston/app.py -s 1

# Queue-based strategy + OSM
python3 Aston/app.py -s 2
```

Default is `-s 1` during the validation phase (through 2026-06-05).
