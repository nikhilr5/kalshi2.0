# Plan: New Analysis Recorder for Weekly BTC Markets

## Context

The app now uses a Kalshi-derived vol smile for theos instead of Deribit. To validate and tune this approach, we need to record the raw inputs (IV, spot, T, strike, market data) at regular intervals so smoothed IV and theos can be recomputed offline with different parameters. The existing recorder in `backtesting/recorder.py` is Deribit-heavy and records to a separate 55GB database. This new recorder is simpler, focused only on weekly BTC markets (Friday 5pm ET expiry), and stores data in `analysis/data/`.

## What gets recorded

Per snapshot (every 5s), for each tracked strike:

| Column | Description |
|--------|-------------|
| `ts` | UTC timestamp (ISO) |
| `ticker` | Kalshi market ticker |
| `event_ticker` | Kalshi event ticker |
| `strike` | Display strike price |
| `close_time` | Event close time (ISO) |
| `T` | Time to expiry in years |
| `kalshi_yes_bid` | Kalshi best bid |
| `kalshi_yes_ask` | Kalshi best ask |
| `bid_size` | Bid depth |
| `ask_size` | Ask depth |
| `spot_bid` | Coinbase BTC bid |
| `spot_ask` | Coinbase BTC ask |
| `spot_mid` | Midpoint |
| `mid_iv` | IV from mid price |
| `bid_iv` | IV from bid price |
| `ask_iv` | IV from ask price |

This gives full freedom to fit smiles and compute smoothed IVs with any span offline.

## New file: `analysis/recorder.py`

### Architecture

Simple single-threaded loop (like existing recorder), no PyQt6:

1. **Discover** weekly BTC events via `discover_events_for_series(api, "KXBTC")`
2. **Filter** to strikes within 8% OTM of spot
3. **Connect** Coinbase price feed + Kalshi WS feed
4. **Snapshot loop** every 5s:
   - For each tracked strike, read current book + spot
   - Compute T from close_time
   - Compute mid_iv, bid_iv, ask_iv using `_implied_vol_quadratic`
   - Insert batch into SQLite
5. **Refilter** every 60s to pick up new markets / drop expired ones

### Reused modules (via sys.path to 4RunnerApp2.0):

- `kalshi_api.KalshiAPI` — REST client
- `ws_feed.KalshiWsFeed` — live orderbook (`on_update(ticker, yes_bid, yes_ask, bid_size, ask_size)`)
- `btc_price_feed.CryptoPriceFeed` — spot price (`on_price(price, bid, ask)`)
- `market_discovery.discover_events_for_series`, `parse_strike`, `display_strike`

### IV computation

Copy `_implied_vol_quadratic` from `app.py` (or import from a shared location). It's a pure function — takes `(price, spot, strike, T, r)` → IV decimal. Use `r = 0.043`.

### SQLite schema (in `analysis/data/recorder.db`)

```sql
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    event_ticker TEXT,
    strike REAL,
    close_time TEXT,
    T REAL,
    kalshi_yes_bid REAL,
    kalshi_yes_ask REAL,
    bid_size INTEGER,
    ask_size INTEGER,
    spot_bid REAL,
    spot_ask REAL,
    spot_mid REAL,
    mid_iv REAL,
    bid_iv REAL,
    ask_iv REAL
);
```

Single table, no sessions overhead. Timestamp + ticker is the natural key.

### Usage

```bash
cd analysis/
python recorder.py           # start recording
python recorder.py --export 2026-05-04  # export a day to CSV
```

## Files to create/modify

| File | Action |
|------|--------|
| `analysis/recorder.py` | **Create** — new recorder (~200 lines) |

No modifications to existing files needed.

## Verification

1. Run `python analysis/recorder.py`
2. Confirm it discovers weekly BTC events and connects to feeds
3. Wait 10-15s, then Ctrl+C
4. Check `analysis/data/recorder.db` has snapshot rows with valid IVs
5. Load in Python: `pd.read_sql("SELECT * FROM snapshots LIMIT 10", conn)` — verify columns populated
