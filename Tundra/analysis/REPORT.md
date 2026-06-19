# Tundra weather recorder — build report (2026-06-18)

## 1. What was built

`Tundra/analysis/recorder.py` — standalone, runnable as `python3 recorder.py` from
`Tundra/analysis/`. Mirrors the crypto recorder (`Aston/tools/recorder.py`) in
structure, S3 rotation, and robustness. Reuses the NWS fetch + sliver filter +
bucket parser from `analysis/Aston/weather/weather_lib.py` (loaded by absolute
path, not copied).

Per UTC-day SQLite in `Tundra/analysis/data/WEATHER-{YYMONDD}.db`, three tables:

- **market_snapshots** — every open weather market, every cycle (~5 min):
  `ts, ticker, city, series, event_day, kind, bucket_sub, bucket_kind,
  bucket_lo, bucket_hi, yes_bid, yes_ask, bid_size, ask_size, volume,
  open_interest, last_price`. Bid/ask/size/vol/OI come inline off `get_markets`
  (the `*_dollars`/`*_fp` fields) — **one series = one API call (~15/cycle, not
  170)**. Buckets parsed from `yes_sub_title` to continuous bounds (±0.5°) via
  `weather_lib.parse_bucket`.
- **forecasts** — NWS deterministic daily-high (°F) per high-temp city, one row
  per future event_day, every cycle: `ts, city, station, event_day,
  forecast_high, source`. Forecast cached/re-fetched from NWS every 30 min but
  written every cycle, so snapshot+forecast are contemporaneous. Sliver filter =
  weather_lib rule (keep grid intervals starting 10–20 UTC, ≥6 h).
- **settlements** — on resolve: `result` + observed high read from the market's
  `expiration_value`. `UNIQUE(ticker)`, scanned every 30 min. First scan
  backfills the full settled archive (~5.7k rows) — free historical ground truth
  for calibration.

Robustness: every NWS/Kalshi/aws call is retry-once-then-skip; one bad
market/city/series can't kill the cycle or loop. Graceful SIGINT/SIGTERM → stop
event → clean close. Per-cycle stdout progress.

A correctness bug was caught + fixed in review: `write_settlement` returned
`conn.total_changes` (cumulative for the connection's life) which inflated the
settlement counter → switched to `cursor.rowcount`.

**Run modes:**
```
python3 recorder.py            # all series, 5-min loop, S3 rotation on
python3 recorder.py --once     # one cycle then exit (smoke test)
python3 recorder.py --no-rotate
```

**Test (one `--once` cycle, 2026-06-18):** `snapshots=170 forecasts=112
settlements=5701`. Sample: `KXHIGHNY-26MAY31-B74.5` ("74° to 75°") settled YES
with `observed_high=75.0` — bucket math verified end to end.

## 2. S3 rotation (production-critical)

Mirrors `Aston/tools/recorder.py` ~705–797 exactly: background thread, hourly
scan, each `WEATHER-*.db` older than 24 h (excluding today's), bundle
`.db`+`.db-wal`+`.db-shm`, `aws s3 cp` to **`s3://kalshibtc/weather-archive/`**,
verify with `aws s3 ls`, then `unlink` locally **only on confirmed upload**. Idle
handle closed before delete; every subprocess try/except-guarded.

**Verified against live S3** 2026-06-18: a back-dated fake `WEATHER-26JUN01.db`
was uploaded, verified, deleted locally; today's DB untouched; test object then
removed from S3.

Rotation runs **in-process** (daemon thread), like the crypto recorder — inherits
Terminal AWS creds/TCC, so **no separate cron line needed**. Just run the
recorder.

## 3. Open Kalshi weather universe (2026-06-18)

**15 series, 170 open markets, ~1.32 M volume, ~878 k OI.** Each high-temp city =
12 strike buckets/day (`T` thresholds + `B` ranges); precip = single yes/no.

| Series | City | Type | Mkts | Volume | OI |
|---|---|---|--:|--:|--:|
| KXHIGHLAX | LA | high | 12 | 441,204 | 263,103 |
| KXHIGHNY | NYC | high | 12 | 241,714 | 155,554 |
| KXHIGHMIA | Miami | high | 12 | 121,244 | 80,465 |
| KXHIGHCHI | Chicago | high | 12 | 87,893 | 57,123 |
| KXHIGHTATL | Atlanta | high | 12 | 78,158 | 59,917 |
| KXHIGHTOKC | Oklahoma City | high | 12 | 61,801 | 43,238 |
| KXHIGHTBOS | Boston | high | 12 | 52,130 | 36,127 |
| KXRAINNYC | NYC | precip | 2 | 39,625 | 38,429 |
| KXHIGHPHIL | Philadelphia | high | 12 | 36,822 | 26,787 |
| KXHIGHTDAL | Dallas | high | 12 | 34,386 | 26,523 |
| KXHIGHTDC | Washington DC | high | 12 | 29,106 | 19,125 |
| KXHIGHDEN | Denver | high | 12 | 28,605 | 19,567 |
| KXHIGHTPHX | Phoenix | high | 12 | 26,270 | 19,513 |
| KXHIGHTSEA | Seattle | high | 12 | 24,657 | 17,296 |
| KXHIGHTHOU | Houston | high | 12 | 19,837 | 15,427 |

**14 high-temp cities + 1 precip (NYC).** No low-temp series, no non-NYC precip
open. Two naming generations: **Detailed** `KXHIGH<CITY>` (NY, LAX, CHI, MIA,
PHIL, DEN — rules name the exact site) and **Terse** `KXHIGHT<CITY>` (OKC, BOS,
DAL, DC, HOU, PHX, SEA, ATL — rules name only the city; exact ASOS ambiguous → the
edge risk). **Capacity:** LA + NYC are the franchise (~50% of the complex);
liquidity thins fast past the top ~6 cities.

## 4. Settlement-station mapping (largest edge risk)

Each series settles on the NWS Climatological Report (Daily) for a specific ASOS.
Detailed series state it verbatim in `rules_primary`; terse series give only a
city, so the station below is best-guess — **verify before sizing**.

| City | Series | Rules say | Assumed station | Confidence |
|---|---|---|---|---|
| NYC | KXHIGHNY | "Central Park, New York" | KNYC | HIGH — stated |
| LA | KXHIGHLAX | "Los Angeles Airport, CA" | KLAX | HIGH — stated |
| Chicago | KXHIGHCHI | "Chicago Midway, IL" | KMDW (not KORD) | HIGH — stated |
| Miami | KXHIGHMIA | "Miami International Airport" | KMIA | HIGH — stated |
| Philadelphia | KXHIGHPHIL | "Philadelphia International Airport" | KPHL | HIGH — stated |
| Denver | KXHIGHDEN | "Denver, CO" | KDEN | MED — city only |
| Oklahoma City | KXHIGHTOKC | "Oklahoma City" | KOKC | MED |
| Boston | KXHIGHTBOS | "Boston" | KBOS | MED |
| Dallas | KXHIGHTDAL | "Dallas" | KDAL vs KDFW | MED — verify |
| Washington DC | KXHIGHTDC | "Washington DC" | KDCA | MED |
| Houston | KXHIGHTHOU | "Houston" | KIAH vs KHOU | MED — verify |
| Phoenix | KXHIGHTPHX | "Phoenix" | KPHX | MED |
| Seattle | KXHIGHTSEA | "Seattle" | KSEA | MED |
| Atlanta | KXHIGHTATL | "Atlanta" | KATL | MED |

Highest-risk: **Dallas (KDAL vs KDFW)** and **Houston (KIAH vs KHOU)** — those
airport pairs routinely differ several °F (multiple buckets). Resolve
empirically: once a few days accumulate, match observed daily highs for each
candidate station (IEM CLI product) against the recorded
`settlements.observed_high` — the matching station is the settlement site.

## Next operational priorities
1. Start the recorder (`python3 recorder.py`) so data + forward validation accrue.
2. Nail the **Dallas / Houston** settlement stations empirically (biggest edge risk).
3. Wire **NBM NBP percentiles + IEM CLI** into the model to replace the faked
   Normal (see FEEDS_RESEARCH.md).
