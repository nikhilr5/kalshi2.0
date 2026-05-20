# Kalshi2.0 — project context

## What this repo is

A Kalshi-trading toolkit for 15-min crypto up/down binaries. Four moving
parts:

- **Aston/** — live PyQt6 MM app. Quotes both sides of a Kalshi 15-min
  contract around a HAR-RV-driven theo. Hold-to-close strategy (no
  flatten layer); positions ride to TWAP settlement.
- **Aston/recorder.py** — standalone process that writes per-day SQLite
  files into `analysis/backtesting/data/`. Captures fills, order events,
  spot ticks, kalshi book updates, and theo/sigma state every recompute.
- **PositionManager/** — read-only viewer over those DBs (trade table,
  per-ticker positions, cumulative P&L).
- **analysis/** — Python scripts for offline analysis. `utility.py`
  holds shared helpers (data loading, S3 fetch, implied σ inversion,
  Brier, realized vol). `analysis/Aston/vol_forecasting.py` is the
  current vol-forecast accuracy + Brier-vs-settlement notebook.
- **LiveDashboard/** — pyqtgraph viewer of recorded data: theo + bid +
  ask per day, with market-boundary separators.

## Live trading config (validation phase)

Validation started 2026-05-15. Decision point at **2026-06-05** (3
weeks). Don't suggest strategy changes during this window — measurement
infrastructure only.

Current Aston settings:

- Series: KXETH15M (Ethereum). Stay on ETH for the validation phase.
- Size: 1-lot orders, max position 8.
- Edges: **5¢ ask, 7¢ bid** (asymmetric — vol-seller lean).
- Tolerance: 0.5–1.0¢.
- Auto-off threshold: 90s before close.
- Always-on mode: on (auto-engages on new market roll).

## HAR-RV model

- Estimator: Parkinson (high-low) on 1-min Coinbase candles.
- Horizons: 15m / 30m / 4h / 24h.
- Coefficients (fit on 30 days of ETH-USD candles, 2,756 training rows):
  - β₀ = +0.0314, β_15 = +0.4485, β_30 = +0.1293,
    β_4h = +0.1843, β_24h = +0.1149
  - R² in-sample = 0.474, OOS = 0.566
- Theo: pure N(d2), no TWAP adjustment, no drift correction.

## Baseline results (5 days, 451 settled markets, 2,217 snapshots)

Bucketed snapshots at T-14m/-10m/-5m/-2m/-30s per market; HAR and Mid
scored on identical rows; ticker-clustered bootstrap. Source:
`analysis/Aston/AgentGenerated/brier_root_cause.py`.

- Vol-forecast accuracy: HAR corr 0.62, MAE 10%, RMSE 13% vs
  market-implied corr 0.30, MAE 18%, RMSE 28%. HAR clearly better at
  forecasting realized σ.
- σ-bias: HAR mean σ 33.5% vs realized 32.5% (+1.0% — small over).
  Market implied 37.9% (+5.4% — meaningfully cautious).
- Brier vs Kalshi settlement (overall): HAR 0.125, Market 0.142.
  Gap −0.017, 95% CI [−0.022, −0.012]. HAR beats Mid significantly.
- Brier gap by offset: T-30s −0.062 (huge), T-2m −0.012,
  T-5m/-10m/-14m all ≈ −0.003 to −0.004. **Most of the edge lives in
  the last 30s** — HAR's late-market conviction vs cautious mid.
- Brier by moneyness: excluding deep-OTM resolved markets (|z|>2.0),
  gap is essentially zero (mean −0.0004, CI [−0.005, +0.005]). The
  overall edge is concentrated in deep-OTM markets where HAR correctly
  forecasts ~0 and mid stays at 0.069. In the ATM/near-ATM regime
  where you actually quote, HAR and Mid are tied.
- TWAP-aware theo: no measurable improvement over plain N(d2).

## Data layout

- Recorder files: `analysis/backtesting/data/KX{SERIES}-{YYMONDD}.db`
  (UTC day per file). Tables: `fills`, `order_events`, `spot_ticks`,
  `kalshi_book`, `theo_state`.
- S3 archive: `s3://kalshibtc/archive/`. Daily rotate via LaunchAgent
  `com.aston.daily-rotate` at 4am local; files >24h old get pushed and
  removed locally.
- S3 cache for analysis re-pulls: `analysis/backtesting/_s3_cache/`.
- Settlement cache: `analysis/Aston/.settlements_cache.json` (keyed by
  ticker, value is 0/1 from Kalshi API).

## Defaults / conventions

- All timestamps stored UTC ISO 8601. Local display in CT.
- Prices in dollars (decimal) internally; UI shows cents.
- σ is annualized (24/7 crypto: ANN_MIN = 525,960).
- "theo" = model fair value; "mid" = market midpoint; "implied σ" =
  σ inverted from market mid via N(d2).

## How to help me (general guidance)

- Default to terse responses. Don't restate context unless asked.
- Don't propose strategy changes during the validation window. Suggest
  analysis tools / measurement / plots instead.
- If I ask about daily P&L swings, remind me they're noise at this
  scale and to wait for the weekly cumulative.
- If I propose to flatten manually, ask whether one of the four
  legitimate flatten conditions applies (position too big, regime
  shift, scheduled event, final 60s). Otherwise discourage.
- When writing analysis scripts: helpers go in `analysis/utility.py`,
  scripts stay clean (under ~200 lines).
- Don't add type hints I don't already use, don't add docstrings to
  one-line functions, don't add comments that just restate the code.

## Operational health checks (daily, ~60 seconds)

```bash
# Aston running? — visual on the app
# Recorder running?
ps aux | grep recorder.py | grep -v grep
# S3 rotate working?
launchctl list | grep aston
tail ~/Library/Logs/aston-daily-rotate.log
# Disk space?
df -h /
```

Don't open Aston or PositionManager just to look at P&L.
