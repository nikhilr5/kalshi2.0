# Kalshi daily-high-temperature model vs market — research report

**Date:** 2026-06-18 · **Scope:** 5 cities (NYC, LAX, OKC, BOS, DAL), day-ahead (Jun 19) markets

---

## TL;DR

- Working pipeline: NWS deterministic high-temp forecast -> Normal(forecast, sigma)
  -> per-bucket probability -> compared to live Kalshi market mids across 5 cities x 6
  buckets each (30 day-ahead markets).
- **sigma is MEASURED**, not assumed: day-ahead forecast-error std from the IEM GFS+NAM MOS
  archive vs observed highs, n=78/station. ~1.9-4.0 F (LAX tight, BOS wide).
- Model bucket probs are internally coherent (each city's 6 buckets sum to 1.000; market
  sums to 0.95-1.06). Disagreement is REAL distributional disagreement, not a normalization
  artifact.
- **One robust, falsifiable disagreement stands out: Dallas.** NWS forecasts a 92 F high;
  the market is 57% confident the high is <=87 F -- a ~5 F gap robust to every modeling
  choice. Cleanest "edge" candidate; the forward harness scores it within ~24h.
- **Verdict:** a *plausible* edge exists in the largest disagreements (DAL, LAX), but it is
  NOT validated. The "19/30 actionable buckets" headline overstates tradability -- most are
  low-prob buckets where a crude Normal shape, a 1-2 F forecast/station offset, or thin
  one-lot liquidity could fully explain the gap. **No historical model-vs-market backtest is
  possible** (no archived Kalshi weather prices) -- validation must accrue forward.

---

## Methodology

### Forecast (the mean)
- NWS api.weather.gov: points/{lat,lon} -> forecastGridData -> maxTemperature.values (C->F).
  Matches the human-facing /forecast daytime high exactly (NYC 83, DAL 92 for Jun 19).
- **Data bug found & fixed:** the grid emits short overnight slivers (e.g. 00:00Z/PT2H)
  carrying the previous afternoon's heat; naive daily-max grabbed these (OKC showed a
  spurious 98 F vs the real 82 F). Fix: keep only intervals starting 10-20 UTC spanning >=6h.

### Distribution (the spread)
- High ~ Normal(mean=forecast, sigma=measured day-ahead err std).
- Buckets cover integer observed highs; widened +/-0.5 F so segments tile the line.
  "83 to 84" -> P(high in [82.5,84.5]); "89 or above" -> P(>=88.5); "80 or below" -> P(<=80.5).
  6 segments sum to 1.0 by construction.

### sigma source -- MEASURED (cache/forecast_error_calibration.csv)
IEM GFS+NAM MOS (12Z run -> next-afternoon n_x) vs IEM observed daily max_temp_f,
n=78/station, 2026-04-01..06-18:

| Station | bias F | err std F | RMSE F | sigma used |
|---|---|---|---|---|
| KNYC | -0.65 | 3.07 | 3.12 | 3.1 |
| KLAX | -0.87 | 1.92 | 2.10 | 1.9 |
| KOKC | -1.83 | 3.34 | 3.79 | 3.3 |
| KBOS | -2.14 | 4.00 | 4.51 | 4.0 |
| KDAL | +0.33 | 3.02 | 3.02 | 3.0 |

Cross-station mean ~3.1 F, consistent with the ~2.5 F literature prior (LAX below, BOS above).
The 2.5 F prior was NOT needed.

### Bias correction -- measured but NOT applied (honesty call)
MOS runs cooler than observed (BOS -2.1, OKC -1.8 F). A first pass warmed the model by -bias,
then **removed it**: bias was measured on MOS but the model forecasts off the NWS gridpoint (a
different product), so the correction is not transferable; empirically it DEGRADED BOS/OKC
agreement (pushed BOS mode into a tail artifact). Bias-free, OKC/BOS now agree closely with
the market -- the expected result if the prior "edge" there was a correction artifact. A
per-product NWS-grid-vs-observed bias calibration is future work.

### Settlement-station mapping (flagged)
NYC->KNYC (Central Park) HIGH confidence. LAX/OKC/BOS/DAL assumed from series name, MED
confidence -- verify before sizing. The NWS gridpoint is an AREA forecast (DAL's resolves near
"Highland Park, TX"; DAL=Love Field not DFW); a 1-2 F grid-vs-station offset is plausible and
is part of the unexplained disagreement.

---

## Core result -- LIVE model vs market (day-ahead, Jun 19)

Full table: cache/live_compare_20260618T171800Z.csv. Top buckets by edge-beyond-spread
(disagreement minus the full spread crossed to take it):

| City | Bucket | Fcst | Model P | Mkt mid | Spread | Edge-spread |
|---|---|---|---|---|---|---|
| DAL | 87 or below | 92 | 0.067 | 0.575 | 0.01 | 0.498 |
| LAX | 70 to 71 | 68 | 0.182 | 0.495 | 0.01 | 0.303 |
| DAL | 92 to 93 | 92 | 0.258 | 0.015 | 0.01 | 0.233 |
| LAX | 66 to 67 | 68 | 0.302 | 0.070 | 0.04 | 0.192 |
| DAL | 94 to 95 | 92 | 0.187 | 0.015 | 0.01 | 0.162 |
| NYC | 80 or below | 83 | 0.210 | 0.375 | 0.01 | 0.155 |
| NYC | 81 to 82 | 83 | 0.226 | 0.380 | 0.02 | 0.134 |
| DAL | 88 to 89 | 92 | 0.136 | 0.280 | 0.02 | 0.124 |
| NYC | 85 to 86 | 83 | 0.185 | 0.055 | 0.01 | 0.120 |
| BOS | 81 to 82 | 83 | 0.184 | 0.295 | 0.01 | 0.101 |

19 of 30 buckets disagree beyond the (tiny 1-8c) spread. Treat that count skeptically.

### Per-city summary
| City | Fcst | sigma | mean |model-mkt| | max disagree | Read |
|---|---|---|---|---|---|
| DAL | 92 | 3.0 | 0.221 | 0.508 | Large, robust -- model warm vs market |
| LAX | 68 | 1.9 | 0.121 | 0.313 | Model cooler than market by ~2 |
| NYC | 83 | 3.1 | 0.096 | 0.165 | Mild; market ~1-2 cooler |
| OKC | 85 | 3.3 | 0.046 | 0.110 | Essentially agrees |
| BOS | 83 | 4.0 | 0.077 | 0.111 | Essentially agrees |

Modal bucket: market's most-likely bucket is 1-3 F cooler than model's in 4 of 5 cities (LAX
the exception). Either the public NWS grid is systematically warm vs settlement station / vs
market, OR there is directional edge -- indistinguishable without outcomes.

---

## Forecast-accuracy backtest (FORECAST side only)
- Day-ahead RMSE 2.1-4.5 F, well below each station's observed day-over-day std (3-11 F):
  real skill over persistence.
- Mild cold bias on MOS (-0.7 to -2.1 F); whether it holds on the NWS grid is unmeasured.
- Implied-prob calibration NOT yet measurable -- needs forecast-bucket-prob vs realized
  outcome, which only accrues forward.

---

## Honest verdict on edge
- **Plausible edge?** Narrowly yes, in the LARGEST disagreements (DAL <=87 at 0.067 vs 0.575;
  LAX 70-71 at 0.18 vs 0.495) -- too big for a 1-2 F station offset to fully explain.
- **Hinges on:** (1) whose temp is right -- public NWS grid vs exact settlement station (a
  systematic offset turns "edge" into uncorrected bias; DAL grid ~Highland Park, settles KDAL
  -- verify); (2) the Normal shape -- real high-temp distributions are skewed/bounded/bimodal
  near fronts, so a symmetric Normal mis-prices tails and manufactures apparent edge in
  low-prob buckets (most of the 19/30); (3) liquidity -- one-lot books, 1-8c spreads, mid is
  a shaky fair value.
- Does NOT hinge on sigma being slightly off -- DAL/LAX gaps are in the MEAN, robust to sigma
  2-5 F and to the bias choice.
- **Bottom line:** promising in DAL/LAX, unproven. Do not size yet. Next step is outcomes, not
  a bigger model.

---

## Validation limits
- **No historical model-vs-MARKET backtest is possible** -- Kalshi archives no historical
  weather prices; no aligned (forecast, price, outcome) panel exists. Any historical edge
  number would be fabricated. Market comparison is LIVE/forward ONLY.
- Forecast-accuracy side IS historically validated (n=78/station) but validates the forecast,
  not the trade.
- sigma measured on a spring window and on MOS, applied as a proxy to the NWS grid product.
  Summer convective regimes (OKC/DAL) likely widen sigma -- recheck mid-season.
- Settlement-station mapping verified only for NYC.

---

## Forward plan (the real validation path)
log_weather.py (run daily) appends one row per day-ahead bucket with forecast, model probs,
live market mid/spread, and -- via IEM max_temp_f backfill -- settled observed high and
realized outcome. After ~3-6 weeks this gives the first honest dataset to:
1. model Brier/log-loss vs market Brier on realized outcomes;
2. a calibration plot (model bucket-prob vs realized frequency);
3. settle DAL-type divergences: when model and market diverge >=2sigma, who wins net of
   spread? (the core edge question, answerable only with outcomes);
4. measure NWS-grid-vs-station bias directly, replacing the borrowed MOS bias.

Sample size: ~5 cities x ~1 clean day-ahead market/day = ~35 settled buckets/week. Brier SE
on binary buckets ~0.5/sqrt(n); expect n ~ 200-400 settled buckets (6-10 weeks) before a
model-vs-market Brier gap separates from noise.

---

## Artifacts (analysis/Aston/weather/)
- weather_lib.py -- shared lib (NWS fetch sliver-filtered, bucket parse, Normal-CDF model,
  book mid/spread); retry-once-then-skip on every external call.
- live_model_vs_market.py -- CORE deliverable; live model-vs-market, ranked by edge-beyond-
  spread; measured sigma, bias documented-but-off.
- log_weather.py -- forward harness; daily snapshot + IEM outcome backfill -> weather_log.csv.
- weather_log.csv -- accruing forward log (30 rows seeded 2026-06-18).
- cache/live_compare_<UTC>.csv -- full live comparison table.
- cache/forecast_error_calibration.csv -- measured per-station bias/err-std/RMSE (n=78).
- cache/observed_high_variability.csv -- observed-high persistence std per station.

Reproduce: cd analysis/Aston/weather && python3 live_model_vs_market.py
Accrue validation: add python3 log_weather.py to a daily cron.
