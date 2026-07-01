# Intraday σ-decay curve (2026-06-19)
How wide the daily-high distribution should be as a function of **local time of day** on the settlement day — measured from history, not assumed.

## Method
- **Hourly ASOS temps** (IEM, 2y) → running-max-so-far at each local hour and the day's actual high.
- Estimator at hour *h* = `max(running_max_through_h, same-day NBS forecast)`; **error = estimate − actual high**; σ(station,h) = std(error) over all days.
- The running max is a hard floor (the high can't end below what's already been observed), so σ collapses as the afternoon peak passes.
- n ≈ 684 days per station-hour.

## σ (°F) by local hour
|   hour |   KBOS |   KDAL |   KLAX |   KNYC |   KOKC |
|-------:|-------:|-------:|-------:|-------:|-------:|
|      5 |   2.53 |   2.05 |   2.26 |   1.84 |   1.97 |
|      6 |   2.53 |   2.05 |   2.26 |   1.84 |   1.97 |
|      7 |   2.52 |   2.05 |   2.26 |   1.83 |   1.97 |
|      8 |   2.51 |   2.05 |   2.26 |   1.83 |   1.97 |
|      9 |   2.48 |   2.04 |   2.26 |   1.83 |   1.97 |
|     10 |   2.44 |   2.04 |   2.19 |   1.83 |   1.97 |
|     11 |   2.37 |   2.03 |   1.9  |   1.81 |   1.97 |
|     12 |   2.23 |   1.97 |   1.53 |   1.69 |   1.93 |
|     13 |   1.96 |   1.83 |   1.21 |   1.54 |   1.87 |
|     14 |   1.61 |   1.61 |   1.08 |   1.24 |   1.65 |
|     15 |   1.4  |   1.38 |   1    |   1.02 |   1.38 |
|     16 |   1.24 |   1.13 |   0.94 |   0.88 |   1.13 |
|     17 |   1.17 |   0.92 |   0.94 |   0.83 |   1.01 |
|     18 |   1.09 |   0.87 |   0.91 |   0.8  |   0.98 |
|     19 |   1.01 |   0.87 |   0.88 |   0.78 |   0.98 |
|     20 |   1    |   0.87 |   0.88 |   0.77 |   0.98 |
|     21 |   0.99 |   0.87 |   0.88 |   0.75 |   0.98 |

## bias (°F, estimate − actual) by local hour
|   hour |   KBOS |   KDAL |   KLAX |   KNYC |   KOKC |
|-------:|-------:|-------:|-------:|-------:|-------:|
|      5 |  -1.47 |  -1.06 |  -1.32 |  -1.11 |  -0.92 |
|      6 |  -1.46 |  -1.06 |  -1.32 |  -1.1  |  -0.92 |
|      7 |  -1.46 |  -1.06 |  -1.32 |  -1.09 |  -0.92 |
|      8 |  -1.45 |  -1.06 |  -1.32 |  -1.09 |  -0.92 |
|      9 |  -1.42 |  -1.06 |  -1.31 |  -1.08 |  -0.91 |
|     10 |  -1.36 |  -1.05 |  -1.21 |  -1.07 |  -0.91 |
|     11 |  -1.24 |  -1.04 |  -0.79 |  -1.05 |  -0.91 |
|     12 |  -1.06 |  -1    |  -0.29 |  -0.94 |  -0.88 |
|     13 |  -0.7  |  -0.87 |   0.04 |  -0.67 |  -0.78 |
|     14 |  -0.31 |  -0.57 |   0.19 |  -0.3  |  -0.5  |
|     15 |  -0.04 |  -0.23 |   0.25 |  -0.03 |  -0.11 |
|     16 |   0.1  |   0.08 |   0.28 |   0.12 |   0.2  |
|     17 |   0.17 |   0.28 |   0.28 |   0.16 |   0.33 |
|     18 |   0.2  |   0.31 |   0.28 |   0.17 |   0.35 |
|     19 |   0.25 |   0.32 |   0.29 |   0.18 |   0.35 |
|     20 |   0.25 |   0.32 |   0.29 |   0.19 |   0.35 |
|     21 |   0.26 |   0.32 |   0.29 |   0.2  |   0.35 |

## How to use
On the settlement day, at local hour *h*: mean = `max(high-so-far, forecast − bias)`, and σ = the value from the table for that hour. Plug into the same Normal-CDF bucket math. Early morning ≈ the day-ahead σ; by mid-afternoon σ is a fraction of it and the running-max floor kills the low buckets outright.

## Key points
1. σ **decays through the day** — full (~day-ahead) at dawn → near-0 after the ~3–5pm peak. Using a flat lead-0 σ all day is the bug this fixes.
2. **LA collapses fastest** — floor active 66% by noon, 82% by 2pm (marine-layer burn-off → an early, stable midday peak), so LA σ is ~1.0 by 2pm. The **plains (OKC/DAL) hold uncertainty latest** — floor only ~40% at 2pm because afternoon convective heating/clouds keep the high live into late afternoon. NYC/BOS are in between.
3. The **morning cold bias self-corrects**: ~−1.1°F at dawn → ~0 by mid-afternoon as observation overrides the stale forecast. Use the hour-matched bias, not the flat lead-0 −2°F.
4. **Floor-active %** (running max ≥ forecast) climbs 10%→83% through the day — by ~4pm the high is already set on >80% of days, which is what drives σ to ~0.8°F. Past ~4pm there is essentially nothing left to trade.

## Caveats
- Truth = hourly-METAR daily max, ~1°F vs CLI settlement value.
- σ is across all seasons; convective spread (summer plains) averaged in.
- Estimator uses the static same-day forecast for the *remaining* hours; a real remaining-hours model (NBM hourly) would tighten the midday band further. This curve is therefore a conservative (upper-bound) σ.

## Files
- `cache/intraday_sigma.csv` — station × hour → n, bias, σ, floor-active%
- `cache/intraday_sigma_raw.csv` — every day×hour error
- `build_intraday_sigma.py` — this build
