# Historical forecast-error σ table (2026-06-19)
Empirically-measured daily-high forecast error for the five modeled settlement stations: **σ** (distribution width) and **bias** (systematic offset, forecast − observed) by **station × lead × season**.
## Method
- **Forecast** — IEM MOS archive. Daily high = max over the local calendar day of the model's temp / max-min line. One run per day (NBS 13Z, NBE/MEX/GFS 12Z) → independent daily samples. Truncated horizon-edge days (no afternoon coverage) dropped.
- **Truth** — IEM ASOS daily `max_temp_f` at each station (METAR-derived; ~1°F vs the CLI value Kalshi settles on).
- **error = forecast_high − observed_high**, per local day. σ = std(error), bias = mean(error). Window 2024-06-19 → 2026-06-19 (2y); n ≈ 684/station/lead for NBS, 728 for NBE.
- **Recommended model per lead** — NBS (NBM short) leads 0–2, NBE (NBM extended) leads 3–7. NBM was tightest at every lead vs GFS-MOS; NBE and the independent GFS-extended (MEX) agreed within ~0.2°F, so the σ curve is cross-validated.

## σ (°F) by station × lead — pooled seasons
| station   |    0 |    1 |    2 |    3 |    4 |    5 |    6 |    7 |
|:----------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| KBOS      | 3.07 | 3.28 | 3.38 | 5.11 | 6.05 | 7.07 | 7.56 | 8.03 |
| KDAL      | 2.56 | 2.85 | 3.35 | 4.16 | 4.69 | 4.73 | 6.2  | 5.19 |
| KLAX      | 2.33 | 3.1  | 3.45 | 2.73 | 2.88 | 2.85 | 2.99 | 2.86 |
| KNYC      | 2.39 | 2.55 | 2.93 | 3.5  | 3.93 | 4.53 | 4.88 | 5.56 |
| KOKC      | 2.87 | 3.07 | 3.6  | 4.93 | 5.76 | 6.72 | 7.9  | 8.08 |

## bias (°F, forecast − observed) by station × lead
| station   |     0 |     1 |     2 |     3 |     4 |     5 |     6 |     7 |
|:----------|------:|------:|------:|------:|------:|------:|------:|------:|
| KBOS      | -2.58 | -0.41 | -0.43 | -1.67 | -2.55 | -2.74 | -2.4  | -2.64 |
| KDAL      | -2.04 | -0.27 | -0.23 | -0.58 | -0.34 | -0.5  | -0.27 | -0.7  |
| KLAX      | -2.19 | -0.18 | -0.2  |  0.81 |  1.11 |  1.48 |  1.87 |  2.05 |
| KNYC      | -2.2  | -0.44 | -0.49 |  1.17 |  1.04 |  1.04 |  1.27 |  1.39 |
| KOKC      | -1.9  | -0.38 | -0.43 | -0.62 | -0.23 |  0    |  0.09 | -0.27 |

## σ (°F) by station × season — trading leads (0–2)
| station   |   DJF |   MAM |   JJA |   SON |
|:----------|------:|------:|------:|------:|
| KBOS      |  2.56 |  4.32 |  3.29 |  2.52 |
| KDAL      |  3.53 |  3.27 |  2.33 |  2.07 |
| KLAX      |  3.06 |  2.57 |  2.12 |  3.65 |
| KNYC      |  2.67 |  3.5  |  2.13 |  1.85 |
| KOKC      |  4.05 |  3.32 |  2.53 |  2.25 |

## Key findings
1. **σ grows ~linearly with lead**: ~2.5°F day-of → ~3.4°F at 2 days → ~6–7°F at 7 days. Use the lead-matched σ, never a global one.
2. **Lead-0 cold bias ≈ −2°F** at every station: the same-day morning (13Z) run systematically under-forecasts the afternoon high. Subtract it — it's the single most actionable correction.
3. **Season dominates σ** (swings 2.0–4.5°F within one station): Dallas/OKC are *easiest in summer* (stable heat ridge, σ≈2.5) and hardest in winter (front timing, σ≈3.6–4.1); Boston worst in spring (σ4.5); LA worst in autumn (Santa Ana, σ3.8). Naive 'summer=hard' is wrong for the southern plains.
4. **NBM beats GFS-MOS** at every lead (e.g. lead-1 σ 2.98 vs 3.30) — confirms NBM as the right live feed.
5. **LA is most predictable far out** (σ 3.9→5.1 over leads 3–7) vs OKC/BOS/NYC blowing out to ~7°F — LA carries tradable edge to longer leads; continental stations decay fast.

## Caveats
- Truth is ASOS daily max, not the CLI settlement value (~1°F gap that flips near-the-money buckets). Re-fit on CLI once the recorder accrues settled highs.
- Station map unverified for terse series (esp. **DAL** KDAL vs KDFW, and LA KLAX vs downtown USC) — a wrong station injects a fixed offset that masquerades as bias.
- 2-year window: ~2 samples/season/year. Extend to 5–10y (IEM has MOS to 2003) for tighter seasonal σ.
- **Model handoff (lead 2→3, NBS→NBE)** can make σ non-monotonic — e.g. LA's NBE σ (~2.9) sits below NBS's lead-2 σ (3.45) because NBM-extended is genuinely tighter for LA than NBM-short's hourly max. Real, but don't read the splice as a smooth curve.

## Files
- `cache/sigma_recommended.csv` — station × lead × season → n, bias, σ, rmse
- `cache/sigma_model.json` — nested lookup for the live model to import
- `cache/sigma_table.csv` — full station × model × season × lead grid
- `cache/sigma_history_errors.csv` — every matched day-forecast (48,280 rows)
- `build_sigma_table.py` (pull+align), `finalize_sigma.py` (this)
