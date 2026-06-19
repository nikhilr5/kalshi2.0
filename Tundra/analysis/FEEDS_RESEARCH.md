# Weather data feeds — research (2026-06-18)

⚠️ marks items to confirm before building.

## 0. Framing
- **Settlement source:** NWS **Climatological Report (Daily) = the CLI text
  product**, "Observed Value / Maximum" row, local-standard-time calendar day,
  read off a specific ASOS per city. Settlement can lag if CLI disagrees with
  METAR highs.
- **Goal:** replace `Normal(point_forecast, sigma)` with a **calibrated,
  station-specific probabilistic daily-high**.
- **Two edge levers:** (1) calibrated probability — get the distribution *shape*
  right (favor NBM percentiles / GEFS / ECMWF-ENS members, not point+assumed σ);
  (2) station specificity — settlement is an exact ASOS; a grid cell 5 mi off is
  systematically biased. Station-native products (MOS/NBM text) bake station bias
  in; gridded GRIB does not.

## 1. NWS api.weather.gov gridpoint + NDFD
**Deterministic only** — `maxTemperature` exists, but the only probabilistic
layers are PoP/thunder; **no percentile maxTemperature.** REST/JSON, no key,
requires `User-Agent`. Gridded 2.5 km, no station output. Cadence: Days 1–3
sub-hourly, Days 4–7 at 00/06/12/18/22 UTC. **Value: low** — sanity cross-check,
can't price a threshold probability. (This is the deterministic anchor the
recorder already captures.)

## 2. NOAA NBM — THE KEY FEED
**NBM v5.0** (operational ~2026-05) added **probabilistic MaxT/MinT as
percentiles** (CONUS + AK) via quantile-mapped processing.
- **`NBP` text bulletin = cleanest daily-Tmax percentile source.** Row labels:
  `TXNP1`=10th, `TXNP2`=25th, `TXNP5`=50th, `TXNP7`=75th, `TXNP9`=90th percentile
  of daily max/min (°F); `TXNMN`/`TXNSD` = mean/std. **Station-native, 9,000+
  stations keyed by ICAO** — `KNYC`/`KLAX`/`KMDW` addressable directly. Viewer:
  `blend.mdl.nws.noaa.gov/nbm-text?ele=NBP&sta=KNYC`. ⚠️ TXN window is
  **12Z–06Z UTC, not local-midnight LST** — needs per-station alignment (esp.
  west coast). NBS/NBE/NBX carry deterministic TXN only.
- **GRIB2 qmd product** (`s3://noaa-nbm-grib2-pds`, us-east-1, no key): full
  5%-step percentile curve + `prob >threshold` exceedance fields, refreshed
  hourly. ⚠️ The verified `TMP ... X% level` fields are *instantaneous* 2 m temp;
  the **daily-MaxT** percentile field token + forecast-hours were not fully pinned
  — inventory qmd `.idx` at window-closing hours or read the NBMv5.0 percentiles
  PDF.
- **Cadence:** CONUS qmd hourly; NBP full data at **01/07/13/19 UTC**. Latency
  ~1.5–2 h.
- **Value: highest of all free feeds.** Read `TXNP1/P2/P5/P7/P9` per station for a
  station-calibrated 10/25/50/75/90 daily-high distribution; interpolate for
  arbitrary strikes or use qmd exceedance fields.

## 3. GEFS
**GEFSv12, 31 members** (`c00` + `p01–p30`), ~25 km native. Build a full empirical
Tmax distribution from member-maxima. Access: NOMADS `gefs_atmos_0p25s` filter or
`s3://noaa-gefs-pds` (`--no-sign-request`); IEM does not redistribute GEFS GRIB
(use Herbie/Planetary Computer). 4 cycles/day. **Gridded only** — nearest-point
bias material at coastal/elevation sites. **Value:** genuine distribution (directly
answers P(high ≥ strike)) but needs a station bias-correction layer; best as a
distribution-width source layered on a station-native center.

## 4. MOS (GFS MAV/MEX, NAM MET, LAMP)
**Station-native text, per ASOS** — daily high is the `N/X` line (⚠️ order flips:
00Z run = X/N, 12Z = N/X). **Temperature is a single deterministic value — no
distribution.** This is almost certainly what the current `Normal(point, σ)` is
built on. MAV 4×/day; MEX 2×/day. Archive via **IEM** `mos.json` back to
2003-12-16. Daily Tmax MAE ≈ 2–3 °F day 1 → ~6 °F day 5+. **Fit σ empirically per
station/season/lead — don't use a global σ.** **Value:** excellent station-native
deterministic center + 20-yr calibration archive; you supply the distribution.

## 5. ECMWF (IFS HRES + ENS)
**ENS = 51 members at 9 km native.** As of 2025-10-01 all IFS/AIFS data is
CC-BY-4.0, €0 data cost. ⚠️ Open caveat: the easy free path (`ecmwf-opendata`
client + `s3://ecmwf-forecasts`) is confirmed free at **0.25°** (includes ENS
members + `2t`); unclear whether full 9 km is freely served. Runs 4×/day,
open-data ≈ 7–8 h latency. **Gridded only.** **Value: gold-standard calibrated
probability** (51 members → empirical CDF), but the easy free path is coarse 0.25°
so station bias-correction matters even more. Best as an **ensemble second
opinion** to blend with NBM.

## 6. IEM — CRITICAL FOR BACKTESTING & CALIBRATION
Hub `mesonet.agron.iastate.edu/api/`, free, ~1 req/s. **Gotcha: three different
"daily high" products, only one matches settlement** — METAR-derived max (~1 °F
off, NOT settlement-grade), DSM (whole-°F), and **CLI/CF6 (the official NWS
climate report = the Kalshi settlement value)**. Settlement-grade observed:
`/cgi-bin/afos/retrieve.py?pil=CLIxxx` (⚠️ map each station to its CLI PIL — KNYC →
`CLINYC`; not always the airport ICAO). Forecast/MOS archive:
`/api/1/mos.json?station=...&model=GFS|MEX|NAM|NBS|NBE`. **Value:** the backbone of
the upgrade — pull historical NBM/MOS forecasts + CLI observed highs, fit
per-station/season/lead **bias + σ**, validate calibration (reliability/Brier).
**Backtest against CLI, not `daily.py`/METAR max.**

## 7. Commercial APIs
Deterministic point + PoP for 3 of 4. **Tomorrow.io** has true ensemble temp
percentiles `temperaturePXX` but only for instantaneous `1m`/`1h` temp, not the
daily-max aggregate (max of hourly P90s ≠ P90 of daily max). Visual Crossing best
for backtest convenience (50-yr history, $0.0001/record) but adds nothing
probabilistic; Weatherbit/OpenWeather precip-prob only. **Off-list but notable:
Open-Meteo's free Ensemble API** (ECMWF/GFS, 51 members, percentiles, daily
aggregation) — the only easy hosted source of daily-Tmax percentiles. **Verdict:
no commercial spend justified for the probability problem.**

## 8. RECOMMENDED FEED STACK ($0, station-native + calibrated)
1. **NBM NBP text = primary pricing source** — `TXNP1/P2/P5/P7/P9` per station,
   calibrated daily-high distribution, 01/07/13/19 UTC.
2. **NBM GRIB qmd** (`s3://noaa-nbm-grib2-pds`) for fine strikes / hourly refresh.
3. **GFS MAV/MEX via IEM `mos.json`** — station-native deterministic cross-check +
   2003→ calibration anchor.
4. **IEM for calibration + truth** — CLI text = settlement-grade observed highs;
   MOS/NBM archive = historical forecasts. Fit per-station/season/lead **bias + σ**
   on (forecast − CLI high); validate calibration on held-out days.

**Pitfalls:** NBP TXN window is 12Z–06Z UTC not LST (verify per station, esp.
KLAX); backtest against CLI not METAR max (~1 °F gap flips near-the-money
binaries); re-verify NBM calibration against CLI per station.

**Paid tier:** ECMWF *data* is now free (€0); only enhanced/low-latency
dissemination and the MARS archive (~€6k/yr) cost money — pursue free 0.25° ENS as
an independent second probabilistic opinion. Commercial APIs not worth it. **No
paid tier required to ship the upgrade.**

**Why this beats `Normal(point, σ)`:** empirical/percentile distribution captures
skew + lead-dependent spread; station-specific by construction (NBP/MOS/CLI),
attacking the biggest edge risk; continuously re-calibrated against the exact
settlement series (CLI) via IEM.

## 9. Open items to verify before building (⚠️)
1. NBM GRIB **daily-MaxT** element token + forecast hours.
2. ECMWF free **full-res 9 km** S3 availability (policy €0; portal docs say 0.25°).
3. NBM v5.0 dissemination latency (measure off AWS object timestamps).
4. **CLI PIL per city** (KNYC → `CLINYC`, etc.).
5. NBP TXN window vs LST settlement-day alignment per station.
6. api.weather.gov exact rate limit (backoff on 429).
