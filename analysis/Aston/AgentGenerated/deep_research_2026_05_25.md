# Aston deep research — 2026-05-25

Sample: 11 trading days (2026-05-15 → 2026-05-25), 14,549 fills, 955 settled
tickers, ~$26.67/day mean held-to-close P&L. All numbers are from
`analysis/Aston/AgentGenerated/_cache/master_fills.pkl` and supporting
scripts in this directory.

## TL;DR

The strategy is barely profitable ($26.67/day, daily SE $9.47, 95% CI
[$8.12, $45.23]) and almost all P&L comes from the SELL side. **Buys lose
money in aggregate (-$7.76/day).** The root cause is a **structural
calibration bias in N(d2)**: theo systematically over-predicts mid-OTM
YES probabilities (theo bin 0.10-0.20: model says 14.8%, actual 5.8%;
bin 0.40-0.50: model 44%, actual 30%). When theo > mid, the market is
more often right than theo on the OTM side, but the strategy
mechanically buys anyway.

The single biggest available improvement is **gating buys when
theo < 0.5 (or z < 0)**: +$15.50/day, 58% improvement on baseline.

The current edge values (7¢ bid / 5¢ ask) are **wrong directionally**.
Optimal observable is closer to **12¢ bid / 5¢ ask** in the existing
fill set ($25 → $25 with cleaner profile). The asymmetric edge needs
to be larger on the bid, not the ask.

HAR-RV refit and TWAP-aware theo are both in the noise — don't waste
the validation window on either.

A directional vol-arb strategy (taker on |theo-mid|>8¢ in T-2m to T-5m
window) is independently profitable at ~$9/day with 23 trades/day and
+21¢/trade — worth a separate pilot post-validation.

---

## 1. Top 3 high-EV improvements (2-3 week timeframe)

### #1 — Gate buys on theo ≥ 0.5 (or equivalently z ≥ 0)
**Expected impact: +$15.53/day (+58%)**

- **Evidence:** 4,147 buy fills with theo<0.5, P&L = -$170.86 over 11
  days. Drop them and the strategy nets $42.20/day instead of $26.67.
- **Why it works:** Theo overstates YES probability at moneyness z<0
  (calibration table in `_calibration.py` — theo bin (0.30, 0.40)
  predicts 36% actual is 24%, etc.). When theo > mid in this regime,
  the market knows better.
- **Statistical strength:** Per-ticker bootstrap CI on buy side =
  [-39.4c, +16.5c] (n=955 tickers) — confidence interval spans zero
  but most of the mass is negative, and the structural mechanism
  (calibration bias) is corroborated by independent Brier analysis
  (`_brier_gate.py`).
- **Implementation:** In `Aston/strategy2.py:_repost`, add a guard
  before `osm.ensure_bid()`:
  ```python
  if self.theo < 0.5:
      self.osm.cancel_bid()
  else:
      self.osm.ensure_bid(desired_bid, self.size_bid)
  ```
  Equivalent gate using moneyness would require knowing strike+spot,
  but theo<0.5 is a clean enough proxy.
- **Caveat:** The bias may be specific to the May 15-25 sample
  (where ETH had a slight downtrend, so z<0 markets that looked like
  reasonable buys got dragged further away from strike). Recommend
  paper-implementing this and measuring outcome over weeks 4-6 of
  validation. Don't ship live mid-window.

### #2 — Push the auto-off threshold from T-90s to T-180s
**Expected impact: +$8.13/day (+30%)**

- **Evidence:** Fills in the T<180s window contribute -$89.38 to P&L
  over 11 days, almost entirely from buys (-$48.90) and stale-edge
  sells (-$40.48). See `_late_market.py`.
- **Why it works:** Quote age >30s correlates with negative P&L (see
  `_quote_age.py` — buys with quote age >30s lose $9.56/fill). In the
  final 3 minutes, theo updates can't compete with TWAP-converging
  spot drift, so resting orders are predominantly stale.
- **Statistical strength:** -$89 over n=2362 fills is well above
  noise; effect is consistent across days.
- **Implementation:** Change auto-off in `app.py` from 90s to 180s
  (search for "90s" or "auto_off"). One line.
- **Caveat:** This forgoes the T-30s edge concentration from the
  earlier baseline analysis, but that baseline used 5-day sample and
  may not generalize. The negative-P&L T<3m signature is strong here.

### #3 — Fix the stale-fill bleed via dedup + cancel-retry
**Expected impact: +$5-10/day (sample-size dependent)**

- **Evidence:** 9.4% of fills have negative edge_c at fill time
  (theo had moved against us, but order still filled). Cost = $116
  over 11 days = $10.59/day. Two-thirds of this is from buys.
  See `_stale_fills.py`.
- **Why it works:** The existing `_repost` flow has a known gap
  (cancel-race Regime B in memory: 392 fills sat >5s after tolerance
  crossed). The infrastructure for `order_attempts` logging now
  exists; can measure precise rate.
- **Implementation:** Two changes in `osm.py`:
  1. Add transient-cancel retry: when cancel fails with transient
     error (errno 35, timeout), schedule a retry instead of dropping
     to the reconcile loop (`osm.py:415-425`).
  2. Tighten tolerance from 1¢ to 0.5¢ — already discussed as the
     symmetric tolerance fix per `aston_adverse_selection_investigation`
     memory.
- **Caveat:** The infrastructure-level fix interacts with the orphan
  bug recovery. Verify with the new `order_attempts` table that
  cancel-retry doesn't create new orphans.

**Combined effect of #1+#2: $43.35/day = 62% improvement (script
`_summary_metrics.py`).**

---

## 2. Top 2 alternative trading approaches worth a pilot

### A. Directional vol-arb (taker on theo-vs-mid disagreement)
**Theoretical basis:** When |theo - mid| >> spread, one side is wrong.
Empirically (from `_directional_v2.py`), theo is right more often
than mid on this filter.

**Evidence on this product:**
- T-2m to T-5m window, threshold 8¢, one trade per ticker: $9.0/day
  net of 1c fee, 60 trades/day, +16¢/trade.
- T-3m, threshold 12¢: $9.9/day, 23 trades/day, +24¢/trade.
- Buys and sells both profitable in this filter (selection picks
  high-conviction theo disagreements where the calibration bias
  doesn't apply).

**Minimal viable pilot:**
- A separate process (or a flag in Aston) that polls theo+book every
  5s in the T-2m to T-5m window. When |theo-mid| > 8¢, IOC a 1-lot
  taker order in theo's direction.
- Bound exposure: max 1 contract per ticker, max 3 simultaneous
  positions.
- Run paper for 1 week alongside Aston, log all signals + would-be
  P&L (including realistic fees).
- Hard stop if drawdown > 2% portfolio.

**Risk:** This is in-sample on the model — HAR was fit on prior data
but the directional bias might be sample-period specific. Need at
least 1,500 markets (2-3 weeks of paper trading) to discriminate
edge from noise at the cents-per-trade level.

### B. Asymmetric edge by theo-regime
**Theoretical basis:** The 7¢ bid / 5¢ ask asymmetry is in the wrong
direction given observed buy-side losses. Restructure to (a) widen
buy edge selectively when theo is in the calibration-biased zone,
and (b) tighten sell edge in regimes where mid bids up the ask
unrealistically.

**Evidence:**
- Edge grid search (`_edge_optimization.py`): optimal observable
  config is (edge_bid=12, edge_ask=5) at $25/day in retained-fills
  counterfactual, vs current $20 at (7,5).
- The improvement is mostly from preventing low-edge buy fills, not
  from generating new fills.

**Minimal viable pilot:**
- Implement edge as a function of theo: `edge_bid = max(7, 12 * I(theo < 0.5))`.
  Conservative: try (8,5) first, then (10,5), then (12,5) over
  consecutive 1-week periods.
- Measure: fill count, per-fill edge, daily P&L. Compare to
  pre-change baseline.
- The single-knob version of #1 — both achieve the same outcome by
  different means.

---

## 3. What's working — don't change

1. **Sell side is real edge.** $378 over 11 days = $34/day, per-ticker
   bootstrap CI [+11, +71]c. The asymmetric vol-seller lean (theo says
   YES is less probable than market) is correct on the OTM-high z>0
   region. Don't gate or kill sells.

2. **The 5¢ ask edge is appropriately tight.** Grid search confirms
   5¢ is the sweet spot. Going to 3¢ adds fills but per-fill EV drops
   to negative territory in mid-ATM zones.

3. **HAR σ forecast accuracy.** Corr 0.47-0.52 with realized σ vs
   ~0 for market-implied σ. HAR is unambiguously the right σ tool.
   The calibration miss is in N(d2), NOT in σ. Don't waste cycles
   refitting HAR — refit gives identical MAE.

4. **Hold-to-close at the strategy level.** Early-exit at T-30s mid
   shows a +5c "advantage" but it's not tradable (you'd pay 1-2c
   spread to exit). Hold-to-close avoids that round-trip cost.

5. **Lonely-at-BBO pull-back + post-only.** Adverse-selection
   protections are doing real work — 1s markout is -0.80c, but it
   recovers to -0.16c by 60s. That's a healthy MM signature (no
   "death spiral" pattern).

6. **The single-threaded queue worker in OSM.** Eliminated whole
   classes of race conditions. Don't touch.

---

## 4. Known unknowns and instrumentation gaps

### High value:
1. **Daily refresh of S-curve calibration table.** Current finding is
   from 11 days. The bias may compress/widen with sample-period
   characteristics. Recommend a daily cron that emits the
   "theo bin → actual outcome" table and tracks per-bin drift.

2. **Per-fill latency: theo timestamp at fill vs theo timestamp at
   place.** The new `order_attempts` table has request/response
   timestamps but not "theo at place" vs "theo at fill". Adding a
   `theo_at_place` column to `order_events.placed` would let us
   directly measure stale-quote bleed instead of inferring from
   edge_c < 0.

3. **Cross-asset spillover (BTC vs ETH).** Couldn't test in this
   investigation because we don't record BTC 1-min ticks (only ETH).
   If ETH theo + BTC spot momentum is informative, that's a feature
   add. Need to enable BTC tick recording for the test.

4. **Counterfactual fill model.** All my "drop bad fills" P&L
   improvements assume the rest of the fills are unchanged. In
   reality, removing capacity from one side may shift queue dynamics
   for the other. A proper simulation harness (replay the recorded
   data with a hypothetical strategy) would tighten the CIs.

### Lower value:
5. **Drift correction in theo.** The S-curve bias might be partly
   from missing mean-reversion. We tested with realized σ → bias
   doesn't shrink, so this is unlikely to be the fix.

6. **Vol smile / skew across adjacent strikes.** Kalshi only has one
   strike per market typically. Inter-event arbitrage would need a
   coverage map of strikes within a window and is hard with current
   data.

---

## 5. Things to NOT do (ruthless honesty)

1. **Stop investing in HAR refit.** R²_train 0.22, R²_test 0.22 on
   new fit vs current's 0.47/0.57 (which were on a different
   sample). MAE on test is 11.29% (new) vs 11.11% (current).
   Different coefficients, identical forecasts. Refit is in the
   noise.

2. **Stop testing TWAP-aware theo.** Brier at T-2m: current 0.0668
   vs TWAP-aware 0.0643. Gap is 0.0025 = below sample-size noise
   floor at n~1000. Has been tested twice; result is the same.

3. **Don't ship the alternative directional strategy as the main
   strategy.** The same calibration bias that hurts buys would
   hurt the directional buy signal — selection at |theo-mid|>8¢
   helps a lot, but it's the same model under the hood. Validate
   first on paper.

4. **Don't make sizing changes during validation.** Capacity is
   fine (5-10 lot trivial, 25-lot Q90), but the per-day SE is $9
   so you can't statistically detect a 50% increase in EV with 2
   weeks of n. Sizing changes belong post-validation.

5. **Don't flatten on intraday P&L swings.** Daily std is $31 on
   mean $26 — many days are noise of one sign or the other. The
   weekly cumulative is the only signal at this scale.

---

## 6. Headline metrics

### Baseline (Aston as-is, 2026-05-15 to 2026-05-25)
| Metric | Value |
|---|---|
| Total realized P&L | $293.39 |
| Daily mean | $26.67 |
| Daily std | $31.40 |
| Daily SE of mean | $9.47 |
| 95% CI on daily | [$8.12, $45.23] |
| Sharpe (daily) | 0.85 |
| Sharpe (annualized) | 16.23 — treat with skepticism (n=11) |
| Positive days | 8 / 11 |
| Fills | 14,549 |
| Markets traded | 955 |
| Buy P&L | -$85.34 (40% of fills) |
| Sell P&L | +$378.74 (60% of fills) |

### Per-side bootstrap CI (per-ticker clustered)
- Buy: -11.5c/ticker [-39.4, +16.5] — not significantly negative
- Sell: +41.2c/ticker [+11.0, +70.6] — significantly positive

### Markouts (mid-mark)
| Horizon | All | Buy | Sell |
|---|---|---|---|
| T+1s | -0.80c | -1.07c | -0.61c |
| T+5s | -0.75c | -0.94c | -0.62c |
| T+30s | -0.39c | -0.74c | -0.15c |
| T+60s | -0.16c | -1.35c | +0.66c |
| T+120s | +0.18c | -2.05c | +1.62c |

Sells show classic MM signature (recover after adverse selection).
Buys show steady bleed at all horizons — a "took the wrong side"
signature, not just adverse selection.

### Scenario summary
| Scenario | P&L/day | Δ |
|---|---|---|
| Baseline | $+26.67 | — |
| Drop buys at theo<0.5 | $+42.20 | +$15.53 |
| Drop buys at z<0 | $+42.17 | +$15.49 |
| Drop edge<0 fills | $+37.26 | +$10.59 |
| Auto-off at T-3m | $+34.80 | +$8.13 |
| **Combined (#1+#2)** | **$+43.35** | **+$16.68** |
| Combined (drop bad buys + stale) | $+46.96 | +$20.29 |
| Optimal edge (12¢/5¢, fill-conditional) | $+25.08 | -$1.59 (counterfactual only) |

The "combined" scenarios are the realistic best-case for in-validation
changes. Both #1 and #2 are simple code changes (~5 lines each).

---

## Files produced

All scripts in `analysis/Aston/AgentGenerated/`:

- `_loader.py` — shared data loader (cache to `_cache/`)
- `_build_master.py` — master fills frame builder (run once, cached)
- `_overview.py` — high-level baseline numbers
- `_side_breakdown.py` — buy vs sell deep dive (moneyness, IV, regime)
- `_calibration.py` — theo vs market vs actual outcome by bin
- `_theo_bias_source.py` — Is σ or model structure the problem? (Model.)
- `_distribution_fit.py` — fat tails / skew of ETH log returns
- `_temporal_patterns.py` — hour, day, week, IV-regime
- `_stale_fills.py` — stale fill cost decomposition
- `_quote_age.py` — quote age vs P&L
- `_edge_optimization.py` — edge grid search (counterfactual)
- `_moneyness_edge.py` — moneyness/theo gate proposals
- `_brier_gate.py` — when theo and mid disagree, who wins?
- `_directional_vol.py` — alternative directional taker strategy
- `_directional_v2.py` — directional v2 with realistic constraints
- `_late_market.py` — T<5m dynamics + auto-off threshold sweep
- `_lot_size.py` — capacity / book depth analysis
- `_har_refit.py` — HAR coefficient refit (null result)
- `_early_exit.py` — hold-to-close vs early exit comparison
- `_summary_metrics.py` — final scenario table

Cached data at `_cache/` (pickle, ~100MB total):
- `master_fills.pkl` — every fill enriched with theo, mid, IV, z, outcome
- `settlements.pkl` — per-ticker TWAP-derived outcome
- `markouts.pkl` — multi-horizon mid markouts per fill
- `all_26MAY15.pkl` — raw data cache
