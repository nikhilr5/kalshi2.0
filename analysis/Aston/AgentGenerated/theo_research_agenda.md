# Aston KXETH15M — PnL Research Agenda

Snapshot 2026-05-18. **Optimization target: realized $ PnL per market, net of fees.** Brier is a leading indicator, not the goal. A perfect theo with miscalibrated edges loses money; a mediocre theo with right-sized edges makes money. This agenda is re-ranked around that fact.

All measurement is offline against `/Users/nikhilr5/Desktop/Kalshi2.0/analysis/backtesting/data/KXETH15M-*.db`. Shared helpers live in `/Users/nikhilr5/Desktop/Kalshi2.0/analysis/utility.py`. **No strategy changes ship during the validation window** — everything here is measurement and prepared code, deployable only after 2026-06-05.

The PnL identity we are decomposing:

```
PnL_per_market = Σ_fills [ (theo_at_fill − fill_price) · signed_qty ]      # spread capture (gross edge)
              − Σ_fills [ (theo_t+δ − theo_at_fill) · signed_qty ]         # adverse selection cost
              + Σ_lots  [ (settle_outcome − theo_at_entry) · signed_qty ]  # settlement variance / drift
              − Σ_fills [ fee ]                                            # fees
```

Spread capture is what we *think* we earn. Adverse selection + settlement drift is what the market gives back to us. The gap between the two is the only number that matters.

---

## What to measure first — 2-3 lowest-effort, highest-information experiments this week

These run on data already on disk (`KXETH15M-26MAY15.db` … `KXETH15M-26MAY17.db`, plus today's WAL). All are measurement-only; nothing ships before 2026-06-05.

**M1. Fill-level adverse selection curve (1–2 days of work).** For every recorded fill, compute theo movement at +30s, +60s, +120s, +300s, +to_close. Bucket by side (bid vs ask), time-to-close (>10min, 5–10min, 1–5min, <1min), and distance-from-strike (|S−K|/(σ·√t)). Tells us directly: where do our 5¢/7¢ edges get eaten? This is the single most informative experiment we can run on existing data, and it costs ~150 lines of pandas. **Deliverable:** a `(side × ttc × moneyness) → mean adv_sel ¢` table with bootstrap CIs.

**M2. Per-market PnL attribution (1 day of work).** Decompose realized PnL per market into the four terms in the identity above. Aggregate by week. This tells us *which leg of the PnL identity is dominating* — if settlement variance swamps spread capture, the theo work matters; if adverse selection swamps spread capture, the quoting work matters; if neither dominates and fees + queue are the story, we're in a different game. **Deliverable:** weekly stacked-bar of the 4 components.

**M3. Edge-utilization plot (half a day of work).** For each market, plot the realized fill rate as a function of edge offered. Cross-reference with realized adverse-selection at that edge. The optimal edge minimizes (adv_sel − spread_capture) per market, *not* maximizes fill rate. We currently have one data point (5¢/7¢) but we can synthesize the response curve from quote depth and time-on-book in `order_events`. **Deliverable:** a (side × edge_offered) → (fill_prob, adv_sel_¢) chart.

These three give us, by end of week, a defensible answer to: "is our PnL bleeding from theo error, adverse selection, or wrong edges?" That answer determines which of the Tier-1 items below gets developed for the 2026-06-05 decision.

---

## Tier 1 — Direct PnL levers (ship-decision items for 2026-06-05)

### 1.1 Adverse-selection measurement and decomposition

**Why this is #1.** For a hold-to-close MM on 15-min binaries, adverse selection is the single largest PnL leak we don't currently measure. Every fill captures 5¢ or 7¢ of spread on paper. The question is how much of that the market takes back over the next 30s–15m through informed flow and settlement drift. We have no instrumentation for this today.

**Hypothesis.** Adverse-selection cost is asymmetric (bid fills are worse), time-dependent (worse in the last 5 min as theo converges to 0/1), and moneyness-dependent (worst near-ATM where settlement noise dominates). If true, the cost on the bid side late-window near-ATM eats well over 7¢ and we are paying to be filled.

**Measurement.** Built directly on the recorder schema — fills already record `spot_bid`, `spot_ask`, `kalshi_yes_bid`, `kalshi_yes_ask`, and `is_taker` at fill time. Theo at fill time and at fill+δ comes from `theo_state` via `merge_asof`.

```python
import pandas as pd, numpy as np, sqlite3
con = sqlite3.connect(db)
fills = pd.read_sql("select * from fills", con, parse_dates=["ts"])
theo  = pd.read_sql("select ts, ticker, spot, strike, theo, sigma, seconds_to_expiry from theo_state", con, parse_dates=["ts"])
con.close()

theo = theo.sort_values("ts"); fills = fills.sort_values("ts")
f = pd.merge_asof(fills, theo, on="ts", by="ticker", direction="backward", suffixes=("","_theo"))
# Theo at fill+δ for each δ
for delta_s in [30, 60, 120, 300]:
    f_shift = fills.assign(ts_fwd=fills.ts + pd.Timedelta(seconds=delta_s)).sort_values("ts_fwd")
    j = pd.merge_asof(f_shift, theo, left_on="ts_fwd", right_on="ts", by="ticker", direction="backward")
    f[f"theo_p{delta_s}"] = j["theo"].values

sgn = np.where(f.side == "yes", +1, -1)  # bid (buy yes) +1; ask (sell yes) -1
f["spread_capture_c"] = (f.theo - f.price) * sgn * 100  # in cents
for delta_s in [30, 60, 120, 300]:
    f[f"adv_sel_c_{delta_s}"] = (f[f"theo_p{delta_s}"] - f.theo) * sgn * 100  # positive = we got picked off
f["ttc_bucket"]  = pd.cut(f.seconds_to_expiry, [-1, 60, 300, 600, 99999], labels=["<1m","1-5m","5-10m",">10m"])
f["mny_bucket"]  = pd.cut(np.abs(f.spot - f.strike) / (f.sigma * np.sqrt(f.seconds_to_expiry/525960)/np.sqrt(365*24*60)),
                          [0, 0.25, 0.75, 1.5, 99], labels=["atm","near","far","tail"])
agg = f.groupby(["side","ttc_bucket","mny_bucket"])[["spread_capture_c","adv_sel_c_60","adv_sel_c_300"]].agg(["mean","sem","count"])
```

**Expected $ impact.** Sizing: ~1,500 fills/week at 1-lot ($1 contracts). Spread capture target is 5–7¢/fill = $0.05–0.07/fill = **~$75–105 gross/week**. If adverse selection averages 3¢/fill on the bid side and 1¢/fill on the ask side (rough prior), we give back ~$30/week. If late-window near-ATM bid fills lose 8¢/fill on adverse selection (plausible given TWAP drift), that subset alone is worth $5–15/week to identify and avoid. **Total PnL impact of doing this right: $20–60/week, or 20–60% of gross edge.**

**Risks.** Theo movement isn't only adverse selection — it includes normal σ-decay as t→0. Decompose into (a) deterministic σ-decay component (theta-equivalent) and (b) the surprise component. Use the σ-decay leg of the theo to subtract the expected decay; the residual is true adverse selection.

---

### 1.2 Per-market PnL attribution (the diagnostic we should have built first)

**Why.** We don't currently know whether our weekly PnL is dominated by spread capture, eaten by adverse selection, smeared by settlement variance, or simply fee-drag. Without this, every other item on this list is guessing. This is the master measurement.

**Hypothesis.** None — this is descriptive. Build the decomposition; the data will tell us where to invest.

**Measurement.** Build a per-market table with columns:

| ticker | n_fills | gross_spread_capture_$ | adv_sel_cost_$ (60s & to_close) | settlement_pnl_$ | fees_$ | net_pnl_$ |

Aggregate weekly. Plot the four components as a stacked bar. The four components must sum (within rounding) to realized net PnL — that's the integrity check.

```python
# Per-fill components (one row per fill)
f["spread_capture_$"] = (f.theo - f.price) * sgn * f.count
f["adv_sel_to_close_$"] = (f.theo_at_close - f.theo) * sgn * f.count   # need theo at close per ticker
f["settle_$"] = (f.y_settle - f.theo_at_close) * sgn * f.count          # residual variance term
f["fee_$"] = -f.fee
per_market = f.groupby("ticker")[["spread_capture_$","adv_sel_to_close_$","settle_$","fee_$"]].sum()
per_market["net_$"] = per_market.sum(axis=1)
```

**Expected $ impact.** This is a measurement, not a lever. Its impact is in *redirecting effort* — if it reveals adverse selection is the dominant leak, every Tier-1 hour should go into 1.1 and 1.3. If settlement variance dominates, 1.4 (TWAP) matters most. Expected ROI from running this experiment: it correctly prioritizes the *other* $30–80/week of decisions on this list.

**Risks.** `theo_at_close` requires extending `theo_state` to the boundary — the recorder may stop recording inside the auto-off window (90s before close). Verify the last `theo_state` row per ticker; use it (and label the residual to-close gap explicitly) rather than fabricating it.

---

### 1.3 Edge calibration to fill economics — is 5¢/7¢ leaving money or getting run over?

**Why.** Direct $ lever. The current asymmetric 5¢ ask / 7¢ bid was set by intuition (vol-seller lean). If actual adverse-selection costs disagree with this choice, every fill is mis-priced.

**Hypothesis.** The optimal edge per side is the one that maximizes `E[spread_capture − adv_sel | fill]` × `P(fill | edge)`. Both factors depend on edge. Wider edges → higher per-fill profit but lower fill rate. Narrower edges → more fills but each one is closer to fair value and more vulnerable to adverse selection. The optimum is usually *not* symmetric, and is usually *time-and-moneyness-conditional*.

**Measurement.** Three pieces.

(a) **Fill-conditioned net edge** — by side, ttc bucket, moneyness bucket: `mean(spread_capture − adv_sel_5min)`. If this is positive across the board, we have room to tighten edges to catch more flow. If it's negative in some buckets, we're paying to be filled there and should *widen* edges in those buckets.

(b) **Fill-probability curve** — for each market, compute the distribution of `(quoted_edge_offered)` from `order_events` (placed price vs theo at placement), and the corresponding fill rate. This gives `P(fill | edge_offered)` per side.

(c) **Expected profit per quote** — `EV(edge) = P(fill | edge) · E[spread_capture − adv_sel | fill, edge] − P(fill | edge) · fee`. Maximize over edge per `(side, ttc, mny)` cell.

**Expected $ impact.** A 1¢ improvement in *effective* per-fill economics on 1,500 fills/week = $15/week. A 2¢ rebalancing on the side that's currently mis-priced = $30/week. If we discover an entire bucket (e.g., last-1-min bid fills near ATM) is structurally negative-EV and we widen edges to avoid filling there, expect $10–25/week of avoided losses. **Total: $20–50/week.**

**Risks.**
- Survivorship: we only see fills that happened. Fill-probability has to be modeled from `order_events`, not just observed from `fills`. Specifically, build `P(fill | order resting at edge e, side s, ttc t)` from order-event traces, not from realized fills alone.
- Our own depth on the book biases competitors' behavior — this is mostly negligible at 1-lot size.
- This is the experiment most prone to look-ahead bias in backtest — the edge that "would have been optimal yesterday" is not the edge that will be optimal tomorrow. Use walk-forward weekly refits.

---

### 1.4 Inventory-aware quote skew

**Why.** Currently Aston quotes are symmetric around theo and just *clamp* at max position 8. This is the worst of both worlds — we keep being willing to add at our limit (until clamp), then suddenly stop, instead of becoming *progressively less willing* as inventory builds. A linear skew of theo by `γ · k · σ² · t_remaining` is the textbook AS-lite fix and is usually 5–15% PnL on a hold-to-close binary book.

**Hypothesis.** When inventory is at +k yes-contracts, our effective fair-value is *below* theo (we already want to sell), so ask should be at `theo − γk − ε_ask` (closer in) and bid should be at `theo − γk − ε_bid` (farther out). Symmetric edges around an inventory-shifted center.

**Measurement.** This is design-and-prep during the validation window, not deploy. Offline:
1. From the per-market PnL attribution (1.2), isolate fills that occurred while inventory was non-zero, broken by sign of inventory.
2. Compute realized PnL per fill conditional on inventory at fill time. If fills-while-long-and-buying-more are systematically worse than fills-while-flat-and-buying, that's the smoking gun for skew.
3. Backtest a `γ · k · σ² · t` skew on the recorded book to estimate which fills it would and wouldn't have generated, and what the PnL delta would have been.

**Expected $ impact.** Literature-prior on AS-lite for hold-to-close binary books: 5–15% of net PnL. On a baseline of $50–80/week net, that's **$3–12/week**. Lower than 1.1/1.3 but it stacks with them and is implementation-cheap once 1.3 is in place.

**Risks.**
- Skew interacts with the 5¢/7¢ asymmetry from 1.3 — implementing both naively double-counts the inventory lean. **Sequence: do 1.3 first to set base edges; layer 1.4 skew on top.**
- γ is a tuning parameter; using literature values blindly will under- or over-skew. Estimate γ from historical inventory-PnL pairs.

---

### 1.5 TWAP-Asian binary correction (reframed for $ PnL impact)

**Why this is here and not in Tier 2.** Even though this is a theo-correctness fix and Brier is a downstream symptom, the *PnL channel* for TWAP misspec is direct: in the last 60–180s of contract life, theo collapses too aggressively to 0/1 under European pricing when settlement is on a TWAP. Near-the-money fills in that window take **systematically negative adverse selection** because our theo says (e.g.) 0.05 while the true Asian-binary price is closer to 0.15. We pay to be filled, then watch settlement drift kill us. This is item 1.1's adverse-selection cost in a specific bucket — but the cost has a *correctible structural cause*.

**Hypothesis.** Kalshi KXETH15M settles on a TWAP of the final N minutes. Under TWAP, P(A_T > K) ≠ N(d2) on spot. The variance of the time-average is strictly less than the variance of S(T), with a t-and-τ dependent multiplier. The mispricing concentrates in the last τ minutes of the contract — exactly when adverse selection costs are highest in our 1.1 measurement.

**Measurement.**

Step A — identify N empirically from `spot_ticks` vs `.settlements_cache.json`. Don't assume; measure. (Code as in the previous version of this doc — preserved.)

Step B — implement `asian_binary_theo(S, K, sigma, t_min, tau_min)` in `analysis/utility.py` alongside the existing European theo.

Step C — **the PnL-relevant test**: paired backtest on recorded fills. For each fill, compute (i) realized PnL with current theo, (ii) realized PnL if theo had been Asian — *holding fills constant* (we wouldn't have placed quotes at the same prices, but this gives an upper bound on the PnL gain).

```python
f["theo_eu"] = european_theo(f.spot, f.strike, f.sigma, f.seconds_to_expiry)
f["theo_as"] = asian_binary_theo(f.spot, f.strike, f.sigma, f.seconds_to_expiry, tau=N_HAT*60)
# Quotes would have shifted by (theo_as - theo_eu). At ttc < 5min near-ATM this can be >5¢.
# Fills that happened at (theo_eu - edge) on the ask would not have happened if quote moved up.
# Conservative PnL gain estimate:
f["theo_shift_c"] = (f.theo_as - f.theo_eu) * 100
gain_per_market = f.groupby("ticker").apply(
    lambda d: ((d.theo_shift_c.abs() > 1) * d.adv_sel_c_to_close * np.sign(d.theo_shift_c)).sum()
)
```

**Expected $ impact.** Concentrated in the late-window near-ATM bucket. That bucket is probably 5–15% of fills. If those fills currently average 6¢ of adverse selection and the Asian correction recovers half of it, that's `0.10 × 1500 × 0.03 = $4.5/week` to `0.15 × 1500 × 0.05 = $11/week`. **Range: $5–15/week.** Smaller than 1.1's measurement scope but larger than 1.4. It also has the side benefit of cleaning up Brier — but that's secondary; the $ comes from the late-window adverse-selection reduction.

**Risks.**
- N_HAT misidentification — Kalshi's settlement rule could be different from what we measure. Build N as config, not constant.
- The PnL-gain estimate above is conservative (holds fills constant); the real gain comes from differently-priced quotes generating differently-distributed fills. Need a quote-level fill-probability model to estimate properly — but that's 1.3's deliverable, so do 1.3 first.

---

## Tier 2 — Diagnostic / supporting (Brier-side work, useful because it feeds Tier 1)

These are *measurement tools and calibration steps* for the theo that feeds Tier 1. They are not PnL levers themselves; they are correctness checks on the input. Useful, but not the goal.

### 2.1 Brier decomposition (REL − RES + UNC) and reliability curves

**Repurposed.** This was Tier-2 priority in the prior version. In the new framing it's a *diagnostic for theo correctness*. A theo with high REL (poor calibration) feeds wrong probabilities into the spread-capture term, which means our edge math in 1.3 is on shaky ground. Run it, but don't optimize for it.

**Expected $ impact.** Indirect. If it reveals systematic miscalibration in a specific moneyness bucket, that buckets gets re-prioritized in 1.3/1.5. Probable direct PnL value: **$0/week**. Probable redirect value: **decides whether 1.5 (TWAP) actually matters**.

### 2.2 Isotonic recalibration of theo → probability

**Repurposed.** A non-parametric monotone calibration map fit on training data, applied to the theo at quote time. If REL > 0, this strictly improves the *input* to 1.1/1.3 — but only if we have ≥500 markets to fit on and ≥500 to evaluate on. Pre-2026-06-05, we don't.

**Expected $ impact.** Indirect, via better-calibrated edge math. Possibly $5–10/week if it removes a systematic bias in spread-capture estimation. **Defer to post-validation.**

### 2.3 Vol-estimator choice (Parkinson vs GK vs YZ) and signature plot

**Repurposed.** Diagnostic on whether 1-min bars are the right sampling frequency. The signature plot tells us if microstructure noise is biasing σ. If σ is systematically biased high or low, every theo at every fill is biased — but the bias is roughly multiplicative, so it shows up as a small offset, not a structural error. **Direct PnL impact: <$3/week.** Diagnostic value: nonzero.

### 2.4 Drift term in d2

**Repurposed.** Prior is essentially zero PnL impact. Listed for completeness. **Direct PnL impact: <$1/week. Skip unless 1.2 attribution shows a residual we can't otherwise explain.**

### 2.5 HAR + market-implied σ blend, regime conditioning, microstructure features, path-dependence

**All deferred.** These are theo-improvement candidates with uncertain Brier benefit and uncertain *PnL* benefit. Reconsider only after 1.1–1.5 are deployed and we have a measurement of where residual PnL leaks remain. **Direct PnL impact: unknown, probably $0–5/week each.**

---

## Statistical guardrails (apply to every claim)

- **All PnL CIs via paired bootstrap, 10,000 resamples, 95%.** Required for every "X improves PnL by $Y/week" claim.
- **Walk-forward only.** Fit on weeks ≤ k, evaluate on week k+1. No pooled in-sample/out-of-sample contamination.
- **Account for autocorrelation in daily PnL.** Block bootstrap with block size matching the autocorrelation length (probably 1–3 markets). Naive iid bootstrap will under-state CI width.
- **Sample-size reminder for PnL.** With ~1,500 fills/week at typical edge size, weekly SE($) is ~$25–40 just from settlement variance. Don't read a $30/week PnL "improvement" as signal on <3 weeks of data. **Decision rule: require ΔPnL ≥ 2 · SE(ΔPnL) and consistent sign across paired-weeks.**
- **Cost-adjust everything.** Realized PnL = gross − fees. Quote-economics analysis must include the Kalshi fee schedule applied at the *fill* level. (Verify `fees` column in `fills` is populated.)
- **Selection bias in edge-calibration.** Item 1.3 tries many `(side, ttc, mny, edge)` cells. Apply a deflated-edge correction — the apparent best cell is biased upward; expected forward PnL is lower than backtest PnL.

---

## Execution order through 2026-06-05

| Date           | Action                                                                                                  |
|----------------|----------------------------------------------------------------------------------------------------------|
| 2026-05-18 – 21 | **M1 + M2 + M3** (the three "what to measure first" items above). Produces the adverse-selection table, PnL attribution stacked bar, and edge-utilization chart. |
| 2026-05-22 – 25 | Item **1.1** full build-out — adverse selection by `(side × ttc × mny)` with bootstrap CIs.            |
| 2026-05-26 – 29 | Item **1.3** edge calibration — fill-probability model from `order_events`, EV-per-quote optimization.  |
| 2026-05-28 – 06-02 | Item **1.5** TWAP empirical N + Asian-binary theo offline backtest. PnL-gain estimate.                |
| 2026-05-30 – 06-04 | Item **1.4** inventory-skew design. Item **2.1** Brier-decomposition diagnostic. Final assessment.   |
| 2026-06-05     | Decision point. Items passing the **ΔPnL ≥ 2·SE with consistent paired-week sign** test ship together.    |

Item 1.2 (per-market PnL attribution) is treated as continuous instrumentation — build once, run every day.

---

## The single number to optimize

**Net realized $ PnL per market, fee-adjusted, averaged over a 7-day rolling window.**

Every item above is justified by its contribution to that number. Every CI is on that number. Every decision on 2026-06-05 is whether the change moves that number by ≥ 2·SE with consistent sign across paired weeks. Brier, calibration, σ-correlation are inputs to that number, not substitutes for it.
