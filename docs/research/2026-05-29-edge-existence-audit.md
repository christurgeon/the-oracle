# Phase 0 — Edge-Existence Audit: Results & Routing

**Date:** 2026-05-29
**Produced by:** `scripts/edge_existence_audit.py` (hermetic core `python/analytics/edge_audit.py`)
**Inputs:** `data/shadow_settlements.full.json` (347 settled `weather_edge` trades, 38 trading days, after dropping 27 reconciler-synthetic / missing-signal rows), `state/eval_audit.jsonl`.
**Raw artifact:** `data/research/2026-05-29-edge-existence-audit.json`

---

## Verdict

**Formal decision branch:** `INCONCLUSIVE_DEFAULT_MICROSTRUCTURE` — but the label is a T3-coverage
technicality. The **substantive verdict is decisive: the model has no forecasting edge on the surface it
actually trades.** Three of the four tests point the same way; the fourth (T3) is merely under-powered by
a data-coverage gap and does not contradict.

**Honest reading:** On the markets the bot chose to trade, the **market price is meaningfully better
calibrated than the model** (T2, n=347, decisive). Claimed edge does **not** predict realized return (T1).
The only positive realized ROI lives entirely on the **behavioral greater/less surface**, not the
forecasting between surface (T4). This empirically and independently confirms the program reframe
(spec §1): the binding constraint is unproven forecasting edge, and the durable edge is
microstructure/behavioral.

---

## Test results (pre-registered thresholds; no post-hoc relaxation)

### T1 — Edge monotonicity: **FAIL**
Realized ROI by claimed-`effective_edge` quartile. Not monotone increasing; top bucket stationary-block CI
lower bound is −4.5% (below 0), so `passes = false`. Claimed effective_edge is uncorrelated with realized
return.

| bucket | edge range | n | ROI | day-clustered CI95 |
|---|---|---|---|---|
| 0 (lowest) | [0.070, 0.078] | 87 | **+1.2%** | [−7.6%, +8.8%] |
| 1 | [0.078, 0.098] | 87 | **−5.4%** | [−36.5%, +14.9%] |
| 2 | [0.098, 0.130] | 87 | **−18.1%** | [−57.6%, +16.3%] |
| 3 (highest) | [0.131, 0.269] | 86 | **+30.4%** | [+13.3%, +55.9%] |

Non-monotone sequence (+1.2%, −5.4%, −18.1%, +30.4%) — buckets 0–2 form an inverted pattern. The top
bucket day-clustered CI excludes 0 (lo=+13.3%), but the stationary-block CI lower bound is −4.5%, so the
joint PASS criterion is not met.

### T2 — Model vs market calibration head-to-head: **FAIL (decisive)**
On the same 347 settled outcomes, the **market price scores better than the model on both proper scoring
rules**, with both paired-difference CIs entirely on the market's side:

| metric | model (`model_probability`) | market (`mid_at_signal`) | paired diff CI95 (model−market) |
|---|---|---|---|
| Brier (lower better) | 0.264 | **0.217** | [+0.040, +0.082] → market better |
| LogScore (lower better) | 0.782 | **0.622** | [+0.141, +0.236] → market better |

`model_beats_market = false`. **Mechanism (hand-verified):** the model is overconfident. On traded between
markets mean model P(YES)=0.13 vs market 0.31; YES actually occurs 33.2% of the time. When the model is
confidently wrong (says 0.13, reality 1) its Brier penalty (0.75) dwarfs the market's (0.48). A 61.7%
win rate coexists with worse calibration because the losses are confident and large.

**Selection note (binding interpretation):** T2 is computed on *traded* markets — a sample selected for
maximal model-vs-market disagreement. That is the **decision-relevant** sample: it says *on the markets we
choose to bet, the market knows more than we do.* This is the cleanest possible falsification of a
forecasting edge.

### T3 — Calibrator counterfactual replay: **UNDER-POWERED (2% coverage)**
Replayed `identity` vs the live `deployed` calibrator at live kelly-0.075 sizing. Match coverage = **2.0%**
(7/347) because `state/eval_audit.jsonl` only begins 2026-05-21, so most traded tickers have no joinable
`raw_ens_prob`. On the tiny matched set both configs lose (identity −32.4% n=6; deployed −34.8% n=7). The
`<0.25` coverage triggers the `inconclusive` flag — this is the *sole* reason the formal branch isn't
`NO_FORECASTING_EDGE`. T3 does not contradict T1/T2; it simply can't speak yet.

### T4 — Edge attribution: forecasting surface is the loser
Scope: `weather_edge` only (matches T1/T2). The 7 `favorite_bias` greater/less trades are excluded here
and contributed negligibly to the portfolio separately.

| surface | n | ROI |
|---|---|---|
| between (forecasting) | 334 | −1.9% |
| weather_edge greater/less | 13 | +72.0% |

All positive realized ROI is on the weather_edge greater/less surface (tiny n=13, high variance). The
forecasting between surface — 96% of weather_edge trades — bleeds. This is exactly where the reframe said
the edge isn't, and where it (faintly) is.

---

## Routing (per spec §2 decision rule)

The substantive verdict (no forecasting edge; edge is behavioral) routes to the **microstructure/
behavioral branch**:

1. **Proceed with Phase 1 (NBM-Gap microstructure strategy)** as the primary new lever, alongside the
   already-live `favorite_bias` behavioral pilot. These do not depend on out-forecasting the market.
2. **Do NOT build Phase 2B (GEFS forecast backfill).** Better-calibrating a model that the market already
   out-calibrates (T2) cannot manufacture forecasting edge. Spending months extracting GRIB to certify
   calibrators is the wrong investment given this verdict.
3. **Phase 2A (pooling/accrual) is demoted to diagnostic, not strategic.** It is only worth pursuing if it
   can make the model *beat the market price* (close the T2 gap) — which the current evidence says it does
   not. Re-test T2 after any pooling experiment before treating it as a P&L lever.

## Cheap follow-up (optional, does not change the headline)
Re-run T3 once `eval_audit.jsonl` accrues more `raw_ens_prob` history (it grows +1 day/day from 2026-05-21),
or backfill raw probs from `predictions.jsonl` if that stream reaches further back, to raise T3 match
coverage above 0.25 and convert the formal branch from `INCONCLUSIVE_DEFAULT_MICROSTRUCTURE` to its
substantive value. Headline (T2 decisive) will not change.

## Limitations
- T2's strength is also its scope: it speaks to traded markets, not the full universe. For the *trading*
  question that is correct; for "could a recalibrated model trade a *different* surface profitably" it is
  silent (that is the Phase 1 NBM-gap question).
- The weather_edge greater/less +72% is n=13 — suggestive of the favorite-longshot bias, not a validated
  edge. It is the `favorite_bias` pilot's domain; re-validate there at n≥40 (~2026-06-23).
- The 27 rows dropped for missing model_probability (reconciler synthetics) are a net-winning slice (74.1%
  win rate); the realized-P&L 'bleed' framing on the kept 347 therefore understates the bot's
  actual book.
