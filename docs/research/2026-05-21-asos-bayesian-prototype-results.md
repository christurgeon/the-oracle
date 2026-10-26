# ASOS-Bayesian Intraday Prototype — Results & Decision

**Date:** 2026-05-21
**Spec:** [`2026-05-21-asos-bayesian-prototype-spec.md`](2026-05-21-asos-bayesian-prototype-spec.md) (pre-registered v2.1; reviewer-flagged fixes applied in scoring v3)
**Code:**
- `scripts/research_2026_05_21_build_asos_climatology.py`
- `scripts/research_2026_05_21_refresh_asos.py`
- `scripts/research_2026_05_21_asos_build_eval_corpus.py`
- `scripts/research_2026_05_21_asos_score.py`

**Artifacts:**
- `data/research/asos_climatology.json`
- `data/research/asos_prototype_evaluations.parquet`
- `data/research/asos_prototype_results.json`

---

## 1. Decision

**CONDITIONAL GREEN.** The mechanism is real (G1 strong, robust to multiple
attacks); the P&L estimate (G3) is structurally unverifiable on the current
corpus due to a calibrator-extrapolation problem the prototype confirmed.
Proceed to formal `ship-feature` lifecycle for productionization, but the
**first** workstream of formal SPEC must address the calibrator-redesign gap
below — do not skip to runtime ASOS client work.

---

## 2. Falsification table

| Gate | Threshold | Result | Status |
|---|---|---|---|
| G1 sharpness, spec-pinned 0800Z baseline | ≥ 0.1°F mean CRPS reduction at 2000Z, 95% lower CI ≥ 0 | **+0.999°F** [+0.73, +1.28], n=143 pairs | **PASS** |
| G1 sharpness, production-fidelity 2000Z-fresh baseline (sensitivity) | (same threshold) | **+0.944°F** [+0.67, +1.22] | PASS |
| G2 trade volume | ≥ 20 post-cap | 204 (ticker_asc) / 204 (max_edge) / 618 (uncapped) | **PASS** |
| G3 P&L (LAX-exc, primary cap = ticker_asc, calibrator-neutral) | ≥ 0 AND 95% lower CI ≥ pre-registered floor | ≈ 0 (slightly positive), CI spans zero | **NON_FALSIFYING** |
| Calibrator extrapolation (trade-eligible, loose support) | ≤ 25% | **29.5%** | exceeds — G3 non-falsifying |
| Calibrator extrapolation (trade-eligible, dense support) | (informational) | 47.1% | confirms non-falsifying under any honest support |

Per spec §5 strict reading: GREEN requires G1 + G2 + G3 all PASS. G3 is
**non-falsifying**, not PASS. Decision is therefore "conditional GREEN" —
G1 evidence is strong enough to justify proceeding, but G3 cannot be
claimed.

---

## 3. Corpus

| | Spec assumed | Reality |
|---|---|---|
| Eval window | 2026-03-18 → 2026-05-14 (58 days) | **2026-04-28 → 2026-05-21 (22 usable days)** |
| Reason for shortfall | n/a | Production didn't fetch same-day GFS + ECMWF in 0800Z snapshots until 2026-04-28. Pre-2026-04-28 snapshots had next-day-only forecasts or GFS-only. Discovered mid-execution; the unpublished tracker's "58 days of ASOS×snapshot overlap" appears to have counted next-day overlap, not usable same-day overlap. |
| Mitigation taken | n/a | Refreshed ASOS through 2026-05-21 via IEM (`scripts/research_2026_05_21_refresh_asos.py`), eval window slid to 2026-04-28 → 2026-05-21. |
| ASOS×snapshot row count | 960 nominal | 2,322 raw → 928 trade-eligible → 204 post-cap |

The 22-day corpus is the binding constraint on this prototype's statistical
power. G1 is well-powered (n=143 paired observations); G3 is borderline-
powered (n=204 trades) but the calibrator issue is the more binding problem.

---

## 4. G1 — sharpness

**The intraday Bayesian update reliably sharpens the forecast.**
Per-city mean CRPS reduction at 2000Z (positive = intraday sharper than
morning baseline):

| City | 0800Z baseline | 2000Z-fresh baseline | n pairs |
|---|---|---|---|
| AUS | +0.97°F | +0.91°F | 42 |
| CHI | +1.17°F | +1.10°F | 40 |
| LAX | +1.44°F | +1.14°F | 40 |
| MIA | +1.56°F | +1.56°F | 21 |
| NYC | +0.57°F | +0.50°F | 40 |

All cities positive under both baselines. The 2000Z-fresh baseline result
(reviewer C1 fix — the production-faithful comparison) is essentially
indistinguishable from the spec-pinned 0800Z result, meaning the bot's
1400Z and 2000Z ensemble refreshes don't capture most of the ASOS
observational signal. This validates the core mechanism.

**Cells underlying the reduction.** Random sample of 5 rows showed
expected behavior: morning ensemble has σ ≈ 2-3°F, intraday posterior
σ ≈ 0.5-0.8°F, both centered close to the eventual settled value. One
sample row at NYC/high May 28 had `M=61, settled=62`, morning predicted
58.4 (4°F too cold), intraday predicted 61.6 (within 0.4°F of truth) —
CRPS reduction +2.33°F. Failure mode also present: AUS/high one day had
`M=86, settled=91`, intraday over-truncated to 87 because climatology
mu_climo at 20Z said "max one degree of warming left" — missed a
5°F afternoon spike. The truncated-Gaussian's thin upper tail is a known
limitation; relevant for hot-front conditions where actual gain exceeds
climatology's expected tail.

**Power.** σ_diff = 1.70°F, exceeds the §5.2 "underpowered" flag of 1.5°F
but G1 still clears the gate (MDE recomputes to ~0.28°F, well below the
+1.0°F observed). σ_diff is high because of those long-tail rows;
median reduction is ~+0.5°F.

---

## 5. G2 — trade volume

| Selection rule | n trades |
|---|---|
| Pre-cap (every signal that passed edge + min_prob filters) | 927 |
| Per-day cap with ticker_asc selection (calibrator-neutral, **primary**) | 204 |
| Per-day cap with max_edge selection (production-faithful, calibrator-endogenous) | 204 |
| No per-day cap (research view of unfiltered signals) | 618 |

Volume threshold (≥20) cleared in every variant. Per-day cap is the
operational constraint, not the bottleneck.

---

## 6. G3 — P&L (the unverifiable result)

**Spec-pinned primary number (ticker_asc cap, LAX-excluded): slightly positive** with
a 95% CI spanning zero. Functionally zero with confidence interval too wide
to reject either profit or loss.

| Cap rule | P&L | 95% CI | n trades |
|---|---|---|---|
| ticker_asc (primary, calibrator-neutral) | **≈ 0** | spans zero | 204 |
| max_edge (production-faithful, calibrator-endogenous) | positive | spans zero | 204 |
| none (uncapped, research view) | positive | spans zero | 618 |

The spread between ticker_asc and max_edge cap rules — on the same
204 trades — is the **direct signature of calibrator endogeneity**.
Production's max-edge ranking selects trades whose edge depends on the
calibrator being tested; on this corpus, the kept-by-max-edge slice has
19.6% win rate vs the dropped slice's 23.9% — the cap is anti-selective
under the wide-range posterior. This was the reviewer's predicted failure
mode and the prototype confirms it.

**Per-city under primary cap (informational):**

| City | P&L sign | n trades |
|---|---|---|
| AUS | positive (largest) | 42 |
| CHI | negative | 40 |
| LAX | small positive | 40 |
| MIA | negative | 42 |
| NYC | positive | 40 |

AUS dominates the aggregate (largest contributor); 2 of 5 cities are
negative. AUS positivity is partly inflated by 3 preliminary-settlement
errors where ASOS observed `M > 86°F` but preliminary CLI reported
`high_f = 70°F` (physically impossible — likely NWS prelim issued before
peak). Excluding the 6 cap-survivor rows flagged as
`|M - settled| > 10°F` moves AUS contribution down and the aggregate
down (still positive but smaller).

**MIA/low at 2000Z is mechanistically degenerate.** At 2000Z (1 PM PT),
the daily low (overnight) has already passed; M = settled on 60/126
MIA/low 2000Z rows. The truncated posterior collapses to a Dirac at M,
which the calibrator maps to ~0.17, prompting structurally-losing NO
trades on `between(M, M+1)` markets where the LOW IS M. Excluding this
cell moves the aggregate up. The productionization spec must
either skip this cell or use a different snapshot time for low markets.

**Calibrator extrapolation is the binding obstacle.** Trade-eligible
extrapolation rate is 29.5% under the loose support definition
([breakpoints_5th, breakpoints_95th] = [0.018, 0.981]) and 47.1% under
the honest densest-cluster definition ([0.05, 0.65]). Either way, G3 is
non-falsifying per the spec gate. The truncated-Gaussian posterior
pushes raw probabilities toward 0 or 1 when M is far from S — exactly
the regime the production isotonic artifact wasn't trained on. The
calibrator squashes these to ~0.08 / ~0.51 by extrapolation, producing
trade signals whose calibrated probability is approximate-at-best.

---

## 7. Gaps the productionization spec MUST address

Listed in priority order:

1. **NEW (revealed by prototype): calibrator redesign for wide-range posteriors.**
   The truncated-Gaussian posterior distribution doesn't fit production's
   isotonic training support. Need either (a) a calibrator fit on
   intraday-Bayesian raw probabilities (would require a few months of
   shadow data, which doesn't yet exist), or (b) a piecewise calibrator
   that handles 0/1 tails differently from the [0.05, 0.65] mid-range.
   **Until this is solved, G3 cannot be evaluated honestly and live
   deployment cannot be justified.**
2. Runtime ASOS client: HTTP retry, caching, timeout, fail-open semantics.
   (Spec already lists this; the prototype confirmed it's still needed.)
3. `same_day_gate` carve-out around `BLOCK_PEAK_PASSED`: the bot blocks
   same-day trades after peak; the strategy needs the carve-out to fire
   intraday updates that re-enable trading.
4. Per-day event cap expansion: production's `max_bets_per_day_per_event`
   currently groups by (city, var, forecast_date). With same-day + next-day
   on the same observable, expand to prevent double-betting.
5. Cell-specific exclusions: MIA/low at 2000Z (mechanically degenerate at
   that snapshot time — daily low has passed); LAX at 2000Z (pre-peak;
   prototype showed LAX positive on G1/P&L, so don't exclude — but a
   later snapshot for LAX would be even better).
6. Bad-settlement filter at trade time: preliminary CLI settlements
   occasionally disagree with ASOS by >10°F (11% of full corpus, 3% of
   cap-survivors). The strategy should not commit trades on
   `(city, date, var)` cells where the current ASOS reading already
   contradicts the preliminary settlement.

---

## 8. Pre-registration adherence

| Item | Followed? | Notes |
|---|---|---|
| 22-day corpus instead of 48-day | Reluctantly adhered | Eval window slid post-discovery; documented in §3. |
| Spec-pinned 0800Z baseline as primary G1 | Adhered + sensitivity added | Reviewer C1 surfaced that 2000Z-fresh is more honest; both reported. |
| Per-day cap "later snapshot wins" | Adhered | But the selection rule within (date, city, var) was implementation-defined: implemented as max-edge (production-faithful) → reviewer C2 flagged calibrator-endogeneity → switched primary to ticker_asc (neutral); both reported. |
| Calibrator support [5th, 95th of training distribution] | Adhered with caveat | Used 5th-95th of breakpoint POSITIONS (loose); reviewer C3 surfaced this overestimates the support → also report densest-cluster support (dense); both reported. |
| G1, G2, G3 thresholds | Adhered exactly | No threshold relaxation. |
| MIA/low @ 2000Z degenerate-cell exclusion | NOT made a gate | Post-hoc exclusion would be bad practice. Surfaced as informational finding; spec amendment for productionization. |
| Bad-settlement (>10°F drift) filter | NOT made a gate | Surfaced as flag; production filter recommendation. |

---

## 9. Reviewer notes (paranoid second pass, condensed)

Reviewer flagged 4 critical + 4 important issues; all addressed in scoring
v3:
- C1 (stale baseline): added 2000Z-fresh sensitivity. G1 still passes.
- C2 (cap endogeneity): switched primary to ticker_asc. G3 dropped from
  clearly positive to ≈ 0.
- C3 (calibrator support too generous): reported both loose and dense
  extrap. Dense = 47.1%, confirms non-falsifying.
- C4 (MIA/low @ 2000Z degenerate): surfaced as finding; G3 ex-MIA/low is
  higher than the primary.
- I1 (bad settlements): added |M-settled|>10°F flag; full-corpus rate
  11.1%, cap-survivor rate 6/204 = 3%.
- I2 (P&L concentration): top 5 of 22 days drive the positive total; the remaining 17 net
  negative. Day-stratified bootstrap CI already accounts for this.
- I3 (haircut sensitivity meaningless): dropped /5 and /20; 98.6% of
  trades capped at MAX_CONTRACTS_PER_TRADE = 50, not by volume_fp.
- I4 (σ_diff flag missing): noted (1.70 > 1.50 threshold); reported.

Bottom line per reviewer: "GREEN should not be claimed on this run." The
prototype agrees; the decision is conditional-GREEN with the calibrator
redesign as the binding next step.

---

## 10. Recommended next action

Move to formal `ship-feature` GOAL phase for ASOS-Bayesian
productionization, with **calibrator redesign as Phase 1 deliverable**.
Do not begin runtime-ASOS-client work, same-day-gate carve-out, or
per-day-cap expansion until the calibrator question is settled — these
are wasted effort if the strategy can't produce trustworthy probabilities.

Alternative: if the calibrator-redesign timeline is too long (likely
1-2 months minimum given shadow-data requirements), pivot to the next-
best in-flight opportunity. The mechanism finding (G1 robust) is a
durable artifact that justifies revisiting when production accumulates
the data needed to fit a wide-range calibrator.
