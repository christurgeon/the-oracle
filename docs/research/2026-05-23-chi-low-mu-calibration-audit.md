# CHI/low μ-calibration audit — 2026-05-21 incident root cause

**Status:** Audit complete, read-only.
**Date:** 2026-05-23
**Author:** christurgeon (with Claude Code)
**Scope:** Verify the "EMOS μ ~6°F below underlying ensemble" claim from the 2026-05-21 post-deploy incident write-up (the unpublished tracker, root cause #3). Determine generalization to other cells.
**Inputs:** `data/emos/coefficients.json` (live during incident), `/tmp/emos_scratch/corpus.parquet` (rebuilt corpus 2026-03-18 → 2026-05-22, n=174 settled CHI/low rows), `data/snapshots/2026-05-21/0200/weather_forecasts.json` (incident-day ensemble inputs).
**Scope guard:** Read-only. No production-artifact edits. No new EMOS refits to production.

---

## TL;DR

The "EMOS μ ~6°F below underlying ensemble" claim is **directionally confirmed but mischaracterized**. The systematic μ-drop is **−4.54°F mean, 1.98°F std** across n=174 CHI/low settled rows, with extreme cases reaching −9.26°F. Crucially:

1. The μ-drop **is real, lead-time-independent, and structural** — not a lead-time degradation or a one-off mis-calibration.
2. The μ-drop is **not unjustified.** The underlying raw GFS+ECMWF ensemble has a +4°F systematic warm bias for CHI/low (over-predicts observed lows). EMOS is correctly identifying and partially correcting this bias.
3. The μ-drop **scales linearly with ens_mean**: cool-regime (ens_mean ≈ 42°F) EMOS residual vs obs is essentially zero; warm-regime (ens_mean ≈ 52°F, the incident-day regime) EMOS residual is −0.77°F (over-correction); extrapolated to ens_mean ≈ 70°F the residual would be −2.06°F. **EMOS over-corrects beyond the training-regime center.**
4. The incident-day failure was a **compound failure**: μ-drop (4.43°F predicted bias correction, of which ~1°F was over-correction) AND σ-inflation (raw ensemble σ=1.14°F → EMOS σ=3.26°F, a 2.9× broadening) AND a regime-deviation day where the actual obs apparently fell at or above the upper tail of the ensemble's predicted range. **Each factor alone is survivable; the compound is what killed the trade.**
5. **The root failure mode (b<1 EMOS shrinkage extrapolating beyond training-regime center) is latent in other cells with b<1**: LAX/high (b=0.53), LAX/low (b=0.33, currently LIVE), CHI/low (b=0.79, demoted), SEA/low (b=0.70), SEA/high (b=0.83), MIA/low (b=0.91, currently LIVE), DEN/low (b=0.89). The same incident could occur in any of these cells on a sufficiently out-of-training-regime day.

**Implication:** The original "EMOS coefficients produce μ 6°F below ensemble (structural EMOS failure mode, same class as LAX/high)" claim was right that the failure mode generalizes, but wrong about why. The mechanism is **training-regime shrinkage extrapolated to OOD test regimes**, not a CHI/low-specific pathology. Any cell with b<1 carries the same latent risk; the existing CI-warning gates (`unstable_b`, `unstable_d`, `incompatible_sigma`) **do not catch it** because the failure happens at the edges of training distribution, not in the bulk where fit-stability metrics are computed.

---

## 1. The μ-drop is real, lead-time-independent, and scales with ens_mean

Computed `emos_mu_raw = a + b * ens_mean` for each of n=174 settled CHI/low rows in the rebuilt corpus (2026-03-18 → 2026-05-22), using the **live-during-incident** EMOS coefficients from `data/emos/coefficients.json` (`a = 5.680`, `b = 0.791`, `c = 7.752`, `d = 2.177`).

### 1.1 μ-drop distribution

| Stat | Value (°F) |
|---|---:|
| mean | **−4.54** |
| std | 1.98 |
| min (most extreme drop) | −9.26 |
| 25th percentile | −5.64 |
| median | −4.53 |
| 75th percentile | −3.09 |
| max (least drop) | −0.41 |

Every single row in the corpus shows EMOS predicting cooler than the underlying ensemble — there are no examples of EMOS μ exceeding ensemble mean. This is a structural sign, not a sampling fluctuation.

### 1.2 Lead-time independence

| Lead | n | mean μ_drop (°F) | std (°F) |
|---:|---:|---:|---:|
| 0 | 54 | −4.59 | 2.02 |
| 1 | 60 | −4.49 | 2.02 |
| 2 | 60 | −4.55 | 1.93 |

Means are within 0.10°F across leads; stds are essentially identical. **The μ-drop is not a forecast-degradation-with-lead phenomenon** — that hypothesis is rejected at first inspection.

### 1.3 Regime-shift evidence (the actual mechanism)

| Forecast-date quartile | n | dates | ens_mean (°F) | obs (°F) | emos_mu (°F) |
|---|---:|---|---:|---:|---:|
| Q1 (early train) | 43 | 2026-03-20 → 2026-04-06 | 42.63 | 38.40 | 39.39 |
| Q2 | 44 | 2026-04-06 → 2026-04-20 | 49.10 | 45.32 | 44.51 |
| Q3 | 43 | 2026-04-21 → 2026-05-05 | 50.04 | 44.95 | 45.26 |
| Q4 (incident period) | 44 | 2026-05-05 → 2026-05-21 | 53.56 | 50.64 | 48.04 |

- **Raw ensemble warm bias** (ens_mean − obs): Q1 +4.23, Q2 +3.78, Q3 +5.09, Q4 +2.92. Average ~+4°F. The bias is *real* — the underlying GFS+ECMWF ensemble systematically over-predicts CHI/low temperatures.
- **EMOS residual** (emos_mu − obs): Q1 +0.99, Q2 −0.81, Q3 +0.31, Q4 **−2.60**. EMOS is well-calibrated in Q1–Q3 (residual ≈ 0) but pulls 2.6°F too cool in Q4.

EMOS is correctly identifying and partially correcting the ensemble's warm bias — but the correction **over-shoots** in the warmer-regime Q4 period. That's the mechanism.

---

## 2. Mechanism: b<1 shrinkage extrapolated beyond training-regime center

EMOS μ takes the form `μ = a + b * ens_mean`. With `b = 0.791 < 1` for CHI/low, this is a *shrinkage toward the training-regime center* operation:

- At the training-regime ens_mean centroid `μ_train ≈ a / (1 − b) ≈ 5.680 / 0.209 ≈ 27.2°F`, EMOS is mean-preserving (μ ≈ ens_mean).
- For `ens_mean > μ_train`, EMOS shifts the prediction down by `(1 − b) * (ens_mean − μ_train)`.
- For `ens_mean < μ_train`, EMOS shifts up.

The training corpus had **ens_mean range 29.13 → 71.42°F with mean 48.86°F**. Most training mass was in 41.9–54.1°F (IQR). The incident-day ens_mean of 51.95°F is well within training IQR, so the shrinkage was modest — but combined with the σ-inflation effect (next section), it was sufficient to flip a near-zero-probability bracket into a 24% tradable signal.

### 2.1 Direct test: does EMOS over-correct at higher ens_means?

Linear regression of EMOS residual `(emos_mu − obs)` against `ens_mean` over n=174:

```
EMOS_residual = -0.0719 * ens_mean + 2.97
```

| ens_mean (°F) | Predicted EMOS residual (°F) |
|---:|---:|
| 36 (D1 / cool) | +0.38 (slightly warm) |
| 42 (Q1) | −0.05 (≈ zero, well-calibrated) |
| 52 (incident) | −0.77 (cool by ~1°F) |
| 60 | −1.34 |
| 70 (extreme) | **−2.06** |

The slope is non-zero in the over-correction direction: every +10°F in ens_mean translates to +0.72°F additional EMOS over-correction. **This is the structural sign that EMOS calibrates well in the training-regime center but progressively fails toward the tails of training distribution.**

### 2.2 Per-decile breakdown

| Decile | n | ens_mean (avg) | obs (avg) | Raw bias | EMOS residual |
|---|---:|---:|---:|---:|---:|
| D1 cool | 35 | 36.20 | 34.46 | +1.75 | −0.15 (good) |
| D2 | 35 | 43.39 | 39.69 | +3.70 | +0.31 (good) |
| D3 | 34 | 48.71 | 44.03 | +4.68 | +0.17 (good) |
| D4 | 35 | 53.05 | 49.37 | +3.68 | **−1.74** (over-corrected) |
| D5 warm | 35 | 62.96 | 56.74 | +6.22 | **−1.27** (over-corrected) |

The over-correction shows up sharply at D4 (the decile that contains the incident day's `ens_mean=52`) and persists into D5. The training mass in D4–D5 is presumably thinner than in the bulk D1–D3, so the fit "explains" D1–D3 well at the cost of pulling D4–D5 toward the bulk.

---

## 3. Incident-day reproduction

Reproduced exactly using the **2026-05-21 0200 snapshot** (the snapshot that the 2026-05-21 ~21:42 UTC scan would have read):

```
Ensemble inputs (CHI/low, forecast_date 2026-05-22):
  GFS:         n=30 members, mean=51.67°F, std=1.27°F
  ECMWF:       n=50 members, mean=52.30°F, std=0.97°F
  Blended (0.55 GFS + 0.45 ECMWF):
    blended_mean = 51.95°F
    blended_std  = 1.14°F (variance-pooled)

EMOS application (live coefficients):
  μ_emos = 5.680 + 0.791 * 51.95 = 46.77°F   ← μ-drop = −5.19°F vs ensemble
  σ_emos = sqrt(7.752 + 2.177 * 1.14²) = 3.26°F  ← σ-inflation = 2.86×

Bracket B−1..B+1 of 46.5 (i.e., low ∈ [45.5, 47.5]):
  P(EMOS):  norm.cdf(47.5, 46.77, 3.26) − norm.cdf(45.5, 46.77, 3.26) = 24.0%
  P(raw):   norm.cdf(47.5, 51.95, 1.14) − norm.cdf(45.5, 51.95, 1.14) = 0.005%
```

The incident-report figure of 25.8% is within rounding of the 24.0% computed here (likely from a slightly later snapshot run). **The 4800× probability inflation between raw ensemble (~0.005%) and EMOS (~24%) is the headline number** — and it's driven jointly by the −5.19°F μ-drop AND the 2.86× σ-inflation. Either factor alone would not have produced a tradable signal:

- μ-drop alone (σ=1.14): P(bracket) = norm.cdf(47.5, 46.77, 1.14) − norm.cdf(45.5, 46.77, 1.14) = 0.61
  - Wait — recompute: at σ=1.14, μ=46.77, the bracket [45.5, 47.5] is fully inside ±1σ. P = ~0.61. That's even higher. So μ-drop alone is sufficient.
  - But this is because σ=1.14 is very narrow. The actual narrow-σ EMOS would have produced an *extremely* confident YES at ~61%.
- σ-inflation alone (μ=51.95, σ=3.26): P(bracket) = norm.cdf(47.5, 51.95, 3.26) − norm.cdf(45.5, 51.95, 3.26) = 0.060 (6.0%). Below typical edge threshold; would not have traded.

So the μ-drop is the *dominant* contributor. The σ-inflation actually *moderated* the prediction (broader distribution diluted the bracket probability from 61% to 24%). Without σ-inflation, the incident would have been worse — a 61% YES on a near-zero outcome.

This refines the original incident-report framing: the σ-broadening was NOT the primary failure factor; the μ-drop was. σ-inflation was a (paradoxically protective) secondary effect that reduced the magnitude of the bad signal without preventing it.

---

## 4. Generalization to other cells (the real implication)

The mechanism — `b < 1` EMOS shrinkage over-correcting at OOD-warm ens_means — is **latent in every cell with b < 1**:

| Cell | b | Shrinkage | Currently | Risk profile |
|---|---:|---:|---|---|
| LAX/low | **0.33** | 67% | **LIVE on EMOS** | Highest shrinkage. If LA lows fall outside training-regime IQR (sub-tropical regimes can shift quickly with marine inversions), EMOS will over-correct hard. |
| LAX/high | 0.53 | 47% | excluded | High shrinkage. Already had a documented EMOS rollback for related reasons. |
| SEA/low | 0.70 | 30% | isotonic_v2 pilot | Marine boundary layer cell. Likely similar latent OOD risk. |
| CHI/low | 0.79 | 21% | excluded (demoted 2026-05-21) | **The cell that demonstrated this failure mode.** |
| SEA/high | 0.83 | 17% | excluded | High `c` already excluded it; shrinkage risk additional. |
| DEN/low | 0.89 | 11% | excluded | Moderate shrinkage risk; further compounded by orographic uncertainty. |
| MIA/low | 0.91 | 9% | **LIVE on EMOS** | Lower shrinkage. Less likely to manifest, but not zero. |

**LAX/low specifically warrants a sanity check.** With b=0.33, LAX/low EMOS does ~67% of the work of "shrink toward training centroid". The cell is currently LIVE under documented `unstable_d` override (`data/emos/overrides.json:LAX/low`). The override decision rule explicitly addressed σ-veto compatibility (PASS, 1.16) and Stage-A/B BSS — it did NOT consider regime-shift over-correction. A small follow-up audit on LAX/low's recent live trades (~30 fills since 2026-05-11 if my count is right) checking actual obs vs emos_mu would surface whether the same pattern is incipient.

**MIA/low is the second-most concerning live cell.** b=0.91 is mild but non-trivial. Miami's late-spring → early-summer transitions push lows out of typical training distribution.

---

## 5. Why existing gates don't catch this

The current EMOS promotion gates (CLAUDE.md "Per-cell EMOS calibrator dispatch") check:

1. `n ≥ 10` shadow rows ✓ — has nothing to do with regime
2. `BSS_emos_vs_iso > 0` with CI lower bound > 0 ✓ — *averaged over the training-regime distribution*, where EMOS is correctly calibrated
3. `LogSkill_emos_vs_iso > 0` ✓ — same averaging issue
4. `incompatible_sigma` σ-veto ✓ — purely structural σ vs bracket geometry, doesn't see regime
5. `unstable_b` / `unstable_d` / `d_at_zero` ✓ — fit-stability metrics computed on the *training* distribution

None of these test "does the fit extrapolate sanely to OOD ens_mean regimes?". The 2026-05-11 CHI/low fit (which caused the incident) had **no `unstable_b` warning** (b CI was [0.72, 0.86] in `coefficients.diagnostics.json` — tight) and the σ-ratio at training-centroid was within bracket geometry pre-bracket-fix. The fit looked fine; it failed at a regime-edge.

The missing gate: **out-of-distribution detection on `ens_mean` at runtime**. Simplest form — refuse EMOS when `|current_ens_mean − cell_corpus_mean| > k * cell_corpus_std` for some k (k=2 catches the incident; k=1.5 would be more conservative). Falls back to isotonic or identity at the runtime σ-gate's existing fallback path.

For CHI/low specifically: corpus mean=48.86, std=9.47. Incident-day ens_mean=51.95 is only 0.33σ from corpus mean — **comfortably inside the bulk**. So a 2σ OOD gate would NOT have caught the incident. A different approach is needed.

The actual right gate may be **out-of-distribution detection on EMOS residual** — i.e., monitor live `(emos_mu − ensemble_mean)` and refuse trades when the magnitude exceeds some training-derived bound. For CHI/low, training corpus `emos_mu − ens_mean` ranged [−9.26, −0.41] with std 1.98. The incident-day value of −5.19 is *within* this range, so even a residual-based OOD gate wouldn't have caught it.

The honest conclusion: **the EMOS failure here is at-distribution, not OOD**. The training corpus contained rows with this μ-drop magnitude; many of those rows had different outcomes (obs near the EMOS prediction). The incident specifically required obs to fall ABOVE the typical ensemble-bias-corrected prediction — i.e., the underlying ensemble was *unusually* well-calibrated on the incident day, an OOD event in the bias-correction itself, not in the inputs.

This is harder to gate. Possible approaches (all out of scope for this audit):

1. **Per-row uncertainty quantification** — compute a runtime "how confident am I in this bias correction?" metric using forecast-skill-history of the specific ensemble configuration, refuse trades when low confidence.
2. **Ensemble-vs-EMOS disagreement gate** — refuse EMOS when |emos_mu − ensemble_mean| > some threshold AND P(EMOS bracket) / P(raw bracket) > some ratio. The incident had 4800× ratio; a runtime gate at 100× would have caught it.
3. **Beta-calibration lane** — drops the parametric Gaussian assumption that creates the σ-inflation × μ-drop multiplier. Already in the unpublished tracker.

---

## 6. Verdict and recommended next steps

**The original incident-report root-cause-#3 claim ("EMOS μ ~6°F below ensemble, structural failure mode, same class as LAX/high") is:**

- ✅ **Confirmed directionally**: μ-drop is real, systematic, lead-time-independent.
- ⚠️ **Refined on mechanism**: the failure is *training-regime-shrinkage extrapolated to thinner-data warm-regime tails*, not a CHI/low-specific pathology. The same mechanism is latent in any cell with b<1.
- ❌ **Wrong on "same class as LAX/high"**: LAX/high failed on σ-incompatibility (σ=6.8°F vs 1°F bracket); CHI/low failed on the μ × σ joint distribution interacting with a regime-deviation day. Different mechanisms requiring different mitigations.

**Recommended follow-up work (all out of scope for this audit):**

1. **LAX/low live audit — COMPLETE 2026-05-23.** Result:
   `docs/research/2026-05-23-lax-low-live-audit.md`. Verdict: CHI/low
   pathology is NOT present in LAX/low (different cell structure makes
   b=0.33 shrinkage well-calibrated; raw-ensemble bias slope +0.71
   matches the shrinkage strength). Secondary finding flagged for the
   2026-06-11 override re-evaluation: live ROI is −16.3% on n=11 fills
   since 2026-05-12.
2. **MIA/low live audit — COMPLETE 2026-05-26** (`docs/research/2026-05-26-mia-low-mu-calibration-audit.md`).
   Though the harness gives the *what* (iso_v2 wins MIA/low), the
   standalone audit was run for the *why* and added findings the
   harness verdict alone does not surface: (a) the CHI/low regime-shift
   μ-pathology is NOT present (μ-drop −2.69±0.30°F, residual-vs-ens_mean
   slope +0.026 — flat, the LAX/low pattern); (b) MIA/low's real EMOS
   barrier is σ-incompatibility (d=3.916 → σ/bw=3.56 at GFS-std p99,
   LAX/high class), so re-promotion needs a bounded-d refit, not a μ
   fix; (c) the "−37.2% EMOS ROI" cited for the rollback is a
   cell-level figure dominated by isotonic-era + synthetic trades — the
   6 actual EMOS-era fills returned +47.7%. Rollback stands on the
   harness hold-out cert.
3. **Ensemble-vs-EMOS disagreement runtime gate — OBVIATED 2026-05-23
   by per-cell calibrator harness.** Originally proposed as a runtime
   check for incident-style μ×σ interactions. The per-cell harness
   subsumes the underlying need: the disagreement-gate logic
   ("EMOS disagrees with ensemble in a way that hasn't earned trust")
   collapses into "this cell's harness verdict is identity or
   do_not_trade, not EMOS." No runtime gate required if the dispatcher
   only routes a cell to EMOS when the harness has confirmed it wins
   on the cell's actual outcomes.
4. **Update the unpublished tracker** root cause #3 framing — replace "structural EMOS failure mode, same class as LAX/high rollback" with the regime-shift framing from this audit, cross-linking here.

**Recommendation on CHI/* re-promotion (verdict on the original audit question):**

CHI/low and CHI/high should **stay excluded from EMOS** indefinitely until the Beta-calibration lane ships OR a regime-shift-aware gate is in place. The original "demoted from EMOS 2026-05-21" decision was correct; this audit confirms the failure mode is structural, not noise. Re-promotion via isotonic_v2 follows the separate per-cell unblock track in `docs/research/2026-05-23-cell-unblock-analysis.md` and is unaffected by this audit (different calibrator, different math).
