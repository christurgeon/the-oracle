# LAX/low live EMOS audit — regime-shift exposure check

**Status:** Audit complete, read-only.
**Date:** 2026-05-23
**Author:** christurgeon (with Claude Code)
**Scope:** Apply the CHI/low audit methodology (`docs/research/2026-05-23-chi-low-mu-calibration-audit.md`) to LAX/low — the next-highest-shrinkage cell currently LIVE on EMOS — to test whether the same `b<1` regime-shift over-correction is emerging in live data before the 2026-06-11 override re-evaluation.
**Inputs:** `data/emos/coefficients.json` (live LAX/low: a=37.179, b=0.330, c=3.399, d=0.774), `/tmp/emos_scratch/corpus.parquet` (n=175 LAX/low settled rows), `data/vps_pull_20260523/trades.jsonl` (n=11 live LAX/low fills since 2026-05-12), per-day snapshots for ensemble inputs + obs settlements.

---

## TL;DR

**The CHI/low pathology is NOT manifest in LAX/low.** The b<1 mechanism that hurt CHI/low (training-corpus shrinkage extrapolating into thinner-data warm-regime tails, producing catastrophic μ-drops) is structurally absent in LAX/low because the underlying GFS+ECMWF ensemble warm bias for LAX/low **scales sharply with ens_mean** (regression slope +0.71, vs +0.137 for CHI/low). The shrinkage + bias growth approximately cancel. EMOS residual vs obs stays in [−1.0, +0.4]°F across the full training ens_mean range — well-calibrated.

**Headline numbers:**
- Corpus EMOS residual vs obs (mean): **Q1 +0.04, Q2 +0.08, Q3 +0.08, Q4 −0.83** — much milder than CHI/low's Q4 of **−2.60**.
- EMOS residual regression slope vs ens_mean: **+0.039** (essentially flat). CHI/low's slope was **−0.072** (over-correction at warm regimes).
- Live fills (n=11 since 2026-05-12): mean EMOS error vs known obs (n=7) = **−1.46°F** — within typical calibration noise.
- σ-inflation factor: **~2.0×** (raw σ ≈ 0.4–1.3°F → EMOS σ ≈ 1.87–2.20°F). CHI/low's incident day saw **2.9×** σ-inflation; the lower factor here means the μ-drop and σ effects do NOT compound into incident-magnitude probability inflations.

**Mild concern surfaced (orthogonal to the CHI/low question):** Live P&L on the 11 LAX/low fills is **negative at −16.3% ROI** (n=7 with known outcomes). This is the result of normal calibration misses (e.g., the 2026-05-20 trade where EMOS underweighted the actual-winning bracket by 9pp), not regime-shift failures. **This is a separate signal worth raising at the 2026-06-11 override re-evaluation** but is not what this audit was scoped to find.

**Verdict:** No precautionary measure required for LAX/low before 2026-06-11 on the regime-shift dimension. The override stays in effect under the existing Brier-improvement gate. The live ROI signal is its own follow-up.

---

## 1. Corpus analysis (parallel to CHI/low §1)

Computed `emos_mu_raw = a + b * ens_mean` on n=175 LAX/low rows in the rebuilt corpus (2026-03-19 → 2026-05-21).

### 1.1 μ-drop distribution

| Stat | LAX/low | CHI/low (for comparison) |
|---|---:|---:|
| mean | **−4.20** | −4.54 |
| std | 2.43 | 1.98 |
| min | −11.86 | −9.26 |
| max | +0.00 | −0.41 |

Magnitude is similar — but the comparison that matters is residual vs *obs*, not vs *ensemble mean*, since EMOS is supposed to correct ensemble bias.

### 1.2 Quartile breakdown (the key test)

| Quartile | ens_mean | obs | emos_mu | Raw bias (ens−obs) | **EMOS residual (emos_mu − obs)** |
|---|---:|---:|---:|---:|---:|
| Q1 | 65.33 | 58.67 | 58.72 | +6.66 | **+0.04** ✓ |
| Q2 | 61.68 | 57.43 | 57.51 | +4.25 | **+0.08** ✓ |
| Q3 | 58.64 | 56.43 | 56.51 | +2.21 | **+0.08** ✓ |
| Q4 | 61.34 | 58.23 | 57.40 | +3.11 | **−0.83** ⚠ |

CHI/low's Q4 EMOS residual was **−2.60**, indicating warm-regime over-correction. LAX/low's Q4 is **−0.83** — mild and within typical calibration variance.

### 1.3 Regression: does over-correction grow with ens_mean?

`EMOS_residual = +0.0388 * ens_mean − 2.55`

| ens_mean (°F) | Predicted EMOS residual (°F) |
|---:|---:|
| 55 | −0.42 |
| 60 | −0.23 |
| 65 | −0.03 |
| 70 | +0.16 |

Slope is **positive small**, meaning if anything LAX/low's EMOS gets *less* cool at higher ens_means (the bias-growth effect dominates the shrinkage effect). This is the opposite direction from CHI/low's regression slope of −0.072. **LAX/low does not exhibit warm-regime over-correction.**

### 1.4 Why the difference

CHI/low's raw ensemble bias has a weak slope vs ens_mean (+0.137). At the upper-decile ens_mean of ~62°F the predicted bias is ~+5.8°F, but EMOS shrinks toward training centroid by ~5.3°F — over-shooting by ~0.5°F per decile.

LAX/low's raw ensemble bias has a **steep slope of +0.71**. The same upper-decile ens_mean of ~63°F shows actual bias of ~+6.2°F (per-decile data, D5 mean ens_mean=68.18, bias=+8.78). The EMOS shrinkage (~67% strength because b=0.33) matches this bias growth — they cancel out, leaving residual near zero across the range.

The b<1 mechanism is doing the *correct calibration work* for LAX/low — the cell needed strong shrinkage because the ensemble has strong bias-growth. CHI/low needed milder shrinkage but EMOS gave it the same shape, producing over-correction in the warm tail.

---

## 2. Live trade audit (n=11 fills since 2026-05-12)

Computed `emos_mu` and `emos_sigma` for each live LAX/low fill using snapshot-day forecast inputs + live EMOS coefficients, cross-referenced with settled obs where available.

| Fill date | fc_date | bracket | side | price | blend_μ | emos_μ | μ_drop | emos_σ | obs | emos_err | won? |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-05-12 | 2026-05-12 | 57.5 | NO | 0.48 | 60.76 | 57.21 | −3.55 | 1.96 | ? | ? | ? |
| 2026-05-12 | 2026-05-13 | 58.5 | NO | 0.57 | 58.41 | 56.43 | −1.98 | 1.93 | ? | ? | ? |
| 2026-05-14 | 2026-05-15 | 57.5 | NO | 0.67 | 60.26 | 57.04 | −3.21 | 1.88 | 60.0 | **−2.96** | ✓ |
| 2026-05-15 | 2026-05-16 | 55.5 | YES | 0.04 | 59.69 | 56.86 | −2.84 | 1.95 | 61.0 | **−4.14** | ✗ |
| 2026-05-16 | 2026-05-17 | 56.5 | YES | 0.09 | 60.08 | 56.99 | −3.10 | 1.96 | 58.0 | −1.01 | ✗ |
| 2026-05-17 | 2026-05-18 | 58.5 | NO | 0.71 | 60.46 | 57.11 | −3.35 | 1.99 | ? | ? | ? |
| 2026-05-17 | 2026-05-17 | 56.5 | NO | 0.86 | 60.34 | 57.07 | −3.27 | 1.88 | 58.0 | −0.93 | ✓ |
| 2026-05-18 | 2026-05-19 | 57.5 | NO | 0.56 | 59.86 | 56.91 | −2.95 | 1.92 | 57.0 | −0.09 | ✗ |
| 2026-05-18 | 2026-05-19 | 59.5 | NO | 0.73 | 59.86 | 56.91 | −2.95 | 1.92 | 57.0 | −0.09 | ✓ |
| 2026-05-19 | 2026-05-20 | 60.5 | NO | 0.56 | **66.71** | 59.17 | **−7.54** | 2.20 | 60.0 | −0.83 | ✗ |
| 2026-05-21 | 2026-05-22 | 59.5 | NO | 0.55 | 61.04 | 57.30 | −3.74 | 1.92 | ? | ? | ? |

(4 fills missing obs because snapshot-tree settlement files don't cover those dates in the VPS pull's window — confirmed not findable, not a code bug.)

**Mean EMOS error vs obs (n=7):** **−1.46°F** (EMOS systematically a touch cool). Range: −4.14 to −0.09. Std: 1.49.

**Win rate (n=7):** 3/7 = 43%. **P&L:** −36.6% ROI on the known-outcome subset, or **−16.3% ROI on total wagered including the 4 unknown-outcome fills assumed to net to zero**).

### 2.1 The 2026-05-19 trade (warm-regime test case)

The single most-OOD-warm trade was 2026-05-19's forecast for 2026-05-20: blended ensemble mean **66.71°F** with σ=1.27°F. This is at the upper end of training distribution (Q4 in the corpus max-quartile). EMOS shifted μ to 59.17°F (drop −7.54°F) and inflated σ to 2.20.

Bot bet NO on bracket [59.5, 61.5]. Actual obs: 60.0°F (inside bracket). NO lost.

**The EMOS prediction was actually closer to right than the raw ensemble** for this trade:
- Raw P([59.5, 61.5]) at N(66.71, 1.27) = ~0.000003 (essentially zero)
- EMOS P([59.5, 61.5]) at N(59.17, 2.20) = 0.355
- Actual obs landed inside the bracket → EMOS's 35.5% probability was much closer to "true" than raw's near-zero

So EMOS did its job calibrationally on this warm-regime day. The trade still lost because the bot took NO at $0.56 (market consensus ~44% in-bracket), believing the model probability was below 44%. The bot was wrong; the market was right.

**This is normal calibration noise, NOT the CHI/low pathology.** CHI/low's incident inflated a sub-0.01% raw probability to 24% EMOS probability and bet YES on a near-impossible event. LAX/low's analogous trade inflated 0.0003% to 35.5% — but bet NO, and lost in the normal way (model under-pred a bracket that actually realized). No 4800× catastrophic mis-calibration.

---

## 3. Mechanism summary — why LAX/low is fine and CHI/low isn't

| Property | LAX/low | CHI/low |
|---|---|---|
| EMOS b coefficient | 0.33 (strong shrinkage) | 0.79 (mild shrinkage) |
| Raw ensemble bias slope vs ens_mean | **+0.71** (steep) | +0.137 (shallow) |
| Bias growth matches shrinkage? | **Yes — they cancel** | **No — shrinkage over-shoots** |
| Q4 EMOS residual vs obs | −0.83 (mild) | −2.60 (severe) |
| EMOS residual vs ens_mean slope | +0.039 (essentially flat) | −0.072 (over-correction grows) |
| σ-inflation typical | ~2.0× | ~2.9× (incident-day) |
| Live incident manifest? | **No** | Yes (2026-05-21) |

The shrinkage parameter b=0.33 is doing the right thing for LAX/low because the underlying ensemble has the bias-growth that justifies that level of shrinkage. CHI/low's b=0.79 was fit on a corpus where the ensemble bias was much flatter — the shrinkage over-corrects in regimes the fit didn't capture well.

**Generalization implication:** The right diagnostic is not "is b<1?" but "does the cell's ensemble-bias slope match the b<1 shrinkage strength?". A cell with b=0.91 and bias slope = +0.15 would over-correct (CHI/low pattern). A cell with b=0.33 and bias slope = +0.71 self-cancels (LAX/low pattern). Worth folding into the EMOS promotion gate as a post-fit diagnostic: compute the (a, b) implied bias-correction slope at p99 ens_mean, compare to corpus-observed raw-bias slope, flag if mismatch > ~50%.

---

## 4. Live ROI signal (orthogonal to the audit question)

Surfacing as a secondary finding for the 2026-06-11 LAX/low override re-evaluation:

**Cumulative live LAX/low P&L:** −16.3% ROI across n=11 fills since 2026-05-12.

For comparison: LAX/high was rolled back at **−39.6% ROI** over 11 trades / 7 days (per `data/emos/overrides.json:LAX/high.rollback_reason`). LAX/low at −16.3% over 11 trades / 11 days is concerning by the same yardstick — although smaller in magnitude and concentrated in 4 losing trades (2026-05-15, 2026-05-16, 2026-05-18, 2026-05-19).

The current override rollback trigger (`data/emos/overrides.json:LAX/low.rollback_trigger`) is: *"If at 2026-06-11 refit `unstable_d` persists AND CV-OOS Brier improvement falls below 10pp"*. The Brier improvement on the 2026-05-11 fit was 67.5% — well above the 10pp gate. So the override stays in force regardless of live ROI on the current trigger language.

**Recommendation for the 2026-06-11 re-evaluation:** consider adding a live-ROI guardrail to the rollback trigger, e.g., "OR live P&L since promotion is more negative than 1× initial position" (2% of pilot bankroll). The CHI/low audit established that calibration metrics (Brier, μ-residual) and trade economics (live ROI) can diverge — Brier-only gates are insufficient for promotion durability. This is a follow-up scope, not a gate change to push through pre-2026-06-11.

---

## 5. Verdict

- **The CHI/low regime-shift pathology is NOT present in LAX/low.** Different cell-level structural property (steep raw-bias slope) makes the b=0.33 shrinkage well-calibrated, not over-shooting.
- **No precautionary measure required pre-2026-06-11** on the regime-shift dimension. The override gate's Brier-improvement criterion remains the appropriate primary test.
- **Live ROI of −16.3% is a secondary signal** that should be folded into the 2026-06-11 re-evaluation under a separate gate, not absorbed silently.
- **General-purpose recommendation:** the EMOS promotion gates should add a "bias-slope-vs-shrinkage-strength" diagnostic at fit time. This would have flagged CHI/low's shrinkage mismatch on the 2026-05-11 fit. **OBVIATED 2026-05-23 by per-cell calibrator harness.** The harness (the unpublished tracker) evaluates each cell's actual performance under each candidate calibrator on a pre-registered hold-out. A bias-slope-vs-shrinkage mismatch will surface directly as "EMOS does not win this cell on the hold-out" — same diagnostic outcome via cell-level evidence rather than a fit-time structural check. The standalone diagnostic is no longer separately required.

## 6. MIA/low (deferred)

MIA/low has b=0.91 (mild shrinkage) and is the third currently-LIVE EMOS cell. Lower-priority follow-up per audit scope. Quick read on the corpus: MIA/low's raw bias slope appears flatter than LAX/low's (Miami has stable sub-tropical climate, less bias growth with ens_mean). The CHI/low pattern is *more* likely to be latent in MIA/low than in LAX/low. **Deferred but worth doing before the 2026-06-11 override re-evaluation** — same scope as this audit, ~1 hour additional work. Tracked as a follow-up in the unpublished tracker.
