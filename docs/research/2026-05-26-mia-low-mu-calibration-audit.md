# MIA/low live μ-calibration audit — regime-shift exposure check

**Status:** Audit complete, read-only.
**Date:** 2026-05-26
**Author:** christurgeon (with Claude Code)
**Scope:** The deferred step-2 follow-up from the CHI/low audit (`docs/research/2026-05-23-chi-low-mu-calibration-audit.md`) and the LAX/low audit (`docs/research/2026-05-23-lax-low-live-audit.md` §6). Apply the CHI/low μ-calibration methodology to MIA/low — the third cell that was LIVE on Gaussian-EMOS (b=0.91) — to test whether the `b<1` regime-shift over-correction pathology is present, ahead of the 2026-06-11 EMOS override re-evaluation. MIA/low was **rolled back from EMOS to iso_v2 on 2026-05-24** on the per-cell harness winner certification, so this audit is now retrospective: it explains *why* MIA/low's live result looked the way it did and informs whether MIA/low is a re-promotion candidate at 2026-06-11.
**Inputs:** `data/emos/coefficients.json` (MIA/low: a=4.071, b=0.909, c=1.581, d=3.916, `unstable_d`, ens_std_p99=1.683 — confirmed identical on VPS), `/tmp/emos_scratch/corpus.parquet` (n=175 MIA/low settled GFS rows, 2026-03-19 → 2026-05-22), `data/vps_pull_20260527/trades.jsonl` (fresh pull, all 6 EMOS-era MIA/low live fills + audit-stream metadata), `data/shadow_settlements.full.json` (authoritative Kalshi settlement P&L), `data/vps_pull_20260523/eval_audit.jsonl` (EMOS-era evaluation reasons).
**Scope guard:** Read-only. No production-artifact edits. No new EMOS refits to production.

---

## TL;DR

**The CHI/low regime-shift μ-pathology is NOT present in MIA/low.** MIA/low's EMOS μ is well-calibrated against observed lows across the full training range — the same conclusion as LAX/low, by the same mechanism (the mild `b<1` shrinkage approximately matches the cell's modest ensemble bias). MIA/low's μ never over-corrects in the warm tail the way CHI/low's does.

**Headline μ numbers (n=175 corpus, live coefficients):**
- μ-drop (`emos_mu − ens_mean`): **−2.69 ± 0.30°F** — much milder and far tighter than CHI/low's **−4.54 ± 1.98** (because b=0.909 vs 0.791, and Miami's GFS spread is small).
- EMOS residual vs obs by ens_mean quartile: **Q1 −0.37, Q2 +0.12, Q3 +0.44, Q4 +0.66** — all within ±0.7°F, and *positive* (slightly warm) in the warm tail. CHI/low's Q4 residual was **−2.60** (severe cool over-correction).
- EMOS-residual-vs-ens_mean regression slope: **+0.026** (essentially flat, slightly positive — like LAX/low's +0.039, the opposite sign of CHI/low's −0.072). No warm-regime over-correction.
- Lead-time independent: μ-drop −2.75 / −2.67 / −2.66 across leads 0/1/2.

**The "−37.2% live MIA/low EMOS ROI" cited in `config.yaml` and the harness investigation is a cell-level figure that does NOT reflect EMOS-era trading.** MIA/low traded on EMOS for exactly **6 fills (fc 2026-05-17 → 05-21)** — the complete and only EMOS-era trade set, confirmed against the fresh 2026-05-27 pull. Those 6 fills returned **+47.7% authoritative ROI (4 wins / 2 losses)**. The cell's lifetime loss is dominated by **isotonic-era** trades: a large reconciler-synthetic position loss (`KXLOWTMIA-26MAY11-B79.5`, n=360, strategy=`None`, fc 2026-05-11) and a smaller isotonic loss on fc 2026-05-15 (the GFS+ECMWF blend-disagreement loss already documented in `data/emos/overrides.json:MIA/low.justification`). **Attributing the cell's loss to EMOS is unsupported** — though the rollback itself stands on its own (separate) evidence (see §2.2).

**The real latent EMOS risk for MIA/low is σ-incompatibility, not μ.** d=3.916 (`unstable_d`) gives σ_emos = 3.56°F at the training GFS-std p99 on 1°F brackets (σ/bw = 3.56, far above the 2.0 σ-veto) — the same *class* as the LAX/high rollback (σ/bw=5.76, d=7.19), milder degree. This never bit in the 6 live trades only because every one landed on a **low-GFS-spread day** (GFS ens_std 0.43–0.66°F → σ_emos 1.5–1.8°F, σ/bw < 2.0). The runtime σ-gate (`emos_runtime_sigma_threshold: 2.0`) would reject the ~33% of historical MIA/low days where GFS ens_std exceeds 0.79°F.

**Verdict:** The regime-shift dimension does **not** block MIA/low. But MIA/low should **not** be re-promoted to EMOS at 2026-06-11 without a **bounded-d refit** (cap d so σ_emos(p99)/bw ≤ 2.0), the same fix already prescribed for LAX/high. The current iso_v2 assignment is appropriate. The −37.2% citation should be corrected in `config.yaml` to avoid mis-framing the next re-promotion decision.

---

## 1. Corpus μ-calibration analysis (parallel to CHI/low §1, LAX/low §1)

Computed `emos_mu = a + b·ens_mean` on n=175 MIA/low settled rows (GFS `ens_mean`, since production EMOS feeds the **GFS-only** ensemble mean — `python/strategies/weather_evaluation.py:765`, `ens_mean_raw, ens_std_raw = _stats(gfs_members_finite)`). This is the correct scale: μ-calibration is on the same GFS-only ensemble the runtime uses.

### 1.1 μ-drop distribution

| Stat | MIA/low | LAX/low | CHI/low |
|---|---:|---:|---:|
| mean | **−2.69** | −4.20 | −4.54 |
| std | **0.30** | 2.43 | 1.98 |
| min | −3.28 | −11.86 | −9.26 |
| max | −1.86 | +0.00 | −0.41 |

MIA/low's μ-drop is the mildest and by far the tightest of the three (std 0.30 vs 1.98–2.43). With b=0.909 the shrinkage is only ~9%, and Miami's GFS ensemble is tightly clustered, so the bias correction is small and nearly constant.

### 1.2 Lead-time independence

| Lead | n | mean μ_drop (°F) | std (°F) |
|---:|---:|---:|---:|
| 0 | 54 | −2.75 | 0.26 |
| 1 | 61 | −2.67 | 0.33 |
| 2 | 60 | −2.66 | 0.31 |

Means within 0.10°F across leads. **Not a forecast-degradation-with-lead phenomenon** — same as CHI/low and LAX/low.

### 1.3 Quartile breakdown — the key over-correction test

By **ens_mean regime** (the dimension that exposed CHI/low):

| Quartile | ens_mean | obs | emos_mu | Raw bias (ens−obs) | **EMOS residual (emos_mu − obs)** |
|---|---:|---:|---:|---:|---:|
| Q1 | 70.54 | 68.59 | 68.22 | +1.95 | **−0.37** ✓ |
| Q2 | 73.58 | 70.86 | 70.98 | +2.71 | **+0.12** ✓ |
| Q3 | 75.61 | 72.40 | 72.83 | +3.22 | **+0.44** ✓ |
| Q4 | 78.97 | 75.23 | 75.89 | +3.74 | **+0.66** ✓ |

Every quartile residual is within ±0.7°F. Crucially, the warm-regime quartile (Q4) residual is **+0.66** (EMOS slightly *too warm*), the **opposite** of over-correction. CHI/low's Q4 was −2.60 (severe cool over-correction); LAX/low's Q4 was −0.83 (mild). MIA/low is the cleanest of the three in the warm tail.

The raw ensemble bias grows modestly with ens_mean (+1.95 → +3.74 across quartiles), and the mild b=0.909 shrinkage tracks it well enough that the residual stays near zero and never flips sharply negative.

### 1.4 Regression — does over-correction grow with ens_mean?

```
EMOS_residual = +0.0260 · ens_mean − 1.728   (n=175)
```

| ens_mean (°F) | Predicted EMOS residual (°F) |
|---:|---:|
| 65 | −0.04 |
| 70 | +0.09 |
| 75 | +0.22 |
| 78 (warm extreme) | +0.30 |

The slope is **positive small** — EMOS gets *slightly warmer* (not cooler) at higher ens_means, so there is no warm-tail over-correction risk. Direction and magnitude match LAX/low (+0.039); the opposite of CHI/low (−0.072, which drove the 2026-05-21 incident).

**Conclusion (the question this audit was scoped to answer): the `b<1` regime-shift μ-pathology is structurally absent in MIA/low.** μ is well-calibrated across the full training range. Whatever explains MIA/low's live result, it is not a μ-drop failure.

---

## 2. Live trade audit (the complete EMOS era: n=6 fills)

### 2.1 The EMOS era was net positive

MIA/low's live calibrator was `emos_raw` for exactly **6 entry fills, fc 2026-05-17 → 05-21** — confirmed complete against the fresh 2026-05-27 pull (no EMOS fills on fc 2026-05-22/23 before the 2026-05-24 rollback; those candidates were rejected by `CAL_FLOOR_BLOCK` + `RAW_PROB_GATE`, per `eval_audit.jsonl`). All 6 were **NO** bets on the modal/upper bracket (brackets are 1°F: floor 77 / cap 78, etc.), with authoritative settlement P&L from `shadow_settlements.full.json`:

| fc_date | bracket | side | px | n | emos_μ | emos_σ | GFS ens_std | obs | won? |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| 2026-05-17 | [77,78] | NO | 0.57 | 41 | 76.00 | 1.81 | 0.66 | 73 | ✓ |
| 2026-05-18 | [77,78] | NO | 0.53 | 20 | 76.28 | 1.80 | 0.65 | 75 | ✓ |
| 2026-05-19 | [77,78] | NO | 0.57 | 18 | 76.18 | 1.55 | 0.48 | 75 | ✓ |
| 2026-05-19 | [75,76] | NO | 0.64 | 8 | 76.21 | 1.56 | 0.49 | 75 | ✗ |
| 2026-05-20 | [76,77] | NO | 0.56 | 14 | 75.60 | 1.59 | 0.52 | 73 | ✓ |
| 2026-05-21 | [77,78] | NO | 0.56 | 8 | 75.97 | 1.52 | 0.43 | 78* | ✗ |

**Total: +47.7% ROI (4 wins / 2 losses).** (\*The 2026-05-21 row settled in-bracket per Kalshi; the snapshot corpus obs of 78.0 is a boundary-rounding artifact — see the 2026-05-03 boundary-grazing note (unpublished). There was also a same-day small partial cash-out on the 05-17 position.)

The EMOS-era trading was profitable. The losses were ordinary calibration misses on tight 1°F brackets, not μ-drop catastrophes.

### 2.2 Reconciling the "−37.2% EMOS ROI" claim

`python/config.yaml` (the `emos_enabled_cells` comment) and `docs/research/2026-05-24-harness-zero-winners-investigation.md` (Branch 2) state *"Live MIA/low EMOS ROI was −37.2%."* That figure is a **cell-level** ROI computed over MIA/low's entire live history (`state/trades.jsonl` + `shadow_settlements.full.json`), which is dominated by **non-EMOS** trades:

- Loss on `KXLOWTMIA-26MAY11-B79.5` (n=360, `strategy=None` reconciler-synthetic, fc 2026-05-11, **isotonic era**) — the single largest loss.
- Loss on `KXLOWTMIA-26MAY15-B72.5` (n=33, **isotonic era**, the blend-disagreement loss already cited in the override doc).
- A large synthetic win (`26MAY08-B73.5`, n=400) partially offsets, but the synthetic mega-positions and the isotonic losses swamp the small EMOS-era fills.

So the −37.2% is **not the ROI of EMOS-era trading** — those 6 fills returned +47.7%. The label "EMOS ROI" in `config.yaml` mis-attributes a cell-level (mostly isotonic + synthetic) loss to the calibrator.

**This does not reverse the rollback.** The 2026-05-24 demotion was decided on the per-cell harness winner certification (iso_v2_global beats EMOS on the pre-registered hold-out, BSS LB **+0.031**, 5/5 folds — `docs/research/2026-05-24-harness-zero-winners-investigation.md`), which is the correct, knob-invariant basis. The rollback is sound; only its *corroborating narrative* ("−37.2% EMOS ROI") is wrong. The honest summary is: **EMOS-era MIA/low trading was a small positive sample (n=6, +47.7%), too small to confirm anything, and iso_v2 won the hold-out on the merits.**

---

## 3. The real latent EMOS risk: σ-incompatibility (LAX/high class, not CHI/low class)

MIA/low's `unstable_d` (d=3.916) is the actual structural concern. σ_emos = √(c + d·ens_std²) with GFS-only ens_std:

| Cell | b | d | σ_emos @ GFS-std p99 | σ/bw (1°F) | Class |
|---|---:|---:|---:|---:|---|
| **MIA/low** | 0.909 | **3.916** | **3.56** | **3.56** | σ-veto fail |
| LAX/low (live) | 0.330 | 0.774 | 2.35 | 2.35 | σ-veto fail (borderline) |
| MIA/high | 1.001 | 0.963 | 2.22 | 2.22 | σ-veto fail (borderline) |
| LAX/high (rolled back) | 0.534 | 7.188 | 5.76 | 5.76 | σ-veto fail (severe) |
| CHI/low (excluded) | 0.791 | 2.177 | 6.49 | 6.49 | σ-veto fail (high c) |

MIA/low's σ/bw = 3.56 exceeds the 2.0 σ-veto and is the same *failure class* as LAX/high (σ floods probability mass across 1°F brackets), at a milder degree (d=3.916 vs 7.188).

**Why it never bit in the 6 live trades:** Miami's GFS ensemble is usually tight. σ_emos crosses 2.0 only when GFS ens_std exceeds **0.79°F**. The corpus median GFS ens_std is 0.59°F; **~33% of MIA/low days (57/175) exceed 0.79°F** and would produce σ/bw > 2.0. Every one of the 6 live trades landed on a low-spread day (GFS ens_std 0.43–0.66°F → σ_emos 1.5–1.8 → σ/bw < 2.0), so the σ-veto/runtime gate was never the binding constraint. The runtime σ-gate (`emos_runtime_sigma_threshold: 2.0`, 2026-05-19) is what would have rejected the high-spread days, and it correctly leaves the compatible-σ days tradeable.

**Note on the σ-veto vs the 2026-05-11 promotion:** the override record (`data/emos/overrides.json:MIA/low`) lists only `unstable_d` as the gate trip — `incompatible_sigma` is not recorded, consistent with the fit-time σ-veto and runtime σ-gate both postdating the 2026-05-11 promotion. The σ/bw=3.56 figure is therefore a retrospective computation (matching the "historical σ/bracket" 3.56 in the unpublished tracker and the σ/bracket ratios in `docs/calibration-runbook.md`). Under today's σ-veto, MIA/low fails the structural threshold.

---

## 4. Mechanism summary — MIA/low vs LAX/low vs CHI/low

| Property | MIA/low | LAX/low | CHI/low |
|---|---|---|---|
| EMOS b (shrinkage) | 0.909 (9%) | 0.33 (67%) | 0.791 (21%) |
| μ-drop mean ± std | −2.69 ± 0.30 | −4.20 ± 2.43 | −4.54 ± 1.98 |
| Q4 (warm) EMOS residual vs obs | **+0.66** (slightly warm) | −0.83 (mild cool) | **−2.60** (severe cool) |
| Residual-vs-ens_mean slope | **+0.026** (flat) | +0.039 (flat) | **−0.072** (over-corrects) |
| Regime-shift μ-pathology? | **No** | No | **Yes** |
| EMOS d (σ²-slope) | **3.916** (`unstable_d`) | 0.774 (`unstable_d`) | 2.177 |
| σ_emos(p99)/bw (1°F) | **3.56** (σ-veto fail) | 2.35 (borderline) | 6.49 |
| Dominant live risk | **σ-incompatibility** | live ROI drift (−16.3%) | regime-shift μ-drop |
| Live incident manifest? | No (gated to low-σ days) | No | Yes (2026-05-21) |

MIA/low and CHI/low both have b<1, but their failure modes are orthogonal: CHI/low fails on **μ** (warm-tail over-correction), MIA/low's only EMOS hazard is **σ** (wide σ on high-GFS-spread days). The generalization from the LAX/low audit holds: the right diagnostic is not "is b<1?" but the *joint* (μ-calibration-vs-regime, σ-vs-bracket-geometry) check — and both are now subsumed by the per-cell calibrator harness.

---

## 5. Verdict and recommendations

1. **Regime-shift dimension: clear.** The CHI/low μ-pathology is structurally absent in MIA/low. No precautionary measure on the μ dimension is required pre-2026-06-11.

2. **EMOS re-promotion at 2026-06-11: gated on a bounded-d refit.** MIA/low should not return to `emos_enabled_cells` while d=3.916 keeps σ/bw at 3.56. Re-promotion should require the same **bounded-d constraint already prescribed for LAX/high** (`overrides.json:LAX/high.rollback_reason`: "cap d at cell_ens_std_p99 / bracket_width"), so that σ_emos(p99)/bw ≤ 2.0, **and** a fresh per-cell harness win over iso_v2 on a post-refit hold-out. Absent both, iso_v2 remains the correct assignment.

3. **Data-integrity fix (low effort, recommended).** Correct the `config.yaml` `emos_enabled_cells` comment: the −37.2% is a cell-level ROI dominated by isotonic-era and reconciler-synthetic trades, not the EMOS-era ROI (which was +47.7% on n=6). State the rollback basis as the harness hold-out certification, not the ROI figure. Separately, `data/emos/overrides.json:MIA/low.rolled_back` is still `null` despite the 2026-05-24 demotion — update it to mirror the LAX/high `rolled_back: true` record for audit-trail consistency.

4. **No new standalone follow-ups.** This closes the last open item from the CHI/low audit (step 2). The σ-incompatibility re-promotion gate and the harness re-evaluation are existing, tracked workstreams; nothing here requires a new lane.

**This audit closes the CHI/low-audit step-2 follow-up.** Net finding: MIA/low's brief EMOS era was harmless-to-mildly-positive; its demotion to iso_v2 is correct on the harness merits; and its only structural barrier to future EMOS re-promotion is `unstable_d`/σ-geometry, addressable by a bounded-d refit — not the regime-shift μ failure that took down CHI/low.
