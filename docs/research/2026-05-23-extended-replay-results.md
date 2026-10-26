# Extended Phase-4 Replay Results — 45-day hold-out

**⚠ SUPERSEDED 2026-05-23 by the ROI-extension session.** This doc captures the first 45-day replay run (Brier + Stage-1-only) before per-(cell, strike_type) ROI + std(predicted_prob) + within-day across-bracket std were added to the replay script. Conclusions here labeled "INCONCLUSIVE" for LAX/high, AUS/high, CHI/high, CHI/low were sharpened in the follow-up ROI session to clear FAIL verdicts for all four (with LAX/high/between as the single marginal pass at ROI = −2.4%). The compression hypothesis was confirmed in the ROI session (iso_v2 std_model_prob median 0.018 vs identity 0.141 — 7.8× compression). **Current authoritative artifact:** `data/research/2026-05-23-isotonic-v2-replay-extended-with-roi.json`. The whole cell-unblock workstream this doc was scoped under was then superseded by an unpublished feature workstream (see the unpublished tracker).

**Generated:** 2026-05-23
**Artifact:** `data/research/2026-05-23-isotonic-v2-replay-extended.json` (28d Brier-only)
**ROI-extension artifact:** `data/research/2026-05-23-isotonic-v2-replay-extended-with-roi.json` (current authoritative)
**Baseline (28d):** `data/research/2026-05-22-isotonic-v2-replay.json`
**Parent doc:** `docs/research/2026-05-23-cell-unblock-analysis.md` (§5.3) — also superseded

## 1. Run parameters

| Parameter | 28d baseline | 45d extended |
|---|---|---|
| `--today` | 2026-05-22 | 2026-05-22 |
| `--cutoff` (early/late split) | 2026-04-24 | 2026-04-07 |
| Late-partition window (days) | 28 | 45 |
| `--isotonic` | `data/calibration/isotonic_v2.json` (production pilot artifact) | same |
| `--block-range` | `[0.446, 0.883]` | same |
| `--min-calibrated-prob` | 0.110 (`PILOT_MIN_CALIBRATED_PROB` constant — matches prod `min_calibrated_prob`) | same |

`--today` could not be bumped to 2026-05-23: only `0200/` and `0800/` snapshot
dirs exist for that day, no `2000/` settlements yet.

## 2. Headline counts

| | 28d | 45d |
|---|---|---|
| `all_candidates` | 4,732 | 4,830 |
| `early_partition` (refit) | 2,795 | 1,497 |
| `late_partition` (hold-out) | 1,937 | 3,333 |
| `identity_stage1_passes` | 475 | 751 |
| `isotonic_stage1_passes` | 479 | 783 |
| `identity_after_per_day_cap` | 258 | 419 |
| `isotonic_after_per_day_cap` | 246 | 421 |

In-memory refit converged on n=1,497 early-partition pairs (no warning logged).
Assumption flagged in the spec resolved: early partition is large enough.

## 3. Global Brier (would-have-traded set, after per-day cap)

| Arm | 28d | 45d | Δ |
|---|---|---|---|
| `identity` | 0.2076 | 0.2165 | +0.0089 |
| `isotonic_v2_refit` | 0.2078 | **0.1880** | **−0.0198** |
| **Δ (isotonic − identity)** | **+0.0002** | **−0.0285** | — |

The extended window flipped the global verdict: at 28d, isotonic_v2 was a wash
versus identity; at 45d it beats identity by 0.0285 Brier points on a
391-candidate hold-out set (after cap), driven by `greater` and `less` strikes
(see §6).

## 4. Per-cell results — 7 excluded cells

All numbers below are for the `isotonic_v2_refit` arm in the late partition.

`Brier_climatology = mean_outcome × (1 − mean_outcome)` is the per-cell base-rate
predictor — the bar gate 2 asks isotonic to clear (point estimate only;
bootstrap CIs are NOT in the artifact, flagged below).

| Cell | n_28d | n_45d | Brier_28d | Brier_45d | win_rate_45d | mean_p_45d | mean_outcome_45d | Climatology_Brier | iso − clim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEN/low   | 14 | 28 | 0.1723 | 0.2091 | 0.250 | 0.189 | 0.250 | 0.1875 | **+0.0216** |
| DEN/high  | 18 | 34 | 0.3193 | 0.2173 | 0.294 | 0.191 | 0.294 | 0.2076 | **+0.0097** |
| SEA/high  | 18 | 37 | 0.2694 | 0.2546 | 0.351 | 0.204 | 0.351 | 0.2279 | **+0.0267** |
| LAX/high  | 23 | 39 | 0.1703 | 0.1744 | 0.231 | 0.192 | 0.231 | 0.1775 | −0.0031 |
| AUS/high  | 14 | 25 | 0.2542 | 0.1581 | 0.200 | 0.186 | 0.200 | 0.1600 | −0.0019 |
| CHI/high  | 17 | 29 | 0.1054 | 0.1420 | 0.172 | 0.184 | 0.172 | 0.1427 | −0.0007 |
| CHI/low   | 12 | 19 | 0.2194 | 0.1504 | 0.211 | 0.191 | 0.211 | 0.1662 | −0.0158 |

## 5. Gate evaluation

| Gate | Cell | n≥45? | Brier < clim (point est)? | Mass-collision check | ROI > −10%? | Verdict |
|---|---|---|---|---|---|---|
| | DEN/low   | FAIL (28) | FAIL (+0.0216) | n/a | UNKNOWN | **KEEP EXCLUDED** |
| | DEN/high  | FAIL (34) | FAIL (+0.0097) | n/a | UNKNOWN | **KEEP EXCLUDED** |
| | SEA/high  | FAIL (37) | FAIL (+0.0267) | n/a | UNKNOWN | **KEEP EXCLUDED** |
| | LAX/high  | FAIL (39) | PASS (−0.0031, marginal) | mass survives gates | UNKNOWN | **INCONCLUSIVE** |
| | AUS/high  | FAIL (25) | PASS (−0.0019, marginal) | mass survives gates | UNKNOWN | **INCONCLUSIVE** |
| | CHI/high  | FAIL (29) | PASS (−0.0007, ≈ tied) | mass survives gates | UNKNOWN | **INCONCLUSIVE** |
| | CHI/low   | FAIL (19) | PASS (−0.0158)         | mass survives gates | UNKNOWN | **INCONCLUSIVE** |

### Spec assumptions that didn't hold

1. **Gate 1 (n≥45) fails for ALL 7 cells.** The user's spec anticipated the
   extended window would clear n=45 on most cells (e.g. "LAX/high n=23 → ~37-40").
   It hits 37-40 trades, not 45. After Stage-1 gates + per-day-event cap, the
   45-day window yields 19-39 trades per cell. The cell-by-cell pull-through
   rate is lower than a `1 trade / cell / day` upper bound suggests — the
   per-day cap fires aggressively, the `min_raw_prob_for_between=0.12` filter
   removes many `between` candidates, and DEN/SEA/AUS cells have markets that
   often don't pass the edge threshold.

2. **Gate 2 bootstrap CI: NOT COMPUTABLE from artifact.** The replay script
   emits point-estimate Brier only. The "Brier < climatology with 95% CI
   lower bound > 0" portion of gate 2 cannot be evaluated from this run — it
   needs a script-side bootstrap (resample late-partition rows, re-compute
   Brier vs base rate, take the 2.5/97.5 percentile). Recommend extending
   `replay_isotonic_holdout.py` if the gate matters for final approval.

3. **Gate 3 (per-cell mass collision): partially answered.** The artifact's
   `min_raw_prob_for_between_sensitivity` block confirms global mass survives
   the `min_raw_prob_for_between=0.12` filter (n_passing=1014 → 836 across
   thresholds 0.10-0.15 for isotonic). The per-cell breakdown (`n` in the
   per-cell tables) confirms each excluded cell still produces trades under
   iso_v2_refit even with the `block_raw_prob_range=[0.446, 0.883]` and
   `min_calibrated_prob=0.110` gates active. No null-on-the-cell pathology.

4. **Gate 4 (ROI > −10%): NOT COMPUTABLE from artifact.** Per-cell rollup
   carries `win_rate` and `mean_model_prob` but neither `mean_yes_ask` nor
   per-trade P&L. The LAX/high parallel here is from
   an unpublished calibration-state note and the LAX/high σ-veto note in
   `python/config.yaml` — the EMOS rollback at LAX/high produced live
   ROI=−39.6% despite acceptable Brier. **The 4 INCONCLUSIVE cells in this
   replay are pre-σ-veto candidates; the same failure mode applies until an
   ROI proxy is computed.**

## 6. Per-strike-type — corroborating the global Δ

| Strike | identity Brier (45d) | isotonic Brier (45d) | Δ |
|---|---:|---:|---:|
| `between` | 0.2182 | 0.2594 | +0.0413 |
| `greater` | 0.1930 | 0.1074 | −0.0856 |
| `less`    | 0.2381 | 0.1487 | −0.0894 |

The global isotonic win comes entirely from `greater`/`less` strikes;
`between` markets still lose to identity. This is the live-cell pattern that
motivated keeping AUS/high, CHI/high, CHI/low (high-`between`-fraction cells)
out of the trading set during the 2026-05-22 pilot flip.

## 7. Recommendations

| Cell | Recommendation | Reason |
|---|---|---|
| **DEN/low**  | **Keep excluded** | iso_v2 Brier > climatology (+0.0216) at n=28. Under-prediction (+6pp). Gate 2 fails on point estimate. |
| **DEN/high** | **Keep excluded** | iso_v2 Brier > climatology (+0.0097) at n=34. Under-prediction (+10pp). Gate 2 fails on point estimate. |
| **SEA/high** | **Keep excluded** | Worst case of the seven. iso_v2 Brier > climatology (+0.0267) at n=37. Under-prediction (+15pp). Gate 2 fails decisively. |
| **LAX/high** | **Needs deeper investigation** | Beats climatology by 0.0031 (within noise). Cell has prior live ROI=−39.6% under EMOS — gate 4 (ROI) is the binding constraint, not Brier. Don't unblock without an ROI proxy. |
| **AUS/high** | **Needs deeper investigation** | Beats climatology by 0.0019 (within noise). n=25 below gate 1. Excluded 2026-05-22 pre-emptively as `between`-heavy cell; §6 shows `between` is where iso_v2 loses. Re-evaluate after gate 4 lands. |
| **CHI/high** | **Needs deeper investigation** | Functionally tied with climatology (Δ=−0.0007). Same `between`-heavy concern as AUS/high. |
| **CHI/low**  | **Needs deeper investigation** | Best Brier margin of the marginal cells (Δ=−0.0158) but lowest n (19). Strongest unblock candidate if gates 1 and 4 can be evaluated. |

**Net: zero cells PASS all four gates. Three cells (DEN/low, DEN/high, SEA/high)
clearly stay excluded. Four cells (LAX/high, AUS/high, CHI/high, CHI/low) are
INCONCLUSIVE and blocked on gates 1 (need more days) and 4 (need ROI proxy).**

## 8. Caveats inherited from the replay tool

- **Stage 1 only.** Snapshots store top-of-book `yes_bid`/`yes_ask` but not
  orderbook depth. Stage-2 VWAP / fill-depth rejections are NOT modeled.
  This is calibration-quality validation, NOT a P&L proxy. (Same caveat
  applies to the 28d baseline.)
- **In-memory refit, not the live `isotonic_v2.json`.** The replay re-fits
  isotonic on the early partition. Promotion would still need a freshly-fit
  artifact going through the standard plateau-derivation + smoke-test path.

## 9. Operator next steps (suggested, not actioned)

1. Drop DEN/low, DEN/high, SEA/high from any active "unblock candidates"
   list — extended hold-out confirms calibrator failure.
2. If LAX/high, AUS/high, CHI/high, or CHI/low are to be reconsidered, the
   blocker is **gate 4 (per-trade ROI proxy)**. Extending
   `replay_isotonic_holdout.py` to emit per-cell `(mean_yes_ask, mean_yes_bid)`
   would let you approximate ROI as `(win_rate − mean_yes_ask) / mean_yes_ask`.
3. **Per-cell bootstrap CIs** (gate 2) are the second blocker. A resample loop
   over the late-partition rows in the replay script would close this.
4. No changes to `python/config.yaml` or any production artifact made by this
   run. Output is a recommendation only.
