# Per-cell unblock analysis for excluded weather cells

**Status:** Read-only research pass. No implementation.
**Date:** 2026-05-23
**Author:** christurgeon (with Claude Code)
**Scope:** 7 cells currently in `python/config.yaml:strategies.weather.excluded_city_vars` — DEN/low, DEN/high, SEA/high, LAX/high, AUS/high, CHI/high, CHI/low.
**Companion artifacts:** `data/research/2026-05-22-isotonic-v2-replay.json` (Phase-4 Stage-1-only replay, 28d hold-out under post-fix bracket math), `/tmp/emos_scratch/coefficients.json` (fresh refit on rebuilt corpus 2026-03-18 → 2026-05-22).
**Hard scope guard:** Spec lifted verbatim from the user task — no new isotonic refits, no production-artifact edits, no strategy modules, no spread/SNR/shrinkage tricks, no NWP models beyond GFS+ECMWF.

**⚠ SUPERSEDED 2026-05-23.** This doc captures a moment-in-time analysis of the 7 excluded weather cells under the verdict-A/B/C framework. Its analytical framing (per-cell unblock decisions made via ad-hoc backtest + σ-veto reasoning) was **superseded later the same day** by the per-cell calibrator evaluation harness workstream (an unpublished feature workstream, the unpublished tracker). The harness replaces the manual A/B/C framework with a repeatable per-cell scoring process that handles "winner / don't trade" as first-class outcomes. **Read this doc as historical context, not the going-forward plan.** Specific stale items: §1.4's "calibrator does not beat identity" framing was inverted by the 45d extended replay (which then turned out to be Brier-via-compression, not genuine improvement); §4.3's "4-cell bucket-3 under-prediction failure" collapsed to 2 cells (DEN/high, SEA/high) at 45d with AUS/high + CHI/low revealed as small-sample noise; §5.3's "extend the replay" recommendation has been executed.

**2026-05-23 follow-up:** Documentation drift listed in earlier versions of this doc has been fixed in commit-pending changes to `python/config.yaml`, `docs/calibration-runbook.md`, and the unpublished tracker.

---

## TL;DR

- **All 7 excluded cells fail verdict (B) EMOS-as-currently-implemented (with this codebase's `incompatible_sigma` σ-veto)** — confirmed on a fresh refit against a rebuilt corpus that includes 12 more snapshot days (March 18 → May 22, n_train 144→172 per cell). σ_emos(p99)/bracket_width ratios are 2.5×–4.7× over the 2.0 threshold for the 7 cells; corpus growth does not move these. **The σ-veto is a project choice, not a law of nature** — it's a guardrail added 2026-05-19 after the LAX/high live failure (ROI=−39.6% in 7 days when bracket-incompatible σ flooded Kelly-capped sizing into wrong brackets). Quant firms absolutely use EMOS for short-bracket weather markets; the relevant literature (Gneiting & Raftery 2005, Hagedorn 2012) covers exactly these cells' regimes. The honest framing is that **EMOS-with-a-Gaussian-tail-and-our-σ-veto** is blocked, while **EMOS-the-framework with any of five named unlock paths** would work — see §4 for the list.
- **All 7 excluded cells fail verdict (A) isotonic_v2 on the user-defined n_settled_days ≥ 45 gate** — the live `eval_audit.jsonl` has only one post-fix forecast-date cohort (2026-05-23) at `bracket_window_version=2`; the existing 28-day Phase-4 replay (`data/research/2026-05-22-isotonic-v2-replay.json`) is 17 days short of n=45 and per-cell n is 10–23. Earliest the gate becomes evaluable on live data: **~2026-07-07** (45 days post bracket-fix landing 2026-05-21). **The right next move is NOT to wait** — re-run the Phase-4 replay today with a 2026-04-07 cutoff against existing on-disk snapshots (~5 min of compute) to get a 45-day hold-out window. 45 days of live data through 2026-07-07 is volume-starved under the bounded-exposure pilot (Kelly=0.075 + drawdown reduction 0.50 → ~5-8 trades/cell/45d); the replay matches that n while testing a different season. See §5.3 for the recommendation.
- **Every verdict is therefore (C) KEEP DISABLED, with a date and named structural improvement** — but the cells fall into three distinct structural buckets (not the original "EMOS d=0 group" label, since corrected in `python/config.yaml`):
  1. **High-`c` intrinsic-uncertainty cells** (DEN/high, SEA/high, DEN/low) — large `c` floor → Gaussian σ exceeds bracket geometry independent of spread. Beta-calibration lane or heavier-tail replacement is the named fix.
  2. **High-`d` informed-spread cells** (LAX/high, AUS/high, CHI/high, CHI/low, AUS/low) — `d` is well-identified (CIs strictly positive) but multiplies into bracket-incompatible σ on typical ensemble spreads. Both LAX/high and CHI/low have documented live-failure precedent. Bounded-`d` EMOS variant, regime-conditional EMOS, sizing-by-σ, or composite-bracket aggregation are all named paths to unblock; none built.
  3. **Pre-emptive pilot exclusions** (AUS/high, CHI/high, CHI/low — duplicates buckets above) — re-evaluation scheduled ~2026-06-05 under the existing pilot plan (`docs/research/2026-05-22-isotonic-refit-bounded-exposure-pilot.md` §"Per-cell re-evaluation"). Replay evidence is mixed: CHI/high looks like a "calibrator-OK, base-rate kills it" cell; AUS/high and CHI/low look like under-prediction failures.

The rollout sequence in §2 reflects ROI × confidence × ease-of-rollback ordering under these constraints.

---

## 1. Phase 0 — Artifact-derived ground-truth table

All values pulled directly from on-disk artifacts. The σ-ratio column uses bracket_width = 2.0°F (post-bracket-fix); `σ/bw > 2.0` is the non-overridable promotion veto (CLAUDE.md "Per-cell EMOS calibrator dispatch", `python/config.yaml:emos_runtime_sigma_threshold: 2.0`).

### 1.1 Currently-excluded cells (`excluded_city_vars`)

EMOS coefficients from the **fresh refit** at `/tmp/emos_scratch/coefficients.json` (fit_date 2026-05-22, n_train shown). Existing on-disk `data/emos/coefficients.json` (fit 2026-05-11, n_train 144–150) gives σ-ratios within ±5% of the values below — the corpus growth from 2026-05-11 → 2026-05-22 did not flip any cell's σ-veto verdict.

| Cell | n_train | a | b | c | d | ens_std_p99 | σ_emos@p99 | **σ/bw (post-fix)** | ci_warnings | Config comment (`config.yaml`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| DEN/low | 180 | -2.25 | 0.91 | 16.60 | 0.70 | 4.59 | 5.60 | **2.80** ❌ | `incompatible_sigma` | "σ-veto: σ_emos/bw=2.80; high `c=16.60` floor dominates variance" |
| DEN/high | 180 | -12.81 | 1.17 | 30.37 | 1.62 | 5.03 | 8.45 | **4.23** ❌ | `unstable_d, incompatible_sigma` | "σ-veto: σ_emos/bw=4.23; high `c=30.37` intrinsic-uncertainty floor (Rocky Mountain orographic)" |
| SEA/high | 175 | 8.67 | 0.84 | 15.75 | 1.31 | 3.54 | 5.67 | **2.84** ❌ | `unstable_d, incompatible_sigma` | "σ-veto: σ_emos/bw=2.84; high `c=15.75` floor (Puget Sound marine boundary layer)" |
| LAX/high | 175 | 33.27 | 0.53 | 0.69 | 7.11 | 2.05 | 5.52 | **2.76** ❌ | `unstable_d, incompatible_sigma` | "unstable_d override revoked 2026-05-18; live BSS=-4.0%, EMOS sigma=6.8F incompatible with 1F brackets (pre-fix) — still incompatible with 2F brackets (post-fix bracket-window 2026-05-21): σ_emos/bw=2.76, ROI=-39.6% in 7 days" |
| AUS/high | 178 | 3.45 | 0.96 | 11.45 | 0.78 | 5.84 | 6.18 | **3.09** ❌ | `incompatible_sigma` | "backtest alpha loss under post-2°F-bracket-fix math" |
| CHI/high | 174 | -1.10 | 1.04 | 14.46 | 3.22 | 4.77 | 9.36 | **4.68** ❌ | `unstable_d, incompatible_sigma` | "near-zero wins (identity 0/16, isotonic_v2 2/17) in 28d holdout" |
| CHI/low | 174 | 2.29 | 0.87 | 10.07 | 2.49 | 3.76 | 6.73 | **3.37** ❌ | `unstable_d, incompatible_sigma` | "demoted from EMOS 2026-05-21; no LIVE post-fix isotonic trades (Stage-1 replay shows n=12, +12pp under-prediction)" |

### 1.2 Currently-trading cells (for reference / sanity)

The 3 EMOS-promoted + 4 isotonic_v2-piloted cells:

| Cell | Lane | n_train | σ_emos@p99 | σ/bw | ci_warnings | Override |
|---|---|---:|---:|---:|---|---|
| MIA/high | EMOS | 175 | 2.40 | **1.20** ✓ | none | none (clean) |
| LAX/low | EMOS | 175 | 2.32 | **1.16** ✓ | `unstable_d` | `data/emos/overrides.json#LAX/low` (re_evaluate_at 2026-06-11) |
| MIA/low | EMOS | 175 | 3.29 | **1.64** ✓ | `unstable_d` | `data/emos/overrides.json#MIA/low` (re_evaluate_at 2026-06-11) |
| NYC/high | isotonic_v2 | 172 | 7.96 | **3.98** ❌ | `incompatible_sigma` | demoted from EMOS 2026-05-21 (post-incident); EMOS re-promotion gated |
| NYC/low | isotonic_v2 | 172 | 4.99 | **2.49** ❌ | `incompatible_sigma` | rolled back from EMOS 2026-05-18 (live BSS_emos_vs_iso=-6.6%) |
| AUS/low | isotonic_v2 | 178 | 6.93 | **3.46** ❌ | `unstable_d, incompatible_sigma` | not in `emos_enabled_cells` |
| SEA/low | isotonic_v2 | 63 | 1.91 | **0.96** ✓ | none on refit (`unstable_b, d_at_zero` on 2026-05-11 fit) | low_n at 2026-05-11, now n=63 |

Sanity: SEA/low passes the σ-gate on the fresh refit and the warnings (`unstable_b`, `d_at_zero`) on the 2026-05-11 fit have cleared with corpus growth. That's noteworthy but **out of scope** for this pass (SEA/low is not excluded; it's trading on isotonic_v2 already).

### 1.3 Stage A / Stage B BSS history (where available)

From `data/emos/refit_validation_20260511T234543Z.json` (last EMOS shadow validation against production isotonic, run 2026-05-12). Note this uses the **pre-bracket-fix** isotonic baseline (`data/calibration/isotonic.json`, fit 2026-05-12) — Stage B values below are NOT directly applicable to today's `isotonic_v2`, but they show which cells EMOS structurally beats isotonic-flavor calibrators independent of bracket-window math.

| Cell | Stage A n | Stage A BSS (cand_vs_prod EMOS-on-EMOS) | Stage B (EMOS vs isotonic-2026-05-12) | Stage B verdict (then) |
|---|---:|---|---|---|
| AUS/high | 6724 | +0.016 [+0.014, +0.019] | -0.331 [-0.360, -0.306] | **NOT_BETTER_THAN_ISO** |
| AUS/low | 5338 | +0.006 [+0.004, +0.008] | -0.068 [-0.078, -0.059] | **NOT_BETTER_THAN_ISO** |
| CHI/high | 5799 | +0.024 [+0.023, +0.025] | -0.384 [-0.411, -0.357] | **BLOCKED_WARNINGS** (unstable_d) — and trailing isotonic on Brier |
| CHI/low | 5686 | +0.007 [+0.002, +0.011] | +0.158 [+0.130, +0.183] | **STAGE_B_ELIGIBLE** (promoted 2026-05-18, demoted 2026-05-21) |
| MIA/high | 4447 | +0.188 [+0.185, +0.193] | +0.359 [+0.349, +0.370] | STAGE_B_ELIGIBLE (live) |
| MIA/low | 5053 | +0.115 [+0.113, +0.117] | +0.262 [+0.258, +0.267] | BLOCKED_WARNINGS → override → live |
| NYC/high | 5318 | +0.041 [+0.038, +0.045] | +0.184 [+0.171, +0.197] | BLOCKED_WARNINGS → override → live → demoted 2026-05-21 |
| NYC/low | 4733 | +0.012 [+0.010, +0.015] | +0.143 [+0.141, +0.146] | STAGE_B_ELIGIBLE → promoted → rolled back 2026-05-18 |

Cells absent from the 2026-05-12 validation (DEN/low, DEN/high, SEA/high, LAX/high, SEA/low): these were already excluded or below the validation's `n_min=15` floor.

### 1.4 Phase-4 replay (`data/research/2026-05-22-isotonic-v2-replay.json`)

Stage-1-only replay over 2026-04-24 → 2026-05-22 (28 days) under post-fix bracket math (`bracket_window_version=2`). Pilot config (`kelly_fraction=0.075`, `min_calibrated_prob=0.110`, `block_raw_prob_range=[0.446, 0.883]`). Per-cell stats under the **isotonic_v2** arm:

| Cell | n (after cap) | wins | win_rate | brier | mean_model_prob | Δ (outcome − model) |
|---|---:|---:|---:|---:|---:|---:|
| DEN/low | 14 | 3 | 0.214 | 0.172 | 0.177 | +0.04 (mild over-pred losses) |
| DEN/high | 18 | 8 | 0.444 | 0.319 | 0.196 | **+0.25** (severe under-prediction of wins) |
| SEA/high | 18 | 7 | 0.389 | 0.269 | 0.194 | **+0.20** (severe under-prediction of wins) |
| LAX/high | 23 | 5 | 0.217 | 0.170 | 0.205 | +0.01 (calibrated on Brier, but trade selection lossy — see §4-LAX/high) |
| AUS/high | 14 | 5 | 0.357 | 0.254 | 0.183 | **+0.17** (under-prediction) |
| CHI/high | 17 | 2 | 0.118 | 0.105 | 0.185 | -0.07 (over-prediction; calibrator-OK, base-rate kills it) |
| CHI/low | 12 | 4 | 0.333 | 0.219 | 0.216 | +0.12 (under-prediction) |

**Global Brier on the late partition**: identity 0.2076, isotonic_v2 0.2078 — calibrator does not beat identity on the hold-out, despite the on-fit corpus Brier improvement of 16.87% (isotonic_v2.json metadata). This is the central tension that motivates the bounded-exposure pilot rather than a full revert.

Per-strike-type under isotonic_v2: between win_rate 0.357 (n=129), greater 0.159 (n=63), less 0.241 (n=54). Mean model_prob clusters in [0.17, 0.20] across all three types — consistent with isotonic_v2's narrow calibrated-output band (`values[0]=0.099`, `values[-1]=0.467`).

---

## 2. Executive summary — prioritized rollout sequence

Ranked by **expected ROI × confidence × ease-of-rollback**, conditioned on a future state where the n=45 data gate has been met and the σ-veto blocker has been addressed (neither of which is in scope today). Every entry is a verdict (C) today; the ordering tells you which to revisit first when conditions change.

Each entry is annotated for both pilot knobs (`kelly_fraction=0.075`, `max_total_exposure_pct=0.30`, `min_calibrated_prob=0.110`, `block_raw_prob_range=[0.446, 0.883]`) and post-revert canonical knobs (`kelly_fraction=0.15`, `max_total_exposure_pct=0.80`, `min_calibrated_prob=0.10`, the pre-pilot `block_raw_prob_range=[0.649, 0.967]`).

| Rank | Cell | Best-case revisit lane | Earliest evaluable date | Validity under pilot vs post-revert | Confidence | Ease-of-rollback |
|---:|---|---|---|---|---|---|
| 1 | **CHI/high** | (A) isotonic_v2 — Phase-4 replay shows good Brier (0.105 on n=17) and calibrated mean_prob; problem is low base rate driving the EV negative, not calibrator pathology. | 2026-06-05 (pre-emptive-exclusion revisit date in pilot plan) | Pilot: trades plausibly unprofitable due to ~12% base rate × bounded Kelly. Post-revert: same problem, higher exposure. **The unblock is operational (trade-selection filter), not calibration.** | Medium — n=17 replay is short but consistent | High (just `excluded_city_vars` removal) |
| 2 | **CHI/low** | (A) isotonic_v2 — Phase-4 brier 0.219, n=12; isotonic under-predicts wins by 12pp but cell would still need extension to n≥30 before any confidence | 2026-06-05 (pre-emptive-exclusion revisit) | Pilot: small wagers, under-prediction means model picks NO frequently — if reality is win-rate 0.33, NO trades lose 33% of the time. Post-revert: amplifies the loss. | Low–Medium | High |
| 3 | **AUS/high** | (A) isotonic_v2 — Phase-4 brier 0.254, n=14, under-predicts wins by 17pp. Comparable to CHI/low. | 2026-06-05 | Pilot: same under-prediction concern as CHI/low. AUS_high also has unresolved tech-debt §269 / §382 hypotheses (regime, lead-time, ECMWF behavior) that should be tested first. | Low | High |
| 4 | **LAX/high** | (B) EMOS via Beta-calibration lane OR `d`-bounded EMOS variant (the unpublished tracker) | Conditional on Beta-lane implementation (no scheduled date) | Pilot: irrelevant; structural σ-veto. Post-revert: still σ-vetoed. Beta-calibration lane is the documented path. | Low (depends on lane that doesn't exist yet) | N/A (no flip possible) |
| 5 | **DEN/low** | (A) isotonic_v2 — Phase-4 brier 0.172 on n=14, well-calibrated (mean_model_prob 0.177 vs win_rate 0.214). | 2026-07-07 (n=45 on live data) — earlier via replay extension | Pilot: small n, low-impact even if it loses. Post-revert: amplifies. **The only excluded cell where isotonic_v2 shows clean calibration on the replay.** | Medium (best calibration evidence among excluded cells) | High |
| 6 | **SEA/high** | (B) Beta-calibration lane (σ-veto) OR (A) isotonic_v2 — but Phase-4 shows under-prediction of wins by 20pp (brier 0.269, n=18). | Conditional on Beta-lane | Severely under-predicts on the replay; trades would be NO-biased and lose at the 39% win rate. | Low | High (if (A) is attempted) |
| 7 | **DEN/high** | (B) Beta-calibration lane (σ-veto) — high-`c` orographic uncertainty; isotonic_v2 also struggles (brier 0.319, n=18, +25pp under-prediction). | Conditional on Beta-lane | Worst Phase-4 brier among all excluded cells. Both lanes appear to struggle structurally. | Very low | High (if (A) is attempted) |

**Aggregate**: today's verdict for all 7 cells is (C) KEEP DISABLED. The ordering above is "where to invest revisit attention first" once data + structural improvements arrive. **No cell ranks high enough to justify a near-term unblock attempt under either knob set.**

**On "earliest evaluable date" — extended-replay beats live-wait:** Every cell in the table above lists either "2026-06-05" (pilot revisit) or "2026-07-07" (n=45 live data) as the evaluable date. Both estimates are pessimistic: **extending the existing Phase-4 replay cutoff backward from 2026-04-24 to 2026-04-07 (snapshots are on disk back to 2026-03-18) is a ~5-minute compute job that gives a 45-day post-fix-math hold-out window today**. This unblocks verdict-(A) gate evaluation for every cell whose blocker is "n_settled_days < 45" rather than a structural calibration failure. Under the bounded-exposure pilot, 45 days of live data through 2026-07-07 is expected to yield ~5-8 trades/cell — comparable n to what the extended-replay already provides, and the replay tests a different seasonal regime. Live data is uniquely valuable for catching Stage-2 (orderbook depth, slippage) failure modes the replay cannot model, but those are not the blocker for any of the 7 cells. **Recommendation: re-run the replay now, defer the live-wait decision to per-cell evidence from the extended replay.** Detailed in §5.3.

---

## 3. Per-cell sections

### 3.1 DEN/low

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 5.60 / 2.0 = **2.80** | ❌ FAIL | Non-overridable per CLAUDE.md "Per-cell EMOS calibrator dispatch" |
| EMOS `d` coefficient | 0.70 (was 0.0 in 2026-05-11 fit) | n/a | Corpus growth produced a small positive `d`, but `c=16.60` dominates the σ — moving `d` doesn't unblock. |
| EMOS `ci_warnings` | `incompatible_sigma` only (`d_at_zero` cleared on refit) | n/a | The warning that matters here is the new structural one. |
| Phase-4 replay n on `isotonic_v2` | 14 | ❌ < 45 | Below user's n_settled_days gate. |
| Phase-4 replay brier | 0.172 | OK in absolute | But n=14 too small for stable per-cell BSS inference. |
| Phase-4 replay calibration | mean_model_prob 0.177, win_rate 0.214 | ✓ tight | **Best calibration evidence of any excluded cell on this replay** — the model under-predicts wins by only 4pp, within the n=14 CI half-width. |
| 2026-05-12 Stage B evidence | not present (excluded from validation) | n/a | DEN/* were never in the EMOS Stage B set. |

#### Why not (A) today

The Phase-4 replay's small n=14 falls short of the user's n_settled_days ≥ 45 gate; the BSS vs per-cell climatology base rate cannot be computed at the required CI tightness. **Note however that DEN/low is the only excluded cell whose Phase-4 replay calibration looks acceptable**, suggesting it's the strongest verdict-(A) candidate once data accumulates.

#### Why not (B) today

`incompatible_sigma` σ/bw = 2.80. The σ comes overwhelmingly from `c=16.60` (intrinsic uncertainty floor), not from spread term `d·σ²`. Corpus growth from 150 → 180 rows did not move `c` enough to matter (was 18.00 on 2026-05-11 fit; now 16.60). There is no plausible refit-only path to σ/bw ≤ 2.0.

#### Structural blocker

High intrinsic forecast uncertainty for Denver overnight lows — likely driven by Front Range downslope events, drainage flows, and the surface inversion regime. GFS / ECMWF ensemble spread under-represents this kind of regime-shift uncertainty, which inflates `c` at fit-time. A Beta-calibration lane (the unpublished tracker) would side-step EMOS Gaussianity but isn't built.

#### Re-evaluation conditions

- **Earliest verdict-(A) re-evaluation:** 2026-07-07 (45 days post bracket-fix, on live data) — or **today** by re-running the Phase-4 replay with a 2026-04-07 cutoff to get a ≥45-day hold-out window. Recommended action: run the extended replay and check BSS vs per-cell base rate.
- **Verdict-(B) re-evaluation:** Wait for the Beta-calibration lane to ship. Until then, the σ-veto stands regardless of corpus size.

#### Shipping checklist if unblocked (hypothetical, do not execute today)

1. Remove `DEN_low` from `python/config.yaml:strategies.weather.excluded_city_vars`.
2. Verify `calibration_type: isotonic` and `isotonic_path: data/calibration/isotonic_v2.json` are in effect (already are).
3. 7-day shadow validation under the existing pilot knobs — verify n≥3 trades per (DEN/low, strike_type) accumulate, no rejection-reason spike (especially `EMOS_SIGMA_INCOMPATIBLE`, `BLOCKED_PLATEAU_RANGE`, `MIN_CALIBRATED_PROB`).
4. Tripwires to watch: `daily_trade_report.py --json | jq '.calibration.per_cell.DEN_low'`, per-cell brier in `state/calibration_monitor.json`, post-flip cumulative single-trade loss above a tightened limit on DEN/low (tighter than the global per-trade trigger because n is small).
5. Revert plan: re-add `DEN_low` to `excluded_city_vars`, restart. Named owner: christurgeon.

---

### 3.2 DEN/high

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 8.45 / 2.0 = **4.23** | ❌ FAIL hard | Worst σ-ratio among excluded cells alongside CHI/high. |
| EMOS `d` coefficient | 1.62 with `unstable_d` (CI [0.0, 5.16] from 2026-05-11 diagnostics) | n/a | Wide CI on `d`; `c=30.37` is the dominant variance contributor. |
| EMOS `ci_warnings` | `unstable_d, incompatible_sigma` | n/a | |
| Phase-4 replay n on `isotonic_v2` | 18 | ❌ < 45 | |
| Phase-4 replay brier | **0.319** | very poor | Worst per-cell brier among all 7 excluded cells (and the 13 cells in the replay overall). |
| Phase-4 replay calibration | mean_model_prob 0.196, win_rate 0.444 | ❌ severe under-prediction | Model says 20% probability, reality is 44% — calibrator does not match this regime. |
| 2026-05-12 Stage B evidence | not present | n/a | |

#### Why not (A) today

Beyond the n<45 gate, the replay shows isotonic_v2 systematically and severely under-predicting wins on this cell (+25pp gap between win_rate and model_prob). This is a calibration *failure*, not a data shortage. Under post-revert canonical knobs the cell would lose money faster because Kelly sizing inflates with edge-from-under-prediction.

#### Why not (B) today

σ/bw = 4.23, more than 2× the threshold. The `c=30.37` floor (highest among all cells) is the structural problem: Denver high-temp forecast uncertainty has a large intrinsic floor that the Gaussian-EMOS variance model cannot accommodate while staying bracket-compatible at 2°F bracket width.

#### Structural blocker

Orographic / continental-air-mass regime variance. Rocky Mountain front-range high-temperature forecasts have a long historical tail of "GFS busts on chinook event / cold-front passage" that is captured at fit-time as a high `c`. Open-Meteo GFS ingestion lag for inland Pacific / Rocky Mountain cells is an additional contributor on shorter-lead trades, but `c=30` is a fundamentally physical statement about the cell, not an ingestion problem.

#### Re-evaluation conditions

- **Verdict-(A):** Even with n=45, the +25pp under-prediction visible in the n=18 replay is unlikely to vanish — this looks like a calibrator-failure cell, not a sample-noise cell. A separate root-cause investigation into why isotonic_v2 under-predicts DEN/high wins is the prerequisite before any unblock attempt.
- **Verdict-(B):** Beta-calibration lane only. Even there, the high `c` may translate to a wide-tailed Beta posterior; unclear if it would resolve the bracket-incompatibility geometrically.

#### Shipping checklist

Same template as DEN/low if ever unblocked — but **do not unblock without first resolving the under-prediction root cause**. This is the cell most likely to bleed live P&L on a premature re-enable.

---

### 3.3 SEA/high

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 5.67 / 2.0 = **2.84** | ❌ FAIL | |
| EMOS `d` coefficient | 1.31 with `unstable_d` (CI [0.0, 4.50] from 2026-05-11 diagnostics) | n/a | Wide CI; `c=15.75` is the variance driver. |
| EMOS `ci_warnings` | `unstable_d, incompatible_sigma` | n/a | |
| Phase-4 replay n | 18 | ❌ < 45 | |
| Phase-4 replay brier | 0.269 | poor | |
| Phase-4 replay calibration | mean_model_prob 0.194, win_rate 0.389 | ❌ severe under-prediction | +20pp gap (model says 19%, reality 39%). Same failure mode as DEN/high, slightly less severe. |
| 2026-05-12 Stage B evidence | not present | n/a | |

#### Why not (A)

n<45, plus the same under-prediction failure mode as DEN/high (+20pp gap). The Puget Sound marine boundary layer regime (low cloud, marine push, post-frontal cold-air advection) is poorly represented in the ensemble — fit-time uncertainty floor is high (`c=15.75`).

#### Why not (B)

σ/bw = 2.84. Same `c`-dominated structure as DEN/high.

#### Structural blocker

Marine boundary layer / coastal influence on SEA-Tacoma high-temp forecasts. This is well-known in operational meteorology (NWS Seattle uses ensemble blending with bias correction precisely because of this regime). The Open-Meteo GFS-only feed isn't capturing the regime properly, and ECMWF blending at the current 0.55/0.45 weight does not resolve it (replay was run with both models active).

#### Re-evaluation conditions

- **(A):** Same as DEN/high — need to understand why isotonic_v2 under-predicts before re-enabling.
- **(B):** Beta-lane.

#### Shipping checklist

Do not unblock without the under-prediction root cause resolved. Same template as DEN/low.

---

### 3.4 LAX/high

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 5.52 / 2.0 = **2.76** | ❌ FAIL | |
| EMOS `d` coefficient | 7.11 with `unstable_d` (CI [3.06, 11.12] from 2026-05-11 diagnostics) | n/a | **High and well-identified `d`** — failure mode differs from DEN/SEA. Here the variance term `d·σ²` dominates `c=0.69`. Tiny `b=0.53` indicates strong shrinkage of the GFS mean. |
| EMOS `ci_warnings` | `unstable_d, incompatible_sigma` | n/a | |
| Phase-4 replay n | 23 | ❌ < 45 | Largest n among excluded cells (since LAX/high had been live until 2026-05-18 rollback). |
| Phase-4 replay brier | 0.170 | OK | |
| Phase-4 replay calibration | mean_model_prob 0.205, win_rate 0.217 | ✓ tight | Calibration looks fine in absolute Brier terms. |
| **Live evidence (`data/emos/overrides.json:LAX/high`):** | EMOS ROI=-39.6% over 11 trades / 7 days; live BSS_emos_vs_iso=-4.0% (n=36 signals); loss/win wager ratio 2.41× | ❌❌ | The σ-incompatibility translated into bracket-misaligned EMOS trade selection, not into bad Brier per se. |
| 2026-05-12 Stage A evidence | not present (not in Stage B because `unstable_d`) | n/a | Promoted under documented override 2026-05-11; rolled back 2026-05-18. |

#### Why not (A)

n<45, AND the replay's good Brier (0.170) on calibrated mean_prob (0.205) is the calibrator-quality story; the live failure was a *trade-selection* story (σ-driven over-sizing of bracket-incompatible trades). Re-enabling isotonic_v2 on LAX/high would side-step that specific σ pathology, but the under-prediction patterns visible in other cells (DEN/high, SEA/high, AUS/high) are not present here, so isotonic_v2 unblock might be defensible *if* the n=45 gate clears AND a per-cell ROI check on extended replay > -10%. Worth queuing for the 2026-07-07 evaluable date.

#### Why not (B)

σ/bw = 2.76. Failure mode well-documented in `data/emos/overrides.json` rollback: `d=7.19` produces EMOS σ=6.8°F at `ens_std=2.5°F`, which is incompatible with 1°F brackets (pre-fix) or 2°F brackets (post-fix). The unpublished tracker entry suggests "cap `d` at `cell_ens_std_p99 / bracket_width`" as a structural fix — that's out of scope for this pass.

#### Structural blocker

Strong spread-skill relationship at this cell (`d=7.11` with tight CI) is informative — when GFS spread is wide, the actual outcome variance IS that high — but the resulting σ violates the bracket geometry. EMOS-Gaussian with bounded `d` is the named fix; not in scope.

#### Re-evaluation conditions

- **(A):** 2026-07-07 for n=45 live OR run extended-replay today. **This is the most-likely-to-unblock-on-isotonic cell among the seven**, given good Brier on the existing 23-row replay window. Tempered by: the prior live promotion under EMOS lost money within 7 days, so the "small n" caveat applies in both directions.
- **(B):** Conditional on bounded-`d` EMOS variant or Beta-lane.

#### Shipping checklist (if (A) unblock is attempted ~2026-07-07)

1. Re-run Phase-4 replay extended to 45-day hold-out window (cutoff 2026-04-07).
2. Per-cell BSS vs the LAX/high climatology base rate (computed on `(LAX, high, strike_type, bracket_width_bin)` over the same window) must have 95% bootstrap CI lower bound > 0.
3. Per-cell ROI on the extended replay > -10%.
4. If all gates clear: remove `LAX_high` from `excluded_city_vars`. Set up a tighter rollback trigger: per-trade loss above a tightened limit (vs the global trigger) for the first 5 LAX/high trades.
5. Revert plan: 1-line config flip back to exclusion. Named owner: christurgeon.

---

### 3.5 AUS/high

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 6.18 / 2.0 = **3.09** | ❌ FAIL | |
| EMOS `d` coefficient | 0.78, **no warnings** (CI [0.0, 1.60] from 2026-05-11 diagnostics) | n/a | Clean `d` fit, `c=11.45` dominates σ. |
| EMOS `ci_warnings` | `incompatible_sigma` only | n/a | Cleanest warning profile of any excluded cell. |
| Phase-4 replay n | 14 | ❌ < 45 | |
| Phase-4 replay brier | 0.254 | poor | |
| Phase-4 replay calibration | mean_model_prob 0.183, win_rate 0.357 | ❌ under-prediction | +17pp gap. Similar failure mode to DEN/high, SEA/high. |
| 2026-05-12 Stage B evidence | -0.331 [-0.360, -0.306] vs isotonic-2026-05-12, n=6724 | ❌ | EMOS lost vs isotonic significantly on the pre-bracket-fix corpus. |
| **the unpublished tracker** | "AUS_high recent bias +3.51°F warm, SNR 2.39, BSS_emos_vs_iso = -33.1% (CI [-36.0%, -30.6%])" — hypotheses (regime, lead-time, ECMWF behavior) untested | ❌ | Tech-debt §269 / §382 unresolved. |

#### Why not (A)

n<45, +17pp under-prediction in replay, AND existing unresolved investigation in the unpublished tracker for regime / lead-time / ECMWF hypotheses on the isotonic baseline. The pre-emptive pilot exclusion (2026-05-22) is appropriate for at least the pilot window.

#### Why not (B)

σ/bw = 3.09. Note however that AUS/high has the cleanest EMOS fit profile of any excluded cell (no `unstable_d`, no `d_at_zero`, no `unstable_b`). The σ-incompatibility comes from `c=11.45` driven by Austin's continental high-amplitude diurnal cycle + occasional dryline-passage outliers. Fit is "honest" about the cell's uncertainty, but that honesty produces bracket-incompatible σ.

#### Structural blocker

High `c` from regime variance (cool-front passages, dryline events). Beta-lane or bounded-`d` EMOS would help.

#### Re-evaluation conditions

- **(A):** 2026-06-05 per pilot plan (pre-emptive exclusion revisit date). Recommended sequence:
  1. Resolve the unpublished tracker regime / lead-time hypotheses on the isotonic baseline.
  2. Then re-run Phase-4 replay extended to n≥45 for AUS/high specifically.
  3. Verdict-(A) gate evaluation.
- **(B):** Beta-lane only.

#### Shipping checklist

Standard. Do not unblock without resolving the under-prediction failure mode and the existing AUS/high investigation backlog.

---

### 3.6 CHI/high

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 9.36 / 2.0 = **4.68** | ❌ FAIL hard | Worst σ-ratio of any cell in the corpus. |
| EMOS `d` coefficient | 3.22 with `unstable_d` (CI [1.13, 6.23] from 2026-05-11 diagnostics — strictly positive, well-identified within the wide interval) | n/a | `d·σ²` is the dominant variance contributor; `c=14.46` also large. |
| EMOS `ci_warnings` | `unstable_d, incompatible_sigma` | n/a | |
| Phase-4 replay n | 17 | ❌ < 45 | |
| Phase-4 replay brier | **0.105** | ✓ very good | Surprisingly low — best brier among excluded cells. |
| Phase-4 replay calibration | mean_model_prob 0.185, win_rate **0.118** | ✓ tight (over-prediction by 7pp) | Model says 19% win, reality 12% — slight over-prediction but the model and reality are aligned in *direction* (both are low-base-rate). |
| 2026-05-12 Stage B evidence | -0.384 [-0.411, -0.357] vs isotonic-2026-05-12 | ❌ | EMOS-on-old-isotonic loses badly. |

#### Why not (A) — the most subtle case

n<45, but **CHI/high is the most "calibration looks fine, base rate kills it" cell on the replay**. The model is well-calibrated to a ~12% win rate; isotonic_v2 doesn't break this; the cell's economics are simply unattractive at that base rate with the existing fee/edge structure. The pre-emptive pilot exclusion comment ("0 wins / 16-17 trades both arms in 28d holdout") is slightly inaccurate (2 wins in isotonic_v2 arm) but captures the right intuition: this cell trades poorly because the win rate is low, not because the calibrator is broken.

If the goal is to unblock a cell that won't *blow up* (low downside), CHI/high is the safest of the 7 — but it will also unlikely make money. Both pilot and post-revert knob sets share this conclusion.

#### Why not (B)

σ/bw = 4.68. The post-2026-05-21 incident (the unpublished tracker) describes how CHI/low specifically got an EMOS_SIGMA_INCOMPATIBLE clearance when bracket_width changed 1.0 → 2.0, then immediately fired a far-out-of-range trade on `KXLOWTCHI-26MAY22-B46.5`. That incident motivated the σ-gate hardening to non-overridable. CHI/high carries the same risk pattern at higher σ-ratio.

#### Structural blocker

High-`c` Chicago-area variance (Lake Michigan effects, lake-breeze front, fall/winter advection patterns). The size of `d=3.22` also tells us the spread-skill relationship is informative (GFS spread carries real signal), but that signal compounds the bracket-incompatibility rather than helping it.

#### Re-evaluation conditions

- **(A):** 2026-06-05 per pilot plan. Recommended action: extend replay to 45 days, check per-cell BSS vs CHI/high base rate. Even if BSS LB > 0, ROI is likely marginal due to low base rate.
- **(B):** Beta-lane.

#### Shipping checklist

If (A) unblock is attempted: standard, but expect low trade volume (cell trades less than once per scan day per the replay). Set a tighter cumulative-loss rollback trigger (e.g., a fixed dollar loss over 5 trades) since the upside is limited.

---

### 3.7 CHI/low

**Verdict: (C) KEEP DISABLED.**

#### Evidence

| Gate | Value | Pass? | Note |
|---|---|---|---|
| EMOS σ_emos@p99 / bw=2.0 | 6.73 / 2.0 = **3.37** | ❌ FAIL | |
| EMOS `d` coefficient | 2.49 with `unstable_d` on fresh fit; **was clean (no warnings)** on 2026-05-11 fit (CI [0.41, 4.34]) | n/a | Coefficient stability deteriorated with corpus growth — note the *opposite* of what you'd hope from more data. |
| EMOS `ci_warnings` | `unstable_d, incompatible_sigma` (fresh refit; 2026-05-11 fit had no warnings) | n/a | This is the cell to investigate if you want to understand "did `unstable_d` get worse with more data". |
| Phase-4 replay n | 12 | ❌ < 45 | |
| Phase-4 replay brier | 0.219 | mediocre | |
| Phase-4 replay calibration | mean_model_prob 0.216, win_rate 0.333 | ❌ under-prediction (+12pp) | Similar pattern to AUS/high, less severe. |
| 2026-05-12 Stage B evidence | **+0.158 [+0.130, +0.183] vs isotonic-2026-05-12, n=5686, no warnings**, STAGE_B_ELIGIBLE | ✓ at the time | Cell was promoted to EMOS on 2026-05-18 on this evidence. |
| **Demotion history** | Promoted to EMOS 2026-05-18 with no warnings (BSS=+15.8% on Stage B); demoted 2026-05-21 after the bracket-window-fix-day incident (model μ for CHI/low ~6°F below underlying ensemble; far-out-of-range YES on `KXLOWTCHI-26MAY22-B46.5` at $0.02). Then σ-veto hardening made re-promotion blocked. | ❌ | Most-recently-demoted excluded cell. |

#### Why not (A)

n<45, AND the cell has been excluded since 2026-05-21 with no live isotonic_v2 evidence yet (pre-pilot it was on EMOS; post-pilot it's excluded). The replay's n=12 is the only post-fix data we have, and it shows +12pp under-prediction.

#### Why not (B)

σ/bw = 3.37. Promotion blocked by the post-2026-05-21 σ-veto hardening. Also, the fresh refit reveals `unstable_d` that wasn't present on the 2026-05-11 fit — corpus growth made the `d` confidence interval *wider*, which is the opposite of typical behavior and worth investigating. (Likely root cause: a few high-spread snapshot days in the 2026-05-11 → 2026-05-22 window dominate the bootstrap for this cell.)

#### Structural blocker

Two distinct things going on:
1. σ-incompatibility from `c=10.07` + `d=2.49` at typical `ens_std=3.76°F`. Beta-lane or bounded-`d` would help.
2. The 2026-05-21 incident exposed a μ-calibration failure (EMOS μ ~6°F below underlying ensemble) — this is the same class of failure that took down LAX/high, but driven by a different mechanism. Worth a dedicated audit before any EMOS re-promotion.

#### Re-evaluation conditions

- **(A):** 2026-06-05 per pilot plan, OR extend the replay today. Recommended: run extended replay AND audit the n=12 trades for the under-prediction direction (is the model systematically over-confident on NO?).
- **(B):** Multiple structural blockers (σ-veto + μ-calibration failure + new `unstable_d`). Not viable until both issues addressed.

#### Shipping checklist

If (A) is attempted: standard plus an explicit check that no `EMOS_SIGMA_INCOMPATIBLE` rows appear in the post-flip eval_audit (which would suggest the σ-veto is somehow being bypassed; not expected, but the 2026-05-21 incident was about exactly this).

---

## 4. Structural failure modes — inductively derived from §3

After working through the per-cell evidence, the 7 excluded cells cluster into three structural buckets. This is the **inductive** answer to "why isn't EMOS the universal fix?"

### 4.0 The σ-veto is a project choice — and these are the unlock paths

Before diving into the buckets: the unifying blocker is the `incompatible_sigma` veto (`python/config.yaml:emos_runtime_sigma_threshold: 2.0`), which rejects EMOS trades where `σ_emos(p99) / bracket_width > 2.0`. This is a guardrail this codebase chose, not a property of the world. EMOS-the-framework works fine for these cells — Gneiting & Raftery (2005), Hagedorn (2012), and standard NWS post-processing all use Gaussian-EMOS for daily T_max/T_min in regimes structurally similar to ours.

The veto exists because Gaussian-EMOS with the fitted (a, b, c, d) produces a distribution **wider than the 2°F bracket geometry** for these cells. The modal bracket gets ~30-40% probability against a market seeing ~15-20%, so Kelly-capped sizing piles into the "right modal bracket" — but the win rate still tracks the cell's underlying base rate, not the model's modal mass. The LAX/high live failure (`data/emos/overrides.json:LAX/high`) demonstrated this is real, not theoretical.

**Five named paths that would make EMOS workable for these cells:**

1. **Bounded-`d` EMOS variant** — at fit time, cap `d` at `cell_ens_std_p99 / bracket_width` (or similar) so σ stays inside bracket geometry by construction. Named in the unpublished tracker. Sacrifices some spread-skill information for tractability.
2. **Beta-calibration lane** — drop the Gaussian assumption; use a heavier-tailed parametric calibrator whose mapping to bracket probabilities is decoupled from the σ-bw ratio. Named in the unpublished tracker 3-lane hybrid.
3. **Sizing-by-σ Kelly adjustment** — accept wide-σ EMOS but scale Kelly by `1 / (σ/bw)` so wide-σ cells bet smaller, not bigger. The σ-veto becomes a continuous discount rather than a binary reject.
4. **Regime-conditional EMOS** — separate fits for known-regime subsets (chinook days, cold-front passages, lake-effect days, marine-boundary-layer days). Covered in Hagedorn 2012 §4. Permitted under the project's model-strategy constraints because it's a regime split of the existing GFS+ECMWF ensemble, not a new NWP model.
5. **Composite-bracket aggregation** — bet "between B−2 and B+2" composites instead of single 2°F brackets, so wide-σ EMOS distributions map to wider effective bets where modal probabilities are well-defined. Changes the *product* offered, not the model.

All five are explicitly **out of scope for this read-only pass**. They are listed here so the per-bucket "Improvements" lines below have a single referent rather than each section restating the same point.

### 4.1 Bucket 1 — High-`c` intrinsic-uncertainty cells (3 cells: DEN/high, SEA/high, DEN/low)

These cells have a large EMOS `c` coefficient (16–30), meaning **a large fit-time variance floor** independent of ensemble spread. The Gaussian-EMOS variance model `σ² = c + d·ens_std²` cannot get under the bracket threshold even at `ens_std → 0` because √c alone exceeds the bracket width.

| Cell | √c (°F) | bw (°F) | √c / bw |
|---|---:|---:|---:|
| DEN/high | 5.51 | 2.0 | **2.76** ❌ |
| SEA/high | 3.97 | 2.0 | **1.98** ❌ (just barely passes the strict gate but fails when `d·σ²` adds) |
| DEN/low | 4.07 | 2.0 | **2.04** ❌ |

**Evidence trail:** `data/emos/coefficients.json` and `/tmp/emos_scratch/coefficients.json` `c` values; `python/config.yaml:62-66` exclusion comments identify these as orographic / marine-boundary-layer cells with the σ-veto rationale.

**Why GFS spread is "uninformative" for these cells:** the variance floor is captured by `c`, not by `d`. The model is correctly representing the cells' irreducible uncertainty; the bracket geometry just doesn't tolerate it.

**Best-fit unlock paths from §4.0:** (2) Beta-calibration lane is the cleanest — drops the Gaussian assumption that's the actual culprit (a wide-σ Gaussian over a 2°F bracket has too much mass in the modal bracket). (4) Regime-conditional EMOS could halve `c` if downslope / chinook events were fit separately, but requires regime labeling that's not on disk. (5) Composite-bracket aggregation works if Kalshi offers wider brackets at these cells — pure product question, not a model question.

### 4.2 Bucket 2 — High-`d` informed-spread cells (3–4 cells: LAX/high, CHI/high, CHI/low, AUS/low)

These cells have a non-trivial `c` AND a large `d` (2–7). The variance term `d·ens_std²` dominates σ at typical spreads. The fit is "honest" — when GFS spread widens, actual variance widens proportionally — but the proportionality constant lands EMOS σ above the bracket width.

| Cell | √(c + d·p99²) (°F) | bw (°F) | σ/bw |
|---|---:|---:|---:|
| LAX/high | 5.52 | 2.0 | **2.76** ❌ |
| CHI/high | 9.36 | 2.0 | **4.68** ❌ |
| CHI/low | 6.73 | 2.0 | **3.37** ❌ |
| AUS/low (currently trades on isotonic_v2, not excluded) | 6.93 | 2.0 | **3.46** ❌ |

**Evidence trail:** `coefficients.json` `d` values, `data/emos/overrides.json:LAX/high.rollback_reason` for the documented live-failure precedent at this bucket.

**Why GFS spread *is* informative here, but doesn't help:** `d > 0` with tight CI means the spread-skill relationship is real. The Gaussian-EMOS framework can't exploit it without producing bracket-incompatible σ.

**Best-fit unlock paths from §4.0:** (1) Bounded-`d` EMOS variant is the most targeted fix — these cells fit a real spread-skill relationship; capping `d` mechanically preserves the signal that matters (which way `σ` moves with spread) while losing only the magnitude. (3) Sizing-by-σ is the soft alternative — keep the wide-σ EMOS, bet smaller on wide-σ cells. Both leave the GFS-spread information accessible, where the Beta-lane (2) discards it. LAX/high's unpublished tracker entry names (1) as the preferred path for this bucket.

### 4.3 Bucket 3 — Mixed-mode cells with under-prediction failure on isotonic_v2 (4 cells: AUS/high, SEA/high, DEN/high, CHI/low)

Independent of EMOS structural issues, the Phase-4 replay reveals that **isotonic_v2 systematically under-predicts wins** on a subset of cells:

| Cell | replay n | isotonic_v2 brier | mean_model_prob | win_rate | gap (outcome − model) |
|---|---:|---:|---:|---:|---:|
| AUS/high | 14 | 0.254 | 0.183 | 0.357 | +0.17 |
| SEA/high | 18 | 0.269 | 0.194 | 0.389 | +0.20 |
| DEN/high | 18 | 0.319 | 0.196 | 0.444 | +0.25 |
| CHI/low | 12 | 0.219 | 0.216 | 0.333 | +0.12 |

**Evidence trail:** `data/research/2026-05-22-isotonic-v2-replay.json:per_cell.isotonic_v2_refit`.

**Why this matters:** Verdict (A) for these cells would not be unblocked even if the n=45 gate cleared — the model is mis-calibrated *toward NO* on these cells, which means YES bets are under-priced and NO bets are over-confident-and-wrong-half-the-time. The bounded-exposure pilot knobs limit the bleed but don't address the underlying calibration failure.

This is a different failure mode from bucket 1/2. It's not σ-veto, it's the **isotonic_v2 calibrator squashing predictions into a [0.17, 0.22] band** that's well below the actual base rate for these cells in the post-fix window. Likely root cause: the isotonic_v2 fit corpus had cell-specific base rates close to the global ~17%, but the post-fix hold-out window happens to have higher base rates for these cells — so the calibrator's monotonic mapping over-shrinks predictions toward the global anchor.

**Improvements (out of scope):** Per-cell isotonic fits (instead of global), OR a Bayesian re-calibration that uses cell-specific priors. Both are explicitly out of scope per the project's model-strategy constraints and the hard "no new isotonic refits" constraint in this pass.

### 4.4 Pacific intraday-lag bucket (cross-references an unpublished note)

An unpublished note documents a 3-6h Open-Meteo GFS ingestion lag affecting DEN/high and SEA/high specifically. This is an additional contributor to the high `c` in those cells on shorter-lead trades (lead=0). It does not explain the full σ-veto problem (which holds across all 3 leads pooled in the EMOS fit) but it is a "known data-quality issue" that would partially improve cells in buckets 1 and 3 if the upstream lag were resolved. Documented separately; explicitly out of scope here.

### 4.5 Failure modes *checked and rejected*

Candidate failure modes that the evidence does NOT support:
- **"EMOS d=0 group"** (the former `config.yaml:62` label): only DEN/low has literal `d=0` on the 2026-05-11 fit, and that cleared on the fresh refit (`d=0.70`). The label was wrong about the mechanism — corrected in `python/config.yaml` to reference the σ-veto.
- **Post-bracket-fix corpus shift causing EMOS instability**: the fresh refit on the rebuilt corpus (n_train ≈ 175 per cell, was 145) shows `unstable_d` appearing for CHI/low where it wasn't before, but for every other cell the stability profile is unchanged. Not a uniform corpus-shift story.
- **Corpus contamination**: the corpus rebuild produced 2466 settled rows from the snapshot tree, all of which pass the existing dedup + settlement-merge checks; no contamination signal.
- **"Bracket-fix doubled the window therefore EMOS now passes"**: the σ-veto threshold of 2.0 is a ratio to `bracket_width`. Pre-fix at bw=1.0, the gate caught σ_emos > 2.0. Post-fix at bw=2.0, it catches σ_emos > 4.0. For the 7 excluded cells, σ_emos values are in [4.24, 9.36] — only DEN/low at σ_emos=5.60 lands inside [4.0, x] depending on where x is, but **5.60 / 2.0 = 2.80 still fails**. The bracket fix did NOT structurally unblock anything; it just shifted the threshold.

---

## 5. Open questions / out-of-scope improvements

Improvements I noticed but did NOT pursue, organized by who would tackle them:

### 5.1 Calibration / modeling
- **Bounded-`d` EMOS variant** — cap `d` at `cell_ens_std_p99 / bracket_width` so σ stays bracket-compatible by construction. Named in the unpublished tracker. Would unblock bucket-2 cells.
- **Beta-calibration lane** — referenced in the unpublished tracker 3-lane hybrid. Would address bucket-1 cells.
- **Per-cell isotonic fits** — would address the bucket-3 under-prediction failure mode visible on the replay. Out of scope per the hard constraints; also potentially over-parameterized at current per-cell n.

### 5.2 Data pipeline
- **Open-Meteo GFS ingestion lag fix** — would partially improve DEN/high and SEA/high on short-lead trades. Already in flight elsewhere.

### 5.3 Validation methodology

**Lead recommendation: extend the Phase-4 replay cutoff to 2026-04-07 today.** This is the single most valuable follow-up. Re-run `scripts/replay_isotonic_holdout.py` (or whatever produced `data/research/2026-05-22-isotonic-v2-replay.json`) with `--cutoff 2026-04-07` — snapshots are on disk back to 2026-03-18 — to get a 45-day post-fix-math hold-out window. ~5 min of compute. Unblocks verdict-(A) gate evaluation **today** for every cell whose blocker is sample size rather than structural calibration failure.

Why this beats the "wait 45d for live data" alternative:
- **Live volume is starved.** Bounded-exposure pilot has `kelly_fraction=0.075`, `max_total_exposure_pct=0.30`, AND a current 24.6% drawdown that's multiplicatively halving sizes via `drawdown_size_reduction=0.50`. Per the pilot doc, expected total volume over 14 days is ~20-30 trades across 4 cells (~5-8/cell). Linear-scale to 45 days: ~15-25/cell — same n as the extended replay would give.
- **Live and replay test different things.** Live data is uniquely valuable for Stage-2 failure modes (orderbook depth, slippage, partial fills) that the replay cannot model. But none of the 7 excluded cells' blockers are Stage-2 — they're either σ-veto (structural) or under-prediction (calibration). Live data cannot diagnose either of those faster than replay can.
- **Replay tests a different season.** April-May snapshots capture warming-season regimes. Live data through 2026-07-07 captures late spring → early summer. Both are valuable; the cheap one should run first.

Cells where the existing 28-day replay already gives a clear directional verdict and 45d wouldn't change it: DEN/high (+25pp under-pred, severe), SEA/high (+20pp under-pred, severe), AUS/high (+17pp under-pred). These are **calibrator-failure** cells, not sample-size cells. Cells where extending to 45d could plausibly flip the verdict: DEN/low (n=14 → ~22, calibration looks acceptable), LAX/high (n=23 → ~37, isotonic_v2 Brier was good), CHI/high (n=17 → ~27, calibrated but low base rate).

Companion follow-ups (out of scope today):
- **Per-cell LogSkill calculation on the extended replay** — the unpublished tracker shipped 2026-05-23 adds LogSkill to EMOS promotion gates; can be applied retroactively to the isotonic_v2 replay for verdict-(A) gate parity with the EMOS path.
- **Per-cell climatology base rate computation** — required for verdict-(A) BSS comparison vs the per-cell base rate (not vs global). Would need a GHCN or NWS-history fetch + binning by `(city, var, strike_type, bracket_width_bin)`. Without it, the BSS computation defaults to global base rate, which understates the calibrator's lift.

### 5.4 Operational
- **2026-05-21 CHI/low μ-calibration failure audit** — the bracket-fix-day incident exposed an EMOS μ ~6°F below the underlying ensemble for CHI/low. Root-cause not identified in the unpublished tracker. Worth a follow-up post-mortem before any CHI/* re-promotion.
- **Why did `unstable_d` get worse for CHI/low with corpus growth?** Comparing the 2026-05-11 fit (no warnings) to the 2026-05-22 refit (`unstable_d`) shows CI widening despite more data. Worth investigating for fit-stability reasons.

---

## 6. Process notes (for future-me)

- I deliberately did NOT dispatch 7 parallel calibration-analyst agents. The cells share too much structural context (same n_settled_days gate, same σ-veto threshold, same isotonic_v2 calibrator, same Phase-4 replay window) that 7 independent analyses would have duplicated more than 80% of their reads against the same handful of artifacts. The right unit of independence here is "bucket" (3 buckets in §4), not "cell". A coherent single-author synthesis dominates parallel dispatch for this shape of work — per `superpowers:dispatching-parallel-agents` "Don't use when: failures are related (fix one might fix others)".
- I DID do a read-only EMOS corpus rebuild + refit to `/tmp/emos_scratch/` per the user spec for verdict-(B) verification. The result confirmed the σ-veto holds across corpus growth — no excluded cell flipped its verdict. Production `data/emos/coefficients.json` was NOT touched.
- The original draft framed verdict (B) as "EMOS is structurally blocked" without qualification. **That was wrong** — the blocker is the project's σ-veto, not EMOS the framework. The §4.0 section was added after a user pushback that correctly noted "quant firms absolutely use EMOS for short-bracket weather markets". The five named unlock paths in §4.0 are the corrective. Future-me: do not conflate "this codebase's current guardrail rejects this" with "this is impossible".
- The original draft also under-sold the extended-replay path, treating "wait 45 days for live data" as the default earliest-evaluable date. **That was also wrong** — live data through 2026-07-07 is volume-starved by the bounded-exposure pilot and would yield comparable n to what the extended replay provides today, while being season-restricted. §5.3 now leads with the extended-replay recommendation.
- The most consequential single finding is §4.3 (bucket 3 under-prediction failure on isotonic_v2). This is a NEW failure mode not previously named in the project's calibration runbook; it deserves its own follow-up beyond the cell-unblock question this doc was scoped to.
