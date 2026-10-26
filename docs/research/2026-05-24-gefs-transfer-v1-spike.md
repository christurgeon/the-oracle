# V-1 spike: does a GEFS-trained calibrator transfer to Open-Meteo inputs?

**Status:** spec reviewed (spec-reviewer pass 2026-05-24) and reconciled — 3 Critical + 3 Important findings addressed: per-arm bias-correction transform, init-cycle pairing, power-gate-the-gate + equivalence reframe, a-priori (leakage-safe) gridpoint selection, contrast precondition (P-4), quantified aggregation gate (P-3). D+1/D+2 row availability empirically confirmed (~105 paired dates/cell; SEA/low thin). Ready to implement.
**Parent:** the unpublished feature spec (F-3a corrected, F-3b transfer question).
**Decides:** whether the full GEFS backfill to 2017 is worth building. This is the small bet that authorizes (or kills) the big bet.

> **Step-0 RESULT (2026-05-24): INCONCLUSIVE at n≈63 → do NOT run the full extraction now.** The Step-0 power check (computed from artifact single-arm CIs, no extraction; see Decision boundary) shows the paired transfer-loss floor (~0.06–0.09) exceeds the 0.05 INCONCLUSIVE threshold on essentially every cell. V-1 cannot resolve at the current overlap. **Revisit ~2026-08** as the OM overlap accrues (converges with the pooling timeline). Confirmed by calibration-analyst audit. The full extraction stays parked until then.

## Purpose

The per-cell calibrator harness needs ~5× more effective n than the current ~63-day corpus provides (synthetic test, harness investigation Branch 3). Raw NOAA GEFS is the only historical ensemble source and can deliver ~20× (3,000 days/cell to 2017). The harness verdict is methodologically sound on a GEFS corpus because candidates refit per fold (the unpublished feature spec F-3a, verified in code). The one unresolved risk is **transfer**: production runs the chosen calibrator against live **Open-Meteo** inputs, but a GEFS backfill would *train* it on **GEFS** inputs. If a GEFS-trained calibrator degrades materially when fed OM inputs, the entire backfill is worthless. V-1 measures that degradation directly, cheaply, on the existing ~63-day overlap where we have both sources.

## The production scenario being simulated

Deployment after a hypothetical full backfill: calibrator trained on historical GEFS fractions → applied to live OM fractions at decision time. V-1 reproduces exactly this on the overlap window and compares it to the status-quo (train-on-OM / apply-to-OM).

## Experiment design

### Data (no bracket synthesis needed — use real markets)

The overlap window (2026-03-19 → 2026-05-21) has **real Kalshi markets** in `data/snapshots/`, so V-1 uses the actual bracket grid and actual outcomes — no synthesis, no `BFILL-` tickers. **Row availability is verified** (2026-05-24): per cell there are ~52-56 distinct settled forecast_dates at D+1 and ~52-55 at D+2 (≈105 paired dates/cell pooling both leads), EXCEPT **SEA/low at 23/22 — flag as under-powered, report separately, do not fold into an aggregate verdict.**

For each (city, var, forecast_date, real market) at leads **D+1 and D+2 only** (D+0 excluded — Prereq P-2):

- **OM fraction:** ensemble bracket fraction from the snapshot's `gfs_seamless` members, computed via the **production OM transform stack** — i.e. WITH `bias_correction.json` applied (this is what production actually feeds the calibrator). GFS-only; do NOT blend ECMWF.
- **GEFS fraction:** the same bracket's fraction from raw NOAA GEFS members, with **NO bias correction applied** (see CRITICAL note below) and Prereqs P-1/P-3 applied.
- **Outcome:** realized YES/NO from the GHCN/NWS settlement (source-agnostic; shared).

This yields paired triples `(om_frac, gefs_frac, outcome)` per cell.

**CRITICAL — bias-correction transform must differ per arm (do NOT reuse `load_feature_rows` blindly).** The harness's GFS-stats path applies the production `bias_correction.json`, which is **fit to correct Open-Meteo `gfs_seamless` members**. Applying it to raw GEFS members would correct GEFS *toward* OM and artificially shrink the very seam V-1 measures → false PASS. The GEFS arm must replicate what a real GEFS backfill would feed the calibrator: raw GEFS members, no OM bias table (production has no GEFS-specific bias table). State the exact transform per arm in the implementation; do not inherit the harness default. (Walk producer→consumer per the `pipeline_implicit_conventions_audit` rule.)

**CRITICAL — init-cycle pairing.** The OM fraction must be paired against the GEFS run from the **same GFS init cycle** the OM members actually reflect. `load_snapshot_data` does last-writer-wins across the day's 4 captures, so a `forecast_date`'s OM members typically reflect the *latest* init captured (12z/18z), NOT 00z. Pairing that against a GEFS 00z run injects an init-time skill gap that loads onto "transfer loss" (manufactured FAIL, or a PASS if it cancels the warm seam). **Fix:** read each row's true init from snapshot provenance and extract the matching GEFS cycle; if provenance is unrecoverable per row, restrict V-1 to a single fixed capture (e.g. the 0200-UTC run) on the OM side AND pair with the GEFS init that capture actually used — accounting for GFS publication lag (the 00z GFS run isn't available until ~04-05z, so an 02:00-UTC capture likely reflects the prior 18z init). Confirm the init empirically before pairing; do not assume 00z.

### Primary metric — cross-fit transfer loss (paired, deployment-faithful)

For each cell, K-fold CV over overlap dates (reuse `emos_fit.cv_fold_iter` for parity), for each candidate C ∈ {iso_v2_global, emos} (identity is source-invariant, used only as the skill baseline inside LogSkill):

- **Reference (status quo):** fit C on **OM** fractions of the train fold → score on **OM** fractions of the val fold → `LogSkill_OM→OM`.
- **Transfer (GEFS-backfill scenario):** fit C on **GEFS** fractions of the train fold → score on **OM** fractions of the val fold → `LogSkill_GEFS→OM`.
- **Transfer loss(cell, C) = LogSkill_OM→OM − LogSkill_GEFS→OM.**

Because both arms score on the *same* OM val data, this is a **paired difference** — the per-date forecast noise partially cancels, giving materially lower variance than either absolute LogSkill (this is why cross-fit beats winner-rank at small n). Report transfer loss per (cell, candidate) with a **block-bootstrap CI on the paired difference** (blocks on forecast_date, mirroring the harness's Politis-Romano scheme).

### Secondary cross-check — winner-rank agreement

Fit all candidates on OM → OM-winner per cell. Fit all on GEFS → GEFS-winner per cell. **Qualitative footnote only** — underpowered at this n and risks over-reading on ~3 tradeable cells. Do NOT tabulate next to the primary verdict; mention as corroboration, never override.

## Prerequisites (MUST be applied inside V-1 extraction — not deferred)

These are confounds that would manufacture a *spurious* transfer failure (an extraction artifact masquerading as a real seam — the inverse of the F-3a misread). V-1 is invalid without them.

- **P-1 — A-PRIORI land/representativeness gridpoint selection (resolves V-3; leakage-safe).** For each city, do not use the naive nearest 0.25° gridpoint, and do **NOT** select the gridpoint by "best match to OM on the overlap" — that tunes the GEFS input toward OM using the same data the transfer test scores on, systematically understating transfer loss (soft false-PASS, worst where the seam is largest, e.g. CHI). Instead select **geographically/a-priori**: the nearest GEFS **land** cell to the GHCND station (land-mask + station proximity), frozen before any transfer metric is computed. Record the chosen (lat, lon) per city AND the residual GEFS-vs-OM mean seam — treat a large residual as *evidence*, not something to optimize away.
- **P-2 — HRRR-D+0 exclusion (resolves V-5).** Use leads **D+1 and D+2 only.** At D+0, OM's `gfs_seamless` blends HRRR/NBM (AUS D+0 −3.5°F signature) and diverges from raw GEFS for a non-seam reason. Excluding D+0 makes the OM scoring inputs HRRR-free and comparable. **Production-scope consequence** below.
- **P-3 — TMAX/TMIN aggregation parity (resolves V-4).** Define the exact GEFS forecast-hour window set per (city, lead) that covers the city's **local calendar day**, and take window-max (TMAX) / window-min (TMIN). **Quantified gate:** after P-1, the per-cell GEFS-vs-OM ensemble-mean delta on the overlap must fall within **±1.0°F** of the benign systematic seam (~+0.5°F) — i.e. residual in roughly [−0.5, +1.5]°F. A residual outside that band means the aggregation window is wrong; abort and re-derive the window before proceeding (do not eyeball it).
- **P-4 — Contrast precondition (guards against an uninformative false PASS).** If, after P-1/P-3, the OM and GEFS fractions are nearly identical (the benign-seam case), transfer loss is trivially ≈0 and the test says nothing about generalization. Before a PASS is meaningful, report the per-cell distribution of `|om_frac − gefs_frac|` and require non-trivial contrast: a minimum count of rows (suggested ≥ 20% of the cell's rows) where OM and GEFS land on **opposite sides of the bracket boundary** (i.e. disagree on the YES/NO the bracket implies). If contrast is below floor → verdict is **"inconclusive — overlap too benign,"** not PASS.

## Decision boundary (this is a gate, not a build trigger — and it may return INCONCLUSIVE)

**Step 0 — gate-the-gate (do this FIRST, before trusting any verdict).** Compute the actual achievable block-bootstrap CI half-width on the paired transfer-loss difference at the real per-cell n (~105 paired dates pooling D+1/D+2; fewer per fold). LogSkill SE ≈ 0.06 at n_val≈144; the paired difference is lower-variance (per-date outcome noise cancels — the harness already bootstraps per-row `ll_diff` blocked on forecast_date), but folds are small. **If the achievable CI half-width is ≳ 0.05, the gate below cannot distinguish PASS from FAIL and V-1 is INCONCLUSIVE at current n** — the honest action is to wait for the overlap to accrue (it grows +1 day/day; revisit when n is sufficient, likely converging with pooling's ~2026-08 window) OR reframe as a one-sided bound (below). Do not report a PASS whose CI can't support it.

> **Step-0 EVALUATED 2026-05-24 → INCONCLUSIVE → do not build now.** Computed from the artifact's single-arm LogSkill CIs (no GEFS extraction needed): median single-arm half-width ≈ 0.09 (3× the X=0.03 bound). A paired transfer-loss difference is lower-variance than a single arm but by at most ~√2, so the paired floor is ~0.06–0.09 — above the 0.05 INCONCLUSIVE threshold on essentially every tradeable cell. V-1 therefore cannot resolve at n≈63; INCONCLUSIVE defaults to "do not build." **Revisit when the OM overlap accrues toward the pooling ~2026-08 window.** Triangulated with the synthetic ~5×-more-data finding (independent leg) and confirmed by calibration-analyst audit 2026-05-24.
>
> An OM-self-split probe (`/tmp/om_self_split*.py`) was attempted as a richer estimate but **discarded**: (1) it measures fit-resampling sensitivity, a *different* variance component than V-1's source-difference val-variance, so it's a loose proxy with no clean bounding theorem; (2) the script seeded its fold-shuffle with Python's randomized `hash()`, making its specific numbers non-reproducible. The artifact-CI computation above is the authoritative Step-0 check and needs no script. (Aside: "fit-stability tracks certification" does NOT hold — LAX/high has the tightest CI yet isn't certified, because a tight CI around ~0 means consistent *null*, not a winner.)

**Reframe as an equivalence / magnitude-bound test (not a point test).** The decision question is "can we rule out transfer loss worse than X?" Pre-register X from the harness's real SE scale (suggested X = 0.03 LogSkill). 
- **PASS** iff, for the winning candidate on each non-SEA/low tradeable cell: the block-bootstrap CI **upper bound** on transfer loss is `< X` (we can rule out a damaging loss) AND P-4 contrast floor is met.
- **FAIL** iff any tradeable cell's transfer-loss CI **lower bound** is `> X` (a real damaging loss is established).
- **INCONCLUSIVE** otherwise (CI straddles X). **Inconclusive defaults to "do NOT build"** — a 3-week / hundreds-of-GB build requires positive evidence of acceptable transfer, not absence of evidence against it.

**PASS authorizes the full build; it does NOT commit to it.** A separate post-V-1 go/no-go weighs cost (re-cost below) and the P-2 production-scope limit. **FAIL** → abandon GEFS for the correct reason (transfer failure); rely on pooling + accrual. Do not re-attempt under a different framing.

### Production-scope limit to carry into the go/no-go (from P-2)

Raw GEFS has no HRRR, so a GEFS backfill can only soundly serve D+1/D+2. If the weather strategy's edge is concentrated at **D+0** (same-day), a GEFS-trained calibrator would be deployed against D+0 OM inputs (carrying HRRR signal) it never trained on — so GEFS backfill helps less than the raw day-count implies *even if V-1 passes*. **Action:** characterize the strategy's realized-PnL lead distribution (from `state/trades.jsonl` / attribution) before committing to the full build.

## Scope, cost, tooling

- Reuse the proven `/tmp/grib_test/.venv` (cfgrib + eccodes + boto3, bundled binaries).
- Extraction: ~63 days × 00z cycle × 31 members × D+1/D+2 window fhrs × 2 vars × 7 cities ≈ low-hundred-thousand byte-range requests; ~1-2 h wall-clock with `max_concurrent≈16`. One-time; transient files stream-and-delete.
- This is a research spike, not production code — a self-contained script under `scripts/` or `/tmp`, output a single JSON/markdown report. No runtime wiring, no harness changes.

## Deliverable

A report containing: per-(cell, candidate) transfer-loss point + bootstrap CI; winner-rank agreement table; chosen gridpoint per city (P-1); aggregation-window definition + validation residual (P-3); and an explicit PASS/FAIL against the decision boundary with a recommended post-V-1 go/no-go (including the lead-distribution characterization from the production-scope note).

## Re-costing the full build (only relevant if V-1 passes)

The v2 mechanical-backfill estimate (~22 GB transient, ~85 MB corpus, ~6 h) was for **197 days**. The full backfill to 2017 is ~3,000 days × 4 cycles ≈ **~15× larger** (hundreds of GB transient, much longer run, stream-and-delete mandatory). Re-cost properly at the go/no-go; do not quote the 197-day numbers for the full build.

## Relationship to other work

- **Gates:** the full GEFS backfill in the unpublished feature spec.
- **Complementary to (does NOT block, is NOT blocked by):** the hierarchical-pooling lever (see the unpublished tracker). Pooling proceeds on its own ~2026-08 corpus-growth timeline regardless of V-1's outcome. GEFS revival must not cannibalize pooling.
- **Independent of:** harness Commit B (shipped).
