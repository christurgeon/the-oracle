# Calibration Runbook

**Shipped config** (`python/config.yaml`, `strategies.weather`):
- **Calibrator:** isotonic (`calibration_type: isotonic`). Artifact:
  `isotonic_path: data/calibration/isotonic_v3.json` (fit 2026-07-02, re-anchored
  ERA5→NWS). It applies to every cell that is not in `excluded_city_vars`.
- **EMOS cells:** none (`emos_enabled_cells: []`). EMOS output is logged in shadow
  only. Coefficients: `data/emos/coefficients.json`. Overrides for warnings-bearing
  cells: `data/emos/overrides.json`.
- **Per-type Platt:** `platt_per_strike_type.json` is on disk but unused.

Per-cell selection is configured by `weather.emos_enabled_cells` in
`python/config.yaml`. Future EMOS
promotions follow the per-cell rule: `n≥10`, `BSS>0` with 95% CI lower
bound `>0`, `LogSkill>0` with 95% CI lower bound `>0` (Kelly-aligned
logarithmic scoring rule — proper, knob-invariant, catches tail
mis-calibration; added 2026-05-23), no `unstable_b`/`unstable_d`/
`d_at_zero` coefficient warnings, and no `incompatible_sigma` warning
(non-overridable structural check: `σ_emos(p99) / bracket_width > 2.0`
blocks promotion regardless of BSS or operator override — the LAX/high
failure mode this veto was added to catch). The older global "BSS ≥ 5%
on full corpus" gate is retired.

**History (author's deployment).** Per-type Platt Phase A shipped 2026-04-21.
Phase B (production rollout on the VPS) never happened; the events below
superseded it.
- **Isotonic.** On 2026-05-22 the live calibrator became isotonic_v2
  (`data/calibration/isotonic_v2.json`, see
  `docs/research/2026-05-22-isotonic-refit-bounded-exposure-pilot.md`; the
  bounded-exposure pilot was scrapped 2026-05-24 after no trades went through).
  isotonic_v3 replaced it on 2026-07-04.
- **EMOS** (`data/emos/coefficients.json`, refit 2026-05-11, artifact hash
  `3ca7b94d9d42…`, n=145 per cell) ran live per cell from the Stage A cutover
  on 2026-05-11. Original Stage A validation via
  `scripts/compare_emos_coefficients.py`: MIA/high BSS +18.85% on n=4447,
  NYC/low BSS +1.21% on n=4733 (the NYC/low pass did not generalize to live).
  **NYC/low and LAX/high rolled back 2026-05-18** (NYC/low: live
  BSS_emos_vs_iso=−6.6%, EMOS worse than isotonic and raw; LAX/high:
  ROI=−39.6% in 7 days, d=7.19 produces EMOS σ=6.8°F incompatible with
  1°F brackets — override revoked, cell re-excluded).
  **NYC/high + CHI/low demoted 2026-05-21** as part of the
  bracket-window-fix post-incident fix (bracket_width=2.0 unblocked the
  σ-gate and exposed mis-calibrated μ — see
  `docs/research/2026-05-22-isotonic-refit-bounded-exposure-pilot.md`).
  **MIA/high + MIA/low rolled back to iso_v2 2026-05-24** after the per-cell
  harness certified iso_v2_global as winner for both (LB +0.014, +0.031, 5/5
  folds); MIA/low live EMOS ROI was −37.2%. They had been promoted 2026-05-11
  and 2026-05-16 respectively. LAX/low, under an `unstable_d` override, was the
  last EMOS cell. It was suspended to isotonic on 2026-06-05.

Sections 0–5 describe the per-type Platt mechanics as originally designed.
They are a reference for the Platt artifact and the shadow-vs-live pattern,
**not** the shipped path.

---

## 0. Mental model in 30 seconds

- Live calibrator maps the **raw ensemble fraction** → a calibrated
  probability before the strategy's edge calculation. Three strike types
  (`greater` / `less` / `between`), three independent Platt fits.
- Shadow calibrator runs in parallel on the same raw input, output logged to
  `state/eval_audit.jsonl` (`shadow_prob` field) but **never** applied. Used
  for A/B analysis during Phase B.
- Audit log rotates daily at UTC midnight. Rotated siblings are pruned after
  `audit_log_retention_days` (default 14). Retention and rotation are
  backstopped by the tripwires in `python/analytics/audit_tripwires.py`.

---

## 1. Refit cadence and inputs

Refit triggers (any one is enough):

1. `fit_date` is more than 45 days old (tripwire fires a `WARNING`; 90 days
   escalates to `CRITICAL`).
2. Rolling 7d Brier is > 1.3× 30d baseline for more than 3 consecutive ticks
   (`calibration_monitor` `brier_7d_vs_30d` alert).
3. Per-type n has grown by ≥ 25% since the last fit (more data means a tighter
   MLE; especially relevant for `greater` / `less` where n was just above the
   `min_per_type=300` floor at first fit).

Input requirements:
- ≥ 300 pairs per strike type (hard floor in `_fit_per_type`).
- Both outcome classes present per type (fitter raises on single-class input).
- Spread inflation applied upstream (`use_spread_inflation=True`, default).

### 1.1 Input-contract preflight

**Run before every refit.** The preflight catches silent training/runtime
input-pipeline drift that calibration analyst review wouldn't see. Operator
workflow:

```bash
# Regenerate the data-availability calendar against the current snapshots tree
# and run the contract gate. Exits 0 if the corpus is clean, 1 if any abort
# condition trips. Does NOT mutate any calibrator artifact.
uv run python scripts/_calibration_dry_run.py --regenerate-calendar
```

What to inspect in the printed report:

- **`partial coverage` count must be 0** (post-cutover; pre-cutover triples
  bucket as `single-model era` and don't trip the gate). As of 2026-05-09
  the empirical post-cutover rate is 0.0% — any nonzero number here means
  a real coverage gap that needs investigation **before** using
  `--allow-partial-model-coverage` to override. Common causes: a partial
  Open-Meteo fetch, ECMWF endpoint outage on the snapshot date.
- **`empty (no fc rows)` count must be 0** post-cutover for the same reason.
  Pre-cutover, ~11 cells with settlements but no usable forecast row are
  expected (legacy data gap); these only matter if you're reproducing the
  2026-05-01 frozen artifact, in which case both `--allow-partial-model-coverage`
  AND `--allow-empty-cells` are required.
- **`single-model era` is fine post- to pre-cutover transition.** It's the
  bucket for triples where only one expected ensemble model was published as
  of the forecast date (i.e., dates before the second model's first-seen).
- **`Calendar first-seen dates` echo**: must match what the operator expects.
  If a new model has appeared in snapshots without an explicit human
  acknowledgment, the gate also logs a `WARN` for it — investigate before
  promoting the next refit.

**Calendar file:** `data/calibration/data_availability_calendar.json`. Source
of truth is auto-detect (`method: "auto"` field). Manual edits are supported
for back-dating an explicit cutover, but require setting `method: "manual"`
and explaining the override in the commit message — the gate logs a `WARN`
whenever it loads a `manual` calendar so the override stays visible.

**The `--no-bias-correction` flag is for ablation runs only.** Production
refits must run with bias correction. The artifact's `bias_correction_applied:
false` metadata field makes the override visible to a future runtime-loader
gate (follow-up FU-3). Do not promote a `bias_correction_applied: false`
artifact to live without explicit sign-off.

The follow-up FU-3 (runtime calibrator loader gating) closes the
2026-04-23 operator-mode-mismatch failure mode that this preflight does
**not** catch — implementation deferred.

---

## 2. Refit procedure

```bash
# 1. Refresh snapshots from the VPS if needed.
uv run python scripts/fetch_shadow_settlements.py --pull-vps

# 2. Run the fitter. 5-fold CV surfaces overfitting; spread inflation on.
uv run --extra ml python scripts/fit_platt.py --cross-validate

# 3. Inspect the two artifacts the fitter wrote.
#    - data/calibration/platt.json                (global fit)
#    - data/calibration/platt_per_strike_type.json (per-type fit — unused in shipped config)
```

What to eyeball before promoting:
- `per_type_warnings` is `[]`. A non-empty list means one of the types fell
  under `min_per_type` — investigate before shipping.
- `brier_calibrated` < `brier_raw` for every type. The `greater` type will
  always show the biggest drop (raw ~0.30 → cal ~0.16); if it doesn't, the
  fit is off.
- Coefficients look sane: `a` in [−4, +1], `b` in [0.1, 3.0]. A `b` near 0
  means the calibrator is collapsing to a constant — usually a data issue.

---

## 3. Promoting a new fit to production

1. Commit the new `platt_per_strike_type.json` on a branch. Don't overwrite the
   live file in-place on the VPS — we want a rollback target.
2. Merge to `main`. `deploy/update.sh` does `git pull` into `/opt/oracle` and
   restarts the systemd unit, so the new artifact lands at
   `/opt/oracle/data/calibration/platt_per_strike_type.json` as a side effect
   of the checkout. No separate copy step.
3. Watch `state/eval_audit.jsonl` for the first 30 minutes:
   - `calibrated_prob` should move relative to the previous run.
   - `shadow_prob` stays whatever the old shadow was configured to.
4. Verify `calibrator` in `/status` Telegram reports shows the new `fit_date`.
5. If a tripwire fires (`audit_log_stale`, `calibrator_artifact_stale`,
   `bin_deviation`) within 24h, initiate rollback (§5).

---

## 4. Running a shadow calibrator during Phase B

Use `shadow_calibrator` in `config.yaml` to run an alternative in parallel,
e.g., isotonic-per-strike-type alongside production Platt:

```yaml
strategies:
  weather:
    calibration_type: per_type_platt
    shadow_calibrator: isotonic      # runs in parallel, never applied
```

Analyse divergence by reading `eval_audit.jsonl`:

```bash
jq -c 'select(.shadow_prob!=null) | {r:.raw_ens_prob, l:.calibrated_prob, s:.shadow_prob}' \
  state/eval_audit.jsonl
```

Promotion criterion (same as §1 refit trigger, applied to the shadow's
implied trades): cumulative 30-day Brier of the shadow ≤ live Brier across
≥ 200 pairs. Until that bar clears, shadow stays shadow.

---

## 5. Rollback

Two options, in increasing blast radius:

**A. Config flip on the VPS (fastest, no git needed).** Edit
`/opt/oracle/python/config.yaml` to set `calibration_type: none` (or back to
`isotonic`) and restart. No artifact swap — the production file at
`data/calibration/platt_per_strike_type.json` stays in place for the next
deploy.

```bash
sudo -u oracle sed -i 's/calibration_type: per_type_platt/calibration_type: none/' \
  /opt/oracle/python/config.yaml
sudo systemctl restart oracle-bot
```

Note this leaves the repo-tracked config divergent from the VPS; the next
`deploy/update.sh` will re-pull the committed `calibration_type` and
reactivate the fit. Follow up with option B if the bad fit needs a durable
rollback.

**B. Revert the commit and redeploy.** Because `deploy/update.sh` does a
straight `git pull` — no file copy — the artifact on the VPS is whatever's
tracked in `data/calibration/platt_per_strike_type.json`. To roll back the
fit itself, revert the commit that introduced it (`git revert <hash>`) or
commit the previous artifact contents, push, and run `update.sh`.

**Kill-switch.** Because `PerTypePlattCalibrator` falls back to identity
on any load error, committing a deliberately malformed
`platt_per_strike_type.json` ships as an emergency kill — the next
restart picks up identity without touching config. Verify with `/status`
showing `calibrator: identity`.

---

## 6. Tripwires and their meaning

| Alert code | What it means | First action |
|---|---|---|
| `audit_log_stale:<stream>` | Writer stopped for > `stale_active_seconds`. | SSH in, `ls -la /opt/oracle/state/*.jsonl`. Check disk + permissions. |
| `audit_log_no_rotation:<stream>` | Active file > 1 day old with no rotated siblings. | Rotation regression; check recent changes to `audit_log.py`. |
| `calibrator_artifact_stale:<label>:<fit_date>` | `fit_date` > `stale_calibrator_days` old. | Refit (§2). If a refit is not warranted, run `/ack <code>` — the code embeds `fit_date`, so it stays silent only until the artifact actually changes; a later refit mints a new code and the alert returns on its own. |
| `brier_7d_vs_30d` | Short-window skill drop vs baseline. | Inspect per-city-var Brier; consider refit or temporary `excluded_city_vars` entry. |
| `bin_deviation` | One reliability bin is off by > 15%. | Look at the bin's mean_predicted vs observed; the fix is usually a refit. |
| `city_var_brier` | Specific city/var Brier > 0.25 on n ≥ 20. | Add to `excluded_city_vars` if persistent; unblock after refit shows positive BSS. |

All alerts go through the `CalibrationMonitor` gate (4h cooldown; warning→critical
escalation bypasses the cooldown).

---

## 7. File map (where things live)

| File | Purpose |
|---|---|
| `python/data/calibration.py` | Single source of truth for `Calibrator` implementations + `load_calibrator`. |
| `python/analytics/audit_log.py` | JSONL writer with daily UTC rotation + retention. Per-stream `SCHEMA_VERSION`. |
| `python/analytics/audit_tripwires.py` | Freshness / rotation / artifact-age checks. |
| `python/analytics/calibration_monitor.py` | Rolling Brier + reliability bins + alert dispatch. |
| `scripts/fit_platt.py` | Offline fitter. `--cross-validate`, `--no-spread-inflation`, `--min-per-type`. |
| `scripts/_calibration_common.py` | Shared pair builder — import from here, don't rebuild. |
| `data/calibration/platt.json` | Global Platt artifact (fallback; unused in prod). |
| `data/calibration/platt_per_strike_type.json` | Per-type Platt artifact (unused; the shipped config uses isotonic). |

---

## 8. Known traps

- **Strike-type label drift.** Calibrator keys are `greater`/`less`/`between`
  — *not* the `Direction` enum values (`above`/`below`). Translation happens
  once in `weather_edge.py` via `_DIRECTION_TO_STRIKE_LABEL`; any new caller
  must use the same map. Regression test: `test_fit_platt.py::
  test_artifact_rejected_by_runtime_if_key_renamed`.
- **Single-class fit.** `_fit_platt` raises on input where all outcomes are 0
  or all 1. The fitter CLI surfaces this; don't swallow it.
- **Shadow calibrator raising.** Swallowed in `evaluate_market_pure`; the
  audit row carries `shadow_prob: null` and a one-line `logger.exception`
  fires. Silent *live* crashes are intentional — shadow is diagnostic, not
  a trading signal.
- **EMOS NaN/Inf contract.** `evaluate_market_pure` filters non-finite GFS
  members at extraction (`_extract` in `weather_evaluation.py`) before the
  EMOS shadow block runs both raw and bias-corrected variants. If
  Open-Meteo ever returns NaN/Inf members, the variants skip via the
  `len(finite) >= 2` guard rather than propagating non-finite values into
  `emos_prob_raw`, `emos_mu_raw`, or the `ens_std_extrapolation` tripwire.
  Open-Meteo has never produced non-finite members in production, but the
  gate is the load-bearing safeguard for the 2026-06-06 scope-B promotion
  data quality.
- **`nbm_bracket_position_veto` / `NBM_BRACKET_POSITION_VETO` (shadow-only,
  2026-05-20).** Between-strike gate that compares NOAA's NBM deterministic
  point forecast against the bracket and vetoes when NBM lands on the
  opposite side from the model's directional bet (YES + NBM outside, or
  NO + NBM inside). Ships behind `weather.nbm_bracket_position_veto.enabled =
  false`: the gate computes `nbm_temperature` / `nbm_veto_fired` on every
  between-strike `eval_audit.jsonl` row but does NOT reject trades.
  Companion `NBM_FETCH_FAILED` reason only fires when `fail_open: false`
  (not the shipped default). Promotion gate before any live activation:
  n ≥ 80 vetoed shadow rows AND P&L delta ≥ +$200 net of fees AND win-rate
  gap ≥ 15pp AND no vetoed cell with zero settled trades. The backtest was 97%
  NO-between, so YES-side and tail generalization must be checked before live
  use. As of 2026-09-30 the gate is unreachable: every vetoed shadow row ended
  in `EDGE_BELOW_THRESHOLD` or `KELLY_ZERO`, so the veto would have blocked no
  trade. Killed variants (do not revive): gap-magnitude filter, ECMWF-GFS
  disagreement ratio.

## EMOS scope-B / scope-C decision rules

**Historical anchor for the broken bin (commentary, NOT a runtime invariant):**
- Computed by `scripts/fit_isotonic.py:_calibration_table` on the n=3,240 production-mirrored fit corpus binned by decile of `cal_prob`.
- Values: `(n=51, mean_predicted=0.478, observed_freq=0.314)` for `raw_ensemble_prob ∈ [0.40, 0.50)`.
- Documents *why* `[0.40, 0.50)` is the broken bin; does not constrain any runtime check.

**Runtime invariant (D9, 2026-05-12 — self-anchored):**
- `data/emos/reconciliation_envelope.json` — sealed JSON with `(recomputed_n, recomputed_predicted, recomputed_observed, eval_audit_row_hashes, isotonic_artifact_hash, frozen_file_sha256, fittable_cells_sealed, start/end/cutoff dates, git_rev)`.
- `data/emos/reference_window_eval_audit.jsonl.frozen` — atomically-written snapshot of the post-loader rows passing the bin/cell/cutoff/final filter at seal time.
- Both files are produced locally by `scripts/analyze_emos_shadow.py` on its first run. Neither ships in this repository.
- Seal command (ran once in the author's deployment, before audit-log retention deleted the source rows ~2026-05-22):
  ```
  uv run python scripts/analyze_emos_shadow.py --dry-run \
      --start-date 2026-05-01 --end-date 2026-05-08 \
      --decision-mode 30-day-system
  ```
- Verify (any subsequent run): same command, no `--reseal`. Aborts on tampering (file-hash mismatch), iso-artifact change, fittable_cells drift, or window change.
- Reseal escape hatch: `--reseal` moves both files to `.bak.<ISO_TS>` (NOT deleted) before re-anchoring.

**Pre-registered gates:**
- PRIMARY (binary): EMOS Wilson 95% CI on observed contains EMOS-window predicted in the broken-bin range.
- Track-1 contingency: if Track 1 already closed the bin, PRIMARY degrades to non-regression check; PnL gate becomes the promotion criterion. Non-regression precise definition: `wilson_ci_low(emos_obs, n_emos) ≤ iso_obs ≤ wilson_ci_high(emos_obs, n_emos)`.
- PnL VETO (binding):
  - 30-day system gate: CRD ≥ $0 at 25th percentile of event-day bootstrap AND CRD point estimate ≥ +$5.
  - 60-day per-cell gate: CRD ≥ −$1 AND ≥ 50% bootstrap mass above 0.
- SECONDARY (continuous): aggregate Brier sign agreement, PIT uniformity, disagreement-conditional Brier improvement.
- NO-REGRESSION VETO: Holm-Bonferroni-corrected per-cell Brier degradation check.

**HOLD policy:** N=2 strikes at 30-day cadence; hard-kill (CRD CI excludes 0 in loss direction) is asymmetric — kills scope C immediately.

**Window semantics:** 30-day → system-level; 60-day → per-cell hybrid promotion.

## Weather isotonic refit gates

Before you ship a refit of the weather isotonic artifact (`strategies.weather.isotonic_path`):

1. Run `uv run python scripts/validate_plateau_bounds.py --isotonic <new artifact>`.
   Exit 5 means the new artifact's `values[0]` defeats `min_calibrated_prob`. That is the
   2026-05-19 incident: the floor stopped binding and four long-shot YES trades fired
   within 7 minutes. Raise the floor or reject the fit. Re-derive `block_raw_prob_range`
   from the exit 0/1/2 output.
2. Run `uv run pytest tests/test_calibration_floor_coupling.py` after you change the
   config to point at the new artifact. It repeats the exit-5 check against the committed
   config. CI (`.github/workflows/ci.yml`) runs the full suite on pull requests and on pushes
   to `main`, but run this test by hand before you deploy.
3. There is no boundary-grazer audit gate yet (`docs/tech-debt.md`). Inspect every flat
   plateau bin by hand for NWS-integer boundary grazers.

## EMOS promotion sequence and falsification rule

`weather.emos_enabled_cells` is empty (all-isotonic since 2026-06-05). At any
future promotion attempt, run the three gates in order, one cell at a time, and
never batch-promote: (1) σ/bracket-width veto (`σ_emos(p99) / bracket_width > 2.0`
is non-overridable; never promote a σ-fail cell to the Gaussian lane),
(2) the BSS + LogSkill gate in `scripts/compare_emos_coefficients.py`,
(3) the floor/curve coupling gate. Falsification rule: if the BSS + LogSkill
gate independently rejects zero BSS-admitted candidates across ≥3 cells at
n≥30, deprecate it (it agreed with BSS alone 96.7% of the time at n=8–17/cell).
Re-entry for LAX/high (structural σ failure) and NYC/low (calibration failure,
`BSS_emos_vs_iso` −6.6%) has no defined trigger.

### Preconditions before any EMOS re-promotion

These items cannot take effect while `emos_enabled_cells` is empty.

**Blocking: resolve before the first cell goes back on EMOS.**

1. **Date-cluster bootstrap.** Corpus rows repeat each settled observation once per lead
   and snapshot. A ~5000-row cell holds only about 30–60 independent `forecast_date`s, so
   row-level bootstrap CIs are too narrow. The gate has cleared cells whose live results
   were indistinguishable from null.
   - Run `scripts/fit_emos.py` and `scripts/compare_emos_coefficients.py` with
     `--cluster-by-date`.
   - Cells with fewer than `MIN_CLUSTER_DATES` (10) dates are skipped by the fit and
     scored `INSUFFICIENT_N` by the compare.
   - The CI-width warning thresholds in `emos_fit.bootstrap_cell` (`b` width > 0.4,
     `d` width > 4.0) were tuned on row-level CIs. Expect more `unstable_b`/`unstable_d`
     warnings under clustering until you re-derive them.
2. **Score the traded subset.** The gate scores the pre-cap, pre-floor bracket grid, but
   live trades are the post-cap subset. Stage A ROI fell after promotion (−16.79%, n=64,
   not significant).
   - Add `survived_to_trade` to `eval_audit` rows, and backfill it by joining
     `state/trades.jsonl`.
   - Make `compare_emos_coefficients.py` score that subset by default.
3. **`overrides.json` coverage.** After you add the cell to `emos_enabled_cells`, run
   `uv run pytest tests/test_emos_overrides_guard.py`. It fails in any of these cases:
   - an `unstable_b`/`unstable_d`/`d_at_zero` warning has no active override;
   - the cell carries `incompatible_sigma`;
   - the cell is skipped or missing in the diagnostics.

**Build or decide before promotion.**

4. **Divergence logging.** Emit a structured `emos_divergence` log, with no rejection, in
   either of these cases:
   - `|emos_prob_raw − model_probability| > 0.10`;
   - EMOS gives `p < 0.05` and the bot routes YES.

   Five of 30 EMOS-era fills matched one of these patterns before the 2026-05-10 MIA/high
   double-fill. All five were MIA.
5. **Monitor cutover blindness.** `calibration_monitor` suppresses the 7d-vs-30d Brier
   alert, and logs a line, while a cell's 30d window mixes `live_calibrator_kind`. That
   leaves no coverage for 30 days after a cutover. Compute the 30d window per kind first.
6. **Plateau block before dispatch.** `block_raw_prob_range` fires on the raw probability
   before calibrator dispatch, so it also blocks EMOS cells. It is null today. If a refit
   makes it non-null, count its fires on `emos_raw` rows. Above about 10%, exempt
   `emos_raw` or derive separate bounds.
7. **μ-shift gate (optional).** `CALIBRATOR_DISAGREEMENT` is bypassed for `emos_raw`. The
   σ gate does not catch an `a`-coefficient shift that moves μ far from `ens_mean`.
   - Candidate rule: reject when `|emos_mu − ens_mean| > k · ens_std_p99`, with k as a
     config value (default 2).
   - First collect at least 4 weeks of σ-gate audit data, to see whether it adds
     protection.
8. **Per-lead coefficients (optional).** The runtime loader supports `by_lead`
   coefficients (`python/data/emos.py`), but `fit_emos.py` pools leads 0–2. MIA/high
   losses were concentrated at lead 1. Evaluate per-lead fits once each cell has n≥45 per
   lead.

**After promotion.**

9. **Runtime σ-gate fire rate.** Measure how often `emos_runtime_sigma_threshold` fires on
   the cell's `emos_raw` evaluations. If it fires on more than ~50% of them, demote the
   cell.

**Evidence on record.** From the 2026-05-11 regime-shift sweep and the Stage B replay
(`data/emos/refit_validation_20260511T234543Z.json`):

- **AUS/high.** EMOS loses (`BSS_emos_vs_iso` −33.1%, CI [−36.0%, −30.6%]), even though it
  absorbs +3.5°F of warm GFS bias (SNR 2.39, 19 fills). The cell is in
  `weather.excluded_city_vars`, so you must un-exclude it explicitly. Before you promote
  it, test three hypotheses:
  - the 46-day training window was too short;
  - the bias depends on lead (stratify BSS by lead 0/1/2);
  - ECMWF and GFS disagree on the bias direction.
- **AUS/low.** +2.40°F warm bias (SNR 2.39, 10 fills). Not excluded.
- **CHI/low.** +5.53°F warm bias (SNR 8.46, 19 fills). Stage B gave `BSS_emos_vs_iso`
  +0.158 [+0.130, +0.183] at n=5686. That reversed the earlier −40.8% figure, which was
  measured against the pre-2026-05-01 isotonic baseline.
- **Live σ/bracket ratios at demotion:** NYC/high 7.71, CHI/low 6.49, MIA/low 3.56,
  LAX/low 2.35, MIA/high 2.22.

## Interpreting harness verdicts (post-2026-05-24)

Per-cell harness output at `data/calibration/per_cell_winners.json` (schema_version: 2):

| Verdict | Meaning | Action |
|---|---|---|
| `winner` | Harness certifies this calibrator at corpus n. | Production exclusion still takes precedence; disagreement is the operator decision point. Check `flags.gate_margin_thin` — if true, the winner cleared the gate by less than 0.01; merit a second look. |
| `insufficient_evidence` | Harness cannot certify at current n. **NOT equivalent to "this calibrator is bad."** | Use `python/analytics/attribution.py` against `state/trades.jsonl` for cells with ≥30 live fills (e.g., NYC/high, AUS/low). Survivor-bias caveat: live attribution sees only the trades production filters admitted. |
| `worse_than_identity` | Harness confident no available calibrator beats identity (≥2 eligible candidates, all with LogSkill_p99 < 0). | Strong recommendation to deploy with identity calibration OR exclude the cell. Verify σ-veto is exercised on the cell's EMOS fit before deploying. |
| `insufficient_n` | Cell has too few settled forecast_dates (< 75 val rows). | Wait for corpus growth; harness cannot evaluate. |
| `insufficient_diversity` | Bracket distribution too narrow (< 5 unique strike/floor/cap tuples). | Calibrator selection is undefined; structural change to ticker mix needed. |
| `no_separation` | Two candidates tied within noise. | Default to identity. |

Diagnostic flags (informational, do not gate):
- `flags.pnl_logskill_disagreement` — winner on log score but historical ROI ≤ 0 (only fires when ROI is informative, i.e., not hitting the simulator's bet-fully-lost floor).
- `flags.gate_margin_thin` — winner LB in (+0.01, +0.02); borderline winner.
- `flags.below_power_floor` — n_val_total < 200; verdict is honest but power-limited.

For currently-trading cells where the harness verdict is `insufficient_evidence` (e.g., NYC/high, AUS/low under 2026-05-24 corpus), the decision aid is `python/analytics/attribution.py` against `state/trades.jsonl` — calibrator-specific and live-data-grounded, but survivor-biased by production filters.

## 9. ASOS intraday climatology refit (`intraday_asos_high`)

The `intraday_asos_high` strategy depends on a **remaining-gain climatology**
artifact `data/asos/climatology.json` (`(city, month, hour_utc) → (μ_gain,
σ_gain, n, source)`), distinct from the isotonic/EMOS calibrators above. It maps
"how much more warming after hour H" per cell; it is NOT a probability
calibrator.

- **Build:** `scripts/build_asos_climatology.py`. Hard-aborts if any city's
  source ASOS station ≠ its CLI settlement station (the CHI→MDW trap). Fit gates:
  per-cell `n≥30`, `(city,hour)` pool `n≥60`, `μ_gain≥0`, `σ_gain≥0.5°F`.
- **Leakage discipline (binding):** the fit window MUST hold out the live/eval
  period. When refitting after shadow accrual, exclude the dates used in any
  reliability evaluation from the fit corpus — a climatology fit on the same days
  it's scored against would inflate apparent sharpness.
- **Refit cadence:** re-run when (a) a new month enters the live window without
  coverage, or (b) ≥1 season of fresh ASOS accrues. Per-cell UTC coverage is
  data-bounded (MIA to ~18Z, AUS ~20Z, NYC/CHI ~22Z); a `lookup` miss returns
  None → the strategy skips that cell (fail-safe), so coverage gaps degrade
  gracefully rather than mis-trade.
- **The probability calibrator question is separate from this refit.** The
  strategy is parked (`enabled: false`). It traded the RAW posterior in shadow
  only. A 44-day forward corpus found no edge
  (`docs/research/2026-07-19-new-data-edge-audit.md`); read the root `CLAUDE.md`
  before any recalibrator work. The per-bin reliability table
  (`scripts/intraday_asos_shadow_report.py`, low-p NO bin) remains the tool for
  that question. A climatology refit does not answer it.
