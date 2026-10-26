# Tech Debt

Bounded design items that do not block live trading but will bite if left. Each entry
gives the problem, the evidence, and the action. Delete an entry when it ships.

**EMOS items live elsewhere.** EMOS is suspended (`weather.emos_enabled_cells: []`) and
`weather_edge` is shadow-only, so EMOS-only debt cannot fire. It is listed as
"Preconditions before any EMOS re-promotion" in `docs/calibration-runbook.md`.

---

## Isotonic refit has no boundary-grazer gate

**Files:** `scripts/fit_isotonic.py`, `scripts/validate_plateau_bounds.py`, `python/analytics/outcomes.py`

Isotonic regression has no boundary-distance input, so it averages NWS-integer grazer mass
(`t == floor_strike` or `t == cap_strike`) into flat plateaus. The reverted 2026-05-19 refit
put a plateau at `cal=0.189` over `raw ∈ [0.073, 0.119]` where 89/89 YES labels were grazers
and the strict-interior win rate was 0/341. Every refit can rediscover this in another bin.
`isotonic_v3` (in the shipped config since 2026-07-04) has none of the gates below. The floor-coupling
gate (exit 5, plus `tests/test_calibration_floor_coupling.py`), the plateau-coverage check
in `audit_tripwires.py`, and the MIA/low boundary-distance filter are partial mitigations.

**Action, before the next refit ships:**
1. Grazer audit gate: write per-plateau `grazer_fraction` to the artifact during the fit.
   The validator then fails (designed as exit 6) when `grazer_fraction ≥ 0.5` on a bin with
   `n_yes_in_bin ≥ 5`.
2. Decouple the F1.4 held-out reliability gate from F1.5. `_resolve_fit_date` sets
   `fit_date = max(forecast_date)`, so the holdout is empty and F1.4 always returns
   `warn_insufficient_n`. Hold back a fixed window, or use time-blocked k-fold on
   `forecast_date`.
3. Long term: add `boundary_distance` as an input, or stratify grazer vs interior. Both need
   corpus schema work in `outcomes.py`.

## Spread-inflation asymmetry between isotonic and EMOS corpora

**Files:** `scripts/fit_isotonic.py`, `scripts/build_emos_corpus.py`, `python/strategies/weather_evaluation.py`

The production-mirrored isotonic fit always applies `get_spread_inflation_factor`.
`--no-spread-inflation` is accepted only with `--legacy-gfs-only`. `build_emos_corpus.py`
does not inflate. The live path also inflates before computing `raw_ensemble_prob`. As a
result, every "EMOS beats isotonic" Brier number compares models fit on different inputs.

**Action:** Drop inflation from the isotonic fit (EMOS prices spread through `(c, d)`).
This is a code change, not a flag flip.
Add `raw_pre_inflation_prob` to audit rows. Refit on the symmetric corpus before any EMOS
re-promotion.

## Regime-shift detector absent from the nightly pipeline

No automated check compares recent GFS means with NWS actuals per cell. The 2026-05-11 sweep
was manual and found the CHI/AUS bleeders after 48 fills. `audit_tripwires.py` checks
staleness and plateau coverage, but nothing computes per-cell bias or SNR.

**Action:** After `collect_daily_snapshot.py`, join the last 20 days of
`data/snapshots/<date>/2000/weather_forecasts.json` with `settlements.json`. Compute
per-(city, var) bias and SNR. Send a Telegram CRITICAL alert when a tradeable cell crosses
Δ ≥ 2°F with SNR ≥ 2.

## Bracket boost wiring is cosmetic

**Files:** `python/strategies/weather_edge.py`, `python/config.yaml` (`directional_boost_factor`)

The boost multiplies `effective_edge` after the depth walk, the `fill_edge` gate, and Kelly
sizing. It changes only metadata, tie-breaking, and logs, never whether a trade happens or
its size. The underlying signal looks real: PASSED-NO with `bid_excess > $0.03` won 92.9%
(Wilson [86.1, 96.5]), ROI +0.408, n=10,229 candidate rows. It is pre-cap, in-sample, and
unstratified.

**Action:** Stratify by lead, city, calibrator, and spread first. If the signal survives,
the smallest real wiring is a lower `edge_threshold` for boost-eligible candidates inside the
depth loop. Moving the boost into the depth loop, or giving boost-eligible candidates a higher
Kelly fraction, each needs its own design pass. The YES side (n=12, 3 settled) is unusable and
needs its own validation cycle.

## `walk_forward_replay` leaks future information

**File:** `python/analytics/backtest.py` (`walk_forward_replay`, `BacktestConfig`)

A calibrator passed in is held fixed across folds. The default call passes none.
`BacktestConfig.block_raw_prob_range` defaults to `(0.533, 0.967)`, which came from a
full-corpus artifact. It also no longer mirrors live: the config value is `null` since
isotonic_v3. Walk-forward is a diagnostic only, so fold Sharpes are optimistic by an
unknown amount.

**Action:** Before walk-forward becomes a go/no-go gate, refit the calibrator per fold
(`fit_isotonic` → `validate_plateau_bounds` → `BacktestConfig` override) and take the
default bounds from the live config.

## `fraction_*` class methods use mixed boundary conventions

**File:** `python/data/weather.py`

`fraction_above` is strict (`>`), `fraction_below` is `1 − fraction_at_or_above`, and
`fraction_between` is inclusive at both ends. `_apply_nws_window` (the 2026-05-21 bracket-window fix) applies
±0.5°F shifts, but the overlap survives the shift. A member exactly at `cap + 0.5` counts
in the `between` bracket and also in the next bracket up
(`fraction_at_or_above(floor + 0.5)`). So a full bracket set can sum above 1. Members land
exactly on x.5 only rarely. `fraction_above` has no live caller.

**Action:** Get Kalshi's canonical `>`/`≥` and `<`/`≤` convention at strike boundaries.
Make `fraction_between` half-open to match it, and remove `fraction_above`. This changes
live probability math, so treat it as its own change. Keep the partition tests in
`tests/test_core_math.py` green.

## `build_pairs` legacy path skips the NWS-rounding shift

**Files:** `scripts/_calibration_common.py` (`build_pairs`), `--legacy-gfs-only` in `fit_platt.py` and `fit_isotonic.py`

`build_pairs` compares raw `m > floor` / `m < cap` with no ±0.5°F shift. Production fits use
`build_production_mirrored_pairs`, so the path is dormant. An operator who passes
`--legacy-gfs-only` silently gets pre-fix math. Deletion is not mechanical, because
`tests/test_fit_platt.py` (`TestFitPlattEndToEnd.fit_inputs`) uses `build_pairs` as its real
pair builder. Do not touch the unrelated `build_pairs` in `scripts/fit_favorite_bias_map.py`.

**Action:** Delete `build_pairs` and the flag after you rebuild that fixture on the
production-mirrored path. Treat it as a calibration-tooling task gated on the
`test_fit_platt` and `test_isotonic` suites.

## No dry-run mode for FillsReconciler

**Files:** `python/execution/fills_reconciler.py`, `python/settings.py` (`FillsReconcilerConfig`)

New reconciler behavior can ship only on unit tests; `/disable_reconciler` is reactive.
`scripts/reconcile_fills.py` dry-runs a backfill window, but not the live tick.

**Action:** Add `fills_reconciler.dry_run` (default off). A safe dry run must gate every side
effect, not just `_append_synthetic_rows`:
- `RiskManager.close_position`, `realize_partial`, and `record_strategy_trade`, plus
  `engine._orders` mutation in the manual-close and cash-out paths
- the cursor advance and `_persist_state`, or a later enable never backfills
- the shakedown `seen_oids` files and the in-memory sticky sets
- Telegram alerts: prefix them with `[DRY RUN]`

Design it as its own change, with off-by-default regression tests on each branch.

## Claude CLI runs from the admin's home directory

**Files:** `deploy/update.sh` (step 4), `deploy/oracle-bot.service` (`ProtectHome=read-only`), `docs/deploy-guide.md` Step 8

The deploy guide installs the Claude CLI under `/home/<admin>/.local/bin/claude`, and the
`oracle` user runs it through `ORACLE_CLAUDE_CODE_BIN`. `ProtectHome=read-only` keeps the home
directory readable, so this works. But `update.sh` runs `chmod o+rx` on the binary's target
and its parent directory after each deploy, which opens a directory in the admin's home to
all users. The default path `/home/$SUDO_USER/.local/bin/claude` also breaks for any other
install location.

**Action:** Install the Claude CLI at `/usr/local/bin/claude` (root:root, 0755) and point
`ORACLE_CLAUDE_CODE_BIN` at it. Remove the `chmod` from `update.sh` and update the deploy
guide. This needs a root step on the VPS.

## Kalshi V2 NO/sell bid↔ask mapping is unexercised

When enabled, `favorite_bias` is the only strategy that places live orders, and it buys YES as a taker. The NO and sell paths
have never run against the V2 create-order API. `cancel_order` has run only on orders the bot
cancels itself.

**Action:** Before any NO-side or resting-order strategy goes live, demo-confirm the NO/sell
bid↔ask mapping and `cancel_order` on V2.

## Stale NWS-preliminary values in old snapshots

`collect_daily_snapshot.py` stores `is_preliminary` reports as-is and never refetches the
final report. Three known outcome mismatches with Kalshi: `KXLOWTNYC-26APR25-B42.5`,
`KXLOWTNYC-26APR25-B40.5`, `KXHIGHAUS-26MAY10-B90.5`. Corpus loaders (`_calibration_common.py`,
`build_nws_bias_corpus.py`) now prefer final over preliminary. `calibration_monitor` reads
Kalshi-truth settlements. Only backtests that read a single snapshot slot still
see the stale value.

**Action:** Refetch the final CLI when only a preliminary report is on disk, or repair the
tree once.

## Small or optional items

- **Kelly same-day correlation haircut.** `dollars *= 1/sqrt(1 + n_same_day · ρ)` ships at
  `same_day_correlation_rho: 0.0`, which is a no-op. Measured within-day φ is +0.071
  (n=1,356 pairs over 20 days). Daily exposure is bound by the total-exposure
  cap (`risk.max_total_exposure_pct`), not by Kelly. The haircut matters at about $50K bankroll or with a looser
  cap. At that point, derive ρ from fresh settlements.
- **`post_flip_monitor.sh` T3 is today-only.** Add `--lookback-days N` (default 1) to
  `daily_trade_report.py` and wire 7 into T3. It is moot while `weather_edge` is shadow-only,
  because T3 is weather-scoped and stays empty.
- **`predictions.jsonl` Brier consumer** (optional). It would show pre-cap drift that
  `eval_audit` monitors cannot see. The only active writer is shadow `weather_edge`.
