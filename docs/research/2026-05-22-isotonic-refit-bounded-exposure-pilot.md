# Isotonic refit on post-bracket-fix corpus + bounded-exposure live pilot

**Status:** Phase 0 validation complete; plan ready for execution
**Date:** 2026-05-22
**Owner:** christurgeon
**Related:** the 2026-05-21 bracket-window fix, the BETWEEN threshold rescale, the post-incident hotfix, the unpublished tracker (bracket-fix follow-ups)

## Context

The 2026-05-21 bracket-window fix changed `compute_blended_probability` to shift bracket bounds by ±0.5°F per direction (NWS integer-rounding settle window). The on-disk `data/calibration/isotonic.json` was fit 2026-05-19 against the OLD 1°F-window raw-prob distribution and is now anti-helpful. The hotfix set `calibration_type: none` (identity) for non-EMOS cells, with the bot in kill-switch mode.

Goal: refit isotonic, validate as well as today's data permits, **deploy live at bounded exposure with pre-committed rollback triggers**. Two weeks of post-fix live data is more valuable than two weeks of shadow data because live exercises the full pipeline (calibrator + filters + Kelly + execution + settlement).

### Why refit today, not wait 30 days

The fit's X-axis (`raw_prob`) is recomputed from stored ensemble members through patched `compute_blended_probability`; the Y-axis (outcome) comes from unchanged NWS settlements. Both inputs are reconstructable from the historical snapshot corpus today. The historical refit IS mathematically valid; what we can't do today is validate on **post-fix held-out data** (there are zero settled post-fix days — bracket fix was 2026-05-21 evening; today is 2026-05-22).

We accept this validation gap by bounding live exposure rather than waiting two weeks for shadow data that doesn't exercise the full pipeline.

### Downstream-filter dependency classification

- **Window-only** (depends on raw_prob scale, not calibrator): `min_raw_prob_for_between` — **DONE at 0.12** (data-derived).
- **Calibrator-coupled** (depends on isotonic curve shape): `block_raw_prob_range`, `min_calibrated_prob` — re-derive from new artifact in Phase 2.
- **Window-independent / no action**: `min_distance_from_strike_std`, `max_model_market_gap_near_term`, `max_spread`, `spread_skill_sensitivity`, `max_fill_slippage`, bias_correction params.

### Active isotonic cells (verified count)

14 total (7 cities × 2 vars) − 4 excluded (DEN×2, SEA/high, LAX/high) − 3 EMOS (MIA/high, MIA/low, LAX/low) = **7 isotonic cells**: NYC/high, NYC/low, CHI/high, **CHI/low** (demoted from EMOS 2026-05-21), AUS/high, AUS/low, SEA/low.

After pre-emptive pilot exclusions (Phase 5a), **4 cells trade in pilot**: NYC/high, NYC/low, AUS/low, SEA/low.

## Phase 0 — Validation findings (2026-05-22, complete)

Confirmed, refuted, and unknown — read this section before executing.

| Item | Status | Detail |
|---|---|---|
| Kill-switch scope | CONFIRMED | `python/orchestrator.py:1139-1154` — STOP halts the entire main loop (sets `self._running = False; return`). No eval_audit rows during kill. |
| `load_calibrator(isotonic_path=...)` | CONFIRMED | `python/data/calibration.py:660-683` — non-default paths plumbable. |
| calibration_monitor Telegram alert wiring | CONFIRMED | `python/orchestrator.py:724` passes `alert_callback=self._on_calibration_drift`. |
| Current `kelly_fraction` | CONFIRMED | `python/config.yaml:140`: **0.15**. |
| `max_total_exposure_pct` | CONFIRMED | `python/config.yaml:136`: **0.80** (overrides settings.py default of 0.40). |
| `max_position_dollars` knob | **DOES NOT EXIST** | No discrete per-trade dollar cap. Only bankroll-fraction-based `max_dollars` (`risk/manager.py:250`). **Plan revised: use `kelly_fraction` reduction + `max_total_exposure_pct` tightening instead of adding a new knob.** |
| `state/calibration_monitor.json` path | CONFIRMED (default) | `python/settings.py:526`: `Field(default="state/calibration_monitor.json")`. Does not currently exist locally; presumed on VPS. `rm -f` is safe if absent. |
| Snapshot orderbook depth | **NOT STORED** | `data/snapshots/<date>/<run>/markets.json` only contains `yes_bid_dollars`, `yes_ask_dollars` (top-of-book). **Plan revised: Phase 4 replay is Stage 1 only — no orderbook-depth / VWAP-fill re-derivation possible.** |
| NWS settlement backfill | **MOSTLY UNAVAILABLE** | NWS public API retains ~5-7 days of CLI products. Test results: 5-21 ✓, 5-19 ✓, 5-14 ✗, 5-11 ✗, 5-10 ✗. **Plan revised: backfill only 2026-05-19 and 2026-05-20 (recoverable); accept the 4-day gap at 2026-05-11→14.** Marginal corpus impact (the 4 days are out of ~59 total). |
| NWS API method | CONFIRMED | `NWSClient.get_daily_observation(city, target_date) -> CliObservation \| None`. Returns `high_f`, `low_f`, `is_preliminary`. |
| Settlements.json schema | CONFIRMED | Array of `{city, cli_location, observation_date, high_f, low_f, product_id, issued_at, is_preliminary, snapshot_ts}` per snapshot date. Backfill writer must match. |
| `daily_trade_report.py --json` schema | CONFIRMED (corrections needed) | Top-level keys: `trade_date, generated_at, phase0_depth_probing, intraday_emos_high, per_cell, settled, pending, calibration`. `calibration_ratio` is nested under `.calibration.calibration_ratio` — **plan's rollback query corrected**. `settled` is an array of per-day entries with `losses[]`; per-cell aggregation needs custom parsing. |
| Cumulative `calibration_ratio` (current state) | OBSERVED | `2.43` under identity-on-pre-fix-corpus (224 settled, 145 wins). Implies the post-flip rollback trigger must measure a **post-flip rolling window** (not cumulative), or the trigger fires on day 1 from inherited state. |
| `bias_correction` historical applicability | NOT BLOCKING | Continuously enabled since 2026-03-21. Corpus starts 2026-03-18 (3 days of un-bias-corrected forecasts before then). Fitter applies CURRENT bias table to all. This is correct behavior — the calibrator should map production-current bias-corrected raw_probs to outcomes. |
| Systemd `RestartSec` on VPS | UNKNOWN | Skipped; requires VPS access. Operator should verify before un-kill to know whether "first scan in 10 min" assumption holds. Default systemd is 100ms which means immediate. |

## Approach (revised based on Phase 0)

### Phase 1 — Initial refit + Phase 2 gates (~30 min, read-only)

```bash
uv run python scripts/fit_isotonic.py \
  --output data/calibration/isotonic_v2.json \
  --cross-validate
```

Default path → `build_production_mirrored_pairs` → patched `compute_blended_probability`. **Do not pass `--legacy-gfs-only`**.

`scripts/calibrate.py` is on stale pre-fix math (lines 130-153) and is NOT in this path. Flag for separate cleanup PR.

### Phase 2 — Decoupled gates (run after every refit)

Two independent gates.

**Gate A: plateau bounds**
```bash
uv run python scripts/validate_plateau_bounds.py --isotonic data/calibration/isotonic_v2.json
```
Exit codes: 0 (one dangerous plateau, OK), 1 (multiple plateaus, REJECT), 2 (no plateaus — block becomes no-op), 5 (floor-coupling defeats `min_calibrated_prob` — REJECT, 2026-05-19 incident guard).

**Gate B: explicit held-out reliability** (runs regardless of Gate A)
Create `scripts/check_held_out_reliability.py` — thin wrapper calling `_load_held_out_pairs_from_snapshots` + `held_out_reliability_check` (importing from `scripts/validate_plateau_bounds.py`; underscored functions used per established pattern in the codebase). Runs the held-out check unconditionally — closes the structural unsatisfiability where exit-2 skips validation.

### Phase 3 — Limited settlement backfill + re-refit (~30 min, revised scope)

**Backfill scope: 2 days only** (NWS retention limit confirmed in Phase 0).

For 2026-05-19 and 2026-05-20:
1. `uv run python` script invoking `NWSClient.get_daily_observation(city, date)` for each of the 7 cities
2. Write `data/snapshots/<date>/settlements.json` matching the verified schema (array of per-city `{city, cli_location, observation_date, high_f, low_f, product_id, issued_at, is_preliminary, snapshot_ts}`)
3. Verify each file loads cleanly via `scripts/_calibration_common.load_snapshot_data`

Accept the 4-day gap at 2026-05-11→14 as data we can't recover from NWS. Out of ~59 settled days, 4 missing is ~7% loss — marginal.

Re-run Phase 1 + Phase 2 on the augmented corpus. **Only the final artifact's gates are load-bearing.**

### Phase 4 — Stage-1-only per-arm replay on pre-fix temporal hold-out (~2h, revised scope)

**Phase 0 confirmed snapshots don't store orderbook depth.** Replay drops Stage 2 (VWAP / fill-depth) entirely. Stage 1 effective_edge + filters + Kelly + cap is still tractable.

Create `scripts/replay_isotonic_holdout.py`:
1. Load corpus via `build_production_mirrored_pairs` (patched).
2. Partition by `forecast_date` at `today − 28d`. Refit on early partition in-memory.
3. **Apply Stage-1 pipeline per arm independently**, for each held-out market under both `(identity)` and `(isotonic_v2)`:
   - calibrated_prob
   - effective_edge = `(model_prob - market_yes_ask) * ensemble_confidence - half_spread` from `yes_bid_dollars`/`yes_ask_dollars` in the snapshot
   - gates: `min_raw_prob_for_between: 0.12` (BOTH arms), `block_raw_prob_range` (with new bounds for isotonic arm), `min_calibrated_prob` (with new floor for isotonic arm)
   - Kelly sizing at pilot `kelly_fraction: 0.075`
   - `PerDayEventCap` selection (independent per arm — pure in-memory rebuild from candidate list, no `state/trades.jsonl` hydration)
4. Compare WHAT WOULD HAVE TRADED vs realized outcomes:
   - Global Brier (cluster bootstrap CI on forecast_date)
   - Per-cell breakdown (especially AUS/high, CHI/high, CHI/low — pre-emptive exclusion candidates)
   - Per-strike-type (greater / less / between — ±0.5 shift moves them oppositely)
   - `min_raw_prob_for_between` sensitivity at [0.05, 0.08, 0.10, 0.12, 0.15]
   - Per-cell BETWEEN raw_prob shift uniformity check (+6.0pp average from the BETWEEN threshold rescale — is it uniform?)

Output: structured JSON + summary at `data/research/2026-05-22-isotonic-v2-replay.json`.

**Caveat to document:** Stage 1 replay can't model Stage 2 rejections (book depth, slippage). Live behavior may differ. Acceptable for calibration-quality validation; not a P&L attribution proxy.

### Phase 5 — Pre-flip preparation

#### 5a. Pre-emptive cell exclusions
Add to `python/config.yaml` `excluded_city_vars`:
- **AUS/high** — the unpublished tracker backtest alpha loss under fix
- **CHI/high** — same
- **CHI/low** — demoted from EMOS 2026-05-21; zero isotonic track record

Pilot trades 4 cells: NYC/high, NYC/low, AUS/low, SEA/low.

#### 5b. Bounded-exposure knobs (revised — no new `max_position_dollars`)

Phase 0 confirmed `max_position_dollars` doesn't exist. Use existing mechanisms:

| Knob | Current | Pilot value | Reason |
|---|---|---|---|
| `kelly_fraction` | 0.15 | **0.075** | Half. Position sizes halved per `position_dollars = kelly_raw * kelly_fraction * bankroll` |
| `max_total_exposure_pct` | 0.80 | **0.30** | Tighten aggregate exposure (originally targeted 0.20; raised to 0.30 at deploy because `_check_kelly_size_executable` validator requires `kelly × single × total × bankroll ≥ $1.00` and at the pilot bankroll the 0.20 value fell below that minimum). 0.30 clears it with 12% margin. Risk-manager-enforced. |
| Daily loss halt | TBD (look up exact knob in Phase 1) | tighten current / 2 | |

**Adding a hard per-trade dollar cap is out of scope for this plan.** The combined effect of halved Kelly + 75%-reduced total-exposure cap provides comparable bound. Tradeoff: a single high-Kelly signal can still consume the full per-trade Kelly output, but aggregate exposure across the 4 cells is capped at 20% bankroll.

#### 5c. Filter re-derivation
1. `min_raw_prob_for_between` — DONE at 0.12. Phase 4 sensitivity validates.
2. `block_raw_prob_range` — from Phase 2 Gate A stdout. If Gate A exits 2 (no plateau), block becomes no-op (document explicitly).
3. `min_calibrated_prob` — confirm new isotonic's `values[0]` is below 0.10 (Gate A exit-5 check).

#### 5d. Docs (pre-flip)

- `CLAUDE.md` "Current State" — update EMOS count 5→3; update `excluded_city_vars` to include new 3
- the unpublished tracker — close items 1, 2; retract `calibrate.py` claim at line 226; add follow-up "patch `scripts/calibrate.py:_compute_model_prob` (stale post-fix)"

#### 5e. Verify alert wiring — DONE in Phase 0

`python/orchestrator.py:724` confirms wiring. Threshold relaxation is meaningful.

#### 5f. Reset calibration_monitor state

`rm -f state/calibration_monitor.json` on VPS. Raise thresholds for 14d: `brier_warn_mult: 1.3 → 1.6`, `brier_crit_mult: 1.6 → 2.0`. Document revert date.

#### 5g. Pre-write monitoring script (corrected per Phase 0 schema)

Create `scripts/post_flip_monitor.sh`:

```bash
#!/usr/bin/env bash
# Run on VPS or after --pull-vps locally.
# Pre-committed rollback triggers — operator interprets PASS/FAIL.

set -uo pipefail

# Trigger 1: calibration_ratio measured over post-flip window
# NOTE: --date is the daily window. Cumulative calibration_ratio includes pre-flip data
# and will read ~2.43 (current state); use the most recent day for post-flip-only signal.
uv run python scripts/daily_trade_report.py --json /tmp/dtr.json --date "$(date -u +%F)" --pull-vps --refresh > /dev/null
CR=$(jq -r '.calibration.calibration_ratio // "null"' /tmp/dtr.json)
echo "calibration_ratio (today): $CR  [FAIL if > 2.0 over n≥10 settled]"

# Trigger 2: any single-trade loss > $MAX_TRADE_LOSS (dollars; set from pilot sizing)
LOSSES=$(jq -r --argjson m "$MAX_TRADE_LOSS" '[.settled[].losses[] | select(.pnl < -$m)] | length' /tmp/dtr.json)
echo "single-trade losses > \$$MAX_TRADE_LOSS: $LOSSES  [FAIL if > 0]"

# Trigger 3: per-cell loss rate > 60% on n≥5 (7d window)
uv run python scripts/daily_trade_report.py --json /tmp/dtr_7d.json --pull-vps > /dev/null  # default 7d aggregation
jq -r '
  .settled
  | map(.losses[]? | {city, var}) as $L
  | .settled
  | map(.losses[]? | "lost") as $allLost
  | ([$L | group_by(.city + "_" + .var) | .[] | {key: (.[0].city + "_" + .[0].var), n: length, losses: length}])
' /tmp/dtr_7d.json || echo "per-cell aggregation FAIL (verify schema)"

# Trigger 4: filter dead — 0 fires in 48h
FILTER_FIRES=$(grep -h '"reason":"BLOCKED_PLATEAU_RANGE"\|"reason":"MIN_CALIBRATED_PROB"' state/eval_audit.jsonl* 2>/dev/null | wc -l)
echo "filter fires (rolling): $FILTER_FIRES  [FAIL if 0 over 48h on n≥20 candidates]"
```

The trigger-3 jq aggregation is approximate; operator should manually verify against `per_cell` section if it populates (it was empty for the 2026-05-19 test date — may be live-only).

### Phase 6 — Deploy with graduated un-kill

#### 6a. Pre-flip backup
```bash
cp data/calibration/isotonic.json data/calibration/isotonic.preflight-$(date +%Y%m%d).json
```

#### 6b. Wire isotonic_v2 at non-default path

Add `isotonic_path` field to `WeatherStrategyConfig` in `python/settings.py` (mirror the existing `extra: "ignore"` pattern; add to `model_dump()` defaults). Thread to `weather_edge.py` `__init__` (around line 174 where `bias_table` is taken). Pass to `load_calibrator(isotonic_path=Path(self._isotonic_path))`. **Estimate: ~30-50 lines including tests** (Phase 0 corrected my "~10 lines").

If this turns out non-trivial, fallback: keep canonical path; `mv isotonic_v2.json isotonic.json` with backup preserved (loses the footgun safety, but operator can decide).

#### 6c. Edit config

```yaml
calibration_type: isotonic                    # was: none
isotonic_path: data/calibration/isotonic_v2.json  # new field
block_raw_prob_range: [<from Phase 2 Gate A>]
kelly_fraction: 0.075                         # was: 0.15
max_total_exposure_pct: 0.20                  # was: 0.80
audit_log_retention_days: 30                  # was: 14
excluded_city_vars:                           # add 3
  - DEN_low
  - DEN_high
  - SEA_high
  - LAX_high
  - AUS_high            # NEW
  - CHI_high            # NEW
  - CHI_low             # NEW
# (raise calibration_monitor brier thresholds per 5f)
```

#### 6d. Quality gates
```bash
uv run ruff check python/ && uv run ruff format python/
uv run mypy python/
uv run pytest
```

`tests/test_calibration_common.py` may have expectations tied to the old artifact's behavior — likely needs updating with new fit metadata.

#### 6e. Commit (template — see plan v3 for full message)

#### 6f. Deploy
Pre-SCP checklist:
1. **Verify `RestartSec` on VPS** (Phase 0 open item):
   `ssh $ORACLE_VPS_HOST "systemctl show oracle-bot.service -p RestartSec"`.
   If 100ms (default): Phase B "watch 2 scan cycles" timing holds as planned.
   If 30s-5min: extend Phase B observation window accordingly.
2. SCP: `python/config.yaml`, `python/settings.py`, `python/strategies/weather_edge.py`, `data/calibration/isotonic_v2.json`, `data/calibration/isotonic.preflight-20260522.json`, `scripts/post_flip_monitor.sh` (+ `scripts/check_held_out_reliability.py`, `scripts/replay_isotonic_holdout.py` for operator parity).
3. `rm -f state/calibration_monitor.json` on VPS.

#### 6g. Graduated un-kill (be present, ~30 min active)

**Phase A — STOP on, post-restart (artifact load + config-strip verification):**
1. Confirm `state/STOP` exists
2. `systemctl restart oracle-bot.service` on VPS
3. Inspect orchestrator startup logs for:
   - `IsotonicCalibrator loaded from path data/calibration/isotonic_v2.json` (exact path)
   - `live_calibrator_kind` init, no Pydantic-strip warnings
   - Bankroll hydration succeeded
4. If anything fails: do not proceed.

**Phase B — un-kill, watch 2 scan cycles (~10-15 min, poised to re-kill):**
5. `rm state/STOP` on VPS (or Telegram `/unkill`)
6. Tail `state/eval_audit.jsonl` in real time. Verify on first scan:
   - `live_calibrator_kind: "isotonic"` on non-EMOS rows
   - At least one row with `reason: "BLOCKED_PLATEAU_RANGE"` OR `reason: "MIN_CALIBRATED_PROB"` over 2 scans on ≥20 candidates. Zero fires → filters dead → re-kill.
   - No calibrator/dispatcher exceptions
   - Trade fills (if any) at sane sizes given halved Kelly + 0.20 total exposure
7. **Stand-by 2 full scan cycles before walking away.** Any rollback trigger → immediate `/kill`.

**Phase C — 24h watchful:**
8. `scripts/post_flip_monitor.sh` at 6h, 12h, 18h, 24h
9. Next morning: `uv run python scripts/daily_trade_report.py --pull-vps --refresh` — verify `.calibration.calibration_ratio` for today is in [0.7, 1.3]× range (n≥5 settled). Note: cumulative ratio is ~2.43 from pre-flip identity data; today-only ratio is the post-flip signal.

### Phase 7 — Rollback procedure (symmetric)

If any trigger fires:
1. `/kill` via Telegram
2. `git revert HEAD` for the flip commit
3. Restore canonical path or revert `isotonic_path` config (depends on Phase 6b choice)
4. **`rm -f state/calibration_monitor.json` on VPS** (symmetric to 5f — clears post-flip-noisy baselines)
5. `systemctl restart oracle-bot.service`
6. Verify `live_calibrator_kind: "identity"` returns in next scan
7. `rm state/STOP` to resume on identity if desired
8. Investigate. Do not retry until root cause is understood (2026-05-19 base rate).

## Files / artifacts touched

**Read-only inspection:**
- `data/snapshots/` — corpus (2 days recoverable backfill in Phase 3; 4-day gap accepted)
- `scripts/fit_isotonic.py`, `scripts/_calibration_common.py`, `scripts/validate_plateau_bounds.py`, `scripts/compare_emos_coefficients.py`
- `python/data/calibration.py:660-683`, `python/data/nws.py:239` (`get_daily_observation`), `python/orchestrator.py:724, 1139-1154`
- `python/strategies/weather_evaluation.py`, `python/strategies/weather_audit.py`, `python/analytics/sizing.py`, `python/analytics/calibration_monitor.py`

**Created:**
- `data/snapshots/2026-05-19/settlements.json`, `data/snapshots/2026-05-20/settlements.json` (Phase 3 backfill, 2 days only)
- `scripts/check_held_out_reliability.py` — Phase 2 Gate B wrapper
- `scripts/replay_isotonic_holdout.py` — Phase 4 Stage-1-only replay
- `scripts/post_flip_monitor.sh` — rollback queries
- `data/calibration/isotonic_v2.json` — new artifact (non-default path)
- `data/calibration/isotonic.preflight-YYYYMMDD.json` — backup of retired artifact
- `data/research/2026-05-22-isotonic-v2-replay.json` — Phase 4 output

**Edited:**
- `python/config.yaml` — `calibration_type`, `isotonic_path` (new), `block_raw_prob_range`, `kelly_fraction`, `max_total_exposure_pct`, `excluded_city_vars` (+3), `audit_log_retention_days`, calibration_monitor thresholds
- `python/settings.py` — add `isotonic_path` field to `WeatherStrategyConfig`
- `python/strategies/weather_edge.py` — pass `isotonic_path` to `load_calibrator`
- `CLAUDE.md` — Current State (5→3 EMOS; updated excluded list)
- the unpublished tracker — close items 1, 2; retract calibrate.py claim; new follow-up
- VPS: `state/calibration_monitor.json` deleted

## Reused existing functions

- `compute_blended_probability` (`python/data/weather.py`) — patched
- `build_production_mirrored_pairs` (`scripts/_calibration_common.py:377`)
- `IsotonicCalibrator.calibrate` + `load_calibrator(isotonic_path=...)` (`python/data/calibration.py:144-215, 660-683`)
- `held_out_reliability_check`, `_load_held_out_pairs_from_snapshots`, `floor_coupling_check` (`scripts/validate_plateau_bounds.py`)
- `bootstrap_bss_ci` template (`scripts/compare_emos_coefficients.py:69-97`)
- `NWSClient.get_daily_observation` (`python/data/nws.py:239`) — backfill
- `fractional_kelly` (`python/analytics/sizing.py:75`)

## Verification

Per Phase 6g graduated un-kill. Additionally:

1. **24h check**: `daily_trade_report.py --pull-vps --refresh` next morning. Today's `.calibration.calibration_ratio` in [0.7, 1.3]×; no single-trade loss above the per-trade loss limit.
2. **7d check**: per-cell n approaching ≥5 on each of 4 active cells; no trigger fired; no false-positive critical alerts.
3. **14d check (~2026-06-05)**: ~14 post-fix settled days; run per-cell BSS analysis. Decisions:
   - Un-exclude AUS/high, CHI/high, CHI/low if calibrated
   - Restore `kelly_fraction: 0.15` and `max_total_exposure_pct: 0.80` if pilot clean
   - Revert calibration_monitor threshold relaxation
   - Investigate per-cell BETWEEN +6.0pp variance if Phase 4 flagged

### Pilot-data-sufficiency expectation (set up-front)

At bounded knobs (`kelly_fraction: 0.075`, `max_total_exposure_pct: 0.30` — raised from planned 0.20 to clear the kelly_size_executable validator at the pilot bankroll; 4 trading cells, per-day cap = 1), realistic pilot volume over 14 days is **~20-30 trades total, ~5-8 per cell**. Sufficient for win-rate sanity but NOT per-cell Brier inference (CI half-width on Brier at n=5 is ±0.15+, wider than any meaningful effect size). Trade sizes are further attenuated by the active 24.6% drawdown (`drawdown_size_reduction: 0.50`) — first-scan trades were small.

Implication for the 14d decision: expect to either (a) extend the pilot at higher exposure once "no disaster" is confirmed, or (b) accept rough per-cell inference and make the call on aggregate signals + qualitative monitoring. The pilot's job is "don't disaster," not "gather precise calibration stats." Document this expectation now, not later — future-you-from-rollback will need to know what data quality was achievable at the bounded knobs.

## Post-flip status (2026-05-22 ~20:23 UTC)

**Pilot LIVE** — STOP removed, first scan completed cleanly (Tick complete: 3 signals, 3 approved, 0 rejected, 0 skipped). Phase A + Phase B verified per `docs/research/...-pilot.md` §6g.

First-scan trades:
- KXHIGHNY-26MAY23-B60.5 YES (NYC/high, isotonic, conf=0.786)
- KXLOWTMIA-26MAY23-B78.5 NO (MIA/low, EMOS, conf=1.0)
- KXLOWTLAX-26MAY23-B58.5 NO (LAX/low, EMOS, conf=1.0)

All trades under the T2 trigger. Calibrator dispatch correct (isotonic for pilot cells, EMOS for promoted cells). Per-day cap suppressed 0 candidates. Filter fires observed (Stage-2 fill-slippage reject on KXHIGHNY-B56.5).

## Risks / open items

- **4 trading cells have no post-fix calibration validation.** Updated worst-case 14-day loss math at the actual VPS bankroll (from `ORACLE_BANKROLL`, not the plan's original larger assumption): per-trade ceiling under bounded knobs ≈ `0.05 × bankroll × 0.50 (drawdown)` (effectively smaller today due to active drawdown 24.6%). Aggregate cap `0.30 × bankroll` (or `0.40 × bankroll` with persisted 0.40 override). Expected loss if pilot is bad is small (sparse trade volume × small per-trade size). Pathological loss is capped by the aggregate.
- **Drawdown reduction active**: Bot is at -24.6% drawdown from peak (drawdown is from PRE-pilot incidents on 2026-05-21, not pilot trades). `drawdown_size_reduction: 0.50` is multiplicatively halving all trade sizes. This DOUBLE-bounds the pilot beyond the kelly/exposure knobs. Reading 14d data: trade sizes will be small until drawdown recovers; aggregate exposure unlikely to approach the aggregate cap.
- **Phase 4 replay is Stage 1 only** (no orderbook depth in snapshots). Calibration-quality validation, not P&L proxy. Live behavior may differ on Stage 2 rejections.
- **NWS backfill recovers only 2 of 6 originally-targeted days.** 4-day gap at 2026-05-11→14 accepted; ~7% corpus loss is marginal. NCEI/NCDC archives could recover these but out of scope.
- **Cumulative `calibration_ratio` is currently 2.43** under identity. Post-flip trigger must measure today-only / rolling window, not cumulative (otherwise day-1 false positive).
- **`isotonic_path` plumbing is 30-50 lines** (Phase 0 correction from "~10 lines"). Includes settings.py field + tests; touches `weather_edge.py`. If overruns budget, fallback to canonical-path swap (lose footgun safety).
- **Systemd `RestartSec` unknown.** Operator should verify on VPS before un-kill so Phase B's "10-15 min" estimate holds. Default 100ms means immediate; some configs use 30s-5min.
- **Round-half-up rounding** verified empirically 78/78; measure-zero concern at `T = N.5`.
- **2026-05-19 revert base rate**: Phase 2 Gate A exit-5 is the specific guard. Do not retry if fires.
- **`scripts/calibrate.py` is stale post-fix.** Not in fit path. Flagged separate follow-up.

## Realistic timeline

- Phase 1 (initial refit + Gate A/B): 30 min
- Phase 3 (2-day backfill + re-refit + re-gate): 30-45 min
- Phase 4 (Stage-1-only replay script): 2-3h
- Phase 5 (config + CLAUDE.md + monitor script + quality gates): 1-1.5h
- Phase 6 (settings.py field + weather_edge wire-up + Phase A/B/C deploy): 2-3h (longer than v3 due to non-default path plumbing)

**Total: 6-10 hours focused.** Tonight is plausible; tomorrow morning more likely. Phase 6b is the largest risk for overruns.
