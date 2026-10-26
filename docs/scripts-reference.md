# Scripts Reference

Catalog of the scripts under `scripts/`. Each entry states what the script does, what it reads and writes, and any constraint that is not obvious from the name.

**Sources of truth**

- Flags and defaults: run `uv run python scripts/<name>.py --help`. This file does not list flags, except where a flag changes safety behavior.
- Full behavior: the script's module docstring.
- Commonly used commands: the **Commands** section of `CLAUDE.md`.

**Maintenance rules**

- `tests/test_scripts_reference.py` fails if a script is missing from this file, or if this file names a script that does not exist. Add a row in the same change that adds a script. Delete the row in the same change that deletes it.
- Keep rows descriptive and stable. Do not put measured results, sample counts, or dated findings here. Put them in `docs/research/` and link to them.

**Conventions**

- Run scripts from the repository root with `uv run python scripts/<name>.py`. `scripts/reconcile_fills.py` runs as `uv run python -m scripts.reconcile_fills`.
- `--pull-vps` copies the latest state files (the trade log lands in `data/vps_trades.jsonl`) from the host in `ORACLE_VPS_HOST`. Set it in the environment or `.env` (see `scripts/_vps.py`). `--refresh` re-fetches settlements from Kalshi. Neither is needed for a repeat view of cached data.
- Scripts that can rewrite data default to a dry run, or require an explicit `--out`, `--write`, `--execute`, or `--apply` flag. Scripts marked **live-money artifact** write a file that production loads.

## Reporting and dashboards

| Script | Responsibility |
|--------|---------------|
| `scripts/shadow_dashboard.py` | Daily P&L dashboard: daily table, per-city, per-side, per-strategy, and edge-calibration breakdowns, with optional JSON export (`--json PATH`, e.g. `data/dashboard_report.json`). Reads `data/shadow_settlements.full.json` (written by `fetch_shadow_settlements.py`) and the trade log. `--mode` and `--strategy` slice the report. The per-strategy table shows order-price P&L. It is not the favorite_bias kill-criteria view. |
| `scripts/daily_trade_report.py` | Daily post-mortem and forward-risk report. Sections: settled P&L with per-loss breakdown, settled P&L by strategy, pending-trade risk scorecard, calibration gap, boundary-filter kill watch, Phase 0 depth probes, intraday EMOS HIGH, and per-day cap blocks. `--mode` defaults to `live`. The weather sections (settled P&L, `settled_per_cell`, scorecard, calibration gap) are scoped to `WEATHER_STRATEGY_NAMES` unless `--strategy` is set, because model-free strategies trade the same series. The by-strategy section covers every strategy. The calibration gap spans all settled dates, not only `--date`. |
| `scripts/favorite_bias_report.py` | One-stop favorite_bias report: calibration gap with a sign-inversion tripwire, settled P&L and ROI, the kill-criteria verdict, and a fill-timing cohort view. Settled views are filtered to `strategy == 'favorite_bias'` and the live `[band_low, band_high)` mid band, and are never filtered by the g(mid) map. They read `data/shadow_settlements.full.json`, not `state/trades.jsonl`. The calibration section joins audited buys to the settled result in the same file first, and to snapshot NWS-CLI settlements only for markets that file does not resolve; it prints the newest outcome date per source, so a stale file is visible. `n_backfilled` counts settled rows whose `settlement_source` is `nws_cli_backfill` rather than a Kalshi result. |
| `scripts/favorite_bias_spread_cap_report.py` | Paired check of the favorite_bias `max_spread` gate from live audit quotes. Per event, it compares a one-contract entry at the first in-band audit row (no cap) with the first row whose spread is at or under the cap (none = no trade), at the ask with the taker fee. It prints a retrospective block and a forward block (from `--forward-from`), each with a day-clustered CI and the counts of dropped and delayed entries. The forward block gives a verdict at n ≥ 40 changed events. It works with the gate off. Outcomes come from public `/markets/{ticker}` (or `/historical/markets/{ticker}`), cached in `data/audit/fb_market_results.json`. |
| `scripts/intraday_asos_shadow_report.py` | Per-bin reliability report for the parked `intraday_asos_high` strategy. Deduplicates to one row per settlement event, evaluates the verdict only on the sharp-σ band, and uses Wilson confidence intervals. Calibration is the headline metric. Any P&L shown is labeled non-decision-grade. |
| `scripts/intraday_shadow_report.py` | Calibration report for the parked `intraday_emos_high` strategy. Joins `state/predictions.jsonl` to NWS settlements. Calibration is the headline metric. P&L is labeled non-decision-grade and fill-optimistic. |
| `scripts/post_flip_monitor.sh` | Pre-committed rollback-trigger checks written for the isotonic_v2 flip. It builds on `daily_trade_report.py --json` and `state/eval_audit.jsonl`. Its thresholds have not been revised since that flip. Read them before reusing it for another calibrator change. |

## Settlement data

| Script | Responsibility |
|--------|---------------|
| `scripts/fetch_shadow_settlements.py` | Fetches Kalshi settlement results for every ticker in the trade log and writes `data/shadow_settlements.json` and `data/shadow_settlements.full.json`. `shadow_settlements.json` is **append-only** (`merge_settlements`): Kalshi returns 404 for old markets, so a cached settled outcome always wins over a fetch that lost it. Without this, cumulative cohorts become a rolling window. |
| `scripts/backfill_settlements_from_snapshots.py` | Reconstructs settlement outcomes that were lost before the store became append-only. Derives each outcome from NWS CLI snapshot data with the same `derive_outcome_yes` path the favorite_bias calibration section trusts. Safety: aborts if any already-settled ticker disagrees with its derived outcome; `--strategy` (repeatable) scopes which tickers are rebuilt and defaults to all strategies with a warning; dry run unless `--write`; timestamped backup before writing; atomic write. Prints month-by-month counts so skewed agreement coverage is visible. After `--write`, run `favorite_bias_report.py --refresh` to regenerate `.full.json`. |
| `scripts/reconcile_settlements.py` | Compares the result implied by the NWS CLI temperature with Kalshi's actual result for recently settled weather markets. A mismatch can mean Kalshi changed its settlement station, NWS revised a report, or the parser missed a value. Investigate before trusting calibration data. `--live` fetches settled markets from the Kalshi API. |
| `scripts/heal_settlements.py` | Repairs snapshot `settlements.json` files corrupted when morning partial-day CLI reports (AUS, DEN) were stored as final. Keeps one winner per `(city, observation_date)` by greatest `(issued_at, product_id)`. Orphan groups with no clean record are dropped. Dry run unless `--execute`. Takes a `flock`, backs up every file, writes atomically, and skips today's directory. Idempotent. |

## Data collection

| Script | Responsibility |
|--------|---------------|
| `scripts/collect_daily_snapshot.py` | Collects open weather markets, multi-model ensemble forecasts, NBM point forecasts, and NWS CLI settlements into `data/snapshots/<date>/` (or `<date>/<run-id>/` with `--run-id`). Snapshots are local, gitignored, and not redistributed because of Kalshi's data terms; schedule the script yourself. |
| `scripts/collect_backtest_data.py` | Fetches closed weather markets, candlestick history, and GFS forecasts into `data/backtest/` for the legacy backtest data source. Needs Kalshi credentials in `.env`. |
| `scripts/backfill_forecast_model_field.py` | Adds `model` and `model_version` to old `weather_forecasts.json` records that lack them (they are GFS-only). Idempotent. `--dry-run` previews. |
| `scripts/refresh_asos_corpus.py` | Incrementally extends the per-city ASOS hourly CSVs in `data/research/asos/`, pulling only new IEM rows and deduplicating by `(station, valid)`. Keep request cadence modest, because IEM rate-limits. |
| `scripts/pull_post_mortem_data.py` | Builds a post-mortem bundle for one date under `data/post_mortems/<date>/`: signal logs from the VPS, operational status over SSH, Kalshi `/portfolio/fills`, a reconciliation of the local trade log against the broker, and a manifest. `--no-pull` skips the VPS pull. |

## Calibration

| Script | Responsibility |
|--------|---------------|
| `scripts/calibrate.py` | Offline reliability diagnostics over local `data/snapshots/`: Brier with Murphy decomposition, ECE, calibration curves with Wilson CIs, PIT histogram, per-city bias, lead-time and strike-type slices. Writes `data/calibration_results.json`. Does not fit calibrators, derive plateau bounds, or touch `state/eval_audit.jsonl`. |
| `scripts/fit_isotonic.py` | Fits the isotonic calibrator and writes a JSON lookup table (default `data/calibration/isotonic.json`). **Live-money artifact.** Write a candidate to another path, validate it with `validate_calibrator.py`, then promote it. Requires `--extra ml`. Runs the input-contract preflight before fitting. |
| `scripts/fit_platt.py` | Fits Platt-scaled calibrators: one global (`platt.json`) and one per strike type (`platt_per_strike_type.json`). Shares the corpus builders and the input-contract preflight with `fit_isotonic.py`. |
| `scripts/validate_calibrator.py` | Pre-deploy gate for an isotonic candidate. Holds out the last `--holdout-days` of forecast dates, refits on the rest, and scores the holdout. The load-bearing gate is that the candidate beats raw-input Brier. The live baseline comparison is advisory. Also checks plateau width and output range on the candidate artifact. Requires `--extra ml`. |
| `scripts/validate_plateau_bounds.py` | Finds dangerous flat plateaus in an isotonic artifact and derives `block_raw_prob_range`. Mirrors `check_calibrator_plateau_coverage` in `python/analytics/audit_tripwires.py`. Re-run on every refit. |
| `scripts/run_calibrator_harness.py` | Evaluates candidate calibrators (identity, isotonic, EMOS) per `(city, var)` cell on a pre-registered hold-out split, using BSS and log skill with bootstrap CIs. Writes `data/calibration/per_cell_winners.json`. On demand only. It never edits `config.yaml`. Verdicts: `winner`, `insufficient_evidence`, `worse_than_identity`, `insufficient_n`, `insufficient_diversity`, `no_separation`. |
| `scripts/_calibration_dry_run.py` | Preflight that checks corpus coverage against the input contract without fitting. Its only write is the data-availability calendar, and only with `--regenerate-calendar`. Exit code 1 on any abort condition. |

## Bias correction

The ERA5 pair builds the existing table. The NWS pair builds and validates a table anchored to the NWS CLI values that Kalshi settles on.

| Script | Responsibility |
|--------|---------------|
| `scripts/collect_gfs_bias_data.py` | Downloads GFS deterministic forecasts and ERA5 reanalysis actuals from Open-Meteo into `data/gfs_bias/`. |
| `scripts/analyze_gfs_bias.py` | Computes per-city, per-variable, per-month GFS bias against ERA5 and writes `data/gfs_bias/bias_correction.json`. |
| `scripts/build_nws_bias_corpus.py` | Pairs lead-1 GFS and ECMWF ensemble means from the snapshot corpus with final NWS CLI settlements. Writes `data/nws_bias/forecasts.csv` and `nws_actuals.csv`. Hermetic: makes no API calls. |
| `scripts/fit_nws_bias.py` | Fits `data/nws_bias/bias_correction.json` from that corpus, in the same schema the runtime `BiasTable` loads. Omits EMOS-enabled cells, because the EMOS affine term already absorbs a constant offset. `--exclude-cells` overrides. |
| `scripts/validate_nws_bias.py` | Walk-forward out-of-sample test of the NWS table against no correction and the ERA5 table, using production correction code. A cell is promoted only if it beats both on test MAE. |

## EMOS

| Script | Responsibility |
|--------|---------------|
| `scripts/build_emos_corpus.py` | Builds `data/emos/corpus.parquet` from snapshots. Keeps the latest run per snapshot day and merges settlements with `prefer_latest_settlement`, without dropping preliminaries. GFS only unless `--include-ecmwf`. `ens_mean` and `ens_std` are recomputed with `ddof=1` to bypass Open-Meteo's `ddof=0`. Do not recompute them downstream. |
| `scripts/fit_emos.py` | Offline EMOS fitter. Fits per-`(city, var)` coefficients pooled across leads. Writes `data/emos/coefficients.json` and `coefficients.diagnostics.json`. Hermetic: imports no production runtime code. `--cluster-by-date` gives a date-cluster bootstrap, which is required before any EMOS re-promotion (`docs/calibration-runbook.md`). |
| `scripts/compare_emos_coefficients.py` | Shadow A/B replay of candidate versus production coefficients on `state/eval_audit.jsonl`, `state/forecasts_raw.jsonl`, and snapshot settlements. Reuses `EmosTable.load` and `emos_probability`, so the replay matches runtime. Reports per-cell Brier, bootstrap BSS, and coefficient drift. `--cluster-by-date` gives a date-cluster bootstrap, which is required before any EMOS re-promotion. |
| `scripts/analyze_emos_shadow.py` | Shadow analyzer that applies pre-registered decision rules and emits a promote-or-hold verdict. Writes the reproducibility envelope `data/emos/reconciliation_envelope.json`. `--decision-mode`, `--start-date`, and `--end-date` are required. |

## Favorite-bias

| Script | Responsibility |
|--------|---------------|
| `scripts/fit_favorite_bias_map.py` | Fits the g(mid) map (price to P(YES)) on greater/less lead-1 pairs. Each isotonic block is shrunk toward a fitted `logit(g) = a + b·logit(mid)` trend. `--out` is **required**. The live path `data/calibration/favorite_bias_isotonic.json` is a **live-money artifact**. Write a candidate, validate it with `scripts/research/fb_map_walkforward.py`, then replace the live file. Before trusting a fit, check `q_over_df` and `degenerate_pure_trend` in the output. Do not refit on staleness alone. |
| `scripts/build_fb_climatology.py` | Builds `(city, var, month) → (μ, σ, n)` strike-climatology cells and a `(city, var)_pool` fallback into `data/calibration/fb_climatology.json`. Reads both snapshot layouts and prefers final over preliminary settlements. `--as-of` supports leave-future-out replay. Loaded by `python/data/fb_climatology.py`. |
| `scripts/research/fb_map_walkforward.py` | Expanding-window walk-forward validator for the g(mid) map. Scores the live traded band against the market baseline and the live artifact. Reports log loss, Brier, per-contract ROI, and Kelly-weighted P&L with capital deployed. Per-contract ROI alone is nearly blind to a map change. |

## Backtesting

| Script | Responsibility |
|--------|---------------|
| `scripts/run_backtest.py` | Replays historical markets through the weather strategy with production execution rules and prints a performance report. `--data-source legacy` reads `data/backtest/`. `--data-source snapshot` reads the snapshot corpus. Supports parameter sweeps and walk-forward mode. |

## Operations (live trade state)

| Script | Responsibility |
|--------|---------------|
| `scripts/check_live_config.py` | Pre-restart check of the merged config (`config.yaml` + `ORACLE_CONFIG_OVERLAY` + env), run by `deploy/update.sh`. Runs `validate_strategy_allocations` against the strategies the config enables (which `Settings.load()` alone does not), enforces the reviewed sizing ceilings, and prints `full` and `live` SHA-256 hashes of the redacted settings dump. `--expect-hash FILE` exits non-zero when the `live` hash (overlay-settable subtrees only) differs from the pin. `--dump` prints the redacted dump. |
| `scripts/reconcile_fills.py` | Operator CLI over `FillsReconciler.backfill()` for multi-day windows beyond the periodic reconciler's startup safety window. Dry run by default. `--apply` writes the planned rows. `--allowlist` limits which order IDs are written, `--expect-n` aborts if the planned write count differs, and `--max-window-days` caps the window. Combining `--apply` with `--force-strategy` requires an interactive `yes-i-am-sure` confirmation. |
| `scripts/revert_cash_out.py` | Offline escape hatch for cash-out reconciliation: `--reset-dedup`, `--revert`, or `--apply` for one `--order-id`. Stop `oracle-bot.service` first. |

## Intraday research support (strategies parked)

| Script | Responsibility |
|--------|---------------|
| `scripts/build_asos_climatology.py` | Builds the ASOS remaining-gain climatology `data/asos/climatology.json`, keyed by `(city, var, month, hour_utc)`. Applies per-cell and pooled sample gates. Hard-aborts if a source CSV's station differs from the NWS settlement station for the city (CHI settles on MDW, not ORD). The fit window ends at a fixed date in the script and does not extend automatically. |
| `scripts/salvage_frame_v1_predictions.py` | Quantifies how much of a `predictions.jsonl` was logged under the legacy civil-day frame. Recomputes the running extremum `M` under both frames from local ASOS CSVs and flags rows where it differs. Writes a per-row JSON report. |
| `scripts/verify_intraday_replay.py` | Replays `IntradayEmosHighStrategy` on snapshot rows and diffs its decision fields against the v6 audit fixture. Also asserts the gate-input hash. |
| `scripts/_asos_reliability_bins.py` | Shared pure-Python reliability-bin helper for the ASOS reports. Fixed bin edges. |

## KXRAIN

| Script | Responsibility |
|--------|---------------|
| `scripts/kxrain_backfill_settlements.py` | The KXRAIN corpus's only disaster-recovery path after a lapse longer than the daily sweep's reach. Runs `KxRainSettlementSweep` with a higher page cap (`--max-pages`). The settled listing is newest-first, so market-days older than the sweep's reach are lost once they leave it. Idempotent on `(station, settlement_date)` through `state/kxrain_settled_index.json`. `--dry-run` reports index coverage without calling Kalshi. `--reconcile` adds rows already on disk but missing from the index, then exits without calling Kalshi. A row is refused unless `result` is exactly `"yes"` or `"no"`. An open market has `result: ""` and would otherwise record a fabricated NO win. |
| `scripts/capture_kxrain_fixture.py` | One-shot, read-only live capture of KXRAIN API payloads into `tests/fixtures/kxrain/`. Paced to stay under the Kalshi rate limit. |

## Research and one-off scripts

Analyses tied to a past decision. Each is reproducible but not part of any routine. Most reads are hermetic over the snapshot corpus or local ASOS CSVs.

| Script | Responsibility |
|--------|---------------|
| `scripts/research/_fb_data.py` | Shared data layer for the favorite_bias research scripts. Enforces the repo's methodology rules: outcomes through `derive_outcome_yes`, taker fees at the ask, lead from the snapshot calendar day (never from time-to-expiry), final settlements only. |
| `scripts/research/fb_improvements.py` | Measures each backtestable favorite_bias lever with day-clustered bootstrap CIs. |
| `scripts/research/fb_city_exclusion.py` | Tests which `(city, var)` cells are reliably negative under a pre-registered exclusion rule. Runs on the live cohort and the backtest corpus. |
| `scripts/research/favorite_bias_oos_g0.py` | Out-of-sample replay of the live favorite_bias rule on historical Kalshi ladder events, plus a resting-bid (maker) entry study and a replay-vs-live-audit check. `fetch` pulls in stages (catalog, hourly candles, then 1-minute candles and tape for entered markets only) with unauthenticated GETs, one thread, a fixed request spacing, 429 backoff and a hard stop. Raw responses are cached under `data/research/g0/raw/` (gitignored), so `analyze` runs offline. Rules are pre-registered in `docs/research/2026-10-01-favorite-bias-oos-g0-prereg.md`. |
| `scripts/research/fb_second_horizon.py` | Compares a second favorite_bias at another lead (lead-0, lead-1, lead-2), with ask-drift and power analysis. |
| `scripts/research/newcity_favorite_refresh.py` | Re-tests favorite-longshot viability in the newer weather cities on the current corpus. |
| `scripts/research_2026_06_09_centering_test.py` | New-city intraday centering test: do same-day HIGH mid-quotes miss an observation-anchored posterior by more than the executable-cost bar. Includes a control on mature cities. |
| `scripts/research_2026_06_09_newcity_spread_census.py` | Measures the quoted-spread distribution of same-day temperature markets per city. Writes `data/research/newcity_spread_census.json`. |
| `scripts/research_2026_06_10_asos_continuous_gap.py` | Measures the gap between a continuous-high proxy built from 1-minute ASOS and the hourly-sampled max, and its economic effect. |
| `scripts/_make_step0_slim_fixture.py` | One-shot builder for the slim MIA snapshot fixture under `tests/research/fixtures/snapshots_step0_slim/`. |

## Shared internals

Not run directly. Imported by the scripts above.

| Module | Responsibility |
|--------|---------------|
| `scripts/_calibration_common.py` | Corpus loaders, `prefer_latest_settlement`, production-mirrored pair builders, calibration tables, and the input-contract preflight entry point. |
| `scripts/_calibration_input_contract.py` | Input-contract preflight and per-pair provenance for the calibration fitters. Stdlib only. |
| `scripts/_vps.py` | VPS host, remote state directory, and a best-effort pull routine that skips missing remote files. Behind every `--pull-vps` flag. |
