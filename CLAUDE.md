# CLAUDE.md

## Project Overview

The Oracle is a prediction market trading bot targeting Kalshi (CFTC-regulated, US-accessible). It uses Python for the strategy engine, execution, and the analytics pipeline.

## Commands

```bash
# Extras: dev (pytest/ruff/mypy), telegram, ml (scikit-learn/scipy/pyarrow)
uv sync --extra dev

# Run the orchestrator
uv run python -m python.orchestrator

# Run the TUI dashboard standalone
uv run python -m python.dashboard

# Lint / format / type check / test (see pyproject.toml)
uv run pytest

# Shadow trading P&L dashboard (instant, reads cached data)
uv run python scripts/shadow_dashboard.py

# --pull-vps copies state from the host in ORACLE_VPS_HOST (user@host, in the environment or .env)
# Pull latest trades from VPS + refresh settlements + export JSON
uv run python scripts/shadow_dashboard.py --pull-vps --refresh --json data/dashboard_report.json

# Fetch settlements from Kalshi API (needed before first dashboard run)
uv run python scripts/fetch_shadow_settlements.py --pull-vps

# Daily trade report: settled P&L post-mortems + pending risk scorecard + calibration gap
uv run python scripts/daily_trade_report.py --pull-vps --refresh
uv run python scripts/daily_trade_report.py --date 2026-05-01  # specific date
uv run python scripts/daily_trade_report.py --json /tmp/report.json  # JSON export

# Build EMOS training corpus from snapshots
uv run python scripts/build_emos_corpus.py

# Build ASOS remaining-gain climatology artifact (intraday_asos_high)
uv run python scripts/build_asos_climatology.py

# Refresh ASOS hourly research corpus (all 16 cities, incremental; run before weekly new-city lock-in analyses)
uv run python scripts/refresh_asos_corpus.py

# Intraday ASOS shadow reliability report (per-bin; low-p NO bin kill-criterion)
uv run python scripts/intraday_asos_shadow_report.py

# Recalibration pipeline (isotonic + Platt fits, plateau validation)
uv run python scripts/calibrate.py

# Fit isotonic calibrator
uv run python scripts/fit_isotonic.py

# Fit Platt-scaled calibrators
uv run python scripts/fit_platt.py

# Validate and derive plateau filter bounds
uv run python scripts/validate_plateau_bounds.py

# Run full backtest on historical weather markets
uv run python scripts/run_backtest.py --start-date 2026-03-01 --end-date 2026-04-30

# Collect daily market snapshot
uv run python scripts/collect_daily_snapshot.py

# Reconcile settlement data with trades
uv run python scripts/reconcile_settlements.py

```

## Architecture

All Python source lives under `python/`. The package is imported as `python.*` (e.g., `from python.strategies.base import Strategy`).

### Module Map

Moved to **`python/CLAUDE.md`**, which loads automatically when working under
`python/`. Read it before touching runtime code — many rows carry non-derivable
operational constraints (fail-open / never-raises contracts, incident
references). `scripts/*` behavior is in **`docs/scripts-reference.md`**.

### Data Flow

```
Orchestrator loop:
  1. Check kill switch (state/STOP file)
  2. Run circuit breaker checks per strategy
  3. Each enabled Strategy.generate_signals() → list[TradeSignal]
  4. RiskManager.check_signal() validates each signal (exposure, loss, drawdown)
  5. ExecutionEngine.execute() places order or logs shadow trade
```

### Key Patterns

- **All I/O is async** — `aiohttp` for HTTP and the Kalshi WebSocket, `httpx` for data fetching
- **Strategies extend `Strategy` ABC** — implement `scan()`, `evaluate()`, `generate_signals()`
- **`TradeSignal` is a frozen dataclass** — immutable, passed from strategy → risk manager → execution engine. The `confidence` field is the ensemble spread-skill factor for weather; `favorite_bias` sets it to its model probability and `intraday_asos_high` to `1.0`.
- **Three execution modes** controlled by the `ORACLE_EXECUTION_MODE` env var (wins) or `execution.mode` in `config.yaml` (default `shadow`): `shadow`, `paper`, `live`
- **Config layering**: `python/config.yaml` for strategy/risk parameters, `.env` for secrets and env-specific settings, both unified in `python/settings.py`
- **Effective edge** (weather): Two-stage edge calculation. **Stage 1** (pre-filter): `effective_edge = raw_edge * ensemble_confidence - half_spread`, where `ensemble_confidence = 1/(1 + k * ensemble_std)` and `half_spread = bid_ask_spread / 2`. Controlled by `spread_skill_sensitivity` (k) and `max_spread` config values. **Stage 2** (orderbook depth check): After Kelly sizing, fetches the full orderbook via `get_orderbook()`, computes VWAP fill price by walking ask levels, and recomputes edge as `fill_edge = (model_prob - vwap_fill_price) * ensemble_confidence` for YES (or `(1 - model_prob - vwap_fill_price) * conf` for NO). Kelly is re-run at the fill price to correct position sizing. Markets with stale/empty books (`available < min_book_depth`) are rejected. Falls back to best-level depth cap if the orderbook API call fails. Controlled by `orderbook_depth_check` and `min_book_depth` config values.
- **Limit-order pricing**: All Kalshi orders are submitted as `order_type="limit"` with a 120s server-side TTL (`order_ttl_seconds`). The submitted limit price is computed by `compute_limit_cents()` in `python/execution/engine.py`: spreads ≥2¢ rest inside the spread biased toward the bid (maker-eligible); 1¢ spreads cross to the ask (deterministic taker, guaranteed fill if depth). `is_taker` and per-fill `fee_cents` from `/portfolio/fills` are recorded on each fill (`OrderRecord.is_taker_fills`, `is_maker_fills`, `fees_from_api`) for downstream P&L accounting and edge-math calibration.
- **Kalshi fee integration**: Exchange fees are deducted from edge calculations, Kelly sizing, backtest P&L, and settlement P&L. Fee formula: `ceil(multiplier × C × P × (1−P) × 100) / 100`; taker multiplier=0.07 (index series 0.035). **Makers pay nothing on weather:** every Climate and Weather series (all KXHIGH*/KXLOWT*/KXRAIN the bot trades or observes) is `fee_type: quadratic` (taker-only), verified via `GET /series` on 2026-10-01; only ~160 of ~14.5k series are `quadratic_with_maker_fees` (maker multiplier 0.0175, none of them weather). The author's live fills agree: 128 pure-maker KXHIGH/KXLOWT fills paid $0.05 total (rounding only) vs ~$5.74 at 0.0175. `fees.py` charges the maker rate unless the `ticker` argument is in `MAKER_FEE_FREE_SERIES` (unknown series stay conservative). All live fee estimates (favorite_bias, weather_edge, intraday) use the taker rate. Rounding: `order_fee` ceils each order to the cent; Kalshi bills at $0.0001 precision (observed ~June 2026), so the formula overstates by < 1¢/order — settlement P&L uses Kalshi-reported `fees_from_api` and falls back to the formula only when it is absent. Multipliers are from the Feb 2026 fee schedule — **monitor https://kalshi.com/fee-schedule and series `fee_type` for changes** as fee increases directly erode edge and may require threshold adjustments.
- **Weather market parsing**: Uses Kalshi API fields (`strike_type`, `floor_strike`, `cap_strike`) instead of regex. Supports `greater`, `less`, and `between` market types. Prices read from `yes_bid_dollars` / `yes_ask_dollars` (API v2 format).
- **Isotonic calibration** (weather, default for non-EMOS tradeable cells): Maps raw ensemble fractions to calibrated probabilities via pre-fitted isotonic regression. Fitted offline by `scripts/fit_isotonic.py` (requires `--extra ml`). At runtime, `python/data/calibration.py` applies numpy-free linear interpolation — no sklearn dependency in production. Current artifact: **isotonic_v3** at `data/calibration/isotonic_v3.json` (fit 2026-07-02, deployed 2026-07-04, n=12540, +8.62% CV Brier — `improvement_pct` in the artifact; re-anchored ERA5→NWS). The plateau-block filter (`block_raw_prob_range`, auto-derived by `scripts/validate_plateau_bounds.py`) rejects raw probabilities in dangerous flat plateaus of the live artifact; it is **currently `null`** — the v2 plateau dissolved at v3's larger corpus, so nothing is blocked on that axis. Re-derive it on any refit. `min_calibrated_prob: 0.110` floor applies regardless of calibrator. Refit cadence and tripwires live in `docs/calibration-runbook.md`.
- **Per-cell EMOS calibrator dispatch** (weather, Stage A): For cells listed in `weather.emos_enabled_cells` (currently **empty** — all-isotonic since 2026-06-05; LAX/low suspended) the live calibrator is `emos_prob_raw` from `data/emos/coefficients.json`; all other cells use isotonic. Dispatcher lives in `weather_evaluation.py:evaluate_market_pure`, fires after the EMOS shadow block (so the same `emos_prob_raw` field is computed on every cell) and before the `min_calibrated_prob` floor (so the floor applies to EMOS output too). Per-cell promotion gate: `n≥10` shadow rows, `BSS_emos_vs_iso > 0` with 95% bootstrap CI lower bound `> 0`, `LogSkill_emos_vs_iso > 0` with CI lower bound `> 0` (proper, knob-invariant — catches tail mis-calibration squared-error BSS under-weights), no `unstable_b`/`unstable_d`/`d_at_zero` coefficient warnings, and no `incompatible_sigma` warning (non-overridable structural veto: `σ_emos(p99) / bracket_width > 2.0` blocks promotion regardless of BSS or operator override). The `CALIBRATOR_DISAGREEMENT` gate is bypassed when `live_calibrator_kind == "emos_raw"` (EMOS disagrees with raw by construction); `isotonic_fallback` rows still get gate protection. Audit rows carry `live_calibrator_kind ∈ {isotonic, emos_raw, isotonic_fallback, identity}`. Single-flag rollback: empty `emos_enabled_cells` reverts to all-isotonic, bit-identical.
- **City/variable exclusion** (weather): The `excluded_city_vars` config list (currently `["DEN_low", "DEN_high", "SEA_high", "LAX_high", "AUS_high", "CHI_high", "CHI_low"]`) blocks untradeable city/variable combinations. DEN/* and SEA/high are the EMOS d=0 group, structurally blocked until a structural fix lands. LAX_high was on EMOS briefly but re-excluded after live ROI=−39.6% (d=7.19 → EMOS σ=6.8°F incompatible with 1°F brackets); LAX/low was the last live EMOS cell but was **suspended → isotonic on 2026-06-05** (0/3, −102% ROI; σ≈1.9°F vs 1°F brackets, live `unstable_d` warning, ~2.5°F cold μ bias — same σ-incompatibility failure as LAX_high). AUS_high / CHI_high / CHI_low added as pre-emptive exclusions under isotonic_v2. (Re-eval moot while weather_edge is shadow-only — the excluded cells don't trade regardless.)
- **Per-day event cap** (weather): The `PerDayEventCap` in `python/strategies/per_day_event_cap.py` allows at most `weather.max_bets_per_day_per_event` (default `1`) commits per `(city, var, forecast_date)` per UTC day, persisting across scans by hydrating from `state/trades.jsonl` on startup + UTC-midnight transitions. Counts distinct `order_id`s (placement + fill for one order = 1 commit). Within a single scan, ranks candidates by `(effective_edge desc, confidence desc, ticker asc)` and takes the top-K where K = remaining day budget. Blocked candidates are recorded to `state/per_day_cap_blocks.jsonl` with full would-have-been-signal metadata for retroactive backtesting; rejection reason `PER_DAY_EVENT_CAP_HIT`, funnel field `n_per_day_cap_blocked`. Winner metadata carries `event_candidates_count`, `event_dropped_edges`, `event_dropped_tickers` (backward-compatible with daily report). Rollback: set `max_bets_per_day_per_event: null` and restart. Shipped 2026-05-21.
- **Favorite-bias strategy (model-free; ships `enabled: false`):** `python/strategies/favorite_bias.py` buys YES on Kalshi greater/less weather favorites (market mid in the configured `band_low`–`band_high` window) at lead-1 (day before observation). The favorite-longshot lift is confined to mid ≥ 0.50; below that the contract is an underdog and the lift is absent. An out-of-sample test did not confirm a forward edge (`docs/research/2026-10-01-favorite-bias-oos-g0-results.md`), so do not size up on in-sample results. Model-free: uses a pre-fitted isotonic g(mid) map (`data/calibration/favorite_bias_isotonic.json`, produced by `scripts/fit_favorite_bias_map.py`) instead of GFS forecasts. All signals carry `execution_taker: true` so `compute_limit_cents` crosses to the ask unconditionally (`_TAKER_ALWAYS_CROSS_CENTS = 100`). With `max_book_walk_cents > 0` the strategy reads the orderbook and sets `metadata.limit_price` at the cheapest ask level covering the Kelly size (bounded, each level must still clear `edge_floor`); the engine submits that instead of the frozen top ask. Fail-open to the top ask. The orchestrator's `apply_favorite_bias_deference` hook drops any favorite_bias signal that shares a `(city, var, forecast_date)` event key with a same-tick weather_edge signal or an open position, preventing stack-up. Config block under `strategies.favorite_bias`; enabling it also needs a `risk.strategy_allocations.favorite_bias` entry. **The per-strategy sizing knobs are only the upper of two caps** — `RiskManager.check_signal`'s per-market cap (`risk.max_single_market_pct × effective_max_total_exposure_pct × bankroll`) and drawdown clamp can bind instead and set every position. Never credit a sizing change from a config diff; read the filled dollars in `state/trades.jsonl`. Note `drawdown = 1 − bankroll/peak_bankroll` measures *equity*, so a withdrawal is indistinguishable from a loss — re-anchor `peak_bankroll` after any transfer (`docs/operations-runbook.md`). It also logs strike-climatology features (`z_strike`/`q_clim`, `python/data/fb_climatology.py`) on every audit row — suppress-nothing instrumentation for a future model-as-veto edge — plus a **default-off** tail-strike guard (`max_strike_climatology_z: null`; variance insurance, no demonstrated EV). Report script: `scripts/favorite_bias_report.py`.
- **Intraday EMOS HIGH (weather) — PARKED (`enabled: false`):** `python/strategies/intraday_emos_high.py` trades same-day KXHIGH from a later-snapshot EMOS posterior. Parked because its `predictions.jsonl` logging is edge-selected (only Stage-1 survivors are recorded), so it cannot fill the low-p NO bin a live flip needs, and `intraday_asos_high` covered the same lane with real observations. `enabled: false` skips construction; un-park = `enabled: true`, which returns it in shadow mode (`shadow_only` + the fail-CLOSED `apply_intraday_shadow_drop` hook).
- **Intraday ASOS nowcast (weather) — PARKED (`enabled: false`, allocation entry removed):** `python/strategies/intraday_asos_high.py` trades the raw ASOS-observation posterior (observed max-so-far + remaining-gain climatology, no learned calibrator) behind physical guards; its orchestrator hooks (`apply_intraday_asos_deference`, fail-CLOSED `apply_intraday_asos_shadow_drop`) remain in code. A 44-day forward corpus falsified it: raw posterior Brier 0.1834 vs climatology 0.1806 vs market 0.1185, realizable ROI negative — the posterior disagrees with the book exactly when the book is right. Do not re-pitch a recalibrator: at the post-peak decision the posterior is bimodally extreme, so there is no calibrated moderate-confidence moment to relocate bets into. **Un-park = `enabled: true` AND re-add the `risk.strategy_allocations` entry** (an orphaned allocation key crash-loops the bot; a missing one makes `book_for()` fall back to full equity). Keep `strategies.intraday_asos_high.asos_obs_cache_ttl_seconds` (default 600s, bounds [60,3600]) at ASOS's ~hourly cadence — a 120s refetch caused an IEM 429 storm. Evidence: `docs/research/2026-07-19-new-data-edge-audit.md`.
- **Boundary-distance suppressive filter (weather, Path 7a):** `weather.boundary_filter_cells` (cell → °F threshold) drops BETWEEN candidates whose blended forecast mean (`weather.blended_forecast_mean()`) sits within the threshold of the nearest NWS-rounding boundary — the grazer zone where integer rounding + 1°F brackets force ~100% losses. Fires in `evaluate_market_pure` after `block_raw_prob_range`, before `min_raw_prob_for_between`; emits `RejectionReason.BOUNDARY_FILTER_BLOCK` (code 35). Enabled in the shipped config for **MIA/low @ 0.5°F** (since 2026-05-29); default-off (`{}`) for all other cells. Single-flag rollback: empty the dict + restart.
- **Calibrated-probability floor block (weather):** `min_calibrated_prob` config rejects candidates with post-calibration probability below the floor in live + paper modes; shadow is exempt for continued validation. Applies uniformly to whichever calibrator fired (isotonic OR EMOS) — dispatcher runs first, floor second. The isotonic artifact has a flat floor at ~0.088/0.092 that swallows distinct raw inputs in [0.03, 0.12] (asymmetric small-win/big-loss economics); EMOS is continuous so the floor cuts a different slice there. Current value: `strategies.weather.min_calibrated_prob`.
- **NWS settlement collection**: `python/data/nws.py` fetches official high/low temperatures from NWS Daily Climate Reports (CLI products) — the exact source Kalshi settles weather markets on. The parser detects preliminary (afternoon same-day "VALID TODAY") vs. final (next-morning "VALID FOR YESTERDAY") reports. Retry logic with exponential backoff on 429/500/503. City-to-CLI-location mapping in `CITY_TO_CLI_LOCATION` (e.g., CHI→MDW). Collected daily via `scripts/collect_daily_snapshot.py` → `data/snapshots/<date>/settlements.json`.
- **Daily snapshots**: `scripts/collect_daily_snapshot.py` collects open markets + GFS forecasts + NWS settlements (NWS fetch runs concurrently with Kalshi/weather fetches; timed ~2 h after each GFS init when scheduled). Output: `data/snapshots/<date>/markets.json`, `weather_forecasts.json`, `settlements.json`, plus `index.jsonl`. `data/snapshots/` is gitignored and not distributed: it holds raw Kalshi data, which Kalshi's developer terms do not allow redistributing. Collect your own; a private deployment can symlink it from an ops checkout (`deploy/update.sh`).
- **Kalshi rate limits**: Basic tier is 20 reads/sec, but there's an undocumented sliding window that triggers 429s after ~100 requests with a 5-10 min cooldown. `iter_all_markets` paginates internally (multiple requests per call). Collection scripts use exponential backoff (15s→120s). Collect data in batches of 3-4 series with gaps between runs.
- **Telegram bot**: Optional (`uv sync --extra telegram`). `TelegramNotifier` runs as a background asyncio task using long polling (outbound HTTPS only — no inbound ports on VPS). Owner-only auth: every command handler checks `chat_id == ORACLE_TELEGRAM_OWNER_CHAT_ID`; non-owner messages silently ignored. Push alerts for trade fills (batched per tick), circuit breaker trips, kill switch activation, strategy errors, startup/shutdown; rate-limited (20 msg/min, critical alerts bypass). All Telegram errors are swallowed — never crashes the orchestrator. Config: `telegram.enabled` in config.yaml, `ORACLE_TELEGRAM_BOT_TOKEN` + `ORACLE_TELEGRAM_OWNER_CHAT_ID` in `.env`. **Full owner-only command reference: `docs/operations-runbook.md`.**
- **Shadow P&L dashboard**: `scripts/shadow_dashboard.py` provides daily P&L breakdown from shadow trading data. Reads `data/shadow_settlements.full.json` (pre-computed by `scripts/fetch_shadow_settlements.py`). Shows: summary stats (P&L, ROI, win rate, streaks, best/worst day), daily table (trades, W/L, P&L, fees, wagered, cumulative P&L), per-city breakdown, edge calibration buckets. `--pull-vps` copies the latest trades from the host in `ORACLE_VPS_HOST` over SCP, `--refresh` re-fetches Kalshi settlements, `--json PATH` exports structured JSON with `summary`, `daily` (with per-day `by_city`/`by_side`), `by_city`, and `edge_calibration` sections. `--mode {shadow,live,paper,all}` (default `all`) slices the report by execution mode — useful when shadow and live rows share one `state/trades.jsonl` log. Typical workflow: run `--pull-vps --refresh` to get fresh data, then bare command for instant repeat views.
- **Trade-log row filter** (`is_pnl_relevant_row`): Every consumer of `state/trades.jsonl` MUST use `python.execution.trade_log.is_pnl_relevant_row()` instead of writing its own `event == "fill"` check. This is the schema-evolution gate — adding a new event type only requires updating the helper.
- **Trade-log fidelity**: `state/trades.jsonl` is treated as a write-behind cache. The `FillsReconciler` (every 5 ticks in live mode) reads `/portfolio/fills` from Kalshi and back-fills any synthetic rows missing from the local log, enforcing the dedup invariant (at most one `event=="fill"` row per `order_id`) inside an `asyncio.Lock` shared with the engine. The placement-row event (`event="placement"`) is the durable strategy-attribution side-channel for synthetic rows after process restart. Operator escape-hatches: `/disable_reconciler` (sub-second, process-local) and `scripts/reconcile_fills.py` (multi-day backfill beyond the periodic reconciler's startup safety window).
- **Manual-close reconciliation** (FillsReconciler, default-off behind `manual_close_enabled`): classifies `/portfolio/fills` rows; `SELL_MANUAL_CANDIDATE` triggers cost-basis lookup + `RiskManager.close_position(pnl=realized_pnl)` + `record_strategy_trade` (so per-strategy circuit breakers see manual-close P&L) + `OrderStatus.EXTERNALLY_CLOSED` stamping + WARNING Telegram. See runbook "Manual-close and cash-out classification" for rollout/ops detail. Known v1 limitation: a crash between trade-log append and `close_position` leaves a small `_daily_pnl` attribution gap (Kalshi-authoritative bankroll self-heals on next hydrate).
- **Cash-out reconciliation** (FillsReconciler v1.1, default-off behind `cash_out_apply_enabled`): classifies opposite-side `action="buy"` fills as `FillKind.BUY_CASH_OUT` (position-aware) and routes them through `_apply_cash_out`. Full closes call `RiskManager.close_position` + stamp `EXTERNALLY_CLOSED`; partial closes call `RiskManager.realize_partial` + decrement `OrderRecord.fill_count` proportionally so the residual settles via existing `_settle_orders`. Writes `event="cash_out"` rows; `cost_basis_for_position` subtracts prior cash_out offsets. P&L safety cap `cash_out_max_pnl_apply_dollars: 500.0` falls back to shakedown + CRITICAL alert on breach. Escape-hatch: `scripts/revert_cash_out.py` (offline). Known limitations: cold-start downtime cash-outs inherit v1's attribution gap; partial cash-outs accept a documented win-rate bias on the per-strategy circuit breaker.

## Coding Conventions

- Python 3.13, type hints everywhere
- Use `from __future__ import annotations` in all modules
- Dataclasses for data structures (frozen where immutable)
- Pydantic models for config/settings
- Async methods prefixed with standard Python conventions (no special prefix)
- Logging via `logging.getLogger(__name__)`
- **Comments: aim for none.** Write a comment only when it explains something the
  code cannot — a critical constraint, a non-obvious choice, or an incident this
  guards against. Never narrate what the code already says. Prefer a docstring
  over an inline comment when the explanation is about a whole function or class.
- **Do not commit `docs/superpowers/`.** It holds working plan/spec artifacts,
  gitignored on purpose — re-derive conclusions from data, not old plans.

## Configuration

- **Strategy parameters**: `python/config.yaml` — edge thresholds, risk limits, market filters, Telegram toggle
- **Secrets/environment**: `.env` (see `.env.example`) — API keys, execution mode, bankroll, Telegram bot token + chat ID
- **Settings class**: `python/settings.py` — `Settings` object merges both sources, env vars override with `ORACLE_` prefix
- **Deployment overlay**: `ORACLE_CONFIG_OVERLAY` (env or `.env`) names a YAML file merged over `config.yaml` for a fixed set of live-ops sections (`CONFIG_OVERLAY_ROOTS`); unknown keys and other sections raise. Use it for private live values instead of editing `config.yaml`.

## Deployment State

- **This deployment's live state** (what is enabled, sizing, host, work in flight) is in `CLAUDE.local.md` when present — a private, gitignored file. Read it before reasoning about live behavior. Without it, assume the public defaults: shadow mode, `favorite_bias` and the KXRAIN observer off.
- **Live config** = `python/config.yaml` + the optional `ORACLE_CONFIG_OVERLAY` file (`python/settings.py`). `uv run python scripts/check_live_config.py --dump` prints the merged result.
- Systemd units: `oracle-bot.service` (main bot — use this for all `systemctl`/`journalctl` calls, **not** `oracle.service`), `oracle-daily-report.service` (daily report), `oracle-alert-failure@oracle-daily-report.service` (Telegram alert on failed report). Daily ops: `docs/operations-runbook.md`. Deploy: `docs/deploy-guide.md`.
- **Residual design items:** `docs/tech-debt.md`.
