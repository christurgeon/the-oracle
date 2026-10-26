"""Intraday EMOS HIGH temperature trading strategy.

Same-day KXHIGHMIA and KXHIGHLAX brackets only, on EMOS-live cells. The
strategy is hermetic from ``weather_edge`` in identity, config, and trade
attribution while sharing the production EMOS posterior, gate stack,
sizing, and execution engine via composition over
``python.strategies.weather_evaluation``.

Architecture:

* ``IntradayEmosHighStrategy.__init__`` REQUIRES an injected
  ``EmosTableHolder`` (not Optional). The strategy reads
  ``self._emos_holder.table`` on every access so an orchestrator-owned
  hot-reload propagates atomically.
* ``setup()`` enforces three preconditions and flips
  ``self._setup_completed = True`` at the END only after they all pass.
  Failure raises ``RuntimeError`` for log visibility but is caught by the
  orchestrator's ``_run_strategy_lifecycle_hook``; the strategy then
  early-returns from every scan tick.
* On every scan tick where ``_setup_completed`` is False, the strategy
  emits a per-cell rate-limited Telegram WARNING (one per 30 min; first
  emission per process is immediate).
* ``scan()`` only returns same-day KXHIGHMIA / KXHIGHLAX markets;
  past-date rows are rejected, future-date rows are silently skipped
  (``weather_edge`` owns lead_days > 0).
* ``generate_signals()`` calls ``evaluate_market_pure`` for the gate
  stack, applies the production Stage-2 orderbook depth check, clamps
  contract count at ``max_contracts_per_trade`` AFTER the Stage-2 Kelly
  re-run and BEFORE fee math, enforces the per-cell daily trade-count cap
  (defense-in-depth), and finally collapses same-day signals via
  ``_select_one_per_event``.

Hermetic discipline: this module MUST NOT import from
``python.strategies.weather_edge``. Allowed shared primitives live in
``weather_evaluation`` and ``weather_parsing``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from python.analytics.audit_log import (
    AuditSinks,
    RejectionReason,
    ScanFunnel,
    build_sinks,
    reason_numeric,
)
from python.analytics.fees import fee_per_contract as kalshi_fee
from python.analytics.prediction_log import PredictionLog, PredictionRecord
from python.analytics.sizing import SizingResult, dollars_to_contracts, fractional_kelly
from python.common import DailyVariable, ExecutionMode, Side, et_today
from python.data.bias_correction import BiasTable
from python.data.emos import EmosTableHolder
from python.data.kalshi_client import KalshiClient
from python.data.weather import EnsembleForecast, WeatherClient
from python.execution.trade_log import is_pnl_relevant_row
from python.notifications.base import AlertLevel, Notifier
from python.settings import Settings
from python.strategies.base import Strategy, TradeSignal
from python.strategies.weather_evaluation import (
    _EvalConfig,
    _MarketCandidate,
    _parse_min_calibrated_prob,
    _parse_plateau_range,
    _select_one_per_event,
    evaluate_market_pure,
    postprocess_forecasts,
)
from python.strategies.weather_parsing import (
    _DIRECTION_TO_STRIKE_LABEL,
    DepthCheck,
    OpponentSideView,
    ParsedMarket,
    ScanCandidate,
    _orderbook_price_to_probability,
    _parse_fp_int,
    _parse_orderbook_level,
    compute_fill_from_levels,
    extract_market_price_info,
    parse_weather_market,
)
from python.utils.jsonl import iter_jsonl
from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date

logger = logging.getLogger(__name__)


# The v6-audited cells, mirroring ``WEATHER_SERIES`` (weather_edge.py). Inlined,
# not imported, to keep the module hermetic from weather_edge.
_INTRADAY_SERIES: dict[str, tuple[str, DailyVariable]] = {
    "KXHIGHMIA": ("MIA", DailyVariable.TEMP_MAX),
    "KXHIGHLAX": ("LAX", DailyVariable.TEMP_MAX),
}

# Per-cell rate-limit window for the setup-incomplete WARNING.
_SETUP_INCOMPLETE_WARNING_INTERVAL_SECONDS = 30 * 60


def _cell_label(city: str, var_label: str) -> str:
    """Canonical EMOS cell key — mirrors weather.emos_enabled_cells format."""
    return f"{city}/{var_label}"


class IntradayEmosHighStrategy(Strategy):
    """Same-day KXHIGH trading on EMOS-live cells only."""

    def __init__(
        self,
        config: dict[str, Any],
        kalshi_client: KalshiClient,
        weather_client: WeatherClient,
        bankroll: float = 10_000.0,
        market_filter: dict[str, Any] | None = None,
        *,
        settings: Settings,
        emos_holder: EmosTableHolder,
        bias_table: BiasTable | None = None,
        prediction_log: PredictionLog | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        """Construct the intraday EMOS HIGH strategy.

        ``settings`` is REQUIRED (keyword-only, no default). A forgotten
        kwarg in a future test factory or admin tool would otherwise
        silently get a default ``Settings()`` instance and trigger
        ``_check_intraday_same_day_deconfliction`` against orchestrator-
        independent defaults — a misconfiguration class we don't want a
        live process to swallow. ``emos_holder`` is likewise required.
        """
        super().__init__(name="intraday_emos_high", config=config, bankroll=bankroll)
        self._kalshi = kalshi_client
        self._weather = weather_client
        self._settings = settings
        self._market_filter = market_filter or {}
        self._bias_table = bias_table
        self._prediction_log = prediction_log
        self._notifier: Notifier | None = notifier

        # Required. Reading through the holder makes an orchestrator-owned
        # hot-reload visible immediately.
        self._emos_holder: EmosTableHolder = emos_holder

        raw_cells = list(config.get("enabled_cells", []) or [])
        self._enabled_cells: frozenset[str] = frozenset(raw_cells)

        # Pydantic-bounded at config load.
        self._strike_distance_min_std: float = float(
            config.get("min_distance_from_strike_std", 2.0)
        )
        self._max_contracts_per_trade: int = int(
            config.get("max_contracts_per_trade", 50)
        )
        self._max_trades_per_cell_per_day: int = int(
            config.get("max_trades_per_cell_per_day", 3)
        )
        self._emos_hot_reload_pause_seconds: int = int(
            config.get("emos_hot_reload_pause_seconds", 60)
        )

        # Set only after every setup() precondition passes; scan() and
        # generate_signals() return [] until then.
        self._setup_completed: bool = False

        # Setup-incomplete WARNING throttle: first per (process, cell) fires
        # immediately, then at most one per 30 min.
        self._setup_incomplete_warning_state: dict[str, float] = {}

        self._lead_negative_warning_seen_dates: dict[str, set[date]] = {}

        # Keyed on (cell, et_date); reset via reset_daily_counters().
        self._trades_today: dict[tuple[str, date], int] = {}

        # Independent of weather_edge's set.
        self._traded_tickers: set[str] = set()

        state_dir = Path(self._settings.state_dir)
        self._audit_sinks: AuditSinks = build_sinks(
            state_dir,
            eval_enabled=bool(config.get("eval_audit_log", True)),
            candidate_enabled=bool(config.get("candidate_audit_log", True)),
            funnel_enabled=bool(config.get("scan_funnel_log", True)),
            retention_days=int(config.get("audit_log_retention_days", 14)),
        )

        self._emos_log_dedup: dict[Any, Any] = {}

        # ``_emos_paused_until`` is a monotonic timestamp; ``_last_seen_emos_hash``
        # is the previous tick's artifact_hash, so a swap is detectable.
        self._emos_paused_until: float = 0.0
        self._last_seen_emos_hash: str | None = None

    async def setup(self) -> None:
        """Refuse-to-start preconditions.

        Three checks in order. Each raises ``RuntimeError`` on failure.
        The orchestrator's ``_run_strategy_lifecycle_hook`` catches the
        exception and writes it via ``logger.exception``; no automatic
        Telegram CRITICAL fires (verified against ``python/orchestrator.py``
        + ``python/utils/log_config.py``). The operational "refuse-to-trade"
        behavior comes entirely from
        ``self._setup_completed`` staying False and the per-cell
        rate-limited WARNING in scan() / generate_signals().
        """
        table = self._emos_holder.table
        if not table.is_loaded:
            raise RuntimeError(
                "intraday_emos_high requires a loaded EmosTable. "
                f"fallback_reason={table.fallback_reason!r}. "
                "Strategy refuses to start with identity calibrator — "
                "v6 evidence does not extend to identity."
            )

        missing = sorted(c for c in self._enabled_cells if c not in table.cells)
        if missing:
            raise RuntimeError(
                f"intraday_emos_high: enabled cells {missing} not present in "
                f"EmosTable. Available cells: {sorted(table.cells.keys())}."
            )

        # Backstop for a programmatic override of the in-memory config that
        # bypasses the Pydantic Field constraints in python/settings.py.
        if self._strike_distance_min_std != 2.0:
            raise RuntimeError(
                f"intraday_emos_high: strike_distance_min_std="
                f"{self._strike_distance_min_std} must equal 2.0 (v6 audited "
                f"value). Pydantic Field constraint should have caught this "
                f"earlier — config layer drift suspected."
            )
        if self._max_contracts_per_trade > 50:
            raise RuntimeError(
                f"intraday_emos_high: max_contracts_per_trade="
                f"{self._max_contracts_per_trade} exceeds 50 (v6 audited cap)."
            )

        self._setup_completed = True
        logger.info(
            "intraday_emos_high setup complete: enabled_cells=%s emos_hash=%s",
            sorted(self._enabled_cells),
            table.artifact_hash,
        )

    async def aclose(self) -> None:
        """No per-strategy resources to release.

        ``WeatherClient`` / ``KalshiClient`` / ``Notifier`` are
        orchestrator-owned; ``EmosTableHolder`` is shared with weather_edge.
        """
        return None

    def reset_daily_counters(self) -> None:
        """Reset per-cell daily trade counters at the orchestrator's daily
        boundary (called from ``RiskManager.reset_daily()`` fan-out).
        """
        self._trades_today = {}
        # Prune the WARNING-dedup state so a long-lived process doesn't leak.
        today = et_today()
        pruned_lead: dict[str, set[date]] = {}
        for cell, dates in self._lead_negative_warning_seen_dates.items():
            kept = {d for d in dates if d >= today}
            if kept:
                pruned_lead[cell] = kept
        self._lead_negative_warning_seen_dates = pruned_lead
        now = time.monotonic()
        self._setup_incomplete_warning_state = {
            cell: ts
            for cell, ts in self._setup_incomplete_warning_state.items()
            if (now - ts) < _SETUP_INCOMPLETE_WARNING_INTERVAL_SECONDS
        }

    def mark_traded(self, ticker: str) -> None:
        """Record a ticker as traded so it's skipped in future scans."""
        self._traded_tickers.add(ticker)

    def unmark_traded(self, ticker: str) -> None:
        """Release a ticker so it can be re-considered this session.

        Idempotent: silently no-ops when the ticker isn't in the set.
        Matches the weather_edge contract called by the orchestrator on
        TTL-expiry cancellations.
        """
        self._traded_tickers.discard(ticker)

    def load_state_from_log(self, trade_log_path: Path) -> None:
        """Rehydrate ``_trades_today`` + ``_traded_tickers`` from ``state/trades.jsonl``.

        Without this, a process restart at e.g. 14:00 ET resets the per-cell
        daily cap counter to 0 and re-permits trades on tickers that already
        filled and settled earlier in the same ET day — silently doubling the
        cap and bypassing the per-ticker dedup.

        Filters rows via ``is_pnl_relevant_row`` (the schema-evolution gate)
        AND ``row["strategy"] == "intraday_emos_high"`` AND
        ``parse_ticker_date(ticker) == et_today()``. Derives ``(city, var)``
        via ``parse_ticker_city_var`` to seed the cap counter.

        Tolerant of missing trade-log (no rows → zero counts) and malformed
        rows (skipped silently — defensive at the I/O boundary).
        """
        today = et_today()
        seeded_counts = 0
        seeded_tickers = 0
        try:
            rows = list(iter_jsonl(trade_log_path))
        except OSError:
            logger.warning(
                "intraday_emos_high.load_state_from_log: unreadable trade-log %s "
                "— continuing with empty state",
                trade_log_path,
                exc_info=True,
            )
            return

        for row in rows:
            if not is_pnl_relevant_row(row):
                continue
            if str(row.get("strategy", "")) != self._name:
                continue
            ticker = str(row.get("market_ticker", "") or row.get("ticker", ""))
            if not ticker:
                continue
            ticker_date = parse_ticker_date(ticker)
            if ticker_date is None or ticker_date != today:
                continue
            city_var = parse_ticker_city_var(ticker)
            if city_var is None:
                continue
            cell = _cell_label(city_var[0], city_var[1])
            key = (cell, today)
            self._trades_today[key] = self._trades_today.get(key, 0) + 1
            seeded_counts += 1
            if ticker not in self._traded_tickers:
                self._traded_tickers.add(ticker)
                seeded_tickers += 1

        logger.info(
            "intraday_emos_high.load_state_from_log: hydrated %d cap-counter "
            "increment(s) and %d traded-ticker(s) from %s",
            seeded_counts,
            seeded_tickers,
            trade_log_path,
        )

    async def _emit_setup_incomplete_warning(self, cell: str) -> None:
        """Per-cell rate-limited WARNING when the strategy can't trade
        because ``setup()`` failed.

        First emission per (process, cell) fires immediately; subsequent
        emissions on the same cell are throttled to one per 30 min so a
        failed-setup orchestrator running for hours doesn't spam the
        operator's phone.

        Alert wiring: ``Notifier.send_alert`` per
        ``python/notifications/base.py``. Same call pattern as
        weather_edge.py (emits at ERROR there; here we emit at WARNING).
        """
        if self._notifier is None:
            return
        now = time.monotonic()
        last = self._setup_incomplete_warning_state.get(cell)
        if last is not None and (now - last) < _SETUP_INCOMPLETE_WARNING_INTERVAL_SECONDS:
            return
        self._setup_incomplete_warning_state[cell] = now
        try:
            await self._notifier.send_alert(
                f"intraday_emos_high SKIPPING TICK on cell={cell} — setup() "
                f"did not complete; strategy is registered but not generating "
                f"signals. Inspect orchestrator startup logs for the setup() "
                f"exception.",
                level=AlertLevel.WARNING,
                failure_bucket=f"intraday_setup_incomplete:{cell}",
            )
        except Exception:
            logger.exception(
                "intraday_emos_high setup-incomplete WARNING dispatch failed (cell=%s)",
                cell,
            )

    async def _emit_lead_negative_warning(
        self,
        *,
        cell: str,
        ticker: str,
        lead_days_raw: int,
        utc_today: date,
        et_today_value: date,
    ) -> None:
        """Per-(cell, UTC date) dedup'd WARNING on lead-negative routing."""
        if self._notifier is None:
            return
        seen = self._lead_negative_warning_seen_dates.setdefault(cell, set())
        if utc_today in seen:
            return
        seen.add(utc_today)
        try:
            await self._notifier.send_alert(
                f"intraday_emos_high lead_days_raw={lead_days_raw} detected on "
                f"cell={cell} ticker={ticker} — treating as same-day. "
                f"ET={et_today_value.isoformat()} UTC={utc_today.isoformat()}",
                level=AlertLevel.WARNING,
                failure_bucket=f"intraday_lead_negative:{cell}:{utc_today.isoformat()}",
            )
        except Exception:
            logger.exception(
                "intraday_emos_high lead-negative WARNING dispatch failed (cell=%s)",
                cell,
            )

    async def _emit_hot_reload_warning(
        self,
        *,
        old_hash: str | None,
        new_hash: str | None,
        new_fit_date: str | None,
    ) -> None:
        """WARNING on EMOS hot-reload."""
        if self._notifier is None:
            return
        try:
            await self._notifier.send_alert(
                f"intraday_emos_high paused {self._emos_hot_reload_pause_seconds}s — "
                f"EMOS reloaded (hash={old_hash}->{new_hash}, fit_date={new_fit_date})",
                level=AlertLevel.WARNING,
                failure_bucket="intraday_emos_hot_reload",
            )
        except Exception:
            logger.exception("intraday_emos_high hot-reload WARNING dispatch failed")

    async def scan(  # type: ignore[override]
        self,
        *,
        scan_id: str | None = None,
        funnel: ScanFunnel | None = None,
        scan_today: date | None = None,
    ) -> list[ScanCandidate]:
        """Fetch open KXHIGHMIA + KXHIGHLAX markets; keep same-day rows only.

        Past-date rows are rejected outright. Future-date rows (lead_days
        > 0) are silently skipped — ``weather_edge`` owns those
        (deconfliction).
        """
        if not self._enabled:
            logger.info("intraday_emos_high disabled — skipping scan")
            return []
        if not self._setup_completed:
            # Surfaces the failure to the operator within one scan tick.
            for cell in sorted(self._enabled_cells):
                await self._emit_setup_incomplete_warning(cell)
            return []

        scan_today = scan_today if scan_today is not None else et_today()
        utc_today = datetime.now(UTC).date()

        candidates: list[ScanCandidate] = []
        total_raw = 0
        skipped_past = 0
        skipped_future = 0

        # Liquidity filter knobs (mirrors weather_edge defaults).
        min_volume = self._market_filter.get("min_volume", 0)
        min_oi = self._market_filter.get("min_open_interest", 0)
        min_price = self._market_filter.get("min_price", 0.0)
        max_price = self._market_filter.get("max_price", 1.0)

        for i, (series_ticker, (city_code, daily_var)) in enumerate(
            _INTRADAY_SERIES.items()
        ):
            if i > 0:
                # Stagger requests to avoid Kalshi's 429 rate limit.
                await asyncio.sleep(1.0)

            try:
                raw_markets, _truncated = await self._kalshi.iter_all_markets(
                    status="open",
                    series_ticker=series_ticker,
                )
            except Exception:
                logger.exception(
                    "intraday_emos_high: failed to fetch markets for %s",
                    series_ticker,
                )
                continue

            total_raw += len(raw_markets)
            if funnel is not None:
                funnel.n_raw += len(raw_markets)

            for market in raw_markets:
                parsed = parse_weather_market(market, city_code, daily_var)
                if parsed is None:
                    continue

                cell = _cell_label(parsed.city, parsed.temp_label)
                if cell not in self._enabled_cells:
                    continue

                if parsed.forecast_date < scan_today:
                    skipped_past += 1
                    if funnel is not None:
                        funnel.increment(RejectionReason.PAST_DATE)
                    continue

                # weather_edge owns future-date markets.
                if parsed.forecast_date > scan_today:
                    skipped_future += 1
                    continue

                # Fires when the UTC date is already past forecast_date. The
                # un-clamped lead is surfaced via a WARNING deduped per (cell,
                # UTC date); posterior/gate use lead clamped to 0.
                lead_days_raw = (parsed.forecast_date - utc_today).days
                if lead_days_raw < 0:
                    await self._emit_lead_negative_warning(
                        cell=cell,
                        ticker=parsed.ticker,
                        lead_days_raw=lead_days_raw,
                        utc_today=utc_today,
                        et_today_value=scan_today,
                    )

                if min_volume > 0:
                    vol = _parse_fp_int(market.get("volume_fp"))
                    if vol is not None and vol < min_volume:
                        if funnel is not None:
                            funnel.increment(RejectionReason.VOLUME)
                        continue
                if min_oi > 0:
                    oi = _parse_fp_int(market.get("open_interest_fp"))
                    if oi is not None and oi < min_oi:
                        if funnel is not None:
                            funnel.increment(RejectionReason.OPEN_INTEREST)
                        continue

                price_info = extract_market_price_info(market)
                if (min_price > 0.0 or max_price < 1.0) and price_info is not None:
                    if (
                        price_info.midpoint < min_price
                        or price_info.midpoint > max_price
                    ):
                        if funnel is not None:
                            funnel.increment(RejectionReason.PRICE_GATE)
                        continue

                candidates.append(
                    ScanCandidate(
                        market=market,
                        parsed=parsed,
                        price_info=price_info,
                    )
                )

        logger.info(
            "intraday_emos_high: %d tradeable markets out of %d total "
            "(%d past_date, %d future_date skipped)",
            len(candidates),
            total_raw,
            skipped_past,
            skipped_future,
        )
        return candidates

    async def evaluate(
        self,
        market: dict[str, Any] | ScanCandidate,
    ) -> float | None:
        """Compute the model probability for a single market.

        Delegates to ``evaluate_market_pure`` via composition: the same
        per-cell EMOS dispatcher fires that drives weather_edge,
        so a cell promoted to EMOS in production returns the EMOS
        probability here without any duplication of the dispatch logic.
        """
        parsed: ParsedMarket | None
        if isinstance(market, ScanCandidate):
            sc = market
            parsed = market.parsed
        else:
            parsed = market.get("_parsed")
            if parsed is None:
                logger.warning(
                    "intraday_emos_high.evaluate: market dict missing _parsed: %s",
                    market.get("ticker"),
                )
                return None
            sc = ScanCandidate(
                market=market,
                parsed=parsed,
                price_info=extract_market_price_info(market),
            )

        # Fetch the forecast and post-process (bias correction + medians/stds
        # for the strike-distance gate).
        forecast_cache: dict[
            tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
        ] = {}
        raw_member_sink: dict[
            tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
        ] = {}
        try:
            single = await self._weather.get_multi_model_forecast(
                city=parsed.city,
                forecast_date=parsed.forecast_date,
                daily_variable=parsed.daily_variable,
                models=list(self._config.get("ensemble_models", ["gfs_seamless"])),
            )
        except Exception:
            logger.exception(
                "intraday_emos_high.evaluate: forecast fetch failed for %s on %s",
                parsed.city,
                parsed.forecast_date,
            )
            return None
        if not single:
            return None
        key = (parsed.city, parsed.forecast_date, parsed.daily_variable)
        forecast_cache[key] = single

        (
            forecast_cache,
            bias_corrections,
            raw_ensemble_stds,
            raw_ensemble_medians,
        ) = postprocess_forecasts(
            forecast_cache=forecast_cache,
            bias_table=self._bias_table,
            bias_shrinkage=self._config.get("bias_shrinkage", 0.3),
            bias_max_correction_f=self._config.get("bias_max_correction_f", 5.0),
            spread_inflation_overrides=self._config.get("spread_inflation_factors"),
            raw_member_sink=raw_member_sink,
        )

        cfg = self._build_eval_config()
        candidate = evaluate_market_pure(
            sc,
            forecast_cache,
            bias_corrections,
            raw_ensemble_stds,
            raw_ensemble_medians,
            cfg,
            calibrator=None,
            shadow_calibrator=None,
            weights=self._config.get("ensemble_weights", {"gfs_seamless": 1.0}),
            weather_client=self._weather,
            bankroll=self._bankroll,
            excluded_city_vars=self._config.get("excluded_city_vars", []),
            spread_inflation_overrides=self._config.get("spread_inflation_factors"),
            emos_table=self._emos_holder.table,
            raw_member_sink=raw_member_sink,
            emos_log_dedup=self._emos_log_dedup,
            emos_enabled_cells=self._enabled_cells,
            strategy_name=self._name,
        )
        return candidate.model_prob if candidate is not None else None

    async def generate_signals(self) -> list[TradeSignal]:
        """Full scan -> evaluate -> size pipeline."""
        if not self._enabled:
            return []
        if not self._setup_completed:
            for cell in sorted(self._enabled_cells):
                await self._emit_setup_incomplete_warning(cell)
            return []

        # EMOS hot-reload pause. The orchestrator owns
        # check_and_reload; this strategy only observes the hash. If the
        # holder's current artifact_hash differs from the hash we saw on the
        # previous tick, arm the pause and emit a WARNING. While the pause
        # is active, scan + audit still run but no signals are returned.
        current_hash = self._emos_holder.table.artifact_hash
        if (
            self._last_seen_emos_hash is not None
            and current_hash is not None
            and current_hash != self._last_seen_emos_hash
        ):
            now = time.monotonic()
            self._emos_paused_until = now + float(self._emos_hot_reload_pause_seconds)
            await self._emit_hot_reload_warning(
                old_hash=self._last_seen_emos_hash,
                new_hash=current_hash,
                new_fit_date=self._emos_holder.table.fit_date,
            )
        self._last_seen_emos_hash = current_hash
        hot_reload_paused = time.monotonic() < self._emos_paused_until

        scan_id = uuid.uuid4().hex[:12]
        scan_ts = datetime.now(tz=UTC).timestamp()
        funnel = ScanFunnel()
        scan_today: date = et_today()

        # Day-rollover defense-in-depth: ``reset_daily_counters()`` is the
        # orchestrator's explicit boundary hook (called from
        # RiskManager.reset_daily fan-out). If a future orchestrator change
        # forgets to wire that fan-out, the per-cell daily cap counter
        # would never reset and the strategy would silently top out at
        # ``max_trades_per_cell_per_day`` for the entire process lifetime.
        # Self-clean any stale (cell, date) entries whose date doesn't
        # match the current ET scan day so an orphaned counter can't
        # silently brick the strategy.
        self._trades_today = {
            key: count
            for key, count in self._trades_today.items()
            if key[1] == scan_today
        }

        markets = await self.scan(
            scan_id=scan_id, funnel=funnel, scan_today=scan_today
        )
        if not markets:
            # Emit the funnel anyway so zero-market stagnation is visible.
            self._emit_funnel(funnel, scan_id, scan_ts, n_markets=0)
            return []

        models: list[str] = self._config.get("ensemble_models", ["gfs_seamless"])
        seen_keys: set[tuple[str, date, DailyVariable]] = set()
        for sc in markets:
            seen_keys.add(
                (sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable)
            )
        semaphore = asyncio.Semaphore(4)

        async def _fetch(
            city: str,
            forecast_date: date,
            daily_variable: DailyVariable,
        ) -> dict[str, EnsembleForecast]:
            async with semaphore:
                return await self._weather.get_multi_model_forecast(
                    city=city,
                    forecast_date=forecast_date,
                    daily_variable=daily_variable,
                    models=models,
                )

        keys_list = list(seen_keys)
        forecast_results = await asyncio.gather(
            *(_fetch(c, d, v) for c, d, v in keys_list)
        )
        raw_forecast_cache: dict[
            tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
        ] = {}
        for key, fc_result in zip(keys_list, forecast_results):
            if fc_result:
                raw_forecast_cache[key] = fc_result

        raw_member_sink: dict[
            tuple[str, date, DailyVariable], dict[str, EnsembleForecast]
        ] = {}
        (
            forecast_cache,
            bias_corrections,
            raw_ensemble_stds,
            raw_ensemble_medians,
        ) = postprocess_forecasts(
            forecast_cache=raw_forecast_cache,
            bias_table=self._bias_table,
            bias_shrinkage=self._config.get("bias_shrinkage", 0.3),
            bias_max_correction_f=self._config.get("bias_max_correction_f", 5.0),
            spread_inflation_overrides=self._config.get("spread_inflation_factors"),
            raw_member_sink=raw_member_sink,
        )

        cfg = self._build_eval_config()
        emos_table_for_scan = self._emos_holder.table
        # Snapshotted with the table so a mid-tick hot-reload cannot desync
        # ``emos_fit_date`` from the artifact_hash that drove this batch.
        emos_fit_date_for_scan = emos_table_for_scan.fit_date

        candidates_for_depth: list[tuple[ScanCandidate, _MarketCandidate]] = []
        batch_tickers: set[str] = set()
        for sc in markets:
            ticker = sc.parsed.ticker
            if ticker in self._traded_tickers or ticker in batch_tickers:
                continue
            batch_tickers.add(ticker)
            candidate = evaluate_market_pure(
                sc,
                forecast_cache,
                bias_corrections,
                raw_ensemble_stds,
                raw_ensemble_medians,
                cfg,
                calibrator=None,
                shadow_calibrator=None,
                weights=self._config.get("ensemble_weights", {"gfs_seamless": 1.0}),
                weather_client=self._weather,
                bankroll=self._bankroll,
                excluded_city_vars=self._config.get("excluded_city_vars", []),
                spread_inflation_overrides=self._config.get(
                    "spread_inflation_factors"
                ),
                emos_table=emos_table_for_scan,
                raw_member_sink=raw_member_sink,
                emos_log_dedup=self._emos_log_dedup,
                emos_enabled_cells=self._enabled_cells,
                scan_id=scan_id,
                sinks=self._audit_sinks,
                funnel=funnel,
                scan_today=scan_today,
                strategy_name=self._name,
            )
            if candidate is None:
                continue
            funnel.n_stage1_pass += 1
            candidates_for_depth.append((sc, candidate))

        # Phase 2b (concurrent): Stage-2 orderbook depth check + VWAP recompute.
        _orderbook_concurrency = int(self._config.get("orderbook_concurrency", 2))
        _depth_semaphore = asyncio.Semaphore(_orderbook_concurrency)

        _depth_outcome_t = tuple[
            ScanCandidate,
            _MarketCandidate,
            tuple[float, SizingResult, int, bool, DepthCheck | None] | None,
        ]

        async def _run_depth(
            sc: ScanCandidate,
            candidate: _MarketCandidate,
        ) -> _depth_outcome_t:
            async with _depth_semaphore:
                result = await self._apply_depth_adjustment(
                    candidate,
                    cfg,
                    funnel=funnel,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                )
            return sc, candidate, result

        depth_outcomes: list[_depth_outcome_t] = await asyncio.gather(
            *[_run_depth(sc, c) for sc, c in candidates_for_depth]
        )

        # Pre-cap: ``n_signals_fired`` and ``_record_prediction`` fire here
        # because predictions.jsonl is the calibrator-fitting population and MUST
        # capture unconstrained model output. This funnel invariant holds only
        # while ``n_signals_fired`` is pre-cap:
        #   len(returned_signals) == n_signals_fired - n_event_capped
        #       - n_after_trades_per_cell_per_day_cap - n_hot_reload_paused
        # The daily cap runs after the event collapse so the counter measures
        # actual emissions, not pre-collapse candidates.
        pre_cap_signals: list[tuple[TradeSignal, _MarketCandidate]] = []
        for sc, candidate, result in depth_outcomes:
            if result is None:
                continue
            effective_edge, sizing, contracts, depth_capped, depth_result = result
            signal = self._build_signal(
                candidate,
                effective_edge,
                sizing,
                contracts,
                depth_capped,
                depth_result,
                scan_today=scan_today,
                emos_fit_date=emos_fit_date_for_scan,
            )
            pre_cap_signals.append((signal, candidate))
            funnel.n_signals_fired += 1
            self._record_prediction(signal, candidate)

        # Always on for intraday: keeps the best candidate per
        # (city, market_type, forecast_date).
        pre_cap_signal_list = [sig for sig, _cand in pre_cap_signals]
        collapsed_signals, n_capped = _select_one_per_event(
            pre_cap_signal_list, enabled=True
        )
        if n_capped:
            funnel.n_event_capped += n_capped

        # Per-cell daily cap, ≤ ``max_trades_per_cell_per_day`` ACTUAL emissions
        # (post-collapse). It is the structural backstop should a change to
        # ``_select_one_per_event``'s grouping key (today city + market_type +
        # forecast_date) ever surface >1 signal/cell/day; re-evaluate it then.
        # The ``n_after_trades_per_cell_per_day_cap`` funnel counter records drops.
        #
        # CRITICAL ordering: the ``hot_reload_paused`` early-return MUST precede
        # the cap-counter mutation below. Incrementing ``_trades_today`` for
        # signals that are then discarded inflates the counter and blocks real
        # signals on later same-ET-day ticks.
        if hot_reload_paused:
            # Keeps the funnel invariant above closed.
            funnel.n_hot_reload_paused += len(collapsed_signals)
            logger.warning(
                "intraday_emos_high: hot-reload pause active — suppressing "
                "%d collapsed signal(s) this tick",
                len(collapsed_signals),
            )
            self._emit_funnel(
                funnel, scan_id, scan_ts, n_markets=len(markets) if markets else 0
            )
            return []

        signals: list[TradeSignal] = []
        for signal in collapsed_signals:
            cell = str(signal.metadata.get("cell", ""))
            cap_key = (cell, scan_today)
            already = self._trades_today.get(cap_key, 0)
            if already >= self._max_trades_per_cell_per_day:
                logger.info(
                    "intraday_emos_high: per-cell daily cap hit (cell=%s n=%d)"
                    " — dropping candidate %s",
                    cell,
                    already,
                    signal.market_ticker,
                )
                funnel.increment(RejectionReason.TRADES_PER_CELL_CAP)
                continue
            self._trades_today[cap_key] = already + 1
            signals.append(signal)

        logger.info(
            "intraday_emos_high: generated %d signal(s) from %d market(s) "
            "(%d event-capped)",
            len(signals),
            len(markets),
            n_capped,
        )
        self._emit_funnel(
            funnel, scan_id, scan_ts, n_markets=len(markets) if markets else 0
        )
        return signals

    # -- Helpers -------------------------------------------------------------

    def _emit_funnel(
        self,
        funnel: ScanFunnel,
        scan_id: str,
        scan_ts: float,
        *,
        n_markets: int,
    ) -> None:
        """Write one ``scan_funnel.jsonl`` row for this scan cycle.

        Mirrors ``WeatherEdgeStrategy._emit_funnel``. Swallows every error
        — a failed funnel write must never break signal generation or stall
        the orchestrator.
        """
        if self._audit_sinks.funnel_log is None:
            return
        try:
            self._audit_sinks.funnel_log.emit_funnel(
                funnel,
                scan_id,
                extra={
                    "strategy": self._name,
                    "scan_ts": scan_ts,
                    "n_markets": n_markets,
                },
            )
        except Exception:
            logger.exception("scan_funnel emission failed for scan_id=%s", scan_id)

    def _build_candidate_audit_row(
        self,
        *,
        candidate: _MarketCandidate,
        cfg: _EvalConfig,
        sizing: SizingResult,
        contracts: int,
        depth_capped: bool,
        depth_result: DepthCheck | None,
        opp_view: OpponentSideView | None,
        fill_edge: float | None,
        fill_fee: float | None,
        reason: RejectionReason,
        scan_id: str,
        scan_ts: float | None,
    ) -> dict[str, Any]:
        """Serialize a Stage-1 candidate + Stage-2 outcome into one audit row.

        Mirrors ``WeatherEdgeStrategy._build_candidate_audit_row`` but
        produces a minimal intraday-flavored row — no bracket-metric fields
        (intraday is BETWEEN-only) and a ``strategy`` discriminator so
        consumers can route rows by source.
        """
        price_info = candidate.price_info
        market = candidate.market

        our_side_available = depth_result.available_contracts if depth_result else None
        our_side_levels = depth_result.total_levels if depth_result else None
        our_side_best_price = (
            depth_result.fill_price - depth_result.slippage
            if depth_result and depth_result.fill_price > 0.0
            else None
        )

        row: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "scan_id": scan_id,
            "scan_ts": scan_ts,
            "strategy": self._name,
            "ticker": candidate.parsed.ticker,
            "event": market.get("event_ticker", ""),
            "city": candidate.parsed.city,
            "var": candidate.parsed.temp_label,
            "forecast_date": candidate.parsed.forecast_date.isoformat(),
            "strike_type": candidate.parsed.strike_type.value,
            "floor": candidate.parsed.floor_strike,
            "cap": candidate.parsed.cap_strike,
            "close_time": market.get("close_time", ""),
            # Pricing
            "yes_bid": price_info.yes_bid,
            "yes_ask": price_info.yes_ask,
            "mid": price_info.midpoint,
            "exec_price": (
                candidate.exec_price if not math.isnan(candidate.exec_price) else None
            ),
            "spread": price_info.spread,
            "volume": price_info.volume,
            "oi": price_info.open_interest,
            "yes_bid_size": price_info.yes_bid_size,
            "yes_ask_size": price_info.yes_ask_size,
            # Model
            "raw_ensemble_prob": candidate.raw_ensemble_prob,
            "calibrated_prob": candidate.model_prob,
            "correction_f": candidate.correction_f,
            "ens_std_pooled": candidate.ensemble_std_pooled,
            "ensemble_confidence": candidate.ensemble_confidence,
            "per_model_probs": candidate.per_model_probs,
            "spread_inflation_factor": candidate.spread_inflation_factor,
            "strike_abs_z": candidate.strike_abs_z,
            # Sizing + edge
            "raw_edge": candidate.raw_edge,
            "effective_edge": candidate.effective_edge,
            "fill_edge": fill_edge,
            "fill_price": depth_result.fill_price if depth_result else None,
            "slippage": depth_result.slippage if depth_result else None,
            "fill_fee": fill_fee,
            "kelly_dollars": sizing.dollars,
            "kelly_raw": sizing.kelly_raw,
            "kelly_capped": sizing.capped,
            "contracts": contracts,
            "depth_capped": depth_capped,
            "side": str(sizing.side),
            # Orderbook (both sides)
            "our_side_available": our_side_available,
            "our_side_levels": our_side_levels,
            "our_side_best_price": our_side_best_price,
            "opp_side_available": opp_view.available if opp_view else None,
            "opp_side_levels": opp_view.levels if opp_view else None,
            "opp_side_best_price": opp_view.best_price if opp_view else None,
            # Outcome
            "reason": reason.value,
            "reason_numeric": reason_numeric(reason),
            # Echo config so rows are self-describing when thresholds move.
            "edge_threshold": cfg.edge_threshold,
            "min_book_depth": self._config.get("min_book_depth", 5),
            "live_calibrator_kind": candidate.live_calibrator_kind,
        }
        return row

    def _build_eval_config(self) -> _EvalConfig:
        """Construct the loop-invariant ``_EvalConfig`` for this strategy.

        Pulls from ``self._config`` for strategy-specific knobs and from
        ``self._settings`` for global risk parameters. The
        ``min_distance_from_strike_std`` value is hard-bound to the
        instance's ``self._strike_distance_min_std`` (which the
        precondition #3 check in setup() pins at 2.0).

        The cal-floor is mode-aware: shadow mode keeps the bin observable
        per the existing weather_edge convention (the spec inherits this).
        Plateau-block is read from the strategy config (defaults to the
        weather config block when absent — see config.yaml).
        """
        exec_mode = self._settings.execution.mode
        min_cal_floor = (
            None
            if str(exec_mode) == ExecutionMode.SHADOW.value
            else _parse_min_calibrated_prob(self._config.get("min_calibrated_prob"))
        )
        plateau_range = _parse_plateau_range(self._config.get("block_raw_prob_range"))
        return _EvalConfig(
            max_spread=self._config.get("max_spread", 0.15),
            spread_skill_sensitivity=self._config.get("spread_skill_sensitivity", 0.15),
            edge_threshold=self._config.get("edge_threshold", 0.07),
            max_edge=self._config.get("max_edge", 0.30),
            kelly_fraction=self._settings.risk.kelly_fraction,
            max_pct=self._settings.risk.max_single_market_pct,
            max_take_pct=self._market_filter.get("max_orderbook_take_pct", 1.0),
            depth_check_enabled=self._config.get("orderbook_depth_check", True),
            min_raw_prob_for_between=self._config.get("min_raw_prob_for_between", 0.05),
            min_distance_from_strike_std=self._strike_distance_min_std,
            max_fill_slippage=self._config.get("max_fill_slippage"),
            block_raw_prob_range=plateau_range,
            max_model_market_gap_near_term=None,
            min_calibrated_prob=min_cal_floor,
        )

    async def _apply_depth_adjustment(
        self,
        candidate: _MarketCandidate,
        cfg: _EvalConfig,
        *,
        funnel: ScanFunnel | None = None,
        scan_id: str | None = None,
        scan_ts: float | None = None,
    ) -> tuple[float, SizingResult, int, bool, DepthCheck | None] | None:
        """Stage-2 orderbook depth check + VWAP recompute + max_contracts clamp.

        Mirrors ``WeatherEdgeStrategy._apply_depth_adjustment`` but applies
        the intraday strategy's per-strategy contract cap
        (``max_contracts_per_trade``) AFTER the Stage-2 Kelly re-run and
        BEFORE fee math. The clamp bounds the
        fill-price-corrected Kelly output, not a stale Stage-1 estimate;
        recorded per-signal fees scale against the clamped contract count.

        When ``scan_id`` is provided and the ``candidate_audit`` sink is
        enabled, emits one ``candidate_audit.jsonl`` row at each terminal
        path (``ORDERBOOK_THIN``, ``FILL_SLIPPAGE``, ``FILL_EDGE_NEGATIVE``,
        ``KELLY_SIDE_FLIP``, ``KELLY_ZERO``, ``PASSED``) — mirrors
        ``WeatherEdgeStrategy._apply_depth_adjustment``.

        Returns ``(effective_edge, sizing, contracts, depth_capped,
        depth_result)`` on pass, else ``None``.
        """
        effective_edge = candidate.effective_edge
        sizing = candidate.sizing
        contracts = candidate.contracts
        depth_capped = False
        depth_result: DepthCheck | None = None
        opp_view: OpponentSideView | None = None
        fill_fee_final: float | None = None
        fill_edge_final: float | None = None

        def _emit_candidate(
            reason: RejectionReason,
            *,
            fill_edge: float | None = None,
            fill_fee: float | None = None,
        ) -> None:
            if scan_id is None or self._audit_sinks.candidate_log is None:
                return
            try:
                row = self._build_candidate_audit_row(
                    candidate=candidate,
                    cfg=cfg,
                    sizing=sizing,
                    contracts=contracts,
                    depth_capped=depth_capped,
                    depth_result=depth_result,
                    opp_view=opp_view,
                    fill_edge=fill_edge,
                    fill_fee=fill_fee,
                    reason=reason,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                )
                self._audit_sinks.candidate_log.emit_candidate(row)
            except Exception:
                logger.exception(
                    "candidate_audit emission failed for %s",
                    candidate.parsed.ticker,
                )

        if cfg.depth_check_enabled:
            depth_result, opp_view = await self._check_orderbook_depth(
                candidate.parsed.ticker,
                sizing.side,
                contracts,
            )
            if depth_result is not None:
                if depth_result.is_stale:
                    logger.warning(
                        "%s: orderbook too thin (available=%d, levels=%d) — skipping",
                        candidate.parsed.ticker,
                        depth_result.available_contracts,
                        depth_result.total_levels,
                    )
                    if funnel is not None:
                        funnel.n_orderbook_thin += 1
                    _emit_candidate(RejectionReason.ORDERBOOK_THIN)
                    return None

                # EMOS rows already price spread via (c, d); skipping the
                # ensemble_confidence discount avoids double-counting (as in
                # weather_edge.py).
                fill_fee = kalshi_fee(depth_result.fill_price)
                ec = (
                    1.0
                    if candidate.live_calibrator_kind == "emos_raw"
                    else candidate.ensemble_confidence
                )
                if sizing.side == Side.YES:
                    fill_edge = (
                        candidate.model_prob - depth_result.fill_price
                    ) * ec - fill_fee
                else:
                    fill_edge = (
                        (1.0 - candidate.model_prob) - depth_result.fill_price
                    ) * ec - fill_fee

                fill_fee_final = fill_fee
                fill_edge_final = fill_edge

                if (
                    cfg.max_fill_slippage is not None
                    and depth_result.slippage > cfg.max_fill_slippage
                ):
                    if funnel is not None:
                        funnel.n_fill_slippage += 1
                    _emit_candidate(
                        RejectionReason.FILL_SLIPPAGE,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None
                if fill_edge < cfg.edge_threshold:
                    if funnel is not None:
                        funnel.n_fill_edge_low += 1
                    _emit_candidate(
                        RejectionReason.FILL_EDGE_NEGATIVE,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None

                effective_edge = fill_edge

                fill_market_price = (
                    depth_result.fill_price
                    if sizing.side == Side.YES
                    else 1.0 - depth_result.fill_price
                )
                pre_depth_side = sizing.side
                sizing = fractional_kelly(
                    model_prob=candidate.model_prob,
                    market_price=fill_market_price,
                    kelly_fraction=cfg.kelly_fraction,
                    max_pct=cfg.max_pct,
                    bankroll=self._bankroll,
                    fee_per_contract=fill_fee,
                )
                if sizing.side != Side.NONE and sizing.side != pre_depth_side:
                    logger.critical(
                        "%s: Kelly side flipped post-depth (%s -> %s); "
                        "fill_price=%.4f model_prob=%.4f — skipping",
                        candidate.parsed.ticker,
                        pre_depth_side,
                        sizing.side,
                        depth_result.fill_price,
                        candidate.model_prob,
                    )
                    if funnel is not None:
                        funnel.n_kelly_side_flip += 1
                    _emit_candidate(
                        RejectionReason.KELLY_SIDE_FLIP,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None
                if sizing.side == Side.NONE or sizing.dollars < 1.0:
                    if funnel is not None:
                        funnel.n_fill_kelly_zero += 1
                    _emit_candidate(
                        RejectionReason.KELLY_ZERO,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None
                contracts = dollars_to_contracts(
                    sizing.dollars,
                    fill_market_price,
                    sizing.side,
                )

                if cfg.max_take_pct < 1.0:
                    max_book_contracts = max(
                        int(depth_result.available_contracts * cfg.max_take_pct),
                        1,
                    )
                else:
                    max_book_contracts = depth_result.available_contracts
                if contracts > max_book_contracts:
                    contracts = max_book_contracts
                    depth_capped = True

        # Fallback: best-level depth cap when depth check is disabled or
        # the orderbook API failed.
        if depth_result is None and cfg.max_take_pct < 1.0:
            price_info = candidate.price_info
            depth = (
                price_info.yes_ask_size
                if sizing.side == Side.YES
                else price_info.yes_bid_size
            )
            if depth is not None and depth > 0:
                max_contracts_best = max(int(depth * cfg.max_take_pct), 1)
                if contracts > max_contracts_best:
                    contracts = max_contracts_best
                    depth_capped = True

        # Applied after the Stage-2 Kelly re-run and before fee math so recorded
        # fees scale with the clamped count. v6-audited ceiling = 50.
        if contracts > self._max_contracts_per_trade:
            logger.info(
                "%s: contract cap clamp %d -> %d (max_contracts_per_trade)",
                candidate.parsed.ticker,
                contracts,
                self._max_contracts_per_trade,
            )
            contracts = self._max_contracts_per_trade
            depth_capped = True

        # RiskManager reads ``suggested_dollars`` for the per-market exposure cap;
        # a stale pre-clamp value would eat budget for capital never deployed.
        # Price is side-aligned: the fill price when the depth check ran, else
        # best-level price, else midpoint.
        per_contract_price: float | None = None
        if depth_result is not None and depth_result.fill_price > 0.0:
            per_contract_price = (
                depth_result.fill_price
                if sizing.side == Side.YES
                else 1.0 - depth_result.fill_price
            )
        else:
            price_info = candidate.price_info
            best_price = (
                price_info.yes_ask if sizing.side == Side.YES else price_info.yes_bid
            )
            if best_price is not None:
                per_contract_price = (
                    best_price if sizing.side == Side.YES else 1.0 - best_price
                )
            elif price_info.midpoint > 0.0:
                per_contract_price = (
                    price_info.midpoint
                    if sizing.side == Side.YES
                    else 1.0 - price_info.midpoint
                )
        if (
            contracts > 0
            and per_contract_price is not None
            and per_contract_price > 0.0
        ):
            sizing = SizingResult(
                side=sizing.side,
                dollars=float(contracts) * per_contract_price,
                kelly_raw=sizing.kelly_raw,
                kelly_fraction_used=sizing.kelly_fraction_used,
                capped=sizing.capped,
            )

        _ = opp_view

        _emit_candidate(
            RejectionReason.PASSED,
            fill_edge=fill_edge_final,
            fill_fee=fill_fee_final,
        )
        return effective_edge, sizing, contracts, depth_capped, depth_result

    async def _check_orderbook_depth(
        self,
        ticker: str,
        side: Side,
        target_contracts: int,
    ) -> tuple[DepthCheck | None, OpponentSideView | None]:
        """Fetch the full orderbook and compute realistic fill cost.

        Mirrors ``WeatherEdgeStrategy._check_orderbook_depth`` — orderbook
        math is shared (``compute_fill_from_levels``) but the call site is
        independent so the strategies remain hermetic.
        """
        try:
            data = await self._kalshi.get_orderbook(
                ticker, depth=self._config.get("book_depth_levels", 10)
            )
        except Exception:
            logger.warning(
                "intraday_emos_high: failed to fetch orderbook for %s",
                ticker,
                exc_info=True,
            )
            return None, None

        orderbook = data.get("orderbook_fp", {})
        if side == Side.YES:
            opposite_bids = orderbook.get("no_dollars", [])
            exit_bids = orderbook.get("yes_dollars", [])
        else:
            opposite_bids = orderbook.get("yes_dollars", [])
            exit_bids = orderbook.get("no_dollars", [])

        ask_levels: list[tuple[float, int]] = []
        for level in opposite_bids:
            parsed = _parse_orderbook_level(level)
            if parsed is None:
                continue
            opp_price, qty = parsed
            our_price = _orderbook_price_to_probability(opp_price)
            if our_price <= 0.0 or our_price >= 1.0:
                continue
            ask_levels.append((our_price, qty))
        ask_levels.sort(key=lambda x: x[0])

        opp_available = 0
        opp_levels_n = 0
        opp_best_price: float | None = None
        for level in exit_bids:
            parsed = _parse_orderbook_level(level)
            if parsed is None:
                continue
            price_prob, qty = parsed
            if price_prob <= 0.0 or price_prob >= 1.0:
                continue
            opp_available += qty
            opp_levels_n += 1
            if opp_best_price is None or price_prob > opp_best_price:
                opp_best_price = price_prob

        opp_view = OpponentSideView(
            available=opp_available,
            levels=opp_levels_n,
            best_price=opp_best_price,
        )

        min_depth = self._config.get("min_book_depth", 5)
        depth_check = compute_fill_from_levels(ask_levels, target_contracts, min_depth)
        return depth_check, opp_view

    def _build_signal(
        self,
        candidate: _MarketCandidate,
        effective_edge: float,
        sizing: SizingResult,
        contracts: int,
        depth_capped: bool,
        depth_result: DepthCheck | None,
        *,
        scan_today: date | None = None,
        emos_fit_date: str | None = None,
    ) -> TradeSignal:
        """Construct the final TradeSignal with all per-signal metadata fields.

        ``emos_fit_date`` is the snapshot value captured at the top of
        ``generate_signals`` next to ``emos_table_for_scan``. When omitted
        (back-compat path, e.g. direct ``_build_signal`` calls from tests)
        it falls back to the live holder — same behavior as pre-fix.
        """
        _today = scan_today if scan_today is not None else et_today()
        # The scan snapshot keeps ``emos_fit_date`` in step with the artifact
        # hash; the live holder is only a fallback for direct test calls.
        _emos_fit_date = (
            emos_fit_date
            if emos_fit_date is not None
            else self._emos_holder.table.fit_date
        )
        utc_today = datetime.now(UTC).date()
        lead_days_raw = (candidate.parsed.forecast_date - utc_today).days
        lead_days = max(lead_days_raw, 0)
        cell = _cell_label(candidate.parsed.city, candidate.parsed.temp_label)

        return TradeSignal(
            market_id=candidate.market.get("ticker", ""),
            market_ticker=candidate.market.get("ticker", ""),
            model_probability=candidate.model_prob,
            market_price=candidate.price_info.midpoint,
            confidence=candidate.ensemble_confidence,
            strategy_name=self._name,
            side=sizing.side,
            suggested_dollars=sizing.dollars,
            suggested_contracts=contracts,
            effective_edge=effective_edge,
            mid_at_signal=candidate.price_info.midpoint,
            exec_price=candidate.exec_price,
            metadata={
                "strategy_name": self._name,
                "cell": cell,
                "city": candidate.parsed.city,
                "forecast_date": candidate.parsed.forecast_date.isoformat(),
                "lead_days": lead_days,
                "lead_days_raw": lead_days_raw,
                "market_type": candidate.parsed.temp_label,
                "strike_type": _DIRECTION_TO_STRIKE_LABEL.get(
                    candidate.parsed.strike_type
                ),
                "floor_strike": candidate.parsed.floor_strike,
                "cap_strike": candidate.parsed.cap_strike,
                "description": candidate.parsed.description,
                # Pricing
                "yes_bid": candidate.price_info.yes_bid,
                "yes_ask": candidate.price_info.yes_ask,
                "bid_ask_spread": candidate.price_info.spread,
                "midpoint": candidate.price_info.midpoint,
                "mid_at_signal": candidate.price_info.midpoint,
                "exec_price": candidate.exec_price,
                # Edge / ensemble
                "raw_edge": candidate.raw_edge,
                "effective_edge_stage1": candidate.effective_edge,
                "fill_edge_stage2": effective_edge,
                "effective_edge": effective_edge,
                "ensemble_confidence": candidate.ensemble_confidence,
                "ens_mean": candidate.forecast.mean_temperature,
                "ens_std": candidate.forecast.std_dev,
                "ensemble_std_pooled": candidate.ensemble_std_pooled,
                "strike_abs_z": candidate.strike_abs_z,
                # v6 strike-distance gate inputs. The median is recorded directly
                # because inverting ``ref_strike - strike_abs_z * std`` is
                # sign-ambiguous and undefined for BETWEEN markets.
                "pooled_median": candidate.ensemble_median_pooled,
                "pooled_pstdev": candidate.ensemble_std_pooled,
                # Bias correction recorded for v6 audit. ``None`` if not
                # applied — SNR-gate-revert disclosure.
                "bias_correction_applied_f": candidate.correction_f,
                "gfs_bias_correction_f": candidate.gfs_bias_correction_f,
                "gfs_uncorrected_prob": candidate.gfs_uncorrected_prob,
                "spread_inflation_factor": candidate.spread_inflation_factor,
                "raw_ensemble_prob": candidate.raw_ensemble_prob,
                # Read from the candidate, not the holder, so a mid-scan hot-reload
                # cannot mix artifacts.
                "emos_prob_raw": candidate.emos_prob_raw,
                "emos_prob_corrected": candidate.emos_prob_corrected,
                "emos_mu_raw": candidate.emos_mu_raw,
                "emos_sigma_raw": candidate.emos_sigma_raw,
                "emos_warnings": (
                    list(candidate.emos_warnings) if candidate.emos_warnings else []
                ),
                "emos_artifact_hash": candidate.emos_coeffs_artifact_hash,
                "emos_coeffs_artifact_hash": candidate.emos_coeffs_artifact_hash,
                "emos_coeffs_loaded_at": candidate.emos_coeffs_loaded_at,
                "emos_fit_date": _emos_fit_date,
                "live_calibrator_kind": candidate.live_calibrator_kind,
                # Sizing
                "kelly_raw": sizing.kelly_raw,
                "kelly_capped": sizing.capped,
                "depth_capped": depth_capped,
                "fill_price": (depth_result.fill_price if depth_result else None),
                "fill_slippage": (depth_result.slippage if depth_result else None),
                "book_depth": (
                    depth_result.available_contracts if depth_result else None
                ),
                "book_levels": (depth_result.total_levels if depth_result else None),
                "is_same_day_market": candidate.parsed.forecast_date == _today,
                # Same one-bet-per-event defaults the cap rewrites on
                # multi-candidate events.
                "event_candidates_count": 1,
                "event_dropped_edges": [],
                "event_dropped_tickers": [],
            },
        )

    def _record_prediction(
        self,
        signal: TradeSignal,
        candidate: _MarketCandidate,
    ) -> None:
        """Log a passed-gate signal to the prediction log for calibration
        tracking. Fire-and-forget — never breaks signal generation.
        """
        if self._prediction_log is None:
            return
        try:
            rec = PredictionRecord(
                timestamp=signal.timestamp,
                strategy=signal.strategy_name,
                market_ticker=signal.market_ticker,
                side=signal.side,
                city=candidate.parsed.city,
                daily_variable=candidate.parsed.temp_label,
                strike_type=_DIRECTION_TO_STRIKE_LABEL.get(candidate.parsed.strike_type),
                floor_strike=(
                    float(candidate.parsed.floor_strike)
                    if candidate.parsed.floor_strike is not None
                    else None
                ),
                cap_strike=(
                    float(candidate.parsed.cap_strike)
                    if candidate.parsed.cap_strike is not None
                    else None
                ),
                expiration_date=candidate.parsed.forecast_date,
                raw_ensemble_prob=candidate.raw_ensemble_prob,
                calibrated_prob=signal.model_probability,
                market_price=(
                    signal.mid_at_signal
                    if not math.isnan(signal.mid_at_signal)
                    else signal.market_price
                ),
                exec_price=(
                    signal.exec_price if not math.isnan(signal.exec_price) else None
                ),
                confidence=signal.confidence,
                isotonic_fit_date=None,
                gfs_prob=(
                    candidate.per_model_probs.get("gfs_seamless")
                    if candidate.per_model_probs is not None
                    else None
                ),
                ecmwf_prob=(
                    candidate.per_model_probs.get("ecmwf_ifs025")
                    if candidate.per_model_probs is not None
                    else None
                ),
                ensemble_std_pooled=candidate.ensemble_std_pooled,
                is_same_day_market=bool(signal.metadata.get("is_same_day_market", False)),
                # The monitor partitions rolling Brier by calibrator kind.
                live_calibrator_kind=candidate.live_calibrator_kind,
                # Single-model GFS inputs and raw EMOS output (pre-floor/plateau)
                # for offline diagnostics.
                ens_mean=candidate.forecast.mean_temperature,
                ens_std=candidate.forecast.std_dev,
                emos_prob_raw=candidate.emos_prob_raw,
            )
            self._prediction_log.append(rec)
        except Exception:
            logger.exception(
                "intraday_emos_high prediction_log write failed for %s",
                signal.market_ticker,
            )


# Hermetic from weather_edge: do NOT import python.strategies.weather_edge here
# (TestHermeticDiscipline enforces it).
__all__ = ["IntradayEmosHighStrategy"]
