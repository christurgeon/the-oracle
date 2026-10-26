"""
Weather data-driven strategy for Kalshi temperature markets (KXHIGH / KXLOWT).

Uses GFS 31-member ensemble forecasts from Open-Meteo to compute the model
probability that a city's daily high (or low) temperature will exceed, fall
below, or land within a bracket threshold. Compares this probability to the
Kalshi market price to find mispricings.

Market structure:
  - Series tickers are per-city: KXHIGHNY, KXHIGHCHI, KXHIGHMIA, etc.
  - Each event (city + day) contains ~6 mutually-exclusive markets:
      1 "greater" edge (top), 1 "less" edge (bottom), ~4 "between" brackets.
  - Market dicts include strike_type, floor_strike, cap_strike — used instead
    of regex-parsing the ticker string.
  - Prices use dollar-denominated fields: yes_bid_dollars, yes_ask_dollars.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from python.analytics.audit_log import (
    AuditSinks,
    RejectionReason,
    ScanFunnel,
    build_sinks,
    flush_sinks,
    reason_numeric,
)
from python.analytics.fees import fee_per_contract as kalshi_fee
from python.analytics.prediction_log import PredictionLog, PredictionRecord
from python.analytics.sizing import SizingResult, dollars_to_contracts, fractional_kelly
from python.common import (
    DailyVariable,
    Direction,  # noqa: F401  (re-export for tests/test_shadow_calibrator.py)
    ExecutionMode,
    Side,
    et_today,
)
from python.data.bias_correction import BiasTable
from python.data.calibration import Calibrator, IdentityCalibrator, load_calibrator
from python.data.emos import EmosTable, EmosTableHolder
from python.data.kalshi_client import KalshiClient
from python.data.nws import NWSClient
from python.data.weather import (
    EnsembleForecast,
    WeatherClient,
    compute_blended_probability,
)
from python.notifications.base import AlertLevel, Notifier
from python.settings import Settings
from python.strategies.base import Strategy, TradeSignal

# Re-exports symbols that moved to the weather_parsing, weather_audit, and
# weather_evaluation sibling modules so existing
# ``from python.strategies.weather_edge import …`` imports keep working.
from python.strategies.per_day_event_cap import (
    PerDayEventCap,
    hydrate_per_day_state_from_trades_jsonl,
)
from python.strategies.same_day_gate import SameDayDecision, SameDayGate
from python.strategies.weather_audit import (  # noqa: F401
    _audit_identity_fields,
    _eval_audit_row,
    _hours_to_settlement,
    _log_emos_failure_once,
)
from python.strategies.weather_evaluation import (  # noqa: F401
    _EvalConfig,
    _MarketCandidate,
    _parse_min_calibrated_prob,
    _parse_plateau_range,
    build_forecasts_raw_rows,
    evaluate_market_pure,
    postprocess_forecasts,
)
from python.strategies.weather_parsing import (  # noqa: F401
    _DIRECTION_TO_STRIKE_LABEL,
    DepthCheck,
    MarketPriceInfo,
    OpponentSideView,
    ParsedMarket,
    ScanCandidate,
    _orderbook_price_to_probability,
    _parse_dollar_price,
    _parse_fp_int,
    _parse_orderbook_level,
    compute_fill_from_levels,
    extract_market_price_info,
    parse_weather_market,
)
from python.utils.jsonl import ForecastsRawWriter, PerDayCapBlockWriter, iter_jsonl
from python.utils.ticker_dates import SERIES_PREFIX_TO_CITY_VAR, parse_ticker_date

logger = logging.getLogger(__name__)

_VAR_BY_TAG: dict[str, DailyVariable] = {
    "high": DailyVariable.TEMP_MAX,
    "low": DailyVariable.TEMP_MIN,
}

# Derived from the single source of truth in ``python/utils/ticker_dates.py``;
# the two differ only in var vocabulary (Open-Meteo field names here, plain
# high/low tags there).
WEATHER_SERIES: dict[str, tuple[str, DailyVariable]] = {
    prefix: (city, _VAR_BY_TAG[var]) for prefix, (city, var) in SERIES_PREFIX_TO_CITY_VAR.items()
}


class WeatherEdgeStrategy(Strategy):
    """
    Weather edge strategy for Kalshi temperature markets.

    Pipeline:
        1. scan() — fetch all open weather markets from Kalshi (per-city series)
        2. evaluate(market) — for each market, fetch GFS ensemble forecast,
           compute model probability, compare to market price
        3. generate_signals() — return TradeSignal objects for markets
           where the edge exceeds the threshold (default 8 cents)

    The edge is deterministic: the GFS ensemble forecast fraction IS the
    model probability. No AI inference or subjective judgment involved.
    """

    def __init__(
        self,
        config: dict[str, Any],
        kalshi_client: KalshiClient,
        weather_client: WeatherClient,
        bankroll: float = 10_000.0,
        market_filter: dict[str, Any] | None = None,
        settings: Settings | None = None,
        bias_table: BiasTable | None = None,
        prediction_log: PredictionLog | None = None,
        notifier: Notifier | None = None,
        emos_holder: EmosTableHolder | None = None,
        same_day_count_provider: Callable[[date], int] | None = None,
    ) -> None:
        super().__init__(name="weather_edge", config=config, bankroll=bankroll)
        self._kalshi = kalshi_client
        self._weather = weather_client
        self._settings = settings or Settings()
        self._cities: list[str] = config.get(
            "cities",
            ["NYC", "CHI", "MIA", "LAX", "DEN"],
        )
        self._market_filter = market_filter or {}
        self._bias_table = bias_table
        # Falls back to identity when disabled or the artifact is missing/malformed.
        iso_path_str = config.get("isotonic_path")
        isotonic_path = Path(iso_path_str) if iso_path_str else None
        live_cal = load_calibrator(config, key="calibrator", isotonic_path=isotonic_path)
        self._calibrator: Calibrator | None = (
            None if isinstance(live_cal, IdentityCalibrator) else live_cal
        )
        # Logged for A/B comparison but never used for trading.
        shadow_cal = load_calibrator(config, key="shadow_calibrator", isotonic_path=isotonic_path)
        self._shadow_calibrator: Calibrator | None = (
            None if isinstance(shadow_cal, IdentityCalibrator) else shadow_cal
        )
        self._prediction_log = prediction_log

        self._traded_tickers: set[str] = set()

        state_dir = Path(self._settings.state_dir)
        self._audit_sinks: AuditSinks = build_sinks(
            state_dir,
            eval_enabled=bool(config.get("eval_audit_log", True)),
            candidate_enabled=bool(config.get("candidate_audit_log", True)),
            funnel_enabled=bool(config.get("scan_funnel_log", True)),
            retention_days=int(config.get("audit_log_retention_days", 14)),
            buffered=True,
        )

        # Always callable so test factories that skip setup() never hit
        # AttributeError; setup() upgrades it to an NWS-backed gate when enabled.
        self._same_day_gate: SameDayGate = SameDayGate.disabled()
        self._nws_client: NWSClient | None = None

        self._notifier: Notifier | None = notifier
        # The orchestrator owns EMOS hot-reload: it rebinds
        # ``self._emos_holder.table`` once per tick before ``generate_signals``.
        # Strategies MUST NOT reload; every read goes through the holder so a
        # rebind is visible. When ``None`` (test fixtures), ``setup()`` does a
        # one-shot load with no hot-reload.
        self._emos_holder: EmosTableHolder | None = emos_holder
        self._emos_log_dedup: dict[Any, Any] = {}
        self._forecasts_raw_writer: ForecastsRawWriter | None = None
        self._emos_halt_logged: bool = False
        self._emos_enabled_cells: frozenset[str] = frozenset(
            config.get("emos_enabled_cells", []) or []
        )

        # Returns the open-position count for a forecast_date (production:
        # ``RiskManager.count_same_day_bets``). ``None`` or rho 0.0/None disables
        # the haircut (legacy independent Kelly). INTER-SCAN ONLY: all candidates
        # in one scan see the same pre-scan count.
        self._same_day_count_provider: Callable[[date], int] | None = same_day_count_provider
        self._same_day_correlation_rho: float | None = config.get("same_day_correlation_rho", 0.0)

        max_per_day_raw = self._config.get(
            "max_bets_per_day_per_event",
            self._settings.strategies.weather.max_bets_per_day_per_event,
        )
        if max_per_day_raw is not None:
            max_per_day_raw = int(max_per_day_raw)
            if max_per_day_raw < 1:
                raise ValueError("max_bets_per_day_per_event must be >= 1 or null")
        max_per_day = max_per_day_raw
        blocks_path = Path(
            self._config.get("per_day_cap_blocks_path", state_dir / "per_day_cap_blocks.jsonl")
        )
        self._trades_log_path = Path(
            self._config.get("trades_log_path", state_dir / "trades.jsonl")
        )
        self._per_day_cap = PerDayEventCap(
            max_per_day=max_per_day,
            audit_writer=PerDayCapBlockWriter(blocks_path) if max_per_day is not None else None,
        )

    def _same_day_haircut_args(self, forecast_date: date | None) -> tuple[int, float | None]:
        """Resolve (n_existing_same_day_bets, correlation_rho) for sizing.

        Returns ``(0, None)`` when the haircut path is disabled — no provider
        plumbed in, no rho configured, rho==0.0, or forecast_date is None
        (defensive against pre-date-parse candidates). Both branches map to a
        no-op inside ``fractional_kelly`` so the legacy independent-Kelly
        result is preserved bit-for-bit.
        """
        if self._same_day_count_provider is None:
            return 0, None
        rho = self._same_day_correlation_rho
        if rho is None or rho <= 0.0:
            return 0, None
        if forecast_date is None:
            return 0, None
        try:
            n = int(self._same_day_count_provider(forecast_date))
        except Exception:  # pragma: no cover — defensive
            logger.exception(
                "same_day_count_provider raised; skipping haircut for %s",
                forecast_date,
            )
            return 0, None
        return n, rho

    # Back-compat shim for callers that assign/read ``self.emos_table``; both
    # route through the holder. The holder is SHARED with the orchestrator and
    # intraday_emos_high, so the setter mutates it for every consumer and
    # REQUIRES an existing holder: silently creating one would diverge from the
    # orchestrator's. Callers with none must first set
    # ``self._emos_holder = EmosTableHolder(table=...)``.
    @property
    def emos_table(self) -> EmosTable | None:
        return None if self._emos_holder is None else self._emos_holder.table

    @emos_table.setter
    def emos_table(self, value: EmosTable | None) -> None:
        if self._emos_holder is None:
            raise RuntimeError(
                "WeatherEdgeStrategy.emos_table setter requires an injected "
                "EmosTableHolder. Construct the strategy with emos_holder=<holder>, "
                "or construct an EmosTableHolder explicitly and assign it via "
                "self._emos_holder = ... before mutating emos_table."
            )
        if value is None:
            raise RuntimeError(
                "WeatherEdgeStrategy.emos_table cannot be set to None — the holder "
                "must contain a real EmosTable (use a disabled-sentinel via "
                "EmosTable._disabled(...) if you need a no-op table)."
            )
        self._emos_holder.table = value

    async def setup(self) -> None:
        """Upgrade the disabled SameDayGate sentinel to a live gate iff configured on,
        then load EMOS coefficients. EMOS failures fall back to identity; the
        strategy never refuses to start.
        """
        sd_cfg = self._config.get("same_day_trading") or {}
        enabled = bool(sd_cfg.get("enabled", False))
        allowed = list(sd_cfg.get("allowed_cities") or [])
        if enabled and allowed:
            self._nws_client = NWSClient(
                timeout_seconds=float(sd_cfg.get("nws_request_timeout_seconds", 4.0)),
            )
            await self._nws_client.connect()
            self._same_day_gate = SameDayGate(
                self._nws_client,
                enabled=True,
                allowed_cities=frozenset(allowed),
                cache_ttl_seconds=int(sd_cfg.get("cache_ttl_seconds", 600)),
                near_peak_ttl_seconds=int(sd_cfg.get("near_peak_ttl_seconds", 90)),
                near_peak_window_minutes=int(sd_cfg.get("near_peak_window_minutes", 120)),
                peak_cutoff_minutes=int(sd_cfg.get("peak_cutoff_minutes", 60)),
            )
            logger.info(
                "SameDayGate upgraded: cities=%s ttl=%ss near_peak_ttl=%ss",
                sorted(allowed),
                sd_cfg.get("cache_ttl_seconds", 600),
                sd_cfg.get("near_peak_ttl_seconds", 90),
            )

        # Must run even when same-day gate setup was skipped. Failures fall back
        # to identity; the strategy never refuses to start.
        state_dir = Path(self._settings.state_dir)
        self._forecasts_raw_writer = ForecastsRawWriter(state_dir / "forecasts_raw.jsonl")

        coeffs_path = self._config.get("emos_coefficients_path", "data/emos/coefficients.json")
        shadow_log_on = bool(self._config.get("emos_shadow_log", False))

        # An injected ``emos_holder`` was already loaded by the orchestrator.
        # Re-loading here would create a second EmosTable that diverges from the
        # holder the orchestrator rebinds on hot-reload. The loader below serves
        # tests and callers that omit the holder. A disabled table is accepted:
        # evaluate_market_pure falls back to identity for non-EMOS cells.
        if self._emos_holder is not None:
            injected = self._emos_holder.table
            if injected.is_loaded:
                logger.info(
                    "emos_coeffs_injected artifact_hash=%s loaded_at=%s n_cells=%d",
                    injected.artifact_hash,
                    injected.artifact_loaded_at,
                    len(injected.cells),
                )
            else:
                logger.info(
                    "EMOS coefficients disabled at orchestrator (reason=%s) - identity fallback",
                    injected.fallback_reason,
                )
            return

        try:
            candidate = EmosTable.load(coeffs_path)
            if not candidate.is_loaded:
                raise RuntimeError(f"EMOS load failed: {candidate.fallback_reason}")
            candidate.smoke_test()  # may raise
            # No holder was injected (test / legacy caller path). Construct
            # one explicitly so the setter invariant — holder must already
            # exist — holds. This holder is strategy-owned, NOT shared with
            # the orchestrator; consumers that expect cross-strategy
            # aliasing must pass ``emos_holder=`` to the constructor instead.
            self._emos_holder = EmosTableHolder(table=candidate)
            logger.info(
                "emos_coeffs_loaded artifact_hash=%s loaded_at=%s n_cells=%d",
                candidate.artifact_hash,
                candidate.artifact_loaded_at,
                len(candidate.cells),
            )
        except FileNotFoundError:
            logger.info("EMOS coefficients file not found - identity fallback")
        except Exception as exc:
            if shadow_log_on:
                logger.error("EMOS load/smoke failed: %s - identity fallback", exc)
                if self._notifier is not None:
                    try:
                        await self._notifier.send_alert(
                            f"EMOS load failed: {exc}",
                            level=AlertLevel.ERROR,
                            failure_bucket="emos_load:_global",
                        )
                    except Exception:
                        logger.exception("EMOS load alert dispatch failed")
            else:
                logger.warning("EMOS load/smoke failed: %s - identity fallback", exc)

    async def aclose(self) -> None:
        """Close the SameDayGate (NWSClient if one was constructed).

        The orchestrator's lifecycle loop wraps this call in its own
        try/except + logger.exception, so we don't double-wrap here.
        """
        await self._flush_audit_sinks()
        await self._same_day_gate.aclose()

    async def _flush_audit_sinks(self) -> None:
        """Write buffered audit rows to disk off the event loop. Fail-open."""
        try:
            await flush_sinks(self._audit_sinks)
        except Exception:
            logger.exception("audit-log flush failed")

    def mark_traded(self, ticker: str) -> None:
        """Record a ticker as traded so it's skipped in future scans."""
        self._traded_tickers.add(ticker)

    def unmark_traded(self, ticker: str) -> None:
        """Release a ticker so it can be considered again this session.

        Called by the orchestrator when a previously-placed order is
        cancelled with zero fills (TTL expiry, server-side cancellation).
        Without this, the strategy is stuck waiting for a process restart
        before it can re-attempt the same market.

        Idempotent: silently no-ops when the ticker isn't in the set.
        """
        self._traded_tickers.discard(ticker)

    def load_traded_tickers(self, shadow_log_path: str) -> None:
        """Load previously traded tickers from the shadow log file.

        Tickers whose embedded date is in the past are considered expired
        and skipped; today's and future-dated tickers are preserved so they
        block re-trading after a restart.
        """
        today = et_today()
        loaded = 0
        skipped_expired = 0

        for entry in iter_jsonl(Path(shadow_log_path)):
            ticker = entry.get("market_ticker", "")
            if not ticker:
                continue
            ticker_date = parse_ticker_date(ticker)
            if ticker_date is not None and ticker_date < today:
                skipped_expired += 1
                continue
            self._traded_tickers.add(ticker)
            loaded += 1

        logger.info(
            "Loaded %d traded ticker(s) from shadow log (%d expired, skipped)",
            loaded,
            skipped_expired,
        )

    async def scan(  # type: ignore[override]
        self,
        *,
        scan_id: str | None = None,
        funnel: ScanFunnel | None = None,
        scan_today: date | None = None,
    ) -> list[ScanCandidate]:
        """
        Fetch all open weather markets from Kalshi across configured cities.

        Queries each city-specific series ticker (e.g., KXHIGHNY, KXHIGHCHI)
        and parses markets using API-provided strike fields.

        Returns ``ScanCandidate`` objects (not raw dicts) so downstream code
        never has to reach into ``market["_parsed"]``.

        When called from ``generate_signals`` the optional ``scan_id`` and
        ``funnel`` kwargs thread diagnostic audit context through; standalone
        callers (tests) omit them to avoid a no-op instantiation.
        """
        if not self._enabled:
            logger.info("Weather edge strategy is disabled — skipping scan")
            return []

        series_to_query: list[tuple[str, str, DailyVariable]] = []
        for series_ticker, (city_code, daily_var) in WEATHER_SERIES.items():
            if city_code in self._cities:
                series_to_query.append((series_ticker, city_code, daily_var))

        if not series_to_query:
            logger.warning("No weather series match configured cities: %s", self._cities)
            return []

        logger.info(
            "Scanning %d weather series for cities: %s",
            len(series_to_query),
            [c for _, c, _ in series_to_query],
        )

        min_volume = self._market_filter.get("min_volume", 0)
        min_oi = self._market_filter.get("min_open_interest", 0)
        min_price = self._market_filter.get("min_price", 0.0)
        max_price = self._market_filter.get("max_price", 1.0)
        _sd_cfg = self._config.get("same_day_trading") or {}
        _sd_gate_timeout_s = float(_sd_cfg.get("nws_gate_timeout_seconds", 20.0))

        candidates: list[ScanCandidate] = []
        total_raw = 0
        filtered_volume = 0
        filtered_oi = 0
        filtered_price = 0
        filtered_past_date = 0
        filtered_same_day_blocked = 0

        # A scan can take 10+s (one series per second + depth checks); without
        # snapshotting ET "today" once, a midnight straddle would classify
        # some markets against the old date and some against the new one.
        scan_today = scan_today if scan_today is not None else et_today()

        _scan_id = scan_id or ""
        _eval_emit = (
            self._audit_sinks.eval_log.emit_eval if self._audit_sinks.eval_log is not None else None
        )

        def _emit_scan_reject(
            sc: ScanCandidate,
            reason: RejectionReason,
            price_info: MarketPriceInfo | None,
            *,
            extra: dict[str, Any] | None = None,
        ) -> None:
            if funnel is not None:
                funnel.increment(reason)
            if _eval_emit is None:
                return
            _eval_emit(
                _eval_audit_row(
                    scan_id=_scan_id,
                    sc=sc,
                    price_info=price_info,
                    reason=reason,
                    extra=extra,
                )
            )

        for i, (series_ticker, city_code, daily_var) in enumerate(series_to_query):
            # Stagger requests to avoid Kalshi 429 rate limits
            if i > 0:
                await asyncio.sleep(1.0)

            try:
                raw_markets, markets_truncated = await self._kalshi.iter_all_markets(
                    status="open",
                    series_ticker=series_ticker,
                )
            except Exception:
                logger.exception("Failed to fetch markets for series %s", series_ticker)
                continue

            if markets_truncated:
                logger.critical(
                    "iter_all_markets TRUNCATED for series %s — got %d markets "
                    "before MAX_PAGES cap; weather scan is INCOMPLETE",
                    series_ticker,
                    len(raw_markets),
                )
                if self._notifier is not None:
                    try:
                        # ERROR, not CRITICAL: CRITICAL bypasses TelegramNotifier's
                        # per-(bucket, hour) dedup, which would spam the operator's
                        # phone since this scan runs every 60s.
                        await self._notifier.send_alert(
                            f"WEATHER SCAN TRUNCATED: iter_all_markets hit "
                            f"MAX_PAGES for series {series_ticker} "
                            f"({len(raw_markets)} markets loaded before the cap; "
                            f"some open weather markets may not have been "
                            f"considered for trading this scan).",
                            level=AlertLevel.ERROR,
                            failure_bucket=f"iter_all_markets_truncated:{series_ticker}",
                        )
                    except Exception:
                        logger.exception("Failed to dispatch iter_all_markets truncation alert")

            total_raw += len(raw_markets)
            if funnel is not None:
                funnel.n_raw += len(raw_markets)

            for market in raw_markets:
                parsed = parse_weather_market(market, city_code, daily_var)
                if parsed is None:
                    continue

                # Same-day markets route through SameDayGate after the cheap
                # liquidity gates below, so dust markets don't trigger an NWS
                # fetch. The gate is a disabled sentinel by default — only
                # deployments that opt in via
                # ``strategies.weather.same_day_trading`` ever reach ALLOW.
                if parsed.forecast_date < scan_today:
                    logger.debug(
                        "Skipping %s — date %s is in the past",
                        parsed.ticker,
                        parsed.forecast_date,
                    )
                    filtered_past_date += 1
                    _emit_scan_reject(
                        ScanCandidate(market=market, parsed=parsed, price_info=None),
                        RejectionReason.PAST_DATE,
                        None,
                    )
                    continue

                if min_volume > 0:
                    vol = _parse_fp_int(market.get("volume_fp"))
                    if vol is not None and vol < min_volume:
                        logger.debug(
                            "%s: volume %d < min %d — skipping",
                            parsed.ticker,
                            vol,
                            min_volume,
                        )
                        filtered_volume += 1
                        _emit_scan_reject(
                            ScanCandidate(market=market, parsed=parsed, price_info=None),
                            RejectionReason.VOLUME,
                            None,
                        )
                        continue

                # Markets with few open contracts often have stale bid/ask
                # quotes that create phantom edges.
                if min_oi > 0:
                    oi = _parse_fp_int(market.get("open_interest_fp"))
                    if oi is not None and oi < min_oi:
                        logger.debug(
                            "%s: open_interest %d < min %d — skipping",
                            parsed.ticker,
                            oi,
                            min_oi,
                        )
                        filtered_oi += 1
                        _emit_scan_reject(
                            ScanCandidate(market=market, parsed=parsed, price_info=None),
                            RejectionReason.OPEN_INTEREST,
                            None,
                        )
                        continue

                # Pre-computed here so it's reused in generate_signals.
                price_info = extract_market_price_info(market)

                if min_price > 0.0 or max_price < 1.0:
                    if price_info is not None:
                        if price_info.midpoint < min_price or price_info.midpoint > max_price:
                            logger.debug(
                                "%s: midpoint %.3f outside [%.3f, %.3f] — skipping",
                                parsed.ticker,
                                price_info.midpoint,
                                min_price,
                                max_price,
                            )
                            filtered_price += 1
                            _emit_scan_reject(
                                ScanCandidate(market=market, parsed=parsed, price_info=price_info),
                                RejectionReason.PRICE_GATE,
                                price_info,
                            )
                            continue

                # Runs AFTER liquidity filters so dust markets never trigger an NWS fetch.
                if parsed.forecast_date == scan_today:
                    try:
                        decision = await asyncio.wait_for(
                            self._same_day_gate.check(
                                parsed.city,
                                parsed.forecast_date,
                                parsed.daily_variable,
                            ),
                            timeout=_sd_gate_timeout_s,
                        )
                    except TimeoutError:
                        decision = SameDayDecision.BLOCK_NWS_FETCH_FAILED

                    if decision != SameDayDecision.ALLOW:
                        filtered_same_day_blocked += 1
                        _emit_scan_reject(
                            ScanCandidate(market=market, parsed=parsed, price_info=price_info),
                            RejectionReason.SAME_DAY_BLOCKED,
                            price_info,
                            extra={"same_day_decision": decision.value},
                        )
                        continue

                candidates.append(
                    ScanCandidate(
                        market=market,
                        parsed=parsed,
                        price_info=price_info,
                    )
                )

        logger.info(
            "Found %d tradeable markets out of %d total "
            "(filtered: %d past_date, %d same_day_blocked, %d volume, "
            "%d open_interest, %d price)",
            len(candidates),
            total_raw,
            filtered_past_date,
            filtered_same_day_blocked,
            filtered_volume,
            filtered_oi,
            filtered_price,
        )
        await self._flush_audit_sinks()
        return candidates

    async def evaluate(self, market: dict[str, Any] | ScanCandidate) -> float | None:
        """
        Compute the model probability for a single weather market.

        Accepts either a raw market dict (with ``_parsed`` key from legacy
        callers) or a ``ScanCandidate``.  Uses multi-model blending when
        configured.

        Returns:
            Model probability of YES (0.0 to 1.0), or None on failure.
        """
        parsed: ParsedMarket | None
        if isinstance(market, ScanCandidate):
            parsed = market.parsed
        else:
            parsed = market.get("_parsed")
        if parsed is None:
            logger.warning(
                "Cannot evaluate market — no parsed data: %s",
                market.get("ticker") if isinstance(market, dict) else "unknown",
            )
            return None

        models: list[str] = self._config.get("ensemble_models", ["gfs_seamless"])
        weights: dict[str, float] = self._config.get(
            "ensemble_weights",
            {"gfs_seamless": 1.0},
        )

        try:
            forecasts = await self._weather.get_multi_model_forecast(
                city=parsed.city,
                forecast_date=parsed.forecast_date,
                daily_variable=parsed.daily_variable,
                models=models,
            )
        except Exception:
            logger.exception(
                "Failed to fetch ensemble forecasts for %s on %s (%s)",
                parsed.city,
                parsed.forecast_date,
                parsed.daily_variable,
            )
            return None

        if not forecasts:
            return None

        model_prob, per_model = compute_blended_probability(
            forecasts,
            weights,
            direction=parsed.strike_type,
            floor_strike=parsed.floor_strike,
            cap_strike=parsed.cap_strike,
        )

        primary = next(iter(forecasts.values()))
        logger.info(
            "Model probability for %s: %.3f (%s, mean=%.1f°F, models=%s)",
            parsed.ticker,
            model_prob,
            parsed.description,
            primary.mean_temperature,
            list(forecasts.keys()),
        )

        return model_prob

    async def generate_signals(self) -> list[TradeSignal]:
        """
        Run the full scan -> evaluate -> size pipeline.

        For each weather market where the model probability diverges from the
        market price by more than the edge threshold, generate a TradeSignal
        with Kelly-sized position.

        Decomposed into three phases:
          1. ``_prefetch_forecasts`` — parallel GFS fetch + bias correction
          2. ``_evaluate_market`` — model prob, edge, Kelly sizing → candidate
          3. ``_apply_depth_adjustment`` — orderbook VWAP, re-Kelly, depth cap
        """
        if not self._enabled:
            return []

        # One scan cycle — audit streams group by this id.
        scan_id = uuid.uuid4().hex[:12]
        scan_ts = datetime.now(tz=UTC).timestamp()
        funnel = ScanFunnel()

        # Threaded through every stage so a midnight crossing cannot classify
        # the same market against two different dates within one scan.
        scan_today: date = et_today()

        # Snapshot the holder's table once so every eval_audit row in this
        # scan_id shares one artifact_hash even if the holder is rebound
        # mid-scan. Never reload here; the orchestrator owns hot-reload.
        emos_halt = (Path(self._settings.state_dir) / "EMOS_HALT").exists()
        emos_table_for_scan: EmosTable | None
        if emos_halt:
            emos_table_for_scan = None
            if not self._emos_halt_logged:
                logger.info("EMOS_HALT present — EMOS shadow paused for this scan")
                self._emos_halt_logged = True
        else:
            self._emos_halt_logged = False
            emos_table_for_scan = self._emos_holder.table if self._emos_holder is not None else None

        # 3-tuple keys (city, var, scan_id) clear at the scan boundary. 4-tuple
        # keys (city, var, failure_bucket, utc_hour) persist across scans and
        # roll over inside _log_emos_failure_once at hour boundaries.
        self._emos_log_dedup = {
            k: v
            for k, v in self._emos_log_dedup.items()
            if not (isinstance(k, tuple) and len(k) == 3)
        }

        raw_member_sink: (
            dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] | None
        ) = {} if (emos_table_for_scan is not None and emos_table_for_scan.is_loaded) else None

        markets = await self.scan(scan_id=scan_id, funnel=funnel, scan_today=scan_today)
        if not markets:
            # Still emit the funnel so we can see stagnation (zero markets).
            self._emit_funnel(funnel, scan_id, scan_ts, n_markets=0)
            await self._flush_audit_sinks()
            return []

        (
            forecast_cache,
            bias_corrections,
            raw_ensemble_stds,
            raw_ensemble_medians,
        ) = await self._prefetch_forecasts(markets, raw_member_sink=raw_member_sink)

        # NBM-bracket-position veto (shadow-only): prefetch the deterministic
        # NBM point forecast for every unique (city, forecast_date) carrying a
        # BETWEEN-strike market. Consumed by ``evaluate_market_pure`` to
        # compute ``nbm_veto_fired`` on every between candidate (audit-only
        # while ``enabled=False``). The gate is a no-op when ``nbm_cache`` is
        # None or empty.
        nbm_veto_cfg = self._config.get("nbm_bracket_position_veto") or {}
        nbm_cache: dict[tuple[str, date, DailyVariable], float | None] | None = None
        # Fetched whenever the block is configured, regardless of ``enabled``:
        # the audit row records ``nbm_veto_fired`` either way. An operator who
        # wants to bypass the fetch (accepting null audit fields) can drop
        # the config block.
        if isinstance(nbm_veto_cfg, dict) and nbm_veto_cfg:
            timeout_s = float(nbm_veto_cfg.get("fetch_timeout_seconds", 4.0))
            nbm_cache = await self._prefetch_nbm(markets, timeout_seconds=timeout_s)

        if self._forecasts_raw_writer is not None and raw_member_sink:
            rows = build_forecasts_raw_rows(
                scan_id=scan_id,
                raw_member_sink=raw_member_sink,
                bias_corrections=bias_corrections,
                utc_today=datetime.now(UTC).date(),
            )
            if rows:
                await asyncio.to_thread(self._forecasts_raw_writer.write_batch, rows)

        bracket_cfg = self._config.get("bracket_monitor", {})
        bracket_metrics_map: dict[str, Any] = {}
        enhancer = None

        # Computed once even when the directional boost is disabled, so
        # candidate_audit can always carry per-event bracket context.
        from python.analytics.bracket_metrics import (
            BracketSumAnalyzer,
            DirectionalSignalEnhancer,
        )

        weather_maker_free = bracket_cfg.get("weather_maker_free", True)
        groups = BracketSumAnalyzer.group_by_event(markets)
        for evt, group in groups.items():
            metrics = BracketSumAnalyzer.compute_metrics(
                evt,
                group,
                weather_maker_free=weather_maker_free,
            )
            if metrics is not None:
                bracket_metrics_map[evt] = metrics

        if bracket_cfg.get("enabled", False):
            for evt, metrics in bracket_metrics_map.items():
                logger.info(
                    "BRACKET SUM: %s | n=%d bid=%.3f ask=%.3f mid=%.3f "
                    "| bid_excess=%.3f | net@10=%.4f%s",
                    evt,
                    metrics.n_brackets,
                    metrics.bid_sum,
                    metrics.ask_sum,
                    metrics.mid_sum,
                    metrics.bid_excess,
                    metrics.net_profit_by_count.get(10, 0.0),
                    " [maker_free]" if weather_maker_free else "",
                )

            enhancer = DirectionalSignalEnhancer(
                bid_threshold=bracket_cfg.get("directional_boost_threshold", 0.03),
                ask_threshold=bracket_cfg.get("directional_boost_threshold", 0.03),
                boost_factor=bracket_cfg.get("directional_boost_factor", 1.15),
                max_boost=bracket_cfg.get("max_boost", 1.30),
            )

        # Calibrated-probability floor: only enforced in live/paper. Shadow
        # mode keeps the bin observable for validation.
        exec_mode = self._settings.execution.mode
        min_cal_floor = (
            None
            if str(exec_mode) == ExecutionMode.SHADOW.value
            else _parse_min_calibrated_prob(self._config.get("min_calibrated_prob"))
        )

        cfg = _EvalConfig(
            max_spread=self._config.get("max_spread", 0.15),
            spread_skill_sensitivity=self._config.get("spread_skill_sensitivity", 0.0),
            edge_threshold=self._config.get("edge_threshold", 0.08),
            max_edge=self._config.get("max_edge", 1.0),
            kelly_fraction=self._settings.risk.kelly_fraction,
            # Pre-risk-manager ceiling only; RiskManager.check_signal applies the
            # binding cap (max_pct × effective_max_total_exposure_pct × bankroll).
            max_pct=self._settings.risk.max_single_market_pct,
            max_take_pct=self._market_filter.get("max_orderbook_take_pct", 1.0),
            depth_check_enabled=self._config.get("orderbook_depth_check", True),
            min_raw_prob_for_between=self._config.get("min_raw_prob_for_between", 0.02),
            min_distance_from_strike_std=self._config.get("min_distance_from_strike_std", 0.0),
            max_fill_slippage=self._config.get("max_fill_slippage"),
            block_raw_prob_range=_parse_plateau_range(self._config.get("block_raw_prob_range")),
            max_model_market_gap_near_term=self._config.get("max_model_market_gap_near_term"),
            min_calibrated_prob=min_cal_floor,
            emos_runtime_sigma_threshold=self._config.get("emos_runtime_sigma_threshold", 2.0),
            # The gate writes audit fields regardless of ``enabled``; the flag
            # controls whether it also rejects the candidate.
            nbm_bracket_position_veto_enabled=bool(
                (self._config.get("nbm_bracket_position_veto") or {}).get("enabled", False)
            ),
            nbm_fail_open=bool(
                (self._config.get("nbm_bracket_position_veto") or {}).get("fail_open", True)
            ),
            # Empty by default; the validated WeatherStrategyConfig dict
            # carries the per-cell threshold map when an operator enables a cell.
            boundary_filter_cells=dict(self._config.get("boundary_filter_cells") or {}),
        )

        # Stage-1 rejection funnel/audit emission happens inside
        # ``evaluate_market_pure``; counters stay serial because there is no
        # await in this loop.
        candidates_for_depth: list[tuple[ScanCandidate, _MarketCandidate, Any | None]] = []
        batch_tickers: set[str] = set()
        for sc in markets:
            ticker = sc.parsed.ticker
            if ticker in self._traded_tickers:
                logger.debug("%s: already traded — skipping", ticker)
                continue
            if ticker in batch_tickers:
                logger.debug("%s: duplicate in current batch — skipping", ticker)
                continue
            batch_tickers.add(ticker)

            n_same_day, rho_same_day = self._same_day_haircut_args(
                getattr(sc.parsed, "forecast_date", None)
            )
            candidate = self._evaluate_market(
                sc,
                forecast_cache,
                bias_corrections,
                raw_ensemble_stds,
                raw_ensemble_medians,
                cfg,
                scan_id=scan_id,
                funnel=funnel,
                emos_table=emos_table_for_scan,
                raw_member_sink=raw_member_sink,
                emos_log_dedup=self._emos_log_dedup,
                scan_today=scan_today,
                n_existing_same_day_bets=n_same_day,
                correlation_rho=rho_same_day,
                nbm_cache=nbm_cache,
            )
            if candidate is None:
                continue

            if funnel is not None:
                funnel.n_stage1_pass += 1

            event_ticker = sc.market.get("event_ticker", "")
            _bm = bracket_metrics_map.get(event_ticker)
            candidates_for_depth.append((sc, candidate, _bm))

        # Stage-1 above has no await; flush here rather than writing inline.
        await self._flush_audit_sinks()

        # Depth adjustment runs concurrently under a bounded semaphore so 80+
        # markets don't serialize behind a 0.5s sleep per /orderbook call.
        # ScanFunnel counters stay race-free (cooperative async). Signal order is
        # non-deterministic, but ``_select_one_per_event`` tie-breaks on
        # (-effective_edge, -confidence, market_ticker), so output is reproducible.
        _orderbook_concurrency = int(self._config.get("orderbook_concurrency", 2))
        _orderbook_settle_delay = float(self._config.get("orderbook_settle_delay_seconds", 0.0))
        _depth_semaphore = asyncio.Semaphore(_orderbook_concurrency)

        async def _run_depth(
            sc: ScanCandidate,
            candidate: _MarketCandidate,
            bracket_metric: Any | None,
        ) -> tuple[
            ScanCandidate,
            _MarketCandidate,
            Any | None,
            tuple[float, SizingResult, int, bool, DepthCheck | None] | None,
        ]:
            async with _depth_semaphore:
                if _orderbook_settle_delay > 0.0:
                    await asyncio.sleep(_orderbook_settle_delay)
                n_same_day, rho_same_day = self._same_day_haircut_args(
                    getattr(sc.parsed, "forecast_date", None)
                )
                result = await self._apply_depth_adjustment(
                    candidate,
                    cfg,
                    scan_id=scan_id,
                    scan_ts=scan_ts,
                    funnel=funnel,
                    bracket_metric=bracket_metric,
                    n_existing_same_day_bets=n_same_day,
                    correlation_rho=rho_same_day,
                )
            return sc, candidate, bracket_metric, result

        signals: list[TradeSignal] = []
        depth_outcomes = await asyncio.gather(
            *[_run_depth(sc, c, bm) for sc, c, bm in candidates_for_depth]
        )
        for sc, candidate, _bm, result in depth_outcomes:
            if result is None:
                continue

            effective_edge, sizing, contracts, depth_capped, depth_result = result

            bracket_boost = 1.0
            if enhancer is not None and _bm is not None:
                bracket_boost = enhancer.compute_boost(sizing.side, _bm)
                if bracket_boost > 1.0:
                    pre_boost = effective_edge
                    effective_edge *= bracket_boost
                    logger.info(
                        "%s: bracket-boosted edge %.3f -> %.3f (x%.2f)",
                        candidate.parsed.ticker,
                        pre_boost,
                        effective_edge,
                        bracket_boost,
                    )

            signal = self._build_signal(
                candidate,
                effective_edge,
                sizing,
                contracts,
                depth_capped,
                depth_result,
                bracket_boost=bracket_boost,
                bracket_bid_excess=_bm.bid_excess if _bm else None,
                bracket_bid_sum=_bm.bid_sum if _bm else None,
                scan_today=scan_today,
            )
            logger.info(
                "SIGNAL: %s | %s %s @ exec=%.2f mid=%.2f model=%.3f "
                "raw_edge=%.3f eff_edge=%.3f spread=%s conf=%.3f | "
                "$%.2f (%d contracts) | %s",
                signal.market_ticker,
                signal.side,
                candidate.parsed.description,
                candidate.exec_price,
                candidate.price_info.midpoint,
                candidate.model_prob,
                candidate.raw_edge,
                effective_edge,
                f"{candidate.price_info.spread:.3f}"
                if candidate.price_info.spread is not None
                else "N/A",
                candidate.ensemble_confidence,
                sizing.dollars,
                contracts,
                "CAPPED" if sizing.capped else "kelly",
            )
            self._record_prediction(signal, candidate)
            signals.append(signal)
            if funnel is not None:
                funnel.n_signals_fired += 1

            # Emit a PASSED row on the eval_audit stream so rejection/pass
            # distributions are scoreable from a single source. candidate_audit
            # already carries a full PASSED row (including fill_edge/fill_fee);
            # this mirrors just the common fields so downstream analyses don't
            # have to cross-join streams. ``fill_edge``/``fee`` are left
            # ``None`` here — the authoritative values live in candidate_audit.
            # Fire-and-forget, same contract as ``_record_prediction``.
            if self._audit_sinks.eval_log is not None:
                try:
                    self._audit_sinks.eval_log.emit_eval(
                        _eval_audit_row(
                            scan_id=scan_id or "",
                            sc=sc,
                            price_info=candidate.price_info,
                            reason=RejectionReason.PASSED,
                            raw_ens_prob=candidate.raw_ensemble_prob,
                            calibrated_prob=candidate.model_prob,
                            shadow_prob=candidate.shadow_prob,
                            correction_f=candidate.correction_f,
                            ens_std=candidate.ensemble_std_pooled,
                            strike_abs_z=candidate.strike_abs_z,
                            exec_price=candidate.exec_price
                            if not math.isnan(candidate.exec_price)
                            else None,
                            raw_edge=candidate.raw_edge,
                            effective_edge=effective_edge,
                            fill_edge=None,
                            fee=None,
                            # Must be set here, not just on REJECTED rows —
                            # otherwise EMOS-promoted trades have NULL
                            # ``live_calibrator_kind`` on the row that traded,
                            # breaking the monthly per-cell re-evaluation's
                            # filter on ``live_calibrator_kind == "emos_raw"``.
                            live_calibrator_kind=candidate.live_calibrator_kind,
                            emos_prob_raw=candidate.emos_prob_raw,
                            emos_prob_corrected=candidate.emos_prob_corrected,
                            emos_mu_raw=candidate.emos_mu_raw,
                            emos_sigma_raw=candidate.emos_sigma_raw,
                            emos_mu_corr=candidate.emos_mu_corr,
                            emos_sigma_corr=candidate.emos_sigma_corr,
                            emos_cdf_at_floor_raw=candidate.emos_cdf_at_floor_raw,
                            emos_cdf_at_cap_raw=candidate.emos_cdf_at_cap_raw,
                            emos_cdf_at_floor_corr=candidate.emos_cdf_at_floor_corr,
                            emos_cdf_at_cap_corr=candidate.emos_cdf_at_cap_corr,
                            emos_warnings=list(candidate.emos_warnings),
                            emos_lead=candidate.emos_lead,
                            emos_disabled_reason=candidate.emos_disabled_reason,
                            emos_coeffs_artifact_hash=candidate.emos_coeffs_artifact_hash,
                            emos_coeffs_loaded_at=candidate.emos_coeffs_loaded_at,
                            isotonic_artifact_hash=candidate.isotonic_artifact_hash,
                            gfs_bias_correction_f=candidate.gfs_bias_correction_f,
                            # ``None`` for non-between markets or an unavailable NBM
                            # entry under fail-open.
                            nbm_temperature=candidate.nbm_temperature,
                            nbm_veto_fired=candidate.nbm_veto_fired,
                        )
                    )
                except Exception:
                    logger.exception(
                        "eval_audit PASSED emission failed for %s",
                        candidate.parsed.ticker,
                    )

        # Calibration audit streams (predictions.jsonl, eval_audit.jsonl)
        # intentionally fire before this cap so the calibrator sees the
        # unconstrained model output.
        now_utc = datetime.now(UTC)
        utc_today = now_utc.date().isoformat()
        self._per_day_cap.maybe_reset(
            utc_today,
            hydrate_fn=lambda d: hydrate_per_day_state_from_trades_jsonl(
                utc_today=d,
                trades_path=self._trades_log_path,
            ),
        )
        scan_ts_iso = datetime.fromtimestamp(scan_ts, tz=UTC).isoformat()
        signals, n_blocked = self._per_day_cap.filter_and_select(
            signals,
            scan_id=scan_id,
            scan_ts=scan_ts_iso,
        )
        if funnel is not None and n_blocked:
            funnel.n_per_day_cap_blocked += n_blocked
        # The cap commit happens in the orchestrator after execute_batch, once
        # the order_id is known.

        logger.info(
            "Weather edge: generated %d signal(s) from %d market(s) (%d suppressed by per-day cap)",
            len(signals),
            len(markets),
            n_blocked,
        )
        self._emit_funnel(funnel, scan_id, scan_ts, n_markets=len(markets))
        await self._flush_audit_sinks()
        return signals

    async def _prefetch_forecasts(
        self,
        markets: list[ScanCandidate],
        *,
        raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]
        | None = None,
    ) -> tuple[
        dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]],
        dict[tuple[str, date, DailyVariable], float],
        dict[tuple[str, date, DailyVariable], float],
        dict[tuple[str, date, DailyVariable], float],
    ]:
        """Fetch ensemble forecasts from all configured models in parallel.

        Returns:
            (forecast_cache, bias_corrections, raw_ensemble_stds,
             raw_ensemble_medians) — forecast_cache maps
            ``(city, date, daily_variable)`` to ``{model_name: EnsembleForecast}``;
            bias_corrections maps the same key to the applied GFS correction in
            °F (0.0 when uncorrected); raw_ensemble_stds maps the same key to
            the pre-inflation pooled ensemble std_dev in °F (for
            spread_skill_sensitivity); raw_ensemble_medians maps the same key
            to the pre-inflation pooled ensemble median (for the
            strike-proximity gate).
        """
        models: list[str] = self._config.get("ensemble_models", ["gfs_seamless"])

        seen_keys: set[tuple[str, date, DailyVariable]] = set()
        for sc in markets:
            key = (sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable)
            seen_keys.add(key)

        unique_keys = list(seen_keys)
        # Per-scan concurrency hint. NOTE: the AUTHORITATIVE limiter on outbound
        # Open-Meteo requests is the client-level semaphore inside WeatherClient
        # (weather.forecast_concurrency); this Semaphore(4) only bounds how many
        # multi-model coroutines are in flight here and is no longer the real
        # rate-limit guard.
        semaphore = asyncio.Semaphore(4)

        # Fail-soft bounds so a slow or 429-storming Open-Meteo cannot push a
        # tick past the systemd WatchdogSec. A cell with no forecast is omitted
        # and retried next tick.
        per_fetch_timeout = float(self._config.get("forecast_fetch_timeout_seconds", 15.0))
        overall_timeout = float(self._config.get("forecast_prefetch_timeout_seconds", 45.0))

        async def _fetch(
            city: str,
            forecast_date: date,
            daily_variable: DailyVariable,
        ) -> dict[str, EnsembleForecast]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._weather.get_multi_model_forecast(
                            city=city,
                            forecast_date=forecast_date,
                            daily_variable=daily_variable,
                            models=models,
                        ),
                        timeout=per_fetch_timeout,
                    )
                except Exception as exc:  # noqa: BLE001
                    # TimeoutError, WeatherAPIError, RuntimeError, etc. — surface
                    # an empty dict so the cell is skipped for this tick.
                    logger.warning(
                        "Ensemble fetch failed/timed out for %s on %s (%s): %s",
                        city,
                        forecast_date,
                        daily_variable.value,
                        exc,
                    )
                    return {}

        try:
            forecast_results = await asyncio.wait_for(
                asyncio.gather(*(_fetch(city, dt, var) for city, dt, var in unique_keys)),
                timeout=overall_timeout,
            )
        except TimeoutError:
            # Overall ceiling hit — abandon the whole fan-out for this tick. The
            # gather is cancelled by wait_for; we degrade to no forecasts rather
            # than starve the watchdog.
            logger.warning(
                "Ensemble prefetch exceeded overall budget (%.1fs) for %d cell(s) "
                "— skipping forecasts this tick",
                overall_timeout,
                len(unique_keys),
            )
            forecast_results = [{} for _ in unique_keys]
        forecast_cache: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]] = {}
        for key, result in zip(unique_keys, forecast_results):
            if result:  # skip empty dicts (all models failed)
                forecast_cache[key] = result

        return postprocess_forecasts(
            forecast_cache=forecast_cache,
            bias_table=self._bias_table,
            bias_shrinkage=self._config.get("bias_shrinkage", 0.3),
            bias_max_correction_f=self._config.get("bias_max_correction_f", 5.0),
            spread_inflation_overrides=self._config.get("spread_inflation_factors"),
            raw_member_sink=raw_member_sink,
        )

    async def _prefetch_nbm(
        self,
        markets: list[ScanCandidate],
        *,
        timeout_seconds: float,
    ) -> dict[tuple[str, date, DailyVariable], float | None]:
        """Fetch the NBM deterministic point forecast for every unique
        ``(city, forecast_date, daily_variable)`` triple on the scan's
        BETWEEN-strike markets.

        Mirrors the dedup + bounded-concurrency pattern in
        ``_prefetch_forecasts``: collect unique keys via ``seen_keys`` (one
        fetch per scan per cell), gather under ``asyncio.Semaphore(4)``, and
        wrap each fetch in ``asyncio.wait_for(..., timeout=timeout_seconds)``.

        Only BETWEEN-strike candidates participate — the veto rule applies
        only to between markets, so fetching for greater/less candidates
        would be pure overhead.

        ``daily_variable`` is part of the key because KXHIGH (TEMP_MAX) and
        KXLOWT (TEMP_MIN) brackets resolve against different observations
        and therefore need different NBM forecasts. Conflating them would
        silently compare a low-temp bracket against the NBM high forecast.

        Returns a cache mapping
        ``(city, forecast_date, daily_variable) → float | None``.
        ``float`` = successful fetch (NBM temperature in °F).
        ``None`` = fetch raised or timed out — surfaced to
        ``evaluate_market_pure`` which routes through ``cfg.nbm_fail_open``.

        Triples with no between-strike markets in the current scan are NOT
        in the returned dict (the gate inside ``evaluate_market_pure``
        treats absent keys as "gate not evaluated").
        """
        seen_keys: set[tuple[str, date, DailyVariable]] = set()
        for sc in markets:
            if sc.parsed.strike_type != Direction.BETWEEN:
                continue
            seen_keys.add((sc.parsed.city, sc.parsed.forecast_date, sc.parsed.daily_variable))

        if not seen_keys:
            return {}

        unique_keys = list(seen_keys)
        # Match _prefetch_forecasts' bounded concurrency so a 7-city × 3-day
        # fan-out doesn't slam the Open-Meteo NBM endpoint.
        semaphore = asyncio.Semaphore(4)

        async def _fetch_one(
            city: str, fc_date: date, daily_variable: DailyVariable
        ) -> tuple[tuple[str, date, DailyVariable], float | None]:
            async with semaphore:
                try:
                    forecast = await asyncio.wait_for(
                        self._weather.get_nbm_point_forecast(
                            city=city,
                            forecast_date=fc_date,
                            daily_variable=daily_variable,
                        ),
                        timeout=timeout_seconds,
                    )
                    return (city, fc_date, daily_variable), float(forecast.temperature)
                except Exception as exc:  # noqa: BLE001
                    # Timeouts, httpx/WeatherAPIError, "client not connected" and
                    # value-coercion failures all become a ``None`` cache value so
                    # the gate routes through ``cfg.nbm_fail_open``.
                    logger.warning(
                        "NBM fetch failed for %s on %s (%s): %s",
                        city,
                        fc_date,
                        daily_variable.value,
                        exc,
                    )
                    return (city, fc_date, daily_variable), None

        results = await asyncio.gather(*[_fetch_one(c, d, v) for c, d, v in unique_keys])
        cache: dict[tuple[str, date, DailyVariable], float | None] = {}
        for key, value in results:
            cache[key] = value
        return cache

    def _evaluate_market(
        self,
        sc: ScanCandidate,
        forecast_cache: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]],
        bias_corrections: dict[tuple[str, date, DailyVariable], float],
        raw_ensemble_stds: dict[tuple[str, date, DailyVariable], float],
        raw_ensemble_medians: dict[tuple[str, date, DailyVariable], float],
        cfg: _EvalConfig,
        *,
        scan_id: str | None = None,
        funnel: ScanFunnel | None = None,
        emos_table: EmosTable | None = None,
        raw_member_sink: dict[tuple[str, date, DailyVariable], dict[str, EnsembleForecast]]
        | None = None,
        emos_log_dedup: dict[Any, Any] | None = None,
        scan_today: date | None = None,
        n_existing_same_day_bets: int = 0,
        correlation_rho: float | None = None,
        nbm_cache: dict[tuple[str, date, DailyVariable], float | None] | None = None,
    ) -> _MarketCandidate | None:
        """Delegate to ``evaluate_market_pure`` with this instance's state.

        Kept as a thin wrapper so tests and the backtest engine share the same
        evaluation logic; see ``evaluate_market_pure`` for the full pipeline.

        ``emos_table`` / ``raw_member_sink`` / ``emos_log_dedup`` are forwarded
        to ``evaluate_market_pure`` for the EMOS shadow path; the backtest
        path defaults all three to ``None`` (identity fallback).

        ``n_existing_same_day_bets`` and ``correlation_rho`` enable the
        same-day Kelly haircut; both are forwarded into ``fractional_kelly``
        through ``evaluate_market_pure``. Default 0 / ``None`` preserves the
        legacy independent-Kelly path (no haircut).
        """
        isotonic_hash = (
            getattr(self._calibrator, "artifact_hash", None) if self._calibrator else None
        )
        return evaluate_market_pure(
            sc,
            forecast_cache,
            bias_corrections,
            raw_ensemble_stds,
            raw_ensemble_medians,
            cfg,
            calibrator=self._calibrator,
            shadow_calibrator=self._shadow_calibrator,
            weights=self._config.get("ensemble_weights", {"gfs_seamless": 1.0}),
            weather_client=self._weather,
            bankroll=self._bankroll,
            excluded_city_vars=self._config.get("excluded_city_vars", []),
            spread_inflation_overrides=self._config.get("spread_inflation_factors"),
            emos_table=emos_table,
            raw_member_sink=raw_member_sink,
            emos_log_dedup=emos_log_dedup,
            emos_enabled_cells=self._emos_enabled_cells,
            isotonic_artifact_hash=isotonic_hash,
            scan_id=scan_id,
            sinks=self._audit_sinks,
            funnel=funnel,
            scan_today=scan_today,
            n_existing_same_day_bets=n_existing_same_day_bets,
            correlation_rho=correlation_rho,
            nbm_cache=nbm_cache,
        )

    async def _apply_depth_adjustment(
        self,
        candidate: _MarketCandidate,
        cfg: _EvalConfig,
        *,
        scan_id: str | None = None,
        scan_ts: float | None = None,
        funnel: ScanFunnel | None = None,
        bracket_metric: Any | None = None,
        n_existing_same_day_bets: int = 0,
        correlation_rho: float | None = None,
    ) -> tuple[float, SizingResult, int, bool, DepthCheck | None] | None:
        """Apply orderbook depth check and fallback depth cap.

        Returns ``(effective_edge, sizing, contracts, depth_capped, depth_result)``
        or ``None`` if the market should be skipped (stale book or fill-edge
        below threshold).

        When ``scan_id`` is provided and the ``candidate_audit`` sink is
        enabled, emits one ``candidate_audit.jsonl`` row per call at each
        of the four terminal paths (``ORDERBOOK_THIN``,
        ``FILL_EDGE_NEGATIVE``, ``KELLY_ZERO``, ``PASSED``).
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
                    bracket_metric=bracket_metric,
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

                fill_fee = kalshi_fee(depth_result.fill_price)
                # Mirrors weather_evaluation.py: EMOS already prices spread via
                # (c, d), so ensemble_confidence is 1.0 for "emos_raw" rows.
                # isotonic_fallback rows keep the legacy discount.
                ec = (
                    1.0
                    if candidate.live_calibrator_kind == "emos_raw"
                    else candidate.ensemble_confidence
                )
                if sizing.side == Side.YES:
                    fill_edge = (candidate.model_prob - depth_result.fill_price) * ec - fill_fee
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
                    logger.info(
                        "%s: fill slippage %.3f > max %.3f (VWAP=%.3f, side=%s) — skipping",
                        candidate.parsed.ticker,
                        depth_result.slippage,
                        cfg.max_fill_slippage,
                        depth_result.fill_price,
                        sizing.side,
                    )
                    if funnel is not None:
                        funnel.n_fill_slippage += 1
                    _emit_candidate(
                        RejectionReason.FILL_SLIPPAGE,
                        fill_edge=fill_edge,
                        fill_fee=fill_fee,
                    )
                    return None

                if fill_edge < cfg.edge_threshold:
                    logger.info(
                        "%s: fill-adjusted edge %.3f < threshold %.3f "
                        "(VWAP=%.3f, slippage=%.3f, fee=%.3f, side=%s) — skipping",
                        candidate.parsed.ticker,
                        fill_edge,
                        cfg.edge_threshold,
                        depth_result.fill_price,
                        depth_result.slippage,
                        fill_fee,
                        sizing.side,
                    )
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
                    n_existing_same_day_bets=n_existing_same_day_bets,
                    correlation_rho=correlation_rho,
                )
                # Safety guard: a depth-induced flip from YES↔NO means the
                # VWAP fill price moved through model_prob, so the original
                # sizing was based on an obsolete view of the trade. Drop
                # the candidate rather than executing the inverted bet.
                if sizing.side not in (Side.NONE,) and sizing.side != pre_depth_side:
                    logger.critical(
                        "%s: Kelly side flipped post-depth (%s → %s); "
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
                    max_contracts = max(
                        int(depth_result.available_contracts * cfg.max_take_pct),
                        1,
                    )
                else:
                    max_contracts = depth_result.available_contracts
                if contracts > max_contracts:
                    logger.info(
                        "%s: capping contracts %d → %d (book_depth=%d, take=%.0f%%)",
                        candidate.parsed.ticker,
                        contracts,
                        max_contracts,
                        depth_result.available_contracts,
                        cfg.max_take_pct * 100,
                    )
                    contracts = max_contracts
                    depth_capped = True

        # Best-level depth cap when depth check is disabled or the orderbook API call failed.
        if depth_result is None and cfg.max_take_pct < 1.0:
            price_info = candidate.price_info
            depth = price_info.yes_ask_size if sizing.side == Side.YES else price_info.yes_bid_size
            if depth is not None and depth > 0:
                max_contracts = max(int(depth * cfg.max_take_pct), 1)
                if contracts > max_contracts:
                    logger.info(
                        "%s: capping contracts %d → %d (depth=%d, take=%.0f%%)",
                        candidate.parsed.ticker,
                        contracts,
                        max_contracts,
                        depth,
                        cfg.max_take_pct * 100,
                    )
                    contracts = max_contracts
                    depth_capped = True

        _emit_candidate(
            RejectionReason.PASSED,
            fill_edge=fill_edge_final,
            fill_fee=fill_fee_final,
        )
        return effective_edge, sizing, contracts, depth_capped, depth_result

    def _record_prediction(
        self,
        signal: TradeSignal,
        candidate: _MarketCandidate,
    ) -> None:
        """Log a passed-gate signal to the prediction log for calibration tracking.

        Fire-and-forget: any failure here must never break signal generation.
        """
        if self._prediction_log is None:
            return
        try:
            isotonic_fit_date: str | None = None
            if self._calibrator is not None:
                md = self._calibrator.metadata
                if isinstance(md, dict):
                    raw = md.get("fit_date")
                    if raw is not None:
                        isotonic_fit_date = str(raw)
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
                exec_price=(signal.exec_price if not math.isnan(signal.exec_price) else None),
                confidence=signal.confidence,
                isotonic_fit_date=isotonic_fit_date,
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
                # The calibrator that produced ``calibrated_prob`` for this
                # row. The downstream monitor partitions rolling Brier by
                # this so isotonic and EMOS rows don't mix.
                live_calibrator_kind=candidate.live_calibrator_kind,
            )
            self._prediction_log.append(rec)
        except Exception:
            logger.exception("prediction_log write failed for %s", signal.market_ticker)

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
        bracket_metric: Any | None,
        fill_edge: float | None,
        fill_fee: float | None,
        reason: RejectionReason,
        scan_id: str,
        scan_ts: float | None,
    ) -> dict[str, Any]:
        """Serialize the pre-depth-check candidate + depth-check outcome into one audit row.

        Called at all four ``_apply_depth_adjustment`` terminal paths.
        """
        price_info = candidate.price_info
        market = candidate.market

        our_side_available = depth_result.available_contracts if depth_result else None
        our_side_levels = depth_result.total_levels if depth_result else None
        # "Best" ask = fill price minus slippage (the top-level price).
        our_side_best_price = (
            depth_result.fill_price - depth_result.slippage
            if depth_result and depth_result.fill_price > 0.0
            else None
        )

        bracket_n = getattr(bracket_metric, "n_brackets", None)
        bracket_bid_sum = getattr(bracket_metric, "bid_sum", None)
        bracket_ask_sum = getattr(bracket_metric, "ask_sum", None)
        bracket_bid_excess = getattr(bracket_metric, "bid_excess", None)
        bracket_n_missing_bids = getattr(bracket_metric, "n_missing_bids", None)

        row: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "scan_id": scan_id,
            "scan_ts": scan_ts,
            **_audit_identity_fields(candidate.parsed, market),
            "close_time": market.get("close_time", ""),
            "yes_bid": price_info.yes_bid,
            "yes_ask": price_info.yes_ask,
            "mid": price_info.midpoint,
            "exec_price": candidate.exec_price if not math.isnan(candidate.exec_price) else None,
            "spread": price_info.spread,
            "volume": price_info.volume,
            "oi": price_info.open_interest,
            "yes_bid_size": price_info.yes_bid_size,
            "yes_ask_size": price_info.yes_ask_size,
            "pre_isotonic_prob": candidate.raw_ensemble_prob,
            "calibrated_prob": candidate.model_prob,
            "shadow_prob": candidate.shadow_prob,
            "correction_f": candidate.correction_f,
            # ``ens_std_primary_model`` is the std of the *first* forecast only
            # (typically GFS), for diagnostic comparisons. ``ens_std_pooled``
            # is the across-model pooled std and drives ``ensemble_confidence``
            # — spread-skill analyses must use the pooled value, not the
            # primary one. (The eval_audit stream calls the pooled value
            # ``ens_std``; this stream keeps both explicit.)
            "ens_std_primary_model": candidate.forecast.std_dev,
            "ens_std_pooled": candidate.ensemble_std_pooled,
            "ensemble_confidence": candidate.ensemble_confidence,
            "per_model_probs": candidate.per_model_probs,
            "spread_inflation_factor": candidate.spread_inflation_factor,
            "strike_abs_z": candidate.strike_abs_z,
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
            "our_side_available": our_side_available,
            "our_side_levels": our_side_levels,
            "our_side_best_price": our_side_best_price,
            "opp_side_available": opp_view.available if opp_view else None,
            "opp_side_levels": opp_view.levels if opp_view else None,
            "opp_side_best_price": opp_view.best_price if opp_view else None,
            "bracket_n": bracket_n,
            "bracket_bid_sum": bracket_bid_sum,
            "bracket_ask_sum": bracket_ask_sum,
            "bracket_bid_excess": bracket_bid_excess,
            "bracket_n_missing_bids": bracket_n_missing_bids,
            "reason": reason.value,
            "reason_numeric": reason_numeric(reason),
            # Echo config so rows are self-describing when the threshold moves.
            "edge_threshold": cfg.edge_threshold,
            "min_book_depth": self._config.get("min_book_depth", 5),
            # Mirrors the field on eval_audit + TradeSignal metadata so the
            # monthly per-cell re-evaluation can join the three streams cleanly.
            "live_calibrator_kind": candidate.live_calibrator_kind,
        }
        return row

    def _emit_funnel(
        self,
        funnel: ScanFunnel,
        scan_id: str,
        scan_ts: float,
        *,
        n_markets: int,
    ) -> None:
        """Write one ``scan_funnel.jsonl`` row for this scan cycle.

        Swallows every error — a failed funnel write must never break
        signal generation or stall the orchestrator.
        """
        if self._audit_sinks.funnel_log is None:
            return
        try:
            self._audit_sinks.funnel_log.emit_funnel(
                funnel,
                scan_id,
                extra={
                    "scan_ts": scan_ts,
                    "n_markets": n_markets,
                },
            )
        except Exception:
            logger.exception("scan_funnel emission failed for scan_id=%s", scan_id)

    def _build_signal(
        self,
        candidate: _MarketCandidate,
        effective_edge: float,
        sizing: SizingResult,
        contracts: int,
        depth_capped: bool,
        depth_result: DepthCheck | None,
        *,
        bracket_boost: float = 1.0,
        bracket_bid_excess: float | None = None,
        bracket_bid_sum: float | None = None,
        scan_today: date | None = None,
    ) -> TradeSignal:
        """Construct the final TradeSignal from a validated candidate."""
        _today = scan_today if scan_today is not None else et_today()
        is_same_day_market = candidate.parsed.forecast_date == _today
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
                "city": candidate.parsed.city,
                "forecast_date": candidate.parsed.forecast_date.isoformat(),
                "market_type": candidate.parsed.temp_label,
                "strike_type": candidate.parsed.strike_type,
                "floor_strike": candidate.parsed.floor_strike,
                "cap_strike": candidate.parsed.cap_strike,
                "description": candidate.parsed.description,
                "kelly_raw": sizing.kelly_raw,
                "kelly_capped": sizing.capped,
                "yes_bid": candidate.price_info.yes_bid,
                "yes_ask": candidate.price_info.yes_ask,
                "bid_ask_spread": candidate.price_info.spread,
                "exec_price": candidate.exec_price,
                "midpoint": candidate.price_info.midpoint,
                "ensemble_std_dev": candidate.forecast.std_dev,
                "ensemble_std_pooled": candidate.ensemble_std_pooled,
                "ensemble_confidence": candidate.ensemble_confidence,
                "ensemble_model": candidate.forecast.model,
                "per_model_probs": candidate.per_model_probs,
                "raw_edge": candidate.raw_edge,
                "effective_edge": effective_edge,
                "volume": candidate.price_info.volume,
                "open_interest": candidate.price_info.open_interest,
                "yes_ask_size": candidate.price_info.yes_ask_size,
                "yes_bid_size": candidate.price_info.yes_bid_size,
                "depth_capped": depth_capped,
                "fill_price": (depth_result.fill_price if depth_result else None),
                "fill_slippage": (depth_result.slippage if depth_result else None),
                "book_depth": (depth_result.available_contracts if depth_result else None),
                "book_levels": (depth_result.total_levels if depth_result else None),
                "close_time": candidate.market.get("close_time", ""),
                # 'bias_correction_f' here is a SEPARATE field from
                # 'gfs_bias_correction_f' in the audit-row schema: this one is
                # the realized correction attached to the trade signal; that
                # one is the pre-floor-block emission for forecasts_raw / eval_audit.
                "bias_correction_f": candidate.correction_f,
                # ``None`` when no bias correction was applied — see
                # ``_MarketCandidate.gfs_uncorrected_prob`` for semantics.
                "gfs_uncorrected_prob": candidate.gfs_uncorrected_prob,
                "spread_inflation_factor": candidate.spread_inflation_factor,
                "bracket_boost": bracket_boost,
                "bracket_bid_excess": bracket_bid_excess,
                "bracket_bid_sum": bracket_bid_sum,
                "isotonic_calibrated": self._calibrator is not None,
                "raw_ensemble_prob": candidate.raw_ensemble_prob,
                "strike_abs_z": candidate.strike_abs_z,
                "is_same_day_market": is_same_day_market,
                # The one-bet-per-event cap rewrites these on the winner of
                # multi-candidate events; defaults here for the common case.
                "event_candidates_count": 1,
                "event_dropped_edges": [],
                "event_dropped_tickers": [],
                # Always emitted so state/trades.jsonl can be joined to signal
                # metadata for counterfactual P&L. ``emos_warnings`` is copied
                # so mutating the signal cannot alter the candidate.
                "emos_prob_raw": candidate.emos_prob_raw,
                "emos_prob_corrected": candidate.emos_prob_corrected,
                "emos_mu_raw": candidate.emos_mu_raw,
                "emos_sigma_raw": candidate.emos_sigma_raw,
                "emos_warnings": (list(candidate.emos_warnings) if candidate.emos_warnings else []),
                "emos_coeffs_artifact_hash": candidate.emos_coeffs_artifact_hash,
                # Read from the candidate (pinned per-evaluation), NOT from
                # self.emos_table — a mid-scan hot reload would otherwise
                # produce a signal where the hash came from the pre-swap
                # table and loaded_at from the post-swap table.
                "emos_coeffs_loaded_at": candidate.emos_coeffs_loaded_at,
                "live_calibrator_kind": candidate.live_calibrator_kind,
            },
        )

    # Back-compat alias: several tests call this through the class
    # (`WeatherEdgeStrategy._extract_market_price_info(...)`); the module-level
    # `extract_market_price_info` (in weather_parsing) is the canonical form.
    _extract_market_price_info = staticmethod(extract_market_price_info)

    async def _check_orderbook_depth(
        self,
        ticker: str,
        side: Side,
        target_contracts: int,
    ) -> tuple[DepthCheck | None, OpponentSideView | None]:
        """
        Fetch the full orderbook and compute realistic fill cost.

        Converts the Kalshi orderbook (YES bids / NO bids) into ask levels
        for the target side and computes VWAP fill price. Also returns a
        summary of the opposite-side book (i.e. the side we'd exit on)
        from the same API response, so the candidate audit can capture
        both sides of the quote without an extra API round-trip.

        Returns ``(DepthCheck, OpponentSideView)`` with both ``None`` iff
        the API call itself fails (caller falls back to best-level depth
        cap).
        """
        try:
            data = await self._kalshi.get_orderbook(
                ticker, depth=self._config.get("book_depth_levels", 10)
            )
        except Exception:
            logger.warning(
                "Failed to fetch orderbook for %s",
                ticker,
                exc_info=True,
            )
            return None, None

        orderbook = data.get("orderbook_fp", {})

        # To buy our side, we cross against the opposite side's bids.
        # Kalshi ``orderbook_fp`` has ``yes_dollars`` and ``no_dollars`` —
        # each a list of [price_dollars_str, qty_str] pairs.
        # A NO bid at price P creates a YES ask at (1 - P).
        # A YES bid at price P creates a NO ask at (1 - P).
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

        # Defensive against API returning levels in unexpected order.
        ask_levels.sort(key=lambda x: x[0])

        # Summarise the exit side of the book — the bids we'd hit to
        # close the position, quoted from our perspective.
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
