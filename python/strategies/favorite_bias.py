"""Model-free favorite-bias strategy: buy YES on underpriced greater/less favorites.

No forecast/EMOS — the only learned input is the price->P(YES) isotonic map g(mid).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from python.analytics.fees import fee_per_contract
from python.analytics.sizing import dollars_to_contracts, fractional_kelly
from python.common import Side
from python.data.calibration import IsotonicCalibrator
from python.data.fb_climatology import FbClimatologyTable, clim_yes_prob, strike_z
from python.settings import Settings
from python.strategies.base import Strategy, TradeSignal
from python.strategies.per_day_event_cap import (
    PerDayEventCap,
    hydrate_per_day_state_from_trades_jsonl,
)
from python.strategies.weather_parsing import (
    _orderbook_price_to_probability,
    _parse_orderbook_level,
)
from python.utils.ticker_dates import parse_ticker_city_var, parse_ticker_date

logger = logging.getLogger(__name__)


def _append_jsonl_atomic(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row atomically (PIPE_BUF single-write), mirroring
    python/utils/jsonl.py:ForecastsRawWriter's per-record approach. There is
    no ``append_jsonl`` helper in jsonl.py — writer classes only."""
    import json as _json
    import os as _os

    path.parent.mkdir(parents=True, exist_ok=True)
    data = (_json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
    fd = _os.open(str(path), _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
    try:
        _os.write(fd, data)
    finally:
        _os.close(fd)


@dataclass(frozen=True)
class FBMarket:
    ticker: str
    city: str
    var: str  # "high" | "low"
    strike_type: str
    floor_strike: float | None
    cap_strike: float | None
    yes_bid: float
    yes_ask: float
    mid: float
    volume_fp: int
    forecast_date: date
    close_time: str


def _to_float(v: Any) -> float | None:
    """Coerce a value to float, tolerating the live Kalshi API's string-typed
    numerics (e.g. yes_ask_dollars == "0.5500"). Returns None for None/""/garbage."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_market(m: dict[str, Any]) -> FBMarket | None:
    t = m.get("ticker", "")
    # The live Kalshi /markets API returns prices as STRINGS ("0.5500"); the
    # snapshot collector normalizes them to floats. Coerce defensively BEFORE any
    # numeric comparison — a raw string here would raise TypeError on `<`.
    ya = _to_float(m.get("yes_ask_dollars"))
    yb = _to_float(m.get("yes_bid_dollars"))
    # Require a genuine two-sided quote: 0 < yb < ya < 1
    # (matches fitter population; a zero-bid market has no real fill quality signal)
    if ya is None or yb is None or not (0.0 < yb < ya < 1.0):
        return None
    # Derive city and var from the ticker — live Kalshi /markets API payloads do NOT
    # carry `city` or `daily_variable` fields (those are snapshot-collector additions).
    # parse_ticker_city_var returns None for non-weather tickers (e.g. KXINX).
    cv = parse_ticker_city_var(t)
    if cv is None:
        return None
    city, var = cv
    fd = parse_ticker_date(t)
    if fd is None:
        return None
    try:
        volume_fp = int(float(m.get("volume_fp", 0) or 0))
    except (TypeError, ValueError):
        volume_fp = 0
    return FBMarket(
        ticker=t,
        city=city,
        var=var,
        strike_type=m.get("strike_type", ""),
        floor_strike=_to_float(m.get("floor_strike")),
        cap_strike=_to_float(m.get("cap_strike")),
        yes_bid=yb,
        yes_ask=ya,
        mid=(ya + yb) / 2.0,
        volume_fp=volume_fp,
        forecast_date=fd,
        close_time=m.get("close_time") or str(m.get("expiration_ts", "")),
    )


@dataclass(frozen=True)
class BookWalkPlan:
    limit_price: float
    vwap: float
    contracts: int  # visible contracts at or below limit_price, capped at the target


def yes_ask_levels(orderbook: dict[str, Any]) -> list[tuple[float, int]]:
    """YES ask ladder ``[(price, qty)]`` ascending, derived from the NO bids."""
    levels: list[tuple[float, int]] = []
    for level in orderbook.get("orderbook_fp", {}).get("no_dollars", []) or []:
        parsed = _parse_orderbook_level(level)
        if parsed is None:
            continue
        price = round(_orderbook_price_to_probability(parsed[0]), 4)
        if 0.0 < price < 1.0:
            levels.append((price, parsed[1]))
    levels.sort()
    return levels


def plan_book_walk(
    ask_levels: list[tuple[float, int]],
    *,
    model_prob: float,
    signal_ask: float,
    target_contracts: int,
    max_walk_cents: int,
    edge_floor: float,
) -> BookWalkPlan | None:
    """Walk the YES ask ladder up to ``target_contracts``.

    A level is eligible only if it is within ``max_walk_cents`` of the
    signal-time ask AND its own marginal edge (g - price - taker fee) still
    clears ``edge_floor``, so every contract bought is one the strategy would
    have bought at the ask. The limit is the cheapest level that covers the
    target (never deeper than needed). ``None`` = no eligible depth.
    """
    ceiling = signal_ask + max_walk_cents / 100.0 + 1e-9
    filled = 0
    cost = 0.0
    limit: float | None = None
    for price, qty in ask_levels:
        if price > ceiling:
            break
        if model_prob - price - fee_per_contract(price, is_maker=False) < edge_floor:
            break
        take = min(qty, target_contracts - filled)
        if take <= 0:
            break
        filled += take
        cost += take * price
        limit = price
        if filled >= target_contracts:
            break
    if limit is None or filled <= 0:
        return None
    return BookWalkPlan(limit_price=limit, vwap=cost / filled, contracts=filled)


def _lead_days(ticker: str, scan_today: date) -> int | None:
    fd = parse_ticker_date(ticker)
    return (fd - scan_today).days if fd else None


class FavoriteBiasStrategy(Strategy):
    # Kalshi ticker quirk: KXHIGH* has no extra T (NY, CHI, MIA, LAX, DEN, AUS,
    # PHIL) but SEA/BOS use KXHIGHT* with an extra T. KXLOWTSEA (launched
    # 2026-06) is collection-only and not yet in this tuple.
    SERIES = (
        "KXHIGHNY",
        "KXHIGHCHI",
        "KXHIGHMIA",
        "KXHIGHLAX",
        "KXHIGHDEN",
        "KXHIGHAUS",
        "KXHIGHTSEA",
        "KXHIGHTBOS",
        "KXHIGHPHIL",
        "KXLOWTNYC",
        "KXLOWTCHI",
        "KXLOWTMIA",
        "KXLOWTLAX",
        "KXLOWTDEN",
        "KXLOWTAUS",
        "KXLOWTBOS",
        "KXLOWTPHIL",
    )

    def __init__(
        self,
        config: dict[str, Any],
        kalshi_client: Any,
        bankroll: float = 10_000.0,
        trades_log_path: str | None = None,
    ) -> None:
        super().__init__(name="favorite_bias", config=config, bankroll=bankroll)
        self._kalshi = kalshi_client
        self._band_low = float(config.get("band_low", 0.40))
        self._band_high = float(config.get("band_high", 0.75))
        self._min_vol = int(config.get("min_volume_fp", 200))
        self._edge_floor = float(config.get("edge_floor", 0.02))
        _ms = config.get("max_spread")
        self._max_spread: float | None = float(_ms) if _ms is not None else None
        self._lead = int(config.get("lead", 1))
        self._strike_types = set(config.get("strike_types", ["greater", "less"]))
        self._kelly_fraction = float(config.get("kelly_fraction", 0.05))
        self._max_pct = float(config.get("max_pct", 0.05))
        self._max_contracts = int(config.get("max_contracts_per_trade", 25))
        self._max_book_walk_cents = int(config.get("max_book_walk_cents", 0) or 0)
        audit_path = config.get("audit_path")
        self._audit_path = (
            Path(audit_path)
            if audit_path
            else Path(Settings().state_dir) / "favorite_bias_audit.jsonl"
        )
        self._taker = bool(config.get("execution_taker", True))
        _rt = config.get("resting_ttl_seconds")
        self._resting_ttl_seconds: int | None = int(_rt) if _rt is not None else None
        iso_path = config.get("isotonic_path")
        self._cal = IsotonicCalibrator(Path(iso_path) if iso_path else None)
        if not self._cal.is_loaded:
            logger.error(
                "favorite_bias: g(mid) artifact in fallback (%s) — strategy will "
                "produce no edge; disable until fixed.",
                self._cal.fallback_reason,
            )
        # Strike-climatology instrumentation (spec 2026-07-03). Fail-open: a
        # disabled table -> null features, guard never fires, trading unchanged.
        self._clim = FbClimatologyTable.load(config.get("climatology_path"))
        if not self._clim.is_loaded:
            logger.warning(
                "favorite_bias: climatology artifact disabled (%s) — clim features "
                "will be null; trading is UNAFFECTED.",
                self._clim.fallback_reason,
            )
        _mz = config.get("max_strike_climatology_z")
        self._max_strike_clim_z: float | None = float(_mz) if _mz is not None else None
        self._clim_min_n = int(config.get("clim_min_n", 15))
        self._trades_log_path = (
            Path(trades_log_path)
            if trades_log_path
            else Path(Settings().state_dir) / "trades.jsonl"
        )
        max_per_day = config.get("max_bets_per_day_per_event", 1)
        # audit_writer=None: cap blocks are deliberately not logged for this strategy.
        self._per_day_cap = PerDayEventCap(max_per_day=max_per_day, audit_writer=None)
        self._traded_tickers: set[str] = set()
        self._cities: set[str] = set(config.get("cities", []))
        self._excluded_city_vars: set[str] = set(config.get("excluded_city_vars", []))
        if self._excluded_city_vars:
            # Excluded cells otherwise vanish silently from signal generation.
            logger.info(
                "favorite_bias: (city,var) exclusion ACTIVE — %s will not be traded",
                sorted(self._excluded_city_vars),
            )

    def mark_traded(self, ticker: str) -> None:
        """Record a ticker as traded so it's skipped in future scans."""
        self._traded_tickers.add(ticker)

    def unmark_traded(self, ticker: str) -> None:
        """Release a ticker so it can be considered again this session.

        Called by the orchestrator when a previously-placed order is cancelled
        with zero fills (TTL expiry, server-side cancellation). Idempotent.
        """
        self._traded_tickers.discard(ticker)

    # ---- pure helpers (unit-tested) ----
    def _g(self, mid: float) -> float:
        return self._cal.calibrate(mid)

    def _passes_band(self, mid: float) -> bool:
        return self._band_low <= mid < self._band_high

    @staticmethod
    def _strike_for(fm: FBMarket) -> float | None:
        """Strike by type: greater->floor_strike, less->cap_strike (never both)."""
        if fm.strike_type == "greater":
            return fm.floor_strike
        if fm.strike_type == "less":
            return fm.cap_strike
        return None  # between is not traded here

    def _clim_features(self, fm: FBMarket) -> dict[str, Any]:
        """z_strike / q_clim / clim_source / clim_n vs (city,var,month). Never raises."""
        out: dict[str, Any] = {
            "z_strike": None,
            "q_clim": None,
            "clim_source": None,
            "clim_n": None,
            "climatology_artifact_hash": self._clim.artifact_hash,
        }
        strike = self._strike_for(fm)
        if strike is None:
            return out
        entry = self._clim.lookup(fm.city, fm.var, fm.forecast_date.month)
        if entry is None:
            return out
        out["clim_source"] = entry.source
        out["clim_n"] = entry.n
        out["z_strike"] = strike_z(strike, entry.mu, entry.sigma)
        out["q_clim"] = clim_yes_prob(strike, fm.strike_type, entry.mu, entry.sigma)
        return out

    def _build_signal(self, fm: FBMarket | None, *, scan_today: date) -> TradeSignal | None:
        if fm is None:
            return None
        if self._cities and fm.city not in self._cities:
            return None
        if f"{fm.city}_{fm.var}" in self._excluded_city_vars:
            return None
        if fm.strike_type not in self._strike_types:
            return None
        if not self._passes_band(fm.mid):
            return None
        if fm.volume_fp < self._min_vol:
            return None
        if (fm.forecast_date - scan_today).days != self._lead:
            return None
        p = self._g(fm.mid)
        fee = fee_per_contract(fm.yes_ask, is_maker=False)
        edge = p - fm.yes_ask - fee
        clim = self._clim_features(fm)
        decision = "buy" if edge >= self._edge_floor else "skip"
        # Epsilon so a quote spread exactly at the cap passes despite float subtraction.
        if (
            decision == "buy"
            and self._max_spread is not None
            and fm.yes_ask - fm.yes_bid > self._max_spread + 1e-9
        ):
            decision = "skip_spread"
        # Tail-strike suppression guard (default-off). Variance insurance, no
        # demonstrated EV. Fires ONLY on a month-source cell with enough obs and
        # a strike deep in the climatological tail; fails open on every other case.
        if (
            decision == "buy"
            and self._max_strike_clim_z is not None
            and clim["clim_source"] == "month"
            and clim["clim_n"] is not None
            and clim["clim_n"] >= self._clim_min_n
            and clim["z_strike"] is not None
            and abs(clim["z_strike"]) >= self._max_strike_clim_z
        ):
            decision = "skip_clim_tail"
        self._audit(fm, p, edge, decision, scan_today, clim)
        if decision != "buy":
            return None
        sizing = fractional_kelly(
            model_prob=p,
            market_price=fm.yes_ask,
            kelly_fraction=self._kelly_fraction,
            max_pct=self._max_pct,
            bankroll=self._bankroll,
            fee_per_contract=fee,
        )
        if sizing.side != Side.YES:
            return None
        contracts = min(
            self._max_contracts,
            dollars_to_contracts(sizing.dollars, fm.yes_ask, Side.YES),
        )
        if contracts < 1:
            return None
        return TradeSignal(
            market_id=fm.ticker,
            market_ticker=fm.ticker,
            model_probability=p,
            market_price=fm.mid,
            confidence=p,
            strategy_name=self._name,
            side=Side.YES,
            suggested_dollars=sizing.dollars,
            suggested_contracts=contracts,
            effective_edge=edge,  # REQUIRED: orchestrator sort key
            mid_at_signal=fm.mid,
            exec_price=fm.yes_ask,
            metadata={
                "city": fm.city,
                "var": fm.var,
                "strike_type": fm.strike_type,
                "floor_strike": fm.floor_strike,
                "cap_strike": fm.cap_strike,
                "forecast_date": fm.forecast_date.isoformat(),
                "market_type": fm.var,  # per-day cap event key parity
                "lead": self._lead,
                "mid": fm.mid,
                "yes_bid": fm.yes_bid,
                "yes_ask": fm.yes_ask,
                "fill_price": fm.yes_ask,  # taker fill — load-bearing
                "execution_taker": self._taker,
                "resting_ttl_seconds": self._resting_ttl_seconds,
                "g_of_mid": p,
                "edge": edge,
                "volume_fp": fm.volume_fp,
                "kelly_raw": sizing.kelly_raw,
                "kelly_capped": sizing.capped,
                "map_version": self._cal.artifact_hash,
                "z_strike": clim["z_strike"],
                "q_clim": clim["q_clim"],
                "clim_source": clim["clim_source"],
                "clim_n": clim["clim_n"],
                "climatology_artifact_hash": clim["climatology_artifact_hash"],
                "close_time": fm.close_time,
                "event_candidates_count": 1,
                "event_dropped_edges": [],
                "event_dropped_tickers": [],
            },
        )

    async def _apply_book_walk(self, sig: TradeSignal) -> TradeSignal:
        """Price the order through the visible ask ladder (bounded). Fail-open.

        Without this the limit sits at the frozen top-of-book ask, so an order
        larger than that one level fills partially and the rest rarely fills in
        the resting TTL (live: orders of 31+ contracts filled 30% vs 91% for
        6-12). Any failure, or no eligible depth, returns the signal unchanged.
        """
        signal_ask = float(sig.metadata["yes_ask"])
        try:
            levels = yes_ask_levels(await self._kalshi.get_orderbook(sig.market_ticker))
        except Exception:
            logger.warning(
                "favorite_bias: orderbook fetch failed for %s — limit stays at signal ask",
                sig.market_ticker,
                exc_info=True,
            )
            return sig
        p = sig.model_probability
        target = sig.suggested_contracts
        # Re-run Kelly at the walked VWAP (worse than the ask) and re-plan with
        # the smaller target so the walk goes no deeper than the final size needs.
        for _ in range(3):
            plan = plan_book_walk(
                levels,
                model_prob=p,
                signal_ask=signal_ask,
                target_contracts=target,
                max_walk_cents=self._max_book_walk_cents,
                edge_floor=self._edge_floor,
            )
            if plan is None:
                return sig
            sizing = fractional_kelly(
                model_prob=p,
                market_price=plan.vwap,
                kelly_fraction=self._kelly_fraction,
                max_pct=self._max_pct,
                bankroll=self._bankroll,
                fee_per_contract=fee_per_contract(plan.vwap, is_maker=False),
            )
            if sizing.side != Side.YES:
                return sig
            resized = min(
                self._max_contracts, dollars_to_contracts(sizing.dollars, plan.vwap, Side.YES)
            )
            if resized < 1:
                return sig
            if resized >= target:
                break
            target = resized
        else:
            return sig
        return replace(
            sig,
            suggested_dollars=sizing.dollars,
            suggested_contracts=target,
            metadata={
                **sig.metadata,
                "fill_price": plan.vwap,
                "limit_price": plan.limit_price,
                "book_walk_cents": int(round((plan.limit_price - signal_ask) * 100)),
                "book_walk_visible_contracts": plan.contracts,
                "kelly_raw": sizing.kelly_raw,
                "kelly_capped": sizing.capped,
            },
        )

    def _audit(
        self,
        fm: FBMarket,
        p: float,
        edge: float,
        decision: str,
        scan_today: date,
        clim: dict[str, Any],
    ) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "market_ticker": fm.ticker,
            "city": fm.city,
            "var": fm.var,
            "strike_type": fm.strike_type,
            "floor_strike": fm.floor_strike,
            "cap_strike": fm.cap_strike,
            "lead": self._lead,
            "mid": fm.mid,
            "yes_bid": fm.yes_bid,
            "yes_ask": fm.yes_ask,
            "g_of_mid": p,
            "edge": edge,
            "decision": decision,
            "volume_fp": fm.volume_fp,
            "map_version": self._cal.artifact_hash,
            **clim,
        }
        self._audit_write(row)

    def _audit_write(self, row: dict[str, Any]) -> None:
        try:
            _append_jsonl_atomic(self._audit_path, row)
        except Exception:  # audit must never crash signal gen
            logger.debug("favorite_bias audit write failed", exc_info=True)

    # ---- Strategy ABC ----
    async def scan(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for series in self.SERIES:
            try:
                markets, _trunc = await self._kalshi.iter_all_markets(
                    status="open", series_ticker=series
                )
                out.extend(markets)
            except Exception:
                logger.warning("favorite_bias scan failed for %s", series, exc_info=True)
        return out

    async def evaluate(self, market: dict[str, Any]) -> float | None:
        fm = _parse_market(market)
        return self._g(fm.mid) if fm else None

    async def generate_signals(self) -> list[TradeSignal]:
        if not self._enabled or not self._cal.is_loaded:
            return []
        scan_today = datetime.now(UTC).date()
        markets = await self.scan()
        signals: list[TradeSignal] = []
        seen: set[str] = set()
        n_in_band = 0  # parsed greater/less markets whose mid sits in the favorite band
        for m in markets:
            t = m.get("ticker", "")
            if t in seen or t in self._traded_tickers:
                continue
            seen.add(t)
            fm = _parse_market(m)
            if (
                fm is not None
                and fm.strike_type in self._strike_types
                and self._passes_band(fm.mid)
            ):
                n_in_band += 1
            sig = self._build_signal(fm, scan_today=scan_today)
            if sig is not None:
                signals.append(sig)
        utc_today = scan_today.isoformat()
        self._per_day_cap.maybe_reset(
            utc_today,
            hydrate_fn=lambda d: hydrate_per_day_state_from_trades_jsonl(
                utc_today=d,
                trades_path=self._trades_log_path,
                relevant_strategy="favorite_bias",
            ),
        )
        scan_ts_iso = datetime.now(UTC).isoformat()
        signals, n_blocked = self._per_day_cap.filter_and_select(
            signals, scan_id="favorite_bias", scan_ts=scan_ts_iso
        )
        if self._max_book_walk_cents > 0:
            signals = [await self._apply_book_walk(sig) for sig in signals]
        # Distinguishes a healthy-but-idle scan (0 in-band) from a dead strategy.
        logger.info(
            "favorite_bias: generated %d signal(s) from %d market(s) "
            "(%d in-band, %d suppressed by per-day cap)",
            len(signals),
            len(markets),
            n_in_band,
            n_blocked,
        )
        # The cap commit happens in the orchestrator after execute_batch, once
        # the order_id is known.
        return signals
