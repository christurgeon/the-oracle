"""
Abstract strategy interface for The Oracle.

Every trading strategy (weather, AI/news, arbitrage, etc.) implements this
base class so the orchestrator can treat them uniformly.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from python.common import Side

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeSignal:
    """
    A trading opportunity identified by a strategy.

    The execution engine consumes these to place (or shadow-log) orders.

    Pricing convention:
      - ``market_price`` is the YES-space midpoint at signal time (analytics use this).
      - ``mid_at_signal`` is the same, kept as a distinct field so strategies can
        populate it explicitly without relying on the historical overloading of
        ``market_price``. Always YES-space midpoint.
      - ``exec_price`` is the side-denominated executable fill price
        (YES ask for YES-side, NO ask — i.e. 1 - YES bid — for NO-side).
        Attribution decomposes ``(exec_price - side_mid)`` as slippage cost.
    """

    market_id: str
    market_ticker: str
    model_probability: float
    market_price: float
    confidence: float
    strategy_name: str
    side: Side
    suggested_dollars: float
    suggested_contracts: int
    effective_edge: float = 0.0
    mid_at_signal: float = float("nan")
    exec_price: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def position_key(self) -> str:
        """Canonical identifier used for position and exposure tracking."""
        return self.market_id or self.market_ticker

    @property
    def edge(self) -> float:
        """Absolute edge: |model_probability - market_price|."""
        return abs(self.model_probability - self.market_price)


class Strategy(abc.ABC):
    """
    Abstract base class for all trading strategies.

    Lifecycle:
        1. Orchestrator calls ``scan()`` to discover relevant markets.
        2. For each candidate, ``evaluate(market)`` computes a model probability.
        3. ``generate_signals()`` runs the full pipeline and returns actionable
           TradeSignal objects (already filtered by edge threshold).
    """

    def __init__(self, name: str, config: dict[str, Any], *, bankroll: float = 10_000.0) -> None:
        self._name = name
        self._config = config
        self._enabled: bool = config.get("enabled", True)
        self._bankroll = bankroll

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def configured_to_trade(self) -> bool:
        """True when config permits this strategy to place live orders.

        Reads STATIC CONFIG ONLY. Deliberately ignores runtime circuit-breaker
        state: a breaker trips precisely when something is going wrong, so a
        breaker-tripped strategy must keep alerting. Do NOT add a tracker check.

        Does NOT gate execution — the per-strategy shadow-drop hooks own that.
        This exists so alerting can ask one question in one place instead of
        re-deriving it per call site.
        """
        return self._enabled and not self._config.get("shadow_only", False)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def bankroll(self) -> float:
        """Current bankroll used for Kelly sizing."""
        return self._bankroll

    def update_bankroll(self, bankroll: float) -> None:
        """Update the bankroll used for Kelly sizing.

        Called by the orchestrator each tick to keep strategy sizing
        in sync with the risk manager's live bankroll.
        """
        clamped = max(0.0, bankroll)
        if clamped != self._bankroll:
            logger.debug(
                "Strategy %s bankroll updated: %.2f -> %.2f",
                self._name,
                self._bankroll,
                clamped,
            )
            self._bankroll = clamped

    async def setup(self) -> None:
        """Optional async post-construction init. Default no-op.

        Called by the orchestrator after construction and after shared
        clients (Kalshi, weather) are connected. Override for resources
        that need an awaitable init (e.g. a per-strategy NWSClient).
        """
        return None

    async def aclose(self) -> None:
        """Optional async teardown. Default no-op.

        Called by the orchestrator during shutdown, before HTTP clients
        owned by the orchestrator are closed. Override to release any
        per-strategy resources (e.g. close a per-strategy NWSClient).
        """
        return None

    @abc.abstractmethod
    async def scan(self) -> list[dict[str, Any]]:
        """
        Discover candidate markets for this strategy.

        Returns a list of market dicts (as returned by the Kalshi API or
        filtered/enriched by the strategy).
        """
        ...

    @abc.abstractmethod
    async def evaluate(self, market: dict[str, Any]) -> float | None:
        """
        Compute the model's probability of YES for a single market.

        Returns:
            A float in [0, 1] representing model probability, or None if
            the strategy cannot produce an estimate for this market.
        """
        ...

    @abc.abstractmethod
    async def generate_signals(self) -> list[TradeSignal]:
        """
        Run the full scan -> evaluate -> size pipeline.

        Returns:
            A list of TradeSignal objects that pass the strategy's internal
            edge and confidence thresholds.
        """
        ...

    def _passes_edge_threshold(self, model_prob: float, market_price: float) -> bool:
        """Check whether the absolute edge exceeds the strategy's threshold."""
        threshold: float = self._config.get("edge_threshold", 0.08)
        return abs(model_prob - market_price) >= threshold

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        return f"<Strategy {self._name} [{status}]>"
