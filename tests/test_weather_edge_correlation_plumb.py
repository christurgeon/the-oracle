"""Tests for the F4.1 same-day Kelly haircut plumbing in WeatherEdgeStrategy.

Invariants pinned here:
  1. ``_same_day_haircut_args`` returns ``(0, None)`` (haircut disabled) when
     - no count provider is plumbed in
     - ``correlation_rho`` is None or 0.0
     - the queried forecast_date is None
     - the provider raises (defensive fail-open)
  2. When both are plumbed in correctly, ``_same_day_haircut_args`` returns
     the live count and the configured rho.
  3. The strategy's ``_evaluate_market`` shim forwards
     ``n_existing_same_day_bets`` / ``correlation_rho`` into
     ``evaluate_market_pure`` (and thus into ``fractional_kelly``) on every
     candidate.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from python.common import DailyVariable, Direction, ExecutionMode
from python.settings import Settings
from python.strategies.weather_edge import (
    MarketPriceInfo,
    ParsedMarket,
    ScanCandidate,
    WeatherEdgeStrategy,
)


def _tomorrow() -> date:
    return date.today() + timedelta(days=1)


def _bare_config(rho: float | None = 0.0) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "edge_threshold": 0.07,
        "max_edge": 0.30,
        "cities": ["NYC"],
        "ensemble_models": ["gfs_seamless"],
        "ensemble_weights": {"gfs_seamless": 1.0},
        "calibration_type": "none",
        "candidate_audit_log": False,
        "eval_audit_log": False,
        "scan_funnel_log": False,
        "orderbook_depth_check": False,
    }
    if rho is not None:
        cfg["same_day_correlation_rho"] = rho
    return cfg


def _make_settings(mode: ExecutionMode = ExecutionMode.LIVE) -> Settings:
    s = Settings()
    s.execution.mode = mode.value  # type: ignore[assignment]
    return s


def _make_strategy(
    *,
    rho: float | None,
    same_day_count_provider: Any = None,
) -> WeatherEdgeStrategy:
    return WeatherEdgeStrategy(
        config=_bare_config(rho=rho),
        kalshi_client=AsyncMock(),
        weather_client=AsyncMock(),
        settings=_make_settings(),
        same_day_count_provider=same_day_count_provider,
    )


# ── _same_day_haircut_args unit ───────────────────────────────────────────


class TestSameDayHaircutArgs:
    def test_returns_zero_none_when_no_provider(self) -> None:
        strat = _make_strategy(rho=0.2, same_day_count_provider=None)
        assert strat._same_day_haircut_args(_tomorrow()) == (0, None)

    def test_returns_zero_none_when_rho_zero(self) -> None:
        strat = _make_strategy(rho=0.0, same_day_count_provider=lambda d: 5)
        assert strat._same_day_haircut_args(_tomorrow()) == (0, None)

    def test_returns_zero_none_when_rho_none(self) -> None:
        strat = _make_strategy(rho=None, same_day_count_provider=lambda d: 5)
        assert strat._same_day_haircut_args(_tomorrow()) == (0, None)

    def test_returns_zero_none_when_forecast_date_none(self) -> None:
        strat = _make_strategy(rho=0.2, same_day_count_provider=lambda d: 5)
        assert strat._same_day_haircut_args(None) == (0, None)

    def test_returns_count_and_rho_when_both_plumbed(self) -> None:
        called_with: list[date] = []

        def provider(d: date) -> int:
            called_with.append(d)
            return 7

        strat = _make_strategy(rho=0.2, same_day_count_provider=provider)
        d = _tomorrow()
        n, rho = strat._same_day_haircut_args(d)
        assert n == 7
        assert rho == 0.2
        assert called_with == [d]

    def test_provider_exception_is_defensive_noop(self) -> None:
        def bad_provider(d: date) -> int:
            raise RuntimeError("simulated risk-manager outage")

        strat = _make_strategy(rho=0.2, same_day_count_provider=bad_provider)
        # Should not propagate; haircut becomes a no-op for this candidate.
        assert strat._same_day_haircut_args(_tomorrow()) == (0, None)

    def test_provider_returns_zero_is_passed_through(self) -> None:
        """When count is 0 (legitimately first bet of the day), the result is
        (0, rho). fractional_kelly itself collapses this to no-haircut via
        same_day_correlation_haircut(0, …) == 1.0, but the values flowing
        through must be the unmodified result."""
        strat = _make_strategy(rho=0.2, same_day_count_provider=lambda d: 0)
        n, rho = strat._same_day_haircut_args(_tomorrow())
        assert n == 0
        assert rho == 0.2


# ── _evaluate_market plumb (integration with evaluate_market_pure) ──────


class TestSizeCallPassesCountAndRho:
    """End-to-end plumb: when ``_evaluate_market`` is called with haircut
    args, those args reach the underlying ``evaluate_market_pure`` call as
    kwargs that map 1:1 to the fractional_kelly invocation inside.

    We monkey-patch ``evaluate_market_pure`` on the strategy module so we
    can inspect the kwargs it receives without needing real forecast data.
    """

    @pytest.mark.parametrize(
        "n_existing, rho",
        [(0, 0.2), (3, 0.2), (10, 0.2), (5, 0.0)],
    )
    def test_evaluate_market_forwards_haircut_kwargs(
        self, tmp_path: Path, n_existing: int, rho: float
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_eval(*args: Any, **kwargs: Any) -> Any:
            captured["n"] = kwargs.get("n_existing_same_day_bets")
            captured["rho"] = kwargs.get("correlation_rho")
            return None

        strat = _make_strategy(rho=rho, same_day_count_provider=lambda d: n_existing)

        sc = ScanCandidate(
            market={"ticker": "KXHIGHNY-99APR01-T55", "event_ticker": "KXHIGHNY-99APR01"},
            parsed=ParsedMarket(
                ticker="KXHIGHNY-99APR01-T55",
                city="NYC",
                forecast_date=_tomorrow(),
                strike_type=Direction.ABOVE,
                floor_strike=55.0,
                cap_strike=None,
                daily_variable=DailyVariable.TEMP_MAX,
            ),
            price_info=MarketPriceInfo(
                midpoint=0.5,
                yes_bid=0.45,
                yes_ask=0.55,
                spread=0.10,
                yes_bid_size=10,
                yes_ask_size=10,
                volume=100,
                open_interest=50,
            ),
        )

        with patch(
            "python.strategies.weather_edge.evaluate_market_pure",
            side_effect=fake_eval,
        ):
            # _same_day_haircut_args returns the right tuple — assemble it
            # the same way generate_signals does, then call _evaluate_market.
            n_arg, rho_arg = strat._same_day_haircut_args(sc.parsed.forecast_date)
            strat._evaluate_market(
                sc,
                forecast_cache={},
                bias_corrections={},
                raw_ensemble_stds={},
                raw_ensemble_medians={},
                cfg=None,  # not consumed by the fake
                n_existing_same_day_bets=n_arg,
                correlation_rho=rho_arg,
            )

        # rho==0 is the explicit "haircut disabled" path → ``_same_day_haircut_args``
        # returns ``(0, None)``. For any rho>0, the live count flows through —
        # including n=0, where downstream ``same_day_correlation_haircut(0, …)``
        # collapses to 1.0 (no-op at the math layer).
        if rho == 0.0:
            assert captured["n"] == 0
            assert captured["rho"] is None
        else:
            assert captured["n"] == n_existing
            assert captured["rho"] == rho

    def test_no_provider_means_no_haircut_kwargs(self, tmp_path: Path) -> None:
        """Strategy constructed WITHOUT a count provider must always send
        (n=0, rho=None) through, regardless of the config rho value."""
        captured: dict[str, Any] = {}

        def fake_eval(*args: Any, **kwargs: Any) -> Any:
            captured["n"] = kwargs.get("n_existing_same_day_bets")
            captured["rho"] = kwargs.get("correlation_rho")
            return None

        # Config has rho=0.2 but NO provider — should disable.
        strat = _make_strategy(rho=0.2, same_day_count_provider=None)

        sc = ScanCandidate(
            market={"ticker": "KXHIGHNY-99APR01-T55", "event_ticker": "KXHIGHNY-99APR01"},
            parsed=ParsedMarket(
                ticker="KXHIGHNY-99APR01-T55",
                city="NYC",
                forecast_date=_tomorrow(),
                strike_type=Direction.ABOVE,
                floor_strike=55.0,
                cap_strike=None,
                daily_variable=DailyVariable.TEMP_MAX,
            ),
            price_info=MarketPriceInfo(
                midpoint=0.5,
                yes_bid=0.45,
                yes_ask=0.55,
                spread=0.10,
                yes_bid_size=10,
                yes_ask_size=10,
                volume=100,
                open_interest=50,
            ),
        )

        with patch(
            "python.strategies.weather_edge.evaluate_market_pure",
            side_effect=fake_eval,
        ):
            n_arg, rho_arg = strat._same_day_haircut_args(sc.parsed.forecast_date)
            strat._evaluate_market(
                sc,
                forecast_cache={},
                bias_corrections={},
                raw_ensemble_stds={},
                raw_ensemble_medians={},
                cfg=None,
                n_existing_same_day_bets=n_arg,
                correlation_rho=rho_arg,
            )

        assert captured["n"] == 0
        assert captured["rho"] is None
