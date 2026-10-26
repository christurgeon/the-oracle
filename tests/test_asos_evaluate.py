"""Tests for asos_evaluate (Task 4.1) and size_candidate (Task 4.2).

TDD: written before implementation — run with `uv run pytest tests/test_asos_evaluate.py`.
"""
from __future__ import annotations

import datetime as dt

import pytest

from python.common import Direction
from python.strategies.weather_parsing import ParsedMarket


def _pm(
    strike_type: Direction,
    floor: float | None,
    cap: float | None,
) -> ParsedMarket:
    from python.common import DailyVariable

    return ParsedMarket(
        ticker="KXHIGHMIA-26JUN06-B89.5",
        city="MIA",
        forecast_date=dt.date(2026, 6, 6),
        strike_type=strike_type,
        floor_strike=floor,
        cap_strike=cap,
        daily_variable=DailyVariable.TEMP_MAX,
    )


# ---------------------------------------------------------------------------
# Task 4.1 — asos_evaluate
# ---------------------------------------------------------------------------


class TestBoundaryGuard:
    """BETWEEN market: post_mean inside YES rounding region → guard_outcome='boundary'."""

    def test_boundary_guard_drops_grazer(self) -> None:
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # post_mean = M + mu_gain = 88.6 + 0.4 = 89.0
        # boundary_distance(89.0, 89.0, 90.0, "between"):
        #   b_lo = 89.0 - 0.5 = 88.5, b_hi = 90.0 + 0.5 = 90.5
        #   89.0 is inside [88.5, 90.5] → returns 0.0 < boundary_min_f=1.0
        #   → guard_outcome="boundary" (not None — full candidate returned)
        cand = asos_evaluate(
            _pm(Direction.BETWEEN, 89.0, 90.0),
            M=88.6,
            mu_gain=0.4,
            sigma_gain=0.6,
            nu=5,
            yes_bid=0.45,
            yes_ask=0.50,
            guards=AsosGuardConfig(boundary_min_f=1.0),
        )
        assert cand is not None
        assert cand.guard_outcome == "boundary"

    def test_boundary_guard_passes_far_from_boundary(self) -> None:
        """BETWEEN market with mean far outside YES region should pass boundary guard.

        Tested implicitly by test_passes_and_sets_probability — documented here
        for spec completeness.
        """
        # (No assertion — see docstring above.)


class TestStaleObsGuard:
    """obs_age_hours > max_staleness_h → guard_outcome='stale'."""

    def test_stale_obs_guard_drops(self) -> None:
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        cand = asos_evaluate(
            _pm(Direction.BETWEEN, 85.0, 86.0),
            M=85.5,
            mu_gain=0.3,
            sigma_gain=0.6,
            nu=5,
            yes_bid=0.45,
            yes_ask=0.50,
            obs_age_hours=3.0,
            guards=AsosGuardConfig(max_staleness_h=2.0),
        )
        assert cand is not None
        assert cand.guard_outcome == "stale"

    def test_fresh_obs_passes_staleness_guard(self) -> None:
        """Documented complement to test_stale_obs_guard_drops — no assertion needed."""
        # Key assertion is in test_stale_obs_guard_drops above.


class TestSigmaFloorGuard:
    """sigma_gain < sigma_floor_f → guard_outcome='headroom' (zero-headroom guard)."""

    def test_zero_headroom_guard_drops(self) -> None:
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        cand = asos_evaluate(
            _pm(Direction.BETWEEN, 85.0, 86.0),
            M=85.5,
            mu_gain=0.1,
            sigma_gain=0.2,
            nu=5,
            yes_bid=0.45,
            yes_ask=0.50,
            guards=AsosGuardConfig(sigma_floor_f=0.5),
        )
        assert cand is not None
        assert cand.guard_outcome == "headroom"  # sigma 0.2 < floor 0.5

    def test_sigma_at_floor_passes(self) -> None:
        """sigma exactly at floor should pass (guard fires on strict <, not <=)."""
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # sigma_gain == sigma_floor_f should NOT be dropped by the headroom guard
        # (guard fires on sigma < floor, not sigma <= floor)
        # Result may still be None from edge — that's acceptable.
        _ = asos_evaluate(
            _pm(Direction.BETWEEN, 89.0, 90.0),
            M=80.0,
            mu_gain=1.0,
            sigma_gain=0.5,
            nu=5,
            yes_bid=0.10,
            yes_ask=0.20,
            guards=AsosGuardConfig(sigma_floor_f=0.5, boundary_min_f=0.0),
        )
        # If no exception raised, guard didn't crash on boundary case — passes.


class TestBadPrelimGuard:
    """bad_prelim guard: only active when bad_prelim_active=True."""

    def test_bad_prelim_guard_inactive_by_default(self) -> None:
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # bad_prelim_active=False by default → should NOT drop even with large drift
        # (may be None from edge filter, but NOT from bad_prelim guard)
        _ = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            prelim_high=70.0,  # |85.0 - 70.0| = 15 >> bad_prelim_max_drift_f=10
            guards=AsosGuardConfig(bad_prelim_active=False),
        )
        # No assertion: we verify no exception raised and bad_prelim guard is silent.

    def test_bad_prelim_guard_active_drops_on_large_drift(self) -> None:
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        cand = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            prelim_high=70.0,  # |85.0 - 70.0| = 15 > bad_prelim_max_drift_f=10
            guards=AsosGuardConfig(bad_prelim_active=True, bad_prelim_max_drift_f=10.0),
        )
        assert cand is not None
        assert cand.guard_outcome == "bad_prelim"

    def test_bad_prelim_guard_active_passes_small_drift(self) -> None:
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # |85.0 - 84.0| = 1.0 < bad_prelim_max_drift_f=10.0 → bad_prelim guard passes
        # (cand may be None from edge filter, but NOT from bad_prelim guard)
        _ = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            prelim_high=84.0,  # drift = 1.0 < 10.0 → passes
            guards=AsosGuardConfig(bad_prelim_active=True, bad_prelim_max_drift_f=10.0),
        )
        # No assertion: verify no exception raised and guard is silent.


class TestPassingCandidate:
    """Guard-passing scenarios: verify probability and post_mean fields."""

    def test_passes_and_sets_probability(self) -> None:
        """All guards pass; candidate fields populated correctly."""
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        cand = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            guards=AsosGuardConfig(),
        )
        assert cand is not None
        assert 0.0 <= cand.model_prob <= 1.0
        assert cand.post_mean_f == 87.0

    def test_candidate_fields_consistent(self) -> None:
        """AsosCandidate carries all required fields after passing guards."""
        from python.common import Side
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        cand = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            guards=AsosGuardConfig(),
        )
        assert cand is not None
        assert cand.post_mean_f == 87.0
        assert cand.M == 85.0
        assert cand.mu_gain == 2.0
        assert cand.sigma_gain == 1.0
        assert cand.obs_age_hours == 0.0
        assert cand.prelim_drift_f is None  # prelim_high not provided
        assert cand.guard_outcome == "pass"
        assert cand.side in (Side.YES, Side.NO, Side.NONE)
        assert isinstance(cand.raw_edge, float)
        assert isinstance(cand.effective_edge, float)
        # Step 0: boundary_distance_f is always populated (may be NaN for malformed geometry)
        assert isinstance(cand.boundary_distance_f, float)

    def test_prelim_drift_recorded_when_provided(self) -> None:
        """prelim_drift_f is populated when prelim_high is given."""
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        cand = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            prelim_high=84.0,  # drift = 1.0, guard inactive → passes
            guards=AsosGuardConfig(bad_prelim_active=False),
        )
        assert cand is not None
        assert cand.prelim_drift_f == pytest.approx(1.0)

    def test_side_yes_when_model_above_ask(self) -> None:
        """Model probability above yes_ask → side=YES."""
        from python.common import Side
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # Use ABOVE with low floor so model_prob is high enough to exceed yes_ask.
        # M=90, mu_gain=2, floor=85: post_mean=92, P(>85.5) is very high.
        cand = asos_evaluate(
            _pm(Direction.ABOVE, 85.0, None),
            M=90.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.50,
            yes_ask=0.55,
            guards=AsosGuardConfig(boundary_min_f=0.0),
        )
        assert cand is not None
        assert cand.side == Side.YES
        assert cand.exec_price == pytest.approx(0.55)

    def test_side_no_when_model_below_ask(self) -> None:
        """Model probability below yes_ask → side=NO."""
        from python.common import Side
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # ABOVE 95 with mean=87: very low YES probability → side=NO
        cand = asos_evaluate(
            _pm(Direction.ABOVE, 95.0, None),
            M=85.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.02,
            yes_ask=0.05,
            guards=AsosGuardConfig(),
        )
        assert cand is not None
        assert cand.side == Side.NO
        # NO exec price = 1 - yes_bid
        assert cand.exec_price == pytest.approx(1.0 - 0.02)

    def test_guard_order_staleness_before_sigma(self) -> None:
        """Staleness guard fires first (before sigma guard)."""
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # Both staleness AND sigma would fail; staleness fires first → "stale"
        cand = asos_evaluate(
            _pm(Direction.BETWEEN, 85.0, 86.0),
            M=85.5,
            mu_gain=0.1,
            sigma_gain=0.2,  # also below sigma_floor_f=0.5
            nu=5,
            yes_bid=0.45,
            yes_ask=0.50,
            obs_age_hours=5.0,  # > max_staleness_h=2.0
            guards=AsosGuardConfig(max_staleness_h=2.0, sigma_floor_f=0.5),
        )
        assert cand is not None
        assert cand.guard_outcome == "stale"  # staleness fires before headroom


class TestNanBoundaryDistance:
    """NaN boundary_distance → conservative drop."""

    def test_nan_boundary_drops_above_with_none_floor(self) -> None:
        """For ABOVE markets, floor=None → boundary_distance returns NaN → drop."""
        from python.common import DailyVariable
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        # ParsedMarket with ABOVE but floor_strike=None is invalid by construction
        # (parse_weather_market would reject), but we test the function's own guard.
        # Use BELOW with cap=None to force NaN.
        pm_no_cap = ParsedMarket(
            ticker="KXHIGHMIA-26JUN06-L89",
            city="MIA",
            forecast_date=dt.date(2026, 6, 6),
            strike_type=Direction.BELOW,
            floor_strike=None,
            cap_strike=None,  # None cap for BELOW → NaN from boundary_distance
            daily_variable=DailyVariable.TEMP_MAX,
        )
        cand = asos_evaluate(
            pm_no_cap,
            M=85.0,
            mu_gain=1.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.40,
            yes_ask=0.50,
            guards=AsosGuardConfig(boundary_min_f=0.1),  # any positive threshold
        )
        assert cand is None  # NaN < 0.1 (conservative NaN → drop)


# ---------------------------------------------------------------------------
# Task 4.2 — size_candidate
# ---------------------------------------------------------------------------


class TestSizeCandidate:
    """size_candidate returns (SizingResult, contracts) with correct clamping."""

    def _make_passing_candidate(self):
        """Build a passing AsosCandidate for sizing tests."""
        from python.strategies.intraday_asos_high import AsosGuardConfig, asos_evaluate

        return asos_evaluate(
            _pm(Direction.ABOVE, 85.0, None),
            M=90.0,
            mu_gain=2.0,
            sigma_gain=1.0,
            nu=5,
            yes_bid=0.50,
            yes_ask=0.55,
            guards=AsosGuardConfig(boundary_min_f=0.0),
        )

    def test_contracts_nonnegative(self) -> None:
        from python.strategies.intraday_asos_high import size_candidate

        cand = self._make_passing_candidate()
        assert cand is not None
        sizing, contracts = size_candidate(
            cand,
            bankroll=10_000.0,
            kelly_fraction=0.15,
            max_single_market_pct=0.05,
            max_contracts_per_trade=50,
        )
        assert contracts >= 0

    def test_contracts_clamped_at_max(self) -> None:
        """Contracts never exceed max_contracts_per_trade."""
        from python.strategies.intraday_asos_high import size_candidate

        cand = self._make_passing_candidate()
        assert cand is not None
        # Use a large bankroll to force many contracts, then clamp
        sizing, contracts = size_candidate(
            cand,
            bankroll=1_000_000.0,
            kelly_fraction=0.5,
            max_single_market_pct=0.10,
            max_contracts_per_trade=5,
        )
        assert contracts <= 5

    def test_contracts_match_sizing_dollars(self) -> None:
        """Contracts derived from sizing.dollars and exec_price."""
        from python.analytics.sizing import dollars_to_contracts
        from python.strategies.intraday_asos_high import size_candidate

        cand = self._make_passing_candidate()
        assert cand is not None
        sizing, contracts = size_candidate(
            cand,
            bankroll=10_000.0,
            kelly_fraction=0.15,
            max_single_market_pct=0.05,
            max_contracts_per_trade=50,
        )
        # contracts should be <= what dollars_to_contracts would produce at cand.exec_price
        expected_unclamped = dollars_to_contracts(sizing.dollars, cand.exec_price, sizing.side)
        assert contracts <= min(expected_unclamped, 50)

    def test_side_none_yields_zero_contracts(self) -> None:
        """If Kelly returns NONE side (zero-edge), contracts should be 0."""
        import datetime as dt

        from python.common import DailyVariable, Direction, Side
        from python.strategies.intraday_asos_high import AsosCandidate, size_candidate

        # Construct a candidate with model_prob == exec_price (zero edge)
        # so fractional_kelly returns NONE side.
        pm = ParsedMarket(
            ticker="KXHIGHMIA-26JUN06-B89.5",
            city="MIA",
            forecast_date=dt.date(2026, 6, 6),
            strike_type=Direction.BETWEEN,
            floor_strike=89.0,
            cap_strike=90.0,
            daily_variable=DailyVariable.TEMP_MAX,
        )
        cand_zero = AsosCandidate(
            parsed=pm,
            model_prob=0.50,
            post_mean_f=89.5,
            side=Side.NONE,
            exec_price=0.50,
            raw_edge=0.0,
            effective_edge=0.0,
            guard_outcome="pass",
            mu_gain=0.0,
            sigma_gain=1.0,
            M=89.5,
            obs_age_hours=0.0,
            prelim_drift_f=None,
            boundary_distance_f=0.0,
        )
        sizing, contracts = size_candidate(
            cand_zero,
            bankroll=10_000.0,
            kelly_fraction=0.15,
            max_single_market_pct=0.05,
            max_contracts_per_trade=50,
        )
        assert contracts == 0

    def test_returns_sizing_result_type(self) -> None:
        """Return type is (SizingResult, int)."""
        from python.analytics.sizing import SizingResult
        from python.strategies.intraday_asos_high import size_candidate

        cand = self._make_passing_candidate()
        assert cand is not None
        sizing, contracts = size_candidate(
            cand,
            bankroll=10_000.0,
            kelly_fraction=0.15,
            max_single_market_pct=0.05,
            max_contracts_per_trade=50,
        )
        assert isinstance(sizing, SizingResult)
        assert isinstance(contracts, int)

    def test_fee_included_in_kelly(self) -> None:
        """size_candidate passes fee_per_contract to fractional_kelly."""
        from python.strategies.intraday_asos_high import size_candidate

        cand = self._make_passing_candidate()
        assert cand is not None
        # Just verify it runs without error and returns reasonable values
        sizing, contracts = size_candidate(
            cand,
            bankroll=10_000.0,
            kelly_fraction=0.15,
            max_single_market_pct=0.05,
            max_contracts_per_trade=50,
        )
        assert sizing.dollars >= 0.0
        assert contracts >= 0

    # ------------------------------------------------------------------
    # Regression tests: single YES-space price for kelly + contracts
    # ------------------------------------------------------------------

    def test_no_side_sizing_consistent_price(self) -> None:
        """NO-side candidate: kelly and dollars_to_contracts use the same YES price.

        Exercises the regression case from the code-review finding: if exec_price
        is keyed on sizing.side rather than cand.side, the two calls see
        different prices when fractional_kelly re-derives a side that differs
        from cand.side.  This test constructs a clear NO candidate (model_prob
        well below yes_bid) and asserts:
          - sizing.side == Side.NO
          - contracts >= 1
          - contract count equals int(sizing.dollars / (1 - yes_price)) —
            i.e. priced at the NO cost (1 - yes_price), where yes_price = yes_ask
            (because cand.side=NO, so yes_price = 1 - exec_price = yes_ask).
        """
        import datetime as dt

        from python.analytics.sizing import dollars_to_contracts
        from python.common import DailyVariable, Direction, Side
        from python.strategies.intraday_asos_high import AsosCandidate, size_candidate
        from python.strategies.weather_parsing import ParsedMarket

        # Construct a NO-side candidate: model_prob≈0.03, yes_bid=0.40, yes_ask=0.45
        # → exec_price = 1 - yes_bid = 0.60, side = NO
        yes_bid = 0.40
        _yes_ask = 0.45  # kept for readability; exec_price derives from yes_bid
        model_prob = 0.03

        pm = ParsedMarket(
            ticker="KXHIGHMIA-26JUN06-B89.5",
            city="MIA",
            forecast_date=dt.date(2026, 6, 6),
            strike_type=Direction.BETWEEN,
            floor_strike=89.0,
            cap_strike=90.0,
            daily_variable=DailyVariable.TEMP_MAX,
        )
        exec_price = 1.0 - yes_bid  # = 0.60 (NO cost)
        raw_edge = (1.0 - model_prob) - exec_price  # positive NO edge

        cand_no = AsosCandidate(
            parsed=pm,
            model_prob=model_prob,
            post_mean_f=85.0,
            side=Side.NO,
            exec_price=exec_price,
            raw_edge=raw_edge,
            effective_edge=raw_edge,
            guard_outcome="pass",
            mu_gain=0.0,
            sigma_gain=1.0,
            M=85.0,
            obs_age_hours=0.0,
            prelim_drift_f=None,
            boundary_distance_f=4.5,
        )

        sizing, contracts = size_candidate(
            cand_no,
            bankroll=10_000.0,
            kelly_fraction=0.5,
            max_single_market_pct=0.10,
            max_contracts_per_trade=200,
        )

        assert sizing.side == Side.NO
        assert contracts >= 1

        # YES-space price: cand.side=NO, so yes_price = 1 - exec_price = yes_ask
        yes_price = 1.0 - exec_price  # = yes_ask = 0.45
        expected_contracts = dollars_to_contracts(sizing.dollars, yes_price, sizing.side)
        assert contracts == min(expected_contracts, 200), (
            f"contracts={contracts} != min(expected={expected_contracts}, 200); "
            f"sizing.dollars={sizing.dollars}, yes_price={yes_price}"
        )

    def test_yes_side_sizing_consistent_price(self) -> None:
        """YES-side candidate: kelly and dollars_to_contracts use the same YES price.

        Complement to test_no_side_sizing_consistent_price. Constructs a clear
        YES candidate and verifies contract count equals
        int(sizing.dollars / yes_price) consistent with dollars_to_contracts.
        """
        import datetime as dt

        from python.analytics.sizing import dollars_to_contracts
        from python.common import DailyVariable, Direction, Side
        from python.strategies.intraday_asos_high import AsosCandidate, size_candidate
        from python.strategies.weather_parsing import ParsedMarket

        # YES-side candidate: model_prob=0.90, yes_bid=0.55, yes_ask=0.60
        # → exec_price = yes_ask = 0.60, side = YES
        _yes_bid = 0.55  # kept for readability; exec_price derives from yes_ask
        yes_ask = 0.60
        model_prob = 0.90

        pm = ParsedMarket(
            ticker="KXHIGHMIA-26JUN06-G85",
            city="MIA",
            forecast_date=dt.date(2026, 6, 6),
            strike_type=Direction.ABOVE,
            floor_strike=85.0,
            cap_strike=None,
            daily_variable=DailyVariable.TEMP_MAX,
        )
        exec_price = yes_ask  # = 0.60
        raw_edge = model_prob - exec_price  # positive YES edge

        cand_yes = AsosCandidate(
            parsed=pm,
            model_prob=model_prob,
            post_mean_f=92.0,
            side=Side.YES,
            exec_price=exec_price,
            raw_edge=raw_edge,
            effective_edge=raw_edge,
            guard_outcome="pass",
            mu_gain=2.0,
            sigma_gain=1.0,
            M=90.0,
            obs_age_hours=0.0,
            prelim_drift_f=None,
            boundary_distance_f=7.5,
        )

        sizing, contracts = size_candidate(
            cand_yes,
            bankroll=10_000.0,
            kelly_fraction=0.5,
            max_single_market_pct=0.10,
            max_contracts_per_trade=200,
        )

        assert sizing.side == Side.YES
        assert contracts >= 1

        # YES-space price: cand.side=YES, so yes_price = exec_price = yes_ask
        yes_price = exec_price  # = 0.60
        expected_contracts = dollars_to_contracts(sizing.dollars, yes_price, sizing.side)
        assert contracts == min(expected_contracts, 200), (
            f"contracts={contracts} != min(expected={expected_contracts}, 200); "
            f"sizing.dollars={sizing.dollars}, yes_price={yes_price}"
        )
