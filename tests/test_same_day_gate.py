"""Unit tests for SameDayGate."""

from __future__ import annotations

from datetime import date

import pytest

from python.common import DailyVariable
from python.data.nws import NWSAPIError
from python.strategies.same_day_gate import SameDayDecision, SameDayGate


@pytest.mark.asyncio
async def test_disabled_factory_returns_block_disabled_without_calling_nws():
    """SameDayGate.disabled() short-circuits before any NWS client method."""
    gate = SameDayGate.disabled()
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_DISABLED
    # Closing the disabled sentinel must not raise.
    await gate.aclose()


# ---------------------------------------------------------------------------
# Peak-passed cutoff tests
# ---------------------------------------------------------------------------

from datetime import UTC, datetime  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402


def _make_gate(*, allowed=("NYC",), clock_value=None, nws_client=None):
    nws = nws_client or AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=None)
    nws.close = AsyncMock()
    clock = (lambda: clock_value) if clock_value is not None else datetime.now
    return SameDayGate(
        nws_client=nws,
        enabled=True,
        allowed_cities=frozenset(allowed),
        clock=clock,
    ), nws


@pytest.mark.asyncio
async def test_peak_passed_high_blocks_without_nws_call():
    """16:01 NYC local with var=TEMP_MAX → BLOCK_PEAK_PASSED, no NWS call."""
    # 16:01 EDT == 20:01 UTC on 2026-05-01 (DST in effect).
    clock_utc = datetime(2026, 5, 1, 20, 1, tzinfo=UTC)
    gate, nws = _make_gate(clock_value=clock_utc)
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_PEAK_PASSED
    nws.get_daily_observation.assert_not_called()


@pytest.mark.asyncio
async def test_peak_passed_low_blocks_without_nws_call():
    """07:01 NYC local with var=TEMP_MIN → BLOCK_PEAK_PASSED, no NWS call."""
    # 07:01 EDT == 11:01 UTC on 2026-05-01.
    clock_utc = datetime(2026, 5, 1, 11, 1, tzinfo=UTC)
    gate, nws = _make_gate(clock_value=clock_utc)
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MIN)
    assert decision == SameDayDecision.BLOCK_PEAK_PASSED
    nws.get_daily_observation.assert_not_called()


@pytest.mark.asyncio
async def test_pre_peak_high_does_not_short_circuit():
    """10:00 NYC with var=TEMP_MAX should not return BLOCK_PEAK_PASSED."""
    # 10:00 EDT == 14:00 UTC on 2026-05-01.
    clock_utc = datetime(2026, 5, 1, 14, 0, tzinfo=UTC)
    gate, _ = _make_gate(clock_value=clock_utc)
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision != SameDayDecision.BLOCK_PEAK_PASSED


@pytest.mark.asyncio
async def test_dst_lax_pre_cutoff_high_pacific_time():
    """20:30 UTC on a DST-active day → LAX local 13:30 PDT (pre-16:00 cutoff)."""
    clock_utc = datetime(2026, 4, 28, 20, 30, tzinfo=UTC)
    gate, _ = _make_gate(allowed=("LAX",), clock_value=clock_utc)
    decision = await gate.check("LAX", date(2026, 4, 28), DailyVariable.TEMP_MAX)
    assert decision != SameDayDecision.BLOCK_PEAK_PASSED


@pytest.mark.asyncio
async def test_dst_nyc_post_cutoff_high_eastern_time():
    """20:30 UTC on a DST-active day → NYC local 16:30 EDT (past 16:00 cutoff)."""
    clock_utc = datetime(2026, 4, 28, 20, 30, tzinfo=UTC)
    gate, _ = _make_gate(allowed=("NYC",), clock_value=clock_utc)
    decision = await gate.check("NYC", date(2026, 4, 28), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_PEAK_PASSED


@pytest.mark.asyncio
async def test_unknown_city_raises_keyerror():
    """A city missing from CITY_TIMEZONE must raise rather than default to UTC."""
    clock_utc = datetime(2026, 5, 1, 14, 0, tzinfo=UTC)
    gate, _ = _make_gate(allowed=("UNKNOWN",), clock_value=clock_utc)
    with pytest.raises(KeyError):
        await gate.check("UNKNOWN", date(2026, 5, 1), DailyVariable.TEMP_MAX)


@pytest.mark.asyncio
async def test_local_date_after_forecast_date_is_past_cutoff():
    """If local clock has already rolled past the forecast date, cutoff is trivially passed."""
    # 06:00 UTC May 2 → 02:00 EDT May 2; forecast_date is May 1 → past.
    clock_utc = datetime(2026, 5, 2, 6, 0, tzinfo=UTC)
    gate, _ = _make_gate(clock_value=clock_utc)
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_PEAK_PASSED


# ---------------------------------------------------------------------------
# NWS fetch path tests
# ---------------------------------------------------------------------------

from python.data.nws import CliObservation  # noqa: E402


def _obs(*, high: int | None, low: int | None, preliminary: bool = False, dt=date(2026, 5, 1)):
    return CliObservation(
        city="NYC",
        cli_location="NYC",
        observation_date=dt,
        high_f=high,
        low_f=low,
        product_id="prod-1",
        issued_at="2026-05-01T20:00:00+00:00",
        is_preliminary=preliminary,
    )


def _pre_peak_clock():
    # 14:00 UTC on 2026-05-01 → 10:00 EDT NYC (pre-15:00 high, post-06:00 low).
    return datetime(2026, 5, 1, 14, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_final_cli_with_high_value_blocks():
    """Final CLI with a populated MAX value blocks for var=TEMP_MAX."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=_obs(high=85, low=None))
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=_pre_peak_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS


@pytest.mark.asyncio
async def test_preliminary_cli_with_high_value_also_blocks():
    """Preliminary CLI is treated identically to final."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(
        return_value=_obs(high=85, low=None, preliminary=True),
    )
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=_pre_peak_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS


@pytest.mark.asyncio
async def test_cli_with_high_none_returns_allow_for_high_market():
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=_obs(high=None, low=42))
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=_pre_peak_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.ALLOW


@pytest.mark.asyncio
async def test_cli_with_low_none_returns_allow_for_low_market():
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=_obs(high=85, low=None))
    # Use a clock pre-low-peak (var=TEMP_MIN peak is 06:00).
    # 09:00 UTC = 05:00 EDT.
    pre_low_clock = lambda: datetime(2026, 5, 1, 9, 0, tzinfo=UTC)  # noqa: E731
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=pre_low_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MIN)
    assert decision == SameDayDecision.ALLOW


@pytest.mark.asyncio
async def test_cross_stat_isolation_high_populated_low_none():
    """high_f populated but var=TEMP_MIN with low_f=None → ALLOW."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=_obs(high=85, low=None))
    pre_low_clock = lambda: datetime(2026, 5, 1, 9, 0, tzinfo=UTC)  # noqa: E731
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=pre_low_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MIN)
    assert decision == SameDayDecision.ALLOW


@pytest.mark.asyncio
async def test_cli_none_returns_allow():
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=None)
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=_pre_peak_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.ALLOW


@pytest.mark.asyncio
async def test_nws_api_error_no_cache_returns_block_fetch_failed():
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(side_effect=NWSAPIError("boom"))
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=_pre_peak_clock,
    )
    decision = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert decision == SameDayDecision.BLOCK_NWS_FETCH_FAILED


@pytest.mark.asyncio
async def test_extract_value_raises_on_unsupported_variable():
    """An unknown DailyVariable inside the gate fetch path raises ValueError."""
    from python.strategies.same_day_gate import _extract_value

    obs = _obs(high=85, low=42)
    # Synthesize an unknown variant via StrEnum subclass.
    import enum

    class _Fake(enum.StrEnum):
        UNSUPPORTED = "precip"
    with pytest.raises(ValueError, match="unsupported daily_variable"):
        _extract_value(obs, _Fake.UNSUPPORTED)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cache + near-peak TTL tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positive_cache_does_not_call_nws_again():
    """Cached BLOCK at any age short-circuits — no second HTTP call."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=_obs(high=85, low=None))
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        clock=_pre_peak_clock,
    )
    d1 = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    d2 = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert d1 == d2 == SameDayDecision.BLOCK_NWS_OBSERVATION_EXISTS
    assert nws.get_daily_observation.call_count == 1


@pytest.mark.asyncio
async def test_negative_cache_within_ttl_skips_nws():
    """Recent ALLOW (negative cache) within TTL avoids a re-fetch."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=None)
    times = [datetime(2026, 5, 1, 14, 0, tzinfo=UTC),
             datetime(2026, 5, 1, 14, 1, tzinfo=UTC)]
    it = iter(times)
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        cache_ttl_seconds=600,
        clock=lambda: next(it),
    )
    d1 = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    d2 = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert d1 == d2 == SameDayDecision.ALLOW
    assert nws.get_daily_observation.call_count == 1


@pytest.mark.asyncio
async def test_negative_cache_expires_and_refetches():
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=None)
    # 700s apart > default 600s TTL.
    times = [datetime(2026, 5, 1, 13, 0, 0, tzinfo=UTC),
             datetime(2026, 5, 1, 13, 11, 40, tzinfo=UTC)]
    it = iter(times)
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        cache_ttl_seconds=600,
        clock=lambda: next(it),
    )
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert nws.get_daily_observation.call_count == 2


@pytest.mark.asyncio
async def test_near_peak_ttl_is_tighter_than_default():
    """Within ±near_peak_window_minutes of peak, cache TTL drops to near_peak_ttl_seconds.

    Setup: clock 1h before HIGH peak (14:00 NYC = 18:00 UTC), cache aged 100s.
    Default TTL=600s would keep cache; near-peak TTL=90s expires it → re-fetch.
    """
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=None)
    times = [datetime(2026, 5, 1, 18, 0, 0, tzinfo=UTC),
             datetime(2026, 5, 1, 18, 1, 40, tzinfo=UTC)]
    it = iter(times)
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        cache_ttl_seconds=600,
        near_peak_window_minutes=120,
        near_peak_ttl_seconds=90,
        clock=lambda: next(it),
    )
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert nws.get_daily_observation.call_count == 2


@pytest.mark.asyncio
async def test_outside_near_peak_window_uses_default_ttl():
    """Outside ±2h, an entry aged 100s is still fresh (default TTL=600s)."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(return_value=None)
    # 04:00 NYC (08:00 UTC) — way outside ±2h of either peak.
    times = [datetime(2026, 5, 1, 8, 0, 0, tzinfo=UTC),
             datetime(2026, 5, 1, 8, 1, 40, tzinfo=UTC)]
    it = iter(times)
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        cache_ttl_seconds=600,
        near_peak_window_minutes=120,
        near_peak_ttl_seconds=90,
        clock=lambda: next(it),
    )
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert nws.get_daily_observation.call_count == 1


@pytest.mark.asyncio
async def test_nws_error_with_stale_negative_cache_blocks_fetch_failed():
    """NWSAPIError + stale negative cache → BLOCK_NWS_FETCH_FAILED."""
    nws = AsyncMock()
    nws.get_daily_observation = AsyncMock(side_effect=[None, NWSAPIError("boom")])
    # 1000s apart, well past 600s default.
    times = [datetime(2026, 5, 1, 14, 0, 0, tzinfo=UTC),
             datetime(2026, 5, 1, 14, 16, 40, tzinfo=UTC)]
    it = iter(times)
    gate = SameDayGate(
        nws, enabled=True, allowed_cities=frozenset(("NYC",)),
        cache_ttl_seconds=600,
        clock=lambda: next(it),
    )
    await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    d2 = await gate.check("NYC", date(2026, 5, 1), DailyVariable.TEMP_MAX)
    assert d2 == SameDayDecision.BLOCK_NWS_FETCH_FAILED
