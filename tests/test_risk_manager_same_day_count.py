"""Tests for ``RiskManager.count_same_day_bets``.

Invariants pinned here:
  - Only open positions (present in ``_positions``) count. Closed positions
    are popped by ``close_position`` so the dict-based count is correct by
    construction; we still pin the behavior to guard against a refactor that
    flips to a soft-tombstone semantics later.
  - Only positions whose ticker resolves to the queried forecast_date count.
  - Non-weather tickers (or unparseable ones) contribute 0 — never throttle
    non-weather strategies that share the same RiskManager.
"""

from __future__ import annotations

from datetime import date

import pytest

from python.risk.manager import RiskManager


@pytest.fixture
def rm(default_settings) -> RiskManager:  # type: ignore[no-untyped-def]
    return RiskManager(default_settings)


class TestCountSameDayBets:
    def test_empty_positions_returns_zero(self, rm: RiskManager) -> None:
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 0

    def test_includes_open_positions_on_date(self, rm: RiskManager) -> None:
        """Three open positions on 2026-05-18 ⇒ count == 3 for that date,
        and 0 for any other date."""
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)
        rm.record_fill("KXLOWTMIA-26MAY18-B68", "weather_edge", dollars=15.0)
        rm.record_fill("KXHIGHCHI-26MAY18-T72", "weather_edge", dollars=20.0)

        assert rm.count_same_day_bets(date(2026, 5, 18)) == 3
        assert rm.count_same_day_bets(date(2026, 5, 19)) == 0
        assert rm.count_same_day_bets(date(2026, 5, 17)) == 0

    def test_excludes_closed_positions(self, rm: RiskManager) -> None:
        """``close_position`` pops the entry; the count must reflect the
        live dict, not historical fills."""
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)
        rm.record_fill("KXLOWTMIA-26MAY18-B68", "weather_edge", dollars=15.0)
        rm.record_fill("KXHIGHCHI-26MAY18-T72", "weather_edge", dollars=20.0)
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 3

        # Close one; the dict should drop it.
        rm.close_position("KXLOWTMIA-26MAY18-B68", pnl=0.0)
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 2

    def test_excludes_other_dates(self, rm: RiskManager) -> None:
        """Multi-date fixture: only the queried date is counted."""
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)
        rm.record_fill("KXHIGHNY-26MAY19-B59.5", "weather_edge", dollars=12.0)
        rm.record_fill("KXHIGHNY-26MAY20-B60.5", "weather_edge", dollars=14.0)
        rm.record_fill("KXLOWTCHI-26MAY18-T48", "weather_edge", dollars=8.0)

        assert rm.count_same_day_bets(date(2026, 5, 18)) == 2
        assert rm.count_same_day_bets(date(2026, 5, 19)) == 1
        assert rm.count_same_day_bets(date(2026, 5, 20)) == 1
        assert rm.count_same_day_bets(date(2026, 5, 21)) == 0

    def test_unparseable_ticker_does_not_count(self, rm: RiskManager) -> None:
        """A non-weather series (e.g. KXINX with its hour suffix decoded
        differently) or a malformed ticker must contribute 0 so the haircut
        can never throttle a non-weather strategy sharing the RiskManager.

        ``parse_ticker_date`` returns ``None`` for KXINX with the hour
        suffix because the regex matrix is anchored to weather-series
        prefixes for the intraday format. We verify by simulating such a
        position and asserting it doesn't count against any date."""
        # KXINX format with hour suffix is intentionally non-weather; the
        # ticker_dates parser only recognizes the intraday H<HHMM> shape
        # when prefixed by KXHIGH/KXLOWT, so this should fail to parse.
        rm.record_fill("KXINX-26MAY18H1600-B4000", "spx_bracket", dollars=50.0)
        rm.record_fill("garbage-not-a-ticker", "weather_edge", dollars=5.0)
        # Plus one real weather position so we know the count machinery works
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)

        # Only the weather row should count
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 1

    def test_intraday_weather_ticker_counts(self, rm: RiskManager) -> None:
        """Intraday KXHIGH (KXHIGHMIA-26MAY18H1700-B85) IS a weather
        series — the parser handles it, so the count should include it."""
        rm.record_fill("KXHIGHMIA-26MAY18H1700-B85", "intraday_emos_high", dollars=8.0)
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)

        assert rm.count_same_day_bets(date(2026, 5, 18)) == 2

    def test_release_exposure_reduces_count_when_zero(self, rm: RiskManager) -> None:
        """``release_exposure`` pops the entry when reduced to zero;
        the count must drop in lockstep."""
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)
        rm.record_fill("KXLOWTMIA-26MAY18-B68", "weather_edge", dollars=15.0)
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 2

        # Fully release one
        rm.release_exposure("KXHIGHNY-26MAY18-B58.5", dollars=10.0)
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 1

    def test_partial_release_keeps_entry_and_count(self, rm: RiskManager) -> None:
        """A partial release leaves the entry in the dict, so the count
        is unchanged. Matches the dict-based ``_positions`` semantics."""
        rm.record_fill("KXHIGHNY-26MAY18-B58.5", "weather_edge", dollars=10.0)
        rm.release_exposure("KXHIGHNY-26MAY18-B58.5", dollars=5.0)
        assert rm.count_same_day_bets(date(2026, 5, 18)) == 1
