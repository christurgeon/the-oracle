"""Tests for python/data/asos.py — Task 2.1 (fail-open client) + 2.2 (IEM parse).

Run with: uv run pytest tests/test_asos_client.py -v
"""
from __future__ import annotations

import datetime as dt

from python.data.asos import AsosClient

# ---------------------------------------------------------------------------
# Task 2.1 — fail-open contract
# ---------------------------------------------------------------------------


async def test_fail_open_returns_none_on_fetch_error(monkeypatch):
    c = AsosClient()

    async def boom(*a, **k):
        raise RuntimeError("IEM down")

    monkeypatch.setattr(c, "_fetch_rows", boom)
    # Must NOT raise — returns None so the orchestrator stays alive.
    assert await c.max_so_far(city="MIA", local_day=dt.date(2026, 6, 6)) is None


async def test_max_so_far_computes_max_and_age(monkeypatch):
    rows = [
        ("MIA", dt.datetime(2026, 6, 6, 16, 51, tzinfo=dt.UTC), 88.0),
        ("MIA", dt.datetime(2026, 6, 6, 17, 51, tzinfo=dt.UTC), 90.0),
    ]
    c = AsosClient(now_fn=lambda: dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC))

    async def fake(*a, **k):
        return rows

    monkeypatch.setattr(c, "_fetch_rows", fake)
    obs = await c.max_so_far(city="MIA", local_day=dt.date(2026, 6, 6))
    assert obs is not None
    assert obs.M == 90.0  # noqa: N803
    assert abs(obs.age_hours - (9 / 60)) < 0.05  # ~9min since 17:51 (now=18:00)
    assert obs.n_obs == 2


async def test_max_so_far_unknown_city_returns_none():
    c = AsosClient()
    result = await c.max_so_far(city="XYZ", local_day=dt.date(2026, 6, 6))
    assert result is None


async def test_max_so_far_empty_rows_returns_none(monkeypatch):
    c = AsosClient()

    async def empty(*a, **k):
        return []

    monkeypatch.setattr(c, "_fetch_rows", empty)
    result = await c.max_so_far(city="MIA", local_day=dt.date(2026, 6, 6))
    assert result is None


async def test_max_so_far_caches_result(monkeypatch):
    """Second call within TTL returns cached result without calling _fetch_rows."""
    call_count = 0
    rows = [("MIA", dt.datetime(2026, 6, 6, 17, 51, tzinfo=dt.UTC), 90.0)]

    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    async def counting(*a, **k):
        nonlocal call_count
        call_count += 1
        return rows

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows", counting)

    obs1 = await c.max_so_far(city="MIA", local_day=dt.date(2026, 6, 6))
    obs2 = await c.max_so_far(city="MIA", local_day=dt.date(2026, 6, 6))
    assert obs1 == obs2
    assert call_count == 1  # only fetched once


def test_default_cache_ttl_matches_hourly_cadence() -> None:
    """Default obs-cache TTL is 600s (10 min), aligned with ASOS's ~hourly
    publish cadence. The prior 120s default refetched IEM every ~2 ticks ->
    the 2026-06-06 IEM 429 storm. Regression guard for that fix."""
    assert AsosClient()._cache_ttl_s == 600.0


def test_cache_ttl_is_overridable() -> None:
    """Construction site (orchestrator) can override the TTL from config."""
    assert AsosClient(cache_ttl_s=900.0)._cache_ttl_s == 900.0


# ---------------------------------------------------------------------------
# Task 2.2 — fixture-based IEM CSV parse test (no network in CI)
# ---------------------------------------------------------------------------

# Minimal IEM CSV response mimicking the real format (station is the ICAO
# 4-letter code KMIA in the raw response; the client renames to "MIA").
IEM_FIXTURE_CSV = """\
station,valid,tmpf
KMIA,2026-06-06 14:53,84.20
KMIA,2026-06-06 15:53,86.00
KMIA,2026-06-06 16:53,M
KMIA,2026-06-06 17:53,88.00
KMIA,2026-06-06 18:53,T
"""


async def test_fetch_rows_parses_iem_fixture(monkeypatch):
    """_fetch_rows must parse the IEM CSV shape, skip M/T/empty rows, and
    return UTC datetime objects paired with float temperatures."""
    c = AsosClient()

    # Patch the HTTP client to return the fixture text
    async def fake_fetch(station, local_day):
        # Call the real _parse_iem_csv helper directly (avoids HTTP)
        from python.data.asos import _parse_iem_csv

        return _parse_iem_csv(IEM_FIXTURE_CSV, station)

    monkeypatch.setattr(c, "_fetch_rows", fake_fetch)

    rows = await c._fetch_rows("KMIA", dt.date(2026, 6, 6))

    assert len(rows) == 3  # two M/T rows skipped
    stations, valids, temps = zip(*rows)
    assert all(s == "MIA" for s in stations)
    assert all(isinstance(v, dt.datetime) for v in valids)
    assert all(v.tzinfo is dt.UTC for v in valids)
    assert list(temps) == [84.20, 86.00, 88.00]


async def test_fetch_rows_parses_iem_fixture_directly():
    """Parse the IEM fixture CSV via _parse_iem_csv directly."""
    from python.data.asos import _parse_iem_csv

    rows = _parse_iem_csv(IEM_FIXTURE_CSV, "KMIA")

    assert len(rows) == 3
    assert rows[0] == ("MIA", dt.datetime(2026, 6, 6, 14, 53, tzinfo=dt.UTC), 84.20)
    assert rows[1] == ("MIA", dt.datetime(2026, 6, 6, 15, 53, tzinfo=dt.UTC), 86.00)
    assert rows[2] == ("MIA", dt.datetime(2026, 6, 6, 17, 53, tzinfo=dt.UTC), 88.00)


async def test_parse_iem_csv_handles_seconds_in_valid():
    """IEM sometimes returns valid with seconds; strip to HH:MM."""
    from python.data.asos import _parse_iem_csv

    csv_with_seconds = "station,valid,tmpf\nKMIA,2026-06-06 14:53:00,84.20\n"
    rows = _parse_iem_csv(csv_with_seconds, "KMIA")
    assert len(rows) == 1
    assert rows[0][1] == dt.datetime(2026, 6, 6, 14, 53, tzinfo=dt.UTC)


# ---------------------------------------------------------------------------
# Fix 1 — _filter_to_local_day: prior-local-evening excluded from M
# ---------------------------------------------------------------------------


def test_filter_to_local_day_excludes_prior_day_evening():
    """A reading at 2026-06-06 04:00 UTC is prior local day for LA (PDT=-7h).

    LA local time at 04:00 UTC is 2026-06-05 21:00 PDT — previous local day.
    It must be EXCLUDED when filtering to local_day=2026-06-06 (LAX city).
    """
    from python.data.asos import _filter_to_local_day

    # 2026-06-06 04:00 UTC = 2026-06-05 21:00 PDT → prior local day
    prior_evening_utc = dt.datetime(2026, 6, 6, 4, 0, tzinfo=dt.UTC)
    # 2026-06-06 14:00 UTC = 2026-06-06 07:00 PDT → current local day
    current_morning_utc = dt.datetime(2026, 6, 6, 14, 0, tzinfo=dt.UTC)

    rows = [
        ("LAX", prior_evening_utc, 95.0),   # warm prior-evening → must be excluded
        ("LAX", current_morning_utc, 72.0),  # current local morning → must be included
    ]

    filtered = _filter_to_local_day(rows, "LAX", dt.date(2026, 6, 6))
    assert len(filtered) == 1
    assert filtered[0][2] == 72.0, (
        "prior-local-day evening reading (95°F) must be excluded from the local-day frame"
    )


def test_filter_to_local_day_excludes_next_day_early_utc():
    """A reading at 2026-06-07 03:00 UTC is next local day for LA.

    LA local time at 2026-06-07 03:00 UTC is 2026-06-06 20:00 PDT — same local
    day as 2026-06-06, so it SHOULD be included.
    """
    from python.data.asos import _filter_to_local_day

    # 2026-06-07 03:00 UTC = 2026-06-06 20:00 PDT → still local day 2026-06-06
    still_local_utc = dt.datetime(2026, 6, 7, 3, 0, tzinfo=dt.UTC)
    # 2026-06-07 08:00 UTC = 2026-06-07 01:00 PDT → next local day 2026-06-07
    next_day_utc = dt.datetime(2026, 6, 7, 8, 0, tzinfo=dt.UTC)

    rows = [
        ("LAX", still_local_utc, 80.0),  # still local day 06/06 → include
        ("LAX", next_day_utc, 65.0),     # local day 06/07 → exclude
    ]

    filtered = _filter_to_local_day(rows, "LAX", dt.date(2026, 6, 6))
    assert len(filtered) == 1
    assert filtered[0][2] == 80.0


def test_max_so_far_excludes_prior_local_day_reading(monkeypatch):
    """max_so_far MUST exclude a warm prior-local-day-evening UTC reading.

    Feed rows where the warmest reading (95°F) falls in the prior local day
    for LAX (PDT offset) — 04:00 UTC = 21:00 PDT on local_day - 1.
    Assert M == 72.0 (not 95.0).
    """
    # 2026-06-06 04:00 UTC = 2026-06-05 21:00 PDT → prior local day
    prior_evening_utc = dt.datetime(2026, 6, 6, 4, 0, tzinfo=dt.UTC)
    # 2026-06-06 14:00 UTC = 2026-06-06 07:00 PDT → current local day
    current_morning_utc = dt.datetime(2026, 6, 6, 14, 0, tzinfo=dt.UTC)

    rows = [
        ("LAX", prior_evening_utc, 95.0),   # warm prior-evening → must be excluded from M
        ("LAX", current_morning_utc, 72.0),  # current local morning
    ]

    c = AsosClient(now_fn=lambda: dt.datetime(2026, 6, 6, 15, 0, tzinfo=dt.UTC))

    async def fake_fetch(station, local_day):
        return rows

    monkeypatch.setattr(c, "_fetch_rows", fake_fetch)

    import asyncio

    obs = asyncio.get_event_loop().run_until_complete(
        c.max_so_far(city="LAX", local_day=dt.date(2026, 6, 6))
    )
    assert obs is not None
    assert obs.M == 72.0, (
        f"M={obs.M} but expected 72.0: prior-local-day evening (95°F) must not contaminate M"
    )
    assert obs.n_obs == 1


# ---------------------------------------------------------------------------
# Stage 0a — LST (local standard time) climate-day frame
#
# NWS CLI climate days run midnight-to-midnight LOCAL STANDARD TIME (no DST).
# CITY_TZ uses FIXED standard-time offsets year-round; these tests pin that
# behavior against the civil-frame regression that mis-assigned the
# 00:00–01:00 LDT boundary hour during DST.
# ---------------------------------------------------------------------------


def test_city_tz_uses_fixed_standard_offsets_not_dst():
    """CITY_TZ must be fixed standard-time offsets (no DST), per spec §0a.

    A civil ``ZoneInfo`` would return a DST offset in summer (e.g. EDT=-4 for
    NYC); the LST frame requires the fixed standard offset (EST=-5) year-round.
    """
    from python.data.asos import CITY_STD_OFFSET_HOURS, CITY_TZ

    assert CITY_STD_OFFSET_HOURS == {
        "NYC": -5,
        "CHI": -6,
        "LAX": -8,
        "MIA": -5,
        "AUS": -6,
    }
    # Offset must be identical in summer (DST window) and winter — i.e. fixed.
    summer = dt.datetime(2026, 7, 1, 12, 0, tzinfo=dt.UTC)
    winter = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
    for city, off in CITY_STD_OFFSET_HOURS.items():
        tz = CITY_TZ[city]
        expected = dt.timedelta(hours=off)
        assert tz.utcoffset(summer) == expected, f"{city} summer offset shifted (DST leak)"
        assert tz.utcoffset(winter) == expected, f"{city} winter offset wrong"


def test_filter_to_local_day_lst_frame_boundary_hour_during_dst():
    """A DST-period boundary-hour ob is assigned to the LST climate day, not civil.

    NYC, 2026-07-15 04:51 UTC: civil EDT = 00:51 (→ would be 07-15), but
    LST EST = 23:51 (→ belongs to climate day 07-14). The LST frame must EXCLUDE
    it from local_day=2026-07-15 and the 05:51 UTC ob (00:51 EST → 07-15) must
    be INCLUDED. This is the exact contamination memo finding #1 documents.
    """
    from python.data.asos import _filter_to_local_day

    boundary_utc = dt.datetime(2026, 7, 15, 4, 51, tzinfo=dt.UTC)  # 23:51 EST → 07-14
    after_midnight_utc = dt.datetime(2026, 7, 15, 5, 51, tzinfo=dt.UTC)  # 00:51 EST → 07-15

    rows = [
        ("NYC", boundary_utc, 79.0),  # LST day 07-14 → must be excluded
        ("NYC", after_midnight_utc, 70.0),  # LST day 07-15 → must be included
    ]
    filtered = _filter_to_local_day(rows, "NYC", dt.date(2026, 7, 15))
    assert len(filtered) == 1
    assert filtered[0][2] == 70.0, (
        "civil frame would keep the 79°F boundary-hour ob; the LST frame must drop it"
    )


# ---------------------------------------------------------------------------
# Fix 2 — station-identity assertion checks VALUES, not just keys
# ---------------------------------------------------------------------------


def test_station_assertion_passes_for_correct_map():
    """The live ASOS_STATION map must satisfy the strengthened assertion without error."""
    from python.data.asos import ASOS_STATION
    from python.data.nws import CITY_TO_CLI_LOCATION

    # This mirrors the assertion that must be at module level in asos.py.
    for city, icao in ASOS_STATION.items():
        short = icao[1:] if (len(icao) == 4 and icao.startswith("K")) else icao
        expected = CITY_TO_CLI_LOCATION[city]
        assert short == expected, (
            f"ASOS_STATION[{city!r}]={icao!r} -> short={short!r} but "
            f"CITY_TO_CLI_LOCATION[{city!r}]={expected!r}"
        )


def test_station_assertion_raises_for_wrong_station():
    """A map with CHI -> KORD (O'Hare) instead of KMDW must fail the assertion.

    The assert at import time catches this mismatch — we reproduce the logic
    here to guarantee a bad map raises AssertionError with a clear message.
    """
    from python.data.nws import CITY_TO_CLI_LOCATION

    bad_map = {
        "NYC": "KNYC",
        "CHI": "KORD",  # wrong: should be KMDW (Midway)
        "LAX": "KLAX",
        "MIA": "KMIA",
        "AUS": "KAUS",
    }

    mismatches = []
    for city, icao in bad_map.items():
        short = icao[1:] if (len(icao) == 4 and icao.startswith("K")) else icao
        expected = CITY_TO_CLI_LOCATION[city]
        if short != expected:
            mismatches.append(
                f"ASOS_STATION[{city!r}]={icao!r} -> station {short!r} != "
                f"settlement station {expected!r} for {city}"
            )
    assert mismatches, "Expected CHI/KORD to be detected as a mismatch"
    assert any("CHI" in m for m in mismatches)


# ---------------------------------------------------------------------------
# Stage 0b — batched multi-station IEM fetch (one request per TTL window)
#
# Today: one station per IEM request, cache keyed (station, local_day). When N
# cells are evaluated per tick, the first tick after the TTL expires fires N
# near-simultaneous requests → the 2026-06-06 429 storm (5 stations) and
# guaranteed breakage at any expansion. max_so_far_batch fetches all stations
# in ONE request and warms the cache so per-city max_so_far calls hit cache.
# ---------------------------------------------------------------------------


# Multi-station CSV: two stations interleaved in one response (the real IEM
# batch shape — repeated station= params return one CSV with a station column).
IEM_MULTI_FIXTURE_CSV = """\
station,valid,tmpf
KMIA,2026-06-06 14:53,84.20
KAUS,2026-06-06 14:53,90.00
KMIA,2026-06-06 15:53,86.00
KAUS,2026-06-06 15:53,M
KMIA,2026-06-06 16:53,88.00
KAUS,2026-06-06 16:53,95.00
"""


def test_parse_iem_csv_multi_demuxes_by_station():
    """_parse_iem_csv_multi splits a multi-station CSV into per-station rows."""
    from python.data.asos import _parse_iem_csv_multi

    by_station = _parse_iem_csv_multi(IEM_MULTI_FIXTURE_CSV)
    assert set(by_station) == {"MIA", "AUS"}
    assert [t for _s, _v, t in by_station["MIA"]] == [84.20, 86.00, 88.00]
    # AUS 15:53 is "M" (missing) → skipped.
    assert [t for _s, _v, t in by_station["AUS"]] == [90.00, 95.00]
    # Short-code keying and UTC tz preserved.
    assert all(v.tzinfo is dt.UTC for _s, v, _t in by_station["MIA"])


def test_parse_iem_csv_single_wrapper_still_selects_one_station():
    """The single-station wrapper selects only the requested station's rows."""
    from python.data.asos import _parse_iem_csv

    rows = _parse_iem_csv(IEM_MULTI_FIXTURE_CSV, "KMIA")
    assert [t for _s, _v, t in rows] == [84.20, 86.00, 88.00]
    assert all(s == "MIA" for s, _v, _t in rows)


async def test_max_so_far_batch_one_request_per_ttl_window(monkeypatch):
    """K stations → exactly ONE underlying fetch; a 2nd call within TTL → ZERO."""
    fetch_calls: list[list[str]] = []

    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    async def fake_batch(stations, local_day):
        fetch_calls.append(list(stations))
        from python.data.asos import _parse_iem_csv_multi

        return _parse_iem_csv_multi(IEM_MULTI_FIXTURE_CSV)

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", fake_batch)

    res1 = await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert res1 == {"MIA": 88.0, "AUS": 95.0}
    assert len(fetch_calls) == 1, "K stations must collapse to a single fetch"
    # Both stations requested in the ONE call.
    assert set(fetch_calls[0]) == {"KMIA", "KAUS"}

    # Second batch call within the TTL: cache warm → zero additional fetches.
    res2 = await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert res2 == {"MIA": 88.0, "AUS": 95.0}
    assert len(fetch_calls) == 1, "a 2nd call within TTL must make ZERO new fetches"


async def test_batch_warms_cache_so_per_city_max_so_far_is_free(monkeypatch):
    """After a batch prefetch, per-city max_so_far calls hit cache (zero fetch).

    This is the exact N→1 mechanism the strategy relies on.
    """
    batch_calls = 0
    single_calls = 0

    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    async def fake_batch(stations, local_day):
        nonlocal batch_calls
        batch_calls += 1
        from python.data.asos import _parse_iem_csv_multi

        return _parse_iem_csv_multi(IEM_MULTI_FIXTURE_CSV)

    async def fake_single(station, local_day):
        nonlocal single_calls
        single_calls += 1
        return []

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", fake_batch)
    monkeypatch.setattr(c, "_fetch_rows", fake_single)

    await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert batch_calls == 1

    obs_mia = await c.max_so_far(city="MIA", local_day=dt.date(2026, 6, 6))
    obs_aus = await c.max_so_far(city="AUS", local_day=dt.date(2026, 6, 6))
    assert obs_mia is not None and obs_mia.M == 88.0
    assert obs_aus is not None and obs_aus.M == 95.0
    assert single_calls == 0, "per-city max_so_far must hit the warm cache, not fetch"
    assert batch_calls == 1, "no extra batch fetch either"


async def test_batch_fail_open_on_missing_station(monkeypatch):
    """A station missing from the response → None for that city; others resolve."""
    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    # Response only carries KMIA; KAUS was requested but absent.
    partial_csv = "station,valid,tmpf\nKMIA,2026-06-06 16:53,88.00\n"

    async def fake_batch(stations, local_day):
        from python.data.asos import _parse_iem_csv_multi

        return _parse_iem_csv_multi(partial_csv)

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", fake_batch)

    res = await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert res["MIA"] == 88.0
    assert res["AUS"] is None, "missing station → None, others still resolve"


async def test_batch_fail_open_on_malformed_response(monkeypatch):
    """A malformed CSV (bad header) → None for every city; never raises."""
    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    async def fake_batch(stations, local_day):
        from python.data.asos import _parse_iem_csv_multi

        return _parse_iem_csv_multi("garbage,not,a,valid,header\nfoo,bar\n")

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", fake_batch)

    res = await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert res == {"MIA": None, "AUS": None}


async def test_batch_fail_open_on_fetch_error_leaves_cache_cold(monkeypatch):
    """A batch request that raises → all None; cache untouched so retry refetches."""
    call_count = 0

    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    async def boom(stations, local_day):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("IEM down")

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", boom)

    res = await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert res == {"MIA": None, "AUS": None}
    assert call_count == 1, "one batched request, no per-station fan-out on error"
    # Cache left cold → a subsequent call re-attempts the fetch.
    await c.max_so_far_batch(cities=["MIA", "AUS"], local_day=dt.date(2026, 6, 6))
    assert call_count == 2, "failed batch must not poison the cache"


async def test_batch_unknown_city_returns_none(monkeypatch):
    """An unmapped city resolves to None and is not fetched."""
    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    async def fake_batch(stations, local_day):
        # Only KMIA should be requested (XYZ is unmapped).
        assert stations == ["KMIA"]
        return {"MIA": [("MIA", dt.datetime(2026, 6, 6, 16, 53, tzinfo=dt.UTC), 88.0)]}

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", fake_batch)

    res = await c.max_so_far_batch(cities=["MIA", "XYZ"], local_day=dt.date(2026, 6, 6))
    assert res == {"MIA": 88.0, "XYZ": None}


async def test_batch_per_city_lst_frame_correct_max(monkeypatch):
    """Per-city M honors the LST local-day filter (prior-evening reading excluded).

    LAX 04:00 UTC = 21:00 PST prior local day → must NOT contaminate M.
    """
    fixed_now = dt.datetime(2026, 6, 6, 18, 0, tzinfo=dt.UTC)

    # KLAX rows: a warm prior-local-day-evening reading + a current-day reading.
    rows = {
        "LAX": [
            ("LAX", dt.datetime(2026, 6, 6, 4, 0, tzinfo=dt.UTC), 95.0),   # prior LST day
            ("LAX", dt.datetime(2026, 6, 6, 14, 0, tzinfo=dt.UTC), 72.0),  # current LST day
        ]
    }

    async def fake_batch(stations, local_day):
        assert stations == ["KLAX"]
        return rows

    c = AsosClient(now_fn=lambda: fixed_now)
    monkeypatch.setattr(c, "_fetch_rows_batch", fake_batch)

    res = await c.max_so_far_batch(cities=["LAX"], local_day=dt.date(2026, 6, 6))
    assert res["LAX"] == 72.0, "prior-local-day evening (95°F) must be excluded from M"


def test_build_iem_url_repeats_station_param():
    """The batch URL carries one station= param per requested station."""
    from python.data.asos import AsosClient

    url = AsosClient._build_iem_url(["KMIA", "KAUS"], dt.date(2026, 6, 6))
    assert url.count("station=KMIA") == 1
    assert url.count("station=KAUS") == 1
    # Repeated report_type preserved (3 and 4), single GET.
    assert "report_type=3" in url and "report_type=4" in url
