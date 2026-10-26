"""Shared Kalshi ticker-date parsing.

Extracted from ``python/strategies/weather_edge.py`` so non-weather strategies
(AI news) and the shadow broker can reuse it without importing
a strategy module.
"""

from __future__ import annotations

import re
from datetime import date

__all__ = [
    "MONTH_MAP",
    "EVENT_DATE_RE",
    "TICKER_DATE_RE",
    "INTRADAY_EVENT_DATE_RE",
    "INTRADAY_TICKER_DATE_RE",
    "SERIES_PREFIX_TO_CITY_VAR",
    "KNOWN_CITY_CODES",
    "parse_ticker_date",
    "parse_ticker_city_var",
]

MONTH_MAP: dict[str, int] = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$", re.IGNORECASE)

TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})-", re.IGNORECASE)

# Intraday weather event-ticker date, e.g. "KXHIGHMIA-26MAY15H1700"
# (spec v1.3 §4.3: ``KXHIGH<CITY>-<YYMMDD>H<HHMM>``). The resolution-hour
# segment is NOT part of the forecast date.
#
# Gated on the ``KXHIGH``/``KXLOWT`` weather-series prefix on purpose:
# unrelated series (e.g. ``KXINX-26MAR20H1600``) share the
# ``<YYMMDD>H<HHMM>`` layout but encode dates with different semantics,
# and existing callers (``shadow_broker``) rely on ``parse_ticker_date``
# returning ``None`` for those. Adding a new intraday-weather series in
# the future requires extending this prefix anchor.
INTRADAY_EVENT_DATE_RE = re.compile(
    r"^(?:KXHIGH|KXLOWT)[A-Z]+-(\d{2})([A-Z]{3})(\d{2})H\d{4}$", re.IGNORECASE
)

INTRADAY_TICKER_DATE_RE = re.compile(
    r"^(?:KXHIGH|KXLOWT)[A-Z]+-(\d{2})([A-Z]{3})(\d{2})H\d{4}-", re.IGNORECASE
)


def parse_ticker_date(ticker: str) -> date | None:
    """Extract a calendar date from a Kalshi ticker.

    Supports four ticker shapes (case-insensitive):

    * Market ticker, daily KXHIGH/KXLOWT — ``KXHIGHNY-26APR14-B85.5``
    * Event ticker, daily KXHIGH/KXLOWT — ``KXHIGHNY-26MAR14``
    * Market ticker, intraday KXHIGH    — ``KXHIGHMIA-26MAY15H1700-B85``
    * Event ticker, intraday KXHIGH     — ``KXHIGHMIA-26MAY15H1700``

    Returns ``None`` for anything unparseable — callers should treat that
    as "date unknown," not "invalid," since exotic tickers (e.g. KXINX
    with a resolution-hour suffix that isn't preceded by a forecast date)
    may encode dates differently.
    """
    # Try the daily-format regexes first (broadest match), then fall back
    # to the intraday-format regexes. The intraday regexes are strictly
    # longer matches anchored on ``H<HHMM>`` so they can't false-match a
    # daily ticker.
    m = (
        TICKER_DATE_RE.search(ticker)
        or EVENT_DATE_RE.search(ticker)
        or INTRADAY_TICKER_DATE_RE.search(ticker)
        or INTRADAY_EVENT_DATE_RE.search(ticker)
    )
    if m is None:
        return None

    year_str, month_str, day_str = m.group(1), m.group(2).upper(), m.group(3)
    month = MONTH_MAP.get(month_str)
    if month is None:
        return None

    try:
        return date(2000 + int(year_str), month, int(day_str))
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Ticker → (city, var) extraction
# ---------------------------------------------------------------------------

# Maps each known Kalshi temperature series prefix to its (city_code, var)
# tuple. This is the single source of truth for the series keyspace:
# ``weather_edge.WEATHER_SERIES`` is derived from it, and every reporting
# consumer resolves cities through ``parse_ticker_city_var``. It lives here,
# not in the strategy package, because settlement-row parsers in
# ``daily_trade_report.py`` and other consumers run outside the strategy
# module's import graph.
#
# Quirks:
#   - ``KXHIGHNY`` uses suffix "NY" but the canonical city code is "NYC".
#   - ``KXHIGHTSEA`` / ``KXHIGHTBOS`` use the "KXHIGHT" prefix (extra T).
#   - ``KXLOWT<CITY>`` always uses the full city code (NYC, not NY).
SERIES_PREFIX_TO_CITY_VAR: dict[str, tuple[str, str]] = {
    "KXHIGHNY": ("NYC", "high"),
    "KXHIGHCHI": ("CHI", "high"),
    "KXHIGHMIA": ("MIA", "high"),
    "KXHIGHLAX": ("LAX", "high"),
    "KXHIGHDEN": ("DEN", "high"),
    "KXHIGHAUS": ("AUS", "high"),
    "KXHIGHPHIL": ("PHIL", "high"),
    # KXHIGHT<CITY> is Kalshi's standard daily-high prefix for cities added from
    # 2026 onward (SEA/BOS and all new cities); legacy cities use KXHIGH<CITY>.
    "KXHIGHTSEA": ("SEA", "high"),
    "KXHIGHTBOS": ("BOS", "high"),
    "KXLOWTNYC": ("NYC", "low"),
    "KXLOWTCHI": ("CHI", "low"),
    "KXLOWTMIA": ("MIA", "low"),
    "KXLOWTLAX": ("LAX", "low"),
    "KXLOWTDEN": ("DEN", "low"),
    "KXLOWTAUS": ("AUS", "low"),
    "KXLOWTPHIL": ("PHIL", "low"),
    "KXLOWTBOS": ("BOS", "low"),
    "KXLOWTSEA": ("SEA", "low"),
    # New cities (collection-only).
    "KXHIGHTATL": ("ATL", "high"),
    "KXHIGHTDAL": ("DAL", "high"),
    "KXHIGHTHOU": ("HOU", "high"),
    "KXHIGHTLV": ("LV", "high"),
    "KXHIGHTMIN": ("MIN", "high"),
    "KXHIGHTNOLA": ("NOLA", "high"),
    "KXHIGHTOKC": ("OKC", "high"),
    "KXHIGHTPHX": ("PHX", "high"),
    "KXHIGHTSATX": ("SATX", "high"),
    "KXHIGHTSFO": ("SFO", "high"),
    "KXHIGHTDC": ("DC", "high"),
    "KXLOWTATL": ("ATL", "low"),
    "KXLOWTDAL": ("DAL", "low"),
    "KXLOWTHOU": ("HOU", "low"),
    "KXLOWTLV": ("LV", "low"),
    "KXLOWTMIN": ("MIN", "low"),
    "KXLOWTNOLA": ("NOLA", "low"),
    "KXLOWTOKC": ("OKC", "low"),
    "KXLOWTPHX": ("PHX", "low"),
    "KXLOWTSATX": ("SATX", "low"),
    "KXLOWTSFO": ("SFO", "low"),
    "KXLOWTDC": ("DC", "low"),
}

# Canonical city codes the ticker parser can emit (source of truth for any
# caller that needs to validate a city against what a real ticker resolves to —
# e.g. config-layer "{CITY}_{var}" exclusion keys). None contain an underscore,
# so "{city}_{var}" round-trips unambiguously.
KNOWN_CITY_CODES: frozenset[str] = frozenset(c for c, _ in SERIES_PREFIX_TO_CITY_VAR.values())

# Resolve longest prefixes first so ``KXHIGHTSEA`` matches before
# ``KXHIGH...`` would (defensive — the current map has no shorter-key
# false-prefix collision because no city code begins with "TSEA" or
# "TBOS", but pinning sort order makes the contract explicit).
_SERIES_PREFIXES_ORDERED: tuple[str, ...] = tuple(
    sorted(SERIES_PREFIX_TO_CITY_VAR, key=len, reverse=True)
)


def parse_ticker_city_var(ticker: str) -> tuple[str, str] | None:
    """Extract ``(city, var)`` from a Kalshi temperature ticker.

    Returns ``(city, var)`` where ``var ∈ {"high", "low"}`` for any of:

    * Daily KXHIGH market/event — ``KXHIGHMIA-26MAY15`` /
      ``KXHIGHMIA-26MAY15-B85``
    * Daily KXLOWT market/event — ``KXLOWTNYC-26MAY15`` /
      ``KXLOWTNYC-26MAY15-B58``
    * Intraday KXHIGH market/event (spec v1.3 §4.3) —
      ``KXHIGHMIA-26MAY15H1700`` / ``KXHIGHMIA-26MAY15H1700-B85``

    Returns ``None`` for empty/garbage input, unknown city codes,
    non-weather series (e.g. ``KXINX``), or tickers missing the
    ``-<YYMMDD>`` date segment.

    Intraday ``KXLOWT`` tickers (``KXLOWT<CITY>-<YYMMDD>H<HHMM>...``) are
    handled by the same regex/prefix machinery; the helper does not gate
    on series semantics, so adding low-temp intraday markets in the
    future requires no change here.

    Required by spec v1.3 §9.2(c) (MINOR #2): settlement rows in
    ``state/trades.jsonl`` do NOT carry ``metadata["cell"]`` (verified
    in ``python/execution/engine.py``). The daily-report
    consec-loss helper recovers ``cell = "<city>/<var>"`` via this
    function in the preferred single-row-scan path.
    """
    if not ticker:
        return None

    upper = ticker.upper()

    # The series prefix runs up to the first hyphen.
    dash_idx = upper.find("-")
    if dash_idx <= 0:
        return None
    prefix = upper[:dash_idx]

    city_var = SERIES_PREFIX_TO_CITY_VAR.get(prefix)
    if city_var is None:
        # Defensive longest-first fallback: if the prefix-up-to-dash
        # didn't match exactly, try each known prefix as a startswith
        # check. Today this is unreachable (every known series prefix
        # is dash-terminated), but it keeps the contract robust against
        # an unexpected Kalshi quirk.
        for known in _SERIES_PREFIXES_ORDERED:  # pragma: no cover
            if upper.startswith(known + "-"):
                city_var = SERIES_PREFIX_TO_CITY_VAR[known]
                break
        if city_var is None:
            return None

    # Confirm the ticker actually carries a date segment — otherwise a
    # bare series-prefix string like "KXHIGHMIA" would falsely resolve
    # to ("MIA", "high"). Re-use ``parse_ticker_date`` so the matrix of
    # accepted date formats stays in lockstep.
    if parse_ticker_date(ticker) is None:
        return None

    return city_var
