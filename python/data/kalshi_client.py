"""
Kalshi REST API v2 + WebSocket client.

Fully async using aiohttp. Implements RSA-PSS request signing, rate limit
tracking, and WebSocket auto-reconnect with heartbeat.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

if TYPE_CHECKING:
    from python.data.contract_watcher import ContractWatcher

logger = logging.getLogger(__name__)

# Maximum number of pagination pages before aborting to prevent infinite loops.
MAX_PAGES = 100


@dataclass(frozen=True)
class PaginationResult:
    """Full result of a paginated Kalshi fetch.

    The richer return shape (vs the legacy ``(rows, truncated)`` tuple) lets the
    FillsReconciler reuse this loop and still get ``cycle_aborted`` (which feeds its
    per-mode critical alert) and ``pages_consumed`` (for logging).
    """

    rows: list[dict[str, Any]]
    truncated: bool  # hit max_pages with a pending non-empty cursor
    cycle_aborted: bool  # server returned an already-seen cursor (API contract break)
    pages_consumed: int  # number of fetches performed (cycling fetch counts)


async def paginate_detailed(
    *,
    method_name: str,
    list_field: str,
    fetch: Callable[[str | None], Awaitable[dict[str, Any]]],
    page_delay: float,
    filter_repr: str,
    max_pages: int = MAX_PAGES,
) -> PaginationResult:
    """Drive a Kalshi paginated endpoint; return a :class:`PaginationResult`.

    The single cursor-pagination loop shared by ``KalshiClient._iter_paginated``
    (markets / positions / orders) AND the FillsReconciler (via a ``fetch`` lambda
    over ``get_fills``). Free function so callers only need to supply a ``fetch``
    callable — no instance/private-method coupling.

    ``truncated`` is True iff the loop hit ``max_pages`` with a pending non-empty
    cursor. ``cycle_aborted`` is True iff the server returned an already-seen cursor
    (A->B->A) — the cycling page's rows are DISCARDED (the cursor returns the same
    rows every call, so extending would inflate sums; this is the 2026-05-11
    reconciler-inflation fix, now shared). Cycle and truncation are mutually
    exclusive (cycle breaks first). ``not cursor`` treats both ``None`` and ``""``
    as terminal. ``page_delay`` is in SECONDS and applies only between fetches.
    """
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    truncated = False
    cycle_aborted = False
    pages_consumed = 0
    while True:
        if pages_consumed >= max_pages:
            # Cap hit. Truncated only if a fresh page was still pending.
            truncated = bool(cursor)
            if truncated:
                logger.warning(
                    "%s: reached MAX_PAGES=%d page limit — aborting pagination "
                    "to prevent infinite loop (%s) — RESULT IS TRUNCATED",
                    method_name, max_pages, filter_repr,
                )
            break
        if pages_consumed > 0 and page_delay > 0:
            await asyncio.sleep(page_delay)
        resp = await fetch(cursor)
        pages_consumed += 1
        page_rows = resp.get(list_field, [])
        cursor = resp.get("cursor")
        # Cycle check BEFORE extend so the repeating page is discarded.
        if cursor and cursor in seen_cursors:
            logger.warning(
                "%s: cursor cycle detected (%r already seen) — aborting "
                "pagination (%s page=%d)",
                method_name, cursor, filter_repr, pages_consumed,
            )
            cycle_aborted = True
            break
        rows.extend(page_rows)
        if not cursor:  # None or "" → terminal
            break
        seen_cursors.add(cursor)
    return PaginationResult(
        rows=rows,
        truncated=truncated,
        cycle_aborted=cycle_aborted,
        pages_consumed=pages_consumed,
    )

# Valid inclusive bounds for a limit-order price in cents.
_MIN_LIMIT_CENTS = 1
_MAX_LIMIT_CENTS = 99


def _build_connector() -> aiohttp.TCPConnector:
    """Build a TCPConnector that resolves DNS off the event loop's thread pool.

    aiohttp's default ``ThreadedResolver`` runs ``socket.getaddrinfo`` in the
    event loop's default ``ThreadPoolExecutor`` — only ~6 threads on the 2-CPU
    VPS — where it competes with every ``asyncio.to_thread`` call (the
    CalibrationMonitor settlements walk, Phase0 writes, heartbeat writes).
    Under load that pool saturates, DNS resolution queues behind it, and the
    event loop stalls long enough to miss the systemd watchdog ping → SIGABRT
    (the 2026-06-06 21:38 UTC crash). ``AsyncResolver`` (aiodns / c-ares) does
    DNS fully async on the event loop with zero thread-pool involvement.

    Falls back to the default-resolver connector (with a warning) if aiodns is
    not installed, so the bot never fails to start on an under-provisioned env.
    Must be called inside a running event loop — TCPConnector binds to it.

    ``keepalive_timeout`` is set above the 60s orchestrator ``poll_interval_seconds``
    so a pooled REST connection survives the idle gap between ticks instead of
    expiring (aiohttp default 15s) and forcing a fresh DNS + TLS handshake every
    tick. Reusing a server-closed (stale) socket is self-healing: ``_request``
    catches ``aiohttp.ClientError`` and retries. It is a no-op for the WS session
    (a single long-lived connection isn't pooled) but harmless there. We do not
    set ``enable_cleanup_closed`` — it is ignored and warns on Python ≥3.12.7 /
    3.13 (the CPython SSL bug it worked around is fixed).
    """
    # 60s poll_interval + 15s margin; keep below any unknown server idle timeout.
    keepalive_timeout = 75.0
    try:
        resolver = aiohttp.AsyncResolver()
    except RuntimeError:
        logger.warning(
            "aiodns not installed; falling back to aiohttp ThreadedResolver. "
            "DNS will run in the default thread pool and can stall the event "
            "loop under contention — install aiodns to avoid watchdog timeouts."
        )
        return aiohttp.TCPConnector(keepalive_timeout=keepalive_timeout)
    return aiohttp.TCPConnector(resolver=resolver, keepalive_timeout=keepalive_timeout)

# ---------------------------------------------------------------------------
# RSA-PSS request signing (Kalshi auth scheme)
# ---------------------------------------------------------------------------


def _load_private_key(path: str, password: bytes | None = None) -> rsa.RSAPrivateKey:
    """Load an RSA private key from a PEM file.

    ``password`` decrypts an encrypted PEM (C-KALSHI-PEM).
    ``None`` (the default) loads an unencrypted PEM, preserving prior behavior.
    """
    key_bytes = Path(path).read_bytes()
    private_key = serialization.load_pem_private_key(key_bytes, password=password)
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise TypeError("Kalshi requires an RSA private key")
    return private_key


def _sign_request(
    private_key: rsa.RSAPrivateKey,
    timestamp_ms: str,
    method: str,
    path: str,
) -> str:
    """
    Create an RSA-PSS signature for a Kalshi API request.

    The message to sign is: ``timestamp_ms + method + path``
    """
    message = (timestamp_ms + method.upper() + path).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: float = 10.0) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            # Prune old timestamps
            self._timestamps = [t for t in self._timestamps if now - t < self._window]
            if len(self._timestamps) >= self._max:
                sleep_for = self._window - (now - self._timestamps[0])
                if sleep_for > 0:
                    logger.debug("Rate limit: sleeping %.2fs", sleep_for)
                    await asyncio.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RETRYABLE_HTTP_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503})
_SAFE_RETRY_METHODS: frozenset[str] = frozenset({"GET", "DELETE"})


def _compute_backoff(attempt: int, base_delay: float, max_delay: float = 30.0) -> float:
    """Exponential backoff with ±25% jitter, capped at *max_delay*."""
    jitter: float = random.uniform(0.75, 1.25)
    raw: float = base_delay * (2**attempt) * jitter
    return min(raw, max_delay)


def _get_retry_delay(
    status: int,
    retry_after: str | None,
    attempt: int,
    base_delay: float,
) -> float:
    """Return the delay before the next retry attempt.

    For 429 responses the ``Retry-After`` header is preferred (capped at 120 s).
    If absent, 429 uses a longer backoff series (5 s base) than 5xx errors.
    """
    if status == 429 and retry_after is not None:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    if status == 429:
        return _compute_backoff(attempt, max(base_delay * 5, 5.0), max_delay=120.0)
    # 5xx — standard short backoff
    return _compute_backoff(attempt, base_delay, max_delay=30.0)


# ---------------------------------------------------------------------------
# Order creation — Kalshi V2 endpoint
# ---------------------------------------------------------------------------

# Kalshi deprecated the legacy "POST /portfolio/orders" create endpoint on
# ~2026-06-19 (it now returns HTTP 410 "deprecated_v1_order_endpoint"). Orders
# are created via the V2 single-book endpoint, which takes a bid/ask side and a
# fixed-point *dollar* price denominated in YES terms.
_V2_ORDERS_PATH = "/portfolio/events/orders"


def build_v2_order_body(
    *,
    ticker: str,
    side: str,
    action: str,
    count: int,
    yes_price: int | None,
    no_price: int | None,
    expiration_ts: int | None,
    self_trade_prevention_type: str = "taker_at_cross",
) -> dict[str, Any]:
    """Map our legacy order parameters onto the Kalshi V2 create-order body.

    The V2 endpoint uses a single (YES) book: ``side`` is ``bid`` (acquire YES
    exposure) or ``ask`` (shed YES exposure), and ``price`` is the YES price as
    a fixed-point dollar string. Buying NO @ p is economically selling YES @
    (1 - p), so a NO buy maps to an ``ask`` at the complement price.

    DOCS-VERIFIED: only the (side=yes, action=buy) -> "bid" mapping is confirmed
    against the published example request. The NO / sell mappings are derived
    from economic equivalence and MUST be confirmed against the demo API before
    any NO-placing or sell strategy trades live. The only live order-placing
    strategy today (favorite_bias) is YES-buy-only.

    Assumes inputs already passed ``KalshiClient._validate_order`` (side in
    {yes, no}, the matching price is set, count >= 1).
    """
    acquires_yes = (side == "yes" and action == "buy") or (
        side == "no" and action == "sell"
    )
    v2_side = "bid" if acquires_yes else "ask"
    # YES-denominated price in cents: NO price p ≡ YES price (100 - p).
    yes_price_cents = yes_price if side == "yes" else (100 - (no_price or 0))
    body: dict[str, Any] = {
        "ticker": ticker,
        "side": v2_side,
        "count": f"{count:.2f}",
        "price": f"{(yes_price_cents or 0) / 100:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": self_trade_prevention_type,
    }
    if expiration_ts is not None:
        # expiration_time requires time_in_force=good_till_canceled (Kalshi).
        body["expiration_time"] = expiration_ts
    return body


# ---------------------------------------------------------------------------
# Kalshi REST client
# ---------------------------------------------------------------------------


class KalshiClient:
    """
    Async client for the Kalshi REST API v2.

    Usage::

        async with KalshiClient(api_key="...", private_key_path="key.pem") as client:
            markets = await client.get_markets(status="open")
    """

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limit: int = 100,
        rate_window: float = 10.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        contract_watcher: ContractWatcher | None = None,
        private_key_password: bytes | None = None,
    ) -> None:
        self.api_key = api_key
        self._private_key = _load_private_key(private_key_path, password=private_key_password)
        self.base_url = base_url.rstrip("/")
        # Kalshi signs the FULL request path including the API prefix (e.g.
        # "/trade-api/v2/portfolio/orders"), not just the endpoint suffix.
        # The WebSocket already does this; REST historically signed only the
        # suffix and silently passed through public read endpoints.
        self._base_path = urlparse(self.base_url).path
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiter = RateLimiter(rate_limit, rate_window)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._backoff_until: float = 0.0  # Global 429 backoff gate (monotonic)
        self._last_skew_check: float = 0.0  # Throttle clock skew detection
        self._contract_watcher = contract_watcher

    def backoff_remaining_seconds(self) -> float:
        """Seconds left on the global 429 backoff gate; ``0.0`` when it is clear."""
        return max(0.0, self._backoff_until - time.monotonic())

    # -- Lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> KalshiClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=_build_connector(),
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -- Auth headers --------------------------------------------------------

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        sig = _sign_request(self._private_key, ts_ms, method, f"{self._base_path}{path}")
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # -- Low-level request ---------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        *,
        expected_error_statuses: set[int] | None = None,
    ) -> dict[str, Any]:
        """Execute an authenticated API request with rate limiting and retry.

        Retry policy is method-aware:
        - **GET / DELETE**: retry on 429, 5xx, timeout, and connection errors.
        - **POST**: retry on 429 only (server rejected before processing).
          Timeout / 5xx on POST means order status is unknown — do NOT retry.

        A global backoff gate (``_backoff_until``) prevents concurrent requests
        from stampeding the API during a 429 cooldown.

        ``expected_error_statuses`` lists HTTP statuses the caller anticipates
        and handles itself (e.g. the post-placement ``get_order`` 404
        read-after-write race). Such a response is logged at DEBUG instead of
        ERROR — it still raises ``KalshiAPIError`` so the caller can react, but
        it does not pollute the journal with per-order ERROR noise that would
        mask a genuine failure.
        """
        if self._session is None or self._session.closed:
            raise RuntimeError("Client not connected — use `async with` or call connect()")

        is_safe = method.upper() in _SAFE_RETRY_METHODS
        url = f"{self.base_url}{path}"

        for attempt in range(self._max_retries + 1):
            # Only consume a rate-limit slot on the first attempt — retries
            # should not burn tokens while the server is telling us to wait.
            if attempt == 0:
                await self._rate_limiter.acquire()

            # Honour the global 429 backoff gate set by any prior request.
            now = time.monotonic()
            if self._backoff_until > now:
                wait = self._backoff_until - now
                logger.info("Global 429 backoff: waiting %.1fs", wait)
                await asyncio.sleep(wait)
                # Many concurrent waiters wake at the same instant when the
                # backoff lifts. Re-acquire a rate-limit slot so they don't
                # stampede the server and re-trigger the cooldown.
                await self._rate_limiter.acquire()

            # Fresh auth headers per attempt (timestamps must not go stale).
            headers = self._auth_headers(method, path)

            try:
                async with self._session.request(
                    method, url, headers=headers, params=params, json=json_body
                ) as resp:
                    # Read body once to avoid double-read issues.
                    raw_text = await resp.text()
                    json_ok = True
                    try:
                        body: dict[str, Any] = json.loads(raw_text)
                    except (json.JSONDecodeError, ValueError):
                        json_ok = False
                        body = {"error": f"Invalid JSON response: {raw_text[:500]}"}

                    if resp.status < 400:
                        if not json_ok:
                            raise KalshiAPIError(resp.status, body)
                        # Clock skew detection (throttled to once per 60s)
                        now_mono = time.monotonic()
                        if now_mono - self._last_skew_check > 60.0:
                            self._last_skew_check = now_mono
                            server_date = resp.headers.get("Date")
                            if server_date:
                                try:
                                    server_dt = parsedate_to_datetime(server_date)
                                    skew = abs(time.time() - server_dt.timestamp())
                                    if skew > 30:
                                        logger.warning(
                                            "Clock skew detected: %.1fs difference "
                                            "with Kalshi server (auth may fail if >60s)",
                                            skew,
                                        )
                                        if self._contract_watcher is not None:
                                            self._contract_watcher.warn(
                                                "clock_skew",
                                                f"Clock skew with Kalshi server: {skew:.1f}s. "
                                                f"Auth signatures may fail if skew exceeds 60s.",
                                            )
                                except Exception:
                                    logger.debug("Clock skew check failed", exc_info=True)
                        return body

                    # ----- Error path -----
                    is_retryable_status = resp.status in _RETRYABLE_HTTP_STATUSES
                    # POST: only 429 is safe to retry (rate-limit rejection).
                    can_retry = (is_retryable_status and is_safe) or (
                        resp.status == 429 and not is_safe
                    )

                    if can_retry and attempt < self._max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        delay = _get_retry_delay(
                            resp.status, retry_after, attempt, self._retry_base_delay
                        )
                        if resp.status == 429:
                            self._backoff_until = time.monotonic() + delay
                        logger.warning(
                            "Kalshi %s %s → %d, retry %d/%d in %.1fs",
                            method,
                            path,
                            resp.status,
                            attempt + 1,
                            self._max_retries,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if expected_error_statuses and resp.status in expected_error_statuses:
                        # Caller anticipates and handles this status — keep it out
                        # of the ERROR stream (still raise so the caller reacts).
                        msg = "Kalshi API %s %s → %s (anticipated): %s"
                        logger.debug(msg, method, path, resp.status, body)
                    else:
                        logger.error("Kalshi API %s %s → %s: %s", method, path, resp.status, body)
                    raise KalshiAPIError(resp.status, body)

            except (TimeoutError, aiohttp.ClientError) as exc:
                # Only retry connection/timeout errors for safe methods.
                if is_safe and attempt < self._max_retries:
                    delay = _compute_backoff(attempt, self._retry_base_delay)
                    logger.warning(
                        "Kalshi %s %s → %s, retry %d/%d in %.1fs",
                        method,
                        path,
                        type(exc).__name__,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise KalshiAPIError(0, {"error": f"{type(exc).__name__}: {exc}"}) from exc

        # Unreachable — the loop always raises or returns — but satisfy mypy.
        raise RuntimeError("Retry loop exited unexpectedly")  # pragma: no cover

    # -- Markets -------------------------------------------------------------

    async def get_markets(
        self,
        *,
        status: str | None = None,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        List markets.  Returns ``{"markets": [...], "cursor": "..."}``
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/markets", params=params)

    async def get_market(self, ticker: str) -> dict[str, Any]:
        """Get details for a single market by ticker."""
        return await self._request("GET", f"/markets/{ticker}")

    async def get_event(self, event_ticker: str) -> dict[str, Any]:
        """Get event details (an event groups related markets)."""
        return await self._request("GET", f"/events/{event_ticker}")

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        """Get the order book for a market."""
        return await self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    async def get_market_trades(
        self,
        *,
        ticker: str,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Public trade prints for one market, newest first.

        Goes through ``_request`` although no-auth, so it shares the rate
        limiter and 429 gate; an unmetered side channel trips the sliding window.
        """
        params: dict[str, Any] = {"ticker": ticker, "limit": limit}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/markets/trades", params=params)

    async def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict[str, Any]:
        """
        Fetch OHLC candlestick price history for a market.

        Args:
            series_ticker: The series this market belongs to (e.g. "KXHIGHNY").
            ticker: The market ticker (e.g. "KXHIGHNY-26MAR14-T56").
            start_ts: Unix timestamp (seconds) for start of range.
            end_ts: Unix timestamp (seconds) for end of range.
            period_interval: Candle size in minutes (1, 60, or 1440).

        Returns:
            {"candlesticks": [...]} where each candle has end_period_ts,
            yes_price (open/high/low/close in cents), no_price, volume, open_interest.
        """
        params: dict[str, Any] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        return await self._request(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params=params,
        )

    # -- Orders --------------------------------------------------------------

    @staticmethod
    def _validate_limit_price(field_name: str, value: int | None) -> None:
        """Validate a limit-order price for the active side.

        ``field_name`` is ``"yes_price"`` or ``"no_price"``; the side name in
        the "required" message is derived from it. Raises
        :class:`OrderValidationError` on a missing or out-of-range price.
        """
        side_name = "YES" if field_name == "yes_price" else "NO"
        if value is None:
            raise OrderValidationError(
                f"{field_name} is required for limit orders on the {side_name} side"
            )
        if not isinstance(value, int) or not (_MIN_LIMIT_CENTS <= value <= _MAX_LIMIT_CENTS):
            raise OrderValidationError(
                f"{field_name} must be an integer in [1, 99], got {value!r}"
            )

    def _validate_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str,
        order_type: str,
        count: int,
        yes_price: int | None,
        no_price: int | None,
        expiration_ts: int | None,
        sell_position_floor: int | None,
    ) -> None:
        """Validate order parameters before sending to the Kalshi API.

        Raises :class:`OrderValidationError` for any invalid parameter.
        """
        if not isinstance(ticker, str) or not ticker:
            raise OrderValidationError(f"ticker must be a non-empty string, got {ticker!r}")

        if side not in ("yes", "no"):
            raise OrderValidationError(f"side must be 'yes' or 'no', got {side!r}")

        if action not in ("buy", "sell"):
            raise OrderValidationError(f"action must be 'buy' or 'sell', got {action!r}")

        if order_type not in ("limit", "market"):
            raise OrderValidationError(
                f"order_type must be 'limit' or 'market', got {order_type!r}"
            )

        if not isinstance(count, int) or count < 1:
            raise OrderValidationError(f"count must be a positive integer, got {count!r}")

        # Price validation for limit orders
        if order_type == "limit":
            if side == "yes":
                self._validate_limit_price("yes_price", yes_price)
            else:  # side == "no"
                self._validate_limit_price("no_price", no_price)

        # Optional parameter validation
        if expiration_ts is not None:
            if not isinstance(expiration_ts, int) or expiration_ts <= 0:
                raise OrderValidationError(
                    f"expiration_ts must be a positive integer, got {expiration_ts!r}"
                )

        if sell_position_floor is not None:
            if not isinstance(sell_position_floor, int) or sell_position_floor < 0:
                raise OrderValidationError(
                    f"sell_position_floor must be a non-negative integer, "
                    f"got {sell_position_floor!r}"
                )

    async def place_order(
        self,
        *,
        ticker: str,
        side: str,
        action: str = "buy",
        order_type: str = "limit",
        count: int = 1,
        yes_price: int | None = None,
        no_price: int | None = None,
        expiration_ts: int | None = None,
        sell_position_floor: int | None = None,
    ) -> dict[str, Any]:
        """
        Place an order on a market.

        Args:
            ticker: Market ticker.
            side: "yes" or "no".
            action: "buy" or "sell".
            order_type: "limit" or "market".
            count: Number of contracts.
            yes_price: Limit price in cents (1-99) for YES side.
            no_price: Limit price in cents (1-99) for NO side.
            expiration_ts: Unix timestamp for order expiry (optional).
            sell_position_floor: For decrease-only orders, set floor to current position.
        """
        self._validate_order(
            ticker=ticker,
            side=side,
            action=action,
            order_type=order_type,
            count=count,
            yes_price=yes_price,
            no_price=no_price,
            expiration_ts=expiration_ts,
            sell_position_floor=sell_position_floor,
        )
        if order_type != "limit":
            # The V2 create endpoint has no "market" type; a marketable limit
            # must be constructed upstream if ever needed. No live caller places
            # market orders, so reject loudly rather than guess a mapping.
            raise OrderValidationError(
                "market orders are not supported via the Kalshi V2 create "
                "endpoint; submit a marketable limit instead"
            )
        body = build_v2_order_body(
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            yes_price=yes_price,
            no_price=no_price,
            expiration_ts=expiration_ts,
        )
        # sell_position_floor (legacy decrease-only floor) has no V2 equivalent
        # wired here; no live caller passes it. Map to reduce_only if ever needed.
        return await self._request("POST", _V2_ORDERS_PATH, json_body=body)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an open order by ID."""
        return await self._request("DELETE", f"/portfolio/orders/{order_id}")

    async def get_orders(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List orders, optionally filtered by ticker or status."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/portfolio/orders", params=params)

    async def get_order(
        self, order_id: str, *, expected_error_statuses: set[int] | None = None
    ) -> dict[str, Any]:
        """Get a single order by ID.

        ``expected_error_statuses`` is forwarded to ``_request`` so callers that
        anticipate a status (e.g. the post-placement 404 read-after-write race)
        can keep it out of the ERROR log. See ``_request`` for the rationale.
        """
        return await self._request(
            "GET",
            f"/portfolio/orders/{order_id}",
            expected_error_statuses=expected_error_statuses,
        )

    # -- Portfolio / Positions -----------------------------------------------

    async def get_positions(
        self,
        *,
        ticker: str | None = None,
        settlement_status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List current positions."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if settlement_status:
            params["settlement_status"] = settlement_status
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", "/portfolio/positions", params=params)

    async def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        return await self._request("GET", "/portfolio/balance")

    async def get_fills(
        self,
        *,
        ticker: str | None = None,
        order_id: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
        min_ts: int | None = None,
        max_ts: int | None = None,
    ) -> dict[str, Any]:
        """List trade fills."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        return await self._request("GET", "/portfolio/fills", params=params)

    # -- Paginated helpers ---------------------------------------------------

    async def iter_all_markets(
        self,
        *,
        status: str | None = None,
        series_ticker: str | None = None,
        page_delay: float = 1.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetch all pages of ``/markets``.

        See :meth:`_iter_paginated` for the truncation / cursor-cycle
        contract. ``page_delay`` exists to avoid Kalshi's undocumented
        sliding-window rate limit.
        """
        return await self._iter_paginated(
            method_name="iter_all_markets",
            list_field="markets",
            fetch=lambda cursor: self.get_markets(
                status=status,
                series_ticker=series_ticker,
                cursor=cursor,
            ),
            page_delay=page_delay,
            filter_repr=f"status={status!r} series_ticker={series_ticker!r}",
        )

    async def _iter_paginated(
        self,
        *,
        method_name: str,
        list_field: str,
        fetch: Callable[[str | None], Awaitable[dict[str, Any]]],
        page_delay: float,
        filter_repr: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Drive a Kalshi paginated endpoint; return ``(rows, truncated)``.

        Thin back-compat wrapper over the shared :func:`paginate_detailed` — existing
        callers keep the ``(rows, truncated)`` contract and inherit the
        discard-cycling-page fix for free.
        """
        result = await paginate_detailed(
            method_name=method_name,
            list_field=list_field,
            fetch=fetch,
            page_delay=page_delay,
            filter_repr=filter_repr,
        )
        return result.rows, result.truncated

    async def iter_all_positions(
        self,
        *,
        ticker: str | None = None,
        settlement_status: str | None = None,
        page_delay: float = 1.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetch all pages of ``/portfolio/positions``.

        See :meth:`_iter_paginated` for the truncation / cursor-cycle
        contract. ``page_delay`` exists to avoid Kalshi's undocumented
        sliding-window rate limit.

        v2 response schema (each row in ``market_positions``)::

            {
                "ticker": "KXHIGHDEN-26APR28-B54.5",
                "position_fp": "-16.00",          # signed string; +YES / -NO; may be fractional
                "market_exposure_dollars": "11.000000",  # positive magnitude (string)
                "fees_paid_dollars": "0.070000",
                "realized_pnl_dollars": "0.000000",
                "resting_orders_count": 0,
                "total_traded_dollars": "11.000000",
                "last_updated_ts": "..."
            }

        Note that legacy fields ``market_ticker``, integer ``position``,
        and ``market_average_price`` are **not** present in v2; readers
        that touched those fields silently filtered every row out.
        """
        return await self._iter_paginated(
            method_name="iter_all_positions",
            list_field="market_positions",
            fetch=lambda cursor: self.get_positions(
                ticker=ticker,
                settlement_status=settlement_status,
                cursor=cursor,
            ),
            page_delay=page_delay,
            filter_repr=f"ticker={ticker!r} settlement_status={settlement_status!r}",
        )

    async def iter_all_orders(
        self,
        *,
        ticker: str | None = None,
        status: str | None = None,
        page_delay: float = 1.0,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetch all pages of ``/portfolio/orders``.

        See :meth:`_iter_paginated` for the truncation / cursor-cycle
        contract.
        """
        return await self._iter_paginated(
            method_name="iter_all_orders",
            list_field="orders",
            fetch=lambda cursor: self.get_orders(
                ticker=ticker,
                status=status,
                cursor=cursor,
            ),
            page_delay=page_delay,
            filter_repr=f"ticker={ticker!r} status={status!r}",
        )


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------


_WS_CLOSE_TYPES = (
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSING,
    aiohttp.WSMsgType.CLOSED,
)


class KalshiWebSocket:
    """
    WebSocket client for real-time Kalshi market data.

    Handles auto-reconnect and heartbeat.

    Note: ``listen()`` may raise ``SubscribeFailed`` out of the async
    generator if the Kalshi subscribe ACK does not arrive within
    ``subscribe_timeout`` seconds; the orchestrator restarts the listener
    task on the next health-check tick.

    Usage::

        ws = KalshiWebSocket(api_key="...", private_key_path="key.pem")
        async for message in ws.listen(channels=["orderbook_delta"], market_tickers=["KXHIGH-..."]):
            process(message)
    """

    # Maximum seconds we wait for a subscribe ACK from Kalshi after sending
    # the subscribe frame. Kalshi typically replies in <1 s on a healthy
    # connection; 5 s gives generous headroom without exceeding the
    # orchestrator tick cadence.
    _DEFAULT_SUBSCRIBE_TIMEOUT: float = 5.0

    # Default cap on frames buffered while waiting for the subscribe ACK.
    # Orchestrator subscribes to ``orderbook_delta`` + ``ticker`` with no
    # ``market_tickers`` filter, so Kalshi flushes a snapshot for every
    # active market on subscribe — easily 100+ frames. 4096 frames @ ~500
    # bytes ≈ 2 MB; trivial memory cost, covers any plausible snapshot.
    # Hitting the cap means the protocol is misbehaving — fail closed.
    _DEFAULT_PENDING_FRAMES_CAP: int = 4096

    # Idle bound (seconds) on the main receive loop. If no *data* frame arrives
    # within this window the client reconnects. Must stay below the risk
    # STALE_DATA cutoff (300s) so recovery precedes a trading halt. Pong-immune:
    # enforced via asyncio.wait_for around receive(), NOT receive(timeout=)
    # (aiohttp resets a per-read timeout on every heartbeat pong — see listen()).
    _DEFAULT_IDLE_TIMEOUT: float = 90.0

    # Channels that Kalshi REQUIRES a non-empty market_tickers filter for.
    # Empirically verified against production: subscribing to
    # ``orderbook_delta`` with ``market_tickers=None`` returns
    # ``{"type":"error","msg":{"code":2,"msg":"Params required"}}``. The
    # docs claim the filter is optional for every channel, but the live
    # API disagrees for orderbook_delta (the volume on a global fan-out
    # would be enormous, so this is a sensible enforcement on Kalshi's
    # side). The ``ticker`` channel accepts a global subscribe.
    _REQUIRES_MARKET_TICKERS: frozenset[str] = frozenset({"orderbook_delta"})

    def __init__(
        self,
        api_key: str,
        private_key_path: str,
        ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2",
        reconnect_delay: float = 5.0,
        heartbeat_interval: float = 10.0,
        subscribe_timeout: float | None = None,
        pending_frames_cap: int | None = None,
        private_key_password: bytes | None = None,
        idle_timeout: float | None = None,
    ) -> None:
        self.api_key = api_key
        self._private_key = _load_private_key(private_key_path, password=private_key_password)
        self.ws_url = ws_url
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        self.idle_timeout = (
            idle_timeout if idle_timeout is not None else self._DEFAULT_IDLE_TIMEOUT
        )
        self.subscribe_timeout = (
            subscribe_timeout
            if subscribe_timeout is not None
            else self._DEFAULT_SUBSCRIBE_TIMEOUT
        )
        self.pending_frames_cap = (
            pending_frames_cap
            if pending_frames_cap is not None
            else self._DEFAULT_PENDING_FRAMES_CAP
        )
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._callbacks: list[tuple[Callable[[dict[str, Any]], Any], bool]] = []
        self._last_message_time: float = 0.0
        # Monotonic timestamp of the most recent SUCCESSFUL subscribe (ACK
        # received). Distinct from _last_message_time (data frames): stays fresh
        # on a quiet market that re-subscribes every idle cycle, but goes stale
        # when the pipe can't re-establish. Read by the orchestrator watchdog as
        # the pipe-health signal. None until the first successful subscribe.
        self._last_subscribe_ok_monotonic: float | None = None
        # Frames received while waiting for the subscribe ACK that are NOT
        # the ACK itself (e.g. an early ticker snapshot). Drained by
        # ``listen()`` before entering ``async for msg in ws``. Deque so
        # the FIFO drain is O(1); a Kalshi all-markets snapshot can fill
        # this with thousands of frames.
        self._pending_frames: deque[dict[str, Any]] = deque()

    @property
    def last_message_age(self) -> float:
        """Seconds since the last message was received. 0 if no messages yet."""
        if self._last_message_time == 0.0:
            return 0.0
        return time.monotonic() - self._last_message_time

    @property
    def last_subscribe_ok_age(self) -> float | None:
        """Seconds since the most recent successful subscribe, or None if a
        subscribe has never succeeded.

        Pipe-health signal for the orchestrator watchdog: a quiet market keeps
        this small (each idle-timeout reconnect re-subscribes), while a broken
        pipe lets it climb. Callers MUST treat None as pipe-UNHEALTHY (the
        cold-start / never-connected case), not as healthy.
        """
        if self._last_subscribe_ok_monotonic is None:
            return None
        return time.monotonic() - self._last_subscribe_ok_monotonic

    @property
    def received_any_message(self) -> bool:
        """True once any data/pending frame has been received on this client.

        Distinguishes "never received" from "just received", which
        ``last_message_age``'s 0.0 sentinel conflates — the orchestrator
        watchdog needs the distinction to handle the cold-start stall case.
        """
        return self._last_message_time > 0.0

    def on_message(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        """Register a callback for incoming messages."""
        is_async = asyncio.iscoroutinefunction(callback)
        self._callbacks.append((callback, is_async))

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        """Open an authenticated WebSocket connection."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                connector=_build_connector(),
            )

        # Build auth headers for the WebSocket handshake
        ts_ms = str(int(time.time() * 1000))
        # For WebSocket, sign against the ws path
        path = "/trade-api/ws/v2"
        sig = _sign_request(self._private_key, ts_ms, "GET", path)
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        }
        ws = await self._session.ws_connect(
            self.ws_url,
            headers=headers,
            heartbeat=self.heartbeat_interval,
        )
        logger.info("WebSocket connected to %s", self.ws_url)
        return ws

    async def _subscribe(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        channels: list[str],
        market_tickers: list[str] | None = None,
    ) -> None:
        """Send subscription commands and wait for Kalshi's ACK frame.

        ACK detection is **fail-open** on every match dimension to avoid the
        worse-than-the-bug failure mode of timing out on a successful
        subscribe just because Kalshi's protocol drifted from our
        assumptions. The principle is: only fail closed when we see an
        explicit error frame addressed to our request_id; otherwise be
        permissive about ACK shape.

        The acceptance rules, in order:
          * Error frame with ``id == request_id``: raise ``SubscribeFailed``.
          * Error frame with mismatched/missing id: log WARNING, keep
            waiting (stray error from a different request, not our reject).
          * TEXT frame with ``id is None``: log a one-time WARNING and
            ACCEPT as the ACK (protocol may have drifted to drop ids).
          * TEXT frame with ``id == request_id`` and ``type != "error"``:
            ACCEPT as the ACK.
          * TEXT frame with ``id != request_id``: buffer for ``listen()``
            to replay (likely an early snapshot frame).

        Raises:
            SubscribeFailed: Kalshi returned an error frame for our id, the
                ACK timeout elapsed, the socket closed before the ACK
                arrived, or the pre-ACK frame buffer overflowed. **Not**
                caught by ``listen``: it propagates out of the generator
                and kills ``_ws_task`` so the orchestrator's existing
                ``_check_ws_health`` Telegram-alert path fires.

        Side effect: any TEXT frame received while waiting for the ACK that
        is **not** the ACK itself (e.g. an early ticker snapshot Kalshi
        flushes immediately on subscribe) is appended to
        ``self._pending_frames`` so ``listen()`` can yield it before
        consuming new frames.
        """
        # Channels that REQUIRE a non-empty market_tickers filter. Kalshi
        # rejects orderbook_delta subscribes without one with
        # ``{"type":"error","msg":{"code":2,"msg":"Params required"}}``;
        # pre-A3 this rejection was silently consumed by the listen loop
        # because there was no ACK reader, so the bug had been live since
        # Phase 2 without surfacing. We now fail loudly at construction
        # time so the bad shape never reaches the wire.
        #
        # Empirically verified channels that REJECT a global subscribe:
        #   - orderbook_delta
        # Channels that ACCEPT a global subscribe (all-markets fan-out):
        #   - ticker (and the other lifecycle channels per the docs).
        # Add to ``_REQUIRES_MARKET_TICKERS`` if Kalshi tightens additional
        # channels in future. The fail-mode is a clear ValueError, not a
        # Kalshi error round-trip, so future regressions are caught at
        # the unit-test boundary rather than in production logs.
        if any(ch in self._REQUIRES_MARKET_TICKERS for ch in channels) and not market_tickers:
            raise ValueError(
                "Kalshi requires a non-empty market_tickers filter for "
                f"channels={channels!r} (got market_tickers={market_tickers!r}); "
                "Kalshi rejects with code=2 'Params required' otherwise. "
                "Either add a market_tickers list, or remove "
                f"{sorted(set(channels) & self._REQUIRES_MARKET_TICKERS)} from channels."
            )

        request_id = 1
        cmd: dict[str, Any] = {
            "id": request_id,
            "cmd": "subscribe",
            "params": {"channels": channels},
        }
        if market_tickers:
            cmd["params"]["market_tickers"] = market_tickers
        await ws.send_json(cmd)
        logger.debug(
            "Subscribe sent: channels=%s tickers=%s", channels, market_tickers
        )

        # Drain frames until we see an ACK (id-less or id == request_id with
        # type != "error") or an explicit error for our id, or the deadline
        # elapses. We wrap receive() in asyncio.wait_for rather than using its
        # native ``timeout=`` because aiohttp resets the per-read timeout on
        # every heartbeat pong (PING/PONG are consumed inside receive() via an
        # internal ``continue``), so ``receive(timeout=remaining)`` would never
        # honor the deadline on a pong-alive but ACK-silent stream. wait_for
        # cancels the whole coroutine, enforcing the wall-clock deadline.
        # CONTINUATION frames are protocol-legal but Kalshi does not fragment.
        deadline = time.monotonic() + self.subscribe_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SubscribeFailed(
                    f"subscribe ACK timeout after {self.subscribe_timeout:.1f}s "
                    f"(channels={channels}, tickers={market_tickers})"
                )
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except TimeoutError as exc:
                raise SubscribeFailed(
                    f"subscribe ACK timeout after {self.subscribe_timeout:.1f}s "
                    f"(channels={channels}, tickers={market_tickers})"
                ) from exc

            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.ERROR,
            ):
                raise SubscribeFailed(
                    f"connection closed during subscribe (msg.type={msg.type!r}, "
                    f"data={msg.data!r})"
                )
            if msg.type != aiohttp.WSMsgType.TEXT:
                # Binary / continuation — ignore and keep waiting.
                continue

            try:
                data = json.loads(msg.data)
            except (TypeError, ValueError):
                logger.warning("Non-JSON frame during subscribe: %r", msg.data)
                continue

            msg_id = data.get("id")
            msg_type = data.get("type")

            # Error-frame branch: only treat as our subscribe rejection if
            # the id matches. A stray error from another request gets a
            # WARNING and we keep waiting (fail-open on id correlation —
            # we already know this is an error frame, so we use the strict
            # rule only here to avoid mis-attributing it).
            if msg_type == "error":
                if msg_id == request_id:
                    err_body = data.get("msg") or data.get("error") or {}
                    code = (
                        err_body.get("code")
                        if isinstance(err_body, dict)
                        else None
                    )
                    err_msg = (
                        err_body.get("msg")
                        if isinstance(err_body, dict)
                        else str(err_body)
                    ) or "unknown error"
                    raise SubscribeFailed(
                        f"Kalshi rejected subscribe: code={code} msg={err_msg!r} "
                        f"(channels={channels}, tickers={market_tickers})",
                        code=code,
                    )
                logger.warning(
                    "Received error frame with id=%r (expected %r) during subscribe: %r",
                    msg_id,
                    request_id,
                    data,
                )
                continue

            # Fail-open id correlation: an id-less non-error frame is
            # accepted as the ACK with a one-time WARNING. We have no
            # captured Kalshi fixture and cannot afford the
            # worse-than-the-bug failure mode of timing out on a
            # successful subscribe just because Kalshi dropped the echoed
            # id field.
            if msg_id is None:
                logger.warning(
                    "Accepted id-less ACK frame during subscribe — "
                    "protocol may have drifted: %r",
                    data,
                )
                sid = None
                if isinstance(data.get("msg"), dict):
                    sid = data["msg"].get("sid")
                logger.info(
                    "Subscribed: channels=%s tickers=%s type=%s sid=%s",
                    channels,
                    market_tickers,
                    msg_type,
                    sid,
                )
                return

            if msg_id == request_id:
                # Fail-open ACK match: any non-error frame echoing our id
                # is accepted as the ACK. The listener loop owns
                # ws.receive() from here.
                sid = None
                if isinstance(data.get("msg"), dict):
                    sid = data["msg"].get("sid")
                logger.info(
                    "Subscribed: channels=%s tickers=%s type=%s sid=%s",
                    channels,
                    market_tickers,
                    msg_type,
                    sid,
                )
                return

            # id present but doesn't match: a frame from a prior or
            # different request, or an unsolicited server push. Buffer for
            # the listener to replay — it might be a snapshot.
            if len(self._pending_frames) >= self.pending_frames_cap:
                raise SubscribeFailed(
                    "snapshot buffer overflow during subscribe "
                    f"(cap={self.pending_frames_cap}, channels={channels}, "
                    f"tickers={market_tickers})"
                )
            self._pending_frames.append(data)

    async def listen(
        self,
        channels: list[str],
        market_tickers: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yield messages from the WebSocket, reconnecting on transport errors.

        This is an infinite async generator — use ``async for`` or cancel
        the task to stop.

        Reconnect is handled internally for:
          * Network/transport errors (``aiohttp.ClientError``,
            ``ConnectionError``, ``OSError``).
          * Server-initiated CLOSE/ERROR frames during the listen loop.

        ``SubscribeFailed`` is **not** caught here — it propagates out and
        terminates the generator. The orchestrator's ``_check_ws_health``
        increments ``_ws_restart_count`` and, after ``_WS_ALERT_THRESHOLD``
        consecutive restarts, fires a CRITICAL Telegram alert via the
        existing path. This routes user-facing observability through the
        single existing alert wiring rather than building a parallel one.
        """
        self._running = True
        reconnect_attempts = 0
        while self._running:
            try:
                ws = await self._connect()
                self._ws = ws
                # Reset the pending-frames buffer for this new connection.
                self._pending_frames.clear()
                await self._subscribe(ws, channels, market_tickers)
                reconnect_attempts = 0
                self._last_subscribe_ok_monotonic = time.monotonic()

                # Forward any frames the subscribe path buffered before the
                # ACK arrived, so callbacks and consumers see them in order.
                while self._pending_frames:
                    data = self._pending_frames.popleft()
                    self._last_message_time = time.monotonic()
                    for cb, is_async in self._callbacks:
                        if is_async:
                            await cb(data)
                        else:
                            cb(data)
                    yield data

                # Explicit receive loop (not ``async for msg in ws``) so we can
                # bound the idle wait. ``asyncio.wait_for`` cancels the WHOLE
                # ``receive()`` coroutine at wall-clock ``idle_timeout`` — unlike
                # ``receive(timeout=)``, whose per-read timeout aiohttp resets on
                # every heartbeat pong, so it would never fire on a half-open
                # (pong-alive, data-silent) stream. Breaking falls through to the
                # reconnect block below, which re-subscribes and gets a fresh
                # snapshot, restoring ``_last_message_time`` freshness.
                while True:
                    try:
                        msg = await asyncio.wait_for(
                            ws.receive(), timeout=self.idle_timeout
                        )
                    except TimeoutError:
                        logger.warning(
                            "WebSocket idle for %.0fs (no data frames received) — "
                            "reconnecting",
                            self.idle_timeout,
                        )
                        break
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        self._last_message_time = time.monotonic()
                        data = json.loads(msg.data)
                        for cb, is_async in self._callbacks:
                            if is_async:
                                await cb(data)
                            else:
                                cb(data)
                        yield data
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.warning("WebSocket error frame: %s", msg)
                        break
                    elif msg.type in _WS_CLOSE_TYPES:
                        logger.info("WebSocket closed by remote")
                        break
                    # else (BINARY / other): ignore and keep listening

            except (aiohttp.ClientError, ConnectionError, OSError) as exc:
                logger.warning("WebSocket connection error: %s", exc)
            except asyncio.CancelledError:
                self._running = False
                raise
            # SubscribeFailed intentionally NOT caught — it propagates out
            # of the generator so the orchestrator's _ws_restart_count path
            # fires its CRITICAL Telegram alert.
            finally:
                # Close the previous WebSocket to prevent resource leaks
                if self._ws and not self._ws.closed:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass  # best-effort cleanup

            if self._running:
                reconnect_attempts += 1
                delay = _compute_backoff(
                    reconnect_attempts,
                    self.reconnect_delay,
                    max_delay=120.0,
                )
                logger.info(
                    "Reconnecting in %.1fs (attempt %d)...",
                    delay,
                    reconnect_attempts,
                )
                await asyncio.sleep(delay)

    async def close(self) -> None:
        """Gracefully shut down the WebSocket."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an error response."""

    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status = status
        self.body = body
        message = body.get("message", body.get("error", str(body)))
        super().__init__(f"Kalshi API {status}: {message}")


class SubscribeFailed(Exception):  # noqa: N818
    """Raised when a Kalshi WebSocket subscribe does not receive a valid ACK.

    Distinguishes four failure modes:
      * Kalshi returned an error frame (auth, invalid channel, rate limit, ...).
      * The ACK timeout elapsed with no response.
      * The socket closed mid-subscribe.
      * The pre-ACK frame buffer overflowed (Kalshi flushed too many
        snapshot frames before ACKing).

    This exception is **not** caught inside ``KalshiWebSocket.listen``; it
    propagates out of the async generator, terminating the listener task.
    The orchestrator's ``_check_ws_health`` then restarts the task and, after
    ``_WS_ALERT_THRESHOLD`` consecutive restarts, fires a CRITICAL Telegram
    alert. This is intentional: it reuses the existing user-facing alert
    path rather than building a parallel one inside the WS class.
    """

    def __init__(self, reason: str, *, code: int | None = None) -> None:
        self.reason = reason
        self.code = code
        super().__init__(reason)


class OrderValidationError(ValueError):
    """Raised when place_order receives invalid parameters before sending to the API."""
