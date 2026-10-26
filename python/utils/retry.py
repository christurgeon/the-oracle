"""
Shared async retry utility for HTTP requests.

Provides a configurable decorator / context helper for retrying async HTTP
calls with exponential backoff, ``Retry-After`` header support, and
customizable retryable conditions.

Usage with httpx::

    from python.utils.retry import async_retry_request

    response = await async_retry_request(
        lambda: client.get(url, params=params),
        max_retries=3,
        retry_delays=[1.0, 2.0, 4.0],
        retryable_statuses={429, 500, 502, 503},
        error_label="Open-Meteo",
    )

The helper raises ``RetryExhaustedError`` (wrapping the last exception) when
all attempts are exhausted.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _jittered(delay: float, jitter: float) -> float:
    """Add up to ``jitter`` fraction of *delay* to *delay* (full-jitter upward).

    ``jitter=0.0`` (default) returns *delay* unchanged. ``jitter=0.5`` returns a
    value uniformly in ``[delay, delay*1.5]``. Jitter only ever *extends* the
    wait so it never violates a server ``Retry-After`` directive, while
    de-synchronizing retry storms across a concurrent fan-out.
    """
    if jitter <= 0.0 or delay <= 0.0:
        return delay
    return delay + delay * jitter * random.random()


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None) -> None:
        super().__init__(message)
        self.last_exception = last_exception


def _get_delay(
    attempt: int,
    retry_delays: list[float],
    response: httpx.Response | None = None,
    respect_retry_after: bool = True,
) -> float:
    """Compute the delay before the next retry attempt.

    If ``respect_retry_after`` is True and the response contains a
    ``Retry-After`` header with a numeric value, that value is used
    (capped at 120 s).  Otherwise falls back to the pre-configured
    delay sequence.
    """
    if respect_retry_after and response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass

    if attempt < len(retry_delays):
        return retry_delays[attempt]
    # Fall back to the last configured delay when attempt exceeds the list.
    return retry_delays[-1] if retry_delays else 1.0


async def async_retry_request(
    request_fn: Callable[[], Awaitable[httpx.Response]],
    *,
    max_retries: int = 3,
    retry_delays: list[float] | None = None,
    retryable_statuses: set[int] | frozenset[int] | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (httpx.RequestError,),
    respect_retry_after: bool = True,
    error_label: str = "HTTP",
    on_retry: Callable[[int, int, float, str], Any] | None = None,
    jitter: float = 0.0,
    max_elapsed_seconds: float | None = None,
) -> httpx.Response:
    """Execute *request_fn* with retry on transient failures.

    Args:
        request_fn: A zero-argument async callable that returns an
            ``httpx.Response``.  Called once per attempt.
        max_retries: Maximum number of *retries* (total attempts =
            ``max_retries + 1``).
        retry_delays: Pre-configured delay sequence in seconds.  Element
            ``i`` is used for attempt ``i``.  If the list is shorter than
            ``max_retries``, the last element is repeated.  Defaults to
            ``[1.0, 2.0, 4.0]``.
        retryable_statuses: HTTP status codes that should trigger a retry
            (e.g. ``{429, 500, 502, 503}``).  Defaults to
            ``{429, 500, 502, 503}``.
        retryable_exceptions: Exception types (beyond HTTP status errors)
            that should trigger a retry.  Defaults to
            ``(httpx.RequestError,)``.
        respect_retry_after: When True, honour the ``Retry-After`` response
            header (numeric values only, capped at 120 s).
        error_label: Human-readable label used in log messages and error
            text (e.g. ``"Open-Meteo"``, ``"NWS API"``).
        on_retry: Optional callback invoked before each retry sleep.
            Receives ``(attempt, max_retries, delay, reason)`` where
            *attempt* is the 1-based retry number and *reason* is a short
            description of why the retry was triggered.
        jitter: Fraction of the computed delay to add as upward random jitter
            (``0.0`` = disabled). De-synchronizes retry storms across a
            concurrent fan-out so retries don't re-burst in lockstep.
        max_elapsed_seconds: Hard ceiling on cumulative retry sleep time per
            request (``None`` = unbounded). When the next sleep would exceed
            this budget, the request gives up immediately and raises
            ``RetryExhaustedError`` rather than sleeping past the bound — this
            stops a string of large ``Retry-After`` values from letting one
            request run away and starve a watchdog/deadline.

    Returns:
        The successful ``httpx.Response``.

    Raises:
        RetryExhaustedError: When all attempts are exhausted.  The
            ``last_exception`` attribute holds the final exception.
    """
    if retry_delays is None:
        retry_delays = [1.0, 2.0, 4.0]
    if retryable_statuses is None:
        retryable_statuses = {429, 500, 502, 503}

    last_exc: Exception | None = None
    elapsed = 0.0  # cumulative retry sleep, for the max_elapsed_seconds budget

    for attempt in range(max_retries + 1):
        try:
            response = await request_fn()
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            if status in retryable_statuses and attempt < max_retries:
                delay = _jittered(
                    _get_delay(attempt, retry_delays, exc.response, respect_retry_after),
                    jitter,
                )
                if max_elapsed_seconds is not None and elapsed + delay > max_elapsed_seconds:
                    raise RetryExhaustedError(
                        f"{error_label} returned {status} and retry budget "
                        f"({max_elapsed_seconds:.1f}s) exhausted: "
                        f"{exc.response.text[:500]}",
                        last_exception=exc,
                    ) from exc
                reason = f"{error_label} {status}"
                if on_retry is not None:
                    on_retry(attempt + 1, max_retries, delay, reason)
                logger.warning(
                    "%s %d — retry %d/%d in %.1fs",
                    error_label, status, attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
                elapsed += delay
                continue
            raise RetryExhaustedError(
                f"{error_label} returned {status}: "
                f"{exc.response.text[:500]}",
                last_exception=exc,
            ) from exc
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = _jittered(_get_delay(attempt, retry_delays), jitter)
                if max_elapsed_seconds is not None and elapsed + delay > max_elapsed_seconds:
                    raise RetryExhaustedError(
                        f"{error_label} request failed and retry budget "
                        f"({max_elapsed_seconds:.1f}s) exhausted: {exc}",
                        last_exception=exc,
                    ) from exc
                reason = f"{error_label} {type(exc).__name__}"
                if on_retry is not None:
                    on_retry(attempt + 1, max_retries, delay, reason)
                logger.warning(
                    "%s %s — retry %d/%d in %.1fs",
                    error_label, type(exc).__name__,
                    attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
                elapsed += delay
                continue
            raise RetryExhaustedError(
                f"{error_label} request failed: {exc}",
                last_exception=exc,
            ) from exc

    # Unreachable — the loop always raises or returns — but satisfy mypy.
    raise RetryExhaustedError(  # pragma: no cover
        f"{error_label} failed after {max_retries} retries",
        last_exception=last_exc,
    )


# Shared defaults for the GET-with-retry helper.
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    error_label: str,
    retryable_statuses: set[int] | frozenset[int],
    exc_type: type[Exception],
    params: dict[str, Any] | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
    jitter: float = 0.0,
    max_elapsed_seconds: float | None = None,
) -> Any:
    """GET *url* with retry, parse JSON, and wrap exhaustion in *exc_type*.

    Centralizes the ``async_retry_request(...) → .json()`` pattern with a
    module-specific error type. On retry exhaustion, raises
    ``exc_type(str(exc))`` chained from the last underlying exception —
    matching the historical per-module ``*APIError`` behavior.

    Args:
        client: An open ``httpx.AsyncClient``.
        url: The URL to GET.
        error_label: Human-readable label for log/error text.
        retryable_statuses: HTTP status codes that trigger a retry. Threaded
            through explicitly — callers must pass their own set (Open-Meteo
            and NWS deliberately differ).
        exc_type: Exception class raised on retry exhaustion.
        params: Optional query parameters.
        max_retries: Maximum retries (defaults to 3).
        retry_delays: Backoff delay sequence (defaults to ``(1.0, 2.0, 4.0)``).
        jitter: Upward backoff jitter fraction (``0.0`` = disabled). Forwarded
            to ``async_retry_request``.
        max_elapsed_seconds: Hard ceiling on cumulative retry sleep per request
            (``None`` = unbounded). Forwarded to ``async_retry_request``.

    Returns:
        The parsed JSON body.
    """
    try:
        response = await async_retry_request(
            lambda: client.get(url, params=params),
            max_retries=max_retries,
            retry_delays=list(retry_delays),
            retryable_statuses=retryable_statuses,
            error_label=error_label,
            jitter=jitter,
            max_elapsed_seconds=max_elapsed_seconds,
        )
    except RetryExhaustedError as exc:
        raise exc_type(str(exc)) from exc.last_exception
    return response.json()
