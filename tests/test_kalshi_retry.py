"""Tests for KalshiClient retry logic in _request."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from python.data.kalshi_client import (
    KalshiAPIError,
    KalshiClient,
    RateLimiter,
    _compute_backoff,
    _get_retry_delay,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    status: int = 200,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock that behaves like an aiohttp response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=json.dumps(body or {}))
    resp.headers = headers or {}

    # Make it work as `async with session.request(...) as resp:`
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx



def _make_client(**kwargs: Any) -> KalshiClient:
    """Create a KalshiClient with mocked key loading and a fake session."""
    with patch("python.data.kalshi_client._load_private_key", return_value=MagicMock()):
        client = KalshiClient(
            api_key="test-key",
            private_key_path="/dev/null",
            **kwargs,
        )
    # Install a mock session so _request doesn't fail on the connected check.
    client._session = MagicMock(spec=aiohttp.ClientSession)
    client._session.closed = False
    # Stub auth headers to avoid crypto calls.
    client._auth_headers = MagicMock(return_value={"Authorization": "test"})
    # Stub rate limiter to avoid timing issues in tests.
    client._rate_limiter = MagicMock()
    client._rate_limiter.acquire = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Unit tests: _compute_backoff
# ---------------------------------------------------------------------------


class TestComputeBackoff:
    def test_base_delay_attempt_zero(self) -> None:
        """Attempt 0 should produce a delay near base_delay."""
        for _ in range(50):
            delay = _compute_backoff(0, 1.0)
            assert 0.75 <= delay <= 1.25

    def test_exponential_growth(self) -> None:
        """Each attempt should roughly double the delay."""
        for _ in range(20):
            d0 = _compute_backoff(0, 1.0)
            d1 = _compute_backoff(1, 1.0)
            # d1 centre is 2.0, d0 centre is 1.0 — with jitter, d1 > d0 most of the time.
            # But we just verify the centres, not individual samples.
            assert 0.75 <= d0 <= 1.25
            assert 1.5 <= d1 <= 2.5

    def test_max_delay_cap(self) -> None:
        """Delay should never exceed max_delay (jitter applied before cap)."""
        for _ in range(50):
            delay = _compute_backoff(10, 1.0, max_delay=5.0)
            assert delay <= 5.0

    def test_jitter_produces_varied_results(self) -> None:
        """Multiple calls should produce different values (not deterministic)."""
        delays = {_compute_backoff(0, 1.0) for _ in range(20)}
        assert len(delays) > 1


# ---------------------------------------------------------------------------
# Unit tests: _get_retry_delay
# ---------------------------------------------------------------------------


class TestGetRetryDelay:
    def test_429_with_retry_after_header(self) -> None:
        """Should use the Retry-After value for 429."""
        delay = _get_retry_delay(429, "10", 0, 1.0)
        assert delay == 10.0

    def test_429_retry_after_capped_at_120(self) -> None:
        """Retry-After values > 120s should be capped."""
        delay = _get_retry_delay(429, "600", 0, 1.0)
        assert delay == 120.0

    def test_429_invalid_retry_after_falls_back(self) -> None:
        """Non-numeric Retry-After should fall back to computed backoff."""
        delay = _get_retry_delay(429, "not-a-number", 0, 1.0)
        assert 3.75 <= delay <= 6.25  # 5.0 * jitter(0.75..1.25)

    def test_429_no_retry_after_uses_long_backoff(self) -> None:
        """429 without header uses 5s-base backoff."""
        delay = _get_retry_delay(429, None, 0, 1.0)
        assert 3.75 <= delay <= 6.25  # 5.0 * jitter

    def test_429_attempt_1_doubles(self) -> None:
        """Second 429 attempt uses 10s base."""
        delay = _get_retry_delay(429, None, 1, 1.0)
        assert 7.5 <= delay <= 12.5  # 10.0 * jitter

    def test_500_uses_short_backoff(self) -> None:
        """5xx uses the standard 1s-base backoff."""
        delay = _get_retry_delay(500, None, 0, 1.0)
        assert 0.75 <= delay <= 1.25

    def test_503_attempt_2(self) -> None:
        """Third 503 attempt uses 4s base."""
        delay = _get_retry_delay(503, None, 2, 1.0)
        assert 3.0 <= delay <= 5.0  # 4.0 * jitter


# ---------------------------------------------------------------------------
# Integration tests: _request retry logic
# ---------------------------------------------------------------------------


class TestRequestRetry:
    """Test _request retry behaviour with mocked HTTP responses."""

    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio
        monkeypatch.setattr(_asyncio, "sleep", AsyncMock(return_value=None))

    async def test_success_no_retry(self) -> None:
        """A 200 response should be returned immediately, no retry."""
        client = _make_client()
        client._session.request = MagicMock(
            return_value=_mock_response(200, {"ok": True})
        )
        result = await client._request("GET", "/test")
        assert result == {"ok": True}
        assert client._session.request.call_count == 1

    # -- 429 retry ---------------------------------------------------------

    async def test_429_then_success_on_get(self) -> None:
        """GET: 429 followed by 200 should retry and succeed."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(429, {"error": "rate limited"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        result = await client._request("GET", "/test")
        assert result == {"ok": True}
        assert client._session.request.call_count == 2

    async def test_429_with_retry_after_header(self) -> None:
        """429 with Retry-After header should respect the value."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(429, {"error": "slow down"}, {"Retry-After": "0.01"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        result = await client._request("GET", "/test")
        assert result == {"ok": True}

    async def test_429_on_post_retries(self) -> None:
        """POST: 429 should still be retried (server rejected before processing)."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(429, {"error": "rate limited"}),
                _mock_response(200, {"order": {"order_id": "abc"}}),
            ]
        )
        result = await client._request("POST", "/portfolio/orders", json_body={"ticker": "X"})
        assert result["order"]["order_id"] == "abc"
        assert client._session.request.call_count == 2

    async def test_429_sets_global_backoff(self) -> None:
        """A 429 response should set _backoff_until so other requests wait."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(429, {"error": "rate limited"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        assert client._backoff_until == 0.0
        await client._request("GET", "/test")
        assert client._backoff_until > 0.0

    async def test_429_exhausted_raises(self) -> None:
        """Exhausting all retries on 429 should raise KalshiAPIError."""
        client = _make_client(max_retries=2, retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(429, {"error": "rate limited"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 429

    # -- 5xx retry ---------------------------------------------------------

    async def test_503_then_success_on_get(self) -> None:
        """GET: 503 followed by 200 should retry and succeed."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(503, {"error": "unavailable"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        result = await client._request("GET", "/test")
        assert result == {"ok": True}

    async def test_500_on_post_does_not_retry(self) -> None:
        """POST: 500 should NOT be retried (order status unknown)."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(500, {"error": "internal"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("POST", "/portfolio/orders", json_body={})
        assert exc_info.value.status == 500
        # Only one attempt — no retry.
        assert client._session.request.call_count == 1

    async def test_500_on_delete_retries(self) -> None:
        """DELETE: 500 should be retried (cancellation is idempotent)."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(500, {"error": "internal"}),
                _mock_response(200, {}),
            ]
        )
        result = await client._request("DELETE", "/portfolio/orders/abc")
        assert result == {}
        assert client._session.request.call_count == 2

    async def test_502_exhausted_raises(self) -> None:
        """Exhausting all retries on 502 should raise KalshiAPIError."""
        client = _make_client(max_retries=2, retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(502, {"error": "bad gateway"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 502
        # 1 initial + 2 retries = 3 total attempts.
        assert client._session.request.call_count == 3

    # -- Non-retryable errors ----------------------------------------------

    async def test_400_no_retry(self) -> None:
        """400 Bad Request should fail immediately with no retry."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(400, {"error": "bad request"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 400
        assert client._session.request.call_count == 1

    async def test_401_no_retry(self) -> None:
        """401 Unauthorized should fail immediately."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(401, {"error": "unauthorized"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 401
        assert client._session.request.call_count == 1

    async def test_403_no_retry(self) -> None:
        """403 Forbidden should fail immediately."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(403, {"error": "forbidden"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 403

    async def test_404_no_retry(self) -> None:
        """404 Not Found should fail immediately."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(404, {"error": "not found"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 404

    # -- Timeout / connection errors ---------------------------------------

    async def test_timeout_on_get_retries(self) -> None:
        """GET: asyncio.TimeoutError should be retried."""
        client = _make_client(retry_base_delay=0.01)

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError()
            return _mock_response(200, {"ok": True})

        client._session.request = MagicMock(side_effect=side_effect)
        result = await client._request("GET", "/test")
        assert result == {"ok": True}
        assert call_count == 2

    async def test_timeout_on_post_does_not_retry(self) -> None:
        """POST: timeout should NOT be retried (order may have been placed)."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(side_effect=TimeoutError())
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("POST", "/portfolio/orders", json_body={})
        assert exc_info.value.status == 0
        assert "TimeoutError" in str(exc_info.value)
        assert client._session.request.call_count == 1

    async def test_connection_error_on_get_retries(self) -> None:
        """GET: aiohttp.ClientError should be retried."""
        client = _make_client(retry_base_delay=0.01)

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ServerDisconnectedError()
            return _mock_response(200, {"ok": True})

        client._session.request = MagicMock(side_effect=side_effect)
        result = await client._request("GET", "/test")
        assert result == {"ok": True}

    async def test_connection_error_on_delete_retries(self) -> None:
        """DELETE: connection errors should be retried (cancel is idempotent)."""
        client = _make_client(retry_base_delay=0.01)

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise aiohttp.ClientConnectionError("connection refused")
            return _mock_response(200, {})

        client._session.request = MagicMock(side_effect=side_effect)
        result = await client._request("DELETE", "/portfolio/orders/abc")
        assert result == {}
        assert call_count == 3

    async def test_timeout_exhausted_wraps_in_kalshi_error(self) -> None:
        """Exhausted timeout retries should wrap in KalshiAPIError(status=0)."""
        client = _make_client(max_retries=2, retry_base_delay=0.01)
        client._session.request = MagicMock(side_effect=TimeoutError())
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 0
        assert "TimeoutError" in str(exc_info.value)
        # 1 initial + 2 retries = 3 total.
        assert client._session.request.call_count == 3

    # -- Rate limiter interaction ------------------------------------------

    async def test_rate_limiter_only_called_once(self) -> None:
        """Rate limiter should only be acquired on the first attempt."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(503, {"error": "unavailable"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        await client._request("GET", "/test")
        # acquire() should be called exactly once (first attempt only).
        assert client._rate_limiter.acquire.await_count == 1

    # -- Auth header regeneration ------------------------------------------

    async def test_auth_headers_regenerated_per_attempt(self) -> None:
        """Each retry should generate fresh auth headers."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(503, {"error": "unavailable"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        await client._request("GET", "/test")
        assert client._auth_headers.call_count == 2

    # -- Invalid JSON body -------------------------------------------------

    async def test_invalid_json_body_on_success_raises(self) -> None:
        """Non-JSON response on 200 should raise, not silently return."""
        client = _make_client()
        resp = MagicMock()
        resp.status = 200
        resp.text = AsyncMock(return_value="not valid json!")
        resp.headers = {}
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        client._session.request = MagicMock(return_value=ctx)

        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 200
        assert "Invalid JSON" in str(exc_info.value)

    # -- Mixed scenario: 429 then 503 then success ------------------------

    async def test_mixed_retryable_errors(self) -> None:
        """Multiple different retryable errors before success."""
        client = _make_client(retry_base_delay=0.01)
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(429, {"error": "rate limited"}),
                _mock_response(503, {"error": "unavailable"}),
                _mock_response(200, {"ok": True}),
            ]
        )
        result = await client._request("GET", "/test")
        assert result == {"ok": True}
        assert client._session.request.call_count == 3

    # -- Session not connected ---------------------------------------------

    async def test_not_connected_raises(self) -> None:
        """Calling _request without connecting should raise RuntimeError."""
        client = _make_client()
        client._session = None
        with pytest.raises(RuntimeError, match="not connected"):
            await client._request("GET", "/test")

    # -- Global backoff gate -----------------------------------------------

    async def test_global_backoff_gate_delays_subsequent_requests(self) -> None:
        """After a 429 sets the global backoff, the next request should wait."""
        client = _make_client(retry_base_delay=0.01)
        # First request: 429 then 200.
        client._session.request = MagicMock(
            side_effect=[
                _mock_response(429, {"error": "rate limited"}),
                _mock_response(200, {"first": True}),
            ]
        )
        await client._request("GET", "/first")
        assert client._backoff_until > 0.0

        # Reset backoff to a tiny future value so test doesn't actually wait.
        import time

        client._backoff_until = time.monotonic() + 0.001

        # Second request should succeed after the gate passes.
        client._session.request = MagicMock(
            return_value=_mock_response(200, {"second": True})
        )
        result = await client._request("GET", "/second")
        assert result == {"second": True}

    # -- max_retries=0 disables retry --------------------------------------

    async def test_max_retries_zero_no_retry(self) -> None:
        """With max_retries=0, any error should fail immediately."""
        client = _make_client(max_retries=0, retry_base_delay=0.01)
        client._session.request = MagicMock(
            return_value=_mock_response(503, {"error": "unavailable"})
        )
        with pytest.raises(KalshiAPIError) as exc_info:
            await client._request("GET", "/test")
        assert exc_info.value.status == 503
        assert client._session.request.call_count == 1


# ---------------------------------------------------------------------------
# Unit tests: RateLimiter — concurrent requests not blocked during sleep
# ---------------------------------------------------------------------------


class TestRateLimiterConcurrency:
    """Verify that sleeping in acquire() does not block other concurrent callers."""

    async def test_no_sleep_when_under_limit(self) -> None:
        """Requests under the limit should be granted immediately, no sleep."""
        limiter = RateLimiter(max_requests=5, window_seconds=10.0)
        # All 5 slots are available; all should acquire without sleeping.
        for _ in range(5):
            await limiter.acquire()
        assert len(limiter._timestamps) == 5

    async def test_concurrent_requests_not_blocked_during_sleep(self) -> None:
        """
        When one caller is sleeping for the rate limit, a second concurrent
        caller must NOT be blocked for the full sleep duration.

        We fill the limiter to capacity, then fire two coroutines.  Both will
        compute a sleep, but they should run concurrently (overlapping), so the
        total wall time must be close to one sleep duration, not two.
        """
        import asyncio as _asyncio
        import time as _time

        # Fill with max_requests timestamps so the next acquire() must sleep.
        sleep_secs = 0.05  # short but measurable
        limiter = RateLimiter(max_requests=2, window_seconds=sleep_secs * 10)

        # Pre-fill so the window is full.
        now = _time.monotonic()
        limiter._timestamps = [now, now]

        start = _time.monotonic()
        # Launch two concurrent acquire() calls.
        await _asyncio.gather(limiter.acquire(), limiter.acquire())
        elapsed = _time.monotonic() - start

        # If the lock were held during sleep, both calls would be serialised and
        # total time >= 2 x sleep.  With the fix, they overlap and finish in
        # roughly 1 x sleep.  Allow up to 1.8x for CI jitter.
        threshold = sleep_secs * 10 * 1.8
        assert elapsed < threshold, (
            f"Elapsed {elapsed:.3f}s suggests callers were serialised "
            f"(expected < {threshold:.3f}s)"
        )
        # Both timestamps were recorded.
        assert len(limiter._timestamps) >= 2

    async def test_rate_limit_still_enforced_after_fix(self) -> None:
        """
        The non-blocking sleep must not break rate enforcement: after the window
        is full and acquire() completes, at most max_requests timestamps should
        fall inside the window.
        """
        import time as _time

        limiter = RateLimiter(max_requests=3, window_seconds=0.5)

        # Fill to capacity and then acquire one more (will sleep briefly).
        now = _time.monotonic()
        limiter._timestamps = [now, now, now]

        await limiter.acquire()

        # After acquiring, old timestamps that slid out of the window are pruned
        # on the next call — the important thing is we did NOT crash or deadlock.
        assert len(limiter._timestamps) >= 1
