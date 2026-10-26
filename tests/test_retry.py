"""
Tests for python/utils/retry.py — shared async retry utility.

Covers:
- Immediate success (no retries needed)
- Retry on retryable HTTP status codes (429, 500, 502, 503)
- Retry on network errors (httpx.RequestError)
- Non-retryable status codes raise immediately
- Max retries exhausted raises RetryExhaustedError
- Retry-After header is respected
- Custom delay sequences
- Custom retryable statuses
- Custom retryable exception types
- on_retry callback is invoked
- Delay falls back to last element when attempts exceed delay list
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from python.utils.retry import RetryExhaustedError, _get_delay, async_retry_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status: int = 200, text: str = "OK", headers: dict | None = None) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = text
    resp.headers = headers or {}
    resp.json.return_value = {"status": "ok"}
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status}",
            request=MagicMock(spec=httpx.Request),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# _get_delay tests
# ---------------------------------------------------------------------------


class TestGetDelay:
    """Unit tests for the _get_delay helper."""

    def test_uses_delay_sequence(self) -> None:
        delays = [1.0, 2.0, 4.0]
        assert _get_delay(0, delays) == 1.0
        assert _get_delay(1, delays) == 2.0
        assert _get_delay(2, delays) == 4.0

    def test_falls_back_to_last_delay(self) -> None:
        delays = [1.0, 2.0]
        assert _get_delay(5, delays) == 2.0

    def test_retry_after_header_respected(self) -> None:
        resp = _mock_response(429, headers={"Retry-After": "10"})
        delay = _get_delay(0, [1.0, 2.0], response=resp, respect_retry_after=True)
        assert delay == 10.0

    def test_retry_after_header_capped_at_120(self) -> None:
        resp = _mock_response(429, headers={"Retry-After": "999"})
        delay = _get_delay(0, [1.0], response=resp, respect_retry_after=True)
        assert delay == 120.0

    def test_retry_after_header_ignored_when_disabled(self) -> None:
        resp = _mock_response(429, headers={"Retry-After": "10"})
        delay = _get_delay(0, [1.0, 2.0], response=resp, respect_retry_after=False)
        assert delay == 1.0

    def test_retry_after_invalid_value_falls_back(self) -> None:
        resp = _mock_response(429, headers={"Retry-After": "not-a-number"})
        delay = _get_delay(0, [5.0], response=resp, respect_retry_after=True)
        assert delay == 5.0

    def test_no_response_uses_delay_sequence(self) -> None:
        delay = _get_delay(0, [3.0], response=None, respect_retry_after=True)
        assert delay == 3.0


# ---------------------------------------------------------------------------
# async_retry_request tests
# ---------------------------------------------------------------------------


class TestAsyncRetryRequest:
    """Integration tests for the async_retry_request function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        resp = _mock_response(200)
        request_fn = AsyncMock(return_value=resp)

        result = await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[0.0, 0.0, 0.0],
        )

        assert result is resp
        assert request_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429(self) -> None:
        fail_resp = _mock_response(429, text="Rate limited")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])

        result = await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[0.0, 0.0, 0.0],
        )

        assert result is ok_resp
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_500(self) -> None:
        fail_resp = _mock_response(500, text="Server Error")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])

        result = await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[0.0, 0.0, 0.0],
        )

        assert result is ok_resp
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_502(self) -> None:
        fail_resp = _mock_response(502, text="Bad Gateway")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])

        result = await async_retry_request(
            request_fn,
            max_retries=2,
            retry_delays=[0.0, 0.0],
        )

        assert result is ok_resp
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_503(self) -> None:
        fail_resp = _mock_response(503, text="Service Unavailable")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])

        result = await async_retry_request(
            request_fn,
            max_retries=2,
            retry_delays=[0.0, 0.0],
        )

        assert result is ok_resp
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_status(self) -> None:
        """A 403 is not retryable by default and should raise immediately."""
        fail_resp = _mock_response(403, text="Forbidden")
        request_fn = AsyncMock(return_value=fail_resp)

        with pytest.raises(RetryExhaustedError, match="403"):
            await async_retry_request(
                request_fn,
                max_retries=3,
                retry_delays=[0.0, 0.0, 0.0],
            )

        assert request_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self) -> None:
        fail_resp = _mock_response(500, text="Server Error")
        request_fn = AsyncMock(return_value=fail_resp)

        with pytest.raises(RetryExhaustedError) as exc_info:
            await async_retry_request(
                request_fn,
                max_retries=2,
                retry_delays=[0.0, 0.0],
                error_label="TestAPI",
            )

        assert "TestAPI" in str(exc_info.value)
        assert exc_info.value.last_exception is not None
        # 1 initial + 2 retries = 3 total
        assert request_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_request_error(self) -> None:
        """Network errors (httpx.RequestError) trigger retries."""
        conn_error = httpx.ConnectError("Connection refused")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[conn_error, ok_resp])

        result = await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[0.0, 0.0, 0.0],
        )

        assert result is ok_resp
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_on_request_error(self) -> None:
        conn_error = httpx.ConnectError("Connection refused")
        request_fn = AsyncMock(side_effect=conn_error)

        with pytest.raises(RetryExhaustedError, match="Connection refused"):
            await async_retry_request(
                request_fn,
                max_retries=1,
                retry_delays=[0.0],
            )

        # 1 initial + 1 retry = 2
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_custom_retryable_statuses(self) -> None:
        """Only custom retryable statuses trigger retry."""
        fail_429 = _mock_response(429, text="Rate limited")
        request_fn = AsyncMock(return_value=fail_429)

        # 429 is NOT in our custom set, so no retry
        with pytest.raises(RetryExhaustedError):
            await async_retry_request(
                request_fn,
                max_retries=3,
                retry_delays=[0.0],
                retryable_statuses={500},
            )

        assert request_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self) -> None:
        """Custom exception types trigger retry."""
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[ValueError("oops"), ok_resp])

        result = await async_retry_request(
            request_fn,
            max_retries=2,
            retry_delays=[0.0, 0.0],
            retryable_exceptions=(ValueError,),
        )

        assert result is ok_resp
        assert request_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_on_retry_callback(self) -> None:
        """The on_retry callback is invoked before each retry sleep."""
        fail_resp = _mock_response(500, text="Error")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, fail_resp, ok_resp])
        callback_calls: list[tuple[int, int, float, str]] = []

        def on_retry(attempt: int, max_retries: int, delay: float, reason: str) -> None:
            callback_calls.append((attempt, max_retries, delay, reason))

        await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[0.0, 0.0, 0.0],
            on_retry=on_retry,
            error_label="TestAPI",
        )

        assert len(callback_calls) == 2
        assert callback_calls[0] == (1, 3, 0.0, "TestAPI 500")
        assert callback_calls[1] == (2, 3, 0.0, "TestAPI 500")

    @pytest.mark.asyncio
    async def test_error_label_in_log_message(self) -> None:
        """The error_label appears in the RetryExhaustedError message."""
        fail_resp = _mock_response(500, text="fail")
        request_fn = AsyncMock(return_value=fail_resp)

        with pytest.raises(RetryExhaustedError, match="MyService"):
            await async_retry_request(
                request_fn,
                max_retries=0,
                retry_delays=[],
                error_label="MyService",
            )

    @pytest.mark.asyncio
    async def test_retry_after_header_used(self) -> None:
        """When Retry-After header is present, it determines the delay."""
        fail_resp = _mock_response(429, text="Rate limited", headers={"Retry-After": "0"})
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])
        callback_calls: list[tuple[int, int, float, str]] = []

        def on_retry(attempt: int, max_retries: int, delay: float, reason: str) -> None:
            callback_calls.append((attempt, max_retries, delay, reason))

        await async_retry_request(
            request_fn,
            max_retries=1,
            retry_delays=[99.0],  # Would be 99s without Retry-After
            on_retry=on_retry,
        )

        # Should have used 0.0 from Retry-After, not 99.0
        assert callback_calls[0][2] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_retries_then_success(self) -> None:
        """Successful after multiple consecutive failures."""
        fail_resp = _mock_response(503, text="Unavailable")
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(
            side_effect=[fail_resp, fail_resp, fail_resp, ok_resp],
        )

        result = await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[0.0, 0.0, 0.0],
        )

        assert result is ok_resp
        assert request_fn.call_count == 4

    @pytest.mark.asyncio
    async def test_zero_max_retries(self) -> None:
        """With max_retries=0, any failure raises immediately."""
        fail_resp = _mock_response(500, text="Error")
        request_fn = AsyncMock(return_value=fail_resp)

        with pytest.raises(RetryExhaustedError):
            await async_retry_request(
                request_fn,
                max_retries=0,
                retry_delays=[],
            )

        assert request_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_last_exception_preserved(self) -> None:
        """RetryExhaustedError.last_exception holds the final exception."""
        fail_resp = _mock_response(500, text="Internal Server Error")
        request_fn = AsyncMock(return_value=fail_resp)

        with pytest.raises(RetryExhaustedError) as exc_info:
            await async_retry_request(
                request_fn,
                max_retries=1,
                retry_delays=[0.0],
            )

        assert isinstance(exc_info.value.last_exception, httpx.HTTPStatusError)
        assert exc_info.value.last_exception.response.status_code == 500


class TestRetryJitter:
    """Jitter de-synchronizes retry storms across the concurrent fan-out."""

    @pytest.mark.asyncio
    async def test_jitter_extends_delay_within_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With jitter f, the configured delay d sleeps in [d, d*(1+f)]."""
        slept: list[float] = []

        async def _record_sleep(delay: float) -> None:
            slept.append(delay)

        # Deterministic jitter: always take the maximum of the jitter window.
        monkeypatch.setattr("python.utils.retry.random.random", lambda: 1.0)
        monkeypatch.setattr("python.utils.retry.asyncio.sleep", _record_sleep)

        fail_resp = _mock_response(429)
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])

        await async_retry_request(
            request_fn,
            max_retries=3,
            retry_delays=[2.0],
            jitter=0.5,
        )

        assert len(slept) == 1
        # base 2.0 + full jitter window 2.0*0.5 = 3.0
        assert slept[0] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_no_jitter_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """jitter defaults to 0.0 — delays are unchanged (back-compat)."""
        slept: list[float] = []

        async def _record_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr("python.utils.retry.random.random", lambda: 1.0)
        monkeypatch.setattr("python.utils.retry.asyncio.sleep", _record_sleep)

        fail_resp = _mock_response(429)
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, ok_resp])

        await async_retry_request(request_fn, max_retries=3, retry_delays=[2.0])

        assert slept == [2.0]


class TestRetryElapsedBudget:
    """max_elapsed_seconds bounds total retry wall-clock per request."""

    @pytest.mark.asyncio
    async def test_gives_up_when_budget_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A long delay sequence must not sleep past the elapsed budget."""
        slept: list[float] = []

        async def _record_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr("python.utils.retry.asyncio.sleep", _record_sleep)

        fail_resp = _mock_response(429, text="rate limited")
        request_fn = AsyncMock(return_value=fail_resp)

        with pytest.raises(RetryExhaustedError):
            await async_retry_request(
                request_fn,
                max_retries=5,
                retry_delays=[100.0, 100.0, 100.0, 100.0, 100.0],
                max_elapsed_seconds=0.5,
            )

        # The first 100s sleep would already blow the 0.5s budget, so we give
        # up immediately without sleeping at all.
        assert sum(slept) <= 0.5

    @pytest.mark.asyncio
    async def test_unbounded_when_budget_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (None) preserves the historical unbounded-retry behavior."""
        slept: list[float] = []

        async def _record_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr("python.utils.retry.asyncio.sleep", _record_sleep)

        fail_resp = _mock_response(429)
        ok_resp = _mock_response(200)
        request_fn = AsyncMock(side_effect=[fail_resp, fail_resp, ok_resp])

        await async_retry_request(
            request_fn,
            max_retries=5,
            retry_delays=[1.0, 2.0],
        )

        assert slept == [1.0, 2.0]
