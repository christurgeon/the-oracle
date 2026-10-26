"""Immediate-fill detection must read both V1 and V2 create-order response shapes.

The legacy create response carried ``fill_count_fp``; the Kalshi V2 create-order
response (POST /portfolio/events/orders) reports synchronous fills as
``fill_count`` (a fixed-point contract string). ``_live_execute`` uses this to
decide whether to drive the immediate-fill bookkeeping path for taker orders
(e.g. favorite_bias, which crosses the ask and fills on placement).
"""

from __future__ import annotations

from python.execution.engine import ExecutionEngine


def test_reads_v2_fill_count() -> None:
    assert ExecutionEngine._immediate_fill_count({"fill_count": "3.00"}) == 3.0


def test_reads_legacy_fill_count_fp() -> None:
    assert ExecutionEngine._immediate_fill_count({"fill_count_fp": "2.00"}) == 2.0


def test_zero_when_absent() -> None:
    assert ExecutionEngine._immediate_fill_count({"order_id": "x"}) == 0.0


def test_prefers_v2_fill_count_when_both_present() -> None:
    # On the V2 create response fill_count is authoritative.
    assert (
        ExecutionEngine._immediate_fill_count({"fill_count": "5.00", "fill_count_fp": "0"})
        == 5.0
    )
