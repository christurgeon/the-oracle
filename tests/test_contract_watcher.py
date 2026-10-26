"""Test Kalshi schema contracts and validation."""

from __future__ import annotations

import time

from python.data.kalshi_schemas import (
    SCHEMA_BALANCE,
    SCHEMA_FILL,
    SCHEMA_MARKET_SETTLEMENT,
    SCHEMA_ORDER_RESPONSE,
    SCHEMA_ORDER_STATUS_CHECK,
    SCHEMA_POSITION_ROW,
    KalshiSchema,
)
from python.notifications.base import AlertLevel


def test_schemas_importable() -> None:
    assert isinstance(SCHEMA_BALANCE, KalshiSchema)
    assert isinstance(SCHEMA_FILL, KalshiSchema)
    assert isinstance(SCHEMA_MARKET_SETTLEMENT, KalshiSchema)
    assert isinstance(SCHEMA_ORDER_RESPONSE, KalshiSchema)
    assert isinstance(SCHEMA_ORDER_STATUS_CHECK, KalshiSchema)
    assert isinstance(SCHEMA_POSITION_ROW, KalshiSchema)


def test_schema_required_fields_are_frozensets() -> None:
    assert isinstance(SCHEMA_BALANCE.required, frozenset)
    assert "balance" in SCHEMA_BALANCE.required


def test_schema_extra_fields_not_in_required() -> None:
    assert "created_time" not in SCHEMA_ORDER_RESPONSE.required
    assert "market_id" not in SCHEMA_FILL.required


def test_v2_create_order_response_passes_schema() -> None:
    """The Kalshi V2 create-order response omits ``status`` (it returns
    order_id/fill_count/remaining_count/... instead). The order-response schema
    must not require ``status`` or every live placement logs a false
    'missing required: status' drift warning."""
    resp = {
        "order_id": "abc123",
        "client_order_id": "c1",
        "fill_count": "0.00",
        "remaining_count": "10.00",
        "ts_ms": 1700000000000,
    }
    for field in SCHEMA_ORDER_RESPONSE.required:
        assert field in resp, (
            f"SCHEMA_ORDER_RESPONSE.required field {field!r} missing from V2 response"
        )


def test_realistic_fill_passes_schema() -> None:
    fill = {
        "count_fp": "1.00",
        "side": "yes",
        "yes_price_dollars": "0.65",
        "is_taker": True,
        "fee_cost": "0.045500",
        "created_time": "2026-04-28T12:00:00Z",
        "order_id": "abc123",
    }
    for field in SCHEMA_FILL.required:
        assert field in fill, f"SCHEMA_FILL.required field {field!r} missing from sample"


def test_realistic_balance_passes_schema() -> None:
    resp = {"balance": 19850, "payout": 0}
    for field in SCHEMA_BALANCE.required:
        assert field in resp


def test_realistic_position_passes_schema() -> None:
    row = {
        "ticker": "KXHIGHDEN-26APR28-B54.5",
        "position_fp": "-16.00",
        "market_exposure_dollars": "11.000000",
        "fees_paid_dollars": "0.070000",
        "resting_orders_count": 0,
    }
    for field in SCHEMA_POSITION_ROW.required:
        assert field in row


# --- check() basic detection -------------------------------------------------

def test_check_missing_required_field_fires_alert() -> None:
    fired: list[tuple[str, AlertLevel]] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append((msg, lvl)))

    watcher.check({}, SCHEMA_BALANCE, context="test_balance")

    assert len(fired) == 1
    msg, lvl = fired[0]
    assert "[CONTRACT]" in msg
    assert "balance" in msg
    assert "test_balance" in msg
    assert lvl == AlertLevel.WARNING


def test_check_type_mismatch_fires_alert() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    # balance is present but wrong type (list instead of int/float)
    watcher.check({"balance": [1, 2, 3]}, SCHEMA_BALANCE, context="test_balance")

    assert len(fired) == 1
    assert "type mismatch" in fired[0]
    assert "balance" in fired[0]


def test_check_valid_data_does_not_fire() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    watcher.check({"balance": 19850}, SCHEMA_BALANCE, context="test_balance")

    assert fired == []


def test_check_extra_fields_ignored() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    watcher.check(
        {"balance": 100, "new_field_kalshi_added": "surprise"},
        SCHEMA_BALANCE,
        context="test_balance",
    )

    assert fired == []


def test_check_no_alert_fn_does_not_raise() -> None:
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=None)
    # Should not raise even with missing required field
    watcher.check({}, SCHEMA_BALANCE, context="test_balance")


def test_check_payload_sample_in_message() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    watcher.check({"unexpected": "value"}, SCHEMA_BALANCE, context="test_balance")

    assert len(fired) == 1
    assert "Payload sample" in fired[0]


# --- Deduplication -----------------------------------------------------------

def test_check_deduplicates_within_ttl() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(
        alert_fn=lambda msg, lvl: fired.append(msg),
        dedup_ttl=3600.0,
    )

    watcher.check({}, SCHEMA_BALANCE, context="test_balance")
    watcher.check({}, SCHEMA_BALANCE, context="test_balance")

    assert len(fired) == 1  # second call suppressed


def test_check_fires_again_after_ttl(monkeypatch) -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(
        alert_fn=lambda msg, lvl: fired.append(msg),
        dedup_ttl=60.0,
    )

    watcher.check({}, SCHEMA_BALANCE, context="test_balance")
    assert len(fired) == 1

    # Advance monotonic time past TTL
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + 120.0)

    watcher.check({}, SCHEMA_BALANCE, context="test_balance")
    assert len(fired) == 2  # re-fired after TTL


def test_check_different_contexts_are_independent() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    watcher.check({}, SCHEMA_BALANCE, context="startup")
    watcher.check({}, SCHEMA_BALANCE, context="reconcile")

    assert len(fired) == 2  # different codes, both fire


# --- warn() ------------------------------------------------------------------

def test_warn_fires_alert_with_correct_format() -> None:
    fired: list[tuple[str, AlertLevel]] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append((msg, lvl)))

    watcher.warn("clock_skew", "Clock skew: 45.0s detected")

    assert len(fired) == 1
    msg, lvl = fired[0]
    assert "[CONTRACT]" in msg
    assert "clock_skew" in msg
    assert "Clock skew: 45.0s detected" in msg
    assert lvl == AlertLevel.WARNING


def test_warn_deduplicates_by_code() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    watcher.warn("clock_skew", "first")
    watcher.warn("clock_skew", "second")

    assert len(fired) == 1


def test_warn_different_codes_both_fire() -> None:
    fired: list[str] = []
    from python.data.contract_watcher import ContractWatcher
    watcher = ContractWatcher(alert_fn=lambda msg, lvl: fired.append(msg))

    watcher.warn("clock_skew", "msg1")
    watcher.warn("unknown_order_status:new_status", "msg2")

    assert len(fired) == 2
