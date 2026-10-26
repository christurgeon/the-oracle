"""Loose Kalshi API response contracts — only the fields we actually use."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class KalshiSchema:
    """Minimal contract for one Kalshi response shape.

    required: fields that MUST be present — alert if absent.
    type_checks: alert if field is present but the wrong type.
                 May overlap with required — presence and type are checked independently.
                 Side-conditional or optional fields that are not in required go here too.
    Extra fields Kalshi adds are silently ignored.
    """

    required: frozenset[str]
    type_checks: MappingProxyType[str, tuple[type, ...]]


# --- Fill row (each item in fills_resp["fills"]) ----------------------------
SCHEMA_FILL = KalshiSchema(
    required=frozenset({"count_fp", "side"}),
    type_checks=MappingProxyType({
        "count_fp": (int, float, str),
        "side": (str,),
        "yes_price_dollars": (int, float, str),
        "no_price_dollars": (int, float, str),
        "is_taker": (bool,),
        "fee_cost": (int, float, str),
        "created_time": (str,),
    }),
)

# --- Order placement response (resp["order"]) --------------------------------
# The Kalshi V2 create-order response (POST /portfolio/events/orders) returns
# order_id/fill_count/remaining_count/average_fill_price/ts_ms and NO "status"
# field, so only order_id is required. "status" is kept as an optional type
# check for the legacy/GET-order shapes that still include it.
SCHEMA_ORDER_RESPONSE = KalshiSchema(
    required=frozenset({"order_id"}),
    type_checks=MappingProxyType({
        "order_id": (str,),
        "status": (str,),
        "fill_count": (int, float, str),
        "remaining_count": (int, float, str),
    }),
)

# --- Order status check during fill polling (resp["order"]) ------------------
SCHEMA_ORDER_STATUS_CHECK = KalshiSchema(
    required=frozenset({"status"}),
    type_checks=MappingProxyType({
        "status": (str,),
        "fill_count_fp": (int, float, str),
    }),
)

# --- Account balance (balance_resp) ------------------------------------------
SCHEMA_BALANCE = KalshiSchema(
    required=frozenset({"balance"}),
    type_checks=MappingProxyType({
        "balance": (int, float),
    }),
)

# --- Position row (each item in market_positions) ----------------------------
SCHEMA_POSITION_ROW = KalshiSchema(
    required=frozenset({"ticker", "position_fp", "market_exposure_dollars"}),
    type_checks=MappingProxyType({
        "ticker": (str,),
        "position_fp": (int, float, str),
        "market_exposure_dollars": (int, float, str),
    }),
)

# --- Settlement market response (resp["market"]) -----------------------------
SCHEMA_MARKET_SETTLEMENT = KalshiSchema(
    required=frozenset({"status", "result"}),
    type_checks=MappingProxyType({
        "status": (str,),
        "result": (str,),
    }),
)
