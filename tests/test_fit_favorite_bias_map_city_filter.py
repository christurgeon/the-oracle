"""g(mid) corpus must stay frozen to the validated favorite_bias cities — new
Kalshi cities are collection-only until Phase B (spec 2026-06-08 §7)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import scripts.fit_favorite_bias_map as fit


def _market(ticker: str, ya: float, yb: float) -> dict:
    # lead-1: snapshot taken on 2026-06-01, forecast date 2026-06-02
    ts = datetime(2026, 6, 1, 12, tzinfo=UTC).timestamp()
    return {
        "ticker": ticker,
        "strike_type": "greater",
        "floor_strike": 85.0,
        "cap_strike": None,
        "snapshot_ts": ts,
        "yes_ask_dollars": ya,
        "yes_bid_dollars": yb,
    }


def test_build_pairs_excludes_new_cities(monkeypatch) -> None:
    # MIA = validated (original 9); PHX = new collection-only city.
    markets = [
        _market("KXHIGHMIA-26JUN02-T85", 0.60, 0.55),
        _market("KXHIGHTPHX-26JUN02-T85", 0.70, 0.65),
    ]
    settle = {
        ("MIA", "2026-06-02"): {"high_f": 88.0},
        ("PHX", "2026-06-02"): {"high_f": 105.0},
    }
    monkeypatch.setattr(fit, "load_snapshot_data", lambda _d: ([], settle, markets))
    pairs = fit.build_pairs(Path("ignored"))
    # Only MIA (mid 0.575) survives; PHX (mid 0.675) is filtered out.
    assert len(pairs) == 1, pairs
    assert abs(pairs[0][0] - 0.575) < 1e-9, pairs
    assert pairs[0][1] == 1  # 88F > 85F greater -> YES


def test_corpus_cities_are_the_validated_nine() -> None:
    assert fit._CORPUS_CITIES == frozenset(
        {"NYC", "CHI", "MIA", "LAX", "DEN", "AUS", "SEA", "BOS", "PHIL"}
    )
    for new_city in ("ATL", "DAL", "HOU", "LV", "MIN", "NOLA", "OKC", "PHX", "SATX", "SFO", "DC"):
        assert new_city not in fit._CORPUS_CITIES
