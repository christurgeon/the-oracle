from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.data.emos import EmosTable


def _payload_with_by_lead():
    return {
        "fit_date": "2026-05-04",
        "training_window_days": 46,
        "lead_handling": "by_lead",
        "schema_version": 1,
        "cells": {
            "DEN/high": {
                "by_lead": {
                    "0": {
                        "a": 0.0, "b": 1.0, "c": 1.0, "d": 0.0,
                        "n_train": 50, "leads_pooled": [0],
                    },
                    "1": {
                        "a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0,
                        "n_train": 50, "leads_pooled": [1],
                    },
                    "2": {
                        "a": 2.0, "b": 1.0, "c": 1.0, "d": 0.0,
                        "n_train": 50, "leads_pooled": [2],
                    },
                },
            },
        },
    }


def _payload_pooled():
    return {
        "fit_date": "2026-05-04", "schema_version": 1,
        "cells": {
            "AUS/high": {
                "a": -4.0, "b": 1.05, "c": 15.0, "d": 0.4,
                "n_train": 127, "leads_pooled": [0, 1, 2],
                "ens_std_p99": 6.5,
            },
        },
    }


def test_get_pooled_returns_same_for_all_leads(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    c0 = table.get(city="AUS", var="high", lead=0)
    c1 = table.get(city="AUS", var="high", lead=1)
    c2 = table.get(city="AUS", var="high", lead=2)
    assert c0 is not None and c1 is not None and c2 is not None
    assert c0.a == c1.a == c2.a == -4.0


def test_get_by_lead_returns_lead_specific(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_with_by_lead()))
    table = EmosTable.load(p)
    c0 = table.get(city="DEN", var="high", lead=0)
    c1 = table.get(city="DEN", var="high", lead=1)
    c2 = table.get(city="DEN", var="high", lead=2)
    assert c0 is not None and c0.a == 0.0
    assert c1 is not None and c1.a == 1.0
    assert c2 is not None and c2.a == 2.0


def test_get_lead_out_of_range_returns_none(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    assert table.get(city="AUS", var="high", lead=3) is None
    assert table.get(city="AUS", var="high", lead=-1) is None


def test_get_unknown_cell_returns_none(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    assert table.get(city="ZZZ", var="high", lead=0) is None


def test_get_rejects_wrong_var_shape(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    with pytest.raises(KeyError):
        table.get(city="AUS", var="TEMP_MAX", lead=0)  # NOT canonical "high"


def test_training_ens_std_p99_returns_finite_when_present(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    assert table.training_ens_std_p99(city="AUS", var="high") == 6.5


def test_training_ens_std_p99_returns_none_for_legacy_artifact(tmp_path: Path):
    payload = _payload_pooled()
    del payload["cells"]["AUS/high"]["ens_std_p99"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(payload))
    table = EmosTable.load(p)
    assert table.training_ens_std_p99(city="AUS", var="high") is None


def test_training_ens_std_p99_unknown_cell_returns_none(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    assert table.training_ens_std_p99(city="ZZZ", var="high") is None


def test_training_ens_std_p99_wrong_var_shape_returns_none(tmp_path: Path):
    """Unlike get(), training_ens_std_p99 swallows KeyError and returns None."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_payload_pooled()))
    table = EmosTable.load(p)
    assert table.training_ens_std_p99(city="AUS", var="TEMP_MAX") is None
