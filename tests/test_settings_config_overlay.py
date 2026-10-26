"""Tests for the ORACLE_CONFIG_OVERLAY deployment overlay and its pre-restart check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from python.orchestrator import configured_strategy_flags, validate_strategy_allocations
from python.settings import CONFIG_OVERLAY_ENV, Settings, config_overlay_path
from scripts import check_live_config

_BASE: dict[str, Any] = {
    "strategies": {
        "weather": {"enabled": True, "edge_threshold": 0.07},
        "favorite_bias": {"enabled": False, "kelly_fraction": 0.2},
    },
    "risk": {"strategy_allocations": {"weather_edge": 0.0, "favorite_bias": 0.0}},
}


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def base_config(tmp_path: Path) -> Path:
    return _write(tmp_path / "config.yaml", _BASE)


def _load_with(
    tmp_path: Path, base_config: Path, overlay: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> Settings:
    path = _write(tmp_path / "overlay.yaml", overlay)
    monkeypatch.setenv(CONFIG_OVERLAY_ENV, str(path))
    return Settings.load(config_path=base_config)


def test_no_overlay_loads_base(base_config: Path) -> None:
    settings = Settings.load(config_path=base_config)
    assert settings.strategies.favorite_bias.enabled is False
    assert settings.risk.strategy_allocations == {"weather_edge": 0.0, "favorite_bias": 0.0}


def test_overlay_merges_into_submodels(
    tmp_path: Path, base_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _load_with(
        tmp_path, base_config, {"strategies": {"favorite_bias": {"enabled": True}}}, monkeypatch
    )
    assert settings.strategies.favorite_bias.enabled is True
    assert settings.strategies.favorite_bias.kelly_fraction == 0.2
    assert settings.strategies.weather.edge_threshold == 0.07


def test_overlay_replaces_map_fields_whole(
    tmp_path: Path, base_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _load_with(
        tmp_path,
        base_config,
        {"risk": {"strategy_allocations": {"weather_edge": 0.0}}},
        monkeypatch,
    )
    assert settings.risk.strategy_allocations == {"weather_edge": 0.0}


def test_overlay_null_replaces_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _write(
        tmp_path / "config.yaml",
        {
            "strategies": {
                "weather": {"enabled": True, "edge_threshold": 0.07},
                "favorite_bias": {"max_strike_climatology_z": 2.0},
            }
        },
    )
    settings = _load_with(
        tmp_path,
        base,
        {"strategies": {"favorite_bias": {"max_strike_climatology_z": None}}},
        monkeypatch,
    )
    assert settings.strategies.favorite_bias.max_strike_climatology_z is None


def test_overlay_unknown_key_raises(
    tmp_path: Path, base_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match=r"unknown key strategies\.favorite_bias\.enabld"):
        _load_with(
            tmp_path, base_config, {"strategies": {"favorite_bias": {"enabld": True}}}, monkeypatch
        )


def test_overlay_outside_allowed_roots_raises(
    tmp_path: Path, base_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match=r"may not set strategies\.weather"):
        _load_with(
            tmp_path,
            base_config,
            {"strategies": {"weather": {"edge_threshold": 0.05}}},
            monkeypatch,
        )


def test_overlay_missing_file_raises(
    tmp_path: Path, base_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CONFIG_OVERLAY_ENV, str(tmp_path / "absent.yaml"))
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        Settings.load(config_path=base_config)


def test_overlay_path_falls_back_to_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(f"{CONFIG_OVERLAY_ENV}=/srv/ops/config.live.yaml\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CONFIG_OVERLAY_ENV, raising=False)
    assert config_overlay_path() == Path("/srv/ops/config.live.yaml")


def test_empty_env_value_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(f"{CONFIG_OVERLAY_ENV}=/srv/ops/config.live.yaml\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(CONFIG_OVERLAY_ENV, "")
    assert config_overlay_path() is None


def test_configured_flags_catch_allocation_for_disabled_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _write(
        tmp_path / "config.yaml",
        {
            "strategies": {
                "weather": {"enabled": True, "edge_threshold": 0.07},
                "favorite_bias": {"enabled": False},
            },
            "risk": {"strategy_allocations": {"weather_edge": 0.0, "favorite_bias": 0.75}},
        },
    )
    settings = Settings.load(config_path=base)
    assert configured_strategy_flags(settings) == [("weather_edge", True)]
    with pytest.raises(ValueError, match="unknown strategy"):
        validate_strategy_allocations(
            settings.risk.strategy_allocations, configured_strategy_flags(settings)
        )


def test_sizing_violations_flag_kelly_past_ceiling(base_config: Path) -> None:
    settings = Settings.load(config_path=base_config)
    assert check_live_config.sizing_violations(settings) == []
    settings.strategies.favorite_bias.kelly_fraction = 0.9
    assert check_live_config.sizing_violations(settings) == [
        "favorite_bias.kelly_fraction=0.9 > 0.60"
    ]


def test_live_hash_ignores_keys_outside_overlay_roots(base_config: Path) -> None:
    settings = Settings.load(config_path=base_config)
    dump = check_live_config.redacted_dump(settings)
    live = check_live_config.digest(check_live_config.live_subset(dump))
    dump["strategies"]["weather"]["edge_threshold"] = 0.5
    assert check_live_config.digest(check_live_config.live_subset(dump)) == live
    dump["risk"]["kelly_fraction"] = 0.5
    assert check_live_config.digest(check_live_config.live_subset(dump)) != live


def test_redacted_dump_excludes_secrets(base_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORACLE_TELEGRAM_BOT_TOKEN", "123:secret")
    dump = check_live_config.redacted_dump(Settings.load(config_path=base_config))
    assert "telegram_bot_token" not in dump
    assert "kalshi_api_key" not in dump


def test_main_fails_on_pinned_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pin = tmp_path / "config.live.sha256"
    pin.write_text("0" * 64 + "\n")
    assert check_live_config.main(["--expect-hash", str(pin)]) == 1
    assert "!= pinned" in capsys.readouterr().err


def test_main_passes_on_matching_pin(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert check_live_config.main([]) == 0
    live = next(
        line.split()[1] for line in capsys.readouterr().out.splitlines() if line.startswith("live ")
    )
    pin = tmp_path / "config.live.sha256"
    pin.write_text(f"{live}\n")
    assert check_live_config.main(["--expect-hash", str(pin)]) == 0
