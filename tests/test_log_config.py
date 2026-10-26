"""Tests for the reusable structured logging module."""

from __future__ import annotations

import json
import logging

import pytest
import structlog

from python.utils.log_config import (
    LoggingConfig,
    bind_contextvars,
    setup_logging,
    unbind_contextvars,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:  # noqa: PT004
    """Reset root logger handlers and structlog state between tests."""
    yield
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    structlog.reset_defaults()


def test_json_renderer_produces_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(LoggingConfig(renderer="json"))
    logging.getLogger("test.json").info("hello %s", "world")

    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data["event"] == "hello world"
    assert data["level"] == "info"
    assert data["logger"] == "test.json"
    assert "timestamp" in data


def test_console_renderer_produces_output(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(LoggingConfig(renderer="console"))
    logging.getLogger("test.console").info("hello console")

    captured = capsys.readouterr()
    assert "hello console" in captured.out


def test_module_levels_override() -> None:
    setup_logging(LoggingConfig(module_levels={"test.noisy": "WARNING"}))

    noisy = logging.getLogger("test.noisy")
    assert noisy.getEffectiveLevel() == logging.WARNING

    normal = logging.getLogger("test.normal")
    assert normal.getEffectiveLevel() == logging.INFO


def test_log_file_output(tmp_path: pytest.TempPathFactory) -> None:
    log_path = str(tmp_path / "test.log")  # type: ignore[operator]
    setup_logging(LoggingConfig(renderer="console", log_file=log_path))

    logging.getLogger("test.file").info("file message")

    with open(log_path) as f:
        content = f.read()
    data = json.loads(content.strip())
    assert data["event"] == "file message"
    assert data["level"] == "info"


def test_contextvars_binding(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(LoggingConfig(renderer="json"))

    bind_contextvars(strategy="weather")
    logging.getLogger("test.ctx").info("with context")
    unbind_contextvars("strategy")

    captured = capsys.readouterr()
    data = json.loads(captured.out.strip())
    assert data["strategy"] == "weather"
    assert data["event"] == "with context"


def test_default_config() -> None:
    setup_logging()
    root = logging.getLogger()
    assert root.level == logging.INFO
    assert len(root.handlers) == 2  # console + in-memory log buffer
