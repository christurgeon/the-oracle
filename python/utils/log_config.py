"""Structured logging setup using structlog's stdlib bridge."""

from __future__ import annotations

import collections
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

__all__ = [
    "LoggingConfig",
    "setup_logging",
    "bind_contextvars",
    "unbind_contextvars",
    "clear_contextvars",
    "get_log_buffer",
]


class RecentLogBuffer(logging.Handler):
    """Ring buffer that stores recent formatted log lines in memory.

    Used by the /agent command to provide log context to Claude
    without depending on journalctl permissions or log file access.
    """

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self.buffer: collections.deque[str] = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(self.format(record))
        except Exception:
            self.handleError(record)

    def get_lines(self, n: int | None = None) -> list[str]:
        """Return the last *n* log lines (all if *n* is None)."""
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]


_log_buffer = RecentLogBuffer(maxlen=500)


def get_log_buffer() -> RecentLogBuffer:
    """Return the module-level log ring buffer."""
    return _log_buffer


class LoggingConfig(BaseModel):
    """Configuration for structured logging."""

    level: str = "INFO"
    renderer: Literal["console", "json"] = "console"
    module_levels: dict[str, str] = {}
    log_file: str | None = None
    log_file_max_bytes: int = 50_000_000  # 50 MB
    log_file_backup_count: int = 5


def _make_formatter(
    pre_chain: list[structlog.types.Processor],
    renderer: Literal["console", "json"],
) -> structlog.stdlib.ProcessorFormatter:
    """Build a ProcessorFormatter with the given renderer."""
    if renderer == "json":
        final: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        final = structlog.dev.ConsoleRenderer()
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final,
        ],
    )


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Configure structlog + stdlib logging."""
    if config is None:
        config = LoggingConfig()

    # foreign_pre_chain enriches stdlib logging.getLogger() calls;
    # structlog.configure() processors enrich structlog.get_logger() calls.
    pre_chain: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=[*pre_chain, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_make_formatter(pre_chain, config.renderer))

    # In-memory ring buffer for /agent (always active, plain-text format)
    _log_buffer.setFormatter(_make_formatter(pre_chain, "console"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(_log_buffer)
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    if config.log_file:
        Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            config.log_file,
            maxBytes=config.log_file_max_bytes,
            backupCount=config.log_file_backup_count,
        )
        file_handler.setFormatter(_make_formatter(pre_chain, "json"))
        root.addHandler(file_handler)

    for module, level_str in config.module_levels.items():
        logging.getLogger(module).setLevel(
            getattr(logging, level_str.upper(), logging.INFO)
        )
