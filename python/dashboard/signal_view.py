"""
Signal monitor widget for The Oracle TUI dashboard.

Displays a DataTable of currently active trade signals with
strategy, market, side, model probability, market price, edge,
confidence, and suggested size.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable, Label


@dataclass
class SignalRow:
    """Data for a single row in the signal monitor table."""

    signal_id: str  # unique key for upsert
    strategy: str
    market: str
    side: str
    model_prob: float
    market_price: float
    edge: float
    confidence: float
    size: float  # dollars
    timestamp: datetime | None = None

    @property
    def age_str(self) -> str:
        """Human-readable age of the signal."""
        if self.timestamp is None:
            return "--"
        delta = datetime.now(UTC) - self.timestamp
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s"
        if secs < 3600:
            return f"{secs // 60}m"
        return f"{secs // 3600}h"


def _row_cells(sig: SignalRow) -> tuple[str, ...]:
    """Format a ``SignalRow`` into the ordered tuple of display cells."""
    return (
        sig.strategy,
        sig.market,
        sig.side,
        f"{sig.model_prob:.1%}",
        f"{sig.market_price:.1%}",
        f"{sig.edge:+.1%}",
        f"{sig.confidence:.0%}",
        f"${sig.size:,.0f}",
        sig.age_str,
    )


# Column definitions: (key, label)
_COLUMNS: list[tuple[str, str]] = [
    ("strategy", "Strategy"),
    ("market", "Market"),
    ("side", "Side"),
    ("model_prob", "Model P"),
    ("market_price", "Mkt P"),
    ("edge", "Edge"),
    ("confidence", "Conf"),
    ("size", "Size ($)"),
    ("age", "Age"),
]


class SignalView(Widget):
    """
    Signal monitor panel.

    Shows a DataTable of currently active signals that strategies have
    produced. The orchestrator calls ``update_signals()`` to refresh the
    full set, or ``add_signal()`` / ``remove_signal()`` for incremental
    changes.
    """

    DEFAULT_CSS = """
    SignalView {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
    }
    SignalView #signal-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    SignalView DataTable {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._signal_keys: dict[str, object] = {}  # signal_id -> row_key

    def compose(self) -> ComposeResult:
        yield Label("SIGNAL MONITOR", id="signal-title")
        yield DataTable(id="signal-table")

    def on_mount(self) -> None:
        """Set up table columns."""
        table = self.query_one("#signal-table", DataTable)
        for key, label in _COLUMNS:
            table.add_column(label, key=key)

    # -- Public API -----------------------------------------------------------

    def update_signals(self, signals: list[SignalRow]) -> None:
        """
        Replace all signals in the table with the given list.

        This is the simplest interface — the orchestrator builds the
        full list and pushes it every cycle.
        """
        table = self.query_one("#signal-table", DataTable)
        table.clear()
        self._signal_keys.clear()

        for sig in signals:
            row_key = table.add_row(*_row_cells(sig), key=sig.signal_id)
            self._signal_keys[sig.signal_id] = row_key

    def add_signal(self, signal: SignalRow) -> None:
        """Add or update a single signal row."""
        table = self.query_one("#signal-table", DataTable)

        row_data = _row_cells(signal)

        if signal.signal_id in self._signal_keys:
            # Update existing row
            for col_idx, (col_key, _) in enumerate(_COLUMNS):
                table.update_cell(signal.signal_id, col_key, row_data[col_idx])
        else:
            row_key = table.add_row(*row_data, key=signal.signal_id)
            self._signal_keys[signal.signal_id] = row_key

    def remove_signal(self, signal_id: str) -> None:
        """Remove a signal from the table."""
        if signal_id in self._signal_keys:
            table = self.query_one("#signal-table", DataTable)
            table.remove_row(signal_id)
            del self._signal_keys[signal_id]

    def clear_signals(self) -> None:
        """Remove all signals from the table."""
        table = self.query_one("#signal-table", DataTable)
        table.clear()
        self._signal_keys.clear()
