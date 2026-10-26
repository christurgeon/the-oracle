"""
Risk status widget for The Oracle TUI dashboard.

Displays risk limits vs current usage with progress bars, per-strategy
circuit breaker states, and the global kill switch indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

from python.dashboard.portfolio_view import _safe_query

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RiskGaugeData:
    """Current value and limit for a single risk metric."""

    label: str
    current: float
    limit: float

    @property
    def pct(self) -> float:
        """Percentage of limit consumed (0-100)."""
        if self.limit <= 0:
            return 0.0
        return min((self.current / self.limit) * 100, 100.0)


@dataclass
class CircuitBreakerState:
    """State of a per-strategy circuit breaker."""

    strategy_name: str
    enabled: bool = True
    reason: str = ""


# ---------------------------------------------------------------------------
# Sub-widgets
# ---------------------------------------------------------------------------


class _RiskGauge(Static):
    """A labelled progress bar showing usage vs limit."""

    DEFAULT_CSS = """
    _RiskGauge {
        height: auto;
        padding: 0 0 1 0;
    }
    _RiskGauge .gauge-header {
        height: 1;
    }
    _RiskGauge .gauge-label {
        width: 1fr;
        color: $text-muted;
    }
    _RiskGauge .gauge-value {
        width: auto;
        text-style: bold;
    }
    _RiskGauge ProgressBar {
        padding: 0;
    }
    _RiskGauge .gauge-ok {
        color: $success;
    }
    _RiskGauge .gauge-warning {
        color: $warning;
    }
    _RiskGauge .gauge-danger {
        color: $error;
    }
    """

    def __init__(self, data: RiskGaugeData, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._data = data

    def compose(self) -> ComposeResult:
        with Horizontal(classes="gauge-header"):
            yield Label(self._data.label, classes="gauge-label")
            val_id = f"{self.id}-val" if self.id else None
            yield Label(self._value_text(), classes="gauge-value", id=val_id)
        yield ProgressBar(
            total=100,
            show_eta=False,
            show_percentage=False,
            id=f"{self.id}-bar" if self.id else None,
        )

    def on_mount(self) -> None:
        bar = self.query_one(ProgressBar)
        bar.update(progress=self._data.pct)

    def update_data(self, data: RiskGaugeData) -> None:
        """Update the gauge with new data."""
        self._data = data
        try:
            val_label = self.query_one(".gauge-value", Label)
            val_label.update(self._value_text())
            bar = self.query_one(ProgressBar)
            bar.update(progress=data.pct)
        except Exception:
            pass

    def _value_text(self) -> str:
        return f"{self._data.current:.1f} / {self._data.limit:.1f}"


class _StatusIndicator(Static):
    """A labelled on/off indicator (for kill switch, circuit breakers)."""

    DEFAULT_CSS = """
    _StatusIndicator {
        height: 1;
        padding: 0 1;
    }
    _StatusIndicator .status-label {
        width: 1fr;
        color: $text-muted;
    }
    _StatusIndicator .status-on {
        color: $success;
        text-style: bold;
    }
    _StatusIndicator .status-off {
        color: $error;
        text-style: bold;
    }
    """

    def __init__(
        self,
        label: str,
        active: bool = True,
        detail: str = "",
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label_text = label
        self._active = active
        self._detail = detail

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Label(self._label_text, classes="status-label")
            status_class = "status-on" if self._active else "status-off"
            status_text = self._active_text()
            status_id = f"{self.id}-status" if self.id else None
            yield Label(status_text, classes=status_class, id=status_id)

    def _active_text(self) -> str:
        if self._active:
            return "ACTIVE"
        reason = f"  ({self._detail})" if self._detail else ""
        return f"DISABLED{reason}"

    def update_status(self, active: bool, detail: str = "") -> None:
        """Update the indicator state."""
        self._active = active
        self._detail = detail
        try:
            status_label = self.query_one(f"#{self.id}-status", Label)
            status_label.update(self._active_text())
            status_label.remove_class("status-on", "status-off")
            status_label.add_class("status-on" if active else "status-off")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class RiskView(Widget):
    """
    Risk status panel.

    Shows risk limit gauges (exposure, daily loss, drawdown), the global
    kill switch status, and per-strategy circuit breaker states.
    """

    DEFAULT_CSS = """
    RiskView {
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }
    RiskView #risk-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    RiskView #cb-section-title {
        text-style: bold;
        color: $text-muted;
        padding-top: 1;
        padding-bottom: 0;
    }
    RiskView #kill-section {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(
        self,
        *,
        exposure_limit: float = 40.0,
        daily_loss_limit: float = 5.0,
        drawdown_limit: float = 15.0,
        strategy_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._exposure_data = RiskGaugeData("Exposure", 0.0, exposure_limit)
        self._daily_loss_data = RiskGaugeData("Daily Loss", 0.0, daily_loss_limit)
        self._drawdown_data = RiskGaugeData("Drawdown", 0.0, drawdown_limit)
        self._strategy_names = strategy_names or ["weather"]
        self._kill_switch_active = False

    def compose(self) -> ComposeResult:
        yield Label("RISK STATUS", id="risk-title")
        yield _RiskGauge(self._exposure_data, id="gauge-exposure")
        yield _RiskGauge(self._daily_loss_data, id="gauge-daily-loss")
        yield _RiskGauge(self._drawdown_data, id="gauge-drawdown")

        yield Label("Kill Switch", id="kill-section")
        yield _StatusIndicator(
            "Global Kill Switch",
            active=False,
            detail="INACTIVE",
            id="kill-switch",
        )

        yield Label("Circuit Breakers", id="cb-section-title")
        for name in self._strategy_names:
            yield _StatusIndicator(
                name,
                active=True,
                id=f"cb-{name}",
            )

    # -- Public API -----------------------------------------------------------

    def update_risk(
        self,
        *,
        exposure_pct: float | None = None,
        daily_loss_pct: float | None = None,
        drawdown_pct: float | None = None,
    ) -> None:
        """
        Update risk gauge values.

        Values are in percentage points (e.g. 12.5 means 12.5%).
        """
        if exposure_pct is not None:
            self._exposure_data.current = exposure_pct
            if (g := _safe_query(self, "#gauge-exposure", _RiskGauge)) is not None:
                g.update_data(self._exposure_data)

        if daily_loss_pct is not None:
            self._daily_loss_data.current = daily_loss_pct
            if (g := _safe_query(self, "#gauge-daily-loss", _RiskGauge)) is not None:
                g.update_data(self._daily_loss_data)

        if drawdown_pct is not None:
            self._drawdown_data.current = drawdown_pct
            if (g := _safe_query(self, "#gauge-drawdown", _RiskGauge)) is not None:
                g.update_data(self._drawdown_data)

    def update_kill_switch(self, active: bool) -> None:
        """Update the kill switch indicator."""
        self._kill_switch_active = active
        if (indicator := _safe_query(self, "#kill-switch", _StatusIndicator)) is not None:
            if active:
                indicator.update_status(False, "TRIGGERED")
            else:
                indicator.update_status(False, "INACTIVE")

    def update_circuit_breaker(
        self,
        strategy_name: str,
        enabled: bool,
        reason: str = "",
    ) -> None:
        """Update a single circuit breaker indicator."""
        if (indicator := _safe_query(self, f"#cb-{strategy_name}", _StatusIndicator)) is not None:
            indicator.update_status(enabled, reason)

    def update_circuit_breakers(self, states: list[CircuitBreakerState]) -> None:
        """Batch-update all circuit breaker states."""
        for state in states:
            self.update_circuit_breaker(
                state.strategy_name,
                state.enabled,
                state.reason,
            )
