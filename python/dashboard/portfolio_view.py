"""
Portfolio summary widget for The Oracle TUI dashboard.

Displays bankroll, P&L, exposure, and position metrics in a
compact panel that the orchestrator can update via public methods.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.dom import DOMNode
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static


def _safe_query[W: Widget](node: DOMNode, selector: str, kind: type[W]) -> W | None:
    """Return ``node.query_one(selector, kind)`` or ``None`` on lookup failure.

    Preserves the pre-mount-tolerant swallow semantics shared across the
    dashboard widgets: a widget queried before its children are mounted (or
    any other lookup error) yields ``None`` rather than raising.
    """
    try:
        return node.query_one(selector, kind)
    except Exception:
        return None


class _MetricCard(Static):
    """A single key-value metric displayed with label and value."""

    DEFAULT_CSS = """
    _MetricCard {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }
    _MetricCard .metric-label {
        color: $text-muted;
        text-style: dim;
    }
    _MetricCard .metric-value {
        color: $text;
        text-style: bold;
    }
    """

    def __init__(
        self,
        label: str,
        value: str = "--",
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._label_text = label
        self._value_text = value

    def compose(self) -> ComposeResult:
        yield Label(self._label_text, classes="metric-label")
        val_id = f"{self.id}-val" if self.id else None
        yield Label(self._value_text, classes="metric-value", id=val_id)

    def update_value(self, value: str) -> None:
        """Update the displayed value."""
        self._value_text = value
        val_label = self.query_one(".metric-value", Label)
        val_label.update(value)


class PortfolioView(Widget):
    """
    Portfolio summary panel.

    Shows bankroll, unrealized P&L, realized P&L, total exposure,
    daily P&L, max drawdown, and open position count.
    """

    DEFAULT_CSS = """
    PortfolioView {
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }
    PortfolioView #portfolio-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    PortfolioView .portfolio-row {
        height: auto;
        padding: 0;
    }
    PortfolioView .pnl-positive {
        color: $success;
    }
    PortfolioView .pnl-negative {
        color: $error;
    }
    """

    # Reactive properties for data binding
    bankroll: reactive[float] = reactive(10_000.0)
    unrealized_pnl: reactive[float] = reactive(0.0)
    realized_pnl: reactive[float] = reactive(0.0)
    daily_pnl: reactive[float] = reactive(0.0)
    total_exposure_pct: reactive[float] = reactive(0.0)
    max_drawdown_pct: reactive[float] = reactive(0.0)
    open_positions: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Label("PORTFOLIO", id="portfolio-title")
        with Horizontal(classes="portfolio-row"):
            yield _MetricCard("Bankroll", id="mc-bankroll")
            yield _MetricCard("Daily P&L", id="mc-daily-pnl")
            yield _MetricCard("Unrealized P&L", id="mc-unreal-pnl")
        with Horizontal(classes="portfolio-row"):
            yield _MetricCard("Realized P&L", id="mc-real-pnl")
            yield _MetricCard("Exposure", id="mc-exposure")
            yield _MetricCard("Drawdown", id="mc-drawdown")
        with Horizontal(classes="portfolio-row"):
            yield _MetricCard("Open Positions", id="mc-positions")

    def on_mount(self) -> None:
        """Render initial values on mount."""
        self._refresh_all()

    def _refresh_all(self) -> None:
        """Push current reactive values into the metric cards."""
        self._update_card("mc-bankroll", f"${self.bankroll:,.2f}")
        self._update_pnl_card("mc-daily-pnl", self.daily_pnl)
        self._update_pnl_card("mc-unreal-pnl", self.unrealized_pnl)
        self._update_pnl_card("mc-real-pnl", self.realized_pnl)
        self._update_card("mc-exposure", f"{self.total_exposure_pct:.1f}%")
        self._update_card("mc-drawdown", f"{self.max_drawdown_pct:.1f}%")
        self._update_card("mc-positions", str(self.open_positions))

    def _update_card(self, card_id: str, value: str) -> None:
        """Update a metric card by id."""
        if (card := _safe_query(self, f"#{card_id}", _MetricCard)) is not None:
            card.update_value(value)

    def _update_pnl_card(self, card_id: str, value: float) -> None:
        """Update a P&L card with color coding."""
        sign = "+" if value >= 0 else ""
        text = f"{sign}${value:,.2f}"
        self._update_card(card_id, text)

    # -- Reactive watchers ---------------------------------------------------

    def watch_bankroll(self, value: float) -> None:
        if self.is_mounted:
            self._update_card("mc-bankroll", f"${value:,.2f}")

    def watch_daily_pnl(self, value: float) -> None:
        if self.is_mounted:
            self._update_pnl_card("mc-daily-pnl", value)

    def watch_unrealized_pnl(self, value: float) -> None:
        if self.is_mounted:
            self._update_pnl_card("mc-unreal-pnl", value)

    def watch_realized_pnl(self, value: float) -> None:
        if self.is_mounted:
            self._update_pnl_card("mc-real-pnl", value)

    def watch_total_exposure_pct(self, value: float) -> None:
        if self.is_mounted:
            self._update_card("mc-exposure", f"{value:.1f}%")

    def watch_max_drawdown_pct(self, value: float) -> None:
        if self.is_mounted:
            self._update_card("mc-drawdown", f"{value:.1f}%")

    def watch_open_positions(self, value: int) -> None:
        if self.is_mounted:
            self._update_card("mc-positions", str(value))

    # -- Public update method for orchestrator --------------------------------

    def update_portfolio(
        self,
        *,
        bankroll: float | None = None,
        unrealized_pnl: float | None = None,
        realized_pnl: float | None = None,
        daily_pnl: float | None = None,
        total_exposure_pct: float | None = None,
        max_drawdown_pct: float | None = None,
        open_positions: int | None = None,
    ) -> None:
        """
        Batch-update portfolio metrics.

        The orchestrator calls this method to push new data into the widget.
        Only provided fields are updated; others retain their current value.
        """
        if bankroll is not None:
            self.bankroll = bankroll
        if unrealized_pnl is not None:
            self.unrealized_pnl = unrealized_pnl
        if realized_pnl is not None:
            self.realized_pnl = realized_pnl
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
        if total_exposure_pct is not None:
            self.total_exposure_pct = total_exposure_pct
        if max_drawdown_pct is not None:
            self.max_drawdown_pct = max_drawdown_pct
        if open_positions is not None:
            self.open_positions = open_positions
