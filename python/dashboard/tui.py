"""
Terminal UI dashboard for The Oracle.

A Textual application with multiple panels: portfolio summary, trade log,
signal monitor, risk status, and strategy performance.  Run standalone
for visual testing:

    uv run python -m python.dashboard.tui
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Footer, Header, Label, Static

from python.dashboard.portfolio_view import PortfolioView, _safe_query
from python.dashboard.risk_view import CircuitBreakerState, RiskView
from python.dashboard.signal_view import SignalRow, SignalView

# ---------------------------------------------------------------------------
# Trade-log row type
# ---------------------------------------------------------------------------


@dataclass
class TradeLogRow:
    """A single entry in the trade log table."""

    timestamp: datetime
    market: str
    side: str
    model_prob: float
    market_price: float
    edge: float
    size: float
    status: str  # filled | shadow | rejected | partial | pending

    @property
    def time_str(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Strategy performance row type
# ---------------------------------------------------------------------------


@dataclass
class StrategyPerfRow:
    """Per-strategy performance summary."""

    strategy: str
    trade_count: int
    win_rate: float
    pnl: float
    brier_score: float | None = None  # None = not yet computed


# ---------------------------------------------------------------------------
# Trade log widget
# ---------------------------------------------------------------------------


_TRADE_COLUMNS: list[tuple[str, str]] = [
    ("time", "Time"),
    ("market", "Market"),
    ("side", "Side"),
    ("model_prob", "Model P"),
    ("mkt_price", "Mkt P"),
    ("edge", "Edge"),
    ("size", "Size ($)"),
    ("status", "Status"),
]


class TradeLogView(Static):
    """Scrollable table of recent trades and signals."""

    DEFAULT_CSS = """
    TradeLogView {
        height: 1fr;
        border: solid $accent;
        padding: 1 2;
    }
    TradeLogView #tradelog-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    TradeLogView DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("TRADE LOG", id="tradelog-title")
        yield DataTable(id="tradelog-table")

    def on_mount(self) -> None:
        table = self.query_one("#tradelog-table", DataTable)
        for key, label in _TRADE_COLUMNS:
            table.add_column(label, key=key)

    def add_trade(self, trade: TradeLogRow) -> None:
        """Prepend a trade to the top of the log."""
        table = self.query_one("#tradelog-table", DataTable)
        table.add_row(
            trade.time_str,
            trade.market,
            trade.side,
            f"{trade.model_prob:.1%}",
            f"{trade.market_price:.1%}",
            f"{trade.edge:+.1%}",
            f"${trade.size:,.0f}",
            trade.status,
        )

    def set_trades(self, trades: list[TradeLogRow]) -> None:
        """Replace all trades in the table."""
        table = self.query_one("#tradelog-table", DataTable)
        table.clear()
        for trade in trades:
            self.add_trade(trade)


# ---------------------------------------------------------------------------
# Strategy performance widget
# ---------------------------------------------------------------------------


_PERF_COLUMNS: list[tuple[str, str]] = [
    ("strategy", "Strategy"),
    ("trades", "Trades"),
    ("win_rate", "Win Rate"),
    ("pnl", "P&L"),
    ("brier", "Brier"),
]


class StrategyPerfView(Static):
    """Per-strategy performance summary table."""

    DEFAULT_CSS = """
    StrategyPerfView {
        height: auto;
        border: solid $accent;
        padding: 1 2;
    }
    StrategyPerfView #perfview-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    StrategyPerfView DataTable {
        height: auto;
        max-height: 12;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("STRATEGY PERFORMANCE", id="perfview-title")
        yield DataTable(id="perf-table")

    def on_mount(self) -> None:
        table = self.query_one("#perf-table", DataTable)
        for key, label in _PERF_COLUMNS:
            table.add_column(label, key=key)

    def update_performance(self, rows: list[StrategyPerfRow]) -> None:
        """Replace all strategy rows."""
        table = self.query_one("#perf-table", DataTable)
        table.clear()
        for row in rows:
            brier_str = f"{row.brier_score:.4f}" if row.brier_score is not None else "--"
            sign = "+" if row.pnl >= 0 else ""
            table.add_row(
                row.strategy,
                str(row.trade_count),
                f"{row.win_rate:.1%}",
                f"{sign}${row.pnl:,.2f}",
                brier_str,
                key=row.strategy,
            )


# ---------------------------------------------------------------------------
# Mode indicator widget
# ---------------------------------------------------------------------------


class ModeIndicator(Static):
    """Small widget that displays the current execution mode."""

    DEFAULT_CSS = """
    ModeIndicator {
        dock: top;
        width: 100%;
        height: 1;
        background: $panel;
        color: $text;
        text-align: center;
        text-style: bold;
        padding: 0 2;
    }
    ModeIndicator.mode-shadow {
        background: $primary-darken-2;
    }
    ModeIndicator.mode-paper {
        background: $warning-darken-2;
    }
    ModeIndicator.mode-live {
        background: $error-darken-1;
    }
    """

    def __init__(self, mode: str = "shadow", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mode = mode

    def on_mount(self) -> None:
        self._refresh_display()

    def set_mode(self, mode: str) -> None:
        """Update the displayed execution mode."""
        self._mode = mode
        self._refresh_display()

    def _refresh_display(self) -> None:
        mode_upper = self._mode.upper()
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        self.update(f"THE ORACLE  |  {mode_upper} MODE  |  {now}")
        self.remove_class("mode-shadow", "mode-paper", "mode-live")
        self.add_class(f"mode-{self._mode}")


# ---------------------------------------------------------------------------
# Main Textual App
# ---------------------------------------------------------------------------


class OracleDashboard(App[None]):
    """
    The Oracle — Terminal Trading Dashboard.

    Panels:
      - Header bar with app name, mode, and time
      - Portfolio summary (bankroll, P&L, exposure, drawdown)
      - Trade log (scrollable table of recent orders)
      - Signal monitor (active trade signals)
      - Risk status (limit gauges, kill switch, circuit breakers)
      - Strategy performance (per-strategy stats)
    """

    TITLE = "The Oracle"
    SUB_TITLE = "Prediction Market Trading Bot"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh_mock", "Refresh", show=True),
        Binding("d", "toggle_dark", "Toggle Dark", show=True),
    ]

    CSS = """
    #app-grid {
        layout: grid;
        grid-size: 2 3;
        grid-columns: 2fr 1fr;
        grid-rows: auto 1fr auto;
        grid-gutter: 1;
        padding: 1;
        height: 100%;
    }
    #portfolio-panel {
        column-span: 2;
    }
    #tradelog-panel {
        row-span: 1;
    }
    #right-stack {
        row-span: 1;
        height: 100%;
    }
    #bottom-left {
        height: auto;
    }
    #bottom-right {
        height: auto;
    }
    """

    def __init__(
        self,
        mode: str = "shadow",
        strategy_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._mode = mode
        self._strategy_names = strategy_names or ["weather"]

    def compose(self) -> ComposeResult:
        yield Header()
        yield ModeIndicator(mode=self._mode, id="mode-bar")

        with Container(id="app-grid"):
            yield PortfolioView(id="portfolio-panel")

            yield TradeLogView(id="tradelog-panel")

            with Vertical(id="right-stack"):
                yield SignalView(id="signal-panel")
                yield RiskView(
                    strategy_names=self._strategy_names,
                    id="risk-panel",
                )

            yield StrategyPerfView(id="bottom-left")

        yield Footer()

    def on_mount(self) -> None:
        """Populate with mock data for visual testing."""
        self._load_mock_data()
        self.set_interval(30, self._update_time)

    # -- Actions --------------------------------------------------------------

    def action_refresh_mock(self) -> None:
        """Reload mock data (bound to 'r' key)."""
        self._load_mock_data()

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    # -- Time update ----------------------------------------------------------

    def _update_time(self) -> None:
        """Refresh the mode bar timestamp."""
        if (mode_bar := _safe_query(self, "#mode-bar", ModeIndicator)) is not None:
            mode_bar._refresh_display()

    # -- Public API for orchestrator ------------------------------------------

    @property
    def portfolio(self) -> PortfolioView:
        """Access the portfolio widget."""
        return self.query_one("#portfolio-panel", PortfolioView)

    @property
    def trade_log(self) -> TradeLogView:
        """Access the trade log widget."""
        return self.query_one("#tradelog-panel", TradeLogView)

    @property
    def signal_monitor(self) -> SignalView:
        """Access the signal monitor widget."""
        return self.query_one("#signal-panel", SignalView)

    @property
    def risk_panel(self) -> RiskView:
        """Access the risk status widget."""
        return self.query_one("#risk-panel", RiskView)

    @property
    def strategy_perf(self) -> StrategyPerfView:
        """Access the strategy performance widget."""
        return self.query_one("#bottom-left", StrategyPerfView)

    def set_execution_mode(self, mode: str) -> None:
        """Update the execution mode display."""
        self._mode = mode
        if (mode_bar := _safe_query(self, "#mode-bar", ModeIndicator)) is not None:
            mode_bar.set_mode(mode)

    # -- Mock data for standalone testing -------------------------------------

    def _load_mock_data(self) -> None:
        """Populate all panels with representative mock data."""
        now = datetime.now(UTC)

        self.portfolio.update_portfolio(
            bankroll=10_245.50,
            daily_pnl=45.50,
            unrealized_pnl=12.30,
            realized_pnl=33.20,
            total_exposure_pct=18.7,
            max_drawdown_pct=2.3,
            open_positions=4,
        )

        trades = [
            TradeLogRow(now, "KXHIGHNYC-25MAR12-T52", "YES", 0.88, 0.75, 0.13, 150.00, "shadow"),
            TradeLogRow(now, "KXHIGHCHI-25MAR12-T40", "NO",  0.25, 0.42, 0.17, 85.00,  "shadow"),
            TradeLogRow(now, "KXHIGHMIA-25MAR12-T78", "YES", 0.92, 0.83, 0.09, 120.00, "filled"),
            TradeLogRow(now, "INX-25MAR-B5400",       "NO",  0.35, 0.55, 0.20, 200.00, "shadow"),
            TradeLogRow(now, "FED-25MAR-T450",        "YES", 0.72, 0.60, 0.12, 95.00,  "rejected"),
            TradeLogRow(now, "KXHIGHLAX-25MAR12-T68", "YES", 0.81, 0.70, 0.11, 110.00, "shadow"),
            TradeLogRow(now, "KXHIGHDEN-25MAR12-T55", "NO",  0.30, 0.45, 0.15, 75.00,  "shadow"),
            TradeLogRow(now, "TSA-25W11-B2M",         "YES", 0.78, 0.65, 0.13, 130.00, "shadow"),
        ]
        self.trade_log.set_trades(trades)

        signals = [
            SignalRow(
                "s1", "weather", "KXHIGHNYC-25MAR12-T52",
                "YES", 0.88, 0.75, 0.13, 0.91, 150.00, now,
            ),
            SignalRow(
                "s2", "weather", "KXHIGHCHI-25MAR12-T40",
                "NO", 0.25, 0.42, 0.17, 0.87, 85.00, now,
            ),
            SignalRow(
                "s3", "weather", "KXHIGHMIA-25MAR12-T78",
                "YES", 0.92, 0.83, 0.09, 0.95, 120.00, now,
            ),
        ]
        self.signal_monitor.update_signals(signals)

        self.risk_panel.update_risk(
            exposure_pct=18.7,
            daily_loss_pct=0.8,
            drawdown_pct=2.3,
        )
        self.risk_panel.update_kill_switch(active=False)
        self.risk_panel.update_circuit_breakers([
            CircuitBreakerState("weather", enabled=True),
        ])

        perf = [
            StrategyPerfRow("weather",  42, 0.667, 325.40, 0.1823),
        ]
        self.strategy_perf.update_performance(perf)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the dashboard for visual testing."""
    app = OracleDashboard(
        mode="shadow",
        strategy_names=["weather"],
    )
    app.run()


if __name__ == "__main__":
    main()
