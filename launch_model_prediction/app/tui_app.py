"""Textual TUI for the Lunch Attendance Prediction System."""

from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)

from textual_plotext import PlotextPlot

from app.actions import (
    ActionResult,
    evaluate,
    get_status,
    parse_batch,
    parse_single,
    predict,
    train,
    view_and_validate_data,
)
from shared import JSON_DIR, MODELS_DIR

__all__ = ["LunchTUIApp"]

_CATS = ["Erw", "Ki", "MA", "MA-Ki"]
_WEEKDAY_NAMES = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]


# ── Helpers ───────────────────────────────────────────────────────────────

def _log_markup(log: RichLog, text: str) -> None:
    """Write markup-coloured text into a RichLog."""
    log.write(Text.from_markup(text))


# ── Navigation mixin ────────────────────────────────────────────────────

class NavigableScreen(Screen):
    """Base screen with h/j/k/l and arrow-key navigation + back binding."""

    BINDINGS = [
        ("h", "back", "Back"),
        ("q", "quit", "Quit"),
    ]

    def _focus_next(self) -> None:
        """Move focus to the next focusable widget."""
        self.screen.focus_next()

    def _focus_previous(self) -> None:
        """Move focus to the previous focusable widget."""
        self.screen.focus_previous()

    # Per-screen, we wire j/k as pressing the focused widget.
    # To avoid shadowing DataTable inner navigation we skip j/k when
    # a DataTable has focus.

    def action_focus_next(self) -> None:
        from textual.widgets import DataTable as _DataTable
        focused = self.app.focused
        if isinstance(focused, _DataTable):
            # DataTable uses up/down natively – do nothing so arrows keep working.
            return
        self._focus_next()

    def action_focus_previous(self) -> None:
        from textual.widgets import DataTable as _DataTable
        focused = self.app.focused
        if isinstance(focused, _DataTable):
            return
        self._focus_previous()

    def action_back(self) -> None:
        """Pop screen unless we are on the main menu."""
        if not isinstance(self, MainMenuScreen):
            self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit()


# ── Result helper ───────────────────────────────────────────────────────

class _ResultMixin:
    def _show(self, result: ActionResult, log: RichLog) -> None:
        log.clear()
        if result.success:
            _log_markup(log, f"[green]Success:[/green] {result.message}")
        else:
            _log_markup(log, f"[red]Error:[/red] {result.message}")


# ── Main Menu ───────────────────────────────────────────────────────────

class MainMenuScreen(NavigableScreen):
    """Root menu with navigation buttons."""

    BINDINGS = [
        ("j", "focus_next", "Down"),
        ("k", "focus_previous", "Up"),
        ("l", "press_focused", "Enter"),
        ("down", "focus_next", "Down"),
        ("up", "focus_previous", "Up"),
        ("right", "press_focused", "Enter"),
        ("enter", "press_focused", "Enter"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main-container"):
            yield Static(
                "\n[bold magenta]\U0001f37d  Lunch Attendance Prediction System[/]\n",
                classes="title",
            )
            with Container(id="menu-buttons"):
                yield Button("1. Parse Single Image", id="btn-parse-single")
                yield Button("2. Parse Batch of Images", id="btn-parse-batch")
                yield Button("3. View & Validate Data", id="btn-data-view")
                yield Button("4. Train Model", id="btn-train")
                yield Button("5. Predict Month", id="btn-predict")
                yield Button("6. Evaluate Model", id="btn-evaluate")
                yield Button("7. System Status", id="btn-status")
                yield Button("q. Quit", id="btn-quit", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        mapping = {
            "btn-parse-single": "parse_single",
            "btn-parse-batch": "parse_batch",
            "btn-data-view": "data_view",
            "btn-train": "train",
            "btn-predict": "predict",
            "btn-evaluate": "evaluate",
            "btn-status": "status",
            "btn-quit": "quit",
        }
        if bid == "btn-quit":
            self.app.exit()
        elif bid in mapping:
            self.app.push_screen(mapping[bid])

    def action_press_focused(self) -> None:
        focused = self.app.focused
        if isinstance(focused, Button):
            focused.press()


# ── Screens with uniform Back navigation ────────────────────────────────

class _BackableScreen(NavigableScreen, _ResultMixin):
    """Mixin that provides a Back button and uniform compose pattern."""

    BINDINGS = [
        ("j", "focus_next", "Down"),
        ("k", "focus_previous", "Up"),
        ("l", "press_focused", "Enter"),
        ("down", "focus_next", "Down"),
        ("up", "focus_previous", "Up"),
        ("right", "press_focused", "Enter"),
        ("enter", "press_focused", "Enter"),
        ("h", "back", "Back"),
        ("left", "back", "Back"),
        ("escape", "back", "Back"),
    ]

    def action_press_focused(self) -> None:
        focused = self.app.focused
        if isinstance(focused, Button):
            focused.press()
        elif isinstance(focused, Input):
            focused.action_submit()


# ── Parse Single ────────────────────────────────────────────────────────

class ParseSingleScreen(_BackableScreen):
    """Screen to parse a single image."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]Parse Single Image[/bold]")
            yield Label("Image path:")
            yield Input(placeholder="data/images/menu_01.jpg", id="input-image")
            with Horizontal():
                yield Button("Parse", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        image_path = Path(self.query_one("#input-image", Input).value.strip())
        log = self.query_one("#log", RichLog)
        if not image_path.exists():
            self._show(ActionResult(False, f"File not found: {image_path}"), log)
            return
        self._show(ActionResult(True, "Running OCR..."), log)
        result = parse_single(image_path)
        self._show(result, log)


# ── Parse Batch ─────────────────────────────────────────────────────────

class ParseBatchScreen(_BackableScreen):
    """Screen to parse a batch of images."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]Parse Batch of Images[/bold]")
            yield Label("Images directory:")
            yield Input(placeholder="data/images", id="input-dir")
            with Horizontal():
                yield Button("Parse Batch", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        dir_path = Path(self.query_one("#input-dir", Input).value.strip() or "data/images")
        log = self.query_one("#log", RichLog)
        if not dir_path.exists() or not dir_path.is_dir():
            self._show(ActionResult(False, f"Directory not found: {dir_path}"), log)
            return
        self._show(ActionResult(True, "Running batch OCR..."), log)
        result = parse_batch(dir_path)
        self._show(result, log)
        if result.success and result.data:
            paths = result.data.get("paths", [])
            for p in paths:
                _log_markup(log, f"  - {p}")


# ── Data View (with charts) ─────────────────────────────────────────────

class DataViewScreen(_BackableScreen):
    """Screen to view, validate, and chart loaded JSON data."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]View & Validate Data[/bold]")
            with Horizontal():
                yield Button("Load Data", id="btn-load", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
            yield Static("[bold]Preview (first 10 rows):[/bold]")
            yield DataTable(id="preview-table")

            with Horizontal():
                with Vertical(classes="chart-box"):
                    yield Static("[bold]Avg Attendance per Weekday[/bold]")
                    yield PlotextPlot(id="weekday-chart")
                with Vertical(classes="chart-box"):
                    yield Static("[bold]Weekly Trend[/bold]")
                    yield PlotextPlot(id="trend-chart")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        log = self.query_one("#log", RichLog)
        table = self.query_one("#preview-table", DataTable)
        table.clear()
        table.columns.clear()

        result = view_and_validate_data()
        self._show(result, log)
        if not result.success or not result.data:
            return

        report = result.data

        # ── Summary log with proper colours ──
        _log_markup(log, "")
        _log_markup(log, f"[cyan]Total sheets:[/cyan] {report['total_sheets']}")
        _log_markup(log, f"[cyan]Valid sheets:[/cyan] {report['valid_sheets']}")
        _log_markup(log, f"[cyan]Total rows:[/cyan] {report['rows']}")
        if report.get("issues"):
            _log_markup(log, "\n[yellow]Validation Issues:[/yellow]")
            for issue in report["issues"]:
                _log_markup(log, f"  [yellow]-[/yellow] {issue}")
        else:
            _log_markup(log, "\n[green]No validation issues found.[/green]")

        # ── Preview table (essential columns only) ──
        sample = report.get("sample", [])
        if sample:
            table.add_column("Date")
            table.add_column("Weekday")
            table.add_column("Holiday")
            for cat in _CATS:
                table.add_column(cat)
            _WD_NAMES = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]
            for row in sample:
                raw_date = row.get("date")
                date_str = (
                    raw_date.strftime("%d.%m.%Y")
                    if hasattr(raw_date, "strftime")
                    else str(raw_date)
                )
                wd_idx = row.get("day_of_week", 0)
                weekday = (
                    _WD_NAMES[int(wd_idx)]
                    if isinstance(wd_idx, int) and 0 <= int(wd_idx) <= 4
                    else str(wd_idx)
                )
                holiday = "Yes" if row.get("is_holiday") else "No"
                vals = [date_str, weekday, holiday]
                vals += [str(row.get(cat, "")) for cat in _CATS]
                table.add_row(*vals)

        # ── Charts ──
        self._render_weekday_chart(report.get("weekday_totals", {}))
        self._render_trend_chart(
            report.get("weekly_labels", []),
            report.get("weekly_trends", {}),
        )

    def _render_weekday_chart(self, totals: dict[str, dict[str, float]]) -> None:
        widget = self.query_one("#weekday-chart", PlotextPlot)
        plt = widget.plt
        plt.clear_data()
        if not totals:
            plt.title("No data")
            widget.refresh()
            return

        names = list(totals.keys())
        # Grouped bar: one bar per category, clustered by weekday
        bar_width = 2
        for ci, cat in enumerate(_CATS):
            vals = [totals.get(wd, {}).get(cat, 0.0) for wd in names]
            offset = [n + ci * bar_width for n in range(len(names))]
            plt.bar(
                offset,
                vals,
                label=cat,
                width=bar_width,
            )
        plt.xticks(range(len(names)), names)
        plt.title("Avg Attendance per Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Avg Count")
        plt.plotsize(40, 12)
        widget.refresh()

    def _render_trend_chart(self, labels: list[str], trends: dict[str, list[float]]) -> None:
        widget = self.query_one("#trend-chart", PlotextPlot)
        plt = widget.plt
        plt.clear_data()
        if not labels or not trends:
            plt.title("No data")
            widget.refresh()
            return

        x_ticks = range(len(labels))
        for cat in _CATS:
            vals = trends.get(cat, [])
            if not vals:
                continue
            plt.plot(x_ticks, vals, label=cat, marker="small")

        # Thin out x labels if many
        if len(labels) > 20:
            step = len(labels) // 10 or 1
            tick_pos = list(range(0, len(labels), step))
            tick_lbl = [labels[i] for i in tick_pos]
            plt.xticks(tick_pos, tick_lbl)
        else:
            plt.xticks(x_ticks, labels)

        plt.title("Weekly Attendance Trend")
        plt.xlabel("Week")
        plt.ylabel("Total Count")
        plt.plotsize(40, 12)
        widget.refresh()


# ── Train ───────────────────────────────────────────────────────────────

class TrainScreen(_BackableScreen):
    """Screen to train the model."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]Train Model[/bold]")
            with Horizontal():
                yield Button("Train", id="btn-train", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        log = self.query_one("#log", RichLog)
        log.clear()
        _log_markup(log, "[cyan]Training in progress...[/cyan]")
        result = train(cross_validate=True)
        self._show(result, log)
        if result.success and result.data:
            cv = result.data.get("cv_scores", {})
            if isinstance(cv, dict) and cv:
                _log_markup(log, "\n[bold]Cross-Validation MAE per Category[/bold]")
                for cat, vals in cv.items():
                    if isinstance(vals, list):
                        _log_markup(
                            log, f"  {cat}: {[round(v, 2) for v in vals]}"
                        )
                    else:
                        _log_markup(log, f"  {cat}: {vals}")
            _log_markup(
                log, f"\nModel saved to: {result.data.get('model_path')}"
            )


# ── Predict ─────────────────────────────────────────────────────────────

class PredictScreen(_BackableScreen):
    """Screen to predict attendance for a month."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]Predict Month[/bold]")
            yield Label("Month (MM.YYYY):")
            yield Input(placeholder="04.2026", id="input-month")
            with Horizontal():
                yield Button("Predict", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
            yield Static("[bold]Predicted Attendance:[/bold]")
            yield DataTable(id="pred-table")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        month = self.query_one("#input-month", Input).value.strip()
        log = self.query_one("#log", RichLog)
        table = self.query_one("#pred-table", DataTable)
        table.clear()
        table.columns.clear()

        result = predict(month)
        self._show(result, log)
        if not result.success or not result.data:
            return

        table.add_column("Week")
        table.add_column("Date")
        table.add_column("Weekday")
        for cat in _CATS:
            table.add_column(cat)
        for row in result.data.get("table", []):
            table.add_row(
                row["week"],
                row["date"],
                row["weekday"],
                *[str(row[cat]) for cat in _CATS],
            )


# ── Evaluate ────────────────────────────────────────────────────────────

class EvaluateScreen(_BackableScreen):
    """Screen to evaluate the trained model."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]Evaluate Model[/bold]")
            with Horizontal():
                yield Button("Evaluate", id="btn-run", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
            yield DataTable(id="eval-table")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        log = self.query_one("#log", RichLog)
        table = self.query_one("#eval-table", DataTable)
        table.clear()
        table.columns.clear()

        result = evaluate()
        self._show(result, log)
        if not result.success or not result.data:
            return

        metrics = result.data.get("metrics", {})
        table.add_column("Category")
        table.add_column("MAE")
        table.add_column("RMSE")
        for cat in _CATS:
            vals = metrics.get(cat, {})
            table.add_row(
                cat,
                f"{vals.get('MAE', 0):.2f}",
                f"{vals.get('RMSE', 0):.2f}",
            )


# ── Status ──────────────────────────────────────────────────────────────

class StatusScreen(_BackableScreen):
    """Screen showing system status."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="screen-container"):
            yield Static("[bold]System Status[/bold]")
            with Horizontal():
                yield Button("Refresh", id="btn-refresh", variant="primary")
                yield Button("Back", id="btn-back")
            yield RichLog(id="log")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        log = self.query_one("#log", RichLog)
        log.clear()
        result = get_status()
        self._show(result, log)
        if result.success and result.data:
            d = result.data
            _log_markup(log, f"[cyan]Historical Weeks:[/cyan]   {d['weeks']}")
            _log_markup(
                log,
                "[cyan]Can Train Model:[/cyan]    "
                f"{'[green]Yes[/green]' if d['can_train'] else '[red]No[/red]'}"
            )
            _log_markup(
                log,
                "[cyan]Model Trained:[/cyan]      "
                f"{'[green]Yes[/green]' if d['model_exists'] else '[red]No[/red]'}"
            )


# ── App ─────────────────────────────────────────────────────────────────

class LunchTUIApp(App[None]):
    """Textual App wrapping the lunch prediction system."""

    CSS = """
    Screen { align: center middle; }
    #main-container { width: 60; height: auto; border: solid green; padding: 1 2; }
    #screen-container { width: 90; height: auto; border: solid blue; padding: 1 2; }
    #menu-buttons { layout: vertical; height: auto; }
    Button { width: 100%; margin: 1; }
    .title { content-align: center middle; text-style: bold; }
    DataTable { height: 12; margin: 1 0; }
    RichLog { height: 8; border: solid yellow; margin: 1 0; }
    PlotextPlot { height: 14; margin: 1 0; }
    .chart-box { width: 1fr; height: auto; border: solid gray; padding: 1; }
    """

    SCREENS = {
        "parse_single": ParseSingleScreen,
        "parse_batch": ParseBatchScreen,
        "data_view": DataViewScreen,
        "train": TrainScreen,
        "predict": PredictScreen,
        "evaluate": EvaluateScreen,
        "status": StatusScreen,
    }

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())
