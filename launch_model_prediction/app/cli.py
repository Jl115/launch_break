"""CLI commands using Typer and Rich.

Thin layer; all business logic lives in app.actions.
"""
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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
from shared import CATEGORY_LABELS, JSON_DIR, MODELS_DIR

__all__ = ["app"]

app = typer.Typer(help="Lunch Attendance Prediction System", rich_markup_mode="rich")
console = Console()


@app.command("parse")
def parse_command(
    image: Path = typer.Argument(help="Path to a scanned menu image"),
    output_dir: Path = typer.Option(JSON_DIR, help="Directory to write JSON output"),
) -> None:
    """Parse a single scanned menu image into structured JSON."""
    result = parse_single(image, output_dir)
    if result.success:
        console.print(f"[green]✅ {result.message}[/green]")
    else:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)


@app.command("parse-batch")
def parse_batch_command(
    images_dir: Path = typer.Option(help="Directory containing scanned images"),
    output_dir: Path = typer.Option(JSON_DIR, help="Directory to write JSON output"),
) -> None:
    """Parse all images in a directory into structured JSON."""
    result = parse_batch(images_dir, output_dir)
    if result.success:
        console.print(f"[green]✅ {result.message}[/green]")
        for p in result.data.get("paths", []):
            console.print(f"  - {p}")
    else:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)


@app.command("data")
def data_command(
    data_dir: Path = typer.Option(JSON_DIR, help="Directory with historical JSON data"),
) -> None:
    """View and validate loaded JSON data."""
    result = view_and_validate_data(data_dir)
    if not result.success:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)

    report = result.data
    console.print(f"[cyan]Total sheets:[/cyan] {report['total_sheets']}")
    console.print(f"[cyan]Valid sheets:[/cyan] {report['valid_sheets']}")
    console.print(f"[cyan]Total rows:[/cyan] {report['rows']}")

    if report.get("issues"):
        console.print("\n[yellow]Validation Issues:[/yellow]")
        for issue in report["issues"]:
            console.print(f"  - {issue}")
    else:
        console.print("\n[green]No validation issues found.[/green]")

    # Show table preview (only essential columns)
    sample = report.get("sample", [])
    if sample:
        df_table = Table(title="Data Preview (first 10 rows)")
        df_table.add_column("Date")
        df_table.add_column("Weekday")
        df_table.add_column("Holiday")
        for cat in CATEGORY_LABELS:
            df_table.add_column(cat, justify="right")

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
            vals += [str(row.get(cat, "")) for cat in CATEGORY_LABELS]
            df_table.add_row(*vals)
        console.print(df_table)


@app.command("train")
def train_command(
    data_dir: Path = typer.Option(JSON_DIR, help="Directory with historical JSON data"),
    output: Path = typer.Option(MODELS_DIR / "predictor.jobml", help="Model output path"),
    cross_validate: bool = typer.Option(True, help="Run TimeSeriesSplit CV before training"),
) -> None:
    """Train the attendance prediction model."""
    console.print("[cyan]Training...[/cyan]")
    result = train(data_dir, output, cross_validate)
    if not result.success:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]✅ {result.message}[/green]")
    cv = result.data.get("cv_scores", {})
    if isinstance(cv, dict) and cv:
        table = Table(title="Cross-Validation MAE per Category")
        table.add_column("Category", style="bold")
        for cat, vals in cv.items():
            if isinstance(vals, list):
                table.add_column(cat, justify="right")
        # Re-structure for display
        splits = max((len(v) for v in cv.values() if isinstance(v, list)), default=0)
        for i in range(splits):
            row_vals = []
            for cat in CATEGORY_LABELS:
                v = cv.get(cat, [])
                row_vals.append(f"{v[i]:.2f}" if isinstance(v, list) and i < len(v) else "")
            table.add_row(f"Split {i + 1}", *row_vals)
        console.print(table)


@app.command("predict")
def predict_command(
    month: str = typer.Argument(help="Target month as YYYY-MM (e.g. 2026-04)"),
    model_path: Path = typer.Option(MODELS_DIR / "predictor.jobml", help="Trained model path"),
) -> None:
    """Predict attendance for each weekday category in a given month."""
    result = predict(month, model_path)
    if not result.success:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"🍽️ Predicted Attendance for {month}")
    table.add_column("Week", style="bold")
    table.add_column("Date", style="bold")
    table.add_column("Weekday", style="bold")
    for cat in CATEGORY_LABELS:
        table.add_column(cat, justify="right")
    for row in result.data.get("table", []):
        table.add_row(
            row["week"],
            row["date"],
            row["weekday"],
            *[str(row[cat]) for cat in CATEGORY_LABELS],
        )
    console.print(table)


@app.command("evaluate")
def evaluate_command(
    data_dir: Path = typer.Option(JSON_DIR, help="Directory with historical JSON data"),
    model_path: Path = typer.Option(MODELS_DIR / "predictor.jobml", help="Trained model path"),
    test_size: float = typer.Option(0.2, help="Fraction to hold out for evaluation"),
) -> None:
    """Evaluate a saved model on historical data."""
    result = evaluate(data_dir, model_path, test_size)
    if not result.success:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)

    metrics = result.data.get("metrics", {})
    table = Table(title="Model Evaluation (Hold-Out Test Set)")
    table.add_column("Category", style="bold")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    for cat in CATEGORY_LABELS:
        vals = metrics.get(cat, {})
        table.add_row(cat, f"{vals.get('MAE', 0):.2f}", f"{vals.get('RMSE', 0):.2f}")
    console.print(table)


@app.command("status")
def status_command() -> None:
    """Show system status: data availability, model presence."""
    result = get_status()
    if not result.success:
        console.print(f"[red]❌ {result.message}[/red]")
        raise typer.Exit(code=1)

    d = result.data
    table = Table(title="📊 System Status")
    table.add_column("Item", style="bold")
    table.add_column("Status", justify="right")
    table.add_row("Historical Weeks", str(d["weeks"]))
    table.add_row("Can Train Model", "[green]Yes[/green]" if d["can_train"] else "[red]No[/red]")
    table.add_row(
        "Model Trained", "[green]Yes[/green]" if d["model_exists"] else "[red]No[/red]"
    )
    console.print(table)


if __name__ == "__main__":
    app()
