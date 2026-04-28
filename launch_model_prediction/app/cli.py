"""CLI commands using Typer and Rich."""
from datetime import datetime
from pathlib import Path

import holidays
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from app.config import AppConfig
from app.container import Container
from features.data_ingestion import FeatureEngineer, MenuRepository
from features.ocr_parser import OcrParserService
from features.prediction import ModelEvaluator, ModelTrainer, SklearnLunchPredictor
from shared import (
    JSON_DIR,
    MODELS_DIR,
    CATEGORY_LABELS,
    InsufficientDataError,
    OcrError,
    ParseError,
    get_logger,
)

__all__ = ["app"]

app = typer.Typer(help="Lunch Attendance Prediction System", rich_markup_mode="rich")
console = Console()
logger = get_logger(__name__)


def _load_container() -> Container:
    return Container()


@app.command("parse")
def parse_command(
    image: Path = typer.Argument(help="Path to a scanned menu image"),
    output_dir: Path = typer.Option(JSON_DIR, help="Directory to write JSON output"),
) -> None:
    """Parse a single scanned menu image into structured JSON."""
    container = _load_container()
    try:
        json_path = container.ocr_service.parse_image(image, output_dir)
        console.print(f"[green]✅ Successfully parsed to {json_path}[/green]")
    except (OcrError, ParseError) as exc:
        console.print(f"[red]❌ {exc}[/red]")
        raise typer.Exit(code=1)


@app.command("parse-batch")
def parse_batch_command(
    images_dir: Path = typer.Option(help="Directory containing scanned images"),
    output_dir: Path = typer.Option(JSON_DIR, help="Directory to write JSON output"),
) -> None:
    """Parse all images in a directory into structured JSON."""
    container = _load_container()
    results = container.ocr_service.parse_batch(images_dir, output_dir)
    console.print(f"[green]✅ Parsed {len(results)} images to {output_dir}[/green]")


@app.command("train")
def train_command(
    data_dir: Path = typer.Option(JSON_DIR, help="Directory with historical JSON data"),
    output: Path = typer.Option(MODELS_DIR / "predictor.jobml", help="Model output path"),
    cross_validate: bool = typer.Option(True, help="Run TimeSeriesSplit CV before training"),
) -> None:
    """Train the attendance prediction model."""
    container = _load_container()
    sheets = container.menu_repository.load_all(data_dir)
    if not sheets:
        console.print("[red]❌ No JSON data found. Run parse first.[/red]")
        raise typer.Exit(code=1)

    df = container.feature_engineer.build_dataset(sheets)
    if df.empty:
        console.print("[red]❌ Could not build dataset from JSONs.[/red]")
        raise typer.Exit(code=1)

    if cross_validate:
        console.print("[cyan]Running TimeSeriesSplit Cross-Validation...[/cyan]")
        try:
            cv_scores = container.model_trainer.cross_validate(df)
            table = Table(title="Cross-Validation MAE per Category")
            table.add_column("Category", style="bold")
            for i in range(container.config.n_splits_cv):
                table.add_column(f"Split {i+1}", justify="right")
            for cat in CATEGORY_LABELS:
                vals = cv_scores.get(cat, [])
                table.add_row(cat, *[f"{v:.2f}" for v in vals])
            console.print(table)
        except InsufficientDataError as exc:
            console.print(f"[yellow]⚠️ {exc}[/yellow]")
            console.print("[yellow]Skipping CV, proceeding with full training...[/yellow]")

    console.print("[cyan]Training final model...[/cyan]")
    try:
        predictor = container.model_trainer.train(df, output)
        console.print(f"[green]✅ Model trained and saved to {output}[/green]")
    except InsufficientDataError as exc:
        console.print(f"[red]❌ {exc}[/red]")
        raise typer.Exit(code=1)


@app.command("predict")
def predict_command(
    month: str = typer.Argument(help="Target month as YYYY-MM (e.g. 2026-04)"),
    model_path: Path = typer.Option(MODELS_DIR / "predictor.jobml", help="Trained model path"),
) -> None:
    """Predict attendance for each weekday category in a given month."""
    try:
        target = datetime.strptime(month, "%Y-%m")
    except ValueError:
        console.print("[red]❌ Invalid month format. Use YYYY-MM.[/red]")
        raise typer.Exit(code=1)

    if not model_path.exists():
        console.print(f"[red]❌ Model not found at {model_path}. Run train first.[/red]")
        raise typer.Exit(code=1)

    predictor = SklearnLunchPredictor()
    predictor.load(model_path)

    # Build feature rows for each weekday in that month
    year, month_num = target.year, target.month
    rows: list[dict] = []
    for day in pd.date_range(start=f"{year}-{month_num:02d}-01", periods=31, freq="D"):
        if day.month != month_num:
            break
        if day.weekday() >= 5:  # skip weekends
            continue
        weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]
        wd_name = weekdays[day.weekday()]
        row = {
            "date": day,
            "year": day.year,
            "month": day.month,
            "week_of_year": day.isocalendar().week,
            "day_of_week": day.weekday(),
            "is_holiday": day.date() in holidays.CH(subdiv="BE", years=range(2020, 2031)),
        }
        # One-hot encode weekday
        for w in weekdays:
            row[f"wd_{w}"] = 1 if w == wd_name else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        console.print("[yellow]⚠️ No weekdays to predict in this month.[/yellow]")
        return

    preds = predictor.predict(df)
    # Aggregate by weekday category (average)
    df = df.join(preds)
    aggregated = df.groupby("day_of_week")[["Erw", "Ki", "MA", "MA-Ki"]].mean().round(0).astype(int)
    weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]

    table = Table(title=f"🍽️ Predicted Attendance for {month}")
    table.add_column("Weekday", style="bold")
    for cat in CATEGORY_LABELS:
        table.add_column(cat, justify="right")
    for i, row in aggregated.iterrows():
        table.add_row(
            weekday_names[int(i)],
            *[str(row[c]) for c in CATEGORY_LABELS],
        )
    console.print(table)


@app.command("evaluate")
def evaluate_command(
    data_dir: Path = typer.Option(JSON_DIR, help="Directory with historical JSON data"),
    model_path: Path = typer.Option(MODELS_DIR / "predictor.jobml", help="Trained model path"),
    test_size: float = typer.Option(0.2, help="Fraction to hold out for evaluation"),
) -> None:
    """Evaluate a saved model on historical data."""
    if not model_path.exists():
        console.print(f"[red]❌ Model not found at {model_path}.[/red]")
        raise typer.Exit(code=1)

    container = _load_container()
    sheets = container.menu_repository.load_all(data_dir)
    df = container.feature_engineer.build_dataset(sheets)
    if df.empty or len(df) < 10:
        console.print("[red]❌ Not enough data for evaluation.[/red]")
        raise typer.Exit(code=1)

    split_idx = int(len(df) * (1 - test_size))
    X = df.drop(columns=["date", *CATEGORY_LABELS])
    y = df[CATEGORY_LABELS]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    predictor = SklearnLunchPredictor()
    predictor.load(model_path)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(predictor, X_test, y_test)

    table = Table(title="Model Evaluation (Hold-Out Test Set)")
    table.add_column("Category", style="bold")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    for cat in CATEGORY_LABELS:
        table.add_row(
            cat,
            f"{metrics[cat]['MAE']:.2f}",
            f"{metrics[cat]['RMSE']:.2f}",
        )
    console.print(table)


@app.command("status")
def status_command() -> None:
    """Show system status: data availability, model presence."""
    container = _load_container()
    sheets = container.menu_repository.load_all()
    num_weeks = len(sheets)
    can_train = num_weeks >= 20
    model_exists = (MODELS_DIR / "predictor.jobml").exists()

    table = Table(title="📊 System Status")
    table.add_column("Item", style="bold")
    table.add_column("Status", justify="right")
    table.add_row("Historical Weeks", str(num_weeks))
    table.add_row("Can Train Model", "[green]Yes[/green]" if can_train else "[red]No[/red]")
    table.add_row("Model Trained", "[green]Yes[/green]" if model_exists else "[red]No[/red]")
    console.print(table)


if __name__ == "__main__":
    app()
