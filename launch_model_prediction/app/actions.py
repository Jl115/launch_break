"""Reusable business actions for CLI and TUI.

All commands live here so both Typer and Textual can invoke them
without duplicating logic.
"""
from datetime import datetime
from pathlib import Path
from typing import Any

import holidays
import pandas as pd
from rich.console import Console

from app.container import Container
from features.prediction import ModelEvaluator, SklearnLunchPredictor
from shared import CATEGORY_LABELS, JSON_DIR, MODELS_DIR, InsufficientDataError

__all__ = [
    "ActionResult",
    "parse_single",
    "parse_batch",
    "view_and_validate_data",
    "train",
    "predict",
    "evaluate",
    "get_status",
]


class ActionResult:
    """Simple wrapper for action outcomes."""

    def __init__(self, success: bool, message: str = "", data: Any = None) -> None:
        self.success = success
        self.message = message
        self.data = data


def _load_container() -> Container:
    return Container()


def parse_single(image: Path, output_dir: Path = JSON_DIR) -> ActionResult:
    container = _load_container()
    try:
        json_path = container.ocr_service.parse_image(image, output_dir)
        return ActionResult(True, f"Parsed to {json_path}", {"path": json_path})
    except Exception as exc:
        return ActionResult(False, str(exc))


def parse_batch(images_dir: Path, output_dir: Path = JSON_DIR) -> ActionResult:
    container = _load_container()
    try:
        results = container.ocr_service.parse_batch(images_dir, output_dir)
        return ActionResult(
            True, f"Parsed {len(results)} images", {"paths": results, "count": len(results)}
        )
    except Exception as exc:
        return ActionResult(False, str(exc))


def view_and_validate_data(data_dir: Path = JSON_DIR) -> ActionResult:
    container = _load_container()
    sheets = container.menu_repository.load_all(data_dir)
    if not sheets:
        return ActionResult(False, "No JSON data found.")

    df = container.feature_engineer.build_dataset(sheets)
    if df.empty:
        return ActionResult(False, "Could not build dataset from JSONs.")

    # Validation report
    issues: list[str] = []
    total_sheets = len(sheets)
    valid_sheets = 0
    for sheet in sheets:
        if sheet.header.date:
            valid_sheets += 1
        else:
            issues.append(f"Sheet missing date: {sheet.header.facility}")
        for day_name, daily in sheet.schedule.items():
            if daily.summary_bottom is None and daily.status != "geschlossen":
                issues.append(f"{day_name}: missing summary_bottom")

    # ---- Chart data ------------------------------------------------------
    _WEEKDAY_NAMES = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]

    # 1. Bar chart: average attendance per weekday, per category
    weekday_totals: dict[str, dict[str, float]] = {}
    if {
        "day_of_week",
        *CATEGORY_LABELS,
    }.issubset(df.columns):
        means = df.groupby("day_of_week")[[*CATEGORY_LABELS]].mean()
        for i, wd in enumerate(_WEEKDAY_NAMES):
            if i in means.index:
                weekday_totals[wd] = {
                    cat: float(round(means.loc[i, cat], 1)) for cat in CATEGORY_LABELS
                }
            else:
                weekday_totals[wd] = {cat: 0.0 for cat in CATEGORY_LABELS}

    # 2. Line chart: total attendance per week (sum of all categories)
    weekly_trends: dict[str, list[float]] = {cat: [] for cat in CATEGORY_LABELS}
    weekly_labels: list[str] = []
    if {"year", "week_of_year"}.issubset(df.columns):
        weekly = df.groupby(["year", "week_of_year"])[[*CATEGORY_LABELS]].sum().reset_index()
        weekly = weekly.sort_values(["year", "week_of_year"])
        for _, row in weekly.iterrows():
            weekly_labels.append(f"{int(row['year'])}-W{int(row['week_of_year']):02d}")
            for cat in CATEGORY_LABELS:
                weekly_trends[cat].append(float(row[cat]))
    # ----------------------------------------------------------------------

    report = {
        "total_sheets": total_sheets,
        "valid_sheets": valid_sheets,
        "rows": len(df),
        "columns": list(df.columns),
        "sample": df.head(10).to_dict("records"),
        "issues": issues,
        "weekday_totals": weekday_totals,
        "weekly_labels": weekly_labels,
        "weekly_trends": weekly_trends,
        "df": df,
    }
    return ActionResult(True, f"Loaded {total_sheets} sheets, {len(df)} rows", report)


def train(
    data_dir: Path = JSON_DIR,
    output: Path = MODELS_DIR / "predictor.jobml",
    cross_validate: bool = True,
) -> ActionResult:
    container = _load_container()
    sheets = container.menu_repository.load_all(data_dir)
    if not sheets:
        return ActionResult(False, "No JSON data found. Run parse first.")

    df = container.feature_engineer.build_dataset(sheets)
    if df.empty:
        return ActionResult(False, "Could not build dataset from JSONs.")

    cv_scores: dict[str, Any] = {}
    if cross_validate:
        try:
            cv_scores = container.model_trainer.cross_validate(df)
        except InsufficientDataError as exc:
            cv_scores = {"warning": str(exc)}

    try:
        predictor = container.model_trainer.train(df, output)
        return ActionResult(
            True,
            f"Model trained and saved to {output}",
            {"cv_scores": cv_scores, "model_path": output},
        )
    except InsufficientDataError as exc:
        return ActionResult(False, str(exc))


def predict(month_str: str, model_path: Path = MODELS_DIR / "predictor.jobml") -> ActionResult:
    try:
        target = datetime.strptime(month_str, "%m.%Y")
    except ValueError:
        return ActionResult(False, "Invalid month format. Use MM.YYYY (e.g. 04.2026).")

    if not model_path.exists():
        return ActionResult(False, f"Model not found at {model_path}. Run train first.")

    predictor = SklearnLunchPredictor()
    predictor.load(model_path)

    year, month_num = target.year, target.month
    rows: list[dict] = []
    for day in pd.date_range(start=f"{year}-{month_num:02d}-01", periods=31, freq="D"):
        if day.month != month_num:
            break
        if day.weekday() >= 5:
            continue
        weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]
        wd_name = weekdays[day.weekday()]
        row = {
            "date": day,
            "year": day.year,
            "month": day.month,
            "week_of_year": day.isocalendar().week,
            "day_of_week": day.weekday(),
            "is_holiday": day.date()
            in holidays.country_holidays("CH", subdiv="BE", years=range(2020, 2031)),
        }
        for w in weekdays:
            row[f"wd_{w}"] = 1 if w == wd_name else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return ActionResult(False, "No weekdays to predict in this month.")

    preds = predictor.predict(df)
    df = df.join(preds)

    # Build per-week, per-day rows with Swiss date format (dd.mm.yyyy)
    weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]

    df["iso_week"] = df["date"].apply(lambda d: d.isocalendar().week)
    unique_weeks = sorted(df["iso_week"].unique())
    week_label_map = {w: f"Week {i + 1}" for i, w in enumerate(unique_weeks)}

    table_data = []
    for _, row in df.iterrows():
        dt = row["date"]
        date_str = f"{dt.day:02d}.{dt.month:02d}.{dt.year}"  # dd.mm.yyyy
        iso_week = row["iso_week"]
        table_data.append({
            "week": week_label_map[iso_week],
            "date": date_str,
            "weekday": weekday_names[int(row["day_of_week"])],
            **{cat: int(row[cat]) for cat in CATEGORY_LABELS},
        })

    return ActionResult(
        True,
        f"Predictions for {month_str}",
        {"month": month_str, "table": table_data},
    )


def evaluate(
    data_dir: Path = JSON_DIR,
    model_path: Path = MODELS_DIR / "predictor.jobml",
    test_size: float = 0.2,
) -> ActionResult:
    if not model_path.exists():
        return ActionResult(False, f"Model not found at {model_path}.")

    container = _load_container()
    sheets = container.menu_repository.load_all(data_dir)
    df = container.feature_engineer.build_dataset(sheets)
    if df.empty or len(df) < 10:
        return ActionResult(False, "Not enough data for evaluation.")

    split_idx = int(len(df) * (1 - test_size))
    X = df.drop(columns=["date", *CATEGORY_LABELS])
    y = df[CATEGORY_LABELS]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    predictor = SklearnLunchPredictor()
    predictor.load(model_path)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(predictor, X_test, y_test)

    return ActionResult(True, "Evaluation complete", {"metrics": metrics})


def get_status() -> ActionResult:
    container = _load_container()
    sheets = container.menu_repository.load_all()
    num_weeks = len(sheets)
    can_train = num_weeks >= 20
    model_exists = (MODELS_DIR / "predictor.jobml").exists()
    return ActionResult(
        True,
        "System status retrieved",
        {
            "weeks": num_weeks,
            "can_train": can_train,
            "model_exists": model_exists,
        },
    )
