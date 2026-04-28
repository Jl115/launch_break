"""Data ingestion feature: feature engineering for ML pipeline."""
from datetime import datetime

import holidays
import numpy as np
import pandas as pd

from entities import MenuSheet
from shared import CATEGORY_LABELS, get_logger

__all__ = ["FeatureEngineer"]

logger = get_logger(__name__)

_WEEKDAY_ORDER = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]
_CH_HOLIDAYS = holidays.CH(subdiv="BE", years=range(2020, 2031))


def _parse_date(header_date: str) -> datetime | None:
    """Attempt to parse a German date string like '2.3. - 6.3.2020'."""
    cleaned = header_date.strip().replace(" ", "")
    # Try start date
    if "-" in cleaned:
        start = cleaned.split("-")[0]
        for fmt in ("%d.%m.%Y", "%d.%m.", "%d.%m"):
            try:
                return datetime.strptime(start, fmt)
            except ValueError:
                continue
    else:
        for fmt in ("%d.%m.%Y", "%d.%m"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
    return None


class FeatureEngineer:
    """Transform MenuSheet entities into a DataFrame ready for ML."""

    def build_dataset(self, sheets: list[MenuSheet]) -> pd.DataFrame:
        """Build ML dataset from historical menu sheets.

        Args:
            sheets: List of MenuSheet entities.

        Returns:
            DataFrame with temporal features and attendance targets.
        """
        rows: list[dict] = []
        for sheet in sheets:
            week_start = _parse_date(sheet.header.date)
            if week_start is None:
                logger.warning("Could not parse date: %s", sheet.header.date)
                continue
            for day_name, daily in sheet.schedule.items():
                if daily.status == "geschlossen":
                    continue
                summary = daily.summary_bottom or {}
                if not summary:
                    continue
                # Determine actual date: offset from week_start
                day_idx = _WEEKDAY_ORDER.index(day_name) if day_name in _WEEKDAY_ORDER else 0
                actual_date = week_start + pd.Timedelta(days=day_idx)
                row = {
                    "date": actual_date,
                    "weekday": day_name,
                    "year": actual_date.year,
                    "month": actual_date.month,
                    "week_of_year": actual_date.isocalendar().week,
                    "day_of_week": actual_date.weekday(),
                    "is_holiday": actual_date.date() in _CH_HOLIDAYS,
                    "Erw": summary.get("Erw", 0),
                    "Ki": summary.get("Ki", 0),
                    "MA": summary.get("MA", 0),
                    "MA-Ki": summary.get("MA-Ki", 0),
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # One-hot encode weekday
        df = pd.get_dummies(df, columns=["weekday"], prefix="wd")
        # Sort chronologically
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Built dataset with shape %s", df.shape)
        return df
