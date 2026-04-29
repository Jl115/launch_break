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


import re


_MONTH_NAME_MAP: dict[str, int] = {
    "januar": 1,
    "februar": 2,
    "maerz": 3,
    "märz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}


def _extract_year(full: str) -> int | None:
    """Look for a 4-digit or 2-digit year at the end of the string."""
    # 4-digit year
    m = re.search(r"(\d{4})(?:\s|$)", full)
    if m:
        return int(m.group(1))
    # 2-digit year (not preceded by a day number with dot)
    m = re.search(r"(?<!\d\.\s)(\d{2})(?:\s|$)", full)
    if m:
        y = int(m.group(1))
        return 2000 + y if y < 50 else 1900 + y
    return None


def _extract_month_name(full: str) -> int | None:
    """Check for German month names in any case."""
    lowered = full.lower()
    for name, num in _MONTH_NAME_MAP.items():
        if name in lowered:
            return num
    return None


def _extract_month_name(full: str) -> int | None:
    """Check for German month names in any case."""
    lowered = full.lower()
    for name, num in _MONTH_NAME_MAP.items():
        if name in lowered:
            return num
    return None


def _parse_date(header_date: str) -> datetime | None:
    """Robust parser for Swiss/German PDF date headers.

    Supports examples found in production data:
      - '2.3. - 6.3.2020'            → 02.03.2020
      - '4.11. - 8.11.24'            → 04.11.2024
      - '02.09. - 06.09.2024'        → 02.09.2024
      - '2. - 6. Juni 2025'          → 02.06.2025
      - '3. - 7. März 2025'          → 03.03.2025
      - '28.4. - 2.5.25'            → 28.04.2025
      - '22.4.2024'                 → 22.04.2024
      - '9. - 13. 9. 2024'          → 09.09.2024
      - '10. - 14.6. 2024'           → 10.06.2024
      - '20. - 24. 3. 2024'         → 20.03.2024
      - '9. — 13.9.2024'           → 09.09.2024
      - '10. - 14. Februar 2025'   → 10.02.2025
      - '19. - 23. August 2024'     → 19.08.2024
    """
    raw = header_date.strip()

    # ── Normalise dashes & remove ALL spaces ─────────────────────────────
    raw = re.sub(r"\s*[–—\-]\s*", "-", raw)
    raw = raw.replace(" ", "")

    # ── 1. Try plain standalone date (no range) ──────────────────────────
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    # ── 2. Split range ────────────────────────────────────────────────────
    if "-" not in raw:
        # No range, no standalone match → try month-name fallback
        month = _extract_month_name(raw)
        year = _extract_year(raw)
        if month and year:
            day_match = re.match(r"(\d{1,2})", raw)
            if day_match:
                try:
                    return datetime(year, month, int(day_match.group(1)))
                except ValueError:
                    pass
        return None

    parts = raw.split("-", 1)
    start_raw = parts[0]
    end_raw = parts[1]

    # ── 3. Try parsing start directly ─────────────────────────────────────
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d.%m."):
        try:
            parsed = datetime.strptime(start_raw, fmt)
            # strptime defaults to 1900 for %d.%m. – fix if we can find a real year
            if parsed.year == 1900:
                yr = _extract_year(end_raw) or _extract_year(raw)
                if yr:
                    parsed = parsed.replace(year=yr)
            return parsed
        except ValueError:
            continue

    # ── 4. Start lacks month → parse end for month/year ──────────────────
    end_parsed = None
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            end_parsed = datetime.strptime(end_raw, fmt)
            break
        except ValueError:
            continue

    end_month = None
    end_year = None
    if end_parsed is not None:
        end_month = end_parsed.month
        end_year = end_parsed.year
    else:
        # Try month name + year extraction from anywhere in the raw string
        end_month = _extract_month_name(raw)
        end_year = _extract_year(raw)

    # If we still don't know month/year, give up
    if end_month is None or end_year is None:
        # Last resort: month name + year with no range
        month = _extract_month_name(raw)
        year = _extract_year(raw)
        if month and year:
            day_match = re.match(r"(\d{1,2})", raw)
            if day_match:
                try:
                    return datetime(year, month, int(day_match.group(1)))
                except ValueError:
                    pass
        return None

    # Extract start day from start_raw
    day_match = re.match(r"(\d{1,2})", start_raw)
    if not day_match:
        # Garbage like 'AB.' – can't determine start day
        return None

    try:
        return datetime(end_year, end_month, int(day_match.group(1)))
    except ValueError:
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
