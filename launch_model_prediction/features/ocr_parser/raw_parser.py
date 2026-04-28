"""OCR feature: raw text parser for menu sheets.

Transforms noisy OCR output into structured MenuSheet data using
regex heuristics aligned with the known form layout.
"""
import re
from pathlib import Path

from entities import DailyMenu, Header, MenuSheet, Signup
from shared import ParseError, get_logger

__all__ = ["RawTextParser"]

logger = get_logger(__name__)

_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"]
_SUMMARY_KEYS = ["Erw", "Ki", "MA", "MA-Ki"]

# OCR frequently miss-labels these keys (case-insensitive mapping).
_OCR_KEY_MAP: dict[str, str] = {
    "emu": "Erw",
    "ci": "Ki",
    "ki": "Ki",
    "k": "Ki",
    "da": "MA",
    "ma": "MA",
    "da-ci": "MA-Ki",
    "ma-ki": "MA-Ki",
    "daci": "MA-Ki",
    "erw": "Erw",
    "erwachsene": "Erw",
}


def _extract_date(text: str) -> str:
    pattern = re.compile(
        r"(\d{1,2})\.\s*(\d{1,2})\.\s*(?:\-\s*(?:.*?\.)?)?\s*(\d{4})"
    )
    m = pattern.search(text)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    m2 = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    if m2:
        return f"{m2.group(1)}.{m2.group(2)}.{m2.group(3)}"
    return ""


def _extract_facility(text: str) -> str:
    lines = text.splitlines()
    if lines:
        first = lines[0].strip()
        if "familien" in first.lower() or "kitas" in first.lower():
            return first
    m = re.search(r"(Familien\s*zentrum[^\n]*)", text, re.I)
    if m:
        return m.group(1).strip()
    return "Familienzentrum"


def _count_closed_days(text: str) -> int:
    """Heuristic: count how many days are marked geschlossen near the top."""
    cutoff = int(len(text) * 0.4)
    top = text[:cutoff]
    occurrences = len(re.findall(r"geschlossen", top, re.I))
    return occurrences


def _parse_summary_columns(text: str) -> dict[str, list[int]]:
    """Extract per-day counts from the bottom summary table.

    Looks for rows like:
        Em 7  Em 1  Em 8  Em 7
        Ci 13 Ci 3  Ci 13 Ci 11
    Returns dict mapping summary key -> list of per-day integer values.
    """
    result: dict[str, list[int]] = {}
    lines = text.splitlines()

    for key in _SUMMARY_KEYS:
        for line in lines:
            matches = re.findall(rf"\b{re.escape(key)}\b\s*[\-\|\:\.]?\s*(\d+)", line, re.I)
            if len(matches) >= 2:
                if key not in result:
                    result[key] = [int(m) for m in matches]

    for bad_key, good_key in _OCR_KEY_MAP.items():
        if good_key in result:
            continue
        for line in lines:
            matches = re.findall(
                rf"\b{re.escape(bad_key)}\b\s*[\-\|\:\.]?\s*(\d+)", line, re.I
            )
            if len(matches) >= 2:
                result[good_key] = [int(m) for m in matches]
                break

    return result


def _parse_signups(text: str) -> list[Signup]:
    """Extract all signup entries from the global names table."""
    signups: list[Signup] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if re.match(r"^[A-Za-zäöüÄÖÜß\- ]+\s*$", stripped) and len(stripped) >= 2:
            name = stripped
            erw = None
            ki = None
            j = i + 1
            while j < len(lines) and (erw is None or ki is None):
                n = lines[j].strip()
                if n.isdigit():
                    if erw is None:
                        erw = int(n)
                    elif ki is None:
                        ki = int(n)
                j += 1
            if erw is not None and ki is not None:
                signups.append(Signup(name=name, erwachsene=erw, kinder=ki))
            i = j
        else:
            i += 1
    return signups


class RawTextParser:
    """Parse noisy OCR text into a MenuSheet entity."""

    def parse(self, raw_text: str) -> MenuSheet:
        """Parse raw OCR text.

        Args:
            raw_text: The raw (cleaned) OCR text.

        Returns:
            Structured MenuSheet.
        """
        header = Header(
            facility=_extract_facility(raw_text),
            document_type="Menu - Anmeldeliste",
            date=_extract_date(raw_text),
        )

        closed_count = _count_closed_days(raw_text)
        summary_columns = _parse_summary_columns(raw_text)
        signups = _parse_signups(raw_text) or None

        schedule: dict[str, DailyMenu] = {}
        open_day_idx = 0
        for day_idx, day in enumerate(_WEEKDAYS):
            if day_idx < closed_count:
                schedule[day] = DailyMenu(status="geschlossen")
                continue

            summary: dict[str, int] = {}
            for key, values in summary_columns.items():
                if open_day_idx < len(values):
                    summary[key] = values[open_day_idx]
                elif values:
                    summary[key] = values[-1]

            open_day_idx += 1
            schedule[day] = DailyMenu(
                status="open",
                signups=signups,
                summary_bottom=summary or None,
            )

        if not schedule:
            raise ParseError("Could not extract any weekday schedule from OCR")

        weekly: dict[str, int] = {}
        for key in _SUMMARY_KEYS:
            weekly[key] = sum(
                (dm.summary_bottom or {}).get(key, 0) for dm in schedule.values()
            )
        return MenuSheet(header=header, schedule=schedule, weekly_totals=weekly)
