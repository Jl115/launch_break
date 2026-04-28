"""Domain Pydantic models for the lunch attendance prediction system."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "Header",
    "Signup",
    "DailyMenu",
    "MenuSheet",
    "DailyAttendance",
    "AttendancePrediction",
]


class Header(BaseModel):
    """Header metadata parsed from a menu sheet."""

    facility: str
    document_type: str
    date: str


class Signup(BaseModel):
    """Individual signup entry."""

    name: str
    erwachsene: int = Field(..., ge=0)
    kinder: int = Field(..., ge=0)


class DailyMenu(BaseModel):
    """Menu and attendance for a single weekday."""

    status: str | None = None
    menu: list[str] | None = None
    signups: list[Signup] | None = None
    summary_bottom: dict[str, int] | None = None


class MenuSheet(BaseModel):
    """Full weekly menu sheet with all days."""

    header: Header
    schedule: dict[str, DailyMenu]
    weekly_totals: dict[str, int] | None = None

    @field_validator("schedule")
    @classmethod
    def _extract_weekly_totals(cls, v: dict[str, DailyMenu]) -> dict[str, DailyMenu]:
        """Ensure schedule keys are valid weekday names."""
        allowed = {"Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag"}
        invalid = set(v.keys()) - allowed
        if invalid:
            raise ValueError(f"Invalid weekday keys: {invalid}")
        return v


class DailyAttendance(BaseModel):
    """Attendance summary for a single day."""

    date: str               # ISO date string
    weekday: str            # e.g. 'Montag'
    erw: int = Field(..., ge=0)
    ki: int = Field(..., ge=0)
    ma: int = Field(..., ge=0)
    ma_ki: int = Field(..., ge=0)


class AttendancePrediction(BaseModel):
    """Attendance prediction for a future month."""

    month: str              # e.g. '2026-04'
    predictions: list[DailyAttendance] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
