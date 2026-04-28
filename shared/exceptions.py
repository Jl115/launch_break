"""Shared exceptions."""

__all__ = [
    "AppError",
    "OcrError",
    "ParseError",
    "ModelError",
    "InsufficientDataError",
]


class AppError(Exception):
    """Base application error."""


class OcrError(AppError):
    """Raised when OCR fails or returns unusable data."""


class ParseError(AppError):
    """Raised when raw text cannot be parsed into structured data."""


class ModelError(AppError):
    """Raised when ML model training or inference fails."""


class InsufficientDataError(AppError):
    """Raised when fewer than the required number of historical records exist."""
