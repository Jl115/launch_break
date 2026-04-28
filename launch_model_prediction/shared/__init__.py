"""Shared layer: utilities reusable across the entire application."""
from shared.constants import (
    CATEGORY_LABELS,
    DATA_DIR,
    IMAGES_DIR,
    JSON_DIR,
    MODELS_DIR,
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_HOST,
    OLLAMA_PRIMARY_MODEL,
    ROOT_DIR,
)
from shared.exceptions import (
    AppError,
    InsufficientDataError,
    ModelError,
    OcrError,
    ParseError,
)
from shared.logger import get_logger
from shared.utils import clean_ocr_output

__all__ = [
    "CATEGORY_LABELS",
    "DATA_DIR",
    "IMAGES_DIR",
    "JSON_DIR",
    "MODELS_DIR",
    "OLLAMA_FALLBACK_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_PRIMARY_MODEL",
    "ROOT_DIR",
    "AppError",
    "InsufficientDataError",
    "ModelError",
    "OcrError",
    "ParseError",
    "get_logger",
    "clean_ocr_output",
]
