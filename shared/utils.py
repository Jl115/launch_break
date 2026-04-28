"""Shared utilities."""
import re

__all__ = ["clean_ocr_output", "parse_german_date_range"]

_ANSI_ESCAPE = re.compile(r"(?:\x1b\[[0-9;]*[a-zA-Z]|\x0f|\x1b\]\d+;.*?\x07|\x1b\[\?\d+[hl])")
_GROUNDING_TAG = re.compile(r"<\|ref\|>.*?<\|/ref\|>", re.S)
_DET_TAG = re.compile(r"<\|det\|>.*?<\|/det\|>", re.S)

_MONTH_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12,
}


def clean_ocr_output(text: str) -> str:
    """Remove ANSI codes and grounding/det tags from OCR output."""
    text = _ANSI_ESCAPE.sub("", text)
    text = _GROUNDING_TAG.sub("", text)
    text = _DET_TAG.sub("", text)
    text = text.replace("\x0f", "").replace("\x1b", "").replace("\x07", "")
    return text.strip()
