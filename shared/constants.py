"""Shared constants."""
from pathlib import Path

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "IMAGES_DIR",
    "JSON_DIR",
    "MODELS_DIR",
    "OLLAMA_HOST",
    "OLLAMA_PRIMARY_MODEL",
    "OLLAMA_FALLBACK_MODEL",
    "CATEGORY_LABELS",
]

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
IMAGES_DIR: Path = DATA_DIR / "images"
JSON_DIR: Path = DATA_DIR / "json"
MODELS_DIR: Path = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST: str = "http://localhost:11434"
OLLAMA_PRIMARY_MODEL: str = "deepseek-ocr:3b"
OLLAMA_FALLBACK_MODEL: str = "qwen3-vl:235b-cloud"

CATEGORY_LABELS: list[str] = ["Erw", "Ki", "MA", "MA-Ki"]
