"""Data ingestion feature: JSON repository for loading historical data."""
import json
from abc import ABC, abstractmethod
from pathlib import Path

from entities import MenuSheet
from shared import JSON_DIR, get_logger

__all__ = ["MenuRepository"]

logger = get_logger(__name__)


class Repository(ABC):
    """Abstract repository interface."""

    @abstractmethod
    def load_all(self, directory: Path | None = None) -> list[MenuSheet]:
        """Load all available menu sheets."""


class MenuRepository(Repository):
    """Concrete repository that loads JSON files into MenuSheet entities."""

    def __init__(self, default_dir: Path = JSON_DIR) -> None:
        self.default_dir = default_dir

    def load_all(self, directory: Path | None = None) -> list[MenuSheet]:
        target = directory or self.default_dir
        if not target.exists():
            logger.warning("JSON directory %s does not exist", target)
            return []

        sheets: list[MenuSheet] = []
        for fp in sorted(target.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                sheet = MenuSheet.model_validate(data)
                sheets.append(sheet)
                logger.debug("Loaded %s", fp.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", fp.name, exc)
        logger.info("Loaded %d menu sheets from %s", len(sheets), target)
        return sheets
