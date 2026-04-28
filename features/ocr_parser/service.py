"""OCR feature service: orchestrates image → text → JSON."""
import json
from abc import ABC, abstractmethod
from pathlib import Path

from entities import MenuSheet
from features.ocr_parser.ollama_client import OllamaClient, OcrClient
from features.ocr_parser.raw_parser import RawTextParser
from shared import JSON_DIR, OcrError, ParseError, get_logger

__all__ = ["OcrParserService"]

logger = get_logger(__name__)


class OcrParserService(ABC):
    """Abstract interface for OCR parsing service."""

    @abstractmethod
    def parse_image(self, image_path: Path, output_dir: Path | None = None) -> Path:
        """Parse a single image into JSON.

        Args:
            image_path: Path to the scanned menu sheet image.
            output_dir: Directory to write the resulting JSON.

        Returns:
            Path to the written JSON file.
        """

    @abstractmethod
    def parse_batch(self, images_dir: Path, output_dir: Path | None = None) -> list[Path]:
        """Parse all images in a directory.

        Args:
            images_dir: Directory containing .jpg/.png images.
            output_dir: Directory to write JSON results.

        Returns:
            List of paths to written JSON files.
        """


class _ConcreteOcrParserService(OcrParserService):
    """Concrete implementation combining Ollama OCR + raw text parsing."""

    def __init__(
        self,
        client: OcrClient | None = None,
        raw_parser: RawTextParser | None = None,
    ) -> None:
        self._client = client or OllamaClient()
        self._parser = raw_parser or RawTextParser()

    def parse_image(self, image_path: Path, output_dir: Path | None = None) -> Path:
        target_dir = output_dir or JSON_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info("OCR started for %s", image_path.name)
        raw_text = self._client.extract_text(image_path)
        logger.info("OCR raw length: %d chars", len(raw_text))

        try:
            sheet = self._parser.parse(raw_text)
        except ParseError as exc:
            logger.error("Parsing failed for %s: %s", image_path.name, exc)
            # Save raw text for debugging
            debug_path = target_dir / f"{image_path.stem}.raw.txt"
            debug_path.write_text(raw_text, encoding="utf-8")
            raise ParseError(
                f"Failed to parse {image_path.name}. Raw text saved to {debug_path}. "
                "You may need a stronger OCR model or manual correction."
            ) from exc

        json_path = target_dir / f"{image_path.stem}.json"
        json_path.write_text(
            json.dumps(sheet.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("JSON saved to %s", json_path)
        return json_path

    def parse_batch(self, images_dir: Path, output_dir: Path | None = None) -> list[Path]:
        results: list[Path] = []
        for img in sorted(images_dir.glob("*")):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                results.append(self.parse_image(img, output_dir))
            except (OcrError, ParseError):
                logger.exception("Skipping %s due to error", img.name)
        return results
