"""OCR feature: client that communicates with Ollama for text extraction."""
import base64
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from shared import (
    OLLAMA_FALLBACK_MODEL,
    OLLAMA_HOST,
    OLLAMA_PRIMARY_MODEL,
    OcrError,
    get_logger,
)

__all__ = ["OcrClient", "OllamaClient"]

logger = get_logger(__name__)


class OcrClient(ABC):
    """Abstract interface for OCR clients."""

    @abstractmethod
    def extract_text(self, image_path: Path) -> str:
        """Return cleaned text extracted from the image.

        Args:
            image_path: Path to the image file.

        Returns:
            Cleaned raw text.

        Raises:
            OcrError: On failure.
        """


class OllamaClient(OcrClient):
    """Concrete OCR client using Ollama chat API.

    Uses deepseek-ocr:3b as primary, falls back to qwen3-vl:235b-cloud
    automatically if the primary model returns only layout tags.
    """

    _PRIMARY_PROMPT = (
        "Extract the text in the image.\n"
        "<|grounding|>Given the layout of the image."
    )
    _FALLBACK_PROMPT = (
        "Extract ALL text from this image exactly as it appears. "
        "Preserve the table structure, names, numbers, and German words. "
        "Do not summarize or describe the image, only transcribe the text content."
    )

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        primary_model: str = OLLAMA_PRIMARY_MODEL,
        fallback_model: str = OLLAMA_FALLBACK_MODEL,
        timeout: int = 300,
    ) -> None:
        self.host = host.rstrip("/")
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.timeout = timeout

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call(self, model: str, image_path: Path, prompt: str) -> str:
        b64_img = self._encode_image(image_path)
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64_img],
                }
            ],
        }
        url = f"{self.host}/api/chat"
        logger.info("Sending %s to %s (model=%s)", image_path.name, url, model)
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OcrError(f"HTTP error during OCR: {exc}") from exc

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        if not text:
            raise OcrError(f"OCR returned empty content for {image_path}")
        return text

    @staticmethod
    def _is_layout_only(text: str) -> bool:
        """Check if output contains only layout tags and no actual text."""
        stripped = text.strip()
        if not stripped:
            return True
        # If text is only grounding/det tags or very short (< 100 chars)
        if len(stripped) < 100:
            return True
        # If text contains mostly ref/det tags
        tag_chars = stripped.count("<") + stripped.count(">") + stripped.count("|")
        if tag_chars > len(stripped) * 0.3:
            return True
        return False

    def extract_text(self, image_path: Path) -> str:
        """Extract text using primary model, fallback if needed."""
        for model, prompt in (
            (self.primary_model, self._PRIMARY_PROMPT),
            (self.fallback_model, self._FALLBACK_PROMPT),
        ):
            try:
                raw = self._call(model, image_path, prompt)
                logger.info("%s raw length: %d", model, len(raw))
                if self._is_layout_only(raw) and model == self.primary_model:
                    logger.warning(
                        "%s returned only layout tags (%d chars), trying fallback",
                        model,
                        len(raw),
                    )
                    continue
                logger.info("%s succeeded for %s", model, image_path.name)
                return raw
            except OcrError:
                logger.warning("Model %s failed for %s", model, image_path.name)
                time.sleep(1)
        raise OcrError(f"All OCR models failed for {image_path}")
