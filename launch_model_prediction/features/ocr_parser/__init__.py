"""OCR parser feature exports."""
from features.ocr_parser.ollama_client import OcrClient, OllamaClient
from features.ocr_parser.raw_parser import RawTextParser
from features.ocr_parser.service import OcrParserService

__all__ = [
    "OcrClient",
    "OllamaClient",
    "RawTextParser",
    "OcrParserService",
]
