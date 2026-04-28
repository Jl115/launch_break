"""Application configuration using Pydantic Settings."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["AppConfig"]


class AppConfig(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_primary_model: str = Field(default="deepseek-ocr:3b", alias="OLLAMA_PRIMARY_MODEL")
    ollama_fallback_model: str = Field(
        default="qwen3-vl:235b-cloud", alias="OLLAMA_FALLBACK_MODEL"
    )
    data_dir: Path = Field(default=Path("data"))
    images_dir: Path = Field(default=Path("data/images"))
    json_dir: Path = Field(default=Path("data/json"))
    models_dir: Path = Field(default=Path("models"))
    min_training_rows: int = Field(default=100)
    n_splits_cv: int = Field(default=3)
    log_level: str = Field(default="INFO")
