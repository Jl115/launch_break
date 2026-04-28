"""Manual dependency injection container."""
from app.config import AppConfig
from features.data_ingestion import FeatureEngineer, MenuRepository
from features.ocr_parser.service import OcrParserService, _ConcreteOcrParserService
from features.prediction import ModelTrainer

__all__ = ["Container"]


class Container:
    """Wires together all application services."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.ocr_service: OcrParserService = _ConcreteOcrParserService()
        self.menu_repository = MenuRepository(self.config.json_dir)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(
            min_rows=self.config.min_training_rows,
            n_splits=self.config.n_splits_cv,
        )
