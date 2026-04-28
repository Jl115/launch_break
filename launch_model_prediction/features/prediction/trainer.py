"""Prediction feature: training orchestration."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from features.prediction.model import LunchPredictor, SklearnLunchPredictor
from shared import InsufficientDataError, ModelError, get_logger

__all__ = ["Trainer", "ModelTrainer"]

logger = get_logger(__name__)

# Minimum rows needed: 20 weeks * 5 days = 100 rows
_MIN_ROWS = 100
_TARGET_COLS = ["Erw", "Ki", "MA", "MA-Ki"]


class Trainer(ABC):
    """Abstract trainer interface."""

    @abstractmethod
    def train(self, df: pd.DataFrame, model_path: Path | None = None) -> LunchPredictor:
        """Train and return a fitted predictor. Optionally save it."""


class ModelTrainer(Trainer):
    """Concrete trainer with TimeSeriesSplit cross-validation."""

    def __init__(self, min_rows: int = _MIN_ROWS, n_splits: int = 3) -> None:
        self.min_rows = min_rows
        self.n_splits = n_splits

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        if len(df) < self.min_rows:
            raise InsufficientDataError(
                f"Need at least {self.min_rows} historical daily records "
                f"(roughly {self.min_rows // 5} weeks). Got {len(df)}."
            )
        # Drop non-feature, non-target columns
        drop_cols = {"date"} & set(df.columns)
        X = df.drop(columns=list(drop_cols | set(_TARGET_COLS)))
        y = df[_TARGET_COLS]
        return X, y

    def train(self, df: pd.DataFrame, model_path: Path | None = None) -> LunchPredictor:
        """Train the lunch attendance predictor.

        Args:
            df: Feature-engineered DataFrame.
            model_path: If provided, save the artifact here.

        Returns:
            Fitted LunchPredictor.
        """
        X, y = self._prepare(df)
        predictor = SklearnLunchPredictor()
        predictor.train(X, y)
        if model_path:
            predictor.save(model_path)
        return predictor

    def cross_validate(self, df: pd.DataFrame) -> dict[str, list[float]]:
        """Run TimeSeriesSplit cross-validation.

        Returns:
            Dict mapping category name to list of MAE scores.
        """
        X, y = self._prepare(df)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores: dict[str, list[float]] = {col: [] for col in _TARGET_COLS}
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            predictor = SklearnLunchPredictor()
            predictor.train(X_train, y_train)
            preds = predictor.predict(X_test)
            for col in _TARGET_COLS:
                mae = np.mean(np.abs(preds[col].values - y_test[col].values))
                scores[col].append(mae)
        return scores
