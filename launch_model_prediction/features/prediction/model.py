"""Prediction feature: ML model definition and persistence."""
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from shared import ModelError, get_logger

__all__ = ["LunchPredictor", "SklearnLunchPredictor"]

logger = get_logger(__name__)

_TARGET_COLS = ["Erw", "Ki", "MA", "MA-Ki"]


class LunchPredictor(ABC):
    """Interface for the lunch attendance predictor."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Train the model on historical data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict attendance. Returns DataFrame with same columns as y."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Deserialize model from disk."""


class SklearnLunchPredictor(LunchPredictor):
    """Concrete predictor using RandomForest in MultiOutputRegressor."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1,
            )
        )
        self.feature_names: list[str] = []
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        if X.empty or y.empty:
            raise ModelError("Cannot train on empty dataset")
        self.feature_names = list(X.columns)
        logger.info("Training on %d samples, %d features", len(X), len(X.columns))
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Training complete")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ModelError("Model has not been trained")
        # Ensure column order
        X = X[self.feature_names]
        preds = self.model.predict(X)
        return pd.DataFrame(preds, columns=_TARGET_COLS, index=X.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        joblib.dump(artifact, path)
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.is_trained = artifact["is_trained"]
        logger.info("Model loaded from %s", path)
