"""Prediction feature: evaluation metrics."""
import numpy as np
import pandas as pd

from features.prediction.model import LunchPredictor
from shared import get_logger

__all__ = ["ModelEvaluator"]

logger = get_logger(__name__)

_TARGET_COLS = ["Erw", "Ki", "MA", "MA-Ki"]


class ModelEvaluator:
    """Evaluate a trained predictor on a hold-out set."""

    def evaluate(self, predictor: LunchPredictor, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute MAE and RMSE per category.

        Args:
            predictor: Trained predictor.
            X: Feature matrix.
            y: True target values.

        Returns:
            Nested dict: {category: {metric: value}}.
        """
        preds = predictor.predict(X)
        results: dict[str, dict[str, float]] = {}
        for col in _TARGET_COLS:
            true_vals = y[col].values
            pred_vals = preds[col].values
            mae = float(np.mean(np.abs(pred_vals - true_vals)))
            rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
            results[col] = {"MAE": mae, "RMSE": rmse}
        return results
