"""Machine learning model artifact loader and inference tool for the Risk Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class RiskModelTool:
    """Encapsulates ML model artifacts and provides inference capabilities as an agent tool."""

    def __init__(self, model_dir: Path | str | None = None):
        if model_dir is None:
            self.model_dir = Path(__file__).resolve().parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self._load_artifacts()
        self.model_name = type(self.model).__name__

    def _load_artifacts(self) -> None:
        """Load trained ML model, preprocessor, and metadata artifacts."""
        model_path = self.model_dir / "best_model.pkl"
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        features_path = self.model_dir / "feature_columns.json"
        threshold_path = self.model_dir / "risk_threshold.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing at: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor file missing at: {preprocessor_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_columns: list[str] = json.load(f)

        with open(threshold_path, "r", encoding="utf-8") as f:
            self.risk_threshold: float = float(json.load(f)["threshold"])

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Execute preprocessor and ML model inference to compute high-risk probability."""
        feature_frame = pd.DataFrame(
            [{feat: features.get(feat, np.nan) for feat in self.feature_columns}],
            columns=self.feature_columns,
        )
        processed = self.preprocessor.transform(feature_frame)
        probabilities = self.model.predict_proba(processed)
        return float(probabilities[0, 1])


__all__ = ["RiskModelTool"]

