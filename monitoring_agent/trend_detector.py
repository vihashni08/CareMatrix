"""Lightweight descriptive trend detection for the four monitored vitals."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import TREND_MIN_POINTS, TREND_MIN_RELATIVE_CHANGE, TREND_WINDOW_SECONDS
except ImportError:  # pragma: no cover - supports direct module execution
    from config import TREND_MIN_POINTS, TREND_MIN_RELATIVE_CHANGE, TREND_WINDOW_SECONDS


def classify_trend(
    values: pd.Series | np.ndarray | list[float],
    min_points: int = TREND_MIN_POINTS,
    relative_change: float = TREND_MIN_RELATIVE_CHANGE,
) -> str:
    """Classify a recent valid sequence as increasing, decreasing, or stable.

    The medians of the first and last thirds of the sequence make this robust to
    a single noisy sample. Fewer than ``min_points`` valid readings are reported
    as ``insufficient_data`` rather than inferring a trend.
    """
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < min_points:
        return "insufficient_data"

    segment_size = max(1, len(series) // 3)
    beginning = float(series.iloc[:segment_size].median())
    ending = float(series.iloc[-segment_size:].median())
    scale = max(abs(beginning), 1e-9)
    change = (ending - beginning) / scale

    if change >= relative_change:
        return "increasing"
    if change <= -relative_change:
        return "decreasing"
    return "stable"


def calculate_trends(
    df: pd.DataFrame,
    window: int = TREND_WINDOW_SECONDS,
    min_points: int = TREND_MIN_POINTS,
    relative_change: float = TREND_MIN_RELATIVE_CHANGE,
) -> pd.DataFrame:
    """Return a descriptive trend label for each vital at every timestamp."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("calculate_trends expects a pandas DataFrame.")
    if window < 1 or min_points < 1:
        raise ValueError("Trend window and minimum points must be positive integers.")

    trends = pd.DataFrame(index=df.index)
    for vital in df.columns:
        labels = []
        for position in range(len(df)):
            recent_values = df[vital].iloc[max(0, position - window + 1) : position + 1]
            labels.append(classify_trend(recent_values, min_points, relative_change))
        trends[vital] = pd.Series(labels, index=df.index, dtype="object")
    return trends
