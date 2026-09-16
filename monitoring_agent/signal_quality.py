"""Conservative signal-quality and possible-artifact classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import (
        ARTIFACT_SPIKE_THRESHOLD,
        SIGNAL_QUALITY_MIN_VALID_POINTS,
        SIGNAL_QUALITY_WINDOW_SECONDS,
    )
except ImportError:  # pragma: no cover - supports direct module execution
    from config import (
        ARTIFACT_SPIKE_THRESHOLD,
        SIGNAL_QUALITY_MIN_VALID_POINTS,
        SIGNAL_QUALITY_WINDOW_SECONDS,
    )


def _valid_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert values to numeric and mark only impossible readings as missing."""
    values = df.copy()
    for vital in values.columns:
        values[vital] = pd.to_numeric(values[vital], errors="coerce")
    return values.mask(invalid_measurement_mask(df))


def invalid_measurement_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Return only truly invalid measurements, separate from missing values.

    Missing raw values are not labelled invalid. They may have a limited signal
    quality after preprocessing, but can remain supporting evidence when the
    full multi-vital and persistence criteria are met.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("invalid_measurement_mask expects a pandas DataFrame.")

    invalid = pd.DataFrame(False, index=df.index, columns=df.columns)
    numeric_values = pd.DataFrame(index=df.index)
    for vital in df.columns:
        raw = df[vital]
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_values[vital] = numeric
        # A non-empty value that cannot be made finite/numeric is invalid;
        # ordinary missing values stay distinct from invalid measurements.
        invalid[vital] = raw.notna() & (~np.isfinite(numeric.fillna(0)) | numeric.isna())

    if "HR" in invalid:
        invalid["HR"] |= numeric_values["HR"] <= 0
    if "MAP" in invalid:
        invalid["MAP"] |= numeric_values["MAP"] <= 0
    if "SpO2" in invalid:
        invalid["SpO2"] |= (numeric_values["SpO2"] < 0) | (numeric_values["SpO2"] > 100)
    if "RR" in invalid:
        invalid["RR"] |= numeric_values["RR"] < 0
    return invalid.fillna(False).astype(bool)


def assess_signal_quality(
    df: pd.DataFrame,
    window: int = SIGNAL_QUALITY_WINDOW_SECONDS,
    min_valid_points: int = SIGNAL_QUALITY_MIN_VALID_POINTS,
    spike_threshold: float = ARTIFACT_SPIKE_THRESHOLD,
) -> pd.DataFrame:
    """Classify each sample as good, suspicious, or insufficient_data.

    A suspicious sample is an abrupt change from its previous neighbour that
    immediately returns close to the prior level at the next sample. Original
    measurements are never deleted or replaced by this function.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("assess_signal_quality expects a pandas DataFrame.")
    if window < 1 or min_valid_points < 1:
        raise ValueError("Signal-quality window and minimum points must be positive integers.")
    if spike_threshold <= 0:
        raise ValueError("Artifact spike threshold must be positive.")

    valid_values = _valid_numeric_values(df)
    quality = pd.DataFrame("good", index=df.index, columns=df.columns, dtype="object")

    for vital in valid_values.columns:
        series = valid_values[vital]
        valid_count = series.notna().rolling(window=window, min_periods=1).sum()
        quality.loc[series.isna() | (valid_count < min_valid_points), vital] = "insufficient_data"

        # An isolated spike has two abrupt adjacent changes in opposite
        # directions: previous -> current and current -> next.
        previous = series.shift(1)
        following = series.shift(-1)
        prev_scale = previous.abs().replace(0, np.nan)
        next_scale = following.abs().replace(0, np.nan)
        previous_change = (series - previous).abs() / prev_scale
        next_change = (series - following).abs() / next_scale
        returns_toward_previous = (following - previous).abs() / prev_scale < spike_threshold
        isolated_spike = (
            series.notna()
            & previous.notna()
            & following.notna()
            & (previous_change >= spike_threshold)
            & (next_change >= spike_threshold)
            & returns_toward_previous
        )
        # A confirmed isolated spike is more informative than the early-window
        # count alone, so it receives the explicit suspicious classification.
        quality.loc[isolated_spike, vital] = "suspicious"
    return quality
