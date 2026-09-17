"""Conservative preprocessing for one-second VitalDB vital-sign samples."""

import numpy as np
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of vital-sign data without discarding extremes.

    Short interruptions (up to five samples) are interpolated linearly.  Any
    remaining gaps are filled from neighbouring observations so the downstream
    rolling calculations can continue.  Physiological extremes are deliberately
    retained: they can be genuine deviations rather than bad data.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("preprocess_data expects a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot preprocess empty case data.")

    # Work on a copy so callers retain the original VitalDB values unchanged.
    cleaned = df.copy()

    # Turn invalid numerical values into missing values before interpolation.
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # VitalDB normally supplies numeric arrays, but coercion handles malformed
    # values safely while keeping the original row/index order unchanged.
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Apply only explicit physiological validity rules. Invalid device values
    # become missing values for the existing conservative gap handling below.
    # This is deliberately not statistical outlier removal: unusual values that
    # remain physiologically possible are kept for deviation monitoring.
    if "HR" in cleaned:
        cleaned.loc[cleaned["HR"] <= 0, "HR"] = np.nan
    if "MAP" in cleaned:
        cleaned.loc[cleaned["MAP"] <= 0, "MAP"] = np.nan
    if "SpO2" in cleaned:
        cleaned.loc[(cleaned["SpO2"] < 0) | (cleaned["SpO2"] > 100), "SpO2"] = np.nan
    if "RR" in cleaned:
        cleaned.loc[cleaned["RR"] < 0, "RR"] = np.nan

    all_missing = cleaned.columns[cleaned.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            "The following required vital-sign columns contain no usable data: "
            + ", ".join(all_missing)
        )

    # Fill only short internal gaps (up to 5 samples) using linear
    # interpolation. Long gaps, leading uncalibrated periods, and trailing
    # disconnected periods remain NaN rather than being propagated by
    # unrestricted forward/backward filling, preventing invalid baselines.
    cleaned = cleaned.interpolate(method="linear", limit=5, limit_area="inside")

    return cleaned
