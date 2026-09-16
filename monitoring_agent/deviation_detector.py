"""Deviation and persistence calculations for the Monitoring Agent."""

import numpy as np
import pandas as pd

try:
    from .config import DEVIATION_THRESHOLDS, PERSISTENCE_SECONDS
except ImportError:  # pragma: no cover - supports direct module execution
    from config import DEVIATION_THRESHOLDS, PERSISTENCE_SECONDS



def calculate_deviation(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Calculate safe relative absolute deviations from a patient baseline.

    A zero (or unavailable) baseline produces ``NaN`` rather than an infinite
    value.  Such samples cannot exceed a threshold and therefore cannot create
    an alert.
    """
    if not isinstance(df, pd.DataFrame) or not isinstance(baseline, pd.DataFrame):
        raise TypeError("df and baseline must both be pandas DataFrames.")
    if not df.index.equals(baseline.index):
        raise ValueError("df and baseline must have matching time indexes.")
    if list(df.columns) != list(baseline.columns):
        raise ValueError("df and baseline must have matching vital-sign columns.")

    denominator = baseline.abs().replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = (df - baseline).abs().divide(denominator)
    return deviation.replace([np.inf, -np.inf], np.nan)


def apply_persistence(alert_series: pd.Series, duration: int = PERSISTENCE_SECONDS) -> pd.Series:
    """Mark samples where a candidate alert has lasted ``duration`` samples.

    With one-second VitalDB samples, the default duration represents ten
    consecutive seconds. The returned series remains true after the tenth
    sample while the same candidate run continues.
    """
    if not isinstance(alert_series, pd.Series):
        raise TypeError("alert_series must be a pandas Series.")
    if not isinstance(duration, int) or duration < 1:
        raise ValueError("Persistence duration must be a positive integer.")

    candidates = alert_series.fillna(False).astype(bool)
    run_groups = (~candidates).cumsum()
    run_length = candidates.groupby(run_groups).cumsum()
    return (candidates & (run_length >= duration)).rename("persistent_alert")
