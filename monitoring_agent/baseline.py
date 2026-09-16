"""Patient-specific rolling baselines for physiological monitoring."""

import pandas as pd


def calculate_baseline(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate each vital's rolling median from the preceding observations.

    A baseline is available only after ten earlier samples (`min_periods=10`).
    The one-sample shift ensures the value being evaluated is not included in
    its own baseline.  The monitoring agent suppresses alerts until every
    required vital has an established baseline.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("calculate_baseline expects a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot calculate a baseline from empty data.")
    if not isinstance(window, int) or window < 10:
        raise ValueError("Baseline window must be an integer of at least 10 samples.")

    return df.shift(1).rolling(window=window, min_periods=10).median()
