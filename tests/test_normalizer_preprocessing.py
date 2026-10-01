"""Unit tests for ObservationNormalizer preprocessing, sorting, and interpolation."""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from data.normalizer import ObservationNormalizer


class TestObservationNormalizerPreprocessing(unittest.TestCase):
    def test_chronological_sorting_and_duplicate_timestamp_dropping(self):
        """Observations must be sorted chronologically and duplicate timestamps deduplicated (keeping last)."""
        df = pd.DataFrame({
            "timestamp": [10.0, 5.0, 10.0, 1.0],
            "HR": [70.0, 75.0, 80.0, 65.0],
            "MAP": [85.0, 90.0, 95.0, 80.0],
        })

        obs = ObservationNormalizer.normalize_dataframe(df, time_column="timestamp")
        timestamps = [o.timestamp for o in obs]
        self.assertEqual(timestamps, [1.0, 5.0, 10.0])

        # Verify duplicate at t=10.0 kept the last record (HR=80, MAP=95)
        last_obs = obs[-1]
        self.assertEqual(last_obs.timestamp, 10.0)
        self.assertEqual(last_obs.HR, 80.0)
        self.assertEqual(last_obs.MAP, 95.0)

    def test_preserves_physiological_extremes_without_outlier_removal(self):
        """Physiological extremes representing acute clinical deterioration must be preserved, not trimmed."""
        # Extreme tachycardia (HR=190), severe hypotension (MAP=42, SBP=55), severe hypoxemia (SpO2=71%)
        df = pd.DataFrame({
            "timestamp": [0.0, 1.0, 2.0],
            "HR": [110.0, 185.0, 210.0],
            "MAP": [65.0, 48.0, 38.0],
            "SBP": [90.0, 65.0, 50.0],
            "DBP": [52.0, 39.0, 32.0],
            "SpO2": [92.0, 78.0, 71.0],
            "RR": [22.0, 38.0, 44.0],
        })

        obs = ObservationNormalizer.normalize_dataframe(df, time_column="timestamp")
        self.assertEqual(len(obs), 3)

        # Ensure extreme values are retained exactly
        self.assertEqual(obs[2].HR, 210.0)
        self.assertEqual(obs[2].MAP, 38.0)
        self.assertEqual(obs[2].SBP, 50.0)
        self.assertEqual(obs[2].DBP, 32.0)
        self.assertEqual(obs[2].SpO2, 71.0)
        self.assertEqual(obs[2].RR, 44.0)

    def test_bounded_interpolation_up_to_five_samples(self):
        """Missing values for gaps <= 5 samples are linearly interpolated; gaps > 5 are not filled beyond 5."""
        # Gap of 3 missing values (samples 1, 2, 3) between 100.0 and 140.0
        df_small_gap = pd.DataFrame({
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "HR": [100.0, np.nan, np.nan, np.nan, 140.0],
        })
        obs_small = ObservationNormalizer.normalize_dataframe(df_small_gap, time_column="timestamp")
        # All 5 values should now be present
        hr_vals = [o.HR for o in obs_small]
        self.assertTrue(all(v is not None for v in hr_vals))
        self.assertEqual(hr_vals[0], 100.0)
        self.assertEqual(hr_vals[1], 110.0)
        self.assertEqual(hr_vals[2], 120.0)
        self.assertEqual(hr_vals[3], 130.0)
        self.assertEqual(hr_vals[4], 140.0)

        # Gap of 7 missing values: only first 5 are interpolated, 6th and 7th remain None
        s = pd.Series([100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 180.0])
        interp_s = ObservationNormalizer.clean_and_interpolate_series(s, max_consecutive_missing=5)
        # Index 0..5 have values, 6 and 7 are NaN
        self.assertFalse(np.isnan(interp_s.iloc[5]))
        self.assertTrue(np.isnan(interp_s.iloc[6]))
        self.assertTrue(np.isnan(interp_s.iloc[7]))


if __name__ == "__main__":
    unittest.main()
