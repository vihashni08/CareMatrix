"""Observation normalizer standardizing timestamps, units, and physiological ranges."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from data.schemas import CommonPatientObservation


class ObservationNormalizer:
    """Normalizes heterogeneous clinical telemetry observations into common CareMatrix format."""

    @staticmethod
    def normalize_temperature(val: float | None) -> float | None:
        """Standardize temperature to degrees Celsius.
        
        If value > 50, assumes Fahrenheit and converts: C = (F - 32) * 5 / 9.
        """
        if val is None or np.isnan(val):
            return None
        val_f = float(val)
        if val_f > 50.0:  # Clearly Fahrenheit (e.g., 98.6°F -> 37.0°C)
            return round((val_f - 32.0) * 5.0 / 9.0, 2)
        if 20.0 <= val_f <= 45.0:
            return round(val_f, 2)
        return val_f

    @staticmethod
    def normalize_spo2(val: float | None) -> float | None:
        """Standardize SpO2 to percentage (0 - 100)."""
        if val is None or np.isnan(val):
            return None
        val_f = float(val)
        # If recorded as fraction (0.70 - 1.00), scale to percentage
        if 0.0 < val_f <= 1.0:
            return round(val_f * 100.0, 1)
        if 0.0 <= val_f <= 100.0:
            return round(val_f, 1)
        return None

    @staticmethod
    def normalize_blood_pressure(
        map_val: float | None,
        sbp_val: float | None,
        dbp_val: float | None,
    ) -> tuple[float | None, float | None, float | None]:
        """Validate and harmonize MAP, SBP, and DBP.
        
        If MAP is missing but SBP and DBP are provided, estimates MAP = DBP + (SBP - DBP)/3.
        Does NOT invent SBP or DBP if missing.
        """
        m = float(map_val) if map_val is not None and not np.isnan(map_val) else None
        s = float(sbp_val) if sbp_val is not None and not np.isnan(sbp_val) else None
        d = float(dbp_val) if dbp_val is not None and not np.isnan(dbp_val) else None

        # Filter impossible negative or extreme non-physiological values
        if m is not None and (m <= 0 or m > 300):
            m = None
        if s is not None and (s <= 0 or s > 350):
            s = None
        if d is not None and (d <= 0 or d > 250):
            d = None

        # Calculate estimated MAP if missing
        if m is None and s is not None and d is not None and s >= d:
            m = round(d + (s - d) / 3.0, 1)

        return m, s, d

    @classmethod
    def normalize_record(
        cls,
        raw: dict[str, Any],
        source: str = "unknown",
        patient_id: str | int = 0,
        relative_timestamp: float | None = None,
    ) -> CommonPatientObservation:
        """Normalize a single raw observation record."""
        # 1. Timestamp handling
        ts = raw.get("timestamp")
        if relative_timestamp is not None:
            t = float(relative_timestamp)
        elif ts is not None and not np.isnan(ts):
            t = float(ts)
        else:
            t = 0.0

        # 2. Heart rate
        hr = raw.get("HR") or raw.get("heart_rate") or raw.get("heartrate") or raw.get("pulse")
        hr_norm = float(hr) if hr is not None and not np.isnan(hr) and 0 < float(hr) < 350 else None

        # 3. Respiratory rate
        rr = raw.get("RR") or raw.get("respiratory_rate") or raw.get("resprate")
        rr_norm = float(rr) if rr is not None and not np.isnan(rr) and 0 < float(rr) < 120 else None

        # 4. SpO2
        spo2 = raw.get("SpO2") or raw.get("spo2") or raw.get("o2sat") or raw.get("sao2")
        spo2_norm = cls.normalize_spo2(spo2)

        # 5. Blood pressure
        map_val = raw.get("MAP") or raw.get("map") or raw.get("mean_arterial_pressure")
        sbp_val = raw.get("SBP") or raw.get("sbp") or raw.get("systolic")
        dbp_val = raw.get("DBP") or raw.get("dbp") or raw.get("diastolic")
        m_norm, s_norm, d_norm = cls.normalize_blood_pressure(map_val, sbp_val, dbp_val)

        # 6. Temperature
        bt = raw.get("BT") or raw.get("bt") or raw.get("temp") or raw.get("temperature")
        bt_norm = cls.normalize_temperature(bt)

        metadata = dict(raw.get("metadata", {}))
        metadata["raw_source"] = source
        metadata["original_timestamp"] = raw.get("charttime") or raw.get("timestamp")

        return CommonPatientObservation(
            timestamp=t,
            HR=hr_norm,
            MAP=m_norm,
            SpO2=spo2_norm,
            RR=rr_norm,
            BT=bt_norm,
            SBP=s_norm,
            DBP=d_norm,
            source=source,
            patient_id=patient_id,
            metadata=metadata,
        )

    @classmethod
    def clean_and_interpolate_series(
        cls,
        series: pd.Series,
        max_consecutive_missing: int = 5,
    ) -> pd.Series:
        """Perform bounded linear interpolation for small gaps (<= max_consecutive_missing samples)."""
        numeric_series = pd.to_numeric(series, errors="coerce")
        return numeric_series.interpolate(
            method="linear",
            limit=max_consecutive_missing,
            limit_direction="forward",
        )

    @classmethod
    def sort_and_deduplicate_observations(
        cls,
        observations: list[CommonPatientObservation],
    ) -> list[CommonPatientObservation]:
        """Sort observations chronologically and drop duplicate timestamps (keep last)."""
        if not observations:
            return []
        sorted_obs = sorted(observations, key=lambda o: o.timestamp)
        deduped: dict[float, CommonPatientObservation] = {}
        for obs in sorted_obs:
            deduped[obs.timestamp] = obs
        return list(deduped.values())

    @classmethod
    def normalize_dataframe(
        cls,
        df: pd.DataFrame,
        source: str = "unknown",
        patient_id: str | int = 0,
        time_column: str | None = None,
        interpolate_max_gap: int = 5,
    ) -> list[CommonPatientObservation]:
        """Normalize a complete pandas DataFrame into sequential CommonPatientObservations.

        Applies:
        1. Chronological sorting and duplicate timestamp removal (keeping last).
        2. Preservation of physiological extremes without outlier removal.
        3. Bounded interpolation (<= interpolate_max_gap samples) for transient sensor dropout.
        """
        observations: list[CommonPatientObservation] = []
        if df.empty:
            return observations

        working_df = df.copy()

        # 1. Chronological sorting and deduplication on time column if present
        if time_column and time_column in working_df.columns:
            working_df = working_df.sort_values(by=time_column).drop_duplicates(subset=[time_column], keep="last")
        elif pd.api.types.is_numeric_dtype(working_df.index):
            working_df = working_df.sort_index().loc[~working_df.index.duplicated(keep="last")]

        # 2. Bounded interpolation for missing continuous vitals (<= 5 samples)
        # Note: physiological extremes are strictly preserved without clipping or trimming
        vital_cols = [
            c for c in [
                "HR", "heart_rate", "heartrate", "pulse",
                "RR", "respiratory_rate", "resprate",
                "SpO2", "spo2", "o2sat",
                "MAP", "map", "mean_arterial_pressure",
                "SBP", "sbp", "systolic",
                "DBP", "dbp", "diastolic",
                "BT", "bt", "temp", "temperature",
            ] if c in working_df.columns
        ]
        for col in vital_cols:
            working_df[col] = cls.clean_and_interpolate_series(working_df[col], max_consecutive_missing=interpolate_max_gap)

        # Determine timestamp indexing
        if time_column and time_column in working_df.columns:
            ts_series = working_df[time_column]
            # If datetime, compute elapsed seconds
            if pd.api.types.is_datetime64_any_dtype(ts_series) or isinstance(ts_series.iloc[0], str):
                try:
                    dt = pd.to_datetime(ts_series)
                    start_time = dt.iloc[0]
                    elapsed = (dt - start_time).dt.total_seconds().values
                except Exception:
                    elapsed = np.arange(len(working_df), dtype=float)
            else:
                elapsed = pd.to_numeric(ts_series, errors="coerce").fillna(0.0).values
        elif isinstance(working_df.index, pd.DatetimeIndex):
            start_time = working_df.index[0]
            elapsed = (working_df.index - start_time).total_seconds().values
        elif pd.api.types.is_numeric_dtype(working_df.index):
            elapsed = working_df.index.values.astype(float)
        else:
            elapsed = np.arange(len(working_df), dtype=float)

        for i, (_, row) in enumerate(working_df.iterrows()):
            t = float(elapsed[i]) if i < len(elapsed) else float(i)
            obs = cls.normalize_record(
                raw=row.to_dict(),
                source=source,
                patient_id=patient_id,
                relative_timestamp=t,
            )
            observations.append(obs)

        return cls.sort_and_deduplicate_observations(observations)


__all__ = ["ObservationNormalizer"]

