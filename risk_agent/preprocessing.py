"""Feature engineering and patient demographic context gathering for the Risk Agent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from communication.events import MonitoringEvent

VITAL_COLUMNS_7 = ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]


def gather_patient_context(
    case_id: int,
    cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Gather demographic context (age, sex, bmi, asa, emop) with offline fallback."""
    if cache is not None and case_id in cache:
        return cache[case_id]

    context: dict[str, float] = {
        "age": np.nan,
        "sex": np.nan,
        "bmi": np.nan,
        "asa": np.nan,
        "emop": np.nan,
    }

    try:
        cases_df = pd.read_csv("https://api.vitaldb.net/cases", timeout=2.0)
        patient = cases_df.loc[cases_df["caseid"] == int(case_id)]
        if not patient.empty:
            row = patient.iloc[0]
            sex_str = str(row.get("sex", "")).upper()
            context["age"] = float(pd.to_numeric(row.get("age"), errors="coerce"))
            context["sex"] = 1.0 if sex_str == "M" else 0.0 if sex_str == "F" else np.nan
            context["bmi"] = float(pd.to_numeric(row.get("bmi"), errors="coerce"))
            context["asa"] = float(pd.to_numeric(row.get("asa"), errors="coerce"))
            context["emop"] = float(pd.to_numeric(row.get("emop"), errors="coerce"))
    except Exception:
        # Standard offline fallback demographic values
        context["age"] = 60.0
        context["sex"] = 1.0
        context["bmi"] = 24.5
        context["asa"] = 2.0
        context["emop"] = 0.0

    if cache is not None:
        cache[case_id] = context
    return context


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    """Extract 7-vital statistical features from a time window."""
    features: dict[str, float] = {}
    for vital in VITAL_COLUMNS_7:
        if vital not in window.columns:
            continue
        values = pd.to_numeric(window[vital], errors="coerce").dropna()
        if values.empty:
            continue

        features[f"{vital}_mean"] = float(values.mean())
        features[f"{vital}_min"] = float(values.min())
        features[f"{vital}_max"] = float(values.max())
        features[f"{vital}_std"] = float(values.std()) if len(values) > 1 else 0.0
        features[f"{vital}_latest"] = float(values.iloc[-1])
        features[f"{vital}_change"] = float(values.iloc[-1] - values.iloc[0])
        features[f"{vital}_slope"] = (
            float(np.polyfit(np.arange(len(values)), values.to_numpy(), 1)[0])
            if len(values) > 1
            else 0.0
        )
    return features


def construct_features(
    event: MonitoringEvent,
    patient_context: dict[str, float],
    feature_columns: list[str],
    window_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Construct feature dictionary aligned with trained model feature columns."""
    features: dict[str, float] = {col: np.nan for col in feature_columns}

    if window_df is not None and not window_df.empty:
        features.update(extract_window_features(window_df))
    else:
        for vital in event.affected_vitals:
            curr = event.current_values.get(vital, np.nan)
            summary = event.vital_summary.get(vital, {})
            init = summary.get("initial_value", curr)
            mn = summary.get("minimum_value", curr)
            mx = summary.get("maximum_value", curr)

            features[f"{vital}_latest"] = float(curr)
            features[f"{vital}_min"] = float(mn)
            features[f"{vital}_max"] = float(mx)
            features[f"{vital}_mean"] = float((mn + mx + curr) / 3.0) if pd.notna(curr) else np.nan
            features[f"{vital}_std"] = float(abs(mx - mn) / 2.0) if pd.notna(mx) and pd.notna(mn) else 0.0
            features[f"{vital}_change"] = float(curr - init) if pd.notna(curr) and pd.notna(init) else 0.0

            trend = event.trends.get(vital, "stable")
            features[f"{vital}_slope"] = 1.0 if trend == "increasing" else -1.0 if trend == "decreasing" else 0.0

    features.update(patient_context)
    return features


__all__ = [
    "VITAL_COLUMNS_7",
    "construct_features",
    "extract_window_features",
    "gather_patient_context",
]

