"""Simple patient-specific physiological deviation Monitoring Agent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:  # Supports both package imports and direct script execution.
    from .alert import create_alert
    from .baseline import calculate_baseline
    from .deviation_detector import (
        DEVIATION_THRESHOLDS,
        apply_persistence,
        calculate_deviation,
    )
    from .preprocessing import preprocess_data
except ImportError:  # pragma: no cover - convenience for direct execution
    from alert import create_alert
    from baseline import calculate_baseline
    from deviation_detector import DEVIATION_THRESHOLDS, apply_persistence, calculate_deviation
    from preprocessing import preprocess_data


TRACK_MAPPING = {
    "Solar8000/HR": "HR",
    "Solar8000/ART_MBP": "MAP",
    "Solar8000/PLETH_SPO2": "SpO2",
    "Solar8000/RR": "RR",
}
VITAL_COLUMNS = list(TRACK_MAPPING.values())


def _get_vitaldb() -> Any:
    """Import VitalDB only when live dataset access is requested."""
    try:
        import vitaldb
    except ImportError as exc:
        raise RuntimeError(
            "VitalDB is not installed. Activate your virtual environment and run "
            "'pip install -r requirements.txt'."
        ) from exc
    return vitaldb


def find_suitable_cases(limit: int | None = None) -> list[int]:
    """Find VitalDB cases containing all four required numeric vital tracks."""
    vitaldb = _get_vitaldb()
    try:
        case_ids = list(vitaldb.find_cases(list(TRACK_MAPPING)))
    except Exception as exc:
        raise RuntimeError(
            "Unable to query VitalDB for cases. Check your network connection and retry."
        ) from exc

    if not case_ids:
        raise RuntimeError("VitalDB returned no cases with HR, MAP, SpO2, and RR tracks.")
    return [int(case_id) for case_id in case_ids[:limit]] if limit else [int(case_id) for case_id in case_ids]


def load_case_data(case_id: int) -> pd.DataFrame:
    """Load one VitalDB case as one-second HR, MAP, SpO2, and RR samples."""
    vitaldb = _get_vitaldb()
    try:
        values = vitaldb.load_case(int(case_id), list(TRACK_MAPPING), interval=1)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load VitalDB case {case_id}. Check the case ID and network connection."
        ) from exc

    if values is None or np.size(values) == 0:
        raise ValueError(f"VitalDB case {case_id} returned no data for the required tracks.")

    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(VITAL_COLUMNS):
        raise ValueError(
            f"VitalDB case {case_id} returned an unexpected shape {values.shape}; "
            f"expected N rows by {len(VITAL_COLUMNS)} tracks."
        )
    df = pd.DataFrame(values, columns=VITAL_COLUMNS)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1, name="timestamp")
    return df


def monitor_dataframe(
    case_id: int,
    df: pd.DataFrame,
    baseline_window: int = 60,
    persistence_duration: int = 10,
) -> dict[str, Any]:
    """Run the full monitoring pipeline on a prepared patient dataframe.

    This function enables offline/synthetic tests without changing the live
    VitalDB workflow used by :func:`run_monitoring`.
    """
    missing = [column for column in VITAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Input data is missing required columns: " + ", ".join(missing))

    raw_df = df.loc[:, VITAL_COLUMNS].copy()
    if raw_df.empty:
        raise ValueError("Cannot monitor empty case data.")
    if raw_df.index.name is None:
        raw_df.index.name = "timestamp"

    processed_df = preprocess_data(raw_df)
    baseline = calculate_baseline(processed_df, window=baseline_window)
    deviation = calculate_deviation(processed_df, baseline)

    threshold_violations = pd.DataFrame(
        {
            vital: deviation[vital].ge(DEVIATION_THRESHOLDS[vital]).fillna(False)
            for vital in VITAL_COLUMNS
        },
        index=processed_df.index,
    )
    baseline_ready = baseline.notna().all(axis=1)
    deviation_count = threshold_violations.sum(axis=1).astype(int)
    candidate_alert = ((deviation_count >= 2) & baseline_ready).rename("candidate_alert")
    persistent_alert = apply_persistence(candidate_alert, duration=persistence_duration)

    monitoring_results = processed_df.copy()
    for vital in VITAL_COLUMNS:
        monitoring_results[f"{vital}_baseline"] = baseline[vital]
        monitoring_results[f"{vital}_deviation"] = deviation[vital]
    monitoring_results["deviation_count"] = deviation_count
    monitoring_results["candidate_alert"] = candidate_alert
    monitoring_results["persistent_alert"] = persistent_alert

    # Emit one event when each persistent run begins, avoiding an alert for every
    # second of one continuous physiological deviation.
    persistent_start = persistent_alert & ~persistent_alert.shift(1, fill_value=False)
    alerts: list[dict[str, Any]] = []
    for timestamp in monitoring_results.index[persistent_start]:
        affected_vitals = [
            vital for vital in VITAL_COLUMNS if bool(threshold_violations.at[timestamp, vital])
        ]
        deviation_values = {vital: deviation.at[timestamp, vital] for vital in affected_vitals}
        alerts.append(
            create_alert(
                case_id=case_id,
                timestamp=timestamp,
                affected_vitals=affected_vitals,
                deviation_values=deviation_values,
                duration=persistence_duration,
            )
        )

    return {
        "df": processed_df,
        "baseline": baseline,
        "deviation": deviation,
        "monitoring_results": monitoring_results,
        "alerts": alerts,
    }


def run_monitoring(case_id: int) -> dict[str, Any]:
    """Load one VitalDB case and return all Phase 1 monitoring outputs."""
    df = load_case_data(case_id)
    return monitor_dataframe(case_id=case_id, df=df)
