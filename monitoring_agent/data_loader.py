"""Data loader and VitalDB interface utilities for the Monitoring Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from monitoring_agent.state import VITAL_COLUMNS

TRACK_MAPPING: dict[str, str] = {
    "Solar8000/HR": "HR",
    "Solar8000/ART_MBP": "MAP",
    "Solar8000/PLETH_SPO2": "SpO2",
    "Solar8000/RR": "RR",
}


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
    """Load one VitalDB case as one-second HR, MAP, SpO2, and RR samples with local cache fallback."""
    try:
        vitaldb = _get_vitaldb()
        values = vitaldb.load_case(int(case_id), list(TRACK_MAPPING), interval=1)
        if values is not None and np.size(values) > 0:
            values = np.asarray(values)
            if values.ndim == 2 and values.shape[1] == len(VITAL_COLUMNS):
                df = pd.DataFrame(values, columns=VITAL_COLUMNS)
                df.index = pd.RangeIndex(start=0, stop=len(df), step=1, name="timestamp")
                return df
    except Exception:
        pass

    # Fallback to locally cached processed file if present
    local_file = Path(__file__).resolve().parent.parent / "data" / "processed" / f"patient_{case_id}.csv"
    if local_file.exists():
        df_local = pd.read_csv(local_file)
        if "timestamp" in df_local.columns:
            df_local = df_local.set_index("timestamp")
        return df_local[VITAL_COLUMNS]

    raise RuntimeError(
        f"Unable to load VitalDB case {case_id}. Check the case ID and network connection."
    )


def synthetic_patient_data() -> pd.DataFrame:
    """Return a deterministic case with an 11-second multi-vital deviation."""
    sample_count = 100
    data = pd.DataFrame(
        {
            "HR": np.full(sample_count, 70.0),
            "MAP": np.full(sample_count, 85.0),
            "SpO2": np.full(sample_count, 98.0),
            "RR": np.full(sample_count, 14.0),
        },
        index=pd.RangeIndex(sample_count, name="timestamp"),
    )
    # Enough simultaneous deviations to exceed the configured thresholds for
    # longer than the severity-based persistence requirement.
    data.loc[70:80, ["HR", "MAP", "SpO2"]] = [95.0, 60.0, 90.0]
    return data[VITAL_COLUMNS]

