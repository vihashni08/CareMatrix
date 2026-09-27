"""Data retrieval and normalization utilities for the Data Analysis Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WINDOW_SECONDS = 300
SAMPLE_INTERVAL = 5

VITAL_TRACKS: list[str] = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/RR",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/BT",
]

VITAL_COLUMNS: list[str] = ["HR", "SpO2", "RR", "SBP", "DBP", "MAP", "BT"]


def normalise_frame(data: Any) -> pd.DataFrame:
    """Normalize physiological data matrix or dataframe into a standard Time + vitals DataFrame."""
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if "Time" not in frame.columns:
            frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)
    else:
        frame = pd.DataFrame(np.asarray(data), columns=VITAL_COLUMNS)
        frame.insert(0, "Time", np.arange(len(frame)) * SAMPLE_INTERVAL)

    for vital in VITAL_COLUMNS:
        if vital not in frame:
            frame[vital] = np.nan
        frame[vital] = pd.to_numeric(frame[vital], errors="coerce")

    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce")
    return frame.dropna(subset=["Time"])


def load_case_data(case_id: int) -> pd.DataFrame:
    """Retrieve patient vital tracks from VitalDB with local processed file fallback."""
    try:
        import vitaldb
        data = vitaldb.load_case(int(case_id), VITAL_TRACKS, interval=SAMPLE_INTERVAL)
        if data is not None and np.size(data):
            return normalise_frame(data)
    except Exception:
        pass

    local_path = Path(__file__).resolve().parent.parent / "data" / "processed" / f"patient_{case_id}.csv"
    if local_path.exists():
        return normalise_frame(pd.read_csv(local_path))

    raise ValueError(f"No VitalDB or local physiological data is available for case {case_id}.")


__all__ = [
    "SAMPLE_INTERVAL",
    "VITAL_COLUMNS",
    "VITAL_TRACKS",
    "WINDOW_SECONDS",
    "load_case_data",
    "normalise_frame",
]

