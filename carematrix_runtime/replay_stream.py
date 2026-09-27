"""Patient vital sign stream generator for continuous replay simulation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np
import pandas as pd

from monitoring_agent.data_loader import VITAL_COLUMNS, load_case_data, synthetic_patient_data


class PatientStreamReplayer:
    """Progressive vital signs stream generator for continuous patient monitoring replay."""

    def __init__(
        self,
        case_id: int | str = 4,
        source_df: pd.DataFrame | None = None,
        adapter: Any | None = None,
        delay_seconds: float = 0.0,
        loop: bool = False,
    ):
        self.case_id = int(case_id) if str(case_id).isdigit() else case_id
        self.delay_seconds = float(delay_seconds)
        self.loop = loop
        self.adapter = adapter
        self._stopped = False

        if source_df is not None:
            self.df = source_df.copy()
        elif adapter is not None and hasattr(adapter, "load_case"):
            obs_list = adapter.load_case(self.case_id)
            rows = [obs.to_monitoring_sample() for obs in obs_list]
            self.df = pd.DataFrame(rows)
            if "timestamp" in self.df.columns:
                self.df = self.df.set_index("timestamp")
        else:
            self.df = self._load_data(self.case_id if isinstance(self.case_id, int) else 0)

    def _load_data(self, case_id: int) -> pd.DataFrame:
        """Load patient data from VitalDB, local processed file, or synthetic data."""
        if case_id == 0:
            return synthetic_patient_data()

        try:
            return load_case_data(case_id)
        except Exception:
            # Fallback to local processed file if available
            local_path = (
                Path(__file__).resolve().parents[1]
                / "data"
                / "processed"
                / f"patient_{case_id}.csv"
            )
            if local_path.exists():
                df = pd.read_csv(local_path)
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                return df[VITAL_COLUMNS]

            # Fallback to synthetic data if no data found
            return synthetic_patient_data()

    def stream(
        self,
        max_samples: int | None = None,
        start_sample: int = 0,
    ) -> Generator[dict[str, Any], None, None]:
        """Progressively yield one vital-sign sample at a time."""
        self._stopped = False
        count = 0
        df_to_stream = self.df.iloc[start_sample:] if start_sample > 0 else self.df

        while not self._stopped:
            for idx, row in df_to_stream.iterrows():
                if self._stopped:
                    return
                sample = {
                    "timestamp": float(idx) if isinstance(idx, (int, float, np.number)) else float(count),
                    "HR": float(row.get("HR", np.nan)),
                    "MAP": float(row.get("MAP", np.nan)),
                    "SpO2": float(row.get("SpO2", np.nan)),
                    "RR": float(row.get("RR", np.nan)),
                }

                yield sample
                count += 1

                if max_samples is not None and count >= max_samples:
                    return

                if self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)

            if not self.loop:
                break

    def stop(self) -> None:
        """Stop the streaming generator."""
        self._stopped = True
