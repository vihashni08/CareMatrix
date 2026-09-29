"""Patient vital sign stream generator for continuous replay simulation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from data.adapters.vitaldb_adapter import VITAL_COLUMNS_7, VitalDBAdapter
from monitoring_agent.data_loader import VITAL_COLUMNS, load_case_data, synthetic_patient_data


class PatientStreamReplayer:
    """Progressive vital signs stream generator for continuous patient monitoring replay."""

    def __init__(
        self,
        case_id: int | str | None = None,
        source_df: pd.DataFrame | None = None,
        adapter: Any | None = None,
        delay_seconds: float = 0.0,
        speed_multiplier: float = 1.0,
        loop: bool = False,
    ):
        if case_id is None and source_df is None:
            case_id = 6
        self.case_id = int(case_id) if (case_id is not None and str(case_id).isdigit()) else case_id
        effective_delay = float(delay_seconds)
        if speed_multiplier > 0 and effective_delay > 0:
            effective_delay = effective_delay / float(speed_multiplier)
        self.delay_seconds = effective_delay
        self.speed_multiplier = float(speed_multiplier)
        self.loop = loop
        self._stopped = False

        if source_df is not None:
            self.df = source_df.copy()
            self.adapter = adapter
        else:
            self.adapter = adapter or VitalDBAdapter()
            if hasattr(self.adapter, "load_case"):
                obs_list = self.adapter.load_case(self.case_id)
                rows = [obs.to_monitoring_sample() for obs in obs_list]
                self.df = pd.DataFrame(rows)
                if "timestamp" in self.df.columns:
                    self.df = self.df.set_index("timestamp")
            else:
                self.df = self._load_fallback_data(self.case_id if isinstance(self.case_id, int) else 0)

    def _load_fallback_data(self, case_id: int) -> pd.DataFrame:
        """Load patient data from local processed file or synthetic fallback."""
        if case_id == 0:
            return synthetic_patient_data()

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
            return df[[c for c in VITAL_COLUMNS_7 if c in df.columns]]

        return synthetic_patient_data()

    def stream(
        self,
        max_samples: int | None = None,
        start_sample: int = 0,
    ) -> Generator[dict[str, Any], None, None]:
        """Progressively yield one vital-sign sample at a time containing all 7 vitals."""
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
                    "SBP": float(row.get("SBP", np.nan)),
                    "DBP": float(row.get("DBP", np.nan)),
                    "BT": float(row.get("BT", np.nan)),
                }
                if self.case_id is not None:
                    sample["patient_id"] = self.case_id
                    sample["case_id"] = self.case_id

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


__all__ = ["PatientStreamReplayer"]
