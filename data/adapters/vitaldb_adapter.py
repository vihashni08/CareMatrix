"""VitalDB dataset adapter integrating VitalDB API, local processed files, and synthetic cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

from data.adapters.base_adapter import BaseDatasetAdapter
from data.normalizer import ObservationNormalizer
from data.schemas import CommonPatientObservation
from monitoring_agent.data_loader import VITAL_COLUMNS, load_case_data, synthetic_patient_data


class VitalDBAdapter(BaseDatasetAdapter):
    """Adapter for VitalDB intraoperative telemetry and synthetic patient simulations."""

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "processed"

    @property
    def dataset_name(self) -> str:
        return "VitalDB"

    def list_cases(self) -> list[int | str]:
        """List known available cases: synthetic case 0 plus any local processed patient CSVs."""
        cases: list[int | str] = [0]
        if self.data_dir.exists():
            for f in self.data_dir.glob("patient_*.csv"):
                try:
                    num = int(f.stem.replace("patient_", ""))
                    if num not in cases:
                        cases.append(num)
                except ValueError:
                    cases.append(f.stem)
        if 4 not in cases:
            cases.append(4)  # Standard default real case
        return sorted(cases, key=lambda x: (isinstance(x, str), x))

    def load_case(self, case_id: int | str) -> list[CommonPatientObservation]:
        """Load patient data from VitalDB, local file, or synthetic benchmark generator."""
        case_num = int(case_id) if str(case_id).isdigit() else 0

        if case_num == 0:
            df = synthetic_patient_data()
            return ObservationNormalizer.normalize_dataframe(
                df,
                source="VitalDB_Synthetic",
                patient_id=0,
                time_column="timestamp" if "timestamp" in df.columns else None,
            )

        # 1. Try local processed file
        local_path = self.data_dir / f"patient_{case_num}.csv"
        if local_path.exists():
            df = pd.read_csv(local_path)
            time_col = "timestamp" if "timestamp" in df.columns else None
            return ObservationNormalizer.normalize_dataframe(
                df,
                source="VitalDB_Local",
                patient_id=case_num,
                time_column=time_col,
            )

        # 2. Try online VitalDB API
        try:
            df = load_case_data(case_num)
            return ObservationNormalizer.normalize_dataframe(
                df,
                source="VitalDB_API",
                patient_id=case_num,
            )
        except Exception:
            # Safe fallback to synthetic data if network unavailable
            df = synthetic_patient_data()
            return ObservationNormalizer.normalize_dataframe(
                df,
                source="VitalDB_Fallback_Synthetic",
                patient_id=case_num,
            )


__all__ = ["VitalDBAdapter"]

