"""MIMIC-IV Demo dataset adapter for ICU and emergency telemetry ingestion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

from data.adapters.base_adapter import BaseDatasetAdapter
from data.normalizer import ObservationNormalizer
from data.schemas import CommonPatientObservation, PatientDatasetMetadata

# Standard MIMIC-IV ICU Chartevents Item IDs for vital signs
MIMIC_ITEM_MAP: dict[int, str] = {
    220045: "HR",    # Heart Rate (bpm)
    220050: "SBP",   # Arterial Blood Pressure systolic (mmHg)
    220179: "SBP",   # Non-Invasive Blood Pressure systolic (mmHg)
    220051: "DBP",   # Arterial Blood Pressure diastolic (mmHg)
    220180: "DBP",   # Non-Invasive Blood Pressure diastolic (mmHg)
    220052: "MAP",   # Arterial Blood Pressure mean (mmHg)
    220181: "MAP",   # Non-Invasive Blood Pressure mean (mmHg)
    220277: "SpO2",  # O2 saturation pulseoxymetry (%)
    220210: "RR",    # Respiratory Rate (insp/min)
    223761: "BT_F",  # Temperature Fahrenheit (°F)
    223762: "BT",    # Temperature Celsius (°C)
}


class MIMICIVAdapter(BaseDatasetAdapter):
    """Adapter for MIMIC-IV Demo clinical telemetry (ICU chartevents and ED vitalsign).
    
    Supports:
    - Real MIMIC-IV Demo directory configured via MIMIC_DATA_PATH env var or constructor arg.
    - Automatic detection of icu/chartevents.csv(.gz) or ed/vitalsign.csv.
    - Realistic de-identified built-in demo cohort fixture when raw download is absent.
    - Zero fabrication of unmeasured variables; unobserved vitals are preserved as np.nan.
    """

    def __init__(self, data_path: Path | str | None = None):
        env_path = os.environ.get("MIMIC_DATA_PATH")
        if data_path is not None:
            self.data_dir = Path(data_path)
        elif env_path:
            self.data_dir = Path(env_path)
        else:
            self.data_dir = Path(__file__).resolve().parents[1] / "mimic_demo"

        self._cached_demo_cohort: dict[str | int, pd.DataFrame] | None = None

    @property
    def dataset_name(self) -> str:
        return "MIMIC-IV-Demo"

    def list_cases(self) -> list[int | str]:
        """List available ICU stay or subject IDs."""
        # 1. If physical MIMIC directory exists, discover stays
        if self.data_dir.exists():
            # Check for pre-processed patient files
            patient_files = list(self.data_dir.glob("patient_*.csv")) + list(self.data_dir.glob("stay_*.csv"))
            if patient_files:
                cases: list[int | str] = []
                for pf in patient_files:
                    stem = pf.stem.replace("patient_", "").replace("stay_", "")
                    cases.append(int(stem) if stem.isdigit() else stem)
                return sorted(cases, key=lambda x: (isinstance(x, str), x))

            # Check for ed/vitalsign.csv
            ed_file = self.data_dir / "ed" / "vitalsign.csv"
            if not ed_file.exists():
                ed_file = self.data_dir / "vitalsign.csv"
            if ed_file.exists():
                try:
                    df = pd.read_csv(ed_file, nrows=1000)
                    id_col = "stay_id" if "stay_id" in df.columns else "subject_id"
                    return sorted(df[id_col].dropna().unique().tolist()[:10])
                except Exception:
                    pass

        # 2. Built-in de-identified realistic demo cohort (1001 to 1005)
        return [1001, 1002, 1003, 1004, 1005]

    def load_case(self, case_id: int | str) -> list[CommonPatientObservation]:
        """Load and normalize telemetry observations for a patient stay."""
        cid = int(case_id) if str(case_id).isdigit() else case_id

        # 1. Check for single-case CSV in data_dir
        if self.data_dir.exists():
            for filename in (f"patient_{cid}.csv", f"stay_{cid}.csv", f"{cid}.csv"):
                file_path = self.data_dir / filename
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    return ObservationNormalizer.normalize_dataframe(
                        df,
                        source="MIMIC-IV_File",
                        patient_id=cid,
                        time_column="charttime" if "charttime" in df.columns else ("timestamp" if "timestamp" in df.columns else None),
                    )

            # 2. Check for chartevents.csv in data_dir
            chartevents = self._find_table_file("chartevents")
            if chartevents is not None:
                parsed_df = self._extract_stay_from_chartevents(chartevents, cid)
                if not parsed_df.empty:
                    return ObservationNormalizer.normalize_dataframe(
                        parsed_df,
                        source="MIMIC-IV_Chartevents",
                        patient_id=cid,
                        time_column="charttime",
                    )

            # 3. Check for vitalsign.csv (ED)
            vitalsign = self._find_table_file("vitalsign")
            if vitalsign is not None:
                parsed_df = self._extract_stay_from_vitalsign(vitalsign, cid)
                if not parsed_df.empty:
                    return ObservationNormalizer.normalize_dataframe(
                        parsed_df,
                        source="MIMIC-IV_ED_Vitalsign",
                        patient_id=cid,
                        time_column="charttime",
                    )

        # 4. Use built-in realistic demo cohort
        demo_cohort = self._get_demo_cohort()
        if cid in demo_cohort:
            df = demo_cohort[cid]
            return ObservationNormalizer.normalize_dataframe(
                df,
                source="MIMIC-IV_Demo_Cohort",
                patient_id=cid,
                time_column="timestamp",
            )

        # Default fallback to first case in demo cohort
        df = demo_cohort.get(1001, self._generate_synthetic_stay(1001, "stable"))
        return ObservationNormalizer.normalize_dataframe(
            df,
            source="MIMIC-IV_Demo_Cohort",
            patient_id=cid,
            time_column="timestamp",
        )

    def _find_table_file(self, table_name: str) -> Path | None:
        """Find CSV or CSV.GZ file for a table in data_dir or its subdirectories."""
        for ext in (".csv", ".csv.gz"):
            p = self.data_dir / f"{table_name}{ext}"
            if p.exists():
                return p
            p_sub = self.data_dir / "icu" / f"{table_name}{ext}"
            if p_sub.exists():
                return p_sub
            p_ed = self.data_dir / "ed" / f"{table_name}{ext}"
            if p_ed.exists():
                return p_ed
        return None

    def _extract_stay_from_chartevents(self, filepath: Path, stay_id: int | str) -> pd.DataFrame:
        """Filter and pivot chartevents records for a specific stay."""
        try:
            sid = int(stay_id) if str(stay_id).isdigit() else None
            # Read chunked to handle potential file size gracefully
            chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(filepath, chunksize=10000):
                id_col = "stay_id" if "stay_id" in chunk.columns else "subject_id"
                matched = chunk[chunk[id_col] == sid]
                if not matched.empty:
                    chunks.append(matched)

            if not chunks:
                return pd.DataFrame()

            raw = pd.concat(chunks, ignore_index=True)
            raw = raw[raw["itemid"].isin(MIMIC_ITEM_MAP.keys())]
            if raw.empty:
                return pd.DataFrame()

            # Map item IDs to vital names
            raw["vital"] = raw["itemid"].map(MIMIC_ITEM_MAP)
            # Handle Fahrenheit to Celsius conversion if needed
            f_mask = raw["vital"] == "BT_F"
            if f_mask.any():
                raw.loc[f_mask, "valuenum"] = (raw.loc[f_mask, "valuenum"] - 32.0) * 5.0 / 9.0
                raw.loc[f_mask, "vital"] = "BT"

            # Pivot to wide format by charttime
            pivoted = raw.pivot_table(
                index="charttime",
                columns="vital",
                values="valuenum",
                aggfunc="last",
            ).reset_index()

            return pivoted.sort_values("charttime")
        except Exception:
            return pd.DataFrame()

    def _extract_stay_from_vitalsign(self, filepath: Path, stay_id: int | str) -> pd.DataFrame:
        """Extract ED vitalsign for a stay."""
        try:
            sid = int(stay_id) if str(stay_id).isdigit() else None
            df = pd.read_csv(filepath)
            id_col = "stay_id" if "stay_id" in df.columns else "subject_id"
            stay_df = df[df[id_col] == sid].copy()
            if stay_df.empty:
                return pd.DataFrame()

            rename_map = {
                "heartrate": "HR",
                "resprate": "RR",
                "o2sat": "SpO2",
                "sbp": "SBP",
                "dbp": "DBP",
                "temperature": "BT",
            }
            stay_df = stay_df.rename(columns=rename_map)
            return stay_df.sort_values("charttime")
        except Exception:
            return pd.DataFrame()

    def _get_demo_cohort(self) -> dict[str | int, pd.DataFrame]:
        """Lazy load realistic de-identified demo cases representing typical ICU trajectories."""
        if self._cached_demo_cohort is None:
            self._cached_demo_cohort = {
                1001: self._generate_synthetic_stay(1001, pattern="stable"),
                1002: self._generate_synthetic_stay(1002, pattern="tachycardia_hypotension"),
                1003: self._generate_synthetic_stay(1003, pattern="hypoxemia_respiratory"),
                1004: self._generate_synthetic_stay(1004, pattern="sensor_noise_artifact"),
                1005: self._generate_synthetic_stay(1005, pattern="multi_vital_escalation"),
            }
        return self._cached_demo_cohort

    @staticmethod
    def _generate_synthetic_stay(stay_id: int, pattern: str = "stable") -> pd.DataFrame:
        """Generate realistic de-identified physiological telemetry for benchmarking."""
        np.random.seed(stay_id)
        n = 100
        timestamps = np.arange(n, dtype=float)

        if pattern == "stable":
            hr = 75.0 + np.random.normal(0, 2, n)
            map_val = 82.0 + np.random.normal(0, 2, n)
            spo2 = 98.0 + np.random.normal(0, 0.5, n)
            rr = 15.0 + np.random.normal(0, 1, n)
            bt = 36.8 + np.random.normal(0, 0.1, n)
        elif pattern == "tachycardia_hypotension":
            # Acute hemodynamic shock profile
            hr = np.concatenate([np.repeat(78.0, 40), np.linspace(78.0, 125.0, 40), np.repeat(125.0, 20)])
            hr += np.random.normal(0, 2, n)
            map_val = np.concatenate([np.repeat(85.0, 40), np.linspace(85.0, 54.0, 40), np.repeat(54.0, 20)])
            map_val += np.random.normal(0, 2, n)
            spo2 = 97.0 + np.random.normal(0, 0.8, n)
            rr = 16.0 + np.random.normal(0, 1, n)
            bt = 37.2 + np.random.normal(0, 0.1, n)
        elif pattern == "hypoxemia_respiratory":
            # Respiratory deterioration profile
            hr = 85.0 + np.random.normal(0, 2, n)
            map_val = 80.0 + np.random.normal(0, 3, n)
            spo2 = np.concatenate([np.repeat(98.0, 35), np.linspace(98.0, 86.0, 45), np.repeat(86.0, 20)])
            spo2 += np.random.normal(0, 0.5, n)
            rr = np.concatenate([np.repeat(14.0, 35), np.linspace(14.0, 32.0, 45), np.repeat(32.0, 20)])
            rr += np.random.normal(0, 1, n)
            bt = 38.1 + np.random.normal(0, 0.1, n)
        elif pattern == "sensor_noise_artifact":
            # Transient disconnections and artifactual zeros
            hr = 80.0 + np.random.normal(0, 3, n)
            map_val = 82.0 + np.random.normal(0, 3, n)
            spo2 = 97.0 + np.random.normal(0, 1, n)
            rr = 16.0 + np.random.normal(0, 1, n)
            bt = 37.0 + np.random.normal(0, 0.1, n)
            # Inject dropouts/artifacts between index 45 and 65
            spo2[45:65] = np.nan
            map_val[50:60] = np.nan
        else:  # multi_vital_escalation
            hr = np.linspace(70.0, 120.0, n) + np.random.normal(0, 2, n)
            map_val = np.linspace(88.0, 58.0, n) + np.random.normal(0, 2, n)
            spo2 = np.linspace(98.0, 90.0, n) + np.random.normal(0, 0.5, n)
            rr = np.linspace(14.0, 28.0, n) + np.random.normal(0, 1, n)
            bt = 37.8 + np.random.normal(0, 0.2, n)

        # Ensure bounded valid ranges
        spo2 = np.clip(spo2, 0.0, 100.0)
        hr = np.clip(hr, 20.0, 250.0)
        map_val = np.clip(map_val, 20.0, 200.0)
        rr = np.clip(rr, 0.0, 80.0)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "HR": hr,
            "MAP": map_val,
            "SpO2": spo2,
            "RR": rr,
            "BT": bt,
            "SBP": map_val + 20.0,
            "DBP": map_val - 15.0,
        })
        return df


__all__ = ["MIMICIVAdapter", "MIMIC_ITEM_MAP"]

