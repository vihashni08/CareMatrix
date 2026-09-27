"""Common data representations and schemas for clinical patient telemetry across datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class CommonPatientObservation:
    """Standardized physiological observation record across VitalDB, MIMIC-IV, and synthetic streams.
    
    Adheres strictly to the CareMatrix monitoring conventions:
    HR in bpm, MAP/SBP/DBP in mmHg, SpO2 in %, RR in insp/min, BT in °C.
    Unavailable physiological parameters are represented as np.nan or None.
    """

    timestamp: float
    HR: float | None = None
    MAP: float | None = None
    SpO2: float | None = None
    RR: float | None = None
    BT: float | None = None
    SBP: float | None = None
    DBP: float | None = None
    source: str = "unknown"
    patient_id: str | int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_monitoring_sample(self) -> dict[str, Any]:
        """Convert to the standard dictionary consumed by MonitoringAgent and event buffers."""
        return {
            "timestamp": float(self.timestamp),
            "HR": float(self.HR) if self.HR is not None and not np.isnan(self.HR) else np.nan,
            "MAP": float(self.MAP) if self.MAP is not None and not np.isnan(self.MAP) else np.nan,
            "SpO2": float(self.SpO2) if self.SpO2 is not None and not np.isnan(self.SpO2) else np.nan,
            "RR": float(self.RR) if self.RR is not None and not np.isnan(self.RR) else np.nan,
            "BT": float(self.BT) if self.BT is not None and not np.isnan(self.BT) else np.nan,
            "SBP": float(self.SBP) if self.SBP is not None and not np.isnan(self.SBP) else np.nan,
            "DBP": float(self.DBP) if self.DBP is not None and not np.isnan(self.DBP) else np.nan,
            "patient_id": self.patient_id,
            "source": self.source,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert entire observation record to dictionary."""
        d = self.to_monitoring_sample()
        d["metadata"] = dict(self.metadata)
        return d


@dataclass
class PatientDatasetMetadata:
    """Metadata describing a patient case or clinical telemetry stream."""

    dataset_name: str
    patient_id: str | int
    sample_count: int
    duration_seconds: float
    available_vitals: list[str]
    missingness_rates: dict[str, float] = field(default_factory=dict)
    ground_truth_label: str | int | None = None
    extra_attributes: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CommonPatientObservation",
    "PatientDatasetMetadata",
]

