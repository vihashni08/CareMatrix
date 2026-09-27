"""Base dataset adapter defining interface for clinical data ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator
import numpy as np

from data.schemas import CommonPatientObservation, PatientDatasetMetadata


class BaseDatasetAdapter(ABC):
    """Abstract interface for dataset adapters converting clinical data into CareMatrix telemetry."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return human-readable dataset identifier."""
        pass

    @abstractmethod
    def list_cases(self) -> list[int | str]:
        """Return list of available patient case or stay identifiers."""
        pass

    @abstractmethod
    def load_case(self, case_id: int | str) -> list[CommonPatientObservation]:
        """Load and normalize sequential observations for a patient case."""
        pass

    def get_metadata(self, case_id: int | str) -> PatientDatasetMetadata:
        """Generate summary metadata and missingness profile for a case."""
        observations = self.load_case(case_id)
        if not observations:
            return PatientDatasetMetadata(
                dataset_name=self.dataset_name,
                patient_id=case_id,
                sample_count=0,
                duration_seconds=0.0,
                available_vitals=[],
            )

        vitals = ["HR", "MAP", "SpO2", "RR", "BT", "SBP", "DBP"]
        counts: dict[str, int] = {v: 0 for v in vitals}
        total = len(observations)

        for obs in observations:
            for v in vitals:
                val = getattr(obs, v, None)
                if val is not None and not np.isnan(val):
                    counts[v] += 1

        available = [v for v, c in counts.items() if c > 0]
        missingness = {v: round(1.0 - (c / total), 3) for v, c in counts.items()}

        duration = observations[-1].timestamp - observations[0].timestamp if total > 1 else 0.0

        return PatientDatasetMetadata(
            dataset_name=self.dataset_name,
            patient_id=case_id,
            sample_count=total,
            duration_seconds=float(duration),
            available_vitals=available,
            missingness_rates=missingness,
        )

    def stream_case(
        self,
        case_id: int | str,
        max_samples: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream normalized samples ready for direct consumption by MonitoringAgent."""
        observations = self.load_case(case_id)
        count = 0
        for obs in observations:
            yield obs.to_monitoring_sample()
            count += 1
            if max_samples is not None and count >= max_samples:
                break


__all__ = ["BaseDatasetAdapter"]

