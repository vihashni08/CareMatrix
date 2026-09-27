"""Patient-specific monitoring state management across sequential observations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from monitoring_agent.alert_state import PhysiologicalEventTracker
from monitoring_agent.config import (
    ALERT_COOLDOWN_SECONDS,
    BASELINE_WINDOW_SECONDS,
    MIN_BASELINE_SAMPLES,
    PERSISTENCE_SECONDS,
    RECOVERY_DURATION_SECONDS,
    SEVERITY_MIN_VITALS,
    SEVERITY_MULTIPLIERS,
    SEVERITY_PERSISTENCE,
)

VITAL_COLUMNS = ["HR", "MAP", "SpO2", "RR"]
SEVERITY_LEVELS = ("mild", "moderate", "severe", "critical")
_SEVERITY_NUM = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3, "critical": 4}
_NUM_SEVERITY = {v: k for k, v in _SEVERITY_NUM.items()}


class PatientMonitoringState:
    """Maintains progressive state across sequential physiological observations."""

    def __init__(self, case_id: int):
        self.case_id = int(case_id)
        # Buffer of raw observations up to max required window (e.g. 120 samples)
        self.raw_buffer: list[dict[str, Any]] = []
        self.max_buffer_size: int = max(BASELINE_WINDOW_SECONDS * 2, 120)

        # Event tracking
        self.event_tracker = PhysiologicalEventTracker(case_id)
        self.next_event_id: int = 1
        self.active_event_id: int | None = None
        self.active_event_state: str = "normal"  # "normal", "deviating", "alert_started", "alert_active", "recovering", "alert_recovered"
        self.recovery_counter: int = 0
        self.cooldown_until_sample: int = -1

        # Persistence tracking
        self.consecutive_candidate_samples: int = 0
        self.candidate_start_sample_idx: int | None = None
        self.candidate_start_timestamp: float | None = None

        # Statistics and counters
        self.total_observations: int = 0
        self.total_escalations: int = 0
        self.total_recoveries: int = 0
        self.last_heartbeat: float = 0.0

        # Latest computed sample metadata
        self.latest_timestamp: float | None = None
        self.latest_raw_values: dict[str, float] = {}
        self.latest_clean_values: dict[str, float] = {}
        self.latest_baselines: dict[str, float] = {}
        self.latest_deviations: dict[str, float] = {}
        self.latest_trends: dict[str, str] = {}
        self.latest_signal_quality: dict[str, str] = {}
        self.latest_vital_severity: dict[str, str] = {}
        self.latest_overall_severity: str = "normal"
        self.latest_candidate_alert: bool = False
        self.latest_persistent_alert: bool = False
        self.latest_decision: str = "CONTINUE_MONITORING"

    def buffer_dataframe(self) -> pd.DataFrame:
        """Return the recent observations buffer as a pandas DataFrame."""
        if not self.raw_buffer:
            return pd.DataFrame(columns=VITAL_COLUMNS)
        df = pd.DataFrame(self.raw_buffer)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df[VITAL_COLUMNS]

    def add_observation(self, sample: dict[str, Any]) -> None:
        """Append a new observation to the rolling buffer."""
        clean_sample: dict[str, Any] = {}
        for k, v in sample.items():
            if k == "timestamp":
                clean_sample[k] = float(v)
            elif k in VITAL_COLUMNS:
                try:
                    clean_sample[k] = float(v) if v is not None and not np.isnan(float(v)) else np.nan
                except (ValueError, TypeError):
                    clean_sample[k] = np.nan
            else:
                clean_sample[k] = v

        if "timestamp" not in clean_sample:
            clean_sample["timestamp"] = float(self.total_observations)

        self.raw_buffer.append(clean_sample)
        if len(self.raw_buffer) > self.max_buffer_size:
            self.raw_buffer.pop(0)

        self.total_observations += 1
        self.latest_timestamp = clean_sample["timestamp"]
        self.latest_raw_values = {
            vital: clean_sample.get(vital, np.nan) for vital in VITAL_COLUMNS
        }

    def get_snapshot(self) -> dict[str, Any]:
        """Export state snapshot for supervisor checkpointing and recovery."""
        return {
            "case_id": self.case_id,
            "raw_buffer": deepcopy(self.raw_buffer),
            "next_event_id": self.next_event_id,
            "active_event_id": self.active_event_id,
            "active_event_state": self.active_event_state,
            "recovery_counter": self.recovery_counter,
            "cooldown_until_sample": self.cooldown_until_sample,
            "consecutive_candidate_samples": self.consecutive_candidate_samples,
            "candidate_start_sample_idx": self.candidate_start_sample_idx,
            "candidate_start_timestamp": self.candidate_start_timestamp,
            "total_observations": self.total_observations,
            "total_escalations": self.total_escalations,
            "total_recoveries": self.total_recoveries,
            "active_events": deepcopy(self.event_tracker.active_events),
            "completed_events": deepcopy(self.event_tracker.completed_events),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a supervisor checkpoint."""
        self.case_id = snapshot["case_id"]
        self.raw_buffer = deepcopy(snapshot["raw_buffer"])
        self.next_event_id = snapshot["next_event_id"]
        self.active_event_id = snapshot["active_event_id"]
        self.active_event_state = snapshot["active_event_state"]
        self.recovery_counter = snapshot["recovery_counter"]
        self.cooldown_until_sample = snapshot["cooldown_until_sample"]
        self.consecutive_candidate_samples = snapshot["consecutive_candidate_samples"]
        self.candidate_start_sample_idx = snapshot["candidate_start_sample_idx"]
        self.candidate_start_timestamp = snapshot["candidate_start_timestamp"]
        self.total_observations = snapshot["total_observations"]
        self.total_escalations = snapshot["total_escalations"]
        self.total_recoveries = snapshot["total_recoveries"]
        self.event_tracker.active_events = deepcopy(snapshot["active_events"])
        self.event_tracker.completed_events = deepcopy(snapshot["completed_events"])

