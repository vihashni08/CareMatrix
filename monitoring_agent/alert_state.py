"""Deterministic lifecycle state handling for persistent monitoring alerts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

try:
    from .config import ALERT_COOLDOWN_SECONDS, RECOVERY_DURATION_SECONDS
except ImportError:  # pragma: no cover - supports direct module execution
    from config import ALERT_COOLDOWN_SECONDS, RECOVERY_DURATION_SECONDS


def manage_alert_lifecycle(
    candidate_alert: pd.Series,
    persistent_alert: pd.Series,
    cooldown: int = ALERT_COOLDOWN_SECONDS,
    recovery_duration: int = RECOVERY_DURATION_SECONDS,
) -> pd.DataFrame:
    """Track alert start, active, recovery, and normal states without duplicates.

    ``alert_started`` occurs only when the existing persistence signal first
    becomes true. An active event enters ``recovering`` when the multi-vital
    condition ends, and closes only after recovery confirmation. A new event
    can begin after recovery and cooldown.
    """
    if not candidate_alert.index.equals(persistent_alert.index):
        raise ValueError("Candidate and persistent alerts must have matching indexes.")
    if cooldown < 0:
        raise ValueError("Alert cooldown cannot be negative.")
    if recovery_duration < 1:
        raise ValueError("Recovery duration must be at least one sample.")

    candidate = candidate_alert.fillna(False).astype(bool)
    persistent = persistent_alert.fillna(False).astype(bool)
    states: list[str] = []
    event_ids: list[int | None] = []
    active = False
    active_event_id: int | None = None
    next_event_id = 1
    cooldown_until_position = -1
    recovery_count = 0

    for position, is_candidate in enumerate(candidate):
        if active:
            if is_candidate:
                recovery_count = 0
                states.append("alert_active")
                event_ids.append(active_event_id)
            else:
                recovery_count += 1
                event_ids.append(active_event_id)
                if recovery_count >= recovery_duration:
                    states.append("alert_recovered")
                    active = False
                    active_event_id = None
                    cooldown_until_position = position + cooldown
                    recovery_count = 0
                else:
                    states.append("recovering")
        elif persistent.iloc[position] and position >= cooldown_until_position:
            active = True
            active_event_id = next_event_id
            next_event_id += 1
            states.append("alert_started")
            event_ids.append(active_event_id)
        elif is_candidate:
            states.append("deviating")
            event_ids.append(None)
        else:
            states.append("normal")
            event_ids.append(None)

    return pd.DataFrame(
        {"alert_state": states, "active_alert_id": event_ids}, index=candidate.index
    )


class PhysiologicalEventTracker:
    """Maintain compact summaries for active and completed monitoring events.

    The tracker stores only event-level statistics per affected vital. It never
    copies the complete VitalDB time series into an event object.
    """

    def __init__(self, case_id: int):
        self.case_id = int(case_id)
        self.active_events: dict[int, dict[str, Any]] = {}
        self.completed_events: list[dict[str, Any]] = []

    def _event_id(self, numeric_id: int) -> str:
        return f"case_{self.case_id}_event_{numeric_id:03d}"

    @staticmethod
    def _summary_from_detail(detail: Mapping[str, Any], timestamp: Any) -> dict[str, Any]:
        current = float(detail["current"])
        deviation = float(detail["relative_deviation"])
        return {
            "initial_value": current,
            "initial_baseline": float(detail["baseline"]),
            "initial_relative_deviation": deviation,
            "initial_direction": detail["direction"],
            "initial_trend": detail["trend"],
            "initial_signal_quality": detail["signal_quality"],
            "start_timestamp": timestamp,
            "latest_value": current,
            "latest_baseline": float(detail["baseline"]),
            "latest_relative_deviation": deviation,
            "latest_direction": detail["direction"],
            "current_trend": detail["trend"],
            "signal_quality": detail["signal_quality"],
            "minimum_value": current,
            "maximum_value": current,
            "peak_absolute_deviation": deviation,
            "latest_timestamp": timestamp,
        }

    @staticmethod
    def _update_summary(summary: dict[str, Any], detail: Mapping[str, Any], timestamp: Any) -> None:
        current = float(detail["current"])
        deviation = float(detail["relative_deviation"])
        summary["latest_value"] = current
        summary["latest_baseline"] = float(detail["baseline"])
        summary["latest_relative_deviation"] = deviation
        summary["latest_direction"] = detail["direction"]
        summary["current_trend"] = detail["trend"]
        summary["signal_quality"] = detail["signal_quality"]
        summary["minimum_value"] = min(summary["minimum_value"], current)
        summary["maximum_value"] = max(summary["maximum_value"], current)
        summary["peak_absolute_deviation"] = max(summary["peak_absolute_deviation"], deviation)
        summary["latest_timestamp"] = timestamp

    def start(
        self,
        numeric_id: int,
        timestamp: Any,
        affected_vitals: list[str],
        details: Mapping[str, Mapping[str, Any]],
        severity: str = "moderate",
    ) -> dict[str, Any]:
        """Start one event and return its compact alert-start snapshot."""
        event = {
            "event_id": self._event_id(numeric_id),
            "case_id": self.case_id,
            "start_timestamp": timestamp,
            "severity": severity,
            "affected_vitals": list(affected_vitals),
            "vital_summary": {
                vital: self._summary_from_detail(details[vital], timestamp)
                for vital in affected_vitals
            },
        }
        self.active_events[numeric_id] = event
        return self.snapshot(numeric_id, timestamp, "alert_started")

    def update(
        self,
        numeric_id: int,
        timestamp: Any,
        details: Mapping[str, Mapping[str, Any]],
        event_state: str = "alert_active",
    ) -> dict[str, Any] | None:
        """Update latest/min/max/peak values and return the current snapshot."""
        event = self.active_events.get(numeric_id)
        if event is None:
            return None
        for vital in event["affected_vitals"]:
            self._update_summary(event["vital_summary"][vital], details[vital], timestamp)
        return self.snapshot(numeric_id, timestamp, event_state)

    def recover(
        self,
        numeric_id: int,
        timestamp: Any,
        details: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        """Apply final readings, close an event, and return its full summary."""
        snapshot = self.update(numeric_id, timestamp, details, "alert_recovered")
        if snapshot is None:
            return None
        completed = deepcopy(snapshot)
        completed["end_timestamp"] = timestamp
        self.completed_events.append(completed)
        self.active_events.pop(numeric_id, None)
        return completed

    def snapshot(self, numeric_id: int, timestamp: Any, event_state: str) -> dict[str, Any]:
        """Return a detached, compact current view suitable for handoff."""
        event = self.active_events[numeric_id]
        return {
            "event_id": event["event_id"],
            "case_id": self.case_id,
            "event_state": event_state,
            "severity": event.get("severity", "moderate"),
            "start_timestamp": event["start_timestamp"],
            "current_timestamp": timestamp,
            "duration_seconds": int(timestamp - event["start_timestamp"] + 1),
            "affected_vitals": list(event["affected_vitals"]),
            "vital_summary": deepcopy(event["vital_summary"]),
        }

    def current_events(self, timestamp: Any) -> list[dict[str, Any]]:
        """Expose only the latest compact state for events still active."""
        return [
            self.snapshot(numeric_id, timestamp, "alert_active")
            for numeric_id in self.active_events
        ]
