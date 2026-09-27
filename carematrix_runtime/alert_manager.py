"""CareMatrix Clinical Alert Lifecycle Manager.

Manages transition of alerts through states:
NEW → ACTIVE → ACKNOWLEDGED → RESOLVED
Prevents duplicate alert spam for the same active condition,
tracks clinician acknowledgment, and resolves alerts upon physiological recovery.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from communication.events import CareCoordinationEvent, MonitoringEvent


class AlertLifecycleState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class ClinicalAlert:
    """Represents an active or resolved clinical alert with full provenance."""

    alert_id: str
    patient_id: int
    event_id: str
    severity: str
    affected_vitals: list[str]
    state: AlertLifecycleState
    start_time: float
    last_update_time: float
    current_values: dict[str, Any] = field(default_factory=dict)
    acknowledged_at: float | None = None
    acknowledged_by: str | None = None
    resolved_at: float | None = None
    resolution_reason: str | None = None
    priority: str = "URGENT"
    care_action_id: str | None = None
    care_action_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value if isinstance(self.state, AlertLifecycleState) else str(self.state)
        d["duration_seconds"] = round((self.resolved_at or time.time()) - self.start_time, 1)
        return d


class AlertManager:
    """Central repository and state machine for clinical alerts."""

    def __init__(self):
        self._active_alerts: dict[str, ClinicalAlert] = {}
        self._alert_history: list[ClinicalAlert] = []
        self._lock = threading.RLock()

    def handle_monitoring_event(self, event: MonitoringEvent) -> ClinicalAlert | None:
        """Process incoming monitoring alert or recovery event."""
        patient_id = int(event.patient_id)
        event_type = str(getattr(event, "event_type", "alert_started")).lower()

        # Check for recovery event
        if "recovery" in event_type or "recovered" in event_type:
            self.resolve_patient_alerts(patient_id, reason="Physiological deviation resolved; vitals normalized")
            return None

        with self._lock:
            # Check if an alert for this patient with overlapping affected vitals is already active
            for existing in list(self._active_alerts.values()):
                if existing.patient_id == patient_id and existing.state in (AlertLifecycleState.NEW, AlertLifecycleState.ACTIVE, AlertLifecycleState.ACKNOWLEDGED):
                    if set(existing.affected_vitals) == set(event.affected_vitals):
                        # Ongoing active condition: update timestamp and values rather than spamming a new alert
                        existing.last_update_time = time.time()
                        existing.current_values = dict(event.current_values)
                        existing.severity = event.severity
                        if existing.state == AlertLifecycleState.NEW:
                            existing.state = AlertLifecycleState.ACTIVE
                        return existing

            # Create new clinical alert
            alert_id = f"alert_{patient_id}_{int(event.timestamp)}_{uuid.uuid4().hex[:4]}"
            alert = ClinicalAlert(
                alert_id=alert_id,
                patient_id=patient_id,
                event_id=event.event_id,
                severity=event.severity,
                affected_vitals=list(event.affected_vitals),
                state=AlertLifecycleState.NEW,
                start_time=time.time(),
                last_update_time=time.time(),
                current_values=dict(event.current_values),
                priority="URGENT" if event.severity in ("severe", "critical") else "ELEVATED",
                metadata={"recommended_action": event.recommended_action},
            )
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            return alert

    def handle_care_action(self, action_event: CareCoordinationEvent) -> None:
        """Attach care coordination details to the corresponding active alert."""
        with self._lock:
            for alert in list(self._active_alerts.values()):
                if alert.patient_id == action_event.patient_id and alert.event_id == action_event.event_id:
                    alert.care_action_id = action_event.action_id
                    alert.care_action_type = action_event.action_type
                    alert.priority = action_event.priority
                    break

    def acknowledge_alert(self, alert_id: str, clinician_id: str = "clinician_on_duty") -> ClinicalAlert | None:
        """Mark an active alert as acknowledged by a clinician."""
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert is not None:
                alert.state = AlertLifecycleState.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = clinician_id
                return alert
            return None

    def resolve_alert(self, alert_id: str, reason: str = "Clinical resolution confirmed") -> ClinicalAlert | None:
        """Resolve a specific alert."""
        with self._lock:
            alert = self._active_alerts.get(alert_id)
            if alert is not None:
                alert.state = AlertLifecycleState.RESOLVED
                alert.resolved_at = time.time()
                alert.resolution_reason = reason
                del self._active_alerts[alert_id]
                return alert
            return None

    def resolve_patient_alerts(self, patient_id: int, reason: str = "Patient recovery detected") -> list[ClinicalAlert]:
        """Resolve all active alerts for a patient upon recovery."""
        resolved = []
        now = time.time()
        with self._lock:
            for alert_id, alert in list(self._active_alerts.items()):
                if alert.patient_id == patient_id:
                    alert.state = AlertLifecycleState.RESOLVED
                    alert.resolved_at = now
                    alert.resolution_reason = reason
                    resolved.append(alert)
                    del self._active_alerts[alert_id]
            return resolved

    def get_active_alerts(self, patient_id: int | None = None) -> list[dict[str, Any]]:
        """Return list of active alerts, optionally filtered by patient."""
        with self._lock:
            alerts = list(self._active_alerts.values())
        if patient_id is not None:
            alerts = [a for a in alerts if a.patient_id == patient_id]
        return [a.to_dict() for a in alerts]

    def get_alert_history(self, patient_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return historical alerts."""
        with self._lock:
            alerts = list(self._alert_history)
        if patient_id is not None:
            alerts = [a for a in alerts if a.patient_id == patient_id]
        return [a.to_dict() for a in alerts[-limit:]]
