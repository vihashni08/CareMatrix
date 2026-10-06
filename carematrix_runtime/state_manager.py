"""Centralized Multi-Patient State Manager for CareMatrix.

Maintains thread-safe real-time state for all monitored patients, including
vital sign time-series buffers, latest agent assessments, active alerts,
and care coordination actions.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any

from carematrix_runtime.alert_manager import AlertManager
from communication.events import (
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringEvent,
    RiskDecisionEvent,
)


class PatientStateRecord:
    """Thread-safe state container for a single monitored patient."""

    def __init__(self, patient_id: int, name: str | None = None, history_length: int = 120):
        self.patient_id = patient_id
        self.name = name or f"Patient {patient_id}"
        self.history_length = history_length
        self._lock = threading.RLock()

        self.latest_vitals: dict[str, Any] = {}
        self.vital_history: collections.deque[dict[str, Any]] = collections.deque(maxlen=history_length)

        self.current_risk_level: str = "LOW RISK"
        self.current_risk_probability: float | None = None
        self.risk_model_name: str | None = None

        self.latest_monitoring_event: dict[str, Any] | None = None
        self.latest_risk_decision: dict[str, Any] | None = None
        self.latest_data_analysis: dict[str, Any] | None = None
        self.latest_clinical_reasoning: dict[str, Any] | None = None
        self.latest_care_action: dict[str, Any] | None = None

        self.status: str = "STABLE"  # "STABLE", "SURVEILLANCE", "ALERT", "RECOVERING"
        self.scenario: str = "STABLE"
        self.last_updated: float = time.time()

    def update_observation(self, obs: dict[str, Any]) -> None:
        """Record a live incoming vital sign observation."""
        with self._lock:
            self.latest_vitals = dict(obs)
            self.vital_history.append(dict(obs))
            self.scenario = obs.get("scenario", self.scenario)
            self.last_updated = time.time()

    def update_monitoring_event(self, event: MonitoringEvent) -> None:
        """Update state with monitoring agent alert/recovery."""
        with self._lock:
            d = event.to_dict()
            self.latest_monitoring_event = d
            event_type = str(getattr(event, "event_type", "")).lower()
            if "recovery" in event_type or "recovered" in event_type:
                self.status = "RECOVERING"
                self.current_risk_level = "LOW RISK"
            elif "started" in event_type or "active" in event_type:
                self.status = "ALERT"
            self.last_updated = time.time()

    def update_risk_decision(self, event: RiskDecisionEvent) -> None:
        """Update state with risk prediction outcome."""
        with self._lock:
            d = event.to_dict()
            self.latest_risk_decision = d
            self.current_risk_level = event.risk_level
            self.current_risk_probability = event.risk_probability
            self.risk_model_name = event.model_name
            if "HIGH" in event.risk_level:
                self.status = "ALERT"
            self.last_updated = time.time()

    def update_data_analysis(self, event: DataAnalysisEvent) -> None:
        """Update state with analytical evidence metrics."""
        with self._lock:
            self.latest_data_analysis = event.to_dict()
            self.last_updated = time.time()

    def update_clinical_reasoning(self, event: ClinicalReasoningEvent) -> None:
        """Update state with clinical reasoning assessment."""
        with self._lock:
            self.latest_clinical_reasoning = event.to_dict()
            if event.priority == "URGENT":
                self.status = "ALERT"
            elif event.priority == "ELEVATED" and self.status != "ALERT":
                self.status = "SURVEILLANCE"
            elif event.priority == "ROUTINE" and self.status not in ("ALERT", "RECOVERING"):
                self.status = "STABLE"
            self.last_updated = time.time()

    def update_care_action(self, event: CareCoordinationEvent) -> None:
        """Update state with care coordination workflow action."""
        with self._lock:
            self.latest_care_action = event.to_dict()
            self.last_updated = time.time()

    def to_summary_dict(self, active_alerts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Return patient summary for dashboard list view."""
        with self._lock:
            alerts = active_alerts if active_alerts is not None else []
            return {
                "patient_id": self.patient_id,
                "name": self.name,
                "status": self.status,
                "scenario": self.scenario,
                "risk_level": self.current_risk_level,
                "risk_probability": round(self.current_risk_probability, 3) if self.current_risk_probability is not None else None,
                "risk_model_name": self.risk_model_name,
                "active_alert_count": len(alerts),
                "has_active_alert": len(alerts) > 0,
                "latest_vitals": self.latest_vitals,
                "last_updated": round(self.last_updated, 2),
                "care_action_priority": (self.latest_care_action or {}).get("priority", "ROUTINE"),
                "care_action_type": (self.latest_care_action or {}).get("action_type", "CONTINUE_ROUTINE_MONITORING"),
            }

    def to_detail_dict(self, active_alerts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Return complete detailed patient view."""
        with self._lock:
            summary = self.to_summary_dict(active_alerts=active_alerts)
            summary.update({
                "latest_monitoring_event": self.latest_monitoring_event,
                "latest_risk_decision": self.latest_risk_decision,
                "latest_data_analysis": self.latest_data_analysis,
                "latest_clinical_reasoning": self.latest_clinical_reasoning,
                "latest_care_action": self.latest_care_action,
                "active_alerts": active_alerts or [],
                "agent_pipeline": {
                    "monitoring": "active" if self.latest_monitoring_event else "idle",
                    "risk": "active" if self.latest_risk_decision else "idle",
                    "data_analysis": "active" if self.latest_data_analysis else "idle",
                    "clinical_reasoning": "active" if self.latest_clinical_reasoning else "idle",
                    "care_coordination": "active" if self.latest_care_action else "idle",
                },
            })
            return summary


class PatientStateManager:
    """Central repository managing state records for all monitored beds."""

    def __init__(self, alert_manager: AlertManager | None = None):
        self.alert_manager = alert_manager or AlertManager()
        self._patients: dict[int, PatientStateRecord] = {}
        self._lock = threading.RLock()

        # Initialize default demo beds
        self.get_or_create(101, name="Bed 101 (Post-Op General)")
        self.get_or_create(102, name="Bed 102 (ICU Sepsis Surveillance)")
        self.get_or_create(103, name="Bed 103 (Telemetry Floor Cardiac)")

    def get_or_create(self, patient_id: int, name: str | None = None) -> PatientStateRecord:
        with self._lock:
            if patient_id not in self._patients:
                self._patients[patient_id] = PatientStateRecord(patient_id, name=name)
            return self._patients[patient_id]

    def record_observation(self, obs: dict[str, Any]) -> None:
        pid = int(obs.get("patient_id", obs.get("case_id", 101)))
        record = self.get_or_create(pid)
        record.update_observation(obs)

    def handle_event(self, topic: str, event: Any) -> None:
        """Route event from EventQueue to appropriate patient record and alert manager."""
        pid = getattr(event, "patient_id", getattr(event, "case_id", None))
        if pid is None:
            return
        pid = int(pid)
        record = self.get_or_create(pid)

        if isinstance(event, MonitoringEvent):
            record.update_monitoring_event(event)
            self.alert_manager.handle_monitoring_event(event)

        elif isinstance(event, RiskDecisionEvent):
            record.update_risk_decision(event)

        elif isinstance(event, DataAnalysisEvent):
            record.update_data_analysis(event)

        elif isinstance(event, ClinicalReasoningEvent):
            record.update_clinical_reasoning(event)

        elif isinstance(event, CareCoordinationEvent):
            record.update_care_action(event)
            self.alert_manager.handle_care_action(event)

    def get_patient_summary_list(self) -> list[dict[str, Any]]:
        with self._lock:
            summaries = []
            for pid, p in list(self._patients.items()):
                active_alerts = self.alert_manager.get_active_alerts(patient_id=pid)
                summaries.append(p.to_summary_dict(active_alerts=active_alerts))
            return summaries

    def get_patient_detail(self, patient_id: int) -> dict[str, Any] | None:
        with self._lock:
            p = self._patients.get(patient_id)
            if p is None:
                return None
            active_alerts = self.alert_manager.get_active_alerts(patient_id=patient_id)
            return p.to_detail_dict(active_alerts=active_alerts)

    def get_patient_vital_history(self, patient_id: int) -> list[dict[str, Any]]:
        with self._lock:
            p = self._patients.get(patient_id)
            if p is None:
                return []
            return list(p.vital_history)
