"""Patient-level clinical reasoning state and context tracking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClinicalReasoningState:
    """Maintains patient clinical reasoning history, prior assessments, and active alerts."""

    case_id: int
    current_risk_status: str = "NORMAL"
    recent_findings: list[str] = field(default_factory=list)
    previous_recommendations: list[str] = field(default_factory=list)
    active_alerts: list[str] = field(default_factory=list)
    previous_events: list[dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    last_priority: str | None = None
    last_evaluated_timestamp: float | None = None

    def record_assessment(
        self,
        event_id: str,
        timestamp: float,
        priority: str,
        risk_level: str,
        summary: str,
        findings: list[str],
        recommendations: list[str],
    ) -> None:
        """Record completed clinical reasoning cycle."""
        self.total_evaluations += 1
        self.last_priority = priority
        self.current_risk_status = risk_level
        self.recent_findings = list(findings)
        self.previous_recommendations = list(recommendations)
        self.last_evaluated_timestamp = timestamp

        record = {
            "event_id": event_id,
            "timestamp": timestamp,
            "priority": priority,
            "risk_level": risk_level,
            "summary": summary,
            "findings": list(findings),
            "recommendations": list(recommendations),
        }
        self.previous_events.append(record)
        if event_id not in self.active_alerts:
            self.active_alerts.append(event_id)

    def get_snapshot(self) -> dict[str, Any]:
        """Export state snapshot for supervisor checkpointing."""
        return {
            "case_id": self.case_id,
            "current_risk_status": self.current_risk_status,
            "recent_findings": deepcopy(self.recent_findings),
            "previous_recommendations": deepcopy(self.previous_recommendations),
            "active_alerts": list(self.active_alerts),
            "previous_events": deepcopy(self.previous_events),
            "total_evaluations": self.total_evaluations,
            "last_priority": self.last_priority,
            "last_evaluated_timestamp": self.last_evaluated_timestamp,
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from supervisor checkpoint."""
        self.case_id = snapshot.get("case_id", self.case_id)
        self.current_risk_status = snapshot.get("current_risk_status", "NORMAL")
        self.recent_findings = deepcopy(snapshot.get("recent_findings", []))
        self.previous_recommendations = deepcopy(snapshot.get("previous_recommendations", []))
        self.active_alerts = list(snapshot.get("active_alerts", []))
        self.previous_events = deepcopy(snapshot.get("previous_events", []))
        self.total_evaluations = snapshot.get("total_evaluations", 0)
        self.last_priority = snapshot.get("last_priority")
        self.last_evaluated_timestamp = snapshot.get("last_evaluated_timestamp")


__all__ = ["ClinicalReasoningState"]
