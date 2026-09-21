"""State management for the CareMatrix Risk Agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from communication.events import MonitoringEvent, RiskDecisionEvent


class RiskAgentState:
    """Maintains state for the Risk Agent across events, retries, and context."""

    def __init__(self):
        self.received_events: list[dict[str, Any]] = []
        self.decision_history: list[dict[str, Any]] = []
        self.active_evaluations: dict[str, str] = {}
        self.retry_counts: dict[str, int] = {}
        self.failure_count: int = 0
        self.total_processed: int = 0
        self.patient_context_cache: dict[int, dict[str, Any]] = {}
        self.last_heartbeat: float = 0.0

    def record_received_event(self, event: MonitoringEvent) -> None:
        """Log receipt of a monitoring event."""
        self.received_events.append(event.to_dict())
        self.active_evaluations[event.event_id] = "evaluating"
        self.total_processed += 1

    def record_decision(self, decision_event: RiskDecisionEvent) -> None:
        """Log a risk decision and mark event completed."""
        self.decision_history.append(decision_event.to_dict())
        self.active_evaluations[decision_event.event_id] = decision_event.decision

    def record_failure(self, event_id: str, error_msg: str) -> None:
        """Record an inference or processing failure."""
        self.failure_count += 1
        self.active_evaluations[event_id] = f"failed: {error_msg}"

    def get_retry_count(self, event_id: str) -> int:
        return self.retry_counts.get(event_id, 0)

    def increment_retry(self, event_id: str) -> int:
        current = self.retry_counts.get(event_id, 0) + 1
        self.retry_counts[event_id] = current
        return current

    def get_snapshot(self) -> dict[str, Any]:
        """Export snapshot for supervisor state recovery."""
        return {
            "received_events": deepcopy(self.received_events),
            "decision_history": deepcopy(self.decision_history),
            "active_evaluations": deepcopy(self.active_evaluations),
            "retry_counts": deepcopy(self.retry_counts),
            "failure_count": self.failure_count,
            "total_processed": self.total_processed,
            "patient_context_cache": deepcopy(self.patient_context_cache),
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.received_events = deepcopy(snapshot.get("received_events", []))
        self.decision_history = deepcopy(snapshot.get("decision_history", []))
        self.active_evaluations = deepcopy(snapshot.get("active_evaluations", {}))
        self.retry_counts = deepcopy(snapshot.get("retry_counts", {}))
        self.failure_count = snapshot.get("failure_count", 0)
        self.total_processed = snapshot.get("total_processed", 0)
        self.patient_context_cache = deepcopy(snapshot.get("patient_context_cache", {}))

