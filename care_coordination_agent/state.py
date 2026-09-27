"""State management for the Care Coordination Agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from care_coordination_agent.schemas import CareActionType, CoordinationStatus
from communication.events import CareCoordinationEvent, ClinicalReasoningEvent


@dataclass
class PatientCoordinationState:
    """Tracks active and historical care coordination actions for a single patient."""

    patient_id: int
    active_actions: dict[str, CareCoordinationEvent] = field(default_factory=dict)
    action_history: list[CareCoordinationEvent] = field(default_factory=list)
    last_action_timestamp: float | None = None
    last_action_type: str | None = None
    last_priority: str | None = None
    unresolved_condition: bool = False
    consecutive_duplicate_count: int = 0

    def is_duplicate(
        self,
        event: ClinicalReasoningEvent,
        action_type: CareActionType,
        suppress_window_seconds: float = 60.0,
    ) -> bool:
        """Check if an active, unresolved action for this condition already exists.

        Prevents repeated duplicate alarms and task spam for the same ongoing clinical state.
        """
        now = time.time()
        # If there is already an active action of the same action_type and priority
        for act in self.active_actions.values():
            if act.status in (CoordinationStatus.ACTIVE.value, CoordinationStatus.NEW.value):
                if act.action_type == action_type.value and act.priority == event.priority:
                    # Same condition still active and unresolved
                    return True

        # Also check recent timestamp window if last action was identical
        if self.last_action_type == action_type.value and self.last_priority == event.priority:
            if self.last_action_timestamp and (now - self.last_action_timestamp) < suppress_window_seconds:
                return True

        return False

    def record_action(self, action_event: CareCoordinationEvent) -> None:
        """Record newly generated care action."""
        self.active_actions[action_event.action_id] = action_event
        self.action_history.append(action_event)
        self.last_action_timestamp = action_event.created_at
        self.last_action_type = action_event.action_type
        self.last_priority = action_event.priority
        self.unresolved_condition = action_event.priority in ("URGENT", "ELEVATED")

    def acknowledge_action(self, action_id: str, clinician_id: str = "clinician_on_duty") -> CareCoordinationEvent | None:
        """Mark an active action as acknowledged by clinical staff."""
        if action_id in self.active_actions:
            act = self.active_actions[action_id]
            act.status = CoordinationStatus.ACKNOWLEDGED.value
            act.metadata["acknowledged_by"] = clinician_id
            act.metadata["acknowledged_at"] = time.time()
            return act
        return None

    def resolve_condition(self, resolution_reason: str = "Patient physiological stabilization") -> list[CareCoordinationEvent]:
        """Resolve all active actions when patient recovers or condition stabilizes."""
        resolved = []
        for action_id, act in list(self.active_actions.items()):
            if act.status in (CoordinationStatus.NEW.value, CoordinationStatus.ACTIVE.value, CoordinationStatus.ACKNOWLEDGED.value):
                act.status = CoordinationStatus.RESOLVED.value
                act.metadata["resolved_at"] = time.time()
                act.metadata["resolution_reason"] = resolution_reason
                resolved.append(act)
                del self.active_actions[action_id]
        self.unresolved_condition = False
        return resolved

    def to_dict(self) -> dict[str, Any]:
        """Return serializable state dict."""
        return {
            "patient_id": self.patient_id,
            "active_actions_count": len(self.active_actions),
            "total_actions": len(self.action_history),
            "last_action_type": self.last_action_type,
            "last_priority": self.last_priority,
            "unresolved_condition": self.unresolved_condition,
        }
