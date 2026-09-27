"""Decision engine for the Care Coordination Agent."""

from __future__ import annotations

import time
import uuid
from typing import Any

from care_coordination_agent.action_policy import ActionPolicy
from care_coordination_agent.schemas import (
    CareActionType,
    CareCoordinationPlan,
    CoordinationPriority,
    CoordinationStatus,
)
from care_coordination_agent.state import PatientCoordinationState
from communication.events import CareCoordinationEvent, ClinicalReasoningEvent


class CareCoordinationEngine:
    """Evaluates validated clinical reasoning outputs and generates coordinated workflow actions."""

    def __init__(self, action_policy: ActionPolicy | None = None):
        self.policy = action_policy or ActionPolicy()

    def coordinate(
        self,
        event: ClinicalReasoningEvent,
        state: PatientCoordinationState,
    ) -> CareCoordinationEvent:
        """Determine and synthesize appropriate workflow actions from clinical reasoning."""
        patient_id = int(getattr(event, "case_id", 0) or getattr(event, "patient_id", 0) or 0)
        event_id = str(getattr(event, "event_id", "unknown_event"))
        timestamp = float(getattr(event, "timestamp", 0.0) or time.time())

        # 1. Deterministic Action Policy Evaluation
        plan: CareCoordinationPlan = self.policy.evaluate(event)

        # 2. Check Deduplication against active state
        is_dup = state.is_duplicate(event, plan.action_type)

        # If duplicate ongoing condition, preserve status as ACTIVE without duplicate notification spam
        status = CoordinationStatus.ACTIVE.value if is_dup else CoordinationStatus.NEW.value

        action_event = CareCoordinationEvent(
            patient_id=patient_id,
            event_id=event_id,
            timestamp=timestamp,
            action_type=plan.action_type.value,
            priority=plan.priority.value,
            status=status,
            reason=plan.reason,
            clinician_review_required=plan.clinician_review_required,
            suggested_orders=plan.suggested_orders,
            escalation_pathway=plan.escalation_pathway,
            clinical_summary=plan.clinical_summary,
            evidence_consistency=plan.evidence_consistency,
            data_reliability=plan.data_reliability,
            confidence=plan.confidence,
            source="Care Coordination Agent",
            correlation_id=event_id,
            action_id=f"action_{patient_id}_{int(timestamp)}_{uuid.uuid4().hex[:6]}",
            metadata={
                "is_duplicate_suppressed": is_dup,
                "policy_notes": plan.notes,
                "original_clinical_priority": event.priority,
            },
        )

        # Record in state
        state.record_action(action_event)
        return action_event
