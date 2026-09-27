"""Schemas and data models for CareMatrix Care Coordination Agent."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from communication.events import CareActionType, CareCoordinationEvent, CoordinationStatus


class CoordinationPriority(str, Enum):
    ROUTINE = "ROUTINE"
    ELEVATED = "ELEVATED"
    URGENT = "URGENT"


class ActionState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    COMPLETED = "COMPLETED"


@dataclass
class CareCoordinationPlan:
    """Deterministic care workflow plan generated from clinical reasoning evidence."""

    action_type: CareActionType
    priority: CoordinationPriority
    reason: str
    clinician_review_required: bool
    suggested_orders: list[str]
    escalation_pathway: str
    clinical_summary: str
    evidence_consistency: str
    data_reliability: str
    confidence: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value if isinstance(self.action_type, CareActionType) else str(self.action_type),
            "priority": self.priority.value if isinstance(self.priority, CoordinationPriority) else str(self.priority),
            "reason": self.reason,
            "clinician_review_required": self.clinician_review_required,
            "suggested_orders": list(self.suggested_orders),
            "escalation_pathway": self.escalation_pathway,
            "clinical_summary": self.clinical_summary,
            "evidence_consistency": self.evidence_consistency,
            "data_reliability": self.data_reliability,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


__all__ = [
    "ActionState",
    "CareActionType",
    "CareCoordinationEvent",
    "CareCoordinationPlan",
    "CoordinationPriority",
    "CoordinationStatus",
]
