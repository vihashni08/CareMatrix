"""Communication layer for CareMatrix agent-to-agent event messaging."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    MONITORING_ALERT = "monitoring_alert"
    MONITORING_RECOVERY = "monitoring_recovery"
    RISK_DECISION = "risk_decision"
    DATA_ANALYSIS = "data_analysis"
    CLINICAL_REASONING = "clinical_reasoning"
    CARE_COORDINATION = "care_coordination"
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_FAILURE = "agent_failure"


class PerformativeType(str, Enum):
    """FIPA-inspired communicative performatives for agent-to-agent negotiation."""

    INFORM = "INFORM"
    REQUEST = "REQUEST"
    PROPOSE = "PROPOSE"
    CHALLENGE = "CHALLENGE"
    AGREE = "AGREE"


class MonitoringDecision(str, Enum):
    CONTINUE_MONITORING = "CONTINUE_MONITORING"
    NO_ACTION = "NO_ACTION"
    ESCALATE_TO_RISK = "ESCALATE_TO_RISK"
    RECOVERY = "RECOVERY"


class RiskDecision(str, Enum):
    LOW_RISK = "LOW_RISK"
    HIGH_RISK = "HIGH_RISK"
    RETRY = "RETRY"
    ESCALATE = "ESCALATE"


class CareActionType(str, Enum):
    CONTINUE_ROUTINE_MONITORING = "CONTINUE_ROUTINE_MONITORING"
    SCHEDULE_CLINICIAN_REVIEW = "SCHEDULE_CLINICIAN_REVIEW"
    TRIGGER_URGENT_CLINICAL_ALERT = "TRIGGER_URGENT_CLINICAL_ALERT"
    REQUEST_DATA_VERIFICATION = "REQUEST_DATA_VERIFICATION"


class CoordinationStatus(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    COMPLETED = "COMPLETED"


@dataclass
class MonitoringEvent:
    """Structured event published by the Monitoring Agent when escalation or recovery occurs."""

    patient_id: int
    event_id: str
    timestamp: float
    event_type: str  # "alert_started", "alert_active", "alert_recovered"
    severity: str    # "mild", "moderate", "severe", "critical"
    affected_vitals: list[str]
    current_values: dict[str, float]
    baseline_values: dict[str, float]
    deviation_values: dict[str, float]
    trends: dict[str, str]
    signal_quality: dict[str, str]
    persistence_duration: int
    recommended_action: str
    vital_details: dict[str, Any] = field(default_factory=dict)
    vital_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    performative: str = PerformativeType.INFORM.value
    challenge_round: int = 0
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["case_id"] = self.patient_id
        d["duration_seconds"] = self.persistence_duration
        return d


@dataclass
class RiskDecisionEvent:
    """Structured event published by the Risk Agent after model evaluation."""

    patient_id: int
    event_id: str
    timestamp: float
    decision: str  # "LOW_RISK", "HIGH_RISK", "RETRY", "ESCALATE"
    risk_probability: float
    risk_level: str
    threshold: float
    model_name: str
    features_used: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    status: str = "success"  # "success", "failed", "retry"
    recommended_action: str = ""
    retry_count: int = 0
    # Context copied from the monitoring event.  These fields let downstream
    # agents analyse the same assessment without reinterpreting model output.
    severity: str = "unknown"
    affected_vitals: list[str] = field(default_factory=list)
    window_start: float | None = None
    window_end: float | None = None
    # Adaptive orchestration metadata
    analysis_level: str = "standard"  # "standard", "detailed", "verification"
    requested_checks: list[str] = field(default_factory=list)
    verification_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    performative: str = PerformativeType.INFORM.value
    challenge_round: int = 0
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    @property
    def case_id(self) -> int:
        return self.patient_id

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["case_id"] = self.patient_id
        return d


@dataclass
class DataAnalysisEvent:
    """Structured, non-diagnostic analytical evidence for Clinical Reasoning."""

    case_id: int
    event_id: str
    timestamp: float
    risk_level: str
    trend_metrics: dict[str, Any] = field(default_factory=dict)
    pattern_identified: list[str] = field(default_factory=list)
    important_changes: list[str] = field(default_factory=list)
    data_quality_flag: bool = False
    analysis_status: str = "complete"
    source: str = "Data Analysis Agent"
    data_quality_details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    # Adaptive orchestration & verification execution fields
    executed_checks: list[str] = field(default_factory=list)
    evidence_consistency: str = "consistent"  # "consistent", "conflicting", "uncertain"
    conflict_flags: list[str] = field(default_factory=list)
    verification_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    performative: str = PerformativeType.INFORM.value
    challenge_round: int = 0
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalReasoningEvent:
    """Structured, non-diagnostic clinical synthesis, evidence report, and prioritized recommendations."""

    case_id: int
    event_id: str
    timestamp: float
    risk_level: str
    priority: str  # "URGENT", "ELEVATED", "ROUTINE"
    clinical_summary: str
    findings: list[str]
    recommended_actions: list[str]
    escalation_required: bool = False
    data_reliability: str = "HIGH"  # "HIGH", "COMPROMISED"
    confidence: float = 1.0
    source: str = "Clinical Reasoning Agent"
    # Cross-agent consistency & evidence verification fields
    evidence_consistency: str = "CONSISTENT"  # "SUPPORTING", "CONFLICTING", "UNCERTAIN"
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    verification_required: bool = False
    # Rich Clinical Intelligence Report fields
    executive_summary: str = ""
    clinical_status: dict[str, Any] = field(default_factory=dict)
    key_findings: list[str] = field(default_factory=list)
    physiological_analysis: list[str] = field(default_factory=list)
    temporal_analysis: list[str] = field(default_factory=list)
    risk_interpretation: str = ""
    evidence_synthesis: str = ""
    medical_evidence: list[dict[str, Any]] = field(default_factory=list)
    clinical_interpretation: str = ""
    uncertainties: list[str] = field(default_factory=list)
    monitoring_priorities: list[str] = field(default_factory=list)
    escalation_rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    performative: str = PerformativeType.INFORM.value
    challenge_round: int = 0
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_clinical_report(self) -> dict[str, Any]:
        """Return the structured 15-section Clinical Intelligence Report."""
        return {
            "executive_summary": self.executive_summary or self.clinical_summary,
            "clinical_status": self.clinical_status or {
                "risk_level": self.risk_level,
                "priority": self.priority,
                "confidence": self.confidence,
                "data_reliability": self.data_reliability,
                "evidence_consistency": self.evidence_consistency,
            },
            "key_findings": self.key_findings or self.findings,
            "physiological_analysis": self.physiological_analysis,
            "temporal_analysis": self.temporal_analysis,
            "risk_interpretation": self.risk_interpretation,
            "supporting_evidence": self.supporting_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "evidence_synthesis": self.evidence_synthesis or self.clinical_summary,
            "medical_evidence": self.medical_evidence,
            "clinical_interpretation": self.clinical_interpretation or self.clinical_summary,
            "uncertainties": self.uncertainties,
            "recommended_actions": self.recommended_actions,
            "monitoring_priorities": self.monitoring_priorities,
            "escalation_rationale": self.escalation_rationale,
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["patient_id"] = self.case_id
        d["clinical_report"] = self.to_clinical_report()
        return d


@dataclass
class AgentHeartbeatEvent:
    """Heartbeat event sent by agents to the Supervisor."""

    agent_name: str
    timestamp: float = field(default_factory=time.time)
    status: str = "healthy"  # "healthy", "degraded", "recovering"
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentFailureEvent:
    """Event emitted when an agent encounters an error or failure."""

    agent_name: str
    error_message: str
    timestamp: float = field(default_factory=time.time)
    recoverable: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CareCoordinationEvent:
    """Structured care coordination action published by Care Coordination Agent."""

    patient_id: int
    event_id: str
    timestamp: float
    action_type: str
    priority: str
    status: str
    reason: str
    clinician_review_required: bool
    suggested_orders: list[str]
    escalation_pathway: str
    clinical_summary: str
    evidence_consistency: str
    data_reliability: str
    confidence: float
    source: str = "Care Coordination Agent"
    correlation_id: str = ""
    action_id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex[:6]}")
    metadata: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

