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
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_FAILURE = "agent_failure"


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
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
