"""Metrics tracking for CareMatrix runtime."""

from __future__ import annotations

import time
from typing import Any


class SystemMetricsTracker:
    """Tracks metrics for agent execution, observations, decisions, and system performance."""

    def __init__(self) -> None:
        self.observations_count = 0
        self.bypassed_decisions = 0
        self.escalated_decisions = 0
        self.alerts_resolved = 0
        self.monitoring_events_count = 0
        self.risk_decisions: dict[str, int] = {}
        self.data_analysis_count = 0
        self.quality_flags_count = 0
        self.verification_required_count = 0
        self.clinical_reasoning_count = 0
        self.care_actions_count = 0
        self.agent_executions: dict[str, dict[str, Any]] = {}

    def record_observation(self, patient_id: int) -> None:
        self.observations_count += 1

    def record_agent_execution(self, agent_name: str, latency: float, success: bool = True) -> None:
        if agent_name not in self.agent_executions:
            self.agent_executions[agent_name] = {"count": 0, "total_latency": 0.0, "success_count": 0}
        self.agent_executions[agent_name]["count"] += 1
        self.agent_executions[agent_name]["total_latency"] += latency
        if success:
            self.agent_executions[agent_name]["success_count"] += 1

    def record_monitoring_event(self, event_type: str, candidate: bool = False, persistent: bool = False) -> None:
        self.monitoring_events_count += 1

    def record_risk_decision(self, risk_level: str) -> None:
        self.risk_decisions[risk_level] = self.risk_decisions.get(risk_level, 0) + 1

    def record_data_analysis(self, quality_flag: bool = False, verification_required: bool = False) -> None:
        self.data_analysis_count += 1
        if quality_flag:
            self.quality_flags_count += 1
        if verification_required:
            self.verification_required_count += 1

    def record_clinical_reasoning(self, mode: str, rag_used: bool = False, override: bool = False) -> None:
        self.clinical_reasoning_count += 1

    def record_care_action(self, action_type: str, priority: str, review_required: bool = False, duplicate_suppressed: bool = False) -> None:
        self.care_actions_count += 1

    def record_adaptive_decision(self, bypassed: bool) -> None:
        if bypassed:
            self.bypassed_decisions += 1
        else:
            self.escalated_decisions += 1

    def record_alert_resolution(self) -> None:
        self.alerts_resolved += 1

    def get_summary(self) -> dict[str, Any]:
        return {
            "observations_count": self.observations_count,
            "bypassed_decisions": self.bypassed_decisions,
            "escalated_decisions": self.escalated_decisions,
            "alerts_resolved": self.alerts_resolved,
            "monitoring_events_count": self.monitoring_events_count,
            "risk_decisions": self.risk_decisions,
            "data_analysis_count": self.data_analysis_count,
            "clinical_reasoning_count": self.clinical_reasoning_count,
            "care_actions_count": self.care_actions_count,
            "agent_executions": self.agent_executions,
        }
