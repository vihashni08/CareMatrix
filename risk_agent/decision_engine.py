"""Autonomous decision engine for the CareMatrix Risk Agent."""

from __future__ import annotations

from typing import Any

from communication.events import MonitoringEvent, RiskDecision, RiskDecisionEvent


class RiskDecisionEngine:
    """Evaluates ML model inference results, patient context, and failure conditions."""

    def __init__(self, default_threshold: float = 0.16, max_retries: int = 3):
        self.default_threshold = default_threshold
        self.max_retries = max_retries

    def validate_event(self, event: MonitoringEvent) -> tuple[bool, str]:
        """Validate monitoring event structure and required fields."""
        if not isinstance(event, MonitoringEvent):
            return False, f"Expected MonitoringEvent object, got {type(event).__name__}"
        if event.patient_id is None:
            return False, "Missing patient_id in monitoring event"
        if not event.event_id:
            return False, "Missing event_id in monitoring event"
        if not event.affected_vitals:
            return False, "Missing affected_vitals in monitoring event"
        return True, "Valid"

    def decide_from_prediction(
        self,
        event: MonitoringEvent,
        probability: float,
        threshold: float,
        model_name: str,
        features: dict[str, float],
    ) -> RiskDecisionEvent:
        """Formulate an explicit decision based on ML model probability vs threshold."""
        vitals_str = ", ".join(event.affected_vitals)

        if probability >= threshold:
            decision = RiskDecision.HIGH_RISK
            risk_level = "HIGH RISK"
            reason = (
                f"{model_name} classified alert as HIGH RISK. "
                f"Calculated risk probability ({probability * 100:.2f}%) exceeds the "
                f"trained threshold ({threshold:.2f}). Affected vitals: {vitals_str} (Severity: {event.severity})."
            )
            recommended_action = "Initiate high-risk clinical protocol and notify attending physician."
        else:
            decision = RiskDecision.LOW_RISK
            risk_level = "LOW RISK"
            reason = (
                f"{model_name} classified alert as LOW RISK. "
                f"Calculated risk probability ({probability * 100:.2f}%) is within safe threshold "
                f"({threshold:.2f}) despite physiological deviation in {vitals_str}."
            )
            recommended_action = "Continue close physiological monitoring without immediate clinical escalation."

        return RiskDecisionEvent(
            patient_id=event.patient_id,
            event_id=event.event_id,
            timestamp=event.timestamp,
            decision=decision.value,
            risk_probability=float(probability),
            risk_level=risk_level,
            threshold=float(threshold),
            model_name=model_name,
            features_used=features,
            reason=reason,
            status="success",
            recommended_action=recommended_action,
            severity=event.severity,
            affected_vitals=list(event.affected_vitals),
            # The established Risk Agent feature window remains five minutes.
            window_start=max(0.0, float(event.timestamp) - 300.0),
            window_end=float(event.timestamp),
        )

    def decide_from_failure(
        self,
        event: MonitoringEvent,
        error: Exception,
        retry_count: int,
        model_name: str,
    ) -> RiskDecisionEvent:
        """Handle ML inference or context failures gracefully."""
        if retry_count < self.max_retries:
            decision = RiskDecision.RETRY
            status = "retry"
            reason = f"Transient failure during risk inference: {error}. Retry attempt {retry_count}/{self.max_retries} scheduled."
            recommended_action = "Retry feature extraction and inference."
        else:
            decision = RiskDecision.ESCALATE
            status = "failed"
            reason = f"Exhausted {self.max_retries} retries on model evaluation: {error}. Escalating to clinical team as safety fallback."
            recommended_action = "Manual clinician evaluation required due to automated risk tool failure."

        return RiskDecisionEvent(
            patient_id=event.patient_id,
            event_id=event.event_id,
            timestamp=event.timestamp,
            decision=decision.value,
            risk_probability=-1.0,
            risk_level="INDETERMINATE / ESCALATED" if decision == RiskDecision.ESCALATE else "RETRY_PENDING",
            threshold=self.default_threshold,
            model_name=model_name,
            reason=reason,
            status=status,
            recommended_action=recommended_action,
            retry_count=retry_count,
            severity=event.severity,
            affected_vitals=list(event.affected_vitals),
            window_start=max(0.0, float(event.timestamp) - 300.0),
            window_end=float(event.timestamp),
        )

