"""Transparent, deterministic action policies for clinical workflow coordination.

Maps validated ClinicalReasoningEvent fields to appropriate clinical care pathways
without fabricating diagnoses or overriding validated clinical evidence.
"""

from __future__ import annotations

from typing import Any

from care_coordination_agent.schemas import (
    CareActionType,
    CareCoordinationPlan,
    CoordinationPriority,
)
from communication.events import ClinicalReasoningEvent


class ActionPolicy:
    """Configurable deterministic policy mapping clinical reasoning outputs to care actions."""

    def __init__(
        self,
        urgent_escalation_pathway: str = "Rapid Response / Attending Physician Notification",
        elevated_escalation_pathway: str = "Intermediate Inpatient Surveillance & Primary Nurse Review",
        routine_escalation_pathway: str = "Standard Continuous Monitoring Protocol",
    ):
        self.urgent_pathway = urgent_escalation_pathway
        self.elevated_pathway = elevated_escalation_pathway
        self.routine_pathway = routine_escalation_pathway

    def evaluate(self, event: ClinicalReasoningEvent) -> CareCoordinationPlan:
        """Evaluate a ClinicalReasoningEvent against deterministic safety and workflow rules."""
        raw_priority = str(getattr(event, "priority", "ROUTINE")).upper()
        risk_level = str(getattr(event, "risk_level", "LOW RISK")).upper()
        verification_req = bool(getattr(event, "verification_required", False))
        data_reliability = str(getattr(event, "data_reliability", "HIGH")).upper()
        evidence_consistency = str(getattr(event, "evidence_consistency", "CONSISTENT")).upper()
        summary = str(getattr(event, "clinical_summary", ""))
        rec_actions = list(getattr(event, "recommended_actions", []) or [])
        confidence = float(getattr(event, "confidence", 1.0) or 1.0)

        # --------------------------------------------------------------------
        # Rule 1: Data Quality Compromised / Sensor Verification Required
        # If signal quality is questionable, request verification and avoid
        # aggressive interventions on artifact noise.
        # --------------------------------------------------------------------
        if verification_req or data_reliability == "COMPROMISED":
            verification_orders = [
                "Perform manual bedside verification of vital signs (pulse, manual BP, respiratory rate).",
                "Inspect sensor attachments, electrode pads, and pulse oximeter probe placement.",
                "Check pressure transducer zeroing and arterial line calibration if applicable.",
                "Re-evaluate automated risk score once signal fidelity is re-established.",
            ]
            # Prioritize safety: do not escalate to high-dose pharmacotherapy on bad sensors
            # But alert clinician if risk was predicted high
            priority = CoordinationPriority.URGENT if "HIGH" in risk_level else CoordinationPriority.ELEVATED
            return CareCoordinationPlan(
                action_type=CareActionType.REQUEST_DATA_VERIFICATION,
                priority=priority,
                reason=(
                    "Data quality flag active or signal verification required. "
                    "Bedside measurement and sensor integrity confirmation required prior to clinical escalation."
                ),
                clinician_review_required=True,
                suggested_orders=verification_orders + rec_actions,
                escalation_pathway="Bedside Signal Quality Verification Protocol",
                clinical_summary=summary,
                evidence_consistency=evidence_consistency,
                data_reliability=data_reliability,
                confidence=min(confidence, 0.60),
                notes=["Verification required: sensor noise or missing vital data detected."],
            )

        # --------------------------------------------------------------------
        # Rule 2: Conflicting Cross-Agent Evidence
        # If monitoring trends conflict with risk classification, schedule
        # clinician arbitration rather than fabricating certainty.
        # --------------------------------------------------------------------
        if evidence_consistency == "CONFLICTING":
            conflict_orders = [
                "Schedule priority clinician evaluation to arbitrate conflicting clinical evidence.",
                "Review multi-hour physiological trend history and bedside nursing documentation.",
                "Obtain confirmatory 12-lead ECG or lab panel if clinically indicated.",
            ]
            return CareCoordinationPlan(
                action_type=CareActionType.SCHEDULE_CLINICIAN_REVIEW,
                priority=CoordinationPriority.ELEVATED,
                reason=(
                    "Cross-agent conflict detected: physiological trend metrics conflict with risk prediction model. "
                    "Clinician review required to establish ground-truth clinical status."
                ),
                clinician_review_required=True,
                suggested_orders=conflict_orders + rec_actions,
                escalation_pathway=self.elevated_pathway,
                clinical_summary=summary,
                evidence_consistency=evidence_consistency,
                data_reliability=data_reliability,
                confidence=min(confidence, 0.70),
                notes=["Conflicting evidence: deterministic safety rules enforce clinician arbitration."],
            )

        # --------------------------------------------------------------------
        # Rule 3: URGENT Priority (Confirmed Deterioration or Acute Risk)
        # Immediate notification and escalation pathway.
        # --------------------------------------------------------------------
        if raw_priority == "URGENT" or "HIGH" in risk_level and raw_priority != "ROUTINE":
            urgent_orders = [
                "Trigger immediate bedside alert to attending physician / Rapid Response Team.",
                "Increase continuous telemetry observation frequency to high-density monitoring.",
                "Verify functional intravenous access and emergency bedside medications.",
                "Re-assess vital signs every 5 minutes until hemodynamic stabilization.",
            ]
            return CareCoordinationPlan(
                action_type=CareActionType.TRIGGER_URGENT_CLINICAL_ALERT,
                priority=CoordinationPriority.URGENT,
                reason="High clinical risk / acute physiological deterioration confirmed by clinical reasoning.",
                clinician_review_required=True,
                suggested_orders=urgent_orders + rec_actions,
                escalation_pathway=self.urgent_pathway,
                clinical_summary=summary,
                evidence_consistency=evidence_consistency,
                data_reliability=data_reliability,
                confidence=confidence,
                notes=["Urgent clinical alert active: immediate bedside review required."],
            )

        # --------------------------------------------------------------------
        # Rule 4: ELEVATED Priority (Moderate Trend Deterioration)
        # Increase surveillance frequency and schedule clinician assessment.
        # --------------------------------------------------------------------
        if raw_priority == "ELEVATED":
            elevated_orders = [
                "Notify primary care nurse of elevated physiological surveillance status.",
                "Increase vital sign observation frequency to every 15 minutes.",
                "Schedule physician clinical review within 30 minutes.",
                "Review baseline fluid balance and recent medication administrations.",
            ]
            return CareCoordinationPlan(
                action_type=CareActionType.SCHEDULE_CLINICIAN_REVIEW,
                priority=CoordinationPriority.ELEVATED,
                reason="Elevated physiological surveillance indicated by clinical reasoning assessment.",
                clinician_review_required=True,
                suggested_orders=elevated_orders + rec_actions,
                escalation_pathway=self.elevated_pathway,
                clinical_summary=summary,
                evidence_consistency=evidence_consistency,
                data_reliability=data_reliability,
                confidence=confidence,
                notes=["Elevated surveillance scheduled: increased vital observation."],
            )

        # --------------------------------------------------------------------
        # Rule 5: ROUTINE Priority (Stable Physiological State)
        # Standard observation interval, no alarm escalation.
        # --------------------------------------------------------------------
        routine_orders = [
            "Maintain standard continuous floor/ICU telemetry surveillance.",
            "Record routine vital sign observations per standard hospital shift protocol.",
        ]
        return CareCoordinationPlan(
            action_type=CareActionType.CONTINUE_ROUTINE_MONITORING,
            priority=CoordinationPriority.ROUTINE,
            reason="Patient vitals stable and within acceptable baseline limits; low risk trajectory.",
            clinician_review_required=False,
            suggested_orders=routine_orders + rec_actions,
            escalation_pathway=self.routine_pathway,
            clinical_summary=summary or "Stable physiological monitoring; no clinical intervention required.",
            evidence_consistency=evidence_consistency,
            data_reliability=data_reliability,
            confidence=confidence,
            notes=["Routine floor monitoring maintained."],
        )
