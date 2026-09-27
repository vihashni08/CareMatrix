"""Unit tests for the CareMatrix Care Coordination Agent."""

from __future__ import annotations

import time
import unittest

from care_coordination_agent import (
    ActionPolicy,
    CareActionType,
    CareCoordinationAgent,
    CareCoordinationEngine,
    CareCoordinationEvent,
    CoordinationPriority,
    CoordinationStatus,
    PatientCoordinationState,
)
from communication.event_queue import EventQueue
from communication.events import ClinicalReasoningEvent


class TestCareCoordinationAgent(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.agent = CareCoordinationAgent(event_queue=self.queue, verbose=False)
        self.engine = CareCoordinationEngine()

    def tearDown(self):
        self.agent.stop()
        self.queue.shutdown()

    def _make_clinical_event(
        self,
        case_id: int = 101,
        event_id: str = "test_event_001",
        risk_level: str = "HIGH RISK",
        priority: str = "URGENT",
        clinical_summary: str = "Acute tachycardia with hypotension noted.",
        findings: list[str] | None = None,
        recommended_actions: list[str] | None = None,
        data_reliability: str = "HIGH",
        verification_required: bool = False,
        evidence_consistency: str = "CONSISTENT",
    ) -> ClinicalReasoningEvent:
        return ClinicalReasoningEvent(
            case_id=case_id,
            event_id=event_id,
            timestamp=time.time(),
            risk_level=risk_level,
            priority=priority,
            clinical_summary=clinical_summary,
            findings=findings or ["Tachycardia (HR=125)", "Hypotension (MAP=55)"],
            recommended_actions=recommended_actions or ["Initiate IV fluid bolus", "Notify attending"],
            escalation_required=(priority == "URGENT"),
            data_reliability=data_reliability,
            confidence=0.90,
            evidence_consistency=evidence_consistency,
            supporting_evidence=["HR elevated", "MAP depressed"],
            conflicting_evidence=[],
            verification_required=verification_required,
        )

    # 1. ROUTINE priority produces CONTINUE_ROUTINE_MONITORING
    def test_routine_priority_produces_routine_monitoring(self):
        ev = self._make_clinical_event(risk_level="LOW RISK", priority="ROUTINE", clinical_summary="Vitals stable.")
        state = PatientCoordinationState(patient_id=101)
        action = self.engine.coordinate(ev, state)

        self.assertEqual(action.action_type, CareActionType.CONTINUE_ROUTINE_MONITORING.value)
        self.assertEqual(action.priority, CoordinationPriority.ROUTINE.value)
        self.assertFalse(action.clinician_review_required)
        self.assertIn("Standard Continuous Monitoring", action.escalation_pathway)

    # 2. ELEVATED priority produces SCHEDULE_CLINICIAN_REVIEW
    def test_elevated_priority_schedules_clinician_review(self):
        ev = self._make_clinical_event(risk_level="MODERATE", priority="ELEVATED")
        state = PatientCoordinationState(patient_id=101)
        action = self.engine.coordinate(ev, state)

        self.assertEqual(action.action_type, CareActionType.SCHEDULE_CLINICIAN_REVIEW.value)
        self.assertEqual(action.priority, CoordinationPriority.ELEVATED.value)
        self.assertTrue(action.clinician_review_required)

    # 3. URGENT priority produces TRIGGER_URGENT_CLINICAL_ALERT
    def test_urgent_priority_triggers_urgent_clinical_alert(self):
        ev = self._make_clinical_event(risk_level="HIGH RISK", priority="URGENT")
        state = PatientCoordinationState(patient_id=101)
        action = self.engine.coordinate(ev, state)

        self.assertEqual(action.action_type, CareActionType.TRIGGER_URGENT_CLINICAL_ALERT.value)
        self.assertEqual(action.priority, CoordinationPriority.URGENT.value)
        self.assertTrue(action.clinician_review_required)
        self.assertIn("Rapid Response", action.escalation_pathway)

    # 4. Compromised data quality forces REQUEST_DATA_VERIFICATION
    def test_compromised_data_quality_forces_verification(self):
        ev = self._make_clinical_event(
            risk_level="HIGH RISK",
            priority="URGENT",
            data_reliability="COMPROMISED",
            verification_required=True,
        )
        state = PatientCoordinationState(patient_id=101)
        action = self.engine.coordinate(ev, state)

        self.assertEqual(action.action_type, CareActionType.REQUEST_DATA_VERIFICATION.value)
        self.assertTrue(action.clinician_review_required)
        self.assertIn("Signal Quality Verification", action.escalation_pathway)
        self.assertTrue(any("manual" in ord.lower() for ord in action.suggested_orders))

    # 5. Conflicting evidence schedules clinician review
    def test_conflicting_evidence_schedules_clinician_review(self):
        ev = self._make_clinical_event(
            risk_level="HIGH RISK",
            priority="ELEVATED",
            evidence_consistency="CONFLICTING",
        )
        state = PatientCoordinationState(patient_id=101)
        action = self.engine.coordinate(ev, state)

        self.assertEqual(action.action_type, CareActionType.SCHEDULE_CLINICIAN_REVIEW.value)
        self.assertTrue(action.clinician_review_required)
        self.assertIn("conflict", action.reason.lower())

    # 6. Deduplication suppresses repeat alert spam for identical ongoing condition
    def test_duplicate_action_suppression(self):
        ev1 = self._make_clinical_event(event_id="ev_001", priority="URGENT")
        ev2 = self._make_clinical_event(event_id="ev_002", priority="URGENT")

        state = PatientCoordinationState(patient_id=101)
        action1 = self.engine.coordinate(ev1, state)
        self.assertEqual(action1.status, CoordinationStatus.NEW.value)
        self.assertFalse(action1.metadata.get("is_duplicate_suppressed", False))

        # Second identical urgent event while first is unresolved
        action2 = self.engine.coordinate(ev2, state)
        self.assertEqual(action2.status, CoordinationStatus.ACTIVE.value)
        self.assertTrue(action2.metadata.get("is_duplicate_suppressed", False))

    # 7. Worker thread consumes from clinical_decisions and publishes care_coordination_events
    def test_worker_thread_event_driven_pipeline(self):
        self.agent.start()
        clinical_ev = self._make_clinical_event(case_id=105, risk_level="MODERATE", priority="ELEVATED")

        self.queue.publish("clinical_decisions", clinical_ev)

        # Wait for consumer
        coord_event = None
        for _ in range(30):
            coord_event = self.queue.poll("care_coordination_events")
            if coord_event is not None:
                break
            time.sleep(0.05)

        self.assertIsNotNone(coord_event)
        self.assertEqual(coord_event.patient_id, 105)
        self.assertEqual(coord_event.priority, "ELEVATED")
        self.assertEqual(coord_event.action_type, CareActionType.SCHEDULE_CLINICIAN_REVIEW.value)

    # 8. Heartbeat emission and metrics
    def test_heartbeat_and_metrics(self):
        hb = self.agent.emit_heartbeat()
        self.assertEqual(hb.agent_name, "CareCoordinationAgent")
        self.assertEqual(hb.status, "healthy")
        self.assertIn("monitored_patients", hb.metrics)
        self.assertIn("active_actions", hb.metrics)

    # 9. State snapshot and worker restart
    def test_state_snapshot_and_restart(self):
        clinical_ev = self._make_clinical_event(case_id=202, priority="URGENT")
        self.agent.process_event(clinical_ev)

        snapshot = self.agent.get_state_snapshot()
        self.assertIn(202, snapshot["patient_states"])

        # Restart worker with snapshot
        success = self.agent.restart_worker(snapshot)
        self.assertTrue(success)
        self.assertTrue(self.agent.is_healthy)
        self.assertIn(202, self.agent.patient_states)

    # 10. Recovery resolves active actions
    def test_recovery_resolves_active_actions(self):
        clinical_ev = self._make_clinical_event(case_id=303, priority="URGENT")
        self.agent.process_event(clinical_ev)

        # Patient has active actions
        state = self.agent.patient_states[303]
        self.assertTrue(len(state.active_actions) > 0)

        # Trigger recovery
        resolved = self.agent.handle_recovery(303, reason="Vitals returned to baseline")
        self.assertTrue(len(resolved) > 0)
        self.assertEqual(len(state.active_actions), 0)
        self.assertFalse(state.unresolved_condition)


if __name__ == "__main__":
    unittest.main()
