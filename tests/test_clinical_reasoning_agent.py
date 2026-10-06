"""Unit tests for the CareMatrix Clinical Reasoning Agent."""

from __future__ import annotations

import time
import unittest
from unittest.mock import Mock

from communication.event_queue import EventQueue
from communication.events import ClinicalReasoningEvent, DataAnalysisEvent
from clinical_reasoning_agent.schemas import LLMReasoningResult
from clinical_reasoning_agent import (
    ClinicalReasoningAgent,
    ClinicalReasoningEngine,
    ClinicalReasoningPriority,
    ClinicalReasoningState,
)


class ClinicalReasoningAgentTests(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.agent = ClinicalReasoningAgent(event_queue=self.queue, enable_llm=False)

    def tearDown(self):
        self.agent.stop()
        self.queue.shutdown()

    def make_analysis_event(
        self,
        event_id: str = "case_4_event_014",
        risk_level: str = "HIGH RISK",
        data_quality_flag: bool = False,
        metrics: dict | None = None,
        patterns: list[str] | None = None,
        analysis_status: str = "complete",
    ) -> DataAnalysisEvent:
        if metrics is None:
            metrics = {
                "HR": {"trend": "increasing", "latest_value": 98.0, "change_over_window": 14.0},
                "MAP": {"trend": "decreasing", "latest_value": 60.0, "change_over_window": -18.0},
            }
        if patterns is None:
            patterns = [
                "HR shows a increasing trend in the assessment window.",
                "MAP shows a decreasing trend in the assessment window.",
                "Simultaneous directional changes are present across multiple affected vitals.",
            ]
        return DataAnalysisEvent(
            case_id=4,
            event_id=event_id,
            timestamp=300.0,
            risk_level=risk_level,
            trend_metrics=metrics,
            pattern_identified=patterns,
            important_changes=["HR: increasing", "MAP: decreasing"],
            data_quality_flag=data_quality_flag,
            analysis_status=analysis_status,
        )

    # 1. HIGH-Risk Reasoning
    def test_high_risk_produces_urgent_clinical_decision(self):
        """Verify HIGH RISK event generates URGENT priority, escalation required, and bedside actions."""
        event = self.make_analysis_event(risk_level="HIGH RISK")
        result = self.agent.process_event(event)

        self.assertIsInstance(result, ClinicalReasoningEvent)
        self.assertEqual(result.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertEqual(result.risk_level, "HIGH RISK")
        self.assertTrue(result.escalation_required)
        self.assertEqual(result.data_reliability, "HIGH")
        self.assertGreater(len(result.recommended_actions), 0)
        self.assertTrue(any("bedside" in a.lower() for a in result.recommended_actions))
        self.assertEqual(len(self.queue.get_history("clinical_decisions")), 1)

    def test_high_risk_invokes_available_llm_without_global_enable_flag(self):
        """High-risk synthesis uses an available Gemini reasoner and still builds a report."""
        reasoner = Mock()
        reasoner.is_available = True
        reasoner.reason.return_value = LLMReasoningResult(
            clinical_summary="High-risk deterioration requires immediate bedside review.",
            supporting_evidence=["MAP remains critically low."],
            conflicting_evidence=[],
            key_findings=["Hypotension with tachycardia."],
            risk_interpretation="High-risk vital pattern.",
            priority="URGENT",
            recommended_actions=["Request immediate clinician assessment."],
            confidence=0.95,
        )
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            llm_reasoner=reasoner,
            enable_llm=False,
            enable_rag=False,
        )

        result = agent.process_event(self.make_analysis_event(risk_level="HIGH RISK"))

        reasoner.reason.assert_called_once()
        self.assertEqual(result.clinical_summary, "High-risk deterioration requires immediate bedside review.")
        report = result.to_clinical_report()
        self.assertIn("High-risk deterioration", report["executive_summary"])
        self.assertEqual(report["clinical_status"]["priority"], "URGENT")

    # 2. LOW-Risk Reasoning (Stable Trends)
    def test_low_risk_stable_produces_routine_decision(self):
        """Verify LOW RISK with stable vitals generates ROUTINE priority and no urgent escalation."""
        event = self.make_analysis_event(
            risk_level="LOW RISK",
            metrics={"HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 1.0}},
            patterns=["LOW RISK classification retained; available affected-vital data is stable."],
        )
        result = self.agent.process_event(event)

        self.assertEqual(result.priority, ClinicalReasoningPriority.ROUTINE.value)
        self.assertEqual(result.risk_level, "LOW RISK")
        self.assertFalse(result.escalation_required)
        self.assertTrue(any("standard" in a.lower() or "surveillance" in a.lower() for a in result.recommended_actions))

    # 3. Worsening Trends with Safe Risk Level
    def test_worsening_trends_elevate_surveillance(self):
        """Verify LOW RISK with directional worsening trends generates ELEVATED surveillance."""
        event = self.make_analysis_event(
            risk_level="LOW RISK",
            metrics={
                "HR": {"trend": "increasing", "latest_value": 88.0, "change_over_window": 12.0},
                "MAP": {"trend": "decreasing", "latest_value": 68.0, "change_over_window": -10.0},
            },
            patterns=["LOW RISK classification retained; trends are reported as supporting evidence only."],
        )
        result = self.agent.process_event(event)

        self.assertEqual(result.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertFalse(result.escalation_required)
        self.assertTrue(any("observe" in a.lower() or "monitor" in a.lower() for a in result.recommended_actions))

    # 4. Poor / Degraded Data Quality
    def test_compromised_data_quality_flags_sensor_check(self):
        """Verify data quality flag lowers confidence and recommends sensor verification."""
        event = self.make_analysis_event(risk_level="LOW RISK", data_quality_flag=True)
        result = self.agent.process_event(event)

        self.assertEqual(result.data_reliability, "COMPROMISED")
        self.assertEqual(result.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertLess(result.confidence, 0.90)
        self.assertTrue(any("sensor" in a.lower() or "probe" in a.lower() for a in result.recommended_actions))

    # 5. Malformed Events Handling
    def test_malformed_event_handled_safely(self):
        """Verify unexpected object or corrupt payload does not crash the agent worker."""
        result = self.agent.process_event(object())

        self.assertIsInstance(result, ClinicalReasoningEvent)
        self.assertEqual(result.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertEqual(result.data_reliability, "COMPROMISED")
        self.assertTrue(any("pipeline" in a.lower() or "vital" in a.lower() for a in result.recommended_actions))

    # 6. Event-Driven Asynchronous Pipeline
    def test_worker_thread_event_driven_pipeline(self):
        """Verify worker thread consumes from clinical_reasoning_events and emits to clinical_decisions."""
        self.agent.start()
        event = self.make_analysis_event(event_id="test_cr_async_evt")

        self.queue.publish("clinical_reasoning_events", event)

        decision = None
        for _ in range(20):
            decisions = self.queue.get_history("clinical_decisions")
            if decisions:
                decision = decisions[0]
                break
            time.sleep(0.05)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.event_id, "test_cr_async_evt")

    # 7. Patient State Context Across Successive Events
    def test_patient_state_tracks_history(self):
        """Verify patient state maintains successive assessments, alert count, and history."""
        event1 = self.make_analysis_event(event_id="evt_1", risk_level="LOW RISK")
        self.agent.process_event(event1)

        event2 = self.make_analysis_event(event_id="evt_2", risk_level="HIGH RISK")
        result2 = self.agent.process_event(event2)

        patient_state = self.agent.patient_states[4]
        self.assertEqual(patient_state.total_evaluations, 2)
        self.assertEqual(patient_state.last_priority, ClinicalReasoningPriority.URGENT.value)
        self.assertIn("evt_1", patient_state.active_alerts)
        self.assertIn("evt_2", patient_state.active_alerts)
        self.assertTrue(any("historical context" in f.lower() for f in result2.findings))

    # 8. Heartbeat Emission
    def test_heartbeat_emission(self):
        """Verify agent emits heartbeat to heartbeats queue."""
        hb = self.agent.emit_heartbeat()
        self.assertEqual(hb.agent_name, "ClinicalReasoningAgent")
        self.assertEqual(hb.status, "healthy")
        self.assertEqual(len(self.queue.get_history("heartbeats")), 1)


if __name__ == "__main__":
    unittest.main()
