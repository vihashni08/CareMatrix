"""Unit tests for Episodic Patient Memory and Clinician Feedback Loop."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from carematrix_runtime.alert_manager import AlertManager
from carematrix_runtime.patient_memory import EpisodeSummary, PatientMemory
from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import create_app
from clinical_reasoning_agent.clinical_reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning_agent.prompt_builder import build_evidence_package, build_reasoning_prompt
from clinical_reasoning_agent.reasoning import ClinicalReasoningEngine
from communication.event_log import EventLogWriter
from communication.event_queue import EventQueue
from communication.events import (
    CareActionType,
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringDecision,
    MonitoringEvent,
    PerformativeType,
    RiskDecisionEvent,
)
from evaluation.feedback_report import analyze_patient_memory, generate_feedback_report


class TestPatientMemory(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_events.db")
        self.memory = PatientMemory(db_path=self.db_path, max_episodes_per_patient=20)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_reasoning_event(
        self,
        case_id: int = 101,
        event_id: str = "reasoning_001",
        priority: str = "URGENT",
        timestamp: float | None = None,
    ) -> ClinicalReasoningEvent:
        return ClinicalReasoningEvent(
            case_id=case_id,
            event_id=event_id,
            timestamp=timestamp or time.time(),
            risk_level="HIGH RISK",
            priority=priority,
            clinical_summary=f"Acute vital deterioration for patient {case_id}",
            findings=["Hypotension MAP=55", "Tachycardia HR=128"],
            recommended_actions=["Vasopressor titration", "Notify ICU team"],
            escalation_required=True,
            data_reliability="HIGH",
            confidence=0.92,
            source="Clinical Reasoning Agent",
        )

    def _make_coordination_event(
        self,
        patient_id: int = 101,
        event_id: str = "reasoning_001",
        action_id: str = "action_001",
        priority: str = "URGENT",
        timestamp: float | None = None,
    ) -> CareCoordinationEvent:
        return CareCoordinationEvent(
            patient_id=patient_id,
            event_id=event_id,
            action_id=action_id,
            timestamp=timestamp or time.time(),
            action_type=CareActionType.TRIGGER_URGENT_CLINICAL_ALERT.value,
            priority=priority,
            status="ACTIVE",
            reason="Confirmed hemodynamic instability",
            clinician_review_required=True,
            suggested_orders=["Arterial blood gas", "Lactate panel"],
            escalation_pathway="ICU Rapid Response",
            clinical_summary=f"Urgent alert for patient {patient_id}",
            evidence_consistency="CONSISTENT",
            data_reliability="HIGH",
            confidence=0.95,
        )

    # ------------------------------------------------------------------------
    # 1. Accumulation & Bounded Capacity (Max 20 FIFO)
    # ------------------------------------------------------------------------
    def test_memory_accumulates_and_bounds_to_max_20(self):
        patient_id = 101
        # Add 25 episodes
        for i in range(25):
            ev = self._make_reasoning_event(
                case_id=patient_id,
                event_id=f"ev_{i:03d}",
                timestamp=1000.0 + i * 10.0,
            )
            self.memory.record_clinical_reasoning(ev)

        episodes = self.memory.get_episodes(patient_id)
        # Must be capped at exactly max_episodes_per_patient (20)
        self.assertEqual(len(episodes), 20)

        # Must be FIFO: oldest 5 (0 to 4) dropped; first should be index 5, last index 24
        self.assertEqual(episodes[0].event_id, "ev_005")
        self.assertEqual(episodes[-1].event_id, "ev_024")

    def test_multi_patient_isolation(self):
        # Patient 101 gets 3 episodes, Patient 102 gets 2 episodes
        for i in range(3):
            self.memory.record_clinical_reasoning(self._make_reasoning_event(case_id=101, event_id=f"p101_{i}"))
        for j in range(2):
            self.memory.record_clinical_reasoning(self._make_reasoning_event(case_id=102, event_id=f"p102_{j}"))

        self.assertEqual(len(self.memory.get_episodes(101)), 3)
        self.assertEqual(len(self.memory.get_episodes(102)), 2)
        self.assertEqual(self.memory.get_all_patient_ids(), [101, 102])

    # ------------------------------------------------------------------------
    # 2. Clinician Feedback Correlation & Latency Metrics
    # ------------------------------------------------------------------------
    def test_clinician_feedback_acknowledgment_and_resolution(self):
        t0 = 1000.0
        reasoning_ev = self._make_reasoning_event(case_id=101, event_id="rev_101", priority="URGENT", timestamp=t0)
        coord_ev = self._make_coordination_event(
            patient_id=101, event_id="rev_101", action_id="act_101", priority="URGENT", timestamp=t0 + 2.0
        )

        self.memory.record_clinical_reasoning(reasoning_ev)
        self.memory.record_care_coordination(coord_ev)

        # Confirm initial state
        eps = self.memory.get_episodes(101)
        self.assertEqual(len(eps), 1)
        ep = eps[0]
        self.assertEqual(ep.status, "PENDING")
        self.assertFalse(ep.acknowledged)
        self.assertFalse(ep.resolved)

        # 1. Clinician Acknowledges via action_id after 45 seconds
        t_ack = t0 + 45.0
        self.memory.record_clinician_feedback(
            patient_id=101,
            event_id="rev_101",
            action="ACKNOWLEDGE",
            clinician_id="Dr. House",
            care_action_id="act_101",
            priority="URGENT",
            timestamp=t_ack,
        )

        self.assertTrue(ep.acknowledged)
        self.assertEqual(ep.status, "ACKNOWLEDGED")
        self.assertEqual(ep.acknowledged_by, "Dr. House")
        self.assertAlmostEqual(ep.time_to_acknowledge_seconds, 45.0, places=1)

        # 2. Clinician Resolves after 300 seconds
        t_res = t0 + 300.0
        self.memory.record_clinician_feedback(
            patient_id=101,
            event_id="rev_101",
            action="RESOLVE",
            clinician_id="Dr. House",
            reason="Patient stabilized with IV Norepinephrine",
            care_action_id="act_101",
            priority="URGENT",
            timestamp=t_res,
        )

        self.assertTrue(ep.resolved)
        self.assertEqual(ep.status, "RESOLVED")
        self.assertEqual(ep.resolution_reason, "Patient stabilized with IV Norepinephrine")
        self.assertAlmostEqual(ep.time_to_resolve_seconds, 300.0, places=1)
        self.assertFalse(ep.is_false_positive)
        self.assertFalse(ep.is_override)

    def test_clinician_false_positive_and_override_flagging(self):
        t0 = 2000.0
        ev = self._make_reasoning_event(case_id=101, event_id="fp_ev", priority="URGENT", timestamp=t0)
        self.memory.record_clinical_reasoning(ev)

        # Resolve with false-positive flag
        self.memory.record_clinician_feedback(
            patient_id=101,
            event_id="fp_ev",
            action="RESOLVE",
            clinician_id="RN Smith",
            reason="Artifact: arterial catheter flushed right before reading. False positive alert.",
            is_false_positive=True,
            priority="URGENT",
            timestamp=t0 + 60.0,
        )

        ep = self.memory.get_episodes(101)[0]
        self.assertTrue(ep.resolved)
        self.assertTrue(ep.is_false_positive)
        self.assertEqual(ep.status, "FALSE_POSITIVE")

    # ------------------------------------------------------------------------
    # 3. SQLite Persistence & Replay Reconstruction
    # ------------------------------------------------------------------------
    def test_sqlite_persistence_and_replay(self):
        t0 = 5000.0
        # Create initial memory and record events with feedback
        ev = self._make_reasoning_event(case_id=105, event_id="persist_ev_105", priority="URGENT", timestamp=t0)
        self.memory.record_clinical_reasoning(ev)
        self.memory.record_clinician_feedback(
            patient_id=105,
            event_id="persist_ev_105",
            action="ACKNOWLEDGE",
            clinician_id="Dr. Watson",
            priority="URGENT",
            timestamp=t0 + 35.0,
        )
        self.memory.record_clinician_feedback(
            patient_id=105,
            event_id="persist_ev_105",
            action="RESOLVE",
            clinician_id="Dr. Watson",
            reason="Resolved by bedside intervention",
            priority="URGENT",
            timestamp=t0 + 120.0,
        )

        # Instantiate a second PatientMemory pointing to the same SQLite DB
        replayed_memory = PatientMemory(db_path=self.db_path)
        replayed_eps = replayed_memory.get_episodes(105)

        self.assertEqual(len(replayed_eps), 1)
        r_ep = replayed_eps[0]
        self.assertEqual(r_ep.event_id, "persist_ev_105")
        self.assertEqual(r_ep.patient_id, 105)
        self.assertTrue(r_ep.acknowledged)
        self.assertEqual(r_ep.acknowledged_by, "Dr. Watson")
        self.assertAlmostEqual(r_ep.time_to_acknowledge_seconds, 35.0, places=1)
        self.assertTrue(r_ep.resolved)
        self.assertEqual(r_ep.status, "RESOLVED")

    # ------------------------------------------------------------------------
    # 4. Prompt Context Formatting & Safety Invariant
    # ------------------------------------------------------------------------
    def test_prompt_context_formatting(self):
        t0 = time.time()
        self.memory.record_clinical_reasoning(
            self._make_reasoning_event(case_id=101, event_id="prompt_ev", priority="URGENT", timestamp=t0)
        )
        self.memory.record_clinician_feedback(
            patient_id=101,
            event_id="prompt_ev",
            action="ACKNOWLEDGE",
            clinician_id="Dr. Grey",
            timestamp=t0 + 50.0,
        )
        self.memory.record_clinician_feedback(
            patient_id=101,
            event_id="prompt_ev",
            action="RESOLVE",
            reason="Artifact flagged as false positive by attending",
            is_false_positive=True,
            timestamp=t0 + 100.0,
        )

        context_str = self.memory.format_memory_for_prompt(101)
        self.assertIn("Past response timing", context_str)
        self.assertIn("Clinical discernment", context_str)
        self.assertIn("SAFETY MANDATE", context_str)
        self.assertIn("NEVER be suppressed or downgraded", context_str)

    def test_deterministic_safety_invariant_holds_with_negative_history(self):
        """CRITICAL: Past false positives must NEVER suppress or downgrade a fresh physiological deterioration."""
        # 1. Setup patient with a history of false positive alerts
        t0 = time.time() - 3600
        for i in range(3):
            ev_id = f"fp_hist_{i}"
            self.memory.record_clinical_reasoning(
                self._make_reasoning_event(case_id=101, event_id=ev_id, priority="URGENT", timestamp=t0 + i * 100)
            )
            self.memory.record_clinician_feedback(
                patient_id=101,
                event_id=ev_id,
                action="RESOLVE",
                reason="Sensor motion artifact false alarm",
                is_false_positive=True,
                timestamp=t0 + i * 100 + 30,
            )

        # 2. Build evidence package for fresh acute shock
        current_vitals = {"HR": 132.0, "MAP": 52.0, "RR": 28.0, "SPO2": 88.0, "TEMP": 38.6}
        trends = {"HR": "increasing", "MAP": "decreasing", "RR": "increasing", "SPO2": "decreasing"}
        signal_quality = {"HR": "reliable", "MAP": "reliable", "RR": "reliable", "SPO2": "reliable"}
        analysis_ev = DataAnalysisEvent(
            case_id=101,
            event_id="da_test_001",
            timestamp=time.time(),
            risk_level="HIGH RISK",
            trend_metrics={"HR": {"trend": "increasing"}, "MAP": {"trend": "decreasing"}},
            pattern_identified=["acute_hypotension", "tachycardia"],
            data_quality_flag=False,
            evidence_consistency="consistent",
            verification_required=False,
        )

        pkg = build_evidence_package(
            event=analysis_ev,
            patient_memory=self.memory,
        )

        # Verify episodic memory was included in evidence package
        self.assertIn("episodic_memory", pkg)
        self.assertEqual(len(pkg["episodic_memory"]), 3)
        self.assertIn("SAFETY MANDATE", pkg["episodic_memory_summary"])

        # 3. Deterministic Safety Arbiter evaluates
        engine = ClinicalReasoningEngine()
        result = engine.evaluate(analysis_ev)

        # Invariant check: Deterministic baseline MUST remain URGENT escalation
        self.assertEqual(result.priority, "URGENT")
        self.assertTrue(result.escalation_required)

    # ------------------------------------------------------------------------
    # 5. Server REST API & Feedback Recording Integration
    # ------------------------------------------------------------------------
    def test_server_acknowledge_and_resolve_updates_patient_memory(self):
        runtime = CareMatrixRuntime(stream_interval_seconds=0.1, verbose=False, event_log_path=self.db_path)
        app = create_app(runtime)
        client = app.test_client()

        # Generate a simulated severe monitoring event and alert
        mon_ev = MonitoringEvent(
            patient_id=101,
            event_id="mon_test_001",
            timestamp=time.time(),
            event_type="alert_started",
            severity="severe",
            affected_vitals=["MAP", "HR"],
            current_values={"MAP": 50.0, "HR": 130.0},
            baseline_values={"MAP": 80.0, "HR": 70.0},
            deviation_values={"MAP": -30.0, "HR": 60.0},
            trends={"MAP": "decreasing", "HR": "increasing"},
            signal_quality={"MAP": "reliable", "HR": "reliable"},
            persistence_duration=4,
            recommended_action="Vasopressor therapy",
        )
        alert = runtime.alert_manager.handle_monitoring_event(mon_ev)
        alert_id = alert.alert_id

        # 1. Test POST /api/alerts/<id>/acknowledge
        ack_res = client.post(
            f"/api/alerts/{alert_id}/acknowledge",
            json={"clinician_id": "Dr. Sarah"},
        )
        self.assertEqual(ack_res.status_code, 200)
        self.assertTrue(ack_res.json["success"])

        # Check PatientMemory
        eps = runtime.patient_memory.get_episodes(101)
        self.assertTrue(len(eps) >= 1)
        target_ep = next((e for e in eps if e.event_id == "mon_test_001" or e.action_id == alert.care_action_id), None)
        self.assertIsNotNone(target_ep)
        self.assertTrue(target_ep.acknowledged)
        self.assertEqual(target_ep.acknowledged_by, "Dr. Sarah")

        # 2. Test POST /api/alerts/<id>/resolve
        res_res = client.post(
            f"/api/alerts/{alert_id}/resolve",
            json={
                "clinician_id": "Dr. Sarah",
                "reason": "Successfully resuscitated",
                "is_false_positive": False,
            },
        )
        self.assertEqual(res_res.status_code, 200)
        self.assertTrue(res_res.json["success"])
        self.assertTrue(target_ep.resolved)

        # 3. Test GET /api/patients/101/memory
        mem_res = client.get("/api/patients/101/memory")
        self.assertEqual(mem_res.status_code, 200)
        self.assertEqual(mem_res.json["patient_id"], 101)
        self.assertGreaterEqual(mem_res.json["count"], 1)

    # ------------------------------------------------------------------------
    # 6. Offline Feedback Report Evaluation
    # ------------------------------------------------------------------------
    def test_feedback_report_generation(self):
        t0 = time.time()
        # Add 1 concordant urgent episode (ack in 30s)
        ev1 = self._make_reasoning_event(case_id=101, event_id="rep_ev_1", priority="URGENT", timestamp=t0 - 300)
        self.memory.record_clinical_reasoning(ev1)
        self.memory.record_clinician_feedback(
            patient_id=101, event_id="rep_ev_1", action="ACKNOWLEDGE", priority="URGENT", timestamp=t0 - 270
        )
        self.memory.record_clinician_feedback(
            patient_id=101, event_id="rep_ev_1", action="RESOLVE", priority="URGENT", timestamp=t0 - 200
        )

        # Add 1 false-positive urgent episode (ack in 50s)
        ev2 = self._make_reasoning_event(case_id=101, event_id="rep_ev_2", priority="URGENT", timestamp=t0 - 150)
        self.memory.record_clinical_reasoning(ev2)
        self.memory.record_clinician_feedback(
            patient_id=101, event_id="rep_ev_2", action="ACKNOWLEDGE", priority="URGENT", timestamp=t0 - 100
        )
        self.memory.record_clinician_feedback(
            patient_id=101,
            event_id="rep_ev_2",
            action="RESOLVE",
            is_false_positive=True,
            priority="URGENT",
            timestamp=t0 - 80,
        )

        # Add 1 elevated episode (ack in 120s)
        ev3 = self._make_reasoning_event(case_id=102, event_id="rep_ev_3", priority="ELEVATED", timestamp=t0 - 200)
        self.memory.record_clinical_reasoning(ev3)
        self.memory.record_clinician_feedback(
            patient_id=102, event_id="rep_ev_3", action="ACKNOWLEDGE", priority="ELEVATED", timestamp=t0 - 80
        )

        result, report_md = generate_feedback_report(patient_memory=self.memory)

        self.assertEqual(result.total_patients, 2)
        self.assertEqual(result.total_episodes, 3)
        self.assertEqual(result.total_with_feedback, 3)
        self.assertEqual(result.acknowledged_count, 3)
        self.assertEqual(result.false_positive_count, 1)
        self.assertEqual(result.agreement_count, 2)
        # Agreement rate: 2/3 = 66.7%
        self.assertAlmostEqual(result.agreement_rate_pct, 66.67, places=1)
        # Urgent median ack latency: median of 30s and 50s = 40s
        urgent_stats = result.latency_by_priority["URGENT"]
        self.assertEqual(urgent_stats.count, 2)
        self.assertAlmostEqual(urgent_stats.median_seconds, 40.0, places=1)

        # Report Markdown checks
        self.assertIn("CareMatrix Clinician Feedback & Episodic Memory Evaluation Report", report_md)
        self.assertIn("Clinician Agreement (Concordant)", report_md)
        self.assertIn("URGENT", report_md)


if __name__ == "__main__":
    unittest.main()
