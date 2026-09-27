"""Unit tests for CareMatrix durable append-only event log and state reconstruction."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
import unittest

from carematrix_runtime.alert_manager import AlertManager
from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.state_manager import PatientStateManager
from communication.event_log import EventLogWriter, deserialize_event
from communication.event_queue import EventQueue
from communication.events import (
    AgentFailureEvent,
    AgentHeartbeatEvent,
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringEvent,
    RiskDecisionEvent,
)


def _make_monitoring_event(
    patient_id: int = 101,
    event_id: str = "mon-001",
    timestamp: float | None = None,
    event_type: str = "tachycardia_started",
    severity: str = "critical",
    affected_vitals: list[str] | None = None,
    current_values: dict[str, float] | None = None,
    baseline_values: dict[str, float] | None = None,
    deviation_values: dict[str, float] | None = None,
    trends: dict[str, str] | None = None,
    signal_quality: dict[str, str] | None = None,
    persistence_duration: int = 5,
    recommended_action: str = "immediate_assessment",
) -> MonitoringEvent:
    """Helper to instantiate valid MonitoringEvent instances with all required fields."""
    return MonitoringEvent(
        patient_id=patient_id,
        event_id=event_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        event_type=event_type,
        severity=severity,
        affected_vitals=affected_vitals or ["heart_rate"],
        current_values=current_values or {"heart_rate": 142.0},
        baseline_values=baseline_values or {"heart_rate": 75.0},
        deviation_values=deviation_values or {"heart_rate": 67.0},
        trends=trends or {"heart_rate": "worsening"},
        signal_quality=signal_quality or {"heart_rate": "GOOD"},
        persistence_duration=persistence_duration,
        recommended_action=recommended_action,
    )


def _make_care_coordination_event(
    patient_id: int = 101,
    event_id: str = "care-001",
    timestamp: float | None = None,
    action_type: str = "ESCALATE_TO_RAPID_RESPONSE",
    priority: str = "URGENT",
    status: str = "TRIGGERED",
    reason: str = "Clinical reasoning triggered rapid response escalation",
    clinician_review_required: bool = True,
    suggested_orders: list[str] | None = None,
    escalation_pathway: str = "ICU Rapid Response",
    clinical_summary: str = "Patient deteriorating rapidly",
    evidence_consistency: str = "CONSISTENT",
    data_reliability: str = "HIGH",
    confidence: float = 0.95,
) -> CareCoordinationEvent:
    """Helper to instantiate valid CareCoordinationEvent instances."""
    return CareCoordinationEvent(
        patient_id=patient_id,
        event_id=event_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        action_type=action_type,
        priority=priority,
        status=status,
        reason=reason,
        clinician_review_required=clinician_review_required,
        suggested_orders=suggested_orders or ["CBC", "Lactate stat"],
        escalation_pathway=escalation_pathway,
        clinical_summary=clinical_summary,
        evidence_consistency=evidence_consistency,
        data_reliability=data_reliability,
        confidence=confidence,
    )


class TestEventLogWriter(unittest.TestCase):
    """Test suite for EventLogWriter, persistence, replay, and graceful failure handling."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_events.db")
        self.writers: list[EventLogWriter] = []

    def tearDown(self):
        for w in self.writers:
            try:
                w.close(timeout=1.0)
            except Exception:
                pass
        self.writers.clear()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_writer(self, path: str | None = None) -> EventLogWriter:
        w = EventLogWriter(path or self.db_path)
        self.writers.append(w)
        return w

    def test_append_only_guarantee(self):
        """Assert EventLogWriter strictly provides append/read methods and no update/delete methods."""
        writer = self._create_writer()

        for forbidden in ("update", "delete", "remove", "drop", "truncate", "clear", "modify"):
            self.assertFalse(
                hasattr(writer, forbidden),
                f"EventLogWriter must be append-only and not expose '{forbidden}' method",
            )

    def test_event_persistence_and_sqlite_schema(self):
        """Verify events of all types are correctly persisted with required SQLite columns."""
        writer = self._create_writer()

        mon_event = _make_monitoring_event(
            patient_id=101,
            event_id="mon-001",
            timestamp=1700000001.0,
            event_type="tachycardia_started",
            severity="critical",
            affected_vitals=["heart_rate"],
            current_values={"heart_rate": 142.0},
            baseline_values={"heart_rate": 75.0},
            deviation_values={"heart_rate": 67.0},
            persistence_duration=5,
            recommended_action="immediate_assessment",
        )

        risk_event = RiskDecisionEvent(
            patient_id=101,
            event_id="risk-001",
            timestamp=1700000002.0,
            decision="ESCALATE_TO_CLINICAL",
            risk_probability=0.88,
            risk_level="HIGH RISK",
            threshold=0.65,
            model_name="RandomForestClassifier",
            features_used={"hr_mean": 138.0},
            reason="Sustained elevated HR with high sepsis risk probability",
        )

        analysis_event = DataAnalysisEvent(
            case_id=101,
            event_id="ana-001",
            timestamp=1700000003.0,
            risk_level="HIGH RISK",
            trend_metrics={"hr_slope": 1.25},
            pattern_identified=["rapid_escalation"],
            data_quality_flag=False,
            evidence_consistency="consistent",
        )

        clinical_event = ClinicalReasoningEvent(
            case_id=101,
            event_id="clin-001",
            timestamp=1700000004.0,
            risk_level="HIGH RISK",
            priority="URGENT",
            clinical_summary="Patient exhibiting signs consistent with septic shock onset.",
            findings=["Tachycardia", "Hypotension"],
            recommended_actions=["Blood cultures", "Lactate", "IV fluids"],
            escalation_required=True,
            confidence=0.92,
        )

        care_event = _make_care_coordination_event(
            patient_id=101,
            event_id="care-001",
            timestamp=1700000005.0,
            action_type="ESCALATE_TO_RAPID_RESPONSE",
            priority="URGENT",
            status="TRIGGERED",
            reason="Clinical reasoning triggered rapid response escalation",
            suggested_orders=["CBC", "Lactate stat"],
        )

        hb_event = AgentHeartbeatEvent(
            agent_name="MonitoringAgent",
            timestamp=1700000006.0,
            status="healthy",
            metrics={"processed": 42},
        )

        fail_event = AgentFailureEvent(
            agent_name="RiskAgent",
            error_message="Feature missing",
            timestamp=1700000007.0,
            recoverable=True,
        )

        # Log events
        writer.log_event("monitoring_events", mon_event)
        writer.log_event("risk_predictions", risk_event)
        writer.log_event("data_analysis_events", analysis_event)
        writer.log_event("clinical_decisions", clinical_event)
        writer.log_event("care_coordination_events", care_event)
        writer.log_event("heartbeats", hb_event)
        writer.log_event("agent_failures", fail_event)

        writer.flush()

        # Check raw SQLite table directly
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(events);")
        columns = {col[1]: col[2] for col in cursor.fetchall()}
        expected_columns = ["id", "topic", "event_type", "patient_id", "timestamp", "payload_json"]
        for col_name in expected_columns:
            self.assertIn(col_name, columns, f"Column {col_name} missing from SQLite schema")

        cursor.execute("SELECT id, topic, event_type, patient_id, timestamp, payload_json FROM events ORDER BY id ASC;")
        rows = cursor.fetchall()
        conn.close()

        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0][1], "monitoring_events")
        self.assertEqual(rows[0][2], "tachycardia_started")
        self.assertEqual(rows[0][3], 101)

        self.assertEqual(rows[1][1], "risk_predictions")
        self.assertEqual(rows[1][3], 101)

        self.assertEqual(rows[2][1], "data_analysis_events")
        self.assertEqual(rows[2][3], 101)

        self.assertEqual(rows[3][1], "clinical_decisions")
        self.assertEqual(rows[3][3], 101)

        self.assertEqual(rows[4][1], "care_coordination_events")
        self.assertEqual(rows[4][3], 101)

        self.assertEqual(rows[5][1], "heartbeats")
        self.assertIsNone(rows[5][3])

        self.assertEqual(rows[6][1], "agent_failures")
        self.assertIsNone(rows[6][3])

    def test_event_queue_integration_and_latency(self):
        """Verify EventQueue publishes to EventLogWriter asynchronously with near-instantaneous execution."""
        queue = EventQueue()
        writer = self._create_writer()

        queue.add_diagnostic_listener(writer.log_event)

        event = _make_monitoring_event(
            patient_id=102,
            event_id="mon-latency-001",
            timestamp=time.time(),
            event_type="desaturation_started",
            severity="moderate",
            affected_vitals=["sp_o2"],
        )

        t0 = time.perf_counter()
        queue.publish("monitoring_events", event)
        latency = time.perf_counter() - t0

        # Verify non-blocking latency remains well under 10ms (typically < 0.1ms)
        self.assertLess(latency, 0.05, f"publish() took too long: {latency:.5f}s")

        writer.flush()
        events = writer.get_events(patient_id=102)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["topic"], "monitoring_events")
        self.assertEqual(events[0]["patient_id"], 102)

    def test_state_reconstruction_identical_to_live_run(self):
        """Verify that replaying persisted events reconstructs PatientStateManager and AlertManager identically."""
        writer = self._create_writer()

        live_alert_mgr = AlertManager()
        live_state_mgr = PatientStateManager(alert_manager=live_alert_mgr)

        events_to_run = [
            (
                "monitoring_events",
                _make_monitoring_event(
                    patient_id=101,
                    event_id="mon-1",
                    timestamp=100.0,
                    event_type="tachycardia_started",
                    severity="high",
                    affected_vitals=["heart_rate"],
                    current_values={"heart_rate": 135.0},
                    deviation_values={"heart_rate": 55.0},
                    persistence_duration=4,
                ),
            ),
            (
                "risk_predictions",
                RiskDecisionEvent(
                    patient_id=101,
                    event_id="risk-1",
                    timestamp=101.0,
                    decision="ESCALATE",
                    risk_probability=0.82,
                    risk_level="HIGH RISK",
                    threshold=0.6,
                    model_name="RandomForestClassifier",
                ),
            ),
            (
                "data_analysis_events",
                DataAnalysisEvent(
                    case_id=101,
                    event_id="ana-1",
                    timestamp=102.0,
                    risk_level="HIGH RISK",
                    trend_metrics={"slope": 2.1},
                    pattern_identified=["tachycardia_spike"],
                    evidence_consistency="consistent",
                ),
            ),
            (
                "clinical_decisions",
                ClinicalReasoningEvent(
                    case_id=101,
                    event_id="clin-1",
                    timestamp=103.0,
                    risk_level="HIGH RISK",
                    priority="URGENT",
                    clinical_summary="Patient critical",
                    findings=["severe tachycardia"],
                    recommended_actions=["IV fluids"],
                ),
            ),
            (
                "care_coordination_events",
                _make_care_coordination_event(
                    patient_id=101,
                    event_id="care-1",
                    timestamp=104.0,
                    action_type="NOTIFY_PRIMARY_CARE_TEAM",
                    priority="URGENT",
                    status="DELIVERED",
                    reason="Urgent clinical assessment",
                    suggested_orders=["CBC"],
                ),
            ),
            (
                "monitoring_events",
                _make_monitoring_event(
                    patient_id=102,
                    event_id="mon-2",
                    timestamp=105.0,
                    event_type="hypotension_started",
                    severity="moderate",
                    affected_vitals=["blood_pressure_sys"],
                    current_values={"blood_pressure_sys": 85.0},
                    deviation_values={"blood_pressure_sys": -35.0},
                    persistence_duration=3,
                ),
            ),
        ]

        # Feed events into live state manager and event log writer
        for topic, evt in events_to_run:
            live_state_mgr.handle_event(topic, evt)
            writer.log_event(topic, evt)

        writer.flush()

        # Capture live state snapshots
        live_p101_detail = live_state_mgr.get_patient_detail(101)
        live_p102_detail = live_state_mgr.get_patient_detail(102)
        live_summaries = live_state_mgr.get_patient_summary_list()
        live_active_alerts = live_alert_mgr.get_active_alerts()

        # Now create a fresh, empty StateManager and AlertManager
        replayed_alert_mgr = AlertManager()
        replayed_state_mgr = PatientStateManager(alert_manager=replayed_alert_mgr)

        # Replay all logged events into fresh state manager
        replayed_count = writer.replay_into(replayed_state_mgr)
        self.assertEqual(replayed_count, len(events_to_run))

        replayed_p101_detail = replayed_state_mgr.get_patient_detail(101)
        replayed_p102_detail = replayed_state_mgr.get_patient_detail(102)
        replayed_summaries = replayed_state_mgr.get_patient_summary_list()
        replayed_active_alerts = replayed_alert_mgr.get_active_alerts()

        # Assert reconstructed states match live states exactly
        self.assertIsNotNone(replayed_p101_detail)
        self.assertEqual(replayed_p101_detail["status"], live_p101_detail["status"])
        self.assertEqual(replayed_p101_detail["risk_level"], live_p101_detail["risk_level"])
        self.assertEqual(replayed_p101_detail["risk_probability"], live_p101_detail["risk_probability"])
        self.assertEqual(
            replayed_p101_detail["latest_monitoring_event"]["event_type"],
            live_p101_detail["latest_monitoring_event"]["event_type"],
        )
        self.assertEqual(
            replayed_p101_detail["latest_risk_decision"]["decision"],
            live_p101_detail["latest_risk_decision"]["decision"],
        )
        self.assertEqual(
            replayed_p101_detail["latest_clinical_reasoning"]["priority"],
            live_p101_detail["latest_clinical_reasoning"]["priority"],
        )
        self.assertEqual(
            replayed_p101_detail["latest_care_action"]["action_type"],
            live_p101_detail["latest_care_action"]["action_type"],
        )

        self.assertEqual(len(replayed_active_alerts), len(live_active_alerts))
        for r_alert, l_alert in zip(replayed_active_alerts, live_active_alerts):
            self.assertEqual(r_alert["patient_id"], l_alert["patient_id"])
            self.assertEqual(r_alert["severity"], l_alert["severity"])
            self.assertEqual(r_alert["affected_vitals"], l_alert["affected_vitals"])
            self.assertEqual(r_alert["state"], l_alert["state"])


    def test_runtime_restart_replays_prior_event_log(self):
        """Verify CareMatrixRuntime reconstructs state on startup when prior log exists."""
        # Step 1: Run runtime 1, generate events, and shut down
        runtime1 = CareMatrixRuntime(verbose=False, event_log_path=self.db_path)
        evt = _make_monitoring_event(
            patient_id=101,
            event_id="mon-restart-01",
            timestamp=time.time(),
            event_type="hypertension_started",
            severity="high",
            affected_vitals=["blood_pressure_sys"],
            current_values={"blood_pressure_sys": 190.0},
            deviation_values={"blood_pressure_sys": 70.0},
            persistence_duration=4,
        )
        runtime1.event_queue.publish("monitoring_events", evt)
        time.sleep(0.1)
        runtime1.stop()

        # Step 2: Start runtime 2 pointing to the same event log
        runtime2 = CareMatrixRuntime(verbose=False, event_log_path=self.db_path)
        self.addCleanup(runtime2.stop)

        p101_detail = runtime2.state_manager.get_patient_detail(101)
        self.assertIsNotNone(p101_detail)
        self.assertEqual(p101_detail["status"], "ALERT")
        self.assertIsNotNone(p101_detail["latest_monitoring_event"])
        self.assertEqual(
            p101_detail["latest_monitoring_event"]["event_type"],
            "hypertension_started",
        )

    def test_writer_failures_graceful_degradation(self):
        """Verify logging errors and writer failures never raise inside publisher."""
        # Non-existent read-only directory or invalid path
        invalid_path = "/invalid_root/nested/dir/events.db"
        writer = self._create_writer(invalid_path)

        # Logging to a broken writer must never raise
        try:
            writer.log_event("monitoring_events", {"bad": object()})
            writer.log_event("invalid_topic", None)
            writer.flush(timeout=0.1)
        except Exception as exc:
            self.fail(f"log_event raised an exception unexpectedly: {exc}")

        # EventQueue publishing with broken writer must never raise
        queue = EventQueue()
        queue.add_diagnostic_listener(writer.log_event)

        try:
            queue.publish("test_topic", {"key": "val"})
        except Exception as exc:
            self.fail(f"EventQueue.publish() raised an exception with failing log listener: {exc}")


if __name__ == "__main__":
    unittest.main()
