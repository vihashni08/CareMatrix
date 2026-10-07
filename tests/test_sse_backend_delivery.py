"""Tests for SSE backend event delivery, topic mapping, and metadata sanitization."""

from __future__ import annotations

import json
import time
import unittest

from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import create_app
from communication.events import (
    AgentFailureEvent,
    AgentHeartbeatEvent,
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringDecision,
    MonitoringEvent,
    RiskDecisionEvent,
)


class TestSSEBackendDelivery(unittest.TestCase):
    def setUp(self):
        self.runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
        self.app = create_app(runtime=self.runtime)
        self.client = self.app.test_client()

    def tearDown(self):
        self.runtime.stop()

    def test_topic_routing_and_sse_payload_structures(self):
        """Verify EventQueue published topics map to expected SSE event types."""
        captured_sse = []

        # Intercept the server's _broadcast_sse via test client listener
        def mock_listener(topic, event):
            pass

        # Use an SSE client generator test
        response = self.client.get("/api/events/stream")
        self.assertEqual(response.status_code, 200)

        # First item in generator is connection_established
        gen = iter(response.response)
        first_chunk = next(gen).decode("utf-8")
        self.assertIn("connection_established", first_chunk)

        # Simulate agent events published through the EventQueue
        now = time.time()

        # 1. MonitoringEvent on "monitoring_events"
        m_event = MonitoringEvent(
            patient_id=101,
            event_id="evt_mon_001",
            timestamp=now,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR"],
            current_values={"HR": 130.0},
            baseline_values={"HR": 75.0},
            deviation_values={"HR": 55.0},
            trends={"HR": "RISING"},
            signal_quality={"HR": "good"},
            persistence_duration=15,
            recommended_action="ESCALATE_TO_RISK",
            metadata={
                "agentic_mode": True,
                "tool_trace": [
                    {
                        "iteration": 1,
                        "thought": "I should call preprocess on this patient vitals.",
                        "action": "call_tool",
                        "tool_name": "preprocess",
                        "tool_args": {},
                        "status": "success",
                    },
                    {
                        "iteration": 2,
                        "thought": "Now calculate baseline.",
                        "action": "finish",
                        "tool_name": "",
                        "tool_args": {},
                        "status": "finished",
                        "rationale": "High heart rate confirmed.",
                    },
                ],
            },
        )
        self.runtime.event_queue.publish("monitoring_events", m_event)

        # 2. RiskDecisionEvent on "risk_decisions"
        r_event = RiskDecisionEvent(
            patient_id=101,
            event_id="evt_risk_001",
            timestamp=now,
            decision="HIGH_RISK",
            risk_probability=0.87,
            risk_level="HIGH RISK",
            threshold=0.65,
            model_name="RandomForestClassifier",
            severity="moderate",
            affected_vitals=["HR"],
            metadata={"agentic_mode": False},
        )
        self.runtime.event_queue.publish("risk_decisions", r_event)

        # 3. Negotiation challenge on "risk_challenges"
        self.runtime.event_queue.publish("risk_challenges", r_event)

        # 4. DataAnalysisEvent on "data_analysis_events"
        d_event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_da_001",
            timestamp=now,
            risk_level="HIGH RISK",
            trend_metrics={"HR_slope": 1.2},
            pattern_identified=["tachycardia_sustained"],
            important_changes=["HR increased from 75 to 130"],
            metadata={
                "agentic_mode": True,
                "tool_trace": [
                    {
                        "iteration": 1,
                        "thought": "Let us check physiological trends.",
                        "action": "call_tool",
                        "tool_name": "analyze_trends",
                        "tool_args": {},
                        "confidence": 0.9,
                    }
                ],
            },
        )
        self.runtime.event_queue.publish("data_analysis_events", d_event)

        # 5. ClinicalReasoningEvent on "clinical_decisions"
        c_event = ClinicalReasoningEvent(
            case_id=101,
            event_id="evt_cr_001",
            timestamp=now,
            risk_level="HIGH RISK",
            priority="URGENT",
            clinical_summary="Patient shows acute tachycardia with rapid onset.",
            findings=["HR > 120 bpm"],
            recommended_actions=["Administer IV fluids", "Repeat ECG"],
            metadata={
                "agentic_mode": True,
                "tool_trace": [
                    {
                        "iteration": 1,
                        "thought": "Internal LLM reasoning monologue: search evidence guidelines.",
                        "action": "call_tool",
                        "tool_name": "retrieve_guidelines",
                        "tool_args": {"query": "tachycardia"},
                    }
                ],
            },
        )
        self.runtime.event_queue.publish("clinical_decisions", c_event)

        # 6. CareCoordinationEvent on "care_coordination_events"
        coord_event = CareCoordinationEvent(
            patient_id=101,
            event_id="evt_coord_001",
            timestamp=now,
            action_type="TRIGGER_URGENT_CLINICAL_ALERT",
            priority="URGENT",
            status="NEW",
            reason="Confirmed physiological deterioration",
            clinician_review_required=True,
            suggested_orders=["ECG", "IV fluids"],
            escalation_pathway="ICU_RAPID_RESPONSE",
            clinical_summary="Patient needs immediate review.",
            evidence_consistency="SUPPORTING",
            data_reliability="HIGH",
            confidence=0.92,
        )
        self.runtime.event_queue.publish("care_coordination_events", coord_event)

        # Read the stream chunks and parse messages
        received_messages = []
        for _ in range(6):
            chunk = next(gen).decode("utf-8")
            for line in chunk.strip().split("\n"):
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    received_messages.append(payload)

        received_types = [m["type"] for m in received_messages]
        self.assertIn("monitoring_alert", received_types)
        self.assertIn("risk_prediction", received_types)
        self.assertIn("agent_negotiation", received_types)
        self.assertIn("data_analysis", received_types)
        self.assertIn("clinical_reasoning", received_types)
        self.assertIn("care_coordination", received_types)

        # Verify chain-of-thought sanitization
        for m in received_messages:
            data = m["data"]
            if m["type"] == "monitoring_alert":
                tool_trace = data.get("metadata", {}).get("tool_trace", [])
                self.assertTrue(len(tool_trace) > 0)
                for step in tool_trace:
                    self.assertNotIn("thought", step, "Raw internal thought should be stripped from tool_trace")
                    self.assertIn("action", step)
                    self.assertIn("tool_name", step)
            elif m["type"] == "data_analysis":
                tool_trace = data.get("metadata", {}).get("tool_trace", [])
                for step in tool_trace:
                    self.assertNotIn("thought", step)
                    self.assertIn("tool_name", step)
            elif m["type"] == "clinical_reasoning":
                tool_trace = data.get("metadata", {}).get("tool_trace", [])
                for step in tool_trace:
                    self.assertNotIn("thought", step)
                    self.assertIn("tool_name", step)

    def test_health_endpoint_enrichment(self):
        """Verify /api/health exposes detailed agent metrics and backward-compatible keys."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Backward compatible keys
        self.assertIn("status", data)
        self.assertIn("all_healthy", data)
        self.assertIn("total_agents", data)
        self.assertIn("risk_model_name", data)
        self.assertIn("supervisor", data)
        self.assertIn("runtime_metrics", data)
        self.assertIn("active_alert_count", data)
        self.assertIn("timestamp", data)

        # Enriched keys
        self.assertIn("agents", data)
        self.assertIn("failure_log", data)
        self.assertIsInstance(data["agents"], dict)
        self.assertIn("MonitoringAgent", data["agents"])
        self.assertIn("RiskAgent", data["agents"])
        self.assertIn("DataAnalysisAgent", data["agents"])
        self.assertIn("ClinicalReasoningAgent", data["agents"])
        self.assertIn("CareCoordinationAgent", data["agents"])

        mon_agent_info = data["agents"]["MonitoringAgent"]
        self.assertIn("status", mon_agent_info)
        self.assertIn("last_heartbeat_ago_seconds", mon_agent_info)
        self.assertIn("restart_count", mon_agent_info)

    def test_multi_patient_isolation_in_events(self):
        """Verify patient_id / case_id is preserved accurately in events without cross-patient leakage."""
        m_event_102 = MonitoringEvent(
            patient_id=102,
            event_id="evt_mon_102",
            timestamp=time.time(),
            event_type="alert_started",
            severity="high",
            affected_vitals=["SpO2"],
            current_values={"SpO2": 88.0},
            baseline_values={"SpO2": 97.0},
            deviation_values={"SpO2": -9.0},
            trends={"SpO2": "FALLING"},
            signal_quality={"SpO2": "good"},
            persistence_duration=20,
            recommended_action="ESCALATE_TO_RISK",
        )

        response = self.client.get("/api/events/stream")
        gen = response.response
        _ = next(gen)  # connection_established

        self.runtime.event_queue.publish("monitoring_events", m_event_102)

        chunk = next(gen).decode("utf-8")
        found_data = None
        for line in chunk.strip().split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                if payload["type"] == "monitoring_alert":
                    found_data = payload["data"]

        self.assertIsNotNone(found_data)
        self.assertEqual(found_data["patient_id"], 102)
        self.assertEqual(found_data["case_id"], 102)
        self.assertIn("SpO2", found_data["affected_vitals"])


if __name__ == "__main__":
    unittest.main()
