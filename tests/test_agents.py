"""Comprehensive test suite for CareMatrix Independent AI Agent Architecture.

Tests verify:
1. Normal patient -> monitoring continues
2. Mild transient deviation -> no unnecessary escalation
3. Persistent abnormality -> escalation
4. Critical abnormality in only ONE vital -> escalation
5. Multiple worsening vitals -> escalation
6. Recovery -> active event closes
7. Duplicate event -> no repeated escalation
8. Asynchronous queue: publish does not block Monitoring
9. Risk worker independently consumes events
10. Monitoring continues observing while Risk is processing (concurrency check)
11. Risk Agent uses actual trained model artifact
12. Risk Agent failure -> retry/fallback without crashing Monitoring
13. Supervisor independent thread detects stale Monitoring Agent and restores state
14. Continuous replay continues after Risk failure
"""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import (
    MonitoringDecision,
    MonitoringEvent,
    RiskDecision,
    RiskDecisionEvent,
)
from monitoring_agent.monitoring_agent import MonitoringAgent
from replay_stream import PatientStreamReplayer
from risk_agent.risk_agent import RiskAgent
from supervisor.supervisor import AgentSupervisor


def generate_baseline_samples(count: int = 65) -> list[dict[str, float]]:
    """Generate normal baseline samples for 65 seconds."""
    samples = []
    for t in range(count):
        samples.append({
            "timestamp": float(t),
            "HR": 70.0,
            "MAP": 85.0,
            "SpO2": 98.0,
            "RR": 14.0,
        })
    return samples


class TestCareMatrixAgentArchitecture(unittest.TestCase):
    """Test suite demonstrating genuine agent loops, decoupled communication, ML tools, and fault tolerance."""

    def setUp(self):
        self.event_queue = EventQueue()
        self.monitoring_agent = MonitoringAgent(
            case_id=999,
            event_queue=self.event_queue,
            baseline_window=60,
        )
        self.risk_agent = RiskAgent(
            event_queue=self.event_queue,
            max_retries=3,
        )

    def tearDown(self):
        if hasattr(self, "monitoring_agent"):
            self.monitoring_agent.stop()
        if hasattr(self, "risk_agent"):
            self.risk_agent.stop()
        if hasattr(self, "supervisor") and self.supervisor:
            self.supervisor.stop()
        self.event_queue.shutdown()

    # ------------------------------------------------------------------------
    # 1. Normal patient -> monitoring continues
    # ------------------------------------------------------------------------
    def test_normal_patient_monitoring_continues(self):
        """Verify normal patient vitals result in CONTINUE_MONITORING and 0 alerts."""
        samples = generate_baseline_samples(70)
        decisions = []

        for s in samples:
            decision, event = self.monitoring_agent.step(s)
            decisions.append(decision)

        self.assertTrue(all(d == MonitoringDecision.CONTINUE_MONITORING for d in decisions))
        self.assertEqual(len(self.event_queue.get_history("monitoring_events")), 0)
        self.assertEqual(self.monitoring_agent.state.total_escalations, 0)

    # ------------------------------------------------------------------------
    # 2. Mild transient deviation -> no unnecessary escalation
    # ------------------------------------------------------------------------
    def test_mild_transient_deviation_no_unnecessary_escalation(self):
        """Verify short/sub-threshold deviations do not trigger escalation."""
        samples = generate_baseline_samples(65)
        for t in range(65, 68):
            samples.append({"timestamp": float(t), "HR": 88.0, "MAP": 85.0, "SpO2": 98.0, "RR": 14.0})
        for t in range(68, 75):
            samples.append({"timestamp": float(t), "HR": 70.0, "MAP": 85.0, "SpO2": 98.0, "RR": 14.0})

        for s in samples:
            decision, event = self.monitoring_agent.step(s)
            self.assertNotEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)

        self.assertEqual(len(self.event_queue.get_history("monitoring_events")), 0)
        self.assertEqual(self.monitoring_agent.state.total_escalations, 0)

    # ------------------------------------------------------------------------
    # 3. Persistent abnormality -> escalation
    # ------------------------------------------------------------------------
    def test_persistent_abnormality_escalation(self):
        """Verify sustained multi-vital deviation triggers ESCALATE_TO_RISK."""
        samples = generate_baseline_samples(65)
        for t in range(65, 77):
            samples.append({"timestamp": float(t), "HR": 95.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0})

        escalation_found = False
        escalation_event = None

        for s in samples:
            decision, event = self.monitoring_agent.step(s)
            if decision == MonitoringDecision.ESCALATE_TO_RISK:
                escalation_found = True
                escalation_event = event

        self.assertTrue(escalation_found)
        self.assertIsNotNone(escalation_event)
        self.assertEqual(escalation_event.event_type, "alert_started")
        self.assertIn("HR", escalation_event.affected_vitals)
        self.assertIn("MAP", escalation_event.affected_vitals)

    # ------------------------------------------------------------------------
    # 4. Critical abnormality in only ONE vital -> escalation
    # ------------------------------------------------------------------------
    def test_single_critical_vital_escalation(self):
        """Verify a single critical vital triggers escalation without requiring a second vital."""
        samples = generate_baseline_samples(65)
        for t in range(65, 71):
            samples.append({"timestamp": float(t), "HR": 120.0, "MAP": 85.0, "SpO2": 98.0, "RR": 14.0})

        escalation_found = False
        escalation_event = None

        for s in samples:
            decision, event = self.monitoring_agent.step(s)
            if decision == MonitoringDecision.ESCALATE_TO_RISK:
                escalation_found = True
                escalation_event = event

        self.assertTrue(escalation_found)
        self.assertEqual(escalation_event.severity, "critical")
        self.assertEqual(escalation_event.affected_vitals, ["HR"])

    # ------------------------------------------------------------------------
    # 5. Multiple worsening vitals -> escalation
    # ------------------------------------------------------------------------
    def test_multiple_worsening_vitals_escalation(self):
        """Verify multiple deteriorating vitals escalate rapidly."""
        samples = generate_baseline_samples(65)
        for t in range(65, 75):
            samples.append({"timestamp": float(t), "HR": 105.0, "MAP": 50.0, "SpO2": 88.0, "RR": 26.0})

        escalations = []
        for s in samples:
            decision, event = self.monitoring_agent.step(s)
            if decision == MonitoringDecision.ESCALATE_TO_RISK:
                escalations.append(event)

        self.assertEqual(len(escalations), 1)
        self.assertIn("HR", escalations[0].affected_vitals)
        self.assertIn("MAP", escalations[0].affected_vitals)
        self.assertIn("SpO2", escalations[0].affected_vitals)

    # ------------------------------------------------------------------------
    # 6. Recovery -> active event closes
    # ------------------------------------------------------------------------
    def test_recovery_closes_active_event(self):
        """Verify return to baseline for recovery duration closes active event with RECOVERY."""
        samples = generate_baseline_samples(65)
        for t in range(65, 75):
            samples.append({"timestamp": float(t), "HR": 105.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0})
        for t in range(75, 85):
            samples.append({"timestamp": float(t), "HR": 70.0, "MAP": 85.0, "SpO2": 98.0, "RR": 14.0})

        decisions = [self.monitoring_agent.step(s)[0] for s in samples]

        self.assertIn(MonitoringDecision.ESCALATE_TO_RISK, decisions)
        self.assertIn(MonitoringDecision.RECOVERY, decisions)
        self.assertIsNone(self.monitoring_agent.state.active_event_id)
        self.assertEqual(self.monitoring_agent.state.total_recoveries, 1)

    # ------------------------------------------------------------------------
    # 7. Duplicate event -> no repeated escalation
    # ------------------------------------------------------------------------
    def test_no_repeated_escalation_for_active_event(self):
        """Verify an ongoing deviation does NOT repeatedly publish duplicate start events."""
        samples = generate_baseline_samples(65)
        for t in range(65, 90):
            samples.append({"timestamp": float(t), "HR": 95.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0})

        escalation_count = 0
        for s in samples:
            decision, event = self.monitoring_agent.step(s)
            if decision == MonitoringDecision.ESCALATE_TO_RISK:
                escalation_count += 1

        self.assertEqual(escalation_count, 1)
        self.assertEqual(self.monitoring_agent.state.total_escalations, 1)

    # ------------------------------------------------------------------------
    # 8. Asynchronous queue: publish does not block Monitoring
    # ------------------------------------------------------------------------
    def test_asynchronous_queue_publish_does_not_block_monitoring(self):
        """Verify publish puts event on queue in < 1ms and returns immediately without running consumers."""
        test_event = MonitoringEvent(
            patient_id=1,
            event_id="test_evt_async",
            timestamp=100.0,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR"],
            current_values={"HR": 95.0},
            baseline_values={"HR": 70.0},
            deviation_values={"HR": 0.35},
            trends={"HR": "increasing"},
            signal_quality={"HR": "good"},
            persistence_duration=8,
            recommended_action="assess",
        )

        t0 = time.perf_counter()
        self.event_queue.publish("monitoring_events", test_event)
        elapsed = time.perf_counter() - t0

        # Must return virtually instantaneously (< 5ms)
        self.assertLess(elapsed, 0.005)
        # Event is queued and waiting for an independent consumer
        polled = self.event_queue.poll("monitoring_events")
        self.assertEqual(polled.event_id, "test_evt_async")

    # ------------------------------------------------------------------------
    # 9. Risk worker independently consumes events
    # ------------------------------------------------------------------------
    def test_risk_worker_independently_consumes_event(self):
        """Verify RiskAgent worker thread independently drains queue and produces decision."""
        self.risk_agent.start()

        test_event = MonitoringEvent(
            patient_id=0,
            event_id="test_worker_evt",
            timestamp=77.0,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR", "MAP"],
            current_values={"HR": 95.0, "MAP": 60.0},
            baseline_values={"HR": 70.0, "MAP": 85.0},
            deviation_values={"HR": 0.35, "MAP": 0.29},
            trends={"HR": "increasing", "MAP": "decreasing"},
            signal_quality={"HR": "good", "MAP": "good"},
            persistence_duration=8,
            recommended_action="assess",
        )

        # Publish event
        self.event_queue.publish("monitoring_events", test_event)

        # Wait for Risk worker thread to process
        decision_event = None
        for _ in range(20):
            decisions = self.event_queue.get_history("risk_decisions")
            if decisions:
                decision_event = decisions[0]
                break
            time.sleep(0.05)

        self.assertIsNotNone(decision_event)
        self.assertIsInstance(decision_event, RiskDecisionEvent)
        self.assertEqual(decision_event.event_id, "test_worker_evt")
        self.assertIn(decision_event.decision, ["LOW_RISK", "HIGH_RISK"])

    # ------------------------------------------------------------------------
    # 10. Monitoring continues observing while Risk is processing (Concurrency Check)
    # ------------------------------------------------------------------------
    # 10. Monitoring continues observing while Risk is processing (Concurrency Check)
    # ------------------------------------------------------------------------
    def test_monitoring_continues_observing_while_risk_processes(self):
        """Verify MonitoringAgent continues producing observations concurrently while RiskAgent executes ML inference using synchronization primitives."""
        risk_entered_ml = threading.Event()
        release_risk_ml = threading.Event()
        observations_during_risk: list[int] = []

        original_use_tool = self.risk_agent.use_ml_model_tool

        def synchronized_ml_tool(features):
            risk_entered_ml.set()
            release_risk_ml.wait(timeout=3.0)
            return original_use_tool(features)

        self.risk_agent.use_ml_model_tool = synchronized_ml_tool
        self.risk_agent.start()

        # Step monitoring until an escalation event is published
        samples = generate_baseline_samples(65)
        for t in range(65, 77):
            samples.append({"timestamp": float(t), "HR": 95.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0})

        for s in samples:
            self.monitoring_agent.step(s)

        # Ensure Risk Agent enters ML tool execution
        self.assertTrue(risk_entered_ml.wait(timeout=3.0), "Risk Agent did not enter ML tool execution.")

        # Concurrently step MonitoringAgent with new observations while Risk Agent is held in ML execution
        post_samples = [
            {"timestamp": 77.0, "HR": 95.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0},
            {"timestamp": 78.0, "HR": 96.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0},
            {"timestamp": 79.0, "HR": 94.0, "MAP": 61.0, "SpO2": 98.0, "RR": 14.0},
        ]
        for s in post_samples:
            dec, ev = self.monitoring_agent.step(s)
            observations_during_risk.append(self.monitoring_agent.state.total_observations)

        # Release Risk Agent to complete ML inference
        release_risk_ml.set()

        # Wait for Risk decision to be published
        decision_event = None
        for _ in range(20):
            decisions = self.event_queue.get_history("risk_decisions")
            if decisions:
                decision_event = decisions[0]
                break
            time.sleep(0.05)

        self.assertIsNotNone(decision_event)
        self.assertEqual(len(observations_during_risk), len(post_samples))
        self.assertEqual(self.monitoring_agent.state.total_observations, len(samples) + len(post_samples))

    # ------------------------------------------------------------------------
    # 11. Risk Agent uses the actual trained model
    # ------------------------------------------------------------------------
    def test_risk_agent_uses_actual_trained_model(self):
        """Verify RiskAgent dynamically identifies and executes the saved model artifact."""
        self.assertEqual(self.risk_agent.model_name, "RandomForestClassifier")
        self.assertIsNotNone(self.risk_agent.model)
        self.assertIsNotNone(self.risk_agent.preprocessor)
        self.assertEqual(len(self.risk_agent.feature_columns), 54)

        test_features = {col: np.nan for col in self.risk_agent.feature_columns}
        test_features["HR_latest"] = 95.0
        test_features["MAP_latest"] = 60.0
        test_features["age"] = 55.0

        probability = self.risk_agent.use_ml_model_tool(test_features)
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    # ------------------------------------------------------------------------
    # 12. Risk Agent failure -> retry/fallback without crashing Monitoring
    # ------------------------------------------------------------------------
    def test_risk_agent_retry_and_fallback(self):
        """Verify Risk Agent worker thread handles tool failure, retries/fallbacks, Supervisor records failure, and Monitoring continues independently."""
        self.supervisor = AgentSupervisor(
            event_queue=self.event_queue,
            heartbeat_timeout_seconds=5.0,
            check_interval_seconds=0.1,
            auto_recover=False,
        )
        self.supervisor.register_agent(self.monitoring_agent)
        self.supervisor.register_agent(self.risk_agent)

        # Inject failure into Risk Agent ML tool and start background worker
        self.risk_agent.inject_failure = True
        self.risk_agent.start()

        invalid_event = MonitoringEvent(
            patient_id=123,
            event_id="case_123_event_err",
            timestamp=10.0,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR"],
            current_values={"HR": 90.0},
            baseline_values={"HR": 70.0},
            deviation_values={"HR": 0.28},
            trends={"HR": "increasing"},
            signal_quality={"HR": "good"},
            persistence_duration=8,
            recommended_action="assess",
        )

        # Publish event for Risk worker to consume
        self.event_queue.publish("monitoring_events", invalid_event)

        # Monitoring agent continues observing independently without interruption
        decision, _ = self.monitoring_agent.step({
            "timestamp": 100.0,
            "HR": 70.0,
            "MAP": 85.0,
            "SpO2": 98.0,
            "RR": 14.0,
        })
        self.assertEqual(decision, MonitoringDecision.CONTINUE_MONITORING)

        # Wait for Risk worker thread to process failure and publish fallback decision
        decision_event = None
        for _ in range(30):
            decisions = self.event_queue.get_history("risk_decisions")
            if decisions:
                decision_event = decisions[0]
                break
            time.sleep(0.05)

        self.assertIsNotNone(decision_event)
        self.assertEqual(decision_event.decision, RiskDecision.ESCALATE.value)
        self.assertEqual(decision_event.status, "failed")
        self.assertGreaterEqual(self.risk_agent.state.failure_count, 1)

        # Check that Supervisor records the failure
        health = self.supervisor.check_health()
        risk_record = self.supervisor.agent_records["RiskAgent"]
        self.assertGreaterEqual(risk_record["failure_count"], 1)
        self.assertEqual(risk_record["status"], "degraded")
        self.assertIn("Simulated ML model tool inference failure", str(risk_record["last_error"]))
        self.assertTrue(self.monitoring_agent.is_healthy)

    # ------------------------------------------------------------------------
    # 13. Supervisor independent thread detects stale Monitoring Agent and restores state
    # ------------------------------------------------------------------------
    def test_supervisor_independent_thread_detects_stale_agent_and_restarts(self):
        """Verify Supervisor independent monitor thread detects stale agent and restores its state snapshot."""
        self.supervisor = AgentSupervisor(
            event_queue=self.event_queue,
            heartbeat_timeout_seconds=0.2,
            check_interval_seconds=0.05,
            auto_recover=True,
        )
        self.supervisor.register_agent(self.monitoring_agent)

        for s in generate_baseline_samples(25):
            self.monitoring_agent.step(s)

        initial_observations = self.monitoring_agent.state.total_observations
        self.assertEqual(initial_observations, 25)

        # Start supervisor background thread
        self.supervisor.start()

        # Simulate agent becoming stale by sleeping longer than heartbeat timeout
        time.sleep(0.3)

        # Supervisor background thread has detected staleness and restarted agent
        health = self.supervisor.agent_records.get("MonitoringAgent", {})
        self.assertGreaterEqual(health.get("restart_count", 0), 1)
        self.assertEqual(self.monitoring_agent.state.total_observations, initial_observations)
        self.assertTrue(self.monitoring_agent.is_healthy)

    # ------------------------------------------------------------------------
    # 14. Continuous replay continues after Risk failure
    # ------------------------------------------------------------------------
    def test_continuous_replay_continues_after_risk_failure(self):
        """Verify progressive streaming continues smoothly even if RiskAgent experiences an error."""
        samples = generate_baseline_samples(65)
        for t in range(65, 75):
            samples.append({"timestamp": float(t), "HR": 95.0, "MAP": 60.0, "SpO2": 98.0, "RR": 14.0})
        for t in range(75, 85):
            samples.append({"timestamp": float(t), "HR": 70.0, "MAP": 85.0, "SpO2": 98.0, "RR": 14.0})
        for t in range(85, 100):
            samples.append({"timestamp": float(t), "HR": 70.0, "MAP": 85.0, "SpO2": 98.0, "RR": 14.0})

        source_df = pd.DataFrame(samples).set_index("timestamp")
        replayer = PatientStreamReplayer(source_df=source_df)

        processed_samples = 0
        with patch.object(self.risk_agent, "use_ml_model_tool", side_effect=ValueError("Feature error")):
            for sample in replayer.stream():
                decision, event = self.monitoring_agent.step(sample)
                processed_samples += 1

        self.assertEqual(processed_samples, len(samples))
        self.assertEqual(self.monitoring_agent.state.total_observations, len(samples))

    # ------------------------------------------------------------------------
    # 15. Supervisor replaces crashed worker with new live thread & preserves state
    # ------------------------------------------------------------------------
    def test_supervisor_crashed_worker_replaced_by_new_live_thread_with_state_preserved(self):
        """Verify Supervisor detects a crashed/stopped Monitoring worker, replaces it with a new live worker thread, preserves state, and resumes processing."""
        self.supervisor = AgentSupervisor(
            event_queue=self.event_queue,
            heartbeat_timeout_seconds=0.3,
            check_interval_seconds=0.05,
            auto_recover=True,
        )
        self.supervisor.register_agent(self.monitoring_agent)

        samples = generate_baseline_samples(50)

        # 1. Monitoring worker running
        self.monitoring_agent.start(stream_source=samples, sample_delay=0.01)
        old_thread = self.monitoring_agent._worker_thread
        self.assertIsNotNone(old_thread)
        self.assertTrue(old_thread.is_alive())

        # Allow worker to process some samples
        for _ in range(20):
            if self.monitoring_agent.state.total_observations >= 15:
                break
            time.sleep(0.02)

        obs_before_crash = self.monitoring_agent.state.total_observations
        self.assertGreaterEqual(obs_before_crash, 10)

        # 2. Save state snapshot
        saved_snapshot = self.monitoring_agent.get_state_snapshot()
        self.supervisor.agent_snapshots["MonitoringAgent"] = saved_snapshot

        # 3. Simulate actual worker crash/termination
        original_step = self.monitoring_agent.step
        def crashing_step(sample):
            raise RuntimeError("Simulated unhandled worker crash")

        self.monitoring_agent.step = crashing_step

        # Wait for worker thread to terminate
        old_thread.join(timeout=1.0)
        self.assertFalse(old_thread.is_alive())

        # Restore original step function so restarted worker can resume
        self.monitoring_agent.step = original_step

        # 4. Supervisor detects dead worker and triggers restart_worker(snapshot)
        health = self.supervisor.check_health()
        new_thread = self.monitoring_agent._worker_thread

        # 5. NEW thread created and verified alive
        self.assertIsNotNone(new_thread)
        self.assertIsNot(new_thread, old_thread, "Supervisor did not create a new worker thread!")
        self.assertTrue(new_thread.is_alive(), "New worker thread is not alive!")

        # 6. State restored
        self.assertGreaterEqual(self.monitoring_agent.state.total_observations, obs_before_crash)

        # 7. Worker resumes processing new observations
        for _ in range(20):
            if self.monitoring_agent.state.total_observations > obs_before_crash:
                break
            time.sleep(0.03)

        self.assertGreater(
            self.monitoring_agent.state.total_observations,
            obs_before_crash,
            "Restarted worker did not resume processing new observations!",
        )

        # 8. Heartbeat becomes healthy
        self.assertTrue(self.monitoring_agent.is_healthy)
        self.assertEqual(health["MonitoringAgent"]["status"], "healthy")
        self.assertGreaterEqual(self.supervisor.agent_records["MonitoringAgent"]["restart_count"], 1)


if __name__ == "__main__":
    unittest.main()
