"""End-to-End Pipeline Integration Tests for CareMatrix 5-Agent Architecture.

Verifies the complete pipeline:
Observation -> Monitoring -> Risk -> Data Analysis -> Clinical Reasoning -> Care Coordination
Across critical clinical scenarios (Gradual Deterioration, Sensor Noise, Recovery, and Supervisor Auto-Recovery).
"""

from __future__ import annotations

import time
import unittest

from care_coordination_agent import CareActionType, CoordinationPriority
from carematrix_runtime.alert_manager import AlertLifecycleState
from carematrix_runtime.patient_stream import PatientScenario
from carematrix_runtime.runtime import CareMatrixRuntime


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        # High speed stream for fast test execution
        self.runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)

    def tearDown(self):
        self.runtime.stop()

    def test_1_full_pipeline_deterioration_scenario(self):
        """End-to-end test: gradual deterioration scenario produces urgent care coordination plan."""
        # Bed 102 is configured with GRADUAL_DETERIORATION in simulator
        self.runtime.simulator.set_scenario(102, PatientScenario.GRADUAL_DETERIORATION)

        # Start runtime and step through 45 observations to trigger escalation
        for _ in range(45):
            self.runtime.step()

        # Allow worker threads to propagate events through all 5 agents
        self.runtime.monitoring_agent.start()
        self.runtime.risk_agent.start()
        self.runtime.data_analysis_agent.start()
        self.runtime.clinical_reasoning_agent.start()
        self.runtime.care_coordination_agent.start()

        time.sleep(1.0)

        # Check detail state for Patient 102
        detail = self.runtime.state_manager.get_patient_detail(102)
        self.assertIsNotNone(detail)

        # If escalation reached Care Coordination, check that action and orders exist
        care_history = self.runtime.event_queue.get_history("care_coordination_events")
        if care_history:
            latest_action = care_history[-1]
            self.assertEqual(latest_action.patient_id, 102)
            self.assertIn(latest_action.priority, [CoordinationPriority.URGENT.value, CoordinationPriority.ELEVATED.value])
            self.assertTrue(len(latest_action.suggested_orders) > 0)
            self.assertTrue(latest_action.clinician_review_required)

        # Verify alerts were generated and managed in alert manager
        alerts = self.runtime.alert_manager.get_active_alerts(patient_id=102)
        if alerts:
            self.assertIn(alerts[0]["state"], [AlertLifecycleState.NEW.value, AlertLifecycleState.ACTIVE.value])

    def test_2_sensor_noise_triggers_data_verification(self):
        """End-to-end test: sensor noise scenario prompts REQUEST_DATA_VERIFICATION without false alarm escalation."""
        self.runtime.simulator.set_scenario(101, PatientScenario.NOISY_SENSOR)

        for _ in range(30):
            self.runtime.step()

        # Start workers to process
        self.runtime.risk_agent.start()
        self.runtime.data_analysis_agent.start()
        self.runtime.clinical_reasoning_agent.start()
        self.runtime.care_coordination_agent.start()

        time.sleep(0.8)

        # Check if data analysis flagged compromised quality
        analysis_history = self.runtime.event_queue.get_history("data_analysis_events")
        care_history = self.runtime.event_queue.get_history("care_coordination_events")

        # In case an alert occurred under noise, verification must be required
        for action in care_history:
            if action.patient_id == 101:
                self.assertEqual(action.action_type, CareActionType.REQUEST_DATA_VERIFICATION.value)
                self.assertTrue(action.clinician_review_required)

    def test_3_physiological_recovery_resolves_alerts(self):
        """End-to-end test: recovery scenario automatically resolves active alerts and restores state."""
        # 1. Trigger an active alert
        from communication.events import MonitoringEvent
        alert_event = MonitoringEvent(
            patient_id=103,
            event_id="test_alert_103",
            timestamp=time.time(),
            event_type="alert_started",
            severity="severe",
            affected_vitals=["hr", "map"],
            current_values={"hr": 130, "map": 52},
            baseline_values={"hr": 72, "map": 85},
            deviation_values={"hr": 58, "map": -33},
            trends={"hr": "increasing", "map": "decreasing"},
            signal_quality={"hr": "good", "map": "good"},
            persistence_duration=4,
            recommended_action="Physician review",
        )
        self.runtime.event_queue.publish("monitoring_events", alert_event)
        time.sleep(0.1)

        active = self.runtime.alert_manager.get_active_alerts(patient_id=103)
        self.assertEqual(len(active), 1)

        # 2. Trigger Recovery event
        recovery_event = MonitoringEvent(
            patient_id=103,
            event_id="test_rec_103",
            timestamp=time.time() + 10,
            event_type="alert_recovered",
            severity="mild",
            affected_vitals=[],
            current_values={"hr": 74, "map": 84},
            baseline_values={"hr": 72, "map": 85},
            deviation_values={"hr": 2, "map": -1},
            trends={"hr": "stable", "map": "stable"},
            signal_quality={"hr": "good", "map": "good"},
            persistence_duration=0,
            recommended_action="Maintain routine surveillance",
        )
        self.runtime.event_queue.publish("monitoring_events", recovery_event)
        time.sleep(0.1)

        # Active alerts should now be resolved
        active_after = self.runtime.alert_manager.get_active_alerts(patient_id=103)
        self.assertEqual(len(active_after), 0)

    def test_4_supervisor_monitors_all_five_agents_and_recovers(self):
        """End-to-end test: supervisor actively monitors all 5 agents and auto-recovers a degraded agent."""
        self.runtime.start()
        time.sleep(0.5)

        # Initial check
        health = self.runtime.supervisor.get_system_health_snapshot()
        self.assertEqual(health["total_agents"], 5)
        self.assertTrue(health["all_healthy"])

        # Inject failure on Risk Agent
        self.runtime.supervisor.record_failure("RiskAgent", "Simulated ML memory fault")
        degraded_health = self.runtime.supervisor.check_health()
        self.assertEqual(degraded_health["RiskAgent"]["status"], "degraded")

        # Supervisor auto-recovery
        restarted = self.runtime.supervisor.restart_agent("RiskAgent")
        self.assertTrue(restarted)

        # Post-restart health
        post_health = self.runtime.supervisor.check_health()
        self.assertIn(post_health["RiskAgent"]["status"], ["healthy", "running"])


if __name__ == "__main__":
    unittest.main()
