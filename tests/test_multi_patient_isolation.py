"""Multi-Patient Telemetry Isolation and State Concurrency Tests.

Verifies that concurrent monitoring of distinct patient beds (Bed 101, Bed 102, Bed 103, Bed 104)
maintains strict mathematical, memory, and algorithmic isolation without cross-contamination.
"""

from __future__ import annotations

import unittest

from carematrix_runtime.patient_stream import PatientProfile, PatientScenario, PatientStreamSimulator
from carematrix_runtime.runtime import CareMatrixRuntime
from communication.events import MonitoringDecision, MonitoringEvent


class TestMultiPatientIsolation(unittest.TestCase):
    def setUp(self):
        self.profiles = [
            PatientProfile(patient_id=101, name="Bed 101 - Stable ICU", scenario=PatientScenario.STABLE),
            PatientProfile(patient_id=102, name="Bed 102 - Deteriorating ICU", scenario=PatientScenario.PERSISTENT_ABNORMALITY),
            PatientProfile(patient_id=103, name="Bed 103 - Artifact ICU", scenario=PatientScenario.NOISY_SENSOR),
            PatientProfile(patient_id=104, name="Bed 104 - Post-Op Recovery", scenario=PatientScenario.RECOVERY),
        ]
        self.sim = PatientStreamSimulator(profiles=self.profiles)
        self.runtime = CareMatrixRuntime(stream_interval_seconds=0.001, verbose=False)
        self.runtime.simulator = self.sim

    def tearDown(self):
        self.runtime.stop()

    def test_isolated_monitoring_states_and_buffers(self):
        """Verify each patient has an independent PatientMonitoringState and buffer."""
        # Feed 15 observations for all 4 patients
        for _ in range(15):
            for p in self.profiles:
                obs = self.sim.generate_observation(p.patient_id)
                self.runtime.state_manager.record_observation(obs)
                self.runtime.monitoring_agent.observe(obs)

        # Check monitoring states dictionary
        agent_states = self.runtime.monitoring_agent.patient_states
        self.assertEqual(len(agent_states), 4)

        for p in self.profiles:
            pid = p.patient_id
            self.assertIn(pid, agent_states)
            p_state = agent_states[pid]
            self.assertEqual(p_state.case_id, pid)
            self.assertEqual(p_state.total_observations, 15)

            # Check DataFrame buffer isolation
            df = p_state.buffer_dataframe()
            self.assertEqual(len(df), 15)

    def test_divergent_patient_decisions(self):
        """Verify abnormal patient does not cause stable patient to falsely escalate."""
        # 15 baseline steps where both beds are stable
        p101 = self.sim.get_patient(101)
        p102 = self.sim.get_patient(102)
        p101.set_scenario(PatientScenario.STABLE)
        p102.set_scenario(PatientScenario.STABLE)

        for _ in range(15):
            obs1 = self.sim.generate_observation(101)
            obs2 = self.sim.generate_observation(102)
            self.runtime.monitoring_agent.observe(obs1)
            self.runtime.monitoring_agent.observe(obs2)

        # Now induce persistent abnormality ONLY in Bed 102
        p102.set_scenario(PatientScenario.PERSISTENT_ABNORMALITY)
        decisions_101 = []
        decisions_102 = []

        for _ in range(25):
            obs1 = self.sim.generate_observation(101)
            obs2 = self.sim.generate_observation(102)
            dec1, _ = self.runtime.monitoring_agent.observe(obs1)
            dec2, _ = self.runtime.monitoring_agent.observe(obs2)
            decisions_101.append(dec1)
            decisions_102.append(dec2)

        # Bed 101 must remain CONTINUE_MONITORING throughout (100% stable)
        self.assertTrue(all(d == MonitoringDecision.CONTINUE_MONITORING for d in decisions_101))

        # Bed 102 must encounter deviation and not remain normal
        state_102 = self.runtime.monitoring_agent.patient_states[102]
        self.assertTrue(
            state_102.latest_overall_severity != "normal"
            or state_102.consecutive_candidate_samples > 0
            or any(d != MonitoringDecision.CONTINUE_MONITORING for d in decisions_102)
        )

    def test_alert_manager_patient_filtering(self):
        """Verify alert queries are strictly partitioned by patient_id."""
        event_101 = MonitoringEvent(
            patient_id=101,
            event_id="evt_iso_101",
            timestamp=100.0,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR", "MAP"],
            current_values={"HR": 115.0, "MAP": 58.0},
            baseline_values={"HR": 72.0, "MAP": 85.0},
            deviation_values={"HR": 0.59, "MAP": 0.31},
            trends={"HR": "increasing", "MAP": "decreasing"},
            signal_quality={"HR": "good", "MAP": "good"},
            persistence_duration=10,
            recommended_action="assess_patient_risk",
        )
        self.runtime.alert_manager.handle_monitoring_event(event_101)

        active_101 = self.runtime.alert_manager.get_active_alerts(patient_id=101)
        active_102 = self.runtime.alert_manager.get_active_alerts(patient_id=102)

        self.assertEqual(len(active_101), 1)
        self.assertEqual(len(active_102), 0)
        self.assertEqual(active_101[0]["patient_id"], 101)


if __name__ == "__main__":
    unittest.main()
