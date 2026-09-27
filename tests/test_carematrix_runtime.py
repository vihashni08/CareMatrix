"""Unit tests for CareMatrix Continuous Monitoring Runtime and Adaptive Execution."""

from __future__ import annotations

import time
import unittest

from carematrix_runtime import (
    CareMatrixRuntime,
    PatientProfile,
    PatientScenario,
    PatientStateManager,
    PatientStreamSimulator,
)
from communication.events import MonitoringDecision


class TestCareMatrixRuntime(unittest.TestCase):
    def test_simulator_scenarios(self):
        """Verify simulator generates correct physiological profiles across all 7 scenarios."""
        sim = PatientStreamSimulator()

        # 1. Stable
        sim.set_scenario(101, PatientScenario.STABLE)
        obs_stable = sim.generate_observation(101)
        self.assertAlmostEqual(obs_stable["HR"], 72.0, delta=5.0)
        self.assertAlmostEqual(obs_stable["MAP"], 85.0, delta=8.0)

        # 2. Gradual Deterioration
        sim.set_scenario(101, PatientScenario.GRADUAL_DETERIORATION)
        for _ in range(45):
            obs_det = sim.generate_observation(101)
        self.assertGreater(obs_det["HR"], 100.0)
        self.assertLess(obs_det["MAP"], 65.0)

        # 3. Sudden Abnormality
        sim.set_scenario(101, PatientScenario.SUDDEN_ABNORMALITY)
        for _ in range(8):
            obs_sudden = sim.generate_observation(101)
        self.assertGreater(obs_sudden["HR"], 100.0)
        self.assertLess(obs_sudden["MAP"], 60.0)

        # 4. Recovery
        sim.set_scenario(101, PatientScenario.RECOVERY)
        for _ in range(30):
            obs_rec = sim.generate_observation(101)
        self.assertAlmostEqual(obs_rec["HR"], 72.0, delta=6.0)

        # 5. Noisy Sensor
        sim.set_scenario(101, PatientScenario.NOISY_SENSOR)
        has_noise = False
        for _ in range(20):
            obs_noise = sim.generate_observation(101)
            if obs_noise["SpO2"] is not None and obs_noise["SpO2"] < 90.0:
                has_noise = True
                break
        self.assertTrue(has_noise)

        # 6. Missing Data
        sim.set_scenario(101, PatientScenario.MISSING_DATA)
        has_missing = False
        for _ in range(10):
            obs_miss = sim.generate_observation(101)
            if obs_miss["BT"] is None or obs_miss["MAP"] is None or obs_miss["SpO2"] is None:
                has_missing = True
                break
        self.assertTrue(has_missing)

        # 7. Persistent Abnormality
        sim.set_scenario(101, PatientScenario.PERSISTENT_ABNORMALITY)
        obs_pers = sim.generate_observation(101)
        self.assertGreater(obs_pers["HR"], 110.0)
        self.assertLess(obs_pers["MAP"], 60.0)

    def test_adaptive_execution_bypasses_stable_observations(self):
        """Verify Monitoring Agent acts as gatekeeper and bypasses downstream pipeline on stable vitals."""
        runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
        # Ensure all beds are stable
        for pid in runtime.simulator.patients:
            runtime.simulator.set_scenario(pid, PatientScenario.STABLE)

        # Run 25 steps
        for _ in range(25):
            runtime.step()

        # All stable cycles should be bypassed
        self.assertGreater(runtime.metrics["routine_bypassed_cycles"], 20)
        self.assertEqual(runtime.metrics["escalated_cycles"], 0)

        runtime.stop()

    def test_supervisor_supervises_all_five_agents(self):
        """Verify Supervisor registers and tracks all 5 agents."""
        runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
        runtime.start()

        time.sleep(0.5)
        health = runtime.supervisor.get_system_health_snapshot()

        self.assertEqual(health["total_agents"], 5)
        expected_agents = {
            "MonitoringAgent",
            "RiskAgent",
            "DataAnalysisAgent",
            "ClinicalReasoningAgent",
            "CareCoordinationAgent",
        }
        self.assertEqual(set(health["agents"].keys()), expected_agents)
        self.assertTrue(health["all_healthy"])

        runtime.stop()

    def test_state_manager_multi_patient(self):
        """Verify centralized state manager maintains independent records per patient."""
        sm = PatientStateManager()

        obs1 = {"patient_id": 101, "Time": 1.0, "HR": 75.0, "MAP": 82.0, "SpO2": 98.0, "RR": 14.0, "BT": 36.8}
        obs2 = {"patient_id": 102, "Time": 1.0, "HR": 115.0, "MAP": 55.0, "SpO2": 92.0, "RR": 24.0, "BT": 38.5}

        sm.record_observation(obs1)
        sm.record_observation(obs2)

        p1 = sm.get_patient_detail(101)
        p2 = sm.get_patient_detail(102)

        self.assertEqual(p1["latest_vitals"]["HR"], 75.0)
        self.assertEqual(p2["latest_vitals"]["HR"], 115.0)
        self.assertEqual(len(sm.get_patient_summary_list()), 3)


if __name__ == "__main__":
    unittest.main()
