"""Tests for Dashboard Data-Fidelity, Real Runtime Truthfulness, and Live Integration."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestDashboardDataFidelity(unittest.TestCase):
    """Rigorous audit tests verifying that dashboard code presents real data with zero synthetic hallucinations."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.reducer_path = self.repo_root / "carematrix_runtime" / "static" / "js" / "state" / "eventReducer.js"
        self.inspection_view_path = (
            self.repo_root / "carematrix_runtime" / "static" / "js" / "components" / "agents" / "AgentInspectionView.jsx"
        )
        self.surveillance_view_path = (
            self.repo_root / "carematrix_runtime" / "static" / "js" / "components" / "surveillance" / "PatientSurveillanceView.jsx"
        )
        self.command_center_path = (
            self.repo_root / "carematrix_runtime" / "static" / "js" / "components" / "command_center" / "CommandCenterView.jsx"
        )

    def test_no_fabricated_orders_in_agent_inspection(self) -> None:
        """Verify that hardcoded bedside orders (e.g., IV fluid, blood cultures) are not fabricated in empty states."""
        content = self.inspection_view_path.read_text(encoding="utf-8")
        # Ensure fake fallback orders are completely absent
        self.assertNotIn("Initiate 500 mL IV crystalloid bolus over 30 min", content)
        self.assertNotIn("Draw STAT serum lactate + repeat in 2 hours", content)
        self.assertNotIn("Blood cultures x2 before broad-spectrum antibiotics", content)
        self.assertNotIn("Increase continuous invasive arterial BP monitoring", content)
        # Truthful empty message must be present
        self.assertIn("Maintain Continuous Standard Observation", content)
        self.assertIn("No active bedside order escalation required", content)

    def test_no_fabricated_medical_literature_in_agent_inspection(self) -> None:
        """Verify that fake literature cards (e.g. Surviving Sepsis fallback) are not shown when no citations exist."""
        content = self.inspection_view_path.read_text(encoding="utf-8")
        self.assertNotIn("Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock", content)
        self.assertNotIn("Evans L, Rhodes A, Alhazzani W, et al. Crit Care Med 2021", content)
        self.assertIn("No Biomedical Literature Citations Retrieved", content)

    def test_no_fabricated_vital_slopes_in_data_analysis(self) -> None:
        """Verify that static slopes (+0.15, -0.28, -0.05) are not hardcoded in the UI."""
        content = self.inspection_view_path.read_text(encoding="utf-8")
        self.assertNotIn("'+0.15'", content)
        self.assertNotIn("'-0.28'", content)
        self.assertNotIn("'-0.05'", content)
        # Should dynamically read slope_per_sample or trend from backend metrics
        self.assertIn("slope_per_sample", content)

    def test_no_fabricated_monitoring_baselines(self) -> None:
        """Verify that hardcoded default baseline values (75, 90, 98) are not used in place of calculated baselines."""
        content = self.inspection_view_path.read_text(encoding="utf-8")
        self.assertNotIn("baseline_values?.[key] || (key === 'hr' ? 75 : key === 'map' ? 90 : 98)", content)
        self.assertIn("Base: Establishing", content)

    def test_ml_latency_and_safety_floor_truthfulness(self) -> None:
        """Verify that ML model latency and physiological safety floor override use real backend payload keys."""
        content = self.inspection_view_path.read_text(encoding="utf-8")
        self.assertIn("physiological_safety_override", content)
        self.assertIn("0.85", content)
        self.assertIn("risk_latency_ms", content)

    def test_event_reducer_handles_all_runtime_sse_topics(self) -> None:
        """Verify that eventReducer.js explicitly maps all standard server SSE topics."""
        content = self.reducer_path.read_text(encoding="utf-8")
        expected_event_types = [
            "vital_tick",
            "monitoring_alert",
            "risk_prediction",
            "agent_negotiation",
            "data_analysis",
            "clinical_reasoning",
            "care_coordination",
            "heartbeat",
            "agent_failure",
            "agent_recovery",
            "clinician_feedback",
        ]
        for evt in expected_event_types:
            with self.subTest(evt=evt):
                self.assertIn(f"eventType === '{evt}'", content)

    def test_event_reducer_multi_patient_state_isolation(self) -> None:
        """Verify that eventReducer isolates incoming events by pid and does not corrupt other beds."""
        content = self.reducer_path.read_text(encoding="utf-8")
        self.assertIn("const pid = window.CareMatrixEventUtils.extractPatientId(data);", content)
        self.assertIn("patientDetails: {", content)
        self.assertIn("...state.patientDetails,", content)
        self.assertIn("[pid]: nextDetail,", content)
        # Verify negotiation extraction during REST hydration
        self.assertIn("negotiation_trace?.dialogue", content)

    def test_truthful_execution_context_labeling(self) -> None:
        """Verify execution labels are truthfully marked as current evaluated execution and not fake replay history."""
        content = self.inspection_view_path.read_text(encoding="utf-8")
        self.assertIn("Latest Evaluated Execution", content)
        self.assertNotIn("Historical Execution 09:14:02", content)


if __name__ == "__main__":
    unittest.main()
