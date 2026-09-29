"""Unit tests for Phase 4 pipeline behavior fixes:
1. Acute physiological collapse vs sensor artifact differentiation.
2. Sustained sensor disconnect alerting.
3. Per-stage and cumulative latency tracking across agents.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import (
    CareActionType,
    MonitoringDecision,
    MonitoringEvent,
    RiskDecision,
    RiskDecisionEvent,
)
from monitoring_agent.decision_engine import MonitoringDecisionEngine
from monitoring_agent.monitoring_agent import MonitoringAgent
from monitoring_agent.state import PatientMonitoringState, VITAL_COLUMNS
from data_analysis_agent.analyzer import assess_window_quality
from data_analysis_agent.data_analysis_agent import DataAnalysisAgent
from clinical_reasoning_agent.clinical_reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning_agent.reasoning import ClinicalReasoningEngine
from care_coordination_agent.care_coordination_agent import CareCoordinationAgent
from care_coordination_agent.action_policy import ActionPolicy


class PipelineBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()

    def test_sustained_step_change_not_flagged_as_sensor_artifact(self):
        """Acute multi-vital collapse must not be flagged as a sensor artifact."""
        # 60 samples: baseline stable MAP=80, HR=75, then at sample 30 step drop to MAP=55, HR=110
        samples = 60
        df = pd.DataFrame({
            "Time": np.arange(samples) * 5.0,
            "HR": [75.0] * 30 + [110.0] * 30,
            "MAP": [80.0] * 30 + [55.0] * 30,
            "SpO2": [98.0] * 30 + [91.0] * 30,
            "RR": [14.0] * 30 + [24.0] * 30,
            "BT": [36.8] * samples,
        })
        flagged, details = assess_window_quality(df)
        self.assertFalse(flagged, "Sustained multi-vital physiological deterioration should not be flagged as an artifact")
        self.assertFalse(details["vitals"]["MAP"]["possible_artifact"])
        self.assertFalse(details["vitals"]["HR"]["possible_artifact"])

    def test_transient_spike_flagged_as_sensor_artifact(self):
        """Single-sample transient glitch/spike must be flagged as a sensor artifact."""
        samples = 60
        hr_vals = [75.0] * samples
        hr_vals[30] = 220.0  # single-sample isolated spike reversing back
        df = pd.DataFrame({
            "Time": np.arange(samples) * 5.0,
            "HR": hr_vals,
            "MAP": [80.0] * samples,
            "SpO2": [98.0] * samples,
            "RR": [14.0] * samples,
            "BT": [36.8] * samples,
        })
        flagged, details = assess_window_quality(df)
        self.assertTrue(flagged, "Isolated spike that immediately reverses must be flagged as artifact")
        self.assertTrue(details["vitals"]["HR"]["possible_artifact"])

    def test_sustained_sensor_disconnect_triggers_escalation(self):
        """15+ consecutive missing samples should trigger sensor disconnect escalation."""
        state = PatientMonitoringState(case_id=101)
        engine = MonitoringDecisionEngine(cooldown_seconds=10, max_missing_samples=15)

        # Establish baseline first (60 samples of valid data)
        for i in range(60):
            obs = {"timestamp": i * 5.0, "HR": 75.0, "MAP": 80.0, "SpO2": 98.0, "RR": 14.0}
            state.add_observation(obs)

        baselines = {"HR": 75.0, "MAP": 80.0, "SpO2": 98.0, "RR": 14.0}

        # Feed 14 missing HR samples -> should continue monitoring (under threshold)
        for i in range(60, 74):
            obs = {"timestamp": i * 5.0, "HR": np.nan, "MAP": 80.0, "SpO2": 98.0, "RR": 14.0}
            state.add_observation(obs)
            decision, event, _ = engine.evaluate(
                state=state,
                latest_clean_values={"HR": np.nan, "MAP": 80.0, "SpO2": 98.0, "RR": 14.0},
                latest_baselines=baselines,
                latest_deviations={"HR": np.nan, "MAP": 0.0, "SpO2": 0.0, "RR": 0.0},
                latest_trends={"HR": "insufficient_data", "MAP": "stable", "SpO2": "stable", "RR": "stable"},
                latest_signal_quality={"HR": "bad", "MAP": "good", "SpO2": "good", "RR": "good"},
                invalid_mask={"HR": True, "MAP": False, "SpO2": False, "RR": False},
            )
            self.assertEqual(decision, MonitoringDecision.CONTINUE_MONITORING)

        # 15th missing sample -> should trigger sensor disconnect escalation
        obs = {"timestamp": 74 * 5.0, "HR": np.nan, "MAP": 80.0, "SpO2": 98.0, "RR": 14.0}
        state.add_observation(obs)
        decision, event, _ = engine.evaluate(
            state=state,
            latest_clean_values={"HR": np.nan, "MAP": 80.0, "SpO2": 98.0, "RR": 14.0},
            latest_baselines=baselines,
            latest_deviations={"HR": np.nan, "MAP": 0.0, "SpO2": 0.0, "RR": 0.0},
            latest_trends={"HR": "insufficient_data", "MAP": "stable", "SpO2": "stable", "RR": "stable"},
            latest_signal_quality={"HR": "disconnected", "MAP": "good", "SpO2": "good", "RR": "good"},
            invalid_mask={"HR": True, "MAP": False, "SpO2": False, "RR": False},
        )
        self.assertEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNotNone(event)
        self.assertTrue(event.metadata.get("sensor_disconnected"))
        self.assertTrue(event.metadata.get("verification_required"))
        self.assertIn("HR", event.affected_vitals)

    def test_corroborated_high_risk_does_not_force_data_verification(self):
        """Corroborated high-risk deterioration must not be forced to REQUEST_DATA_VERIFICATION."""
        # Simulated DataAnalysisEvent with HIGH RISK and worsening vitals (SUPPORTING consistency)
        from communication.events import DataAnalysisEvent

        analysis_event = DataAnalysisEvent(
            case_id=242,
            event_id="case_242_event_001",
            timestamp=300.0,
            risk_level="HIGH RISK",
            trend_metrics={
                "HR": {"trend": "increasing", "latest_value": 115.0, "change_over_window": 35.0},
                "MAP": {"trend": "decreasing", "latest_value": 52.0, "change_over_window": -28.0},
            },
            pattern_identified=["Hemodynamic discordance verified: compensatory tachycardia with concurrent arterial hypotension."],
            important_changes=["HR: increasing", "MAP: decreasing"],
            data_quality_flag=False,
            analysis_status="complete",
            executed_checks=["trends", "variability", "patterns", "verification"],
            evidence_consistency="SUPPORTING",
            conflict_flags=[],
            verification_required=False,
            metadata={"probability": 0.85},
        )

        reasoning_agent = ClinicalReasoningAgent(self.queue, enable_llm=False)
        reasoning_event = reasoning_agent.process_event(analysis_event)

        self.assertEqual(reasoning_event.priority, "URGENT")
        self.assertFalse(reasoning_event.verification_required, "Corroborated high risk must not require data verification")

        # Now pass to CareCoordination
        coordination_agent = CareCoordinationAgent(self.queue)
        action_event = coordination_agent.process_event(reasoning_event)

        self.assertEqual(action_event.action_type, CareActionType.TRIGGER_URGENT_CLINICAL_ALERT.value)

    def test_per_stage_and_cumulative_latency_logging(self):
        """Each agent must record stage_latency_ms and cumulative_latency_ms."""
        # Setup pipeline mock data
        from communication.events import DataAnalysisEvent

        analysis_event = DataAnalysisEvent(
            case_id=242,
            event_id="case_242_event_001",
            timestamp=300.0,
            risk_level="LOW RISK",
            trend_metrics={"HR": {"trend": "stable", "latest_value": 75.0, "change_over_window": 0.0}},
            pattern_identified=[],
            important_changes=[],
            data_quality_flag=False,
            analysis_status="complete",
            executed_checks=["trends"],
            evidence_consistency="SUPPORTING",
            conflict_flags=[],
            verification_required=False,
            metadata={"cumulative_latency_ms": 12.5, "stage_latency_ms": 4.2},
        )

        reasoning_agent = ClinicalReasoningAgent(self.queue, enable_llm=False)
        reasoning_event = reasoning_agent.process_event(analysis_event)

        self.assertIn("stage_latency_ms", reasoning_event.metadata)
        self.assertIn("cumulative_latency_ms", reasoning_event.metadata)
        self.assertGreaterEqual(reasoning_event.metadata["cumulative_latency_ms"], 12.5)

        coord_agent = CareCoordinationAgent(self.queue)
        action_event = coord_agent.process_event(reasoning_event)

        self.assertIn("stage_latency_ms", action_event.metadata)
        self.assertIn("cumulative_latency_ms", action_event.metadata)
        self.assertGreater(action_event.metadata["cumulative_latency_ms"], reasoning_event.metadata["cumulative_latency_ms"])


if __name__ == "__main__":
    unittest.main()
