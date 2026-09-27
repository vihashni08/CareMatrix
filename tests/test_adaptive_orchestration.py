"""Comprehensive unit tests for Adaptive Multi-Agent Orchestration and Cross-Agent Verification.

Covers:
1. LOW risk + stable data -> standard analysis, routine monitoring, supporting consistency.
2. HIGH risk + worsening trend -> detailed analysis, urgent priority, supporting consistency.
3. Poor data quality -> verification required, reduced confidence, uncertain consistency.
4. HIGH risk + stable evidence -> conflict detected, reduced confidence, elevated priority, verification required.
5. LOW risk + worsening evidence -> conflict detected, elevated priority, verification required.
6. Supporting evidence maintains high confidence (>= 0.95).
7. Conflicting evidence reduces confidence (<= 0.70).
8. Malformed event handling safety (fallback graceful, non-crashing).
9. Dynamic check requirements generation (standard vs detailed, checks selection).
10. End-to-end asynchronous event queue propagation of adaptive & verification fields.
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import (
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringEvent,
    RiskDecisionEvent,
)
from communication.orchestration import (
    determine_analysis_requirements,
    verify_cross_agent_consistency,
)
from clinical_reasoning_agent import (
    ClinicalReasoningAgent,
    ClinicalReasoningEngine,
    ClinicalReasoningPriority,
)
from data_analysis_agent import DataAnalysisAgent
from risk_agent import RiskAgent


class TestAdaptiveOrchestrationAndVerification(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.reasoning_engine = ClinicalReasoningEngine()
        self.clinical_agent = ClinicalReasoningAgent(event_queue=self.queue, enable_llm=False)

    def tearDown(self):
        self.clinical_agent.stop()
        self.queue.shutdown()

    # 1. LOW Risk + Stable Data -> standard analysis, routine monitoring
    def test_scenario_1_low_risk_stable_data(self):
        """LOW risk with stable data requests standard analysis and produces routine monitoring."""
        analysis_level, requested_checks, verify_req = determine_analysis_requirements(
            decision="LOW_RISK",
            probability=0.05,
            severity="LOW",
            affected_vitals=["HR"],
        )
        self.assertEqual(analysis_level, "standard")
        self.assertFalse(verify_req)
        self.assertIn("trends", requested_checks)

        event = DataAnalysisEvent(
            case_id=1,
            event_id="evt_low_stable",
            timestamp=100.0,
            risk_level="LOW RISK",
            trend_metrics={"HR": {"trend": "stable", "latest_value": 70.0, "change_over_window": 0.5}},
            pattern_identified=["Stable baseline vitals."],
            data_quality_flag=False,
            analysis_status="complete",
        )
        decision = self.clinical_agent.process_event(event)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.ROUTINE.value)
        self.assertEqual(decision.evidence_consistency, "SUPPORTING")
        self.assertFalse(decision.verification_required)
        self.assertGreaterEqual(decision.confidence, 0.90)

    # 2. HIGH Risk + Worsening Trend -> detailed analysis, urgent priority, supporting
    def test_scenario_2_high_risk_worsening_trends(self):
        """HIGH risk with worsening vitals requests detailed checks and results in URGENT priority."""
        analysis_level, requested_checks, verify_req = determine_analysis_requirements(
            decision="HIGH_RISK",
            probability=0.65,
            severity="HIGH",
            affected_vitals=["HR", "MAP"],
        )
        self.assertEqual(analysis_level, "detailed")
        self.assertTrue(verify_req)
        self.assertIn("verification", requested_checks)

        event = DataAnalysisEvent(
            case_id=2,
            event_id="evt_high_worsening",
            timestamp=200.0,
            risk_level="HIGH RISK",
            trend_metrics={
                "HR": {"trend": "increasing", "latest_value": 115.0, "change_over_window": 25.0},
                "MAP": {"trend": "decreasing", "latest_value": 55.0, "change_over_window": -20.0},
            },
            pattern_identified=["Simultaneous directional changes across multiple affected vitals."],
            data_quality_flag=False,
            analysis_status="complete",
        )
        decision = self.clinical_agent.process_event(event)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertEqual(decision.evidence_consistency, "SUPPORTING")
        self.assertTrue(decision.escalation_required)
        self.assertFalse(decision.verification_required)
        self.assertGreaterEqual(decision.confidence, 0.90)

    # 3. Poor Data Quality -> verification required, reduced confidence, uncertain consistency
    def test_scenario_3_poor_data_quality_triggers_uncertainty(self):
        """Sensor/quality defects produce UNCERTAIN consistency, reduced confidence, and sensor check."""
        event = DataAnalysisEvent(
            case_id=3,
            event_id="evt_poor_quality",
            timestamp=300.0,
            risk_level="LOW RISK",
            trend_metrics={},
            pattern_identified=[],
            data_quality_flag=True,
            analysis_status="partial_analysis",
        )
        decision = self.clinical_agent.process_event(event)
        self.assertEqual(decision.evidence_consistency, "UNCERTAIN")
        self.assertEqual(decision.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertTrue(decision.verification_required)
        self.assertEqual(decision.data_reliability, "COMPROMISED")
        self.assertLessEqual(decision.confidence, 0.60)
        self.assertTrue(any("sensor" in a.lower() or "probe" in a.lower() for a in decision.recommended_actions))

    # 4. HIGH Risk + Stable Evidence -> conflict detected, reduced confidence, verification required
    def test_scenario_4_high_risk_stable_trends_conflict(self):
        """HIGH risk combined with completely stable vitals triggers CONFLICTING flag and reduces confidence."""
        event = DataAnalysisEvent(
            case_id=4,
            event_id="evt_high_stable_conflict",
            timestamp=400.0,
            risk_level="HIGH RISK",
            trend_metrics={
                "HR": {"trend": "stable", "latest_value": 75.0, "change_over_window": 0.0},
                "MAP": {"trend": "stable", "latest_value": 85.0, "change_over_window": 0.0},
            },
            pattern_identified=["Monitored vitals remain stable."],
            data_quality_flag=False,
            analysis_status="complete",
        )
        decision = self.clinical_agent.process_event(event)
        self.assertEqual(decision.evidence_consistency, "CONFLICTING")
        self.assertTrue(decision.verification_required)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertLessEqual(decision.confidence, 0.75)
        self.assertTrue(len(decision.conflicting_evidence) > 0)

    # 5. LOW Risk + Worsening Evidence -> conflict detected, elevated priority, verification required
    def test_scenario_5_low_risk_worsening_trends_conflict(self):
        """LOW risk combined with rapid multi-vital deterioration triggers CONFLICTING flag and ELEVATED priority."""
        event = DataAnalysisEvent(
            case_id=5,
            event_id="evt_low_worsening_conflict",
            timestamp=500.0,
            risk_level="LOW RISK",
            trend_metrics={
                "HR": {"trend": "increasing", "latest_value": 120.0, "change_over_window": 30.0},
                "MAP": {"trend": "decreasing", "latest_value": 50.0, "change_over_window": -25.0},
            },
            pattern_identified=["Simultaneous directional changes across multiple affected vitals."],
            data_quality_flag=False,
            analysis_status="complete",
        )
        decision = self.clinical_agent.process_event(event)
        self.assertEqual(decision.evidence_consistency, "CONFLICTING")
        self.assertTrue(decision.verification_required)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertLessEqual(decision.confidence, 0.70)
        self.assertTrue(len(decision.conflicting_evidence) > 0)

    # 6. Supporting evidence maintains high confidence
    def test_scenario_6_supporting_evidence_high_confidence(self):
        """Aligned model predictions and trends maintain confidence >= 0.90."""
        consistency, supporting, conflicting, flags, verify_req, conf = verify_cross_agent_consistency(
            risk_level="HIGH RISK",
            metrics={"HR": {"trend": "increasing", "latest_value": 110.0, "change_over_window": 15.0}},
            patterns=["HR worsening"],
            data_quality_flag=False,
            analysis_status="complete",
        )
        self.assertEqual(consistency, "SUPPORTING")
        self.assertGreaterEqual(conf, 0.90)
        self.assertFalse(verify_req)
        self.assertEqual(len(conflicting), 0)

    # 7. Conflicting evidence reduces confidence
    def test_scenario_7_conflicting_evidence_reduces_confidence(self):
        """Mismatched model predictions and trends lower confidence <= 0.70."""
        consistency, supporting, conflicting, flags, verify_req, conf = verify_cross_agent_consistency(
            risk_level="HIGH RISK",
            metrics={"HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 0.0}},
            patterns=["Vitals stable"],
            data_quality_flag=False,
            analysis_status="complete",
        )
        self.assertEqual(consistency, "CONFLICTING")
        self.assertLessEqual(conf, 0.70)
        self.assertTrue(verify_req)
        self.assertTrue(len(conflicting) > 0)

    # 8. Malformed event handling safety
    def test_scenario_8_malformed_event_handled_safely(self):
        """Non-DataAnalysisEvent payload does not raise unhandled exception and yields degraded result."""
        result = self.clinical_agent.process_event(None)  # type: ignore[arg-type]
        self.assertIsInstance(result, ClinicalReasoningEvent)
        self.assertEqual(result.evidence_consistency, "UNCERTAIN")
        self.assertTrue(result.verification_required)
        self.assertEqual(result.data_reliability, "COMPROMISED")

    # 9. Dynamic check requirements calculation
    def test_scenario_9_dynamic_analysis_requirements(self):
        """Checks calculation adjusts based on risk, probability, and affected vitals."""
        # Multi-vital low risk
        lvl1, checks1, v1 = determine_analysis_requirements("LOW_RISK", 0.08, "LOW", ["HR", "MAP"])
        self.assertEqual(lvl1, "standard")
        self.assertIn("patterns", checks1)
        self.assertFalse(v1)

        # Single vital low risk
        lvl2, checks2, v2 = determine_analysis_requirements("LOW_RISK", 0.05, "LOW", ["HR"])
        self.assertEqual(lvl2, "standard")
        self.assertNotIn("patterns", checks2)
        self.assertFalse(v2)

        # High probability above threshold 0.16
        lvl3, checks3, v3 = determine_analysis_requirements("LOW_RISK", 0.18, "MEDIUM", ["HR"])
        self.assertEqual(lvl3, "detailed")
        self.assertIn("verification", checks3)
        self.assertTrue(v3)

    # 10. Asynchronous Queue Pipeline Verification
    def test_scenario_10_queue_pipeline_adaptive_propagation(self):
        """Verify adaptive fields propagate across the queue between agents."""
        risk_event = RiskDecisionEvent(
            patient_id=10,
            event_id="evt_queue_10",
            timestamp=300.0,
            risk_probability=0.75,
            risk_level="HIGH RISK",
            decision="HIGH_RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            severity="HIGH",
            affected_vitals=["HR", "MAP"],
            window_start=0.0,
            window_end=300.0,
            analysis_level="detailed",
            requested_checks=["trends", "variability", "patterns", "verification"],
            verification_required=True,
        )

        def mock_loader(case_id: int) -> pd.DataFrame:
            times = np.arange(61) * 5.0
            return pd.DataFrame({
                "Time": times,
                "HR": np.linspace(80, 130, len(times)),
                "SpO2": np.full(len(times), 98.0),
                "RR": np.full(len(times), 14.0),
                "SBP": np.full(len(times), 120.0),
                "DBP": np.full(len(times), 75.0),
                "MAP": np.linspace(85, 55, len(times)),
                "BT": np.full(len(times), 36.7),
            })

        analysis_agent = DataAnalysisAgent(event_queue=self.queue, data_loader=mock_loader)
        analysis_event = analysis_agent.process_event(risk_event)

        self.assertIn("trends", analysis_event.executed_checks)
        self.assertIn("verification", analysis_event.executed_checks)
        self.assertEqual(analysis_event.evidence_consistency, "SUPPORTING")

        clinical_event = self.clinical_agent.process_event(analysis_event)
        self.assertEqual(clinical_event.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertEqual(clinical_event.evidence_consistency, "SUPPORTING")
        self.assertTrue(clinical_event.escalation_required)


if __name__ == "__main__":
    unittest.main()
