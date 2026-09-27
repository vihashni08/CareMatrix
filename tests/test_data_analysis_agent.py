import unittest

import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import RiskDecisionEvent
from data_analysis_agent import DataAnalysisAgent


def sample_frame():
    points = 61
    return pd.DataFrame({
        "Time": np.arange(points) * 5,
        "HR": np.linspace(70, 80, points), "SpO2": np.full(points, 98.0),
        "RR": np.full(points, 14.0), "SBP": np.full(points, 120.0),
        "DBP": np.full(points, 75.0), "MAP": np.linspace(85, 75, points),
        "BT": np.full(points, 36.7),
    })


class DataAnalysisAgentTests(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.agent = DataAnalysisAgent(self.queue, data_loader=lambda _: sample_frame())

    def event(self, event_id="case_4_event_013", level="LOW RISK"):
        return RiskDecisionEvent(
            patient_id=4, event_id=event_id, timestamp=300.0,
            decision=level.replace(" ", "_"), risk_probability=.10 if level == "LOW RISK" else .25,
            risk_level=level, threshold=.16, model_name="RandomForestClassifier",
            severity="moderate", affected_vitals=["HR", "MAP"], window_start=0, window_end=300,
        )

    def test_low_risk_analysis_preserves_classification_and_publishes(self):
        result = self.agent.process_event(self.event())
        self.assertEqual(result.risk_level, "LOW RISK")
        self.assertEqual(result.analysis_status, "complete")
        self.assertFalse(result.data_quality_flag)
        self.assertEqual(self.queue.get_history("clinical_reasoning_events")[0].event_id, result.event_id)

    def test_high_risk_analysis_reports_multi_vital_patterns(self):
        result = self.agent.process_event(self.event("case_4_event_014", "HIGH RISK"))
        self.assertEqual(result.risk_level, "HIGH RISK")
        self.assertTrue(any("multiple affected vitals" in p for p in result.pattern_identified))
        self.assertIn("HR", result.trend_metrics)

    def test_missing_window_returns_traceable_partial_event(self):
        self.agent.data_loader = lambda _: sample_frame().iloc[0:0]
        result = self.agent.process_event(self.event())
        self.assertEqual(result.analysis_status, "partial_analysis")
        self.assertEqual(result.event_id, "case_4_event_013")
        self.assertTrue(result.data_quality_flag)

    def test_malformed_input_returns_partial_event_without_crashing(self):
        result = self.agent.process_event(object())
        self.assertEqual(result.analysis_status, "partial_analysis")
        self.assertEqual(result.event_id, "unknown_risk_event")


if __name__ == "__main__":
    unittest.main()
