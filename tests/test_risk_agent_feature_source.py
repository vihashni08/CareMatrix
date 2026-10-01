"""Tests verifying the Risk Agent feature_source metadata tracking.

Verifies:
  - feature_source == "window" when a valid >=5 sample window DataFrame is supplied.
  - feature_source == "approximated_fallback" when window_df is None or has <5 samples.
  - feature_source is preserved in RiskDecisionEvent.metadata.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import MonitoringEvent
from risk_agent.preprocessing import construct_features, gather_patient_context
from risk_agent.risk_agent import RiskAgent


class TestRiskAgentFeatureSource(unittest.TestCase):
    """Test suite verifying feature source tagging and propagation."""

    def setUp(self):
        self.event_queue = EventQueue()
        self.risk_agent = RiskAgent(event_queue=self.event_queue)

        self.event = MonitoringEvent(
            patient_id=242,
            event_id="EVT_TEST_001",
            timestamp=100.0,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR", "MAP"],
            current_values={"HR": 95.0, "MAP": 62.0},
            baseline_values={"HR": 72.0, "MAP": 85.0},
            deviation_values={"HR": 23.0, "MAP": -23.0},
            trends={"HR": "increasing", "MAP": "decreasing"},
            signal_quality={"HR": "good", "MAP": "good"},
            persistence_duration=15,
            recommended_action="assess_risk",
            vital_summary={
                "HR": {"initial_value": 72.0, "minimum_value": 70.0, "maximum_value": 98.0},
                "MAP": {"initial_value": 85.0, "minimum_value": 60.0, "maximum_value": 88.0},
            },
        )

    def tearDown(self):
        self.risk_agent.stop()
        self.event_queue.shutdown()

    def test_feature_source_approximated_fallback_when_window_none(self):
        """When window_df is None, feature_source must be 'approximated_fallback'."""
        context = gather_patient_context(242)
        feats, src = construct_features(
            self.event,
            context,
            self.risk_agent.feature_columns,
            window_df=None,
            return_source=True,
        )
        self.assertEqual(src, "approximated_fallback")

        decision = self.risk_agent.process_event(self.event, window_df=None)
        self.assertEqual(decision.metadata.get("feature_source"), "approximated_fallback")

    def test_feature_source_approximated_fallback_when_window_too_short(self):
        """When window_df has fewer than 5 samples, fall back to 'approximated_fallback'."""
        short_df = pd.DataFrame({
            "Time": [0, 5, 10],
            "HR": [70.0, 72.0, 75.0],
            "MAP": [85.0, 84.0, 82.0],
        })
        context = gather_patient_context(242)
        feats, src = construct_features(
            self.event,
            context,
            self.risk_agent.feature_columns,
            window_df=short_df,
            min_window_samples=5,
            return_source=True,
        )
        self.assertEqual(src, "approximated_fallback")

    def test_feature_source_window_when_valid_window_supplied(self):
        """When window_df has >= 5 samples, feature_source must be 'window'."""
        valid_df = pd.DataFrame({
            "Time": np.arange(10) * 5.0,
            "HR": [70.0, 72.0, 74.0, 78.0, 82.0, 85.0, 88.0, 90.0, 92.0, 95.0],
            "MAP": [85.0, 83.0, 80.0, 78.0, 75.0, 72.0, 70.0, 68.0, 65.0, 62.0],
            "SpO2": [98.0] * 10,
            "RR": [14.0] * 10,
            "SBP": [120.0] * 10,
            "DBP": [80.0] * 10,
            "BT": [36.8] * 10,
        })
        context = gather_patient_context(242)
        feats, src = construct_features(
            self.event,
            context,
            self.risk_agent.feature_columns,
            window_df=valid_df,
            min_window_samples=5,
            return_source=True,
        )
        self.assertEqual(src, "window")

        decision = self.risk_agent.process_event(self.event, window_df=valid_df)
        self.assertEqual(decision.metadata.get("feature_source"), "window")


if __name__ == "__main__":
    unittest.main()
