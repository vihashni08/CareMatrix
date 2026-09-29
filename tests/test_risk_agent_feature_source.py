"""Integration tests for Risk Agent live feature source and window extraction.

Verifies that in the live event-driven path:
1. RiskAgent receives the real rolling buffered window from MonitoringAgent/Runtime.
2. RiskDecisionEvent.metadata records feature_source == "window" once the buffer has sufficient samples.
3. The resulting probability matches the probability computed directly on the buffered window.
4. When buffer has fewer than MIN_WINDOW_SAMPLES (or no buffer exists), the agent cleanly falls back
   to approximated statistics and records feature_source == "approximated_fallback".
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import pandas as pd

from carematrix_runtime import CareMatrixRuntime
from communication.events import MonitoringEvent
from data.adapters.vitaldb_adapter import VitalDBAdapter
from risk_agent.model import RiskModelTool
from risk_agent.preprocessing import (
    MIN_WINDOW_SAMPLES,
    construct_features,
    extract_window_features,
    gather_patient_context,
)
from risk_agent.risk_agent import RiskAgent


class TestRiskAgentFeatureSource(unittest.TestCase):
    def test_vitaldb_case4_live_runtime_uses_window_feature_source(self):
        """Replay VitalDB Case 4 (samples 800-899) through runtime worker loop.

        Asserts that the live worker loop feeds the real buffered window into RiskAgent,
        tags feature_source == 'window', and matches direct window feature extraction probability.
        """
        adapter = VitalDBAdapter()
        samples = adapter.load_case(4)[800:900]

        runtime = CareMatrixRuntime(stream_interval_seconds=0.001, verbose=False)
        runtime.monitoring_agent.baseline_window = 20
        runtime.monitoring_agent.persistence_duration = 3
        runtime.state_manager.get_or_create(4, name="VitalDB-Case-4")

        runtime.start()

        for s in samples:
            d = s.to_monitoring_sample()
            d["case_id"] = 4
            d["patient_id"] = 4
            runtime.state_manager.record_observation(d)
            runtime.monitoring_agent.observe(d)
            time.sleep(0.001)

        # Wait for asynchronous worker loop to process escalation and publish decision
        deadline = time.time() + 4.0
        r_hist = []
        while time.time() < deadline:
            r_hist = runtime.event_queue.get_history("risk_decisions")
            if len(r_hist) > 0:
                break
            time.sleep(0.05)

        # Fetch the buffered window used at escalation
        buffered_window = runtime.provide_analysis_window(4)
        runtime.stop()

        self.assertGreater(len(r_hist), 0, "Expected at least one RiskDecisionEvent to be published")
        decision_event = r_hist[0]

        # Verify feature_source is "window"
        self.assertEqual(
            decision_event.metadata.get("feature_source"),
            "window",
            "Expected feature_source to be 'window' when buffer has >= 5 samples",
        )

        # Compute probability directly from buffered window
        model_tool = RiskModelTool()
        context = gather_patient_context(4)
        features = {col: np.nan for col in model_tool.feature_columns}
        features.update(extract_window_features(buffered_window))
        features.update(context)
        direct_probability = model_tool.predict_proba(features)

        # Verify probability matches within small tolerance
        self.assertAlmostEqual(
            decision_event.risk_probability,
            direct_probability,
            delta=0.02,
            msg=f"Live probability ({decision_event.risk_probability}) did not match direct window probability ({direct_probability})",
        )

    def test_cold_start_fallback_when_samples_below_threshold(self):
        """Verify that when buffer has < MIN_WINDOW_SAMPLES, fallback approximation is used."""
        # 3 samples (< MIN_WINDOW_SAMPLES = 5)
        small_window = pd.DataFrame({
            "Time": [0.0, 5.0, 10.0],
            "HR": [75.0, 80.0, 85.0],
            "MAP": [85.0, 82.0, 80.0],
            "SpO2": [98.0, 97.0, 96.0],
            "RR": [14.0, 15.0, 16.0],
        })

        mock_event = MonitoringEvent(
            patient_id=101,
            event_id="test_cold_start_001",
            timestamp=10.0,
            event_type="alert_started",
            severity="moderate",
            affected_vitals=["HR"],
            current_values={"HR": 85.0},
            baseline_values={"HR": 75.0},
            deviation_values={"HR": 10.0},
            trends={"HR": "increasing"},
            signal_quality={"HR": "good"},
            persistence_duration=3,
            recommended_action="Investigate HR deviation",
            vital_summary={"HR": {"initial_value": 75.0, "minimum_value": 75.0, "maximum_value": 85.0}},
        )

        agent = RiskAgent()
        context = {"age": 60.0, "sex": 1.0, "bmi": 24.5, "asa": 2.0, "emop": 0.0}

        # Passing small window (< 5 samples)
        features, source = agent.construct_features(
            event=mock_event,
            patient_context=context,
            window_df=small_window,
            return_source=True,
        )
        self.assertEqual(source, "approximated_fallback")

        # Passing None
        features_none, source_none = agent.construct_features(
            event=mock_event,
            patient_context=context,
            window_df=None,
            return_source=True,
        )
        self.assertEqual(source_none, "approximated_fallback")

        # Process event with small window
        decision_event = agent.process_event(mock_event, window_df=small_window)
        self.assertEqual(decision_event.metadata.get("feature_source"), "approximated_fallback")

    def test_runtime_wires_provide_analysis_window_to_data_analysis_and_risk(self):
        """Verify CareMatrixRuntime initializes both RiskAgent and DataAnalysisAgent with provide_analysis_window."""
        runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
        self.assertEqual(runtime.risk_agent.window_provider, runtime.provide_analysis_window)
        self.assertEqual(runtime.data_analysis_agent.data_loader, runtime.provide_analysis_window)
        runtime.stop()


if __name__ == "__main__":
    unittest.main()
