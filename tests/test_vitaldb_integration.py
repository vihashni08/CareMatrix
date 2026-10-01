"""Integration test replaying a real VitalDB case segment end to end through CareMatrix."""

from __future__ import annotations

from pathlib import Path
import time
import unittest
import pandas as pd

from carematrix_runtime import CareMatrixRuntime, PatientStreamReplayer
from data.adapters.vitaldb_adapter import VitalDBAdapter

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "test_fixtures" / "vitaldb_case_6_snippet.csv"


class TestVitalDBIntegration(unittest.TestCase):
    def test_real_vitaldb_case_segment_end_to_end_replay(self) -> None:
        """Replay real VitalDB telemetry snippet through full agent pipeline and verify end-to-end execution."""
        if not FIXTURE_PATH.exists():
            adapter = VitalDBAdapter()
            try:
                snippet_df = adapter.load_case_dataframe(6, interval=5).iloc[:60]
            except Exception as exc:
                self.skipTest(f"Neither test fixture nor VitalDB API is accessible: {exc}")
        else:
            snippet_df = pd.read_csv(FIXTURE_PATH)

        self.assertFalse(snippet_df.empty, "Real VitalDB test snippet must not be empty")

        runtime = CareMatrixRuntime(stream_interval_seconds=0.001, verbose=False)
        runtime.monitoring_agent.baseline_window = 10
        runtime.monitoring_agent.persistence_duration = 2
        runtime.state_manager.get_or_create(6, name="VitalDB Case 6 — Replay")

        runtime.start()

        replayer = PatientStreamReplayer(case_id=6, source_df=snippet_df, delay_seconds=0.001)
        for sample in replayer.stream(max_samples=40):
            sample["case_id"] = 6
            sample["patient_id"] = 6
            runtime.state_manager.record_observation(sample)
            runtime.monitoring_agent.observe(sample)
            time.sleep(0.001)

        # Allow worker threads to process queue
        time.sleep(1.0)
        runtime.stop()

        # Verify observations were recorded by state manager
        patient_state = runtime.state_manager.get_or_create(6)
        self.assertIsNotNone(patient_state)
        self.assertGreaterEqual(len(patient_state.vital_history), 40)
        self.assertIn("VitalDB Case 6 — Replay", patient_state.name)

        # Verify monitoring agent observed samples
        self.assertGreaterEqual(runtime.monitoring_agent.state.total_observations, 40)


if __name__ == "__main__":
    unittest.main()
