"""Unit tests for dataset adapters, common patient schema, and observation normalizer."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from data.adapters import BaseDatasetAdapter, MIMICIVAdapter, VitalDBAdapter
from data.normalizer import ObservationNormalizer
from data.schemas import CommonPatientObservation, PatientDatasetMetadata
from carematrix_runtime import PatientStreamReplayer


class TestDatasetAdapters(unittest.TestCase):
    def test_1_common_patient_observation_schema(self):
        """CommonPatientObservation converts properly to standard monitoring dictionary."""
        obs = CommonPatientObservation(
            timestamp=15.0,
            HR=82.0,
            MAP=75.0,
            SpO2=98.0,
            RR=16.0,
            BT=37.0,
            source="vitaldb",
            patient_id=4,
        )
        sample = obs.to_monitoring_sample()
        self.assertEqual(sample["timestamp"], 15.0)
        self.assertEqual(sample["HR"], 82.0)
        self.assertEqual(sample["MAP"], 75.0)
        self.assertEqual(sample["SpO2"], 98.0)
        self.assertEqual(sample["RR"], 16.0)
        self.assertEqual(sample["BT"], 37.0)
        self.assertEqual(sample["patient_id"], 4)
        self.assertEqual(sample["source"], "vitaldb")

    def test_2_normalizer_temperature_conversion(self):
        """Fahrenheit values > 50°F convert to Celsius; valid Celsius values are preserved."""
        # 98.6°F -> 37.0°C
        self.assertAlmostEqual(ObservationNormalizer.normalize_temperature(98.6), 37.0, places=1)
        # 36.8°C -> 36.8°C
        self.assertAlmostEqual(ObservationNormalizer.normalize_temperature(36.8), 36.8, places=1)
        # None -> None
        self.assertIsNone(ObservationNormalizer.normalize_temperature(None))

    def test_3_normalizer_spo2_scaling(self):
        """Fractional SpO2 (0.0 - 1.0) scales to percentage (0 - 100)."""
        self.assertEqual(ObservationNormalizer.normalize_spo2(0.96), 96.0)
        self.assertEqual(ObservationNormalizer.normalize_spo2(98.0), 98.0)
        self.assertIsNone(ObservationNormalizer.normalize_spo2(None))

    def test_4_normalizer_blood_pressure_map_estimation(self):
        """When MAP is missing but SBP/DBP provided, estimated MAP = DBP + (SBP - DBP)/3."""
        # SBP=120, DBP=80 -> MAP ~ 80 + 40/3 = 93.3
        m, s, d = ObservationNormalizer.normalize_blood_pressure(None, 120.0, 80.0)
        self.assertAlmostEqual(m, 93.3, places=1)
        self.assertEqual(s, 120.0)
        self.assertEqual(d, 80.0)

        # Extreme impossible values filtered out
        m_bad, s_bad, d_bad = ObservationNormalizer.normalize_blood_pressure(500.0, -10.0, 300.0)
        self.assertIsNone(m_bad)
        self.assertIsNone(s_bad)
        self.assertIsNone(d_bad)

    def test_5_vitaldb_adapter_loading(self):
        """VitalDB adapter loads synthetic case 0 and real cases into common observations."""
        adapter = VitalDBAdapter()
        cases = adapter.list_cases()
        self.assertIn(0, cases)

        observations = adapter.load_case(0)
        self.assertGreater(len(observations), 10)
        self.assertIsInstance(observations[0], CommonPatientObservation)
        self.assertEqual(observations[0].patient_id, 0)
        self.assertIn("HR", [k for k in ("HR", "MAP") if getattr(observations[0], k) is not None])

        # Metadata extraction
        meta = adapter.get_metadata(0)
        self.assertIsInstance(meta, PatientDatasetMetadata)
        self.assertEqual(meta.patient_id, 0)
        self.assertIn("HR", meta.available_vitals)

    def test_6_mimic_adapter_demo_cohort(self):
        """MIMIC-IV adapter provides de-identified demo cases with realistic trajectories."""
        adapter = MIMICIVAdapter()
        cases = adapter.list_cases()
        self.assertGreaterEqual(len(cases), 5)
        self.assertIn(1001, cases)

        obs_1001 = adapter.load_case(1001)
        self.assertEqual(len(obs_1001), 100)
        self.assertIsInstance(obs_1001[0], CommonPatientObservation)
        self.assertEqual(obs_1001[0].source, "MIMIC-IV_Demo_Cohort")
        self.assertEqual(obs_1001[0].patient_id, 1001)

    def test_7_mimic_missing_vital_preservation_without_fabrication(self):
        """MIMIC-IV adapter preserves unmeasured variables as NaN without fabricating values."""
        adapter = MIMICIVAdapter()
        # Case 1004 has sensor noise / dropouts
        obs_1004 = adapter.load_case(1004)
        sample_dict_45 = obs_1004[50].to_monitoring_sample()

        # SpO2 should be NaN during dropout window (indices 45-65 in case 1004)
        self.assertTrue(np.isnan(sample_dict_45["SpO2"]))

    def test_8_mimic_data_path_environment_variable(self):
        """MIMIC-IV adapter respects MIMIC_DATA_PATH environment variable or constructor arg."""
        custom_path = "/tmp/carematrix_test_mimic_dir"
        adapter = MIMICIVAdapter(data_path=custom_path)
        self.assertEqual(adapter.data_dir, Path(custom_path))

        with unittest.mock.patch.dict(os.environ, {"MIMIC_DATA_PATH": "/tmp/env_mimic_dir"}):
            adapter_env = MIMICIVAdapter()
            self.assertEqual(adapter_env.data_dir, Path("/tmp/env_mimic_dir"))

    def test_9_replayer_adapter_integration(self):
        """PatientStreamReplayer streams normalized samples directly from an adapter."""
        adapter = MIMICIVAdapter()
        replayer = PatientStreamReplayer(case_id=1001, adapter=adapter, delay_seconds=0.0)

        stream = replayer.stream(max_samples=10)
        samples = list(stream)
        self.assertEqual(len(samples), 10)
        self.assertIn("HR", samples[0])
        self.assertIn("MAP", samples[0])
        self.assertEqual(samples[0]["timestamp"], 0.0)
        self.assertEqual(samples[1]["timestamp"], 1.0)


if __name__ == "__main__":
    unittest.main()

