"""Tests for dataset-backed patient loading and API endpoints.

Tests verify that:
  - CareMatrixRuntime.add_dataset_patient() streams real case observations through
    the pipeline and produces valid MonitoringEvents.
  - GET /api/datasets returns case IDs from both VitalDB and MIMIC adapters.
  - POST /api/patients/dataset correctly registers a new patient and returns
    the expected data_source field.
  - GET /api/patients/<id> returns data_source for all patients (simulated and real).
"""

from __future__ import annotations

import json
import threading
import time
import unittest

from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import create_app
from data.adapters.mimic_adapter import MIMICIVAdapter
from data.adapters.vitaldb_adapter import VitalDBAdapter


class TestAddDatasetPatient(unittest.TestCase):
    """Unit tests for CareMatrixRuntime.add_dataset_patient()."""

    def setUp(self):
        self.rt = CareMatrixRuntime(stream_interval_seconds=0.05, verbose=False)
        self.rt.start()

    def tearDown(self):
        self.rt.stop()

    def test_add_vitaldb_patient_returns_correct_shape(self):
        """add_dataset_patient() must return patient_id, case_id, data_source."""
        info = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        self.assertIn("patient_id", info)
        self.assertIn("case_id", info)
        self.assertIn("data_source", info)
        self.assertIn("adapter", info)
        self.assertEqual(info["case_id"], 4)
        self.assertEqual(info["adapter"], "vitaldb")
        self.assertIn("VitalDB", info["data_source"])

    def test_add_mimic_patient_returns_correct_shape(self):
        """add_dataset_patient() for MIMIC must label data_source correctly."""
        info = self.rt.add_dataset_patient(case_id=1001, adapter_name="mimic")
        self.assertIn("MIMIC", info["data_source"])
        self.assertEqual(info["case_id"], 1001)

    def test_new_patient_registered_in_state_manager(self):
        """The new patient_id must appear in the state manager."""
        info = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        pid = info["patient_id"]
        # Wait briefly for at least one observation to arrive
        time.sleep(0.3)
        detail = self.rt.state_manager.get_patient_detail(pid)
        self.assertIsNotNone(detail, f"Patient {pid} not found in state_manager")

    def test_observations_flow_into_state_manager(self):
        """After a short delay, the dataset patient should have received observations."""
        info = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        pid = info["patient_id"]
        time.sleep(0.5)
        history = self.rt.state_manager.get_patient_vital_history(pid)
        self.assertGreater(len(history), 0, "No observations received for dataset-backed patient")

    def test_observation_shape_has_uppercase_vitals(self):
        """Observations from the replayer must have uppercase vital keys (HR, MAP, SpO2, RR)."""
        info = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        pid = info["patient_id"]
        time.sleep(0.5)
        history = self.rt.state_manager.get_patient_vital_history(pid)
        self.assertTrue(len(history) > 0)
        obs = history[0]
        # At least one of the key vital signs must be present in uppercase
        has_vitals = any(k in obs for k in ("HR", "MAP", "SpO2", "RR"))
        self.assertTrue(has_vitals, f"Observation missing uppercase vital keys: {list(obs.keys())}")

    def test_patient_id_collision_avoidance(self):
        """Each call to add_dataset_patient() for DIFFERENT cases must return a unique patient_id >= 200."""
        info1 = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        info2 = self.rt.add_dataset_patient(case_id=0, adapter_name="vitaldb")
        self.assertNotEqual(info1["patient_id"], info2["patient_id"])
        self.assertGreaterEqual(info1["patient_id"], 200)
        self.assertGreaterEqual(info2["patient_id"], 200)

    def test_add_dataset_patient_idempotent(self):
        """Loading the same adapter+case_id twice must reuse the existing patient_id and spawn no extra thread."""
        initial_threads = [t.name for t in threading.enumerate() if "CareMatrix-Dataset-" in t.name]
        info1 = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        after_first = [t.name for t in threading.enumerate() if "CareMatrix-Dataset-" in t.name]
        self.assertEqual(len(after_first), len(initial_threads) + 1)

        info2 = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        after_second = [t.name for t in threading.enumerate() if "CareMatrix-Dataset-" in t.name]

        self.assertEqual(info1["patient_id"], info2["patient_id"])
        self.assertEqual(len(after_second), len(after_first), "A second thread was erroneously spawned for the same case!")

    def test_stream_error_handling_marks_data_source(self):
        """If a dataset stream encounters an unhandled exception, data_source reflects the error."""
        info = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        pid = info["patient_id"]

        class BrokenReplayer:
            def stream(self):
                raise RuntimeError("Telemetry sensor hardware disconnected")
            def stop(self):
                pass

        with self.rt._dataset_lock:
            self.rt._dataset_sources[pid]["replayer"] = BrokenReplayer()

        # Execute stream loop to trigger exception handling
        self.rt._dataset_stream_loop(pid)

        source = self.rt.get_patient_data_source(pid)
        self.assertTrue(source.endswith("(stream error)"), f"Expected '(stream error)' in {source}")

    def test_get_patient_data_source_simulated(self):
        """Simulated beds 101-103 must return 'simulated' from get_patient_data_source."""
        src = self.rt.get_patient_data_source(101)
        self.assertEqual(src, "simulated")

    def test_get_patient_data_source_dataset(self):
        """Dataset-backed patients must return a non-'simulated' data_source label."""
        info = self.rt.add_dataset_patient(case_id=4, adapter_name="vitaldb")
        src = self.rt.get_patient_data_source(info["patient_id"])
        self.assertNotEqual(src, "simulated")
        self.assertIn("VitalDB", src)


class TestAdapterListCases(unittest.TestCase):
    """Unit tests for adapter list_cases() — used by GET /api/datasets."""

    def test_vitaldb_list_cases_returns_list(self):
        adapter = VitalDBAdapter()
        cases = adapter.list_cases()
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)
        # Case 4 is always included as the default real case
        self.assertIn(4, cases)

    def test_mimic_list_cases_returns_list(self):
        adapter = MIMICIVAdapter()
        cases = adapter.list_cases()
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)


class TestDatasetAPIEndpoints(unittest.TestCase):
    """Integration tests for the three new API endpoints."""

    def setUp(self):
        self.rt = CareMatrixRuntime(stream_interval_seconds=0.05, verbose=False)
        self.rt.start()
        self.app = create_app(runtime=self.rt)
        self.client = self.app.test_client()

    def tearDown(self):
        self.rt.stop()

    def test_get_datasets_returns_vitaldb_only(self):
        """GET /api/datasets exposes only the supported VitalDB adapter."""
        resp = self.client.get("/api/datasets")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("vitaldb", data)
        self.assertNotIn("mimic", data)
        self.assertIsInstance(data["vitaldb"], list)
        self.assertGreater(len(data["vitaldb"]), 0)

    def test_post_dataset_patient_vitaldb(self):
        """POST /api/patients/dataset must create a new patient and return 201 with data_source."""
        resp = self.client.post(
            "/api/patients/dataset",
            data=json.dumps({"case_id": 4, "adapter": "vitaldb"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        data = json.loads(resp.data)
        self.assertIn("data_source", data)
        self.assertIn("VitalDB", data["data_source"])

    def test_post_dataset_patient_rejects_unsupported_adapter(self):
        """POST /api/patients/dataset rejects adapters no longer exposed by the API."""
        resp = self.client.post(
            "/api/patients/dataset",
            data=json.dumps({"case_id": 1001, "adapter": "mimic"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("Unsupported adapter", data["error"])

    def test_post_dataset_patient_missing_case_id(self):
        """POST /api/patients/dataset without case_id must return 400."""
        resp = self.client.post(
            "/api/patients/dataset",
            data=json.dumps({"adapter": "vitaldb"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_get_patient_detail_includes_data_source_simulated(self):
        """GET /api/patients/101 (simulated) must return data_source='simulated'."""
        resp = self.client.get("/api/patients/101")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("data_source", data)
        self.assertEqual(data["data_source"], "simulated")

    def test_get_patient_detail_includes_data_source_dataset(self):
        """GET /api/patients/<dataset_pid> must return a non-'simulated' data_source."""
        # First create a dataset patient
        post_resp = self.client.post(
            "/api/patients/dataset",
            data=json.dumps({"case_id": 4, "adapter": "vitaldb"}),
            content_type="application/json",
        )
        self.assertEqual(post_resp.status_code, 201)
        created = json.loads(post_resp.data)
        pid = created.get("patient_id")
        self.assertIsNotNone(pid)

        # Now retrieve it
        get_resp = self.client.get(f"/api/patients/{pid}")
        self.assertEqual(get_resp.status_code, 200)
        data = json.loads(get_resp.data)
        self.assertIn("data_source", data)
        self.assertNotEqual(data["data_source"], "simulated")

    def test_list_patients_includes_data_source(self):
        """GET /api/patients must include data_source field for every patient."""
        resp = self.client.get("/api/patients")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn("patients", data)
        for p in data["patients"]:
            self.assertIn("data_source", p, f"Patient {p.get('patient_id')} missing data_source")


if __name__ == "__main__":
    unittest.main()
