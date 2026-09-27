"""Unit tests for the CareMatrix Flask API and SSE Server."""

from __future__ import annotations

import json
import time
import unittest

from carematrix_runtime.patient_stream import PatientScenario
from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import create_app


class TestServerAPI(unittest.TestCase):
    def setUp(self):
        self.runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
        self.runtime.start()
        self.app = create_app(runtime=self.runtime)
        self.client = self.app.test_client()

    def tearDown(self):
        self.runtime.stop()

    def test_dashboard_route_serves_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"CareMatrix", res.data)
        self.assertIn(b"root", res.data)

    def test_get_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)
        self.assertIn("supervisor", data)
        self.assertIn("runtime_metrics", data)
        self.assertTrue(data["all_healthy"])
        self.assertEqual(data["total_agents"], 5)

    def test_get_patients_endpoint(self):
        res = self.client.get("/api/patients")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("patients", data)
        self.assertGreaterEqual(len(data["patients"]), 3)
        self.assertIn("patient_id", data["patients"][0])

    def test_get_patient_detail_endpoint(self):
        res = self.client.get("/api/patients/101")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["patient_id"], 101)
        self.assertIn("status", data)
        self.assertIn("agent_pipeline", data)

    def test_get_patient_vitals_history(self):
        # Generate some observations first
        self.runtime.step()
        self.runtime.step()

        res = self.client.get("/api/patients/101/vitals?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["patient_id"], 101)
        self.assertIsInstance(data["vitals"], list)
        self.assertGreaterEqual(len(data["vitals"]), 2)

    def test_trigger_scenario_endpoint(self):
        payload = {"patient_id": 101, "scenario": "GRADUAL_DETERIORATION"}
        res = self.client.post("/api/scenarios/trigger", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["scenario"], PatientScenario.GRADUAL_DETERIORATION.value)

        # Verify invalid scenario rejected
        bad_res = self.client.post("/api/scenarios/trigger", json={"patient_id": 101, "scenario": "INVALID_SCENARIO"})
        self.assertEqual(bad_res.status_code, 400)

    def test_alerts_lifecycle_endpoints(self):
        # Trigger an alert manually
        from communication.events import MonitoringEvent
        event = MonitoringEvent(
            patient_id=101,
            event_id="test_alert_evt",
            timestamp=time.time(),
            event_type="alert_started",
            severity="severe",
            affected_vitals=["hr", "map"],
            current_values={"hr": 125, "map": 55},
            baseline_values={"hr": 72, "map": 85},
            deviation_values={"hr": 53, "map": -30},
            trends={"hr": "increasing", "map": "decreasing"},
            signal_quality={"hr": "good", "map": "good"},
            persistence_duration=5,
            recommended_action="Notify physician",
        )
        self.runtime.alert_manager.handle_monitoring_event(event)

        # GET /api/alerts
        res = self.client.get("/api/alerts?patient_id=101")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["active_count"], 1)
        alert_id = data["active"][0]["alert_id"]

        # POST /api/alerts/<id>/acknowledge
        ack_res = self.client.post(
            f"/api/alerts/{alert_id}/acknowledge",
            json={"clinician_id": "Dr. House"}
        )
        self.assertEqual(ack_res.status_code, 200)
        ack_data = ack_res.get_json()
        self.assertTrue(ack_data["success"])
        self.assertEqual(ack_data["alert"]["state"], "ACKNOWLEDGED")
        self.assertEqual(ack_data["alert"]["acknowledged_by"], "Dr. House")

        # POST /api/alerts/<id>/resolve
        res_res = self.client.post(
            f"/api/alerts/{alert_id}/resolve",
            json={"reason": "Patient stable"}
        )
        self.assertEqual(res_res.status_code, 200)

        # Verify active count is now 0
        res2 = self.client.get("/api/alerts?patient_id=101")
        self.assertEqual(res2.get_json()["active_count"], 0)


if __name__ == "__main__":
    unittest.main()
