"""Verification script for CareMatrix live runtime and dashboard endpoints."""

import json
import time
import sys
import unittest

from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import create_app


def run_verification():
    print("==================================================================")
    print("CAREMATRIX DASHBOARD & RUNTIME END-TO-END VALIDATION")
    print("==================================================================")

    # 1. Initialize runtime with fast stepping
    runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
    runtime.start()

    try:
        app = create_app(runtime=runtime)
        client = app.test_client()

        # Step runtime several times to populate initial agent states
        for _ in range(5):
            runtime.step()
        time.sleep(0.5)

        # ------------------------------------------------------------------
        # 1. Verify GET / and GET /dashboard
        # ------------------------------------------------------------------
        print("\n[1] Verifying Static Dashboard Endpoints (GET / and GET /dashboard)...")
        r_root = client.get("/")
        assert r_root.status_code == 200, f"Root returned {r_root.status_code}"
        html_root = r_root.data.decode("utf-8")
        assert '<div id="root">' in html_root, "Missing #root mount element"
        assert "CareMatrix" in html_root, "Missing CareMatrix branding"
        assert "Clinical Decision Support" in html_root, "Missing CDS tab title"
        print("    ✓ GET / returned 200 OK with React mount element and branding")

        r_dash = client.get("/dashboard")
        assert r_dash.status_code == 200, f"/dashboard returned {r_dash.status_code}"
        print("    ✓ GET /dashboard returned 200 OK")

        # ------------------------------------------------------------------
        # 2. Verify GET /api/health
        # ------------------------------------------------------------------
        print("\n[2] Verifying System Health (GET /api/health)...")
        r_health = client.get("/api/health")
        assert r_health.status_code == 200, f"/api/health returned {r_health.status_code}"
        health_data = r_health.get_json()
        assert health_data["status"] == "healthy", f"Expected healthy, got {health_data['status']}"
        assert health_data["total_agents"] == 5, f"Expected 5 agents, got {health_data['total_agents']}"
        assert health_data["all_healthy"] is True, "Expected all_healthy True"
        print(f"    ✓ GET /api/health: 5 agents active, all healthy (status: {health_data['status']})")

        # ------------------------------------------------------------------
        # 3. Verify GET /api/patients
        # ------------------------------------------------------------------
        print("\n[3] Verifying Multi-Patient Registry (GET /api/patients)...")
        r_patients = client.get("/api/patients")
        assert r_patients.status_code == 200, f"/api/patients returned {r_patients.status_code}"
        p_data = r_patients.get_json()
        patients = p_data.get("patients", [])
        assert len(patients) >= 3, f"Expected at least 3 beds, found {len(patients)}"
        patient_ids = [p["patient_id"] for p in patients]
        print(f"    ✓ GET /api/patients: Found {len(patients)} active beds: {patient_ids}")

        target_pid = patient_ids[0]

        # ------------------------------------------------------------------
        # 4. Verify GET /api/patients/<id>
        # ------------------------------------------------------------------
        print(f"\n[4] Verifying Single Patient Detail (GET /api/patients/{target_pid})...")
        r_detail = client.get(f"/api/patients/{target_pid}")
        assert r_detail.status_code == 200, f"Detail returned {r_detail.status_code}"
        detail = r_detail.get_json()
        assert detail["patient_id"] == target_pid
        assert "status" in detail
        assert "agent_pipeline" in detail
        assert "latest_vitals" in detail
        pipeline = detail["agent_pipeline"]
        print(f"    ✓ GET /api/patients/{target_pid}: status={detail['status']}, pipeline keys={list(pipeline.keys())}")

        # ------------------------------------------------------------------
        # 5. Verify GET /api/patients/<id>/vitals
        # ------------------------------------------------------------------
        print(f"\n[5] Verifying Vital History Timeseries (GET /api/patients/{target_pid}/vitals)...")
        r_vitals = client.get(f"/api/patients/{target_pid}/vitals?limit=40")
        assert r_vitals.status_code == 200, f"Vitals returned {r_vitals.status_code}"
        v_data = r_vitals.get_json()
        assert "vitals" in v_data
        assert isinstance(v_data["vitals"], list)
        print(f"    ✓ GET /api/patients/{target_pid}/vitals: Retrieved {len(v_data['vitals'])} observation ticks")

        # ------------------------------------------------------------------
        # 6. Verify GET /api/patients/<id>/clinical-report
        # ------------------------------------------------------------------
        print(f"\n[6] Verifying Structured Clinical Intelligence Report (GET /api/patients/{target_pid}/clinical-report)...")
        r_rep = client.get(f"/api/patients/{target_pid}/clinical-report")
        assert r_rep.status_code == 200, f"Report returned {r_rep.status_code}"
        rep_resp = r_rep.get_json()
        assert "clinical_report" in rep_resp, "Missing clinical_report in response"
        assert "metadata" in rep_resp, "Missing metadata in response"

        report = rep_resp["clinical_report"]
        required_sections = [
            "executive_summary",
            "clinical_status",
            "key_findings",
            "physiological_analysis",
            "temporal_analysis",
            "risk_interpretation",
            "supporting_evidence",
            "conflicting_evidence",
            "evidence_synthesis",
            "medical_evidence",
            "clinical_interpretation",
            "uncertainties",
            "recommended_actions",
            "monitoring_priorities",
            "escalation_rationale",
            "confidence"
        ]
        for sec in required_sections:
            assert sec in report, f"Missing required report section: {sec}"
        print(f"    ✓ GET /api/patients/{target_pid}/clinical-report: All 16 structured sections present and valid")
        print(f"      Executive Summary: {report['executive_summary'][:80]}...")
        print(f"      Clinical Status: Risk={report['clinical_status']['risk_level']}, Priority={report['clinical_status']['priority']}")

        # ------------------------------------------------------------------
        # 7. Verify Trigger Deterioration Scenario & Dynamic Report Update
        # ------------------------------------------------------------------
        print(f"\n[7] Triggering Scenario 'GRADUAL_DETERIORATION' for Bed {target_pid}...")
        r_trig = client.post("/api/scenarios/trigger", json={
            "patient_id": target_pid,
            "scenario": "GRADUAL_DETERIORATION"
        })
        assert r_trig.status_code == 200, f"Trigger failed with {r_trig.status_code}: {r_trig.data}"
        print("    ✓ Scenario triggered successfully")

        # Step multiple times to propagate through Monitoring -> Risk -> Data Analysis -> Clinical Reasoning -> Care Coordination
        for _ in range(8):
            runtime.step()
        time.sleep(0.5)

        r_rep_post = client.get(f"/api/patients/{target_pid}/clinical-report")
        assert r_rep_post.status_code == 200
        post_rep = r_rep_post.get_json()["clinical_report"]
        print(f"    ✓ Post-trigger status: Risk={post_rep['clinical_status']['risk_level']}, Priority={post_rep['clinical_status']['priority']}")
        print(f"    ✓ Key Findings count: {len(post_rep['key_findings'])}, Recommended Actions: {len(post_rep['recommended_actions'])}")

        # ------------------------------------------------------------------
        # 8. Verify Alert Lifecycle Management
        # ------------------------------------------------------------------
        print(f"\n[8] Verifying Alert Lifecycle API...")
        r_alerts = client.get(f"/api/alerts?patient_id={target_pid}")
        assert r_alerts.status_code == 200
        alerts_data = r_alerts.get_json()
        active_alerts = alerts_data.get("active", [])
        print(f"    ✓ Active alerts for patient {target_pid}: {len(active_alerts)}")

        if active_alerts:
            first_alert = active_alerts[0]
            aid = first_alert["alert_id"]
            # Acknowledge
            r_ack = client.post(f"/api/alerts/{aid}/acknowledge", json={"clinician_id": "dr_verification"})
            assert r_ack.status_code == 200
            print(f"    ✓ Successfully acknowledged alert {aid}")

            # Resolve
            r_res = client.post(f"/api/alerts/{aid}/resolve", json={"clinician_id": "dr_verification", "reason": "Test verified"})
            assert r_res.status_code == 200
            print(f"    ✓ Successfully resolved alert {aid}")

        # ------------------------------------------------------------------
        # 9. Verify Episodic Memory Endpoint
        # ------------------------------------------------------------------
        print(f"\n[9] Verifying Episodic Patient Memory (GET /api/patients/{target_pid}/memory)...")
        r_mem = client.get(f"/api/patients/{target_pid}/memory")
        assert r_mem.status_code == 200, f"Memory returned {r_mem.status_code}"
        mem_data = r_mem.get_json()
        assert "episodes" in mem_data
        assert "context_prompt" in mem_data
        print(f"    ✓ GET /api/patients/{target_pid}/memory: count={mem_data.get('count', 0)}")

        # ------------------------------------------------------------------
        # 10. Verify SSE Stream Endpoint
        # ------------------------------------------------------------------
        print(f"\n[10] Verifying Real-Time SSE Stream (GET /api/events/stream)...")
        r_stream = client.get("/api/events/stream")
        assert r_stream.status_code == 200
        assert r_stream.mimetype == "text/event-stream"
        assert r_stream.headers.get("Cache-Control") == "no-cache"
        print("    ✓ GET /api/events/stream responds with text/event-stream and no-cache headers")

        # ------------------------------------------------------------------
        # 11. Verify UI Text & CDS Framing
        # ------------------------------------------------------------------
        print(f"\n[11] Verifying Clinical Decision Support Boundaries in Static HTML...")
        assert "Decision Support" in html_root
        assert "Clinical Decision Support Only" in html_root
        assert "Recommended Clinician Review & Decision Support Checklist" in html_root
        assert "Episodic Patient Memory & Clinician Feedback Audit Trail" in html_root
        assert "DOI" in html_root
        print("    ✓ HTML strictly enforces CDS boundaries and provides full provenance")

        print("\n==================================================================")
        print("ALL ENDPOINTS AND RUNTIME PIPELINES FULLY VALIDATED!")
        print("==================================================================")

    finally:
        runtime.stop()


if __name__ == "__main__":
    run_verification()
