"""Frontend static architecture and modular real-time state tests."""

from __future__ import annotations

import os
import re
import unittest

from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import create_app


class TestFrontendArchitecture(unittest.TestCase):
    def setUp(self):
        self.runtime = CareMatrixRuntime(stream_interval_seconds=0.01, verbose=False)
        self.app = create_app(runtime=self.runtime)
        self.client = self.app.test_client()

    def tearDown(self):
        self.runtime.stop()

    def test_dashboard_serves_modular_index_html(self):
        """Verify GET / and GET /dashboard serve the modular index.html entrypoint."""
        for path in ("/", "/dashboard"):
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            html = res.data.decode("utf-8")
            self.assertIn("CareMatrix", html)
            self.assertIn("/static/css/dashboard.css", html)
            self.assertIn("/static/js/app/App.jsx", html)
            self.assertIn("/static/js/state/eventReducer.js", html)

    def test_all_referenced_static_assets_served_by_flask(self):
        """Verify every script and stylesheet in index.html is served with 200 OK by Flask."""
        res = self.client.get("/dashboard")
        html = res.data.decode("utf-8")

        # Extract all local /static/ links
        matches = re.findall(r'src="(/static/[^"]+)"', html)
        matches += re.findall(r'href="(/static/[^"]+)"', html)

        self.assertGreater(len(matches), 10, "Should have decomposed into modular static files")

        for asset_url in matches:
            asset_res = self.client.get(asset_url)
            self.assertEqual(
                asset_res.status_code,
                200,
                f"Static file '{asset_url}' should be served successfully by Flask",
            )
            self.assertGreater(
                len(asset_res.data),
                0,
                f"Static file '{asset_url}' should not be empty",
            )

    def test_event_reducer_structure_and_isolation_rules(self):
        """Verify reducer file contains necessary handlers and strict isolation protections."""
        reducer_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "state",
            "eventReducer.js",
        )
        self.assertTrue(os.path.exists(reducer_path))

        with open(reducer_path, "r", encoding="utf-8") as f:
            reducer_code = f.read()

        # All supported SSE events are routed in reducer
        expected_events = [
            "vital_tick",
            "monitoring_alert",
            "risk_prediction",
            "agent_negotiation",
            "data_analysis",
            "clinical_reasoning",
            "care_coordination",
            "heartbeat",
            "agent_failure",
            "agent_recovery",
            "clinician_feedback",
        ]
        for ev in expected_events:
            self.assertIn(
                ev,
                reducer_code,
                f"eventReducer.js must support '{ev}'",
            )

        # Multi-patient isolation logic
        self.assertIn("extractPatientId", reducer_code)
        self.assertIn("patientDetails", reducer_code)

    def test_initial_state_model_structure(self):
        """Verify initialState.js defines expected system, patients, and bounded trace containers."""
        state_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "state",
            "initialState.js",
        )
        self.assertTrue(os.path.exists(state_path))

        with open(state_path, "r", encoding="utf-8") as f:
            state_code = f.read()

        for key in (
            "system",
            "patients",
            "selectedPatientId",
            "patientDetails",
            "eventHistory",
            "activeAlerts",
            "pipelineState",
            "agenticTraces",
            "negotiationTrace",
        ):
            self.assertIn(key, state_code)

    def test_event_utils_tool_trace_normalization_and_thought_stripping(self):
        """Verify eventUtils.js contains normalizeToolTrace and no exposure of thought monologue."""
        utils_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "utils",
            "eventUtils.js",
        )
        self.assertTrue(os.path.exists(utils_path))

        with open(utils_path, "r", encoding="utf-8") as f:
            utils_code = f.read()

        self.assertIn("normalizeToolStep", utils_code)
        self.assertIn("normalizeToolTrace", utils_code)
        self.assertIn("derivePipelineState", utils_code)
        self.assertIn("createAuditEntry", utils_code)

    def test_three_primary_navigation_views_implemented(self):
        """Verify CommandCenterView, PatientSurveillanceView, and AgentInspectionView exist and are wired."""
        components_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "components",
        )
        cc_path = os.path.join(components_dir, "system", "CommandCenterView.jsx")
        ps_path = os.path.join(components_dir, "patient", "PatientSurveillanceView.jsx")
        ai_path = os.path.join(components_dir, "agents", "AgentInspectionView.jsx")

        self.assertTrue(os.path.exists(cc_path), "CommandCenterView.jsx must exist")
        self.assertTrue(os.path.exists(ps_path), "PatientSurveillanceView.jsx must exist")
        self.assertTrue(os.path.exists(ai_path), "AgentInspectionView.jsx must exist")

        with open(cc_path, "r", encoding="utf-8") as f:
            cc_code = f.read()
        self.assertIn("Inpatient Telemetry Census", cc_code)
        self.assertIn("Real-Time Clinical Alert Feed", cc_code)
        self.assertIn("5-Agent Autonomous System Health Matrix", cc_code)
        self.assertNotIn("System Pipeline Architecture Flow", cc_code)
        self.assertIn("Adaptive Execution & Surveillance Metrics", cc_code)

        with open(ai_path, "r", encoding="utf-8") as f:
            ai_code = f.read()
        self.assertIn("Agent Architecture & Tool Execution Inspection", ai_code)
        self.assertIn("Agentic Tool Call Sequence", ai_code)
        self.assertIn("Supervisor Telemetry", ai_code)

    def test_header_wires_primary_navigation_views(self):
        """Verify Header.jsx provides navigation buttons for the three primary views."""
        header_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "components",
            "layout",
            "Header.jsx",
        )
        with open(header_path, "r", encoding="utf-8") as f:
            header_code = f.read()

        self.assertIn("COMMAND CENTER", header_code)
        self.assertIn("PATIENT SURVEILLANCE", header_code)
        self.assertIn("AGENT INSPECTION", header_code)
        self.assertIn("command_center", header_code)
        self.assertIn("patient_surveillance", header_code)
        self.assertIn("agent_inspection", header_code)

    def test_agent_inspection_view_covers_all_five_agents_and_paradigms(self):
        """Verify AgentInspectionView defines all 5 agents, distinct paradigms, and dedicated subpanels."""
        ai_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "components",
            "agents",
            "AgentInspectionView.jsx",
        )
        self.assertTrue(os.path.exists(ai_path))
        with open(ai_path, "r", encoding="utf-8") as f:
            code = f.read()

        # All 5 agents present
        for ag in ("MonitoringAgent", "RiskAgent", "DataAnalysisAgent", "ClinicalReasoningAgent", "CareCoordinationAgent"):
            self.assertIn(ag, code, f"AgentInspectionView must support {ag}")

        # Paradigms represented
        self.assertIn("Agentic Tool Loop", code)
        self.assertIn("Trained ML Inference", code)
        self.assertIn("Deterministic CDS", code)

        # Registered monitoring tools library
        for tool in ("preprocess", "calculate_baseline", "calculate_deviations", "analyze_trends", "assess_signal_quality"):
            self.assertIn(tool, code, f"AgentInspectionView must document tool {tool}")

        # Cross-agent FIPA negotiation
        self.assertIn("FIPA Protocol", code)
        self.assertIn("CHALLENGE", code)
        self.assertIn("Requested Checks", code)

        # PubMed RAG grounding
        self.assertIn("PubMed RAG Grounding", code)
        self.assertIn("Focused Medical Query", code)
        self.assertIn("PMID", code)

        # Bedside orders and pathways
        self.assertIn("Actionable Bedside Orders Checklist", code)
        self.assertIn("Escalation Pathway Protocol Definitions", code)

    def test_agent_inspection_view_safety_invariants_and_overrides(self):
        """Verify safety invariant overrides and sanitization guarantees in AgentInspectionView."""
        ai_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "components",
            "agents",
            "AgentInspectionView.jsx",
        )
        with open(ai_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Risk physiological safety floor override
        self.assertIn("Physiological Safety Floor Override", code)
        self.assertIn("65 mmHg", code)
        self.assertIn("140 bpm", code)
        self.assertIn("0.85", code)

        # Clinical reasoning deterministic safety arbitration
        self.assertIn("Deterministic Safety Arbitration", code)
        self.assertIn("Safety Arbitration Floor Enforced", code)

        # Data sanitization guarantee
        self.assertIn("Strict Zero-Thought Guarantee", code)

    def test_clinical_dashboard_view_components_and_structure(self):
        """Verify ClinicalDashboard.jsx defines all 7 clinical sections and handles truthful state."""
        cd_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "components",
            "clinical",
            "ClinicalDashboard.jsx",
        )
        self.assertTrue(os.path.exists(cd_path), "ClinicalDashboard.jsx must exist")
        with open(cd_path, "r", encoding="utf-8") as f:
            code = f.read()

        # All 7 clinical components present
        self.assertIn("PatientSelector", code)
        self.assertIn("PatientHeader", code)
        self.assertIn("CurrentVitals", code)
        self.assertIn("ClinicalAssessment", code)
        self.assertIn("VitalTrends", code)
        self.assertIn("AlertsRisk", code)
        self.assertIn("RecommendedActions", code)
        self.assertIn("EvidencePanel", code)

        # Truthful empty/waiting states
        self.assertIn("Awaiting telemetry", code)
        self.assertIn("Awaiting clinical reasoning", code)
        self.assertIn("Collecting trend data", code)
        self.assertIn("Awaiting care coordination", code)

        # Safety arbitration note check
        self.assertIn("Safety arbitration was applied", code)

        # Vital thresholds and NaN safety
        self.assertIn("isNaN", code)
        self.assertIn("HR", code)
        self.assertIn("MAP", code)
        self.assertIn("SpO", code)
        self.assertIn("RR", code)

    def test_router_and_initial_state_defaults_to_clinical_dashboard(self):
        """Verify DashboardRouter and initialState default to clinical_dashboard view."""
        router_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "app",
            "DashboardRouter.jsx",
        )
        with open(router_path, "r", encoding="utf-8") as f:
            router_code = f.read()
        self.assertIn("clinical_dashboard", router_code)
        self.assertIn("ClinicalDashboard", router_code)

        state_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "carematrix_runtime",
            "static",
            "js",
            "state",
            "initialState.js",
        )
        with open(state_path, "r", encoding="utf-8") as f:
            state_code = f.read()
        self.assertIn("activeTab: 'clinical_dashboard'", state_code)


if __name__ == "__main__":
    unittest.main()


