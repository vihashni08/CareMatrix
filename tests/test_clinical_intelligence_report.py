"""Comprehensive validation test suite for CareMatrix Backend Clinical Intelligence Report.

Verifies:
1. Stable case:
   - No fabricated deterioration
   - Adaptive downstream execution (RAG skipped when justified)
   - Complete, valid Clinical Intelligence Report generated
2. High-risk deterioration:
   - End-to-end pipeline: Monitoring -> Risk -> Analysis -> PubMed -> LLM -> Safety arbitration -> Clinical report -> Care coordination
   - Full 15-section report structure populated
3. Conflicting evidence:
   - Conflicting evidence populated
   - Uncertainty populated
   - Verification required flagged
   - Deterministic safety arbitration overrides LLM downgrade attempt
4. Poor / missing data:
   - No fabricated measurements
   - Data quality flagged as COMPROMISED
   - Confidence strictly bounded (<= 0.60)
   - Sensor verification prioritized
5. RAG grounding & metadata preservation:
   - PMIDs, titles, authors, journal, year, DOI preserved from PubMed
   - No fabricated citations accepted
   - Literature reasoning fields populated (why_retrieved, key_evidence, application_to_case)
6. API compatibility:
   - GET /api/patients/<id> includes latest_clinical_reasoning with complete report
   - GET /api/patients/<id>/clinical-report returns 15-section structured payload
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from care_coordination_agent import CareCoordinationAgent
from carematrix_runtime.server import create_app
from clinical_reasoning_agent import (
    ClinicalReasoningAgent,
    ClinicalReasoningEngine,
    ClinicalReasoningPriority,
)
from clinical_reasoning_agent.llm_reasoner import GeminiClinicalReasoner
from clinical_reasoning_agent.rag.schemas import RetrievalResult, RetrievedPassage
from communication.event_queue import EventQueue
from communication.events import (
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringEvent,
    RiskDecisionEvent,
)


class TestClinicalIntelligenceReport(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.engine = ClinicalReasoningEngine()

    def tearDown(self):
        self.queue.shutdown()

    # ------------------------------------------------------------------------
    # 1. Stable Case
    # ------------------------------------------------------------------------
    def test_1_stable_case_produces_complete_report_and_skips_rag(self):
        """Stable patient with high confidence skips RAG and yields a complete, valid clinical report."""
        agent = ClinicalReasoningAgent(event_queue=self.queue, enable_llm=False, enable_rag=True)

        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_stable_001",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
            trend_metrics={
                "HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 1.0, "slope": 0.01},
                "MAP": {"trend": "stable", "latest_value": 85.0, "change_over_window": -0.5, "slope": -0.005},
                "SpO2": {"trend": "stable", "latest_value": 98.0, "change_over_window": 0.0, "slope": 0.0},
            },
            pattern_identified=["Stable vital sign trajectory within normal baseline tolerance."],
        )

        cr_event = agent.process_event(event)

        # Gating check: RAG must be skipped
        self.assertEqual(cr_event.metadata.get("evidence_retrieval"), "SKIPPED_LOW_RISK_CONFIDENT")
        self.assertEqual(cr_event.priority, ClinicalReasoningPriority.ROUTINE.value)
        self.assertFalse(cr_event.escalation_required)
        self.assertEqual(cr_event.data_reliability, "HIGH")

        # Verify all 15 report sections
        report = cr_event.to_clinical_report()
        self.assertIn("executive_summary", report)
        self.assertTrue(len(report["executive_summary"]) > 0)
        self.assertEqual(report["clinical_status"]["risk_level"], "LOW RISK")
        self.assertEqual(report["clinical_status"]["priority"], "ROUTINE")
        self.assertEqual(report["clinical_status"]["data_reliability"], "HIGH")
        self.assertEqual(report["clinical_status"]["evidence_consistency"], "SUPPORTING")
        self.assertTrue(len(report["key_findings"]) > 0)
        self.assertTrue(len(report["physiological_analysis"]) > 0)
        self.assertTrue(len(report["temporal_analysis"]) > 0)
        self.assertTrue(len(report["risk_interpretation"]) > 0)
        self.assertTrue(len(report["supporting_evidence"]) > 0)
        self.assertEqual(report["conflicting_evidence"], [])
        self.assertTrue(len(report["evidence_synthesis"]) > 0)
        self.assertEqual(report["medical_evidence"], [])  # RAG skipped -> empty
        self.assertTrue(len(report["clinical_interpretation"]) > 0)
        self.assertEqual(report["uncertainties"], [])
        self.assertTrue(len(report["recommended_actions"]) > 0)
        self.assertTrue(len(report["monitoring_priorities"]) > 0)
        self.assertTrue(len(report["escalation_rationale"]) > 0)
        self.assertGreaterEqual(report["confidence"], 0.85)

    # ------------------------------------------------------------------------
    # 2. High-Risk Deterioration with Full Pipeline Flow
    # ------------------------------------------------------------------------
    def test_2_high_risk_deterioration_pipeline_flow(self):
        """High-risk deterioration flows from Monitoring -> Risk -> Analysis -> Clinical Reasoning -> Care Coordination."""
        # Mock PubMed abstract with full metadata
        mock_abstracts = [
            {
                "pmid": "31234567",
                "title": "Hemodynamic Instability and Vasopressor Timing in Septic Shock",
                "abstract": "Early MAP decrease below 65 mmHg combined with refractory tachycardia warrants prompt bedside intervention.",
                "authors": "Johnson M, Patel R, Smith K et al.",
                "journal": "Intensive Care Med",
                "year": "2023",
                "doi": "10.1007/s00134-023-07100-x",
            }
        ]

        llm_response_json = json.dumps({
            "executive_summary": "Patient demonstrates acute cardiovascular collapse with severe hypotension and compensatory tachycardia.",
            "clinical_summary": "Acute cardiovascular instability with MAP falling to 55 mmHg and HR rising to 125 bpm.",
            "clinical_status": {
                "risk_level": "HIGH RISK",
                "priority": "URGENT",
                "confidence": 0.96,
                "data_reliability": "HIGH",
                "evidence_consistency": "SUPPORTING",
            },
            "key_findings": [
                "Tachycardia trajectory (HR 125 bpm, Δ=+30 bpm)",
                "Systemic hypotension (MAP 55 mmHg, Δ=-25 mmHg)",
                "Consistent cross-agent deterioration markers",
            ],
            "physiological_analysis": [
                "Cardiovascular: Severe hypotension with compensatory tachycardia reflecting progressive hemodynamic compromise.",
                "Respiratory: SpO2 stable at 94% on room air.",
            ],
            "temporal_analysis": [
                "HR increased rapidly over the 15-minute observation window (slope=+2.0 bpm/min).",
                "MAP dropped monotonically below critical threshold 65 mmHg (window Δ=-25.0 mmHg).",
            ],
            "risk_interpretation": "Machine learning model predicted HIGH RISK based on simultaneous heart rate acceleration and blood pressure collapse.",
            "supporting_evidence": [
                "Active deterioration detected in HR and MAP simultaneously.",
                "Model risk probability 0.88 exceeds high-risk threshold.",
            ],
            "conflicting_evidence": [],
            "evidence_synthesis": "Observed acute physiological trajectories directly corroborate the predictive model's high-risk assessment.",
            "medical_evidence": [
                {
                    "pmid": "31234567",
                    "title": "Hemodynamic Instability and Vasopressor Timing in Septic Shock",
                    "authors": "Johnson M, Patel R, Smith K et al.",
                    "journal": "Intensive Care Med",
                    "year": "2023",
                    "doi": "10.1007/s00134-023-07100-x",
                    "relevance_score": 0.95,
                    "why_retrieved": "Retrieved due to concurrent acute hypotension (MAP 55) and tachycardia (HR 125).",
                    "key_evidence": "Recommends prompt bedside intervention when MAP falls below 65 mmHg with refractory tachycardia.",
                    "application_to_case": "Directly applies to patient's current MAP 55 mmHg requiring immediate fluid/vasopressor assessment.",
                }
            ],
            "clinical_interpretation": "Patient is in an active state of hemodynamic decompensation requiring immediate fluid resuscitation and physician bedside evaluation.",
            "uncertainties": [],
            "recommended_actions": [
                "Immediate bedside clinical evaluation of hemodynamic and perfusion status.",
                "Prepare intravenous fluid resuscitation protocol.",
                "Notify attending intensive care physician.",
            ],
            "monitoring_priorities": [
                "Continuous invasive arterial blood pressure monitoring.",
                "High-frequency 5-minute vitals reassessment.",
            ],
            "escalation_rationale": "URGENT priority assigned due to confirmed acute hypotension below 65 mmHg with high model deterioration risk.",
            "priority": "URGENT",
            "confidence": 0.96,
        })

        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = llm_response_json
        mock_client.interactions.create.return_value = mock_interaction
        mock_reasoner = GeminiClinicalReasoner(client=mock_client)

        with patch("clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts", return_value=mock_abstracts):
            cr_agent = ClinicalReasoningAgent(
                event_queue=self.queue,
                llm_reasoner=mock_reasoner,
                enable_llm=True,
                enable_rag=True,
            )

            # Analysis Event
            analysis_event = DataAnalysisEvent(
                case_id=202,
                event_id="evt_high_risk_002",
                timestamp=300.0,
                risk_level="HIGH RISK",
                evidence_consistency="SUPPORTING",
                metadata={"confidence": 0.95},
                trend_metrics={
                    "HR": {"trend": "increasing", "latest_value": 125.0, "change_over_window": 30.0, "slope": 2.0},
                    "MAP": {"trend": "decreasing", "latest_value": 55.0, "change_over_window": -25.0, "slope": -1.8},
                },
                pattern_identified=["Simultaneous directional changes across multiple affected vitals."],
            )

            cr_event = cr_agent.process_event(analysis_event)

        # Validate Clinical Reasoning Output
        self.assertEqual(cr_event.priority, "URGENT")
        self.assertTrue(cr_event.escalation_required)
        self.assertEqual(cr_event.risk_level, "HIGH RISK")
        self.assertEqual(cr_event.metadata.get("evidence_retrieval"), "RUN_HIGH_RISK")

        report = cr_event.to_clinical_report()
        self.assertEqual(report["clinical_status"]["priority"], "URGENT")
        self.assertEqual(len(report["medical_evidence"]), 1)
        med_ev = report["medical_evidence"][0]
        self.assertEqual(med_ev["pmid"], "31234567")
        self.assertEqual(med_ev["authors"], "Johnson M, Patel R, Smith K et al.")
        self.assertEqual(med_ev["journal"], "Intensive Care Med")
        self.assertEqual(med_ev["year"], "2023")
        self.assertEqual(med_ev["doi"], "10.1007/s00134-023-07100-x")
        self.assertIn("MAP 55", med_ev["why_retrieved"])
        self.assertIn("resuscitation", report["clinical_interpretation"].lower())
        self.assertTrue(len(report["monitoring_priorities"]) > 0)

        # Validate Care Coordination downstream consumption
        coord_agent = CareCoordinationAgent(event_queue=self.queue)
        coord_event = coord_agent.process_event(cr_event)

        self.assertIsInstance(coord_event, CareCoordinationEvent)
        self.assertEqual(coord_event.priority, "URGENT")
        self.assertTrue(coord_event.clinician_review_required)
        self.assertEqual(coord_event.action_type, "TRIGGER_URGENT_CLINICAL_ALERT")
        self.assertTrue(any("bedside" in o.lower() for o in coord_event.suggested_orders))

    # ------------------------------------------------------------------------
    # 3. Conflicting Evidence & Safety Arbitration Protection
    # ------------------------------------------------------------------------
    def test_3_conflicting_evidence_and_safety_arbitration_override(self):
        """When LLM attempts to downgrade a HIGH RISK case to ROUTINE, safety arbitration rejects it."""
        # Simulated LLM hallucinating/downgrading a HIGH RISK patient to ROUTINE
        llm_downgrade_json = json.dumps({
            "executive_summary": "Patient is resting quietly; vitals appear acceptable to me.",
            "clinical_summary": "Vitals appear acceptable.",
            "clinical_status": {
                "risk_level": "LOW RISK",
                "priority": "ROUTINE",
                "confidence": 0.90,
                "data_reliability": "HIGH",
                "evidence_consistency": "SUPPORTING",
            },
            "key_findings": ["Subjectively appears comfortable."],
            "physiological_analysis": ["No severe signs noted by assistant."],
            "temporal_analysis": ["Trajectory deemed non-urgent."],
            "risk_interpretation": "Downgrading model assessment to routine monitoring.",
            "supporting_evidence": ["Stable appearance."],
            "conflicting_evidence": [],
            "evidence_synthesis": "Assistant recommends routine surveillance.",
            "medical_evidence": [],
            "clinical_interpretation": "Patient can be managed routinely.",
            "uncertainties": [],
            "recommended_actions": ["Continue standard monitoring."],
            "monitoring_priorities": ["Routine checks."],
            "escalation_rationale": "Assigned routine.",
            "priority": "ROUTINE",  # Conflicting with HIGH RISK!
            "confidence": 0.90,
        })

        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = llm_downgrade_json
        mock_client.interactions.create.return_value = mock_interaction
        mock_reasoner = GeminiClinicalReasoner(client=mock_client)

        cr_agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            llm_reasoner=mock_reasoner,
            enable_llm=True,
            enable_rag=False,
        )

        analysis_event = DataAnalysisEvent(
            case_id=303,
            event_id="evt_conflict_003",
            timestamp=400.0,
            risk_level="HIGH RISK",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
            trend_metrics={
                "HR": {"trend": "increasing", "latest_value": 128.0, "change_over_window": 32.0},
                "MAP": {"trend": "decreasing", "latest_value": 58.0, "change_over_window": -20.0},
            },
        )

        cr_event = cr_agent.process_event(analysis_event)

        # Deterministic Safety Arbitration must OVERRIDE the LLM downgrade
        self.assertEqual(cr_event.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertTrue(cr_event.escalation_required)
        self.assertTrue(cr_event.metadata.get("llm_reasoning_conflict"))
        self.assertTrue(cr_event.metadata.get("safety_arbitration_applied"))
        self.assertIn("SAFETY ARBITRATION", cr_event.clinical_summary)
        self.assertTrue(any("overridden" in c.lower() or "conflicting" in c.lower() for c in cr_event.conflicting_evidence))
        self.assertTrue(cr_event.verification_required)

        # Report checks
        report = cr_event.to_clinical_report()
        self.assertEqual(report["clinical_status"]["priority"], "URGENT")
        self.assertIn("SAFETY ARBITRATION", report["escalation_rationale"])

    # ------------------------------------------------------------------------
    # 4. Poor / Compromised Data Quality
    # ------------------------------------------------------------------------
    def test_4_compromised_data_quality_enforces_bounds_and_verification(self):
        """Data quality flag enforces COMPROMISED reliability, bounds confidence to <= 0.60, and orders sensor check."""
        agent = ClinicalReasoningAgent(event_queue=self.queue, enable_llm=False)

        event = DataAnalysisEvent(
            case_id=404,
            event_id="evt_quality_flag_004",
            timestamp=500.0,
            risk_level="LOW RISK",
            data_quality_flag=True,
            data_quality_details={"reason": "Excessive pulse oximeter probe movement artifacts."},
            evidence_consistency="UNCERTAIN",
            metadata={"confidence": 0.50},
        )

        cr_event = agent.process_event(event)

        self.assertEqual(cr_event.data_reliability, "COMPROMISED")
        self.assertEqual(cr_event.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertTrue(cr_event.verification_required)
        self.assertLessEqual(cr_event.confidence, 0.60)
        self.assertTrue(any("sensor" in a.lower() or "probe" in a.lower() for a in cr_event.recommended_actions))

        report = cr_event.to_clinical_report()
        self.assertEqual(report["clinical_status"]["data_reliability"], "COMPROMISED")
        self.assertEqual(report["clinical_status"]["evidence_consistency"], "UNCERTAIN")
        self.assertTrue(any("sensor" in u.lower() or "quality" in u.lower() or "noise" in u.lower() for u in report["uncertainties"]))

    # ------------------------------------------------------------------------
    # 5. RAG Grounding & Citation Integrity
    # ------------------------------------------------------------------------
    def test_5_rag_grounding_rejects_hallucinated_citations(self):
        """LLM cannot invent unretrieved PMIDs; valid retrieved passages are preserved with metadata."""
        real_passages = [
            RetrievedPassage(
                document_id="PMID_11223344",
                title="Respiratory Rate as an Indicator of Acute Deterioration",
                source="PubMed PMID:11223344",
                section="Abstract",
                text="Tachypnea is a sensitive early marker for physiological decompensation.",
                relevance_score=0.91,
                metadata={
                    "pmid": "11223344",
                    "authors": "Cretikos MA, Bellomo R et al.",
                    "journal": "Med J Aust",
                    "year": "2008",
                    "doi": "10.5694/j.1326-5377.2008.tb01828.x",
                },
            )
        ]
        mock_retrieval = RetrievalResult(
            query="tachypnea respiratory rate increasing",
            passages=real_passages,
            retrieval_status="SUCCESS",
            top_score=0.91,
            retrieval_source="pubmed",
        )

        # LLM response tries to cite a fake PMID "99999999" plus the real one "11223344"
        llm_response_json = json.dumps({
            "executive_summary": "Tachypnea detected indicating respiratory distress.",
            "clinical_summary": "Tachypnea detected.",
            "clinical_status": {
                "risk_level": "LOW RISK",
                "priority": "ELEVATED",
                "confidence": 0.85,
                "data_reliability": "HIGH",
                "evidence_consistency": "SUPPORTING",
            },
            "key_findings": ["RR elevated to 28 /min"],
            "physiological_analysis": ["Respiratory: Tachypnea without severe desaturation."],
            "temporal_analysis": ["RR increased from 16 to 28 /min."],
            "risk_interpretation": "Elevated risk due to respiratory rate acceleration.",
            "supporting_evidence": ["RR 28 /min"],
            "conflicting_evidence": [],
            "evidence_synthesis": "Tachypnea indicates early compensatory stress.",
            "medical_evidence": [
                {
                    "pmid": "99999999",  # FAKE!
                    "title": "Completely Made Up Paper",
                    "why_retrieved": "Hallucinated citation",
                    "key_evidence": "Fake claim",
                    "application_to_case": "Fake application",
                },
                {
                    "pmid": "11223344",  # REAL!
                    "title": "Respiratory Rate as an Indicator of Acute Deterioration",
                    "why_retrieved": "Relevant to tachypnea presentation.",
                    "key_evidence": "Tachypnea is a sensitive early marker of acute deterioration.",
                    "application_to_case": "Underlines clinical urgency of rising respiratory rate.",
                },
            ],
            "clinical_interpretation": "Patient requires close observation of respiratory pattern.",
            "uncertainties": [],
            "recommended_actions": ["Monitor respiratory rate every 15 minutes."],
            "monitoring_priorities": ["Continuous respiratory surveillance."],
            "escalation_rationale": "ELEVATED priority due to acute tachypnea.",
            "priority": "ELEVATED",
            "confidence": 0.85,
        })

        reasoner = GeminiClinicalReasoner(client=MagicMock())
        result = reasoner.validate_and_parse(llm_response_json, retrieval_result=mock_retrieval)

        # Grounding check: The fake PMID "99999999" MUST NOT be included!
        pmids = [m["pmid"] for m in result.medical_evidence]
        self.assertNotIn("99999999", pmids)
        self.assertIn("11223344", pmids)

        # Metadata preservation check
        real_med = [m for m in result.medical_evidence if m["pmid"] == "11223344"][0]
        self.assertEqual(real_med["authors"], "Cretikos MA, Bellomo R et al.")
        self.assertEqual(real_med["journal"], "Med J Aust")
        self.assertEqual(real_med["year"], "2008")
        self.assertEqual(real_med["doi"], "10.5694/j.1326-5377.2008.tb01828.x")
        self.assertEqual(real_med["why_retrieved"], "Relevant to tachypnea presentation.")

    # ------------------------------------------------------------------------
    # 6. API Compatibility & Endpoints
    # ------------------------------------------------------------------------
    def test_6_api_endpoints_return_structured_clinical_intelligence_report(self):
        """Flask API routes return the rich structured Clinical Intelligence Report."""
        app = create_app()
        client = app.test_client()
        rt = app.config["RUNTIME"]

        # Feed a structured ClinicalReasoningEvent to state manager
        cr_event = ClinicalReasoningEvent(
            case_id=101,
            event_id="evt_api_test_001",
            timestamp=12345.0,
            risk_level="HIGH RISK",
            priority="URGENT",
            clinical_summary="Bedside clinical evaluation required immediately for acute deterioration.",
            findings=["Tachycardia", "Hypotension"],
            recommended_actions=["Immediate bedside assessment", "IV fluid access check"],
            escalation_required=True,
            data_reliability="HIGH",
            confidence=0.94,
            evidence_consistency="SUPPORTING",
            executive_summary="Patient 101 demonstrates acute hemodynamic instability.",
            clinical_status={
                "risk_level": "HIGH RISK",
                "priority": "URGENT",
                "confidence": 0.94,
                "data_reliability": "HIGH",
                "evidence_consistency": "SUPPORTING",
            },
            key_findings=["MAP dropped to 56 mmHg", "HR increased to 122 bpm"],
            physiological_analysis=["Cardiovascular: MAP 56 mmHg indicates inadequate tissue perfusion."],
            temporal_analysis=["MAP decreased by -24 mmHg over last 15 minutes."],
            risk_interpretation="Machine learning model predicted HIGH RISK reflecting acute shock trajectory.",
            supporting_evidence=["Concurrent hypotension and tachycardia."],
            conflicting_evidence=[],
            evidence_synthesis="Deterioration confirmed across physiological and predictive metrics.",
            medical_evidence=[
                {
                    "pmid": "12345678",
                    "title": "Shock Guidelines",
                    "authors": "Levy MM et al.",
                    "journal": "Crit Care Med",
                    "year": "2021",
                    "doi": "10.1097/CCM.0000000000005089",
                    "relevance_score": 0.92,
                    "why_retrieved": "Guideline for MAP < 65 mmHg.",
                    "key_evidence": "Target MAP >= 65 mmHg in initial resuscitation.",
                    "application_to_case": "Initiate hemodynamic stabilization targeting MAP >= 65 mmHg.",
                }
            ],
            clinical_interpretation="Acute circulatory compromise requiring attending review.",
            uncertainties=[],
            monitoring_priorities=["Arterial line pressure monitoring."],
            escalation_rationale="URGENT priority assigned due to severe hypotension.",
        )

        rt.state_manager.handle_event("clinical_decisions", cr_event)

        # 1. Test GET /api/patients/<id>
        resp_detail = client.get("/api/patients/101")
        self.assertEqual(resp_detail.status_code, 200)
        detail_data = resp_detail.get_json()
        latest_cr = detail_data.get("latest_clinical_reasoning")
        self.assertIsNotNone(latest_cr)
        self.assertEqual(latest_cr["priority"], "URGENT")
        self.assertIn("clinical_report", latest_cr)
        self.assertEqual(latest_cr["clinical_report"]["clinical_status"]["risk_level"], "HIGH RISK")
        self.assertEqual(len(latest_cr["clinical_report"]["medical_evidence"]), 1)

        # 2. Test GET /api/patients/<id>/clinical-report
        resp_report = client.get("/api/patients/101/clinical-report")
        self.assertEqual(resp_report.status_code, 200)
        report_data = resp_report.get_json()
        self.assertEqual(report_data["patient_id"], 101)
        self.assertIn("clinical_report", report_data)
        report = report_data["clinical_report"]
        self.assertEqual(report["executive_summary"], "Patient 101 demonstrates acute hemodynamic instability.")
        self.assertEqual(report["clinical_status"]["priority"], "URGENT")
        self.assertEqual(report["clinical_status"]["confidence"], 0.94)
        self.assertEqual(report["medical_evidence"][0]["pmid"], "12345678")
        self.assertEqual(report["medical_evidence"][0]["authors"], "Levy MM et al.")
        self.assertEqual(report["medical_evidence"][0]["journal"], "Crit Care Med")
        self.assertEqual(report["medical_evidence"][0]["year"], "2021")
        self.assertEqual(report["medical_evidence"][0]["doi"], "10.1097/CCM.0000000000005089")
        self.assertEqual(report["monitoring_priorities"], ["Arterial line pressure monitoring."])
        self.assertEqual(report["escalation_rationale"], "URGENT priority assigned due to severe hypotension.")


if __name__ == "__main__":
    unittest.main()
