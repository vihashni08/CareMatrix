"""End-to-End Multi-Agent Integration Tests for the Agentic CareMatrix Architecture.

Validates the full continuous pipeline across:
  Observations -> Monitoring Agent (LLM tools)
               -> Risk Agent (ML inference)
               <-> Data Analysis Agent (LLM tools) [Challenge-Negotiation]
               -> Clinical Reasoning Agent (LLM tools + RAG)
               -> Care Coordination Agent

Covers:
1. Complete 5-agent asynchronous pipeline execution through EventQueue.
2. Verification of tool traces and agentic metadata preserved without collision.
3. Multi-patient concurrent stream isolation (Bed 101 vs Bed 102).
4. Cross-agent challenge-response negotiation flow with trace retention.
5. Clinical Reasoning dynamic RAG tool grounding and synthesis.
6. Failure isolation and deterministic fallback recovery.
7. End-to-end safety arbitration overriding unsafe LLM downgrades.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from care_coordination_agent import CareCoordinationAgent
from carematrix_runtime.patient_memory import PatientMemory
from carematrix_runtime.state_manager import PatientStateManager
from clinical_reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning_agent.llm_reasoner import (
    ClinicalReasoningToolDecision,
    GeminiClinicalReasoner,
)
from clinical_reasoning_agent.rag.schemas import RetrievalResult, RetrievedPassage
from clinical_reasoning_agent.schemas import LLMReasoningResult
from communication.event_queue import EventQueue
from communication.events import (
    CareCoordinationEvent,
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    MonitoringDecision,
    MonitoringEvent,
    PerformativeType,
    RiskDecision,
    RiskDecisionEvent,
)
from communication.llm_client import LLMTimeoutError
from data_analysis_agent import DataAnalysisAgent
from data_analysis_agent.data_analysis_llm_reasoner import (
    DataAnalysisToolDecision,
    GeminiDataAnalysisReasoner,
)
from monitoring_agent.monitoring_agent import MonitoringAgent
from risk_agent import RiskAgent
from risk_agent.risk_llm_reasoner import (
    GeminiRiskReasoner,
    RiskChallengeResponse,
    RiskLLMProposal,
)


def make_deteriorating_window(points: int = 61) -> pd.DataFrame:
    """Return a DataFrame reflecting severe hemodynamic and respiratory deterioration."""
    return pd.DataFrame({
        "Time": np.arange(points) * 5.0,
        "HR": np.linspace(72, 128, points),    # Marked tachycardia
        "SpO2": np.linspace(98, 87, points),   # Hypoxemia
        "RR": np.linspace(14, 30, points),     # Tachypnea
        "SBP": np.linspace(125, 82, points),   # Hypotension
        "DBP": np.linspace(80, 48, points),
        "MAP": np.linspace(95, 59, points),
        "BT": np.full(points, 38.6),           # Febrile
    })


def make_stable_window(points: int = 61) -> pd.DataFrame:
    """Return a DataFrame reflecting stable physiological state."""
    return pd.DataFrame({
        "Time": np.arange(points) * 5.0,
        "HR": np.full(points, 72.0),
        "SpO2": np.full(points, 98.0),
        "RR": np.full(points, 14.0),
        "SBP": np.full(points, 120.0),
        "DBP": np.full(points, 78.0),
        "MAP": np.full(points, 92.0),
        "BT": np.full(points, 36.8),
    })


class TestAgenticEndToEndIntegration(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.state_manager = PatientStateManager()
        self.queue.add_diagnostic_listener(self.state_manager.handle_event)

        # In-memory episodic memory for clinical reasoning
        self.patient_memory = PatientMemory(
            db_path=None,
            event_queue=self.queue,
            max_episodes_per_patient=10,
        )

        # Monitored windows keyed by patient id
        self.patient_windows: dict[int, pd.DataFrame] = {}

        def window_loader(pid):
            pid_int = int(pid) if str(pid).isdigit() else pid
            return self.patient_windows.get(pid_int, make_stable_window())

        # 1. Monitoring Agent (agentic enabled)
        self.monitoring_agent = MonitoringAgent(
            case_id=101,
            event_queue=self.queue,
            baseline_window=10,
            persistence_duration=2,
            verbose=False,
            enable_llm=True,
            max_tool_iterations=4,
        )

        # 2. Risk Agent
        self.risk_agent = RiskAgent(
            event_queue=self.queue,
            verbose=False,
            enable_llm=False,
            window_provider=window_loader,
        )

        # 3. Data Analysis Agent (agentic enabled)
        self.data_analysis_agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=window_loader,
            verbose=False,
            enable_llm=True,
            max_tool_iterations=4,
        )

        # 4. Clinical Reasoning Agent (agentic enabled)
        self.clinical_reasoning_agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=True,
            verbose=False,
            patient_memory=self.patient_memory,
            max_tool_iterations=4,
        )

        # 5. Care Coordination Agent
        self.care_coordination_agent = CareCoordinationAgent(
            event_queue=self.queue,
            verbose=False,
        )

    def tearDown(self):
        # Ensure all worker threads are safely terminated
        for agent in [
            self.monitoring_agent,
            self.risk_agent,
            self.data_analysis_agent,
            self.clinical_reasoning_agent,
            self.care_coordination_agent,
        ]:
            if hasattr(agent, "stop"):
                agent.stop()
            if hasattr(agent, "join"):
                try:
                    agent.join(timeout=0.5)
                except Exception:
                    pass
        self.queue.shutdown()

    def test_complete_five_agent_asynchronous_pipeline_flow(self):
        """Test full event-driven execution across all 5 agents with background worker threads.

        Flow:
          Monitoring Agent processes persistent abnormal observations ->
          MonitoringEvent (monitoring_events) ->
          Risk Agent evaluates features & ML model ->
          RiskDecisionEvent (risk_decisions) ->
          Data Analysis Agent performs agentic analysis ->
          DataAnalysisEvent (clinical_reasoning_events) ->
          Clinical Reasoning Agent executes agentic synthesis ->
          ClinicalReasoningEvent (clinical_decisions) ->
          Care Coordination Agent activates protocol ->
          CareCoordinationEvent (care_coordination_events)
        """
        # Assign deteriorating vitals window for Patient 101
        self.patient_windows[101] = make_deteriorating_window()

        # Start downstream background worker threads
        self.risk_agent.start()
        self.data_analysis_agent.start()
        self.clinical_reasoning_agent.start()
        self.care_coordination_agent.start()

        # Feed observations into Monitoring Agent to establish baseline and trigger persistence threshold
        t_base = 0.0
        for i in range(12):
            obs = {
                "patient_id": 101,
                "timestamp": t_base + i * 5.0,
                "HR": 75.0,
                "SpO2": 98.0,
                "RR": 14.0,
                "SBP": 120.0,
                "DBP": 80.0,
                "MAP": 93.0,
                "BT": 36.8,
            }
            self.monitoring_agent.observe(obs)

        # Now feed severe abnormalities to trigger critical escalation
        for i in range(7):
            obs = {
                "patient_id": 101,
                "timestamp": t_base + (12 + i) * 5.0,
                "HR": 135.0,
                "SpO2": 85.0,
                "RR": 32.0,
                "SBP": 80.0,
                "DBP": 45.0,
                "MAP": 56.0,
                "BT": 38.9,
            }
            self.monitoring_agent.observe(obs)

        # Wait for the complete pipeline to cascade through EventQueue to CareCoordination
        deadline = time.time() + 4.0
        final_care_event: CareCoordinationEvent | None = None
        while time.time() < deadline:
            care_events = self.queue.get_history("care_coordination_events")
            matching = [e for e in care_events if getattr(e, "patient_id", None) == 101]
            if matching:
                final_care_event = matching[-1]
                break
            time.sleep(0.05)

        self.assertIsNotNone(
            final_care_event,
            "End-to-end pipeline failed: CareCoordinationEvent was not published within deadline.",
        )

        # 1. Verify all 5 stages published to their respective topics
        monitoring_events = [e for e in self.queue.get_history("monitoring_events") if getattr(e, "patient_id", None) == 101]
        risk_events = [e for e in self.queue.get_history("risk_decisions") if getattr(e, "patient_id", None) == 101]
        analysis_events = [e for e in self.queue.get_history("clinical_reasoning_events") if getattr(e, "case_id", None) == 101]
        clinical_events = [e for e in self.queue.get_history("clinical_decisions") if getattr(e, "case_id", None) == 101]

        self.assertGreaterEqual(len(monitoring_events), 1, "MonitoringEvent missing")
        self.assertGreaterEqual(len(risk_events), 1, "RiskDecisionEvent missing")
        self.assertGreaterEqual(len(analysis_events), 1, "DataAnalysisEvent missing")
        self.assertGreaterEqual(len(clinical_events), 1, "ClinicalReasoningEvent missing")

        # 2. Verify agentic mode and metadata exist without collision
        monitoring_meta = monitoring_events[-1].metadata
        self.assertIn(monitoring_meta.get("agentic_mode"), ("BOUNDED_LLM_TOOLS", "LLM_FALLBACK_DETERMINISTIC", "DETERMINISTIC_FALLBACK"))

        analysis_meta = analysis_events[-1].metadata
        self.assertIn(analysis_meta.get("agentic_mode"), ("LLM_AGENTIC_TOOLS", "LLM_FALLBACK_DETERMINISTIC", "DETERMINISTIC"))

        clinical_meta = clinical_events[-1].metadata
        self.assertIn(clinical_meta.get("agentic_mode"), ("BOUNDED_LLM_TOOLS", "LLM_FALLBACK_DETERMINISTIC", "DETERMINISTIC"))

        # 3. Verify PatientStateManager recorded the final state
        patient_record = self.state_manager.get_patient_detail(101)
        self.assertIsNotNone(patient_record)
        self.assertEqual(patient_record["patient_id"], 101)
        self.assertIn(patient_record["status"], ("ALERT", "SURVEILLANCE"))

    def test_multi_patient_concurrent_stream_isolation(self):
        """Verify that concurrent evaluations for Patient 101 and Patient 102 maintain strict isolation.

        Patient 101: Severely deteriorating (High Risk, Alert)
        Patient 102: Stable physiological vitals (Low Risk, Stable)
        """
        self.patient_windows[101] = make_deteriorating_window()
        self.patient_windows[102] = make_stable_window()

        self.risk_agent.start()
        self.data_analysis_agent.start()
        self.clinical_reasoning_agent.start()
        self.care_coordination_agent.start()

        # Establish baseline for Patient 101
        t0 = 0.0
        for i in range(12):
            self.monitoring_agent.observe({
                "patient_id": 101, "timestamp": t0 + i * 5.0,
                "HR": 72.0, "SpO2": 98.0, "RR": 14.0, "SBP": 120.0, "DBP": 80.0, "MAP": 93.0, "BT": 36.8,
            })
        # Step Patient 101 into severe deterioration
        for i in range(7):
            self.monitoring_agent.observe({
                "patient_id": 101, "timestamp": t0 + (12 + i) * 5.0,
                "HR": 135.0, "SpO2": 85.0, "RR": 32.0, "SBP": 80.0, "DBP": 45.0, "MAP": 56.0, "BT": 38.9,
            })

        # Step Patient 102 with stable data
        for i in range(15):
            self.monitoring_agent.observe({
                "patient_id": 102, "timestamp": t0 + i * 5.0,
                "HR": 70.0, "SpO2": 99.0, "RR": 13.0, "SBP": 118.0, "DBP": 76.0, "MAP": 90.0, "BT": 36.6,
            })

        # Wait for asynchronous pipeline completion for patient 101
        deadline = time.time() + 3.0
        while time.time() < deadline:
            care_events = self.queue.get_history("care_coordination_events")
            if any(getattr(e, "patient_id", None) == 101 for e in care_events):
                break
            time.sleep(0.05)

        # Inspect isolated states in StateManager
        p101 = self.state_manager.get_patient_detail(101)
        p102 = self.state_manager.get_patient_detail(102)

        self.assertIsNotNone(p101)
        self.assertIsNotNone(p102)

        # Patient 101 should be escalated/alerted
        self.assertIn(p101["status"], ("ALERT", "SURVEILLANCE"))
        # Patient 102 should remain STABLE
        self.assertEqual(p102["status"], "STABLE")

        # Verify Monitoring Agent maintains distinct state buckets
        with self.monitoring_agent._patient_state_lock:
            s101 = self.monitoring_agent.patient_states.get(101)
            s102 = self.monitoring_agent.patient_states.get(102)
            self.assertIsNotNone(s101)
            self.assertIsNotNone(s102)
            self.assertNotEqual(s101.latest_clean_values.get("HR"), s102.latest_clean_values.get("HR"))

    def test_cross_agent_challenge_negotiation_with_tool_trace_retention(self):
        """Verify that when DataAnalysis challenges Risk, the negotiation resolves and traces are preserved."""
        # Provide stable data window for patient 105
        self.patient_windows[105] = make_stable_window()

        self.risk_agent.start()
        self.data_analysis_agent.start()
        self.clinical_reasoning_agent.start()

        # Emit an initial conflicting risk decision: Model says HIGH RISK, but vitals are stable
        conflict_risk = RiskDecisionEvent(
            patient_id=105,
            event_id="case_105_negotiation_test",
            timestamp=300.0,
            decision="HIGH_RISK",
            risk_probability=0.88,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            window_start=0,
            window_end=300,
            performative=PerformativeType.INFORM.value,
            challenge_round=0,
        )
        self.risk_agent.state.record_decision(conflict_risk, features={"HR_mean": 72.0})

        # Inject into risk_decisions topic
        self.queue.publish("risk_decisions", conflict_risk)

        # Wait for the negotiation dialogue to complete and reach clinical reasoning
        deadline = time.time() + 6.0
        clinical_event = None
        while time.time() < deadline:
            c_events = self.queue.get_history("clinical_decisions")
            matching = [e for e in c_events if getattr(e, "case_id", None) == 105]
            if matching and matching[-1].metadata.get("negotiation_trace"):
                clinical_event = matching[-1]
                break
            time.sleep(0.05)

        self.assertIsNotNone(clinical_event, "Negotiation cycle did not reach clinical decisions")
        neg_trace = clinical_event.metadata.get("negotiation_trace", {})
        self.assertEqual(neg_trace.get("status"), "RESOLVED")
        self.assertEqual(neg_trace.get("challenge_round"), 1)

        # Verify DataAnalysis resolved event was recorded
        analysis_events = [e for e in self.queue.get_history("clinical_reasoning_events") if getattr(e, "case_id", None) == 105]
        self.assertTrue(len(analysis_events) >= 1)
        resolved_analysis = analysis_events[-1]
        self.assertIn("negotiation_trace", resolved_analysis.metadata)

    def test_clinical_reasoning_rag_grounding_integration(self):
        """Verify Clinical Reasoning Agent formulates RAG queries and grounds decisions in retrieved evidence."""
        mock_passage = RetrievedPassage(
            document_id="guideline_sepsis",
            title="Surviving Sepsis Campaign Guidelines",
            source="Surviving Sepsis Campaign 2021",
            section="Resuscitation",
            text="Administer 30 mL/kg IV crystalloid fluid for hypotension or lactate >= 4 mmol/L within 3 hours.",
            relevance_score=0.92,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.side_effect = [
            ClinicalReasoningToolDecision(
                thought="Fetch medical literature for sepsis resuscitation.",
                action="tool",
                tool_name="retrieve_medical_evidence",
                tool_args={"query": "sepsis fluid resuscitation"},
            ),
            ClinicalReasoningToolDecision(thought="Finish analysis", action="finish"),
        ]
        mock_reasoner.reason.return_value = LLMReasoningResult(
            clinical_summary="Patient exhibits sepsis pattern requiring urgent crystalloid resuscitation.",
            supporting_evidence=["Hypotension with tachycardia", "Surviving Sepsis Campaign protocol indicated"],
            conflicting_evidence=[],
            key_findings=["Hypotension with tachycardia", "Surviving Sepsis Campaign protocol indicated"],
            risk_interpretation="High clinical risk",
            priority="URGENT",
            recommended_actions=["Consider clinician evaluation for 30 mL/kg IV crystalloid fluid."],
            confidence=0.94,
            knowledge_sources=[{"title": "Surviving Sepsis Campaign Guidelines", "source": "Surviving Sepsis Campaign 2021"}],
            retrieval_status="SUCCESS",
        )

        self.clinical_reasoning_agent.llm_reasoner = mock_reasoner
        self.clinical_reasoning_agent.tools["retrieve_medical_evidence"] = lambda event_or_query, top_k=2: RetrievalResult(
            query="sepsis fluid resuscitation",
            passages=[mock_passage],
            retrieval_status="SUCCESS",
            top_score=0.92,
        )

        analysis_event = DataAnalysisEvent(
            case_id=101,
            event_id="case_101_rag_test",
            timestamp=time.time(),
            risk_level="HIGH RISK",
            data_quality_flag=False,
            conflict_flags=[],
            evidence_consistency="SUPPORTING",
            verification_required=False,
            metadata={
                "tool_trace": [{"tool": "statistical_analysis", "status": "success"}],
                "hemodynamic_assessment": "Severe septic shock pattern with profound hypotension",
            },
        )

        clinical_result = self.clinical_reasoning_agent.process_event(analysis_event)

        self.assertIsNotNone(clinical_result)
        self.assertEqual(clinical_result.case_id, 101)
        self.assertIn("knowledge_sources", clinical_result.metadata)
        self.assertTrue(any("Surviving Sepsis Campaign" in str(s) for s in clinical_result.metadata["knowledge_sources"]))
        self.assertTrue(any("crystalloid" in f.lower() or "sepsis" in f.lower() for f in clinical_result.findings))

    def test_failure_isolation_and_graceful_fallback(self):
        """Verify that an LLM timeout/failure in Data Analysis does not crash the system, falling back safely."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = LLMTimeoutError("Gemini API connection timed out")
        self.data_analysis_agent.llm_reasoner = mock_reasoner

        risk_event = RiskDecisionEvent(
            patient_id=101,
            event_id="case_101_fail_safe",
            timestamp=time.time(),
            decision="HIGH_RISK",
            risk_probability=0.85,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            window_start=0,
            window_end=300,
            performative=PerformativeType.INFORM.value,
            challenge_round=0,
        )

        # Process event; should execute deterministic fallback without throwing
        analysis_result = self.data_analysis_agent.process_event(risk_event)

        self.assertIsNotNone(analysis_result)
        self.assertEqual(analysis_result.metadata.get("agentic_mode"), "LLM_FALLBACK_DETERMINISTIC")
        self.assertIn("fallback_error", analysis_result.metadata)
        self.assertIn("Gemini API connection timed out", analysis_result.metadata["fallback_error"])

    def test_safety_arbitration_overrides_unsafe_llm_downgrade(self):
        """Verify that deterministic safety rules override an unsafe LLM routine downgrade for severe vital collapse."""
        analysis_event = DataAnalysisEvent(
            case_id=101,
            event_id="case_101_safety_override",
            timestamp=time.time(),
            risk_level="HIGH RISK",
            data_quality_flag=False,
            conflict_flags=[],
            evidence_consistency="SUPPORTING",
            verification_required=False,
            metadata={"tool_trace": []},
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.return_value = ClinicalReasoningToolDecision(
            thought="Review complete.",
            action="finish",
        )
        mock_reasoner.reason.return_value = LLMReasoningResult(
            clinical_summary="Patient appears stable; routine vital checks recommended.",
            supporting_evidence=["Normal sinus rhythm"],
            conflicting_evidence=[],
            key_findings=["Mild fluctuations only"],
            risk_interpretation="Low risk",
            priority="ROUTINE",  # Unsafe downgrade!
            recommended_actions=["Continue routine vital checks."],
            confidence=0.75,
        )
        self.clinical_reasoning_agent.llm_reasoner = mock_reasoner

        clinical_result = self.clinical_reasoning_agent.process_event(analysis_event)

        # Safety arbitration must elevate or preserve urgent/elevated priority
        self.assertIn(clinical_result.priority, ("URGENT", "ELEVATED"))
        self.assertTrue(clinical_result.metadata.get("safety_arbitration_applied", False))
        self.assertTrue(clinical_result.metadata.get("llm_safety_conflict", False))


if __name__ == "__main__":
    unittest.main()
