"""Unit and integration tests for bounded LLM-driven tool selection in ClinicalReasoningAgent.

Covers:
A. Patient context selection:
   - Gemini selects get_patient_context.
   - Memory is queried and patient context is returned into state.
B. Query formulation:
   - Gemini selects build_focused_medical_query.
   - De-identified query focused on vitals and trends is returned.
C. Medical evidence retrieval:
   - Gemini selects retrieve_medical_evidence.
   - Authoritative RAG literature passages are fetched.
D. Tool-result feedback:
   - Results from previous tool executions are passed into state summary and next reasoning turns.
E. Clinical evidence evaluation:
   - Gemini selects evaluate_clinical_evidence.
   - Non-diagnostic recommendations and deterministic evaluation are computed.
F. Multi-tool sequence:
   - Gemini executes a bounded sequence of distinct clinical tools.
G. Finish behavior:
   - Gemini selects action='finish' after gathering required information.
H. Gemini failure / timeout fallback:
   - Clean deterministic pipeline fallback without crashing.
I. Malformed / invalid Gemini response:
   - Schema validation error handled gracefully with deterministic fallback.
J. Tool execution failure:
   - Safe error handling in tool trace and graceful recovery.
K. Maximum tool-call limit bounding:
   - Enforces max_tool_iterations cap safely.
L. Deterministic safety arbitration override:
   - If LLM or proposal attempts to downgrade an urgent/high-risk case to routine,
     deterministic safety arbiter strictly enforces high priority / urgent escalation.
M. RAG retrieval integration:
   - Verified that retrieved passages are passed to synthesis prompt and evidence package.
N. State persistence:
   - ClinicalReasoningState records assessments and maintains tool trace.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from communication.event_queue import EventQueue
from communication.events import (
    ClinicalReasoningEvent,
    DataAnalysisEvent,
    PerformativeType,
)
from communication.llm_client import LLMTimeoutError, LLMSchemaValidationError
from clinical_reasoning_agent.clinical_reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning_agent.llm_reasoner import (
    ClinicalReasoningToolDecision,
    GeminiClinicalReasoner,
)
from clinical_reasoning_agent.schemas import LLMReasoningResult
from clinical_reasoning_agent.state import ClinicalReasoningState
from clinical_reasoning_agent.rag.schemas import RetrievalResult, RetrievedPassage
from carematrix_runtime.patient_memory import PatientMemory


class ClinicalReasoningAgenticToolsTests(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.patient_memory = PatientMemory()

    def tearDown(self):
        self.queue.shutdown()

    def make_analysis_event(
        self,
        case_id: int = 101,
        event_id: str = "analysis_evt_001",
        risk_level: str = "HIGH RISK",
        data_quality_flag: bool = False,
        metrics: dict | None = None,
        patterns: list[str] | None = None,
    ) -> DataAnalysisEvent:
        if metrics is None:
            metrics = {
                "HR": {"trend": "increasing", "latest_value": 115.0, "change_over_window": 25.0},
                "MAP": {"trend": "decreasing", "latest_value": 58.0, "change_over_window": -20.0},
            }
        if patterns is None:
            patterns = [
                "HR shows an increasing trend in the assessment window.",
                "MAP shows a decreasing trend in the assessment window.",
                "Simultaneous directional changes are present across multiple vitals.",
            ]
        return DataAnalysisEvent(
            case_id=case_id,
            event_id=event_id,
            timestamp=1200.0,
            risk_level=risk_level,
            trend_metrics=metrics,
            pattern_identified=patterns,
            important_changes=["HR: increasing", "MAP: decreasing"],
            data_quality_flag=data_quality_flag,
            analysis_status="complete",
            evidence_consistency="SUPPORTING",
            metadata={
                "confidence": 0.85,
                "affected_vitals": ["HR", "MAP"],
            },
        )

    def make_llm_result(
        self,
        clinical_summary: str = "Patient exhibits hemodynamic compromise.",
        priority: str = "URGENT",
        recommended_actions: list[str] | None = None,
        confidence: float = 0.9,
        key_findings: list[str] | None = None,
        risk_interpretation: str = "High risk hemodynamic deterioration",
        supporting_evidence: list[str] | None = None,
        conflicting_evidence: list[str] | None = None,
        medical_evidence: list[dict] | None = None,
        retrieval_status: str = "NO_RELEVANT_EVIDENCE",
    ) -> LLMReasoningResult:
        return LLMReasoningResult(
            clinical_summary=clinical_summary,
            supporting_evidence=supporting_evidence or ["MAP remains low."],
            conflicting_evidence=conflicting_evidence or [],
            key_findings=key_findings or ["Tachycardia", "Hypotension"],
            risk_interpretation=risk_interpretation,
            priority=priority,
            recommended_actions=recommended_actions or ["Consider clinician evaluation."],
            confidence=confidence,
            medical_evidence=medical_evidence or [],
            retrieval_status=retrieval_status,
        )

    # -------------------------------------------------------------------------
    # TEST A: Patient Context Selection
    # -------------------------------------------------------------------------
    def test_a_patient_context_selection(self):
        """Gemini selects get_patient_context and receives patient history."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            patient_memory=self.patient_memory,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        decisions = [
            ClinicalReasoningToolDecision(
                thought="Need to check past clinical episodes and feedback for this patient.",
                action="tool",
                tool_name="get_patient_context",
                tool_args={"limit": 5},
                confidence=0.9,
            ),
            ClinicalReasoningToolDecision(
                thought="Patient context retrieved, ready to finish.",
                action="finish",
                rationale="Sufficient clinical history gathered.",
                confidence=0.95,
            ),
        ]
        mock_reasoner.select_tool.side_effect = decisions
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Patient exhibits tachycardia with hypotension.",
            priority="URGENT",
            recommended_actions=["Consider clinician evaluation for cardiovascular support."],
            confidence=0.9,
            key_findings=["Tachycardia", "Hypotension"],
            risk_interpretation="High risk hemodynamic deterioration",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        self.assertIsInstance(result, ClinicalReasoningEvent)
        self.assertEqual(result.metadata.get("agentic_mode"), "LLM_AGENTIC_TOOLS")
        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["tool_name"], "get_patient_context")
        self.assertEqual(trace[0]["status"], "success")
        self.assertEqual(trace[1]["action"], "finish")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST B: Focused Medical Query Formulation
    # -------------------------------------------------------------------------
    def test_b_query_formulation(self):
        """Gemini selects build_focused_medical_query to construct search query."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        decisions = [
            ClinicalReasoningToolDecision(
                thought="Need to construct focused medical literature query for hemodynamic instability.",
                action="tool",
                tool_name="build_focused_medical_query",
                confidence=0.88,
            ),
            ClinicalReasoningToolDecision(
                thought="Query built, concluding tool gathering.",
                action="finish",
                rationale="Query formulated.",
                confidence=0.9,
            ),
        ]
        mock_reasoner.select_tool.side_effect = decisions
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Analysis complete with focused query.",
            priority="URGENT",
            recommended_actions=["Consider bedside evaluation."],
            confidence=0.88,
            key_findings=["HR increasing", "MAP decreasing"],
            risk_interpretation="Deteriorating state",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        self.assertIsInstance(result, ClinicalReasoningEvent)
        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(trace[0]["tool_name"], "build_focused_medical_query")
        self.assertIn("query", trace[0]["output"])
        self.assertTrue(len(trace[0]["output"]["query"]) > 0)
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST C: Medical Evidence Retrieval via RAG
    # -------------------------------------------------------------------------
    def test_c_medical_evidence_retrieval(self):
        """Gemini selects retrieve_medical_evidence and literature passages are retrieved."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=True,
            max_tool_iterations=3,
        )

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="tachycardia hypotension",
            passages=[
                RetrievedPassage(
                    document_id="doc_sepsis_01",
                    title="Management of Septic Shock and Hemodynamic Instability",
                    source="guidelines",
                    section="Hemodynamics",
                    text="Sepsis guidelines recommend prompt resuscitation and blood pressure restoration.",
                    relevance_score=0.92,
                    metadata={"pmid": "33333333"},
                )
            ],
            retrieval_status="SUCCESS",
            top_score=0.92,
        )
        agent.retriever = mock_retriever
        agent.tools["retrieve_medical_evidence"] = lambda event_or_query, top_k=2: mock_retriever.retrieve(event_or_query)

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        decisions = [
            ClinicalReasoningToolDecision(
                thought="Retrieve clinical literature on septic shock and hemodynamic trends.",
                action="tool",
                tool_name="retrieve_medical_evidence",
                tool_args={"top_k": 2},
                confidence=0.91,
            ),
            ClinicalReasoningToolDecision(
                thought="Evidence retrieved, completing cycle.",
                action="finish",
                rationale="Passages retrieved successfully.",
                confidence=0.95,
            ),
        ]
        mock_reasoner.select_tool.side_effect = decisions
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Hemodynamic instability consistent with sepsis guidelines.",
            priority="URGENT",
            recommended_actions=["Consider clinician evaluation for prompt hemodynamic support."],
            confidence=0.92,
            key_findings=["Hypotension", "Tachycardia"],
            risk_interpretation="Septic shock risk",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        self.assertIsInstance(result, ClinicalReasoningEvent)
        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(trace[0]["tool_name"], "retrieve_medical_evidence")
        self.assertTrue(trace[0]["output"]["has_evidence"])
        self.assertEqual(trace[0]["output"]["passages_count"], 1)
        self.assertEqual(result.metadata.get("evidence_retrieval"), "RUN_AGENTIC_TOOL_RAG")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST D: Tool Result Feedback into Next Gemini Turn
    # -------------------------------------------------------------------------
    def test_d_tool_result_feedback(self):
        """Verifies that results from executed tools are reflected in state_summary for subsequent turns."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            patient_memory=self.patient_memory,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True

        recorded_summaries = []

        def fake_select_tool(**kwargs):
            recorded_summaries.append(dict(kwargs.get("state_summary", {})))
            if len(recorded_summaries) == 1:
                return ClinicalReasoningToolDecision(
                    thought="First get patient context.",
                    action="tool",
                    tool_name="get_patient_context",
                    tool_args={"limit": 3},
                )
            elif len(recorded_summaries) == 2:
                return ClinicalReasoningToolDecision(
                    thought="Now build medical query.",
                    action="tool",
                    tool_name="build_focused_medical_query",
                )
            else:
                return ClinicalReasoningToolDecision(
                    thought="All information gathered, finish.",
                    action="finish",
                )

        mock_reasoner.select_tool.side_effect = fake_select_tool
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Patient assessment synthesized.",
            priority="URGENT",
            recommended_actions=["Consider clinical evaluation."],
            confidence=0.85,
            key_findings=["Hypotension"],
            risk_interpretation="High risk",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        agent.process_event(event)

        self.assertEqual(len(recorded_summaries), 3)
        # Turn 1: No patient context yet
        self.assertFalse(recorded_summaries[0]["patient_context_retrieved"])
        self.assertFalse(recorded_summaries[0]["medical_query_built"])

        # Turn 2: Patient context retrieved, no query yet
        self.assertTrue(recorded_summaries[1]["patient_context_retrieved"])
        self.assertFalse(recorded_summaries[1]["medical_query_built"])

        # Turn 3: Both patient context and query present
        self.assertTrue(recorded_summaries[2]["patient_context_retrieved"])
        self.assertTrue(recorded_summaries[2]["medical_query_built"])
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST E: Clinical Evidence Evaluation
    # -------------------------------------------------------------------------
    def test_e_evaluate_clinical_evidence(self):
        """Gemini invokes evaluate_clinical_evidence and deterministic output is computed."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=2,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        decisions = [
            ClinicalReasoningToolDecision(
                thought="Run deterministic evidence evaluation first.",
                action="tool",
                tool_name="evaluate_clinical_evidence",
            ),
            ClinicalReasoningToolDecision(
                thought="Finish after baseline evaluation.",
                action="finish",
            ),
        ]
        mock_reasoner.select_tool.side_effect = decisions
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Evaluation confirmed.",
            priority="URGENT",
            recommended_actions=["Consider bedside evaluation."],
            confidence=0.9,
            key_findings=["Severe hypotension"],
            risk_interpretation="Critical condition",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(trace[0]["tool_name"], "evaluate_clinical_evidence")
        self.assertIn("priority", trace[0]["output"])
        self.assertEqual(trace[0]["output"]["priority"], "URGENT")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST F: Dynamic Multi-Tool Sequence
    # -------------------------------------------------------------------------
    def test_f_dynamic_multi_tool_sequence(self):
        """Gemini executes a full 4-tool sequence: context -> query -> retrieve -> finish."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            patient_memory=self.patient_memory,
            enable_llm=True,
            enable_rag=True,
            max_tool_iterations=5,
        )

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            query="sepsis shock",
            passages=[
                RetrievedPassage(
                    document_id="doc_1",
                    title="Shock Guidelines",
                    source="guidelines",
                    section="Shock",
                    text="Titrate vasopressors to maintain MAP > 65.",
                    relevance_score=0.89,
                )
            ],
            retrieval_status="SUCCESS",
            top_score=0.89,
        )
        agent.retriever = mock_retriever
        agent.tools["retrieve_medical_evidence"] = lambda event_or_query, top_k=2: mock_retriever.retrieve(event_or_query)

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        decisions = [
            ClinicalReasoningToolDecision(thought="Get context", action="tool", tool_name="get_patient_context"),
            ClinicalReasoningToolDecision(thought="Build query", action="tool", tool_name="build_focused_medical_query"),
            ClinicalReasoningToolDecision(thought="Retrieve RAG", action="tool", tool_name="retrieve_medical_evidence"),
            ClinicalReasoningToolDecision(thought="Done", action="finish"),
        ]
        mock_reasoner.select_tool.side_effect = decisions
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Multi-tool analysis complete.",
            priority="URGENT",
            recommended_actions=["Consider clinician evaluation for vasopressor titration."],
            confidence=0.92,
            key_findings=["Hypotension"],
            risk_interpretation="Shock risk",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(len(trace), 4)
        tool_names = [t.get("tool_name") for t in trace[:3]]
        self.assertEqual(tool_names, ["get_patient_context", "build_focused_medical_query", "retrieve_medical_evidence"])
        self.assertEqual(trace[3]["action"], "finish")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST G: Finish Behavior
    # -------------------------------------------------------------------------
    def test_g_immediate_finish_behavior(self):
        """Gemini immediately chooses finish when existing evidence is already clear."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.return_value = ClinicalReasoningToolDecision(
            thought="Incoming vitals and trends are self-explanatory and sufficient.",
            action="finish",
            rationale="No further literature or memory needed.",
            confidence=0.95,
        )
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Hemodynamic compromise verified directly from telemetry trends.",
            priority="URGENT",
            recommended_actions=["Consider bedside evaluation."],
            confidence=0.95,
            key_findings=["Tachycardia", "Hypotension"],
            risk_interpretation="Acute deterioration",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["action"], "finish")
        self.assertEqual(result.metadata.get("agentic_mode"), "LLM_AGENTIC_TOOLS")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST H: Gemini Failure / Timeout Deterministic Fallback
    # -------------------------------------------------------------------------
    def test_h_gemini_timeout_fallback(self):
        """When Gemini times out or raises an error in tool selection, agent falls back to deterministic evaluation."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.side_effect = LLMTimeoutError("Gemini tool-selection timed out after 3.0s")
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event(risk_level="HIGH RISK")
        result = agent.process_event(event)

        self.assertIsInstance(result, ClinicalReasoningEvent)
        self.assertEqual(result.metadata.get("agentic_mode"), "LLM_FALLBACK_DETERMINISTIC")
        self.assertIn("timed out", result.metadata.get("fallback_error", ""))
        self.assertEqual(result.priority, "URGENT")  # Deterministic safety rule for HIGH RISK
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST I: Malformed / Invalid Gemini Response
    # -------------------------------------------------------------------------
    def test_i_malformed_gemini_response(self):
        """When Gemini returns unparseable garbage or invalid schema, agent falls back gracefully."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.side_effect = LLMSchemaValidationError("Malformed JSON in response")
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        self.assertEqual(result.metadata.get("agentic_mode"), "LLM_FALLBACK_DETERMINISTIC")
        self.assertIn("Malformed JSON", result.metadata.get("fallback_error", ""))
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST J: Tool Execution Failure Handling
    # -------------------------------------------------------------------------
    def test_j_tool_execution_failure_handling(self):
        """When a selected tool raises an exception, the error is recorded and agent recovers without crash."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        # Break one tool intentionally
        agent.tools["get_patient_context"] = MagicMock(side_effect=RuntimeError("Database connection lost"))

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.return_value = ClinicalReasoningToolDecision(
            thought="Fetch patient memory.",
            action="tool",
            tool_name="get_patient_context",
        )
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Deterministic evaluation recovered after tool error.",
            priority="URGENT",
            recommended_actions=["Consider bedside review."],
            confidence=0.85,
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        self.assertIsInstance(result, ClinicalReasoningEvent)
        trace = result.metadata.get("tool_trace", [])
        self.assertTrue(len(trace) > 0)
        self.assertEqual(trace[0]["status"], "error")
        self.assertIn("Database connection lost", trace[0]["error"])
        # Deterministic evaluation was still provided
        self.assertEqual(result.priority, "URGENT")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST K: Maximum Tool-Call Limit Bounding
    # -------------------------------------------------------------------------
    def test_k_max_iterations_bounding(self):
        """Tool loop terminates strictly at max_tool_iterations even if Gemini keeps calling tools."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=3,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        # Always request build_focused_medical_query
        mock_reasoner.select_tool.return_value = ClinicalReasoningToolDecision(
            thought="Keep calling query tool forever.",
            action="tool",
            tool_name="build_focused_medical_query",
        )
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Iteration capped report.",
            priority="URGENT",
            recommended_actions=["Consider bedside evaluation."],
            confidence=0.8,
            key_findings=["Hypotension"],
            risk_interpretation="Risk",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        trace = result.metadata.get("tool_trace", [])
        self.assertEqual(len(trace), 3)
        self.assertEqual(mock_reasoner.select_tool.call_count, 3)
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST L: Deterministic Safety Arbitration Overrides Unsafe LLM Proposal
    # -------------------------------------------------------------------------
    def test_l_deterministic_safety_arbitration_overrides_llm(self):
        """Even if Gemini proposes ROUTINE priority for a HIGH RISK patient, deterministic safety arbiter overrides it to URGENT."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=2,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.return_value = ClinicalReasoningToolDecision(
            thought="Sufficient information.",
            action="finish",
        )
        # Unsafe LLM attempt: ROUTINE priority for HIGH RISK case!
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Patient appears relatively stable, routine observation.",
            priority="ROUTINE",
            recommended_actions=["Continue routine vital checks."],
            confidence=0.70,
            key_findings=["Mild fluctuations"],
            risk_interpretation="Low perceived severity",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event(risk_level="HIGH RISK")
        result = agent.process_event(event)

        # Safety arbiter MUST override priority to URGENT
        self.assertEqual(result.priority, "URGENT")
        self.assertTrue(result.metadata.get("safety_arbitration_applied"))
        self.assertTrue(result.metadata.get("llm_safety_conflict"))
        self.assertIn("LLM Reasoning conflict flagged", result.findings[-1])
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST M: RAG Evidence Grounding
    # -------------------------------------------------------------------------
    def test_m_rag_evidence_grounding_passed_to_prompt(self):
        """Retrieved RAG passages from agentic loop are forwarded to final synthesis."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=True,
            max_tool_iterations=3,
        )

        mock_passage = RetrievedPassage(
            document_id="guideline_sepsis",
            title="Surviving Sepsis Campaign Guidelines",
            source="pubmed",
            section="Resuscitation",
            text="Fluid resuscitation and early antibiotic therapy within one hour.",
            relevance_score=0.95,
            metadata={"pmid": "28108562"},
        )
        agent.tools["retrieve_medical_evidence"] = lambda event_or_query, top_k=2: RetrievalResult(
            query="sepsis resuscitation",
            passages=[mock_passage],
            retrieval_status="SUCCESS",
            top_score=0.95,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        decisions = [
            ClinicalReasoningToolDecision(
                thought="Retrieve sepsis literature.",
                action="tool",
                tool_name="retrieve_medical_evidence",
            ),
            ClinicalReasoningToolDecision(thought="Finish", action="finish"),
        ]
        mock_reasoner.select_tool.side_effect = decisions
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Patient requires resuscitation consistent with sepsis bundle.",
            priority="URGENT",
            recommended_actions=["Consider clinician evaluation for fluid resuscitation."],
            confidence=0.95,
            key_findings=["Hypotension", "Tachycardia"],
            risk_interpretation="Septic shock",
            medical_evidence=[
                {
                    "title": "Surviving Sepsis Campaign Guidelines",
                    "pmid": "28108562",
                    "citation": "Surviving Sepsis Campaign Guidelines. PubMed PMID: 28108562.",
                }
            ],
            retrieval_status="SUCCESS",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event()
        result = agent.process_event(event)

        # Verify that reasoner.reason was called with the retrieval_result
        _, kwargs = mock_reasoner.reason.call_args
        self.assertIsNotNone(kwargs.get("retrieval_result"))
        self.assertTrue(kwargs["retrieval_result"].has_evidence)
        self.assertEqual(result.metadata.get("retrieval_status"), "SUCCESS")
        agent.stop()

    # -------------------------------------------------------------------------
    # TEST N: State Assessment Recording and Event Queue Dispatch
    # -------------------------------------------------------------------------
    def test_n_state_recording_and_event_publishing(self):
        """Clinical reasoning agent records assessments in ClinicalReasoningState and emits to queue."""
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=True,
            enable_rag=False,
            max_tool_iterations=2,
        )

        mock_reasoner = MagicMock(spec=GeminiClinicalReasoner)
        mock_reasoner.is_available = True
        mock_reasoner.select_tool.return_value = ClinicalReasoningToolDecision(
            thought="Direct finish.",
            action="finish",
        )
        mock_reasoner.reason.return_value = self.make_llm_result(
            clinical_summary="Standard clinical assessment.",
            priority="URGENT",
            recommended_actions=["Consider clinician evaluation."],
            confidence=0.88,
            key_findings=["Hypotension"],
            risk_interpretation="High risk",
        )
        agent.llm_reasoner = mock_reasoner

        event = self.make_analysis_event(case_id=77, event_id="evt_77")
        result = agent.process_event(event)

        patient_state = agent.patient_states.get(77)
        self.assertIsNotNone(patient_state)
        self.assertEqual(len(patient_state.previous_events), 1)
        self.assertEqual(patient_state.previous_events[0]["priority"], "URGENT")
        self.assertEqual(patient_state.last_agentic_mode, "LLM_AGENTIC_TOOLS")
        self.assertTrue(len(patient_state.latest_tool_trace) > 0)
        agent.stop()


if __name__ == "__main__":
    unittest.main()
