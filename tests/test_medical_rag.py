"""Unit tests for Phase 4: Evidence-Grounded Medical RAG for the Clinical Reasoning Agent.

Covers:
1. Document loading & metadata validation.
2. Document chunking & metadata preservation.
3. Local vector indexing & cosine similarity search.
4. Relevant clinical query retrieves matching guidance (e.g. tachycardia + hypotension -> shock guideline).
5. Irrelevant query rejection (retrieval_status="NO_RELEVANT_EVIDENCE").
6. Metadata preservation on retrieved passages (document_id, title, section, score).
7. Empty knowledge base handling without crash.
8. Retriever failure handling with graceful fallback to standard LLM / deterministic reasoning.
9. No relevant evidence produces retrieval_status="NO_RELEVANT_EVIDENCE" with empty passages.
10. RAG context correctly partitioned from patient evidence in prompt.
11. Gemini output citations match retrieved sources.
12. RAG guidance cannot override deterministic HIGH RISK prediction or verification_required.
13. RAG failure falls back safely.
14. End-to-end integration: ClinicalReasoningAgent emits reasoning_mode="LLM_RAG" with citations.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from communication.event_queue import EventQueue
from communication.events import ClinicalReasoningEvent, DataAnalysisEvent
from clinical_reasoning_agent import (
    ClinicalReasoningAgent,
    ClinicalReasoningEngine,
    ClinicalReasoningPriority,
)
from clinical_reasoning_agent.llm_reasoner import (
    GeminiClinicalReasoner,
)
from clinical_reasoning_agent.prompt_builder import (
    build_evidence_package,
    build_reasoning_prompt,
)
from clinical_reasoning_agent.rag import (
    DocumentChunker,
    DocumentLoader,
    LocalMedicalEmbedder,
    LocalVectorStore,
    MedicalDocument,
    MedicalRetriever,
    RetrievalResult,
    RetrievedPassage,
    build_focused_medical_query,
)
from clinical_reasoning_agent.schemas import (
    LLMReasoningResult,
    ReasoningMode,
)


class TestMedicalRAG(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.engine = ClinicalReasoningEngine()
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(chunk_size_chars=400, overlap_chars=50)

    def tearDown(self):
        self.queue.shutdown()

    def make_event(
        self,
        event_id: str = "case_0_evt_rag_001",
        case_id: int = 0,
        risk_level: str = "HIGH RISK",
        data_quality_flag: bool = False,
        metrics: dict | None = None,
        patterns: list[str] | None = None,
        analysis_status: str = "complete",
    ) -> DataAnalysisEvent:
        if metrics is None:
            metrics = {
                "HR": {"trend": "increasing", "latest_value": 118.0, "change_over_window": 25.0},
                "MAP": {"trend": "decreasing", "latest_value": 58.0, "change_over_window": -18.0},
            }
        if patterns is None:
            patterns = ["Acute tachycardia with concurrent arterial hypotension."]
        return DataAnalysisEvent(
            case_id=case_id,
            event_id=event_id,
            timestamp=120.0,
            risk_level=risk_level,
            trend_metrics=metrics,
            pattern_identified=patterns,
            important_changes=["HR increasing", "MAP decreasing"],
            data_quality_flag=data_quality_flag,
            analysis_status=analysis_status,
        )

    def make_rag_llm_json(
        self,
        priority: str = "URGENT",
        confidence: float = 0.94,
        citations: list[dict] | None = None,
    ) -> str:
        if citations is None:
            citations = [
                {
                    "document_id": "GUIDELINE_HEMODYNAMIC_001",
                    "title": "Hemodynamic Deterioration, Acute Hypotension, and Shock Monitoring",
                    "section": "Mean Arterial Pressure (MAP) and Tachycardia Thresholds",
                }
            ]
        return json.dumps({
            "clinical_summary": "Patient demonstrates acute hemodynamic decompensation consistent with shock protocols.",
            "supporting_evidence": ["HR 118 bpm", "MAP 58 mmHg below perfusion threshold"],
            "conflicting_evidence": [],
            "key_findings": ["Tachycardia", "Hypotension", "Sub-65 MAP threshold"],
            "risk_interpretation": "Cardiovascular deterioration requiring immediate intervention per guideline.",
            "priority": priority,
            "recommended_actions": [
                "Establish large-bore venous access.",
                "Initiate balanced crystalloid bolus per Surviving Sepsis Guideline.",
            ],
            "confidence": confidence,
            "uncertainties": [],
            "knowledge_sources": citations,
            "retrieved_evidence": ["MAP below 65 mmHg is a critical clinical threshold associated with impaired autoregulation."],
        })

    # 1. Document loading & metadata validation
    def test_1_document_loading_and_metadata_validation(self):
        """Curated guidelines load correctly with required clinical metadata."""
        docs = self.loader.load_curated_guidelines()
        self.assertGreaterEqual(len(docs), 5)
        for doc in docs:
            self.assertIsInstance(doc, MedicalDocument)
            self.assertTrue(doc.doc_id.startswith("GUIDELINE_"))
            self.assertTrue(len(doc.title) > 0)
            self.assertTrue(len(doc.source) > 0)
            self.assertTrue(len(doc.text) > 50)
            self.assertIn("target_vitals", doc.metadata)

    # 2. Document chunking & metadata preservation
    def test_2_document_chunking_and_metadata_preservation(self):
        """Document chunking splits text into chunks while preserving provenance metadata."""
        docs = self.loader.load_curated_guidelines()
        chunks = self.chunker.chunk_documents(docs)
        self.assertGreater(len(chunks), len(docs))
        for chunk in chunks:
            self.assertTrue(chunk.doc_id.startswith("GUIDELINE_"))
            self.assertTrue(len(chunk.title) > 0)
            self.assertTrue(len(chunk.section) > 0)
            self.assertTrue(len(chunk.text) > 0)
            self.assertTrue(chunk.chunk_id.endswith(("_chk0", "_chk1", "_chk2", "_chk3", "_chk4")))

    # 3. Local vector indexing & cosine similarity search
    def test_3_local_vector_indexing_and_cosine_similarity(self):
        """Vector store indexes chunks and calculates cosine similarity without external servers."""
        docs = self.loader.load_curated_guidelines()
        chunks = self.chunker.chunk_documents(docs)
        embedder = LocalMedicalEmbedder()
        store = LocalVectorStore(embedder=embedder, min_similarity=0.05)
        store.index_chunks(chunks)
        self.assertEqual(store.total_chunks, len(chunks))

        results = store.search("hypotension mean arterial pressure shock", top_k=3)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertIsInstance(top_result, RetrievedPassage)
        self.assertGreater(top_result.similarity_score, 0.05)
        self.assertTrue("MAP" in top_result.text or "pressure" in top_result.text.lower() or "shock" in top_result.text.lower())

    # 4. Relevant clinical query retrieves matching guidance
    def test_4_relevant_clinical_query_retrieves_matching_guidance(self):
        """Tachycardia and hypotension query retrieves shock/hemodynamic instability guideline."""
        retriever = MedicalRetriever(min_similarity=0.10, enable_pubmed=False)
        query = "hypotension arterial pressure MAP shock fluid resuscitation"
        result = retriever.retrieve(query, top_k=2)

        self.assertEqual(result.retrieval_status, "SUCCESS")
        self.assertGreater(len(result.passages), 0)
        doc_ids = [p.doc_id for p in result.passages]
        self.assertIn("GUIDELINE_HEMODYNAMIC_001", doc_ids)

    # 5. Irrelevant query rejection
    def test_5_irrelevant_query_rejection(self):
        """Completely irrelevant non-clinical query returns NO_RELEVANT_EVIDENCE."""
        retriever = MedicalRetriever(min_similarity=0.25)
        query = "astronomy orbital mechanics telescope galaxy photography"
        result = retriever.retrieve(query, top_k=3)

        self.assertEqual(result.retrieval_status, "NO_RELEVANT_EVIDENCE")
        self.assertEqual(len(result.passages), 0)

    # 6. Metadata preservation on retrieved passages
    def test_6_metadata_preservation_on_retrieved_passages(self):
        """Retrieved passages preserve doc_id, title, section, and similarity_score."""
        retriever = MedicalRetriever(min_similarity=0.05, enable_pubmed=False)
        result = retriever.retrieve("acute hypoxia oxygen desaturation SpO2", top_k=1)

        self.assertEqual(result.retrieval_status, "SUCCESS")
        self.assertEqual(len(result.passages), 1)
        p = result.passages[0]
        self.assertEqual(p.doc_id, "GUIDELINE_RESPIRATORY_003")
        self.assertIn("Respiratory", p.title)
        self.assertIsInstance(p.similarity_score, float)
        self.assertGreater(p.similarity_score, 0.0)

    # 7. Empty knowledge base handling without crash
    def test_7_empty_knowledge_base_handling(self):
        """Retriever with empty document list handles search gracefully without crash."""
        empty_loader = MagicMock()
        empty_loader.load_documents.return_value = []
        retriever = MedicalRetriever(loader=empty_loader, min_similarity=0.1, enable_pubmed=False)

        result = retriever.retrieve("hemodynamic shock", top_k=3)
        self.assertEqual(result.retrieval_status, "NO_RELEVANT_EVIDENCE")
        self.assertEqual(len(result.passages), 0)

    # 8. Retriever failure handling with graceful fallback
    def test_8_retriever_failure_handling_with_graceful_fallback(self):
        """Retriever exception triggers RETRIEVAL_FAILED status without crashing."""
        retriever = MedicalRetriever(enable_pubmed=False)
        with patch.object(retriever.vector_store, "search", side_effect=RuntimeError("Store corruption")):
            result = retriever.retrieve("tachycardia", top_k=2)
            self.assertEqual(result.retrieval_status, "RETRIEVAL_FAILED")
            self.assertEqual(len(result.passages), 0)

    # 9. No relevant evidence produces retrieval_status="NO_RELEVANT_EVIDENCE"
    def test_9_no_relevant_evidence_status(self):
        """When similarity scores are below threshold, result is NO_RELEVANT_EVIDENCE."""
        retriever = MedicalRetriever(min_similarity=0.99, enable_pubmed=False)  # impossibly high threshold
        result = retriever.retrieve("hemodynamics", top_k=2)
        self.assertEqual(result.retrieval_status, "NO_RELEVANT_EVIDENCE")
        self.assertEqual(len(result.passages), 0)

    # 10. RAG context correctly separated from patient evidence in prompt
    def test_10_rag_context_separated_from_patient_evidence_in_prompt(self):
        """Prompt distinctly separates PATIENT EVIDENCE from RETRIEVED MEDICAL KNOWLEDGE."""
        event = self.make_event()
        evidence = build_evidence_package(event)
        retriever = MedicalRetriever(min_similarity=0.05)
        query = build_focused_medical_query(event)
        retrieval_result = retriever.retrieve(query, top_k=2)

        prompt = build_reasoning_prompt(evidence, retrieval_result=retrieval_result)
        self.assertIn("=== SECTION 1: PATIENT EVIDENCE", prompt)
        self.assertIn("=== SECTION 2: RETRIEVED MEDICAL KNOWLEDGE", prompt)
        self.assertIn("NOT Patient Facts", prompt)
        self.assertIn("GUIDELINE_", prompt)

    # 11. Gemini output citations match retrieved sources
    def test_11_gemini_output_citations_match_retrieved_sources(self):
        """Gemini clinical reasoner validates citations against retrieved documents."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = self.make_rag_llm_json(
            citations=[
                {
                    "document_id": "GUIDELINE_HEMODYNAMIC_001",
                    "title": "Hemodynamic Deterioration, Acute Hypotension, and Shock Monitoring",
                    "section": "Target MAP",
                }
            ]
        )
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        event = self.make_event()
        evidence = build_evidence_package(event)

        retrieval_result = RetrievalResult(
            query="hypotension shock",
            passages=[
                RetrievedPassage(
                    document_id="GUIDELINE_HEMODYNAMIC_001",
                    title="Hemodynamic Deterioration, Acute Hypotension, and Shock Monitoring",
                    source="Intensive Care Guidelines",
                    section="Target MAP",
                    text="MAP below 65 mmHg warrants immediate resuscitation.",
                    relevance_score=0.85,
                    metadata={},
                )
            ],
            retrieval_status="SUCCESS",
        )

        result = reasoner.reason(evidence, retrieval_result=retrieval_result)
        self.assertIsInstance(result, LLMReasoningResult)
        self.assertEqual(result.retrieval_status, "SUCCESS")
        self.assertEqual(len(result.knowledge_sources), 1)
        self.assertEqual(result.knowledge_sources[0]["document_id"], "GUIDELINE_HEMODYNAMIC_001")
        self.assertEqual(len(result.retrieved_evidence), 1)

    # 12. RAG guidance cannot override deterministic HIGH RISK prediction or verification_required
    def test_12_rag_guidance_cannot_override_deterministic_high_risk(self):
        """RAG guidance cannot downgrade HIGH RISK priority or bypass cross-agent verification."""
        event = self.make_event(risk_level="HIGH RISK")
        deterministic = self.engine.analyze_case(event)

        # LLM suggests ROUTINE citing some guideline
        llm_result = LLMReasoningResult(
            clinical_summary="Patient is stable per routine guideline.",
            supporting_evidence=[],
            conflicting_evidence=[],
            key_findings=["Stable"],
            risk_interpretation="Low risk according to literature.",
            priority="ROUTINE",  # Conflicting with HIGH RISK
            recommended_actions=["Routine checks."],
            confidence=0.90,
            uncertainties=[],
            knowledge_sources=[{"document_id": "GUIDELINE_NEWS2_005", "title": "NEWS2", "section": "Routine"}],
            retrieved_evidence=["NEWS2 routine observation schedule."],
            retrieval_status="SUCCESS",
        )

        output = self.engine.arbitrate(
            deterministic=deterministic,
            llm_result=llm_result,
            risk_level="HIGH RISK",
            data_quality_flag=False,
            fallback_error=None,
        )

        # Deterministic safety rule enforces ELEVATED/URGENT and verification_required
        self.assertIn(output.priority, (ClinicalReasoningPriority.URGENT.value, ClinicalReasoningPriority.ELEVATED.value))
        self.assertNotEqual(output.priority, ClinicalReasoningPriority.ROUTINE.value)
        self.assertTrue(output.verification_required)
        self.assertTrue(output.llm_reasoning_conflict)
        self.assertEqual(output.reasoning_mode, ReasoningMode.LLM_RAG.value)
        self.assertTrue(any("GUIDELINE_NEWS2_005" in str(s) for s in output.knowledge_sources))

    # 13. RAG failure falls back safely
    def test_13_rag_failure_falls_back_safely(self):
        """When RAG retrieval fails, reasoner proceeds with standard LLM synthesis without failure."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = self.make_rag_llm_json(citations=[])
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        event = self.make_event()
        evidence = build_evidence_package(event)

        failed_retrieval = RetrievalResult(
            query="shock",
            passages=[],
            retrieval_status="RETRIEVAL_FAILED",
            error_message="Vector index read error",
        )

        result = reasoner.reason(evidence, retrieval_result=failed_retrieval)
        self.assertIsInstance(result, LLMReasoningResult)
        self.assertEqual(result.retrieval_status, "RETRIEVAL_FAILED")
        self.assertEqual(len(result.knowledge_sources), 0)

        # Arbitrator should recognize that retrieval failed and produce LLM_ASSISTED
        deterministic = self.engine.analyze_case(event)
        output = self.engine.arbitrate(
            deterministic=deterministic,
            llm_result=result,
            risk_level="HIGH RISK",
            data_quality_flag=False,
            fallback_error=None,
        )
        self.assertEqual(output.reasoning_mode, ReasoningMode.LLM_ASSISTED.value)

    # 14. End-to-end integration: ClinicalReasoningAgent emits reasoning_mode="LLM_RAG" with citations
    @patch("clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts", return_value=None)
    def test_14_end_to_end_agent_emits_llm_rag_with_citations(self, _mock_pubmed):
        """ClinicalReasoningAgent processes DataAnalysisEvent, runs RAG, and emits ClinicalReasoningEvent with LLM_RAG."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = self.make_rag_llm_json(
            citations=[
                {
                    "document_id": "GUIDELINE_HEMODYNAMIC_001",
                    "title": "Hemodynamic Deterioration, Acute Hypotension, and Shock Monitoring",
                    "section": "Mean Arterial Pressure (MAP) and Tachycardia Thresholds",
                }
            ]
        )
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            llm_reasoner=reasoner,
            enable_llm=True,
            enable_rag=True,
        )

        event = self.make_event()
        agent.process_event(event)

        history = self.queue.get_history("clinical_decisions")
        self.assertEqual(len(history), 1)
        decision: ClinicalReasoningEvent = history[0]

        self.assertEqual(decision.metadata.get("reasoning_mode"), "LLM_RAG")
        self.assertEqual(decision.metadata.get("retrieval_status"), "SUCCESS")
        sources = decision.metadata.get("knowledge_sources", [])
        self.assertGreater(len(sources), 0)
        self.assertEqual(sources[0]["document_id"], "GUIDELINE_HEMODYNAMIC_001")
        retrieved = decision.metadata.get("retrieved_evidence", [])
        self.assertGreater(len(retrieved), 0)


if __name__ == "__main__":
    unittest.main()
