"""Tests for Risk-Gated Evidence Retrieval with Confidence-Triggered Deepening.

Verifies:
1. Low-risk + SUPPORTING + high confidence skips RAG (SKIPPED_LOW_RISK_CONFIDENT).
2. Low-risk + CONFLICTING triggers RAG (RUN_CONFIDENCE_ESCALATION).
3. Low-risk + UNCERTAIN triggers RAG (RUN_CONFIDENCE_ESCALATION).
4. Low-risk + low confidence (< threshold) triggers RAG (RUN_CONFIDENCE_ESCALATION).
5. High-risk always runs RAG (RUN_HIGH_RISK).
6. Fail-safe behavior on missing fields:
   - Missing/unparseable risk_level -> runs RAG (RUN_HIGH_RISK).
   - Missing evidence_consistency -> runs RAG (RUN_CONFIDENCE_ESCALATION as UNCERTAIN).
   - Missing confidence -> runs RAG (RUN_CONFIDENCE_ESCALATION with confidence=0.0).
7. Safety invariant: LLM/RAG can never downgrade a deterministic escalation.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock

from clinical_reasoning_agent.clinical_reasoning_agent import (
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    ClinicalReasoningAgent,
)
from clinical_reasoning_agent.rag.retriever import MedicalRetriever
from clinical_reasoning_agent.rag.schemas import RetrievalResult
from clinical_reasoning_agent.schemas import (
    ClinicalReasoningPriority,
    LLMReasoningResult,
)
from communication.events import DataAnalysisEvent


class SpyRetriever(MedicalRetriever):
    """MedicalRetriever spy that counts retrieve() invocations."""

    def __init__(self):
        super().__init__()
        self.retrieve_call_count = 0

    def retrieve(self, event: DataAnalysisEvent) -> RetrievalResult:
        self.retrieve_call_count += 1
        return super().retrieve(event)


class TestProgressiveEvidenceRetrieval(unittest.TestCase):
    """Test suite for progressive, risk-gated evidence retrieval."""

    def setUp(self):
        self.spy_retriever = SpyRetriever()
        self.agent = ClinicalReasoningAgent(
            retriever=self.spy_retriever,
            enable_llm=False,  # Test deterministic baseline gating directly
            enable_rag=True,
            verbose=False,
            retrieval_confidence_threshold=0.80,
        )

    def test_low_risk_supporting_high_confidence_skips_rag(self):
        """Low-risk + SUPPORTING + confidence >= 0.80 must skip RAG."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_low_stable",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
            trend_metrics={"HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 0.0}},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "SKIPPED_LOW_RISK_CONFIDENT")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 0)
        self.assertIn("RAG skipped", res.metadata.get("evidence_retrieval_reason", ""))

    def test_low_risk_conflicting_triggers_rag(self):
        """Low-risk + CONFLICTING must escalate to RAG via confidence escalation."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_low_conflicting",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="CONFLICTING",
            metadata={"confidence": 0.65},
            trend_metrics={
                "HR": {"trend": "increasing", "latest_value": 115.0, "change_over_window": 25.0},
                "MAP": {"trend": "decreasing", "latest_value": 55.0, "change_over_window": -20.0},
            },
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_CONFIDENCE_ESCALATION")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)
        self.assertIn("CONFLICTING", res.metadata.get("evidence_retrieval_reason", ""))

    def test_low_risk_uncertain_triggers_rag(self):
        """Low-risk + UNCERTAIN (e.g. data quality compromised) must trigger RAG."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_low_uncertain",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="UNCERTAIN",
            data_quality_flag=True,
            metadata={"confidence": 0.55},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_CONFIDENCE_ESCALATION")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)

    def test_low_risk_low_confidence_triggers_rag(self):
        """Low-risk + SUPPORTING but confidence below threshold (< 0.80) must trigger RAG."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_low_poor_confidence",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.70},  # Below 0.80 threshold
            trend_metrics={"HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 0.0}},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_CONFIDENCE_ESCALATION")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)
        self.assertIn("low cross-agent confidence", res.metadata.get("evidence_retrieval_reason", ""))

    def test_high_risk_always_runs_rag(self):
        """High-risk cases must always trigger RAG on the critical path."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_high_risk",
            timestamp=100.0,
            risk_level="HIGH RISK",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
            trend_metrics={"HR": {"trend": "increasing", "latest_value": 130.0, "change_over_window": 40.0}},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_HIGH_RISK")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)

    # ------------------------------------------------------------------------
    # Fail-Safe Tests: Missing Values Must Trigger Retrieval
    # ------------------------------------------------------------------------
    def test_missing_risk_level_fails_safe_to_high_risk(self):
        """Missing or unparseable risk_level must fail-safe to high-risk retrieval."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_missing_risk",
            timestamp=100.0,
            risk_level="",  # Empty / missing
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_HIGH_RISK")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)

    def test_unparseable_risk_level_fails_safe_to_high_risk(self):
        """Unparseable risk_level string must fail-safe to high-risk retrieval."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_bad_risk",
            timestamp=100.0,
            risk_level="INDETERMINATE_CORRUPTED",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_HIGH_RISK")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)

    def test_missing_evidence_consistency_fails_safe_to_uncertain(self):
        """Missing evidence_consistency must fail-safe to UNCERTAIN and trigger retrieval."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_missing_consistency",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="",  # Missing
            metadata={"confidence": 0.95},
        )

        res = self.agent.process_event(event)

        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_CONFIDENCE_ESCALATION")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)
        self.assertIn("UNCERTAIN", res.metadata.get("evidence_retrieval_reason", ""))

    def test_missing_confidence_fails_safe_to_zero(self):
        """Missing confidence must fail-safe to 0.0 and trigger confidence escalation."""
        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_missing_conf",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="SUPPORTING",
            metadata={},  # No confidence in metadata or event
        )
        # Force event.confidence to None if it has a default
        event.confidence = None

        res = self.agent.process_event(event)

        # Since confidence is 0.0, it is < threshold (0.80) -> triggers escalation
        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_CONFIDENCE_ESCALATION")
        self.assertEqual(self.spy_retriever.retrieve_call_count, 1)

    # ------------------------------------------------------------------------
    # Safety Invariant: LLM Cannot Downgrade Deterministic Escalations
    # ------------------------------------------------------------------------
    def test_safety_invariant_high_risk_llm_downgrade_rejected(self):
        """For high-risk cases, LLM suggesting ROUTINE must be rejected by safety arbiter."""
        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.reason.return_value = LLMReasoningResult(
            clinical_summary="LLM mistakenly thinks patient is fine.",
            supporting_evidence=[],
            conflicting_evidence=[],
            key_findings=[],
            risk_interpretation="Routine observation sufficient.",
            priority=ClinicalReasoningPriority.ROUTINE.value,
            recommended_actions=["Continue routine care"],
            confidence=0.99,
        )

        agent = ClinicalReasoningAgent(
            retriever=self.spy_retriever,
            llm_reasoner=mock_llm,
            enable_llm=True,
            enable_rag=True,
            verbose=False,
        )

        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_high_risk_invariant",
            timestamp=100.0,
            risk_level="HIGH RISK",
            evidence_consistency="SUPPORTING",
            trend_metrics={"HR": {"trend": "increasing", "latest_value": 140.0, "change_over_window": 45.0}},
        )

        res = agent.process_event(event)

        # Safety baseline must hold: priority cannot be downgraded to ROUTINE
        self.assertNotEqual(res.priority, ClinicalReasoningPriority.ROUTINE.value)
        self.assertEqual(res.metadata.get("llm_reasoning_conflict"), True)
        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_HIGH_RISK")

    def test_safety_invariant_confidence_escalation_llm_downgrade_rejected(self):
        """For confidence-escalated cases with deterministic verification, LLM cannot clear verification."""
        mock_llm = MagicMock()
        mock_llm.is_available = True
        mock_llm.reason.return_value = LLMReasoningResult(
            clinical_summary="LLM dismisses sensor conflict.",
            supporting_evidence=[],
            conflicting_evidence=[],
            key_findings=[],
            risk_interpretation="No action needed.",
            priority=ClinicalReasoningPriority.ROUTINE.value,
            recommended_actions=[],
            confidence=0.95,
        )

        agent = ClinicalReasoningAgent(
            retriever=self.spy_retriever,
            llm_reasoner=mock_llm,
            enable_llm=True,
            enable_rag=True,
            verbose=False,
        )

        event = DataAnalysisEvent(
            case_id=101,
            event_id="evt_conflict_invariant",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="CONFLICTING",
            verification_required=True,
            data_quality_flag=True,
        )

        res = agent.process_event(event)

        # Data quality / conflict verification cannot be suppressed
        self.assertTrue(res.verification_required)
        self.assertEqual(res.data_reliability, "COMPROMISED")
        self.assertEqual(res.metadata.get("evidence_retrieval"), "RUN_CONFIDENCE_ESCALATION")

    def test_prompt_length_recorded_in_metadata(self):
        """prompt_length_chars must be measured and recorded in metadata for both run and skipped."""
        event_skip = DataAnalysisEvent(
            case_id=101,
            event_id="evt_skip_len",
            timestamp=100.0,
            risk_level="LOW RISK",
            evidence_consistency="SUPPORTING",
            metadata={"confidence": 0.95},
        )
        res_skip = self.agent.process_event(event_skip)
        len_skip = res_skip.metadata.get("prompt_length_chars")
        self.assertIsNotNone(len_skip)
        self.assertGreater(len_skip, 0)

        event_run = DataAnalysisEvent(
            case_id=101,
            event_id="evt_run_len",
            timestamp=100.0,
            risk_level="HIGH RISK",
            evidence_consistency="SUPPORTING",
            trend_metrics={"HR": {"trend": "increasing", "latest_value": 135.0, "change_over_window": 40.0}},
            metadata={"confidence": 0.95},
        )
        res_run = self.agent.process_event(event_run)
        len_run = res_run.metadata.get("prompt_length_chars")
        self.assertIsNotNone(len_run)
        # RAG run with retrieved passages will have a longer prompt than skipped RAG
        self.assertGreater(len_run, len_skip)

    def test_event_queue_end_to_end_all_three_statuses(self):
        """End-to-end integration test through real EventQueue verifying all three retrieval statuses."""
        from communication import EventQueue
        import time

        queue = EventQueue()
        agent = ClinicalReasoningAgent(
            event_queue=queue,
            enable_llm=False,
            enable_rag=True,
            verbose=False,
            retrieval_confidence_threshold=0.80,
        )
        agent.start()

        try:
            # 1. Low risk + SUPPORTING + confidence 0.90 -> SKIPPED_LOW_RISK_CONFIDENT
            evt1 = DataAnalysisEvent(
                case_id=101,
                event_id="evt_e2e_1",
                timestamp=1.0,
                risk_level="LOW RISK",
                evidence_consistency="SUPPORTING",
                metadata={"confidence": 0.90},
            )
            queue.publish("clinical_reasoning_events", evt1)

            # 2. Low risk + CONFLICTING + confidence 0.65 -> RUN_CONFIDENCE_ESCALATION
            evt2 = DataAnalysisEvent(
                case_id=102,
                event_id="evt_e2e_2",
                timestamp=2.0,
                risk_level="LOW RISK",
                evidence_consistency="CONFLICTING",
                metadata={"confidence": 0.65},
            )
            queue.publish("clinical_reasoning_events", evt2)

            # 3. High risk -> RUN_HIGH_RISK
            evt3 = DataAnalysisEvent(
                case_id=103,
                event_id="evt_e2e_3",
                timestamp=3.0,
                risk_level="HIGH RISK",
                evidence_consistency="SUPPORTING",
                metadata={"confidence": 0.95},
                trend_metrics={"HR": {"trend": "increasing", "latest_value": 130.0, "change_over_window": 35.0}},
            )
            queue.publish("clinical_reasoning_events", evt3)

            # Wait for consumer thread to process all 3 events
            t0 = time.time()
            decisions = []
            while time.time() - t0 < 3.0:
                decisions = queue.get_history("clinical_decisions")
                if len(decisions) >= 3:
                    break
                time.sleep(0.05)

            self.assertEqual(len(decisions), 3)
            statuses = {d.case_id: d.metadata.get("evidence_retrieval") for d in decisions}
            self.assertEqual(statuses[101], "SKIPPED_LOW_RISK_CONFIDENT")
            self.assertEqual(statuses[102], "RUN_CONFIDENCE_ESCALATION")
            self.assertEqual(statuses[103], "RUN_HIGH_RISK")
        finally:
            agent.stop()


if __name__ == "__main__":
    unittest.main()
