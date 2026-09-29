"""Unit tests for Phase 3: LLM-Assisted Clinical Reasoning and Safety Arbitration.

Covers:
1. Valid Gemini response parsing and structured result generation.
2. Structured output schema validation (types, priorities, ranges).
3. Invalid / malformed JSON response triggers safe deterministic fallback.
4. Missing required fields in Gemini response triggers safe fallback.
5. Gemini API failure (exception) triggers safe deterministic fallback.
6. Gemini request timeout triggers safe fallback.
7. LLM output conflicting with deterministic risk (LLM suggests ROUTINE for HIGH RISK)
   triggers safety arbitration, preserves safety baseline, and flags llm_reasoning_conflict=True.
8. High-risk + supporting evidence synthesis via LLM.
9. Low-risk + stable evidence synthesis via LLM.
10. High-risk + conflicting evidence maintains verification_required=True.
11. Poor data quality forces data_reliability="COMPROMISED" and bounds confidence.
12. Deterministic mode when no API key / LLM disabled.
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
    LLMReasonerError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from clinical_reasoning_agent.prompt_builder import (
    build_evidence_package,
    build_reasoning_prompt,
)
from clinical_reasoning_agent.schemas import (
    LLMReasoningResult,
    ReasoningMode,
)


class TestLLMClinicalReasoning(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.engine = ClinicalReasoningEngine()

    def tearDown(self):
        self.queue.shutdown()

    def make_event(
        self,
        event_id: str = "case_0_evt_001",
        case_id: int = 0,
        risk_level: str = "HIGH RISK",
        data_quality_flag: bool = False,
        metrics: dict | None = None,
        patterns: list[str] | None = None,
        analysis_status: str = "complete",
    ) -> DataAnalysisEvent:
        if metrics is None:
            metrics = {
                "HR": {"trend": "increasing", "latest_value": 110.0, "change_over_window": 20.0},
                "MAP": {"trend": "decreasing", "latest_value": 60.0, "change_over_window": -15.0},
            }
        if patterns is None:
            patterns = ["Simultaneous directional changes across multiple affected vitals."]
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

    def make_valid_llm_json(
        self,
        priority: str = "URGENT",
        confidence: float = 0.95,
        summary: str = "Patient shows acute tachycardia and hypotension requiring immediate clinical review.",
    ) -> str:
        return json.dumps({
            "clinical_summary": summary,
            "supporting_evidence": ["HR increased to 110 bpm", "MAP decreased to 60 mmHg"],
            "conflicting_evidence": [],
            "key_findings": ["Tachycardia trajectory", "MAP decline"],
            "risk_interpretation": "Cardiovascular deterioration meeting high-risk threshold.",
            "priority": priority,
            "recommended_actions": [
                "Immediate bedside hemodynamic evaluation.",
                "Prepare fluid resuscitation protocol.",
            ],
            "confidence": confidence,
            "uncertainties": [],
        })

    # 1. Valid Gemini response parsing
    def test_1_valid_gemini_response_parsing(self):
        """Valid JSON from Gemini parses correctly into LLMReasoningResult."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = self.make_valid_llm_json()
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        event = self.make_event()
        evidence = build_evidence_package(event)

        result = reasoner.reason(evidence)
        self.assertIsInstance(result, LLMReasoningResult)
        self.assertEqual(result.priority, "URGENT")
        self.assertAlmostEqual(result.confidence, 0.95)
        self.assertEqual(len(result.recommended_actions), 2)
        self.assertTrue(any("tachycardia" in f.lower() for f in result.key_findings))

    # 2. Structured output schema validation
    def test_2_structured_output_schema_validation(self):
        """Reasoner validates priority types, confidence bounds, and required keys."""
        reasoner = GeminiClinicalReasoner(client=MagicMock())

        # Invalid priority string
        invalid_priority_json = json.dumps({
            "clinical_summary": "Test",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "key_findings": [],
            "risk_interpretation": "Test",
            "priority": "INVALID_PRIORITY",
            "recommended_actions": [],
            "confidence": 0.8,
        })
        with self.assertRaises(LLMSchemaValidationError):
            reasoner.validate_and_parse(invalid_priority_json)

        # Invalid confidence bounds (> 1.0)
        invalid_conf_json = json.dumps({
            "clinical_summary": "Test",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "key_findings": [],
            "risk_interpretation": "Test",
            "priority": "ROUTINE",
            "recommended_actions": [],
            "confidence": 1.5,
        })
        with self.assertRaises(LLMSchemaValidationError):
            reasoner.validate_and_parse(invalid_conf_json)

    # 3. Invalid JSON response triggers safe deterministic fallback
    def test_3_invalid_json_fallback(self):
        """Malformed JSON from Gemini triggers safe fallback without agent crash."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = "Not valid JSON at all!"
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event()
        decision = agent.process_event(event)

        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.LLM_FALLBACK.value)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertTrue(decision.escalation_required)
        self.assertIn("llm_fallback_reason", decision.metadata)

    # 4. Missing required fields triggers fallback
    def test_4_missing_required_fields_fallback(self):
        """Incomplete JSON payload triggers schema validation error and deterministic fallback."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        # Missing 'recommended_actions' and 'priority'
        mock_interaction.output_text = json.dumps({
            "clinical_summary": "Summary only",
            "confidence": 0.9,
        })
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event()
        decision = agent.process_event(event)

        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.LLM_FALLBACK.value)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)

    # 5. Gemini API failure triggers safe fallback
    def test_5_gemini_api_failure_fallback(self):
        """Network or API error when invoking Gemini triggers deterministic fallback."""
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = RuntimeError("Google API 503 Service Unavailable")

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event()
        decision = agent.process_event(event)

        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.LLM_FALLBACK.value)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertIn("503 Service Unavailable", decision.metadata.get("llm_fallback_reason", ""))

    # 6. Timeout triggers safe fallback
    def test_6_gemini_timeout_fallback(self):
        """Request timeout when invoking Gemini triggers deterministic fallback."""
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = TimeoutError("Request timed out after 10.0s")

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event()
        decision = agent.process_event(event)

        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.LLM_FALLBACK.value)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)

    # 7. LLM output conflicting with deterministic risk (LLM says ROUTINE for HIGH RISK)
    def test_7_llm_conflicts_with_deterministic_risk_safety_arbitration(self):
        """Deterministic safety overrides LLM when LLM proposes ROUTINE for HIGH RISK."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        # LLM erroneously claims ROUTINE priority for HIGH RISK patient
        mock_interaction.output_text = self.make_valid_llm_json(
            priority="ROUTINE",
            confidence=0.90,
            summary="Patient seems fine, routine observation.",
        )
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event(risk_level="HIGH RISK")
        decision = agent.process_event(event)

        # Deterministic safety rule enforces URGENT priority and overrides LLM
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertTrue(decision.metadata.get("llm_reasoning_conflict"))
        self.assertIn(decision.metadata.get("reasoning_mode"), [ReasoningMode.LLM_ASSISTED.value, ReasoningMode.LLM_RAG.value])
        self.assertTrue(any("SAFETY ARBITRATION" in decision.clinical_summary for _ in [1]))
        self.assertTrue(any("conflicting" in c.lower() for c in decision.conflicting_evidence))

    # 8. High-risk + supporting evidence via LLM
    def test_8_high_risk_supporting_evidence_llm(self):
        """HIGH RISK with supporting vitals produces LLM-assisted URGENT clinical decision."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = self.make_valid_llm_json(
            priority="URGENT",
            confidence=0.96,
            summary="Critical cardiovascular instability confirmed by multi-vital deterioration.",
        )
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event(risk_level="HIGH RISK")
        decision = agent.process_event(event)

        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertIn(decision.metadata.get("reasoning_mode"), [ReasoningMode.LLM_ASSISTED.value, ReasoningMode.LLM_RAG.value])
        self.assertFalse(decision.metadata.get("llm_reasoning_conflict"))
        self.assertTrue(decision.escalation_required)
        self.assertIn("Critical cardiovascular instability", decision.clinical_summary)

    # 9. Low-risk + stable evidence via LLM
    @patch("clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts", return_value=None)
    def test_9_low_risk_stable_evidence_llm(self, _mock_pubmed):
        """LOW RISK with stable vitals produces LLM-assisted ROUTINE monitoring decision."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = json.dumps({
            "clinical_summary": "Vital parameters stable. Standard automated surveillance recommended.",
            "supporting_evidence": ["Stable heart rate and blood pressure."],
            "conflicting_evidence": [],
            "key_findings": ["All metrics within normal baseline tolerance."],
            "risk_interpretation": "Patient remains physiologically stable under model evaluation.",
            "priority": "ROUTINE",
            "recommended_actions": ["Continue standard monitoring."],
            "confidence": 0.95,
            "uncertainties": [],
        })
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event(
            risk_level="LOW RISK",
            metrics={"HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 0.0}},
            patterns=["Stable baseline vitals."],
        )
        decision = agent.process_event(event)

        self.assertEqual(decision.priority, ClinicalReasoningPriority.ROUTINE.value)
        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.LLM_ASSISTED.value)
        self.assertFalse(decision.escalation_required)
        self.assertFalse(decision.metadata.get("llm_reasoning_conflict"))

    # 10. High-risk + conflicting evidence maintains verification_required=True
    @patch("clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts", return_value=None)
    def test_10_high_risk_conflicting_evidence_preserves_verification(self, _mock_pubmed):
        """HIGH RISK with conflicting evidence enforces verification_required=True regardless of LLM."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = json.dumps({
            "clinical_summary": "Model indicates high risk but vital trends appear stable.",
            "supporting_evidence": [],
            "conflicting_evidence": ["All vital signs stable"],
            "key_findings": ["Discordance between model score and flat physiological trajectories."],
            "risk_interpretation": "Potential false positive or early phase pre-symptomatic deviation.",
            "priority": "ELEVATED",
            "recommended_actions": ["Verify sensor placement", "Re-evaluate in 5 minutes"],
            "confidence": 0.70,
            "uncertainties": ["Signal flatline versus true physiological stability"],
        })
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event(
            risk_level="HIGH RISK",
            metrics={"HR": {"trend": "stable", "latest_value": 72.0, "change_over_window": 0.0}},
            patterns=["Monitored vitals remain stable."],
        )
        decision = agent.process_event(event)

        self.assertEqual(decision.priority, ClinicalReasoningPriority.ELEVATED.value)
        self.assertTrue(decision.verification_required)
        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.LLM_ASSISTED.value)

    # 11. Poor data quality forces data_reliability="COMPROMISED"
    def test_11_poor_data_quality_enforces_compromised_reliability(self):
        """Data quality flag forces COMPROMISED reliability and bounds confidence <= 0.60."""
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_text = json.dumps({
            "clinical_summary": "Sensor noise detected across leads.",
            "supporting_evidence": [],
            "conflicting_evidence": ["Data artifact flag active"],
            "key_findings": ["Lead impedance artifact"],
            "risk_interpretation": "Cannot reliably compute risk due to noise.",
            "priority": "ELEVATED",
            "recommended_actions": ["Inspect leads and probes."],
            "confidence": 0.90,  # LLM overconfidently claims 0.90
            "uncertainties": ["Severe sensor missingness"],
        })
        mock_client.interactions.create.return_value = mock_interaction

        reasoner = GeminiClinicalReasoner(client=mock_client)
        agent = ClinicalReasoningAgent(event_queue=self.queue, llm_reasoner=reasoner)

        event = self.make_event(risk_level="LOW RISK", data_quality_flag=True)
        decision = agent.process_event(event)

        # Safety boundary forces COMPROMISED reliability and clamps confidence to <= 0.60
        self.assertEqual(decision.data_reliability, "COMPROMISED")
        self.assertTrue(decision.verification_required)
        self.assertLessEqual(decision.confidence, 0.60)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.ELEVATED.value)

    # 12. Deterministic mode when no API key / LLM disabled
    def test_12_deterministic_mode_when_llm_disabled(self):
        """When LLM is disabled or unconfigured, agent operates in pure DETERMINISTIC mode."""
        reasoner = GeminiClinicalReasoner(api_key=None, client=None)
        agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            llm_reasoner=reasoner,
            enable_llm=False,
        )

        event = self.make_event()
        decision = agent.process_event(event)

        self.assertEqual(decision.metadata.get("reasoning_mode"), ReasoningMode.DETERMINISTIC.value)
        self.assertEqual(decision.priority, ClinicalReasoningPriority.URGENT.value)
        self.assertTrue(decision.escalation_required)
        self.assertFalse(decision.metadata.get("llm_reasoning_conflict"))


if __name__ == "__main__":
    unittest.main()

