"""Unit tests for Monitoring Agent LLM-assisted reasoning and safety arbitration."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from communication.event_queue import EventQueue
from communication.events import MonitoringDecision, MonitoringEvent
from communication.llm_client import (
    LLMSchemaValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from monitoring_agent.monitoring_agent import MonitoringAgent
from monitoring_agent.monitoring_llm_reasoner import (
    GeminiMonitoringReasoner,
    MonitoringClassification,
    MonitoringLLMProposal,
)
from monitoring_agent.decision_engine import MonitoringDecisionEngine
from monitoring_agent.state import PatientMonitoringState


class TestMonitoringLLMReasoning(unittest.TestCase):
    """Test suite for GeminiMonitoringReasoner and MonitoringDecisionEngine arbitration."""

    def setUp(self):
        self.queue = EventQueue()
        self.reasoner = GeminiMonitoringReasoner(api_key="mock_key")

    def tearDown(self):
        self.queue.shutdown()

    def make_context(
        self,
        overall_severity: str = "critical",
        is_candidate: bool = True,
        is_persistent: bool = True,
        vitals: list[str] | None = None,
    ) -> dict:
        vitals = vitals or ["heart_rate", "blood_pressure_sys"]
        vital_details = {
            "heart_rate": {
                "current": 145.0,
                "baseline": 75.0,
                "relative_deviation": 0.93,
                "severity": "critical",
                "trend": "increasing",
                "signal_quality": "good",
            },
            "blood_pressure_sys": {
                "current": 82.0,
                "baseline": 120.0,
                "relative_deviation": -0.32,
                "severity": "moderate",
                "trend": "decreasing",
                "signal_quality": "good",
            },
        }
        return {
            "overall_severity": overall_severity,
            "is_candidate": is_candidate,
            "is_persistent": is_persistent,
            "alert_eligible_vitals": vitals,
            "vital_details": vital_details,
        }

    def test_valid_gemini_response_parsing(self):
        """Verify valid JSON from Gemini parses into MonitoringLLMProposal."""
        raw_json = json.dumps({
            "classification": "GENUINE_DETERIORATION",
            "rationale": "Sustained tachycardia with concurrent hypotension indicates hemodynamic instability.",
            "confidence": 0.92,
            "recommended_action": "Notify clinical team for rapid evaluation.",
        })
        proposal = self.reasoner.validate_and_parse(raw_json)
        self.assertEqual(proposal.classification, MonitoringClassification.GENUINE_DETERIORATION.value)
        self.assertEqual(proposal.confidence, 0.92)
        self.assertIn("hemodynamic instability", proposal.rationale)

    def test_schema_validation_invalid_classification(self):
        """Unknown classification value raises LLMSchemaValidationError."""
        raw_json = json.dumps({
            "classification": "UNKNOWN_CATEGORY",
            "rationale": "Valid rationale.",
            "confidence": 0.8,
        })
        with self.assertRaises(LLMSchemaValidationError):
            self.reasoner.validate_and_parse(raw_json)

    def test_schema_validation_confidence_bounds(self):
        """Confidence score outside [0, 1] raises LLMSchemaValidationError."""
        raw_json = json.dumps({
            "classification": "GENUINE_DETERIORATION",
            "rationale": "Valid rationale.",
            "confidence": 1.5,
        })
        with self.assertRaises(LLMSchemaValidationError):
            self.reasoner.validate_and_parse(raw_json)

    def test_schema_validation_missing_required_fields(self):
        """Missing rationale raises LLMSchemaValidationError."""
        raw_json = json.dumps({
            "classification": "GENUINE_DETERIORATION",
            "confidence": 0.9,
        })
        with self.assertRaises(LLMSchemaValidationError):
            self.reasoner.validate_and_parse(raw_json)

    def test_safety_arbitration_deterministic_escalation_overrides_llm_disagreement(self):
        """Deterministic ESCALATE_TO_RISK MUST NOT be downgraded if LLM proposes SENSOR_ARTIFACT."""
        engine = MonitoringDecisionEngine()
        context = self.make_context(overall_severity="critical", is_persistent=True)

        event = MonitoringEvent(
            patient_id=101,
            event_id="mon-test-01",
            timestamp=100.0,
            event_type="alert_started",
            severity="critical",
            affected_vitals=["heart_rate"],
            current_values={"heart_rate": 150.0},
            baseline_values={"heart_rate": 75.0},
            deviation_values={"heart_rate": 75.0},
            trends={"heart_rate": "increasing"},
            signal_quality={"heart_rate": "good"},
            persistence_duration=5,
            recommended_action="assess_patient_risk",
            metadata={"reason": "Persistent severe tachycardia"},
        )

        # Mock LLM proposal claiming this is just a sensor artifact
        llm_proposal = MonitoringLLMProposal(
            classification="SENSOR_ARTIFACT",
            rationale="Sudden jump looks like artifact.",
            confidence=0.85,
            recommended_action="Check electrode attachment.",
        )

        final_decision, final_event, final_context = engine.arbitrate(
            decision=MonitoringDecision.ESCALATE_TO_RISK,
            event=event,
            context=context,
            llm_proposal=llm_proposal,
        )

        # Invariant: final decision is STILL ESCALATE_TO_RISK
        self.assertEqual(final_decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNotNone(final_event)
        self.assertTrue(final_event.metadata.get("llm_monitoring_conflict"))
        self.assertTrue(final_event.metadata.get("verification_required"))
        self.assertIn("SAFETY ARBITRATION", final_event.metadata.get("reason", ""))
        self.assertEqual(final_event.metadata.get("reasoning_mode"), "LLM_ARBITRATED")

    def test_llm_agreement_enhances_metadata(self):
        """When LLM agrees with escalation, metadata is enriched and conflict is False."""
        engine = MonitoringDecisionEngine()
        context = self.make_context(overall_severity="critical", is_persistent=True)

        event = MonitoringEvent(
            patient_id=101,
            event_id="mon-test-02",
            timestamp=100.0,
            event_type="alert_started",
            severity="critical",
            affected_vitals=["heart_rate"],
            current_values={"heart_rate": 150.0},
            baseline_values={"heart_rate": 75.0},
            deviation_values={"heart_rate": 75.0},
            trends={"heart_rate": "increasing"},
            signal_quality={"heart_rate": "good"},
            persistence_duration=5,
            recommended_action="assess_patient_risk",
            metadata={"reason": "Persistent severe tachycardia"},
        )

        llm_proposal = MonitoringLLMProposal(
            classification="GENUINE_DETERIORATION",
            rationale="Confirmed acute physiological decline.",
            confidence=0.96,
        )

        final_decision, final_event, _ = engine.arbitrate(
            decision=MonitoringDecision.ESCALATE_TO_RISK,
            event=event,
            context=context,
            llm_proposal=llm_proposal,
        )

        self.assertEqual(final_decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertFalse(final_event.metadata.get("llm_monitoring_conflict"))
        self.assertFalse(final_event.metadata.get("verification_required"))
        self.assertEqual(final_event.metadata.get("reasoning_mode"), "LLM_ASSISTED")
        self.assertEqual(final_event.metadata.get("llm_rationale"), "Confirmed acute physiological decline.")
        self.assertEqual(final_event.metadata.get("llm_confidence"), 0.96)

    def test_agent_integration_with_mocked_llm(self):
        """Verify MonitoringAgent steps correctly with LLM enabled and mocked Gemini reasoner."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.propose.return_value = MonitoringLLMProposal(
            classification="GENUINE_DETERIORATION",
            rationale="Multi-vital instability observed.",
            confidence=0.94,
        )

        agent = MonitoringAgent(
            case_id=101,
            event_queue=self.queue,
            baseline_window=5,
            persistence_duration=2,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        # Baseline observations
        for i in range(5):
            agent.step({
                "patient_id": 101,
                "timestamp": float(i),
                "HR": 75.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Deterioration observation
        decision, event = agent.step({
            "patient_id": 101,
            "timestamp": 10.0,
            "HR": 150.0,
            "MAP": 60.0,
            "SpO2": 88.0,
            "RR": 28.0,
        })

        self.assertTrue(mock_reasoner.propose.called)

    def test_agent_fallback_on_llm_error(self):
        """Verify LLM failure cleanly falls back to deterministic decision without crashing."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.propose.side_effect = LLMTimeoutError("Request timed out")

        agent = MonitoringAgent(
            case_id=101,
            event_queue=self.queue,
            baseline_window=5,
            persistence_duration=2,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        # Agent should handle observation without raising LLMTimeoutError
        decision, event = agent.step({
            "patient_id": 101,
            "timestamp": 1.0,
            "HR": 75.0,
            "MAP": 90.0,
            "SpO2": 98.0,
            "RR": 16.0,
        })
        self.assertIn(decision, (MonitoringDecision.CONTINUE_MONITORING, MonitoringDecision.NO_ACTION))



if __name__ == "__main__":
    unittest.main()
