"""Tests for cross-agent challenge-response negotiation and bounded real-time arbitration."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from clinical_reasoning_agent import ClinicalReasoningAgent
from communication.event_queue import EventQueue
from communication.events import (
    DataAnalysisEvent,
    MonitoringEvent,
    PerformativeType,
    RiskDecision,
    RiskDecisionEvent,
)
from data_analysis_agent import DataAnalysisAgent
from risk_agent import RiskAgent
from risk_agent.risk_llm_reasoner import (
    GeminiRiskReasoner,
    RiskChallengeResponse,
    RiskLLMProposal,
)


def make_stable_window():
    """Return a DataFrame where vitals remain completely stable."""
    points = 61
    return pd.DataFrame({
        "Time": np.arange(points) * 5,
        "HR": np.full(points, 75.0),
        "SpO2": np.full(points, 98.0),
        "RR": np.full(points, 14.0),
        "SBP": np.full(points, 120.0),
        "DBP": np.full(points, 75.0),
        "MAP": np.full(points, 85.0),
        "BT": np.full(points, 36.7),
    })


def make_worsening_window():
    """Return a DataFrame where multiple vitals exhibit worsening trajectories."""
    points = 61
    return pd.DataFrame({
        "Time": np.arange(points) * 5,
        "HR": np.linspace(70, 115, points),    # Tachycardia
        "SpO2": np.linspace(98, 89, points),   # Desaturation
        "RR": np.linspace(14, 28, points),     # Tachypnea
        "SBP": np.linspace(120, 85, points),   # Hypotension
        "DBP": np.linspace(80, 50, points),
        "MAP": np.linspace(90, 60, points),
        "BT": np.full(points, 37.0),
    })


class TestAgentNegotiation(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()
        self.risk_agent = RiskAgent(event_queue=self.queue, verbose=False)
        self.analysis_agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_stable_window(),
            verbose=False,
        )
        self.clinical_agent = ClinicalReasoningAgent(
            event_queue=self.queue,
            enable_llm=False,
            enable_rag=False,
            verbose=False,
        )

    def test_conflicting_scenario_triggers_challenge_round(self):
        """Verify that a model HIGH RISK prediction alongside stable vitals triggers a CHALLENGE event."""
        # Initial model assessment claims HIGH RISK
        high_risk_event = RiskDecisionEvent(
            patient_id=101,
            event_id="case_101_evt_1",
            timestamp=time.time(),
            decision="HIGH_RISK",
            risk_probability=0.88,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            severity="moderate",
            affected_vitals=["HR"],
            window_start=0,
            window_end=300,
            performative=PerformativeType.INFORM.value,
            challenge_round=0,
        )

        # DataAnalysisAgent evaluates stable window data against HIGH RISK
        analysis_result = self.analysis_agent.process_event(high_risk_event)

        # Evidence should be conflicting (HIGH RISK with stable vitals)
        self.assertEqual(analysis_result.evidence_consistency, "CONFLICTING")
        self.assertEqual(analysis_result.performative, PerformativeType.CHALLENGE.value)
        self.assertEqual(analysis_result.challenge_round, 1)

        # Challenge must be published to 'risk_challenges'
        challenges = self.queue.get_history("risk_challenges")
        self.assertGreaterEqual(len(challenges), 1)
        challenge_on_queue = challenges[0]
        self.assertEqual(challenge_on_queue.performative, PerformativeType.CHALLENGE.value)
        self.assertEqual(challenge_on_queue.challenge_round, 1)
        self.assertEqual(challenge_on_queue.event_id, "case_101_evt_1")

    def test_risk_agent_handles_challenge_and_responds_with_agree(self):
        """Verify RiskAgent responds to a challenge by re-evaluating and producing AGREE/PROPOSE."""
        # 1. Establish prior decision
        prior_decision = RiskDecisionEvent(
            patient_id=101,
            event_id="case_101_evt_2",
            timestamp=time.time(),
            decision="HIGH_RISK",
            risk_probability=0.82,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            features_used={"HR_mean": 95.0, "MAP_mean": 65.0},
            performative=PerformativeType.INFORM.value,
            challenge_round=0,
        )
        self.risk_agent.state.record_decision(prior_decision, features={"HR_mean": 95.0, "MAP_mean": 65.0})

        # 2. Challenge from DataAnalysisAgent
        challenge = DataAnalysisEvent(
            case_id=101,
            event_id="case_101_evt_2",
            timestamp=time.time(),
            risk_level="HIGH RISK",
            conflict_flags=["HIGH_RISK_WITH_STABLE_TRENDS"],
            evidence_consistency="CONFLICTING",
            verification_required=True,
            metadata={"conflicting_evidence": ["Monitored vitals remain stable over assessment window."]},
            performative=PerformativeType.CHALLENGE.value,
            challenge_round=1,
        )

        # 3. Handle challenge
        response = self.risk_agent.handle_challenge(challenge)
        self.assertIsNotNone(response)
        self.assertEqual(response.challenge_round, 1)
        self.assertEqual(response.performative, PerformativeType.AGREE.value)
        self.assertIn("HIGH_RISK", response.decision)
        self.assertIn("confirmed", response.reason.lower())

        # Response must be published back to risk_decisions
        decisions_on_queue = self.queue.get_history("risk_decisions")
        self.assertGreaterEqual(len(decisions_on_queue), 1)
        self.assertEqual(decisions_on_queue[-1].performative, PerformativeType.AGREE.value)

    def test_challenge_round_cap_strictly_enforced(self):
        """Hard real-time safety bound: cap challenge-response to at most ONE round per event chain."""
        # A challenge with challenge_round > 1 must be refused by RiskAgent
        excess_challenge = DataAnalysisEvent(
            case_id=101,
            event_id="case_101_evt_excess",
            timestamp=time.time(),
            risk_level="HIGH RISK",
            conflict_flags=["HIGH_RISK_WITH_STABLE_TRENDS"],
            performative=PerformativeType.CHALLENGE.value,
            challenge_round=2,  # Past round 1
        )
        resp = self.risk_agent.handle_challenge(excess_challenge)
        self.assertIsNone(resp, "RiskAgent must refuse challenges past round 1")

        # DataAnalysisAgent must also refuse to challenge again when challenge_round >= 1
        already_challenged_risk_event = RiskDecisionEvent(
            patient_id=101,
            event_id="case_101_evt_excess",
            timestamp=time.time(),
            decision="HIGH_RISK",
            risk_probability=0.85,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            performative=PerformativeType.AGREE.value,
            challenge_round=1,  # Already at round 1
        )
        self.queue.clear()
        analysis_result = self.analysis_agent.process_event(already_challenged_risk_event)
        # It must NOT publish another challenge to risk_challenges
        self.assertEqual(len(self.queue.get_history("risk_challenges")), 0)
        self.assertEqual(analysis_result.performative, PerformativeType.INFORM.value)

    def test_clinical_reasoning_surfaces_negotiation_trace(self):
        """Verify that ClinicalReasoningEvent.metadata contains the full negotiation dialogue and resolution."""
        # 1. Simulate initial Risk decision
        initial_risk = RiskDecisionEvent(
            patient_id=102,
            event_id="case_102_evt_3",
            timestamp=300.0,
            decision="HIGH_RISK",
            risk_probability=0.78,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            window_start=0,
            window_end=300,
            performative=PerformativeType.INFORM.value,
            challenge_round=0,
        )
        self.risk_agent.state.record_decision(initial_risk, features={"HR_mean": 80.0})

        # 2. DataAnalysisAgent detects conflict and challenges
        challenge_event = self.analysis_agent.process_event(initial_risk)
        self.assertEqual(challenge_event.performative, PerformativeType.CHALLENGE.value)

        # 3. RiskAgent processes challenge and responds
        risk_response = self.risk_agent.handle_challenge(challenge_event)
        self.assertIsNotNone(risk_response)

        # 4. DataAnalysisAgent processes resolved risk decision
        final_analysis = self.analysis_agent.process_event(risk_response)
        self.assertIn("negotiation_trace", final_analysis.metadata)

        # 5. ClinicalReasoningAgent synthesizes final decision
        clinical_event = self.clinical_agent.process_event(final_analysis)

        # Assert negotiation trace is surfaced
        self.assertIn("negotiation_trace", clinical_event.metadata)
        trace = clinical_event.metadata["negotiation_trace"]
        self.assertEqual(trace["status"], "RESOLVED")
        self.assertEqual(trace["challenge_round"], 1)
        self.assertGreaterEqual(len(trace["dialogue"]), 2)
        self.assertEqual(trace["dialogue"][0]["performative"], PerformativeType.CHALLENGE.value)
        self.assertEqual(trace["dialogue"][1]["performative"], PerformativeType.AGREE.value)
        self.assertIn("resolution_summary", trace)

        # Clinical findings should reflect resolution
        self.assertTrue(any("Challenge Resolution" in f for f in clinical_event.findings))

    def test_mocked_llm_risk_challenge_resolution_propose(self):
        """Verify RiskAgent with LLM reasoning enabled proposes revision when advised by LLM."""
        mock_llm = MagicMock(spec=GeminiRiskReasoner)
        mock_llm.is_available = True
        mock_llm.re_evaluate_challenge.return_value = RiskChallengeResponse(
            decision="PROPOSE",
            revised_risk_level="LOW RISK",
            rationale="Short-term vital stability overrides transient baseline noise; downgrading risk.",
            confidence=0.92,
        )

        llm_risk_agent = RiskAgent(
            event_queue=self.queue,
            enable_llm=True,
            llm_reasoner=mock_llm,
            verbose=False,
        )

        prior_decision = RiskDecisionEvent(
            patient_id=103,
            event_id="case_103_evt_4",
            timestamp=time.time(),
            decision="HIGH_RISK",
            risk_probability=0.22,
            risk_level="HIGH RISK",
            threshold=0.16,
            model_name="RandomForestClassifier",
            performative=PerformativeType.INFORM.value,
            challenge_round=0,
        )
        llm_risk_agent.state.record_decision(prior_decision, features={"HR_mean": 72.0})

        challenge = DataAnalysisEvent(
            case_id=103,
            event_id="case_103_evt_4",
            timestamp=time.time(),
            risk_level="HIGH RISK",
            conflict_flags=["HIGH_RISK_WITH_STABLE_TRENDS"],
            performative=PerformativeType.CHALLENGE.value,
            challenge_round=1,
        )

        response = llm_risk_agent.handle_challenge(challenge)
        self.assertIsNotNone(response)
        self.assertEqual(response.performative, PerformativeType.PROPOSE.value)
        self.assertEqual(response.decision, "LOW_RISK")
        self.assertEqual(response.risk_level, "LOW RISK")
        self.assertIn("downgrading risk", response.reason)
        mock_llm.re_evaluate_challenge.assert_called_once()

    def test_asynchronous_worker_thread_negotiation(self):
        """Verify full negotiation cycle operates autonomously via background worker threads."""
        self.risk_agent.start()
        self.analysis_agent.start()
        self.clinical_agent.start()

        try:
            # Emit a conflicting initial risk decision
            initial_decision = RiskDecisionEvent(
                patient_id=105,
                event_id="async_case_105",
                timestamp=300.0,
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
            # Store in risk agent state so it can correlate
            self.risk_agent.state.record_decision(initial_decision, features={"HR_mean": 75.0})

            # Publish to trigger DataAnalysisAgent
            self.queue.publish("risk_decisions", initial_decision)

            # Wait briefly for autonomous workers to negotiate:
            # DataAnalysis (CHALLENGE) -> RiskAgent (AGREE) -> DataAnalysis (RESOLVE) -> ClinicalReasoning
            deadline = time.time() + 2.0
            found_clinical = None
            while time.time() < deadline:
                clinical_events = self.queue.get_history("clinical_decisions")
                matching = [e for e in clinical_events if getattr(e, "event_id", "") == "async_case_105"]
                if matching and matching[-1].metadata.get("negotiation_trace"):
                    found_clinical = matching[-1]
                    break
                time.sleep(0.05)

            self.assertIsNotNone(found_clinical, "Autonomous worker negotiation did not complete in time")
            trace = found_clinical.metadata.get("negotiation_trace", {})
            self.assertEqual(trace.get("status"), "RESOLVED")
            self.assertEqual(trace.get("challenge_round"), 1)

        finally:
            self.risk_agent.stop()
            self.analysis_agent.stop()
            self.clinical_agent.stop()
            self.risk_agent.join(timeout=1.0)
            self.analysis_agent.join(timeout=1.0)
            self.clinical_agent.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
