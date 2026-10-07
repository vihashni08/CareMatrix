"""Unit and integration tests for bounded LLM-driven tool selection in DataAnalysisAgent.

Covers:
A. Normal/stable analysis:
   - Gemini selects appropriate analytical tools (e.g. assess_window_quality -> extract_metrics_and_patterns -> finish).
   - Tools execute and results are fed back into subsequent decisions.
   - Agent produces valid complete DataAnalysisEvent with tool trace.
B. Poor data quality:
   - Gemini selects assess_window_quality, observes compromised quality flag, and subsequent decisions respond.
C. Trend/pattern analysis:
   - Gemini dynamically selects calculate_trend or extract_metrics_and_patterns.
D. Consistency verification:
   - verify_cross_agent_consistency runs and aligns/conflicts correctly.
E. Multiple analytical requirements:
   - Sequential tool calls in bounded loop.
F. Gemini unavailable/timeout:
   - Clean deterministic pipeline fallback without breaking flow.
G. Invalid/malformed Gemini decision:
   - Safe parsing error fallback to deterministic pipeline.
H. Tool execution failure:
   - Safe error handling in trace and recovery.
I. Maximum tool-call limit:
   - Loop bounds safely at max_tool_iterations.
J. Tool-result feedback:
   - Verified that subsequent reasoning steps receive prior tool execution results.
K. Existing challenge/negotiation behavior:
   - Agent continues to challenge conflicting risk decisions and negotiate.
L. Existing agent/pipeline compatibility:
   - Normal event publication and state maintenance.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from communication.event_queue import EventQueue
from communication.events import (
    DataAnalysisEvent,
    PerformativeType,
    RiskDecisionEvent,
)
from communication.llm_client import LLMTimeoutError, LLMSchemaValidationError
from data_analysis_agent.data_analysis_agent import DataAnalysisAgent
from data_analysis_agent.data_analysis_llm_reasoner import (
    DataAnalysisToolDecision,
    GeminiDataAnalysisReasoner,
)


def make_sample_window(points: int = 61, hr_trend: str = "stable", map_trend: str = "stable") -> pd.DataFrame:
    """Generate sample observation window DataFrame."""
    hr_vals = (
        np.full(points, 75.0)
        if hr_trend == "stable"
        else (np.linspace(70, 110, points) if hr_trend == "increasing" else np.linspace(110, 70, points))
    )
    map_vals = (
        np.full(points, 85.0)
        if map_trend == "stable"
        else (np.linspace(85, 55, points) if map_trend == "decreasing" else np.linspace(55, 85, points))
    )
    return pd.DataFrame({
        "Time": np.arange(points) * 5.0,
        "HR": hr_vals,
        "MAP": map_vals,
        "SpO2": np.full(points, 98.0),
        "RR": np.full(points, 14.0),
        "SBP": np.full(points, 120.0),
        "DBP": np.full(points, 75.0),
        "BT": np.full(points, 36.7),
    })


class TestDataAnalysisAgenticTools(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()

    def make_risk_event(
        self,
        event_id: str = "case_10_evt_001",
        risk_level: str = "LOW RISK",
        affected_vitals: list[str] | None = None,
        challenge_round: int = 0,
        performative: str = PerformativeType.INFORM.value,
    ) -> RiskDecisionEvent:
        return RiskDecisionEvent(
            patient_id=10,
            event_id=event_id,
            timestamp=300.0,
            decision=risk_level.replace(" ", "_"),
            risk_probability=0.12 if risk_level == "LOW RISK" else 0.85,
            risk_level=risk_level,
            threshold=0.16,
            model_name="RandomForestClassifier",
            severity="moderate",
            affected_vitals=affected_vitals or ["HR", "MAP"],
            window_start=0.0,
            window_end=300.0,
            performative=performative,
            challenge_round=challenge_round,
        )

    # ------------------------------------------------------------------------
    # A. Normal / Stable Analysis Tool Flow
    # ------------------------------------------------------------------------
    def test_normal_stable_analysis_agentic_flow(self):
        """Verify LLM dynamically selects quality assessment -> metrics extraction -> finish."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = [
            DataAnalysisToolDecision(
                thought="First check observation window data quality.",
                action="tool",
                tool_name="assess_window_quality",
                tool_args={},
            ),
            DataAnalysisToolDecision(
                thought="Window quality is nominal. Extract metrics and trajectories.",
                action="tool",
                tool_name="extract_metrics_and_patterns",
                tool_args={"requested_checks": ["trends", "patterns"]},
            ),
            DataAnalysisToolDecision(
                thought="Metrics extracted and stable. Analysis complete.",
                action="finish",
                rationale="Stable physiological trajectories corroborate low risk.",
                confidence=0.96,
            ),
        ]

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event()
        result = agent.process_event(risk_event)

        self.assertEqual(result.analysis_status, "complete")
        self.assertEqual(result.risk_level, "LOW RISK")
        self.assertFalse(result.data_quality_flag)
        self.assertIn("HR", result.trend_metrics)
        self.assertEqual(result.metadata["agentic_mode"], "LLM_AGENTIC_TOOLS")

        trace = result.metadata["tool_trace"]
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0]["tool_name"], "assess_window_quality")
        self.assertEqual(trace[0]["status"], "success")
        self.assertEqual(trace[1]["tool_name"], "extract_metrics_and_patterns")
        self.assertEqual(trace[1]["status"], "success")
        self.assertEqual(trace[2]["action"], "finish")

    # ------------------------------------------------------------------------
    # B. Poor Data Quality Flow
    # ------------------------------------------------------------------------
    def test_poor_data_quality_handling(self):
        """Verify LLM detects compromised data quality and responds accordingly."""
        # Create window with heavy missing data
        poor_window = make_sample_window(points=61)
        poor_window.loc[10:50, "HR"] = np.nan

        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = [
            DataAnalysisToolDecision(
                thought="Evaluate window completeness first.",
                action="tool",
                tool_name="assess_window_quality",
                tool_args={},
            ),
            DataAnalysisToolDecision(
                thought="Quality is compromised due to high missingness on HR. Finish and flag quality issue.",
                action="finish",
                rationale="Sensor data quality degraded, flagging for clinical review.",
                confidence=0.90,
            ),
        ]

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: poor_window,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event()
        result = agent.process_event(risk_event)

        self.assertTrue(result.data_quality_flag)
        self.assertEqual(result.metadata["agentic_mode"], "LLM_AGENTIC_TOOLS")
        trace = result.metadata["tool_trace"]
        self.assertEqual(trace[0]["tool_name"], "assess_window_quality")
        self.assertTrue(trace[0]["output"]["quality_flagged"])

    # ------------------------------------------------------------------------
    # C. Dynamic Trend & Pattern Analysis Flow
    # ------------------------------------------------------------------------
    def test_trend_and_pattern_tool_selection(self):
        """Verify LLM selects specific vital trend calculation tool."""
        window = make_sample_window(points=61, hr_trend="increasing", map_trend="decreasing")

        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = [
            DataAnalysisToolDecision(
                thought="Check HR trend trajectory specifically.",
                action="tool",
                tool_name="calculate_trend",
                tool_args={"vital": "HR"},
            ),
            DataAnalysisToolDecision(
                thought="Extract multi-vital patterns including variability and verification.",
                action="tool",
                tool_name="extract_metrics_and_patterns",
                tool_args={"requested_checks": ["trends", "patterns", "verification"]},
            ),
            DataAnalysisToolDecision(
                thought="Hemodynamic discordance verified. Complete analysis.",
                action="finish",
                rationale="Tachycardia with hypotension pattern observed.",
                confidence=0.98,
            ),
        ]

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: window,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event(risk_level="HIGH RISK")
        result = agent.process_event(risk_event)

        trace = result.metadata["tool_trace"]
        tool_names = [t["tool_name"] for t in trace if t.get("tool_name")]
        self.assertIn("calculate_trend", tool_names)
        self.assertIn("extract_metrics_and_patterns", tool_names)
        self.assertEqual(result.trend_metrics["HR"]["trend"], "increasing")

    # ------------------------------------------------------------------------
    # D. Cross-Agent Consistency Verification Tool
    # ------------------------------------------------------------------------
    def test_verify_cross_agent_consistency_tool(self):
        """Verify Gemini can invoke verify_cross_agent_consistency tool directly."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = [
            DataAnalysisToolDecision(
                thought="Extract metrics and patterns first.",
                action="tool",
                tool_name="extract_metrics_and_patterns",
                tool_args={},
            ),
            DataAnalysisToolDecision(
                thought="Now explicitly run cross-agent consistency verification.",
                action="tool",
                tool_name="verify_cross_agent_consistency",
                tool_args={},
            ),
            DataAnalysisToolDecision(
                thought="Consistency verification completed.",
                action="finish",
                rationale="Cross-agent alignment confirmed.",
            ),
        ]

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event(risk_level="LOW RISK")
        result = agent.process_event(risk_event)

        trace = result.metadata["tool_trace"]
        tool_names = [t["tool_name"] for t in trace if t.get("tool_name")]
        self.assertIn("verify_cross_agent_consistency", tool_names)

    # ------------------------------------------------------------------------
    # E. Tool Result Feedback into Next Step
    # ------------------------------------------------------------------------
    def test_tool_result_feedback_into_subsequent_decision(self):
        """Verify the state and previously executed tools are fed back into subsequent LLM turns."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)

        def mock_select(case_id, event_id, risk_level, affected_vitals, window_summary, executed_tools, state_summary, iteration, max_iterations):
            if iteration == 0:
                self.assertNotIn("assess_window_quality", executed_tools)
                self.assertFalse(state_summary["quality_assessed"])
                return DataAnalysisToolDecision(
                    thought="Run quality check.",
                    action="tool",
                    tool_name="assess_window_quality",
                )
            elif iteration == 1:
                # Feedback verification: assess_window_quality must be in executed_tools and state_summary updated
                self.assertIn("assess_window_quality", executed_tools)
                self.assertTrue(state_summary["quality_assessed"])
                return DataAnalysisToolDecision(
                    thought="Quality checked. Now extract patterns.",
                    action="tool",
                    tool_name="extract_metrics_and_patterns",
                )
            else:
                self.assertIn("extract_metrics_and_patterns", executed_tools)
                self.assertTrue(state_summary["metrics_extracted"])
                return DataAnalysisToolDecision(
                    thought="All needed tools executed. Finish.",
                    action="finish",
                    rationale="Complete.",
                )

        mock_reasoner.select_tool.side_effect = mock_select

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event()
        agent.process_event(risk_event)

        self.assertEqual(mock_reasoner.select_tool.call_count, 3)

    # ------------------------------------------------------------------------
    # F. Gemini Unavailable / Timeout Fallback
    # ------------------------------------------------------------------------
    def test_gemini_unavailable_timeout_fallback(self):
        """Verify LLM timeout cleanly falls back to deterministic analysis without crashing."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = LLMTimeoutError("Gemini API timed out")

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event()
        result = agent.process_event(risk_event)

        self.assertEqual(result.analysis_status, "complete")
        self.assertEqual(result.metadata["agentic_mode"], "LLM_FALLBACK_DETERMINISTIC")
        self.assertIn("fallback_error", result.metadata)
        self.assertIn("Gemini API timed out", result.metadata["fallback_error"])
        self.assertIn("HR", result.trend_metrics)

    # ------------------------------------------------------------------------
    # G. Malformed Gemini Decision Fallback
    # ------------------------------------------------------------------------
    def test_malformed_gemini_decision_fallback(self):
        """Verify invalid JSON schema from Gemini safely triggers fallback."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = LLMSchemaValidationError("Missing required field")

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event()
        result = agent.process_event(risk_event)

        self.assertEqual(result.analysis_status, "complete")
        self.assertEqual(result.metadata["agentic_mode"], "LLM_FALLBACK_DETERMINISTIC")

    # ------------------------------------------------------------------------
    # H. Tool Execution Failure Handling
    # ------------------------------------------------------------------------
    def test_tool_execution_failure_handling(self):
        """Verify unknown or failing tool is recorded in trace and agent safely recovers."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.return_value = DataAnalysisToolDecision(
            thought="Calling nonexistent tool.",
            action="tool",
            tool_name="nonexistent_analysis_tool",
            tool_args={},
        )

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        risk_event = self.make_risk_event()
        result = agent.process_event(risk_event)

        # Agent should safely fall back and complete
        self.assertEqual(result.analysis_status, "complete")
        trace = agent.patient_states[10].latest_tool_trace
        self.assertTrue(len(trace) > 0)
        self.assertEqual(trace[0]["status"], "error")
        self.assertIn("Unknown tool", trace[0]["error"])

    # ------------------------------------------------------------------------
    # I. Maximum Tool-Call Limit
    # ------------------------------------------------------------------------
    def test_maximum_tool_call_limit(self):
        """Verify loop terminates at max_tool_iterations without infinite looping."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        # LLM keeps calling assess_window_quality repeatedly
        mock_reasoner.select_tool.return_value = DataAnalysisToolDecision(
            thought="Keep assessing window quality.",
            action="tool",
            tool_name="assess_window_quality",
            tool_args={},
        )

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
            max_tool_iterations=3,
        )

        risk_event = self.make_risk_event()
        result = agent.process_event(risk_event)

        self.assertEqual(mock_reasoner.select_tool.call_count, 3)
        self.assertEqual(len(result.metadata["tool_trace"]), 3)
        self.assertEqual(result.analysis_status, "complete")

    # ------------------------------------------------------------------------
    # J. Challenge & Negotiation Preservation
    # ------------------------------------------------------------------------
    def test_agentic_analysis_preserves_challenge_and_negotiation(self):
        """Verify that agentic tool use still triggers CHALLENGE when HIGH RISK conflicts with stable trends."""
        mock_reasoner = MagicMock(spec=GeminiDataAnalysisReasoner)
        mock_reasoner.select_tool.side_effect = [
            DataAnalysisToolDecision(
                thought="Assess quality.",
                action="tool",
                tool_name="assess_window_quality",
            ),
            DataAnalysisToolDecision(
                thought="Extract metrics.",
                action="tool",
                tool_name="extract_metrics_and_patterns",
            ),
            DataAnalysisToolDecision(
                thought="Finish analysis.",
                action="finish",
            ),
        ]

        agent = DataAnalysisAgent(
            event_queue=self.queue,
            data_loader=lambda _: make_sample_window(hr_trend="stable", map_trend="stable"),
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        # High risk prediction but window vitals are stable -> CONFLICTING
        high_risk_event = self.make_risk_event(risk_level="HIGH RISK", challenge_round=0)
        challenge_result = agent.process_event(high_risk_event)

        self.assertEqual(challenge_result.evidence_consistency, "CONFLICTING")
        self.assertEqual(challenge_result.performative, PerformativeType.CHALLENGE.value)
        self.assertEqual(challenge_result.challenge_round, 1)

        challenges = self.queue.get_history("risk_challenges")
        self.assertGreaterEqual(len(challenges), 1)
        self.assertEqual(challenges[0].event_id, high_risk_event.event_id)
        self.assertEqual(challenge_result.metadata["agentic_mode"], "LLM_AGENTIC_TOOLS")
        self.assertTrue(len(challenge_result.metadata["tool_trace"]) >= 2)


if __name__ == "__main__":
    unittest.main()
