"""Unit and integration tests for bounded LLM-driven tool selection in MonitoringAgent.

Covers:
1. Normal/stable observations
2. Significant abnormal observation
3. Multiple abnormalities
4. Poor/invalid signal quality
5. Gemini unavailable/failure
6. Tool execution failure
7. Maximum tool-call limit
8. Correct tool selection/execution
9. Existing deterministic safety escalation
"""

import unittest
from unittest.mock import MagicMock, call
import pandas as pd
import numpy as np

from communication.event_queue import EventQueue
from communication.events import MonitoringDecision, MonitoringEvent
from communication.llm_client import LLMTimeoutError, LLMSchemaValidationError
from monitoring_agent.monitoring_agent import MonitoringAgent
from monitoring_agent.monitoring_llm_reasoner import (
    GeminiMonitoringReasoner,
    MonitoringClassification,
    ToolSelectionDecision,
    MonitoringLLMProposal,
)


class TestMonitoringAgenticTools(unittest.TestCase):
    def setUp(self):
        self.queue = EventQueue()

    def test_1_normal_stable_observation_dynamic_tool_calling(self):
        """Test normal/stable observation where LLM chooses baseline -> finish."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        # 1st call: LLM selects calculate_baseline
        # 2nd call: LLM decides observation is stable and finishes
        mock_reasoner.select_tool.side_effect = [
            ToolSelectionDecision(
                thought="Need to inspect baseline first.",
                action="tool",
                tool_name="calculate_baseline",
                tool_args={},
            ),
            ToolSelectionDecision(
                thought="Baseline is consistent and patient vitals are stable.",
                action="finish",
                final_decision="CONTINUE_MONITORING",
                classification="STABLE",
                rationale="Normal vitals, within baseline boundaries.",
                confidence=0.95,
            ),
        ] * 15

        agent = MonitoringAgent(
            case_id=201,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        # Feed 11 normal observations so baseline exists
        for i in range(10):
            agent.step({
                "patient_id": 201,
                "timestamp": float(i),
                "HR": 72.0,
                "MAP": 88.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Observation 11
        decision, event = agent.step({
            "patient_id": 201,
            "timestamp": 11.0,
            "HR": 73.0,
            "MAP": 89.0,
            "SpO2": 98.0,
            "RR": 16.0,
        })

        self.assertEqual(decision, MonitoringDecision.CONTINUE_MONITORING)
        self.assertIsNone(event)
        # Verify tool trace was recorded in state
        trace = agent.state.latest_tool_trace
        self.assertTrue(len(trace) >= 2)
        self.assertEqual(trace[0]["tool_name"], "calculate_baseline")
        self.assertEqual(trace[0]["status"], "success")
        self.assertEqual(trace[1]["action"], "finish")
        self.assertEqual(agent.state.last_agentic_mode, "LLM_AGENTIC_TOOLS")

    def test_2_significant_abnormal_observation_tool_selection(self):
        """Test abnormal vital triggers calculate_deviation and trends, culminating in escalation."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.side_effect = [
            ToolSelectionDecision(
                thought="Severe tachycardia detected, calculating relative deviation.",
                action="tool",
                tool_name="calculate_deviation",
                tool_args={},
            ),
            ToolSelectionDecision(
                thought="Calculating trends to confirm trajectory.",
                action="tool",
                tool_name="calculate_trends",
                tool_args={},
            ),
            ToolSelectionDecision(
                thought="Acute tachycardia confirmed. Escalating.",
                action="finish",
                final_decision="ESCALATE_TO_RISK",
                classification="GENUINE_DETERIORATION",
                rationale="Heart rate jumped over 150 bpm with significant positive deviation.",
                confidence=0.98,
            ),
        ] * 25

        agent = MonitoringAgent(
            case_id=202,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        for i in range(12):
            agent.step({
                "patient_id": 202,
                "timestamp": float(i),
                "HR": 75.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Severe tachycardia sustained for 5 samples (SEVERITY_PERSISTENCE['severe'] = 5)
        decision, event = None, None
        for i in range(12, 17):
            decision, event = agent.step({
                "patient_id": 202,
                "timestamp": float(i),
                "HR": 155.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        self.assertEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNotNone(event)
        self.assertIn("HR", event.affected_vitals)
        self.assertEqual(event.metadata["agentic_mode"], "LLM_AGENTIC_TOOLS")
        # Check tool trace in event metadata
        self.assertTrue(len(event.metadata["tool_trace"]) >= 2)
        trace_tools = [s["tool_name"] for s in event.metadata["tool_trace"] if s.get("tool_name")]
        self.assertIn("calculate_deviation", trace_tools)

    def test_3_multiple_abnormalities_workflow(self):
        """Test multi-vital deterioration where Gemini assesses deviation then finishes."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.side_effect = [
            ToolSelectionDecision(
                thought="Multiple vitals abnormal: MAP crashing and HR spiking. Calculating deviation.",
                action="tool",
                tool_name="calculate_deviation",
                tool_args={},
            ),
            ToolSelectionDecision(
                thought="Deviation confirmed shock state. Escalating immediately.",
                action="finish",
                final_decision="ESCALATE_TO_RISK",
                classification="GENUINE_DETERIORATION",
                rationale="Cardiovascular collapse with severe hypotension and tachycardia.",
                confidence=0.99,
            ),
        ] * 10

        agent = MonitoringAgent(
            case_id=203,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        for i in range(12):
            agent.step({
                "patient_id": 203,
                "timestamp": float(i),
                "HR": 75.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        decision, event = None, None
        for i in range(12, 17):
            decision, event = agent.step({
                "patient_id": 203,
                "timestamp": float(i),
                "HR": 140.0,
                "MAP": 52.0,
                "SpO2": 89.0,
                "RR": 26.0,
            })

        self.assertEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNotNone(event)
        self.assertIn("MAP", event.affected_vitals)

    def test_4_poor_signal_quality_handling(self):
        """Test poor/invalid signal quality leading to tool selection of assess_signal_quality."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.side_effect = [
            ToolSelectionDecision(
                thought="Possible sensor artifact on SpO2 reading of 0. Checking signal quality.",
                action="tool",
                tool_name="assess_signal_quality",
                tool_args={"window": 10},
            ),
            ToolSelectionDecision(
                thought="Checking invalid measurement mask.",
                action="tool",
                tool_name="invalid_measurement_mask",
                tool_args={},
            ),
            ToolSelectionDecision(
                thought="Artifact confirmed. Discarding without escalation.",
                action="finish",
                final_decision="CONTINUE_MONITORING",
                classification="SENSOR_ARTIFACT",
                rationale="SpO2 disconnect artifact.",
                confidence=0.95,
            ),
        ] * 15

        agent = MonitoringAgent(
            case_id=204,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        for i in range(12):
            agent.step({
                "patient_id": 204,
                "timestamp": float(i),
                "HR": 75.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Transient artifact
        decision, event = agent.step({
            "patient_id": 204,
            "timestamp": 13.0,
            "HR": 75.0,
            "MAP": 90.0,
            "SpO2": 0.0,  # Physio implausible
            "RR": 16.0,
        })

        # Decision is non-escalation (CONTINUE_MONITORING or NO_ACTION for candidate/invalid) and event is None
        self.assertNotEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNone(event)
        trace_tools = [s["tool_name"] for s in agent.state.latest_tool_trace if s.get("tool_name")]
        self.assertIn("assess_signal_quality", trace_tools)
        self.assertIn("invalid_measurement_mask", trace_tools)

    def test_5_gemini_unavailable_fallback(self):
        """Test that Gemini timeout or exception triggers clean deterministic pipeline fallback."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.side_effect = LLMTimeoutError("Gemini API connection timed out")
        mock_reasoner.propose.side_effect = LLMTimeoutError("Gemini API connection timed out")

        agent = MonitoringAgent(
            case_id=205,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        for i in range(12):
            agent.step({
                "patient_id": 205,
                "timestamp": float(i),
                "HR": 75.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        decision, event = None, None
        for i in range(12, 17):
            decision, event = agent.step({
                "patient_id": 205,
                "timestamp": float(i),
                "HR": 160.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Deterministic safety rules should trigger escalation despite LLM failure
        self.assertEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNotNone(event)
        self.assertEqual(event.metadata["agentic_mode"], "LLM_FALLBACK_DETERMINISTIC")
        self.assertIn("llm_fallback_reason", event.metadata)
        self.assertIn("Gemini API connection timed out", event.metadata["llm_fallback_reason"])

    def test_6_tool_execution_failure_handling(self):
        """Test tool execution failure is recorded in trace and agent safely recovers."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.return_value = ToolSelectionDecision(
            thought="Running nonexistent tool.",
            action="tool",
            tool_name="nonexistent_analytical_tool",
            tool_args={},
        )

        agent = MonitoringAgent(
            case_id=206,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        decision, event = agent.step({
            "patient_id": 206,
            "timestamp": 1.0,
            "HR": 75.0,
            "MAP": 90.0,
            "SpO2": 98.0,
            "RR": 16.0,
        })

        self.assertEqual(decision, MonitoringDecision.CONTINUE_MONITORING)
        trace = agent.state.latest_tool_trace
        self.assertTrue(len(trace) > 0)
        self.assertEqual(trace[0]["status"], "error")
        self.assertIn("Unknown tool", trace[0]["error"])

    def test_7_maximum_tool_call_limit(self):
        """Test agent bounded loop halts when max_tool_iterations is reached without infinite looping."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.return_value = ToolSelectionDecision(
            thought="Keep cleaning.",
            action="tool",
            tool_name="preprocess",
            tool_args={},
        )

        agent = MonitoringAgent(
            case_id=207,
            event_queue=self.queue,
            baseline_window=10,
            max_tool_iterations=3,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        decision, event = agent.step({
            "patient_id": 207,
            "timestamp": 1.0,
            "HR": 75.0,
            "MAP": 90.0,
            "SpO2": 98.0,
            "RR": 16.0,
        })

        self.assertEqual(decision, MonitoringDecision.CONTINUE_MONITORING)
        self.assertEqual(mock_reasoner.select_tool.call_count, 3)
        self.assertEqual(len(agent.state.latest_tool_trace), 3)

    def test_8_correct_tool_selection_and_execution_feedback(self):
        """Verify that executed tool result is fed back into next decision context."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)

        def mock_select(patient_id, observations_count, latest_observation, executed_tools, state_summary, iteration, max_iterations):
            if iteration == 0:
                self.assertNotIn("preprocess", executed_tools)
                self.assertFalse(state_summary["has_clean_values"])
                return ToolSelectionDecision(
                    thought="Clean data first.",
                    action="tool",
                    tool_name="preprocess",
                )
            elif iteration == 1:
                # Feedback check: preprocess must now be recorded in executed_tools and state_summary
                self.assertIn("preprocess", executed_tools)
                self.assertTrue(state_summary["has_clean_values"])
                return ToolSelectionDecision(
                    thought="Data cleaned, now compute baseline.",
                    action="tool",
                    tool_name="calculate_baseline",
                )
            else:
                self.assertIn("calculate_baseline", executed_tools)
                self.assertTrue(state_summary["baseline_established"])
                return ToolSelectionDecision(
                    thought="All needed tools executed.",
                    action="finish",
                    final_decision="CONTINUE_MONITORING",
                    classification="STABLE",
                    rationale="Vitals are well within limits.",
                )

        mock_reasoner.select_tool.side_effect = mock_select

        agent = MonitoringAgent(
            case_id=208,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        agent.step({
            "patient_id": 208,
            "timestamp": 1.0,
            "HR": 75.0,
            "MAP": 90.0,
            "SpO2": 98.0,
            "RR": 16.0,
        })

        self.assertEqual(mock_reasoner.select_tool.call_count, 3)

    def test_9_existing_deterministic_safety_escalation(self):
        """Test that deterministic safety guardrails override Gemini if Gemini hallucinates STABLE on lethal vitals."""
        mock_reasoner = MagicMock(spec=GeminiMonitoringReasoner)
        mock_reasoner.select_tool.return_value = ToolSelectionDecision(
            thought="I think everything is fine despite HR=190.",
            action="finish",
            final_decision="CONTINUE_MONITORING",
            classification="STABLE",
            rationale="Looks harmless.",
            confidence=0.99,
        )

        agent = MonitoringAgent(
            case_id=209,
            event_queue=self.queue,
            baseline_window=10,
            enable_llm=True,
            llm_reasoner=mock_reasoner,
        )

        for i in range(12):
            agent.step({
                "patient_id": 209,
                "timestamp": float(i),
                "HR": 75.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Severe life-threatening tachycardia sustained for 5 samples
        decision, event = None, None
        for i in range(12, 17):
            decision, event = agent.step({
                "patient_id": 209,
                "timestamp": float(i),
                "HR": 190.0,
                "MAP": 90.0,
                "SpO2": 98.0,
                "RR": 16.0,
            })

        # Deterministic safety arbitration MUST enforce escalation and flag safety arbitration conflict
        self.assertEqual(decision, MonitoringDecision.ESCALATE_TO_RISK)
        self.assertIsNotNone(event)
        self.assertTrue(event.metadata["llm_monitoring_conflict"])
        self.assertTrue(event.metadata["verification_required"])


if __name__ == "__main__":
    unittest.main()
