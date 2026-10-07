"""Unit tests for CareMatrix minimal tool abstraction layer.

Verifies:
1. Tool encapsulation: name, description, input schema, output schema, callable implementation.
2. Safe error handling: exceptions wrapped in ToolResult without crashing callers.
3. Transparent direct callable interface: tool(x, y) works identically to original functions.
4. Dict compatibility for existing agent code: registry['tool_name'] behaves as expected.
5. Monitoring tools registry and execution.
6. Data Analysis tools registry and execution.
7. Clinical Reasoning tools registry and execution.
8. Risk tools registry and execution.
9. Agent integration: MonitoringAgent, DataAnalysisAgent, ClinicalReasoningAgent, RiskAgent expose standardized tools.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from carematrix_runtime.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    create_clinical_reasoning_tools,
    create_data_analysis_tools,
    create_monitoring_tools,
    create_risk_tools,
)
from carematrix_runtime.patient_memory import PatientMemory
from clinical_reasoning_agent.clinical_reasoning_agent import ClinicalReasoningAgent
from communication.events import MonitoringEvent, RiskDecisionEvent
from data_analysis_agent.data_analysis_agent import DataAnalysisAgent
from monitoring_agent.monitoring_agent import MonitoringAgent
from risk_agent.risk_agent import RiskAgent


class TestToolAbstractionLayer(unittest.TestCase):
    """Test suite validating the lightweight native tool abstraction."""

    def test_tool_definition_and_safe_execution(self):
        """Test Tool creation, parameter inference, successful run, and error handling."""
        def sample_add(a: int, b: int) -> int:
            return a + b

        tool = Tool(
            name="sample_add",
            description="Adds two integers.",
            func=sample_add,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

        self.assertEqual(tool.name, "sample_add")
        self.assertEqual(tool.description, "Adds two integers.")
        self.assertIn("a", tool.parameters["properties"])

        # 1. Direct call
        self.assertEqual(tool(2, 3), 5)

        # 2. Safe execution via execute()
        res = tool.execute(a=10, b=20)
        self.assertTrue(res.success)
        self.assertEqual(res.data, 30)
        self.assertIsNone(res.error)

        # 3. Safe error handling on invalid invocation
        def sample_faulty(val: int) -> float:
            if val == 0:
                raise ZeroDivisionError("division by zero in sample_faulty")
            return 10.0 / val

        tool_faulty = Tool(
            name="sample_faulty",
            description="Computes reciprocal.",
            func=sample_faulty,
        )
        res_fail = tool_faulty.execute(val=0)
        self.assertFalse(res_fail.success)
        self.assertIsNone(res_fail.data)
        self.assertIn("ZeroDivisionError", str(res_fail.error))

    def test_tool_registry_dict_compatibility(self):
        """Verify ToolRegistry implements dict protocol for backward-compatible agent access."""
        registry = ToolRegistry(name="test_registry")
        tool1 = Tool(name="tool1", description="First tool", func=lambda x: x * 2)
        registry.register(tool1)

        self.assertIn("tool1", registry)
        self.assertNotIn("non_existent", registry)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry["tool1"](5), 10)
        self.assertEqual(list(registry.keys()), ["tool1"])
        self.assertEqual(len(registry.values()), 1)

        # Gemini declarations export
        decls = registry.to_gemini_tool_declarations()
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["name"], "tool1")

    def test_monitoring_tools_registration_and_execution(self):
        """Verify monitoring tools are registered and execute properly."""
        tools = create_monitoring_tools(baseline_window=10)
        expected_names = [
            "preprocess",
            "calculate_baseline",
            "calculate_deviation",
            "calculate_trends",
            "assess_signal_quality",
            "invalid_measurement_mask",
        ]
        for name in expected_names:
            self.assertIn(name, tools)

        # Execute preprocessing via tool
        df_raw = pd.DataFrame({
            "Time": [0.0, 5.0, 10.0],
            "HR": [72.0, np.nan, 75.0],
            "MAP": [85.0, 84.0, 83.0],
            "SpO2": [98.0, 98.0, 98.0],
            "RR": [14.0, 14.0, 14.0],
        })
        res_prep = tools.execute("preprocess", df=df_raw)
        self.assertTrue(res_prep.success)
        self.assertFalse(res_prep.data["HR"].isna().any())

        # Direct call check
        res_direct = tools["preprocess"](df_raw)
        self.assertFalse(res_direct["HR"].isna().any())

    def test_data_analysis_tools_registration_and_execution(self):
        """Verify data analysis tools are registered and execute properly."""
        tools = create_data_analysis_tools()
        expected_names = [
            "calculate_trend",
            "assess_window_quality",
            "extract_metrics_and_patterns",
            "verify_cross_agent_consistency",
        ]
        for name in expected_names:
            self.assertIn(name, tools)

        # Execute calculate_trend
        s = pd.Series([70.0, 75.0, 80.0, 85.0, 90.0])
        res_trend = tools.execute("calculate_trend", values=s)
        self.assertTrue(res_trend.success)
        slope, direction = res_trend.data
        self.assertEqual(direction, "increasing")

    def test_clinical_reasoning_tools_registration_and_execution(self):
        """Verify clinical reasoning tools are registered and execute properly."""
        patient_memory = PatientMemory(db_path=None)
        tools = create_clinical_reasoning_tools(patient_memory=patient_memory)
        expected_names = [
            "build_focused_medical_query",
            "retrieve_medical_evidence",
            "get_patient_context",
            "evaluate_clinical_evidence",
            "arbitrate_clinical_reasoning",
        ]
        for name in expected_names:
            self.assertIn(name, tools)

        # Execute get_patient_context
        res_ctx = tools.execute("get_patient_context", patient_id=101, limit=3)
        self.assertTrue(res_ctx.success)
        self.assertEqual(res_ctx.data["patient_id"], 101)

        # Execute retrieve_medical_evidence
        res_rag = tools.execute("retrieve_medical_evidence", event_or_query="tachycardia hypotension", top_k=2)
        self.assertTrue(res_rag.success)
        self.assertIsNotNone(res_rag.data)

    def test_risk_tools_registration_and_execution(self):
        """Verify risk tools are registered and execute properly."""
        tools = create_risk_tools()
        expected_names = [
            "gather_patient_context",
            "construct_features",
            "predict_risk_probability",
        ]
        for name in expected_names:
            self.assertIn(name, tools)

        # Execute gather_patient_context
        res_ctx = tools.execute("gather_patient_context", case_id=1)
        self.assertTrue(res_ctx.success)
        self.assertIn("age", res_ctx.data)

        # Execute predict_risk_probability with zeroed feature vector
        sample_features = {f: 0.0 for f in tools.get("construct_features").func.__code__.co_varnames}
        res_pred = tools.execute("predict_risk_probability", features=sample_features)
        self.assertTrue(res_pred.success)
        self.assertIsInstance(res_pred.data, float)

    def test_agents_integrate_tools_layer(self):
        """Verify all agent classes have tools registry exposed and operational."""
        # 1. MonitoringAgent
        m_agent = MonitoringAgent(case_id=101)
        self.assertIsInstance(m_agent.tools, ToolRegistry)
        self.assertIn("preprocess", m_agent.tools)
        self.assertIn("calculate_baseline", m_agent.tools)

        # 2. DataAnalysisAgent
        da_agent = DataAnalysisAgent()
        self.assertIsInstance(da_agent.tools, ToolRegistry)
        self.assertIn("calculate_trend", da_agent.tools)

        # 3. ClinicalReasoningAgent
        cr_agent = ClinicalReasoningAgent(enable_llm=False)
        self.assertIsInstance(cr_agent.tools, ToolRegistry)
        self.assertIn("retrieve_medical_evidence", cr_agent.tools)

        # 4. RiskAgent
        r_agent = RiskAgent()
        self.assertIsInstance(r_agent.tools, ToolRegistry)
        self.assertIn("predict_risk_probability", r_agent.tools)


if __name__ == "__main__":
    unittest.main()

