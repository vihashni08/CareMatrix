"""Unified tools module for CareMatrix multi-agent architecture.

Provides a lightweight, framework-free native tool abstraction for genuine LLM tool use.
Exposes standardized tools for:
- Monitoring Agent (preprocessing, baseline, deviation, trends, signal quality)
- Data Analysis Agent (trends, window quality, metrics and patterns, consistency)
- Clinical Reasoning Agent (focused query builder, RAG evidence retrieval, patient context, clinical evaluation, arbitration)
- Risk Agent (patient context, feature construction, ML model inference)
"""

from __future__ import annotations

from carematrix_runtime.tools.base import Tool, ToolRegistry, ToolResult
from carematrix_runtime.tools.clinical_reasoning_tools import create_clinical_reasoning_tools
from carematrix_runtime.tools.data_analysis_tools import create_data_analysis_tools
from carematrix_runtime.tools.monitoring_tools import create_monitoring_tools
from carematrix_runtime.tools.risk_tools import create_risk_tools

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "create_clinical_reasoning_tools",
    "create_data_analysis_tools",
    "create_monitoring_tools",
    "create_risk_tools",
]

