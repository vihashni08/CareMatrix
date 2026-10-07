"""Standardized tools for CareMatrix Data Analysis Agent.

Wraps existing functions:
- statistical analysis: calculate statistical summaries (mean, min, max, std)
- trend analysis: linear slope and categorical direction classification
- pattern analysis: multi-vital pattern detection (e.g. simultaneous changes, hemodynamic discordance)
- historical comparison: window changes and metric comparisons over time
- consistency analysis: cross-agent evidence alignment and conflict detection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from carematrix_runtime.tools.base import Tool, ToolRegistry
from communication.events import RiskDecisionEvent
from communication.orchestration import verify_cross_agent_consistency
from data_analysis_agent.analyzer import (
    assess_window_quality,
    calculate_trend,
    extract_metrics_and_patterns,
)


def create_data_analysis_tools() -> ToolRegistry:
    """Create and register all data analysis agent tools."""
    registry = ToolRegistry(name="data_analysis")

    # 1. Trend analysis (single vital)
    def _calculate_trend(values: pd.Series) -> Tuple[float, str]:
        return calculate_trend(values)

    registry.register(
        Tool(
            name="calculate_trend",
            description="Calculate linear slope and classify robust descriptive trend (increasing, decreasing, stable) for a vital sign series.",
            func=_calculate_trend,
            parameters={
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "description": "pandas Series of sequential vital sign measurements",
                    }
                },
                "required": ["values"],
            },
            output_schema={
                "type": "array",
                "description": "Tuple of (slope: float, direction: str)",
            },
            tags=["data_analysis", "trends"],
        )
    )

    # 2. Window Quality & Artifact Assessment
    def _assess_window_quality(
        window: pd.DataFrame,
        expected_samples: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        return assess_window_quality(window, expected_samples=expected_samples)

    registry.register(
        Tool(
            name="assess_window_quality",
            description="Assess data completeness, missingness ratio, sample sufficiency, and potential sensor artifact spikes over an observation window.",
            func=_assess_window_quality,
            parameters={
                "type": "object",
                "properties": {
                    "window": {
                        "type": "object",
                        "description": "DataFrame containing vital observations across the sliding window",
                    },
                    "expected_samples": {
                        "type": "integer",
                        "description": "Expected number of sample ticks in window",
                    },
                },
                "required": ["window"],
            },
            output_schema={
                "type": "array",
                "description": "Tuple of (quality_flagged: bool, quality_details: dict)",
            },
            tags=["data_analysis", "quality"],
        )
    )

    # 3. Statistical Analysis, Historical Comparison & Multi-Vital Pattern Analysis
    def _extract_metrics_and_patterns(
        window: pd.DataFrame,
        risk: RiskDecisionEvent,
        requested_checks: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
        return extract_metrics_and_patterns(
            window=window,
            risk=risk,
            requested_checks=requested_checks,
        )

    registry.register(
        Tool(
            name="extract_metrics_and_patterns",
            description="Extract statistical metrics (mean, min, max, std), window historical changes, and multi-vital patterns (variability, concordance, discordance).",
            func=_extract_metrics_and_patterns,
            parameters={
                "type": "object",
                "properties": {
                    "window": {
                        "type": "object",
                        "description": "Observation window DataFrame",
                    },
                    "risk": {
                        "type": "object",
                        "description": "Upstream RiskDecisionEvent triggering analysis",
                    },
                    "requested_checks": {
                        "type": "array",
                        "description": "List of checks to execute, e.g. ['trends', 'variability', 'patterns', 'verification']",
                    },
                },
                "required": ["window", "risk"],
            },
            output_schema={
                "type": "array",
                "description": "Tuple of (metrics: dict, patterns: list, changes: list, executed_checks: list)",
            },
            tags=["data_analysis", "statistics", "patterns"],
        )
    )

    # 4. Consistency analysis (Cross-agent verification)
    def _verify_cross_agent_consistency(
        risk_level: str,
        metrics: Dict[str, Any],
        patterns: List[str],
        data_quality_flag: bool,
        analysis_status: str,
    ) -> Tuple[str, List[str], List[str], List[str], bool, float]:
        return verify_cross_agent_consistency(
            risk_level=risk_level,
            metrics=metrics,
            patterns=patterns,
            data_quality_flag=data_quality_flag,
            analysis_status=analysis_status,
        )

    registry.register(
        Tool(
            name="verify_cross_agent_consistency",
            description="Verify consistency between Risk Agent predictions and Data Analysis temporal findings, detecting alignment or conflict.",
            func=_verify_cross_agent_consistency,
            parameters={
                "type": "object",
                "properties": {
                    "risk_level": {
                        "type": "string",
                        "description": "Risk classification ('HIGH RISK' or 'LOW RISK')",
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Extracted vital statistical metrics dictionary",
                    },
                    "patterns": {
                        "type": "array",
                        "description": "List of observed physiological pattern descriptions",
                    },
                    "data_quality_flag": {
                        "type": "boolean",
                        "description": "True if data quality issues were flagged in the window",
                    },
                    "analysis_status": {
                        "type": "string",
                        "description": "Status of data analysis stage (e.g. 'complete')",
                    },
                },
                "required": ["risk_level", "metrics", "patterns", "data_quality_flag", "analysis_status"],
            },
            output_schema={
                "type": "array",
                "description": "Tuple of (consistency, supporting_ev, conflicting_ev, conflict_flags, verification_req, confidence)",
            },
            tags=["data_analysis", "consistency"],
        )
    )

    return registry

