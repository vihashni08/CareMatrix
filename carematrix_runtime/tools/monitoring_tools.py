"""Standardized tools for CareMatrix Monitoring Agent.

Wraps existing functions:
- preprocessing: clean and normalize raw vital telemetry
- baseline calculation: rolling baseline window calculation
- deviation calculation: compute relative deviation against baselines
- trend analysis: evaluate directional trends over sliding window
- signal-quality assessment: evaluate invalid readings and sensor signal noise
"""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from carematrix_runtime.tools.base import Tool, ToolRegistry
from monitoring_agent.baseline import calculate_baseline
from monitoring_agent.config import BASELINE_WINDOW_SECONDS
from monitoring_agent.deviation_detector import calculate_deviation
from monitoring_agent.preprocessing import preprocess_data
from monitoring_agent.signal_quality import assess_signal_quality, invalid_measurement_mask
from monitoring_agent.trend_detector import calculate_trends


def create_monitoring_tools(baseline_window: int = BASELINE_WINDOW_SECONDS) -> ToolRegistry:
    """Create and register all monitoring agent tools."""
    registry = ToolRegistry(name="monitoring")

    # 1. Preprocessing tool
    def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
        return preprocess_data(df)

    registry.register(
        Tool(
            name="preprocess",
            description="Clean and preprocess raw vital signs telemetry data by forward-filling and handling edge invalidities.",
            func=_preprocess,
            parameters={
                "type": "object",
                "properties": {
                    "df": {
                        "type": "object",
                        "description": "DataFrame containing timestamp and vital columns (HR, MAP, SpO2, RR, SBP, DBP, BT)",
                    }
                },
                "required": ["df"],
            },
            output_schema={
                "type": "object",
                "description": "Cleaned vital signs DataFrame",
            },
            tags=["monitoring", "preprocessing"],
        )
    )

    # 2. Baseline calculation tool
    def _calculate_baseline(df: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
        w = window if window is not None else baseline_window
        return calculate_baseline(df, window=w)

    registry.register(
        Tool(
            name="calculate_baseline",
            description="Calculate rolling baseline medians for physiological vital signs over a specified historical window.",
            func=_calculate_baseline,
            parameters={
                "type": "object",
                "properties": {
                    "df": {
                        "type": "object",
                        "description": "Preprocessed DataFrame with physiological measurements",
                    },
                    "window": {
                        "type": "integer",
                        "description": f"Rolling window size in samples (default: {baseline_window})",
                    },
                },
                "required": ["df"],
            },
            output_schema={
                "type": "object",
                "description": "DataFrame containing rolling baseline values for each vital sign",
            },
            tags=["monitoring", "baseline"],
        )
    )

    # 3. Deviation calculation tool
    def _calculate_deviation(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        return calculate_deviation(df, baseline)

    registry.register(
        Tool(
            name="calculate_deviation",
            description="Calculate relative deviation percentage of current vital signs against established patient baselines.",
            func=_calculate_deviation,
            parameters={
                "type": "object",
                "properties": {
                    "df": {
                        "type": "object",
                        "description": "Current physiological measurements DataFrame",
                    },
                    "baseline": {
                        "type": "object",
                        "description": "Established rolling baselines DataFrame",
                    },
                },
                "required": ["df", "baseline"],
            },
            output_schema={
                "type": "object",
                "description": "DataFrame containing normalized relative deviation fractions per vital",
            },
            tags=["monitoring", "deviation"],
        )
    )

    # 4. Trend analysis tool
    def _calculate_trends(
        df: pd.DataFrame,
        window: int = 10,
        min_points: int = 5,
        relative_change: float = 0.03,
    ) -> pd.DataFrame:
        return calculate_trends(
            df=df,
            window=window,
            min_points=min_points,
            relative_change=relative_change,
        )

    registry.register(
        Tool(
            name="calculate_trends",
            description="Calculate robust categorical physiological trajectories (increasing, decreasing, stable) using windowed medians.",
            func=_calculate_trends,
            parameters={
                "type": "object",
                "properties": {
                    "df": {
                        "type": "object",
                        "description": "Preprocessed vitals DataFrame",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Observation window size in samples (default 10)",
                    },
                    "min_points": {
                        "type": "integer",
                        "description": "Minimum valid points required to compute trend (default 5)",
                    },
                    "relative_change": {
                        "type": "number",
                        "description": "Minimum fractional change threshold to classify non-stable trend (default 0.03)",
                    },
                },
                "required": ["df"],
            },
            output_schema={
                "type": "object",
                "description": "DataFrame containing trend classification strings per vital column",
            },
            tags=["monitoring", "trend"],
        )
    )

    # 5. Signal quality assessment tool
    def _assess_signal_quality(
        df: pd.DataFrame,
        window: int = 10,
        min_valid_points: int = 5,
        spike_threshold: float = 0.5,
    ) -> pd.DataFrame:
        return assess_signal_quality(
            df=df,
            window=window,
            min_valid_points=min_valid_points,
            spike_threshold=spike_threshold,
        )

    registry.register(
        Tool(
            name="assess_signal_quality",
            description="Assess sensor signal quality categorizing readings into good, noisy, missing, or artifact states.",
            func=_assess_signal_quality,
            parameters={
                "type": "object",
                "properties": {
                    "df": {
                        "type": "object",
                        "description": "Raw observations DataFrame",
                    },
                    "window": {
                        "type": "integer",
                        "description": "Quality assessment window (default 10)",
                    },
                    "min_valid_points": {
                        "type": "integer",
                        "description": "Minimum valid measurements threshold (default 5)",
                    },
                    "spike_threshold": {
                        "type": "number",
                        "description": "Relative jump threshold indicating non-physiological spike (default 0.5)",
                    },
                },
                "required": ["df"],
            },
            output_schema={
                "type": "object",
                "description": "DataFrame containing categorical signal quality indicators",
            },
            tags=["monitoring", "signal_quality"],
        )
    )

    # 6. Invalid measurement mask tool
    def _invalid_measurement_mask(df: pd.DataFrame) -> pd.DataFrame:
        return invalid_measurement_mask(df)

    registry.register(
        Tool(
            name="invalid_measurement_mask",
            description="Generate boolean mask marking physiologically implausible or disconnected measurements.",
            func=_invalid_measurement_mask,
            parameters={
                "type": "object",
                "properties": {
                    "df": {
                        "type": "object",
                        "description": "Vitals DataFrame to inspect",
                    }
                },
                "required": ["df"],
            },
            output_schema={
                "type": "object",
                "description": "Boolean DataFrame mask indicating invalid measurement entries",
            },
            tags=["monitoring", "signal_quality"],
        )
    )

    return registry

