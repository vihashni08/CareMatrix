"""Standardized tools for CareMatrix Risk Agent.

Wraps existing functions:
- patient context gathering: demographic and physiological priors (age, sex, bmi, asa, emop)
- feature construction: tabular feature vector extraction from vital windows and events
- ML model inference: preprocessor transformation and probability estimation from trained artifacts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from carematrix_runtime.tools.base import Tool, ToolRegistry
from communication.events import MonitoringEvent
from risk_agent.model import RiskModelTool
from risk_agent.preprocessing import construct_features, gather_patient_context


def create_risk_tools(
    model_tool: Optional[RiskModelTool] = None,
    model_dir: Optional[Union[Path, str]] = None,
) -> ToolRegistry:
    """Create and register all risk agent tools."""
    registry = ToolRegistry(name="risk")

    active_model_tool = model_tool or RiskModelTool(model_dir=model_dir)

    # 1. Patient context gathering tool
    def _gather_patient_context(
        case_id: Union[int, str],
        cache: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return gather_patient_context(case_id, cache=cache)

    registry.register(
        Tool(
            name="gather_patient_context",
            description="Gather demographic context (age, sex, bmi, asa, emop) for a patient with offline fallback.",
            func=_gather_patient_context,
            parameters={
                "type": "object",
                "properties": {
                    "case_id": {
                        "type": "integer",
                        "description": "Patient or case identifier",
                    },
                },
                "required": ["case_id"],
            },
            output_schema={
                "type": "object",
                "description": "Dictionary of patient demographic features",
            },
            tags=["risk", "context"],
        )
    )

    # 2. Feature construction tool
    def _construct_features(
        event: MonitoringEvent,
        patient_context: Dict[str, float],
        feature_columns: Optional[List[str]] = None,
        window_df: Optional[pd.DataFrame] = None,
        min_window_samples: int = 5,
        return_source: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], str]]:
        cols = feature_columns or active_model_tool.feature_columns
        return construct_features(
            event=event,
            patient_context=patient_context,
            feature_columns=cols,
            window_df=window_df,
            min_window_samples=min_window_samples,
            return_source=return_source,
        )

    registry.register(
        Tool(
            name="construct_features",
            description="Construct ML feature vector aligned with trained model schema using event telemetry and patient context.",
            func=_construct_features,
            parameters={
                "type": "object",
                "properties": {
                    "event": {
                        "type": "object",
                        "description": "MonitoringEvent triggering risk prediction",
                    },
                    "patient_context": {
                        "type": "object",
                        "description": "Demographic context dictionary",
                    },
                    "window_df": {
                        "type": "object",
                        "description": "Optional observation window DataFrame",
                    },
                },
                "required": ["event", "patient_context"],
            },
            output_schema={
                "type": "object",
                "description": "Dictionary mapping feature names to numerical values",
            },
            tags=["risk", "features"],
        )
    )

    # 3. ML Model inference tool
    def _predict_risk_probability(features: Dict[str, Any]) -> float:
        return active_model_tool.predict_proba(features)

    registry.register(
        Tool(
            name="predict_risk_probability",
            description="Execute trained preprocessor and ML model inference (RandomForest/GBM) to compute high-risk probability.",
            func=_predict_risk_probability,
            parameters={
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Dictionary of feature values matching trained model feature schema",
                    }
                },
                "required": ["features"],
            },
            output_schema={
                "type": "number",
                "description": "Probability score between 0.0 and 1.0",
            },
            tags=["risk", "ml_inference"],
        )
    )

    return registry

