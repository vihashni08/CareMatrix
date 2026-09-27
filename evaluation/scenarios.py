"""Standardized clinical evaluation scenarios for RAG, verification, and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from communication.events import DataAnalysisEvent


@dataclass
class ClinicalBenchmarkScenario:
    """A standardized physiological scenario used to evaluate agent behavior."""

    scenario_id: str
    name: str
    description: str
    risk_level: str
    metrics: dict[str, dict[str, Any]]
    patterns: list[str]
    data_quality_flag: bool = False
    expected_guideline_ids: list[str] = field(default_factory=list)
    expected_priority: str = "URGENT"

    def to_data_analysis_event(self, case_id: int = 999) -> DataAnalysisEvent:
        """Instantiate a DataAnalysisEvent representing this clinical state."""
        return DataAnalysisEvent(
            case_id=case_id,
            event_id=f"scenario_{self.scenario_id}",
            timestamp=120.0,
            risk_level=self.risk_level,
            trend_metrics=self.metrics,
            pattern_identified=self.patterns,
            important_changes=[f"{v} {m.get('trend')}" for v, m in self.metrics.items()],
            data_quality_flag=self.data_quality_flag,
            analysis_status="complete",
        )


STANDARD_CLINICAL_SCENARIOS: dict[str, ClinicalBenchmarkScenario] = {
    "tachycardia": ClinicalBenchmarkScenario(
        scenario_id="SCENARIO_TACHYCARDIA",
        name="Acute Tachycardia",
        description="Persistent heart rate elevation exceeding 115 bpm with stable blood pressure.",
        risk_level="HIGH RISK",
        metrics={
            "HR": {"trend": "increasing", "latest_value": 120.0, "change_over_window": 25.0},
            "MAP": {"trend": "stable", "latest_value": 82.0, "change_over_window": 0.0},
        },
        patterns=["Isolated upward heart rate trajectory."],
        expected_guideline_ids=["GUIDELINE_TACHYCARDIA_002"],
        expected_priority="ELEVATED",
    ),
    "hypotension": ClinicalBenchmarkScenario(
        scenario_id="SCENARIO_HYPOTENSION",
        name="Hypotension and Circulatory Shock",
        description="Mean Arterial Pressure collapse under 60 mmHg requiring immediate resuscitation.",
        risk_level="HIGH RISK",
        metrics={
            "HR": {"trend": "increasing", "latest_value": 118.0, "change_over_window": 30.0},
            "MAP": {"trend": "decreasing", "latest_value": 56.0, "change_over_window": -24.0},
        },
        patterns=["Simultaneous directional changes across multiple affected vitals."],
        expected_guideline_ids=["GUIDELINE_HEMODYNAMIC_001"],
        expected_priority="URGENT",
    ),
    "respiratory_deterioration": ClinicalBenchmarkScenario(
        scenario_id="SCENARIO_RESPIRATORY",
        name="Acute Hypoxemia and Respiratory Distress",
        description="Falling oxygen saturation under 88% coupled with tachypnea > 30 insp/min.",
        risk_level="HIGH RISK",
        metrics={
            "SpO2": {"trend": "decreasing", "latest_value": 87.0, "change_over_window": -11.0},
            "RR": {"trend": "increasing", "latest_value": 32.0, "change_over_window": 16.0},
        },
        patterns=["Acute respiratory decompensation trajectory."],
        expected_guideline_ids=["GUIDELINE_RESPIRATORY_003"],
        expected_priority="URGENT",
    ),
    "sensor_artifact": ClinicalBenchmarkScenario(
        scenario_id="SCENARIO_SENSOR_ARTIFACT",
        name="Signal Noise and Sensor Disconnect",
        description="High data sparsity and sudden dropout indicating probe displacement.",
        risk_level="HIGH RISK",
        metrics={
            "SpO2": {"trend": "uncertain", "latest_value": 0.0, "change_over_window": -98.0},
            "MAP": {"trend": "uncertain", "latest_value": 0.0, "change_over_window": -80.0},
        },
        patterns=["Telemetry discontinuity and zero-value spike."],
        data_quality_flag=True,
        expected_guideline_ids=["GUIDELINE_ARTIFACT_004"],
        expected_priority="ELEVATED",
    ),
    "multi_vital_deterioration": ClinicalBenchmarkScenario(
        scenario_id="SCENARIO_NEWS2_MULTI",
        name="Multi-Vital Aggregate Deterioration",
        description="Global physiologic failure across hemodynamics, respiration, and temperature.",
        risk_level="HIGH RISK",
        metrics={
            "HR": {"trend": "increasing", "latest_value": 128.0, "change_over_window": 35.0},
            "MAP": {"trend": "decreasing", "latest_value": 52.0, "change_over_window": -28.0},
            "SpO2": {"trend": "decreasing", "latest_value": 89.0, "change_over_window": -9.0},
            "RR": {"trend": "increasing", "latest_value": 34.0, "change_over_window": 18.0},
        },
        patterns=["Simultaneous directional changes across multiple affected vitals."],
        expected_guideline_ids=["GUIDELINE_NEWS2_005", "GUIDELINE_HEMODYNAMIC_001"],
        expected_priority="URGENT",
    ),
    "irrelevant_query": ClinicalBenchmarkScenario(
        scenario_id="SCENARIO_IRRELEVANT",
        name="Irrelevant Non-Clinical Query",
        description="Query unrelated to medical vitals to verify rejection of spurious retrieval.",
        risk_level="LOW RISK",
        metrics={},
        patterns=["Astrophysics satellite orbital calculation telemetry parameters."],
        expected_guideline_ids=[],
        expected_priority="ROUTINE",
    ),
}


def get_all_scenarios() -> list[ClinicalBenchmarkScenario]:
    """Return all standard benchmark scenarios."""
    return list(STANDARD_CLINICAL_SCENARIOS.values())


def get_scenario(key: str) -> ClinicalBenchmarkScenario | None:
    """Retrieve benchmark scenario by key."""
    return STANDARD_CLINICAL_SCENARIOS.get(key)


__all__ = [
    "ClinicalBenchmarkScenario",
    "STANDARD_CLINICAL_SCENARIOS",
    "get_all_scenarios",
    "get_scenario",
]

