"""Quantitative evaluation framework for the CareMatrix multi-agent system."""

from __future__ import annotations

from evaluation.agent_metrics import (
    LatencySummary,
    LatencyTracker,
    RAGEvaluationResult,
    RAGEvaluator,
    VerificationEvaluationResult,
    VerificationEvaluator,
)
from evaluation.dataset_runner import (
    DatasetRunner,
    OrchestrationComparisonResult,
)
from evaluation.metrics import (
    ClassificationMetricsResult,
    GROUND_TRUTH_UNAVAILABLE,
    compute_classification_metrics,
)
from evaluation.report import generate_evaluation_report
from evaluation.scenarios import (
    ClinicalBenchmarkScenario,
    STANDARD_CLINICAL_SCENARIOS,
    get_all_scenarios,
    get_scenario,
)

__all__ = [
    "ClassificationMetricsResult",
    "ClinicalBenchmarkScenario",
    "DatasetRunner",
    "GROUND_TRUTH_UNAVAILABLE",
    "LatencySummary",
    "LatencyTracker",
    "OrchestrationComparisonResult",
    "RAGEvaluationResult",
    "RAGEvaluator",
    "STANDARD_CLINICAL_SCENARIOS",
    "VerificationEvaluationResult",
    "VerificationEvaluator",
    "compute_classification_metrics",
    "generate_evaluation_report",
    "get_all_scenarios",
    "get_scenario",
]

