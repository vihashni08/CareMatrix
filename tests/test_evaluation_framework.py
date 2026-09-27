"""Unit tests for CareMatrix quantitative evaluation framework."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from evaluation.agent_metrics import (
    LatencySummary,
    LatencyTracker,
    RAGEvaluationResult,
    RAGEvaluator,
    VerificationEvaluationResult,
    VerificationEvaluator,
)
from evaluation.dataset_runner import DatasetRunner, OrchestrationComparisonResult
from evaluation.metrics import (
    ClassificationMetricsResult,
    GROUND_TRUTH_UNAVAILABLE,
    compute_classification_metrics,
)
from evaluation.report import generate_evaluation_report
from evaluation.scenarios import (
    STANDARD_CLINICAL_SCENARIOS,
    get_all_scenarios,
    get_scenario,
)
from clinical_reasoning_agent.rag.schemas import RetrievalResult, RetrievedPassage


class TestEvaluationFramework(unittest.TestCase):
    def test_1_classification_metrics_computation(self):
        """Classification metrics calculate accuracy, precision, recall, and F1 accurately."""
        y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
        y_prob = [0.9, 0.1, 0.8, 0.3, 0.2, 0.7, 0.85, 0.15]

        res = compute_classification_metrics(y_true, y_pred, y_prob)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.sample_count, 8)
        self.assertGreater(res.accuracy, 0.6)
        self.assertGreater(res.precision, 0.6)
        self.assertGreater(res.recall, 0.6)
        self.assertGreater(res.f1, 0.6)
        self.assertIsNotNone(res.roc_auc)
        self.assertEqual(res.confusion_matrix["TP"], 3)
        self.assertEqual(res.confusion_matrix["TN"], 3)

    def test_2_ground_truth_unavailable_handling(self):
        """When reference outcomes are unavailable, evaluator reports GROUND_TRUTH_UNAVAILABLE."""
        res_none = compute_classification_metrics(None, [1, 0, 1])
        self.assertEqual(res_none.status, GROUND_TRUTH_UNAVAILABLE)
        self.assertIn("not available", res_none.message.lower())

        res_empty = compute_classification_metrics([], [])
        self.assertEqual(res_empty.status, GROUND_TRUTH_UNAVAILABLE)

    def test_3_latency_tracker_statistics(self):
        """LatencyTracker records execution latencies and computes mean, median, min, max, std."""
        tracker = LatencyTracker()
        tracker.record("risk", 0.010)  # 10 ms
        tracker.record("risk", 0.020)  # 20 ms
        tracker.record("risk", 0.030)  # 30 ms

        summary = tracker.get_stage_summary("risk")
        self.assertEqual(summary.count, 3)
        self.assertAlmostEqual(summary.mean_ms, 20.0, places=1)
        self.assertAlmostEqual(summary.median_ms, 20.0, places=1)
        self.assertAlmostEqual(summary.min_ms, 10.0, places=1)
        self.assertAlmostEqual(summary.max_ms, 30.0, places=1)

    def test_4_verification_evaluator_tracking(self):
        """VerificationEvaluator tracks distribution and preserves discordant conflict evidence."""
        evaluator = VerificationEvaluator()
        evaluator.record_decision(
            event_id="evt_01",
            case_id=101,
            consistency="SUPPORTING",
            verification_required=False,
            conflicting_evidence=[],
        )
        evaluator.record_decision(
            event_id="evt_02",
            case_id=102,
            consistency="CONFLICTING",
            verification_required=True,
            conflicting_evidence=["Model predicted HIGH RISK but vital signs are flat."],
        )

        summary = evaluator.summarize()
        self.assertEqual(summary.total_events, 2)
        self.assertEqual(summary.supporting_count, 1)
        self.assertEqual(summary.conflicting_count, 1)
        self.assertEqual(summary.verification_required_count, 1)
        self.assertEqual(summary.verification_rate, 0.5)
        self.assertEqual(len(summary.recorded_conflicts), 1)
        self.assertEqual(summary.recorded_conflicts[0]["event_id"], "evt_02")

    def test_5_rag_evaluator_metrics(self):
        """RAGEvaluator computes retrieval success rates and verifies document provenance."""
        evaluator = RAGEvaluator()

        res_success = RetrievalResult(
            query="tachycardia",
            passages=[
                RetrievedPassage(
                    document_id="GUIDELINE_TACHYCARDIA_002",
                    title="Acute Tachycardia",
                    source="ACC",
                    section="General",
                    text="Guidelines on tachycardia...",
                    relevance_score=0.85,
                )
            ],
            retrieval_status="SUCCESS",
        )
        evaluator.record_retrieval(res_success, latency_seconds=0.005, cited_doc_ids=["GUIDELINE_TACHYCARDIA_002"])

        res_none = RetrievalResult(
            query="astronomy",
            passages=[],
            retrieval_status="NO_RELEVANT_EVIDENCE",
        )
        evaluator.record_retrieval(res_none, latency_seconds=0.002)

        summary = evaluator.summarize()
        self.assertEqual(summary.total_queries, 2)
        self.assertEqual(summary.success_count, 1)
        self.assertEqual(summary.no_relevant_evidence_count, 1)
        self.assertEqual(summary.success_rate, 0.5)
        self.assertEqual(summary.provenance_validity_rate, 1.0)
        self.assertEqual(summary.citation_validation_failures, 0)
        self.assertIsNotNone(summary.retrieval_latencies)

    def test_6_clinical_benchmark_scenarios(self):
        """Standard clinical scenarios convert into DataAnalysisEvents for benchmarking."""
        scenarios = get_all_scenarios()
        self.assertGreaterEqual(len(scenarios), 6)

        hypo = get_scenario("hypotension")
        self.assertIsNotNone(hypo)
        self.assertEqual(hypo.risk_level, "HIGH RISK")
        event = hypo.to_data_analysis_event(case_id=77)
        self.assertEqual(event.case_id, 77)
        self.assertIn("HR", event.trend_metrics)
        self.assertIn("MAP", event.trend_metrics)

    def test_7_dataset_runner_adaptive_vs_baseline(self):
        """DatasetRunner executes cohort evaluation comparing adaptive depth vs fixed baseline."""
        mock_adapter = MagicMock()
        mock_adapter.dataset_name = "MockClinical"
        mock_adapter.list_cases.return_value = [1, 2]

        from data.schemas import CommonPatientObservation
        # Case with normal observations
        obs_case = [
            CommonPatientObservation(timestamp=float(i), HR=72.0, MAP=85.0, SpO2=98.0, RR=14.0)
            for i in range(20)
        ]
        mock_adapter.load_case.return_value = obs_case

        runner = DatasetRunner(adapter=mock_adapter, enable_llm=False, enable_rag=False, max_samples_per_case=20)
        results = runner.run_cohort_evaluation(case_ids=[1, 2])

        self.assertEqual(results["dataset_name"], "MockClinical")
        self.assertEqual(results["cases_evaluated"], 2)
        orch: OrchestrationComparisonResult = results["orchestration_comparison"]
        self.assertEqual(orch.total_cases, 2)
        self.assertGreaterEqual(orch.baseline_total_analyses, orch.adaptive_total_analyses)
        self.assertGreaterEqual(orch.unnecessary_analyses_avoided, 0)

    def test_8_report_generation(self):
        """generate_evaluation_report formats structured markdown tables."""
        orch = OrchestrationComparisonResult(
            total_cases=2,
            baseline_total_analyses=4,
            adaptive_total_analyses=2,
            unnecessary_analyses_avoided=2,
            efficiency_gain_pct=50.0,
            baseline_runtime_ms=100.0,
            adaptive_runtime_ms=50.0,
            runtime_saved_ms=50.0,
            runtime_reduction_pct=50.0,
            cases_requiring_verification=0,
            cases_with_conflicts=0,
        )
        results = {
            "dataset_name": "TestDataset",
            "cases_evaluated": 2,
            "orchestration_comparison": orch,
            "latency_summaries": {
                "risk": LatencySummary("risk", 2, 12.5, 12.5, 10.0, 15.0, 2.5),
            },
            "verification_summary": VerificationEvaluationResult(2, 2, 0, 0, 0, 0.0, []),
            "risk_metrics": {"status": GROUND_TRUTH_UNAVAILABLE, "message": "No outcomes available."},
        }

        report = generate_evaluation_report(results)
        self.assertIn("# CareMatrix Quantitative Evaluation Report: TestDataset", report)
        self.assertIn("2 avoided (50.0%)", report)
        self.assertIn("GROUND_TRUTH_UNAVAILABLE", report)


if __name__ == "__main__":
    unittest.main()
