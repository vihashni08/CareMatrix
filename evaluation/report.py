"""Evaluation reporting utilities rendering Markdown and formatted console summaries."""

from __future__ import annotations

from typing import Any
from evaluation.agent_metrics import LatencySummary, RAGEvaluationResult, VerificationEvaluationResult
from evaluation.dataset_runner import OrchestrationComparisonResult
from evaluation.metrics import ClassificationMetricsResult


def generate_evaluation_report(results: dict[str, Any]) -> str:
    """Generate comprehensive Markdown evaluation report."""
    dataset_name = results.get("dataset_name", "Clinical Telemetry")
    cases = results.get("cases_evaluated", 0)

    lines: list[str] = [
        f"# CareMatrix Quantitative Evaluation Report: {dataset_name}",
        "",
        "## 1. Executive Summary",
        f"- **Dataset Evaluated**: {dataset_name}",
        f"- **Cohort Size**: {cases} patient cases",
        "",
        "## 2. Adaptive Orchestration vs. Fixed Baseline Comparison",
        "Empirical measurement of evidence-aware analysis depth vs. fixed unconditional detailed pipeline:",
        "",
    ]

    orch: OrchestrationComparisonResult | None = results.get("orchestration_comparison")
    if orch:
        lines.extend([
            "| Metric | Baseline (Fixed Tier) | Adaptive (CareMatrix) | Efficiency Impact |",
            "|---|---|---|---|",
            f"| **Analysis Operations** | {orch.baseline_total_analyses} | {orch.adaptive_total_analyses} | **{orch.unnecessary_analyses_avoided} avoided ({orch.efficiency_gain_pct}%)** |",
            f"| **Execution Time** | {orch.baseline_runtime_ms:.1f} ms | {orch.adaptive_runtime_ms:.1f} ms | **{orch.runtime_saved_ms:.1f} ms saved ({orch.runtime_reduction_pct}%)** |",
            f"| **Verification Required** | N/A | {orch.cases_requiring_verification} cases | Dynamically escalated |",
            f"| **Evidence Conflicts** | N/A | {orch.cases_with_conflicts} cases | Preserved and flagged |",
            "",
        ])

    lines.extend([
        "## 3. Operational Latency Performance Across Agents",
        "System latency benchmarks measured per lifecycle invocation:",
        "",
        "| Pipeline Stage | Invocations | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std Dev (ms) |",
        "|---|---|---|---|---|---|---|",
    ])

    latencies: dict[str, LatencySummary] = results.get("latency_summaries", {})
    for stage, s in latencies.items():
        if s.count > 0:
            lines.append(
                f"| `{s.stage_name}` | {s.count} | {s.mean_ms:.2f} | {s.median_ms:.2f} | {s.min_ms:.2f} | {s.max_ms:.2f} | {s.std_ms:.2f} |"
            )

    verif: VerificationEvaluationResult | None = results.get("verification_summary")
    if verif:
        lines.extend([
            "",
            "## 4. Cross-Agent Verification Distribution",
            f"- **Total Multi-Agent Events**: {verif.total_events}",
            f"- **Supporting Evidence (Concordant)**: {verif.supporting_count} ({(verif.supporting_count / max(1, verif.total_events) * 100):.1f}%)",
            f"- **Conflicting Evidence (Discordant)**: {verif.conflicting_count} ({(verif.conflicting_count / max(1, verif.total_events) * 100):.1f}%)",
            f"- **Uncertain Evidence**: {verif.uncertain_count} ({(verif.uncertain_count / max(1, verif.total_events) * 100):.1f}%)",
            f"- **Verification Required Triggered**: {verif.verification_required_count} ({verif.verification_rate * 100:.1f}%)",
            "",
        ])
        if verif.recorded_conflicts:
            lines.append("### Recorded Discordant Conflict Evidence")
            for c in verif.recorded_conflicts[:5]:
                lines.append(f"- **Event `{c['event_id']}` (Case {c['case_id']})**: {c['consistency']}")
                for ce in c["conflicting_evidence"]:
                    lines.append(f"  * {ce}")

    rag: RAGEvaluationResult | None = results.get("rag_summary")
    if rag:
        lines.extend([
            "",
            "## 5. Evidence-Grounded Medical RAG Evaluation",
            "| Metric | Observed Value | Standard Requirement |",
            "|---|---|---|",
            f"| **Retrieval Success Rate** | {rag.success_rate * 100:.1f}% | Target: > 80% on clinical findings |",
            f"| **No Relevant Evidence Rate** | {rag.no_relevant_rate * 100:.1f}% | Correctly rejects spurious queries |",
            f"| **Mean Retrieved Passages** | {rag.avg_passages_retrieved:.1f} | Top-k capacity: 1 - 2 passages |",
            f"| **Mean Relevance Score** | {rag.avg_relevance_score:.3f} | Cosine similarity threshold >= 0.15 |",
            f"| **Source Provenance Validity** | {rag.provenance_validity_rate * 100:.1f}% | 100% matched to curated guidelines |",
            f"| **Citation Validation Failures** | {rag.citation_validation_failures} | 0 fabricated citations allowed |",
        ])

    risk_res = results.get("risk_metrics", {})
    lines.extend([
        "",
        "## 6. Risk Agent Model Validation Status",
    ])
    if isinstance(risk_res, ClassificationMetricsResult) and risk_res.status == "SUCCESS":
        lines.extend([
            f"- **Accuracy**: {risk_res.accuracy:.4f}",
            f"- **Precision**: {risk_res.precision:.4f}",
            f"- **Recall**: {risk_res.recall:.4f}",
            f"- **F1 Score**: {risk_res.f1:.4f}",
            f"- **AUROC**: {risk_res.roc_auc if risk_res.roc_auc is not None else 'N/A'}",
            f"- **Confusion Matrix**: {risk_res.confusion_matrix}",
        ])
    else:
        msg = risk_res.get("message", "Target ground-truth labels unavailable.") if isinstance(risk_res, dict) else risk_res.message
        lines.extend([
            f"- **Status**: `GROUND_TRUTH_UNAVAILABLE`",
            f"- **Audit Note**: {msg}",
            "- **Preservation Notice**: RandomForestClassifier model weights (`best_model.pkl`), preprocessor (`preprocessor.pkl`), and 0.16 threshold were NOT retrained or modified.",
        ])

    return "\n".join(lines)


__all__ = ["generate_evaluation_report"]

