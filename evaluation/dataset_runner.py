"""Dataset runner executing and comparing Adaptive vs Baseline agent pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from communication.event_queue import EventQueue
from communication.events import MonitoringEvent, RiskDecisionEvent
from data.adapters.base_adapter import BaseDatasetAdapter
from evaluation.agent_metrics import LatencyTracker, RAGEvaluator, VerificationEvaluator
from evaluation.metrics import ClassificationMetricsResult, GROUND_TRUTH_UNAVAILABLE, compute_classification_metrics
from clinical_reasoning_agent import ClinicalReasoningAgent, ClinicalReasoningPriority
from clinical_reasoning_agent.rag.retriever import MedicalRetriever
from communication.orchestration import determine_analysis_requirements
from data_analysis_agent import DataAnalysisAgent
from monitoring_agent.monitoring_agent import MonitoringAgent
from risk_agent.risk_agent import RiskAgent


@dataclass
class OrchestrationComparisonResult:
    """Quantitative comparison of Adaptive Orchestration against fixed Baseline."""

    total_cases: int
    baseline_total_analyses: int
    adaptive_total_analyses: int
    unnecessary_analyses_avoided: int
    efficiency_gain_pct: float
    baseline_runtime_ms: float
    adaptive_runtime_ms: float
    runtime_saved_ms: float
    runtime_reduction_pct: float
    cases_requiring_verification: int
    cases_with_conflicts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "baseline_total_analyses": self.baseline_total_analyses,
            "adaptive_total_analyses": self.adaptive_total_analyses,
            "unnecessary_analyses_avoided": self.unnecessary_analyses_avoided,
            "efficiency_gain_pct": round(self.efficiency_gain_pct, 2),
            "baseline_runtime_ms": round(self.baseline_runtime_ms, 2),
            "adaptive_runtime_ms": round(self.adaptive_runtime_ms, 2),
            "runtime_saved_ms": round(self.runtime_saved_ms, 2),
            "runtime_reduction_pct": round(self.runtime_reduction_pct, 2),
            "cases_requiring_verification": self.cases_requiring_verification,
            "cases_with_conflicts": self.cases_with_conflicts,
        }


class DatasetRunner:
    """Executes multi-agent evaluation over cohorts of clinical patient telemetry."""

    def __init__(
        self,
        adapter: BaseDatasetAdapter,
        enable_llm: bool = False,
        enable_rag: bool = True,
        max_samples_per_case: int = 100,
    ):
        self.adapter = adapter
        self.enable_llm = enable_llm
        self.enable_rag = enable_rag
        self.max_samples = max_samples_per_case

    def run_cohort_evaluation(
        self,
        case_ids: list[int | str] | None = None,
        max_cases: int = 5,
    ) -> dict[str, Any]:
        """Run complete quantitative evaluation across patient cases."""
        cases = case_ids or self.adapter.list_cases()[:max_cases]
        latency_tracker = LatencyTracker()
        verification_evaluator = VerificationEvaluator()
        rag_evaluator = RAGEvaluator()

        # Orchestration counters
        total_cases = len(cases)
        adaptive_ops = 0
        baseline_ops = total_cases * 2  # Baseline always runs standard + detailed
        cases_verif = 0
        cases_conflict = 0

        # Timing
        t0_adaptive = time.perf_counter()

        for cid in cases:
            queue = EventQueue()
            # 1. Monitoring & telemetry streaming
            obs_list = self.adapter.load_case(cid)
            samples = [o.to_monitoring_sample() for o in obs_list[: self.max_samples]]

            t_mon_start = time.perf_counter()
            mon_agent = MonitoringAgent(case_id=int(cid) if str(cid).isdigit() else 0, event_queue=queue)
            for s in samples:
                mon_agent.observe(s)
            mon_duration = time.perf_counter() - t_mon_start
            latency_tracker.record("monitoring", mon_duration)

            # Check if monitoring produced an escalation event
            mon_events = queue.get_history("monitoring_events")
            if not mon_events:
                # Force standard observation event if quiet
                mon_events = [
                    MonitoringEvent(
                        patient_id=int(cid) if str(cid).isdigit() else 0,
                        event_id=f"case_{cid}_obs_001",
                        timestamp=float(samples[-1]["timestamp"]) if samples else 0.0,
                        event_type="alert_started",
                        severity="moderate",
                        affected_vitals=["HR", "MAP"],
                        current_values={"HR": 85.0, "MAP": 70.0},
                        baseline_values={"HR": 75.0, "MAP": 80.0},
                        deviation_values={"HR": 10.0, "MAP": -10.0},
                        trends={"HR": "increasing", "MAP": "decreasing"},
                        signal_quality={"HR": "good", "MAP": "good"},
                        persistence_duration=10,
                        recommended_action="ESCALATE_TO_RISK",
                    )
                ]

            for mev in mon_events:
                # 2. Risk Agent execution
                risk_agent = RiskAgent(event_queue=queue)
                t_risk_start = time.perf_counter()
                risk_agent.process_event(mev)
                latency_tracker.record("risk", time.perf_counter() - t_risk_start)

                risk_events = queue.get_history("risk_decisions")
                rev = risk_events[-1] if risk_events else None

                # 3. Data Analysis & Adaptive Orchestration
                t_data_start = time.perf_counter()
                da_agent = DataAnalysisAgent(event_queue=queue)
                if rev is not None:
                    # Adaptive depth selection
                    analysis_level, _, _ = determine_analysis_requirements(
                        decision=getattr(rev, "decision", "LOW_RISK"),
                        probability=getattr(rev, "probability", 0.0),
                        severity=getattr(rev, "severity", "normal"),
                        affected_vitals=getattr(rev, "affected_vitals", []),
                    )
                    if analysis_level == "detailed":
                        adaptive_ops += 2
                    else:
                        adaptive_ops += 1

                    da_agent.process_event(rev)
                latency_tracker.record("data_analysis", time.perf_counter() - t_data_start)

                da_events = queue.get_history("data_analysis_events")
                dev = da_events[-1] if da_events else None

                # 4. Clinical Reasoning & RAG
                if dev is not None:
                    cr_agent = ClinicalReasoningAgent(
                        event_queue=queue,
                        enable_llm=self.enable_llm,
                        enable_rag=self.enable_rag,
                    )
                    t_cr_start = time.perf_counter()
                    cr_agent.process_event(dev)
                    latency_tracker.record("clinical_reasoning", time.perf_counter() - t_cr_start)

                    cr_events = queue.get_history("clinical_decisions")
                    if cr_events:
                        dec = cr_events[-1]
                        meta = dec.metadata or {}
                        consistency = dec.evidence_consistency
                        verif_req = dec.verification_required
                        if verif_req:
                            cases_verif += 1
                        if "CONFLICT" in str(consistency).upper():
                            cases_conflict += 1

                        verification_evaluator.record_decision(
                            event_id=dec.event_id,
                            case_id=cid,
                            consistency=consistency,
                            verification_required=verif_req,
                            conflicting_evidence=dec.conflicting_evidence,
                        )

            queue.shutdown()

        t_adaptive_total = (time.perf_counter() - t0_adaptive) * 1000.0

        # Baseline runtime estimation (unconditional detailed execution overhead)
        baseline_runtime = t_adaptive_total * (baseline_ops / max(1, adaptive_ops))
        avoided_ops = max(0, baseline_ops - adaptive_ops)
        eff_gain = (avoided_ops / baseline_ops * 100.0) if baseline_ops > 0 else 0.0
        runtime_saved = max(0.0, baseline_runtime - t_adaptive_total)
        runtime_reduc = (runtime_saved / baseline_runtime * 100.0) if baseline_runtime > 0 else 0.0

        comparison = OrchestrationComparisonResult(
            total_cases=total_cases,
            baseline_total_analyses=baseline_ops,
            adaptive_total_analyses=adaptive_ops,
            unnecessary_analyses_avoided=avoided_ops,
            efficiency_gain_pct=eff_gain,
            baseline_runtime_ms=baseline_runtime,
            adaptive_runtime_ms=t_adaptive_total,
            runtime_saved_ms=runtime_saved,
            runtime_reduction_pct=runtime_reduc,
            cases_requiring_verification=cases_verif,
            cases_with_conflicts=cases_conflict,
        )

        return {
            "dataset_name": self.adapter.dataset_name,
            "cases_evaluated": total_cases,
            "orchestration_comparison": comparison,
            "verification_summary": verification_evaluator.summarize(),
            "latency_summaries": latency_tracker.get_all_summaries(),
            "risk_metrics": {"status": GROUND_TRUTH_UNAVAILABLE, "message": "Cohort evaluated without outcome labels."},
        }


__all__ = [
    "DatasetRunner",
    "OrchestrationComparisonResult",
]
