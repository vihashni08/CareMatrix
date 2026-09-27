"""Agent-level performance, latency, verification, and RAG evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any
import numpy as np

from clinical_reasoning_agent.rag.document_loader import CURATED_MEDICAL_KNOWLEDGE
from clinical_reasoning_agent.rag.schemas import RetrievalResult


@dataclass
class LatencySummary:
    """Statistical summary of operational latencies for a pipeline stage."""

    stage_name: str
    count: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "std_ms": round(self.std_ms, 2),
        }


class LatencyTracker:
    """Collects and aggregates timing measurements across all agent lifecycle stages."""

    def __init__(self):
        self._timings: dict[str, list[float]] = {
            "monitoring": [],
            "risk": [],
            "data_analysis": [],
            "clinical_reasoning": [],
            "rag_retrieval": [],
            "gemini_reasoning": [],
            "end_to_end": [],
        }

    def record(self, stage: str, duration_seconds: float) -> None:
        """Record a single latency observation in seconds."""
        if stage not in self._timings:
            self._timings[stage] = []
        self._timings[stage].append(duration_seconds * 1000.0)  # Store in ms

    def get_stage_summary(self, stage: str) -> LatencySummary:
        """Calculate statistics for a specific stage."""
        values = self._timings.get(stage, [])
        if not values:
            return LatencySummary(
                stage_name=stage,
                count=0,
                mean_ms=0.0,
                median_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                std_ms=0.0,
            )
        arr = np.array(values, dtype=float)
        return LatencySummary(
            stage_name=stage,
            count=len(arr),
            mean_ms=float(np.mean(arr)),
            median_ms=float(np.median(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            std_ms=float(np.std(arr)),
        )

    def get_all_summaries(self) -> dict[str, LatencySummary]:
        """Calculate statistics for all recorded stages."""
        return {stage: self.get_stage_summary(stage) for stage in self._timings}


@dataclass
class VerificationEvaluationResult:
    """Summary of cross-agent consistency verification outcomes."""

    total_events: int = 0
    supporting_count: int = 0
    conflicting_count: int = 0
    uncertain_count: int = 0
    verification_required_count: int = 0
    verification_rate: float = 0.0
    recorded_conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "supporting_count": self.supporting_count,
            "conflicting_count": self.conflicting_count,
            "uncertain_count": self.uncertain_count,
            "verification_required_count": self.verification_required_count,
            "verification_rate": round(self.verification_rate, 4),
            "recorded_conflicts_count": len(self.recorded_conflicts),
        }


class VerificationEvaluator:
    """Evaluates cross-agent verification consistency and preserves conflict evidence."""

    def __init__(self):
        self.total_events: int = 0
        self.supporting_count: int = 0
        self.conflicting_count: int = 0
        self.uncertain_count: int = 0
        self.verification_required_count: int = 0
        self.conflicts: list[dict[str, Any]] = []

    def record_decision(
        self,
        event_id: str,
        case_id: int | str,
        consistency: str,
        verification_required: bool,
        conflicting_evidence: list[str] | None = None,
    ) -> None:
        """Record an evaluation decision."""
        self.total_events += 1
        c_upper = str(consistency).upper()

        if "SUPPORT" in c_upper:
            self.supporting_count += 1
        elif "CONFLICT" in c_upper:
            self.conflicting_count += 1
        else:
            self.uncertain_count += 1

        if verification_required:
            self.verification_required_count += 1

        if (conflicting_evidence and len(conflicting_evidence) > 0) or "CONFLICT" in c_upper:
            self.conflicts.append({
                "event_id": event_id,
                "case_id": case_id,
                "consistency": consistency,
                "conflicting_evidence": list(conflicting_evidence or []),
            })

    def summarize(self) -> VerificationEvaluationResult:
        """Produce structured verification assessment."""
        v_rate = (self.verification_required_count / self.total_events) if self.total_events > 0 else 0.0
        return VerificationEvaluationResult(
            total_events=self.total_events,
            supporting_count=self.supporting_count,
            conflicting_count=self.conflicting_count,
            uncertain_count=self.uncertain_count,
            verification_required_count=self.verification_required_count,
            verification_rate=v_rate,
            recorded_conflicts=list(self.conflicts),
        )


@dataclass
class RAGEvaluationResult:
    """Quantitative performance metrics for Medical RAG retrieval."""

    total_queries: int = 0
    success_count: int = 0
    no_relevant_evidence_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    no_relevant_rate: float = 0.0
    avg_passages_retrieved: float = 0.0
    avg_relevance_score: float = 0.0
    provenance_validity_rate: float = 1.0
    citation_validation_failures: int = 0
    retrieval_latencies: LatencySummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "success_count": self.success_count,
            "no_relevant_evidence_count": self.no_relevant_evidence_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 4),
            "no_relevant_rate": round(self.no_relevant_rate, 4),
            "avg_passages_retrieved": round(self.avg_passages_retrieved, 2),
            "avg_relevance_score": round(self.avg_relevance_score, 3),
            "provenance_validity_rate": round(self.provenance_validity_rate, 4),
            "citation_validation_failures": self.citation_validation_failures,
        }


class RAGEvaluator:
    """Evaluates Medical RAG retrieval quality, latency, provenance, and citation fidelity."""

    def __init__(self):
        self.known_doc_ids = {d["document_id"] for d in CURATED_MEDICAL_KNOWLEDGE}
        self.queries_count: int = 0
        self.success_count: int = 0
        self.no_relevant_count: int = 0
        self.error_count: int = 0
        self.passage_counts: list[int] = []
        self.scores: list[float] = []
        self.provenance_checks: list[bool] = []
        self.citation_failures: int = 0
        self.latencies: list[float] = []

    def record_retrieval(
        self,
        result: RetrievalResult,
        latency_seconds: float | None = None,
        cited_doc_ids: list[str] | None = None,
    ) -> None:
        """Record a single retrieval result and inspect provenance."""
        self.queries_count += 1
        if latency_seconds is not None:
            self.latencies.append(latency_seconds * 1000.0)

        status = getattr(result, "retrieval_status", "NO_RELEVANT_EVIDENCE")
        if status == "SUCCESS":
            self.success_count += 1
        elif status == "NO_RELEVANT_EVIDENCE":
            self.no_relevant_count += 1
        else:
            self.error_count += 1

        passages = getattr(result, "passages", []) or []
        self.passage_counts.append(len(passages))

        retrieved_ids = set()
        for p in passages:
            doc_id = getattr(p, "document_id", "")
            retrieved_ids.add(doc_id)
            self.provenance_checks.append(doc_id in self.known_doc_ids)
            score = getattr(p, "relevance_score", 0.0)
            self.scores.append(float(score))

        # Check citation alignment: if citations were made, do they match retrieved passages?
        if cited_doc_ids:
            for cid in cited_doc_ids:
                if cid not in retrieved_ids:
                    self.citation_failures += 1

    def summarize(self) -> RAGEvaluationResult:
        """Compute aggregate RAG performance statistics."""
        n = self.queries_count
        succ_rate = (self.success_count / n) if n > 0 else 0.0
        no_rel_rate = (self.no_relevant_count / n) if n > 0 else 0.0
        avg_pass = float(np.mean(self.passage_counts)) if self.passage_counts else 0.0
        avg_score = float(np.mean(self.scores)) if self.scores else 0.0
        prov_rate = (sum(self.provenance_checks) / len(self.provenance_checks)) if self.provenance_checks else 1.0

        lat_summary = None
        if self.latencies:
            arr = np.array(self.latencies)
            lat_summary = LatencySummary(
                stage_name="rag_retrieval",
                count=len(arr),
                mean_ms=float(np.mean(arr)),
                median_ms=float(np.median(arr)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
                std_ms=float(np.std(arr)),
            )

        return RAGEvaluationResult(
            total_queries=n,
            success_count=self.success_count,
            no_relevant_evidence_count=self.no_relevant_count,
            error_count=self.error_count,
            success_rate=succ_rate,
            no_relevant_rate=no_rel_rate,
            avg_passages_retrieved=avg_pass,
            avg_relevance_score=avg_score,
            provenance_validity_rate=prov_rate,
            citation_validation_failures=self.citation_failures,
            retrieval_latencies=lat_summary,
        )


__all__ = [
    "LatencySummary",
    "LatencyTracker",
    "RAGEvaluationResult",
    "RAGEvaluator",
    "VerificationEvaluationResult",
    "VerificationEvaluator",
]
