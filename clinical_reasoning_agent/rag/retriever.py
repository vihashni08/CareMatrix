"""Medical retriever coordinating focused query construction and local passage lookup."""

from __future__ import annotations

from typing import Any

from communication.events import DataAnalysisEvent
from clinical_reasoning_agent.rag.chunker import DocumentChunker
from clinical_reasoning_agent.rag.document_loader import DocumentLoader
from clinical_reasoning_agent.rag.schemas import RetrievalResult, RetrievedPassage
from clinical_reasoning_agent.rag.vector_store import LocalVectorStore


def build_focused_medical_query(event: DataAnalysisEvent) -> str:
    """Extract key physiological abnormalities to construct a focused clinical search query.

    Never includes patient identifiers (e.g. patient_id or case_id).
    Focuses strictly on vital trends, risk status, sensor quality, and deterioration patterns.
    """
    risk_level = getattr(event, "risk_level", "LOW RISK")
    metrics = getattr(event, "trend_metrics", {}) or {}
    patterns = getattr(event, "pattern_identified", []) or []
    quality_flag = bool(getattr(event, "data_quality_flag", False))
    consistency = getattr(event, "evidence_consistency", "SUPPORTING")

    query_parts: list[str] = [risk_level]

    # Map directional vital changes to clinical search terms
    for vital, m in metrics.items():
        trend = m.get("trend")
        if vital == "HR" and trend == "increasing":
            query_parts.extend(["tachycardia", "heart rate increasing", "cardiovascular"])
        elif vital == "MAP" and trend == "decreasing":
            query_parts.extend(["hypotension", "MAP decreasing", "shock", "blood pressure collapse"])
        elif vital == "SpO2" and trend == "decreasing":
            query_parts.extend(["hypoxemia", "oxygen desaturation", "SpO2 falling"])
        elif vital == "RR" and trend == "increasing":
            query_parts.extend(["tachypnea", "respiratory rate increasing", "respiratory distress"])
        elif trend in ("increasing", "decreasing"):
            query_parts.append(f"{vital} {trend}")

    # Add data quality troubleshooting terms if flagged
    if quality_flag:
        query_parts.extend(["sensor artifact", "signal noise", "probe calibration", "transducer error"])

    # Add deterioration pattern keywords
    for p in patterns:
        p_lower = p.lower()
        if "simultaneous directional changes" in p_lower or "multiple affected vitals" in p_lower:
            query_parts.extend(["multi-vital deterioration", "NEWS2 escalation"])
            break

    return " ".join(query_parts).strip()


class MedicalRetriever:
    """Retrieves authoritative medical literature passages matching patient physiological patterns."""

    def __init__(
        self,
        vector_store: LocalVectorStore | None = None,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
        min_similarity: float = 0.15,
        top_k: int = 2,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.vector_store = vector_store or LocalVectorStore(min_similarity=min_similarity)
        self.min_similarity = min_similarity
        self.top_k = top_k
        self._is_initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Load and index curated medical documents."""
        try:
            documents = self.loader.load_documents()
            chunks = self.chunker.chunk_documents(documents)
            self.vector_store.add_chunks(chunks)
            self._is_initialized = True
        except Exception:
            self._is_initialized = False

    def retrieve(self, event_or_query: DataAnalysisEvent | str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant clinical guidance for a patient event or raw query string."""
        if isinstance(event_or_query, str):
            query = event_or_query.strip()
        else:
            query = build_focused_medical_query(event_or_query)

        effective_top_k = top_k if top_k is not None else self.top_k

        if not query or not self._is_initialized:
            return RetrievalResult(
                query=query,
                passages=[],
                retrieval_status="NO_RELEVANT_EVIDENCE",
                top_score=0.0,
            )

        try:
            passages = self.vector_store.search(
                query=query,
                top_k=effective_top_k,
                min_similarity=self.min_similarity,
            )

            if passages:
                return RetrievalResult(
                    query=query,
                    passages=passages,
                    retrieval_status="SUCCESS",
                    top_score=passages[0].relevance_score,
                )
            return RetrievalResult(
                query=query,
                passages=[],
                retrieval_status="NO_RELEVANT_EVIDENCE",
                top_score=0.0,
            )

        except Exception as exc:
            return RetrievalResult(
                query=query,
                passages=[],
                retrieval_status="RETRIEVAL_FAILED",
                top_score=0.0,
                error_message=str(exc),
            )


__all__ = [
    "MedicalRetriever",
    "build_focused_medical_query",
]
