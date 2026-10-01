"""Medical retriever coordinating focused query construction and local passage lookup."""

from __future__ import annotations

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
    # consistency is extracted but influences query indirectly through caller gating
    _ = getattr(event, "evidence_consistency", "SUPPORTING")

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
    """Retrieves authoritative medical literature passages matching patient physiological patterns.

    Retrieval strategy (in order):

    1. **PubMed** (live) — calls ``fetch_pubmed_abstracts`` with the focused query.
       On success, wraps each abstract as a ``RetrievedPassage`` and returns
       ``retrieval_source="pubmed"``.
    2. **Curated TF-IDF fallback** — used when PubMed is unreachable, times out,
       or returns zero results.  The same local vector store that was previously
       the sole source; ``retrieval_source="curated_fallback"``.

    PubMed is only called when the existing progressive retrieval gate (in
    ``ClinicalReasoningAgent``) has already decided that retrieval should run.
    This method does not bypass or add any gating condition.
    """

    def __init__(
        self,
        vector_store: LocalVectorStore | None = None,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
        min_similarity: float = 0.15,
        top_k: int = 2,
        enable_pubmed: bool = True,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.vector_store = vector_store or LocalVectorStore(min_similarity=min_similarity)
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.enable_pubmed = enable_pubmed
        self._is_initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Load and index curated medical documents (offline fallback corpus)."""
        try:
            documents = self.loader.load_documents()
            chunks = self.chunker.chunk_documents(documents)
            self.vector_store.add_chunks(chunks)
            self._is_initialized = True
        except Exception:
            self._is_initialized = False

    # ------------------------------------------------------------------
    # PubMed path (primary)
    # ------------------------------------------------------------------
    def _retrieve_pubmed(self, query: str, effective_top_k: int) -> RetrievalResult | None:
        """Attempt live PubMed retrieval.

        Returns a populated ``RetrievalResult`` (``retrieval_source="pubmed"``)
        on success, or ``None`` if PubMed returned nothing or failed (including
        any exception raised by ``fetch_pubmed_abstracts``).
        """
        try:
            from clinical_reasoning_agent.rag.pubmed_client import fetch_pubmed_abstracts  # noqa: PLC0415
            candidate_k = max(effective_top_k, 3)
            abstracts = fetch_pubmed_abstracts(query, max_results=candidate_k)
        except Exception:
            # Import error, unexpected exception from the client, or any other failure.
            return None

        if not abstracts:
            return None

        passages: list[RetrievedPassage] = []
        for i, ab in enumerate(abstracts):
            pmid = ab.get("pmid", "UNKNOWN")
            title = ab.get("title", "Untitled PubMed Article")
            abstract_text = ab.get("abstract", "")
            if not abstract_text:
                continue
            authors = ab.get("authors", "")
            journal = ab.get("journal", "")
            year = ab.get("year", "")
            doi = ab.get("doi", "")
            relevance_score = max(0.95 - i * 0.08, 0.50)
            passages.append(
                RetrievedPassage(
                    document_id=f"PMID_{pmid}",
                    title=title,
                    source=f"PubMed PMID:{pmid}",
                    section="Abstract",
                    text=abstract_text,
                    relevance_score=relevance_score,
                    metadata={
                        "pmid": pmid,
                        "retrieval_source": "pubmed",
                        "authors": authors,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                    },
                )
            )

        if not passages:
            return None

        selected_passages = passages[:effective_top_k]

        return RetrievalResult(
            query=query,
            passages=selected_passages,
            retrieval_status="SUCCESS",
            top_score=selected_passages[0].relevance_score,
            retrieval_source="pubmed",
        )

    # ------------------------------------------------------------------
    # Curated TF-IDF path (offline fallback)
    # ------------------------------------------------------------------
    def _retrieve_curated(self, query: str, effective_top_k: int) -> RetrievalResult:
        """Retrieve from the local curated TF-IDF corpus (offline fallback)."""
        if not query or not self._is_initialized:
            return RetrievalResult(
                query=query,
                passages=[],
                retrieval_status="NO_RELEVANT_EVIDENCE",
                top_score=0.0,
                retrieval_source="curated_fallback",
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
                    retrieval_source="curated_fallback",
                )
            return RetrievalResult(
                query=query,
                passages=[],
                retrieval_status="NO_RELEVANT_EVIDENCE",
                top_score=0.0,
                retrieval_source="curated_fallback",
            )
        except Exception as exc:
            return RetrievalResult(
                query=query,
                passages=[],
                retrieval_status="RETRIEVAL_FAILED",
                top_score=0.0,
                error_message=str(exc),
                retrieval_source="curated_fallback",
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def retrieve(self, event_or_query: DataAnalysisEvent | str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant clinical guidance for a patient event or raw query string.

        Tries PubMed first (when ``enable_pubmed=True``); falls back to the
        curated local TF-IDF corpus on any PubMed failure or empty result.
        """
        if isinstance(event_or_query, str):
            query = event_or_query.strip()
        else:
            query = build_focused_medical_query(event_or_query)

        effective_top_k = top_k if top_k is not None else self.top_k

        # 1. Try PubMed (primary source)
        if self.enable_pubmed and query:
            pubmed_result = self._retrieve_pubmed(query, effective_top_k)
            if pubmed_result is not None:
                return pubmed_result

        # 2. Fall back to curated local corpus
        return self._retrieve_curated(query, effective_top_k)


__all__ = [
    "MedicalRetriever",
    "build_focused_medical_query",
]
