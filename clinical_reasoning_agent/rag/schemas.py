"""Data models and schema definitions for the Medical RAG module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MedicalDocument:
    """An authoritative clinical document loaded into the knowledge base."""

    document_id: str
    title: str
    source: str
    source_type: str  # e.g., "clinical_guideline", "physiological_protocol", "consensus_statement"
    publication_date: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.document_id


@dataclass
class DocumentChunk:
    """A granular, section-bounded chunk of a medical document for retrieval."""

    chunk_id: str
    document_id: str
    title: str
    source: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.document_id


@dataclass
class RetrievedPassage:
    """A retrieved medical passage scored for relevance against a patient finding query."""

    document_id: str
    title: str
    source: str
    section: str
    text: str
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return self.document_id

    @property
    def similarity_score(self) -> float:
        return self.relevance_score

    @property
    def pmid(self) -> str:
        return str(self.metadata.get("pmid", "") or "")

    @property
    def authors(self) -> str:
        return str(self.metadata.get("authors", "") or "")

    @property
    def journal(self) -> str:
        return str(self.metadata.get("journal", "") or "")

    @property
    def year(self) -> str:
        return str(self.metadata.get("year", "") or "")

    @property
    def doi(self) -> str:
        return str(self.metadata.get("doi", "") or "")

    def to_citation_dict(self) -> dict[str, Any]:
        """Convert passage to a citation reference for downstream records."""
        d = {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "section": self.section,
            "relevance_score": round(self.relevance_score, 3),
        }
        if self.pmid:
            d["pmid"] = self.pmid
        if self.authors:
            d["authors"] = self.authors
        if self.journal:
            d["journal"] = self.journal
        if self.year:
            d["year"] = self.year
        if self.doi:
            d["doi"] = self.doi
        return d


@dataclass
class RetrievalResult:
    """Result of querying the medical retriever."""

    query: str
    passages: list[RetrievedPassage] = field(default_factory=list)
    retrieval_status: str = "NO_RELEVANT_EVIDENCE"  # "SUCCESS", "NO_RELEVANT_EVIDENCE", "RETRIEVAL_FAILED"
    top_score: float = 0.0
    error_message: str | None = None
    retrieval_source: str = "curated_fallback"  # "pubmed" | "curated_fallback"

    @property
    def has_evidence(self) -> bool:
        return self.retrieval_status == "SUCCESS" and len(self.passages) > 0


__all__ = [
    "DocumentChunk",
    "MedicalDocument",
    "RetrievalResult",
    "RetrievedPassage",
]
