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

    def to_citation_dict(self) -> dict[str, Any]:
        """Convert passage to a lightweight citation reference for downstream records."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "section": self.section,
            "relevance_score": round(self.relevance_score, 3),
        }


@dataclass
class RetrievalResult:
    """Result of querying the medical retriever."""

    query: str
    passages: list[RetrievedPassage] = field(default_factory=list)
    retrieval_status: str = "NO_RELEVANT_EVIDENCE"  # "SUCCESS", "NO_RELEVANT_EVIDENCE", "ERROR"
    top_score: float = 0.0
    error_message: str | None = None

    @property
    def has_evidence(self) -> bool:
        return self.retrieval_status == "SUCCESS" and len(self.passages) > 0


__all__ = [
    "DocumentChunk",
    "MedicalDocument",
    "RetrievalResult",
    "RetrievedPassage",
]
