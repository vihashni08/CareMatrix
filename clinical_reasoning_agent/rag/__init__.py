"""Evidence-Grounded Medical Retrieval-Augmented Generation (RAG) module for CareMatrix."""

from clinical_reasoning_agent.rag.chunker import DocumentChunker
from clinical_reasoning_agent.rag.document_loader import CURATED_MEDICAL_KNOWLEDGE, DocumentLoader
from clinical_reasoning_agent.rag.embeddings import LocalMedicalEmbedder
from clinical_reasoning_agent.rag.retriever import MedicalRetriever, build_focused_medical_query
from clinical_reasoning_agent.rag.schemas import (
    DocumentChunk,
    MedicalDocument,
    RetrievalResult,
    RetrievedPassage,
)
from clinical_reasoning_agent.rag.vector_store import LocalVectorStore

__all__ = [
    "CURATED_MEDICAL_KNOWLEDGE",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentLoader",
    "LocalMedicalEmbedder",
    "LocalVectorStore",
    "MedicalDocument",
    "MedicalRetriever",
    "RetrievalResult",
    "RetrievedPassage",
    "build_focused_medical_query",
]

