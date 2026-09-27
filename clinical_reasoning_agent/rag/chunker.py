"""Document chunking utility for medical guidelines."""

from __future__ import annotations

import re
from typing import Any

from clinical_reasoning_agent.rag.schemas import DocumentChunk, MedicalDocument


class DocumentChunker:
    """Chunks clinical documents into retrieval units preserving provenance metadata."""

    def __init__(
        self,
        max_chunk_words: int = 150,
        overlap_words: int = 25,
        chunk_size_chars: int | None = None,
        overlap_chars: int | None = None,
    ):
        if chunk_size_chars is not None:
            self.max_chunk_words = max(10, chunk_size_chars // 5)
        else:
            self.max_chunk_words = max_chunk_words

        if overlap_chars is not None:
            self.overlap_words = max(2, overlap_chars // 5)
        else:
            self.overlap_words = overlap_words

    def chunk_document(self, document: MedicalDocument) -> list[DocumentChunk]:
        """Split a single medical document into granular chunks."""
        words = document.text.split()
        if not words:
            return []

        # If document is within max size, preserve as single coherent chunk
        if len(words) <= self.max_chunk_words:
            return [
                DocumentChunk(
                    chunk_id=f"{document.document_id}_chk0",
                    document_id=document.document_id,
                    title=document.title,
                    source=document.source,
                    section=document.section,
                    text=document.text,
                    metadata=dict(document.metadata),
                )
            ]

        chunks: list[DocumentChunk] = []
        start = 0
        idx = 0
        step = max(1, self.max_chunk_words - self.overlap_words)

        while start < len(words):
            chunk_words = words[start : start + self.max_chunk_words]
            chunk_text = " ".join(chunk_words).strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}_chk{idx}",
                    document_id=document.document_id,
                    title=document.title,
                    source=document.source,
                    section=document.section,
                    text=chunk_text,
                    metadata=dict(document.metadata),
                )
            )
            idx += 1
            start += step

        return chunks

    def chunk_documents(self, documents: list[MedicalDocument]) -> list[DocumentChunk]:
        """Chunk a collection of documents."""
        all_chunks: list[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


__all__ = ["DocumentChunker"]
