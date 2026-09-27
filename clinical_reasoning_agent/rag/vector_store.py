"""In-memory vector store for medical knowledge passages."""

from __future__ import annotations

import numpy as np

from clinical_reasoning_agent.rag.embeddings import LocalMedicalEmbedder
from clinical_reasoning_agent.rag.schemas import DocumentChunk, RetrievedPassage


class LocalVectorStore:
    """In-memory vector store computing cosine similarity against clinical document chunks."""

    def __init__(self, embedder: LocalMedicalEmbedder | None = None, min_similarity: float = 0.15):
        self.embedder = embedder or LocalMedicalEmbedder()
        self.min_similarity = min_similarity
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and index document chunks."""
        if not chunks:
            self.chunks = []
            self.embeddings = None
            return

        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        self.embeddings = self.embedder.transform(texts)
        self.chunks = list(chunks)

    index_chunks = add_chunks

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 2,
        min_similarity: float | None = None,
    ) -> list[RetrievedPassage]:
        """Retrieve top-k passages matching query with score >= min_similarity."""
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        threshold = min_similarity if min_similarity is not None else self.min_similarity
        query_vec = self.embedder.embed_query(query)

        # Cosine similarity is dot product because vectors are L2-normalized
        scores = np.dot(self.embeddings, query_vec)
        if len(scores) == 0:
            return []

        sorted_indices = np.argsort(scores)[::-1]

        results: list[RetrievedPassage] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score < threshold:
                break
            chunk = self.chunks[idx]
            results.append(
                RetrievedPassage(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    source=chunk.source,
                    section=chunk.section,
                    text=chunk.text,
                    relevance_score=score,
                    metadata=dict(chunk.metadata),
                )
            )
            if len(results) >= top_k:
                break

        return results


__all__ = ["LocalVectorStore"]
