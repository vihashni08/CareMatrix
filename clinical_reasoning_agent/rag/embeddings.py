"""Local embedding vectorizer for medical knowledge retrieval."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class LocalMedicalEmbedder:
    """Computes lightweight, normalized term-frequency embedding vectors for clinical text."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            stop_words="english",
            token_pattern=r"(?u)\b[A-Za-z0-9_]{2,}\b",
        )
        self._is_fitted = False

    def fit(self, texts: list[str]) -> None:
        """Fit vocabulary on knowledge base corpus texts."""
        if not texts:
            return
        self.vectorizer.fit(texts)
        self._is_fitted = True

    def transform(self, texts: list[str]) -> np.ndarray:
        """Vectorize a list of texts into L2-normalized float arrays."""
        if not self._is_fitted:
            raise RuntimeError("Embedder must be fitted on corpus before transforming.")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray()

    def embed_query(self, query: str) -> np.ndarray:
        """Compute normalized vector for a single search query."""
        if not self._is_fitted:
            raise RuntimeError("Embedder must be fitted on corpus before querying.")
        vec = self.vectorizer.transform([query]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


__all__ = ["LocalMedicalEmbedder"]

