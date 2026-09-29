"""Tests for the PubMed RAG integration (no real network calls)."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from clinical_reasoning_agent.rag import pubmed_client
from clinical_reasoning_agent.rag.pubmed_client import (
    clear_cache,
    fetch_pubmed_abstracts,
)
from clinical_reasoning_agent.rag.retriever import MedicalRetriever
from clinical_reasoning_agent.rag.schemas import RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAMPLE_ABSTRACTS = [
    {
        "pmid": "12345678",
        "title": "Hypotension and Tachycardia in Shock",
        "abstract": "This study examines haemodynamic instability in septic shock patients.",
    },
    {
        "pmid": "87654321",
        "title": "Tachycardia Monitoring in ICU",
        "abstract": "Heart rate trends as early warning signals in intensive care monitoring.",
    },
]

_QUERY = "hypotension tachycardia shock monitoring"


def _make_retriever(enable_pubmed: bool = True) -> MedicalRetriever:
    """Return a MedicalRetriever with PubMed controlled by the flag."""
    return MedicalRetriever(enable_pubmed=enable_pubmed)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestPubMedRAG(unittest.TestCase):
    def setUp(self) -> None:
        clear_cache()

    def tearDown(self) -> None:
        clear_cache()

    # -----------------------------------------------------------------------
    # (a) Valid abstracts → retrieval_source == "pubmed", passages from mock
    # -----------------------------------------------------------------------
    def test_valid_abstracts_uses_pubmed(self) -> None:
        """When fetch_pubmed_abstracts returns results, source should be 'pubmed'."""
        with patch(
            "clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts",
            return_value=_SAMPLE_ABSTRACTS,
        ):
            retriever = _make_retriever(enable_pubmed=True)
            result: RetrievalResult = retriever.retrieve(_QUERY)

        self.assertEqual(result.retrieval_source, "pubmed")
        self.assertEqual(result.retrieval_status, "SUCCESS")
        self.assertTrue(result.has_evidence)
        # Passages come from the mock, not the curated corpus
        pmids = {p.metadata.get("pmid") for p in result.passages}
        self.assertIn("12345678", pmids)
        self.assertIn("87654321", pmids)

    # -----------------------------------------------------------------------
    # (b) Timeout / network error → fallback to curated_fallback, no exception
    # -----------------------------------------------------------------------
    def test_timeout_falls_back_to_curated(self) -> None:
        """When fetch_pubmed_abstracts raises (simulated timeout), fall back silently."""
        with patch(
            "clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts",
            side_effect=TimeoutError("simulated timeout"),
        ):
            retriever = _make_retriever(enable_pubmed=True)
            # Should NOT raise — must fall back gracefully
            result: RetrievalResult = retriever.retrieve(_QUERY)

        self.assertEqual(result.retrieval_source, "curated_fallback")
        # No exception propagated — this is the critical assertion

    # -----------------------------------------------------------------------
    # (c) Returns None → fallback to curated_fallback
    # -----------------------------------------------------------------------
    def test_none_return_falls_back_to_curated(self) -> None:
        """When fetch_pubmed_abstracts returns None, fall back to curated corpus."""
        with patch(
            "clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts",
            return_value=None,
        ):
            retriever = _make_retriever(enable_pubmed=True)
            result: RetrievalResult = retriever.retrieve(_QUERY)

        self.assertEqual(result.retrieval_source, "curated_fallback")

    # -----------------------------------------------------------------------
    # (d) Cache prevents a second network call within TTL
    # -----------------------------------------------------------------------
    def test_cache_prevents_second_network_call(self) -> None:
        """Identical queries within TTL must use the cache, not call the network again."""
        call_count = 0
        original_fn = pubmed_client._http_get

        def counting_http_get(url, params, timeout):
            nonlocal call_count
            call_count += 1
            raise OSError("network unavailable")  # force None result

        with patch.object(pubmed_client, "_http_get", side_effect=counting_http_get):
            clear_cache()
            # First call — hits "network"
            result1 = fetch_pubmed_abstracts(_QUERY, max_results=3)
            # Second call — should use cache, NOT hit network again
            result2 = fetch_pubmed_abstracts(_QUERY, max_results=3)

        self.assertEqual(call_count, 1, "Network should only be called once within TTL")
        self.assertIsNone(result1)
        self.assertIsNone(result2)

    # -----------------------------------------------------------------------
    # (e) enable_pubmed=False → always uses curated fallback, never calls network
    # -----------------------------------------------------------------------
    def test_pubmed_disabled_uses_curated(self) -> None:
        """When enable_pubmed=False, retriever must never call fetch_pubmed_abstracts."""
        with patch(
            "clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts",
            side_effect=AssertionError("should not be called"),
        ):
            retriever = _make_retriever(enable_pubmed=False)
            result: RetrievalResult = retriever.retrieve(_QUERY)

        self.assertEqual(result.retrieval_source, "curated_fallback")

    # -----------------------------------------------------------------------
    # (f) retrieval_source propagated when PubMed succeeds end-to-end
    # -----------------------------------------------------------------------
    def test_retrieval_source_field_present_on_result(self) -> None:
        """RetrievalResult must always carry retrieval_source regardless of path."""
        with patch(
            "clinical_reasoning_agent.rag.pubmed_client.fetch_pubmed_abstracts",
            return_value=_SAMPLE_ABSTRACTS,
        ):
            retriever = _make_retriever(enable_pubmed=True)
            result = retriever.retrieve(_QUERY)

        self.assertIn(result.retrieval_source, ("pubmed", "curated_fallback"))

    # -----------------------------------------------------------------------
    # (g) Cache TTL expiry triggers a new network call
    # -----------------------------------------------------------------------
    def test_cache_expires_after_ttl(self) -> None:
        """After TTL, an expired cache entry should trigger a new network call."""
        # Minimal XML stubs returned by _http_get
        _ESEARCH_XML = b"""<?xml version="1.0"?>
<eSearchResult><IdList><Id>11111111</Id></IdList></eSearchResult>"""
        _EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation><PMID>11111111</PMID>
      <Article><ArticleTitle>Test Title</ArticleTitle>
        <Abstract><AbstractText>Test abstract text.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""

        call_count = 0

        def fake_http_get(url, params, timeout):
            nonlocal call_count
            call_count += 1
            if "esearch" in url:
                return _ESEARCH_XML
            return _EFETCH_XML

        original_ttl = pubmed_client._CACHE_TTL_SECONDS
        try:
            pubmed_client._CACHE_TTL_SECONDS = 0.05  # 50 ms — expire quickly
            clear_cache()

            with patch.object(pubmed_client, "_http_get", side_effect=fake_http_get):
                # First call — hits the network (2 HTTP calls: esearch + efetch)
                r1 = fetch_pubmed_abstracts(_QUERY, max_results=1)
                calls_after_first = call_count

                time.sleep(0.1)  # let cache expire (> 50 ms)

                # Second call — cache expired, should hit network again
                r2 = fetch_pubmed_abstracts(_QUERY, max_results=1)

            self.assertEqual(calls_after_first, 2, "First call should make esearch + efetch requests")
            self.assertEqual(call_count, 4, "Second call after TTL expiry should make another 2 requests")
            self.assertIsNotNone(r1)
            self.assertIsNotNone(r2)
        finally:
            pubmed_client._CACHE_TTL_SECONDS = original_ttl


if __name__ == "__main__":
    unittest.main()
