#!/usr/bin/env python3
"""Live PubMed connectivity verification script.

Calls PubMed E-utilities via CareMatrix pubmed_client without mocking.
Run directly with python3 to verify live connectivity to NCBI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so imports work when executed directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clinical_reasoning_agent.rag.pubmed_client import fetch_pubmed_abstracts  # noqa: E402

QUERY = "hypotension tachycardia shock monitoring"


def main() -> int:
    print(f"Executing live PubMed query: {QUERY!r}")
    print("Calling NCBI E-utilities (esearch + efetch)...")

    results = fetch_pubmed_abstracts(QUERY, max_results=3, timeout=5.0)

    if results is None:
        print("\n[RESULT] No results / failed to retrieve abstracts from PubMed.")
        print("PubMed may be unreachable, timed out, returned an error, or yielded zero results.")
        print("CareMatrix will safely fall back to the curated offline guidelines.")
        return 1

    print(f"\n[RESULT] Successfully retrieved {len(results)} abstract(s) from PubMed:")
    for idx, article in enumerate(results, start=1):
        pmid = article.get("pmid", "UNKNOWN")
        title = article.get("title", "Untitled")
        abstract = article.get("abstract", "")
        preview = (abstract[:160] + "...") if len(abstract) > 160 else abstract
        print(f"\n--- [{idx}] PMID: {pmid} ---")
        print(f"Title:    {title}")
        print(f"Abstract: {preview}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
