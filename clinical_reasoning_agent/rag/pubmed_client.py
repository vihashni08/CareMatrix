"""PubMed E-utilities client for live abstract retrieval.

Uses NCBI esearch + efetch to fetch recent abstracts for a clinical query.
On any failure (network error, timeout, empty/malformed XML, HTTP error,
rate-limit) returns None so the caller can fall back to the curated corpus.

Environment variables (both optional):
    NCBI_API_KEY       – NCBI API key (raises rate limit to 10 req/s; omit at 3 req/s).
    NCBI_CONTACT_EMAIL – Contact e-mail sent as the ``email`` parameter per NCBI policy.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET
from urllib import parse, request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# NCBI E-utilities base URLs
# ---------------------------------------------------------------------------
_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_TOOL_NAME = "CareMatrix"

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS: float = 3600.0  # 1 hour

_cache_lock = threading.Lock()
# Maps cache_key -> (insert_monotonic_time, result_or_None)
_cache: dict[str, tuple[float, list[dict[str, str]] | None]] = {}

_MISSING = object()  # sentinel for "not in cache"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_get(key: str) -> object:
    """Return cached value, or ``_MISSING`` sentinel on miss/expiry."""
    with _cache_lock:
        entry = _cache.get(key)
    if entry is None:
        return _MISSING
    ts, value = entry
    if time.monotonic() - ts > _CACHE_TTL_SECONDS:
        with _cache_lock:
            _cache.pop(key, None)
        return _MISSING
    return value


def _cache_set(key: str, value: list[dict[str, str]] | None) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), value)


def clear_cache() -> None:
    """Clear the in-memory abstract cache (useful in tests)."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------
def _load_env_if_needed() -> None:
    """Load NCBI settings from .env file if not already present in environment."""
    try:
        root = Path(__file__).resolve().parent.parent.parent
        env_path = root / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception:
        pass


# ---------------------------------------------------------------------------
# NCBI base parameters
# ---------------------------------------------------------------------------
def _ncbi_params() -> dict[str, str]:
    """Return base parameters required by every NCBI E-utilities request."""
    _load_env_if_needed()
    params: dict[str, str] = {"tool": _TOOL_NAME}
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    email = os.environ.get("NCBI_CONTACT_EMAIL", "carematrix@example.com").strip()
    params["email"] = email
    return params


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------
def _http_get(url: str, params: dict[str, str], timeout: float) -> bytes:
    """Perform a GET request and return the raw response body."""
    full_url = url + "?" + parse.urlencode(params)
    req = request.Request(full_url, headers={"User-Agent": f"{_TOOL_NAME}/1.0"})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------
def _parse_pmids(xml_bytes: bytes) -> list[str]:
    """Extract PMIDs from an esearch XML response."""
    try:
        root = ET.fromstring(xml_bytes)
        return [el.text for el in root.findall(".//Id") if el.text]
    except ET.ParseError:
        return []


def _parse_abstracts(xml_bytes: bytes) -> list[dict[str, str]]:
    """Extract title, abstract text, PMID, authors, journal, year, and DOI from an efetch PubmedArticleSet XML."""
    results: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return results

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None and pmid_el.text else "UNKNOWN"

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Untitled"

        # Structured abstracts contain multiple <AbstractText Label="..."> elements
        abstract_parts: list[str] = []
        for ab_el in article.findall(".//AbstractText"):
            label = ab_el.get("Label", "")
            text = "".join(ab_el.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)

        abstract_text = " ".join(abstract_parts).strip()
        if not abstract_text:
            continue

        # Extract authors
        authors_list: list[str] = []
        for auth in article.findall(".//Author"):
            last_name = auth.find("LastName")
            fore_name = auth.find("ForeName") or auth.find("Initials")
            if last_name is not None and last_name.text:
                l_text = last_name.text.strip()
                f_text = f" {fore_name.text.strip()}" if fore_name is not None and fore_name.text else ""
                authors_list.append(f"{l_text}{f_text}")
        if len(authors_list) > 3:
            authors_str = f"{', '.join(authors_list[:3])} et al."
        elif authors_list:
            authors_str = ", ".join(authors_list)
        else:
            authors_str = ""

        # Extract journal
        journal_el = article.find(".//Journal/Title")
        if journal_el is None or not (journal_el.text or "").strip():
            journal_el = article.find(".//Journal/ISOAbbreviation")
        journal_str = "".join(journal_el.itertext()).strip() if journal_el is not None else ""

        # Extract publication year
        year_el = article.find(".//JournalIssue/PubDate/Year")
        if year_el is None or not (year_el.text or "").strip():
            year_el = article.find(".//PubDate/Year")
        if year_el is None or not (year_el.text or "").strip():
            year_el = article.find(".//PubDate/MedlineDate")
        year_str = "".join(year_el.itertext()).strip()[:4] if year_el is not None else ""

        # Extract DOI
        doi_str = ""
        for el in article.findall(".//ELocationID"):
            if el.get("EIdType") == "doi" and el.text:
                doi_str = el.text.strip()
                break
        if not doi_str:
            for el in article.findall(".//ArticleId"):
                if el.get("IdType") == "doi" and el.text:
                    doi_str = el.text.strip()
                    break

        results.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "authors": authors_str,
            "journal": journal_str,
            "year": year_str,
            "doi": doi_str,
        })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_pubmed_abstracts(
    query: str,
    max_results: int = 3,
    timeout: float = 3.0,
) -> list[dict[str, str]] | None:
    """Fetch PubMed abstracts matching *query* via NCBI E-utilities.

    Calls ``esearch.fcgi`` to retrieve PMIDs, then ``efetch.fcgi``
    (``rettype=abstract``, ``retmode=xml``) to fetch the abstracts.
    Results are cached in memory with a 1-hour TTL.

    Returns:
        A list of dicts with keys ``"pmid"``, ``"title"``, ``"abstract"``,
        or ``None`` on any failure (network error, timeout, empty or
        malformed XML, HTTP error, rate-limit).  Never raises.

    Args:
        query:       PubMed search query string.
        max_results: Maximum number of abstracts to retrieve (default 3).
        timeout:     HTTP timeout in seconds applied *per request* (default 3.0).
    """
    if not query or not query.strip():
        return None

    cache_key = f"{query.strip()}|{max_results}"
    cached = _cache_get(cache_key)
    if cached is not _MISSING:
        return cached  # type: ignore[return-value]

    base = _ncbi_params()

    try:
        # Step 1 - esearch: retrieve PMIDs
        esearch_params: dict[str, str] = {
            **base,
            "db": "pubmed",
            "term": query.strip(),
            "retmax": str(max_results),
            "retmode": "xml",
            "sort": "relevance",
        }
        esearch_bytes = _http_get(_ESEARCH_URL, esearch_params, timeout)
        pmids = _parse_pmids(esearch_bytes)
        if not pmids:
            _cache_set(cache_key, None)
            return None

        # Step 2 - efetch: retrieve abstracts for the PMIDs
        efetch_params: dict[str, str] = {
            **base,
            "db": "pubmed",
            "id": ",".join(pmids[:max_results]),
            "rettype": "abstract",
            "retmode": "xml",
        }
        efetch_bytes = _http_get(_EFETCH_URL, efetch_params, timeout)
        abstracts = _parse_abstracts(efetch_bytes)
        if not abstracts:
            _cache_set(cache_key, None)
            return None

        _cache_set(cache_key, abstracts)
        return abstracts

    except (URLError, OSError, TimeoutError, Exception):
        # Broad catch-all: never let a network failure propagate into the
        # agent pipeline.  Cache as None so we don't retry within the TTL.
        _cache_set(cache_key, None)
        return None


# ---------------------------------------------------------------------------
# Build PubMed query from a DataAnalysisEvent
# ---------------------------------------------------------------------------
def build_pubmed_query(event: object) -> str:  # event: DataAnalysisEvent
    """Construct a PubMed-compatible search query from a DataAnalysisEvent.

    Delegates keyword-mapping to ``build_focused_medical_query`` from
    ``retriever`` to avoid duplicating the vital-trend to clinical-term logic.
    The returned string is suitable for passing directly to
    ``fetch_pubmed_abstracts``.
    """
    # Import here to avoid circular imports at module load time.
    from clinical_reasoning_agent.rag.retriever import build_focused_medical_query  # noqa: PLC0415

    return build_focused_medical_query(event)


__all__ = [
    "build_pubmed_query",
    "clear_cache",
    "fetch_pubmed_abstracts",
]
