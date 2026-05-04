import os
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime, timedelta
import requests


# =============================================================================
# Multi-platform paper utilities (arxiv, biorxiv, nature, chemrxiv, doi.org)
# =============================================================================

# Regex patterns per domain for extracting paper URLs from text.
# Bare (no scheme) patterns are included for arxiv since documents sometimes
# contain links without the https:// prefix.
_PLATFORM_PATTERNS = {
    'arxiv.org': [
        r'https?://(?:www\.)?arxiv\.org/abs/[\w\.\-]+',
        r'https?://(?:www\.)?arxiv\.org/pdf/[\w\.\-]+',
        r'(?<!//)arxiv\.org/abs/[\w\.\-]+',
        r'(?<!//)arxiv\.org/pdf/[\w\.\-]+',
    ],
    'biorxiv.org': [
        r'https?://(?:www\.)?biorxiv\.org/content/[\w\./\-]+',
    ],
    'nature.com': [
        r'https?://(?:www\.)?nature\.com/articles/[\w\.\-]+',
    ],
    'chemrxiv.org': [
        r'https?://(?:www\.)?chemrxiv\.org/engage/[\w\./\-]+',
        r'https?://(?:www\.)?chemrxiv\.org/[\w\./\-]+',
    ],
    'doi.org': [
        r'https?://doi\.org/[\w\./\-]+',
    ],
}


def extract_paper_links_from_text(text: str, domains: Optional[List[str]] = None) -> List[str]:
    """
    Extract paper links from document text across multiple platforms.

    Args:
        text: Document text content.
        domains: List of domain strings to search for (e.g. ['arxiv.org', 'biorxiv.org']).
                 If None, searches all supported platforms.

    Returns:
        Deduplicated list of paper URLs found in the text.
    """
    if domains is None:
        active_domains = list(_PLATFORM_PATTERNS.keys())
    else:
        # Always include doi.org when any domain is requested, since papers
        # on biorxiv/nature/chemrxiv may be linked via doi.org
        active_domains = list(domains)
        if 'doi.org' not in active_domains:
            active_domains.append('doi.org')

    links = []
    for domain in active_domains:
        for pattern in _PLATFORM_PATTERNS.get(domain, []):
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize bare URLs (no scheme) to https://
                if not match.startswith('http'):
                    match = 'https://' + match
                links.append(match)

    return list(set(links))


def normalize_arxiv_url(url):
    """
    Normalize arxiv URLs to a standard format for comparison.

    Args:
        url (str): Raw URL from browsing history or document

    Returns:
        str: Normalized arxiv paper ID (e.g., "2301.00001"), or None
    """
    patterns = [
        r'arxiv\.org/abs/([\w\.\-]+)',
        r'arxiv\.org/pdf/([\w\.\-]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            paper_id = match.group(1)
            if paper_id.endswith('.pdf'):
                paper_id = paper_id[:-4]
            # Strip arxiv version suffix (e.g. "v1", "v12") so that
            # https://arxiv.org/abs/2305.14314v2 and
            # https://arxiv.org/abs/2305.14314 normalize to the same id.
            # Browser-recorded histories often capture versioned URLs after
            # arxiv's canonical redirect.
            paper_id = re.sub(r'v\d+$', '', paper_id)
            return paper_id
    return None


def extract_paper_id(url: str) -> Optional[Tuple[str, str]]:
    """
    Extract a typed paper identifier from a URL.

    Supports arxiv, biorxiv, nature, chemrxiv, and doi.org URLs.

    Args:
        url: A paper URL.

    Returns:
        A tuple (id_type, id_value) such as ("ARXIV", "2301.00001") or
        ("DOI", "10.1101/..."), or None if the URL is not recognized.
    """
    # arxiv
    arxiv_id = normalize_arxiv_url(url)
    if arxiv_id:
        return ("ARXIV", arxiv_id)

    # biorxiv DOI  (e.g. biorxiv.org/content/10.1101/2023.01.01.123456v1)
    biorxiv_match = re.search(r'biorxiv\.org/content/(10\.\d{4,}/[\w\.\-]+)', url)
    if biorxiv_match:
        doi = re.sub(r'v\d+$', '', biorxiv_match.group(1))  # strip version suffix
        return ("DOI", doi)

    # nature  (e.g. nature.com/articles/s41586-023-06415-8)
    nature_match = re.search(r'nature\.com/articles/([\w\.\-]+)', url)
    if nature_match:
        return ("DOI", f"10.1038/{nature_match.group(1)}")

    # doi.org direct  (e.g. doi.org/10.1234/something)
    doi_match = re.search(r'doi\.org/(10\.\d{4,}/[\w\.\-/]+)', url)
    if doi_match:
        return ("DOI", doi_match.group(1))

    # chemrxiv article-details link (no DOI available)
    chemrxiv_match = re.search(r'chemrxiv\.org/engage/chemrxiv/article-details/([\w\-]+)', url)
    if chemrxiv_match:
        return ("CHEMRXIV_ID", chemrxiv_match.group(1))

    return None


def paper_id_to_ss_identifier(paper_id: Tuple[str, str]) -> Optional[str]:
    """
    Convert a paper ID tuple to a Semantic Scholar batch-API identifier.

    Args:
        paper_id: Tuple from extract_paper_id, e.g. ("ARXIV", "2301.00001").

    Returns:
        String like "ARXIV:2301.00001" or "DOI:10.1101/...", or None if the
        id type cannot be looked up (e.g. CHEMRXIV_ID).
    """
    id_type, id_value = paper_id
    if id_type in ("ARXIV", "DOI"):
        return f"{id_type}:{id_value}"
    return None


def search_s2_by_title(title: str, fields: str = 'citationCount,title,publicationDate,abstract,externalIds') -> Optional[dict]:
    """
    Search Semantic Scholar for a paper by title. Used as a fallback when
    paper IDs (e.g. ChemRxiv) can't be directly looked up via batch API.

    Args:
        title: The paper title to search for.
        fields: S2 fields to return.

    Returns:
        Paper dict if a good match is found, None otherwise.
    """
    from rapidfuzz import fuzz

    response = _s2_request_with_backoff(
        'get',
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={'query': title, 'fields': fields, 'limit': 3},
    )
    if response is None or response.status_code != 200:
        return None

    data = response.json()
    results = data.get('data', [])
    for paper in results:
        s2_title = paper.get('title', '')
        if fuzz.ratio(title.lower(), s2_title.lower()) >= 85:
            return paper
    return None


def _s2_request_with_backoff(method, url, max_retries=3, **kwargs):
    """
    Make a Semantic Scholar API request with rate limiting (1 req/s) and backoff.

    Args:
        method: 'get' or 'post'
        url: The API URL
        max_retries: Number of retries on 429/5xx errors
        **kwargs: Passed to requests.get/post

    Returns:
        requests.Response object, or None on total failure.
    """
    import time as _time

    s2_headers = kwargs.pop('headers', {})
    s2_api_key = os.environ.get("S2_API_KEY")
    if s2_api_key:
        s2_headers["x-api-key"] = s2_api_key

    # Rate limit: wait 1s between requests
    if not hasattr(_s2_request_with_backoff, '_last_call'):
        _s2_request_with_backoff._last_call = 0
    elapsed = _time.time() - _s2_request_with_backoff._last_call
    if elapsed < 1.0:
        _time.sleep(1.0 - elapsed)

    for attempt in range(max_retries + 1):
        try:
            if method == 'post':
                response = requests.post(url, headers=s2_headers, **kwargs)
            else:
                response = requests.get(url, headers=s2_headers, **kwargs)

            _s2_request_with_backoff._last_call = _time.time()

            if response.status_code == 200:
                return response
            elif response.status_code in (429, 500, 502, 503) and attempt < max_retries:
                wait_time = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"  S2 API {response.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                _time.sleep(wait_time)
            else:
                return response
        except Exception as e:
            if attempt < max_retries:
                wait_time = 5 * (2 ** attempt)
                print(f"  S2 request error: {e}, retrying in {wait_time}s...")
                _time.sleep(wait_time)
            else:
                print(f"  S2 request failed after {max_retries + 1} attempts: {e}")
                return None

    return None


def fetch_papers_from_semantic_scholar(
    paper_ids: List[Tuple[str, str]],
    fields: str = 'citationCount,title,publicationDate,abstract,externalIds',
    fallback_titles: Optional[List[Optional[str]]] = None,
) -> List[Optional[dict]]:
    """
    Batch-fetch paper metadata from Semantic Scholar.

    Args:
        paper_ids: List of (id_type, id_value) tuples from extract_paper_id.
        fields: Comma-separated Semantic Scholar fields to request.
        fallback_titles: Optional list (same length as paper_ids) of paper titles.
            Used as fallback for IDs that can't be batch-fetched (e.g. ChemRxiv).

    Returns:
        List of paper dicts (or None entries for papers not found).
        Returns an empty list on API errors.
    """
    # Separate papers into batch-fetchable and those needing title search
    batch_ids = []
    batch_indices = []
    title_search_indices = []

    for i, pid in enumerate(paper_ids):
        ss_id = paper_id_to_ss_identifier(pid)
        if ss_id:
            batch_ids.append(ss_id)
            batch_indices.append(i)
        else:
            title_search_indices.append(i)

    results = [None] * len(paper_ids)

    # Batch fetch supported IDs
    if batch_ids:
        response = _s2_request_with_backoff(
            'post',
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={'fields': fields},
            json={"ids": batch_ids},
        )
        if response is not None:
            data = response.json()
            if isinstance(data, list):
                for idx, paper in zip(batch_indices, data):
                    results[idx] = paper
            else:
                error_msg = data.get('message', data) if isinstance(data, dict) else data
                print(f"Semantic Scholar API error: {error_msg}")

    # Title search fallback for unsupported IDs (e.g. ChemRxiv)
    if title_search_indices and fallback_titles:
        for idx in title_search_indices:
            if idx < len(fallback_titles) and fallback_titles[idx]:
                paper = search_s2_by_title(fallback_titles[idx], fields=fields)
                if paper:
                    print(f"  Found '{fallback_titles[idx]}' via title search on S2")
                    results[idx] = paper

    return results


def s2_batch_fetch_by_arxiv_ids(
    arxiv_ids: List[str],
    fields: str = 'citationCount,title,publicationDate,externalIds',
) -> Optional[list]:
    """
    Batch-fetch paper metadata from Semantic Scholar using arXiv IDs.
    Includes rate limiting (1 req/s) and exponential backoff on failures.

    Args:
        arxiv_ids: List of arXiv paper IDs (e.g., ['2305.14314', '2309.12307'])
        fields: Comma-separated Semantic Scholar fields to request.

    Returns:
        List of paper dicts on success, or None on API errors.
    """
    if not arxiv_ids:
        return None

    response = _s2_request_with_backoff(
        'post',
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params={'fields': fields},
        json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]},
    )
    if response is None:
        return None
    result = response.json()
    if isinstance(result, list):
        return result

    error_msg = result.get('message', result) if isinstance(result, dict) else result
    print(f"Semantic Scholar API error: {error_msg}")
    return None


def match_paper_links_with_browsing_history(
    gold_text: str,
    browsing_history: Optional[List[str]],
    domains: Optional[List[str]] = None,
    min_papers: int = 5,
) -> Tuple[bool, List, List, int]:
    """
    Check if paper links in the document match those in browsing history.

    Args:
        gold_text: Document text content.
        browsing_history: List of URLs visited during task (can be None).
        domains: Platform domains to search for (passed to extract_paper_links_from_text).
        min_papers: Minimum number of papers expected.

    Returns:
        (links_match, doc_ids, visited_ids, matched_count)
    """
    if browsing_history is None:
        browsing_history = []

    doc_links = extract_paper_links_from_text(gold_text, domains)
    doc_ids = set()
    for link in doc_links:
        pid = extract_paper_id(link)
        if pid:
            doc_ids.add(pid)

    visited_ids = set()
    for url in browsing_history:
        pid = extract_paper_id(url)
        if pid:
            visited_ids.add(pid)

    matched_count = len(doc_ids.intersection(visited_ids))
    links_match = matched_count >= min(min_papers, len(doc_ids)) and len(doc_ids) >= min_papers

    return links_match, list(doc_ids), list(visited_ids), matched_count


# =============================================================================
# General utilities
# =============================================================================

def is_within_x_years(date_string, years, reference_date=None):
    """
    Check if date_string is no more than x years before reference_date.

    Args:
        date_string: Date string in format "YYYY-MM-DD"
        reference_date: datetime object or date string to compare against
        years: Maximum number of years before reference_date

    Returns:
        True if date_string is within x years before reference_date
    """
    date = datetime.strptime(date_string, "%Y-%m-%d")

    if reference_date is None:
        reference_date = datetime.now()
    if isinstance(reference_date, str):
        reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    cutoff_date = reference_date - timedelta(days=years * 365.25)
    return date >= cutoff_date


def get_paper_info_ss(arxiv_id):
    """Query Semantic Scholar for a single arxiv paper."""
    arxiv_id = arxiv_id.replace('arXiv:', '')
    url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
    response = _s2_request_with_backoff('get', url, params={'fields': 'citationCount,title,publicationDate'})
    if response and response.status_code == 200:
        return response.json()
    return None


# =============================================================================
# Legacy arxiv-only wrappers (kept for backward compatibility with instances 1-3)
# =============================================================================

def extract_arxiv_links_from_text(text):
    """
    Extract arxiv.org links from document text.

    Thin wrapper around extract_paper_links_from_text for backward compatibility.

    Args:
        text (str): Document text content

    Returns:
        list: List of arxiv URLs found in the document
    """
    return extract_paper_links_from_text(text, domains=['arxiv.org'])


def match_document_links_with_browsing_history(gold_text, browsing_history):
    """
    Check if arxiv links in the document match those in browsing history.

    Thin wrapper around match_paper_links_with_browsing_history that preserves
    the original return format (paper ID strings instead of tuples).

    Args:
        gold_text (str): Document text content
        browsing_history (list): List of URLs visited during task (can be None)

    Returns:
        tuple: (links_match: bool, doc_paper_ids: list, visited_paper_ids: list, matched_count: int)
    """
    if browsing_history is None:
        browsing_history = []

    # Use the generalized function internally
    links_match, doc_id_tuples, visited_id_tuples, matched_count = (
        match_paper_links_with_browsing_history(gold_text, browsing_history, domains=['arxiv.org'])
    )

    # Convert tuples back to plain arxiv ID strings for backward compat
    doc_paper_ids = [id_val for id_type, id_val in doc_id_tuples if id_type == "ARXIV"]
    visited_paper_ids = [id_val for id_type, id_val in visited_id_tuples if id_type == "ARXIV"]

    return links_match, doc_paper_ids, visited_paper_ids, matched_count
