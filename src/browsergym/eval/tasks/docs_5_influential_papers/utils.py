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


def fetch_papers_from_semantic_scholar(
    paper_ids: List[Tuple[str, str]],
    fields: str = 'citationCount,title,publicationDate,abstract,externalIds',
) -> List[Optional[dict]]:
    """
    Batch-fetch paper metadata from Semantic Scholar.

    Args:
        paper_ids: List of (id_type, id_value) tuples from extract_paper_id.
        fields: Comma-separated Semantic Scholar fields to request.

    Returns:
        List of paper dicts (or None entries for papers not found).
        Returns an empty list on API errors.
    """
    ss_ids = []
    for pid in paper_ids:
        ss_id = paper_id_to_ss_identifier(pid)
        if ss_id:
            ss_ids.append(ss_id)

    if not ss_ids:
        return []

    try:
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={'fields': fields},
            json={"ids": ss_ids},
        )
        result = response.json()
        if isinstance(result, list):
            return result
        error_msg = result.get('message', result) if isinstance(result, dict) else result
        print(f"Semantic Scholar API error: {error_msg}")
        return []
    except Exception as e:
        print(f"Error fetching from Semantic Scholar: {e}")
        return []


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
    params = {'fields': 'citationCount,title,publicationDate'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
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
