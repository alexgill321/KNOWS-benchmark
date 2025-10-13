import re
from typing import List
from urllib.parse import urlparse

def extract_arxiv_links_from_text(text):
    """
    Extract arxiv.org links from document text.

    Args:
        text (str): Document text content

    Returns:
        list: List of arxiv URLs found in the document
    """
    # Pattern to match arxiv URLs in various formats
    arxiv_patterns = [
        r'https?://arxiv\.org/abs/[\w\.-]+',
        r'https?://arxiv\.org/pdf/[\w\.-]+',
        r'arxiv\.org/abs/[\w\.-]+',
        r'arxiv\.org/pdf/[\w\.-]+'
    ]

    arxiv_links = []
    for pattern in arxiv_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize URL format
            if not match.startswith('http'):
                match = 'https://' + match
            arxiv_links.append(match)

    return list(set(arxiv_links))  # Remove duplicates

def normalize_arxiv_url(url):
    """
    Normalize arxiv URLs to a standard format for comparison.

    Args:
        url (str): Raw URL from browsing history or document

    Returns:
        str: Normalized arxiv paper ID (e.g., "2301.00001")
    """
    # Extract paper ID from various arxiv URL formats
    patterns = [
        r'arxiv\.org/abs/([\w\.-]+)',
        r'arxiv\.org/pdf/([\w\.-]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            paper_id = match.group(1)
            # Remove .pdf extension if present
            if paper_id.endswith('.pdf'):
                paper_id = paper_id[:-4]
            return paper_id

    return None

def match_document_links_with_browsing_history(gold_text, browsing_history):
    """
    Check if arxiv links in the document match those in browsing history.

    Args:
        gold_text (str): Document text content
        browsing_history (list): List of URLs visited during task

    Returns:
        tuple: (links_match: bool, doc_links: list, visited_links: list, matched_count: int)
    """
    # Extract arxiv links from document
    doc_arxiv_links = extract_arxiv_links_from_text(gold_text)

    # Extract arxiv links from browsing history
    browsing_arxiv_links = [url for url in browsing_history if 'arxiv.org' in url]

    # Normalize all links to paper IDs for comparison
    doc_paper_ids = set()
    for link in doc_arxiv_links:
        paper_id = normalize_arxiv_url(link)
        if paper_id:
            doc_paper_ids.add(paper_id)

    visited_paper_ids = set()
    for link in browsing_arxiv_links:
        paper_id = normalize_arxiv_url(link)
        if paper_id:
            visited_paper_ids.add(paper_id)

    # Count matches
    matched_papers = doc_paper_ids.intersection(visited_paper_ids)
    matched_count = len(matched_papers)

    # Check if most document links have corresponding visits
    links_match = matched_count >= min(5, len(doc_paper_ids)) and len(doc_paper_ids) >= 5

    return links_match, list(doc_paper_ids), list(visited_paper_ids), matched_count