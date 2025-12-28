"""Web utilities for fetching and downloading content from URLs."""

import os
import re
import requests
from typing import Dict, List, Optional
from urllib.parse import urlparse

# Domains known to block programmatic image downloads (anti-hotlinking, bot protection, etc.)
UNVERIFIABLE_DOMAINS = [
    "ndtvimg.com",
    "researchgate.net",
    "wikimedia.org",
    "wikipedia.org",
    "instagram.com",
    "fbcdn.net",  # Facebook CDN
    "pinimg.com",  # Pinterest
    "twimg.com",  # Twitter
]


def is_unverifiable_url(url: str) -> bool:
    """Check if URL is from a domain known to block programmatic downloads.

    Args:
        url: The URL to check.

    Returns:
        True if the domain is known to block downloads, False otherwise.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Check if any unverifiable domain is in the URL's domain
        return any(blocked in domain for blocked in UNVERIFIABLE_DOMAINS)
    except Exception:
        return False


def download_image_from_url(url: str, temp_dir: str, timeout: int = 15) -> str:
    """Download image from URL to temp directory.

    Args:
        url: The URL to download the image from.
        temp_dir: Directory to save the downloaded image.
        timeout: Request timeout in seconds.

    Returns:
        Path to downloaded image, or None if download failed.
    """
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if content_type.startswith('image/'):
                # Determine extension from content type
                ext = content_type.split('/')[-1].split(';')[0]
                if ext not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                    ext = 'png'
                temp_path = os.path.join(temp_dir, f"url_image_{hash(url)}.{ext}")
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                return temp_path
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
    return None


def extract_id_from_url(url: str, patterns: List[str]) -> Optional[str]:
    """Extract an ID from a URL using regex patterns.

    Useful for extracting identifiers from URLs like arXiv IDs, YouTube video IDs,
    or any other URL-embedded identifier.

    Args:
        url: URL to parse.
        patterns: List of regex patterns, each with a capture group for the ID.
            Patterns are tried in order; first match wins.

    Returns:
        Extracted ID string, or None if no pattern matched.

    Examples:
        >>> arxiv_patterns = [
        ...     r'arxiv\\.org/(?:abs|pdf)/(\\d{4}\\.\\d{4,5})(?:v\\d+)?',
        ...     r'(\\d{4}\\.\\d{4,5})(?:v\\d+)?\\.pdf',
        ... ]
        >>> extract_id_from_url('https://arxiv.org/abs/2301.12345', arxiv_patterns)
        '2301.12345'
    """
    if not url or not patterns:
        return None

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def is_url_from_domain(url: str, domain: str, case_sensitive: bool = False) -> bool:
    """Check if URL is from a specific domain.

    Args:
        url: The URL to check.
        domain: The domain to match (e.g., 'usda.gov', 'fdc.nal.usda.gov').
        case_sensitive: Whether to perform case-sensitive matching.

    Returns:
        True if the URL contains the specified domain, False otherwise.
    """
    if not url or not domain:
        return False

    url_check = url if case_sensitive else url.lower()
    domain_check = domain if case_sensitive else domain.lower()

    return domain_check in url_check


def fetch_api_with_retry(
    url: str,
    timeout: int = 10,
    max_retries: int = 3,
    headers: Optional[Dict[str, str]] = None
) -> Optional[Dict]:
    """
    Fetch JSON data from an API with exponential backoff retry logic.

    Handles rate limiting (429 status) with exponential backoff.

    Args:
        url: API endpoint URL.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for rate-limited requests.
        headers: Optional HTTP headers.

    Returns:
        JSON response as dict, or None if fetch failed.
    """
    import time

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited - exponential backoff
                wait_time = 2 ** attempt
                print(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            else:
                return None

        except Exception as e:
            print(f"Error fetching API data from {url}: {e}")
            return None

    print(f"Failed to fetch data after {max_retries} retries")
    return None


def fetch_page_title(url: str, timeout: int = 10, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Fetch page title from any webpage via HTML parsing.

    Attempts to extract the page title from the <title> tag first,
    then falls back to the first <h1> tag if no title is found.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        headers: Optional HTTP headers to send with request.

    Returns:
        Page title or h1 text, or None if fetch failed.
    """
    try:
        from bs4 import BeautifulSoup

        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        request_headers = headers or default_headers

        response = requests.get(url, timeout=timeout, headers=request_headers)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the title element
        title = soup.find('title')
        if title:
            title_text = title.get_text().strip()
            # Clean up the title - often contains site name after separator
            if '|' in title_text:
                title_text = title_text.split('|')[0].strip()
            if title_text:
                return title_text

        # Try h1 as fallback
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()

        return None

    except Exception as e:
        print(f"Error fetching page {url}: {e}")
        return None
