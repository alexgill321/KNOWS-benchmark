"""Web utilities for fetching and downloading content from URLs."""

import os
import re
import requests
import html2text
from typing import Dict, List, Optional, Tuple
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


def validate_url_format(url: str) -> Tuple[bool, str]:
    """
    Validate URL format without making HTTP requests.

    Checks that the URL has a valid scheme (http/https), a valid domain,
    and proper structure. Useful when websites block programmatic access.

    Args:
        url: URL to validate.

    Returns:
        tuple: (is_valid: bool, details: str)
            - is_valid: True if URL has valid format
            - details: Description of result
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False, f"Invalid URL scheme: {parsed.scheme}"
        if not parsed.netloc:
            return False, "URL has no domain"
        if '.' not in parsed.netloc:
            return False, "Invalid domain format"
        return True, "URL format is valid"
    except Exception as e:
        return False, f"URL parsing failed: {str(e)[:50]}"


def validate_url_accessible(url: str, timeout: int = 10, fallback_to_format: bool = True) -> Tuple[bool, str]:
    """
    Check if URL is accessible via HTTP request.

    Performs a HEAD request to check if the URL is reachable and returns
    a success status code (< 400). Falls back to GET request if HEAD fails
    with 403/405 (some websites block HEAD requests). If all HTTP methods fail
    and fallback_to_format is True, validates URL format instead.

    Args:
        url: URL to validate.
        timeout: Request timeout in seconds (default 10).
        fallback_to_format: If True, validate URL format when HTTP fails with 403.

    Returns:
        tuple: (is_accessible: bool, details: str)
            - is_accessible: True if URL returned status < 400 (or has valid format if fallback)
            - details: Description of result or error
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    try:
        # Try HEAD request first (faster, less bandwidth)
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers=headers
        )
        if response.status_code < 400:
            return True, f"URL accessible (status {response.status_code})"

        # If HEAD returns 403 or 405, try GET (some sites block HEAD)
        if response.status_code in (403, 405):
            response = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers=headers,
                stream=True  # Don't download full content
            )
            # Close connection immediately after checking status
            response.close()
            if response.status_code < 400:
                return True, f"URL accessible (status {response.status_code})"

        # If still 403 and fallback enabled, check URL format
        # (some sites block programmatic access but URL is valid)
        if response.status_code == 403 and fallback_to_format:
            is_valid, details = validate_url_format(url)
            if is_valid:
                return True, f"URL format valid (site blocks programmatic access)"
            return False, details

        return False, f"URL returned status {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "URL request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"URL request failed: {str(e)[:50]}"


def fetch_page_text_content(
    url: str,
    timeout: int = 10,
    max_chars: int = 15000,
    headers: Optional[Dict[str, str]] = None
) -> Tuple[Optional[str], str]:
    """Fetch URL and convert HTML to readable text content.

    Removes non-content elements (script, style, nav, header, footer, aside)
    and returns cleaned text suitable for LLM analysis.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        max_chars: Maximum characters to return (truncates if exceeded).
        headers: Optional HTTP headers to send with request.

    Returns:
        Tuple of (text_content or None, status_details).
    """
    try:
        from bs4 import BeautifulSoup

        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        request_headers = headers or default_headers

        response = requests.get(url, timeout=timeout, headers=request_headers)

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Get text and clean whitespace
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()

        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text, "OK"

    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)[:50]}"
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


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

def fetch_url_content(url):
    """
    Fetch and convert URL to markdown text using Playwright for JavaScript rendering.

    Uses Playwright to render JavaScript-heavy pages (like Fandom wikis) before
    extracting content. Falls back to requests for simpler pages.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: Markdown content (truncated to 60k chars), or None if fetch fails.

    Examples:
        >>> content = fetch_url_content("https://example.com/character-info")
        >>> if content:
        ...     print(f"Fetched {len(content)} characters of content")
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Launch headless browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            page = context.new_page()

            # Navigate: try domcontentloaded, fallback to load
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
            except Exception:
                page.goto(url, wait_until="load", timeout=5000)

            # Wait for selector (only reached if navigation succeeded)
            try:
                page.wait_for_selector("main, article, .mw-parser-output, #content", timeout=3000)
            except Exception:
                page.wait_for_timeout(500)
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass
           
            # Get the rendered HTML
            html_content = page.content()

            browser.close()

        if "JavaScript is disabled" in html_content:
            print(f"JavaScript appears to be disabled for {url}")
            return None
        
        # Convert HTML to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = True  # Don't convert hyperlinks to markdown format
        h.ignore_images = True  # Skip image references
        h.body_width = 0  # Don't wrap lines
        markdown = h.handle(html_content)

        # Truncate to ~60k chars (~15k tokens) to prevent excessive LLM usage
        if len(markdown) > 60000:
            markdown = markdown[:60000]
        return markdown

    except Exception as e:
        print(f"Error fetching {url} with Playwright: {e}")
        return None