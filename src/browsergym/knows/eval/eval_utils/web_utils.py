"""Web utilities for fetching and downloading content from URLs."""

import os
import re
import requests
import html2text
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup

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


# Opt-in aggressive HTML cleaning: extra tags + class/id pattern matches on
# top of the default strip. Used by callers that need clean article text
# (e.g. CP2/CP4 content relevance in sheets_45). Disabled by default to keep
# existing callers unchanged.
_AGGRESSIVE_JUNK_TAGS = (
    'noscript', 'iframe', 'form', 'button',
)
_AGGRESSIVE_JUNK_CLASS_ID = re.compile(
    r'\b('
    r'ad|ads|advert|advertisement|sponsored|promo|popup|modal|'
    r'cookie|newsletter|subscribe|signup|'
    r'sidebar|related|recommend|popular|trending|'
    r'comment|disqus|share|social|'
    r'menu|breadcrumb|toolbar'
    r')\b',
    re.IGNORECASE,
)


def _strip_aggressive_junk(soup) -> None:
    for el in soup(_AGGRESSIVE_JUNK_TAGS):
        el.decompose()
    for el in soup.find_all(attrs={'class': _AGGRESSIVE_JUNK_CLASS_ID}):
        el.decompose()
    for el in soup.find_all(attrs={'id': _AGGRESSIVE_JUNK_CLASS_ID}):
        el.decompose()


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


def download_image_from_url(url: str, temp_dir: str, timeout: int = 15, headers: Optional[Dict[str, str]] = None, wayback_fallback: bool = False) -> str:
    """Download image from URL to temp directory.

    Attempts a direct download first. If that fails and wayback_fallback is
    enabled, tries fetching an archived copy from the Wayback Machine.

    Args:
        url: The URL to download the image from.
        temp_dir: Directory to save the downloaded image.
        timeout: Request timeout in seconds.
        headers: Optional HTTP headers (e.g. Referer for hotlink protection).
        wayback_fallback: If True, try the Wayback Machine when direct download fails.

    Returns:
        Path to downloaded image, or None if download failed.
    """
    def _save_image_response(response, source_url):
        """Save a successful image response to disk and return the path."""
        content_type = response.headers.get('Content-Type', '')
        if content_type.startswith('image/'):
            ext = content_type.split('/')[-1].split(';')[0]
            if ext not in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
                ext = 'png'
            temp_path = os.path.join(temp_dir, f"url_image_{hash(source_url)}.{ext}")
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            return temp_path
        return None

    # Wikimedia (Wikipedia/Wikimedia Commons) requires a descriptive, identifying
    # User-Agent per https://meta.wikimedia.org/wiki/User-Agent_policy. Generic
    # browser UAs are rate-limited (HTTP 429), so use a compliant UA for those
    # hosts to avoid throttling.
    parsed_host = urlparse(url).netloc.lower()
    is_wikimedia = ("wikimedia.org" in parsed_host) or ("wikipedia.org" in parsed_host)
    if is_wikimedia:
        default_headers = {
            'User-Agent': (
                'BrowserGym-Knows-Eval/1.0 '
                '(https://github.com/alexgill321/KNOWS-benchmark; eval-bot) '
                'requests/python'
            )
        }
    else:
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    request_headers = headers or default_headers

    # Strategy 1: Direct download
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=True, headers=request_headers)
        if response.status_code == 200:
            result = _save_image_response(response, url)
            if result:
                return result
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")

    # Strategy 2: Wayback Machine archived image
    if wayback_fallback:
        try:
            wb_api = f"https://archive.org/wayback/available?url={url}"
            resp = requests.get(wb_api, timeout=10)
            snapshot = resp.json().get('archived_snapshots', {}).get('closest', {})
            wb_url = snapshot.get('url')
            if wb_url:
                # Rewrite to raw-image variant (im_ flag) so wayback returns
                # original image bytes instead of an HTML viewer page
                wb_url = re.sub(r"(/web/\d+)/", r"\1im_/", wb_url, count=1)
                wb_resp = requests.get(wb_url, timeout=timeout, headers=request_headers)
                if wb_resp.status_code == 200:
                    result = _save_image_response(wb_resp, url)
                    if result:
                        return result
        except Exception as e:
            print(f"Wayback fallback failed for {url}: {e}")

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

    # Wikimedia (Wikipedia/Wikimedia Commons) requires a descriptive User-Agent
    # per https://meta.wikimedia.org/wiki/User-Agent_policy; generic browser UAs
    # are aggressively rate-limited (HTTP 429).
    parsed_host = urlparse(url).netloc.lower()
    is_wikimedia = ("wikimedia.org" in parsed_host) or ("wikipedia.org" in parsed_host)

    for attempt in range(max_retries):
        try:
            if is_wikimedia:
                default_headers = {
                    'User-Agent': (
                        'BrowserGym-Knows-Eval/1.0 '
                        '(https://github.com/alexgill321/KNOWS-benchmark; eval-bot) '
                        'requests/python'
                    )
                }
            else:
                default_headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            request_headers = headers or default_headers
            response = requests.get(url, timeout=timeout, headers=request_headers)

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
    headers: Optional[Dict[str, str]] = None,
    aggressive_strip: bool = False,
) -> Tuple[Optional[str], str]:
    """Fetch URL and convert HTML to readable text content.

    Removes non-content elements (script, style, nav, header, footer, aside)
    and returns cleaned text suitable for LLM analysis. When
    ``aggressive_strip=True``, additionally removes inline ads, sidebars,
    cookie banners, comment widgets, and similar non-article elements.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.
        max_chars: Maximum characters to return (truncates if exceeded).
        headers: Optional HTTP headers to send with request.
        aggressive_strip: When True, also remove inline ads / sidebars /
            comments / forms / iframes via class+id pattern matching.

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

        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        text_default = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()

        if aggressive_strip:
            _strip_aggressive_junk(soup)
            text = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()
            # Safety net: if aggressive strip removed more than half the content,
            # our class/id regex over-matched the article body. Fall back.
            if len(text) < len(text_default) * 0.5:
                text = text_default
        else:
            text = text_default

        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text, "OK"

    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {str(e)[:50]}"
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


def fetch_page_text_content_playwright(
    url: str,
    max_chars: int = 15000,
    timeout: int = 10,
    aggressive_strip: bool = False,
) -> Tuple[Optional[str], str]:
    """Fetch URL using Playwright headless browser for JS-rendered content.

    Renders the page with Chromium, strips non-content elements, isolates the
    main content area, and converts to markdown via html2text. Useful for
    JavaScript-heavy sites (Khan Academy, LibreTexts, etc.) where basic
    requests.get() returns incomplete content.

    Falls back to fetch_page_text_content() if Playwright is unavailable.

    Return signature matches fetch_page_text_content() for drop-in compatibility.

    Args:
        url: URL to fetch.
        max_chars: Maximum characters to return (default 15000).
        timeout: Navigation timeout in seconds.
        aggressive_strip: When True, also remove inline ads / sidebars /
            comments / forms / iframes via class+id pattern matching.

    Returns:
        Tuple of (markdown_content or None, status_details).
    """
    try:
        import html2text
        from bs4 import BeautifulSoup
        from playwright.sync_api import sync_playwright
        try:
            from playwright_stealth import Stealth
        except ImportError:
            Stealth = None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=[
                '--disable-blink-features=AutomationControlled',
            ])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36",
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }
            )
            if Stealth is not None:
                Stealth().apply_stealth_sync(context)
            page = context.new_page()

            # Belt-and-braces: even with stealth, keep this baseline.
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            timeout_ms = timeout * 1000
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception:
                try:
                    page.goto(url, wait_until="load", timeout=timeout_ms // 2)
                except Exception as nav_err:
                    browser.close()
                    return None, f"Navigation failed: {str(nav_err)[:80]}"

            # Wait for content to render
            try:
                page.wait_for_selector(
                    "main, article, .mw-parser-output, #content, .content, body",
                    timeout=3000
                )
            except Exception:
                page.wait_for_timeout(2000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

            html_content = page.content()
            browser.close()

        # Strip non-content elements
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        if aggressive_strip:
            _strip_aggressive_junk(soup)

        # Isolate main content area
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find(class_='mw-parser-output') or
            soup.find(id='content') or
            soup.find(class_='content') or
            soup.body or
            soup
        )

        cleaned_html = str(main_content)

        if "JavaScript is disabled" in cleaned_html:
            return None, "JavaScript appears disabled on this page"

        # Convert to markdown
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        markdown = h.handle(cleaned_html)

        if len(markdown) > max_chars:
            markdown = markdown[:max_chars]

        return markdown, "OK"

    except Exception as e:
        # Fallback to requests-based approach
        return fetch_page_text_content(url, timeout=timeout, max_chars=max_chars)


def _looks_like_deny_page(content: Optional[str]) -> bool:
    """Heuristic: True if `content` is a 200-served bot/access-denied page."""
    if not content:
        return False
    head = content[:5000].lower()
    return any(m in head for m in _DENY_PAGE_MARKERS)


def fetch_with_fallbacks(url: str, max_chars: int = 15000, timeout: int = 15,
                         aggressive_strip: bool = False) -> Tuple[Optional[str], str]:
    """Fetch URL content with multiple fallback strategies.

    Tries in order:
    1. Plain requests + HTML-to-text (fastest, works for most static sites)
    2. Playwright with stealth (for JS-rendered pages)
    3. Playwright retry with longer timeout (2x)
    4. Wayback Machine archived snapshot
    5. archive.today snapshot (via curl-cffi; archive.ph is Cloudflare-fronted)

    Returns on first success (content > 200 chars to avoid error pages).

    Args:
        url: URL to fetch.
        max_chars: Maximum characters to return.
        timeout: Request timeout in seconds.
        aggressive_strip: When True, also remove ads / sidebars / comments
            via class+id pattern matching. Forwarded to each strategy.

    Returns:
        Tuple of (text_content or None, status_details).
    """
    def _is_real_content(c: Optional[str]) -> bool:
        return bool(c and len(c.strip()) > 200 and not _looks_like_deny_page(c))

    content, status = fetch_page_text_content(url, max_chars=max_chars, timeout=timeout,
                                              aggressive_strip=aggressive_strip)
    if _is_real_content(content):
        return content, "OK (requests)"

    content, status = fetch_page_text_content_playwright(url, max_chars=max_chars, timeout=timeout,
                                                        aggressive_strip=aggressive_strip)
    if _is_real_content(content):
        return content, "OK (playwright)"

    content, status = fetch_page_text_content_playwright(url, max_chars=max_chars, timeout=timeout * 2,
                                                        aggressive_strip=aggressive_strip)
    if _is_real_content(content):
        return content, "OK (playwright-retry)"

    try:
        wb_api = f"https://archive.org/wayback/available?url={url}"
        resp = requests.get(wb_api, timeout=10)
        snapshot = resp.json().get('archived_snapshots', {}).get('closest', {})
        wb_url = snapshot.get('url', '')
        if wb_url:
            content, status = fetch_page_text_content(wb_url, timeout=timeout, max_chars=max_chars,
                                                      aggressive_strip=aggressive_strip)
            if _is_real_content(content):
                return content, "OK (wayback)"
    except Exception:
        pass

    try:
        archive_url = f"https://archive.ph/newest/{url}"
        content, status = _fetch_with_curl_cffi(archive_url, max_chars=max_chars, timeout=timeout,
                                                aggressive_strip=aggressive_strip)
        if _is_real_content(content):
            head = content[:1500].lower()
            if 'no results' not in head and 'no archive' not in head:
                return content, "OK (archive.today)"
    except Exception:
        pass

    return None, "All fetch strategies failed"


_CURL_CFFI_PROFILES = ("safari17_0", "chrome120", "chrome131", "edge99")
_DENY_PAGE_MARKERS = (
    "access denied", "permission to access", "verify you are human",
    "request blocked", "forbidden", "are you a robot",
)


def _fetch_with_curl_cffi(url: str, max_chars: int = 15000, timeout: int = 30,
                          aggressive_strip: bool = False) -> Tuple[Optional[str], str]:
    """Fetch URL via curl-cffi, sweeping browser TLS/HTTP2 fingerprints.

    Edmunds-style sites block Chrome fingerprints but accept Safari; some
    Cloudflare hosts are the inverse. Tries each profile until one returns a
    200 that doesn't look like a deny/challenge page. Returns `(content, status)`.

    Args:
        url, max_chars, timeout: as elsewhere.
        aggressive_strip: When True, also remove ads / sidebars / comments
            via class+id pattern matching.
    """
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None, "curl-cffi not installed"

    last_status = "curl-cffi all profiles failed"
    for profile in _CURL_CFFI_PROFILES:
        try:
            resp = curl_requests.get(url, impersonate=profile,
                                     timeout=timeout, allow_redirects=True)
        except Exception as e:
            last_status = f"curl-cffi[{profile}] {str(e)[:60]}"
            continue
        if resp.status_code != 200:
            last_status = f"curl-cffi[{profile}] HTTP {resp.status_code}"
            continue
        body = resp.text or ""
        head = body[:5000].lower()
        if any(m in head for m in _DENY_PAGE_MARKERS):
            last_status = f"curl-cffi[{profile}] deny-page (200 body)"
            continue
        soup = BeautifulSoup(body, 'html.parser')
        for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            el.decompose()

        text_default = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()

        if aggressive_strip:
            _strip_aggressive_junk(soup)
            text = re.sub(r'\s+', ' ', soup.get_text(separator=' ')).strip()
            # Safety net: aggressive over-strips when article body is wrapped
            # in a class matching the junk regex. Fall back to default.
            if len(text) < len(text_default) * 0.5:
                text = text_default
        else:
            text = text_default
        if len(text) > max_chars:
            text = text[:max_chars] + '...'
        if len(text) <= 200:
            last_status = f"curl-cffi[{profile}] response too short"
            continue
        return text, f"OK (curl-cffi {profile})"
    return None, last_status


def fetch_with_fallbacks_extended(url: str, max_chars: int = 15000, timeout: int = 30,
                                  aggressive_strip: bool = False) -> Tuple[Optional[str], str]:
    """`fetch_with_fallbacks` (5 strategies) + curl-cffi profile sweep as a 6th
    strategy for Cloudflare/edmunds-style hosts whose TLS/HTTP2 fingerprint blocks
    Python's `requests` and Playwright. Returns `(content, status)`.

    ``aggressive_strip`` is forwarded to every strategy.
    """
    content, status = fetch_with_fallbacks(url, max_chars=max_chars, timeout=timeout,
                                           aggressive_strip=aggressive_strip)
    if content:
        return content, status
    cf_content, cf_status = _fetch_with_curl_cffi(url, max_chars=max_chars, timeout=timeout,
                                                  aggressive_strip=aggressive_strip)
    if cf_content:
        return cf_content, cf_status
    return None, f"{status}; {cf_status}"


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


def normalize_url_for_comparison(url: str) -> str:
    """Normalize URL for consistent comparison (e.g., with browsing history).

    Removes common URL variations that don't affect content:
    - Trailing slashes
    - Query parameters
    - URL fragments (#anchor)
    - www. prefix
    - Protocol differences (lowercased)

    Args:
        url: URL to normalize.

    Returns:
        Normalized URL string.

    Examples:
        >>> normalize_url_for_comparison('https://www.example.com/page?foo=bar#section')
        'https://example.com/page'
        >>> normalize_url_for_comparison('HTTP://Example.COM/page/')
        'http://example.com/page'
    """
    if not url:
        return ''

    try:
        parsed = urlparse(url)

        # Normalize domain (remove www.)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]

        # Rebuild URL without query params and fragments
        normalized = urlunparse((
            parsed.scheme.lower(),
            domain,
            parsed.path.rstrip('/'),
            '',  # params
            '',  # query
            ''   # fragment
        ))

        return normalized

    except Exception:
        # Fallback: simple lowercase and strip trailing slash
        return url.lower().rstrip('/')


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
                try:
                    page.goto(url, wait_until="load", timeout=5000)
                except:
                    pass

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
    
def download_page_images(
        url, 
        folder,
        timeout: int = 10,
        headers: Optional[Dict[str, str]] = None
    ):
    """
    Download all images from a webpage to a specified folder.

    Args:
        page_url (str): The URL of the webpage to download images from.
        folder (str): The folder to save the downloaded images.

    Returns:
        list: A list of filenames of the downloaded images, or an empty list if no images are found.

    Examples:
        >>> downloaded_files = download_page_images("https://example.com", "./images")
        >>> print(f"Downloaded {len(downloaded_files)} images")
    """
    from PIL import Image
    # Only accept these image extensions
    allowed_exts = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # 1. Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # 2. Get the HTML of the website
    default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    request_headers = headers or default_headers
    response = requests.get(url, timeout=timeout, headers=request_headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. Find all <img> tags
    img_tags = soup.find_all('img')
    # print(f"Found {len(img_tags)} images.")
    downloaded_files = []
    for i, img in enumerate(img_tags):
        # Get the 'src' attribute
        img_url = img.get('src')
        if not img_url:
            continue

        # Handle relative URLs (e.g., /images/pic.jpg -> https://site.com/images/pic.jpg)
        img_url = urljoin(url, img_url)

        try:
            # Extract extension from URL path
            parsed = urlparse(img_url)
            _, ext = os.path.splitext(parsed.path or "")
            ext = ext.lower().lstrip('.') if ext else ''

            # Download the image data
            response = requests.get(img_url, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()

            # Prefer URL extension when valid
            if ext and ext in allowed_exts:
                chosen_ext = ext
            else:
                # Map common content-types to extensions
                ct_map = {
                    'image/png': 'png',
                    'image/jpeg': 'jpg',
                    'image/jpg': 'jpg',
                    'image/bmp': 'bmp',
                    'image/tiff': 'tiff',
                    'image/x-tiff': 'tiff'
                }
                ct = content_type.split(';')[0].strip()
                chosen_ext = ct_map.get(ct)

            # Skip if extension is not allowed
            if not chosen_ext or chosen_ext not in allowed_exts:
                # print(f"Skipping {img_url} (unsupported type)")
                continue
            
            # Create a filename
            filename = os.path.basename(urlparse(img_url).path)
            if not filename or '.' not in filename:
                filename = f"image_{i}.{chosen_ext}"
            else:
                # Ensure correct extension
                name_without_ext = os.path.splitext(filename)[0]
                filename = f"{name_without_ext}.{chosen_ext}"
                
            filepath = os.path.join(folder, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            # Validate that the file is a real image
            try:
                Image.open(filepath).verify()
            except Exception:
                os.remove(filepath)
                continue
            # print(f"Downloaded: {filename}")
            downloaded_files.append(filename)
        except Exception as e:
            # print(f"Could not download {img_url}: {e}")
            pass
    return downloaded_files