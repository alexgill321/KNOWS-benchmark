"""Utility functions for sheets_10_paper_sorting evaluator.

This module contains functions for:
- URL validation (arXiv, Google Drive)
- Author name handling and comparison
- Figure 1 extraction from arXiv source files
- Row color/formatting detection
- Preprocessing helpers (Google Scholar scraping, arXiv search)
"""

import os
import re
import sys
import json
import tarfile
import gzip
import time as _time
import tempfile
import unicodedata
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse, parse_qs
import requests

# Global rate limiter for raw arxiv HTTP requests (HTML, source downloads).
# Ensures no two requests to export.arxiv.org are within 1s of each other.
_last_arxiv_request_time = 0.0

def _arxiv_rate_limit():
    """Wait if needed to ensure at least 5s between arxiv HTTP requests."""
    global _last_arxiv_request_time
    elapsed = _time.time() - _last_arxiv_request_time
    if elapsed < 5.0:
        _time.sleep(5.0 - elapsed)
    _last_arxiv_request_time = _time.time()

def _arxiv_request_with_retry(url: str, max_retries: int = 3, timeout: int = 30,
                               headers: dict = None, stream: bool = False) -> requests.Response:
    """Make an HTTP request to arxiv with rate limiting and retry on 429 or connection errors.

    Args:
        url: URL to request.
        max_retries: Maximum retry attempts on 429 or connection errors.
        timeout: Request timeout in seconds.
        headers: Optional request headers.
        stream: Whether to stream the response.

    Returns:
        requests.Response object.

    Raises:
        requests.RequestException: If request fails after all retries.
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            _arxiv_rate_limit()
            response = requests.get(url, headers=headers, timeout=timeout, stream=stream)
            if response.status_code == 429 and attempt < max_retries:
                wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
                print(f"      arXiv rate limited (429), waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                _time.sleep(wait_time)
                continue
            return response
        except (requests.RequestException, ConnectionError) as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = 10 * (2 ** attempt)
                print(f"      Connection error, waiting {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}")
                _time.sleep(wait_time)
            else:
                raise
    raise last_exception

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from rapidfuzz import fuzz
from src.browsergym.knows.eval.eval_utils.text_utils import fuzzy_match_text

# Task-level constants
# Note: TASK_DIR points to the template level. Instance-specific data is in instance_X/data/
TASK_DIR = os.path.dirname(os.path.abspath(__file__))

# Instance-aware directory helpers
def get_instance_dir(instance: int = 1) -> str:
    """Get the directory for a specific instance."""
    return os.path.join(TASK_DIR, f"instance_{instance}")

def get_data_dir(instance: int = 1) -> str:
    """Get the data directory for a specific instance."""
    return os.path.join(get_instance_dir(instance), "data")

def get_figures_dir(instance: int = 1) -> str:
    """Get the gold figures directory for a specific instance."""
    return os.path.join(get_data_dir(instance), "gold_figures")

# Default dirs for backwards compatibility (instance_1)
DATA_DIR = get_data_dir(1)
FIGURES_DIR = get_figures_dir(1)

# HTTP headers for arXiv requests (avoid 403 errors)
ARXIV_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# ============================================================================
# Shared Helper Functions
# ============================================================================

def ensure_data_directories(instance: int = 1):
    """Create necessary data directories if they don't exist."""
    os.makedirs(get_data_dir(instance), exist_ok=True)
    os.makedirs(get_figures_dir(instance), exist_ok=True)


def load_json(filename: str, instance: int = 1) -> Optional[Dict]:
    """Load JSON file from data directory.

    Args:
        filename: Name of the JSON file (e.g., 'gold_papers.json')
        instance: Instance number (default: 1)

    Returns:
        Parsed JSON data, or None if file doesn't exist.
    """
    filepath = os.path.join(get_data_dir(instance), filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filename: str, instance: int = 1):
    """Save data to JSON file in data directory.

    Args:
        data: Data to save (dict, list, etc.)
        filename: Name of the JSON file (e.g., 'gold_papers.json')
        instance: Instance number (default: 1)
    """
    filepath = os.path.join(get_data_dir(instance), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {filepath}")


def fetch_arxiv_html(arxiv_id: str) -> Tuple[bool, str, str]:
    """Fetch HTML version of an arXiv paper.

    Args:
        arxiv_id: The arXiv paper ID (e.g., '2301.12345').

    Returns:
        Tuple of (success, html_content, message).
    """
    html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

    try:
        response = _arxiv_request_with_retry(html_url, headers=ARXIV_HEADERS, timeout=30)
        if response.status_code == 200:
            return True, response.text, "HTML fetched successfully"
        else:
            return False, "", f"HTTP {response.status_code}"
    except requests.RequestException as e:
        return False, "", f"Request error: {e}"


# ============================================================================
# URL Validation Functions
# ============================================================================

def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """Extract arXiv ID from a URL.

    Handles various URL formats:
    - https://arxiv.org/abs/2301.12345
    - https://arxiv.org/pdf/2301.12345.pdf
    - http://arxiv.org/abs/2301.12345v1
    - https://ar5iv.org/abs/2301.12345
    - http://export.arxiv.org/abs/2301.12345
    - URLs with version numbers (2301.12345v2)

    Args:
        url: The arXiv URL to parse.

    Returns:
        The arXiv ID (e.g., "2301.12345") or None if not found.
    """
    if not url:
        return None

    # Patterns for arXiv IDs
    # New format: YYMM.NNNNN (e.g., 2301.12345)
    # Old format: category/YYMMNNN (e.g., cs.CV/0601001)
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?',  # New format with version
        r'arxiv\.org/(?:abs|pdf)/([\w\-\.]+/\d+)(?:v\d+)?',    # Old format with version
        r'ar5iv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?',   # ar5iv mirror
        r'(\d{4}\.\d{4,5})(?:v\d+)?\.pdf',                     # Just ID in PDF filename
        r'(\d{4}\.\d{4,5})(?:v\d+)?$',                         # Just the ID at end
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)
            # Remove version suffix if present (for patterns that don't capture it)
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
            return arxiv_id

    return None


def extract_drive_file_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from a URL.

    Handles various URL formats:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://docs.google.com/document/d/FILE_ID/edit

    Args:
        url: The Google Drive URL to parse.

    Returns:
        The file ID or None if not found.
    """
    if not url:
        return None

    # Pattern 1: /d/FILE_ID/
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    # Pattern 2: ?id=FILE_ID
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if 'id' in query_params:
        return query_params['id'][0]

    return None


# ============================================================================
# Author Handling Functions
# ============================================================================

def normalize_author_name(name: str) -> str:
    """Normalize an author name for comparison.

    - Converts to lowercase
    - Removes accents/diacritics
    - Removes extra whitespace
    - Handles common variations (Jr., III, etc.)

    Args:
        name: The author name to normalize.

    Returns:
        Normalized author name.
    """
    if not name:
        return ""

    # Convert to lowercase
    name = name.lower().strip()

    # Remove accents/diacritics
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remove common suffixes
    suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]

    # Normalize whitespace
    name = ' '.join(name.split())

    return name


def parse_authors_string(authors_str: str) -> List[str]:
    """Parse a comma-separated author string into a list.

    Handles:
    - Comma-separated names
    - "and" as separator
    - Newlines within the string

    Args:
        authors_str: The author string to parse.

    Returns:
        List of individual author names.
    """
    if not authors_str:
        return []

    # Replace newlines with commas
    authors_str = authors_str.replace('\n', ', ')

    # Replace " and " with comma
    authors_str = re.sub(r'\s+and\s+', ', ', authors_str, flags=re.IGNORECASE)

    # Split by comma
    authors = [a.strip() for a in authors_str.split(',')]

    # Remove empty strings
    authors = [a for a in authors if a]

    return authors


def compare_authors_list(user_authors: List[str], gold_authors: List[str],
                         strict: bool = False) -> Tuple[bool, str]:
    """Compare two author lists.

    Args:
        user_authors: List of authors from user's spreadsheet.
        gold_authors: List of gold standard authors.
        strict: If True, requires exact order and count match.

    Returns:
        Tuple of (is_match, details_message).
    """
    # Import normalize_name from eval_utils
    from src.browsergym.knows.eval.eval_utils.text_utils import normalize_name

    if not user_authors and not gold_authors:
        return True, "Both author lists are empty"

    if not user_authors:
        return False, "User author list is empty"

    if not gold_authors:
        return False, "Gold author list is empty"

    # Normalize all names using eval_utils normalize_name
    user_normalized = [normalize_name(a, remove_suffixes=True) for a in user_authors]
    gold_normalized = [normalize_name(a, remove_suffixes=True) for a in gold_authors]

    if strict:
        # Exact match required
        if user_normalized == gold_normalized:
            return True, f"Exact match: {len(user_authors)} authors"
        else:
            return False, f"Authors don't match exactly. User: {user_authors[:3]}..., Gold: {gold_authors[:3]}..."
    else:
        # Check if first author matches and most authors are present
        first_match = user_normalized[0] == gold_normalized[0] if user_normalized and gold_normalized else False

        # Count matches
        matches = sum(1 for u in user_normalized if u in gold_normalized)
        match_ratio = matches / len(gold_normalized) if gold_normalized else 0

        if first_match and match_ratio >= 0.8:
            return True, f"First author matches, {matches}/{len(gold_normalized)} authors found"
        elif first_match:
            return False, f"First author matches but only {matches}/{len(gold_normalized)} authors found"
        else:
            return False, f"First author doesn't match. User: {user_authors[0] if user_authors else 'N/A'}, Gold: {gold_authors[0] if gold_authors else 'N/A'}"


# ============================================================================
# Figure 1 Extraction Functions
# ============================================================================

# --- Stage 1: arXiv HTML Extraction ---

def convert_svg_to_png(svg_content: str, output_path: str) -> Optional[str]:
    """Convert SVG content to PNG file.

    Args:
        svg_content: Raw SVG content as string.
        output_path: Path to save the PNG file.

    Returns:
        Path to PNG file if successful, None otherwise.
    """
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_path, scale=2)
        return output_path
    except ImportError:
        # Try with svglib as fallback
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            import io

            drawing = svg2rlg(io.StringIO(svg_content))
            if drawing:
                renderPM.drawToFile(drawing, output_path, fmt="PNG")
                return output_path
        except Exception:
            pass
    except Exception:
        pass

    return None


def parse_html_for_figure_1(html_content: str, arxiv_id: str) -> Tuple[bool, Optional[str], str]:
    """Stage 1a: Automatically parse HTML to find Figure 1 image URL or SVG content.

    Looks for <figure class="ltx_figure" id="S1.F1"> or similar patterns.
    Handles both <img> tags and inline <svg> elements (for TikZ figures).

    Args:
        html_content: Raw HTML content from arXiv.
        arxiv_id: The arXiv paper ID.

    Returns:
        Tuple of (success, image_url_or_svg, message).
        For img: returns URL string
        For svg: returns SVG content string prefixed with "SVG:"
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return False, None, "BeautifulSoup not installed"

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find Figure 1 by ID pattern (S1.F1, S2.F1, A1.F1, S0.F1, etc.)
    for figure in soup.find_all('figure', class_='ltx_figure'):
        fig_id = figure.get('id', '')
        # Match patterns like S0.F1, S1.F1, S2.F1, A1.F1, etc. (Section/Appendix + Figure 1)
        if re.search(r'[SA]\d+\.F1$', fig_id):
            # First try to find an <img> tag
            img = figure.find('img')
            if img and img.get('src'):
                img_src = img['src']
                if img_src.startswith('http'):
                    img_url = img_src
                elif img_src.startswith('/'):
                    img_url = f"https://ar5iv.labs.arxiv.org{img_src}"
                else:
                    img_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}/{img_src}"
                return True, img_url, f"Found Figure 1 by ID: {fig_id}"

            # Check for inline SVG (common for TikZ figures)
            svg = figure.find('svg')
            if svg:
                # Return SVG content with prefix to indicate it's SVG data
                svg_content = str(svg)
                return True, f"SVG:{svg_content}", f"Found Figure 1 (SVG) by ID: {fig_id}"

    return False, None, "Figure 1 not found by automatic parsing"


def parse_html_for_figure_1_with_llm(html_content: str, arxiv_id: str, model) -> Tuple[bool, Optional[str], str]:
    """Stage 1b: Use LLM to parse raw HTML and find Figure 1 image URL.

    Passes the raw HTML content to the LLM to identify Figure 1.

    Args:
        html_content: Raw HTML content from arXiv.
        arxiv_id: The arXiv paper ID.
        model: LLM model to use for parsing.

    Returns:
        Tuple of (success, image_url, message).
    """
    # Truncate HTML if too long (keep first 100k chars which should include figures)
    if len(html_content) > 100000:
        html_content = html_content[:100000] + "\n... [truncated]"

    prompt = f"""Analyze this arXiv paper HTML and find the image URL for Figure 1.

Look for:
- <figure> elements with class "ltx_figure"
- Figure 1 typically has id like "S1.F1" or similar
- Inside the figure, find the <img> tag and its src attribute

HTML content:
```html
{html_content}
```

Return ONLY the image src value (e.g., "x1.png") for Figure 1, or "NOT_FOUND" if you cannot find it."""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an HTML parser. Extract the requested information precisely."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    try:
        response = model(messages).strip()

        # Clean up response - remove quotes, extra text
        response_clean = re.sub(r'["\']', '', response).strip()

        if response_clean and response_clean.upper() != "NOT_FOUND":
            # Ensure it looks like a valid image filename
            if re.search(r'\.(png|jpg|jpeg|svg)$', response_clean, re.IGNORECASE):
                if response_clean.startswith('http'):
                    img_url = response_clean
                elif response_clean.startswith('/'):
                    img_url = f"https://ar5iv.labs.arxiv.org{response_clean}"
                else:
                    img_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}/{response_clean}"
                return True, img_url, f"LLM identified Figure 1: {response_clean}"

        return False, None, "LLM could not identify Figure 1"

    except Exception as e:
        return False, None, f"LLM error: {e}"


def _get_svg_cache_dir() -> str:
    """Get the SVG cache directory, creating it if needed."""
    cache_dir = os.path.join(TASK_DIR, '.svg_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def extract_figure_1_from_html(arxiv_id: str, model=None) -> Tuple[bool, Optional[bytes], str]:
    """Stage 1: Extract Figure 1 from arXiv HTML page.

    Uses 2-part approach:
    - 1a: Automatic HTML parsing
    - 1b: LLM parsing fallback (if model provided)

    Handles both regular images (<img> tags) and inline SVG (TikZ figures).
    Caches SVG content locally so retries don't need to re-fetch from arXiv.

    Args:
        arxiv_id: The arXiv paper ID.
        model: Optional LLM model for Stage 1b fallback.

    Returns:
        Tuple of (success, image_bytes, message).
        image_bytes is the PNG data if found.
    """
    # Check SVG cache first — avoids hitting arXiv for previously found SVGs
    svg_cache_path = os.path.join(_get_svg_cache_dir(), f"{arxiv_id.replace('/', '_')}.svg")
    if os.path.exists(svg_cache_path):
        print(f"      Using cached SVG for {arxiv_id}")
        with open(svg_cache_path, 'r') as f:
            svg_content = f.read()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        png_path = convert_svg_to_png(svg_content, tmp_path)
        if png_path and os.path.exists(png_path):
            with open(png_path, 'rb') as f:
                png_bytes = f.read()
            os.unlink(png_path)
            return True, png_bytes, f"Found Figure 1 (SVG, cached)"
        else:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return False, None, f"SVG cached but conversion still failed"

    html_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

    # Headers to avoid 403 errors from arXiv
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    try:
        # Fetch HTML (with rate limiting and retry on 429)
        response = _arxiv_request_with_retry(html_url, headers=headers, timeout=30)
        if response.status_code != 200:
            return False, None, f"HTML not available (status {response.status_code})"

        html_content = response.text

        # Stage 1a: Automatic parsing
        success, img_url_or_svg, msg = parse_html_for_figure_1(html_content, arxiv_id)

        # Stage 1b: LLM fallback
        if not success and model:
            success, img_url_or_svg, msg = parse_html_for_figure_1_with_llm(html_content, arxiv_id, model)

        if not success:
            return False, None, msg

        # Check if result is SVG content (prefixed with "SVG:")
        if img_url_or_svg.startswith("SVG:"):
            svg_content = img_url_or_svg[4:]  # Remove "SVG:" prefix

            # Cache SVG content locally for future retries
            with open(svg_cache_path, 'w') as f:
                f.write(svg_content)

            # Try to convert SVG to PNG
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name

            png_path = convert_svg_to_png(svg_content, tmp_path)
            if png_path and os.path.exists(png_path):
                with open(png_path, 'rb') as f:
                    png_bytes = f.read()
                os.unlink(png_path)  # Clean up temp file
                return True, png_bytes, msg
            else:
                # Clean up temp file if conversion failed
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return False, None, f"{msg} but SVG to PNG conversion failed"
        else:
            # Regular image URL - download it (with rate limiting and retry)
            img_response = _arxiv_request_with_retry(img_url_or_svg, headers=headers, timeout=30)
            if img_response.status_code != 200:
                return False, None, f"Failed to download image: {img_url_or_svg}"

            return True, img_response.content, msg

    except requests.RequestException as e:
        return False, None, f"Request error: {e}"
    except Exception as e:
        return False, None, f"Error: {e}"


# --- Stage 2 & 3: LaTeX and Source-based Extraction ---

def download_arxiv_source(arxiv_id: str, output_dir: str, timeout: int = 60) -> Tuple[bool, str, List[str]]:
    """Download arXiv source files (tar.gz) for a paper.

    URL format: https://arxiv.org/e-print/{arxiv_id}

    Args:
        arxiv_id: The arXiv paper ID.
        output_dir: Directory to extract files to.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (success, message, list_of_extracted_files).
    """
    url = f"https://export.arxiv.org/e-print/{arxiv_id}"

    # Headers to avoid 403 errors from arXiv
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*',
    }

    try:
        response = _arxiv_request_with_retry(url, headers=headers, timeout=timeout, stream=False)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')

        # Create output directory
        paper_dir = os.path.join(output_dir, arxiv_id.replace('/', '_'))
        os.makedirs(paper_dir, exist_ok=True)

        extracted_files = []

        if 'application/x-eprint-tar' in content_type or 'application/gzip' in content_type:
            # It's a gzipped tar file
            tar_path = os.path.join(paper_dir, 'source.tar.gz')
            with open(tar_path, 'wb') as f:
                f.write(response.content)

            # Extract tar.gz
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                    extracted_files = [os.path.join(paper_dir, m.name) for m in tar.getmembers() if m.isfile()]
            except tarfile.TarError:
                # Maybe it's just gzipped, not tar
                try:
                    with gzip.open(tar_path, 'rb') as gz:
                        content = gz.read()
                        tex_path = os.path.join(paper_dir, 'main.tex')
                        with open(tex_path, 'wb') as f:
                            f.write(content)
                        extracted_files = [tex_path]
                except Exception as e:
                    return False, f"Failed to extract gzip: {e}", []

            # Clean up tar file
            if os.path.exists(tar_path):
                os.remove(tar_path)

        elif 'application/x-tex' in content_type or 'text/plain' in content_type:
            # Single TeX file
            tex_path = os.path.join(paper_dir, 'main.tex')
            with open(tex_path, 'wb') as f:
                f.write(response.content)
            extracted_files = [tex_path]
        else:
            # Unknown format, save as-is
            raw_path = os.path.join(paper_dir, 'source_raw')
            with open(raw_path, 'wb') as f:
                f.write(response.content)
            extracted_files = [raw_path]

        return True, f"Downloaded and extracted {len(extracted_files)} files", extracted_files

    except requests.RequestException as e:
        return False, f"Failed to download: {e}", []
    except Exception as e:
        return False, f"Error processing source: {e}", []


def find_files_by_extension(source_dir: str, extensions: List[str],
                            prioritize: Optional[str] = None) -> List[str]:
    """Find all files with given extensions in a source directory.

    Args:
        source_dir: Directory to search.
        extensions: List of extensions to match (e.g., ['.tex', '.png']).
        prioritize: Optional filename to put first if found (e.g., 'main.tex').

    Returns:
        List of file paths matching the extensions.
    """
    found_files = []
    priority_file = None
    ext_lower = [ext.lower() for ext in extensions]

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ext_lower):
                filepath = os.path.join(root, file)
                if prioritize and file.lower() == prioritize.lower():
                    priority_file = filepath
                else:
                    found_files.append(filepath)

    if priority_file:
        found_files.insert(0, priority_file)

    return found_files


def find_tex_files(source_dir: str) -> List[str]:
    """Find all .tex files in a source directory, with main.tex first."""
    return find_files_by_extension(source_dir, ['.tex'], prioritize='main.tex')


def find_png_files(source_dir: str) -> List[str]:
    """Find all PNG image files in a source directory."""
    return find_files_by_extension(source_dir, ['.png'])


def convert_pdf_to_png(pdf_path: str, output_path: Optional[str] = None, dpi: int = 150) -> Optional[str]:
    """Convert a PDF file to PNG.

    Uses pdf2image library (requires poppler).

    Args:
        pdf_path: Path to the PDF file.
        output_path: Optional output path for PNG. If None, uses same path with .png extension.
        dpi: Resolution for the conversion.

    Returns:
        Path to the PNG file, or None if conversion failed.
    """
    try:
        from pdf2image import convert_from_path

        if output_path is None:
            output_path = os.path.splitext(pdf_path)[0] + '.png'

        # Convert first page only
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=dpi)

        if images:
            images[0].save(output_path, 'PNG')
            return output_path

        return None

    except ImportError:
        # Try alternative: use subprocess with pdftoppm if available
        try:
            import subprocess
            if output_path is None:
                output_path = os.path.splitext(pdf_path)[0] + '.png'

            # pdftoppm outputs with a suffix, so we need to handle that
            output_base = os.path.splitext(output_path)[0]
            result = subprocess.run(
                ['pdftoppm', '-png', '-f', '1', '-l', '1', '-r', str(dpi), pdf_path, output_base],
                capture_output=True, timeout=30
            )

            # pdftoppm adds -1 suffix for single page
            expected_output = f"{output_base}-1.png"
            if os.path.exists(expected_output):
                os.rename(expected_output, output_path)
                return output_path

            # Sometimes it doesn't add the suffix
            if os.path.exists(output_path):
                return output_path

            return None
        except Exception:
            return None
    except Exception:
        return None


def find_pdf_files(source_dir: str) -> List[str]:
    """Find all PDF image files in a source directory."""
    return find_files_by_extension(source_dir, ['.pdf'])


def parse_latex_for_figure_1(tex_content: str) -> Optional[str]:
    """Parse LaTeX content to find Figure 1 and extract its image filename.

    Looks for patterns like:
    - \\begin{figure} ... \\includegraphics{filename} ... \\label{fig:1} ... \\end{figure}
    - \\begin{figure} ... \\includegraphics{filename} ... \\caption{...Figure 1...} ... \\end{figure}
    - \\begin{figure} ... \\input{tikz_file} ... \\end{figure} (TikZ figures)
    - \\begin{figure} ... \\begin{tikzpicture} ... \\end{figure} (inline TikZ)
    - The first figure environment in the document (often Figure 1)

    Args:
        tex_content: Content of a .tex file.

    Returns:
        The image filename referenced in Figure 1, or None if not found.
        For TikZ figures, returns "TIKZ:filename" or "TIKZ:inline".
    """
    # Remove comments (lines starting with %)
    lines = tex_content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments but keep the rest
        comment_idx = line.find('%')
        if comment_idx == 0:
            continue  # Skip full comment lines
        elif comment_idx > 0:
            # Check if % is escaped
            if line[comment_idx - 1] != '\\':
                line = line[:comment_idx]
        cleaned_lines.append(line)
    tex_content = '\n'.join(cleaned_lines)

    # Find all figure environments
    # Pattern to match \begin{figure} ... \end{figure} (including figure*)
    figure_pattern = r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'
    figure_matches = re.findall(figure_pattern, tex_content, re.DOTALL | re.IGNORECASE)

    if not figure_matches:
        return None

    # For each figure, check if it's Figure 1
    for idx, figure_content in enumerate(figure_matches):
        is_figure_1 = False

        # Check for label like \label{fig:1}, \label{fig1}, \label{figure1}
        label_patterns = [
            r'\\label\{fig:1\}',
            r'\\label\{fig1\}',
            r'\\label\{figure1\}',
            r'\\label\{fig:one\}',
            r'\\label\{fig_1\}',
        ]
        for lp in label_patterns:
            if re.search(lp, figure_content, re.IGNORECASE):
                is_figure_1 = True
                break

        # Check caption for "Figure 1" or if this is the first figure
        if not is_figure_1:
            caption_match = re.search(r'\\caption\{([^}]*)\}', figure_content, re.DOTALL)
            if caption_match:
                caption_text = caption_match.group(1)
                # Check if caption explicitly mentions "Figure 1" or similar
                if re.search(r'figure\s*1\b', caption_text, re.IGNORECASE):
                    is_figure_1 = True

        # If this is the first figure environment, assume it's Figure 1
        if idx == 0:
            is_figure_1 = True

        if is_figure_1:
            # First, try to extract includegraphics filename
            # Patterns: \includegraphics{file}, \includegraphics[options]{file}
            includegraphics_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
            img_match = re.search(includegraphics_pattern, figure_content)

            if img_match:
                filename = img_match.group(1).strip()
                return filename

            # Check for TikZ via \input{tikz_file}
            input_pattern = r'\\input\{([^}]+)\}'
            input_match = re.search(input_pattern, figure_content)
            if input_match:
                tikz_file = input_match.group(1).strip()
                # Check if it's likely a TikZ file (contains tikz, pgf, or is .tex)
                if 'tikz' in tikz_file.lower() or 'pgf' in tikz_file.lower() or tikz_file.endswith('.tex'):
                    return f"TIKZ:{tikz_file}"

            # Check for inline TikZ picture
            if re.search(r'\\begin\{tikzpicture\}', figure_content):
                return "TIKZ:inline"

            # Check for pgfplots
            if re.search(r'\\begin\{axis\}', figure_content) or re.search(r'\\begin\{tikzpicture\}', figure_content):
                return "TIKZ:inline"

    return None


def resolve_image_path(image_ref: str, source_dir: str, convert_pdf: bool = True) -> Tuple[Optional[str], str]:
    """Resolve an image reference from LaTeX to an actual file path.

    LaTeX \includegraphics often omits the extension, so we need to find
    the actual file. If a PDF is found and convert_pdf=True, converts it to PNG.

    Args:
        image_ref: Image reference from LaTeX (may lack extension).
        source_dir: Directory containing extracted arXiv source.
        convert_pdf: If True, convert PDF files to PNG.

    Returns:
        Tuple of (file_path, message). file_path is None if not found.
    """
    # Clean up the reference
    image_ref = image_ref.strip()

    # Get the base path without extension if it has one
    base_ref = image_ref
    ref_ext = None
    for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg', '.gif']:
        if image_ref.lower().endswith(ext):
            base_ref = image_ref[:-len(ext)]
            ref_ext = ext
            break

    # Extensions to try in order of preference (PNG first, then PDF for conversion)
    extensions_to_try = ['.png', '.pdf', '.jpg', '.jpeg', '']

    # Search for the file
    found_file = None
    found_ext = None

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_lower = file.lower()
            file_base = os.path.splitext(file)[0]
            file_ext = os.path.splitext(file)[1].lower()

            # Check if this file matches the reference
            for ext in extensions_to_try:
                # Try direct match with path
                if file_lower == (base_ref + ext).lower():
                    found_file = os.path.join(root, file)
                    found_ext = file_ext
                    break

                # Try matching just the filename (without subdirectory in ref)
                ref_basename = os.path.basename(base_ref)
                if file_base.lower() == ref_basename.lower() and (ext == '' or file_ext == ext):
                    found_file = os.path.join(root, file)
                    found_ext = file_ext
                    break

            if found_file:
                break
        if found_file:
            break

    if not found_file:
        return None, f"Could not find file matching '{image_ref}'"

    # If it's already a PNG, return it
    if found_ext == '.png':
        return found_file, "Found PNG"

    # If it's a PDF and we should convert
    if found_ext == '.pdf' and convert_pdf:
        png_path = convert_pdf_to_png(found_file)
        if png_path:
            return png_path, f"Converted PDF to PNG: {os.path.basename(png_path)}"
        else:
            return None, f"Found PDF '{os.path.basename(found_file)}' but conversion failed"

    # For other formats (jpg, jpeg), return as-is if acceptable
    if found_ext in ['.jpg', '.jpeg']:
        return found_file, f"Found {found_ext.upper()}"

    return None, f"Found '{os.path.basename(found_file)}' but format {found_ext} not supported"


def extract_figure_1_with_latex_parsing(source_dir: str) -> Tuple[bool, Optional[str], str]:
    """Extract Figure 1 by parsing LaTeX files.

    Args:
        source_dir: Directory containing extracted arXiv source.

    Returns:
        Tuple of (success, figure_1_path, message).
        For TikZ figures, returns (False, None, "Figure 1 is TikZ...") to indicate
        the caller should try arXiv HTML instead.
    """
    tex_files = find_tex_files(source_dir)

    if not tex_files:
        return False, None, "No .tex files found in source"

    # Try each tex file
    for tex_path in tex_files:
        try:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = f.read()
        except Exception as e:
            continue

        # Parse for Figure 1
        image_ref = parse_latex_for_figure_1(tex_content)

        if image_ref:
            # Check if it's a TikZ figure - these can't be extracted from source,
            # caller should use arXiv HTML which renders TikZ as SVG
            if image_ref.startswith("TIKZ:"):
                tikz_ref = image_ref[5:]  # Remove "TIKZ:" prefix
                if tikz_ref == "inline":
                    return False, None, f"Figure 1 is inline TikZ (use arXiv HTML for rendered version)"
                else:
                    return False, None, f"Figure 1 uses TikZ file '{tikz_ref}' (use arXiv HTML for rendered version)"

            # Try to resolve the image reference to an actual file (with PDF conversion)
            resolved_path, resolve_msg = resolve_image_path(image_ref, source_dir, convert_pdf=True)

            if resolved_path:
                return True, resolved_path, f"Found Figure 1 via LaTeX parsing: {resolve_msg}"
            else:
                # Found reference but couldn't resolve
                return False, None, f"Found Figure 1 reference '{image_ref}' but {resolve_msg}"

    return False, None, "Could not find Figure 1 in LaTeX files"


def extract_figure_1_with_llm(source_dir: str, model) -> Tuple[bool, Optional[str], str]:
    """Stage 2: Extract Figure 1 by using LLM to read .tex files.

    The LLM iteratively reads tex files starting with main.tex to find
    Figure 1 and identify its associated image file.

    Args:
        source_dir: Directory containing extracted arXiv source.
        model: LLM model to use for parsing.

    Returns:
        Tuple of (success, figure_1_path, message).
    """
    tex_files = find_tex_files(source_dir)
    png_files = find_png_files(source_dir)
    pdf_files = find_pdf_files(source_dir)

    if not tex_files:
        return False, None, "No .tex files found in source"

    # Combine PNG and PDF files for the prompt
    all_image_files = png_files + pdf_files
    if not all_image_files:
        return False, None, "No image files (PNG or PDF) found in source"

    # Create a summary of available image files
    image_basenames = [os.path.basename(p) for p in all_image_files]
    image_list_str = '\n'.join(f"  - {name}" for name in image_basenames)

    # Read tex files and ask LLM to identify Figure 1
    for tex_path in tex_files[:3]:  # Limit to first 3 tex files
        try:
            with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = f.read()
        except Exception:
            continue

        # Truncate if too long
        if len(tex_content) > 50000:
            tex_content = tex_content[:50000] + "\n... [truncated]"

        tex_filename = os.path.basename(tex_path)

        # Ask LLM to find Figure 1
        prompt = f"""Analyze this LaTeX file and identify the image file used for Figure 1.

Available image files in the source:
{image_list_str}

LaTeX file ({tex_filename}):
```latex
{tex_content}
```

Task: Find the \\begin{{figure}} environment that contains Figure 1 (usually the first figure, or one with \\label{{fig:1}} or similar).
Look for the \\includegraphics command inside that figure environment and identify which image file it references.

Important:
- Match the image reference to one of the available image files listed above
- The LaTeX reference might omit the file extension (.png, .pdf, etc.)
- If you cannot find Figure 1, respond with "NOT_FOUND"

Respond with ONLY the image filename (e.g., "figure1.png" or "figure1.pdf") or "NOT_FOUND". Nothing else."""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a LaTeX parsing assistant. Extract the requested information precisely."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        try:
            response = model(messages)
            response_text = response.strip()

            if response_text and response_text != "NOT_FOUND":
                # Remove any quotes or extra text
                response_clean = re.sub(r'["\']', '', response_text).strip()

                # Use resolve_image_path to find and potentially convert the file
                resolved_path, resolve_msg = resolve_image_path(response_clean, source_dir, convert_pdf=True)

                if resolved_path:
                    return True, resolved_path, f"Found Figure 1 via LLM: {resolve_msg}"

        except Exception as e:
            continue

    return False, None, "LLM could not identify Figure 1 image file"


def extract_figure_1_from_source(source_dir: str, arxiv_id: str,
                                  model=None) -> Tuple[bool, Optional[str], str]:
    """Extract Figure 1 PNG image from arXiv source files using 2-stage approach.

    Stage 1: Automatically parse LaTeX files to find Figure 1 and its image file
    Stage 2: If Stage 1 fails and model provided, use LLM to read .tex files

    Only returns PNG images. Returns None if no PNG for Figure 1 exists.
    If Figure 1 is TikZ-based, skips LLM stage since there's no image file to find.

    Args:
        source_dir: Directory containing extracted arXiv source.
        arxiv_id: The arXiv paper ID (for logging).
        model: Optional LLM model for Stage 2 fallback.

    Returns:
        Tuple of (success, figure_1_path, message).
        success=True and figure_1_path=None means Figure 1 doesn't have a PNG.
    """
    # Stage 1: Automatic LaTeX parsing
    success, fig_path, message = extract_figure_1_with_latex_parsing(source_dir)

    if success and fig_path:
        return True, fig_path, f"[Stage 1] {message}"

    # If Figure 1 is TikZ, don't bother with LLM - there's no image file to find
    # The caller should use arXiv HTML instead which renders TikZ as SVG
    if "TikZ" in message:
        return False, None, f"[Stage 1] {message}"

    # Stage 2: LLM-based extraction (if model provided)
    if model:
        success, fig_path, message = extract_figure_1_with_llm(source_dir, model)

        if success and fig_path:
            return True, fig_path, f"[Stage 2] {message}"

        # If LLM also failed, return the failure message
        return False, None, f"[Stage 2] {message}"

    # No model provided, return Stage 1 failure
    return False, None, f"[Stage 1] {message}"


# ============================================================================
# Section Extraction Functions (for World Model Detection)
# ============================================================================

def extract_section_headers_from_html(html_content: str) -> List[Dict[str, str]]:
    """Extract all section headers from arXiv HTML.

    Args:
        html_content: Raw HTML content from arXiv.

    Returns:
        List of dicts with 'level' (h2, h3, etc.), 'text', and 'index'.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html_content, 'html.parser')
    headers = []
    for i, header in enumerate(soup.find_all(['h2', 'h3', 'h4'])):
        headers.append({
            'level': header.name,
            'text': header.get_text(strip=True),
            'index': i
        })
    return headers


def extract_section_content_from_html(html_content: str, section_text: str) -> Optional[str]:
    """Extract content of a specific section from HTML.

    Finds the section header matching section_text, then collects all content
    until the next header at the same or higher level.

    Args:
        html_content: Raw HTML content from arXiv.
        section_text: Text to match in section header.

    Returns:
        Section content as plain text, or None if not found.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the target header
    target_header = None
    for header in soup.find_all(['h2', 'h3', 'h4']):
        if section_text.lower() in header.get_text(strip=True).lower():
            target_header = header
            break

    if not target_header:
        return None

    target_level = target_header.name  # e.g., 'h2', 'h3'

    # Strategy 1: Check if header is inside a <section> element
    parent_section = target_header.find_parent('section')
    if parent_section:
        # Get all text content from this section (paragraphs, divs, spans)
        content_parts = []
        for elem in parent_section.find_all(['p', 'div', 'span']):
            # Skip if this element contains another section or header
            if elem.find(['section', 'h1', 'h2', 'h3', 'h4']):
                continue
            # Skip elements that are just containers (have child block elements)
            if elem.find(['p', 'div']) and elem.name == 'div':
                continue
            text = elem.get_text(strip=True)
            if text and len(text) > 20 and text not in content_parts:
                content_parts.append(text)
        for ul in parent_section.find_all(['ul', 'ol']):
            for li in ul.find_all('li'):
                text = f"- {li.get_text(strip=True)}"
                if text not in content_parts:
                    content_parts.append(text)
        if content_parts:
            return '\n\n'.join(content_parts)

    # Strategy 2: Use find_all_next() to traverse all following elements
    content_parts = []
    for element in target_header.find_all_next():
        # Stop at next header of same or higher level
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            if element.name <= target_level and element != target_header:
                break
        # Collect paragraph content
        if element.name == 'p':
            text = element.get_text(strip=True)
            if text and len(text) > 20 and text not in content_parts:
                content_parts.append(text)
        # Collect div content that looks like a paragraph
        elif element.name == 'div' and not element.find(['p', 'div', 'section']):
            text = element.get_text(strip=True)
            if text and len(text) > 50 and text not in content_parts:
                content_parts.append(text)
        # Collect list items
        elif element.name == 'li':
            text = f"- {element.get_text(strip=True)}"
            if text not in content_parts:
                content_parts.append(text)

    return '\n\n'.join(content_parts) if content_parts else None


def find_related_work_section(headers: List[Dict], model=None) -> Optional[str]:
    """Find which header is the Related Work section.

    First tries pattern matching, then uses LLM if provided.

    Args:
        headers: List of section headers from extract_section_headers_from_html.
        model: Optional LLM model for fallback identification.

    Returns:
        The section header text, or None if not found.
    """
    # Pattern-based matching
    related_patterns = [
        'related work', 'related literature', 'prior work',
        'previous work', 'background', 'state of the art',
        'related works'
    ]

    for header in headers:
        header_lower = header['text'].lower()
        for pattern in related_patterns:
            if pattern in header_lower:
                return header['text']

    # LLM fallback if no pattern match
    if model and headers:
        header_list = '\n'.join([f"- {h['text']}" for h in headers])
        prompt = f"""Which of these section headers is about Related Work or Prior Work?

Section headers:
{header_list}

Return ONLY the exact section header text, or "NONE" if there isn't one."""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Identify the Related Work section."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        try:
            response = model(messages).strip()
            if response.upper() != "NONE":
                # Verify the response matches an actual header
                for header in headers:
                    if response in header['text'] or header['text'] in response:
                        return header['text']
        except Exception:
            pass

    return None


def extract_section_from_latex(tex_content: str, section_patterns: List[str]) -> Optional[str]:
    """Extract section content from LaTeX source.

    Args:
        tex_content: Raw LaTeX content.
        section_patterns: List of section name patterns to match (e.g., ['related work']).

    Returns:
        Section content as plain text, or None if not found.
    """
    # Remove comments first
    lines = tex_content.split('\n')
    cleaned_lines = []
    for line in lines:
        comment_idx = line.find('%')
        if comment_idx == 0:
            continue  # Skip full comment lines
        elif comment_idx > 0:
            # Check if % is escaped
            if line[comment_idx - 1] != '\\':
                line = line[:comment_idx]
        cleaned_lines.append(line)
    tex_content = '\n'.join(cleaned_lines)

    # Find section start for any of the patterns
    for pattern in section_patterns:
        # Match \section{...pattern...} or \section*{...pattern...}
        regex = rf'\\section\*?\{{([^}}]*{re.escape(pattern)}[^}}]*)\}}'
        match = re.search(regex, tex_content, re.IGNORECASE)

        if match:
            start_pos = match.end()

            # Find next section or end of document
            next_section = re.search(r'\\(?:section|chapter)\*?\{', tex_content[start_pos:])
            end_doc = re.search(r'\\end\{document\}', tex_content[start_pos:])

            if next_section:
                end_pos = start_pos + next_section.start()
            elif end_doc:
                end_pos = start_pos + end_doc.start()
            else:
                end_pos = len(tex_content)

            # Extract and clean content
            content = tex_content[start_pos:end_pos]

            # Remove common LaTeX commands while preserving text
            content = re.sub(r'\\cite[pt]?\{[^}]*\}', '', content)  # Citations
            content = re.sub(r'\\label\{[^}]*\}', '', content)  # Labels
            content = re.sub(r'\\ref\{[^}]*\}', '', content)  # References
            content = re.sub(r'\\textbf\{([^}]*)\}', r'\1', content)  # Bold
            content = re.sub(r'\\textit\{([^}]*)\}', r'\1', content)  # Italic
            content = re.sub(r'\\emph\{([^}]*)\}', r'\1', content)  # Emphasis
            content = re.sub(r'\\[a-z]+\{([^}]*)\}', r'\1', content)  # Other commands
            content = re.sub(r'\\[a-z]+', '', content)  # Commands without args
            content = re.sub(r'[{}]', '', content)  # Remaining braces
            content = re.sub(r'~', ' ', content)  # Non-breaking spaces
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

            return content.strip()

    return None


def detect_world_models_in_text(text: str, model) -> Tuple[bool, str]:
    """Use LLM to check if text contains meaningful world model mentions.

    Args:
        text: Text content from Related Work section.
        model: LLM model to use for analysis.

    Returns:
        Tuple of (has_world_models, explanation).
    """
    # Truncate to avoid token limits
    text_truncated = text[:8000] if len(text) > 8000 else text

    prompt = f"""Analyze this Related Work section from an academic paper.

Does it contain meaningful mentions of "world models" in the context of AI/ML?

Note: We're looking for actual discussions of world models as a concept,
not incidental mentions or different uses of the phrase.

Related Work text:
```
{text_truncated}
```

Answer with:
- YES if there are meaningful world model mentions
- NO if there are no mentions or only incidental ones

Then briefly explain your reasoning."""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You analyze academic papers for specific concept mentions."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    try:
        response = model(messages).strip()
        has_world_models = response.upper().startswith('YES')
        return has_world_models, response
    except Exception as e:
        return False, f"Error: {e}"


def detect_keyword_in_text(text: str, keyword: str, model) -> Tuple[bool, str]:
    """Use LLM to check if text contains meaningful mentions of a keyword.

    Args:
        text: Text content from Related Work section.
        keyword: The keyword/phrase to search for.
        model: LLM model to use for analysis.

    Returns:
        Tuple of (has_keyword, explanation).
    """
    # Truncate to avoid token limits
    text_truncated = text[:8000] if len(text) > 8000 else text

    prompt = f"""Analyze this Related Work section from an academic paper.

Does it contain meaningful mentions of "{keyword}"?

Note: We're looking for actual discussions of "{keyword}" as a concept or technique,
not incidental or passing mentions.

Related Work text:
```
{text_truncated}
```

Answer with:
- YES if there are meaningful mentions of "{keyword}"
- NO if there are no mentions or only incidental ones

Then briefly explain your reasoning."""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You analyze academic papers for specific concept mentions."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    try:
        response = model(messages).strip()
        has_keyword = response.upper().startswith('YES')
        return has_keyword, response
    except Exception as e:
        return False, f"Error: {e}"


def detect_chain_of_thought_in_text(text: str, model) -> Tuple[bool, str]:
    """Use LLM to check if text contains meaningful chain-of-thought mentions.

    Convenience wrapper around detect_keyword_in_text for backward compatibility.
    """
    return detect_keyword_in_text(text, "chain-of-thought", model)


def detect_keyword_simple(text: str, keyword: str) -> bool:
    """Simple regex-based keyword detection in text.

    Args:
        text: Text to check for keyword mentions.
        keyword: The keyword/phrase to search for.

    Returns:
        True if keyword is mentioned.
    """
    if not text or not keyword:
        return False

    # Escape the keyword for regex, but allow hyphens/spaces to be interchangeable
    keyword_pattern = re.escape(keyword).replace(r'\-', r'[- ]').replace(r'\ ', r'[- ]')
    return bool(re.search(keyword_pattern, text, re.IGNORECASE))


def detect_chain_of_thought_simple(text: str) -> bool:
    """Simple regex-based chain-of-thought detection.

    Convenience wrapper around detect_keyword_simple for backward compatibility.
    """
    return detect_keyword_simple(text, "chain-of-thought")


# ============================================================================
# Preprocessing Helper Functions
# ============================================================================

_shared_arxiv_client = None

def _get_arxiv_client():
    """Get or create a shared arxiv client with proper rate limiting."""
    global _shared_arxiv_client
    if _shared_arxiv_client is None:
        import arxiv
        _shared_arxiv_client = arxiv.Client(
            page_size=30,      # arXiv API recommends max 30 per page
            delay_seconds=5,   # Conservative: 5s between requests
            num_retries=5,
        )
    return _shared_arxiv_client


def _arxiv_query_with_retry(search, max_retries: int = 3) -> list:
    """Execute an arxiv search with retry and exponential backoff on 429s.

    Args:
        search: An arxiv.Search object.
        max_retries: Maximum number of retry attempts.

    Returns:
        List of arxiv.Result objects.
    """
    import time

    client = _get_arxiv_client()

    for attempt in range(max_retries + 1):
        try:
            results = list(client.results(search))
            return results
        except Exception as e:
            error_str = str(e)
            if '429' in error_str and attempt < max_retries:
                wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
                print(f"      arXiv rate limited (429), waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise

    return []


def _parse_arxiv_result(result) -> Dict:
    """Convert an arxiv.Result to our standard paper dict."""
    raw_id = result.entry_id.split('/')[-1]
    arxiv_id = re.sub(r'v\d+$', '', raw_id)
    return {
        'arxiv_id': arxiv_id,
        'title': result.title,
        'authors': [str(a) for a in result.authors],
        'abstract': result.summary,
        'pdf_url': result.pdf_url,
    }


def fetch_arxiv_batch(arxiv_ids: List[str]) -> List[Dict]:
    """Fetch metadata for multiple papers by arXiv ID in a single API call.

    Uses the arxiv id_list parameter to batch-fetch papers, avoiding
    per-paper API calls.

    Args:
        arxiv_ids: List of arXiv IDs (e.g., ["2301.12345", "2405.00357"]).

    Returns:
        List of paper dicts with metadata.
    """
    if not arxiv_ids:
        return []

    try:
        import arxiv

        search = arxiv.Search(id_list=arxiv_ids)
        results = _arxiv_query_with_retry(search)

        papers = [_parse_arxiv_result(r) for r in results]
        print(f"      Batch-fetched {len(papers)}/{len(arxiv_ids)} papers from arXiv")
        return papers

    except ImportError:
        print("Warning: arxiv package not installed")
        return []
    except Exception as e:
        print(f"Error batch-fetching from arXiv: {e}")
        return []


def search_arxiv_by_title(title: str, max_results: int = 5) -> Optional[Dict]:
    """Search arXiv for a paper by title.

    Args:
        title: Paper title to search for.
        max_results: Maximum number of results to return.

    Returns:
        Dict with paper info if found, None otherwise.
    """
    try:
        import arxiv

        search = arxiv.Search(
            query=f'ti:"{title}"',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = _arxiv_query_with_retry(search)

        if not results:
            return None

        # Find best match by title similarity
        best_match = None
        best_score = 0

        for result in results:
            is_match, score = fuzzy_match_text(title, result.title, threshold=80)
            if score > best_score:
                best_score = score
                best_match = result

        if best_match and best_score >= 80:
            paper = _parse_arxiv_result(best_match)
            paper['match_score'] = best_score
            return paper

        return None

    except ImportError:
        print("Warning: arxiv package not installed")
        return None
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return None


def search_arxiv_by_author(author_name: str, max_results: int = 50, first_author_only: bool = False) -> List[Dict]:
    """Search arXiv for all papers by an author name.

    Uses the arXiv API with `au:"Author Name"` query syntax.

    When first_author_only is True, fetches up to 500 results from arXiv
    (to get a comprehensive list) and then filters to papers where the
    searched author is first author. The final list is capped at max_results.

    Args:
        author_name: The author's name to search for.
        max_results: Maximum number of results to return after filtering.
        first_author_only: If True, only return papers where this author is
            listed first. Useful for prolific authors to reduce noise.

    Returns:
        List of dicts with paper info (arxiv_id, title, authors, abstract).
    """
    try:
        import arxiv

        search = arxiv.Search(
            query=f'au:"{author_name}"',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = _arxiv_query_with_retry(search)
        papers = [_parse_arxiv_result(r) for r in results]

        if first_author_only:
            author_norm = normalize_author_name(author_name)
            filtered = [p for p in papers if normalize_author_name(p.get('authors', [''])[0]) == author_norm]
            print(f"      Filtered to {len(filtered)}/{len(papers)} first-author papers")
            return filtered[:max_results]

        return papers

    except ImportError:
        print("Warning: arxiv package not installed")
        return []
    except Exception as e:
        print(f"Error searching arXiv by author: {e}")
        return []


def normalize_title_for_matching(title: str) -> str:
    """Normalize a title for comparison.

    - Converts to lowercase
    - Removes punctuation
    - Normalizes whitespace

    Args:
        title: The title to normalize.

    Returns:
        Normalized title string.
    """
    if not title:
        return ""

    # Convert to lowercase
    title = title.lower()

    # Remove common punctuation
    title = re.sub(r'[^\w\s]', ' ', title)

    # Normalize whitespace
    title = ' '.join(title.split())

    return title


def match_gscholar_to_arxiv_papers(
    gscholar_papers: List[Dict],
    arxiv_papers: List[Dict],
    model=None
) -> List[Dict]:
    """Match Google Scholar papers to arXiv papers using multi-stage matching.

    Matching strategy (in order):
    1. Direct URL extraction - Check if eprint_url or pub_url contains arXiv ID
    2. Exact title match - After normalization
    3. Fuzzy title match - Using fuzzy_match_text() with threshold 80
    4. LLM semantic match - For remaining unmatched papers (if model provided)

    Args:
        gscholar_papers: List of papers from Google Scholar scraping.
            Each dict should have: 'title', 'eprint_url', 'pub_url'
        arxiv_papers: List of papers from arXiv author search.
            Each dict should have: 'arxiv_id', 'title', 'authors', 'abstract'
        model: Optional LLM model for semantic matching fallback.

    Returns:
        List of matched arXiv paper dicts with additional 'gscholar_title' field.
    """
    matched_papers = []
    matched_arxiv_ids = set()
    unmatched_gscholar = []

    # Build lookup structures for arXiv papers
    arxiv_by_id = {p['arxiv_id']: p for p in arxiv_papers}
    arxiv_by_normalized_title = {}
    for p in arxiv_papers:
        norm_title = normalize_title_for_matching(p['title'])
        if norm_title:
            arxiv_by_normalized_title[norm_title] = p

    print(f"    Matching {len(gscholar_papers)} GScholar papers against {len(arxiv_papers)} arXiv papers...")

    for gs_paper in gscholar_papers:
        gs_title = gs_paper.get('title', '')
        eprint_url = gs_paper.get('eprint_url', '')
        pub_url = gs_paper.get('pub_url', '')

        matched_arxiv = None
        match_method = None

        # Strategy 1: Direct URL extraction
        for url in [eprint_url, pub_url]:
            arxiv_id = extract_arxiv_id_from_url(url)
            if arxiv_id and arxiv_id in arxiv_by_id:
                matched_arxiv = arxiv_by_id[arxiv_id]
                match_method = "direct_url"
                break

        # Strategy 2: Exact title match (normalized)
        if not matched_arxiv:
            gs_norm_title = normalize_title_for_matching(gs_title)
            if gs_norm_title in arxiv_by_normalized_title:
                matched_arxiv = arxiv_by_normalized_title[gs_norm_title]
                match_method = "exact_title"

        # Strategy 3: Fuzzy title match
        if not matched_arxiv:
            best_match = None
            best_score = 0
            for arxiv_paper in arxiv_papers:
                if arxiv_paper['arxiv_id'] in matched_arxiv_ids:
                    continue
                is_match, score = fuzzy_match_text(gs_title, arxiv_paper['title'], threshold=80)
                if is_match and score > best_score:
                    best_score = score
                    best_match = arxiv_paper

            if best_match:
                matched_arxiv = best_match
                match_method = f"fuzzy_title (score={best_score})"

        # If matched, add to results
        if matched_arxiv and matched_arxiv['arxiv_id'] not in matched_arxiv_ids:
            matched_arxiv_ids.add(matched_arxiv['arxiv_id'])
            result = matched_arxiv.copy()
            result['gscholar_title'] = gs_title
            result['match_method'] = match_method
            matched_papers.append(result)
            print(f"      Matched: '{gs_title[:50]}...' -> {matched_arxiv['arxiv_id']} ({match_method})")
        else:
            unmatched_gscholar.append(gs_paper)

    # Strategy 4: LLM semantic match for remaining arXiv papers
    # Iterate through remaining arXiv papers and check if each matches a GScholar entry
    # Note: Multiple arXiv papers can match the same GScholar entry (e.g., different versions)
    remaining_arxiv = [p for p in arxiv_papers if p['arxiv_id'] not in matched_arxiv_ids]

    if model and remaining_arxiv and gscholar_papers:
        print(f"    Using LLM for {len(remaining_arxiv)} unmatched arXiv papers...")

        # Build list of ALL GScholar titles (not just unmatched) - multiple arXiv can map to same GScholar
        gs_titles_list = [f"[{i}] {p.get('title', '')}" for i, p in enumerate(gscholar_papers)]
        gs_list_str = '\n'.join(gs_titles_list)

        for arxiv_paper in remaining_arxiv:
            arxiv_id = arxiv_paper['arxiv_id']
            arxiv_title = arxiv_paper['title']

            prompt = f"""Does this arXiv paper match any of the Google Scholar entries below?

arXiv paper: [{arxiv_id}] "{arxiv_title}"

Google Scholar entries:
{gs_list_str}

IMPORTANT: Many arXiv papers will NOT have a matching Google Scholar entry. Only match if you are confident it's the same paper (possibly with a restructured title due to revisions). There should be some clear overlap in title wording for a match.

If you find a match, return ONLY the index number (e.g., "3").
If there is NO match (which is common and expected), return "NO_MATCH".

Your answer:"""

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You match paper titles. Return only the index number or NO_MATCH. Most papers will not have a match."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]

            try:
                response = model(messages).strip()
                # Clean response - extract index number
                idx_match = re.search(r'^(\d+)$', response)
                if idx_match:
                    gs_idx = int(idx_match.group(1))
                    if gs_idx < len(gscholar_papers):
                        matched_arxiv_ids.add(arxiv_id)
                        gs_paper = gscholar_papers[gs_idx]
                        gs_title = gs_paper.get('title', '')
                        result = arxiv_paper.copy()
                        result['gscholar_title'] = gs_title
                        result['match_method'] = "llm_semantic"
                        matched_papers.append(result)
                        print(f"      LLM matched: {arxiv_id} -> '{gs_title[:50]}...'")
            except Exception as e:
                print(f"      LLM error for {arxiv_id}: {e}")

    print(f"    Total matched: {len(matched_papers)} papers")
    return matched_papers
