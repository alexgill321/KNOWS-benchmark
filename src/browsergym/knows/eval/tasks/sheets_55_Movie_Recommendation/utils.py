"""Task-specific utilities for sheets_55_Movie_Recommendation.

Uses Playwright to fetch IMDb pages (IMDb blocks simple HTTP requests)
and parses structured JSON-LD data for programmatic verification.
LLM is only used for Oscar award verification where structured data
is not available.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import html2text
from playwright.sync_api import sync_playwright, Browser

from src.browsergym.knows.eval.eval_utils.llm_utils import parse_yes_no

# ---------------------------------------------------------------------------
# IMDb URLs
# ---------------------------------------------------------------------------

IMDB_SEARCH_URL = "https://www.imdb.com/find/?q={query}&s=tt&ttype=ft"
IMDB_TITLE_URL = "https://www.imdb.com/title/{imdb_id}/"
IMDB_AWARDS_URL = "https://www.imdb.com/title/{imdb_id}/awards/"


# ---------------------------------------------------------------------------
# Playwright browser management
# ---------------------------------------------------------------------------

_playwright_instance = None
_browser: Optional[Browser] = None


def get_browser() -> Browser:
    """Get or create a shared Playwright browser instance."""
    global _playwright_instance, _browser
    if _browser is None:
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=True)
    return _browser


def close_browser():
    """Close the shared Playwright browser instance."""
    global _playwright_instance, _browser
    if _browser:
        _browser.close()
        _browser = None
    if _playwright_instance:
        _playwright_instance.stop()
        _playwright_instance = None


def _fetch_page_html_once(url: str, timeout: int = 15000) -> Optional[str]:
    """Single attempt to fetch a page's rendered HTML using Playwright.

    Args:
        url: URL to fetch.
        timeout: Navigation timeout in milliseconds.

    Returns:
        Rendered HTML string, or None if fetch failed.
    """
    context = None
    try:
        browser = get_browser()
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except Exception:
            try:
                page.goto(url, wait_until="load", timeout=timeout)
            except Exception:
                return None
        page.wait_for_timeout(2000)
        try:
            html = page.content()
        except Exception:
            page.wait_for_timeout(3000)
            html = page.content()
        return html
    except Exception as e:
        print(f"[IMDb] Playwright fetch failed for {url}: {e}")
        return None
    finally:
        if context:
            try:
                context.close()
            except Exception:
                pass


def _fetch_page_html(url: str, timeout: int = 15000, max_retries: int = 3) -> Optional[str]:
    """Fetch a page's rendered HTML with retry and exponential backoff.

    Args:
        url: URL to fetch.
        timeout: Navigation timeout in milliseconds per attempt.
        max_retries: Maximum number of attempts (default 3).

    Returns:
        Rendered HTML string, or None if all attempts failed.
    """
    for attempt in range(1, max_retries + 1):
        html = _fetch_page_html_once(url, timeout=timeout)
        if html:
            return html
        if attempt < max_retries:
            backoff = 2 ** attempt  # 2s, 4s
            print(f"[IMDb] Attempt {attempt}/{max_retries} failed for {url}, retrying in {backoff}s...")
            time.sleep(backoff)
    print(f"[IMDb] All {max_retries} attempts failed for {url}")
    return None


# ---------------------------------------------------------------------------
# IMDb data extraction
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    """Normalize movie titles for comparison."""
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title


def _extract_imdb_id_from_search(movie_title: str, year: Optional[str] = None, max_retries: int = 3) -> Optional[str]:
    """Search IMDb via Playwright and extract the best-matching title ID.

    Retries if the page loads but search results haven't rendered yet
    (transient JS rendering issue).

    Args:
        movie_title: Title of the movie.
        year: Optional release year to narrow the search.
        max_retries: Maximum search attempts (default 3).

    Returns:
        IMDb title ID string (e.g. 'tt1375666'), or None if not found.
    """
    query = movie_title.replace(" ", "+")
    if year:
        query += f"+{year}"
    search_url = IMDB_SEARCH_URL.format(query=query)

    for attempt in range(1, max_retries + 1):
        html = _fetch_page_html_once(search_url)
        if not html:
            if attempt < max_retries:
                backoff = 2 ** attempt
                print(f"[IMDb] Search attempt {attempt}/{max_retries} returned no HTML for '{movie_title}', retrying in {backoff}s...")
                time.sleep(backoff)
            continue

        # Extract all title links from search results
        matches = re.findall(r'/title/(tt\d+)/', html)
        if matches:
            # Deduplicate while preserving order (first result is usually best)
            seen = set()
            unique_ids = []
            for imdb_id in matches:
                if imdb_id not in seen:
                    seen.add(imdb_id)
                    unique_ids.append(imdb_id)
            imdb_id = unique_ids[0]
            print(f"[IMDb] Search for '{movie_title}' ({year}) -> {imdb_id}")
            return imdb_id

        # HTML returned but no title IDs — JS may not have rendered search results
        if attempt < max_retries:
            backoff = 2 ** attempt
            print(f"[IMDb] Search attempt {attempt}/{max_retries} got HTML but no results for '{movie_title}', retrying in {backoff}s...")
            time.sleep(backoff)

    print(f"[IMDb] All {max_retries} search attempts failed for '{movie_title}'")
    return None


def _parse_json_ld(html: str) -> Optional[Dict]:
    """Extract JSON-LD structured data from an IMDb page.

    Args:
        html: Full HTML content of the page.

    Returns:
        Parsed JSON-LD dict with movie metadata, or None if not found.
    """
    match = re.search(
        r'<script type="application/ld\+json">(.*?)</script>',
        html, re.DOTALL,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def fetch_imdb_data(movie_title: str, year: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch structured IMDb data for a movie.

    Searches IMDb, fetches the title page, and extracts JSON-LD metadata.
    Validates that the fetched page matches the requested movie.

    Args:
        movie_title: Title of the movie.
        year: Optional release year.

    Returns:
        Tuple of (movie_data dict, imdb_id). movie_data contains keys like:
        - name: Movie title
        - genre: List of genre strings
        - contentRating: MPA rating (e.g. "PG-13")
        - aggregateRating.ratingValue: IMDb score as float
        - datePublished: Release date string
        - duration: ISO 8601 duration (e.g. "PT2H28M")
        Returns (None, imdb_id) if data extraction or validation fails.
    """
    imdb_id = _extract_imdb_id_from_search(movie_title, year)
    if not imdb_id:
        print(f"[IMDb] No search result for '{movie_title}' ({year})")
        return None, None

    title_url = IMDB_TITLE_URL.format(imdb_id=imdb_id)
    print(f"[IMDb] Fetching: {title_url}")
    html = _fetch_page_html(title_url)
    if not html:
        print(f"[IMDb] Failed to fetch page for '{movie_title}'")
        return None, imdb_id

    data = _parse_json_ld(html)
    if not data or data.get("@type") != "Movie":
        print(f"[IMDb] No valid JSON-LD Movie data for '{movie_title}'")
        return None, imdb_id

    # Validate title match — check JSON-LD name and also the full page HTML
    # (handles foreign-language titles like Parasite/Gisaengchung)
    page_title = data.get("name", "")
    norm_search = _normalize_title(movie_title)
    norm_page_title = _normalize_title(page_title)
    title_in_html = norm_search in _normalize_title(html[:15000])
    if norm_search not in norm_page_title and norm_page_title not in norm_search and not title_in_html:
        print(f"[IMDb] Title mismatch: searched '{movie_title}', got '{page_title}'")
        return None, imdb_id

    # Validate year if provided
    if year:
        date_published = data.get("datePublished", "")
        if str(year) not in date_published:
            print(f"[IMDb] Year mismatch: expected {year}, got '{date_published}'")
            return None, imdb_id

    print(f"[IMDb] Got data for '{data.get('name')}' "
          f"(genre={data.get('genre')}, rating={data.get('contentRating')}, "
          f"imdb={data.get('aggregateRating', {}).get('ratingValue')})")
    return data, imdb_id


def fetch_imdb_awards_text(imdb_id: str) -> Optional[str]:
    """Fetch the IMDb awards page and return as plain text.

    Args:
        imdb_id: IMDb title ID (e.g. 'tt1375666').

    Returns:
        Plain text content of the awards page, or None if fetch failed.
    """
    if not imdb_id:
        return None

    awards_url = IMDB_AWARDS_URL.format(imdb_id=imdb_id)
    print(f"[IMDb] Fetching awards: {awards_url}")
    html = _fetch_page_html(awards_url)
    if not html:
        return None

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    text = h.handle(html)

    # Truncate to reasonable size
    if len(text) > 30000:
        text = text[:30000]
    return text


# ---------------------------------------------------------------------------
# Programmatic verification (no LLM needed)
# ---------------------------------------------------------------------------

def verify_genre(imdb_data: Optional[Dict], preferred_genres: List[str]) -> Optional[bool]:
    """Check if a movie belongs to at least one preferred genre.

    Args:
        imdb_data: Parsed JSON-LD data from IMDb.
        preferred_genres: List of preferred genre names.

    Returns:
        True if match found, False if no match, None if data unavailable.
    """
    if not imdb_data:
        return None

    imdb_genres = imdb_data.get("genre", [])
    if isinstance(imdb_genres, str):
        imdb_genres = [imdb_genres]
    if not imdb_genres:
        return None

    imdb_lower = {g.lower() for g in imdb_genres}
    preferred_lower = {g.lower() for g in preferred_genres}

    return bool(imdb_lower & preferred_lower)


def verify_imdb_score(imdb_data: Optional[Dict], sheet_score: float, tolerance: float = 0.1) -> Optional[bool]:
    """Check if the sheet's IMDb score matches the actual score.

    Args:
        imdb_data: Parsed JSON-LD data from IMDb.
        sheet_score: Score value from the spreadsheet.
        tolerance: Allowed deviation (default +/- 0.1).

    Returns:
        True if within tolerance, False if not, None if data unavailable.
    """
    if not imdb_data:
        return None

    rating_obj = imdb_data.get("aggregateRating", {})
    actual_score = rating_obj.get("ratingValue")
    if actual_score is None:
        return None

    try:
        actual = float(actual_score)
    except (ValueError, TypeError):
        return None

    return abs(sheet_score - actual) <= tolerance


def verify_mpa_rating(imdb_data: Optional[Dict], sheet_rating: str) -> Optional[bool]:
    """Check if the sheet's MPA rating matches the actual content rating.

    Args:
        imdb_data: Parsed JSON-LD data from IMDb.
        sheet_rating: MPA rating from the spreadsheet (e.g. "PG-13").

    Returns:
        True if match, False if mismatch, None if data unavailable.
    """
    if not imdb_data:
        return None

    actual_rating = imdb_data.get("contentRating")
    if not actual_rating:
        return None

    return sheet_rating.strip().upper() == actual_rating.strip().upper()


# ---------------------------------------------------------------------------
# LLM-based verification (only for Oscar awards — unstructured data)
# ---------------------------------------------------------------------------

def verify_oscar_awards(
    model: Any,
    movie_title: str,
    qualifying_oscars: List[str],
    awards_text: Optional[str],
) -> Optional[bool]:
    """Verify that a movie won at least one qualifying Oscar.

    Uses LLM to interpret the unstructured awards page text, since Oscar
    category data is not available in structured form from IMDb.

    Args:
        model: LLM model callable.
        movie_title: Title of the movie.
        qualifying_oscars: List of qualifying Oscar category names.
        awards_text: Plain text from the IMDb awards page.

    Returns:
        True if verified, False if not, None if unable to determine.
    """
    if not awards_text:
        return None

    oscar_str = ", ".join(qualifying_oscars)
    system_msg = (
        "You are a movie awards verifier. You are given text scraped from an IMDb "
        "awards page. Base your answer ONLY on the provided text. If the information "
        "is not explicitly stated in the text, answer Unsure. Do NOT use any prior "
        "knowledge. Answer Yes, No, or Unsure."
    )
    user_msg = (
        f"Has '{movie_title}' WON (not just nominated for) at least one of these "
        f"Oscar/Academy Award categories: {oscar_str}?\n\n"
        f"IMDb Awards page text:\n{awards_text[:15000]}\n\n"
        f"Answer Yes, No, or Unsure."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
    ]
    response = str(model(messages)).strip()
    return parse_yes_no(response)
