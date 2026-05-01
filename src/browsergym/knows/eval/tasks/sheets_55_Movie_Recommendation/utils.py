"""Task-specific utilities for sheets_55_Movie_Recommendation.

Uses Playwright to fetch IMDb pages (IMDb blocks simple HTTP requests)
and parses structured JSON-LD data for programmatic verification.
LLM is only used for Oscar award verification where structured data
is not available.
"""

import json
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import html2text
import requests
from playwright.sync_api import sync_playwright, Browser

from src.browsergym.knows.eval.eval_utils.llm_utils import parse_yes_no

# ---------------------------------------------------------------------------
# IMDb URLs
# ---------------------------------------------------------------------------

IMDB_SEARCH_URL = "https://www.imdb.com/find/?q={query}&s=tt&ttype=ft"
IMDB_TITLE_URL = "https://www.imdb.com/title/{imdb_id}/"
IMDB_AWARDS_URL = "https://www.imdb.com/title/{imdb_id}/awards/"
IMDB_SUGGEST_URL = "https://v2.sg.media-imdb.com/suggestion/{first_char}/{query}.json"


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


def _titles_match(search_norm: str, candidate_norm: str) -> bool:
    """Compare two already-normalized titles. Exact match always passes;
    substring match is allowed only when both sides are 4+ chars to avoid
    short-title false positives (e.g., "Up", "It", "M")."""
    if not search_norm or not candidate_norm:
        return False
    if search_norm == candidate_norm:
        return True
    if len(search_norm) >= 4 and len(candidate_norm) >= 4:
        return search_norm in candidate_norm or candidate_norm in search_norm
    return False


def _suggest_imdb_ids(movie_title: str, year: Optional[str] = None) -> List[str]:
    """Look up IMDb tt ID candidates via the suggest autocomplete endpoint.

    Filters results by content type (movies only), title match, and year.
    Returns all matching candidates sorted by rank (most popular first).

    Args:
        movie_title: Title of the movie.
        year: Optional release year to disambiguate remakes.

    Returns:
        List of IMDb title ID strings (e.g. ['tt0120338']), most likely match first.
        Empty list if no candidates match.
    """
    target_norm = re.sub(r'[^\w\s]', '', movie_title.lower()).strip()
    if not target_norm:
        return []

    first_char = target_norm[0]
    query = urllib.parse.quote(target_norm.replace(' ', '_'))
    url = IMDB_SUGGEST_URL.format(first_char=first_char, query=query)

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        entries = resp.json().get("d", [])
    except (requests.RequestException, ValueError):
        return []

    candidates = []
    for entry in entries:
        # Movies only — drop TV shows, video games, music videos, etc.
        if entry.get("q") != "feature" and entry.get("qid") != "movie":
            continue
        # Title match (normalized exact, or accept full title with subtitle)
        entry_norm = re.sub(r'[^\w\s]', '', entry.get("l", "").lower()).strip()
        if target_norm == entry_norm or entry_norm.startswith(target_norm + " ") or target_norm.startswith(entry_norm + " "):
            candidates.append(entry)

    if not candidates:
        return []

    # Lowest rank = most popular = most likely the right one when multiple match
    candidates.sort(key=lambda e: e.get("rank", 1e9))
    ids = [c["id"] for c in candidates if c.get("id")]
    if ids:
        print(f"[IMDb] Suggest API matches for '{movie_title}' ({year}) -> {ids}")
    return ids


def _extract_imdb_ids_from_search(movie_title: str, year: Optional[str] = None, max_retries: int = 3) -> List[str]:
    """Search IMDb via Playwright and return candidate title IDs.

    Retries if the page loads but search results haven't rendered yet
    (transient JS rendering issue).

    Args:
        movie_title: Title of the movie.
        year: Optional release year to narrow the search.
        max_retries: Maximum search attempts (default 3).

    Returns:
        List of unique IMDb title IDs in search-result order (most relevant first).
        Empty list if all search attempts failed.
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
            # Deduplicate while preserving order (first result is usually most relevant)
            seen = set()
            unique_ids = []
            for imdb_id in matches:
                if imdb_id not in seen:
                    seen.add(imdb_id)
                    unique_ids.append(imdb_id)
            print(f"[IMDb] Search for '{movie_title}' ({year}) -> {unique_ids}")
            return unique_ids

        # HTML returned but no title IDs — JS may not have rendered search results
        if attempt < max_retries:
            backoff = 2 ** attempt
            print(f"[IMDb] Search attempt {attempt}/{max_retries} got HTML but no results for '{movie_title}', retrying in {backoff}s...")
            time.sleep(backoff)

    print(f"[IMDb] All {max_retries} search attempts failed for '{movie_title}'")
    return []


def _parse_json_ld(html: str) -> Optional[Dict]:
    """Extract JSON-LD structured data from an IMDb page.

    Args:
        html: Full HTML content of the page.

    Returns:
        Parsed JSON-LD dict with movie metadata, or None if not found.
    """
    matches = re.findall(
        r'<script type="application/ld\+json">(.*?)</script>',
        html, re.DOTALL,
    )
    if not matches:
        return None
    try:
        for tag in matches:
            data = json.loads(tag)
            if isinstance(data, dict) and data.get("@type") == "Movie":
                return data
            
        return None
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
    # Try the suggest endpoint first — fast, structured, can match by year/type.
    # Fall back to Playwright search only if suggest returns no candidates.
    candidates = _suggest_imdb_ids(movie_title, year)
    if not candidates:
        candidates = _extract_imdb_ids_from_search(movie_title, year)
    if not candidates:
        print(f"[IMDb] No search result for '{movie_title}' ({year})")
        return None, None

    norm_search = _normalize_title(movie_title)
    last_imdb_id = candidates[0]

    # Try each candidate; return the first one that passes title + year validation.
    for imdb_id in candidates:
        last_imdb_id = imdb_id
        title_url = IMDB_TITLE_URL.format(imdb_id=imdb_id)
        print(f"[IMDb] Fetching: {title_url}")
        html = _fetch_page_html(title_url)
        if not html:
            print(f"[IMDb] Failed to fetch page for '{movie_title}' ({imdb_id})")
            continue

        data = _parse_json_ld(html)
        if not data or data.get("@type") != "Movie":
            print(f"[IMDb] No valid JSON-LD Movie data for '{movie_title}' ({imdb_id})")
            continue

        # Validate title match — check JSON-LD name and alternateName
        # (handles foreign-language titles like Parasite/Gisaengchung)
        page_title = data.get("name", "")
        alt_names = data.get("alternateName", [])
        if isinstance(alt_names, str):
            alt_names = [alt_names]
        norm_page_title = _normalize_title(page_title)
        norm_alts = [_normalize_title(a) for a in alt_names if a]

        if not (
            _titles_match(norm_search, norm_page_title)
            or any(_titles_match(norm_search, a) for a in norm_alts)
        ):
            print(f"[IMDb] Title mismatch ({imdb_id}): searched '{movie_title}', got '{page_title}' (alt={alt_names})")
            continue

        # Validate year if provided
        if year:
            year_published = data.get("datePublished", "")[:4]
            year_str = str(year)[:4]
            if year_str != year_published:
                print(f"[IMDb] Year mismatch ({imdb_id}): expected {year_str}, got '{year_published}'")
                continue

        print(f"[IMDb] Got data for '{data.get('name')}' ({imdb_id}) "
              f"(genre={data.get('genre')}, rating={data.get('contentRating')}, "
              f"imdb={data.get('aggregateRating', {}).get('ratingValue')})")
        return data, imdb_id

    print(f"[IMDb] No candidate matched for '{movie_title}' ({year}) after trying {len(candidates[:5])} candidate(s)")
    return None, last_imdb_id


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

def get_primary_imdb_genre(imdb_data: Optional[Dict]) -> Optional[str]:
    """Return a movie's primary (first-listed) IMDb genre, lowercased.

    Handles both list and single-string forms, and flattens comma-separated
    strings (some IMDb pages serialize genres as one "Action, Drama" string
    instead of a proper list).

    Args:
        imdb_data: Parsed JSON-LD data from IMDb.

    Returns:
        Lowercased primary genre, or None if data missing or genre list empty.
    """
    if not imdb_data:
        return None
    imdb_genres = imdb_data.get("genre", [])
    if isinstance(imdb_genres, str):
        imdb_genres = [imdb_genres]
    flat = []
    for g in imdb_genres:
        flat.extend(s.strip() for s in str(g).split(",") if s.strip())
    return flat[0].lower() if flat else None


def verify_genre(imdb_data: Optional[Dict], preferred_genres: List[str]) -> Optional[bool]:
    """Check if a movie's primary IMDb genre matches a preferred genre.

    Args:
        imdb_data: Parsed JSON-LD data from IMDb.
        preferred_genres: List of preferred genre names.

    Returns:
        True if match found, False if no match, None if data unavailable.
    """
    first = get_primary_imdb_genre(imdb_data)
    if first is None:
        return None

    # Word-boundary match so compound forms like "Dark Comedy" pass for
    # "Comedy" but "Farce" does not.
    for pref in preferred_genres:
        if re.search(rf"\b{re.escape(pref.lower())}\b", first):
            return True
    return False


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

def _normalize_mpa_rating(rating: Optional[str]) -> Optional[str]:
    """Normalize MPA rating strings for comparison."""
    if not rating or not isinstance(rating, str):
        return "UNRATED"
    
    r = str(rating).upper().replace("-", "").replace(" ", "")
    r = r.split(":")[-1] # Take only what's after the colon (e.g., US:PG13 -> PG13)

    # 2. Map to a "Canonical" form
    mapping = {
        "PG13": "PG-13",
        "TV14": "PG-13",   # TV equivalent
        "TVMA": "R",       # TV equivalent
        "NC17": "NC-17",
        "APPROVED": "G",   # Older movies
        "PASSED": "G",
        "NOTRATED": "UNRATED",
        "NR": "UNRATED"
    }
    
    return mapping.get(r, r)

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

    return _normalize_mpa_rating(sheet_rating) == _normalize_mpa_rating(actual_rating)


# ---------------------------------------------------------------------------
# LLM-based verification (only for Oscar awards — unstructured data)
# ---------------------------------------------------------------------------

# Legacy: superseded by extract_qualifying_oscars_won. Kept for backward
# compatibility; no current call sites in this repo.
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


def extract_qualifying_oscars_won(
    model: Any,
    movie_title: str,
    qualifying_oscars: List[str],
    awards_text: Optional[str],
) -> Optional[set]:
    """Extract which qualifying Oscars a movie actually won.

    Args:
        model: LLM model callable.
        movie_title: Title of the movie.
        qualifying_oscars: List of qualifying Oscar category names.
        awards_text: Plain text from the IMDb awards page.

    Returns:
        Set of canonical category names (from qualifying_oscars) that the movie
        won according to the awards text. Empty set if it won none of them.
        None if the awards text is unavailable.
    """
    if not awards_text:
        return None

    oscar_str = "\n".join(f"- {o}" for o in qualifying_oscars)
    system_msg = (
        "You are a movie awards verifier. Base your answer ONLY on the provided "
        "awards text. Do NOT use any prior knowledge."
    )
    user_msg = (
        f"From the categories below, list all that '{movie_title}' WON (not just "
        f"nominated for) according to the IMDb awards text.\n\n"
        f"Categories:\n{oscar_str}\n\n"
        f"Awards text:\n{awards_text[:15000]}\n\n"
        f"Instructions:\n"
        f"1. A win in the text counts for a category even if the phrasing differs (e.g., 'Best Actress' matches 'Best Performance by an Actress in a Leading Role').\n"
        f"2. Only count winner. DO NOT count nominated.\n"
        f"3. Respond with ONLY the category names from the list, separated by commas.\n"
        f"4. If no matches are found, respond with: None"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_msg}]},
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
    ]
    response = str(model(messages)).strip().rstrip(".")
    if response.lower() in ("none", ""):
        return set()

    canonical = {o.lower(): o for o in qualifying_oscars}
    won = set()
    for piece in response.split(","):
        key = piece.strip().lower().rstrip(".")
        if key in canonical:
            won.add(canonical[key])
    return won

# ---------------------------------------------------------------------------
# Data Normalization & Parsing Utilities
# ---------------------------------------------------------------------------

def parse_duration_to_minutes(raw: str) -> Optional[float]:
    """Parse a duration string to a float number of minutes.

    Handles formats:
        - "2h 15m", "2 hours 15 minutes", "2h"
        - "2:15" (H:MM colon format)
        - "135", "135 min", "135 minutes"

    Args:
        raw: Raw duration string.

    Returns:
        Duration in minutes, or None if unparseable.
    """
    s = str(raw).strip().lower()

    # "Xh Ym" format (Flexible for: h, hr, hrs, hour, hours)
    m = re.match(r'^(\d+)\s*h(?:ours?|r?s?)?(?:\s*(\d+)\s*m(?:in(?:utes?)?|s?)?)?$', s)
    if m:
        return int(m.group(1)) * 60 + (int(m.group(2)) if m.group(2) else 0)

    # "H:MM" colon format
    m = re.match(r'^(\d+):(\d+)$', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # Plain number with optional "min"/"minutes" suffix
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(?:min(?:utes?)?|m)?$', s)
    if m:
        return float(m.group(1))

    return None