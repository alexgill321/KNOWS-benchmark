"""Task-specific utilities for sheets_55_Movie_Recommendation.

Uses Playwright to fetch IMDb pages (IMDb blocks simple HTTP requests)
and parses structured JSON-LD data for programmatic verification.
LLM is only used for Oscar award verification where structured data
is not available.
"""

import json
import html
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
    title = html.unescape(title)
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


def _suggest_imdb_ids_full(movie_title: str, year: Optional[str] = None) -> List[Dict]:
    """Look up IMDb tt ID candidates via the suggest autocomplete endpoint.

    Filters results by content type (movies only), title match, and year.
    Returns full entry dicts sorted by rank (most popular first).

    Args:
        movie_title: Title of the movie.
        year: Optional release year to disambiguate remakes.

    Returns:
        List of suggest entry dicts (with 'id', 'y', 'l', etc.), most likely match first.
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
    return candidates


def _suggest_imdb_ids(movie_title: str, year: Optional[str] = None) -> List[str]:
    """Look up IMDb tt ID candidates via the suggest autocomplete endpoint.

    Returns just the IDs (wrapper around _suggest_imdb_ids_full for backward compat).
    """
    entries = _suggest_imdb_ids_full(movie_title, year)
    return [e["id"] for e in entries if e.get("id")]


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
    suggest_entries = _suggest_imdb_ids_full(movie_title, year)
    candidates = [e["id"] for e in suggest_entries] if suggest_entries else []
    suggest_year_map = {e["id"]: e.get("y") for e in suggest_entries}  # year from suggest API
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

        # Validate year if provided (±1 tolerance for limited vs wide release dates)
        if year:
            year_published = data.get("datePublished", "")[:4]
            year_str = str(year)[:4]
            try:
                if abs(int(year_str) - int(year_published)) > 1:
                    print(f"[IMDb] Year mismatch ({imdb_id}): expected {year_str}, got '{year_published}'")
                    continue
            except ValueError:
                pass

        # Backfill datePublished from suggest API if JSON-LD is missing it
        if not data.get("datePublished") and suggest_year_map.get(imdb_id):
            data["datePublished"] = str(suggest_year_map[imdb_id])

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


def fetch_academy_award_wins(imdb_id: str) -> Optional[List[str]]:
    """Fetch Academy Award wins from IMDb awards page by parsing HTML structure.

    Parses the Academy Awards section directly from the rendered HTML,
    extracting only entries marked as "Winner" with their category names.

    Args:
        imdb_id: IMDb title ID (e.g. 'tt1375666').

    Returns:
        List of Oscar category names won (e.g. ['Best Achievement in Cinematography']),
        or None if fetch failed. Empty list if no wins found.
    """
    if not imdb_id:
        return None

    from bs4 import BeautifulSoup

    awards_url = IMDB_AWARDS_URL.format(imdb_id=imdb_id)
    print(f"[IMDb] Fetching awards (structured): {awards_url}")

    # Fetch with retry — these pages are large and sometimes don't fully render
    html = None
    for attempt in range(2):
        context = None
        try:
            browser = get_browser()
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.goto(awards_url, wait_until="domcontentloaded", timeout=15000)

            # Wait for the Academy Awards section to render
            try:
                page.wait_for_selector(
                    "section.ipc-page-section h3",
                    timeout=5000,
                )
            except Exception:
                pass
            page.wait_for_timeout(1000)

            # Click expand buttons in the Academy Awards section
            try:
                sections = page.query_selector_all("section.ipc-page-section")
                for section in sections:
                    h3 = section.query_selector("h3")
                    if h3 and "Academy Awards" in h3.inner_text():
                        for btn in section.query_selector_all("button"):
                            btn_text = btn.inner_text().lower()
                            if "more" in btn_text or "see all" in btn_text:
                                btn.click()
                                page.wait_for_timeout(1000)
                        break
            except Exception:
                pass

            html = page.content()
        except Exception as e:
            if attempt == 1:
                print(f"[IMDb] Failed to fetch awards page: {e}")
                return None
        finally:
            if context:
                try:
                    context.close()
                except Exception:
                    pass

        # Quick validation: check if we got the section before parsing
        if html and "Academy Awards" in html:
            break
        elif attempt == 0:
            print(f"[IMDb] Retry awards fetch for {imdb_id} (page may not have rendered)")
            time.sleep(1)

    if not html or len(html) < 200:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Find the Academy Awards section
    for h3 in soup.find_all("h3"):
        if "Academy Awards" in h3.get_text():
            section = h3.find_parent("section")
            if not section:
                continue

            wins = []
            items = section.find_all("div", class_="ipc-metadata-list-summary-item__tc")
            for item in items:
                a_tag = item.find("a", class_="ipc-metadata-list-summary-item__t")
                if not a_tag:
                    continue
                title_text = a_tag.get_text(strip=True)
                if "Winner" not in title_text:
                    continue

                # Extract category from the item text
                full_text = item.get_text(separator=" | ", strip=True)
                # Format is: "YEAR Winner | Oscar | Category Name | Person Names..."
                parts = [p.strip() for p in full_text.split("|")]
                # Category is typically the 3rd part (after "YEAR Winner" and "Oscar")
                if len(parts) >= 3:
                    category = parts[2].strip()
                    wins.append(category)

            print(f"[IMDb] Academy Award wins for {imdb_id}: {wins}")
            return wins

    print(f"[IMDb] Academy Awards section not found for {imdb_id}")
    return None


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


def _match_oscar_category(win_text: str, qualifying_oscars: List[str]) -> Optional[str]:
    """Match an IMDb award category string to a qualifying Oscar name.

    Handles IMDb's verbose category names like "Best Achievement in Cinematography"
    mapping to the canonical "Best Cinematography". Also handles full IMDb names
    in the qualifying list (e.g. "Best Actor in a Leading Role").

    Args:
        win_text: Category text from IMDb (e.g. "Best Achievement in Cinematography").
        qualifying_oscars: List of qualifying Oscar names (short or full form).

    Returns:
        Matching qualifying Oscar name, or None if no match.
    """
    win_lower = win_text.lower()

    # First try direct/substring match — handles cases where qualifying_oscars
    # uses full IMDb names (e.g. "Best Actor in a Leading Role")
    for oscar_name in qualifying_oscars:
        oscar_lower = oscar_name.lower()
        if oscar_lower == win_lower:
            return oscar_name
        # Only match if the qualifying name is long enough to be specific (>20 chars)
        # to avoid false positives like "Best Actor" matching "Best Actor in a Supporting Role"
        if len(oscar_lower) > 20 and (oscar_lower in win_lower or win_lower in oscar_lower):
            return oscar_name

    # Regex-based matching for short canonical names
    # Includes both modern and vintage Oscar category names
    match_patterns = {
        "Best Actor": [r"\bactor\b(?!.*(?:support))", r"\bperformance by an actor in a leading\b"],
        "Best Actress": [r"\bactress\b(?!.*(?:support))", r"\bperformance by an actress in a leading\b"],
        "Best Director": [r"\bdirect(?:or|ing)\b"],
        "Best Original Screenplay": [
            r"\boriginal\s+screenplay\b",
            r"\bwriting,?\s+original\b",
            r"\bscreenplay\s+written\s+directly\b",
            r"\bstory\s+and\s+screenplay\b",
            r"\bwritten\s+directly\s+for\s+the\s+screen\b",
        ],
        "Best Adapted Screenplay": [
            r"\badapted\s+screenplay\b",
            r"\bwriting,?\s+adapted\b",
            r"\bscreenplay\s+based\s+on\b",
            r"\bmaterial\s+previously\s+produced\b",
            r"\bscreenplay\s*-?\s*adapted\b",
            r"\bbased\s+on\s+material\s+from\s+another\s+medium\b",
        ],
        "Best Cinematography": [r"\bcinematography\b"],
    }

    for oscar_name in qualifying_oscars:
        patterns = match_patterns.get(oscar_name, [])
        for pattern in patterns:
            if re.search(pattern, win_lower):
                return oscar_name

    return None


def extract_qualifying_oscars_won(
    model: Any,
    movie_title: str,
    qualifying_oscars: List[str],
    awards_text: Optional[str],
    imdb_id: Optional[str] = None,
) -> Optional[set]:
    """Extract which qualifying Oscars a movie actually won.

    Uses programmatic HTML parsing of the IMDb awards page first.
    Falls back to LLM-based extraction from plain text only if
    structured parsing fails.

    Args:
        model: LLM model callable (used as fallback).
        movie_title: Title of the movie.
        qualifying_oscars: List of qualifying Oscar category names.
        awards_text: Plain text from the IMDb awards page (fallback).
        imdb_id: IMDb title ID for structured parsing.

    Returns:
        Set of canonical category names (from qualifying_oscars) that the movie
        won according to the awards page. Empty set if it won none of them.
        None if the awards data is unavailable.
    """
    # Try programmatic parsing first
    if imdb_id:
        wins = fetch_academy_award_wins(imdb_id)
        if wins is not None:
            matched = set()
            for win_category in wins:
                match = _match_oscar_category(win_category, qualifying_oscars)
                if match:
                    matched.add(match)
            print(f"[Oscar] Programmatic result for '{movie_title}': {sorted(matched) if matched else 'none'}")
            return matched

    # Fallback to LLM if structured parsing failed
    if not awards_text and imdb_id:
        awards_text = fetch_imdb_awards_text(imdb_id)
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