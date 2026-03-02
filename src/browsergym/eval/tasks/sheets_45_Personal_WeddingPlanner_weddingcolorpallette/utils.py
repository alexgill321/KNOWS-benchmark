"""Shared utilities for sheets_45 Wedding Color Palette evaluator.

Provides helpers for navigating Google Sheets raw API responses by tab name,
reading column ranges, checking cell formatting, and robustly detecting images.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

# Import general utilities from eval_utils
from src.browsergym.eval.eval_utils.web_utils import (
    validate_url_accessible,
    fetch_page_title,
)
from src.browsergym.eval.eval_utils.parallel_utils import parallel_execute
from src.browsergym.eval.eval_utils.google_sheets_utils import find_urls_in_sheet
from src.browsergym.eval.eval_utils.table_utils import (
    cell_bg_hex as _cell_bg_hex_raw,
    get_background_color,
    get_cell_value,
    get_image_url_from_raw_sheet_cell,
    read_column_values,
)


def cell_bg_hex(sheet_tab: Dict, row_idx: int, col_idx: int) -> Optional[str]:
    """Thin wrapper: calls table_utils.cell_bg_hex with a sheet_tab."""
    return _cell_bg_hex_raw({"sheets": [sheet_tab]}, row_idx, col_idx)


# ---------------------------------------------------------------------------
# Robust image detection
# ---------------------------------------------------------------------------

def detect_any_image(
    sheet_tab: Dict,
    row_idx: int,
    col_idx: int,
) -> Optional[str]:
    """Detect an image in a single cell and return its URL, or None.

    Delegates to ``get_image_url_from_raw_sheet_cell`` which checks:
    formulaValue (``=IMAGE("url")``), stringValue, formattedValue,
    and the hyperlink property.

    Args:
        sheet_tab: Sheet tab dict.
        row_idx: 0-based row.
        col_idx: 0-based column.

    Returns:
        Image URL string, or None.
    """
    return get_image_url_from_raw_sheet_cell({"sheets": [sheet_tab]}, row_idx, col_idx)


# ---------------------------------------------------------------------------
# Colour-category heuristic (blue / green / red)
# ---------------------------------------------------------------------------

# Each set holds lowercase substrings that identify a shade.  Order does not
# matter — we check membership with ``any(kw in name for kw in …)``.
_BLUE_KW = {
    "blue", "navy", "cobalt", "cerulean", "sapphire", "azure", "indigo",
    "periwinkle", "slate", "sky", "midnight", "dusty", "cornflower",
    "denim", "steel", "powder", "ice", "aqua", "aegean", "capri",
    "electric blue", "baby blue", "royal blue", "tiffany",
}
_GREEN_KW = {
    "green", "sage", "olive", "emerald", "forest", "mint", "jade",
    "hunter", "lime", "pistachio", "moss", "eucalyptus", "fern",
    "chartreuse", "seafoam", "teal", "viridian", "celadon", "shamrock",
    "laurel", "basil", "juniper", "kelly", "malachite", "spruce",
}
_RED_KW = {
    "red", "burgundy", "crimson", "scarlet", "ruby", "wine", "maroon",
    "cherry", "garnet", "brick", "coral", "rose", "blush", "merlot",
    "raspberry", "cranberry", "vermilion", "carmine", "claret",
    "sangria", "terracotta", "rust", "auburn", "mahogany", "oxblood",
    "berry", "magenta", "cerise", "cardinal", "firebrick", "tomato",
    "salmon", "flamingo", "fuchsia", "petal", "peony", "poppy",
}


def classify_color_category(color_name: str) -> Optional[str]:
    """Return ``'blue'``, ``'green'``, or ``'red'`` if the name matches, else None.

    Uses exact keyword matching against curated shade lists.
    Multi-word keywords (e.g. ``"baby blue"``) are matched as exact
    substrings; single-word keywords must match a whole word in the name.
    Case-insensitive.

    Args:
        color_name: A single colour-name string (e.g. ``"Dusty Rose"``).

    Returns:
        Category string or None if unrecognised.
    """
    name = color_name.lower().strip()
    if not name:
        return None
    words = set(name.split())

    for kw in _BLUE_KW:
        if " " in kw:
            if kw in name:
                return "blue"
        elif kw in words:
            return "blue"
    for kw in _GREEN_KW:
        if " " in kw:
            if kw in name:
                return "green"
        elif kw in words:
            return "green"
    for kw in _RED_KW:
        if " " in kw:
            if kw in name:
                return "red"
        elif kw in words:
            return "red"
    return None


# ---------------------------------------------------------------------------
# Fuzzy / semantic colour-name matching
# ---------------------------------------------------------------------------


def _normalize_color_name(name: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return " ".join(name.lower().split())


def color_names_match(a: str, b: str) -> bool:
    """Return True if two colour names are semantically equivalent.

    Strategy (in order):
    1. Exact match after normalisation.
    2. One name is a substring of the other (e.g. "Navy" ↔ "Navy Blue").
    3. Both names share a keyword root from the colour-category lists
       (e.g. both contain "cobalt").

    Args:
        a: First colour name.
        b: Second colour name.

    Returns:
        True if the names should be considered the same colour.
    """
    na = _normalize_color_name(a)
    nb = _normalize_color_name(b)
    if not na or not nb:
        return False

    # 1. Exact normalised match
    if na == nb:
        return True

    # 2. Substring containment
    if na in nb or nb in na:
        return True

    # 3. Shared keyword root
    all_kw = _BLUE_KW | _GREEN_KW | _RED_KW
    kw_a = {kw for kw in all_kw if kw in na}
    kw_b = {kw for kw in all_kw if kw in nb}
    if kw_a and kw_b and kw_a & kw_b:
        return True

    return False


def fuzzy_match_color_in_set(name: str, color_set: set) -> bool:
    """Return True if *name* fuzzy-matches any entry in *color_set*.

    *color_set* should contain normalised (lowered/stripped) colour names.

    Args:
        name: Colour name to look up.
        color_set: Set of normalised colour names.

    Returns:
        True if a match is found.
    """
    n = _normalize_color_name(name)
    if n in color_set:
        return True
    for c in color_set:
        if color_names_match(n, c):
            return True
    return False


# ---------------------------------------------------------------------------
# Color list & decoration matrix discovery
# ---------------------------------------------------------------------------

# Common header labels that should not be treated as colour names.
_COLOR_HEADER_LABELS = {
    "color", "colour", "colors", "colours",
    "color name", "colour name", "color names", "colour names",
    "name", "names", "shade", "shades", "hue", "hues",
}


def _is_color_header(val: str) -> bool:
    """Return True if *val* looks like a column header rather than a colour name."""
    return val.lower().strip() in _COLOR_HEADER_LABELS


def find_color_region(
    sheet_tab: Dict,
    col_idx: int = 0,
    max_row: int = 30,
) -> Optional[Dict[str, Any]]:
    """Locate the vertical colour-name list and return its region metadata.

    Scans *col_idx* from row 0 downward, collecting non-empty values that
    are **not** URLs and **not** header labels (e.g. "Color Name").
    Stops at the first empty cell after at least one value has been collected
    (i.e. a contiguous block).

    Args:
        sheet_tab: Sheet tab dict.
        col_idx: Column to scan (default 0 = column A).
        max_row: Stop scanning after this row.

    Returns:
        Dict with ``col``, ``start_row`` (inclusive), ``end_row`` (exclusive),
        and ``names`` (list of colour-name strings), or None if nothing found.
    """
    raw = read_column_values(sheet_tab, col_idx, start_row=0, end_row=max_row)
    names: List[str] = []
    start_row: Optional[int] = None
    end_row = 0
    for idx, val in enumerate(raw):
        val = val.strip()
        if val and not val.startswith(("http://", "https://")) and not _is_color_header(val):
            if start_row is None:
                start_row = idx
            names.append(val)
            end_row = idx + 1
        elif start_row is not None:
            break  # first blank after the block
    if not names or start_row is None:
        return None
    return {
        "col": col_idx,
        "start_row": start_row,
        "end_row": end_row,
        "names": names,
    }


def find_color_list(sheet_tab: Dict, col_idx: int = 0, max_row: int = 30) -> List[str]:
    """Read the vertical colour-name list from the top-left area of a sheet.

    Delegates to ``find_color_region`` and returns only the name strings.

    Args:
        sheet_tab: Sheet tab dict.
        col_idx: Column to scan (default 0 = column A).
        max_row: Stop scanning after this row even if cells are non-empty.

    Returns:
        Ordered list of colour name strings (may be empty).
    """
    region = find_color_region(sheet_tab, col_idx=col_idx, max_row=max_row)
    return region["names"] if region else []


def find_decoration_matrix(
    sheet_tab: Dict,
    color_names: List[str],
    search_start_row: int = 0,
    max_row: int = 80,
) -> Optional[Dict[str, Any]]:
    """Locate the decoration matrix region below the colour list.

    Strategy: scan rows from *search_start_row* looking for a **header row**
    where ≥ 50 % of the non-empty cells match a colour name from
    *color_names*.  The first such row is treated as the matrix header.
    Data rows follow immediately and end at the first fully-empty row.

    Args:
        sheet_tab: Sheet tab dict.
        color_names: The colour names extracted from the colour list.
        search_start_row: Row to begin the scan (typically the end of the
            colour list).
        max_row: Hard upper bound for the scan.

    Returns:
        Dict with keys:
            ``header_row``  – 0-based row index of the header
            ``header_names`` – list of header cell values (all columns)
            ``data_start_row`` – first data row (header_row + 1)
            ``data_end_row`` – exclusive end of data rows
            ``label_col`` – 0-based index of the leftmost column
                            (decoration names)
            ``data_col_start`` – 0-based index of first colour-data column
            ``data_col_end`` – exclusive end of colour-data columns
        or None when no matching header row is found.
    """
    if not color_names:
        return None

    grid = sheet_tab.get("properties", {}).get("gridProperties", {})
    num_cols = grid.get("columnCount", 30)
    sheet_raw = {"sheets": [sheet_tab]}
    color_set = {_normalize_color_name(c) for c in color_names}

    for r_idx in range(search_start_row, max_row):
        # Gather formatted values for every column in this row
        cell_values = [
            (get_cell_value(sheet_raw, r_idx, c_idx) or "").strip()
            for c_idx in range(num_cols)
        ]

        non_empty = [v for v in cell_values if v]
        if len(non_empty) < 2:
            continue

        # Count how many non-empty cells fuzzy-match a colour name
        match_count = sum(
            1 for v in non_empty if fuzzy_match_color_in_set(v, color_set)
        )

        # Require ≥ 50 % match and at least 2 matches
        if match_count < 2 or match_count / len(non_empty) < 0.5:
            continue

        # Found candidate header row.
        # Determine column bounds: the leftmost non-empty cell is the
        # label column; the colour columns follow.
        first_non_empty = next(
            (i for i, v in enumerate(cell_values) if v), 0
        )
        last_non_empty = max(
            (i for i, v in enumerate(cell_values) if v), default=0
        )

        # The label column might be an empty "corner" cell if decorations
        # start one column to the left of the header colours.  In that
        # case the first non-empty cell IS a colour name and the label
        # column is first_non_empty - 1 (but not < 0).
        if fuzzy_match_color_in_set(cell_values[first_non_empty], color_set):
            label_col = max(first_non_empty - 1, 0)
            data_col_start = first_non_empty
        else:
            label_col = first_non_empty
            data_col_start = first_non_empty + 1

        data_col_end = last_non_empty + 1

        # Walk downward to find the data extent.
        data_start = r_idx + 1
        data_end = data_start
        for dr in range(data_start, max_row):
            row_vals = [
                (get_cell_value(sheet_raw, dr, c_idx) or "").strip()
                for c_idx in range(num_cols)
            ]
            if not any(row_vals):
                break
            data_end = dr + 1

        header_names = cell_values[data_col_start:data_col_end]

        return {
            "header_row": r_idx,
            "header_names": header_names,
            "data_start_row": data_start,
            "data_end_row": data_end,
            "label_col": label_col,
            "data_col_start": data_col_start,
            "data_col_end": data_col_end,
        }

    return None


# ---------------------------------------------------------------------------
# Palette-tab helpers (checkpoint 6)
# ---------------------------------------------------------------------------

def find_palette_tab(sheet_raw: Dict[str, Any]) -> Optional[Dict]:
    """Find a sheet tab whose name suggests it holds colour palettes.

    Skips the first tab (the main sheet) and looks for titles containing
    ``"palette"`` (case-insensitive).  Falls back to any second-or-later tab
    if nothing matches by name.

    Args:
        sheet_raw: Full spreadsheet response from ``get_sheet_content()``.

    Returns:
        The matching sheet tab dict, or None if only one tab exists.
    """
    if not sheet_raw:
        return None
    sheets = sheet_raw.get("sheets", [])
    if len(sheets) < 2:
        return None

    # Prefer a tab whose title contains "palette"
    for tab in sheets[1:]:
        title = tab.get("properties", {}).get("title", "")
        if "palette" in title.lower():
            return tab

    # Fallback: return the second tab
    return sheets[1]


def collect_reference_bg_hexes(
    sheet_tab: Dict,
    num_rows: int,
    max_col: int = 10,
) -> Set[str]:
    """Collect every distinct non-white background hex from the first *num_rows*.

    Scans all columns up to *max_col* for each row.  This captures the
    colour swatches placed alongside the colour names regardless of which
    exact column they are in.

    Args:
        sheet_tab: Main sheet tab dict.
        num_rows: Number of rows to scan (typically ``len(color_names)``).
        max_col: Rightmost column to check (exclusive, default 10).

    Returns:
        Set of lowercase ``#rrggbb`` hex strings.
    """
    hexes: Set[str] = set()
    for r in range(num_rows):
        for c in range(max_col):
            h = cell_bg_hex(sheet_tab, r, c)
            if h:
                hexes.add(h.lower())
    return hexes


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` to an (R, G, B) tuple with 0-255 values."""
    hex_str = hex_str.lstrip("#")
    return (
        int(hex_str[0:2], 16),
        int(hex_str[2:4], 16),
        int(hex_str[4:6], 16),
    )


def hex_color_distance(hex1: str, hex2: str) -> float:
    """Euclidean distance between two ``#rrggbb`` colours in RGB space.

    Returns a value in [0, ~441].  Two identical colours give 0.
    """
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def hex_matches_any(
    target: str,
    reference_set: Set[str],
    tolerance: float = 45.0,
) -> bool:
    """Return True if *target* is within *tolerance* of any colour in the set.

    A tolerance of ~45 allows for minor rounding differences between the
    agent's chosen shade and the reference swatch while still rejecting
    unrelated colours.

    Args:
        target: ``#rrggbb`` hex to test.
        reference_set: Set of ``#rrggbb`` hex strings.
        tolerance: Max Euclidean RGB distance to consider a match.

    Returns:
        True when at least one reference colour is close enough.
    """
    target = target.lower()
    for ref in reference_set:
        if hex_color_distance(target, ref) <= tolerance:
            return True
    return False


def count_filled_cells_in_row(
    sheet_tab: Dict,
    row_idx: int,
    max_col: int = 30,
) -> Tuple[int, List[str]]:
    """Count cells with a non-white background fill in a row.

    Args:
        sheet_tab: Sheet tab dict.
        row_idx: 0-based row index.
        max_col: Rightmost column to check (exclusive).

    Returns:
        Tuple of (count, list_of_hex_strings) for the filled cells.
    """
    count = 0
    hexes: List[str] = []
    for c in range(max_col):
        h = cell_bg_hex(sheet_tab, row_idx, c)
        if h:
            count += 1
            hexes.append(h.lower())
    return count, hexes


# ---------------------------------------------------------------------------
# URL verification helpers (using eval_utils.web_utils)
# ---------------------------------------------------------------------------

def url_matches_wedding_article(
    url: str,
    page_title: str,
    model: Any = None,
) -> bool:
    """Check if a URL leads to a wedding colour article.

    Uses keyword matching first, with optional LLM fallback.

    Args:
        url: The article URL.
        page_title: Title fetched from the page (may be empty).
        model: Optional LLM model for fallback matching.

    Returns:
        True if the URL plausibly leads to a wedding colour article.
    """
    combined = f"{url} {page_title}".lower()

    # Known wedding / bridal / design domains — always accept
    wedding_domains = [
        "theknot.com", "weddingwire.com", "brides.com", "marthastewartweddings",
        "bridalguide", "greenweddingshoes", "stylemepretty.com", "junebugweddings",
        "ruffledblog", "oncewed.com", "elizabethannedesigns", "weddingchicks",
        "100layercake", "loverly.com", "zola.com", "minted.com",
    ]
    if any(d in combined for d in wedding_domains):
        return True

    # Accept if URL/title contains ANY wedding keyword (since this task is
    # specifically about wedding colors, a wedding article is a valid source)
    wedding_kw = [
        "wedding", "bridal", "bride", "nuptial", "marriage",
        "ceremony", "reception", "bouquet", "bridesmaid",
    ]
    if any(w in combined for w in wedding_kw):
        return True

    # Accept if URL/title contains BOTH a general article indicator AND
    # a color-related keyword (e.g. a color blog post the agent used as source)
    article_kw = [
        "blog", "article", "post", "guide", "tips", "best",
        "top", "ideas", "inspiration", "trend", "magazine",
    ]
    color_kw = [
        "color", "colour", "palette", "shade", "hue", "scheme",
        "blue", "green", "red", "navy", "sage", "forest",
        "dusty", "slate", "emerald", "burgundy", "coral",
    ]
    if any(a in combined for a in article_kw) and any(c in combined for c in color_kw):
        return True

    # LLM fallback (try even with empty page_title — URL alone can be informative)
    if model is not None:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are verifying whether a URL could be a source article "
                    "for wedding color research. This includes wedding blogs, "
                    "color inspiration articles, design/decor articles, or any "
                    "content about colors that could inform a wedding palette. "
                    "Be lenient — answer 'Yes' for any article, blog, or "
                    "content page that discusses colors, weddings, design, or "
                    "decor. Only answer 'No' for clearly unrelated pages "
                    "(e.g. shopping carts, login pages, error pages)."
                )}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": (
                    f"URL: {url}\n"
                    f"{'Page title: ' + page_title if page_title else '(page title unavailable)'}\n"
                    f"Could this be a source article for wedding color research? "
                    f"Answer Yes or No."
                )}],
            },
        ]
        try:
            response = model(messages)
            if response and "yes" in response.lower():
                return True
        except Exception as e:
            print(f"  LLM error for article matching: {e}")

    return False


def url_matches_paint_store(
    url: str,
    page_title: str,
    model: Any = None,
) -> bool:
    """Check if a URL leads to a paint store or colour reference page.

    Uses keyword matching first, with optional LLM fallback.

    Args:
        url: The paint-store URL.
        page_title: Title fetched from the page (may be empty).
        model: Optional LLM model for fallback matching.

    Returns:
        True if the URL plausibly leads to a paint/color reference.
    """
    combined = f"{url} {page_title}".lower()
    paint_kw = [
        "paint", "sherwin", "benjamin moore", "behr", "valspar",
        "colorhexa", "pantone", "hex", "swatch", "color code",
        "dulux", "farrow", "ppg", "glidden", "myperfectcolor",
    ]
    if any(kw in combined for kw in paint_kw):
        return True

    # LLM fallback
    if model is not None and page_title:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are verifying whether a URL leads to a paint store, "
                    "color swatch page, or color reference website. "
                    "Answer 'Yes' if it is a paint/color reference or 'No' "
                    "if clearly unrelated."
                )}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": (
                    f"URL: {url}\nPage title: {page_title}\n"
                    f"Is this a paint store or color reference page? "
                    f"Answer Yes or No."
                )}],
            },
        ]
        try:
            response = model(messages)
            if response and "yes" in response.lower():
                return True
        except Exception as e:
            print(f"  LLM error for paint store matching: {e}")

    return False


# ---------------------------------------------------------------------------
# Unified URL column validation (shared by checkpoints 2 & 4)
# ---------------------------------------------------------------------------

def validate_and_match_urls(
    sheet_tab: Dict,
    color_names: List[str],
    col_idx: int,
    start_row: int,
    end_row: int,
    relevance_fn: Any,
    model: Any = None,
    relevance_threshold: float = 0.7,
) -> List[str]:
    """Validate a column of URLs: liveness and content relevance.

    Checks every URL found in the specified column range for reachability
    and content relevance.  Presence/coverage checks (whether every colour
    has a URL) are left to the caller so that each evaluation criterion
    maps to exactly one checkpoint step.

    Args:
        sheet_tab: Sheet tab dict.
        color_names: The colour names corresponding to each row.
        col_idx: 0-based column index to scan for URLs.
        start_row: First row (inclusive, 0-based).
        end_row: Last row (exclusive, 0-based).
        relevance_fn: Callable ``(url, page_title, model=) -> bool`` that
            checks whether a URL's content is relevant (e.g.
            ``url_matches_wedding_article`` or ``url_matches_paint_store``).
        model: Optional LLM model passed to *relevance_fn*.
        relevance_threshold: Minimum fraction of reachable URLs that must
            match via *relevance_fn* (default 0.7).

    Returns:
        List of failure description strings (empty if all checks pass).
        Failure strings contain keyword ``"reachable"`` for liveness issues
        and ``"relevant"`` for content-match issues so the caller can
        classify them.
    """
    failures: List[str] = []

    # Scan URLs from the column
    urls_found: List[str] = []
    rows = sheet_tab.get("data", [{}])[0].get("rowData", [])
    for i, r_idx in enumerate(range(start_row, end_row)):
        found = find_urls_in_sheet(rows, start_row=r_idx, num_rows=1,
                                   start_col=col_idx, end_col=col_idx + 1)
        url = found[0] if found else None
        if url:
            urls_found.append(url)

    if not urls_found:
        failures.append(f"No URLs found in col {col_idx}")
        return failures

    # Phase 1: Verify liveness in parallel
    liveness_tasks = [
        {"id": url, "func": validate_url_accessible, "args": (url,)}
        for url in urls_found
    ]
    liveness_results = parallel_execute(liveness_tasks, max_workers=5)

    truly_dead: List[str] = []
    reachable_urls: List[str] = []
    for url in urls_found:
        result = liveness_results.get(url)
        if result is None:
            truly_dead.append(f"{url[:50]} (no response)")
            continue
        accessible, details = result
        if accessible:
            reachable_urls.append(url)
        elif any(str(code) in details for code in [403, 405, 406, 429]):
            reachable_urls.append(url)
        else:
            truly_dead.append(f"{url[:50]} ({details})")

    print(
        f"  [DEBUG] {len(reachable_urls)}/{len(urls_found)} URLs are reachable, "
        f"{len(truly_dead)} dead"
    )

    if truly_dead:
        failures.append(
            f"{len(truly_dead)}/{len(urls_found)} URLs are not reachable: "
            f"{truly_dead[:3]}"
        )

    # Phase 2: Fetch titles + relevance matching in parallel
    if reachable_urls:
        # Deduplicate URLs to avoid redundant network calls; count
        # results against ALL rows (including duplicates) afterwards.
        unique_reachable = list(dict.fromkeys(reachable_urls))

        def _check_relevance(url: str) -> bool:
            title = fetch_page_title(url) or ""
            return relevance_fn(url, title, model=model)

        relevance_tasks = [
            {"id": url, "func": _check_relevance, "args": (url,)}
            for url in unique_reachable
        ]
        relevance_results = parallel_execute(relevance_tasks, max_workers=5)

        # Map results back to ALL reachable URLs (including duplicates)
        matched = sum(
            1 for url in reachable_urls if relevance_results.get(url)
        )

        print(
            f"  [DEBUG] {matched}/{len(reachable_urls)} URLs match relevance check"
        )

        if len(reachable_urls) > 0:
            rel_ratio = matched / len(reachable_urls)
            if rel_ratio < relevance_threshold:
                failures.append(
                    f"Only {matched}/{len(reachable_urls)} links appear "
                    f"relevant ({rel_ratio:.0%})"
                )

    return failures
