"""Task-agnostic shared utilities for the sheets_45 colour-palette evaluators.

Nothing here is specific to a single instance's scenario: colour categories,
topic keywords, store domains, thresholds and counts are all supplied by the
caller (the per-instance config block in each ``evaluator.py``).

Provides helpers for navigating Google Sheets raw API responses, detecting the
colour list / decoration matrix / palette region, classifying colours, matching
URLs against a topic, validating URL columns, and guarded model loading.
"""

import math
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Imports from eval_utils
from src.browsergym.knows.eval.eval_utils.web_utils import (
    fetch_page_title,
    fetch_with_fallbacks_extended,
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import (
    parallel_execute,
    fast_parallel_vlm_calls,
)
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import find_urls_in_sheet
from src.browsergym.knows.eval.eval_utils.table_utils import (
    get_background_color,
    get_cell_value,
    get_image_url_from_raw_sheet_cell,
    read_column_values,
)
from src.browsergym.knows.eval.eval_utils.llm_utils import (
    evaluate_with_llm,
    extract_json_with_llm,
)
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint


# ---------------------------------------------------------------------------
# Module constants (replace magic numbers)
# ---------------------------------------------------------------------------

DEFAULT_COLOR_MAX_ROW = 30          # row cap for the colour-name list scan
DEFAULT_MATRIX_MAX_ROW = 80         # row cap for the decoration-matrix scan
HEADER_MATCH_RATIO = 0.5            # fraction of header cells that must be colours
DEFAULT_HEX_TOLERANCE = 45.0        # max RGB distance for a hex "match"
DEFAULT_RELEVANCE_THRESHOLD = 1.0   # every reachable link must be on-topic
PALETTE_LABEL_MAX_COL = 6           # max col when searching for palette label col
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEX_TEXT_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


# ---------------------------------------------------------------------------
# Sheet-access guards
# ---------------------------------------------------------------------------

def get_row_data(sheet_tab: Dict) -> List[Dict]:
    """Return the ``rowData`` list of a sheet tab, guarding empty ``data``.

    ``sheet_tab["data"]`` may be missing OR present-but-empty; ``[0]`` on the
    latter raises IndexError. This centralises the safe access.

    Args:
        sheet_tab: Sheet tab dict.

    Returns:
        List of row dicts (possibly empty).
    """
    data = sheet_tab.get("data") or [{}]
    return data[0].get("rowData", []) if data else []


# ---------------------------------------------------------------------------
# Guarded, cached model loading
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOAD_FAILED: Set[str] = set()


def get_model(model_id: str, cached_models: Optional[Dict[str, Any]] = None) -> Optional[Callable]:
    """Lazily load an LLM/VLM model, cached and guarded — never raises.

    A failed load is remembered so subsequent calls return ``None`` immediately
    instead of retrying. Callers degrade gracefully when ``None`` is returned.

    Args:
        model_id: Model identifier passed to ``load_model``.
        cached_models: Optional dict of preloaded models keyed by model_id.

    Returns:
        The model callable, or None if it could not be loaded.
    """
    if cached_models and cached_models.get(model_id) is not None:
        _MODEL_CACHE[model_id] = cached_models[model_id]
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    if model_id in _MODEL_LOAD_FAILED:
        return None
    try:
        _MODEL_CACHE[model_id] = load_model(model_id)
        return _MODEL_CACHE[model_id]
    except Exception as e:
        print(f"  [WARN] model load failed for '{model_id}': {e}")
        _MODEL_LOAD_FAILED.add(model_id)
        return None


# ---------------------------------------------------------------------------
# Checkpoint early-exit helper
# ---------------------------------------------------------------------------

def fail_all_steps(
    checkpoint: Checkpoint,
    step_names: List[str],
    reason: str,
    start_time: Optional[float] = None,
) -> Checkpoint:
    """Mark every declared step of a checkpoint as failed with one reason.

    Used by early-exit guards so a checkpoint always reports all N steps even
    when a prerequisite (sheet data, colour list, ...) is missing.

    Args:
        checkpoint: The Checkpoint to populate.
        step_names: Ordered step names mirroring checkpoints.md.
        reason: Failure detail applied to every step.
        start_time: Optional checkpoint start time for execution_time.

    Returns:
        The same checkpoint, populated.
    """
    for idx, name in enumerate(step_names, start=1):
        checkpoint.add_step(name, False, idx, reason)
    if start_time is not None:
        checkpoint.execution_time = time.time() - start_time
    return checkpoint


# ---------------------------------------------------------------------------
# Noise-robust VLM/LLM calls
# ---------------------------------------------------------------------------

def robust_vlm_calls(
    vlm_tasks: List[Dict[str, Any]],
    model: Any,
    samples: int = 2,
    max_workers: int = 10,
) -> Dict[str, bool]:
    """Judge each VLM/LLM task ``samples`` times; an item fails only if every
    sample fails it.

    Model calls are non-deterministic, so a genuinely-correct item can be
    mis-judged on a single call. Honouring a lone dissenting pass means one
    noisy false-negative cannot fail a step that requires every item to pass.

    Args:
        vlm_tasks: Tasks as for ``fast_parallel_vlm_calls`` (``id`` / ``messages``).
        model: The loaded model callable.
        samples: Number of times to judge each task.
        max_workers: Parallel worker cap.

    Returns:
        Dict mapping each task id to ``True`` unless all of its samples
        returned ``False``. Empty input yields ``{}``.
    """
    if not vlm_tasks:
        return {}
    expanded = [
        {"id": f"{t['id']}##{k}", "messages": t["messages"]}
        for t in vlm_tasks
        for k in range(samples)
    ]
    raw = fast_parallel_vlm_calls(expanded, model, max_workers=max_workers)
    return {
        t["id"]: any(raw.get(f"{t['id']}##{k}", False) for k in range(samples))
        for t in vlm_tasks
    }


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def cell_bg_hex(sheet_tab: Dict, row_idx: int, col_idx: int) -> Optional[str]:
    """Return the cell background colour as ``#rrggbb``, or None for white / no fill.

    The Sheets API omits a colour channel whose value is 0, so a missing
    channel must default to 0.0 (not 1.0) — otherwise pure-primary fills such
    as ``{'blue': 1}`` are mis-read as white.

    Args:
        sheet_tab: Sheet tab dict.
        row_idx: 0-based row index.
        col_idx: 0-based column index.

    Returns:
        Lowercase ``#rrggbb`` hex string, or None for a white / unfilled cell.
    """
    bg = get_background_color({"sheets": [sheet_tab]}, row_idx, col_idx)
    if not bg:
        return None
    r = bg.get("red", 0.0)
    g = bg.get("green", 0.0)
    b = bg.get("blue", 0.0)
    if r > 0.98 and g > 0.98 and b > 0.98:
        return None
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255)),
    )


def detect_any_image(sheet_tab: Dict, row_idx: int, col_idx: int) -> Optional[str]:
    """Detect an image in a single cell and return its URL, or None.

    Delegates to ``get_image_url_from_raw_sheet_cell`` which checks
    formulaValue (``=IMAGE("url")``), stringValue, formattedValue and the
    hyperlink property.

    Args:
        sheet_tab: Sheet tab dict.
        row_idx: 0-based row.
        col_idx: 0-based column.

    Returns:
        Image URL string, or None.
    """
    return get_image_url_from_raw_sheet_cell({"sheets": [sheet_tab]}, row_idx, col_idx)


# Known placeholder / stub image generators — these return a generated
# rectangle (often with text), never a real photograph.
_PLACEHOLDER_IMAGE_DOMAINS = (
    "placehold.co", "placehold.it", "placehold.jp", "placeholder.com",
    "dummyimage.com", "fakeimg.pl", "fakeimg.com",
)


def is_placeholder_image_url(url: str) -> bool:
    """Return True if a URL is from a known placeholder / stub image service.

    Such services (placehold.co, dummyimage.com, ...) only ever return a
    generated coloured rectangle — never a real photograph — so the image
    cannot be a genuine decoration picture.

    Args:
        url: The image URL to check.

    Returns:
        True if the URL belongs to a known placeholder-image service.
    """
    if not url:
        return False
    u = url.lower()
    return any(d in u for d in _PLACEHOLDER_IMAGE_DOMAINS)


# ---------------------------------------------------------------------------
# Colour-category classification (parameterised — no hardcoded shade lists)
# ---------------------------------------------------------------------------

def classify_color_category(
    color_name: str,
    categories: Tuple[str, ...],
    model: Any = None,
) -> Optional[str]:
    """Return the category a colour name belongs to, or None.

    Hybrid strategy:
    1. Literal match — if a category word appears in the lowercased name,
       return that category (e.g. "Royal Blue" -> "blue").
    2. Optional LLM fallback — when ``model`` is supplied, ask which category
       (if any) the colour belongs to.

    Args:
        color_name: A single colour-name string.
        categories: Tuple of category names (e.g. ``("blue", "green", "red")``).
        model: Optional LLM model for the fallback.

    Returns:
        The matched category string, or None.
    """
    name = (color_name or "").lower().strip()
    if not name:
        return None

    for cat in categories:
        c = cat.lower().strip()
        if c and c in name:
            return cat

    if model is not None:
        cat_list = ", ".join(categories)
        prompt = (
            f"Colour families: {cat_list}.\n"
            f"Which family does the colour '{color_name}' belong to? Answer with "
            f"the family name exactly, or 'none' if it belongs to none of them."
        )
        try:
            resp = evaluate_with_llm(prompt, model, return_type="str")
            if resp:
                for cat in categories:
                    if cat.lower() in resp:
                        return cat
        except Exception as e:
            print(f"  LLM error classifying '{color_name}': {e}")

    return None


def classify_colors_batch(
    names: List[str],
    categories: Tuple[str, ...],
    model: Any = None,
) -> Dict[str, bool]:
    """Decide, for each colour name, whether it belongs to any of *categories*.

    A literal-match pass runs first; a single batched LLM pass resolves the
    remaining names when a model is available.

    Args:
        names: Colour names to classify.
        categories: Tuple of category names.
        model: Optional LLM model for the fallback.

    Returns:
        Dict mapping each name to True/False (belongs to a category).
    """
    result: Dict[str, bool] = {}
    unresolved: List[str] = []
    for name in names:
        if classify_color_category(name, categories) is not None:
            result[name] = True
        else:
            result[name] = False
            unresolved.append(name)

    if unresolved and model is not None:
        cat_list = ", ".join(categories)
        tasks = []
        for name in unresolved:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": (
                    "You are a colour classification assistant. Given a colour "
                    "name and a list of colour families, answer 'Yes' if the "
                    "colour belongs to ANY of the listed families, else 'No'."
                )}]},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"Colour name: '{name}'\nColour families: {cat_list}\n"
                    f"Does '{name}' belong to any of these families? Answer Yes or No."
                )}]},
            ]
            tasks.append({"id": name, "messages": messages})
        try:
            llm = robust_vlm_calls(tasks, model)
            for name in unresolved:
                result[name] = bool(llm.get(name, False))
        except Exception as e:
            print(f"  LLM batch classification error: {e}")

    return result


# ---------------------------------------------------------------------------
# Fuzzy colour-name matching
# ---------------------------------------------------------------------------

def _normalize_color_name(name: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return " ".join(name.lower().split())


def color_names_match(a: str, b: str) -> bool:
    """Return True if two colour names are equivalent.

    Strategy: exact match after normalisation, or one name contained in the
    other (e.g. "Navy" <-> "Navy Blue").

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
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


def fuzzy_match_color_in_set(name: str, color_set: set) -> bool:
    """Return True if *name* matches any entry in *color_set*.

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


def headers_in_order(
    header_names: List[str],
    color_names: List[str],
) -> Tuple[bool, str]:
    """Check that matrix column headers follow the colour-list order.

    For each non-empty header, finds the index of the colour it matches in
    *color_names*. Headers that match no colour are skipped — membership is
    the concern of a separate checkpoint step. The headers are in order iff
    the matched colour-list indices are strictly increasing.

    Args:
        header_names: Matrix column header cell values, left to right.
        color_names: The extracted colour list, in sheet order.

    Returns:
        Tuple ``(is_ordered, detail)``. ``is_ordered`` is True when fewer
        than two headers match a colour (order cannot be assessed).
    """
    # Prefer exact match; fall back to substring so 'Champagne Gold' resolves
    # to its own entry instead of collapsing onto an earlier 'Champagne'.
    normalised = [_normalize_color_name(c) for c in color_names]
    indexed: List[Tuple[str, int]] = []
    for h in header_names:
        if not h or not h.strip():
            continue
        nh = _normalize_color_name(h)
        idx = next((i for i, c in enumerate(normalised) if c == nh), None)
        if idx is None:
            idx = next(
                (i for i, c in enumerate(color_names) if color_names_match(h, c)),
                None,
            )
        if idx is not None:
            indexed.append((h, idx))

    if len(indexed) < 2:
        return True, (
            f"Only {len(indexed)} header(s) matched a colour - "
            f"order cannot be assessed."
        )

    for i in range(1, len(indexed)):
        if indexed[i][1] <= indexed[i - 1][1]:
            return False, (
                f"Header '{indexed[i][0]}' is out of order relative to "
                f"'{indexed[i - 1][0]}' (colour-list positions "
                f"{indexed[i - 1][1]} then {indexed[i][1]})."
            )

    return True, f"All {len(indexed)} matched headers follow the colour-list order."


# ---------------------------------------------------------------------------
# Colour list & decoration matrix discovery
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


def _scan_color_column(
    sheet_tab: Dict,
    col_idx: int,
    max_row: int,
) -> Optional[Dict[str, Any]]:
    """Scan a single column for a contiguous block of colour names."""
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


def find_color_region(
    sheet_tab: Dict,
    columns: Tuple[int, ...] = (0, 1),
    max_row: int = DEFAULT_COLOR_MAX_ROW,
) -> Optional[Dict[str, Any]]:
    """Locate the vertical colour-name list, scanning candidate columns in order.

    Scans each column in *columns* from row 0 downward, collecting non-empty
    values that are not URLs and not header labels. Stops at the first empty
    cell after a block has started. The first column that yields a block wins.

    Args:
        sheet_tab: Sheet tab dict.
        columns: Candidate column indices to try, in priority order.
        max_row: Stop scanning after this row.

    Returns:
        Dict with ``col``, ``start_row`` (inclusive), ``end_row`` (exclusive)
        and ``names``, or None if nothing found.
    """
    for col_idx in columns:
        region = _scan_color_column(sheet_tab, col_idx, max_row)
        if region is not None:
            return region
    return None


def find_color_list(
    sheet_tab: Dict,
    columns: Tuple[int, ...] = (0, 1),
    max_row: int = DEFAULT_COLOR_MAX_ROW,
) -> List[str]:
    """Read the vertical colour-name list from the top-left area of a sheet.

    Delegates to ``find_color_region`` and returns only the name strings.

    Args:
        sheet_tab: Sheet tab dict.
        columns: Candidate column indices to try, in priority order.
        max_row: Stop scanning after this row.

    Returns:
        Ordered list of colour name strings (may be empty).
    """
    region = find_color_region(sheet_tab, columns=columns, max_row=max_row)
    return region["names"] if region else []


def find_decoration_matrix(
    sheet_tab: Dict,
    color_names: List[str],
    search_start_row: int = 0,
    max_row: int = DEFAULT_MATRIX_MAX_ROW,
) -> Optional[Dict[str, Any]]:
    """Locate the decoration matrix region below the colour list.

    Scans rows from *search_start_row* for a header row where at least
    ``HEADER_MATCH_RATIO`` of the non-empty cells match a colour name. Data
    rows follow and end at the first fully-empty row.

    Args:
        sheet_tab: Sheet tab dict.
        color_names: Colour names extracted from the colour list.
        search_start_row: Row to begin the scan.
        max_row: Hard upper bound for the scan.

    Returns:
        Dict with ``header_row``, ``header_names``, ``data_start_row``,
        ``data_end_row``, ``label_col``, ``data_col_start``, ``data_col_end``,
        or None when no matching header row is found.
    """
    if not color_names:
        return None

    grid = sheet_tab.get("properties", {}).get("gridProperties", {})
    num_cols = grid.get("columnCount", 30)
    sheet_raw = {"sheets": [sheet_tab]}
    color_set = {_normalize_color_name(c) for c in color_names}

    for r_idx in range(search_start_row, max_row):
        cell_values = [
            (get_cell_value(sheet_raw, r_idx, c_idx) or "").strip()
            for c_idx in range(num_cols)
        ]

        non_empty = [v for v in cell_values if v]
        if len(non_empty) < 2:
            continue

        match_count = sum(
            1 for v in non_empty if fuzzy_match_color_in_set(v, color_set)
        )

        if match_count < 2 or match_count / len(non_empty) < HEADER_MATCH_RATIO:
            continue

        first_non_empty = next(
            (i for i, v in enumerate(cell_values) if v), 0
        )
        last_non_empty = max(
            (i for i, v in enumerate(cell_values) if v), default=0
        )

        if fuzzy_match_color_in_set(cell_values[first_non_empty], color_set):
            label_col = max(first_non_empty - 1, 0)
            data_col_start = first_non_empty
        else:
            label_col = first_non_empty
            data_col_start = first_non_empty + 1

        data_col_end = last_non_empty + 1

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
# Palette-tab helpers
# ---------------------------------------------------------------------------

def find_palette_label_col(
    palette_tab: Dict,
    palette_row_indices: List[int],
    max_col: int = PALETTE_LABEL_MAX_COL,
) -> Optional[int]:
    """Locate the palette-row label column (text-only, populated everywhere).

    The label column is the leftmost column where every palette row has a
    non-empty text value AND no background fill — distinguishing a label
    column (e.g. "Romantic Blush" / "Day Ceremony") from a colour cell.

    Args:
        palette_tab: The palette sheet tab dict.
        palette_row_indices: 0-based indices of the palette rows.
        max_col: Rightmost column to inspect (exclusive).

    Returns:
        Column index of the label column, or None if none qualifies.
    """
    if not palette_row_indices:
        return None
    sheet_raw = {"sheets": [palette_tab]}
    for c in range(max_col):
        ok = True
        for r in palette_row_indices:
            val = (get_cell_value(sheet_raw, r, c) or "").strip()
            if not val or cell_bg_hex(palette_tab, r, c) is not None:
                ok = False
                break
        if ok:
            return c
    return None


def find_checklist_tab(
    sheet_raw: Dict[str, Any],
    palette_tab: Optional[Dict] = None,
    title_keywords: Tuple[str, ...] = ("checklist", "planner", "plan", "decor"),
) -> Optional[Dict]:
    """Find a checklist / planner tab beyond the main sheet and palette tab.

    Prefers a tab whose title contains a checklist keyword; otherwise falls
    back to the first non-palette tab past the main sheet. Used by tasks
    that require a second new tab in addition to the palette tab.

    Args:
        sheet_raw: Full spreadsheet response from ``get_sheet_content()``.
        palette_tab: The palette tab dict (to exclude from candidates).
        title_keywords: Lowercase keywords to prefer in a tab title.

    Returns:
        The matching sheet tab dict, or None.
    """
    if not sheet_raw:
        return None
    sheets = sheet_raw.get("sheets", [])
    if len(sheets) < 3:
        return None
    palette_title = (palette_tab.get("properties", {}).get("title", "")
                     if palette_tab else "")
    candidates = [
        t for t in sheets[1:]
        if t.get("properties", {}).get("title", "") != palette_title
    ]
    for tab in candidates:
        title = tab.get("properties", {}).get("title", "").lower()
        if any(kw in title for kw in title_keywords):
            return tab
    return candidates[0] if candidates else None


def find_palette_tab(sheet_raw: Dict[str, Any]) -> Optional[Dict]:
    """Find a sheet tab whose name suggests it holds colour palettes.

    Skips the first tab (the main sheet) and prefers a title containing
    ``"palette"``; otherwise falls back to the second tab.

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

    for tab in sheets[1:]:
        title = tab.get("properties", {}).get("title", "")
        if "palette" in title.lower():
            return tab

    return sheets[1]


def collect_reference_bg_hexes(
    sheet_tab: Dict,
    num_rows: int,
    max_col: int = 10,
) -> Set[str]:
    """Collect every distinct non-white background hex from the first *num_rows*.

    Args:
        sheet_tab: Main sheet tab dict.
        num_rows: Number of rows to scan.
        max_col: Rightmost column to check (exclusive).

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
    """Euclidean distance between two ``#rrggbb`` colours in RGB space."""
    r1, g1, b1 = hex_to_rgb(hex1)
    r2, g2, b2 = hex_to_rgb(hex2)
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


def hex_matches_any(
    target: str,
    reference_set: Set[str],
    tolerance: float = DEFAULT_HEX_TOLERANCE,
) -> bool:
    """Return True if *target* is within *tolerance* of any colour in the set.

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


def _looks_like_color_text(text: str) -> bool:
    """Return True if a cell's text plausibly names/encodes a colour.

    Matches a hex code (``#rrggbb``) or a short, mostly-alphabetic token
    (1-3 words, e.g. "Dusty Rose") — the typical "just text" representation an
    agent uses when it types colours instead of filling cell backgrounds.
    """
    t = (text or "").strip()
    if not t:
        return False
    if _HEX_TEXT_RE.match(t):
        return True
    words = t.split()
    if 1 <= len(words) <= 3 and all(w.replace("-", "").isalpha() for w in words):
        return True
    return False


def find_palette_region(
    palette_tab: Dict,
    max_col: int = 30,
) -> List[Dict[str, Any]]:
    """Detect palette rows by content — background fill OR colour-like text.

    A colour-content cell is either background-filled or holds colour-like
    text. A leading label column (e.g. a palette-name column whose text reads
    like a colour name) is excluded: colour cells begin at the leftmost column
    that carries a background fill in any row, so any text-only column to its
    left is dropped. A row qualifies as a palette row if it has at least 2
    colour-content cells. Detecting rows this way lets the CP6 steps test
    row-count, per-row width and fill-vs-text independently.

    Args:
        palette_tab: The palette sheet tab dict.
        max_col: Rightmost column to inspect (exclusive).

    Returns:
        List of per-row dicts with ``row_idx``, ``content_cols``,
        ``filled_cols``, ``hex_list`` and ``text_cols``.
    """
    rows = get_row_data(palette_tab)
    sheet_raw = {"sheets": [palette_tab]}

    # Pass 1: record each cell's state (fill / colour-text) for every row.
    raw_rows: List[Tuple[int, Dict[int, Tuple[str, Optional[str]]]]] = []
    for r_idx in range(len(rows)):
        cells: Dict[int, Tuple[str, Optional[str]]] = {}
        for c in range(max_col):
            h = cell_bg_hex(palette_tab, r_idx, c)
            if h is not None:
                cells[c] = ("fill", h.lower())
            elif _looks_like_color_text(get_cell_value(sheet_raw, r_idx, c) or ""):
                cells[c] = ("text", None)
        raw_rows.append((r_idx, cells))

    # Leftmost column carrying a background fill anywhere — colour cells start
    # here; any text-only column to its left is a label column and is ignored.
    fill_cols = [c for _, cells in raw_rows
                 for c, (kind, _) in cells.items() if kind == "fill"]
    min_fill_col = min(fill_cols) if fill_cols else 0

    # Pass 2: build the region from colour columns only.
    region: List[Dict[str, Any]] = []
    for r_idx, cells in raw_rows:
        content_cols: List[int] = []
        filled_cols: List[int] = []
        hex_list: List[str] = []
        text_cols: List[int] = []
        for c in sorted(cells):
            if c < min_fill_col:
                continue  # leading label / non-colour column
            kind, h = cells[c]
            content_cols.append(c)
            if kind == "fill":
                filled_cols.append(c)
                hex_list.append(h)
            else:
                text_cols.append(c)
        if len(content_cols) >= 2:
            region.append({
                "row_idx": r_idx,
                "content_cols": content_cols,
                "filled_cols": filled_cols,
                "hex_list": hex_list,
                "text_cols": text_cols,
            })

    return region


# ---------------------------------------------------------------------------
# Generic URL / topic matching
# ---------------------------------------------------------------------------

def url_matches_topic(
    url: str,
    page_text: str,
    topic_keywords: Tuple[str, ...],
    domains: Optional[Tuple[str, ...]] = None,
    exclude_keywords: Optional[Tuple[str, ...]] = None,
    model: Any = None,
    topic_description: Optional[str] = None,
) -> bool:
    """Generic check: does a URL / page relate to a given topic?

    Strategy: (1) allow-listed domain hit, (2) topic-keyword hit with no
    excluded keyword present, (3) optional LLM fallback judging the page text.

    Args:
        url: The URL.
        page_text: Text content fetched from the page (may be empty). The
            actual page body when available, otherwise the page title.
        topic_keywords: Keywords identifying the topic.
        domains: Optional allow-list of domains that always pass.
        exclude_keywords: Keywords that disqualify a URL (e.g. a competing
            store type). A URL containing any of these never passes the
            keyword/domain heuristic.
        model: Optional LLM model for the fallback.
        topic_description: Human-readable topic for the LLM prompt.

    Returns:
        True if the URL plausibly relates to the topic.
    """
    combined = f"{url} {page_text or ''}".lower()
    domains = domains or ()
    exclude_keywords = exclude_keywords or ()

    has_exclude = any(x.lower() in combined for x in exclude_keywords)

    if not has_exclude:
        if any(d.lower() in combined for d in domains):
            return True
        if any(k.lower() in combined for k in topic_keywords):
            return True

    if model is not None:
        desc = topic_description or "the requested topic"
        excl_note = ""
        if exclude_keywords:
            excl_note = (
                f" Answer 'No' if the page is actually about "
                f"{', '.join(exclude_keywords)}."
            )
        snippet = (page_text or "").strip()
        prompt = (
            f"URL: {url}\n"
            f"{'Page content: ' + snippet[:3000] if snippet else '(page content unavailable)'}\n"
            f"Could this page reasonably be about {desc}?{excl_note} "
            f"Answer Yes or No."
        )
        # Judge twice; a lone dissenting pass is honoured so one noisy
        # false-negative cannot fail the link.
        votes = []
        for _ in range(2):
            try:
                votes.append(bool(evaluate_with_llm(prompt, model, return_type="bool")))
            except Exception as e:
                print(f"  LLM error in url_matches_topic: {e}")
        return any(votes)

    return False


def make_topic_matcher(
    topic_keywords: Tuple[str, ...],
    domains: Optional[Tuple[str, ...]] = None,
    exclude_keywords: Optional[Tuple[str, ...]] = None,
    topic_description: Optional[str] = None,
) -> Callable:
    """Build a ``relevance_fn`` closure for ``validate_and_match_urls``.

    Args:
        topic_keywords: Keywords identifying the topic.
        domains: Optional allow-list of domains.
        exclude_keywords: Optional disqualifying keywords.
        topic_description: Human-readable topic for the LLM prompt.

    Returns:
        Callable ``(url, page_text, model=None) -> bool``.
    """
    def _matcher(url: str, page_text: str, model: Any = None) -> bool:
        return url_matches_topic(
            url, page_text, topic_keywords,
            domains=domains, exclude_keywords=exclude_keywords,
            model=model, topic_description=topic_description,
        )
    return _matcher


def validate_and_match_urls(
    sheet_tab: Dict,
    color_names: List[str],
    col_idx: int,
    start_row: int,
    end_row: int,
    relevance_fn: Any,
    model: Any = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    precomputed_urls: Optional[List[str]] = None,
) -> List[str]:
    """Validate a column of URLs: liveness and content relevance.

    Presence/coverage checks are left to the caller so each evaluation
    criterion maps to exactly one checkpoint step.

    Args:
        sheet_tab: Sheet tab dict.
        color_names: Colour names corresponding to each row.
        col_idx: 0-based column index to scan for URLs.
        start_row: First row (inclusive, 0-based).
        end_row: Last row (exclusive, 0-based).
        relevance_fn: Callable ``(url, page_text, model=) -> bool``.
        model: Optional LLM model passed to *relevance_fn*.
        relevance_threshold: Minimum fraction of reachable URLs that must match.
        precomputed_urls: If provided, used directly instead of re-scanning the
            column (avoids a redundant second scan by the caller).

    Returns:
        List of failure description strings (empty if all checks pass).
        Strings contain ``"reachable"`` for liveness issues and ``"relevant"``
        for content-match issues so the caller can classify them.
    """
    failures: List[str] = []

    if precomputed_urls is not None:
        urls_found = list(precomputed_urls)
    else:
        urls_found = []
        rows = get_row_data(sheet_tab)
        for r_idx in range(start_row, end_row):
            found = find_urls_in_sheet(rows, start_row=r_idx, num_rows=1,
                                       start_col=col_idx, end_col=col_idx + 1)
            if found:
                urls_found.append(found[0])

    if not urls_found:
        failures.append(f"No URLs found in col {col_idx}")
        return failures

    # Phase 1: fetch real page content for each unique URL using browser-grade
    # fallbacks (requests -> Playwright -> Wayback -> archive.today ->
    # curl-cffi browser-TLS sweep). A URL is "reachable" only if its actual
    # content could be retrieved; a URL no strategy can fetch is genuinely
    # dead. This defeats bot-protection (Cloudflare 5xx, WAF 404s) that plain
    # requests cannot.
    unique_urls = list(dict.fromkeys(urls_found))
    fetch_tasks = [
        {"id": url, "func": fetch_with_fallbacks_extended,
         "args": (url,), "kwargs": {"max_chars": 4000}}
        for url in unique_urls
    ]
    fetched = parallel_execute(fetch_tasks, max_workers=10)

    content_by_url: Dict[str, str] = {}
    for url in unique_urls:
        result = fetched.get(url)
        content = result[0] if result else None
        if content:
            content_by_url[url] = content

    dead = [u for u in unique_urls if u not in content_by_url]
    dead_rows = sum(1 for u in urls_found if u not in content_by_url)
    print(
        f"  [DEBUG] {len(content_by_url)}/{len(unique_urls)} unique URLs "
        f"retrieved, {len(dead)} not reachable"
    )

    if dead:
        failures.append(
            f"{dead_rows}/{len(urls_found)} URLs are not reachable "
            f"(no content could be retrieved): {dead[:3]}"
        )

    # Phase 2: relevance — judge the actual fetched page content.
    reachable_urls = [u for u in urls_found if u in content_by_url]
    if reachable_urls:
        unique_reachable = list(dict.fromkeys(reachable_urls))

        def _check_relevance(url: str) -> bool:
            return relevance_fn(url, content_by_url[url], model=model)

        relevance_tasks = [
            {"id": url, "func": _check_relevance, "args": (url,)}
            for url in unique_reachable
        ]
        relevance_results = parallel_execute(relevance_tasks, max_workers=10)

        matched = sum(
            1 for url in reachable_urls if relevance_results.get(url)
        )

        print(f"  [DEBUG] {matched}/{len(reachable_urls)} URLs match relevance check")

        rel_ratio = matched / len(reachable_urls)
        if rel_ratio < relevance_threshold:
            failures.append(
                f"Only {matched}/{len(reachable_urls)} links lead to relevant "
                f"content ({rel_ratio:.0%})"
            )

    return failures


def grade_url_column(
    checkpoint_name: str,
    sheet_tab: Dict,
    color_region: Optional[Dict[str, Any]],
    col_offset: int,
    relevance_fn: Any,
    total_steps: int,
    step_names: List[str],
    model: Any = None,
) -> Checkpoint:
    """Shared logic for URL-column checkpoints (article links / store links).

    Scans the URL column once, then checks presence, per-colour coverage,
    liveness and relevance. With ``total_steps >= 4`` liveness and relevance
    are separate steps; otherwise they are combined into one.

    Args:
        checkpoint_name: Human-readable checkpoint name.
        sheet_tab: The main sheet tab dict.
        color_region: Region dict from ``find_color_region`` (or None).
        col_offset: Column offset from the colour-name column.
        relevance_fn: Callable ``(url, title, model=) -> bool``.
        total_steps: Number of evaluation steps for this checkpoint.
        step_names: Ordered step names mirroring checkpoints.md.
        model: Optional LLM model for relevance matching.

    Returns:
        Populated Checkpoint object.
    """
    start = time.time()
    checkpoint = Checkpoint(total=total_steps, result=0, name=checkpoint_name)

    if sheet_tab is None or color_region is None:
        return fail_all_steps(
            checkpoint, step_names, "No colour region available.", start
        )

    col_idx = color_region["col"] + col_offset
    names = color_region["names"]
    rows = get_row_data(sheet_tab)

    urls_found: List[str] = []
    missing_rows: List[str] = []
    for i, r_idx in enumerate(range(color_region["start_row"], color_region["end_row"])):
        found = find_urls_in_sheet(rows, start_row=r_idx, num_rows=1,
                                   start_col=col_idx, end_col=col_idx + 1)
        if found:
            urls_found.append(found[0])
        else:
            missing_rows.append(names[i] if i < len(names) else f"row {r_idx}")

    # Step 1: column contains URLs
    if not urls_found:
        return fail_all_steps(
            checkpoint, step_names, f"No URLs found in col {col_idx}.", start
        )
    checkpoint.add_step(
        step_names[0], True, 1,
        f"{len(urls_found)}/{len(names)} rows have a URL in col {col_idx}.",
    )

    # Step 2: each colour has a corresponding link
    if not missing_rows:
        checkpoint.add_step(
            step_names[1], True, 2, f"All {len(names)} colours have a link.",
        )
    else:
        checkpoint.add_step(
            step_names[1], False, 2,
            f"{len(missing_rows)}/{len(names)} colours missing links: "
            f"{missing_rows[:5]}",
        )

    # Step 3(+): liveness / relevance
    failures = validate_and_match_urls(
        sheet_tab=sheet_tab,
        color_names=names,
        col_idx=col_idx,
        start_row=color_region["start_row"],
        end_row=color_region["end_row"],
        relevance_fn=relevance_fn,
        model=model,
        precomputed_urls=urls_found,
    )
    liveness_failures = [f for f in failures if "reachable" in f]
    relevance_failures = [f for f in failures if "relevant" in f]

    if total_steps >= 4:
        if not liveness_failures:
            checkpoint.add_step(
                step_names[2], True, 3,
                f"All {len(urls_found)} URLs are reachable.",
            )
        else:
            checkpoint.add_step(step_names[2], False, 3, "; ".join(liveness_failures))
        if not relevance_failures:
            checkpoint.add_step(
                step_names[3], True, 4, "Links lead to relevant content.",
            )
        else:
            checkpoint.add_step(step_names[3], False, 4, "; ".join(relevance_failures))
    else:
        all_failures = liveness_failures + relevance_failures
        if not all_failures:
            checkpoint.add_step(
                step_names[2], True, 3,
                f"All {len(urls_found)} URLs are reachable and relevant.",
            )
        else:
            checkpoint.add_step(step_names[2], False, 3, "; ".join(all_failures))

    checkpoint.execution_time = time.time() - start
    return checkpoint


# ---------------------------------------------------------------------------
# colorhexa.com lookup & swatch rendering (checkpoint 3)
# ---------------------------------------------------------------------------

def lookup_colorhexa_name(hex_str: str) -> Optional[str]:
    """Return colorhexa.com's canonical colour name for a hex value, or None.

    colorhexa serves hex pages (e.g. ``/000080``) titled like
    ``"Navy blue / #000080 hex color"``. A browser User-Agent is required —
    colorhexa returns HTTP 403 without one. colorhexa has no name->hex pages,
    so the lookup direction is hex -> name only.

    Args:
        hex_str: A ``#rrggbb`` (or ``rrggbb``) hex string.

    Returns:
        The canonical colour name, or None on failure.
    """
    if not hex_str:
        return None
    h = hex_str.lstrip("#").strip().lower()
    if len(h) != 6:
        return None
    title = fetch_page_title(
        f"https://www.colorhexa.com/{h}",
        headers={"User-Agent": _BROWSER_UA},
    )
    if not title:
        return None
    name = title.split("/")[0].strip()
    return name or None


def render_color_swatch(hex_str: str, out_dir: str, size: int = 128) -> Optional[str]:
    """Write a solid-colour PNG swatch for *hex_str*; return its path or None.

    Args:
        hex_str: A ``#rrggbb`` hex string.
        out_dir: Directory to write the PNG into (created if needed).
        size: Square swatch side length in pixels.

    Returns:
        Path to the written PNG, or None on failure.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        rgb = hex_to_rgb(hex_str)
    except (ValueError, IndexError):
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"swatch_{hex_str.lstrip('#').lower()}.png")
        Image.new("RGB", (size, size), rgb).save(path)
        return path
    except Exception as e:
        print(f"  swatch render failed for {hex_str}: {e}")
        return None


# ---------------------------------------------------------------------------
# Task-config extraction from task.md
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUM = (
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
)
_TASK_CONFIG_INT_FIELDS = (
    "min_articles", "min_colors", "min_decoration_types",
    "min_palette_rows", "palette_cells_per_row",
)


def _parse_number_token(token: Any) -> Optional[int]:
    """Parse a positive int from a digit string, int, or English number word."""
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token if token > 0 else None
    if not isinstance(token, str):
        return None
    t = token.strip().lower()
    if t.isdigit():
        n = int(t)
        return n if n > 0 else None
    return _NUMBER_WORDS.get(t)


def _split_category_list(text: str) -> List[str]:
    """Split a 'X, Y, or Z' colour-category phrase into a clean lowercase list."""
    parts = re.split(r",|\bor\b|\band\b", text)
    cats: List[str] = []
    for p in parts:
        p = p.strip().strip(".").lower()
        if p and p not in ("either", "neither"):
            cats.append(p)
    return cats


def _coerce_categories(value: Any) -> Optional[Tuple[str, ...]]:
    """Coerce an LLM/regex colour-category value into a clean tuple, or None."""
    if not value:
        return None
    if isinstance(value, str):
        items = _split_category_list(value)
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value if str(v).strip()]
    else:
        return None
    items = [i for i in items if i and i not in ("either", "neither", "or", "and")]
    return tuple(items) if items else None


def _regex_task_config(text: str) -> Dict[str, Any]:
    """Best-effort regex extraction of config values from task.md prose."""
    t = " ".join(text.lower().split())
    cfg: Dict[str, Any] = {}

    m = re.search(rf"{_NUM}\s+articles?", t)
    if m:
        cfg["min_articles"] = _parse_number_token(m.group(1))

    m = re.search(rf"{_NUM}\s+(?:[a-z]+\s+){{0,3}}shades?", t)
    if m:
        cfg["min_colors"] = _parse_number_token(m.group(1))

    m = re.search(rf"{_NUM}\s+(?:wedding\s+)?(?:objects?|decorations?)", t)
    if m:
        cfg["min_decoration_types"] = _parse_number_token(m.group(1))

    m = re.search(rf"{_NUM}\s+(?:[a-z]+\s+){{0,2}}(?:rows|palettes?)", t)
    if m:
        cfg["min_palette_rows"] = _parse_number_token(m.group(1))

    m = re.search(rf"choose\s+{_NUM}\s+colou?rs", t)
    if not m:
        m = re.search(rf"{_NUM}[-\s]colou?r\s+palettes?", t)
    if m:
        cfg["palette_cells_per_row"] = _parse_number_token(m.group(1))

    m = re.search(r"shades? of (?:either\s+)?([^.]+?)\.", t)
    if m:
        cats = _split_category_list(m.group(1))
        if cats:
            cfg["color_categories"] = cats

    return cfg


def extract_task_config(task_md_path: str, model: Any = None) -> Dict[str, Any]:
    """Extract evaluator config values from a task.md description.

    Resolves each field LLM-first, then regex. Unresolved fields are returned
    as None so the caller can fail the dependent step with a clear reason
    instead of guessing. Never raises.

    Args:
        task_md_path: Path to the instance's task.md file.
        model: Optional LLM model for the primary extraction tier.

    Returns:
        Dict with keys ``color_categories`` (tuple of str, or None) and the
        integer fields ``min_articles``, ``min_colors``,
        ``min_decoration_types``, ``min_palette_rows``,
        ``palette_cells_per_row`` (int, or None).
    """
    config: Dict[str, Any] = {
        "color_categories": None,
        "min_articles": None,
        "min_colors": None,
        "min_decoration_types": None,
        "min_palette_rows": None,
        "palette_cells_per_row": None,
    }

    try:
        with open(task_md_path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"  [WARN] could not read task.md ({task_md_path}): {e}")
        return config

    # Tier 1: LLM extraction.
    llm_cfg: Dict[str, Any] = {}
    if model is not None and text.strip():
        prompt = (
            "Read the task description below and extract these values. Return "
            "ONLY a JSON object with exactly these keys:\n"
            '  "color_categories": array of colour family name strings\n'
            '  "min_articles": integer (minimum source articles to find)\n'
            '  "min_colors": integer (minimum distinct colour shades)\n'
            '  "min_decoration_types": integer (decoration objects/types)\n'
            '  "min_palette_rows": integer (rows in the palette tab)\n'
            '  "palette_cells_per_row": integer (colours per palette row)\n'
            "Use null for any value not stated.\n\n"
            f"Task description:\n{text}"
        )
        try:
            parsed = extract_json_with_llm(prompt, model, expect_type="object")
            if isinstance(parsed, dict):
                llm_cfg = parsed
        except Exception as e:
            print(f"  [WARN] LLM task-config extraction failed: {e}")

    # Tier 2: regex extraction (per-field fallback).
    regex_cfg = _regex_task_config(text)

    cats = _coerce_categories(llm_cfg.get("color_categories"))
    if cats is None:
        cats = _coerce_categories(regex_cfg.get("color_categories"))
    config["color_categories"] = cats

    for key in _TASK_CONFIG_INT_FIELDS:
        val = _parse_number_token(llm_cfg.get(key))
        if val is None:
            val = _parse_number_token(regex_cfg.get(key))
        config[key] = val

    return config
