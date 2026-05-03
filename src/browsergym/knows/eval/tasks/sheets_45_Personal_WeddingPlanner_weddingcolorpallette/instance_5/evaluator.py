"""Evaluator for the Wedding Color Palette Google Sheets task (instance 5).

Beach / coastal variant: coral / turquoise / sand, fabric/textile store links
instead of paint, beach-specific decorations, Day/Evening palette labels, and a
decoration-palette checklist tab.  28 points across 7 checkpoints.
"""

import os
import shutil
import sys
from typing import Any, List, Optional
import time
import traceback
import argparse

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports from eval_utils
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import get_sheet_content, find_urls_in_sheet
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import fast_parallel_vlm_calls, parallel_download


# Local utils
from src.browsergym.knows.eval.tasks.sheets_45_Personal_WeddingPlanner_weddingcolorpallette.utils import (
    find_color_list,
    find_color_region,
    find_decoration_matrix,
    read_column_values,
    detect_any_image,
    cell_bg_hex,
    find_palette_tab,
    collect_reference_bg_hexes,
    count_filled_cells_in_row,
    hex_matches_any,
    hex_to_rgb,
    validate_and_match_urls,
    _normalize_color_name,
    url_matches_wedding_article,
)
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url, fetch_page_title

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# VLM / LLM model (loaded lazily on first use)
model = None
model_id = "gemini-2.5-flash-google-ai"

# ----- Instance 5 specific configuration -----
COLOR_CATEGORIES = ("coral", "turquoise", "sand")
CATEGORY_LABEL = "/".join(COLOR_CATEGORIES)
MIN_ARTICLES = 2
MIN_COLORS = 10
TOP_LEFT_MAX_ROW = 15
MIN_DECORATION_TYPES = 5
MIN_PALETTE_ROWS = 10
PALETTE_CELLS_PER_ROW = 3
PALETTE_LABELS = ("day ceremony", "evening reception")

# Global variables
sheet_id = None
sheet_raw = None
main_tab = None
color_region = None  # cached by grade_checkpoint_1 for reuse
palette_tab_global = None  # cached by grade_checkpoint_6 so CP7 can skip it


def setup(workspace_doc_id: str):
    """Setup function to initialize the evaluator."""
    global sheet_id, sheet_raw, main_tab, color_region, palette_tab_global

    sheet_id = workspace_doc_id
    print(f"Using workspace document ID: {sheet_id}")
    sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

    if sheet_raw and sheet_raw.get("sheets"):
        main_tab = sheet_raw["sheets"][0]
        tab_title = main_tab.get("properties", {}).get("title", "")
        print(f"Main tab: '{tab_title}'")
    else:
        print("ERROR: could not fetch sheet data")
        main_tab = None

    color_region = None
    palette_tab_global = None


# ---------------------------------------------------------------------------
# Local helper: fabric/textile store URL relevance
# ---------------------------------------------------------------------------

_FABRIC_KEYWORDS = (
    "fabric", "textile", "joann", "spoonflower", "moodfabrics", "fabric.com",
    "michaels", "hobbylobby", "robertkaufman", "moda", "kaufman", "dharma",
    "etsy", "amazon", "linen", "silk", "cotton", "polyester", "rayon",
    "upholstery",
)


def url_matches_fabric_store(url: str, page_title: str, model: Any = None) -> bool:
    """True if a URL plausibly leads to a fabric/textile store or color page.

    Keyword match first, LLM fallback otherwise.  Explicitly excludes URLs that
    look like they belong to a paint store.
    """
    combined = f"{url} {page_title}".lower()

    paint_only = ("paint", "sherwin", "behr", "valspar", "benjamin moore", "dulux", "farrow")
    is_paint = any(p in combined for p in paint_only)
    has_fabric = any(kw in combined for kw in _FABRIC_KEYWORDS)

    if has_fabric and not is_paint:
        return True

    if model is not None:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You verify whether a URL leads to a FABRIC or TEXTILE "
                    "store/page (e.g. Joann, Spoonflower, Mood Fabrics, "
                    "Etsy fabric listings) — NOT a paint store. Answer 'Yes' "
                    "for any fabric/textile/material color reference. Answer "
                    "'No' for paint stores, drink stores, or unrelated pages."
                )}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": (
                    f"URL: {url}\n"
                    f"{'Page title: ' + page_title if page_title else '(page title unavailable)'}\n"
                    f"Is this a fabric or textile store / color page (not a "
                    f"paint store)? Answer Yes or No."
                )}],
            },
        ]
        try:
            response = model(messages)
            if response and "yes" in response.lower():
                return True
        except Exception as e:
            print(f"  LLM error for fabric matching: {e}")

    return False


def grade_checkpoint_1(browsing_history: Optional[List[str]] = None):
    """
    Checkpoint 1: Color Extraction (5 pts)
    The agent found beach/coastal wedding color information from at least two
    articles and color names are listed vertically in the top-left area.

    Outcome Evaluation:
    - All extracted color names appear in a vertical list in a single column.
    - The color list starts in the top-left area of the sheet (rows 1-15).
    - At least 10 unique color names/shades are present.
    - The agent searched for and found at least 2 articles about beach or
      coastal wedding colors.
    - All extracted colors belong to coral, turquoise, or sand categories.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global color_region
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Color Extraction")

    if main_tab is None:
        checkpoint.add_step(
            "Sheet Data Available", False, 1,
            "Sheet data unavailable.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    region = find_color_region(main_tab, col_idx=0, max_row=30)
    if region is None:
        region = find_color_region(main_tab, col_idx=1, max_row=30)

    color_region = region

    if region is None:
        checkpoint.add_step(
            "Vertical Color List", False, 1,
            "No contiguous colour-name block found in columns A or B.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_colors = len(region["names"])
    print(
        f"  [DEBUG] Color list in col {region['col']}, "
        f"rows {region['start_row']}–{region['end_row'] - 1} ({num_colors} names)"
    )

    checkpoint.add_step(
        "Vertical Color List", True, 1,
        f"Color list found in col {region['col']}, "
        f"rows {region['start_row']}–{region['end_row'] - 1}.",
    )

    if region["start_row"] < TOP_LEFT_MAX_ROW:
        checkpoint.add_step(
            "Top-Left Placement", True, 1,
            f"Color list starts at row {region['start_row']}.",
        )
    else:
        checkpoint.add_step(
            "Top-Left Placement", False, 1,
            f"Color list starts at row {region['start_row']} "
            f"(expected < {TOP_LEFT_MAX_ROW}).",
        )

    if num_colors >= MIN_COLORS:
        checkpoint.add_step(
            f"Minimum {MIN_COLORS} Unique Colors", True, 1,
            f"{num_colors} unique color names found.",
        )
    else:
        checkpoint.add_step(
            f"Minimum {MIN_COLORS} Unique Colors", False, 1,
            f"Only {num_colors} unique color names found "
            f"(expected >= {MIN_COLORS}): {region['names']}",
        )

    history = browsing_history or []
    wedding_urls: list = []
    for url in history:
        title = fetch_page_title(url) or ""
        if url_matches_wedding_article(url, title):
            wedding_urls.append(url)
            if len(wedding_urls) >= MIN_ARTICLES:
                break

    if len(wedding_urls) >= MIN_ARTICLES:
        checkpoint.add_step(
            "Article Research", True, 1,
            f"{len(wedding_urls)} wedding-color articles found in "
            f"{len(history)} browsing-history entries.",
        )
    else:
        checkpoint.add_step(
            "Article Research", False, 1,
            f"Only {len(wedding_urls)} wedding-color article(s) found in "
            f"{len(history)} browsing-history entries (expected >= {MIN_ARTICLES}).",
        )

    # Step 5: Colors belong to coral / turquoise / sand (LLM batch judge)
    categorised = 0
    still_uncategorised: List[str] = list(region["names"])
    if num_colors > 0:
        global model
        if model is None:
            model = load_model(model_id)

        cat_upper = ", ".join(c.upper() for c in COLOR_CATEGORIES)
        cat_lower = ", ".join(COLOR_CATEGORIES)
        llm_tasks = []
        for name in region["names"]:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": (
                        "You are a color classification assistant. Given a color "
                        f"name, determine if it belongs to the {cat_upper} "
                        "color family. Answer 'Yes' if it belongs to any of these "
                        "families, or 'No' if it does not."
                    )}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"Does the color '{name}' belong to the {cat_lower} "
                        f"color family? Answer Yes or No."
                    )}],
                },
            ]
            llm_tasks.append({"id": name, "messages": messages})

        print(f"  Running {len(llm_tasks)} LLM color-category checks…")
        llm_results = fast_parallel_vlm_calls(llm_tasks, model, max_workers=10)
        categorised = sum(1 for v in llm_results.values() if v)
        still_uncategorised = [
            n for n in region["names"] if not llm_results.get(n, False)
        ]

    if num_colors > 0:
        ratio = categorised / num_colors
        if ratio >= 0.8:
            checkpoint.add_step(
                "Color Category Match", True, 1,
                f"{categorised}/{num_colors} colors classified into "
                f"{CATEGORY_LABEL} ({ratio:.0%}).",
            )
        else:
            checkpoint.add_step(
                "Color Category Match", False, 1,
                f"Only {categorised}/{num_colors} colors classified into "
                f"{CATEGORY_LABEL} ({ratio:.0%}). Unrecognised: {still_uncategorised[:5]}",
            )
    else:
        checkpoint.add_step(
            "Color Category Match", False, 1,
            "No colors found to categorise.",
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def _grade_url_column(
    checkpoint_name: str,
    col_offset: int,
    relevance_fn,
    total_steps: int = 4,
) -> "Checkpoint":
    """Shared logic for URL-bearing columns (CP2, CP4)."""
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=total_steps, result=0, name=checkpoint_name)

    if main_tab is None or color_region is None:
        checkpoint.add_step(
            "Data Available", False, 1,
            "No color region available.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    col_idx = color_region["col"] + col_offset
    names = color_region["names"]
    rows = main_tab.get("data", [{}])[0].get("rowData", [])

    urls_found: list = []
    missing_rows: list = []
    for i, r_idx in enumerate(range(color_region["start_row"], color_region["end_row"])):
        found = find_urls_in_sheet(rows, start_row=r_idx, num_rows=1,
                                   start_col=col_idx, end_col=col_idx + 1)
        raw_url = found[0] if found else None
        cname = names[i] if i < len(names) else "?"
        print(f"  [DEBUG-URL] {cname}: {raw_url}")
        if raw_url:
            urls_found.append(raw_url)
        else:
            missing_rows.append(cname)

    if urls_found:
        checkpoint.add_step(
            "URLs Present", True, 1,
            f"{len(urls_found)}/{len(names)} rows have a URL in col {col_idx}.",
        )
    else:
        checkpoint.add_step(
            "URLs Present", False, 1,
            f"No URLs found in col {col_idx}.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if not missing_rows:
        checkpoint.add_step(
            "Full Coverage", True, 1,
            f"All {len(names)} colors have a link.",
        )
    else:
        checkpoint.add_step(
            "Full Coverage", False, 1,
            f"{len(missing_rows)}/{len(names)} colors missing links: "
            f"{missing_rows[:5]}",
        )

    global model
    if model is None:
        model = load_model(model_id)

    failures = validate_and_match_urls(
        sheet_tab=main_tab,
        color_names=names,
        col_idx=col_idx,
        start_row=color_region["start_row"],
        end_row=color_region["end_row"],
        relevance_fn=relevance_fn,
        model=model,
    )

    liveness_failures = [f for f in failures if "reachable" in f or "invalid" in f]
    relevance_failures = [f for f in failures if "relevant" in f]

    if total_steps >= 4:
        if not liveness_failures:
            checkpoint.add_step(
                "Links Functional", True, 1,
                f"All {len(urls_found)} URLs are reachable.",
            )
        else:
            checkpoint.add_step(
                "Links Functional", False, 1,
                "; ".join(liveness_failures),
            )

        if not relevance_failures:
            checkpoint.add_step(
                "Content Relevance", True, 1,
                "Links lead to relevant content.",
            )
        else:
            checkpoint.add_step(
                "Content Relevance", False, 1,
                "; ".join(relevance_failures),
            )
    else:
        all_failures = liveness_failures + relevance_failures
        if not all_failures:
            checkpoint.add_step(
                "Links Functional & Relevant", True, 1,
                f"All {len(urls_found)} URLs are reachable and lead to relevant content.",
            )
        else:
            checkpoint.add_step(
                "Links Functional & Relevant", False, 1,
                "; ".join(all_failures),
            )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2: Article Source Links (4 pts)."""
    print("----------------- CHECKPOINT 2 ----------------")
    return _grade_url_column(
        "Article Source Links",
        col_offset=1,
        relevance_fn=url_matches_wedding_article,
        total_steps=4,
    )


def grade_checkpoint_3():
    """Checkpoint 3: Color Cell Formatting (4 pts)."""
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Color Cell Formatting")

    if main_tab is None or color_region is None:
        checkpoint.add_step(
            "Data Available", False, 1,
            "No color region available.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    fill_col = color_region["col"] + 2
    sr = color_region["start_row"]
    er = color_region["end_row"]
    names = color_region["names"]

    filled_count = 0
    hex_set = set()
    missing_fills = []
    color_hex_pairs = []
    for i, r_idx in enumerate(range(sr, er)):
        h = cell_bg_hex(main_tab, r_idx, fill_col)
        if h:
            filled_count += 1
            hex_set.add(h.lower())
            if i < len(names):
                color_hex_pairs.append((names[i], h))
        else:
            missing_fills.append(names[i] if i < len(names) else f"row {r_idx}")

    print(
        f"  [DEBUG] {filled_count}/{len(names)} cells have bg fill, "
        f"{len(hex_set)} distinct hex values"
    )

    if filled_count > 0:
        checkpoint.add_step(
            "Fills Present", True, 1,
            f"{filled_count}/{len(names)} cells have background fill "
            f"({len(hex_set)} distinct hex values).",
        )
    else:
        checkpoint.add_step(
            "Fills Present", False, 1,
            f"No background fills found in col {fill_col}.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if not missing_fills:
        checkpoint.add_step(
            "Full Coverage", True, 1,
            f"All {len(names)} colors have a corresponding fill.",
        )
    else:
        checkpoint.add_step(
            "Full Coverage", False, 1,
            f"{len(missing_fills)}/{len(names)} colors missing fills: "
            f"{missing_fills[:5]}",
        )

    global model
    if model is None:
        model = load_model(model_id)

    if color_hex_pairs:
        hex_tasks = []
        for idx, (cname, chex) in enumerate(color_hex_pairs):
            r, g, b = hex_to_rgb(chex)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": (
                        "You are a color matching expert. You will be given a "
                        "color name and a color described by its hex code and "
                        "RGB values. Determine if the given color is a reasonable "
                        "representation of the named color. Be lenient — if the "
                        "color is in the right general family answer 'Yes'. Only "
                        "answer 'No' if the color is clearly wrong."
                    )}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"Color name: '{cname}'\n"
                        f"Hex: {chex}, RGB: ({r}, {g}, {b})\n"
                        f"Does this color reasonably match the name? "
                        f"Answer Yes or No."
                    )}],
                },
            ]
            hex_tasks.append({"id": f"{idx}_{cname}", "messages": messages})

        print(f"  Running {len(hex_tasks)} hex-to-name LLM checks…")
        hex_results = fast_parallel_vlm_calls(hex_tasks, model, max_workers=10)

        matched = sum(1 for v in hex_results.values() if v)
        mismatched = [
            tid.split("_", 1)[1] for tid, v in hex_results.items() if not v
        ]

        match_ratio = matched / len(color_hex_pairs) if color_hex_pairs else 0
        if match_ratio >= 0.7:
            checkpoint.add_step(
                "Hex Color Match", True, 1,
                f"{matched}/{len(color_hex_pairs)} hex colors match "
                f"their named colors ({match_ratio:.0%}).",
            )
        else:
            checkpoint.add_step(
                "Hex Color Match", False, 1,
                f"Only {matched}/{len(color_hex_pairs)} hex colors match "
                f"their named colors ({match_ratio:.0%}). "
                f"Mismatched: {mismatched[:5]}",
            )
    else:
        checkpoint.add_step(
            "Hex Color Match", False, 1,
            "No color-hex pairs to verify.",
        )

    min_distinct = min(3, len(names))
    if len(hex_set) >= min_distinct:
        checkpoint.add_step(
            "Distinct Colors", True, 1,
            f"{len(hex_set)} distinct hex values (>= {min_distinct}).",
        )
    else:
        checkpoint.add_step(
            "Distinct Colors", False, 1,
            f"Only {len(hex_set)} distinct hex values (expected >= {min_distinct}).",
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4: Fabric/Textile Store References (3 pts)
    Fabric or textile store links are provided in the fourth column next to
    each color (replacing the seed instance's paint-store check).
    """
    print("----------------- CHECKPOINT 4 ----------------")
    return _grade_url_column(
        "Fabric/Textile Store References",
        col_offset=3,
        relevance_fn=url_matches_fabric_store,
        total_steps=3,
    )


def grade_checkpoint_5():
    """
    Checkpoint 5: Beach Wedding Decoration Matrix (4 pts)
    A beach-specific wedding decoration matrix is created below the color
    list with images.

    Outcome Evaluation:
    - At least 5 types of beach wedding decorations are listed in the
      leftmost column (LLM judge for beach-wedding-specific items).
    - Column headers contain the same color names from the original list.
    - At least half of the matrix cells contain images.
    - Images show the specified decoration type in the corresponding color.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Beach Decoration Matrix")

    if main_tab is None:
        checkpoint.add_step(
            "Data Available", False, 1,
            "Sheet data unavailable.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    color_names = find_color_list(main_tab)
    print(
        f"  [DEBUG] Colour list ({len(color_names)} names): "
        f"{color_names[:8]}{'…' if len(color_names) > 8 else ''}"
    )

    if not color_names:
        checkpoint.add_step(
            "Beach Decoration Types", False, 1,
            "Could not find colour list in column A – matrix search aborted.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    search_start = len(color_names) + 1
    matrix = find_decoration_matrix(
        main_tab, color_names, search_start_row=search_start
    )

    if matrix is None:
        checkpoint.add_step(
            "Beach Decoration Types", False, 1,
            f"No decoration matrix found below row {search_start}.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    print(
        f"  [DEBUG] Matrix detected – header row {matrix['header_row']}, "
        f"data rows {matrix['data_start_row']}–{matrix['data_end_row']}, "
        f"label col {matrix['label_col']}, "
        f"data cols {matrix['data_col_start']}–{matrix['data_col_end']}"
    )

    decoration_labels = read_column_values(
        main_tab,
        col_idx=matrix["label_col"],
        start_row=matrix["data_start_row"],
        end_row=matrix["data_end_row"],
    )
    decoration_types = [d for d in decoration_labels if d.strip()]

    # Step 1: ≥5 decorations AND beach-specific (combined per checkpoint outcome)
    beach_specific_count = 0
    non_beach: List[str] = []
    if decoration_types:
        if model is None:
            model = load_model(model_id)
        llm_tasks = []
        for idx, deco in enumerate(decoration_types):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": (
                        "You judge whether a wedding decoration item is "
                        "specifically suited to a BEACH or COASTAL wedding. "
                        "Examples that count as beach-specific: centerpieces, "
                        "bridesmaid dresses, table runners, shell decorations, "
                        "invitations themed for the beach, sand ceremony "
                        "vases, driftwood arches, conch favors, beach welcome "
                        "signs. Generic wedding items still get 'Yes' if they "
                        "are commonly used at beach weddings. Answer 'No' "
                        "only for items that are clearly indoor/winter/snow "
                        "themed and would not appear at a beach wedding."
                    )}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"Decoration item: '{deco}'\n"
                        f"Is this item suitable for a beach/coastal wedding? "
                        f"Answer Yes or No."
                    )}],
                },
            ]
            llm_tasks.append({"id": f"{idx}_{deco}", "messages": messages})

        print(f"  Running {len(llm_tasks)} LLM beach-decoration checks…")
        llm_results = fast_parallel_vlm_calls(llm_tasks, model, max_workers=10)
        beach_specific_count = sum(1 for v in llm_results.values() if v)
        non_beach = [
            tid.split("_", 1)[1]
            for tid, v in llm_results.items()
            if not v
        ]

    enough_decorations = len(decoration_types) >= MIN_DECORATION_TYPES
    enough_beach = (
        len(decoration_types) > 0
        and beach_specific_count / len(decoration_types) >= 0.6
    )

    if enough_decorations and enough_beach:
        checkpoint.add_step(
            "Beach Decoration Types", True, 1,
            f"{len(decoration_types)} decorations, {beach_specific_count} "
            f"beach-specific: {decoration_types[:5]}",
        )
    elif not enough_decorations:
        checkpoint.add_step(
            "Beach Decoration Types", False, 1,
            f"Only {len(decoration_types)} decoration type(s) found "
            f"(expected >= {MIN_DECORATION_TYPES}): {decoration_types}",
        )
    else:
        checkpoint.add_step(
            "Beach Decoration Types", False, 1,
            f"Only {beach_specific_count}/{len(decoration_types)} decorations "
            f"are beach-specific. Non-beach: {non_beach[:5]}",
        )

    # Step 2: Header colour names match
    header_names = matrix["header_names"]
    color_set = {_normalize_color_name(c) for c in color_names}
    non_empty_headers = [h for h in header_names if h.strip()]

    matched_headers = []
    unmatched_headers = []
    for h in non_empty_headers:
        if _normalize_color_name(h) in color_set:
            matched_headers.append(h)
        else:
            unmatched_headers.append(h)

    if unmatched_headers:
        if model is None:
            model = load_model(model_id)

        llm_tasks = []
        color_list_str = ", ".join(sorted(color_set))
        for h in unmatched_headers:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": (
                        "You are a color name matching assistant. Determine if "
                        "a given color name refers to the same color as any "
                        "name in a reference list. Answer 'Yes' if it matches "
                        "any, 'No' otherwise."
                    )}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"Color name: '{h}'\n"
                        f"Reference list: [{color_list_str}]\n"
                        f"Does '{h}' match any color in the reference list? "
                        f"Answer Yes or No."
                    )}],
                },
            ]
            llm_tasks.append({"id": h, "messages": messages})

        print(f"  Running {len(llm_tasks)} LLM header-match checks…")
        llm_results = fast_parallel_vlm_calls(llm_tasks, model, max_workers=10)

        still_unmatched = []
        for h in unmatched_headers:
            if llm_results.get(h, False):
                matched_headers.append(h)
            else:
                still_unmatched.append(h)
        unmatched_headers = still_unmatched

    matched_header_count = len(matched_headers)

    if matched_header_count > 0 and not unmatched_headers:
        checkpoint.add_step(
            "Header Color Match", True, 1,
            f"{matched_header_count} header colours match the original list.",
        )
    elif unmatched_headers:
        checkpoint.add_step(
            "Header Color Match", False, 1,
            f"Matrix headers not in colour list: {sorted(unmatched_headers)}",
        )
    else:
        checkpoint.add_step(
            "Header Color Match", False, 1,
            "No header colour names found in matrix.",
        )

    # Step 3: At least half of grid cells contain images
    total_cells = 0
    image_cells = 0
    image_info = []
    for r_idx in range(matrix["data_start_row"], matrix["data_end_row"]):
        row_offset = r_idx - matrix["data_start_row"]
        decoration = (
            decoration_types[row_offset]
            if row_offset < len(decoration_types)
            else "decoration"
        )
        for c_idx in range(matrix["data_col_start"], matrix["data_col_end"]):
            total_cells += 1
            url = detect_any_image(main_tab, r_idx, c_idx)
            if url:
                image_cells += 1
                col_offset = c_idx - matrix["data_col_start"]
                color_name = (
                    header_names[col_offset]
                    if col_offset < len(header_names)
                    else "unknown"
                )
                image_info.append((url, decoration, color_name))

    if total_cells > 0 and image_cells / total_cells >= 0.5:
        checkpoint.add_step(
            "Image Coverage", True, 1,
            f"{image_cells}/{total_cells} cells contain images "
            f"({image_cells / total_cells:.0%}).",
        )
    elif total_cells == 0:
        checkpoint.add_step(
            "Image Coverage", False, 1,
            "Matrix body has 0 cells.",
        )
    else:
        checkpoint.add_step(
            "Image Coverage", False, 1,
            f"Only {image_cells}/{total_cells} cells contain images "
            f"({image_cells / total_cells:.0%}, expected >= 50%).",
        )

    # Step 4: VLM judge – images show correct decoration in correct colour
    if image_info:
        if model is None:
            model = load_model(model_id)

        temp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "temp_vlm_images"
        )
        os.makedirs(temp_dir, exist_ok=True)

        try:
            max_vlm_checks = 15
            if len(image_info) > max_vlm_checks:
                step = len(image_info) / max_vlm_checks
                sampled = [image_info[int(i * step)] for i in range(max_vlm_checks)]
            else:
                sampled = image_info

            dl_tasks = []
            for idx, (img_url, _, _) in enumerate(sampled):
                dl_tasks.append({
                    "id": f"img_{idx}",
                    "func": download_image_from_url,
                    "args": (img_url, temp_dir),
                })

            downloaded = parallel_download(
                dl_tasks, max_workers=5, use_rate_limit=False
            )

            vlm_tasks = []
            for idx, (_, decoration, color_name) in enumerate(sampled):
                img_path = downloaded.get(f"img_{idx}")
                if img_path and os.path.exists(img_path):
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": (
                                "You are evaluating images in a beach wedding "
                                "decoration matrix. Given an image, a decoration "
                                "type, and a color, determine if the image is "
                                "reasonably related to the decoration type and "
                                "color. Be lenient — answer 'Yes' if the image "
                                "is broadly related to the decoration category "
                                "and the color is present or plausible. Only "
                                "answer 'No' if the image is clearly unrelated "
                                "or the color is completely wrong."
                            )}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": (
                                    f"Decoration type: {decoration}\n"
                                    f"Expected color: {color_name}\n"
                                    f"Is this image reasonably related to a "
                                    f"{decoration} in or featuring {color_name}? "
                                    f"Answer Yes or No."
                                )},
                            ],
                        },
                    ]
                    vlm_tasks.append({"id": f"vlm_{idx}", "messages": messages})

            if vlm_tasks:
                print(f"  Running {len(vlm_tasks)} VLM image checks…")
                vlm_results = fast_parallel_vlm_calls(
                    vlm_tasks, model, max_workers=5
                )

                vlm_passed = sum(1 for v in vlm_results.values() if v)
                vlm_total = len(vlm_tasks)

                if vlm_total > 0:
                    vlm_ratio = vlm_passed / vlm_total
                    print(
                        f"  [DEBUG] VLM verification: {vlm_passed}/{vlm_total} "
                        f"images match ({vlm_ratio:.0%})"
                    )
                    if vlm_ratio >= 0.5:
                        checkpoint.add_step(
                            "VLM Image Verification", True, 1,
                            f"{vlm_passed}/{vlm_total} sampled images show "
                            f"correct decoration in correct colour ({vlm_ratio:.0%}).",
                        )
                    else:
                        checkpoint.add_step(
                            "VLM Image Verification", False, 1,
                            f"Only {vlm_passed}/{vlm_total} sampled images show "
                            f"correct decoration in correct colour "
                            f"({vlm_ratio:.0%}, expected >= 50%).",
                        )
                else:
                    checkpoint.add_step(
                        "VLM Image Verification", False, 1,
                        "No images could be downloaded for VLM verification.",
                    )
            else:
                checkpoint.add_step(
                    "VLM Image Verification", False, 1,
                    "No images could be downloaded for VLM verification.",
                )

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        checkpoint.add_step(
            "VLM Image Verification", False, 1,
            "No images found to verify.",
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


# ---------------------------------------------------------------------------
# Helpers for finding palette / checklist tabs
# ---------------------------------------------------------------------------

def _find_palette_label_col(palette_tab, palette_row_indices: List[int]) -> Optional[int]:
    """Return the leftmost column that holds non-empty text labels (no fill)
    for the palette rows."""
    rows = palette_tab.get("data", [{}])[0].get("rowData", [])
    for col in range(0, 6):
        labels_seen = 0
        for r_idx in palette_row_indices:
            if r_idx < len(rows):
                cells = rows[r_idx].get("values", [])
                if col < len(cells):
                    text = (cells[col].get("formattedValue") or "").strip()
                    if text and not cell_bg_hex(palette_tab, r_idx, col):
                        labels_seen += 1
        if labels_seen >= max(2, len(palette_row_indices) // 2):
            return col
    return None


def _find_checklist_tab(palette_tab) -> Optional[dict]:
    """Find a tab beyond the palette tab that looks like the checklist tab."""
    if not sheet_raw:
        return None
    sheets = sheet_raw.get("sheets", [])
    if len(sheets) < 3:
        return None

    palette_idx = None
    if palette_tab is not None:
        palette_id = palette_tab.get("properties", {}).get("sheetId")
        for i, s in enumerate(sheets):
            if s.get("properties", {}).get("sheetId") == palette_id:
                palette_idx = i
                break

    # Prefer a tab whose title contains 'checklist', 'planner', or 'plan'
    for i, s in enumerate(sheets):
        if i in (0, palette_idx):
            continue
        title = s.get("properties", {}).get("title", "").lower()
        if any(k in title for k in ("checklist", "planner", "plan", "decor")):
            return s

    # Fallback: first non-main, non-palette tab
    for i, s in enumerate(sheets):
        if i in (0, palette_idx):
            continue
        return s
    return None


def grade_checkpoint_6():
    """
    Checkpoint 6: Day/Evening Color Palette Tab (4 pts)
    A new tab contains at least 10 labeled three-color palette combinations.

    Outcome Evaluation:
    - A new sheet/tab was created for color palettes.
    - At least 10 three-color rows exist with exactly 3 background-filled cells.
    - Each palette row is labeled either "Day Ceremony" or "Evening Reception".
    - Color combinations use colors from the original extracted list.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    global model, palette_tab_global
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Day/Evening Palette Tab")

    palette_tab = find_palette_tab(sheet_raw)
    palette_tab_global = palette_tab

    if palette_tab is None:
        tab_names = [
            s.get("properties", {}).get("title", "?")
            for s in (sheet_raw or {}).get("sheets", [])
        ]
        checkpoint.add_step(
            "Palette Tab Exists", False, 1,
            f"No palette tab found. Tabs present: {tab_names}",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    tab_title = palette_tab.get("properties", {}).get("title", "")
    print(f"  [DEBUG] Found palette tab: '{tab_title}'")

    checkpoint.add_step(
        "Palette Tab Exists", True, 1,
        f"Palette tab found: '{tab_title}'.",
    )

    data_blocks = palette_tab.get("data", [{}])
    rows = data_blocks[0].get("rowData", []) if data_blocks else []
    total_rows = len(rows)

    palette_rows = []  # (row_idx, fill_count, hex_list)
    for r_idx in range(total_rows):
        fill_count, hex_list = count_filled_cells_in_row(palette_tab, r_idx)
        if fill_count > 0:
            palette_rows.append((r_idx, fill_count, hex_list))

    print(
        f"  [DEBUG] {len(palette_rows)} row(s) with at least 1 coloured cell "
        f"(out of {total_rows} total rows)"
    )

    # Step 2: ≥10 rows with exactly 3 background-filled cells
    correct_count_rows = [r for r in palette_rows if r[1] == PALETTE_CELLS_PER_ROW]
    enough_rows = len(correct_count_rows) >= MIN_PALETTE_ROWS
    bad_rows = [(r_idx, fc) for r_idx, fc, _ in palette_rows if fc != PALETTE_CELLS_PER_ROW]

    if enough_rows and not bad_rows:
        checkpoint.add_step(
            f"≥{MIN_PALETTE_ROWS} 3-Cell Palette Rows", True, 1,
            f"{len(correct_count_rows)} palette rows have exactly "
            f"{PALETTE_CELLS_PER_ROW} background-filled cells.",
        )
    else:
        examples = bad_rows[:5]
        detail = ", ".join(f"row {r} has {n}" for r, n in examples)
        if len(bad_rows) > 5:
            detail += f" (+{len(bad_rows) - 5} more)"
        checkpoint.add_step(
            f"≥{MIN_PALETTE_ROWS} 3-Cell Palette Rows", False, 1,
            f"{len(correct_count_rows)}/{len(palette_rows)} rows have exactly "
            f"{PALETTE_CELLS_PER_ROW} fills (need >= {MIN_PALETTE_ROWS}). "
            f"Bad rows: {detail}",
        )

    # Step 3: Day/Evening labels for each palette row
    palette_row_indices = [r_idx for r_idx, _, _ in palette_rows]
    label_col = _find_palette_label_col(palette_tab, palette_row_indices)

    label_pairs = []
    if label_col is not None:
        for r_idx in palette_row_indices:
            text = ""
            if r_idx < len(rows):
                cells = rows[r_idx].get("values", [])
                if label_col < len(cells):
                    text = (cells[label_col].get("formattedValue") or "").strip()
            label_pairs.append((r_idx, text))

    bad_labels = []
    good_labels = 0
    for r_idx, text in label_pairs:
        normal = " ".join(text.lower().split())
        if any(lbl in normal for lbl in PALETTE_LABELS):
            good_labels += 1
        else:
            bad_labels.append(f"row {r_idx}: '{text}'")

    if not palette_row_indices:
        checkpoint.add_step(
            "Day/Evening Labels", False, 1,
            "No palette rows to inspect for labels.",
        )
    elif label_col is None:
        checkpoint.add_step(
            "Day/Evening Labels", False, 1,
            "Could not locate a label column on the palette tab.",
        )
    elif good_labels == len(label_pairs):
        checkpoint.add_step(
            "Day/Evening Labels", True, 1,
            f"All {good_labels} palette rows labeled "
            f"'Day Ceremony' or 'Evening Reception'.",
        )
    else:
        checkpoint.add_step(
            "Day/Evening Labels", False, 1,
            f"Only {good_labels}/{len(label_pairs)} palette rows have a valid "
            f"Day/Evening label. Missing/wrong: {bad_labels[:5]}",
        )

    # Step 4: Palette colours come from the original extracted list
    color_names = find_color_list(main_tab) if main_tab else []
    ref_hexes = collect_reference_bg_hexes(
        main_tab, num_rows=len(color_names) + 2
    ) if main_tab and color_names else set()

    if ref_hexes:
        print(
            f"  [DEBUG] Reference hex set ({len(ref_hexes)}): "
            f"{sorted(ref_hexes)[:8]}{'…' if len(ref_hexes) > 8 else ''}"
        )
        total_palette_colors = 0
        matched_count = 0
        for _, _, hex_list in palette_rows:
            for h in hex_list:
                total_palette_colors += 1
                if hex_matches_any(h, ref_hexes):
                    matched_count += 1

        if total_palette_colors > 0:
            ratio = matched_count / total_palette_colors
            if ratio >= 0.8:
                checkpoint.add_step(
                    "Colors From Original List", True, 1,
                    f"{matched_count}/{total_palette_colors} palette colours "
                    f"match the original list ({ratio:.0%}).",
                )
            else:
                checkpoint.add_step(
                    "Colors From Original List", False, 1,
                    f"Only {matched_count}/{total_palette_colors} palette colours "
                    f"match the original list ({ratio:.0%}, expected >= 80%).",
                )
        else:
            checkpoint.add_step(
                "Colors From Original List", False, 1,
                "No palette colours to verify.",
            )
    else:
        checkpoint.add_step(
            "Colors From Original List", False, 1,
            "No reference colours available from main sheet.",
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_7():
    """
    Checkpoint 7: Decoration-Palette Checklist Tab (4 pts)
    A second new tab contains a checklist mapping decoration items to top
    three palette choices.

    Outcome Evaluation:
    - A second new sheet/tab was created.
    - Column A contains a list of wedding decoration items (LLM judge).
    - Columns B, C, D represent three palette choices.
    - Cells in columns B-D indicate the chosen color from each palette
      (filled with colors or color references).
    """
    print("----------------- CHECKPOINT 7 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Decoration-Palette Checklist Tab")

    checklist_tab = _find_checklist_tab(palette_tab_global)

    if checklist_tab is None:
        tab_names = [
            s.get("properties", {}).get("title", "?")
            for s in (sheet_raw or {}).get("sheets", [])
        ]
        checkpoint.add_step(
            "Checklist Tab Exists", False, 1,
            f"No third tab beyond main+palette found. Tabs present: {tab_names}",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    tab_title = checklist_tab.get("properties", {}).get("title", "")
    print(f"  [DEBUG] Checklist tab: '{tab_title}'")

    checkpoint.add_step(
        "Checklist Tab Exists", True, 1,
        f"Checklist tab found: '{tab_title}'.",
    )

    # Identify decoration items in column A (rows 1+ to skip optional header).
    col_a = read_column_values(
        checklist_tab, col_idx=0, start_row=0, end_row=60
    )

    # Skip the very first cell if it looks like a header.
    decoration_items: List[str] = []
    decoration_rows: List[int] = []
    skip_header = bool(col_a) and col_a[0].strip().lower() in (
        "item", "items", "decoration", "decorations", "decor",
        "wedding item", "wedding items",
    )
    start = 1 if skip_header else 0
    for offset, val in enumerate(col_a[start:], start=start):
        text = (val or "").strip()
        if not text:
            if decoration_items:
                break
            continue
        decoration_items.append(text)
        decoration_rows.append(offset)

    print(f"  [DEBUG] Column A items ({len(decoration_items)}): {decoration_items[:5]}")

    # Step 2: Column A contains wedding decoration items (LLM judge sample)
    if not decoration_items:
        checkpoint.add_step(
            "Decoration Items in Column A", False, 1,
            "Column A has no decoration items.",
        )
    else:
        if model is None:
            model = load_model(model_id)
        sample = decoration_items[:8]
        llm_tasks = []
        for idx, item in enumerate(sample):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": (
                        "You judge whether a short label refers to a wedding "
                        "decoration item or wedding-day element (e.g. "
                        "'Centerpieces', 'Bridesmaid Dresses', 'Aisle "
                        "Decorations', 'Place Settings', 'Bouquets'). Answer "
                        "'Yes' for any wedding-related decoration or stylistic "
                        "item. Answer 'No' for off-topic text (color names, "
                        "URLs, palette names, generic placeholders)."
                    )}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"Label: '{item}'\n"
                        f"Is this a wedding decoration item? Answer Yes or No."
                    )}],
                },
            ]
            llm_tasks.append({"id": f"{idx}_{item}", "messages": messages})

        print(f"  Running {len(llm_tasks)} LLM decoration-item checks…")
        llm_results = fast_parallel_vlm_calls(llm_tasks, model, max_workers=10)
        deco_yes = sum(1 for v in llm_results.values() if v)

        threshold_met = (
            len(decoration_items) >= 3 and deco_yes / len(sample) >= 0.6
        )
        if threshold_met:
            checkpoint.add_step(
                "Decoration Items in Column A", True, 1,
                f"{len(decoration_items)} items in column A; "
                f"{deco_yes}/{len(sample)} sampled items judged wedding-related.",
            )
        else:
            checkpoint.add_step(
                "Decoration Items in Column A", False, 1,
                f"{len(decoration_items)} items in column A; only "
                f"{deco_yes}/{len(sample)} sampled items judged "
                f"wedding-related (need >= 3 items and >=60% relevance).",
            )

    # Step 3: Columns B, C, D represent three palette choices (header or content)
    rows_data = checklist_tab.get("data", [{}])[0].get("rowData", [])
    bcd_present_rows = 0
    for r_idx in decoration_rows:
        if r_idx >= len(rows_data):
            continue
        cells = rows_data[r_idx].get("values", [])
        non_empty = 0
        for col in (1, 2, 3):
            text = ""
            bg = cell_bg_hex(checklist_tab, r_idx, col)
            if col < len(cells):
                text = (cells[col].get("formattedValue") or "").strip()
            if bg or text:
                non_empty += 1
        if non_empty == 3:
            bcd_present_rows += 1

    # Optionally inspect a header row above to confirm B/C/D are palette labels.
    header_row_idx = None
    if decoration_rows and decoration_rows[0] > 0:
        header_row_idx = decoration_rows[0] - 1
    palette_header_present = False
    if header_row_idx is not None and header_row_idx < len(rows_data):
        header_cells = rows_data[header_row_idx].get("values", [])
        header_texts = [
            (header_cells[c].get("formattedValue") or "").strip().lower()
            if c < len(header_cells) else ""
            for c in (1, 2, 3)
        ]
        palette_header_present = sum(1 for t in header_texts if t) >= 2

    has_three_columns = decoration_items and (
        bcd_present_rows >= max(2, len(decoration_items) // 2)
        or palette_header_present
    )
    if has_three_columns:
        checkpoint.add_step(
            "Three Palette Choice Columns", True, 1,
            f"Columns B-D populated for {bcd_present_rows}/{len(decoration_items)} "
            f"items"
            f"{' (palette header detected)' if palette_header_present else ''}.",
        )
    else:
        checkpoint.add_step(
            "Three Palette Choice Columns", False, 1,
            f"Columns B-D not consistently populated: "
            f"{bcd_present_rows}/{len(decoration_items)} item rows filled "
            f"and no palette header found.",
        )

    # Step 4: Cells in B-D filled with colors or color references
    color_filled = 0
    color_text = 0
    total_cells = 0
    for r_idx in decoration_rows:
        if r_idx >= len(rows_data):
            continue
        cells = rows_data[r_idx].get("values", [])
        for col in (1, 2, 3):
            total_cells += 1
            if cell_bg_hex(checklist_tab, r_idx, col):
                color_filled += 1
                continue
            text = ""
            if col < len(cells):
                text = (cells[col].get("formattedValue") or "").strip()
            if text and (
                text.lower().startswith("#")
                or any(c in text.lower() for c in COLOR_CATEGORIES)
                or len(text.split()) <= 4
            ):
                color_text += 1

    populated = color_filled + color_text
    if total_cells == 0:
        checkpoint.add_step(
            "Color Choices Populated", False, 1,
            "No B-D cells to inspect.",
        )
    elif populated / total_cells >= 0.5:
        checkpoint.add_step(
            "Color Choices Populated", True, 1,
            f"{populated}/{total_cells} B-D cells contain a color fill or "
            f"color reference ({color_filled} filled, {color_text} text).",
        )
    else:
        checkpoint.add_step(
            "Color Choices Populated", False, 1,
            f"Only {populated}/{total_cells} B-D cells contain a color fill "
            f"or color reference (expected >= 50%).",
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None, cached_models: dict = None, browsing_history: List[str] = None):
    """Grade all checkpoints for the beach wedding color palette task."""
    total_start_time = time.time()

    try:
        setup(workspace_doc_id)

        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        checkpoints: List[Checkpoint] = []

        checkpoints.append(grade_checkpoint_1(browsing_history=browsing_history))
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())
        checkpoints.append(grade_checkpoint_5())
        checkpoints.append(grade_checkpoint_6())
        checkpoints.append(grade_checkpoint_7())

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()

        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate sheets_45 instance 5 (Beach Wedding Color Palette)"
    )
    parser.add_argument("--workspace_doc_id", type=str, help="Google Sheets document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "✓" if step["success"] else "✗"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
