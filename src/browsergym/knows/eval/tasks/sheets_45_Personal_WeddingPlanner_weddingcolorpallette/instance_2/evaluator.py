"""Evaluator for the Wedding Color Palette Google Sheets task."""

import os
import shutil
import sys
from typing import List, Optional
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
    url_matches_paint_store,
)

# Color categories for this instance.
COLOR_CATEGORIES = ("purple", "yellow", "orange")
CATEGORY_LABEL = "/".join(COLOR_CATEGORIES)
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url, fetch_page_title

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# VLM / LLM model (loaded lazily on first use)
model = None
model_id = "gemini-2.5-flash-google-ai"

# Global variables
sheet_id = None
sheet_raw = None
main_tab = None
color_region = None  # cached by grade_checkpoint_1 for reuse


def setup(workspace_doc_id: str):
    """
    Setup function to initialize the evaluator.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, sheet_raw, main_tab, color_region

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


def grade_checkpoint_1(browsing_history: Optional[List[str]] = None):
    """
    Checkpoint 1: Color Extraction (5 pts)
    The agent found and extracted wedding color information from at least three
    articles and color names are listed vertically in the top-left area.

    Outcome Evaluation:
    - All extracted color names appear in a vertical list in a single column.
    - The color list starts in the top-left area of the sheet (rows 1-15 approximately).
    - At least 10 unique color names/shades are present, extracted from articles.
    - The agent searched for and found at least 3 articles about wedding colors.
    - All extracted colors belong to the specified purple, yellow, or orange categories.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global color_region
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Color Extraction")

    if main_tab is None:
        checkpoint.add_step(
            "Sheet Data Available", False, 1,
            "Sheet data unavailable.",
            execution_time=time.time() - checkpoint_start,
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Colors in a vertical list in a single column
    region = find_color_region(main_tab, col_idx=0, max_row=30)
    if region is None:
        region = find_color_region(main_tab, col_idx=1, max_row=30)

    color_region = region  # cache for CP2-4

    if region is None:
        checkpoint.add_step(
            "Vertical Color List", False, 1,
            "No contiguous colour-name block found in columns A or B.",
            execution_time=time.time() - checkpoint_start,
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

    # Step 2: Color list starts in top-left area (rows 1-15)
    if region["start_row"] < 15:
        checkpoint.add_step(
            "Top-Left Placement", True, 1,
            f"Color list starts at row {region['start_row']}.",
        )
    else:
        checkpoint.add_step(
            "Top-Left Placement", False, 1,
            f"Color list starts at row {region['start_row']} (expected < 15).",
        )

    # Step 3: At least 10 unique color names / shades extracted
    if num_colors >= 10:
        checkpoint.add_step(
            "Minimum 10 Unique Colors", True, 1,
            f"{num_colors} unique color names found.",
        )
    else:
        checkpoint.add_step(
            "Minimum 10 Unique Colors", False, 1,
            f"Only {num_colors} unique color names found (expected >= 10): "
            f"{region['names']}",
        )

    # Step 4: Agent visited at least 3 wedding-color articles
    history = browsing_history or []
    wedding_urls: list = []
    for url in history:
        title = fetch_page_title(url) or ""
        if url_matches_wedding_article(url, title):
            wedding_urls.append(url)
            if len(wedding_urls) >= 3:
                break  # no need to check the rest

    if len(wedding_urls) >= 3:
        checkpoint.add_step(
            "Article Research", True, 1,
            f"{len(wedding_urls)} wedding-color articles found in "
            f"{len(history)} browsing-history entries.",
        )
    else:
        checkpoint.add_step(
            "Article Research", False, 1,
            f"Only {len(wedding_urls)} wedding-color article(s) found in "
            f"{len(history)} browsing-history entries (expected >= 3).",
        )

    # Step 5: Colors belong to purple / yellow / orange (LLM batch judge)
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
    """Shared logic for checkpoints 2 (article links) and 4 (paint store links).

    Args:
        checkpoint_name: Human-readable checkpoint name.
        col_offset: Column offset from the colour-name column (1 for articles, 3 for paint).
        relevance_fn: Callable ``(url, title, model=) -> bool`` for content matching.
        total_steps: Number of eval steps for this checkpoint.

    Returns:
        Populated Checkpoint object.
    """
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

    # DEBUG: dump raw URLs so we can see what the agent actually put in
    rows = main_tab.get("data", [{}])[0].get("rowData", [])

    # Gather per-row URL data
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

    # Step 1: Column contains URLs/links
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

    # Step 2: Each color name has a corresponding link
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

    # Step 3: Links are functional
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

    # Separate liveness failures from relevance failures
    liveness_failures = [f for f in failures if "reachable" in f or "invalid" in f]
    relevance_failures = [f for f in failures if "relevant" in f]

    if total_steps >= 4:
        # CP2: separate steps for liveness and relevance
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
                f"Links lead to relevant content.",
            )
        else:
            checkpoint.add_step(
                "Content Relevance", False, 1,
                "; ".join(relevance_failures),
            )
    else:
        # CP4: combined liveness + relevance in one step
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
    """
    Checkpoint 2: Article Source Links (1 pt)
    Article links are placed in the column immediately to the right of each
    color name.

    Outcome Evaluation:
    - The column next to color names contains URLs/links.
    - Each color name has a corresponding article link.
    - Links are functional and lead to the source articles.
    - Proper alignment exists between color names and their source links.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    return _grade_url_column("Article Source Links", col_offset=1, relevance_fn=url_matches_wedding_article, total_steps=4)


def grade_checkpoint_3():
    """
    Checkpoint 3: Color Cell Formatting (1 pt)
    Cells are filled with colors matching the color names in the third column.

    Outcome Evaluation:
    - The column to the right of the links contains cells filled with background colors.
    - The fill colors visually match or closely approximate the named colors (VLM judge).
    - The hex value for the cell color matches the color shade hex found in colorhexa.
    - Each color name has a corresponding colored cell.
    """
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

    # Scan fill column
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

    # Step 1: Column contains cells filled with background colors
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

    # Step 2: Each color name has a corresponding colored cell
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

    # Step 3: Hex value matches the color shade (LLM judge)
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
                        "color is in the right general family (e.g. a blue shade "
                        "for a blue-named color) answer 'Yes'. Only answer 'No' "
                        "if the color is clearly wrong (e.g. a red for a "
                        "blue-named color)."
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

    # Step 4: Distinct fills (colors visually approximate the named colors)
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
    Checkpoint 4: Paint Store References (1 pt)
    Paint store links are provided in the fourth column next to each color.

    Outcome Evaluation:
    - The column to the right contains links to paint stores or color pages.
    - Links are functional and lead to relevant paint/color information.
    - Each color has an associated paint store link.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    return _grade_url_column("Paint Store References", col_offset=3, relevance_fn=url_matches_paint_store, total_steps=3)


def grade_checkpoint_5():
    """
    Checkpoint 5: Wedding Decoration Matrix (1 pt)
    A wedding decoration matrix is created below the color list with images.

    Outcome Evaluation:
    - At least 3 types of wedding decorations are listed in the leftmost column.
    - Column headers contain the same color names from the original list.
    - At least half of the matrix cells contain images.
    - Images show the specified decoration type in the corresponding color.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Wedding Decoration Matrix")

    if main_tab is None:
        checkpoint.add_step(
            "Data Available", False, 1,
            "Sheet data unavailable.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Discover the colour list (checkpoint-1 area)
    color_names = find_color_list(main_tab)
    print(
        f"  [DEBUG] Colour list ({len(color_names)} names): "
        f"{color_names[:8]}{'…' if len(color_names) > 8 else ''}"
    )

    if not color_names:
        checkpoint.add_step(
            "Decoration Types", False, 1,
            "Could not find colour list in column A – matrix search aborted.",
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find the decoration matrix
    search_start = len(color_names) + 1
    matrix = find_decoration_matrix(
        main_tab, color_names, search_start_row=search_start
    )

    if matrix is None:
        checkpoint.add_step(
            "Decoration Types", False, 1,
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

    # Step 1: At least 3 decoration types in leftmost column
    decoration_labels = read_column_values(
        main_tab,
        col_idx=matrix["label_col"],
        start_row=matrix["data_start_row"],
        end_row=matrix["data_end_row"],
    )
    decoration_types = [d for d in decoration_labels if d.strip()]

    if len(decoration_types) >= 3:
        checkpoint.add_step(
            "Decoration Types", True, 1,
            f"{len(decoration_types)} decoration types found: "
            f"{decoration_types[:5]}",
        )
    else:
        checkpoint.add_step(
            "Decoration Types", False, 1,
            f"Only {len(decoration_types)} decoration type(s) found "
            f"(expected >= 3): {decoration_types}",
        )

    # Step 2: Header colour names match the extracted colour list (exact then LLM)
    header_names = matrix["header_names"]
    color_set = {_normalize_color_name(c) for c in color_names}
    non_empty_headers = [h for h in header_names if h.strip()]

    # Pass 1: exact normalised match
    matched_headers = []
    unmatched_headers = []
    for h in non_empty_headers:
        if _normalize_color_name(h) in color_set:
            matched_headers.append(h)
        else:
            unmatched_headers.append(h)

    # Pass 2: LLM fallback for remaining unmatched headers
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
                        "You are a color name matching assistant. You will be "
                        "given a color name and a list of reference color names. "
                        "Determine if the given color name refers to the same "
                        "color as any name in the reference list. Answer 'Yes' "
                        "if it matches any, 'No' otherwise."
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
    image_info = []  # (url, decoration_type, color_name) for VLM checks
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
            # Sample up to 15 images evenly across the matrix
            max_vlm_checks = 15
            if len(image_info) > max_vlm_checks:
                step = len(image_info) / max_vlm_checks
                sampled = [image_info[int(i * step)] for i in range(max_vlm_checks)]
            else:
                sampled = image_info

            # Download images in parallel
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

            # Build VLM tasks for downloaded images
            vlm_tasks = []
            for idx, (_, decoration, color_name) in enumerate(sampled):
                img_path = downloaded.get(f"img_{idx}")
                if img_path and os.path.exists(img_path):
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": (
                                "You are evaluating images in a wedding decoration "
                                "matrix. Given an image, a decoration type, and a "
                                "color, determine if the image is reasonably related "
                                "to the decoration type and color. Be lenient — the "
                                "image does not need to be a perfect match. Answer "
                                "'Yes' if the image is broadly related to the "
                                "decoration category (e.g. any floral arrangement "
                                "for 'flowers', any table setting for 'centerpiece') "
                                "and the color is present or plausible even if not "
                                "dominant. Only answer 'No' if the image is clearly "
                                "unrelated to the decoration type or the color is "
                                "completely wrong (e.g. an all-red image for 'navy')."
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


def grade_checkpoint_6():
    """
    Checkpoint 6: Color Palette Tab (1 pt)
    A separate tab contains at least 10 color palette combinations.

    Outcome Evaluation:
    - A new sheet/tab was created for color palettes.
    - At least 10 rows of color combinations exist.
    - Each row contains exactly 3 colored cells representing a palette.
    - Colors are filled as background colors (not just text).
    - Color combinations use colors from the original extracted list.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Color Palette Tab")

    # Step 1: Palette tab exists
    palette_tab = find_palette_tab(sheet_raw)

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

    # Scan palette rows
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

    # Step 2: At least 10 rows of color combinations
    if len(palette_rows) >= 10:
        checkpoint.add_step(
            "Minimum 10 Rows", True, 1,
            f"{len(palette_rows)} palette rows found.",
        )
    else:
        checkpoint.add_step(
            "Minimum 10 Rows", False, 1,
            f"Only {len(palette_rows)} palette row(s) found (expected >= 10).",
        )

    # Step 3: Each row has exactly 3 colored cells
    bad_rows = [
        (r_idx, fill_count)
        for r_idx, fill_count, _ in palette_rows
        if fill_count != 3
    ]
    if not bad_rows:
        checkpoint.add_step(
            "Exactly 3 Cells Per Row", True, 1,
            f"All {len(palette_rows)} rows have exactly 3 colored cells.",
        )
    else:
        examples = bad_rows[:5]
        detail = ", ".join(f"row {r} has {n}" for r, n in examples)
        if len(bad_rows) > 5:
            detail += f" (+{len(bad_rows) - 5} more)"
        checkpoint.add_step(
            "Exactly 3 Cells Per Row", False, 1,
            f"{len(bad_rows)}/{len(palette_rows)} row(s) do not have "
            f"exactly 3 fills: {detail}",
        )

    # Step 4: Colors are filled as background colors (not just text)
    if palette_rows:
        checkpoint.add_step(
            "Background Color Fills", True, 1,
            f"{len(palette_rows)} rows have background color fills.",
        )
    else:
        checkpoint.add_step(
            "Background Color Fills", False, 1,
            "No rows with background color fills found.",
        )

    # Step 5: Palette colours come from the original extracted list
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


def grade_checkpoints(workspace_doc_id: str = None, cached_models: dict = None, browsing_history: List[str] = None):
    """
    Grade all checkpoints for the wedding color palette task.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
        cached_models: Dictionary of preloaded models keyed by model_id.
        browsing_history: List of URLs visited during task execution.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    try:
        setup(workspace_doc_id)

        # Use cached model if available
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

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()

        # Return a failed result
        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sheets_45 Wedding Color Palette")
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
