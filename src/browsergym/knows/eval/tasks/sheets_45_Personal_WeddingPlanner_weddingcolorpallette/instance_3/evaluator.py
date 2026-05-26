"""Evaluator for the Wedding Color Palette Google Sheets task (instance_3).

Checkpoint thresholds and colour categories are extracted from task.md at
runtime; the evaluation logic is task-agnostic and shared via ``utils.py``.
"""

import os
import shutil
import sys
from typing import Dict, List, Optional
import time
import traceback
import argparse


# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/path/to/KNOWS-benchmark/"
    else:
        return os.getcwd()


BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports from eval_utils
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import get_sheet_content
from src.browsergym.knows.eval.eval_utils.parallel_utils import (
    fast_parallel_vlm_calls,
    parallel_download,
    parallel_execute,
)
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url, fetch_page_title

# Shared task-agnostic utilities
from src.browsergym.knows.eval.tasks.sheets_45_Personal_WeddingPlanner_weddingcolorpallette.utils import (
    DEFAULT_COLOR_MAX_ROW,
    TOP_LEFT_MAX_ROW,
    cell_bg_hex,
    classify_colors_batch,
    collect_reference_bg_hexes,
    detect_any_image,
    extract_task_config,
    fail_all_steps,
    find_color_list,
    find_color_region,
    find_decoration_matrix,
    find_palette_region,
    find_palette_tab,
    get_cell_value,
    get_model,
    grade_url_column,
    headers_in_order,
    hex_matches_any,
    hex_to_rgb,
    is_placeholder_image_url,
    lookup_colorhexa_name,
    apply_describe_then_judge_override,
    make_topic_matcher,
    read_column_values,
    render_color_swatch,
    robust_vlm_calls,
    _normalize_color_name,
)

# ===========================================================================
# Per-instance configuration
# ===========================================================================

# Checkpoint thresholds / counts — extracted from task.md by setup().
# A field left None means task.md extraction could not determine it, and the
# dependent evaluation step fails with a clear reason rather than guessing.
COLOR_CATEGORIES = None
CATEGORY_LABEL = None
MIN_ARTICLES = None
MIN_COLORS = None
MIN_DECORATION_TYPES = None
MIN_PALETTE_ROWS = None
PALETTE_CELLS_PER_ROW = None

# Pass ratio for CP5 Image Coverage — task.md asks for "at least half".
IMAGE_COVERAGE_RATIO = 0.5

# Article-source relevance config (checkpoints 1 & 2).
ARTICLE_KEYWORDS = (
    "wedding", "bridal", "bride", "nuptial", "marriage",
    "ceremony", "reception", "bouquet", "bridesmaid",
)
ARTICLE_DOMAINS = (
    "theknot.com", "weddingwire.com", "brides.com", "marthastewartweddings",
    "bridalguide", "greenweddingshoes", "stylemepretty.com", "junebugweddings",
    "ruffledblog", "oncewed.com", "elizabethannedesigns", "weddingchicks",
    "100layercake", "loverly.com", "zola.com", "minted.com",
)
ARTICLE_DESCRIPTION = (
    "wedding colours, a wedding colour palette, or wedding design/decor"
)

# Store-reference relevance config (checkpoint 4).
STORE_KEYWORDS = (
    "paint", "sherwin", "benjamin moore", "behr", "valspar", "colorhexa",
    "pantone", "swatch", "color code", "colour code", "dulux", "farrow",
    "ppg", "glidden", "myperfectcolor",
)
STORE_DOMAINS = ()
STORE_EXCLUDE_KEYWORDS = ()
STORE_DESCRIPTION = "a paint store, colour swatch page, or colour reference"

# Relevance matchers built once from the config above.
ARTICLE_MATCHER = make_topic_matcher(
    ARTICLE_KEYWORDS, ARTICLE_DOMAINS, topic_description=ARTICLE_DESCRIPTION,
)
STORE_MATCHER = make_topic_matcher(
    STORE_KEYWORDS, STORE_DOMAINS,
    exclude_keywords=STORE_EXCLUDE_KEYWORDS, topic_description=STORE_DESCRIPTION,
)

# Checkpoint names (mirror checkpoints.md).
CP1_NAME = "Color Extraction"
CP2_NAME = "Article Source Links"
CP3_NAME = "Color Cell Formatting"
CP4_NAME = "Paint Store References"
CP5_NAME = "Wedding Decoration Matrix"
CP6_NAME = "Color Palette Tab"

# CP2/CP4 share the same 4-step shape and 13-pt total
# (1 purity + 1 presence + 1 functional + 10 proportional relevance).
CP_URL_TOTAL_STEPS = 4
CP_STORE_TOTAL_STEPS = 4
CP_URL_TOTAL_POINTS = 13

# Ordered step names per checkpoint (mirror checkpoints.md outcome bullets).
STEP_NAMES = {
    "color_extraction": [
        "Vertical Color List", "Top-Left Placement", "Minimum Unique Colors",
        "Article Research", "Color Category Match",
    ],
    "article_links": [
        "URL Column Type", "Per-Row Link Presence", "Links Functional", "Content Relevance",
    ],
    "color_formatting": [
        "Fill Column Type", "Visual Match", "Hex Matches colorhexa", "Full Coverage",
    ],
    "store_links": [
        "URL Column Type", "Per-Color Link Presence", "Links Functional",
        "Content Relevance",
    ],
    "decoration_matrix": [
        "Decoration Types", "Header Color Match", "Header Color Order",
        "Image Coverage", "VLM Image Verification",
    ],
    "palette_tab": [
        "Palette Tab Exists", "Minimum Palette Rows", "Exactly N Cells Per Row",
        "Background Fills Not Text", "Colors From Original List",
    ],
}

# ===========================================================================

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

model_id = "gemini-2.5-flash-google-ai"

# Global state populated by setup().
sheet_id = None
sheet_raw = None
main_tab = None


def setup(workspace_doc_id: str):
    """Initialize the evaluator: fetch the sheet and extract task.md config.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, sheet_raw, main_tab
    global COLOR_CATEGORIES, CATEGORY_LABEL, MIN_ARTICLES, MIN_COLORS
    global MIN_DECORATION_TYPES, MIN_PALETTE_ROWS, PALETTE_CELLS_PER_ROW

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

    # Extract checkpoint thresholds / categories from this instance's task.md.
    task_md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.md")
    cfg = extract_task_config(task_md_path, model=get_model(model_id))
    COLOR_CATEGORIES = cfg["color_categories"]
    CATEGORY_LABEL = "/".join(COLOR_CATEGORIES) if COLOR_CATEGORIES else None
    MIN_ARTICLES = cfg["min_articles"]
    MIN_COLORS = cfg["min_colors"]
    MIN_DECORATION_TYPES = cfg["min_decoration_types"]
    MIN_PALETTE_ROWS = cfg["min_palette_rows"]
    PALETTE_CELLS_PER_ROW = cfg["palette_cells_per_row"]
    print(f"  [CONFIG] extracted from task.md: {cfg}")


def grade_checkpoint_1(
    color_region: Optional[Dict] = None,
    browsing_history: Optional[List[str]] = None,
):
    """Checkpoint 1: Color Extraction (5 pts)."""
    print("----------------- CHECKPOINT 1 ----------------")
    steps = STEP_NAMES["color_extraction"]
    start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name=CP1_NAME)

    try:
        if main_tab is None:
            return fail_all_steps(checkpoint, steps, "Sheet data unavailable.", start)

        region = color_region

        if region is None:
            return fail_all_steps(
                checkpoint, steps,
                "No contiguous colour-name block found in columns A or B.", start,
            )

        names = region["names"]
        num_colors = len(names)
        print(
            f"  [DEBUG] Color list in col {region['col']}, "
            f"rows {region['start_row']}-{region['end_row'] - 1} ({num_colors} names)"
        )

        # Step 1: vertical colour list found
        checkpoint.add_step(
            steps[0], True, 1,
            f"Color list found in col {region['col']}, rows "
            f"{region['start_row']}-{region['end_row'] - 1}.",
        )

        # Step 2: top-left placement
        if region["start_row"] < TOP_LEFT_MAX_ROW:
            checkpoint.add_step(
                steps[1], True, 2,
                f"Color list starts at row {region['start_row']}.",
            )
        else:
            checkpoint.add_step(
                steps[1], False, 2,
                f"Color list starts at row {region['start_row']} "
                f"(expected < {TOP_LEFT_MAX_ROW}).",
            )

        # Step 3: minimum unique colours
        if MIN_COLORS is None:
            checkpoint.add_step(
                steps[2], False, 3,
                "Could not determine the required colour count from task.md.",
            )
        elif num_colors >= MIN_COLORS:
            checkpoint.add_step(
                steps[2], True, 3, f"{num_colors} unique color names found.",
            )
        else:
            checkpoint.add_step(
                steps[2], False, 3,
                f"Only {num_colors} unique color names found "
                f"(expected >= {MIN_COLORS}): {names}",
            )

        model = get_model(model_id)

        # Step 4: article research (visited >= MIN_ARTICLES wedding-colour articles)
        if MIN_ARTICLES is None:
            checkpoint.add_step(
                steps[3], False, 4,
                "Could not determine the required article count from task.md.",
            )
        else:
            history = browsing_history or []
            wedding_urls: List[str] = []
            if history:
                title_tasks = [
                    {"id": url, "func": fetch_page_title, "args": (url,)}
                    for url in dict.fromkeys(history)
                ]
                titles = parallel_execute(title_tasks, max_workers=10)
                for url in history:
                    title = titles.get(url) or ""
                    if ARTICLE_MATCHER(url, title, model=model):
                        wedding_urls.append(url)

            if len(wedding_urls) >= MIN_ARTICLES:
                checkpoint.add_step(
                    steps[3], True, 4,
                    f"{len(wedding_urls)} wedding-color articles found in "
                    f"{len(history)} browsing-history entries.",
                )
            else:
                checkpoint.add_step(
                    steps[3], False, 4,
                    f"Only {len(wedding_urls)} wedding-color article(s) found in "
                    f"{len(history)} browsing-history entries "
                    f"(expected >= {MIN_ARTICLES}).",
                )

        # Step 5: colours belong to the configured categories
        if COLOR_CATEGORIES is None:
            checkpoint.add_step(
                steps[4], False, 5,
                "Could not determine the colour categories from task.md.",
            )
        elif num_colors == 0:
            checkpoint.add_step(steps[4], False, 5, "No colors found to categorise.")
        else:
            results = classify_colors_batch(names, COLOR_CATEGORIES, model=model)
            categorised = sum(1 for v in results.values() if v)
            uncategorised = [n for n, v in results.items() if not v]
            ratio = categorised / num_colors
            if categorised == num_colors:
                checkpoint.add_step(
                    steps[4], True, 5,
                    f"{categorised}/{num_colors} colors classified into "
                    f"{CATEGORY_LABEL} ({ratio:.0%}).",
                )
            else:
                checkpoint.add_step(
                    steps[4], False, 5,
                    f"Only {categorised}/{num_colors} colors classified into "
                    f"{CATEGORY_LABEL} ({ratio:.0%}). Unrecognised: {uncategorised[:5]}",
                )

        checkpoint.execution_time = time.time() - start
        return checkpoint

    except Exception as e:
        traceback.print_exc()
        fresh = Checkpoint(total=5, result=0, name=CP1_NAME)
        return fail_all_steps(fresh, steps, f"Checkpoint error: {e}", start)


def grade_checkpoint_2(color_region: Optional[Dict] = None):
    """Checkpoint 2: Article Source Links (13 pts)."""
    print("----------------- CHECKPOINT 2 ----------------")
    steps = STEP_NAMES["article_links"]
    start = time.time()
    try:
        return grade_url_column(
            CP2_NAME, main_tab, color_region, col_offset=1,
            relevance_fn=ARTICLE_MATCHER, total_steps=CP_URL_TOTAL_STEPS,
            step_names=steps, model=get_model(model_id),
        )
    except Exception as e:
        traceback.print_exc()
        cp = Checkpoint(total=CP_URL_TOTAL_POINTS, result=0, name=CP2_NAME)
        return fail_all_steps(cp, steps, f"Checkpoint error: {e}", start)


def grade_checkpoint_3(color_region: Optional[Dict] = None):
    """Checkpoint 3: Color Cell Formatting (12 pts).

    Step weights: Fill Column Type 1 pt; Visual Match 5 pts proportional;
    Hex Matches colorhexa 5 pts proportional; Full Coverage 1 pt.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    steps = STEP_NAMES["color_formatting"]
    start = time.time()
    checkpoint = Checkpoint(total=12, result=0, name=CP3_NAME)

    try:
        if main_tab is None or color_region is None:
            return fail_all_steps(checkpoint, steps, "No color region available.", start)

        fill_col = color_region["col"] + 2
        sr, er = color_region["start_row"], color_region["end_row"]
        names = color_region["names"]
        sheet_raw_main = {"sheets": [main_tab]}

        filled = []   # (color_name, hex)
        missing = []
        text_only_rows = []   # (color_name, text) — cell has text but no fill
        for i, r_idx in enumerate(range(sr, er)):
            h = cell_bg_hex(main_tab, r_idx, fill_col)
            nm = names[i] if i < len(names) else f"row {r_idx}"
            if h:
                filled.append((nm, h.lower()))
            else:
                missing.append(nm)
                text = (get_cell_value(sheet_raw_main, r_idx, fill_col) or "").strip()
                if text:
                    text_only_rows.append((nm, text))

        # Early-exit: empty fill column fails all 4 steps.
        if not filled:
            return fail_all_steps(
                checkpoint, steps, f"No background fills found in col {fill_col}.", start,
            )

        # Step 1: Fill Column Type — every populated cell in the fill column
        # must have a background fill (not just typed text).
        if not text_only_rows:
            checkpoint.add_step(
                steps[0], True, 1,
                f"Fill column is a colour-fill column "
                f"({len(filled)} fills, no text-only cells).",
            )
        else:
            sample = [f"{nm}: {txt[:40]!r}" for nm, txt in text_only_rows[:3]]
            checkpoint.add_step(
                steps[0], False, 1,
                f"{len(text_only_rows)} cell(s) in col {fill_col} contain text "
                f"instead of a background fill: {sample}",
            )

        model = get_model(model_id)

        # Step 2: fills visually match the named colours (VLM swatch judge)
        visual_passed, visual_total, visual_done = 0, 0, False
        visual_items: list = []
        res: dict = {}
        render_failures: list = []
        if model is not None:
            temp_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "temp_swatches"
            )
            try:
                swatches = []  # (color_name, path_or_None)
                for nm, chex in filled:
                    p = render_color_swatch(chex, temp_dir)
                    swatches.append((nm, p))
                    if not p:
                        render_failures.append(nm)
                rendered = [(nm, p) for nm, p in swatches if p]
                if rendered:
                    vlm_tasks = []
                    for idx, (nm, p) in enumerate(rendered):
                        messages = [
                            {"role": "system", "content": [{"type": "text", "text": (
                                "You judge whether a colour swatch image matches a "
                                "named colour. Answer 'Yes' only if the swatch is "
                                "the named colour or extremely close to it "
                                "(distinguishable only by minor shade variation). "
                                "Answer 'No' if it is noticeably different."
                            )}]},
                            {"role": "user", "content": [
                                {"type": "image", "image": p},
                                {"type": "text", "text": (
                                    f"Does this swatch match the colour '{nm}' "
                                    f"(or is it extremely close to it)? "
                                    f"Answer Yes or No."
                                )},
                            ]},
                        ]
                        vlm_tasks.append({"id": str(idx), "messages": messages})
                    res = robust_vlm_calls(vlm_tasks, model)
                    apply_describe_then_judge_override(res, rendered, model)
                    visual_passed = sum(1 for v in res.values() if v)
                    # Denominator = every filled colour; render failures count
                    # toward total but not toward passes.
                    visual_total = len(filled)
                    visual_items = rendered
                    visual_done = True
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        if not visual_done and model is not None:
            # Fallback: text LLM judge on name vs hex/RGB.
            text_tasks = []
            for idx, (nm, chex) in enumerate(filled):
                r, g, b = hex_to_rgb(chex)
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": (
                        "You judge whether a hex colour matches a named colour. "
                        "Answer 'Yes' only if the hex represents the named colour "
                        "or is extremely close to it. Answer 'No' if it is "
                        "noticeably different."
                    )}]},
                    {"role": "user", "content": [{"type": "text", "text": (
                        f"Color name: '{nm}'\nHex: {chex}, RGB: ({r}, {g}, {b})\n"
                        f"Does this hex match the colour '{nm}' (or is it extremely "
                        f"close)? Answer Yes or No."
                    )}]},
                ]
                text_tasks.append({"id": str(idx), "messages": messages})
            res = robust_vlm_calls(text_tasks, model)
            visual_passed = sum(1 for v in res.values() if v)
            visual_total = len(text_tasks)
            visual_items = filled
            visual_done = True

        if not visual_done:
            checkpoint.add_step(
                steps[1], False, 2,
                "Model unavailable - could not verify visual colour match.",
                max_score=5,
            )
        elif visual_total == 0:
            checkpoint.add_step(
                steps[1], False, 2,
                "No fills available to verify visual colour match.",
                max_score=5,
            )
        else:
            visual_score = round(visual_passed * 5 / visual_total)
            if visual_passed == visual_total:
                checkpoint.add_step(
                    steps[1], True, 2,
                    f"{visual_passed}/{visual_total} fills visually match their "
                    f"named colours ({visual_score}/5 pts).",
                    score=visual_score, max_score=5,
                )
            else:
                visual_score = min(visual_score, 4)   # fail caps below max
                vlm_no = [visual_items[int(k)][0]
                          for k, v in res.items() if not v]
                rf_note = (
                    f" ({len(render_failures)} unrenderable)"
                    if render_failures else ""
                )
                checkpoint.add_step(
                    steps[1], False, 2,
                    f"Only {visual_passed}/{visual_total} fills visually match "
                    f"their named colours{rf_note} ({visual_score}/5 pts). "
                    f"Failed: {(vlm_no + render_failures)[:8]}",
                    score=visual_score, max_score=5,
                )

        # Step 3: hex value matches colorhexa.com's colour for that hex
        unique_hexes = list(dict.fromkeys(h for _, h in filled))
        chx_tasks = [
            {"id": h, "func": lookup_colorhexa_name, "args": (h,)}
            for h in unique_hexes
        ]
        chx_names = parallel_execute(chx_tasks, max_workers=10)
        resolved = [
            (nm, h, chx_names.get(h)) for nm, h in filled if chx_names.get(h)
        ]
        if not resolved:
            checkpoint.add_step(
                steps[2], False, 3,
                "colorhexa unreachable - could not verify any hex value.",
                max_score=5,
            )
        elif model is None:
            checkpoint.add_step(
                steps[2], False, 3,
                "Model unavailable - could not compare colorhexa names.",
                max_score=5,
            )
        else:
            judge_tasks = []
            for idx, (nm, h, chexa) in enumerate(resolved):
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": (
                        "You judge whether two colour descriptions refer to the "
                        "same or a closely related colour. Answer Yes or No."
                    )}]},
                    {"role": "user", "content": [{"type": "text", "text": (
                        f"Color label used in the sheet: '{nm}'.\n"
                        f"colorhexa.com's canonical name for hex {h}: '{chexa}'.\n"
                        f"Are these the same or a closely related colour? "
                        f"Answer Yes or No."
                    )}]},
                ]
                judge_tasks.append({"id": str(idx), "messages": messages})
            jres = robust_vlm_calls(judge_tasks, model)
            matched = sum(1 for v in jres.values() if v)
            hex_score = round(matched * 5 / len(resolved))
            if matched == len(resolved):
                checkpoint.add_step(
                    steps[2], True, 3,
                    f"{matched}/{len(resolved)} hex values are consistent with "
                    f"colorhexa.com ({hex_score}/5 pts).",
                    score=hex_score, max_score=5,
                )
            else:
                hex_score = min(hex_score, 4)   # fail caps below max
                failed_names = [resolved[int(k)][0] for k, v in jres.items() if not v]
                checkpoint.add_step(
                    steps[2], False, 3,
                    f"Only {matched}/{len(resolved)} hex values are consistent "
                    f"with colorhexa.com ({hex_score}/5 pts). "
                    f"Failed: {failed_names[:8]}",
                    score=hex_score, max_score=5,
                )

        # Step 4: each colour name has a corresponding colored cell
        if not missing:
            checkpoint.add_step(
                steps[3], True, 4,
                f"All {len(names)} colors have a corresponding fill.",
            )
        else:
            checkpoint.add_step(
                steps[3], False, 4,
                f"{len(missing)}/{len(names)} colors missing fills: {missing[:5]}",
            )

        checkpoint.execution_time = time.time() - start
        return checkpoint

    except Exception as e:
        traceback.print_exc()
        fresh = Checkpoint(total=12, result=0, name=CP3_NAME)
        return fail_all_steps(fresh, steps, f"Checkpoint error: {e}", start)


def grade_checkpoint_4(color_region: Optional[Dict] = None):
    """Checkpoint 4: Paint Store References (13 pts)."""
    print("----------------- CHECKPOINT 4 ----------------")
    steps = STEP_NAMES["store_links"]
    start = time.time()
    try:
        return grade_url_column(
            CP4_NAME, main_tab, color_region, col_offset=3,
            relevance_fn=STORE_MATCHER, total_steps=CP_STORE_TOTAL_STEPS,
            step_names=steps, model=get_model(model_id),
        )
    except Exception as e:
        traceback.print_exc()
        cp = Checkpoint(total=CP_URL_TOTAL_POINTS, result=0, name=CP4_NAME)
        return fail_all_steps(cp, steps, f"Checkpoint error: {e}", start)


def grade_checkpoint_5(color_region: Optional[Dict] = None):
    """Checkpoint 5: Wedding Decoration Matrix (15 pts).

    Step weights: Decoration Types / Header Color Match / Header Color Order
    are 1 pt each; Image Coverage is 2 pts; VLM Image Verification is 10 pts
    awarded proportionally to the pass count.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    steps = STEP_NAMES["decoration_matrix"]
    start = time.time()
    checkpoint = Checkpoint(total=15, result=0, name=CP5_NAME)

    try:
        if main_tab is None:
            return fail_all_steps(checkpoint, steps, "Sheet data unavailable.", start)

        region = color_region
        if region is None:
            return fail_all_steps(
                checkpoint, steps,
                "Could not find colour list - matrix search aborted.", start,
            )
        color_names = region["names"]

        search_start = region["end_row"]
        matrix = find_decoration_matrix(
            main_tab, color_names, search_start_row=search_start,
        )
        if matrix is None:
            return fail_all_steps(
                checkpoint, steps,
                f"No decoration matrix found below row {search_start}.", start,
            )

        model = get_model(model_id)
        header_names = matrix["header_names"]

        # Step 1: at least MIN_DECORATION_TYPES decoration types
        decoration_labels = read_column_values(
            main_tab, col_idx=matrix["label_col"],
            start_row=matrix["data_start_row"], end_row=matrix["data_end_row"],
        )
        decoration_types = [d for d in decoration_labels if d.strip()]
        if MIN_DECORATION_TYPES is None:
            checkpoint.add_step(
                steps[0], False, 1,
                "Could not determine the required decoration count from task.md.",
            )
        elif len(decoration_types) >= MIN_DECORATION_TYPES:
            checkpoint.add_step(
                steps[0], True, 1,
                f"{len(decoration_types)} decoration types found: "
                f"{decoration_types[:5]}",
            )
        else:
            checkpoint.add_step(
                steps[0], False, 1,
                f"Only {len(decoration_types)} decoration type(s) found "
                f"(expected >= {MIN_DECORATION_TYPES}): {decoration_types}",
            )

        # Step 2: header colour names match the extracted colour list
        color_set = {_normalize_color_name(c) for c in color_names}
        non_empty_headers = [h for h in header_names if h.strip()]
        matched_headers, unmatched_headers = [], []
        for h in non_empty_headers:
            if _normalize_color_name(h) in color_set:
                matched_headers.append(h)
            else:
                unmatched_headers.append(h)

        if unmatched_headers and model is not None:
            color_list_str = ", ".join(sorted(color_set))
            llm_tasks = []
            for h in unmatched_headers:
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": (
                        "You are a colour name matching assistant. Given a colour "
                        "name and a reference list, answer 'Yes' if it refers to "
                        "the same colour as any name in the list, else 'No'."
                    )}]},
                    {"role": "user", "content": [{"type": "text", "text": (
                        f"Color name: '{h}'\nReference list: [{color_list_str}]\n"
                        f"Does '{h}' match any colour in the reference list? "
                        f"Answer Yes or No."
                    )}]},
                ]
                llm_tasks.append({"id": h, "messages": messages})
            llm_results = fast_parallel_vlm_calls(llm_tasks, model, max_workers=10)
            still_unmatched = []
            for h in unmatched_headers:
                if llm_results.get(h, False):
                    matched_headers.append(h)
                else:
                    still_unmatched.append(h)
            unmatched_headers = still_unmatched

        if matched_headers and not unmatched_headers:
            checkpoint.add_step(
                steps[1], True, 2,
                f"{len(matched_headers)} header colours match the original list.",
            )
        elif unmatched_headers:
            checkpoint.add_step(
                steps[1], False, 2,
                f"Matrix headers not in colour list: {sorted(unmatched_headers)}",
            )
        else:
            checkpoint.add_step(
                steps[1], False, 2, "No header colour names found in matrix.",
            )

        # Step 3: matrix columns appear in the same order as the colour list
        if not non_empty_headers:
            checkpoint.add_step(
                steps[2], False, 3,
                "No matrix column headers found to check ordering.",
            )
        else:
            order_ok, order_detail = headers_in_order(header_names, color_names)
            checkpoint.add_step(steps[2], order_ok, 3, order_detail)

        # Step 4: at least half of grid cells contain images
        total_cells, image_cells = 0, 0
        image_info = []  # (url, decoration_type, color_name)
        for r_idx in range(matrix["data_start_row"], matrix["data_end_row"]):
            row_offset = r_idx - matrix["data_start_row"]
            decoration = (
                decoration_types[row_offset]
                if row_offset < len(decoration_types) else "decoration"
            )
            for c_idx in range(matrix["data_col_start"], matrix["data_col_end"]):
                total_cells += 1
                url = detect_any_image(main_tab, r_idx, c_idx)
                if url:
                    image_cells += 1
                    col_offset = c_idx - matrix["data_col_start"]
                    color_name = (
                        header_names[col_offset]
                        if col_offset < len(header_names) else "unknown"
                    )
                    image_info.append((url, decoration, color_name))

        if total_cells == 0:
            checkpoint.add_step(steps[3], False, 4, "Matrix body has 0 cells.",
                                max_score=2)
        elif image_cells / total_cells >= IMAGE_COVERAGE_RATIO:
            checkpoint.add_step(
                steps[3], True, 4,
                f"{image_cells}/{total_cells} cells contain images "
                f"({image_cells / total_cells:.0%}).",
                max_score=2,
            )
        else:
            checkpoint.add_step(
                steps[3], False, 4,
                f"Only {image_cells}/{total_cells} cells contain images "
                f"({image_cells / total_cells:.0%}, expected >= "
                f"{IMAGE_COVERAGE_RATIO:.0%}).",
                max_score=2,
            )

        # Step 5: VLM judge - images show the decoration in the colour
        # (10 pts, awarded proportionally to the number of images that pass).
        if not image_info:
            checkpoint.add_step(steps[4], False, 5, "No images found to verify.",
                                max_score=10)
        elif model is None:
            checkpoint.add_step(
                steps[4], False, 5, "Model unavailable - could not verify images.",
                max_score=10,
            )
        else:
            temp_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "temp_vlm_images"
            )
            os.makedirs(temp_dir, exist_ok=True)
            try:
                sampled = image_info  # judge every image; no stride sampling

                # Placeholder/stub images (placehold.co etc.) are never real
                # decoration photos — fail them without a VLM call.
                placeholder_count = sum(
                    1 for url, _, _ in sampled if is_placeholder_image_url(url)
                )
                real = [
                    item for item in sampled
                    if not is_placeholder_image_url(item[0])
                ]

                # Dedupe downloads by URL: fetch each unique URL once.
                # VLM tasks below are still built per-cell so duplicate URLs
                # across cells get judged independently with their own
                # (decoration, colour) context.
                unique_urls = list(dict.fromkeys(url for url, _, _ in real))
                dl_tasks = [
                    {"id": url, "func": download_image_from_url,
                     "args": (url, temp_dir)}
                    for url in unique_urls
                ]
                downloaded = parallel_download(dl_tasks, max_workers=20, use_rate_limit=False)

                vlm_tasks = []
                for idx, (img_url, decoration, color_name) in enumerate(real):
                    img_path = downloaded.get(img_url)
                    if img_path and os.path.exists(img_path):
                        messages = [
                            {"role": "system", "content": [{"type": "text", "text": (
                                "You verify photos in a wedding decoration matrix. "
                                "Answer 'Yes' ONLY if the image is a real "
                                "photograph that shows specifically the named "
                                "decoration AND that decoration is specifically "
                                "the named colour. Answer 'No' if either the "
                                "decoration type is different (even slightly) OR "
                                "the colour is different — a related or close "
                                "shade is NOT a match (e.g. royal blue is NOT "
                                "navy, light green is NOT emerald green, wine is "
                                "NOT burgundy). Also answer 'No' for placeholders, "
                                "solid-colour swatches, graphics with text, logos, "
                                "or anything that is not a real photograph."
                            )}]},
                            {"role": "user", "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": (
                                    f"Decoration: {decoration}\n"
                                    f"Expected colour: {color_name}\n"
                                    f"Does this image show specifically a "
                                    f"{decoration} in specifically the colour "
                                    f"{color_name}? Answer Yes or No."
                                )},
                            ]},
                        ]
                        vlm_tasks.append({"id": f"vlm_{idx}", "messages": messages})

                vlm_results = (
                    robust_vlm_calls(vlm_tasks, model, max_workers=10)
                    if vlm_tasks else {}
                )
                # Failed downloads count toward the denominator but not the
                # numerator: an image we couldn't fetch can't earn credit.
                download_failures = sum(
                    1 for img_url, _, _ in real
                    if not (downloaded.get(img_url) and os.path.exists(downloaded[img_url]))
                )
                vlm_passed = sum(1 for v in vlm_results.values() if v)
                vlm_total = placeholder_count + download_failures + len(vlm_tasks)

                if vlm_total == 0:
                    checkpoint.add_step(
                        steps[4], False, 5,
                        "No images could be downloaded for VLM verification.",
                        max_score=10,
                    )
                else:
                    vlm_score = round(vlm_passed * 10 / vlm_total)
                    if vlm_passed == vlm_total:
                        checkpoint.add_step(
                            steps[4], True, 5,
                            f"{vlm_passed}/{vlm_total} sampled images are real photos "
                            f"of the correct decoration in the correct colour "
                            f"({vlm_score}/10 pts).",
                            score=vlm_score, max_score=10,
                        )
                    else:
                        vlm_score = min(vlm_score, 9)   # fail caps below max
                        notes = []
                        if placeholder_count:
                            notes.append(f"{placeholder_count} placeholder")
                        if download_failures:
                            notes.append(f"{download_failures} unreachable")
                        breakdown = f" ({'; '.join(notes)})" if notes else ""
                        checkpoint.add_step(
                            steps[4], False, 5,
                            f"Only {vlm_passed}/{vlm_total} sampled images are real "
                            f"photos of the correct decoration in the correct "
                            f"colour{breakdown} ({vlm_score}/10 pts).",
                            score=vlm_score, max_score=10,
                        )
            finally:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        checkpoint.execution_time = time.time() - start
        return checkpoint

    except Exception as e:
        traceback.print_exc()
        fresh = Checkpoint(total=15, result=0, name=CP5_NAME)
        return fail_all_steps(fresh, steps, f"Checkpoint error: {e}", start)


def grade_checkpoint_6():
    """Checkpoint 6: Color Palette Tab (5 pts).

    Five independent steps: tab existence, row count, per-row width, fill type
    (background vs plain text), and colour provenance.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    steps = STEP_NAMES["palette_tab"]
    start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name=CP6_NAME)

    try:
        palette_tab = find_palette_tab(sheet_raw)
        if palette_tab is None:
            tab_names = [
                s.get("properties", {}).get("title", "?")
                for s in (sheet_raw or {}).get("sheets", [])
            ]
            return fail_all_steps(
                checkpoint, steps,
                f"No palette tab found. Tabs present: {tab_names}", start,
            )

        # Step 1: a separate palette tab exists
        tab_title = palette_tab.get("properties", {}).get("title", "")
        checkpoint.add_step(steps[0], True, 1, f"Palette tab found: '{tab_title}'.")

        region = find_palette_region(palette_tab)
        print(f"  [DEBUG] {len(region)} palette row(s) detected")

        # Step 2: at least MIN_PALETTE_ROWS palette rows
        if MIN_PALETTE_ROWS is None:
            checkpoint.add_step(
                steps[1], False, 2,
                "Could not determine the required palette-row count from task.md.",
            )
        elif len(region) >= MIN_PALETTE_ROWS:
            checkpoint.add_step(
                steps[1], True, 2, f"{len(region)} palette rows found.",
            )
        else:
            checkpoint.add_step(
                steps[1], False, 2,
                f"Only {len(region)} palette row(s) found "
                f"(expected >= {MIN_PALETTE_ROWS}).",
            )

        # Step 3: each palette row has exactly PALETTE_CELLS_PER_ROW colour cells
        if PALETTE_CELLS_PER_ROW is None:
            checkpoint.add_step(
                steps[2], False, 3,
                "Could not determine the palette width from task.md.",
            )
        elif not region:
            checkpoint.add_step(
                steps[2], False, 3, "No palette rows found to check width.",
            )
        else:
            bad_rows = [
                (r["row_idx"], len(r["content_cols"]))
                for r in region
                if len(r["content_cols"]) != PALETTE_CELLS_PER_ROW
            ]
            if not bad_rows:
                checkpoint.add_step(
                    steps[2], True, 3,
                    f"All {len(region)} rows have exactly "
                    f"{PALETTE_CELLS_PER_ROW} colour cells.",
                )
            else:
                detail = ", ".join(f"row {r} has {n}" for r, n in bad_rows[:5])
                if len(bad_rows) > 5:
                    detail += f" (+{len(bad_rows) - 5} more)"
                checkpoint.add_step(
                    steps[2], False, 3,
                    f"{len(bad_rows)}/{len(region)} row(s) do not have exactly "
                    f"{PALETTE_CELLS_PER_ROW} colour cells: {detail}",
                )

        # Step 4: colours are background fills, not just typed text
        total_content = sum(len(r["content_cols"]) for r in region)
        total_filled = sum(len(r["filled_cols"]) for r in region)
        if total_content == 0:
            checkpoint.add_step(
                steps[3], False, 4, "No palette colour cells found.",
            )
        else:
            fill_ratio = total_filled / total_content
            if total_filled == total_content:
                checkpoint.add_step(
                    steps[3], True, 4,
                    f"{total_filled}/{total_content} palette cells use background "
                    f"fills ({fill_ratio:.0%}).",
                )
            else:
                checkpoint.add_step(
                    steps[3], False, 4,
                    f"Only {total_filled}/{total_content} palette cells use "
                    f"background fills ({fill_ratio:.0%}); the rest are plain text.",
                )

        # Step 5: palette colours come from the original extracted list
        color_names = find_color_list(main_tab, columns=(0, 1)) if main_tab else []
        ref_hexes = (
            collect_reference_bg_hexes(main_tab, num_rows=len(color_names) + 2)
            if main_tab and color_names else set()
        )
        all_hexes = [h for r in region for h in r["hex_list"]]
        if not ref_hexes:
            checkpoint.add_step(
                steps[4], False, 5, "No reference colours available from main sheet.",
            )
        elif not all_hexes:
            checkpoint.add_step(
                steps[4], False, 5, "No background-filled palette cells to verify.",
            )
        else:
            matched = sum(1 for h in all_hexes if hex_matches_any(h, ref_hexes))
            ratio = matched / len(all_hexes)
            if matched == len(all_hexes):
                checkpoint.add_step(
                    steps[4], True, 5,
                    f"{matched}/{len(all_hexes)} palette colours match the "
                    f"original list ({ratio:.0%}).",
                )
            else:
                checkpoint.add_step(
                    steps[4], False, 5,
                    f"Only {matched}/{len(all_hexes)} palette colours match the "
                    f"original list ({ratio:.0%}).",
                )

        checkpoint.execution_time = time.time() - start
        return checkpoint

    except Exception as e:
        traceback.print_exc()
        fresh = Checkpoint(total=5, result=0, name=CP6_NAME)
        return fail_all_steps(fresh, steps, f"Checkpoint error: {e}", start)


def grade_checkpoints(workspace_doc_id: str = None, cached_models: dict = None,
                      browsing_history: List[str] = None):
    """Grade all checkpoints for the wedding color palette task.

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
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        traceback.print_exc()
        failed = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed.add_step("Setup", False, 1, f"Fatal setup error: {str(e)}")
        return Result([failed], total_execution_time=time.time() - total_start_time)

    # Pre-load the model once (guarded; shared by all checkpoints).
    get_model(model_id, cached_models)

    # Compute the colour region once and thread it through every checkpoint
    # that needs it. CP1 grades it; CP2/3/4/5 use it for column offsets.
    region = (
        find_color_region(main_tab, columns=(0, 1), max_row=DEFAULT_COLOR_MAX_ROW)
        if main_tab is not None else None
    )

    checkpoints: List[Checkpoint] = [
        grade_checkpoint_1(color_region=region, browsing_history=browsing_history),
        grade_checkpoint_2(color_region=region),
        grade_checkpoint_3(color_region=region),
        grade_checkpoint_4(color_region=region),
        grade_checkpoint_5(color_region=region),
        grade_checkpoint_6(),
    ]

    return Result(checkpoints, total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sheets_45 Wedding Color Palette")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Sheets document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history,
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "PASS" if step["success"] else "FAIL"
            print(f"  [{status}] {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
