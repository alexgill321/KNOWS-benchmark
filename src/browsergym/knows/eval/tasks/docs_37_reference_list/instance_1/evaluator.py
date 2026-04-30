import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List

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
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_services_helpers import get_doc_content

# Imports from template-level utils
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute
from src.browsergym.knows.eval.eval_utils.models import load_model

from src.browsergym.knows.eval.tasks.docs_37_reference_list.utils import (
    extract_headings_with_bookmarks,
    extract_bullet_sections,
    extract_reference_links,
    matches_lecture_title_format,
    get_gold_lectures,
    match_valid_category,
    match_text_quiet,
    is_raw_url,
    check_slide_format,
    check_link_name_relevance,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_37_reference_list/instance_1/")
GOLD_DATA_PATH = os.path.join(TASK_DIR, "data/gold_outputs.json")
NOTION_DATA_PATH = os.path.join(TASK_DIR, "data/notion_schedule.json")
PAGE_TITLES_PATH = os.path.join(TASK_DIR, "data/page_titles.json")

model = None
model_id = "gemini-3-flash-google-ai"

# Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

# Global variables
document = None
gold_data = None
notion_schedule = None

# Caches to avoid redundant computation across checkpoints
doc_refs_cache = None
gold_to_doc_cache = None
category_cache = None


def setup_document(workspace_doc_id):
    """Fetch the Google Doc and load gold/notion reference data."""
    global document, gold_data, notion_schedule

    document = get_doc_content(workspace_doc_id, DOCS_SERVICE)

    with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    with open(NOTION_DATA_PATH, "r", encoding="utf-8") as f:
        notion_schedule = json.load(f)


def grade_checkpoint_1():
    """Checkpoint 1 (30pt): Lecture Title — format, heading style, bookmarks."""
    start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Lecture Title")

    # --- Shared preparation ---
    gold_lectures = get_gold_lectures(gold_data)
    doc_headings = extract_headings_with_bookmarks(document)

    total_count = len(gold_lectures)

    # Build matched_headings: gold_lecture -> heading dict or None
    heading_texts = [h["text"] for h in doc_headings]
    matched_headings = {}
    for gold_lecture in gold_lectures:
        matched_text, score = match_text_quiet(gold_lecture, heading_texts, threshold=75)
        if matched_text:
            # Find the heading dict for this matched text
            matched_headings[gold_lecture] = next(
                h for h in doc_headings if h["text"] == matched_text
            )
        else:
            matched_headings[gold_lecture] = None

    # Build notion combined titles: "M/D: title" to match against doc headings holistically
    notion_combined_titles = []
    for entry in notion_schedule:
        try:
            dt = datetime.strptime(entry["lecture_date"], "%B %d, %Y")
            combined = f"{dt.month}/{dt.day}: {entry['lecture_title']}"
            notion_combined_titles.append(combined)
        except (ValueError, KeyError):
            pass

    # --- Step 1: Title format (M/D: title) + gold match + notion cross-reference ---
    step_start = time.time()
    format_pass = 0
    format_details = []
    for gold_lecture in gold_lectures:
        heading = matched_headings.get(gold_lecture)
        if not heading:
            format_details.append(f"Missing: '{gold_lecture}'")
            continue
        if not matches_lecture_title_format(heading["text"]):
            format_details.append(f"Bad format: '{heading['text']}'")
            continue
        # Verify the doc heading closely matches the gold lecture title
        _, gold_score = match_text_quiet(heading["text"], [gold_lecture], threshold=85)
        if gold_score < 85:
            format_details.append(f"Weak gold match ({gold_score}): '{heading['text']}' vs '{gold_lecture}'")
            continue
        # Cross-reference full heading (M/D: title) against notion combined titles
        notion_match, _ = match_text_quiet(heading["text"], notion_combined_titles, threshold=60)
        if not notion_match:
            format_details.append(f"No notion match: '{heading['text']}'")
            continue
        format_pass += 1

    score_1 = calculate_percentage_score(format_pass, total_count, 10)
    checkpoint.add_step(
        name="Title format (M/D: title) with gold + notion match",
        success=(format_pass == total_count),
        step_id=1,
        details=f"{format_pass}/{total_count} lectures in correct format with notion match. "
                + "; ".join(format_details) if format_details else f"{format_pass}/{total_count} all passed",
        score=score_1,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 2: Heading 3 style ---
    step_start = time.time()
    h3_pass = 0
    h3_details = []
    for gold_lecture in gold_lectures:
        heading = matched_headings.get(gold_lecture)
        if not heading:
            h3_details.append(f"Missing: '{gold_lecture}'")
            continue
        if heading["style"] == "HEADING_3":
            h3_pass += 1
        else:
            h3_details.append(f"Wrong style '{heading['style']}': '{heading['text']}'")

    score_2 = calculate_percentage_score(h3_pass, total_count, 10)
    checkpoint.add_step(
        name="Heading 3 style",
        success=(h3_pass == total_count),
        step_id=2,
        details=f"{h3_pass}/{total_count} lectures are Heading 3. "
                + "; ".join(h3_details) if h3_details else f"{h3_pass}/{total_count} all passed",
        score=score_2,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 3: Bookmarked (headingId present) ---
    step_start = time.time()
    bookmark_pass = 0
    bookmark_details = []
    for gold_lecture in gold_lectures:
        heading = matched_headings.get(gold_lecture)
        if not heading:
            bookmark_details.append(f"Missing: '{gold_lecture}'")
            continue
        if heading["has_bookmark"]:
            bookmark_pass += 1
        else:
            bookmark_details.append(f"Not bookmarked: '{heading['text']}'")

    score_3 = calculate_percentage_score(bookmark_pass, total_count, 10)
    checkpoint.add_step(
        name="Bookmarked",
        success=(bookmark_pass == total_count),
        step_id=3,
        details=f"{bookmark_pass}/{total_count} lectures bookmarked. "
                + "; ".join(bookmark_details) if bookmark_details else f"{bookmark_pass}/{total_count} all passed",
        score=score_3,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2 (40pt): Topics bullet lists — categories, bold, non-empty, no invalid."""
    global model, category_cache
    if model is None:
        model = load_model(model_id)

    start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Topics bullet lists")

    sections = extract_bullet_sections(document)
    total_sections = len(sections)

    # Build expected (lecture, list_type) pairs from gold data
    gold_pairs = set()
    for item in gold_data:
        if item.get("list_type"):
            gold_pairs.add((item["lecture"], item["list_type"]))
    total_gold_pairs = len(gold_pairs)

    # Build category cache: category_text -> matched valid category (or None)
    # Exact matching only (no LLM fallback) — categories must be from a fixed set
    category_cache = {}
    for section in sections:
        cat = section["category"]
        if cat not in category_cache:
            category_cache[cat] = match_valid_category(cat)

    # Classify each doc section's category using cache
    valid_sections = []
    invalid_sections = []
    for section in sections:
        if category_cache[section["category"]]:
            valid_sections.append(section)
        else:
            invalid_sections.append(section)

    # Pre-build lecture match cache: section_lecture -> set of gold_lectures it matches
    # This reduces O(gold_pairs * sections) match calls to O(unique_section_lectures * unique_gold_lectures)
    unique_section_lectures = {s["lecture"] for s in sections}
    unique_gold_lectures = {lec for lec, _ in gold_pairs}
    section_to_gold_lectures = {}
    for sl in unique_section_lectures:
        matched_golds = set()
        for gl in unique_gold_lectures:
            if match_text_quiet(sl, [gl], threshold=75)[0]:
                matched_golds.add(gl)
        section_to_gold_lectures[sl] = matched_golds

    # --- Step 1: Valid category titles (gold coverage) ---
    step_start = time.time()
    covered = 0
    coverage_details = []
    for lecture, list_type in sorted(gold_pairs):
        found = any(
            category_cache[s["category"]] == list_type
            for s in sections
            if lecture in section_to_gold_lectures.get(s["lecture"], set())
        )
        if found:
            covered += 1
        else:
            coverage_details.append(f"Missing '{list_type}' under '{lecture}'")

    score_1 = calculate_percentage_score(covered, total_gold_pairs, 10)
    checkpoint.add_step(
        name="Valid category titles (gold coverage)",
        success=(covered == total_gold_pairs),
        step_id=1,
        details=f"{covered}/{total_gold_pairs} expected categories found. "
                + "; ".join(coverage_details[:5]) if coverage_details else f"{covered}/{total_gold_pairs} all found",
        score=score_1,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 2: Category titles in bold ---
    step_start = time.time()
    bold_pass = 0
    bold_details = []
    for section in sections:
        if section["category_is_bold"]:
            bold_pass += 1
        else:
            bold_details.append(f"Not bold: '{section['category']}' under '{section['lecture']}'")

    score_2 = calculate_percentage_score(bold_pass, total_sections, 10)
    checkpoint.add_step(
        name="Category titles in bold",
        success=(bold_pass == total_sections),
        step_id=2,
        details=f"{bold_pass}/{total_sections} category titles are bold. "
                + "; ".join(bold_details[:5]) if bold_details else f"{bold_pass}/{total_sections} all bold",
        score=score_2,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 3: No empty bullet lists ---
    step_start = time.time()
    nonempty_pass = 0
    empty_details = []
    for section in sections:
        if len(section["items"]) > 0:
            nonempty_pass += 1
        else:
            empty_details.append(f"Empty: '{section['category']}' under '{section['lecture']}'")

    score_3 = calculate_percentage_score(nonempty_pass, total_sections, 10)
    checkpoint.add_step(
        name="No empty bullet lists",
        success=(nonempty_pass == total_sections),
        step_id=3,
        details=f"{nonempty_pass}/{total_sections} sections non-empty. "
                + "; ".join(empty_details) if empty_details else f"{nonempty_pass}/{total_sections} all non-empty",
        score=score_3,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 4: No invalid category titles ---
    step_start = time.time()
    valid_count = len(valid_sections)
    invalid_details = [f"Invalid: '{s['category']}' under '{s['lecture']}'" for s in invalid_sections]

    score_4 = calculate_percentage_score(valid_count, total_sections, 10)
    checkpoint.add_step(
        name="No invalid category titles",
        success=(len(invalid_sections) == 0),
        step_id=4,
        details=f"{valid_count}/{total_sections} sections have valid categories. "
                + "; ".join(invalid_details[:5]) if invalid_details else f"{valid_count}/{total_sections} all valid",
        score=score_4,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3 (60pt): Reference links — presence, categorization, slides, format, relevant."""
    global model, doc_refs_cache, gold_to_doc_cache
    if model is None:
        model = load_model(model_id)

    start = time.time()
    checkpoint = Checkpoint(total=60, result=0, name="Reference links")

    doc_refs = extract_reference_links(document)
    doc_refs_cache = doc_refs
    total_gold = len(gold_data)

    # Pre-cache lecture matching: doc_ref_lecture -> set of gold_lectures it matches
    # This avoids calling match_text_quiet for every (gold_item, doc_ref) pair
    unique_doc_lectures = {r["lecture"] for r in doc_refs}
    unique_gold_lectures = {item["lecture"] for item in gold_data}
    doc_lecture_to_gold = {}
    for dl in unique_doc_lectures:
        matched_golds = set()
        for gl in unique_gold_lectures:
            if match_text_quiet(dl, [gl], threshold=75)[0]:
                matched_golds.add(gl)
        doc_lecture_to_gold[dl] = matched_golds

    # Match each gold item to a doc ref by (lecture, url) using cached lecture matches
    gold_to_doc = {}
    for gold_item in gold_data:
        gold_url = gold_item["url"]
        gold_lecture = gold_item["lecture"]
        matched_ref = next(
            (r for r in doc_refs if r["url"] == gold_url
             and gold_lecture in doc_lecture_to_gold.get(r["lecture"], set())),
            None
        )
        gold_to_doc[(gold_lecture, gold_url)] = matched_ref
    gold_to_doc_cache = gold_to_doc

    # --- Step 1: All reference links present without duplicates (per lecture) ---
    step_start = time.time()
    present_count = 0
    presence_details = []
    # Build per-lecture duplicate set: (lecture, url) pairs seen more than once
    seen_per_lecture = set()
    duplicate_per_lecture = set()
    for ref in doc_refs:
        key = (ref["lecture"], ref["url"])
        if key in seen_per_lecture:
            duplicate_per_lecture.add(key)
        seen_per_lecture.add(key)

    for gold_item in gold_data:
        if gold_to_doc[(gold_item["lecture"], gold_item["url"])]:
            key = (gold_item["lecture"], gold_item["url"])
            if key not in duplicate_per_lecture:
                present_count += 1
            else:
                presence_details.append(f"Duplicate in '{gold_item['lecture']}': '{gold_item['name']}'")
        else:
            presence_details.append(f"Missing: '{gold_item['name']}'")

    score_1 = calculate_percentage_score(present_count, total_gold, 10)
    checkpoint.add_step(
        name="All references present without duplicates",
        success=(present_count == total_gold),
        step_id=1,
        details=f"{present_count}/{total_gold} refs present and unique. "
                + "; ".join(presence_details[:5]) if presence_details else f"{present_count}/{total_gold} all present",
        score=score_1,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 2: Correct categorization ---
    step_start = time.time()
    cat_pass = 0
    cat_details = []
    for gold_item in gold_data:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            cat_details.append(f"Missing: '{gold_item['name']}'")
            continue
        expected_type = gold_item.get("list_type", "")
        # Use category_cache from CP2 if available, otherwise compute
        doc_category = (category_cache.get(ref["category"])
                        if category_cache else match_valid_category(ref["category"], model=model))
        if doc_category == expected_type:
            cat_pass += 1
        else:
            cat_details.append(f"Wrong category '{ref['category']}' (expected '{expected_type}'): '{gold_item['name']}'")

    score_2 = calculate_percentage_score(cat_pass, total_gold, 10)
    checkpoint.add_step(
        name="Correct categorization",
        success=(cat_pass == total_gold),
        step_id=2,
        details=f"{cat_pass}/{total_gold} correctly categorized. "
                + "; ".join(cat_details[:5]) if cat_details else f"{cat_pass}/{total_gold} all correct",
        score=score_2,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 3: Each link has a slide number ---
    step_start = time.time()
    slide_present_count = 0
    slide_present_details = []
    for gold_item in gold_data:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            slide_present_details.append(f"Missing: '{gold_item['name']}'")
            continue
        if ref["slide_numbers"]:
            slide_present_count += 1
        else:
            slide_present_details.append(f"No slides: '{gold_item['name']}'")

    score_3 = calculate_percentage_score(slide_present_count, total_gold, 10)
    checkpoint.add_step(
        name="Slide numbers present",
        success=(slide_present_count == total_gold),
        step_id=3,
        details=f"{slide_present_count}/{total_gold} have slide numbers. "
                + "; ".join(slide_present_details[:5]) if slide_present_details else f"{slide_present_count}/{total_gold} all have slides",
        score=score_3,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 4: Hyperlink format (anchor text, not raw URL) ---
    step_start = time.time()
    format_pass = 0
    format_details = []
    for gold_item in gold_data:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            format_details.append(f"Missing: '{gold_item['name']}'")
            continue
        if not is_raw_url(ref["anchor_text"]):
            format_pass += 1
        else:
            format_details.append(f"Raw URL as anchor: '{ref['anchor_text']}'")

    score_4 = calculate_percentage_score(format_pass, total_gold, 10)
    checkpoint.add_step(
        name="Hyperlink format (descriptive anchor text)",
        success=(format_pass == total_gold),
        step_id=4,
        details=f"{format_pass}/{total_gold} use descriptive anchor text. "
                + "; ".join(format_details[:5]) if format_details else f"{format_pass}/{total_gold} all descriptive",
        score=score_4,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 5: Slide number format ("Slide: {n1, n2, ...}") ---
    step_start = time.time()
    slide_fmt_pass = 0
    slide_fmt_details = []
    for gold_item in gold_data:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            slide_fmt_details.append(f"Missing: '{gold_item['name']}'")
            continue
        if check_slide_format(ref["full_text"]):
            slide_fmt_pass += 1
        else:
            slide_fmt_details.append(f"Bad slide format: '{ref['full_text']}'")

    score_5 = calculate_percentage_score(slide_fmt_pass, total_gold, 10)
    checkpoint.add_step(
        name="Slide number format",
        success=(slide_fmt_pass == total_gold),
        step_id=5,
        details=f"{slide_fmt_pass}/{total_gold} have correct slide format. "
                + "; ".join(slide_fmt_details[:5]) if slide_fmt_details else f"{slide_fmt_pass}/{total_gold} all correct format",
        score=score_5,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 6: Link name relevance ---
    step_start_6 = time.time()
    with open(PAGE_TITLES_PATH, "r", encoding="utf-8") as f:
        page_titles = json.load(f)

    # Build unique (anchor_text, url) pairs from already-matched refs in gold_to_doc
    unique_relevance_pairs = {}
    for _, ref in gold_to_doc.items():
        if ref:
            key = (ref["anchor_text"], ref["url"])
            if key not in unique_relevance_pairs:
                unique_relevance_pairs[key] = page_titles.get(ref["url"])

    # Parallel relevance checks using same pattern as Step 6
    relevance_tasks = [
        {
            "id": f"{anchor_text}||{url}",
            "func": check_link_name_relevance,
            "args": (anchor_text, url, model),
            "kwargs": {"page_title": page_title},
        }
        for (anchor_text, url), page_title in unique_relevance_pairs.items()
    ]
    relevance_results = parallel_execute(relevance_tasks, max_workers=10) if relevance_tasks else {}

    # Rebuild relevance_cache from parallel results
    relevance_cache = {
        (anchor_text, url): relevance_results.get(f"{anchor_text}||{url}", False)
        for (anchor_text, url) in unique_relevance_pairs
    }

    relevance_pass = 0
    relevance_details = []
    for gold_item in gold_data:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            relevance_details.append(f"Missing: '{gold_item['name']}'")
            continue
        is_relevant = relevance_cache.get((ref["anchor_text"], ref["url"]), False)
        if is_relevant:
            relevance_pass += 1
        else:
            relevance_details.append(f"Irrelevant name '{ref['anchor_text']}' for '{ref['url']}'")

    score_6 = calculate_percentage_score(relevance_pass, total_gold, 10)
    checkpoint.add_step(
        name="Link names relevant",
        success=(relevance_pass == total_gold),
        step_id=6,
        details=f"{relevance_pass}/{total_gold} have relevant names. "
                + "; ".join(relevance_details[:5]) if relevance_details else f"{relevance_pass}/{total_gold} all relevant",
        score=score_6,
        max_score=10,
        execution_time=time.time() - step_start_6,
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_4():
    """Checkpoint 4 (30pt): Multiple references — bold, dark green 2, no duplicate slides."""
    start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Multiple references")

    # Reuse cached data from CP3 if available
    doc_refs = doc_refs_cache if doc_refs_cache is not None else extract_reference_links(document)

    # Filter gold items that appear in multiple slides (num_slides > 1)
    multi_slide_gold = [item for item in gold_data if item.get("num_slides", 1) > 1]
    total_multi = len(multi_slide_gold)

    if total_multi == 0:
        # No multi-slide references — all steps pass vacuously
        for step_id, name in enumerate(
            ["Bold for multi-slide refs", "Dark green 2 for multi-slide refs", "No duplicate slide numbers"], 1
        ):
            checkpoint.add_step(
                name=name, success=True, step_id=step_id,
                details="No multi-slide references found", score=10, max_score=10,
                execution_time=0,
            )
        checkpoint.execution_time = time.time() - start
        return checkpoint

    # Reuse gold_to_doc from CP3 cache if available, otherwise rebuild
    if gold_to_doc_cache is not None:
        multi_keys = {(g["lecture"], g["url"]) for g in multi_slide_gold}
        gold_to_doc = {k: v for k, v in gold_to_doc_cache.items() if k in multi_keys}
    else:
        gold_to_doc = {}
        for gold_item in multi_slide_gold:
            matched_ref = next(
                (r for r in doc_refs if r["url"] == gold_item["url"]
                 and match_text_quiet(r["lecture"], [gold_item["lecture"]], threshold=75)[0]),
                None
            )
            gold_to_doc[(gold_item["lecture"], gold_item["url"])] = matched_ref

    # --- Step 1: Hyperlinks of multi-slide references are bold ---
    step_start = time.time()
    bold_pass = 0
    bold_details = []
    for gold_item in multi_slide_gold:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            bold_details.append(f"Missing: '{gold_item['name']}'")
            continue
        if ref["link_is_bold"]:
            bold_pass += 1
        else:
            bold_details.append(f"Not bold: '{gold_item['name']}' in '{gold_item['lecture']}'")

    score_1 = calculate_percentage_score(bold_pass, total_multi, 10)
    checkpoint.add_step(
        name="Bold for multi-slide refs",
        success=(bold_pass == total_multi),
        step_id=1,
        details=f"{bold_pass}/{total_multi} multi-slide refs are bold. "
                + "; ".join(bold_details[:5]) if bold_details else f"{bold_pass}/{total_multi} all bold",
        score=score_1,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 2: Hyperlinks of multi-slide references are dark green 2 ---
    step_start = time.time()
    green_pass = 0
    green_details = []
    for gold_item in multi_slide_gold:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            green_details.append(f"Missing: '{gold_item['name']}'")
            continue
        if ref["link_is_dark_green_2"]:
            green_pass += 1
        else:
            green_details.append(f"Not dark green 2: '{gold_item['name']}' in '{gold_item['lecture']}'")

    score_2 = calculate_percentage_score(green_pass, total_multi, 10)
    checkpoint.add_step(
        name="Dark green 2 for multi-slide refs",
        success=(green_pass == total_multi),
        step_id=2,
        details=f"{green_pass}/{total_multi} multi-slide refs are dark green 2. "
                + "; ".join(green_details[:5]) if green_details else f"{green_pass}/{total_multi} all dark green 2",
        score=score_2,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    # --- Step 3: No duplicate slide numbers for multi-slide refs ---
    step_start = time.time()
    no_dup_pass = 0
    dup_details = []
    for gold_item in multi_slide_gold:
        ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
        if not ref:
            dup_details.append(f"Missing: '{gold_item['name']}'")
            continue
        slide_nums = ref["slide_numbers"]
        if len(slide_nums) == len(set(slide_nums)):
            no_dup_pass += 1
        else:
            dup_details.append(f"Duplicate slides {slide_nums}: '{gold_item['name']}'")

    score_3 = calculate_percentage_score(no_dup_pass, total_multi, 10)
    checkpoint.add_step(
        name="No duplicate slide numbers",
        success=(no_dup_pass == total_multi),
        step_id=3,
        details=f"{no_dup_pass}/{total_multi} have unique slide numbers. "
                + "; ".join(dup_details[:5]) if dup_details else f"{no_dup_pass}/{total_multi} all unique",
        score=score_3,
        max_score=10,
        execution_time=time.time() - step_start,
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
    """Main evaluation function: setup + grade all checkpoints."""
    total_start = time.time()

    setup_document(workspace_doc_id)

    checkpoints: List[Checkpoint] = []
    checkpoints.append(grade_checkpoint_1())
    checkpoints.append(grade_checkpoint_2())
    checkpoints.append(grade_checkpoint_3())
    checkpoints.append(grade_checkpoint_4())

    return Result(checkpoints, total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate docs_37_reference_list")
    parser.add_argument("--workspace_doc_id", type=str, required=True, help="Google Docs document ID")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    args = parser.parse_args()

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        cached_models=args.cached_models,
        browsing_history=args.browsing_history,
    )
    
    score = result.final_score

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {score['result']}/{score['total']}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        cp_time = checkpoint.get("execution_time", 0)
        print(f"\n{checkpoint['name']}: {checkpoint['score']} ({cp_time:.2f}s)")
        for step in checkpoint["steps"]:
            status = "PASS" if step["success"] else "FAIL"
            step_time = step.get("execution_time", 0)
            print(f"  [{status}] {step['name']}: {step['details'] or 'No details'} ({step_time:.2f}s)")
    print(f"\nTotal time: {result.total_execution_time:.2f}s")
