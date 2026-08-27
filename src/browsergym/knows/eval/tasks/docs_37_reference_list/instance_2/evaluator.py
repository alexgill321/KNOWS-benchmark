import os
import sys
import re
import json
import time
import argparse
from typing import List

def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()


BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports from eval_utils
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_services_helpers import get_doc_content

# Imports from template-level utils
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute
from src.browsergym.knows.eval.eval_utils.models import load_model

from src.browsergym.knows.eval.tasks.docs_37_reference_list.utils import (
    extract_headings_with_bookmarks,
    extract_bullet_sections,
    extract_reference_links,
    pair_headings_to_gold,
    title_matches,
    match_valid_category,
    match_text_quiet,
    is_raw_url,
    check_slide_format,
    check_link_name_relevance,
    is_dead_link,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/docs_37_reference_list/instance_2/")
GOLD_DATA_PATH = os.path.join(TASK_DIR, "data/gold_outputs.json")
PAGE_TITLES_PATH = os.path.join(TASK_DIR, "data/page_titles.json")
SCHEDULE_PATH = os.path.join(TASK_DIR, "data/schedule.json")

# Instance-2 specific valid categories
VALID_CATEGORIES_2 = {"Tutorials", "Textbooks", "Videos"}

model = None
model_id = "gemini-3-flash-google-ai"

# Maps match_valid_category / check_link_name_relevance return_method values to
# step categories. A method of None means no phase accepted the text; since the
# category cache here is built with a model, the LLM made the final rejection,
# so callers use LLM_VLM_JUDGEMENT as the dict-lookup default.
_METHOD_CATEGORIES = {
    "exact": StepCategory.DETERMINISTIC,
    "llm": StepCategory.LLM_VLM_JUDGEMENT,
    "fuzzy": StepCategory.FUZZY_MATCH,
    "error": StepCategory.EXECUTION_ERROR,
}

# Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

# Global variables
document = None
gold_data = None

# Caches to avoid redundant computation across checkpoints
doc_refs_cache = None
gold_to_doc_cache = None
category_cache = None

def setup_document(workspace_doc_id):
    """Fetch the Google Doc and load gold reference data."""
    global document, gold_data

    document = get_doc_content(workspace_doc_id, DOCS_SERVICE)

    with open(GOLD_DATA_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)


def grade_checkpoint_1():
    """Checkpoint 1 (20pt): Lecture Title — Module N format and heading style."""
    _STEPS = [
        (1, "Title format (Module N: title) with gold match", 10),
        (2, "Heading 3 style", 10),
    ]
    global model
    start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Lecture Title")
    _added = set()

    if not document:
        for sid, sname, smax in _STEPS:
            checkpoint.add_step(
                name=sname, success=False, step_id=sid,
                details="Error fetching document content.",
                score=0, max_score=smax, execution_time=0,
                category=StepCategory.EXECUTION_ERROR,
            )
        checkpoint.execution_time = time.time() - start
        return checkpoint

    try:
        with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
            schedule = json.load(f)
        gold_lectures = []
        for entry in schedule:
            title = entry.get("lecture_title", "").strip()
            if not title or not entry.get("lecture_slides_link"):
                continue
            module_match = re.search(r"\d+", entry.get("module", ""))
            if module_match:
                gold_lectures.append(f"Module {module_match.group()}: {title}")

        if model is None:
            model = load_model(model_id)
        doc_headings = extract_headings_with_bookmarks(document)
        total_count = len(gold_lectures)

        matched_headings = pair_headings_to_gold(
            gold_lectures, doc_headings, model=model
        )

        # --- Step 1: Title format (Module N: title) + gold match ---
        t = time.time()
        try:
            format_pass = 0
            format_details = []
            for gold_lecture in gold_lectures:
                heading = matched_headings.get(gold_lecture)
                if not heading:
                    format_details.append(f"Missing: '{gold_lecture}'")
                    continue
                
                gold_prefix, sep, gold_rest = gold_lecture.partition(":")
                gold_prefix = (gold_prefix + ":") if sep else gold_lecture
                doc_text = heading["text"].strip()
                starts_ok = doc_text.lower().startswith(gold_prefix.lower())
                doc_rest = doc_text[len(gold_prefix):].strip() if starts_ok else ""
                if not starts_ok or not title_matches(doc_rest, gold_rest.strip()):
                    format_details.append(f"Bad format: '{heading['text']}'")
                    continue
                format_pass += 1
            score_1 = calculate_percentage_score(format_pass, total_count, 10)
            checkpoint.add_step(
                name="Title format (Module N: title) with gold match",
                success=(format_pass == total_count), step_id=1,
                details=f"{format_pass}/{total_count} lectures in correct format. "
                        + "; ".join(format_details) if format_details else f"{format_pass}/{total_count} all passed",
                score=score_1, max_score=10, execution_time=time.time() - t,
                category=StepCategory.FUZZY_MATCH,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Title format (Module N: title) with gold match",
                success=False, step_id=1, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(1)

        # --- Step 2: Heading 3 style ---
        t = time.time()
        try:
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
                success=(h3_pass == total_count), step_id=2,
                details=f"{h3_pass}/{total_count} lectures are Heading 3. "
                        + "; ".join(h3_details) if h3_details else f"{h3_pass}/{total_count} all passed",
                score=score_2, max_score=10, execution_time=time.time() - t,
                category=StepCategory.STRUCTURAL,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Heading 3 style", success=False, step_id=2,
                details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(2)

    except Exception as e:
        for sid, sname, smax in _STEPS:
            if sid not in _added:
                checkpoint.add_step(
                    name=sname, success=False, step_id=sid,
                    details=f"Data acquisition failed — {e}",
                    score=0, max_score=smax, execution_time=0,
                    category=StepCategory.EXECUTION_ERROR,
                )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2 (40pt): Topics bullet lists — categories, bold, non-empty, no invalid."""
    _STEPS = [
        (1, "Valid category titles (gold coverage)", 10),
        (2, "Bullet list starts with title in bold", 10),
        (3, "No empty bullet lists", 10),
        (4, "No invalid category titles", 10),
    ]
    global model, category_cache
    if model is None:
        model = load_model(model_id)

    start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Topics bullet lists")
    _added = set()

    if not document:
        for sid, sname, smax in _STEPS:
            checkpoint.add_step(
                name=sname, success=False, step_id=sid,
                details="Error fetching document content.",
                score=0, max_score=smax, execution_time=0,
                category=StepCategory.EXECUTION_ERROR,
            )
        checkpoint.execution_time = time.time() - start
        return checkpoint

    try:
        sections = extract_bullet_sections(document)
        if not sections or len(sections) == 0:
            for step in _STEPS:
                checkpoint.add_step(
                    name=step[1], success=False, step_id=step[0],
                    details="No bullet sections found in the document.",
                    score=0, max_score=step[2], execution_time=0,
                    category=StepCategory.EXECUTION_ERROR,
                )
            checkpoint.execution_time = time.time() - start
            return checkpoint
        
        total_sections = len(sections)

        gold_pairs = set()
        for item in gold_data:
            if item.get("list_type"):
                gold_pairs.add((item["lecture"], item["list_type"]))
        total_gold_pairs = len(gold_pairs)

        category_cache = {}
        for section in sections:
            cat = section["category"]
            if cat not in category_cache:
                category_cache[cat] = match_valid_category(
                    cat, model=model, valid_categories=VALID_CATEGORIES_2, return_method=True
                )

        valid_sections = []
        invalid_sections = []
        for section in sections:
            if category_cache[section["category"]][0]:
                valid_sections.append(section)
            else:
                invalid_sections.append(section)

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
        t = time.time()
        try:
            covered = 0
            coverage_details = []
            coverage_items = []  # (category, success) per gold pair for StepCategory.aggregate
            for lecture, list_type in sorted(gold_pairs):
                matched_section = next(
                    (s for s in sections
                     if lecture in section_to_gold_lectures.get(s["lecture"], set())
                     and category_cache[s["category"]][0] == list_type),
                    None,
                )
                if matched_section is not None:
                    covered += 1
                    coverage_items.append((
                        _METHOD_CATEGORIES.get(category_cache[matched_section["category"]][1], StepCategory.LLM_VLM_JUDGEMENT),
                        True,
                    ))
                else:
                    coverage_details.append(f"Missing '{list_type}' under '{lecture}'")
                    # Rejection is the exact comparison over cached category verdicts
                    coverage_items.append((StepCategory.DETERMINISTIC, False))
            score_1 = calculate_percentage_score(covered, total_gold_pairs, 10)
            checkpoint.add_step(
                name="Valid category titles (gold coverage)",
                success=(covered == total_gold_pairs), step_id=1,
                details=f"{covered}/{total_gold_pairs} expected categories found. "
                        + "; ".join(coverage_details[:5]) if coverage_details else f"{covered}/{total_gold_pairs} all found",
                score=score_1, max_score=10, execution_time=time.time() - t,
                category=StepCategory.aggregate(coverage_items),
            )
        except Exception as e:
            checkpoint.add_step(
                name="Valid category titles (gold coverage)",
                success=False, step_id=1, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(1)

        # --- Step 2: Bullet list starts with the title in bold ---
        t = time.time()
        try:
            format_pass = 0
            format_details = []
            for section in sections:
                is_bold = section["category_is_bold"]
                if is_bold:
                    format_pass += 1
                else:
                    format_details.append(
                        f"'{section['category']}' under '{section['lecture']}' not bold"
                    )
            score_2 = calculate_percentage_score(format_pass, total_sections, 10)
            checkpoint.add_step(
                name="Bullet list starts with title in bold",
                success=(format_pass == total_sections), step_id=2,
                details=f"{format_pass}/{total_sections} category titles are bold bullets. "
                        + "; ".join(format_details[:5]) if format_details else f"{format_pass}/{total_sections} all correct",
                score=score_2, max_score=10, execution_time=time.time() - t,
                category=StepCategory.STRUCTURAL,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Bullet list starts with title in bold",
                success=False, step_id=2, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(2)

        # --- Step 3: No empty bullet lists ---
        t = time.time()
        try:
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
                success=(nonempty_pass == total_sections), step_id=3,
                details=f"{nonempty_pass}/{total_sections} sections non-empty. "
                        + "; ".join(empty_details) if empty_details else f"{nonempty_pass}/{total_sections} all non-empty",
                score=score_3, max_score=10, execution_time=time.time() - t,
                category=StepCategory.STRUCTURAL,
            )
        except Exception as e:
            checkpoint.add_step(
                name="No empty bullet lists",
                success=False, step_id=3, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(3)

        # --- Step 4: No invalid category titles ---
        t = time.time()
        try:
            valid_count = len(valid_sections)
            invalid_details = [f"Invalid: '{s['category']}' under '{s['lecture']}'" for s in invalid_sections]
            invalid_items = [
                (_METHOD_CATEGORIES.get(category_cache[s["category"]][1], StepCategory.LLM_VLM_JUDGEMENT),
                 bool(category_cache[s["category"]][0]))
                for s in sections
            ]
            score_4 = calculate_percentage_score(valid_count, total_sections, 10)
            checkpoint.add_step(
                name="No invalid category titles",
                success=(len(invalid_sections) == 0), step_id=4,
                details=f"{valid_count}/{total_sections} sections have valid categories. "
                        + "; ".join(invalid_details[:5]) if invalid_details else f"{valid_count}/{total_sections} all valid",
                score=score_4, max_score=10, execution_time=time.time() - t,
                category=StepCategory.aggregate(invalid_items),
            )
        except Exception as e:
            checkpoint.add_step(
                name="No invalid category titles",
                success=False, step_id=4, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(4)

    except Exception as e:
        for sid, sname, smax in _STEPS:
            if sid not in _added:
                checkpoint.add_step(
                    name=sname, success=False, step_id=sid,
                    details=f"Data acquisition failed — {e}",
                    score=0, max_score=smax, execution_time=0,
                    category=StepCategory.EXECUTION_ERROR,
                )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3 (90pt): Reference links — presence, no duplicates, categorization,
    slides match gold, hyperlink format, slide format, author format, relevance, active."""
    _STEPS = [
        (1, "All references present", 10),
        (2, "No duplicate reference links", 10),
        (3, "Correct categorization", 10),
        (4, "Slide numbers match gold", 10),
        (5, "Hyperlink format (descriptive anchor text)", 10),
        (6, "Slide number format", 10),
        (7, "Author format (First author) after hyperlink", 10),
        (8, "Link names relevant", 10),
        (9, "No dead hyperlinks", 10),
    ]
    global model, doc_refs_cache, gold_to_doc_cache
    if model is None:
        model = load_model(model_id)

    start = time.time()
    checkpoint = Checkpoint(total=90, result=0, name="Reference links")
    _added = set()

    if not document:
        for sid, sname, smax in _STEPS:
            checkpoint.add_step(
                name=sname, success=False, step_id=sid,
                details="Error fetching document content.",
                score=0, max_score=smax, execution_time=0,
                category=StepCategory.EXECUTION_ERROR,
            )
        checkpoint.execution_time = time.time() - start
        return checkpoint

    try:
        doc_refs = extract_reference_links(document)
        doc_refs_cache = doc_refs
        total_gold = len(gold_data)

        unique_doc_lectures = {r["lecture"] for r in doc_refs}
        unique_gold_lectures = {item["lecture"] for item in gold_data}
        # Resolve each doc lecture to its single best-matching gold lecture.
        doc_lecture_best_gold = {}
        for dl in unique_doc_lectures:
            matched_gl, _ = match_text_quiet(dl, list(unique_gold_lectures), threshold=75)
            doc_lecture_best_gold[dl] = matched_gl

        gold_to_doc = {}
        for r in doc_refs:
            resolved = doc_lecture_best_gold.get(r["lecture"])
            if resolved is None:
                continue
            gold_to_doc.setdefault((resolved, r["url"]), r)
        for gold_item in gold_data:
            gold_to_doc.setdefault((gold_item["lecture"], gold_item["url"]), None)
        gold_to_doc_cache = gold_to_doc

        ref_counts = {}
        for ref in doc_refs:
            key = (ref["lecture"], ref["url"])
            ref_counts[key] = ref_counts.get(key, 0) + 1

        # --- Step 1: All references present ---
        t = time.time()
        found_items = []
        try:
            present_count = 0
            presence_details = []
            for gold_item in gold_data:
                if gold_to_doc[(gold_item["lecture"], gold_item["url"])]:
                    present_count += 1
                    found_items.append(gold_item)
                else:
                    presence_details.append(f"Missing: '{gold_item['name']}'")
            score_1 = calculate_percentage_score(present_count, total_gold, 10)
            checkpoint.add_step(
                name="All references present",
                success=(present_count == total_gold), step_id=1,
                details=f"{present_count}/{total_gold} refs present. "
                        + "; ".join(presence_details[:5]) if presence_details else f"{present_count}/{total_gold} all present",
                score=score_1, max_score=10, execution_time=time.time() - t,
                category=StepCategory.FUZZY_MATCH,
            )
        except Exception as e:
            checkpoint.add_step(
                name="All references present",
                success=False, step_id=1, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(1)

        # --- Step 2: No duplicate reference links ---
        t = time.time()
        try:
            dup_pass = 0
            dup_details = []
            for gold_item in found_items:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                count = ref_counts.get((ref["lecture"], ref["url"]), 0)
                if count <= 1:
                    dup_pass += 1
                else:
                    dup_details.append(f"Appears {count}x in '{ref['lecture']}': '{gold_item['name']}'")
            score_2 = calculate_percentage_score(dup_pass, total_gold, 10)
            checkpoint.add_step(
                name="No duplicate reference links",
                success=(dup_pass == total_gold), step_id=2,
                details=f"{dup_pass}/{total_gold} refs have no duplicates. "
                        + "; ".join(dup_details[:5]) if dup_details else f"{dup_pass}/{total_gold} all unique",
                score=score_2, max_score=10, execution_time=time.time() - t,
                category=StepCategory.DETERMINISTIC,
            )
        except Exception as e:
            checkpoint.add_step(
                name="No duplicate reference links",
                success=False, step_id=2, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(2)

        # --- Step 3: Correct categorization ---
        t = time.time()
        try:
            cat_pass = 0
            cat_details = []
            cat_items = []  # (category, success) per reference for StepCategory.aggregate
            for gold_item in found_items:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                if not ref:
                    cat_details.append(f"Missing: '{gold_item['name']}'")
                    continue
                expected_type = gold_item.get("list_type", "")
                if category_cache:
                    doc_category, cat_method = category_cache.get(ref["category"], (None, None))
                else:
                    doc_category, cat_method = match_valid_category(
                        ref["category"], model=model, valid_categories=VALID_CATEGORIES_2,
                        return_method=True,
                    )
                # A None method means no phase accepted: the LLM made the final
                # rejection (the cache and the direct call both run with a model).
                item_category = _METHOD_CATEGORIES.get(cat_method, StepCategory.LLM_VLM_JUDGEMENT)
                if doc_category == expected_type:
                    cat_pass += 1
                    cat_items.append((item_category, True))
                else:
                    cat_details.append(
                        f"Wrong category '{ref['category']}' (expected '{expected_type}'): '{gold_item['name']}'"
                    )
                    cat_items.append((item_category, False))
            score_3 = calculate_percentage_score(cat_pass, total_gold, 10)
            checkpoint.add_step(
                name="Correct categorization",
                success=(cat_pass == total_gold), step_id=3,
                details=f"{cat_pass}/{total_gold} correctly categorized. "
                        + "; ".join(cat_details[:5]) if cat_details else f"{cat_pass}/{total_gold} all correct",
                score=score_3, max_score=10, execution_time=time.time() - t,
                category=StepCategory.aggregate(cat_items),
            )
        except Exception as e:
            checkpoint.add_step(
                name="Correct categorization",
                success=False, step_id=3, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(3)

        # --- Step 4: Slide numbers match gold ---
        t = time.time()
        try:
            slide_match_count = 0
            slide_match_details = []
            for gold_item in found_items:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                if not ref:
                    slide_match_details.append(f"Missing: '{gold_item['name']}'")
                    continue
                if set(ref["slide_numbers"]) == set(gold_item["slides"]):
                    slide_match_count += 1
                else:
                    slide_match_details.append(
                        f"Slide mismatch '{gold_item['name']}': got {sorted(ref['slide_numbers'])} expected {sorted(gold_item['slides'])}"
                    )
            score_4 = calculate_percentage_score(slide_match_count, total_gold, 10)
            checkpoint.add_step(
                name="Slide numbers match gold",
                success=(slide_match_count == total_gold), step_id=4,
                details=f"{slide_match_count}/{total_gold} have correct slide numbers. "
                        + "; ".join(slide_match_details[:5]) if slide_match_details else f"{slide_match_count}/{total_gold} all match",
                score=score_4, max_score=10, execution_time=time.time() - t,
                category=StepCategory.DETERMINISTIC,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Slide numbers match gold",
                success=False, step_id=4, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(4)

        # --- Step 5: Hyperlink format (anchor text, not raw URL) ---
        t = time.time()
        try:
            format_pass = 0
            format_details = []
            for gold_item in found_items:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                if not ref:
                    format_details.append(f"Missing: '{gold_item['name']}'")
                    continue
                if not is_raw_url(ref["anchor_text"]):
                    format_pass += 1
                else:
                    format_details.append(f"Raw URL as anchor: '{ref['anchor_text']}'")
            score_5 = calculate_percentage_score(format_pass, total_gold, 10)
            checkpoint.add_step(
                name="Hyperlink format (descriptive anchor text)",
                success=(format_pass == total_gold), step_id=5,
                details=f"{format_pass}/{total_gold} use descriptive anchor text. "
                        + "; ".join(format_details[:5]) if format_details else f"{format_pass}/{total_gold} all descriptive",
                score=score_5, max_score=10, execution_time=time.time() - t,
                category=StepCategory.DETERMINISTIC,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Hyperlink format (descriptive anchor text)",
                success=False, step_id=5, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(5)

        # --- Step 6: Slide number format ---
        t = time.time()
        try:
            slide_fmt_pass = 0
            slide_fmt_details = []
            for gold_item in found_items:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                if not ref:
                    slide_fmt_details.append(f"Missing: '{gold_item['name']}'")
                    continue
                if check_slide_format(ref["full_text"]):
                    slide_fmt_pass += 1
                else:
                    slide_fmt_details.append(f"Bad slide format: '{ref['full_text']}'")
            score_6 = calculate_percentage_score(slide_fmt_pass, total_gold, 10)
            checkpoint.add_step(
                name="Slide number format",
                success=(slide_fmt_pass == total_gold), step_id=6,
                details=f"{slide_fmt_pass}/{total_gold} have correct slide format. "
                        + "; ".join(slide_fmt_details[:5]) if slide_fmt_details else f"{slide_fmt_pass}/{total_gold} all correct format",
                score=score_6, max_score=10, execution_time=time.time() - t,
                category=StepCategory.DETERMINISTIC,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Slide number format",
                success=False, step_id=6, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(6)

        # --- Step 7: Author format "(First author)" after hyperlink ---
        t = time.time()
        try:
            author_pass = 0
            author_details = []
            for gold_item in found_items:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                if not ref:
                    author_details.append(f"Missing: '{gold_item['name']}'")
                    continue
                gold_author = gold_item.get("author", "")
                full_text = ref["full_text"]
                anchor_text = ref["anchor_text"]
                # Look for (author) in the text after the anchor
                meta = full_text[full_text.find(anchor_text) + len(anchor_text):] if anchor_text in full_text else full_text
                author_match = re.search(r'\(([^)]+)\)', meta)
                if not author_match:
                    author_details.append(f"No (author) in bullet: '{gold_item['name']}'")
                    continue
                doc_author = author_match.group(1).strip()
                if not gold_author:
                    author_pass += 1
                    continue
                _, score = match_text_quiet(gold_author, [doc_author], threshold=65)
                if score >= 65:
                    author_pass += 1
                else:
                    author_details.append(
                        f"Author mismatch '{doc_author}' (expected '{gold_author}'): '{gold_item['name']}'"
                    )
            score_7 = calculate_percentage_score(author_pass, total_gold, 10)
            checkpoint.add_step(
                name="Author format (First author) after hyperlink",
                success=(author_pass == total_gold), step_id=7,
                details=f"{author_pass}/{total_gold} have correct author format. "
                        + "; ".join(author_details[:5]) if author_details else f"{author_pass}/{total_gold} all correct",
                score=score_7, max_score=10, execution_time=time.time() - t,
                category=StepCategory.FUZZY_MATCH,
            )
        except Exception as e:
            checkpoint.add_step(
                name="Author format (First author) after hyperlink",
                success=False, step_id=7, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(7)

        # --- Step 8: Link name relevance ---
        t = time.time()
        try:
            with open(PAGE_TITLES_PATH, "r", encoding="utf-8") as f:
                page_titles = json.load(f)
            unique_relevance_pairs = {}
            for _, ref in gold_to_doc.items():
                if ref:
                    key = (ref["anchor_text"], ref["url"])
                    if key not in unique_relevance_pairs:
                        unique_relevance_pairs[key] = page_titles.get(ref["url"])
            relevance_tasks = [
                {
                    "id": f"{anchor_text}||{url}",
                    "func": check_link_name_relevance,
                    "args": (anchor_text, url, model),
                    "kwargs": {"page_title": page_title, "return_method": True},
                }
                for (anchor_text, url), page_title in unique_relevance_pairs.items()
            ]
            relevance_results = parallel_execute(relevance_tasks, max_workers=10) if relevance_tasks else {}
            # Cache stores (verdict, method) — method records whether the fuzzy
            # title match or the LLM decided. Verdict behavior is unchanged.
            relevance_cache = {}
            for (anchor_text, url) in unique_relevance_pairs:
                res = relevance_results.get(f"{anchor_text}||{url}", (False, None))
                if not isinstance(res, tuple):
                    res = (bool(res), None)
                relevance_cache[(anchor_text, url)] = res
            relevance_pass = 0
            relevance_details = []
            rel_items = []  # (category, success) per reference for StepCategory.aggregate
            for gold_item in gold_data:
                ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                if not ref:
                    relevance_details.append(f"Missing: '{gold_item['name']}'")
                    continue
                is_relevant, rel_method = relevance_cache.get((ref["anchor_text"], ref["url"]), (False, None))
                rel_items.append((_METHOD_CATEGORIES.get(rel_method, StepCategory.LLM_VLM_JUDGEMENT), bool(is_relevant)))
                if is_relevant:
                    relevance_pass += 1
                else:
                    relevance_details.append(f"Irrelevant name '{ref['anchor_text']}' for '{ref['url']}'")
            score_8 = calculate_percentage_score(relevance_pass, total_gold, 10)
            checkpoint.add_step(
                name="Link names relevant",
                success=(relevance_pass == total_gold), step_id=8,
                details=f"{relevance_pass}/{total_gold} have relevant names. "
                        + "; ".join(relevance_details[:5]) if relevance_details else f"{relevance_pass}/{total_gold} all relevant",
                score=score_8, max_score=10, execution_time=time.time() - t,
                category=StepCategory.aggregate(rel_items),
            )
        except Exception as e:
            checkpoint.add_step(
                name="Link names relevant",
                success=False, step_id=8, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(8)

        # --- Step 9: No dead hyperlinks ---
        t = time.time()
        try:
            # unique_urls = list({r["url"] for r in doc_refs if r.get("url")})
            unique_urls = list({gold_item["url"] for gold_item in found_items if gold_item.get("url")})
            if len(unique_urls) == 0 and len(gold_data) > 0:
                checkpoint.add_step(
                    name="No dead hyperlinks",
                    success=False, step_id=9,
                    details="No urls found",
                    score=0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.EXECUTION_ERROR,
                )
            else:
                dead_link_tasks = [
                    {"id": url, "func": is_dead_link, "args": (url,), "kwargs": {}}
                    for url in unique_urls
                ]
                dead_check_results = parallel_execute(dead_link_tasks, max_workers=20) if dead_link_tasks else {}
                dead_urls = []
                for url in unique_urls:
                    result = dead_check_results.get(url)
                    if isinstance(result, tuple) and result[0]:
                        dead_urls.append((url, result[1]))
                no_dead = len(dead_urls) == 0
                dead_details = [f"{url} ({reason})" for url, reason in dead_urls[:5]]
                checkpoint.add_step(
                    name="No dead hyperlinks",
                    success=no_dead, step_id=9,
                    details="No dead links found" if no_dead else f"{len(dead_urls)} dead link(s): " + "; ".join(dead_details),
                    score=10 if no_dead else 0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.DETERMINISTIC,
                )
        except Exception as e:
            checkpoint.add_step(
                name="No dead hyperlinks",
                success=False, step_id=9, details=f"Step evaluation error: {e}",
                score=0, max_score=10, execution_time=time.time() - t,
                category=StepCategory.EXECUTION_ERROR,
            )
        _added.add(9)

    except Exception as e:
        for sid, sname, smax in _STEPS:
            if sid not in _added:
                checkpoint.add_step(
                    name=sname, success=False, step_id=sid,
                    details=f"Data acquisition failed — {e}",
                    score=0, max_score=smax, execution_time=0,
                    category=StepCategory.EXECUTION_ERROR,
                )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_4():
    """Checkpoint 4 (50pt): Multiple references — bold, italic, dark cyan 1, 12pt, no duplicate slides."""
    _STEPS = [
        (1, "Bold for multi-slide refs", 10),
        (2, "Italic for multi-slide refs", 10),
        (3, "Dark cyan 1 for multi-slide refs", 10),
        (4, "12pt font size for multi-slide refs", 10),
        (5, "No duplicate slide numbers", 10),
    ]
    start = time.time()
    checkpoint = Checkpoint(total=50, result=0, name="Multiple references")
    _added = set()

    if not document:
        for sid, sname, smax in _STEPS:
            checkpoint.add_step(
                name=sname, success=False, step_id=sid,
                details="Error fetching document content.",
                score=0, max_score=smax, execution_time=0,
                category=StepCategory.EXECUTION_ERROR,
            )
        checkpoint.execution_time = time.time() - start
        return checkpoint

    try:
        doc_refs = doc_refs_cache if doc_refs_cache is not None else extract_reference_links(document)
        multi_slide_gold = [item for item in gold_data if len(item.get("slides") or []) > 1]
        total_multi = len(multi_slide_gold)

        if total_multi == 0:
            for sid, sname, smax in _STEPS:
                checkpoint.add_step(
                    name=sname, success=True, step_id=sid,
                    details="No multi-slide references found — step passes vacuously",
                    score=smax, max_score=smax, execution_time=0,
                    category=StepCategory.VACUOUS_PASS,
                )
                _added.add(sid)
        else:
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

            # --- Step 1: Bold (and only the hyperlink) ---
            t = time.time()
            try:
                bold_pass = 0
                bold_details = []
                for gold_item in multi_slide_gold:
                    ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                    if not ref:
                        bold_details.append(f"Missing: '{gold_item['name']}'")
                        continue
                    if ref["link_is_bold"] and not ref.get("non_link_is_bold", False):
                        bold_pass += 1
                    elif not ref["link_is_bold"]:
                        bold_details.append(f"Hyperlink not bold: '{gold_item['name']}' in '{gold_item['lecture']}'")
                    else:
                        bold_details.append(
                            f"Surrounding text also bold (must be only the hyperlink): "
                            f"'{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                score_1 = calculate_percentage_score(bold_pass, total_multi, 10)
                checkpoint.add_step(
                    name="Bold for multi-slide refs",
                    success=(bold_pass == total_multi), step_id=1,
                    details=f"{bold_pass}/{total_multi} multi-slide refs are bold (only hyperlink). "
                            + "; ".join(bold_details[:5]) if bold_details else f"{bold_pass}/{total_multi} all bold",
                    score=score_1, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.STRUCTURAL,
                )
            except Exception as e:
                checkpoint.add_step(
                    name="Bold for multi-slide refs",
                    success=False, step_id=1, details=f"Step evaluation error: {e}",
                    score=0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.EXECUTION_ERROR,
                )
            _added.add(1)

            # --- Step 2: Italic (and only the hyperlink) ---
            t = time.time()
            try:
                italic_pass = 0
                italic_details = []
                for gold_item in multi_slide_gold:
                    ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                    if not ref:
                        italic_details.append(f"Missing: '{gold_item['name']}'")
                        continue
                    if ref.get("link_is_italic") and not ref.get("non_link_is_italic", False):
                        italic_pass += 1
                    elif not ref.get("link_is_italic"):
                        italic_details.append(
                            f"Hyperlink not italic: '{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                    else:
                        italic_details.append(
                            f"Surrounding text also italic (must be only the hyperlink): "
                            f"'{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                score_2 = calculate_percentage_score(italic_pass, total_multi, 10)
                checkpoint.add_step(
                    name="Italic for multi-slide refs",
                    success=(italic_pass == total_multi), step_id=2,
                    details=f"{italic_pass}/{total_multi} multi-slide refs are italic (only hyperlink). "
                            + "; ".join(italic_details[:5]) if italic_details else f"{italic_pass}/{total_multi} all italic",
                    score=score_2, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.STRUCTURAL,
                )
            except Exception as e:
                checkpoint.add_step(
                    name="Italic for multi-slide refs",
                    success=False, step_id=2, details=f"Step evaluation error: {e}",
                    score=0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.EXECUTION_ERROR,
                )
            _added.add(2)

            # --- Step 3: Dark cyan 1 (and only the hyperlink) ---
            t = time.time()
            try:
                cyan_pass = 0
                cyan_details = []
                for gold_item in multi_slide_gold:
                    ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                    if not ref:
                        cyan_details.append(f"Missing: '{gold_item['name']}'")
                        continue
                    if ref.get("link_is_dark_cyan_1") and not ref.get("non_link_is_dark_cyan_1", False):
                        cyan_pass += 1
                    elif not ref.get("link_is_dark_cyan_1"):
                        cyan_details.append(
                            f"Hyperlink not dark cyan 1: '{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                    else:
                        cyan_details.append(
                            f"Surrounding text also dark cyan 1 (must be only the hyperlink): "
                            f"'{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                score_3 = calculate_percentage_score(cyan_pass, total_multi, 10)
                checkpoint.add_step(
                    name="Dark cyan 1 for multi-slide refs",
                    success=(cyan_pass == total_multi), step_id=3,
                    details=f"{cyan_pass}/{total_multi} multi-slide refs are dark cyan 1 (only hyperlink). "
                            + "; ".join(cyan_details[:5]) if cyan_details else f"{cyan_pass}/{total_multi} all dark cyan 1",
                    score=score_3, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.DETERMINISTIC,
                )
            except Exception as e:
                checkpoint.add_step(
                    name="Dark cyan 1 for multi-slide refs",
                    success=False, step_id=3, details=f"Step evaluation error: {e}",
                    score=0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.EXECUTION_ERROR,
                )
            _added.add(3)

            # --- Step 4: 12pt font size (and only the hyperlink) ---
            t = time.time()
            try:
                font_pass = 0
                font_details = []
                for gold_item in multi_slide_gold:
                    ref = gold_to_doc[(gold_item["lecture"], gold_item["url"])]
                    if not ref:
                        font_details.append(f"Missing: '{gold_item['name']}'")
                        continue
                    link_font = ref.get("link_font_size")
                    if link_font == 12 and not ref.get("non_link_is_12pt", False):
                        font_pass += 1
                    elif link_font != 12:
                        font_details.append(
                            f"Hyperlink not 12pt (got {link_font}pt): '{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                    else:
                        font_details.append(
                            f"Surrounding text also 12pt (must be only the hyperlink): "
                            f"'{gold_item['name']}' in '{gold_item['lecture']}'"
                        )
                score_4 = calculate_percentage_score(font_pass, total_multi, 10)
                checkpoint.add_step(
                    name="12pt font size for multi-slide refs",
                    success=(font_pass == total_multi), step_id=4,
                    details=f"{font_pass}/{total_multi} multi-slide refs are 12pt (only hyperlink). "
                            + "; ".join(font_details[:5]) if font_details else f"{font_pass}/{total_multi} all 12pt",
                    score=score_4, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.DETERMINISTIC,
                )
            except Exception as e:
                checkpoint.add_step(
                    name="12pt font size for multi-slide refs",
                    success=False, step_id=4, details=f"Step evaluation error: {e}",
                    score=0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.EXECUTION_ERROR,
                )
            _added.add(4)

            # --- Step 5: No duplicate slide numbers ---
            t = time.time()
            try:
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
                score_5 = calculate_percentage_score(no_dup_pass, total_multi, 10)
                checkpoint.add_step(
                    name="No duplicate slide numbers",
                    success=(no_dup_pass == total_multi), step_id=5,
                    details=f"{no_dup_pass}/{total_multi} have unique slide numbers. "
                            + "; ".join(dup_details[:5]) if dup_details else f"{no_dup_pass}/{total_multi} all unique",
                    score=score_5, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.DETERMINISTIC,
                )
            except Exception as e:
                checkpoint.add_step(
                    name="No duplicate slide numbers",
                    success=False, step_id=5, details=f"Step evaluation error: {e}",
                    score=0, max_score=10, execution_time=time.time() - t,
                    category=StepCategory.EXECUTION_ERROR,
                )
            _added.add(5)

    except Exception as e:
        for sid, sname, smax in _STEPS:
            if sid not in _added:
                checkpoint.add_step(
                    name=sname, success=False, step_id=sid,
                    details=f"Data acquisition failed — {e}",
                    score=0, max_score=smax, execution_time=0,
                    category=StepCategory.EXECUTION_ERROR,
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
    parser = argparse.ArgumentParser(description="Evaluate docs_37_reference_list instance_2")
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
