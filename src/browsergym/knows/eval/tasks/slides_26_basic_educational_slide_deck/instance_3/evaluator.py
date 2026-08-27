import os
import re
import sys
import time
import argparse
import shutil

# Base path resolution
BASE_PATH = None
if os.path.exists("/app/src"):
    BASE_PATH = "/app"
elif os.path.exists("/scratch"):
    BASE_PATH = "."
else:
    BASE_PATH = os.getcwd()
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.text_utils import text_fuzzy_match_contained_long
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_text,
    extract_text_boxes_from_slide,
    extract_slide_images,
    extract_slide_links,
    extract_slide_links_with_positions,
    extract_title_text,
    extract_bullet_point_texts,
    extract_image_source_urls,
    validate_bullet_points,
    download_slide_image,
    get_element_bbox,
    get_text_style_from_shape,
    is_text_big,
    get_slide_dimensions,
    find_url_below_image,
    _extract_links_from_text_element,
)
from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.llm_utils import evaluate_with_llm
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute
from src.browsergym.knows.eval.tasks.slides_26_basic_educational_slide_deck.utils import (
    parse_task_md,
    filter_non_image_links,
    find_small_font_credit,
    match_source_image,
    ensure_model,
    set_cached_model,
    reset_model_state,
    normalize_for_match,
    contains_normalized,
    find_with_flexible_whitespace,
    img_area,
    font_pt,
    placeholder_type,
    bbox_center_x,
    get_master_placeholder_font_pt,
    is_dark_orange,
    fill_failure_steps,
    call_with_timeout,
    join_extras,
    has_min_slides,
    score_credit_match,
    cluster_images_by_phash,
    to_deck_positions,
    EMU_PER_INCH,
    llm_extract_source_url,
    CREDIT_KEYWORDS,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/slides_26_basic_educational_slide_deck/instance_3/")
DATA_DIR = os.path.join(TASK_DIR, "data/")

# Module-level so the catastrophic-failure handler can still emit a complete report
# if parse_task_md raises. Per-CP guards check `_task_parse_error` before running.
_task_details, _task_parse_error = parse_task_md(TASK_DIR)
EXPECTED_TOPIC = _task_details["topic"] if _task_details else None
EXPECTED_PRESENTER = _task_details["presenter"] if _task_details else None
EXPECTED_CONTENT_SLIDES = _task_details["num_content_slides"] if _task_details else None

# Module-level so grade_checkpoints' catastrophic-failure path can recover the spec.
CP1_STEPS = [
    ("Topic Big Bold Font", 5), ("Topic Dark Orange", 2), ("Presenter Name", 3),
    ("Background Image Relevant", 5), ("Image Credit Small Font", 2),
    ("Image Source URL Accessible", 3),
]
CP2_STEPS = [
    ("Slide Count", 5), ("Topic Relevance", 10), ("Source in Lower-Left", 10),
    ("Info From Source", 10), ("Unique Headings", 10), ("Heading Bold Italic", 10),
    ("Bullet Points", 5), ("No Text Overflow", 5),
    ("Content Paraphrased", 10), ("Content Left Side", 10),
]
CP3_STEPS = [("Covers All Concepts", 4), ("Original Text", 3), ("Engagement Prompt", 3)]
CP4_STEPS = [
    ("Image Relevant", 10), ("Image Right Side", 10),
    ("Image Source Credit", 10), ("Image Source URL Matches", 10),
    ("Images Unique", 10),
]

# Global state
model_id = "gemini-3-flash-google-ai"
presentation_id = None
presentation_data = None
DRIVE_SERVICE = None
SLIDES_SERVICE = None


def setup_presentation(workspace_doc_id):
    """Lazy-init Google services + fetch presentation. Leaves presentation_data=None on failure."""
    global presentation_id, presentation_data, DRIVE_SERVICE, SLIDES_SERVICE

    # Reset first so bad input clears any stale state from a prior run.
    presentation_id = None
    presentation_data = None

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    if SLIDES_SERVICE is None:
        services, ok = call_with_timeout(initialize_google_services, 30, "initialising Google services", service_type="slides")
        if not ok:
            return
        DRIVE_SERVICE, SLIDES_SERVICE = services

    print(f"Using workspace presentation ID: {workspace_doc_id}")
    data, ok = call_with_timeout(
        lambda: SLIDES_SERVICE.presentations().get(presentationId=workspace_doc_id).execute(),
        30, f"fetching presentation {workspace_doc_id}",
    )
    if ok:
        presentation_data = data
        presentation_id = workspace_doc_id


def grade_checkpoint_1():
    """Checkpoint 1 (20pt): Title slide."""
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Title Slide")

    if _task_parse_error is not None:
        fill_failure_steps(checkpoint, CP1_STEPS, _task_parse_error)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if presentation_data is None:
        fill_failure_steps(checkpoint, CP1_STEPS, "Failed to load presentation (auth or API failure)")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if not has_min_slides(presentation_data, 1):
        fill_failure_steps(checkpoint, CP1_STEPS, "No slides found in presentation")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    temp_dir = os.path.join(DATA_DIR, "temp_title_images")

    try:
        title_slide = presentation_data['slides'][0]
        text_boxes = extract_text_boxes_from_slide(title_slide)
        slide_w, slide_h = get_slide_dimensions(presentation_data)
        if slide_h is None:
            fill_failure_steps(checkpoint, CP1_STEPS, "Slide dimensions unavailable")
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint
        topic_style = None  # set in step 1, reused by step 2
        bg_image_path = None  # set in step 4, reused by step 6

        # ---- Step 1 (5 pt): Topic in big, bold font (>= 40pt) ----
        step_start = time.time()
        topic_bold_big = False
        topic_text_found = False
        is_big = False
        is_bold = False
        resolved_master_pt = None
        assumed_big = False  # True when fontSize unresolvable and we assume big
        step1_category = StepCategory.DETERMINISTIC
        details = ""
        try:
            # Prefer TITLE placeholder w/ topic; else largest matching box; else multi-box
            # fallback (joined slide text contains topic) using TITLE placeholder for style.
            direct = [tb for tb in text_boxes
                      if contains_normalized(tb.get('text', ''), EXPECTED_TOPIC)]
            matched_tb = None
            match_via = None
            if direct:
                title_subset = [tb for tb in direct
                                if placeholder_type(tb) in ('TITLE', 'CENTERED_TITLE')]
                pool = title_subset or direct
                # Rank by font size first; bbox area tie-breaks for inherited fontSize.
                matched_tb = max(pool, key=lambda tb: (
                    font_pt(tb),
                    tb['bbox'].get('width', 0) * tb['bbox'].get('height', 0),
                ))
                match_via = "title placeholder" if title_subset else "largest matching box"
            elif contains_normalized(extract_slide_text(title_slide), EXPECTED_TOPIC):
                # Topic spans multiple boxes; use a non-empty TITLE placeholder for style.
                matched_tb = next((tb for tb in text_boxes
                                   if placeholder_type(tb) in ('TITLE', 'CENTERED_TITLE')
                                   and (tb.get('text', '') or '').strip()), None)
                # Surface placeholder text so reader can tell if it represents the topic.
                if matched_tb:
                    placeholder_snippet = (matched_tb.get('text', '') or '').strip()[:40]
                    match_via = f"multi-box (TITLE='{placeholder_snippet}' for style)"

            if matched_tb is not None:
                topic_text_found = True
                shape = matched_tb['element'].get('shape', {})
                # Pass presentation_data so themeColor titles get foregroundColor pre-resolved
                # — step 2's is_dark_orange then works without its own resolution path.
                topic_style = get_text_style_from_shape(shape, presentation=presentation_data)
                is_bold = bool(topic_style.get('bold'))
                is_big = is_text_big(topic_style, min_pt=40)
                # Inherited fontSize: try the master/layout size; if that's also unset,
                # assume big enough rather than punishing decks where the API hides it.
                if not is_big and topic_style.get('fontSize') is None:
                    ph_type = shape.get('placeholder', {}).get('type', '')
                    if ph_type in ('TITLE', 'CENTERED_TITLE'):
                        resolved_master_pt = get_master_placeholder_font_pt(presentation_data, ph_type)
                        if resolved_master_pt is not None:
                            is_big = resolved_master_pt >= 40
                        else:
                            is_big = True
                            assumed_big = True
                topic_bold_big = is_bold and is_big

            if topic_text_found:
                font_size = topic_style.get('fontSize')
                if font_size:
                    font_info = f"{font_size['magnitude']}{font_size.get('unit', 'PT')}"
                elif resolved_master_pt is not None:
                    font_info = f"{resolved_master_pt}PT (resolved from master/layout)"
                else:
                    font_info = "inherited (title placeholder, master size unknown)"
                details = f"Topic found via {match_via}. Bold: {is_bold}, Font: {font_info}, Big (>=40pt): {is_big}"
            else:
                details = f"Topic '{EXPECTED_TOPIC}' not found on title slide"
        except Exception as e:
            details = f"Error during topic check: {e}"
            step1_category = StepCategory.EXECUTION_ERROR
        if topic_bold_big and assumed_big:
            step1_category = StepCategory.VACUOUS_PASS
        checkpoint.add_step("Topic Big Bold Font", topic_bold_big, 1, details, max_score=5,
                            execution_time=time.time() - step_start, category=step1_category)

        # ---- Step 2 (2 pt): Topic is in dark orange ----
        step_start = time.time()
        is_orange = False
        step2_category = StepCategory.FUZZY_MATCH  # HSV hue/tolerance color check
        details = ""
        try:
            if topic_style:
                is_orange = is_dark_orange(topic_style, presentation_data=presentation_data)
                fg = topic_style.get('foregroundColor')
                theme_name = topic_style.get('foregroundThemeColor')
                if fg:
                    details = f"RGB: ({fg.get('red', 0):.3f}, {fg.get('green', 0):.3f}, {fg.get('blue', 0):.3f}), dark orange: {is_orange}"
                elif theme_name:
                    details = f"Theme color '{theme_name}' (resolved via master scheme), dark orange: {is_orange}"
                else:
                    details = "Could not determine text color (no rgbColor or themeColor on shape)"
            else:
                details = "Topic text not found, cannot check color"
                step2_category = StepCategory.DEPENDENCY_NOT_EVALUATED
        except Exception as e:
            details = f"Error during color check: {e}"
            step2_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Topic Dark Orange", is_orange, 2, details, max_score=2,
                            execution_time=time.time() - step_start, category=step2_category)

        # ---- Step 3 (3 pt): Presenter's name ----
        step_start = time.time()
        name_found = False
        step3_category = StepCategory.DETERMINISTIC  # normalized text containment
        details = ""
        try:
            name_found = contains_normalized(extract_slide_text(title_slide), EXPECTED_PRESENTER)
            details = f"Found '{EXPECTED_PRESENTER}' on title slide" if name_found else f"'{EXPECTED_PRESENTER}' not found on title slide"
        except Exception as e:
            details = f"Error during presenter name check: {e}"
            step3_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Presenter Name", name_found, 3, details, max_score=3,
                            execution_time=time.time() - step_start, category=step3_category)

        # ---- Step 4 (5 pt): Background image relevant to topic ----
        step_start = time.time()
        bg_success = False
        step4_category = StepCategory.LLM_VLM_JUDGEMENT  # VLM relevance verdict decides
        details = ""
        try:
            # Sort by area DESC so binary_judge_image (alphabetical os.listdir) hits the
            # largest first. pageBackgroundFill is always slide-sized so it goes first.
            image_metas = sorted(
                [info for info in extract_slide_images(title_slide, presentation_id, SLIDES_SERVICE)
                 if info.get('contentUrl')],
                key=img_area, reverse=True,
            )
            image_urls = []
            bg_url = title_slide.get('pageProperties', {}).get('pageBackgroundFill', {}).get('stretchedPictureFill', {}).get('contentUrl')
            if bg_url:
                image_urls.append(bg_url)
            # Regular images only count as backgrounds when sized like one — otherwise a
            # small relevant decoration (logo/thumbnail) would falsely pass the step.
            slide_area = (slide_w or 0) * (slide_h or 0)
            BG_AREA_THRESHOLD = 0.70
            small_image_count = 0
            for info in image_metas:
                if slide_area and img_area(info) >= BG_AREA_THRESHOLD * slide_area:
                    image_urls.append(info['contentUrl'])
                else:
                    small_image_count += 1

            if not image_urls:
                if small_image_count:
                    # Rejected by the area gate — a geometric coverage check.
                    step4_category = StepCategory.SPATIAL
                    details = (f"Title slide has {small_image_count} image(s) but none cover the slide "
                               f"as a background (need ≥{int(BG_AREA_THRESHOLD * 100)}% of slide area "
                               f"or pageBackgroundFill)")
                else:
                    # No images at all — an element-count observation.
                    step4_category = StepCategory.STRUCTURAL
                    details = "No images found on title slide"
            else:
                # Pre-clean: a prior crash could leave stale title_img_*.png around.
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                os.makedirs(temp_dir, exist_ok=True)
                saved_count = 0
                save_failed_count = 0
                for idx, url in enumerate(image_urls):
                    img = download_slide_image(url)
                    if not img:
                        continue
                    # Per-image try/except so one PIL.save failure doesn't poison the batch.
                    # Zero-pad idx so os.listdir alphabetical sort matches numeric order.
                    try:
                        img.save(os.path.join(temp_dir, f"title_img_{idx:03d}.png"))
                        saved_count += 1
                    except Exception as save_err:
                        save_failed_count += 1
                        print(f"Failed to save title image {idx}: {save_err}")

                if saved_count == 0:
                    step4_category = StepCategory.EXECUTION_ERROR
                    details = (f"Found {len(image_urls)} image URL(s) but failed to download/save any"
                               + (f" ({save_failed_count} save errors)" if save_failed_count else ""))
                else:
                    model = ensure_model(model_id)
                    if model is None:
                        step4_category = StepCategory.EXECUTION_ERROR
                        details = "Model unavailable; cannot judge background image relevance"
                    else:
                        matching = binary_judge_image(
                            model, temp_dir,
                            f"Is this image thematically related to '{EXPECTED_TOPIC}'? "
                            f"Answer Yes if it depicts subject matter, objects, scenes, people, or symbols "
                            f"that a viewer would reasonably associate with {EXPECTED_TOPIC}."
                        )
                        if matching:
                            bg_success = True
                            bg_image_path = matching
                            details = f"Found relevant background image: {os.path.basename(matching)}"
                        else:
                            # Leave bg_image_path=None so step 6 doesn't match an unrelated image.
                            details = f"Found {saved_count} image(s) but none are relevant to {EXPECTED_TOPIC}"
        except Exception as e:
            details = f"Error checking background image: {e}"
            step4_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Background Image Relevant", bg_success, 4, details, max_score=5,
                            execution_time=time.time() - step_start, category=step4_category)

        # ---- Step 5 (2 pt): Image credit in small font ----
        step_start = time.time()
        credit_found = False
        step5_category = StepCategory.SPATIAL  # font-size + y-position + keyword gate
        details = ""
        try:
            credit_found, credit_tb = find_small_font_credit(text_boxes, slide_h)
            if credit_found:
                snippet = (credit_tb or {}).get('text', '(text unavailable)')[:80]
                details = f"Found small-font credit text: '{snippet}'"
            else:
                details = "No small-font image credit found on title slide"
        except Exception as e:
            details = f"Error during image credit check: {e}"
            step5_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Image Credit Small Font", credit_found, 5, details, max_score=2,
                            execution_time=time.time() - step_start, category=step5_category)

        # ---- Step 6 (3 pt): Source URL image matches background image ----
        step_start = time.time()
        image_matches = False
        step6_score = 0
        # Tiered exact -> pHash -> VLM matcher; VLM is the final arbiter unless the
        # recorded match method says an earlier tier decided.
        step6_category = StepCategory.LLM_VLM_JUDGEMENT
        details = ""
        try:
            slide_links = extract_slide_links(title_slide)
            model = ensure_model(model_id)
            if model is None:
                details = "Model unavailable; cannot match source URL"
                step6_category = StepCategory.EXECUTION_ERROR
            else:
                os.makedirs(temp_dir, exist_ok=True)
                # If step 4 didn't pick a topical bg, relax to any saved title image
                # (step 6 awards 1/3 partial credit in that case).
                candidate_image = bg_image_path
                relaxed = False
                if candidate_image is None and os.path.isdir(temp_dir):
                    saved = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                             if f.startswith('title_img_')]
                    if saved:
                        candidate_image = saved[0]
                        relaxed = True
                if candidate_image is None:
                    details = "No saved title image available to verify source URL"
                    step6_category = StepCategory.EXECUTION_ERROR
                else:
                    image_matches, details = match_source_image(slide_links, candidate_image,
                                                                 temp_dir, 0, model)
                    # The details string records the deciding tier of the match.
                    method_m = re.search(r"\((exact|perceptual_hash)\)", details or "")
                    if "No image or links available" in (details or ""):
                        step6_category = StepCategory.EXECUTION_ERROR  # no candidate URL to test
                    elif method_m:
                        step6_category = StepCategory.from_match_method(method_m.group(1))
                    else:
                        step6_category = StepCategory.LLM_VLM_JUDGEMENT
                    if image_matches:
                        # Full credit only on a topical-bg match; relaxed match earns 1/3.
                        if relaxed:
                            step6_score = 1
                            details = f"(step 4 found no topical bg; partial credit 1/3) {details}"
                        else:
                            step6_score = 3
        except Exception as e:
            details = f"Error checking source image match: {e}"
            step6_category = StepCategory.EXECUTION_ERROR
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        # success=True only on full credit; partial credit goes through `score=`.
        checkpoint.add_step("Image Source URL Matches", step6_score == 3, 6, details,
                            score=step6_score, max_score=3,
                            execution_time=time.time() - step_start, category=step6_category)

    except Exception as e:
        print(f"Error in checkpoint 1: {e}")
        fill_failure_steps(checkpoint, CP1_STEPS, f"Skipped due to checkpoint error: {e}", only_missing=True)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2 (85pt, 9 steps): Content Slides."""
    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=85, result=0, name="Content Slides")

    if _task_parse_error is not None:
        fill_failure_steps(checkpoint, CP2_STEPS, _task_parse_error)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if presentation_data is None:
        fill_failure_steps(checkpoint, CP2_STEPS, "Failed to load presentation (auth or API failure)")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if not has_min_slides(presentation_data, 3):
        fill_failure_steps(checkpoint, CP2_STEPS, "Not enough slides in presentation")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        slides = presentation_data['slides']
        content_slides = slides[1:-1]
        actual_count = len(content_slides)
        slide_w, slide_h = get_slide_dimensions(presentation_data)
        if slide_w is None:
            fill_failure_steps(checkpoint, CP2_STEPS, "Slide dimensions unavailable")
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Pre-extract per-slide data; one bad slide becomes a stub.
        slide_data = []
        for slide in content_slides:
            try:
                # Plain-text-paragraph slides return [] from extract_bullet_point_texts;
                # fall back to lines from the largest non-title shape so steps 4/8 see
                # body text. Step 7 is unaffected (it re-checks via validate_bullet_points).
                bullets = extract_bullet_point_texts(slide)
                if not bullets:
                    title_norm = (extract_title_text(slide) or '').strip().lower()
                    body_candidates = []
                    for _el in slide.get('pageElements', []):
                        if 'shape' not in _el or 'text' not in _el['shape']:
                            continue
                        if _el['shape'].get('placeholder', {}).get('type', '') in (
                                'TITLE', 'CENTERED_TITLE', 'SUBTITLE'):
                            continue
                        _txt = ''
                        for _tr in _el['shape']['text'].get('textElements', []):
                            _run = _tr.get('textRun')
                            if _run:
                                _txt += _run.get('content', '')
                        _txt = _txt.strip()
                        if not _txt or _txt.lower() == title_norm:
                            continue
                        body_candidates.append((len(_txt), _txt))
                    if body_candidates:
                        body_candidates.sort(reverse=True, key=lambda c: c[0])
                        bullets = [ln.strip() for ln in body_candidates[0][1].splitlines()
                                   if len(ln.strip()) >= 3]
                slide_data.append({
                    'text': extract_slide_text(slide),
                    'text_boxes': extract_text_boxes_from_slide(slide),
                    'title': extract_title_text(slide),
                    'bullets': bullets,
                    'links_flat': extract_slide_links(slide),
                    'links_with_pos': extract_slide_links_with_positions(slide),
                    'source_url': None,
                    'source_content': None,
                    'extraction_failed': False,
                })
            except Exception as e:
                print(f"Error pre-extracting content slide: {e}")
                # Fresh lists per stub so mutations can't bleed across slides.
                slide_data.append({
                    'text': '', 'text_boxes': [], 'title': '', 'bullets': [],
                    'links_flat': [], 'links_with_pos': [],
                    'source_url': None, 'source_content': None,
                    'extraction_failed': True,
                })

        # ---- Step 1 (5 pt): Exactly N content slides ----
        step_start = time.time()
        step1_category = StepCategory.STRUCTURAL  # element-count check
        try:
            count_ok = actual_count == EXPECTED_CONTENT_SLIDES
            details = f"Found {actual_count}/{EXPECTED_CONTENT_SLIDES} content slides"
        except Exception as e:
            count_ok = False
            details = f"Error during slide count: {e}"
            step1_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Slide Count", count_ok, 1, details, max_score=5,
                            execution_time=time.time() - step_start, category=step1_category)

        # ---- Step 2 (10 pt): All content slides related to topic ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step2_category = StepCategory.LLM_VLM_JUDGEMENT  # per-slide LLM relevance verdicts
        try:
            model = ensure_model(model_id)
            if model is None:
                details = "Model unavailable; cannot judge topic relevance"
                step2_category = StepCategory.EXECUTION_ERROR
            else:
                relevance_tasks = [
                    {
                        'id': f'relevance_{i}',
                        'func': evaluate_with_llm,
                        'args': (
                            f"Is the following slide content related to the topic '{EXPECTED_TOPIC}'?\n\n"
                            f"Slide text: {sd['text'][:2000]}",
                            model,
                        ),
                        'kwargs': {'return_type': 'bool'},
                    }
                    for i, sd in enumerate(slide_data)
                ]
                relevance_results = parallel_execute(relevance_tasks, max_workers=5, timeout=120)
                related_count = sum(1 for v in relevance_results.values() if v is True)
                crashed = sum(1 for v in relevance_results.values() if v is None)
                failed_slides = []
                for tid, v in relevance_results.items():
                    if v is True:
                        continue
                    try:
                        failed_slides.append(int(tid.replace('relevance_', '')))
                    except (ValueError, AttributeError):
                        failed_slides.append(tid)
                failed_slides.sort(key=lambda x: (isinstance(x, str), x))
                success = related_count == actual_count
                step_score = calculate_percentage_score(related_count, actual_count)
                details = f"{related_count}/{actual_count} content slides related to topic" + join_extras(
                    f"{crashed} LLM call(s) crashed" if crashed else None,
                    f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
                )
        except Exception as e:
            details = f"Error during topic relevance check: {e}"
            step2_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Topic Relevance", success, 2, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step2_category)

        # ---- Step 3 (10 pt): Source in small font in lower-left corner ----
        # Partial credit: 4-of-4 criteria = full, 3-of-4 = half. Also captures source_url
        # from the best-match text box (used by steps 4 and 8).
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step3_category = StepCategory.SPATIAL  # position + font-size criteria per slide
        try:
            # Bare "http" excluded: a body bullet "Visit https://..." would otherwise match.
            credit_keywords = ("www", ".com", ".org", ".edu", ".net",
                               "source", "reference", "credit", "citation", "cited")
            full_pass_count = 0
            partial_pass_count = 0
            extraction_failed_count = 0
            failed_slides = []
            step3_items = []  # (category, success) per slide for StepCategory.aggregate()
            score_sum = 0.0  # 1.0 per full, 0.5 per partial
            for i, sd in enumerate(slide_data):
                if sd.get('extraction_failed'):
                    extraction_failed_count += 1
                    failed_slides.append(i)
                    step3_items.append((StepCategory.EXECUTION_ERROR, False))
                    continue
                non_image_links = filter_non_image_links(sd['links_flat'])
                best_count = 0
                best_tb = None
                for tb in sd['text_boxes']:
                    matched = score_credit_match(tb, slide_w, slide_h, credit_keywords)
                    if matched > best_count:
                        best_count = matched
                        best_tb = tb

                if best_count == 4:
                    full_pass_count += 1
                    score_sum += 1.0
                    step3_items.append((StepCategory.SPATIAL, True))
                elif best_count == 3:
                    partial_pass_count += 1
                    score_sum += 0.5
                    failed_slides.append(i)
                    step3_items.append((StepCategory.SPATIAL, False))
                else:
                    failed_slides.append(i)
                    step3_items.append((StepCategory.SPATIAL, False))

                # Source URL: prefer link inside the best-match text box (when ≥3 criteria).
                if best_count >= 3 and best_tb is not None:
                    tb_links = _extract_links_from_text_element(best_tb['element'].get('shape', {}).get('text', {}))
                    tb_non_image = filter_non_image_links(tb_links)
                    if tb_non_image:
                        sd['source_url'] = tb_non_image[0]

                # Body-link fallback when the credit box wasn't detected. Step 4's LLM
                # judges traceability, so a wrong link gets rejected there.
                if not sd['source_url'] and non_image_links:
                    # Prefer the lowest-on-slide link (most likely a footer citation).
                    non_image_set = set(non_image_links)
                    positioned = [l for l in sd.get('links_with_pos', []) if l.get('url') in non_image_set]
                    if positioned:
                        sd['source_url'] = max(positioned, key=lambda l: l.get('bbox', {}).get('y', 0)).get('url')
                    else:
                        sd['source_url'] = non_image_links[0]

            success = full_pass_count == actual_count
            step_score = calculate_percentage_score(score_sum, actual_count)
            step3_category = StepCategory.aggregate(step3_items)
            details = f"{full_pass_count}/{actual_count} slides have full credit citation" + join_extras(
                f"{partial_pass_count} partial (3-of-4 criteria)" if partial_pass_count else None,
                f"{extraction_failed_count} slide(s) extraction failed" if extraction_failed_count else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            details = f"Error during source citation check: {e}"
            step3_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Source in Lower-Left", success, 3, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step3_category)

        # ---- LLM fallback for source URLs the regex extractor missed ----
        # Bare-domain citations (e.g. "kids.kiddle.co/page") have no protocol and
        # slip past the regex; ask the LLM in parallel for slides still missing a URL.
        try:
            llm_model = ensure_model(model_id)
            if llm_model is not None:
                llm_url_tasks = [
                    {'id': f'llm_url_{i}', 'func': llm_extract_source_url,
                     'args': (sd['text'], llm_model)}
                    for i, sd in enumerate(slide_data)
                    if not sd.get('source_url') and sd.get('text')
                ]
                if llm_url_tasks:
                    llm_url_results = parallel_execute(llm_url_tasks, max_workers=5, timeout=90)
                    for tid, url in llm_url_results.items():
                        if not url:
                            continue
                        try:
                            slide_idx = int(tid.replace('llm_url_', ''))
                        except (ValueError, AttributeError):
                            continue
                        if 0 <= slide_idx < len(slide_data):
                            slide_data[slide_idx]['source_url'] = url
        except Exception as e:
            print(f"LLM source-URL fallback failed: {e}")

        # ---- Fetch source content for steps 4 and 8 (parallelized) ----
        # Lazy import: slim envs may lack Playwright; the wrapper adds curl-cffi as a
        # 5th strategy for Cloudflare-protected hosts whose TLS fingerprint gets blocked.
        try:
            from src.browsergym.knows.eval.eval_utils.web_utils import (
                fetch_with_fallbacks_extended as fetch_with_fallbacks,
            )
        except Exception as e:
            print(f"Source-content fetcher unavailable: {e}")
            fetch_with_fallbacks = None

        if fetch_with_fallbacks is not None:
            # 30s/strategy (vs. 15s default) avoids false unfetchables on slow-rendering pages.
            fetch_tasks = [
                {'id': f'fetch_{i}', 'func': fetch_with_fallbacks, 'args': (sd['source_url'],),
                 'kwargs': {'timeout': 30}}
                for i, sd in enumerate(slide_data) if sd['source_url']
            ]
            if fetch_tasks:
                try:
                    # 3 workers keeps peak memory bounded (Chromium per fetch); 300s wall.
                    fetch_results = parallel_execute(fetch_tasks, max_workers=3, timeout=300)
                    for i, sd in enumerate(slide_data):
                        res = fetch_results.get(f'fetch_{i}')
                        if res and isinstance(res, tuple):
                            if res[0]:
                                sd['source_content'] = res[0]
                            sd['source_fetch_status'] = res[1] if len(res) > 1 else None
                        elif res is None and sd['source_url']:
                            sd['source_fetch_status'] = 'parallel timeout / crash'
                except Exception as e:
                    print(f"Error fetching source content in parallel: {e}")

        # ---- Step 4 (10 pt): Information actually comes from the cited source ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step4_category = StepCategory.LLM_VLM_JUDGEMENT  # LLM traceability verdicts
        try:
            model = ensure_model(model_id)
            if model is None:
                details = "Model unavailable; cannot verify content-source traceability"
                step4_category = StepCategory.EXECUTION_ERROR
            else:
                source_check_tasks = []
                step4_items = []  # (category, success) per slide for StepCategory.aggregate()
                no_url_count = 0
                unfetchable_count = 0
                no_bullets_count = 0
                pre_skipped = []  # slides that never reached the LLM
                unfetchable_reasons = []  # "deck_pos: status (url=...)" diagnostic strings
                source_system_prompt = (
                    "You are verifying whether slide content could have been derived from a given source. "
                    "Answer 'Yes' if the bullet points cover topics or facts that appear in the source text, "
                    "even if the wording is different. Answer 'No' only if the bullet points discuss topics "
                    "completely unrelated to the source. Respond with ONLY 'yes' or 'no'."
                )
                for i, sd in enumerate(slide_data):
                    if not sd['source_url']:
                        no_url_count += 1
                        pre_skipped.append(i)
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        continue
                    if not sd['source_content']:
                        unfetchable_count += 1
                        pre_skipped.append(i)
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        # Surface the URL so LLM-extractor casing/spacing bugs are visible.
                        status = sd.get('source_fetch_status') or 'unknown'
                        url_snippet = (sd.get('source_url') or '')[:80]
                        unfetchable_reasons.append(f"{i + 2}: {status} (url={url_snippet!r})")
                        continue
                    if not sd['bullets']:
                        no_bullets_count += 1
                        pre_skipped.append(i)
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        continue
                    # Threshold 3 chars so short factual bullets ("Atari 2600") aren't dropped.
                    all_bullets = "\n".join(f"- {b}" for b in sd['bullets'] if len(b.strip()) >= 3)
                    if not all_bullets:
                        no_bullets_count += 1
                        pre_skipped.append(i)
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        continue
                    heading = (sd['title'] or '').strip()
                    source_text = sd['source_content']
                    # Anchor source text on heading position (skip short headings — too noisy).
                    if heading and len(heading) >= 15:
                        idx = source_text.lower().find(heading.lower()[:20])
                        if idx >= 0:
                            source_text = source_text[max(0, idx - 200):]
                    source_check_tasks.append({
                        'id': f'source_{i}',
                        'func': evaluate_with_llm,
                        'args': (
                            f"A student created the following bullet points for a presentation slide about '{heading}'. "
                            f"The student claims these bullet points are based on the webpage text shown below. "
                            f"The bullet points are expected to be paraphrased or summarized, not copied verbatim. "
                            f"Do the bullet points contain information that is consistent with and could reasonably "
                            f"have been learned from the source text?\n\n"
                            f"Bullet points:\n{all_bullets}\n\n"
                            f"Source webpage text:\n{source_text[:10000]}",
                            model,
                        ),
                        'kwargs': {'return_type': 'bool', 'system_prompt': source_system_prompt},
                    })

                source_results = parallel_execute(source_check_tasks, max_workers=5, timeout=120) if source_check_tasks else {}
                info_from_source_count = sum(1 for v in source_results.values() if v is True)
                crashed = sum(1 for v in source_results.values() if v is None)
                failed_slides = list(pre_skipped)
                # Per-slide reasons (False = not traceable, None = LLM crashed) so
                # the failure summary explains every slide, not just fetch failures.
                llm_failure_reasons = []
                for tid, v in source_results.items():
                    # None = LLM call crashed; True/False = LLM verdict.
                    step4_items.append((StepCategory.EXECUTION_ERROR, False) if v is None
                                       else (StepCategory.LLM_VLM_JUDGEMENT, v is True))
                    try:
                        idx = int(tid.replace('source_', ''))
                    except (ValueError, AttributeError):
                        if v is not True:
                            failed_slides.append(tid)
                        continue
                    # Stash verdict for step 8's traceability-gated paraphrase award.
                    if 0 <= idx < len(slide_data):
                        slide_data[idx]['traceable'] = v
                    if v is True:
                        continue
                    failed_slides.append(idx)
                    reason = "LLM crashed" if v is None else "not traceable per LLM"
                    llm_failure_reasons.append(f"{idx + 2}: {reason}")
                failed_slides.sort(key=lambda x: (isinstance(x, str), x))

                success = info_from_source_count == actual_count
                step_score = calculate_percentage_score(info_from_source_count, actual_count)
                step4_category = StepCategory.aggregate(step4_items)
                unfetchable_summary = (
                    f"{unfetchable_count} URL unfetchable [{'; '.join(unfetchable_reasons)}]"
                    if unfetchable_reasons else
                    (f"{unfetchable_count} URL unfetchable" if unfetchable_count else None)
                )
                llm_failure_summary = (
                    f"{len(llm_failure_reasons)} LLM rejected [{'; '.join(llm_failure_reasons)}]"
                    if llm_failure_reasons else None
                )
                details = f"{info_from_source_count}/{actual_count} slides have content traceable to cited source" + join_extras(
                    f"{no_url_count} no source URL" if no_url_count else None,
                    unfetchable_summary,
                    f"{no_bullets_count} no evaluable bullets" if no_bullets_count else None,
                    llm_failure_summary,
                    f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
                )
        except Exception as e:
            details = f"Error during source-content check: {e}"
            step4_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Info From Source", success, 4, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step4_category)

        # ---- Step 5 (10 pt): Unique, clear headings related to subsection ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step5_category = StepCategory.LLM_VLM_JUDGEMENT  # LLM heading-relevance verdicts
        try:
            model = ensure_model(model_id)
            if model is None:
                details = "Model unavailable; cannot judge heading relevance"
                step5_category = StepCategory.EXECUTION_ERROR
            else:
                headings = [(sd['title'] or '').strip() for sd in slide_data]
                non_empty = [h for h in headings if h]
                unique_headings = set(normalize_for_match(h) for h in non_empty)
                all_unique = len(unique_headings) == len(non_empty)
                all_have_headings = len(non_empty) == actual_count

                # Task id encodes original slide index for direct failure-report mapping.
                heading_tasks = [
                    {
                        'id': f'heading_{slide_i}',
                        'func': evaluate_with_llm,
                        'args': (
                            f"Is this heading a valid subsection title for the topic '{EXPECTED_TOPIC}'?\n\n"
                            f"Heading: {heading}",
                            model,
                        ),
                        'kwargs': {'return_type': 'bool'},
                    }
                    for slide_i, heading in enumerate(headings)
                    if heading
                ]
                heading_results = parallel_execute(heading_tasks, max_workers=5, timeout=120) if heading_tasks else {}
                related_headings = sum(1 for v in heading_results.values() if v is True)

                missing_heading_slides = [i for i, h in enumerate(headings) if not h]
                # Duplicate slide indices: every slide whose normalized heading occurs >1 time.
                norm_to_indices = {}
                for i, h in enumerate(headings):
                    if not h:
                        continue
                    norm_to_indices.setdefault(normalize_for_match(h), []).append(i)
                duplicate_slides = [i for idxs in norm_to_indices.values() if len(idxs) > 1 for i in idxs]
                # Non-related: LLM didn't return True for this slide's heading.
                non_related_slides = []
                for tid, v in heading_results.items():
                    if v is True:
                        continue
                    try:
                        non_related_slides.append(int(tid.replace('heading_', '')))
                    except (ValueError, AttributeError):
                        non_related_slides.append(tid)
                failed_slides = sorted(
                    set(missing_heading_slides) | set(duplicate_slides) | set(non_related_slides),
                    key=lambda x: (isinstance(x, str), x),
                )

                success = all_have_headings and all_unique and related_headings == len(non_empty)
                # If the deterministic dedupe alone rejected (all headings present and
                # LLM-approved, but duplicates exist), the decider was deterministic.
                if (not success and all_have_headings and not all_unique
                        and related_headings == len(non_empty)):
                    step5_category = StepCategory.DETERMINISTIC
                # Cap score at unique count so identical headings can't earn full credit.
                effective_related = min(related_headings, len(unique_headings))
                step_score = calculate_percentage_score(effective_related, actual_count)
                details = f"{len(non_empty)}/{actual_count} slides have headings"
                if missing_heading_slides:
                    details += f" (missing on slides: {to_deck_positions(missing_heading_slides)})"
                details += f", {'all unique' if all_unique else f'duplicates found ({len(non_empty) - len(unique_headings)})'}"
                details += f", {related_headings}/{len(non_empty)} related to topic"
                if failed_slides:
                    details += f", failed: slide(s) {to_deck_positions(failed_slides)}"
        except Exception as e:
            details = f"Error during heading check: {e}"
            step5_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Unique Headings", success, 5, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step5_category)

        # ---- Step 6 (10 pt): Heading is bold and italic ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step6_category = StepCategory.DETERMINISTIC  # bold/italic style flags
        try:
            bold_italic_count = 0
            no_heading_count = 0
            no_match_count = 0
            extraction_failed_count = 0
            failed_slides = []
            for i, sd in enumerate(slide_data):
                if sd.get('extraction_failed'):
                    extraction_failed_count += 1
                    failed_slides.append(i)
                    continue
                heading = (sd['title'] or '').strip()
                if not heading:
                    no_heading_count += 1
                    failed_slides.append(i)
                    continue
                # Prefer TITLE placeholder; else exact text; else substring fallback.
                title_match = next((tb for tb in sd['text_boxes']
                                    if placeholder_type(tb) in ('TITLE', 'CENTERED_TITLE')
                                    and contains_normalized(tb.get('text', ''), heading)), None)
                matched_tb = title_match
                if matched_tb is None:
                    matched_tb = next((tb for tb in sd['text_boxes'] if tb.get('text', '').strip() == heading), None)
                if matched_tb is None:
                    matched_tb = next((tb for tb in sd['text_boxes']
                                       if contains_normalized(tb.get('text', ''), heading)), None)
                if matched_tb is None:
                    no_match_count += 1
                    failed_slides.append(i)
                    continue
                style = get_text_style_from_shape(matched_tb['element'].get('shape', {}))
                if bool(style.get('bold')) and bool(style.get('italic')):
                    bold_italic_count += 1
                else:
                    failed_slides.append(i)
            success = bold_italic_count == actual_count
            step_score = calculate_percentage_score(bold_italic_count, actual_count)
            details = f"{bold_italic_count}/{actual_count} slides have bold+italic headings" + join_extras(
                f"{no_heading_count} no heading" if no_heading_count else None,
                f"{no_match_count} heading text not located in any text box" if no_match_count else None,
                f"{extraction_failed_count} slide(s) extraction failed" if extraction_failed_count else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            details = f"Error during heading style check: {e}"
            step6_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Heading Bold Italic", success, 6, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step6_category)

        # ---- Step 7 (5 pt): Information is presented in bullet points ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step7_category = StepCategory.STRUCTURAL  # bullet-count check
        try:
            bullets_count = 0
            extraction_failed_count = 0
            no_bullets_slides = []
            for i, sd in enumerate(slide_data):
                # Skip stubs — no extracted text to find bullets in.
                if sd.get('extraction_failed'):
                    extraction_failed_count += 1
                    continue
                # Accept 2+ bullets — task says "short list" without a specific count.
                has_bullets, _ = validate_bullet_points(content_slides[i], min_count=2)
                if has_bullets:
                    bullets_count += 1
                else:
                    no_bullets_slides.append(i)
            success = bullets_count == actual_count
            step_score = calculate_percentage_score(bullets_count, actual_count, max_points=5)
            details = f"{bullets_count}/{actual_count} slides present information in bullet points" + join_extras(
                f"{extraction_failed_count} slide(s) extraction failed" if extraction_failed_count else None,
                f"no bullets: slide(s) {to_deck_positions(no_bullets_slides)}" if no_bullets_slides else None,
            )
        except Exception as e:
            details = f"Error during bullet point check: {e}"
            step7_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Bullet Points", success, 7, details, score=step_score, max_score=5,
                            execution_time=time.time() - step_start, category=step7_category)

        # ---- Step 8 (5 pt): No text box overflows the slide ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step8_category = StepCategory.SPATIAL  # box geometry against slide bounds
        try:
            no_overflow_count = 0
            extraction_failed_count = 0
            overflow_slides = []
            worst = None  # (EMUs past the edge, edge name, slide index, text snippet)
            for i, sd in enumerate(slide_data):
                # Skip stubs — empty text_boxes would false-positive as "no overflow".
                if sd.get('extraction_failed'):
                    extraction_failed_count += 1
                    continue
                overflow = False
                for tb in sd['text_boxes']:
                    bbox = tb['bbox']
                    exceeded = []
                    over_right = bbox.get('x', 0) + bbox.get('width', 0) - slide_w
                    over_bottom = bbox.get('y', 0) + bbox.get('height', 0) - slide_h
                    if over_right > slide_w * 0.05:
                        exceeded.append(('right', over_right))
                    if over_bottom > slide_h * 0.05:
                        exceeded.append(('bottom', over_bottom))
                    if not exceeded:
                        continue
                    overflow = True
                    edge, over = max(exceeded, key=lambda pair: pair[1])
                    if worst is None or over > worst[0]:
                        snippet = (tb.get('text') or '').strip().replace('\n', ' ')[:40]
                        worst = (over, edge, i, snippet)
                if overflow:
                    overflow_slides.append(i)
                else:
                    no_overflow_count += 1
            success = no_overflow_count == actual_count
            step_score = calculate_percentage_score(no_overflow_count, actual_count, max_points=5)
            details = f"{no_overflow_count}/{actual_count} slides keep every text box within the slide" + join_extras(
                f"{extraction_failed_count} slide(s) extraction failed" if extraction_failed_count else None,
                f"overflows: slide(s) {to_deck_positions(overflow_slides)}" if overflow_slides else None,
                (f"worst {worst[0] / EMU_PER_INCH:.2f}in past the {worst[1]} edge on slide "
                 f"{to_deck_positions([worst[2]])[0]} ({worst[3]!r})") if worst else None,
            )
        except Exception as e:
            details = f"Error during text box overflow check: {e}"
            step8_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("No Text Overflow", success, 8, details, score=step_score, max_score=5,
                            execution_time=time.time() - step_start, category=step8_category)

        # ---- Step 9 (10 pt): Content is paraphrased, not verbatim copied ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step9_category = StepCategory.FUZZY_MATCH  # fuzzy verbatim scan is the first-line check
        try:
            paraphrased_count = 0
            partial_credit_count = 0
            score_sum = 0.0  # 1.0 per confirmed paraphrase, 0.5 when source unfetchable but bullets exist
            no_source_count = 0
            no_bullets_count = 0
            extraction_failed_count = 0
            failed_slides = []
            step9_items = []  # (category, success) per slide for StepCategory.aggregate()
            for i, sd in enumerate(slide_data):
                if sd.get('extraction_failed'):
                    extraction_failed_count += 1
                    failed_slides.append(i)
                    step9_items.append((StepCategory.EXECUTION_ERROR, False))
                    continue
                if not sd['source_content']:
                    no_source_count += 1
                    # Partial credit when bullets exist but source isn't fetchable
                    # (avoids double-penalising — steps 3/4 already cover the missing source).
                    if sd['bullets']:
                        partial_credit_count += 1
                        score_sum += 0.5
                    failed_slides.append(i)
                    step9_items.append((StepCategory.EXECUTION_ERROR, False))
                    continue
                if not sd['bullets']:
                    no_bullets_count += 1
                    failed_slides.append(i)
                    step9_items.append((StepCategory.EXECUTION_ERROR, False))
                    continue
                # Capture matched substring so the LLM second-pass sees the relevant excerpt.
                verbatim_match = None
                for b in sd['bullets']:
                    if len(b.strip()) < 3:
                        continue
                    m, _ = text_fuzzy_match_contained_long(b, sd['source_content'], threshold=90)
                    if m:
                        verbatim_match = m
                        break
                if verbatim_match is not None:
                    model = ensure_model(model_id)
                    if model is None:
                        failed_slides.append(i)
                        step9_items.append((StepCategory.EXECUTION_ERROR, False))
                        continue  # Can't verify; conservatively don't award.
                    # Center a 1000-char window on the fuzzy-flagged substring.
                    src = sd['source_content']
                    needle = verbatim_match[:50]
                    idx = find_with_flexible_whitespace(src, needle)
                    if idx < 0:
                        idx = 0
                    excerpt_start = max(0, idx - 300)
                    excerpt = src[excerpt_start:excerpt_start + 1000]
                    all_bullets = "\n".join(f"- {b}" for b in sd['bullets'] if len(b.strip()) >= 3)
                    is_copy = evaluate_with_llm(
                        f"Is the following slide content a direct verbatim copy from the source, "
                        f"or has it been paraphrased/summarized?\n\n"
                        f"Slide content:\n{all_bullets}\n\n"
                        f"Source text (excerpt):\n{excerpt}",
                        model, return_type="bool",
                        system_prompt="You are evaluating whether text has been copied verbatim. "
                        "Answer 'Yes' if it is mostly verbatim/copied, 'No' if it has been paraphrased or summarized."
                    )
                    # Only award on explicit "not a copy"; None (crash) doesn't grant the point.
                    if is_copy is False:
                        paraphrased_count += 1
                        score_sum += 1.0
                        step9_items.append((StepCategory.LLM_VLM_JUDGEMENT, True))
                    else:
                        failed_slides.append(i)
                        # None = LLM crashed; True = LLM confirmed the copy.
                        step9_items.append((StepCategory.EXECUTION_ERROR, False) if is_copy is None
                                           else (StepCategory.LLM_VLM_JUDGEMENT, False))
                elif sd.get('traceable') is True:
                    # No verbatim overlap + step 4 confirmed traceable = genuine paraphrase.
                    paraphrased_count += 1
                    score_sum += 1.0
                    step9_items.append((StepCategory.FUZZY_MATCH, True))
                else:
                    # No overlap but not traceable either — likely off-source, not paraphrased.
                    failed_slides.append(i)
                    step9_items.append((StepCategory.LLM_VLM_JUDGEMENT, False))
            success = paraphrased_count == actual_count
            step_score = calculate_percentage_score(score_sum, actual_count)
            step9_category = StepCategory.aggregate(step9_items)
            # `no_source_count` includes the partial-credit subset; subtract for clean display.
            no_source_no_bullets = no_source_count - partial_credit_count
            details = f"{paraphrased_count}/{actual_count} slides fully paraphrased" + join_extras(
                f"{partial_credit_count} partial (source unfetchable, bullets present)" if partial_credit_count else None,
                f"{no_source_no_bullets} no source + no bullets" if no_source_no_bullets else None,
                f"{no_bullets_count} no bullets (had source)" if no_bullets_count else None,
                f"{extraction_failed_count} slide(s) extraction failed" if extraction_failed_count else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            details = f"Error during paraphrase check: {e}"
            step9_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Content Paraphrased", success, 9, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step9_category)

        # ---- Step 10 (10 pt): Content placed on the left side ----
        step_start = time.time()
        success, step_score, details = False, 0, ""
        step10_category = StepCategory.SPATIAL  # box-center x-threshold check
        try:
            left_count = 0
            no_content_count = 0
            extraction_failed_count = 0
            failed_slides = []
            for i, sd in enumerate(slide_data):
                if sd.get('extraction_failed'):
                    extraction_failed_count += 1
                    failed_slides.append(i)
                    continue
                heading_norm = normalize_for_match((sd['title'] or '').strip())
                content_boxes = []
                for tb in sd['text_boxes']:
                    text = tb['text'].strip()
                    if not text:
                        continue
                    # Skip title/subtitle placeholders — inherited fonts evade the size filter.
                    if placeholder_type(tb) in ('TITLE', 'CENTERED_TITLE', 'SUBTITLE'):
                        continue
                    # Skip text that IS the heading or a "Heading:" / "Heading—" sub-heading.
                    text_norm = normalize_for_match(text)
                    if heading_norm and (text_norm == heading_norm or any(
                            text_norm.startswith(heading_norm + sep) for sep in (' ', ':', '—', '–'))):
                        continue
                    style = get_text_style_from_shape(tb['element'].get('shape', {}))
                    font_size = style.get('fontSize')
                    if font_size and not is_text_big(style, min_pt=14):
                        continue
                    content_boxes.append(tb)
                if not content_boxes:
                    no_content_count += 1
                    failed_slides.append(i)
                    continue
                # Use box CENTER — symmetric to CP4 step 2's right-side check.
                # A box without bbox info (cx is None) doesn't count as left-aligned.
                centers = [bbox_center_x(tb['bbox']) for tb in content_boxes]
                if all(cx is not None and cx < slide_w * 0.5 for cx in centers):
                    left_count += 1
                else:
                    failed_slides.append(i)
            success = left_count == actual_count
            step_score = calculate_percentage_score(left_count, actual_count)
            details = f"{left_count}/{actual_count} slides have content on the left side" + join_extras(
                f"{no_content_count} slide(s) had no body content boxes" if no_content_count else None,
                f"{extraction_failed_count} slide(s) extraction failed" if extraction_failed_count else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            details = f"Error during content-left-side check: {e}"
            step10_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Content Left Side", success, 10, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step10_category)

    except Exception as e:
        print(f"Error in checkpoint 2: {e}")
        fill_failure_steps(checkpoint, CP2_STEPS, f"Skipped due to checkpoint error: {e}", only_missing=True)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3 (10pt): Summary Slide."""
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Summary Slide")

    if _task_parse_error is not None:
        fill_failure_steps(checkpoint, CP3_STEPS, _task_parse_error)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if presentation_data is None:
        fill_failure_steps(checkpoint, CP3_STEPS, "Failed to load presentation (auth or API failure)")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if not has_min_slides(presentation_data, 3):
        fill_failure_steps(checkpoint, CP3_STEPS, "Not enough slides in presentation")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        slides = presentation_data['slides']
        summary_slide = slides[-1]
        summary_text = extract_slide_text(summary_slide)
        content_slides = slides[1:-1]

        content_headings = []
        all_content_text = []
        for slide in content_slides:
            try:
                heading = (extract_title_text(slide) or '').strip()
                if heading:
                    content_headings.append(heading)
                # 3-char threshold matches step 8 — short factual bullets still trip detection.
                bullets = [b for b in extract_bullet_point_texts(slide) if len(b.strip()) >= 3]
                if not bullets:
                    # Same fallback as CP2 — keeps step 2's corpus non-empty for
                    # plain-text-paragraph slides. See CP2 comment for rationale.
                    title_norm = heading.lower()
                    body_candidates = []
                    for _el in slide.get('pageElements', []):
                        if 'shape' not in _el or 'text' not in _el['shape']:
                            continue
                        if _el['shape'].get('placeholder', {}).get('type', '') in (
                                'TITLE', 'CENTERED_TITLE', 'SUBTITLE'):
                            continue
                        _txt = ''
                        for _tr in _el['shape']['text'].get('textElements', []):
                            _run = _tr.get('textRun')
                            if _run:
                                _txt += _run.get('content', '')
                        _txt = _txt.strip()
                        if not _txt or _txt.lower() == title_norm:
                            continue
                        body_candidates.append((len(_txt), _txt))
                    if body_candidates:
                        body_candidates.sort(reverse=True, key=lambda c: c[0])
                        bullets = [ln.strip() for ln in body_candidates[0][1].splitlines()
                                   if len(ln.strip()) >= 3]
                all_content_text.extend(bullets)
            except Exception as e:
                print(f"Error extracting content slide for summary check: {e}")

        # Steps 1 & 3 share parallel LLM calls; only run them with model + summary text.
        llm_results = {}
        model = ensure_model(model_id) if summary_text else None
        if model is not None:
            llm_tasks = []
            # Skip covers_concepts when no headings to cover (step 1 reports it instead).
            if content_headings:
                headings_list = "\n".join(f"- {h}" for h in content_headings)
                llm_tasks.append({
                    'id': 'covers_concepts',
                    'func': evaluate_with_llm,
                    'args': (
                        f"A presentation on '{EXPECTED_TOPIC}' has the following content slide topics:\n"
                        f"{headings_list}\n\n"
                        f"Does the following summary slide text briefly mention or reference ALL of the above topics?\n\n"
                        f"Summary slide text:\n{summary_text}",
                        model,
                    ),
                    'kwargs': {
                        'return_type': 'bool',
                        'system_prompt': "You are checking if a summary slide covers all topics from a presentation. "
                        "Answer 'Yes' if the summary mentions or references all listed topics, even briefly or indirectly. "
                        "Answer 'No' only if one or more topics are completely missing. Respond with ONLY 'yes' or 'no'."
                    },
                })
            llm_tasks.append({
                'id': 'engagement_prompt',
                'func': evaluate_with_llm,
                'args': (
                    f"Does the following slide text include a statement or prompt encouraging students "
                    f"to ask questions, engage, discuss, or participate?\n\n"
                    f"Slide text:\n{summary_text}",
                    model,
                ),
                'kwargs': {'return_type': 'bool'},
            })
            llm_results = parallel_execute(llm_tasks, max_workers=2, timeout=120)

        # ---- Step 1 (4 pt): Briefly mentions all concepts ----
        step_start = time.time()
        covers_all, details = False, ""
        step1_category = StepCategory.LLM_VLM_JUDGEMENT
        try:
            if not summary_text:
                details = "No summary slide content"
                step1_category = StepCategory.DETERMINISTIC  # empty-artifact reject
            elif not content_headings:
                details = "No content headings to compare against"
                # Content-slide headings missing (already penalised in CP2 step 5).
                step1_category = StepCategory.DEPENDENCY_NOT_EVALUATED
            else:
                raw = llm_results.get('covers_concepts')
                covers_all = raw is True
                if raw is None:
                    cause = "model unavailable" if model is None else "LLM call failed"
                    details = f"Summary covers {len(content_headings)} content topics: undecidable ({cause})"
                    step1_category = StepCategory.EXECUTION_ERROR
                else:
                    details = f"Summary covers all {len(content_headings)} content topics: {covers_all}"
        except Exception as e:
            details = f"Error during covers-all check: {e}"
            step1_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Covers All Concepts", covers_all, 1, details, max_score=4,
                            execution_time=time.time() - step_start, category=step1_category)

        # ---- Step 2 (3 pt): Summary text is original (not verbatim) ----
        step_start = time.time()
        is_original, details = False, ""
        step2_category = StepCategory.FUZZY_MATCH  # fuzzy verbatim scan is the first-line check
        llm_confirmed = False  # an LLM second-pass overturned at least one fuzzy flag
        try:
            if not summary_text:
                details = "No summary slide content"
                step2_category = StepCategory.DETERMINISTIC  # empty-artifact reject
            else:
                combined_content = " ".join(all_content_text)
                if not combined_content.strip():
                    # No corpus → can't verify; conservatively don't award.
                    details = "No content text available to verify originality"
                    # Content-slide text missing (already penalised in CP2 step 7).
                    step2_category = StepCategory.DEPENDENCY_NOT_EVALUATED
                else:
                    is_original = True
                    # Sentences ≥8 chars (filters common words like "Very", "Good"). Split on
                    # `:` and em/en dash too — "Topic: <copied>" patterns count as sentences.
                    summary_sentences = [s.strip() for s in re.split(r'[.?!;:\n—–]', summary_text)
                                         if len(s.strip()) >= 8]
                    # 120s wall budget caps LLM verification storm on long summaries.
                    step_deadline = time.time() + 120
                    for sentence in summary_sentences:
                        if time.time() > step_deadline:
                            details = "Step 2 wall budget (120s) exhausted; trusting fuzzy results so far"
                            step2_category = StepCategory.EXECUTION_ERROR
                            break
                        match, score = text_fuzzy_match_contained_long(sentence, combined_content, threshold=90)
                        if not match:
                            continue
                        # Fuzzy flagged verbatim — LLM second-pass to confirm. On LLM
                        # crash/unavailable, trust fuzzy (≥90% is a strong signal).
                        if model is not None:
                            # Center 2000-char window on the fuzzy match.
                            needle = match[:50]
                            idx = find_with_flexible_whitespace(combined_content, needle)
                            if idx < 0:
                                idx = 0
                            excerpt_start = max(0, idx - 600)
                            excerpt = combined_content[excerpt_start:excerpt_start + 2000]
                            is_copy, ok = call_with_timeout(
                                evaluate_with_llm, 60, "CP3 step 2 verbatim verification",
                                f"Is the following summary sentence a verbatim copy from the content text below, "
                                f"or has it been paraphrased/summarized?\n\n"
                                f"Summary sentence:\n{sentence}\n\n"
                                f"Content text (excerpt):\n{excerpt}",
                                model, return_type="bool",
                                system_prompt="Answer 'Yes' if the sentence is mostly verbatim/copied, "
                                "'No' if it has been paraphrased or summarized."
                            )
                            if not ok:
                                is_copy = None
                            if is_copy is True:
                                is_original = False
                                step2_category = StepCategory.LLM_VLM_JUDGEMENT
                                details = f"Verbatim match confirmed by LLM (fuzzy score {score}): '{sentence[:60]}...'"
                                break
                            elif is_copy is None:
                                is_original = False
                                step2_category = StepCategory.FUZZY_MATCH  # fuzzy flag decided
                                details = f"Verbatim match (score {score}, LLM call failed; trusting fuzzy flag): '{sentence[:60]}...'"
                                break
                            else:
                                llm_confirmed = True  # LLM overturned this fuzzy flag
                        else:
                            is_original = False
                            step2_category = StepCategory.FUZZY_MATCH  # fuzzy flag decided
                            details = f"Verbatim match (score {score}, model unavailable): '{sentence[:60]}...'"
                            break
                    if is_original:
                        # Last mechanism that ran on the accept path: LLM if it
                        # overturned any fuzzy flag, otherwise the fuzzy scan itself.
                        if step2_category != StepCategory.EXECUTION_ERROR and llm_confirmed:
                            step2_category = StepCategory.LLM_VLM_JUDGEMENT
                        details = "Summary text is original (no verbatim matches with content slides)"
        except Exception as e:
            is_original = False
            details = f"Error during originality check: {e}"
            step2_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Original Text", is_original, 2, details, max_score=3,
                            execution_time=time.time() - step_start, category=step2_category)

        # ---- Step 3 (3 pt): Engagement prompt ----
        step_start = time.time()
        has_engagement, details = False, ""
        step3_category = StepCategory.LLM_VLM_JUDGEMENT
        try:
            if not summary_text:
                details = "No summary slide content"
                step3_category = StepCategory.DETERMINISTIC  # empty-artifact reject
            else:
                raw = llm_results.get('engagement_prompt')
                has_engagement = raw is True
                if raw is None:
                    cause = "model unavailable" if model is None else "LLM call failed"
                    details = f"Engagement prompt: undecidable ({cause})"
                    step3_category = StepCategory.EXECUTION_ERROR
                else:
                    details = f"Engagement prompt found: {has_engagement}"
        except Exception as e:
            details = f"Error during engagement check: {e}"
            step3_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Engagement Prompt", has_engagement, 3, details, max_score=3,
                            execution_time=time.time() - step_start, category=step3_category)

    except Exception as e:
        print(f"Error in checkpoint 3: {e}")
        fill_failure_steps(checkpoint, CP3_STEPS, f"Skipped due to checkpoint error: {e}", only_missing=True)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """Checkpoint 4 (50pt, 5 steps × 10 pt each): Visual Elements."""
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=50, result=0, name="Visual Elements")

    if _task_parse_error is not None:
        fill_failure_steps(checkpoint, CP4_STEPS, _task_parse_error)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if presentation_data is None:
        fill_failure_steps(checkpoint, CP4_STEPS, "Failed to load presentation (auth or API failure)")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    if not has_min_slides(presentation_data, 3):
        fill_failure_steps(checkpoint, CP4_STEPS, "Not enough slides in presentation")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    temp_dir = os.path.join(DATA_DIR, "temp_content_images")

    try:
        slides = presentation_data['slides']
        content_slides = slides[1:-1]
        actual_count = len(content_slides)
        slide_w, slide_h = get_slide_dimensions(presentation_data)
        if slide_w is None:
            fill_failure_steps(checkpoint, CP4_STEPS, "Slide dimensions unavailable")
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        os.makedirs(temp_dir, exist_ok=True)

        # Pre-extract per-slide image data; one bad slide becomes a stub.
        slide_image_data = []
        download_failed_count = 0
        for i, slide in enumerate(content_slides):
            try:
                images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
                heading = (extract_title_text(slide) or '').strip()
                text_boxes = extract_text_boxes_from_slide(slide)
                links_with_pos = extract_slide_links_with_positions(slide)
                # objectId -> ALT URLs (step 4 fallback when no link sits below the image).
                # Positional key when objectId missing so multiple no-id images don't collide.
                alt_url_entries = list(extract_image_source_urls(slide))
                alt_url_map = {}
                for ai, a in enumerate(alt_url_entries):
                    key = a.get('objectId') or f'_idx_{ai}'
                    alt_url_map[key] = a.get('source_urls', [])
                # Last-ditch fallback when the picked image's ALT has no URL.
                all_alt_urls = [u for entry in alt_url_entries for u in entry.get('source_urls', [])]

                image_info = None
                download_failed = False
                if images:
                    # Pick largest so a tiny decorative element doesn't displace the topical photo.
                    img_meta = max(images, key=img_area)
                    url = img_meta.get('contentUrl')
                    if url:
                        img = download_slide_image(url)
                        img_path = None
                        if img:
                            # Zero-pad for stable alphabetical ordering.
                            img_path = os.path.join(temp_dir, f"slide_{i:03d}_img.png")
                            img.save(img_path)
                        else:
                            download_failed = True
                        bbox = get_element_bbox({'transform': img_meta.get('transform', {}),
                                                 'size': img_meta.get('size', {})})
                        alt_urls = alt_url_map.get(img_meta.get('objectId', ''), [])
                        image_info = {'url': url, 'path': img_path, 'bbox': bbox, 'alt_urls': alt_urls}

                if download_failed:
                    download_failed_count += 1

                slide_image_data.append({
                    'heading': heading, 'image': image_info, 'text_boxes': text_boxes,
                    'links_with_pos': links_with_pos, 'download_failed': download_failed,
                    'all_alt_urls': all_alt_urls,
                })
            except Exception as e:
                print(f"Error pre-extracting slide {i}: {e}")
                # Fresh lists per stub so mutations can't bleed across slides.
                slide_image_data.append({
                    'heading': '', 'image': None, 'text_boxes': [],
                    'links_with_pos': [], 'download_failed': False, 'all_alt_urls': [],
                })

        # ---- Step 1 (10 pt): Each content slide has a relevant image ----
        step_start = time.time()
        step1_category = StepCategory.LLM_VLM_JUDGEMENT  # VLM relevance verdicts
        try:
            model = ensure_model(model_id)
            if model is None:
                success = False
                step_score = 0
                details = "Model unavailable; cannot judge image relevance"
                step1_category = StepCategory.EXECUTION_ERROR
            else:
                no_image_count = sum(1 for sid in slide_image_data
                                     if not sid['image'] or not sid['image'].get('path'))
                # (category, success) per slide for StepCategory.aggregate();
                # missing/undownloadable images could not be judged.
                step1_items = [(StepCategory.EXECUTION_ERROR, False)] * no_image_count
                relevance_tasks = [
                    {
                        'id': f'img_rel_{i}',
                        'func': binary_judge_image,
                        'args': (
                            model, sid['image']['path'],
                            f"Is this image relevant to the topic '{sid['heading']}' "
                            f"in the context of {EXPECTED_TOPIC}?"
                        ),
                    }
                    for i, sid in enumerate(slide_image_data)
                    if sid['image'] and sid['image']['path']
                ]
                relevant_count = 0
                failed_slides = []
                if relevance_tasks:
                    rel_results = parallel_execute(relevance_tasks, max_workers=5, timeout=180)
                    # binary_judge_image returns matching path (truthy) or None (no match/crash).
                    relevant_count = sum(1 for v in rel_results.values() if v)
                    for tid, v in rel_results.items():
                        step1_items.append((StepCategory.LLM_VLM_JUDGEMENT, bool(v)))
                        if v:
                            continue
                        try:
                            failed_slides.append(int(tid.replace('img_rel_', '')))
                        except (ValueError, AttributeError):
                            failed_slides.append(tid)
                    failed_slides.sort(key=lambda x: (isinstance(x, str), x))

                success = relevant_count == actual_count
                step_score = calculate_percentage_score(relevant_count, actual_count)
                step1_category = StepCategory.aggregate(step1_items)
                details = f"{relevant_count}/{actual_count} slides have relevant images" + join_extras(
                    f"{download_failed_count} image(s) undownloadable" if download_failed_count else None,
                    f"{no_image_count} slide(s) had no image" if no_image_count else None,
                    f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
                )
        except Exception as e:
            success = False
            step_score = 0
            details = f"Error during image relevance check: {e}"
            step1_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Image Relevant", success, 1, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step1_category)

        # ---- Step 2 (10 pt): Image placed on right side ----
        # Image-center vs slide-center: wide right-leaning passes, tiny edge slivers fail.
        step_start = time.time()
        step2_category = StepCategory.SPATIAL  # image-center x-threshold check
        try:
            no_image_count = sum(1 for sid in slide_image_data
                                 if not sid['image'] or not sid['image'].get('bbox'))
            # Image without a bbox (cx is None) doesn't count as right-aligned.
            right_count = 0
            failed_slides = []
            for i, sid in enumerate(slide_image_data):
                if not sid['image'] or not sid['image']['bbox']:
                    failed_slides.append(i)
                    continue
                cx = bbox_center_x(sid['image']['bbox'])
                if cx is not None and cx > slide_w / 2:
                    right_count += 1
                else:
                    failed_slides.append(i)
            success = right_count == actual_count
            step_score = calculate_percentage_score(right_count, actual_count)
            details = f"{right_count}/{actual_count} slides have images on the right side" + join_extras(
                f"{no_image_count} slide(s) had no image" if no_image_count else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            success = False
            step_score = 0
            details = f"Error during image-right-side check: {e}"
            step2_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Image Right Side", success, 2, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step2_category)

        # ---- Step 3 (10 pt): Image source credit in small font (lower-right) ----
        step_start = time.time()
        step3_category = StepCategory.SPATIAL  # below-image position + small-font gate
        try:
            credit_count = 0
            no_image_count = 0
            failed_slides = []
            for i, sid in enumerate(slide_image_data):
                if not sid['image'] or not sid['image']['bbox']:
                    no_image_count += 1
                    failed_slides.append(i)
                    continue
                bbox = sid['image']['bbox']
                img_bottom = bbox.get('y', 0) + bbox.get('height', 0)
                img_left = bbox.get('x', 0)
                img_right = img_left + bbox.get('width', 0)
                # Credit must sit below the image (5% slop) AND its center-x must fall
                # within the image's x-range — rejects wide CP2 S3 source-line false matches.
                found = False
                for tb in sid['text_boxes']:
                    text = (tb.get('text') or '').lower().strip()
                    if not text:
                        continue
                    style = get_text_style_from_shape(tb['element'].get('shape', {}))
                    tb_bbox = tb.get('bbox', {}) or {}
                    # Small-font gate: explicit fontSize <18pt OR (inherited & bbox <10% slide_h).
                    font_size = style.get('fontSize')
                    if font_size:
                        if is_text_big(style, min_pt=18):
                            continue
                    elif tb_bbox.get('height', 0) > slide_h * 0.10:
                        continue
                    if tb_bbox.get('y', 0) < img_bottom - (slide_h * 0.05):
                        continue
                    tb_cx = bbox_center_x(tb_bbox)
                    if tb_cx is None or tb_cx < img_left or tb_cx > img_right:
                        continue
                    if any(kw in text for kw in CREDIT_KEYWORDS):
                        found = True
                        break
                if found:
                    credit_count += 1
                else:
                    failed_slides.append(i)
            success = credit_count == actual_count
            step_score = calculate_percentage_score(credit_count, actual_count)
            details = f"{credit_count}/{actual_count} slides have image source credit beneath image" + join_extras(
                f"{no_image_count} slide(s) had no image" if no_image_count else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            success = False
            step_score = 0
            details = f"Error during image credit check: {e}"
            step3_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Image Source Credit", success, 3, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step3_category)

        # ---- Step 4 (10 pt): Source URL accessible and points to the image ----
        step_start = time.time()
        # Tiered exact -> pHash -> VLM matcher; per-item categories come from the
        # recorded match method, aggregated via StepCategory.aggregate().
        step4_category = StepCategory.LLM_VLM_JUDGEMENT
        try:
            model = ensure_model(model_id)
            if model is None:
                success = False
                step_score = 0
                details = "Model unavailable; cannot verify image source URL"
                step4_category = StepCategory.EXECUTION_ERROR
            else:
                img_match_tasks = []
                step4_items = []  # (category, success) per slide
                no_url_count = 0
                for i, sid in enumerate(slide_image_data):
                    if not sid['image'] or not sid['image']['path']:
                        # No downloadable slide image to compare against.
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        continue
                    # Prefer a positioned link below the image (need non-degenerate bbox —
                    # zero-area bbox would let any link match as "below").
                    bbox = sid['image'].get('bbox') or {}
                    has_valid_bbox = bbox.get('width', 0) > 0 and bbox.get('height', 0) > 0
                    image_url_below = (find_url_below_image(bbox, sid['links_with_pos'])
                                       if has_valid_bbox else None)
                    if image_url_below:
                        candidates = [image_url_below]
                    else:
                        # Fall back: this image's ALT URLs, else any sibling image's ALT URLs.
                        candidates = list(sid['image'].get('alt_urls') or [])
                        if not candidates:
                            candidates = list(sid.get('all_alt_urls') or [])
                    if not candidates:
                        no_url_count += 1
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        continue
                    img_match_tasks.append({
                        'id': f'img_match_{i}',
                        'func': match_source_image,
                        'args': (candidates, sid['image']['path'], temp_dir, i, model),
                    })
                img_match_results = parallel_execute(img_match_tasks, max_workers=5, timeout=240) if img_match_tasks else {}
                match_count = sum(1 for v in img_match_results.values() if v and v[0])
                crashed = sum(1 for v in img_match_results.values() if v is None)
                # Track failing slide indices for debugging (which URL is the culprit?).
                failed_slides = []
                for tid, v in img_match_results.items():
                    if v is None:
                        step4_items.append((StepCategory.EXECUTION_ERROR, False))  # match call crashed
                    else:
                        # match_source_image details record the deciding tier.
                        method_m = re.search(r"\((exact|perceptual_hash)\)", v[1] or "")
                        if v[0]:
                            step4_items.append((
                                StepCategory.from_match_method(method_m.group(1)) if method_m
                                else StepCategory.LLM_VLM_JUDGEMENT, True))
                        elif "No image or links available" in (v[1] or ""):
                            step4_items.append((StepCategory.EXECUTION_ERROR, False))
                        else:
                            # VLM is the tiered matcher's final arbiter on rejects.
                            step4_items.append((StepCategory.LLM_VLM_JUDGEMENT, False))
                    if v is None or v[0]:
                        continue
                    # Defensive: future task_id schema change shouldn't crash the step.
                    try:
                        failed_slides.append(int(tid.replace('img_match_', '')))
                    except (ValueError, AttributeError):
                        failed_slides.append(tid)
                failed_slides.sort(key=lambda x: (isinstance(x, str), x))

                success = match_count == actual_count
                step_score = calculate_percentage_score(match_count, actual_count)
                step4_category = StepCategory.aggregate(step4_items)
                # download_failed_count is reported in step 1; only step-specific extras here.
                details = f"{match_count}/{actual_count} slides have source URL matching the slide image" + join_extras(
                    f"{no_url_count} no URL below image or in ALT text" if no_url_count else None,
                    f"{crashed} match call(s) crashed" if crashed else None,
                    f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
                )
        except Exception as e:
            success = False
            step_score = 0
            details = f"Error during image source URL match: {e}"
            step4_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Image Source URL Matches", success, 4, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step4_category)

        # ---- Step 5 (10 pt): Images are unique across content slides ----
        # Cluster images by perceptual similarity; score by clusters/actual_count so a
        # missing image is already penalized (it can't contribute a unique cluster).
        step_start = time.time()
        step5_category = StepCategory.FUZZY_MATCH  # pHash-similarity clustering
        try:
            present_idx_paths = [(i, sid['image']['path']) for i, sid in enumerate(slide_image_data)
                                 if sid['image'] and sid['image'].get('path')]
            present_indices = [i for i, _ in present_idx_paths]
            present_paths = [p for _, p in present_idx_paths]
            no_image_count = actual_count - len(present_paths)
            cluster_ids = cluster_images_by_phash(present_paths)
            unique_clusters = len(set(cluster_ids))
            duplicates = len(present_paths) - unique_clusters
            # Failed slides: in a non-singleton cluster, or had no image at all.
            failed_slides = [i for i, sid in enumerate(slide_image_data)
                             if not sid['image'] or not sid['image'].get('path')]
            cluster_size = {}
            for cid in cluster_ids:
                cluster_size[cid] = cluster_size.get(cid, 0) + 1
            for slide_i, cid in zip(present_indices, cluster_ids):
                if cluster_size[cid] > 1:
                    failed_slides.append(slide_i)
            failed_slides = sorted(set(failed_slides))
            success = unique_clusters == actual_count
            step_score = calculate_percentage_score(unique_clusters, actual_count)
            details = f"{unique_clusters}/{actual_count} unique image(s) across content slides" + join_extras(
                f"{no_image_count} slide(s) had no image" if no_image_count else None,
                f"{duplicates} duplicate image(s) detected" if duplicates else None,
                f"failed: slide(s) {to_deck_positions(failed_slides)}" if failed_slides else None,
            )
        except Exception as e:
            success = False
            step_score = 0
            details = f"Error during image uniqueness check: {e}"
            step5_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Images Unique", success, 5, details, score=step_score, max_score=10,
                            execution_time=time.time() - step_start, category=step5_category)

    except Exception as e:
        print(f"Error in checkpoint 4: {e}")
        fill_failure_steps(checkpoint, CP4_STEPS, f"Skipped due to checkpoint error: {e}", only_missing=True)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
    """Main entry — always returns a complete Result, even if individual CPs crash."""
    total_start = time.time()

    # Reset cached model so transient failures don't poison batched harnesses.
    reset_model_state()

    if cached_models and model_id in cached_models:
        set_cached_model(cached_models[model_id])

    # Setup must not raise out of grade_checkpoints; per-CP guards report cleanly.
    try:
        setup_presentation(workspace_doc_id)
    except Exception as e:
        print(f"Failed to set up presentation: {e}")

    # If a CP factory crashes, emit a complete fallback Checkpoint.
    cp_specs = [
        (grade_checkpoint_1, "Title Slide", 20, CP1_STEPS),
        (grade_checkpoint_2, "Content Slides", 85, CP2_STEPS),
        (grade_checkpoint_3, "Summary Slide", 10, CP3_STEPS),
        (grade_checkpoint_4, "Visual Elements", 50, CP4_STEPS),
    ]

    checkpoints = []
    for factory, cp_name, total, steps in cp_specs:
        try:
            checkpoints.append(factory())
        except Exception as e:
            print(f"Catastrophic failure in {cp_name}: {e}")
            cp = Checkpoint(total=total, result=0, name=cp_name)
            fill_failure_steps(cp, steps, f"Checkpoint crashed: {e}")
            cp.execution_time = 0
            checkpoints.append(cp)

    return Result(checkpoints, total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate slides_26 basic educational slide deck")
    parser.add_argument("--workspace_doc_id", required=True, help="Google Slides presentation ID")
    parser.add_argument("--browsing_history", nargs="*", default=[], help="List of visited URLs")
    args = parser.parse_args()

    result = grade_checkpoints(args.workspace_doc_id, browsing_history=args.browsing_history)

    print(f"\nFinal Score: {result.final_score}")
    report = result.get_detailed_report()
    for cp in report["checkpoints"]:
        print(f"\n{cp['name']}: {cp['score']} ({cp['execution_time']:.2f}s)")
        for step in cp["steps"]:
            status = "[PASS]" if step["success"] else "[FAIL]"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'} ({step['execution_time']:.2f}s)")
