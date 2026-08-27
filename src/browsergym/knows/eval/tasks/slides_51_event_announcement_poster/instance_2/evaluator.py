import os
import re
import json
import shutil
import sys
import tempfile
import time
import argparse
from typing import List, Dict, Any


# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()


BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, StepCategory, calculate_percentage_score
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_text,
    extract_text_boxes_from_slide,
    get_text_style_from_shape,
    get_paragraph_alignment,
    get_slide_dimensions,
    get_element_bbox,
    download_slide_image,
    is_text_big,
    is_text_color,
    is_text_centered,
    is_text_left_aligned,
    extract_speaker_notes_text,
    get_shape_background_fill,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keyword_exact_match, keywords_exact_match
from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute
from src.browsergym.knows.eval.eval_utils.utils import is_bbox_mostly_inside, bbox_overlap_ratio
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_text_content
from src.browsergym.knows.eval.tasks.slides_51_event_announcement_poster.utils import (
    find_header_box,
    find_subheader_box,
    find_body_box,
    select_summary_body_box,
    classify_citation_group,
    is_color_close,
    COLORS,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/slides_51_event_announcement_poster/instance_2/")
LOGOS_DIR = os.path.join(TASK_DIR, "data/gold_images/logos/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Instance-specific event facts
EVENT_NAME = "Becoming a National Geographic Photographer"
EVENT_DATE = "September 15"
LOCATION_KEYWORDS_ALL = ["1030 North Olive", "Tucson"]
HOST_KEYWORDS_ALL = ["Center for Creative Photography", "University of Arizona"]
ORG_NAME = "Center for Creative Photography at the University of Arizona"
SPEAKER_NAME = "Michael Yamashita"
TOPIC = "Photography"
HEADER_COLOR = COLORS["mint_green"]
SIDEBAR_COLOR = COLORS["light_orange"]
ACHIEVEMENT_COUNT = 5
CONTACT_KEYWORDS = ["arizona.edu", "ccp.arizona.edu", "(520)", "tucson",
                    "center for creative photography", "university of arizona"]
WEBPAGE_KEYWORDS = ["michaelyamashita.com", "yamashitaphoto.com", "yamashita"]
SPEAKER_FACTS = (
    "Michael Yamashita is an American photojournalist who has worked for "
    "National Geographic for over 40 years. He is known for his photographs "
    "of Asia (Japan, China, Vietnam) and his book-length projects on Marco "
    "Polo and the Great Wall of China. He studied Asian Studies at Wesleyan "
    "University before becoming a photographer."
)

_, SLIDES_SERVICE = initialize_google_services(service_type="slides")

# Model for VLM evaluation (used in later checkpoints)
model = None
model_id = "gemini-2.5-flash-google-ai"

presentation_id = None
presentation_data = None

# Cached across checkpoints: populated in CP3 (body_text) and CP4 (sidebar_text),
# reused in CP6 to avoid re-finding text boxes.
body_text = ""
body_box_selected = None
sidebar_text = ""


def setup_presentation(workspace_doc_id):
    """
    Setup presentation processing.

    Args:
        workspace_doc_id (str): Google Slides presentation ID to evaluate.
    """
    global presentation_id, presentation_data

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace presentation ID: {workspace_doc_id}")
    presentation_id = workspace_doc_id

    presentation_data = SLIDES_SERVICE.presentations().get(
        presentationId=presentation_id
    ).execute()


def grade_checkpoint_1():
    """
    Checkpoint 1 (65 pt): Header & Subheader

    Steps:
        1.  Event name at top of page (5 pt)
        2.  Event name centered on slide (5 pt)
        3.  Event name has bold font (5 pt)
        4.  Event name font size >= 20 (5 pt)
        5.  Event name color is Mint Green (5 pt)
        6.  Event name reads expected text (5 pt)
        7.  Subheader below main header (5 pt)
        8.  Subheader is left-aligned (5 pt)
        9.  Subheader is italic, smaller than header (5 pt)
        10. Subheader contains date (5 pt)
        11. Subheader contains location (5 pt)
        12. Subheader contains host (5 pt)
        13. Visual contrast between header and subheader (5 pt)
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=65, result=0, name="Header & Subheader")

    slides = presentation_data.get('slides', [])
    if not slides:
        checkpoint.add_step(
            "Presentation has slides",
            False,
            1,
            "No slides found in presentation",
            max_score = 5,
            category = StepCategory.EXECUTION_ERROR
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    slide_width_emu, slide_height_emu = get_slide_dimensions(presentation_data)
    if slide_width_emu is None or slide_height_emu is None:
        for step_id, name in enumerate([
            "Event Name at Top", "Event Name Centered", "Event Name Bold",
            "Event Name Font Size >= 20", "Event Name Mint Green Color", "Event Name Text Content",
            "Subheader Below Header", "Subheader Left-Aligned", "Subheader Italic & Smaller Font",
            "Subheader Contains Date", "Subheader Contains Location", "Subheader Contains Host",
            "Visual Contrast"], start=1):
            checkpoint.add_step(name, False, step_id, "Slide dimensions unavailable", max_score=5,
                                category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    text_boxes = extract_text_boxes_from_slide(slide)

    # --- Find header and subheader by position/styling (content-independent) ---
    header_box = find_header_box(text_boxes, slide_height_emu, EVENT_NAME)
    subheader_box = find_subheader_box(text_boxes, header_box)

    # Cache styles and common values
    header_style = get_text_style_from_shape(header_box['element']['shape']) if header_box else None
    subheader_style = get_text_style_from_shape(subheader_box['element']['shape']) if subheader_box else None

    header_found = header_box is not None
    sub_found = subheader_box is not None
    bbox = header_box['bbox'] if header_found else None

    header_font_size = 0
    if header_style and header_style.get('fontSize'):
        header_font_size = header_style['fontSize'].get('magnitude', 0)

    sub_font_size = 0
    if subheader_style and subheader_style.get('fontSize'):
        sub_font_size = subheader_style['fontSize'].get('magnitude', 0)

    # --- Step 1: Event name at top of page ---
    at_top = bbox['y'] < slide_height_emu * 0.3 if header_found else False
    checkpoint.add_step(
        "Event Name at Top",
        at_top,
        1,
        f"y={bbox['y']:.0f}, threshold={slide_height_emu * 0.3:.0f}" if header_found else "Header not found",
        max_score = 5,
        category = StepCategory.SPATIAL if header_found else StepCategory.DEPENDENCY_NOT_EVALUATED
    )

    # --- Step 2: Event name centered on slide ---
    if header_found:
        is_centered = is_text_centered(header_box['element']['shape'], bbox, slide_width_emu)
        alignment = get_paragraph_alignment(header_box['element']['shape'])
        center_detail = (
            f"alignment={alignment or 'not set'}, "
            f"box_center_x={bbox['x'] + bbox['width'] / 2:.0f}, slide_center={slide_width_emu / 2:.0f}"
        )
        # Explicit CENTER alignment decides deterministically; otherwise the
        # bbox-position check made the call.
        center_category = (StepCategory.DETERMINISTIC if alignment == 'CENTER'
                           else StepCategory.SPATIAL)
    else:
        is_centered = False
        center_detail = "Header not found"
        center_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    checkpoint.add_step(
        "Event Name Centered",
        is_centered,
        2,
        center_detail,
        max_score = 5,
        category = center_category
    )

    # --- Step 3: Event name has bold font ---
    is_bold = header_style.get('bold', False) if header_style else False
    checkpoint.add_step(
        "Event Name Bold",
        is_bold,
        3,
        f"Bold: {'yes' if is_bold else 'no'}" if header_found else "Header not found",
        max_score = 5,
        category = StepCategory.DETERMINISTIC if header_found else StepCategory.DEPENDENCY_NOT_EVALUATED
    )

    # --- Step 4: Event name font size >= 20 ---
    is_large = is_text_big(header_style, min_pt=20) if header_style else False
    checkpoint.add_step(
        "Event Name Font Size >= 20",
        is_large,
        4,
        f"Font size: {header_font_size}pt" if header_found else "Header not found",
        max_score = 5,
        category = StepCategory.DETERMINISTIC if header_found else StepCategory.DEPENDENCY_NOT_EVALUATED
    )

    # --- Step 5: Event name color is Mint Green ---
    fg_color = header_style.get('foregroundColor') if header_style else None
    if not header_found:
        is_target_color = False
        color_detail = "Header not found"
        color_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    elif fg_color is None:
        is_target_color = False
        color_detail = "No explicit color set (default)"
        # Rejected because no color value exists — no tolerance comparison ran
        color_category = StepCategory.DETERMINISTIC
    else:
        is_target_color = is_text_color(header_style, *HEADER_COLOR, tolerance=0.30)
        color_detail = (f"r={fg_color.get('red', 0):.2f} "
                        f"g={fg_color.get('green', 0):.2f} "
                        f"b={fg_color.get('blue', 0):.2f}")
        color_category = StepCategory.FUZZY_MATCH
    checkpoint.add_step(
        "Event Name Mint Green Color",
        is_target_color,
        5,
        color_detail,
        max_score = 5,
        category = color_category
    )

    # --- Step 6: Event name reads expected text ---
    if header_found:
        name_match = keyword_exact_match(header_box['text'], EVENT_NAME, case_sensitive=False)
        if not name_match:
            name_match = keyword_exact_match(header_box['text'], EVENT_NAME, case_sensitive=False, substring=True)
        name_detail = f"Header text: '{header_box['text'].strip()[:80]}'"
    else:
        full_text = extract_slide_text(slide)
        name_match = keyword_exact_match(full_text, EVENT_NAME, case_sensitive=False, substring=True)
        name_detail = "Header not found, checked full slide text" + (" - found" if name_match else " - not found")
    checkpoint.add_step(
        "Event Name Text Content",
        name_match,
        6,
        name_detail,
        max_score = 5,
        category = StepCategory.DETERMINISTIC
    )

    # --- Step 7: Subheader below main header ---
    if sub_found and header_found:
        sub_center_y = subheader_box['bbox']['y'] + subheader_box['bbox']['height'] / 2
        header_center_y = bbox['y'] + bbox['height'] / 2
        is_below = sub_center_y > header_center_y
        below_detail = f"sub_center={sub_center_y:.0f}, header_center={header_center_y:.0f}"
    else:
        is_below = False
        below_detail = "Header or subheader not found"
    checkpoint.add_step(
        "Subheader Below Header",
        is_below,
        7,
        below_detail,
        max_score = 5,
        category = (StepCategory.STRUCTURAL if (sub_found and header_found)
                    else StepCategory.DEPENDENCY_NOT_EVALUATED)
    )

    # --- Step 8: Subheader is left-aligned ---
    if sub_found:
        is_left = is_text_left_aligned(
            subheader_box['element']['shape'], subheader_box['bbox'], slide_width_emu
        )
        alignment = get_paragraph_alignment(subheader_box['element']['shape'])
        left_detail = (
            f"alignment={alignment or 'not set'}, "
            f"x={subheader_box['bbox']['x']:.0f}, threshold={slide_width_emu * 0.25:.0f}"
        )
        # Explicit paragraph alignment decides deterministically; otherwise the
        # bbox-position fallback made the call.
        left_category = (StepCategory.DETERMINISTIC
                         if alignment in ('LEFT', 'START', 'CENTER', 'RIGHT', 'END', 'JUSTIFIED')
                         else StepCategory.SPATIAL)
    else:
        is_left = False
        left_detail = "Subheader not found"
        left_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    checkpoint.add_step(
        "Subheader Left-Aligned",
        is_left,
        8,
        left_detail,
        max_score = 5,
        category = left_category
    )

    # --- Step 9: Subheader is italic, smaller than header ---
    is_italic = subheader_style.get('italic', False) if subheader_style else False
    is_smaller = sub_font_size < header_font_size if (header_font_size > 0 and sub_font_size > 0) else False
    italic_smaller = is_italic and is_smaller
    checkpoint.add_step(
        "Subheader Italic & Smaller Font",
        italic_smaller,
        9,
        f"Italic: {'yes' if is_italic else 'no'}, font: {sub_font_size}pt < header {header_font_size}pt" if sub_found else "Subheader not found",
        max_score = 5,
        category = StepCategory.DETERMINISTIC if sub_found else StepCategory.DEPENDENCY_NOT_EVALUATED
    )

    # --- Steps 10-12: Subheader content checks ---
    search_text = subheader_box['text'] if sub_found else extract_slide_text(slide)

    # Step 10: Date — accept "September 15", "Sept 15", "9/15", etc.
    date_keywords = ["September 15", "Sept 15", "Sep 15", "9/15", "09/15"]
    matched_date = keywords_exact_match(search_text, date_keywords, case_sensitive=False, substring=True)
    has_date = matched_date is not None
    checkpoint.add_step(
        "Subheader Contains Date",
        has_date,
        10,
        f"Date: {matched_date if matched_date else 'missing'}",
        max_score = 5,
        category = StepCategory.DETERMINISTIC
    )

    # Step 11: Location
    has_location = all(
        keyword_exact_match(search_text, kw, case_sensitive=False, substring=True)
        for kw in LOCATION_KEYWORDS_ALL
    )
    checkpoint.add_step(
        "Subheader Contains Location",
        has_location,
        11,
        f"Location keywords ({', '.join(LOCATION_KEYWORDS_ALL)}): {'found' if has_location else 'missing'}",
        max_score = 5,
        category = StepCategory.DETERMINISTIC
    )

    # Step 12: Host
    has_host = all(
        keyword_exact_match(search_text, kw, case_sensitive=False, substring=True)
        for kw in HOST_KEYWORDS_ALL
    )
    checkpoint.add_step(
        "Subheader Contains Host",
        has_host,
        12,
        f"Host keywords ({', '.join(HOST_KEYWORDS_ALL)}): {'found' if has_host else 'missing'}",
        max_score = 5,
        category = StepCategory.DETERMINISTIC
    )

    # --- Step 13: Visual contrast between header and subheader ---
    if header_style and subheader_style:
        size_diff = header_font_size > sub_font_size and (header_font_size - sub_font_size) >= 4
        h_bold = header_style.get('bold', False)
        s_italic = subheader_style.get('italic', False)
        style_diff = h_bold and s_italic
        contrast = size_diff or style_diff
        contrast_detail = f"Size: {header_font_size}pt vs {sub_font_size}pt | Bold vs italic: {h_bold} vs {s_italic}"
    else:
        contrast = False
        contrast_detail = "Header or subheader style not found"
    checkpoint.add_step(
        "Visual Contrast",
        contrast,
        13,
        contrast_detail,
        max_score = 5,
        category = (StepCategory.DETERMINISTIC if (header_style and subheader_style)
                    else StepCategory.DEPENDENCY_NOT_EVALUATED)
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2 (20 pt): Logo

    Steps:
        1. Logo is the Center for Creative Photography at U of Arizona logo (5 pt)
        2. Logo is inserted in the top right of the slide (5 pt)
        3. Logo appropriately scaled (5 pt)
        4. Logo does not overlap or crowd header/subheader text (5 pt)
    """
    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Logo")

    slides = presentation_data.get('slides', [])
    if not slides:
        for i, name in enumerate(["Org Logo", "Logo Top Right", "Logo Scale", "Logo No Overlap"], 1):
            checkpoint.add_step(
                name,
                False,
                i,
                "No slides found",
                max_score = 5,
                category = StepCategory.EXECUTION_ERROR
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    slide_w, slide_h = get_slide_dimensions(presentation_data)
    if slide_w is None or slide_h is None:
        for step_id, name in enumerate(["Org Logo", "Logo Top Right", "Logo Scale", "Logo No Overlap with Header/Subheader"], start=1):
            checkpoint.add_step(name, False, step_id, "Slide dimensions unavailable", max_score=5,
                                category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # --- Find all image elements with bboxes (position-based, content-independent) ---
    image_elements = []
    for elem in slide.get('pageElements', []):
        if 'image' in elem:
            bbox = get_element_bbox(elem)
            image_elements.append({
                'element': elem,
                'bbox': bbox,
                'contentUrl': elem['image'].get('contentUrl', ''),
            })

    if not image_elements:
        for i, name in enumerate(["Org Logo", "Logo Top Right", "Logo Scale", "Logo No Overlap"], 1):
            checkpoint.add_step(
                name,
                False,
                i,
                "No images found on slide",
                max_score = 5,
                category = StepCategory.DETERMINISTIC
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find logo candidate: image closest to top-right corner of slide
    logo = None
    best_dist = float('inf')
    for img in image_elements:
        b = img['bbox']
        cx = b['x'] + b['width'] / 2
        cy = b['y'] + b['height'] / 2
        dist = ((cx - slide_w) ** 2 + cy ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            logo = img

    logo_bbox = logo['bbox']

    # --- Step 1: The logo matches the expected org (VLM with examples) ---
    step_start = time.time()
    step1_success = False
    step1_details = []
    step1_category = StepCategory.LLM_VLM_JUDGEMENT

    temp_dir = tempfile.mkdtemp()
    try:
        global model
        if model is None:
            model = load_model(model_id)

        pil_img = download_slide_image(logo['contentUrl'])
        if pil_img:
            img_path = os.path.join(temp_dir, "logo.png")
            pil_img.save(img_path)

            result = binary_judge_image(
                model,
                img_path,
                f"Is this the {ORG_NAME} logo?",
                LOGOS_DIR
            )
            step1_success = result is not None
            step1_details.append(f"VLM match against example logos: {'yes' if step1_success else 'no'}")
        else:
            step1_details.append("Failed to download logo image")
            step1_category = StepCategory.EXECUTION_ERROR
    except Exception as e:
        step1_details.append(f"Error during logo check: {e}")
        step1_category = StepCategory.EXECUTION_ERROR
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    checkpoint.add_step(
        "Org Logo",
        step1_success,
        1,
        " | ".join(step1_details),
        max_score = 5,
        execution_time = time.time() - step_start,
        category = step1_category
    )

    # --- Step 2: Logo is inserted in the top right of the slide ---
    step_start = time.time()
    top_right_zone = {
        'x': slide_w * 0.6,
        'y': 0,
        'width': slide_w * 0.4,
        'height': slide_h * 0.4,
    }
    step2_success = is_bbox_mostly_inside(logo_bbox, top_right_zone, threshold=0.6)
    step2_detail = (f"logo bbox ({logo_bbox['x']:.0f},{logo_bbox['y']:.0f}) "
                    f"vs top-right zone ({top_right_zone['x']:.0f},{top_right_zone['y']:.0f})")
    checkpoint.add_step(
        "Logo Top Right",
        step2_success,
        2,
        step2_detail,
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.SPATIAL
    )

    # --- Step 3: Logo appropriately scaled ---
    step_start = time.time()
    logo_area = logo_bbox['width'] * logo_bbox['height']
    slide_area = slide_w * slide_h
    area_pct = (logo_area / slide_area) * 100 if slide_area > 0 else 0
    step3_success = 1.0 <= area_pct <= 20.0
    checkpoint.add_step(
        "Logo Scale",
        step3_success,
        3,
        f"Scale: {area_pct:.1f}% of slide ({'appropriate' if step3_success else 'inappropriate'})",
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.SPATIAL
    )

    # --- Step 4: Logo does not overlap or crowd header/subheader text ---
    step_start = time.time()
    step4_details = []

    # Re-find header and subheader by position/styling
    text_boxes = extract_text_boxes_from_slide(slide)
    header_box = find_header_box(text_boxes, slide_h, EVENT_NAME)
    subheader_box = find_subheader_box(text_boxes, header_box)

    # Check overlap: allow minor bounding box intersections (< 15% of smaller box area)
    OVERLAP_THRESHOLD = 0.15

    overlaps_header = False
    header_overlap_pct = 0.0
    if header_box:
        header_overlap_pct = max(
            bbox_overlap_ratio(logo_bbox, header_box['bbox']),
            bbox_overlap_ratio(header_box['bbox'], logo_bbox),
        )
        overlaps_header = header_overlap_pct > OVERLAP_THRESHOLD
    step4_details.append(f"Overlaps header: {'yes' if overlaps_header else 'no'} ({header_overlap_pct:.0%})")

    overlaps_subheader = False
    sub_overlap_pct = 0.0
    if subheader_box:
        sub_overlap_pct = max(
            bbox_overlap_ratio(logo_bbox, subheader_box['bbox']),
            bbox_overlap_ratio(subheader_box['bbox'], logo_bbox),
        )
        overlaps_subheader = sub_overlap_pct > OVERLAP_THRESHOLD
    step4_details.append(f"Overlaps subheader: {'yes' if overlaps_subheader else 'no'} ({sub_overlap_pct:.0%})")

    step4_success = not overlaps_header and not overlaps_subheader
    checkpoint.add_step(
        "Logo No Overlap with Header/Subheader",
        step4_success,
        4,
        " | ".join(step4_details),
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.SPATIAL
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (30 pt): Central Body — Topic Background

    Steps:
        1. Background summary placed in central body (5 pt)
        2. Summary is 1–2 paragraphs (5 pt)
        3. Summary covers Photography (5 pt)
        4. Summary tied to Michael Yamashita's work (5 pt)
        5. Summary is engaging (5 pt)
        6. At least 2 reputable, stable sources are cited (5 pt)
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Central Body — Topic Background")

    slides = presentation_data.get('slides', [])
    if not slides:
        for i, name in enumerate(["Body Placement", "1-2 Paragraphs", f"Covers {TOPIC}",
                                   f"Tied to {SPEAKER_NAME}", "Engaging Content", "Reputable Sources"], 1):
            checkpoint.add_step(
                name,
                False,
                i,
                "No slides found",
                max_score = 5,
                category = StepCategory.EXECUTION_ERROR
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    slide_w, slide_h = get_slide_dimensions(presentation_data)
    if slide_w is None or slide_h is None:
        for step_id, name in enumerate([
            "Body Summary in Central Area", "1-2 Paragraphs",
            f"Covers {TOPIC}", f"Tied to {SPEAKER_NAME}'s Work",
            "Engaging Content", "At Least 2 Relevant Cited Sources"], start=1):
            checkpoint.add_step(name, False, step_id, "Slide dimensions unavailable", max_score=5,
                                category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    text_boxes = extract_text_boxes_from_slide(slide)

    # --- Find central body text box ---
    global body_text, body_box_selected, model
    if model is None:
        model = load_model(model_id)

    step_start = time.time()
    body_box, body_is_summary, step1_detail = select_summary_body_box(
        text_boxes, slide_w, slide_h, model)
    body_text = body_box['text'].strip() if body_box else ""
    body_box_selected = body_box

    # --- Step 1: Background summary placed in central body ---
    if body_box is None:
        step1_category = StepCategory.SPATIAL
    elif body_is_summary is None:
        step1_category = StepCategory.EXECUTION_ERROR
    else:
        step1_category = StepCategory.LLM_VLM_JUDGEMENT

    checkpoint.add_step(
        "Body Summary in Central Area",
        bool(body_is_summary),
        1,
        step1_detail,
        max_score = 5,
        execution_time = time.time() - step_start,
        category = step1_category
    )

    # --- Step 2: Summary is 1-2 paragraphs ---
    step_start = time.time()
    if not body_text:
        step2_success = False
        step2_detail = "No body text"
    else:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body_text) if p.strip()]
        if len(paragraphs) <= 1:
            single_line_paras = [p.strip() for p in body_text.split('\n') if p.strip()]
            para_count = len(single_line_paras) if len(single_line_paras) > 1 else 1
        else:
            para_count = len(paragraphs)
        step2_success = 1 <= para_count <= 2
        step2_detail = f"Paragraph count: {para_count}"

    checkpoint.add_step(
        "1-2 Paragraphs",
        step2_success,
        2,
        step2_detail,
        max_score = 5,
        execution_time = time.time() - step_start,
        category = (StepCategory.STRUCTURAL if body_text
                    else StepCategory.DEPENDENCY_NOT_EVALUATED)
    )

    # --- Steps 3, 4, 5: LLM checks in parallel ---
    step3_success = False
    step3_details = []
    step4_success = False
    step4_details = []
    step5_success = False
    step5_details = []
    step3_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    step4_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    step5_category = StepCategory.DEPENDENCY_NOT_EVALUATED

    if not body_text:
        step3_details.append("No body text")
        step4_details.append("No body text")
        step5_details.append("No body text")
    else:

        msg_topic = [
            {"role": "system", "content": [{"type": "text", "text":
                "You evaluate event poster content. Answer only Yes or No, then give a one-sentence reason."}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"Does the following text discuss {TOPIC} (the art and craft of taking and "
                f"creating photographs, including techniques, subjects, history, or notable "
                f"photographers)? It does not need to be a standalone tutorial — a poster-style "
                f"description that references these topics counts as Yes.\n\n"
                f"Text:\n{body_text[:2000]}"}]}
        ]

        msg_speaker = [
            {"role": "system", "content": [{"type": "text", "text":
                "You evaluate whether event poster content is tied to a speaker's field of expertise. "
                "Answer only Yes or No, then give a one-sentence reason."}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"Speaker facts: {SPEAKER_FACTS}\n\n"
                f"Is the following event poster summary clearly tied to {SPEAKER_NAME}'s work or "
                f"field of expertise (e.g., photojournalism, National Geographic, Asia, "
                f"documentary photography)? Answer Yes if the summary discusses topics within "
                f"his field, even if not exhaustively.\n\n"
                f"Summary:\n{body_text[:2000]}"}]}
        ]

        msg_engaging = [
            {"role": "system", "content": [{"type": "text", "text":
                "You evaluate event poster content for an academic / cultural-institution setting. "
                "Technical and topical depth is appropriate and expected. Answer only Yes or No, "
                "then give a one-sentence reason."}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"Does the following event poster text make a reasonable effort to be engaging "
                f"and convey why the event is worth attending? It does not need to be marketing "
                f"copy — it just needs to go beyond a dry, purely factual description and show "
                f"some enthusiasm or highlight the value of the topic.\n\n"
                f"Text:\n{body_text[:2000]}"}]}
        ]

        llm_start = time.time()
        llm_tasks = [
            {'id': 'covers_topic', 'func': model, 'args': (msg_topic,)},
            {'id': 'tied_to_speaker', 'func': model, 'args': (msg_speaker,)},
            {'id': 'engaging', 'func': model, 'args': (msg_engaging,)},
        ]
        llm_results = parallel_execute(llm_tasks, max_workers=3)
        print(f"    LLM calls completed in {time.time() - llm_start:.2f}s")

        r3 = llm_results.get('covers_topic')
        step3_success = r3 and r3.strip().lower().startswith("yes")
        step3_details.append(f"LLM: {r3.strip()[:150] if r3 else 'no response'}")
        step3_category = StepCategory.LLM_VLM_JUDGEMENT if r3 else StepCategory.EXECUTION_ERROR

        r4 = llm_results.get('tied_to_speaker')
        step4_success = r4 and r4.strip().lower().startswith("yes")
        step4_details.append(f"LLM: {r4.strip()[:150] if r4 else 'no response'}")
        step4_category = StepCategory.LLM_VLM_JUDGEMENT if r4 else StepCategory.EXECUTION_ERROR

        r5 = llm_results.get('engaging')
        step5_success = r5 and r5.strip().lower().startswith("yes")
        step5_details.append(f"LLM: {r5.strip()[:150] if r5 else 'no response'}")
        step5_category = StepCategory.LLM_VLM_JUDGEMENT if r5 else StepCategory.EXECUTION_ERROR

    checkpoint.add_step(
        f"Covers {TOPIC}",
        step3_success,
        3,
        " | ".join(step3_details),
        max_score = 5,
        category = step3_category
    )
    checkpoint.add_step(
        f"Tied to {SPEAKER_NAME}'s Work",
        step4_success,
        4,
        " | ".join(step4_details),
        max_score = 5,
        category = step4_category
    )
    checkpoint.add_step(
        "Engaging Content",
        step5_success,
        5,
        " | ".join(step5_details),
        max_score = 5,
        category = step5_category
    )

    # --- Step 6: At least 2 reputable, stable sources cited for topic ---
    step_start = time.time()
    step6_details = []
    step6_score = 0
    step6_success = False
    step6_category = StepCategory.LLM_VLM_JUDGEMENT
    notes_text = extract_speaker_notes_text(slide)

    if not body_text or not notes_text:
        step6_details.append("Missing body text or speaker notes")
        step6_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    else:
        if model is None:
            model = load_model(model_id)

        extract_msg = [
            {"role": "system", "content": [{"type": "text", "text":
                "You extract citation URLs from speaker notes. Reply with ONLY a JSON array of URL strings, no other text."}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"Below is a poster's topic summary and its speaker notes. Identify which URLs "
                f"in the speaker notes are explicitly cited as sources for the TOPIC SUMMARY text "
                f"(not for the speaker bio, logo, or other parts). Return a JSON array of just "
                f"those URL strings.\n\n"
                f"TOPIC SUMMARY:\n{body_text[:2000]}\n\n"
                f"SPEAKER NOTES:\n{notes_text[:3000]}"}]}
        ]
        extract_resp = model(extract_msg)

        cited_urls = []
        if extract_resp:
            try:
                match = re.search(r'\[.*?\]', extract_resp, re.DOTALL)
                if match:
                    cited_urls = json.loads(match.group(0))
                    cited_urls = [u for u in cited_urls if isinstance(u, str) and u.startswith('http')]
            except (json.JSONDecodeError, ValueError):
                cited_urls = re.findall(r'https?://[^\s"\',\]]+', extract_resp)

        step6_details.append(f"LLM cited URLs for topic: {len(cited_urls)}")

        if not cited_urls:
            step6_details.append("No URLs cited for topic summary")
        else:
            fetch_tasks = [
                {'id': url, 'func': fetch_page_text_content, 'args': (url,), 'kwargs': {'max_chars': 3000, 'timeout': 15}}
                for url in cited_urls
            ]
            fetch_results = parallel_execute(fetch_tasks, max_workers=3)

            relevance_tasks = []
            for url in cited_urls:
                result = fetch_results.get(url)
                content = result[0] if result and result[0] else ""
                content_block = f"\n\nSite content excerpt:\n{content[:2000]}" if content else "\n\n(Content could not be fetched — evaluate based on URL only.)"
                msg = [
                    {"role": "system", "content": [{"type": "text", "text":
                        "You evaluate whether a cited website is relevant to a specific topic. "
                        "Answer only Yes or No, then give a one-sentence reason."}]},
                    {"role": "user", "content": [{"type": "text", "text":
                        f"Is the following cited website directly relevant to {TOPIC} or "
                        f"{SPEAKER_NAME}'s work (photojournalism, National Geographic, "
                        f"documentary photography, photography techniques, photography history)?\n\n"
                        f"URL: {url}{content_block}"}]}
                ]
                relevance_tasks.append({'id': url, 'func': model, 'args': (msg,)})

            relevance_results = parallel_execute(relevance_tasks, max_workers=3)

            relevant_count = 0
            for url in cited_urls:
                resp = relevance_results.get(url)
                is_relevant = resp and resp.strip().lower().startswith("yes")
                if is_relevant:
                    relevant_count += 1
                step6_details.append(f"  [{'Y' if is_relevant else 'N'}] {url[:70]}")

            step6_score = calculate_percentage_score(min(relevant_count, 2), 2, max_points=5)
            step6_success = step6_score == 5
            step6_details.insert(1, f"Relevant: {relevant_count}/2 ({min(relevant_count, 2) / 2:.0%})")

    checkpoint.add_step(
        "At Least 2 Relevant Cited Sources",
        step6_success,
        6,
        " | ".join(step6_details),
        score = step6_score,
        max_score = 5,
        execution_time = time.time() - step_start,
        category = step6_category
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4 (30 pt): Right Sidebar — Speaker Bio

    Steps:
        1. A shaded right-hand sidebar in Light Orange exists (5 pt)
        2. Right sidebar contains text (5 pt)
        3. Speaker's current affiliation is correct (5 pt)
        4. Speaker's educational background is accurate (5 pt)
        5. 5 distinct, verifiable professional achievements listed (5 pt)
        6. Connection between speaker and Photography (5 pt)
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Right Sidebar — Speaker Bio")

    slides = presentation_data.get('slides', [])
    if not slides:
        for i, name in enumerate(["Light Orange Sidebar", "Sidebar Has Text", "Affiliation",
                                   "Education", "Achievements", "Photography Connection"], 1):
            checkpoint.add_step(
                name,
                False,
                i,
                "No slides found",
                max_score = 5,
                category = StepCategory.EXECUTION_ERROR
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    slide_w, _ = get_slide_dimensions(presentation_data)
    if slide_w is None:
        for step_id, name in enumerate([
            "Light Orange Sidebar Exists", "Sidebar Contains Text",
            "Speaker Affiliation Correct", "Educational Background Accurate",
            f"{ACHIEVEMENT_COUNT}+ Professional Achievements", "Connection to Photography"], start=1):
            checkpoint.add_step(name, False, step_id, "Slide dimensions unavailable", max_score=5,
                                category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    text_boxes = extract_text_boxes_from_slide(slide)

    # --- Find sidebar: right-side text box(es) with the target color background ---
    right_threshold = slide_w * 0.55
    right_text_boxes = []
    for tb in text_boxes:
        b = tb['bbox']
        center_x = b['x'] + b['width'] / 2
        if center_x > right_threshold and len(tb['text'].strip()) >= 30:
            fill = get_shape_background_fill(tb['element'], presentation_data)
            right_text_boxes.append({
                'tb': tb,
                'fill': fill,
                'is_target_color': is_color_close(fill, SIDEBAR_COLOR),
            })

    target_bg_shapes = []
    for elem in slide.get('pageElements', []):
        if 'shape' in elem:
            bbox = get_element_bbox(elem)
            center_x = bbox['x'] + bbox['width'] / 2
            if center_x > right_threshold:
                fill = get_shape_background_fill(elem, presentation_data)
                if is_color_close(fill, SIDEBAR_COLOR):
                    target_bg_shapes.append({'bbox': bbox, 'fill': fill})

    sidebar_box = None
    sidebar_has_target = False

    for rtb in right_text_boxes:
        if rtb['is_target_color']:
            if sidebar_box is None or len(rtb['tb']['text']) > len(sidebar_box['text']):
                sidebar_box = rtb['tb']
                sidebar_has_target = True

    if sidebar_box is None and right_text_boxes:
        for rtb in right_text_boxes:
            tb_bbox = rtb['tb']['bbox']
            for gs in target_bg_shapes:
                gs_bbox = gs['bbox']
                tb_cx = tb_bbox['x'] + tb_bbox['width'] / 2
                tb_cy = tb_bbox['y'] + tb_bbox['height'] / 2
                if (gs_bbox['x'] <= tb_cx <= gs_bbox['x'] + gs_bbox['width'] and
                        gs_bbox['y'] <= tb_cy <= gs_bbox['y'] + gs_bbox['height']):
                    if sidebar_box is None or len(rtb['tb']['text']) > len(sidebar_box['text']):
                        sidebar_box = rtb['tb']
                        sidebar_has_target = True

    if sidebar_box is None and right_text_boxes:
        sidebar_box = max(right_text_boxes, key=lambda x: len(x['tb']['text']))['tb']

    global sidebar_text
    sidebar_text = sidebar_box['text'].strip() if sidebar_box else ""

    # --- Step 1: A shaded right-hand sidebar in Light Orange exists ---
    step_start = time.time()
    target_exists = sidebar_has_target or len(target_bg_shapes) > 0
    checkpoint.add_step(
        "Light Orange Sidebar Exists",
        target_exists,
        1,
        f"Light-orange-filled shapes on right: {len(target_bg_shapes)} | Sidebar has target fill: {sidebar_has_target}",
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.FUZZY_MATCH
    )

    # --- Step 2: Right sidebar contains text ---
    step_start = time.time()
    has_text = len(sidebar_text) >= 30
    checkpoint.add_step(
        "Sidebar Contains Text",
        has_text,
        2,
        f"Sidebar text length: {len(sidebar_text)} chars",
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.DETERMINISTIC
    )

    # --- Steps 3-6: text-based + LLM checks in parallel ---
    step3_success = False
    step3_details = []
    step4_success = False
    step4_details = []
    step5_success = False
    step5_details = []
    step5_score = 0
    step6_success = False
    step6_details = []
    step3_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    step4_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    step5_category = StepCategory.DEPENDENCY_NOT_EVALUATED
    step6_category = StepCategory.DEPENDENCY_NOT_EVALUATED

    if not sidebar_text:
        step3_details.append("No sidebar text")
        step4_details.append("No sidebar text")
        step5_details.append("No sidebar text")
        step6_details.append("No sidebar text")
    else:
        global model
        if model is None:
            model = load_model(model_id)

        system_prompt = (f"You evaluate speaker bios on event posters. "
                         f"{SPEAKER_FACTS} "
                         f"Answer only Yes or No, then give a one-sentence reason.")

        # --- Text-based checks (reliable, deterministic) ---
        # Step 3: Affiliation — must mention "National Geographic" (his lifelong affiliation)
        affil_keywords = ["national geographic"]
        matched_affil = keywords_exact_match(sidebar_text, affil_keywords, substring=True)
        mentions_affiliation = matched_affil is not None
        step3_details.append(f"Text mentions affiliation keyword: {'yes' if mentions_affiliation else 'no'}")

        # Step 4: Education — must mention a degree keyword AND an institution
        edu_keywords_inst = ["wesleyan"]
        matched_edu_inst = keywords_exact_match(sidebar_text, edu_keywords_inst, substring=True)
        has_edu_institution = matched_edu_inst is not None
        step4_details.append(f"Institution: {'yes' if has_edu_institution else 'no'} ({matched_edu_inst or '-'})")

        # Step 6: Photography connection — must mention a related keyword
        photo_keywords = ["photo", "camera", "documentary", "national geographic"]
        matched_photo = keywords_exact_match(sidebar_text, photo_keywords, substring=True)
        mentions_photo = matched_photo is not None
        step6_details.append(f"Text mentions photography keyword: {'yes' if mentions_photo else 'no'}")

        # --- LLM checks (grounded with facts, for nuanced verification) ---
        msg_affiliation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"{SPEAKER_NAME} is best known as a National Geographic photographer. "
                f"Does the following bio explicitly name his current affiliation "
                f"(National Geographic, or a similar professional affiliation), and is it "
                f"stated correctly?\n\n{sidebar_text[:2000]}"}]}
        ]
        msg_education = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"{SPEAKER_NAME} studied at Wesleyan University. "
                f"Does the following bio include his educational background "
                f"(degree and university) and is it accurate?\n\n{sidebar_text[:2000]}"}]}
        ]
        msg_achievements = [
            {"role": "system", "content": [{"type": "text", "text":
                f"You evaluate speaker bios on event posters. "
                f"{SPEAKER_FACTS} "
                "Reply with ONLY a single integer representing the count."}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"How many distinct professional achievements are mentioned in the following "
                f"bio of {SPEAKER_NAME}? Count awards, fellowships, publications, books, "
                f"notable projects, exhibitions, or assignments. "
                f"Reply with ONLY the number.\n\n{sidebar_text[:2000]}"}]}
        ]
        msg_topic_link = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"Does the following bio mention a connection between {SPEAKER_NAME} "
                f"and {TOPIC} (e.g., his work as a photojournalist, his projects, "
                f"or his contributions to the field)?\n\n{sidebar_text[:2000]}"}]}
        ]

        llm_start = time.time()
        llm_tasks = [
            {'id': 'affiliation', 'func': model, 'args': (msg_affiliation,)},
            {'id': 'education', 'func': model, 'args': (msg_education,)},
            {'id': 'achievements', 'func': model, 'args': (msg_achievements,)},
            {'id': 'topic_link', 'func': model, 'args': (msg_topic_link,)},
        ]
        llm_results = parallel_execute(llm_tasks, max_workers=4)
        print(f"    LLM calls completed in {time.time() - llm_start:.2f}s")

        r3 = llm_results.get('affiliation')
        llm_affil = r3 and r3.strip().lower().startswith("yes")
        step3_success = mentions_affiliation and llm_affil
        step3_details.append(f"LLM: {r3.strip()[:120] if r3 else 'no response'}")
        # Keyword tier failing alone decides; otherwise the LLM is the final gate.
        step3_category = (StepCategory.DETERMINISTIC if not mentions_affiliation
                          else StepCategory.LLM_VLM_JUDGEMENT)

        r4 = llm_results.get('education')
        llm_edu = r4 and r4.strip().lower().startswith("yes")
        step4_success = has_edu_institution and llm_edu
        step4_details.append(f"LLM: {r4.strip()[:120] if r4 else 'no response'}")
        step4_category = (StepCategory.DETERMINISTIC if not has_edu_institution
                          else StepCategory.LLM_VLM_JUDGEMENT)

        r5 = llm_results.get('achievements')
        achievement_count = 0
        if r5:
            digits = re.findall(r'\d+', r5.strip())
            achievement_count = int(digits[0]) if digits else 0
        step5_score = calculate_percentage_score(
            min(achievement_count, ACHIEVEMENT_COUNT), ACHIEVEMENT_COUNT, max_points=5
        )
        step5_success = step5_score == 5
        step5_details.append(
            f"Achievements found: {achievement_count}/{ACHIEVEMENT_COUNT} "
            f"({min(achievement_count, ACHIEVEMENT_COUNT) / ACHIEVEMENT_COUNT:.0%})"
        )
        step5_category = StepCategory.LLM_VLM_JUDGEMENT

        r6 = llm_results.get('topic_link')
        llm_topic = r6 and r6.strip().lower().startswith("yes")
        step6_success = mentions_photo and llm_topic
        step6_details.append(f"LLM: {r6.strip()[:120] if r6 else 'no response'}")
        step6_category = (StepCategory.DETERMINISTIC if not mentions_photo
                          else StepCategory.LLM_VLM_JUDGEMENT)

    checkpoint.add_step(
        "Speaker Affiliation Correct",
        step3_success,
        3,
        " | ".join(step3_details),
        max_score = 5,
        category = step3_category
    )
    checkpoint.add_step(
        "Educational Background Accurate",
        step4_success,
        4,
        " | ".join(step4_details),
        max_score = 5,
        category = step4_category
    )
    checkpoint.add_step(
        f"{ACHIEVEMENT_COUNT}+ Professional Achievements",
        step5_success,
        5,
        " | ".join(step5_details),
        score = step5_score,
        max_score = 5,
        category = step5_category
    )
    checkpoint.add_step(
        f"Connection to {TOPIC}",
        step6_success,
        6,
        " | ".join(step6_details),
        max_score = 5,
        category = step6_category
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Checkpoint 5 (15 pt): Footer

    Steps:
        1. Footer is positioned at the bottom right of the poster (5 pt)
        2. Footer contains org contact info (5 pt)
        3. Footer contains speaker's webpage URL (5 pt)
    """
    print("----------------- CHECKPOINT 5 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=15, result=0, name="Footer")

    slides = presentation_data.get('slides', [])
    if not slides:
        for i, name in enumerate(["Footer Position", "Org Contact", "Speaker URL"], 1):
            checkpoint.add_step(
                name,
                False,
                i,
                "No slides found",
                max_score = 5,
                category = StepCategory.EXECUTION_ERROR
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    slide_w, slide_h = get_slide_dimensions(presentation_data)
    if slide_w is None or slide_h is None:
        for step_id, name in enumerate([
            "Footer at Bottom Right", "Org Contact Info",
            f"{SPEAKER_NAME}'s Webpage URL"], start=1):
            checkpoint.add_step(name, False, step_id, "Slide dimensions unavailable", max_score=5,
                                category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    text_boxes = extract_text_boxes_from_slide(slide)

    # --- Find footer: text box in the bottom-right area of the slide ---
    footer_zone = {
        'x': slide_w * 0.40,
        'y': slide_h * 0.90,
        'width': slide_w * 0.60,
        'height': slide_h * 0.10,
    }

    footer_candidates = []
    for tb in text_boxes:
        b = tb['bbox']
        center_y = b['y'] + b['height'] / 2
        center_x = b['x'] + b['width'] / 2
        if center_y > footer_zone['y'] and center_x > footer_zone['x']:
            footer_candidates.append(tb)

    footer_box = None
    if footer_candidates:
        footer_candidates.sort(key=lambda tb: tb['bbox']['y'], reverse=True)
        footer_box = footer_candidates[0]

    footer_text = footer_box['text'].strip() if footer_box else ""

    footer_links = []
    if footer_box:
        elem = footer_box['element']
        if 'shape' in elem and 'text' in elem['shape']:
            text_element = elem['shape']['text']
            for text_run in text_element.get('textElements', []):
                if 'textRun' in text_run:
                    style = text_run['textRun'].get('style', {})
                    if 'link' in style:
                        url = style['link'].get('url', '')
                        if url:
                            footer_links.append(url)
                    content = text_run['textRun'].get('content', '')
                    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
                    footer_links.extend(urls)

    all_footer_text = footer_text + " " + " ".join(footer_links)

    # --- Step 1: Footer positioned at bottom right ---
    step_start = time.time()
    if footer_box is None:
        step1_success = False
        step1_detail = "No text box found in bottom-right area"
    else:
        step1_success = is_bbox_mostly_inside(footer_box['bbox'], footer_zone, threshold=0.5)
        b = footer_box['bbox']
        step1_detail = (f"Footer bbox ({b['x']:.0f},{b['y']:.0f}) vs footer zone "
                        f"({footer_zone['x']:.0f},{footer_zone['y']:.0f}, "
                        f"w={footer_zone['width']:.0f} h={footer_zone['height']:.0f}) | "
                        f"content: {len(footer_text)} chars")

    checkpoint.add_step(
        "Footer at Bottom Right",
        step1_success,
        1,
        step1_detail,
        max_score = 5,
        execution_time = time.time() - step_start,
        category = (StepCategory.SPATIAL if footer_box is not None
                    else StepCategory.DETERMINISTIC)
    )

    # --- Step 2: Contains org contact info ---
    step_start = time.time()
    matched_contact = keywords_exact_match(all_footer_text, CONTACT_KEYWORDS, substring=True)
    step2_success = matched_contact is not None

    checkpoint.add_step(
        "Org Contact Info",
        step2_success,
        2,
        f"Contact keyword matched: {matched_contact if matched_contact else 'none'}",
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.DETERMINISTIC
    )

    # --- Step 3: Contains speaker's webpage URL ---
    step_start = time.time()
    matched_url = keywords_exact_match(all_footer_text, WEBPAGE_KEYWORDS, substring=True)
    step3_success = matched_url is not None

    checkpoint.add_step(
        f"{SPEAKER_NAME}'s Webpage URL",
        step3_success,
        3,
        f"Webpage URL: {'found (' + matched_url + ')' if matched_url else 'not found'}",
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.DETERMINISTIC
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """
    Checkpoint 6 (20 pt): Citations & Callouts

    Steps:
        1. Citations are present in speaker notes (5 pt)
        2. Sources cited correspond to topic summary claims (5 pt)
        3. Sources cited correspond to speaker bio claims (5 pt)
        4. 1-2 callout boxes with quotes/stats/facts about Photography (5 pt)
    """
    print("----------------- CHECKPOINT 6 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Citations & Callouts")

    slides = presentation_data.get('slides', [])
    if not slides:
        for i, name in enumerate(["Notes Have Citations", "Topic Sources", "Bio Sources", "Callout Boxes"], 1):
            checkpoint.add_step(
                name,
                False,
                i,
                "No slides found",
                max_score = 5,
                category = StepCategory.EXECUTION_ERROR
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    notes_text = extract_speaker_notes_text(slide)
    notes_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', notes_text)

    # --- Step 1: Citations present in speaker notes ---
    step_start = time.time()
    step1_success = len(notes_text.strip()) > 0 and len(notes_urls) > 0
    checkpoint.add_step(
        "Citations Present in Speaker Notes",
        step1_success,
        1,
        f"Speaker notes: {len(notes_text)} chars | URLs in notes: {len(notes_urls)}",
        max_score = 5,
        execution_time = time.time() - step_start,
        category = StepCategory.DETERMINISTIC
    )

    if not notes_urls:
        for i, name in enumerate(["Topic Sources", "Bio Sources"], 2):
            checkpoint.add_step(
                name,
                False,
                i,
                "No URLs in speaker notes to evaluate",
                max_score = 5,
                category = StepCategory.DEPENDENCY_NOT_EVALUATED
            )
    else:
        global model
        if model is None:
            model = load_model(model_id)

        url_list_str = "\n".join(f"- {u}" for u in notes_urls)
        classify_msg = [
            {"role": "system", "content": [{"type": "text", "text":
                "You classify citation URLs from speaker notes by which part of a poster they support. "
                "Return ONLY a JSON object mapping each URL to one of: 'topic', 'bio', 'unknown'. "
                "No other text."}]},
            {"role": "user", "content": [{"type": "text", "text":
                f"A poster has a TOPIC SUMMARY about {TOPIC} and a SPEAKER BIO about "
                f"{SPEAKER_NAME}. The speaker notes contain citation URLs. Classify each URL by "
                f"which part of the poster it is cited for.\n\n"
                f"TOPIC SUMMARY:\n{body_text[:1500]}\n\n"
                f"SPEAKER BIO:\n{sidebar_text[:1500]}\n\n"
                f"SPEAKER NOTES:\n{notes_text[:3000]}\n\n"
                f"URLS TO CLASSIFY:\n{url_list_str}\n\n"
                f"Return a JSON object like: "
                f"{{\"https://example.com\": \"topic\", \"https://other.com\": \"bio\"}}. "
                f"Use 'unknown' if you can't tell."}]}
        ]
        classify_resp = model(classify_msg)

        classification = {}
        if classify_resp:
            try:
                match = re.search(r'\{.*\}', classify_resp, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, dict):
                        for url, cat in parsed.items():
                            if isinstance(cat, str) and cat in ('topic', 'bio', 'unknown'):
                                classification[url] = cat
            except (json.JSONDecodeError, ValueError):
                pass

        topic_urls = [u for u, c in classification.items() if c == 'topic']
        bio_urls = [u for u, c in classification.items() if c == 'bio']

        print(f"    Classified: topic={len(topic_urls)} bio={len(bio_urls)}")

        fetch_url_list = list(set(topic_urls + bio_urls))
        fetch_results = {}
        if fetch_url_list:
            fetch_tasks = [
                {'id': url, 'func': fetch_page_text_content, 'args': (url,),
                 'kwargs': {'max_chars': 3000, 'timeout': 15}}
                for url in fetch_url_list
            ]
            fetch_results = parallel_execute(fetch_tasks, max_workers=3)

        verify_tasks = []
        for url in topic_urls:
            result = fetch_results.get(url)
            content = result[0] if result and result[0] else None
            if not content:
                continue
            msg = [
                {"role": "system", "content": [{"type": "text", "text":
                    "You verify whether a cited web page's content supports claims made in a poster text. "
                    "Answer only Yes or No, then give a one-sentence reason."}]},
                {"role": "user", "content": [{"type": "text", "text":
                    f"Does the following web page content support any specific claims made in "
                    f"this poster topic summary?\n\n"
                    f"POSTER TOPIC SUMMARY:\n{body_text[:2000]}\n\n"
                    f"WEB PAGE CONTENT ({url}):\n{content[:2500]}"}]}
            ]
            verify_tasks.append({'id': f'topic::{url}', 'func': model, 'args': (msg,)})

        for url in bio_urls:
            result = fetch_results.get(url)
            content = result[0] if result and result[0] else None
            if not content:
                continue
            msg = [
                {"role": "system", "content": [{"type": "text", "text":
                    "You verify whether a cited web page's content supports claims made in a poster text. "
                    "Answer only Yes or No, then give a one-sentence reason."}]},
                {"role": "user", "content": [{"type": "text", "text":
                    f"Does the following web page content support any specific claims made in "
                    f"this poster speaker bio?\n\n"
                    f"POSTER SPEAKER BIO:\n{sidebar_text[:2000]}\n\n"
                    f"WEB PAGE CONTENT ({url}):\n{content[:2500]}"}]}
            ]
            verify_tasks.append({'id': f'bio::{url}', 'func': model, 'args': (msg,)})

        verify_results = parallel_execute(verify_tasks, max_workers=4) if verify_tasks else {}

        # --- Step 2: Topic sources ---
        step2_success, step2_detail = classify_citation_group(
            topic_urls, fetch_results, verify_results, 'topic'
        )
        # A pass with every URL unfetchable is full credit without any check running.
        topic_all_unfetchable = bool(topic_urls) and all(
            not (fetch_results.get(u) and fetch_results.get(u)[0]) for u in topic_urls
        )
        checkpoint.add_step(
            "Sources for Topic Summary",
            step2_success,
            2,
            step2_detail,
            max_score = 5,
            category = (StepCategory.VACUOUS_PASS if (step2_success and topic_all_unfetchable)
                        else StepCategory.LLM_VLM_JUDGEMENT)
        )

        # --- Step 3: Bio sources ---
        step3_success, step3_detail = classify_citation_group(
            bio_urls, fetch_results, verify_results, 'bio'
        )
        bio_all_unfetchable = bool(bio_urls) and all(
            not (fetch_results.get(u) and fetch_results.get(u)[0]) for u in bio_urls
        )
        checkpoint.add_step(
            "Sources for Speaker Bio",
            step3_success,
            3,
            step3_detail,
            max_score = 5,
            category = (StepCategory.VACUOUS_PASS if (step3_success and bio_all_unfetchable)
                        else StepCategory.LLM_VLM_JUDGEMENT)
        )

    # --- Step 4: Callout boxes (1-2 quotes/stats/facts) ---
    step_start = time.time()
    text_boxes = extract_text_boxes_from_slide(slide)
    slide_w, slide_h = get_slide_dimensions(presentation_data)

    callout_count = 0
    callout_details = []
    if slide_w is not None and slide_h is not None:
        # Identify the structural anchors so we can ignore them when counting callouts
        header_box = find_header_box(text_boxes, slide_h, EVENT_NAME)
        subheader_box = find_subheader_box(text_boxes, header_box)
        # Reuse the body box CP3 settled on so the same box is excluded here
        body_box = (body_box_selected if body_box_selected is not None
                    else find_body_box(text_boxes, slide_w, slide_h))

        # Sidebar text box: largest right-side (>0.55) text box with substantial content
        right_threshold = slide_w * 0.55
        sidebar_candidates = [
            tb for tb in text_boxes
            if (tb['bbox']['x'] + tb['bbox']['width'] / 2) > right_threshold
            and len(tb['text'].strip()) >= 30
        ]
        sidebar_anchor = max(sidebar_candidates, key=lambda tb: len(tb['text'])) if sidebar_candidates else None

        # Footer text box: bottom-right
        footer_anchor = None
        footer_zone_y = slide_h * 0.85
        footer_zone_x = slide_w * 0.40
        footer_candidates = [
            tb for tb in text_boxes
            if (tb['bbox']['y'] + tb['bbox']['height'] / 2) > footer_zone_y
            and (tb['bbox']['x'] + tb['bbox']['width'] / 2) > footer_zone_x
        ]
        if footer_candidates:
            footer_anchor = max(footer_candidates, key=lambda tb: tb['bbox']['y'])

        anchor_ids = set()
        for anchor in (header_box, subheader_box, body_box, sidebar_anchor, footer_anchor):
            if anchor:
                anchor_ids.add(anchor.get('objectId'))

        # Callouts: leftover text boxes with at least 8 chars (filter out tiny artifacts)
        for tb in text_boxes:
            if tb.get('objectId') in anchor_ids:
                continue
            if len(tb['text'].strip()) < 8:
                continue
            callout_count += 1
            callout_details.append(tb['text'].strip()[:60])

    callout_step_success = 1 <= callout_count <= 2
    detail = f"Callout boxes detected: {callout_count}"
    if callout_details:
        detail += " | " + " | ".join(callout_details[:3])
    checkpoint.add_step(
        "1-2 Callout Boxes",
        callout_step_success,
        4,
        detail,
        max_score = 5,
        execution_time = time.time() - step_start,
        category = (StepCategory.STRUCTURAL
                    if (slide_w is not None and slide_h is not None)
                    else StepCategory.EXECUTION_ERROR)
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_7():
    """
    Checkpoint 7 (10 pt): Overall Structure

    Steps:
        1. No text or images are overlapping each other (5 pt)
        2. No text or images are off of the slide (5 pt)
    """
    print("----------------- CHECKPOINT 7 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Overall Structure")

    slides = presentation_data.get('slides', [])
    if not slides:
        checkpoint.add_step(
            "No Overlapping",
            False,
            1,
            "No slides found",
            max_score = 5,
            category = StepCategory.EXECUTION_ERROR
        )
        checkpoint.add_step(
            "All On Slide",
            False,
            2,
            "No slides found",
            max_score = 5,
            category = StepCategory.EXECUTION_ERROR
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[0]
    slide_w, slide_h = get_slide_dimensions(presentation_data)
    if slide_w is None or slide_h is None:
        for step_id, name in enumerate(["No Overlapping Elements", "All Elements On Slide"], start=1):
            checkpoint.add_step(name, False, step_id, "Slide dimensions unavailable", max_score=5,
                                category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    elements = []
    for elem in slide.get('pageElements', []):
        bbox = get_element_bbox(elem)
        if bbox['width'] <= 0 or bbox['height'] <= 0:
            continue
        has_text = 'shape' in elem and 'text' in elem.get('shape', {})
        has_image = 'image' in elem
        if has_text or has_image:
            label = "text" if has_text else "image"
            elements.append({'bbox': bbox, 'type': label, 'id': elem.get('objectId', '?')})

    # --- Step 1: No significant overlapping between elements ---
    step1_start = time.time()
    OVERLAP_THRESHOLD = 0.15
    overlapping_ids = set()
    overlapping_pairs = []

    for i in range(len(elements)):
        a = elements[i]['bbox']
        for j in range(i + 1, len(elements)):
            b = elements[j]['bbox']
            overlap_pct = max(bbox_overlap_ratio(a, b), bbox_overlap_ratio(b, a))
            if overlap_pct > OVERLAP_THRESHOLD:
                overlapping_ids.add(i)
                overlapping_ids.add(j)
                overlapping_pairs.append(
                    f"{elements[i]['type']}({elements[i]['id'][:8]}) & "
                    f"{elements[j]['type']}({elements[j]['id'][:8]}): {overlap_pct:.0%}"
                )

    non_overlapping = len(elements) - len(overlapping_ids)
    step1_score = calculate_percentage_score(non_overlapping, len(elements), max_points=5) if elements else 0
    step1_success = step1_score == 5
    step1_details = [f"Non-overlapping: {non_overlapping}/{len(elements)} "
                     f"({non_overlapping / len(elements):.0%})" if elements else "No elements"]
    for pair in overlapping_pairs[:3]:
        step1_details.append(f"  {pair}")

    checkpoint.add_step(
        "No Overlapping Elements",
        step1_success,
        1,
        " | ".join(step1_details),
        score = step1_score,
        max_score = 5,
        execution_time = time.time() - step1_start,
        category = StepCategory.SPATIAL
    )

    # --- Step 2: All elements on the slide ---
    step2_start = time.time()
    on_slide_count = 0
    off_slide_info = []

    for elem in elements:
        b = elem['bbox']
        cx = b['x'] + b['width'] / 2
        cy = b['y'] + b['height'] / 2
        center_on = 0 <= cx <= slide_w and 0 <= cy <= slide_h
        clipped_w = max(0, min(b['x'] + b['width'], slide_w) - max(b['x'], 0))
        clipped_h = max(0, min(b['y'] + b['height'], slide_h) - max(b['y'], 0))
        total_area = b['width'] * b['height']
        area_on = (clipped_w * clipped_h / total_area) if total_area > 0 else 0
        if center_on and area_on >= 0.85:
            on_slide_count += 1
        else:
            off_slide_info.append(f"{elem['type']}({elem['id'][:8]}): area_on={area_on:.0%}")

    step2_score = calculate_percentage_score(on_slide_count, len(elements), max_points=5) if elements else 0
    step2_success = step2_score == 5
    step2_details = [f"On slide: {on_slide_count}/{len(elements)} "
                     f"({on_slide_count / len(elements):.0%})" if elements else "No elements"]
    for off in off_slide_info[:3]:
        step2_details.append(f"  {off}")

    checkpoint.add_step(
        "All Elements On Slide",
        step2_success,
        2,
        " | ".join(step2_details),
        score = step2_score,
        max_score = 5,
        execution_time = time.time() - step2_start,
        category = StepCategory.SPATIAL
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str, cached_models: Dict[str, Any] = None, browsing_history: List[str] = None):
    total_start = time.time()
    try:
        setup_presentation(workspace_doc_id)

        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]

        checkpoints: List[Checkpoint] = []
        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())
        checkpoints.append(grade_checkpoint_5())
        checkpoints.append(grade_checkpoint_6())
        checkpoints.append(grade_checkpoint_7())

        total_execution_time = time.time() - total_start
        return Result(checkpoints, total_execution_time=total_execution_time)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        failed = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed.add_step("Evaluation", False, 1, f"Fatal error: {e}", execution_time=0,
                        category=StepCategory.EXECUTION_ERROR)
        return Result([failed], total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate event announcement poster presentation")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                        help="Google Slides presentation ID to evaluate")
    parser.add_argument("--cached_models", type=dict, default=None,
                        help="Dictionary of preloaded models")
    args = parser.parse_args()

    print(f"DEBUG mode: {DEBUG}")
    step_start = time.time()

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        cached_models=args.cached_models,
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "[PASS]" if step["success"] else "[FAIL]"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - step_start:.2f} seconds")
