import os
import re
import shutil
import sys
import time

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score, StepCategory
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_exact_match
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download, parallel_execute, parallel_image_match
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_image_source_urls,
    extract_slide_images,
    extract_text_boxes_from_slide,
    get_slide_dimensions,
    get_image_area_percentage_from_api,
    get_text_style_from_shape,
    is_text_big,
)
from src.browsergym.knows.eval.tasks.slides_39_Personal_Lookbook_PaintColors.utils import (
    browser_headers,
    check_browsing_history,
    download_alt_image,
    identify_image_subject_vlm,
    evaluate_image_relevance_vlm,
    find_color_slides,
    get_image_position,
    get_title_text,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/slides_39_Personal_Lookbook_PaintColors/instance_5/")
DATA_DIR = os.path.join(TASK_DIR, "data/")

try:
    _, SLIDES_SERVICE = initialize_google_services(service_type="slides")
except Exception as e:
    print(f"Failed to initialize Google services at module load: {e}")
    SLIDES_SERVICE = None

# VLM model for relevance + topic detection; override via EVAL_MODEL_ID.
model = None
model_id = os.environ.get("EVAL_MODEL_ID", "gemini-3-flash-google-ai")

# Expected room type for this instance, parsed from task.md at setup time.
EXPECTED_ROOM_TYPE = None


def _load_room_type():
    """Load the expected room type from room_type.txt."""
    path = os.path.join(TASK_DIR, "room_type.txt")
    try:
        with open(path, 'r') as f:
            return f.read().strip().lower()
    except Exception as e:
        print(f"Warning: could not read room_type.txt: {e}")
    return None


def setup(presentation_data, workspace_doc_id):
    """
    Set up the evaluation context once before grading any checkpoint.

    Args:
        presentation_data (dict): Raw presentation data from Google Slides API.
        workspace_doc_id (str): Google Slides presentation ID.

    Returns:
        dict: Preprocessed context with keys:
            - slides: list of all slides
            - slide_width_emu: slide width in EMUs
            - slide_height_emu: slide height in EMUs
            - title_slide: the first slide (title slide) or None
            - color_slides: list of color slide dicts from find_color_slides
            - color_names: list of color name strings
            - recommendation_slide_idx: index of recommendation slide or None
            - recommendation_slide: the recommendation slide dict or None
            - workspace_doc_id: passed through for image extraction
            - room_type: expected room type from task.md
    """
    global EXPECTED_ROOM_TYPE
    EXPECTED_ROOM_TYPE = _load_room_type()
    if EXPECTED_ROOM_TYPE:
        print(f"Expected room type: '{EXPECTED_ROOM_TYPE}'")
    else:
        print("Warning: could not determine expected room type from task.md")

    slides = presentation_data.get('slides', [])
    if not isinstance(slides, list):
        print(f"Warning: presentation_data['slides'] is {type(slides).__name__}, not list; treating as empty")
        slides = []
        presentation_data = {**presentation_data, 'slides': []}

    slide_width_emu, slide_height_emu = get_slide_dimensions(presentation_data)
    if slide_width_emu is None or slide_height_emu is None:
        raise ValueError("Slide dimensions unavailable: pageSize missing or malformed")

    title_slide = slides[0] if slides else None

    color_slides, recommendation_slide_idx = find_color_slides(slides)
    color_names = [cs['title'] for cs in color_slides]
    recommendation_slide = (
        slides[recommendation_slide_idx]
        if recommendation_slide_idx is not None and recommendation_slide_idx < len(slides)
        else None
    )

    # Count total content slides (between title and recommendation)
    content_slide_count = 0
    for idx in range(1, len(slides)):
        if idx == recommendation_slide_idx:
            continue
        content_slide_count += 1

    return {
        'presentation_data': presentation_data,
        'slides': slides,
        'slide_width_emu': slide_width_emu,
        'slide_height_emu': slide_height_emu,
        'title_slide': title_slide,
        'color_slides': color_slides,
        'color_names': color_names,
        'content_slide_count': content_slide_count,
        'recommendation_slide_idx': recommendation_slide_idx,
        'recommendation_slide': recommendation_slide,
        'workspace_doc_id': workspace_doc_id,
        'room_type': EXPECTED_ROOM_TYPE,
    }


def grade_checkpoint_1(ctx, browsing_history=None):
    """
    Checkpoint 1 (12 pt): Title slide names the room/project and contains an
    appropriate high-quality image covering at least 70% of the slide space.

    Steps:
        1. Title text matches the VLM-identified room/project (2 pt)
        2. Exactly one image on title slide (2 pt)
        3. Image coverage >= 70% (4 pt)
        4. Browsing history check for the room/project (2 pt)
        5. Image relevance via VLM (2 pt)
    """
    global model
    start = time.time()
    checkpoint = Checkpoint(total=12, result=0, name="Title Slide Image")

    presentation_data = ctx.get('presentation_data')
    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        reason = ctx.get('fetch_error', "No slides found in presentation")
        checkpoint.add_step("Title is Room/Project", False, 1, details=reason, max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("One Image on Title Slide", False, 2, details=reason, max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Coverage >= 70%", False, 3, details=reason, max_score=4, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Browsing History Check", False, 4, details=reason, max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Relevance (VLM)", False, 5, details=reason, max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    title_slide = ctx['title_slide']
    room_type = ctx.get('room_type', '')

    title_text = get_title_text(title_slide)
    images = extract_slide_images(title_slide, ctx['workspace_doc_id'], SLIDES_SERVICE)

    # Identify image subject via VLM for use by browsing-history and relevance steps.
    vlm_topic = ""
    if images:
        if model is None:
            model = load_model(model_id)
        vlm_topic = identify_image_subject_vlm(images, model, DATA_DIR)
    # Reconcile with room_type.txt: if VLM saw a constituent of the authoritative
    # compound (e.g., 'office' for room_type 'home office'), upgrade to the
    # compound so downstream browsing-history / relevance checks match correctly.
    # Genuine mismatches keep the VLM answer.
    if room_type and (not vlm_topic or vlm_topic.lower() in room_type.lower().split()):
        topic = room_type
    else:
        topic = vlm_topic
    ctx['topic'] = topic  # Reused by later checkpoints

    # Step 1: Title text contains the expected room type from task.md (2 pt)
    step_start = time.time()
    step1_category = StepCategory.DETERMINISTIC  # regex word-boundary match
    if not title_text:
        title_pass = False
        title_detail = "No title text found in title position"
    elif not room_type:
        # Fallback: if we couldn't parse room type, just check title is non-empty
        title_pass = True
        title_detail = f"Title: '{title_text}' (room type unknown, accepting any title)"
        step1_category = StepCategory.VACUOUS_PASS  # accepted without a real check
    else:
        title_pass = re.search(rf'\b{re.escape(room_type)}\b', title_text.lower()) is not None
        title_detail = (
            f"Title: '{title_text}', Expected room type: '{room_type}'"
            + (" (match)" if title_pass else " (mismatch)")
        )
    checkpoint.add_step(
        "Title is Room/Project", title_pass, 1,
        details=title_detail,
        max_score=2, execution_time=time.time() - step_start,
        category=step1_category
    )

    # Step 2: Exactly one image on title slide (2 pt)
    step_start = time.time()
    has_one_image = len(images) == 1
    checkpoint.add_step(
        "One Image on Title Slide", has_one_image, 2,
        details=f"Found {len(images)} image(s)" + ("" if has_one_image else " (expected 1)"),
        max_score=2, execution_time=time.time() - step_start,
        category=StepCategory.STRUCTURAL
    )

    if len(images) == 0:
        checkpoint.add_step("Image Coverage >= 70%", False, 3, details="No images on title slide", max_score=4, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Browsing History Check", False, 4, details="No images on title slide", max_score=2, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Image Relevance (VLM)", False, 5, details="No images on title slide", max_score=2, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    # Step 3: Image coverage >= 70% (4 pt)
    step_start = time.time()
    if ctx['slide_width_emu'] is None or ctx['slide_height_emu'] is None:
        checkpoint.add_step("Image Coverage >= 70%", False, 3, details="Slide dimensions unavailable", max_score=4, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Browsing History Check", False, 4, details="Slide dimensions unavailable", max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Relevance (VLM)", False, 5, details="Slide dimensions unavailable", max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - start
        return checkpoint
    image_percentage = get_image_area_percentage_from_api(
        title_slide, ctx['slide_width_emu'], ctx['slide_height_emu']
    )
    meets_70 = image_percentage >= 70.0
    checkpoint.add_step(
        "Image Coverage >= 70%", meets_70, 3,
        details=f"Image covers {image_percentage:.1f}% of slide area",
        max_score=4, execution_time=time.time() - step_start,
        category=StepCategory.SPATIAL
    )

    # Step 4: Browsing history check (2 pt)
    step_start = time.time()
    searched = check_browsing_history(browsing_history, topic) if topic else False
    checkpoint.add_step(
        "Browsing History Check", searched, 4,
        details=f"Agent searched for '{topic}'" if searched else f"No evidence of image search" + (f" for '{topic}'" if topic else ""),
        max_score=2, execution_time=time.time() - step_start,
        category=StepCategory.WEB_VISIT
    )

    # Step 5: Image relevance via VLM (2 pt)
    step_start = time.time()
    all_relevant, num_rel, total_rel = False, 0, 0
    if topic:
        all_relevant, num_rel, total_rel = evaluate_image_relevance_vlm(images, topic, model, DATA_DIR)
    checkpoint.add_step(
        "Image Relevance (VLM)", all_relevant, 5,
        details=f"{num_rel}/{total_rel} image(s) relevant to '{topic}'" if topic else "Could not identify image subject",
        max_score=2, execution_time=time.time() - step_start,
        category=StepCategory.LLM_VLM_JUDGEMENT
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_2(ctx):
    """
    Checkpoint 2 (30 pt): Check that 5-10 color selection slides were created
    with appropriate color names as titles.

    Steps:
        1. Color slide count in 5-10 range (10 pt)
        2. Each slide has a color name title (10 pt, proportional)
        3. Color names are distinct and appropriate for interior design - LLM judge (10 pt, proportional)
    """
    global model
    start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Color Selection Slides")

    presentation_data = ctx.get('presentation_data')
    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        reason = ctx.get('fetch_error', "No slides found in presentation")
        checkpoint.add_step("Color Slide Count (5-10)", False, 1, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Color Name Titles", False, 2, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Colors Distinct & Appropriate (LLM)", False, 3, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    color_slides = ctx['color_slides']
    color_names = ctx['color_names']

    # Step 1: Color slide count in 5-10 range (10 pt)
    step_start = time.time()
    num_colors = len(color_slides)
    in_range = 5 <= num_colors <= 10
    checkpoint.add_step(
        "Color Slide Count (5-10)", in_range, 1,
        details=f"Found {num_colors} color slide(s)",
        max_score=10, execution_time=time.time() - step_start,
        category=StepCategory.STRUCTURAL
    )

    if num_colors == 0:
        checkpoint.add_step("Color Name Titles", False, 2, details="No color slides found", max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Colors Distinct & Appropriate (LLM)", False, 3, details="No color slides found", max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    # Step 2: Each content slide has a title (10 pt, proportional)
    step_start = time.time()
    content_slide_count = ctx['content_slide_count']
    titled_count = len(color_slides)  # color_slides only includes slides with titles
    untitled_count = content_slide_count - titled_count
    all_have_titles = titled_count == content_slide_count
    step2_score = calculate_percentage_score(titled_count, content_slide_count, max_points=10)
    checkpoint.add_step(
        "Color Name Titles", all_have_titles, 2,
        score=step2_score, max_score=10,
        details=f"{titled_count}/{content_slide_count} content slides have titles"
                + (f" ({untitled_count} missing)" if untitled_count > 0 else "")
                + f": {', '.join(color_names)}",
        execution_time=time.time() - step_start,
        category=StepCategory.STRUCTURAL
    )

    # Step 3: Colors are distinct and appropriate for interior design (LLM judge) (10 pt, proportional)
    step_start = time.time()

    # Check uniqueness programmatically (does not require LLM)
    seen = set()
    unique_mask = []
    for name in color_names:
        normalized = name.strip().lower()
        unique_mask.append(normalized not in seen)
        seen.add(normalized)

    # Ask LLM to evaluate each color in a single call
    color_list = "\n".join(f"{i+1}. {name}" for i, name in enumerate(color_names))
    topic = ctx.get('topic', '')
    if topic:
        question = f"Is each of the following a plausible color for painting a wall in a {topic}?"
    else:
        question = "Is each of the following a plausible color for painting a wall?"
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "For each color listed, respond with ONLY the number followed by Yes or No on separate lines. Example:\n1. Yes\n2. No"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{question}\n{color_list}"}]
        }
    ]

    if model is None:
        model = load_model(model_id)
    response = model(messages)

    # Parse LLM response: digit followed (within ~5 non-digits) by yes/no.
    # Tolerates "1. Yes", "(1) Yes", "1: Yes", "Color 1: Yes", "1Yes", etc.
    appropriate_mask = [False] * num_colors
    if response:
        for line in response.strip().split('\n'):
            m = re.search(r'(\d+)\D{0,5}(yes|no)\b', line, re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < num_colors:
                appropriate_mask[idx] = m.group(2).lower() == 'yes'

    # Count colors that are both unique and LLM-appropriate
    pass_count = sum(1 for i in range(num_colors) if unique_mask[i] and appropriate_mask[i])
    all_pass = pass_count == num_colors
    step3_score = calculate_percentage_score(pass_count, num_colors, max_points=10)

    # Category: the LLM verdict decides unless it returned nothing (execution
    # error) or the deterministic uniqueness check alone rejected.
    if not response:
        step3_category = StepCategory.EXECUTION_ERROR
    elif not all_pass and all(appropriate_mask) and not all(unique_mask):
        step3_category = StepCategory.DETERMINISTIC
    else:
        step3_category = StepCategory.LLM_VLM_JUDGEMENT

    details_parts = []
    for i, name in enumerate(color_names):
        status = "pass" if (unique_mask[i] and appropriate_mask[i]) else "fail"
        reasons = []
        if not unique_mask[i]:
            reasons.append("duplicate")
        if not appropriate_mask[i]:
            reasons.append("not appropriate")
        details_parts.append(f"{name} ({status}{': ' + ', '.join(reasons) if reasons else ''})")
    details_str = f"{pass_count}/{num_colors} valid: {'; '.join(details_parts)}"

    checkpoint.add_step(
        "Colors Distinct & Appropriate (LLM)", all_pass, 3,
        score=step3_score, max_score=10,
        details=details_str,
        execution_time=time.time() - step_start,
        category=step3_category
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_3(ctx, browsing_history=None):
    """
    Checkpoint 3 (70 pt): Verify each color slide contains two relevant images
    positioned correctly on the slide.

    Steps:
        1. Agent searched for color + room/project images (10 pt, proportional)
        2. Exactly two images on each color slide (10 pt, proportional)
        3. Image positioning: bottom left + bottom right (10 pt, proportional)
        4. Image relevance to color and room/project (VLM judge) (10 pt, proportional)
        5. Each image has a source URL in its ALT text (10 pt, proportional)
        6. ALT text source URL leads to the same image (10 pt, proportional)
        7. The two images on each color slide are unique (10 pt, proportional)
    """
    global model
    start = time.time()
    checkpoint = Checkpoint(total=70, result=0, name="Color Slide Content")

    presentation_data = ctx.get('presentation_data')
    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        reason = ctx.get('fetch_error', "No slides found in presentation")
        checkpoint.add_step("Browsing History for Color Images", False, 1, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Two Images Per Slide", False, 2, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Positioning (BL + BR)", False, 3, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Relevance (VLM)", False, 4, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("ALT Text Has Source URL", False, 5, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Source Match (ALT URL)", False, 6, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Two Images Are Unique", False, 7, details=reason, max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    color_slides = ctx['color_slides']
    num_colors = len(color_slides)
    topic = ctx.get('topic', '')

    if num_colors == 0:
        reason = ctx.get('fetch_error', "No color slides found")
        # No color slides: downstream checks are skipped, not evaluated.
        checkpoint.add_step("Browsing History for Color Images", False, 1, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Two Images Per Slide", False, 2, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Image Positioning (BL + BR)", False, 3, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Image Relevance (VLM)", False, 4, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("ALT Text Has Source URL", False, 5, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Image Source Match (ALT URL)", False, 6, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Two Images Are Unique", False, 7, details=reason, max_score=10, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    # Step 1: Browsing history - searched for each color + topic (10 pt, proportional)
    step_start = time.time()
    search_pass_count = 0
    search_details = []
    for cs in color_slides:
        color = cs['color']
        search_terms = [color, topic] if topic else [color]
        found = check_browsing_history(browsing_history, search_terms)
        if found:
            search_pass_count += 1
        search_details.append(f"{color} ({'found' if found else 'missing'})")

    step1_score = calculate_percentage_score(search_pass_count, num_colors, max_points=10)
    checkpoint.add_step(
        "Browsing History for Color Images", search_pass_count == num_colors, 1,
        score=step1_score, max_score=10,
        details=f"{search_pass_count}/{num_colors} color searches found: {'; '.join(search_details)}",
        execution_time=time.time() - step_start,
        category=StepCategory.WEB_VISIT
    )

    # Extract images for all color slides once (reused by steps 2-4)
    slide_images = {}
    for cs in color_slides:
        slide_images[cs['index']] = extract_slide_images(
            cs['slide'], ctx['workspace_doc_id'], SLIDES_SERVICE
        )

    # Step 2: Exactly two images per color slide (10 pt, proportional)
    step_start = time.time()
    two_images_count = 0
    image_count_details = []
    for cs in color_slides:
        images = slide_images[cs['index']]
        has_two = len(images) == 2
        if has_two:
            two_images_count += 1
        image_count_details.append(f"{cs['color']}: {len(images)} img(s)")

    step2_score = calculate_percentage_score(two_images_count, num_colors, max_points=10)
    checkpoint.add_step(
        "Two Images Per Slide", two_images_count == num_colors, 2,
        score=step2_score, max_score=10,
        details=f"{two_images_count}/{num_colors} slides have exactly 2 images: {'; '.join(image_count_details)}",
        execution_time=time.time() - step_start,
        category=StepCategory.STRUCTURAL
    )

    # Step 3: Image positioning - bottom left + bottom right (10 pt, proportional)
    step_start = time.time()
    if ctx['slide_width_emu'] is None or ctx['slide_height_emu'] is None:
        checkpoint.add_step("Image Positioning (BL + BR)", False, 3, details="Slide dimensions unavailable", max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Relevance (VLM)", False, 4, details="Slide dimensions unavailable", max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("ALT Text Has Source URL", False, 5, details="Slide dimensions unavailable", max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Image Source Match (ALT URL)", False, 6, details="Slide dimensions unavailable", max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Two Images Are Unique", False, 7, details="Slide dimensions unavailable", max_score=10, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - start
        return checkpoint
    position_pass_count = 0
    position_details = []
    for cs in color_slides:
        images = slide_images[cs['index']]
        if not images:
            position_details.append(f"{cs['color']}: no images")
            continue

        # Check positions using already-extracted image data
        positions = set()
        for img_info in images:
            pos = get_image_position(img_info, ctx['slide_width_emu'], ctx['slide_height_emu'])
            positions.add(pos)

        has_bl_br = 'bottom_left' in positions and 'bottom_right' in positions
        has_exactly_two = len(images) == 2
        if has_bl_br and has_exactly_two:
            position_pass_count += 1
        position_details.append(
            f"{cs['color']}: {', '.join(sorted(positions))}"
            + (f" ({len(images)} images)" if not has_exactly_two else "")
        )

    step3_score = calculate_percentage_score(position_pass_count, num_colors, max_points=10)
    checkpoint.add_step(
        "Image Positioning (BL + BR)", position_pass_count == num_colors, 3,
        score=step3_score, max_score=10,
        details=f"{position_pass_count}/{num_colors} correct: {'; '.join(position_details)}",
        execution_time=time.time() - step_start,
        category=StepCategory.SPATIAL
    )

    # Step 4: Image relevance to color theme and room/project (VLM judge) (10 pt, proportional)
    step_start = time.time()

    if model is None:
        model = load_model(model_id)

    # Build parallel tasks for VLM evaluation. Key by slide index so two
    # slides with the same color name don't collide in the results dict.
    vlm_tasks = []
    skipped_indices = set()
    for cs in color_slides:
        images = slide_images[cs['index']]
        if not images:
            skipped_indices.add(cs['index'])
            continue

        combined_topic = f"{cs['color']} {topic}" if topic else cs['color']
        vlm_tasks.append({
            'id': cs['index'],
            'func': evaluate_image_relevance_vlm,
            'args': (images, combined_topic, model, DATA_DIR),
        })

    vlm_results = parallel_execute(vlm_tasks, max_workers=3) if vlm_tasks else {}

    relevance_pass_count = 0
    relevance_details = []
    for cs in color_slides:
        if cs['index'] in skipped_indices:
            relevance_details.append(f"{cs['color']}: no images")
            continue

        result = vlm_results.get(cs['index'], (False, 0, 0))
        all_relevant, num_rel, total_rel = result
        if all_relevant:
            relevance_pass_count += 1
        relevance_details.append(f"{cs['color']}: {num_rel}/{total_rel} relevant")

    step4_score = calculate_percentage_score(relevance_pass_count, num_colors, max_points=10)
    step4_pass = relevance_pass_count == num_colors
    step4_details = f"{relevance_pass_count}/{num_colors} relevant: {'; '.join(relevance_details)}"
    checkpoint.add_step(
        "Image Relevance (VLM)", step4_pass, 4,
        score=step4_score, max_score=10,
        details=step4_details,
        execution_time=time.time() - step_start,
        category=StepCategory.LLM_VLM_JUDGEMENT
    )

    # Step 5: Each image has a source URL in its ALT text (10 pt, proportional)
    step_start = time.time()
    alt_url_pass_count = 0
    alt_url_details = []

    # Extract image sources per slide (reused in step 6)
    slide_image_sources = {}
    for cs in color_slides:
        slide_image_sources[cs['index']] = extract_image_source_urls(cs['slide'])

    for cs in color_slides:
        image_sources = slide_image_sources[cs['index']]
        total_images = len(image_sources)
        with_url = sum(1 for src in image_sources if src['source_urls'])

        if total_images > 0 and with_url == total_images:
            alt_url_pass_count += 1
        alt_url_details.append(f"{cs['color']}: {with_url}/{total_images} have URLs")

    step5_score = calculate_percentage_score(alt_url_pass_count, num_colors, max_points=10)
    checkpoint.add_step(
        "ALT Text Has Source URL", alt_url_pass_count == num_colors, 5,
        score=step5_score, max_score=10,
        details=f"{alt_url_pass_count}/{num_colors} slides: {'; '.join(alt_url_details)}",
        execution_time=time.time() - step_start,
        category=StepCategory.DETERMINISTIC
    )

    # Step 6: ALT text source URL leads to the same image (10 pt, proportional)
    step_start = time.time()
    source_pass_count = 0
    source_details = []

    temp_dir = os.path.join(DATA_DIR, "temp_images_source_check")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Queue one download per content URL plus one per ALT URL (any-match counts).
        download_tasks = []
        image_pair_map = {}      # object_id -> color name
        alt_count_per_obj = {}   # object_id -> number of ALT URLs queued
        alt_task_url = {}        # alt task id -> URL (for dedup result fan-out)
        url_to_canonical = {}    # ALT URL -> canonical task id that downloads it

        for cs in color_slides:
            images = slide_images[cs['index']]
            image_sources = slide_image_sources[cs['index']]

            obj_to_content = {}
            for img_info in images:
                if img_info.get('objectId') and img_info.get('contentUrl'):
                    obj_to_content[img_info['objectId']] = img_info['contentUrl']

            for img_source in image_sources:
                alt_urls = img_source['source_urls']
                object_id = img_source['objectId']

                if not alt_urls or object_id not in obj_to_content:
                    continue

                image_pair_map[object_id] = cs['color']
                alt_count_per_obj[object_id] = len(alt_urls)
                download_tasks.append({
                    'id': f"{object_id}_content",
                    'func': download_image_from_url,
                    'args': (obj_to_content[object_id], temp_dir),
                })
                for alt_idx, alt_url in enumerate(alt_urls):
                    # Dedup: the same ALT URL across slides gets one download.
                    # Multiple parallel requests to the same Wikimedia URL
                    # can exceed retries on 429; one request + fan-out is
                    # both faster and more reliable.
                    tid = f"{object_id}_alt_{alt_idx}"
                    alt_task_url[tid] = alt_url
                    if alt_url not in url_to_canonical:
                        url_to_canonical[alt_url] = tid
                        download_tasks.append({
                            'id': tid,
                            'func': download_alt_image,
                            'args': (alt_url, temp_dir),
                            'kwargs': {'headers': browser_headers(alt_url), 'wayback_fallback': True},
                        })

        # Phase 1: download all images in parallel.
        downloaded = parallel_download(download_tasks, max_workers=5, use_rate_limit=False) if download_tasks else {}

        # Fan dedup'd ALT downloads back out to every task id that uses that URL.
        for tid, alt_url in alt_task_url.items():
            canonical = url_to_canonical.get(alt_url)
            if canonical and canonical != tid:
                downloaded[tid] = downloaded.get(canonical)

        # Phase 2: build match tasks — content vs every downloaded ALT.
        match_tasks = []
        for object_id in image_pair_map:
            content_path = downloaded.get(f"{object_id}_content")
            if not content_path:
                continue
            for alt_idx in range(alt_count_per_obj.get(object_id, 0)):
                alt_path = downloaded.get(f"{object_id}_alt_{alt_idx}")
                if alt_path:
                    match_tasks.append({
                        'id': f"{object_id}_alt_{alt_idx}",
                        'candidate_path': content_path,
                        'gold_path': alt_path,
                    })

        # Phase 3: compare all candidate pairs in parallel.
        match_results = parallel_image_match(match_tasks, max_workers=5) if match_tasks else {}

        # Per-objectId aggregation
        obj_status = {}  # object_id -> {'downloaded': bool, 'matched': bool}
        for object_id in image_pair_map:
            n_alts = alt_count_per_obj.get(object_id, 0)
            content_ok = downloaded.get(f"{object_id}_content") is not None
            alt_ok = any(
                downloaded.get(f"{object_id}_alt_{i}") is not None
                for i in range(n_alts)
            )
            matched = any(
                match_results.get(f"{object_id}_alt_{i}", (False, None))[0]
                for i in range(n_alts)
            )
            # The deciding tier ('exact' or 'perceptual_hash') of the first
            # matching ALT, threaded through for the step category.
            match_method = next(
                (match_results.get(f"{object_id}_alt_{i}", (False, None))[1]
                 for i in range(n_alts)
                 if match_results.get(f"{object_id}_alt_{i}", (False, None))[0]),
                None,
            )
            obj_status[object_id] = {
                'downloaded': content_ok and alt_ok,
                'matched': matched,
                'method': match_method,
            }

        # Per-color aggregation: track download failures separately from mismatch.
        color_stats = {}  # color -> {'total': N, 'downloaded': N, 'matched': N}
        for object_id, color in image_pair_map.items():
            s = color_stats.setdefault(color, {'total': 0, 'downloaded': 0, 'matched': 0})
            s['total'] += 1
            if obj_status[object_id]['downloaded']:
                s['downloaded'] += 1
            if obj_status[object_id]['matched']:
                s['matched'] += 1

        for cs in color_slides:
            s = color_stats.get(cs['color'], {'total': 0, 'downloaded': 0, 'matched': 0})
            n = s['total']
            if n > 0 and s['matched'] == n:
                source_pass_count += 1
            parts = [f"{s['matched']}/{n} matched"]
            failed_dl = n - s['downloaded']
            if failed_dl > 0:
                parts.append(f"{failed_dl} download failed")
            source_details.append(f"{cs['color']}: {', '.join(parts)}")

        # (category, success) per image for StepCategory.aggregate(): download
        # failures could not be checked; matches carry their deciding tier;
        # rejects were decided by the last tier that ran (perceptual hash).
        step6_items = []
        for object_id in image_pair_map:
            st = obj_status[object_id]
            if not st['downloaded']:
                step6_items.append((StepCategory.EXECUTION_ERROR, False))
            elif st['matched']:
                step6_items.append((StepCategory.from_match_method(st['method']), True))
            else:
                step6_items.append((StepCategory.FUZZY_MATCH, False))
        # Slides with no testable ALT/content pair (no ALT URL or no content
        # URL) fail the step without any comparison having run.
        for cs in color_slides:
            if color_stats.get(cs['color'], {'total': 0})['total'] == 0:
                step6_items.append((StepCategory.EXECUTION_ERROR, False))
        step6_category = StepCategory.aggregate(step6_items)

        # Step 7: Two images on each slide are distinct (10 pt, proportional)
        # Pair the two content images per slide and call them duplicates if
        # they exact- or perceptual-hash match. Must run before `finally`
        # deletes temp_dir.
        step7_start = time.time()
        unique_match_tasks = []
        slide_pair_status = {}  # cs['index'] -> 'pending' | 'too_few' | 'no_paths'
        for cs in color_slides:
            images = slide_images[cs['index']]
            if len(images) != 2:
                slide_pair_status[cs['index']] = 'too_few'
                continue
            obj_a = images[0].get('objectId')
            obj_b = images[1].get('objectId')
            path_a = downloaded.get(f"{obj_a}_content") if obj_a else None
            path_b = downloaded.get(f"{obj_b}_content") if obj_b else None
            if not path_a or not path_b:
                slide_pair_status[cs['index']] = 'no_paths'
                continue
            slide_pair_status[cs['index']] = 'pending'
            unique_match_tasks.append({
                'id': cs['index'],
                'candidate_path': path_a,
                'gold_path': path_b,
            })

        unique_match_results = (
            parallel_image_match(unique_match_tasks, max_workers=5)
            if unique_match_tasks else {}
        )

        unique_pass_count = 0
        unique_details = []
        for cs in color_slides:
            status = slide_pair_status.get(cs['index'])
            if status == 'too_few':
                unique_details.append(f"{cs['color']}: only {len(slide_images[cs['index']])} image(s)")
                continue
            if status == 'no_paths':
                unique_details.append(f"{cs['color']}: download failed")
                continue
            matched, _ = unique_match_results.get(cs['index'], (False, None))
            if matched:
                unique_details.append(f"{cs['color']}: duplicate")
            else:
                unique_pass_count += 1
                unique_details.append(f"{cs['color']}: unique")

        step7_score = calculate_percentage_score(unique_pass_count, num_colors, max_points=10)
        checkpoint.add_step(
            "Two Images Are Unique", unique_pass_count == num_colors, 7,
            score=step7_score, max_score=10,
            details=f"{unique_pass_count}/{num_colors} slides have unique images: {'; '.join(unique_details)}",
            execution_time=time.time() - step7_start,
            category=StepCategory.FUZZY_MATCH,
        )

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    step6_score = calculate_percentage_score(source_pass_count, num_colors, max_points=10)
    checkpoint.add_step(
        "Image Source Match (ALT URL)", source_pass_count == num_colors, 6,
        score=step6_score, max_score=10,
        details=f"{source_pass_count}/{num_colors} slides verified: {'; '.join(source_details)}",
        execution_time=time.time() - step_start,
        category=step6_category
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_4(ctx):
    """
    Checkpoint 4 (10 pt): Check that a final recommendation slide was created
    with the agent's color choice clearly stated.

    Steps:
        1. Recommendation slide exists (2 pt)
        2. "{COLOR} is the best choice" text found, case-insensitive (3 pt)
        3. Chosen color matches a previously presented option (3 pt)
        4. Text in large font (2 pt)
    """
    start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Recommendation Slide")

    presentation_data = ctx.get('presentation_data')
    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        reason = ctx.get('fetch_error', "No slides found in presentation")
        checkpoint.add_step("Recommendation Slide Exists (Last)", False, 1, details=reason, max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step('"{COLOR} is the best choice" Text', False, 2, details=reason, max_score=3, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Color Matches Previous Option", False, 3, details=reason, max_score=3, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Large Font", False, 4, details=reason, max_score=2, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    recommendation_slide = ctx['recommendation_slide']
    recommendation_slide_idx = ctx['recommendation_slide_idx']
    color_names = ctx['color_names']
    num_slides = len(ctx['slides'])

    # Step 1: Recommendation slide exists and is the last slide (2 pt)
    step_start = time.time()
    has_slide = recommendation_slide is not None
    is_last = has_slide and recommendation_slide_idx == num_slides - 1

    if not has_slide:
        detail = "No recommendation slide found"
    elif is_last:
        detail = f"Recommendation slide found at slide {recommendation_slide_idx + 1} (last slide)"
    else:
        detail = f"Recommendation slide at slide {recommendation_slide_idx + 1}, but last slide is {num_slides}"

    checkpoint.add_step(
        "Recommendation Slide Exists (Last)", is_last, 1,
        details=detail,
        max_score=2, execution_time=time.time() - step_start,
        category=StepCategory.STRUCTURAL
    )

    if not has_slide:
        reason = ctx.get('fetch_error', "No recommendation slide found")
        # No recommendation slide: downstream checks are skipped, not evaluated.
        checkpoint.add_step('"{COLOR} is the best choice" Text', False, 2, details=reason, max_score=3, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Color Matches Previous Option", False, 3, details=reason, max_score=3, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.add_step("Large Font", False, 4, details=reason, max_score=2, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    # Find "{COLOR} is the best choice" and identify which color was chosen.
    text_boxes = extract_text_boxes_from_slide(recommendation_slide)

    target_phrase = 'is the best choice'
    best_choice_text = None
    chosen_color = None
    matching_element = None

    for tb in text_boxes:
        text_norm = ' '.join(tb['text'].split())
        text_lower = text_norm.lower()
        if target_phrase not in text_lower:
            continue
        best_choice_text = text_norm
        matching_element = tb['element']

        # Match any known color name anywhere in the text box.
        chosen_color = keywords_exact_match(text_norm, color_names, substring=True)
        break

    # Step 2: "{COLOR} is the best choice" text found, case-insensitive (3 pt)
    step_start = time.time()
    has_text = best_choice_text is not None
    checkpoint.add_step(
        '"{COLOR} is the best choice" Text', has_text, 2,
        details=f'Found: "{best_choice_text}"' if has_text else 'Text "{COLOR} is the best choice" not found',
        max_score=3, execution_time=time.time() - step_start,
        category=StepCategory.DETERMINISTIC
    )

    # Step 3: Chosen color matches a previously presented option (3 pt)
    step_start = time.time()
    color_matches = False
    if chosen_color and color_names:
        chosen_lower = chosen_color.lower()
        color_matches = any(c.strip().lower() == chosen_lower for c in color_names)

    checkpoint.add_step(
        "Color Matches Previous Option", color_matches, 3,
        details=f"'{chosen_color}' matches color slides" if color_matches else (
            f"'{chosen_color}' not found in: {', '.join(color_names)}" if chosen_color else "No color extracted"
        ),
        max_score=3, execution_time=time.time() - step_start,
        category=StepCategory.DETERMINISTIC
    )

    # Step 4: Text in large font (2 pt)
    step_start = time.time()
    text_style = get_text_style_from_shape(matching_element['shape']) if matching_element else {}
    is_large = is_text_big(text_style, min_pt=18, element=matching_element)
    font_size = text_style.get('fontSize', {}).get('magnitude', 0) if text_style else 0
    checkpoint.add_step(
        "Large Font", is_large, 4,
        details=f"Font size: {font_size}pt" if font_size > 0 else "Could not determine font size",
        max_score=2, execution_time=time.time() - step_start,
        # No best-choice text box found (step 2 failed) means there was
        # nothing to style-check; otherwise the font-size threshold decides.
        category=(StepCategory.DETERMINISTIC if matching_element
                  else StepCategory.DEPENDENCY_NOT_EVALUATED)
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
    """
    Grade all checkpoints for the Personal Lookbook Paint Colors task.

    Args:
        workspace_doc_id (str): Google Slides presentation ID.
        cached_models (dict, optional): Dictionary of preloaded models by model_id.
        browsing_history (list, optional): List of URLs visited during task.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start = time.time()

    global model
    if cached_models and model_id in cached_models:
        model = cached_models[model_id]

    # Fetch presentation data and set up context once
    fetch_error = None
    try:
        presentation_data = SLIDES_SERVICE.presentations().get(
            presentationId=workspace_doc_id
        ).execute()
    except Exception as e:
        print(f"Error fetching presentation: {e}")
        presentation_data = {'slides': []}
        fetch_error = f"Failed to fetch presentation: {e}"

    try:
        ctx = setup(presentation_data, workspace_doc_id)
    except Exception as e:
        print(f"setup() raised: {e}")
        ctx = {
            'presentation_data': {'slides': []},
            'slides': [],
            'slide_width_emu': None,
            'slide_height_emu': None,
            'title_slide': None,
            'color_slides': [],
            'color_names': [],
            'content_slide_count': 0,
            'recommendation_slide_idx': None,
            'recommendation_slide': None,
            'workspace_doc_id': workspace_doc_id,
            'fetch_error': f"Setup failed: {e}",
        }
    if fetch_error and 'fetch_error' not in ctx:
        ctx['fetch_error'] = fetch_error

    # (name, total_pts, [(step_name, step_max_score), ...], callable).
    # Step list mirrors each grade_checkpoint_N to synthesize a fully-failed
    # checkpoint on crash.
    cp_specs = [
        ('Title Slide Image', 12, [
            ('Title is Room/Project', 2),
            ('One Image on Title Slide', 2),
            ('Image Coverage >= 70%', 4),
            ('Browsing History Check', 2),
            ('Image Relevance (VLM)', 2),
        ], lambda: grade_checkpoint_1(ctx, browsing_history)),
        ('Color Selection Slides', 30, [
            ('Color Slide Count (5-10)', 10),
            ('Color Name Titles', 10),
            ('Colors Distinct & Appropriate (LLM)', 10),
        ], lambda: grade_checkpoint_2(ctx)),
        ('Color Slide Content', 70, [
            ('Browsing History for Color Images', 10),
            ('Two Images Per Slide', 10),
            ('Image Positioning (BL + BR)', 10),
            ('Image Relevance (VLM)', 10),
            ('ALT Text Has Source URL', 10),
            ('Image Source Match (ALT URL)', 10),
            ('Two Images Are Unique', 10),
        ], lambda: grade_checkpoint_3(ctx, browsing_history)),
        ('Recommendation Slide', 10, [
            ('Recommendation Slide Exists (Last)', 2),
            ('"{COLOR} is the best choice" Text', 3),
            ('Color Matches Previous Option', 3),
            ('Large Font', 2),
        ], lambda: grade_checkpoint_4(ctx)),
    ]

    checkpoints = []
    for name, total, steps, fn in cp_specs:
        try:
            checkpoints.append(fn())
        except Exception as e:
            print(f"Checkpoint '{name}' crashed: {e}")
            cp = Checkpoint(total=total, result=0, name=name)
            for i, (step_name, max_score) in enumerate(steps, start=1):
                cp.add_step(step_name, False, i,
                            details=f"Crashed: {e}", max_score=max_score,
                            category=StepCategory.EXECUTION_ERROR)
            checkpoints.append(cp)

    return Result(checkpoints, total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Personal Lookbook Paint Colors Task")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                        help="Google Slides presentation ID")
    parser.add_argument("--browsing_history", nargs='+',
                        help="List of URLs visited")
    parser.add_argument("--checkpoint", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run specific checkpoint only")

    args = parser.parse_args()

    start_time = time.time()

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history,
    )

    # Print results
    report = result.get_detailed_report()
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    for cp in report["checkpoints"]:
        print(f"\n{cp['name']}: {cp['score']}")
        if cp.get('execution_time'):
            print(f"  Time: {cp['execution_time']:.2f}s")
        for step in cp["steps"]:
            status = "PASS" if step["success"] else "FAIL"
            print(f"  [{status}] {step['name']} ({step['score']}/{step['max_score']}): {step['details']}")

    score = report["final_score"]
    print(f"\nFinal Score: {score['result']}/{score['total']}")
    print(f"Total Time: {time.time() - start_time:.2f}s")
