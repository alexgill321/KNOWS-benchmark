from itertools import chain
import glob
import os
import sys
import time
import argparse
import requests
import re
from typing import List, Dict, Any, Optional
import shutil

# base path helper

def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# imports from eval_utils
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.text_utils import (
    keywords_exact_match,
    keywords_match_robust,
    keyword_exact_match,
)
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_links,
    extract_slide_text,
    extract_text_boxes_from_slide,
    extract_title_text,
    get_slide_background_color,
    colors_are_different,
    extract_slide_images,
    download_slide_image,
    get_text_style_from_shape,
    extract_table_from_slide,
    get_element_bbox
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download, parallel_execute
from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_url_content

from src.browsergym.knows.eval.tasks.slides_42_personal_none_product_comparison.utils import (
    detect_color_name,
    ensure_scheme,
    extract_device_info_with_llm,
    evaluate_device_info_with_llm,
    validate_rankings,
    download_images_from_url
)

# Constants
# Resolve TASK_DIR from this file's location so the path is correct regardless of
# how BASE_PATH is configured at runtime.
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
GOLD_IMAGES_DIR = os.path.join(DATA_DIR, "gold_images")
model = None
model_id = "gemini-3-flash-google-ai"

DRIVE_SERVICE, SLIDES_SERVICE = initialize_google_services(service_type="slides")

# Global
presentation_id = None
presentation_data = None
gold_devices = None

def setup_presentation(workspace_doc_id):
    """
    Setup presentation processing.

    Args:
        workspace_doc_id (str): Google Slides presentation ID to evaluate.
    """
    global presentation_id, presentation_data, gold_devices

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace presentation ID: {workspace_doc_id}")
    presentation_id = workspace_doc_id

    # Fetch presentation data
    presentation_data = SLIDES_SERVICE.presentations().get(presentationId=presentation_id).execute()

    # Load gold characters list
    gold_devices_path = os.path.join(DATA_DIR, "gold_devices.txt")
    if os.path.exists(gold_devices_path):
        with open(gold_devices_path, 'r') as f:
            gold_devices = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: gold_devices.txt not found at {gold_devices_path}")
        gold_devices = []


def grade_checkpoint_1():
    """
    Checkpoint 1 (6pt): Title slide has all required elements.

    Outcome Evaluation:
    - Exact match on "A Gift for James!" found.
    - Title is in bold.
    - Subtitle correctly lists all 3 tablet options from the gold list.
    - Image represents Deloitte, such as an office or branding scene, found.
    - Image is to the right of the title.
    - Deloitte's official colors, Deloitte Green and black, are used.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Title Slide Validation")

    checkpoint_1_step_names = [
        "Title Match",
        "Title Is Bold",
        "Subtitle Includes All Tablets",
        "Deloitte Image Found",
        "Title Left of Image",
        "Deloitte Colors",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(checkpoint_1_step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Assume first slide is the title slide
    title_slide = presentation_data['slides'][0]

    global model
    if model is None:
        model = load_model(model_id)

    # Step 1: Match on 'A Gift for James!' (robust: exact then LLM fallback)
    step_start = time.time()
    title_text = extract_title_text(title_slide)
    title_found_text, title_match_method = keywords_match_robust(title_text, "A Gift for James!", model=model, return_method=True)
    title_found = bool(title_found_text)

    checkpoint.add_step("Title Match", title_found, 1, "Found title 'A Gift for James!'" if title_found else "Title does not match 'A Gift for James!'", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC if title_match_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT)

    # Step 2: Title is bold (any run carrying the title text is bold)
    step_start = time.time()
    text_boxes = extract_text_boxes_from_slide(title_slide)

    title_bold = False
    title_text_box = None
    for text_box in text_boxes:
        text_box_text = text_box.get('text', '')
        # The title may be split across multiple text boxes/runs; accept either
        # an exact match or a substring containment in either direction.
        if title_text and (
            keyword_exact_match(title_text, text_box_text)
            or keyword_exact_match(text_box_text, title_text, substring=True)
            or keyword_exact_match(title_text, text_box_text, substring=True)
        ):
            element = text_box.get('element', {})
            title_style = get_text_style_from_shape(element.get('shape', {}))
            if title_style.get('bold'):
                title_bold = True
                title_text_box = text_box
                break
            # Keep the first matching box as a positional reference even when not bold.
            if title_text_box is None:
                title_text_box = text_box
    checkpoint.add_step("Title Is Bold", title_bold, 2, "Title text is bold" if title_bold else "Title text not bold or could not determine", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)

    # Step 3: Subtitle lists all 3 device options
    step_start = time.time()
    slide_text = extract_slide_text(title_slide)
    devices_matched = True
    unmatched = ""
    device_match_items = []
    for device in gold_devices:
        device_match_text, device_match_method = keywords_match_robust(slide_text, device, model=model, substring=True, return_method=True)
        device_match_items.append((StepCategory.DETERMINISTIC if device_match_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT, bool(device_match_text)))
        if not device_match_text:
            devices_matched = False
            unmatched += device + "; "

    checkpoint.add_step("Subtitle Includes All Tablets", devices_matched, 3, "Subtitle lists all 3 tablets" if devices_matched else f"Subtitle missing tablets: {unmatched}", execution_time=time.time() - step_start, category=StepCategory.aggregate(device_match_items))

    # Steps 4 & 5: Image found and title-left-of-image. Both must record a step
    # regardless of any exceptions while downloading/judging images.
    step_start = time.time()
    images = extract_slide_images(title_slide, presentation_id, SLIDES_SERVICE)
    uni_image_valid = False
    uni_image_detail = "No valid Deloitte image found"
    uni_image_category = StepCategory.LLM_VLM_JUDGEMENT
    matching_image = None
    title_left_of_image = False
    title_left_detail = "Title is not to the left of the Deloitte image"
    title_left_category = StepCategory.DEPENDENCY_NOT_EVALUATED

    temp_dir = os.path.join(DATA_DIR, "temp_images")
    try:
        os.makedirs(temp_dir, exist_ok=True)
        # Download and save each image temporarily
        for idx, img_info in enumerate(images):
            if img_info.get('contentUrl'):
                try:
                    img = download_slide_image(img_info['contentUrl'])
                except Exception as e:
                    print(f"Failed to download slide image {idx}: {e}")
                    continue
                if img:
                    temp_img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                    try:
                        img.save(temp_img_path)
                    except Exception as e:
                        print(f"Failed to save slide image {idx}: {e}")

        if os.path.isdir(temp_dir) and os.listdir(temp_dir):
            try:
                matching_image = binary_judge_image(
                    model,
                    temp_dir,
                    "Is this an image representing Deloitte, such as an office or branding scene?"
                )
            except Exception as e:
                print(f"Deloitte image LLM check failed: {e}")
                matching_image = None
                uni_image_detail = f"Deloitte image check failed: {e}"
                uni_image_category = StepCategory.EXECUTION_ERROR

            if matching_image:
                uni_image_valid = True
                uni_image_detail = "Found an image representing Deloitte"
        else:
            uni_image_detail = "No images available on title slide"
            uni_image_category = StepCategory.DETERMINISTIC
    except Exception as e:
        print(f"Unexpected error during image processing: {e}")
        uni_image_detail = f"Unexpected error during image processing: {e}"
        uni_image_category = StepCategory.EXECUTION_ERROR
    finally:
        # Step 4 always recorded
        checkpoint.add_step("Deloitte Image Found", uni_image_valid, 4, uni_image_detail, execution_time=time.time() - step_start, category=uni_image_category)

        # Step 5: Title-left-of-image (best-effort; always recorded)
        step5_start = time.time()
        try:
            if uni_image_valid and matching_image:
                title_x = None
                title_width = 0.0
                ref_box = title_text_box
                if ref_box is None:
                    for text_box in text_boxes:
                        if title_text and keyword_exact_match(title_text, text_box.get('text', '')):
                            ref_box = text_box
                            break
                if ref_box is not None:
                    title_x = ref_box['bbox']['x']
                    title_width = ref_box['bbox']['width']

                # Map the matching image filename back to the corresponding pageElement.
                img_element = None
                match_filename = os.path.basename(matching_image)
                m = re.match(r"temp_image_(\d+)\.png$", match_filename)
                if m:
                    target_idx = int(m.group(1))
                    image_elements = [el for el in title_slide.get('pageElements', []) if 'image' in el]
                    if 0 <= target_idx < len(image_elements):
                        img_element = image_elements[target_idx]

                if img_element is not None and title_x is not None:
                    image_x = get_element_bbox(img_element)['x']
                    title_center_x = title_x + title_width / 2
                    title_left_of_image = image_x > title_center_x
                    title_left_detail = (
                        "Title is to the left of the Deloitte image" if title_left_of_image
                        else "Title is not to the left of the Deloitte image"
                    )
                    title_left_category = StepCategory.SPATIAL
                else:
                    title_left_detail = "Could not determine title or image position"
                    title_left_category = StepCategory.DEPENDENCY_NOT_EVALUATED
            else:
                title_left_detail = "Skipping position check: no valid Deloitte image"
                title_left_category = StepCategory.DEPENDENCY_NOT_EVALUATED
        except Exception as e:
            print(f"Title-left-of-image check failed: {e}")
            title_left_detail = f"Position check failed: {e}"
            title_left_category = StepCategory.EXECUTION_ERROR
        checkpoint.add_step("Title Left of Image", title_left_of_image, 5, title_left_detail, execution_time=time.time() - step5_start, category=title_left_category)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Step 6: Deloitte's official colors (Deloitte Green and black) used somewhere
    # on the slide (background, title/subtitle text, or shape fills).
    official_colors = [
        ("Deloitte Green", {'r': 0.525, 'g': 0.737, 'b': 0.145}),
        ("black", {'r': 0.0, 'g': 0.0, 'b': 0.0}),
    ]
    step_start = time.time()
    color_valid = False
    color_detail_parts = []
    color_category = StepCategory.FUZZY_MATCH

    def _color_close(rgb_dict, official, threshold=0.2):
        if not rgb_dict:
            return False
        # Accept either {r,g,b} or {red,green,blue} key shapes.
        r = rgb_dict.get('r', rgb_dict.get('red'))
        g = rgb_dict.get('g', rgb_dict.get('green'))
        b = rgb_dict.get('b', rgb_dict.get('blue'))
        if r is None or g is None or b is None:
            return False
        return not colors_are_different({'r': r, 'g': g, 'b': b}, official, threshold=threshold)

    def _match_any_official(rgb_dict, threshold=0.2):
        for name, color in official_colors:
            if _color_close(rgb_dict, color, threshold=threshold):
                return name
        return None

    try:
        bg_color = get_slide_background_color(title_slide, presentation_data)
        bg_match = _match_any_official(bg_color, threshold=0.2)
        if bg_match:
            color_valid = True
            color_detail_parts.append(f"background uses {bg_match}")

        if not color_valid:
            # Check title/subtitle text run colors for an official Deloitte color.
            for text_box in text_boxes:
                element = text_box.get('element', {})
                shape = element.get('shape', {})
                style = get_text_style_from_shape(shape, presentation_data)
                fg = style.get('foregroundColor')
                fg_match = _match_any_official(fg, threshold=0.2)
                if fg_match:
                    color_valid = True
                    color_detail_parts.append(f"text run on '{text_box.get('text', '')[:30]}' uses {fg_match}")
                    break

        if not color_valid:
            # Check shape fills (e.g., colored bars/accents) for an official Deloitte color.
            for element in title_slide.get('pageElements', []):
                shape = element.get('shape', {})
                fill = shape.get('shapeProperties', {}).get('shapeBackgroundFill', {})
                solid = fill.get('solidFill', {})
                color_info = solid.get('color', {})
                rgb = color_info.get('rgbColor', {})
                if rgb:
                    fill_match = _match_any_official(
                        {'r': rgb.get('red', 0), 'g': rgb.get('green', 0), 'b': rgb.get('blue', 0)},
                        threshold=0.2,
                    )
                    if fill_match:
                        color_valid = True
                        color_detail_parts.append(f"a shape fill uses {fill_match}")
                        break
    except Exception as e:
        print(f"Deloitte colors check error: {e}")
        color_detail_parts.append(f"check error: {e}")
        color_category = StepCategory.EXECUTION_ERROR

    color_detail = (
        "Deloitte official color found on slide (" + "; ".join(color_detail_parts) + ")"
        if color_valid else "No Deloitte Green or black detected on background, title text, or shape fills"
    )
    checkpoint.add_step("Deloitte Colors", bool(color_valid), 6, color_detail, execution_time=time.time() - step_start, category=color_category)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2 (2pt): The Challenge and the goal slides meet requirements.

    Outcome Evaluation:
    - At least one line in the slide body explains the challenge of the search.
    - At least one line in the slide body explains the goal of the search.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Challenge & Goal")

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Explains Challenge", False, 1, "No slides found in presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Explains Goal", False, 2, "No slides found in presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Slide 2 is index 1
    slides = presentation_data['slides']
    if len(slides) < 2:
        checkpoint.add_step("Explains Challenge", False, 1, "Challenge slide not found or not in the correct order", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
        checkpoint.add_step("Explains Goal", False, 2, "Challenge slide not found or not in the correct order", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[1]

    # Step 2 & 3: At least one line explaining challenge and one explaining goal
    step_start = time.time()
    slide_text = extract_slide_text(slide)

    if model is None:
        model = load_model(model_id)

    # Step 1: Challenge explanation (own try/except so a failure here doesn't drop step 2).
    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant who evaluates whether the text describes at least one challenge in choosing a tablet for a newly promoted corporate professional. Response with ONLY 'yes' or 'no'."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Is there at least one challenge in choosing a tablet for a newly promoted corporate professional in this text?\n\nText: {slide_text}"}]
            }
        ]
        response = model(messages).strip().lower()
        challenge_ok = 'yes' in response
        checkpoint.add_step("Explains Challenge", challenge_ok, 1, "Found challenge explanation" if challenge_ok else "No challenge explanation found", execution_time=time.time() - step_start, category=StepCategory.LLM_VLM_JUDGEMENT)
    except Exception as e:
        print(f"LLM failed to evaluate challenge: {e}")
        checkpoint.add_step("Explains Challenge", False, 1, f"LLM error while evaluating challenge: {e}", execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)

    # Step 2: Goal explanation (separate try/except).
    step_start = time.time()
    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant who evaluates whether the text describes at least one goal in choosing a tablet for a newly promoted corporate professional. Response with ONLY 'yes' or 'no'."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Is there at least one goal in choosing a tablet for a newly promoted corporate professional in this text?\n\nText: {slide_text}"}]
            }
        ]
        response = model(messages).strip().lower()
        goal_ok = 'yes' in response
        checkpoint.add_step("Explains Goal", goal_ok, 2, "Found goal explanation" if goal_ok else "No goal explanation found", execution_time=time.time() - step_start, category=StepCategory.LLM_VLM_JUDGEMENT)
    except Exception as e:
        print(f"LLM failed to evaluate goal: {e}")
        checkpoint.add_step("Explains Goal", False, 2, f"LLM error while evaluating goal: {e}", execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (5pt): All evaluation criteria are listed.

    Outcome Evaluation:
    - Client presentation capability found.
    - Battery life for long workdays and travel found.
    - Portability and weight for commuting found.
    - Performance for productivity and multitasking found.
    - Security and enterprise compatibility found.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global model
    
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Evaluation Criteria")
    categories = [
        "Client presentation capability",
        "Battery life for long workdays and travel",
        "Portability and weight for commuting",
        "Performance for productivity and multitasking",
        "Security and enterprise compatibility",
    ]
    if not presentation_data or 'slides' not in presentation_data:
        for i, name in enumerate(categories, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Slide 3 index 2
    slides = presentation_data['slides']
    if len(slides) < 3:
        for i, name in enumerate(categories, 1):
            checkpoint.add_step(name, False, i, "Criteria slide not found or not in the correct order", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    
    slide = slides[2]
    slide_text = extract_slide_text(slide)

    category_keyword_list = [
        ["Client presentation", "presentation capability", "presentation", "client-facing"],
        ["Battery life", "battery", "long workdays", "travel"],
        ["Portability", "weight", "commute", "commuting", "lightweight"],
        ["Performance", "productivity", "multitasking", "processor"],
        ["Security", "enterprise compatibility", "enterprise", "encryption"],
    ]
    
    if model is None:
        model = load_model(model_id)
    for i, category in enumerate(categories):
        step_start = time.time()
        is_valid, criteria_match_method = keywords_match_robust(slide_text, category_keyword_list[i], model=model, substring=True, return_method=True)
        checkpoint.add_step(category, bool(is_valid), i + 1, f"Found {category}" if bool(is_valid) else f"Missing {category}", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC if criteria_match_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint
        
def grade_checkpoint_4():
    """
    Checkpoint 4 (24pt): The tablet slides meet the requirements.

    Outcome Evaluation (x3 tablets, 8 pts each):
    - Title of the slide is the device name.
    - Each slide contains at least one source link.
    - Two product images from different angles found.
    - Key features and specifications section found.
    - Pros are listed.
    - Cons are listed.
    - Product images are from the source link(s) in the slide.
    - Product key features are accurate according to the sources."""
    print("----------------- CHECKPOINT 4 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=24, result=0, name="Device Slides")

    # Step names are grouped by emission phase. The success path emits these in
    # phase-major, device-major order (phase 1 for D1/D2/D3, then phase 2 for
    # D1/D2/D3, then phase 3 for D1/D2/D3), so the early-exit mirrors that
    # ordering to keep step_id -> step_name mapping consistent across paths.
    phase_1_steps = [
        "Device Name as Title",
        "Source Link(s) in Slide",
        "Product Images",
        "Product Images From Sources",
    ]
    phase_2_steps = ["Key Features", "Pros", "Cons"]
    phase_3_steps = ["Content From Sources"]
    num_devices = len(gold_devices) if gold_devices else 3

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        step_id = 1
        for phase_steps in (phase_1_steps, phase_2_steps, phase_3_steps):
            for i in range(num_devices):
                for name in phase_steps:
                    checkpoint.add_step(f"Device {i+1} - {name}", False, step_id, "No slides found in the presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
                    step_id += 1
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    if len(slides) < 6:
        step_id = 1
        for phase_steps in (phase_1_steps, phase_2_steps, phase_3_steps):
            for i in range(num_devices):
                for name in phase_steps:
                    checkpoint.add_step(f"Device {i+1} - {name}", False, step_id, "Device slides missing or not in the correct order", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
                    step_id += 1
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    section_validation_task = []

    expected_titles = gold_devices.copy()
    split_pattern = r'[;/|\n*#\t]+|\s{2,}'
    step_id = 1
    all_slides = []
    if model is None:
        model = load_model(model_id)

    image_file_map = {
        "iPad Pro (M4)": "ipad",
        "Samsung Galaxy Tab S10 Ultra": "samsung",
        "Microsoft Surface Pro 11": "surface",
    }

    def _parse_percentage(value):
        """Parse an LLM-returned percentage string into a float in [0, 100].

        Accepts forms like '85', '85%', '~85', 'about 85', '85.5'. Returns None
        when the value cannot be parsed (including the parallel_execute None
        sentinel for failed tasks).
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                v = float(value)
            except (TypeError, ValueError):
                return None
            return max(0.0, min(100.0, v))
        if isinstance(value, str):
            m = re.search(r"-?\d+(?:\.\d+)?", value)
            if not m:
                return None
            try:
                v = float(m.group(0))
            except ValueError:
                return None
            return max(0.0, min(100.0, v))
        return None

    print(f"1. Validate titles, source links, and images")
    for i in range(3):  # for each device slide
        print(f"    Device slide {i+1}")
        slide = slides[3 + i]

        slide_text = extract_slide_text(slide, "\n")
        slide_text_tokens = [part.strip() for part in re.split(split_pattern, slide_text) if part.strip()]

        # 1. Validate that title is the device name
        step_start = time.time()
        slide_title = extract_title_text(slide)
        print(f"        Checking that title is device name...")
        try:
            title_match, title_match_method = keywords_match_robust(expected_titles, slide_title, model=model, return_method=True)
            title_match_category = StepCategory.DETERMINISTIC if title_match_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT
        except Exception as e:
            print(f"        Title match failed: {e}")
            title_match = None
            title_match_category = StepCategory.EXECUTION_ERROR
        if not title_match:
            checkpoint.add_step(f"Device {i+1} - Device Name as Title", False, step_id, "The title is not the device name", execution_time=time.time() - step_start, category=title_match_category)
        else:
            checkpoint.add_step(f"Device {i+1} - Device Name as Title", True, step_id, f"The title is the device name ({title_match})", execution_time=time.time() - step_start, category=title_match_category)
            # Guard against normalized forms that aren't literally in the list.
            if title_match in expected_titles:
                expected_titles.remove(title_match)
        step_id += 1

        # 2. Validate that slide contains at least one source link
        print(f"        Checking that the slide provides at least 1 source...")
        step_start = time.time()
        try:
            slide_links = extract_slide_links(slide)
            slide_links = [ensure_scheme(link) for link in slide_links]
        except Exception as e:
            print(f"        Error extracting slide links: {e}")
            slide_links = []
        checkpoint.add_step(f"Device {i+1} - Source Link(s) in Slide", len(slide_links) > 0, step_id, "The slide contains at least one source" if len(slide_links) > 0 else "No source is found in slide", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
        step_id += 1

        # 3 & 4. Validate product images and source-image match. Both steps are
        # always recorded, even on exceptions, with their own try/finally so a
        # failure in the image flow doesn't drop later device steps.
        print(f"        Checking that images from slide match those in the gold folder...")
        step_start = time.time()

        # Always start each iteration with fresh temp dirs to avoid cross-iteration leakage.
        temp_dir = os.path.join(DATA_DIR, "temp_images")
        url_temp_dir = os.path.join(DATA_DIR, "temp_url_images")
        for d in (temp_dir, url_temp_dir):
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

        valid_images = False
        valid_images_detail = "Product images are missing or not from different angles"
        valid_images_category = None
        found_in_source = False
        found_in_source_detail = "No images in the slide was from the source"
        found_in_source_category = StepCategory.LLM_VLM_JUDGEMENT

        try:
            try:
                images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
            except Exception as e:
                print(f"        Error extracting slide images: {e}")
                images = []

            ref_image_folder = ""
            if title_match and title_match in image_file_map:
                ref_image_folder = image_file_map[title_match]

            slide_image_paths = []
            valid_image_count = 0
            image_from_different_angle = None

            if len(images) == 0:
                valid_images_detail = f"No images found on the slide"
                valid_images_category = StepCategory.DETERMINISTIC
            else:
                os.makedirs(temp_dir, exist_ok=True)
                try:
                    for idx, img_info in enumerate(images):
                        if not img_info.get('contentUrl'):
                            continue
                        try:
                            img = download_slide_image(img_info['contentUrl'])
                        except Exception as e:
                            print(f"        Failed to download slide image {idx}: {e}")
                            continue
                        if img:
                            temp_img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                            try:
                                img.save(temp_img_path)
                                slide_image_paths.append(temp_img_path)
                            except Exception as e:
                                print(f"        Failed to save slide image {idx}: {e}")

                    examples_path = os.path.join(GOLD_IMAGES_DIR, ref_image_folder) if ref_image_folder else None
                    if slide_image_paths and os.path.isdir(examples_path):
                        if len(slide_image_paths) == 1:
                            binary_judge_results = binary_judge_image(
                                model,
                                slide_image_paths[0],
                                "Does this image show exactly two views of the same or similar tablets as shown in the examples?",
                                examples_path,
                            )
                            if binary_judge_results:
                                valid_image_count = 2
                                image_from_different_angle = True
                        elif len(slide_image_paths) > 1:
                            binary_judge_tasks = [
                                {
                                    "id": idx,
                                    "func": binary_judge_image,
                                    "args": (model, img_path, "Is this an image of a tablet of the same or similar model as those in the examples?", examples_path),
                                }
                                for idx, img_path in enumerate(slide_image_paths)
                            ]
                            try:
                                binary_judge_results = parallel_execute(binary_judge_tasks)
                            except Exception as e:
                                print(f"        binary_judge_image parallel_execute failed: {e}")
                                binary_judge_results = {}
                            for j in binary_judge_results:
                                if valid_image_count == 2:
                                    break
                                if binary_judge_results[j]:
                                    valid_image_count += 1

                            # Different-angle check (move first slide image aside as the example reference).
                            temp_example_dir = os.path.join(DATA_DIR, "temp_example")
                            if os.path.exists(temp_example_dir):
                                shutil.rmtree(temp_example_dir, ignore_errors=True)
                            os.makedirs(temp_example_dir, exist_ok=True)
                            try:
                                if slide_image_paths:
                                    shutil.copy2(slide_image_paths[0], temp_example_dir)
                                    try:
                                        os.remove(slide_image_paths[0])
                                    except OSError:
                                        pass
                                try:
                                    image_from_different_angle = binary_judge_image(
                                        model,
                                        temp_dir,
                                        "Is this image showing the tablet from a different perspective or angle compared to the example?",
                                        temp_example_dir,
                                    )
                                except Exception as e:
                                    print(f"        Different-angle check failed: {e}")
                                    image_from_different_angle = None
                                for f in os.listdir(temp_example_dir):
                                    try:
                                        shutil.move(os.path.join(temp_example_dir, f), temp_dir)
                                    except Exception as e:
                                        print(f"        Could not restore example image: {e}")
                            finally:
                                if os.path.exists(temp_example_dir):
                                    shutil.rmtree(temp_example_dir, ignore_errors=True)
                    else:
                        valid_images_detail = "Could not download slide images or example folder is missing"
                        valid_images_category = StepCategory.EXECUTION_ERROR

                    valid_images = valid_image_count >= 2 and bool(image_from_different_angle)
                    if valid_images:
                        valid_images_detail = "Found 2 product images from 2 different angles"
                    elif valid_image_count < 2:
                        valid_images_detail = f"Only {valid_image_count}/2 slide images matched the tablet"
                    elif not image_from_different_angle:
                        valid_images_detail = "Slide images are not from different angles"
                except Exception as e:
                    print(f"        Error evaluating images: {e}")
                    valid_images_detail = f"Error evaluating images: {e}"
                    valid_images_category = StepCategory.EXECUTION_ERROR

            checkpoint.add_step(f"Device {i+1} - Product Images", valid_images, step_id, valid_images_detail, execution_time=time.time() - step_start, category=valid_images_category if valid_images_category else StepCategory.LLM_VLM_JUDGEMENT)
            step_id += 1

            # Source-image match check (must always record a step).
            print(f"        Checking that at least one image from the slide is from the given source...")
            step_start_src = time.time()
            try:
                os.makedirs(url_temp_dir, exist_ok=True)
                image_download_tasks = [
                    {
                        'id': link,
                        'func': download_images_from_url,
                        'args': (link, url_temp_dir),
                    }
                    for link in slide_links
                ]
                if image_download_tasks:
                    try:
                        parallel_execute(image_download_tasks)
                    except Exception as e:
                        print(f"        URL image download failed: {e}")

                slide_temp_dir_has_imgs = os.path.isdir(temp_dir) and bool(os.listdir(temp_dir))
                url_temp_has_imgs = os.path.isdir(url_temp_dir) and bool(os.listdir(url_temp_dir))

                if not slide_links:
                    found_in_source_detail = "No source links on slide; cannot verify"
                    found_in_source_category = StepCategory.EXECUTION_ERROR
                elif not url_temp_has_imgs:
                    found_in_source_detail = "Could not retrieve any images from the source links"
                    found_in_source_category = StepCategory.EXECUTION_ERROR
                elif not slide_temp_dir_has_imgs:
                    found_in_source_detail = "No slide images available to compare against source"
                    found_in_source_category = StepCategory.EXECUTION_ERROR
                else:
                    try:
                        matching_image = binary_judge_image(
                            model,
                            url_temp_dir,
                            "Is this an image of a tablet of the same or similar model as the examples?",
                            temp_dir,
                        )
                    except Exception as e:
                        print(f"        Source-image LLM check failed: {e}")
                        matching_image = None
                        found_in_source_detail = f"Source-image check failed: {e}"
                        found_in_source_category = StepCategory.EXECUTION_ERROR
                    if matching_image:
                        found_in_source = True
                        found_in_source_detail = "At least one image in the slide was found in the source"
            except Exception as e:
                print(f"        Unexpected error in source-image flow: {e}")
                found_in_source_detail = f"Unexpected error: {e}"
                found_in_source_category = StepCategory.EXECUTION_ERROR

            checkpoint.add_step(f"Device {i+1} - Product Images From Sources", found_in_source, step_id, found_in_source_detail, execution_time=time.time() - step_start_src, category=found_in_source_category)
            step_id += 1
        finally:
            for d in (temp_dir, url_temp_dir):
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)

        all_slides.append({
            "title": slide_title,
            "links": slide_links,
            "text_tokens": slide_text_tokens,
        })

    # Validate key features, pros, and cons sections
    for i, slide in enumerate(all_slides):
        slide_title = slide["title"]
        slide_text = "\n".join(slide["text_tokens"])
        task_text = f"""Extract the content for key features, pros, and cons of a tablet from the given slide text.

Respond ONLY with this exact JSON format:

{{
    "key_features": "<semicolon-separated points>",
    "pros": "<semicolon-separated points>",
    "cons": "<semicolon-separated points>"
}}

If no relevant information is found for a certain section, still include it in the response with an empty string as value.

Slide text:

{slide_text}
"""
        section_validation_task.append({
            'id': f"slide_{i}",
            'func': extract_device_info_with_llm,
            'args': (task_text, model),
        })

    print(f"2. Validating features, pros, and cons")
    print(f"    Extracting all slide section contents...")
    step_start = time.time()
    try:
        section_validation_results = parallel_execute(section_validation_task, max_workers=5) or {}
    except Exception as e:
        print(f"    parallel_execute for section validation failed: {e}")
        section_validation_results = {}
    print(f"    Finished extracting section content in {time.time()-step_start}")

    for i, slide in enumerate(all_slides):
        step_start = time.time()
        section_content = section_validation_results.get(f"slide_{i}")
        if not isinstance(section_content, dict):
            section_content = {"key_features": "", "pros": "", "cons": ""}

        has_key_features = bool(section_content.get("key_features"))
        checkpoint.add_step(f"Device {i+1} - Key Features", has_key_features, step_id,
                            f"Key features found for device {i+1}" if has_key_features else f"Missing key features for device {i+1}",
                            execution_time=time.time() - step_start,
                            category=StepCategory.LLM_VLM_JUDGEMENT)
        step_id += 1

        step_start = time.time()
        has_pros = bool(section_content.get("pros"))
        checkpoint.add_step(f"Device {i+1} - Pros", has_pros, step_id,
                            f"Pros found for device {i+1}" if has_pros else f"Missing pros for device {i+1}",
                            execution_time=time.time() - step_start,
                            category=StepCategory.LLM_VLM_JUDGEMENT)
        step_id += 1

        step_start = time.time()
        has_cons = bool(section_content.get("cons"))
        checkpoint.add_step(f"Device {i+1} - Cons", has_cons, step_id,
                            f"Cons found for device {i+1}" if has_cons else f"Missing cons for device {i+1}",
                            execution_time=time.time() - step_start,
                            category=StepCategory.LLM_VLM_JUDGEMENT)
        step_id += 1

        slide["key_features"] = section_content.get("key_features", "")

    # Validate that information is pulled from links
    print(f"3. Verifying that information comes from given source")
    match_threshold = 70
    verifying_task = []
    for i, slide in enumerate(all_slides):
        slide_links = slide["links"]
        features = slide.get("key_features", "")

        url_fetch_tasks = [
            {
                'id': link,
                'func': fetch_url_content,
                'args': (link,),
            }
            for link in slide_links
        ]

        fetched_contents = []
        if url_fetch_tasks:
            print(f"    Downloading web content for Device {i+1}...")
            start_time = time.time()
            try:
                fetch_results = parallel_download(url_fetch_tasks, max_workers=3, use_rate_limit=False) or {}
            except Exception as e:
                print(f"    parallel_download failed for Device {i+1}: {e}")
                fetch_results = {}
            print(f"    Finished downloading web content in {time.time()-start_time}")
            for _url, content in fetch_results.items():
                if content:
                    fetched_contents.append("\n".join([part.strip() for part in re.split(split_pattern, content) if part.strip()]))
        fetched_text = "\n".join(fetched_contents)
        if features and fetched_text:
            task_text = f"""Evaluate how much the following information is supported by the source.

Respond in a single number between 0 and 100.

Information:
{features}

Content:
{fetched_text}
"""
            verifying_task.append({
                'id': f"slide_{i}",
                'func': evaluate_device_info_with_llm,
                'args': (task_text, model, "str"),
            })

    try:
        verifying_results = parallel_execute(verifying_task, max_workers=3) or {}
    except Exception as e:
        print(f"    parallel_execute for source verification failed: {e}")
        verifying_results = {}

    # Iterate slides in order so the "Device i" label aligns with prior CP4 steps,
    # regardless of the LLM completion order.
    for i, slide in enumerate(all_slides):
        step_start = time.time()
        raw_value = verifying_results.get(f"slide_{i}", None)
        if not raw_value:
            print(f"        Device {i+1}: No source content found.")
            checkpoint.add_step(
                f"Device {i+1} - Content From Sources",
                False,
                step_id,
                "No source content found",
                execution_time=time.time() - step_start,
                category=StepCategory.EXECUTION_ERROR,
            )
        else:
            match_percentage = _parse_percentage(raw_value)
            if match_percentage is None:
                checkpoint.add_step(
                    f"Device {i+1} - Content From Sources",
                    False,
                    step_id,
                    f"Could not parse LLM percentage response: {raw_value!r}",
                    execution_time=time.time() - step_start,
                    category=StepCategory.EXECUTION_ERROR,
                )
            else:
                print(f"        Device {i+1}: {match_percentage:.2f}% of slide text found in sources.")
                success = match_percentage >= match_threshold
                checkpoint.add_step(
                    f"Device {i+1} - Content From Sources",
                    success,
                    step_id,
                    f"{match_percentage:.1f}% of listed features found in sources" if success else f"Only {match_percentage:.1f}% of listed features is from sources",
                    execution_time=time.time() - step_start,
                    category=StepCategory.LLM_VLM_JUDGEMENT,
                )
        step_id += 1

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_5():
    """
    Checkpoint 5 (13pt): Side-by-side comparison meets the requirements.

    Outcome Evaluation:

    - Table has exactly 3 columns.
    - All devices are included as column headers.
    - Three colors red, yellow, and green are used for the coding scheme in the table content.
    - Client presentation capability is covered.
    - An appropriate color applied for each cell under Client presentation capability.
    - Battery life for long workdays and travel is covered.
    - An appropriate color applied for each cell under Battery life for long workdays and travel.
    - Portability and weight for commuting is covered.
    - An appropriate color applied for each cell under Portability and weight for commuting.
    - Performance for productivity and multitasking is covered.
    - An appropriate color applied for each cell under Performance for productivity and multitasking.
    - Security and enterprise compatibility is covered.
    - An appropriate color applied for each cell under Security and enterprise compatibility.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global model
    if model is None:
        model = load_model(model_id)
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=13, result=0, name="Comparison Slide")
    
    # Step names ordered to match the success-path emission order: first the two
    # structural checks, then all five "Found <category> Row" steps, then the
    # color-scheme check, then all five per-category "Correct Color Coding"
    # steps. Categories use lowercase to match the canonical `categories` list
    # used later in the success path for keyword matching.
    comparison_step_names = [
        "Table Has 3 Columns",
        "All Tablets as Headers",
        "Found client presentation capability Row",
        "Found battery life Row",
        "Found portability and weight Row",
        "Found performance Row",
        "Found security and enterprise compatibility Row",
        "Green, Yellow, and Red as Color Coding Scheme",
        "client presentation capability - Correct Color Coding",
        "battery life - Correct Color Coding",
        "portability and weight - Correct Color Coding",
        "performance - Correct Color Coding",
        "security and enterprise compatibility - Correct Color Coding",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(comparison_step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    if len(slides) < 7:
        for i, name in enumerate(comparison_step_names, 1):
            checkpoint.add_step(name, False, i, "Comparison slide missing or not in the correct order", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    step_start = time.time()
    slide = slides[6]
    
    # Extract table from slide
    table_data = extract_table_from_slide(slide)
    
    if not table_data:
        for i, name in enumerate(comparison_step_names, 1):
            checkpoint.add_step(name, False, i, "No table found in the slide", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    # Step 1: Verify table has exactly 3 columns
    step_id = 1
    step_start = time.time()
    has_3_columns = table_data['num_columns'] == 3
    checkpoint.add_step("Table Has 3 Columns", has_3_columns, step_id, 
                       "Table has exactly 3 columns" if has_3_columns else f"Table has {table_data['num_columns']} columns, expected 3",
                       execution_time=time.time() - step_start,
                       category=StepCategory.STRUCTURAL)
    step_id += 1
    
    # Step 2: Verify all tablets are column headers
    step_start = time.time()
    headers = table_data.get('headers', [])
    missing_devices = ""
    print(f"1. Validating table headers.")
    for device in gold_devices:
        header_match = keywords_exact_match(device, headers, substring=True)
        if not bool(header_match):
            missing_devices += f"{device}; " 
    headers_valid = len(missing_devices) == 0
    checkpoint.add_step("All Tablets as Headers", headers_valid, step_id,
                       "All tablets found as column headers" if headers_valid else f"Missing header(s) for the following tablets: {missing_devices}",
                       execution_time=time.time() - step_start,
                       category=StepCategory.DETERMINISTIC)
    step_id += 1
    
    print(f"2. Validating category coverage based on the table content.")
    categories = [
        "client presentation capability",
        "battery life",
        "portability and weight",
        "performance",
        "security and enterprise compatibility",
    ]
    color_rank_map = {
        'green': 1,
        'yellow': 2,
        'red': 3
    }
    rows_content = ""
    print(f"    Extracting table content for LLM evaluation...")
    for row in table_data.get('rows', []):
        rows_content += "; ".join([row[device] for device in headers]) + "\n"
    
    task_text = f"""Extract each row in the given table content and assign it to the following categories.
            
IMPORTANT: Each line represents one table row, with each semicolon-separated value corresponding to the devices {", ".join(headers)}, respectively.
Each line should be evaluated to one category ONLY.
                        
Respond ONLY with this exact JSON format:

{{
    "<Category Name>": {{
        "<Device Name>": "<Cell Content>",
    }}    
}}


If the line contains mixed-category information, evaluate based on the most prominent category or the category that best fits the overall content of the line.
The order of the returned categories must correspond to the order of the content in the table.

Categories:
{"; ".join(categories)}

Table Content:
{rows_content}
"""
    print(f"    Sending table content to LLM for category extraction...")
    step_start = time.time()
    category_extraction_failed = False
    try:
        category_map_raw = evaluate_device_info_with_llm(task_text, model, return_type="json")
    except Exception as e:
        print(f"    LLM category extraction failed: {e}")
        category_map_raw = None
    if not isinstance(category_map_raw, dict):
        category_map_raw = {}
        category_extraction_failed = True
    print(f"    LLM finished category extraction in {time.time() - step_start:.2f} seconds.")

    # Normalize category keys (lowercased, stripped) for robust matching against
    # our canonical category list.
    def _norm(s):
        return s.strip().lower() if isinstance(s, str) else ""

    canonical_to_raw = {}
    for raw_key in category_map_raw.keys():
        norm_key = _norm(raw_key)
        for cat in categories:
            if cat in norm_key or norm_key in cat:
                canonical_to_raw.setdefault(cat, raw_key)
                break

    llm_ranking_task = []
    ranking_from_table = {}
    colors_used = set()

    print(f"    Starting category coverage validation and preparing LLM ranking tasks...")
    for category in categories:
        step_start = time.time()
        raw_key = canonical_to_raw.get(category)
        cells_dict = category_map_raw.get(raw_key) if raw_key else None
        # Ensure cells_dict is a {device: cell_text} mapping. If LLM returns a
        # list/string we recover what we can; otherwise treat as missing.
        if isinstance(cells_dict, dict):
            cells_values = [str(v) if v is not None else "" for v in cells_dict.values()]
        elif isinstance(cells_dict, list):
            cells_values = [str(v) if v is not None else "" for v in cells_dict]
        elif isinstance(cells_dict, str) and cells_dict:
            cells_values = [cells_dict]
        else:
            cells_values = []

        if cells_values:
            print(f"        Found category '{category}' in table content. Creating LLM ranking task for this category.")
            checkpoint.add_step(f"Found {category} Row", True, step_id, f"The {category} category is covered in table", execution_time=time.time() - step_start, category=StepCategory.LLM_VLM_JUDGEMENT)
            comparison_content = "\n".join(cells_values)
            ranking_task_text = f"""Given the information for these devices {", ".join(headers)}, respectively, rank them numerically, based on the category {category}, from best to worst, where 1 is best and {len(headers)} is worst.
IMPORTANT:

Values may contains both numerical and qualitative information. When two values have the same numerical ranking, use the qualitative information to determine if they should be ranked the same or if one is better than the other.

Respond with ONLY the rankings in the following JSON format:
{{
"<Device Name>": <Rank>,
}}

Two values can be ranked the same if they are very close or identical.
'Unknown' values are always ranked worst.

Values:

{comparison_content}
"""
            llm_ranking_task.append({
                'id': f'{category}',
                'func': evaluate_device_info_with_llm,
                'args': (ranking_task_text, model, "json"),
            })
        else:
            print(f"        No clear information about category '{category}' found in table content.")
            checkpoint.add_step(f"Found {category} Row", False, step_id, f"No clear information about {category} found in table", execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR if category_extraction_failed else StepCategory.LLM_VLM_JUDGEMENT)
        step_id += 1

    # Color coding extraction: locate each category's row in the table by header
    # text rather than relying on positional index from the LLM response.
    print(f"3a. Extracting color coding scheme from table for validation...")
    step_start = time.time()
    cell_colors = table_data.get('cell_colors', {}) or {}
    table_rows = table_data.get('rows', []) or []

    def _row_text_for(row):
        # Concatenate cell text for category-keyword matching against the row.
        return " ".join(row.get(h, "") for h in headers)

    # Map each canonical category to a row index (0-based among data rows, header excluded).
    category_row_idx = {}
    for cat in categories:
        for r_idx, row in enumerate(table_rows):
            if cat in _row_text_for(row).lower():
                category_row_idx[cat] = r_idx
                break

    for category in categories:
        ranking_from_table[category] = {}
        r_idx = category_row_idx.get(category)
        if r_idx is None:
            continue
        for dev_idx, device in enumerate(headers):
            try:
                color = cell_colors.get((r_idx + 1, dev_idx))
            except Exception as e:
                print(f"    Color lookup failed for {category}/{device}: {e}")
                color = None
            if color is None:
                colors_used.add('unknown')
                ranking_from_table[category][device] = -1
                continue
            try:
                color_str = detect_color_name(color)
            except Exception as e:
                print(f"    detect_color_name failed: {e}")
                color_str = 'unknown'
            colors_used.add(color_str)
            ranking_from_table[category][device] = color_rank_map.get(color_str, -1)

    print(f"    Validating that the table uses Green, Yellow, and Red as the color coding scheme...")
    expected_colors = {'red', 'yellow', 'green'}
    distinct_colors = colors_used - {'unknown'}
    if 'unknown' in colors_used and not distinct_colors.issuperset(expected_colors):
        checkpoint.add_step("Green, Yellow, and Red as Color Coding Scheme", False, step_id, f"Unknown colors found in table: {', '.join(sorted(colors_used))}", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
    elif distinct_colors != expected_colors:
        checkpoint.add_step("Green, Yellow, and Red as Color Coding Scheme", False, step_id, f"Color set does not match required scheme. Colors found: {', '.join(sorted(distinct_colors))}", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
    else:
        checkpoint.add_step("Green, Yellow, and Red as Color Coding Scheme", True, step_id, f"All required colors are used: {', '.join(sorted(distinct_colors))}", execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
    step_id += 1

    start_time = time.time()
    print(f"3b. Validating that the table correctly ranks the devices for each category based on the color coding scheme...")
    llm_ranking_results = {}
    if llm_ranking_task:
        try:
            llm_ranking_results = parallel_execute(llm_ranking_task, max_workers=max(1, len(llm_ranking_task))) or {}
        except Exception as e:
            print(f"    parallel_execute for rankings failed: {e}")
            llm_ranking_results = {}
        print(f"    LLM finished ranking tasks in {time.time() - start_time:.2f} seconds.")

    for category in categories:
        start_time = time.time()
        llm_ranking = llm_ranking_results.get(category)
        table_ranking = ranking_from_table.get(category, {})
        if not isinstance(llm_ranking, dict) or not llm_ranking:
            checkpoint.add_step(f"{category} - Correct Color Coding", False, step_id, f"LLM failed to rank devices for {category}", execution_time=time.time() - start_time, category=StepCategory.EXECUTION_ERROR)
        elif not table_ranking:
            checkpoint.add_step(f"{category} - Correct Color Coding", False, step_id, f"No table row located for {category}", execution_time=time.time() - start_time, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
        else:
            try:
                ranking_consistent = validate_rankings(llm_ranking, table_ranking)
            except Exception as e:
                print(f"    validate_rankings failed for {category}: {e}")
                ranking_consistent = False
            checkpoint.add_step(
                f"{category} - Correct Color Coding",
                bool(ranking_consistent),
                step_id,
                f"Appropriate colors are used to rank values from best to worst for {category}" if ranking_consistent else f"Colors are not correctly assigned for {category}",
                execution_time=time.time() - start_time,
                category=StepCategory.LLM_VLM_JUDGEMENT,
            )
        step_id += 1

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_6():
    """
    Checkpoint 6 (3pt): Recommendation slide meets all requirements.

    Outcome Evaluation:
    - All three tablet options found in the slide.
    - Summaries align with the comparison data.
    - Recommendations based on different professional types or workplace needs are provided.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Recommendation Slide")

    recommendation_step_names = [
        "All Tablets Mentioned",
        "Summaries Align with Comparison Data",
        "Recommendations Based on Professional Types",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(recommendation_step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    if len(slides) < 8:
        for i, name in enumerate(recommendation_step_names, 1):
            checkpoint.add_step(name, False, i, "Recommendation slide missing or not in the correct order", execution_time=time.time() - checkpoint_start, category=StepCategory.STRUCTURAL)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_start = time.time()
    slide = slides[7]
    try:
        slide_text = extract_slide_text(slide)
    except Exception as e:
        print(f"Failed to extract slide text for recommendation slide: {e}")
        slide_text = ""

    if model is None:
        model = load_model(model_id)

    task_text = f"""Extract the following tablet summaries and recommendations from this Google slide text.

IMPORTANT: This text may contain multiple tablets or none at all.
Extract the information for EACH tablet separately.

Respond ONLY with this exact JSON format:
{{
  "summary": [["<device_name>", "<summary_text>"], ...],
  "recommendation": [["<device_name>", "<recommendation_text>"], ...]
}}

If the summary or recommendation for a device is not found, use an empty string for that field.
If there is NO device, still return an object with the two properties set to empty arrays.

Slide text:
{slide_text}"""

    try:
        device_data_raw = extract_device_info_with_llm(task_text, model)
    except Exception as e:
        print(f"LLM device extraction failed: {e}")
        device_data_raw = None
    if not isinstance(device_data_raw, dict):
        device_data_raw = {}

    summary_payload = device_data_raw.get("summary") if isinstance(device_data_raw.get("summary"), list) else []
    recommendation_payload = device_data_raw.get("recommendation") if isinstance(device_data_raw.get("recommendation"), list) else []

    def _normalize_pair_list(items):
        """Coerce a list whose items may be [name, text] pairs or {name, text} dicts
        into a uniform list of (name, text) tuples; ignores malformed entries.
        """
        out = []
        for entry in items:
            if isinstance(entry, (list, tuple)):
                name = entry[0] if len(entry) > 0 else ""
                text = entry[1] if len(entry) > 1 else ""
            elif isinstance(entry, dict):
                # Accept common shapes like {"device_name": ..., "summary": ...} or {"name": ..., "text": ...}.
                name = (
                    entry.get("device_name")
                    or entry.get("name")
                    or entry.get("device")
                    or ""
                )
                text = (
                    entry.get("summary_text")
                    or entry.get("recommendation_text")
                    or entry.get("summary")
                    or entry.get("recommendation")
                    or entry.get("text")
                    or ""
                )
            else:
                continue
            out.append((str(name), str(text) if text is not None else ""))
        return out

    summaries = _normalize_pair_list(summary_payload)
    recommendations = _normalize_pair_list(recommendation_payload)

    device_map = {}
    device_names = []
    for name, text in summaries:
        device_names.append(name)
        device_map.setdefault(name, {"summary": "", "recommendation": ""})
        device_map[name]["summary"] = text
    for name, text in recommendations:
        if name not in device_map:
            device_map[name] = {"summary": "", "recommendation": ""}
            device_names.append(name)
        device_map[name]["recommendation"] = text

    missing_devices = ""
    device_mention_items = []
    print(f"1. Validating devices:")
    for device in gold_devices:
        # Match the gold device against the LLM-extracted device names using
        # the full device string with substring matching (avoids false positives
        # from splitting into single tokens like 'M4' that span multiple devices).
        try:
            match, mention_method = keywords_match_robust(device_names, device, model=model, substring=True, return_method=True)
            mention_category = StepCategory.DETERMINISTIC if mention_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT
        except Exception as e:
            print(f"    Device match failed for {device}: {e}")
            match = None
            mention_category = StepCategory.EXECUTION_ERROR
        device_mention_items.append((mention_category, bool(match)))
        if not match:
            missing_devices += device + "; "
            print(f"    Missing device: {device}")
        else:
            print(f"    Found device: {device}")

    checkpoint.add_step("All Tablets Mentioned", len(missing_devices) == 0, 1, "All three correct tablets discussed in the slide" if len(missing_devices) == 0 else f"Missing information for: {missing_devices}", execution_time=time.time() - step_start, category=StepCategory.aggregate(device_mention_items))

    # Step 2 (Summaries) and Step 3 (Recommendations) - prepared inside their own
    # try/except so any failure still records a step.
    step_start = time.time()
    valid_summaries = False
    summaries_detail = "No summary tasks were executed"
    summaries_category = StepCategory.LLM_VLM_JUDGEMENT
    valid_recommendations = False
    recommendations_detail = "No recommendation tasks were executed"
    recommendations_category = StepCategory.LLM_VLM_JUDGEMENT
    summary_tasks = []
    recommendation_tasks = []
    missing_sum = 0
    missing_rec = 0

    try:
        comparison_slide = slides[6]
        try:
            comparison_table = extract_table_from_slide(comparison_slide)
        except Exception as e:
            print(f"Failed to extract comparison table: {e}")
            comparison_table = None

        print(f"2. Collecting Summaries and Recommendations for Evaluation Tasks...")
        for device, info in device_map.items():
            summary = info.get("summary", "")
            recommendation = info.get("recommendation", "")
            comparison_text = ""
            if comparison_table:
                headers = comparison_table.get('headers', [])
                try:
                    matched_header = keywords_match_robust(headers, device, model=model, substring=True)
                except Exception as e:
                    print(f"    Header match failed for {device}: {e}")
                    matched_header = None
                if matched_header:
                    column_values = [row.get(matched_header, "") for row in comparison_table.get('rows', []) if matched_header in row]
                    comparison_text = "\n".join(column_values)
            if summary:
                summary_tasks.append({
                    'id': f'{device}',
                    'func': evaluate_device_info_with_llm,
                    'args': (f"Is the following summary for {device} consistent with the source information?\n\nSource: {comparison_text}\n\nSummary: {summary}", model),
                })
            else:
                missing_sum += 1
                print(f"    Missing summary for {device}")

            if recommendation:
                recommendation_tasks.append({
                    'id': f'{device}',
                    'func': evaluate_device_info_with_llm,
                    'args': (f"Is the following recommendation of {device} based on a professional type or workplace need?\n\nRecommendation: {recommendation}", model),
                })
            else:
                missing_rec += 1
                print(f"    Missing recommendation for {device}")

        print(f"3. Evaluating Summaries:")
        invalid_summaries = 0
        if summary_tasks:
            try:
                summary_eval_results = parallel_execute(summary_tasks, max_workers=3) or {}
            except Exception as e:
                print(f"    Summary parallel_execute failed: {e}")
                summary_eval_results = {}
            for device, isValid in summary_eval_results.items():
                if isValid:
                    print(f"    Summary for {device} is consistent with source information.")
                else:
                    invalid_summaries += 1
                    print(f"    Summary for {device} is NOT consistent with source information.")
        # Require all gold devices to have a non-empty, consistent summary.
        valid_summaries = invalid_summaries == 0 and missing_sum == 0 and len(summary_tasks) >= len(gold_devices)
        if valid_summaries:
            summaries_detail = "All summaries are consistent with source information"
        else:
            summaries_detail = "Some device summaries are inconsistent with the source information or missing"
    except Exception as e:
        print(f"Unexpected error evaluating summaries: {e}")
        summaries_detail = f"Unexpected error evaluating summaries: {e}"
        summaries_category = StepCategory.EXECUTION_ERROR

    checkpoint.add_step("Summaries Align with Comparison Data", valid_summaries, 2, summaries_detail, execution_time=time.time() - step_start, category=summaries_category)

    step_start = time.time()
    try:
        print(f"4. Evaluating Recommendations:")
        invalid_recommendations = 0
        if recommendation_tasks:
            try:
                rec_eval_results = parallel_execute(recommendation_tasks, max_workers=3) or {}
            except Exception as e:
                print(f"    Recommendation parallel_execute failed: {e}")
                rec_eval_results = {}
            for device, isValid in rec_eval_results.items():
                if isValid:
                    print(f"    Recommendation for {device} is based on professional type/workplace need.")
                else:
                    invalid_recommendations += 1
                    print(f"    Recommendation for {device} is not based on professional type/workplace need.")
        valid_recommendations = invalid_recommendations == 0 and missing_rec == 0 and len(recommendation_tasks) >= len(gold_devices)
        if valid_recommendations:
            recommendations_detail = "All recommendations are valid"
        else:
            recommendations_detail = "Some device recommendations are not valid or missing"
    except Exception as e:
        print(f"Unexpected error evaluating recommendations: {e}")
        recommendations_detail = f"Unexpected error evaluating recommendations: {e}"
        recommendations_category = StepCategory.EXECUTION_ERROR

    checkpoint.add_step("Recommendations Based on Professional Types", valid_recommendations, 3, recommendations_detail, execution_time=time.time() - step_start, category=recommendations_category)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_7():
    """
    Checkpoint 7 (4pt): The overall structure follows the specified task requirements
    and the deck addresses the stated audience and tone.

    Outcome Evaluation:
    - Exactly 8 slides found.
    - All slides follow the required sequence and titles.
    - Tone is professional/confident/easy-to-understand (per task brief).
    - Content is tailored to the newly promoted corporate professional audience and the colleague decision-maker.
    """
    print("----------------- CHECKPOINT 7 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Overall Structure")

    step_names = [
        "Exactly 8 Slides",
        "Correct Titles and Order",
        "Tone Is Professional and Confident",
        "Audience-Tailored Content",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_start = time.time()
    slides = presentation_data.get('slides', [])

    if len(slides) < 8:
        checkpoint.add_step("Exactly 8 Slides", False, 1, f"Found only {len(slides)}/8 slides", execution_time=time.time() - step_start, category=StepCategory.STRUCTURAL)
    else:
        checkpoint.add_step("Exactly 8 Slides", True, 1, "Found exactly 8 slides", execution_time=time.time() - step_start, category=StepCategory.STRUCTURAL)

    expected_titles_keywords = [
        ["A Gift for James!"],
        ["challenge", "goal"],
        ["evaluation", "criteria", "considerations", "factors", "judge"],
        [],
        [],
        [],
        ["comparison", "side by side", "side-by-side"],
        ["recommendation", "suggestion", "advice", "which", "best"],
    ]

    if model is None:
        model = load_model(model_id)

    # Step 2: Correct titles and order. Collect failures rather than returning early
    # so device-title checks always contribute to the result.
    step_start = time.time()
    failures = []
    title_check_items = []
    for i in chain(range(3), range(6, 8)):
        item_category = None
        try:
            title_text = extract_title_text(slides[i])
        except Exception as e:
            print(f"    Failed to extract title from slide {i+1}: {e}")
            title_text = ""
            item_category = StepCategory.EXECUTION_ERROR
        if not title_text:
            failures.append(f"slide {i+1}: missing title")
            title_check_items.append((item_category if item_category else StepCategory.DETERMINISTIC, False))
            continue
        try:
            title_match, title_method = keywords_match_robust(title_text, expected_titles_keywords[i], model=model, substring=True, return_method=True)
            item_category = StepCategory.DETERMINISTIC if title_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT
        except Exception as e:
            print(f"    Title match failed for slide {i+1}: {e}")
            title_match = None
            item_category = StepCategory.EXECUTION_ERROR
        title_check_items.append((item_category, bool(title_match)))
        if not title_match:
            failures.append(f"slide {i+1}: title does not match expected keywords")

    expected_devices = gold_devices.copy()
    for i in range(3, 6):
        item_category = None
        try:
            title_text = extract_title_text(slides[i])
        except Exception as e:
            print(f"    Failed to extract device-slide title {i+1}: {e}")
            title_text = ""
            item_category = StepCategory.EXECUTION_ERROR
        if title_text:
            try:
                device_match, device_method = keywords_match_robust(expected_devices, title_text, model=model, substring=True, description="The same device", return_method=True)
                item_category = StepCategory.DETERMINISTIC if device_method == "exact" else StepCategory.LLM_VLM_JUDGEMENT
            except Exception as e:
                print(f"    Device title match failed for slide {i+1}: {e}")
                device_match = None
                item_category = StepCategory.EXECUTION_ERROR
            if device_match and device_match in expected_devices:
                expected_devices.remove(device_match)
                title_check_items.append((item_category, True))
            else:
                title_check_items.append((item_category, False))
        else:
            title_check_items.append((item_category if item_category else StepCategory.DETERMINISTIC, False))
    if expected_devices:
        failures.append(f"missing device titles: {', '.join(expected_devices)}")

    titles_ok = not failures
    checkpoint.add_step(
        "Correct Titles and Order",
        titles_ok,
        2,
        "Slide titles and order are correct." if titles_ok else "; ".join(failures),
        execution_time=time.time() - step_start,
        category=StepCategory.aggregate(title_check_items),
    )

    # Step 3: Tone (exciting/supportive/easy-to-understand) via LLM check on the full deck text.
    step_start = time.time()
    deck_text_parts = []
    for s in slides:
        t = None
        try:
            title = extract_title_text(s)
            if title != expected_titles_keywords[0][0]:    
                t = extract_slide_text(s)
        except Exception as e:
            print(f"    Failed to read slide text: {e}")
            t = ""
        if t:
            deck_text_parts.append(t)
    deck_text = "\n\n".join(deck_text_parts) if len(deck_text_parts) > 0 else ""

    tone_ok = False
    tone_detail = "Could not evaluate tone"
    tone_category = StepCategory.EXECUTION_ERROR
    if deck_text:
        try:
            tone_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You evaluate presentation deck text. Given the deck text, judge whether the overall tone is professional, confident, and easy to understand (avoiding dry, overly technical, or jargon-heavy language). Respond ONLY with 'yes' or 'no'."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Is the tone of this deck professional, confident, and easy to understand?\n\nDeck text:\n{deck_text}"}],
                },
            ]
            tone_response = model(tone_messages).strip().lower()
            tone_ok = 'yes' in tone_response
            tone_detail = "Tone is professional/confident/easy to understand" if tone_ok else "Tone is not professional/confident/easy to understand"
            tone_category = StepCategory.LLM_VLM_JUDGEMENT
        except Exception as e:
            print(f"    Tone LLM check failed: {e}")
            tone_detail = f"Tone check failed: {e}"
    else:
        tone_detail = "No text found in the deck to evaluate tone"
    checkpoint.add_step("Tone Is Professional and Confident", tone_ok, 3, tone_detail, execution_time=time.time() - step_start, category=tone_category)

    # Step 4: Audience-tailoring (newly promoted corporate professional; colleague decision-maker).
    step_start = time.time()
    audience_ok = False
    audience_detail = "Could not evaluate audience tailoring"
    audience_category = StepCategory.EXECUTION_ERROR
    if deck_text:
        try:
            audience_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You evaluate presentation deck text. Given the deck text, judge whether the content is clearly tailored to a newly promoted corporate professional entering a client-facing role and useful to a colleague making the purchase decision (e.g. references corporate/workplace life, client meetings, travel, business-relatable examples, practical considerations for both audiences). Respond ONLY with 'yes' or 'no'."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Is the content tailored to a newly promoted corporate professional and helpful to a colleague making the purchase decision?\n\nDeck text:\n{deck_text}"}],
                },
            ]
            audience_response = model(audience_messages).strip().lower()
            audience_ok = 'yes' in audience_response
            audience_detail = "Content is tailored to the stated audience" if audience_ok else "Content does not adequately address the stated audience"
            audience_category = StepCategory.LLM_VLM_JUDGEMENT
        except Exception as e:
            print(f"    Audience LLM check failed: {e}")
            audience_detail = f"Audience check failed: {e}"
    else:
        audience_detail = "No text found in the deck to evaluate audience tailoring"
    checkpoint.add_step("Audience-Tailored Content", audience_ok, 4, audience_detail, execution_time=time.time() - step_start, category=audience_category)

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
        failed.add_step("Evaluation", False, 1, f"Fatal error: {e}", execution_time=0, category=StepCategory.EXECUTION_ERROR)
        return Result([failed], total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate personal product comparison presentation")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Slides presentation ID to evaluate")
    # parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    args = parser.parse_args()

    step_start = time.time()

    print(f"DEBUG mode: {DEBUG}")
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        cached_models=args.cached_models,
        # browsing_history=args.browsing_history
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "✓" if step["success"] else "X"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - step_start:.2f} seconds")