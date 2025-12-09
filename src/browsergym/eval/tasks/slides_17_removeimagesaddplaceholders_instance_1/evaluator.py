#!/usr/bin/env python3
"""
Evaluator for slides_17_removeimagesaddplaceholders task.

This evaluator checks:
1. All original images are saved to the specified Drive folder
2. Images are deleted and replaced with red text description placeholders
3. New images from web are placed with URL attribution, covering the placeholders
"""

import os
import sys
import json
import csv
import time
import argparse
import tempfile
import shutil
from typing import List, Dict, Any, Tuple

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

# Imports
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    list_drive_folder_files,
    download_drive_file_as_image
)
from src.browsergym.eval.eval_utils.slides_utils import (
    extract_slide_images,
    download_slide_image,
    extract_slide_links,
    extract_text_boxes_from_slide,
    get_text_style_from_shape,
    is_text_red,
    is_text_big
)
from src.browsergym.eval.eval_utils.image_utils import match_image_tiered, binary_judge_image
from src.browsergym.eval.eval_utils.text_utils import text_fuzzy_match_contained_short
from src.browsergym.eval.eval_utils.utils import is_bbox_mostly_inside
from src.browsergym.eval.eval_utils.models import load_model

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/slides_17_removeimagesaddplaceholders_instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
GOLD_IMAGES_DIR = os.path.join(DATA_DIR, "gold_images/")
GOLD_DESCRIPTIONS_CSV = os.path.join(DATA_DIR, "gold_descriptions.csv")
ORIGINAL_LOCATIONS_JSON = os.path.join(DATA_DIR, "original_image_locations.json")
DRIVE_FOLDER_ID = "19hN98W-JWHjwpoRGg5tT4z75oKMM5i9G"

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Global variables
model = None
model_id = "gemini-2.5-flash-google-ai"
DRIVE_SERVICE = None
SLIDES_SERVICE = None
presentation_id = None
presentation_data = None
gold_descriptions = None
original_locations = None


def load_gold_data():
    """Load gold descriptions and original image locations."""
    global gold_descriptions, original_locations

    # Load gold descriptions
    gold_descriptions = {}
    if os.path.exists(GOLD_DESCRIPTIONS_CSV):
        with open(GOLD_DESCRIPTIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gold_descriptions[row['filename']] = row['description']
        print(f"Loaded {len(gold_descriptions)} gold descriptions")
    else:
        print(f"WARNING: Gold descriptions file not found: {GOLD_DESCRIPTIONS_CSV}")

    # Load original image locations
    if os.path.exists(ORIGINAL_LOCATIONS_JSON):
        with open(ORIGINAL_LOCATIONS_JSON, 'r', encoding='utf-8') as f:
            original_locations = json.load(f)
        print(f"Loaded {len(original_locations)} original image locations")
    else:
        print(f"WARNING: Original locations file not found: {ORIGINAL_LOCATIONS_JSON}")
        original_locations = {}


def setup_presentation(workspace_doc_id):
    """Setup presentation processing."""
    global presentation_id, presentation_data, DRIVE_SERVICE, SLIDES_SERVICE

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace presentation ID: {workspace_doc_id}")
    presentation_id = workspace_doc_id

    # Initialize services
    DRIVE_SERVICE, SLIDES_SERVICE = initialize_google_services(service_type="slides")

    if not SLIDES_SERVICE:
        raise RuntimeError("Failed to initialize Google Slides service")

    # Fetch presentation data
    presentation_data = SLIDES_SERVICE.presentations().get(presentationId=presentation_id).execute()

    # Load gold data
    load_gold_data()


def calculate_percentage_score(success_count: int, total_count: int, max_points: int = 10) -> int:
    """Calculate score based on percentage, rounded to nearest 10%."""
    if total_count == 0:
        return 0
    percentage = success_count / total_count
    rounded_percentage = round(percentage, 1)  # Round to nearest 10%
    return int(rounded_percentage * max_points)


def grade_checkpoint_1():
    """
    Checkpoint 1 (10pt): All original images saved to Drive folder.

    All-or-nothing: Pass only if ALL gold images are found in the Drive folder.

    Uses a tiered batch approach to avoid rate limiting:
    1. Try exact match for ALL pairs first
    2. For unmatched, try perceptual hash for ALL pairs
    3. Only use VLM as last resort for remaining unmatched
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Images Saved to Drive")

    global model

    # Get list of files in the Drive folder
    step_start = time.time()
    drive_files = list_drive_folder_files(DRIVE_FOLDER_ID, DRIVE_SERVICE)

    if not drive_files:
        checkpoint.add_step(
            "All Images in Drive",
            False,
            1,
            "No files found in Drive folder",
            score=0,
            max_score=10,
            execution_time=time.time() - step_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    print(f"Found {len(drive_files)} files in Drive folder")

    # Create temp directory for downloaded images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download all Drive images sequentially
        drive_images = []
        image_files = [f for f in drive_files if f.get('mimeType', '').startswith('image/')]

        print(f"  Downloading {len(image_files)} images sequentially...")
        for file_info in image_files:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    img = download_drive_file_as_image(file_info['id'], DRIVE_SERVICE)
                    if img:
                        temp_path = os.path.join(temp_dir, f"drive_{file_info['id']}.png")
                        img.save(temp_path)
                        drive_images.append(temp_path)
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Backoff: 0.5s, 1s, 1.5s
                    else:
                        print(f"Failed to download {file_info['id']}: {e}")

        print(f"Downloaded {len(drive_images)} images from Drive")

        # Get all gold image files
        gold_image_files = [f for f in os.listdir(GOLD_IMAGES_DIR)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        # Track matches: gold_filename -> (matched, match_method, drive_path)
        matched_gold = {}
        unmatched_gold = set(gold_image_files)

        # Import matching functions
        from src.browsergym.eval.eval_utils.image_utils import image_exact_match, perceptual_hash_match

        # ============ TIER 1: Exact match for ALL pairs ============
        print("  Tier 1: Trying exact match for all pairs...")
        for gold_filename in list(unmatched_gold):
            gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)

            for drive_path in drive_images:
                try:
                    if image_exact_match(drive_path, gold_path):
                        print(f"    Matched {gold_filename} via exact")
                        matched_gold[gold_filename] = ("exact", drive_path)
                        unmatched_gold.discard(gold_filename)
                        break
                except Exception:
                    pass

        print(f"  After exact match: {len(matched_gold)} matched, {len(unmatched_gold)} remaining")

        # ============ TIER 2: Perceptual hash for remaining ============
        if unmatched_gold:
            print("  Tier 2: Trying perceptual hash for remaining...")
            for gold_filename in list(unmatched_gold):
                gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)

                for drive_path in drive_images:
                    try:
                        if perceptual_hash_match(drive_path, gold_path, threshold=15):
                            print(f"    Matched {gold_filename} via perceptual_hash")
                            matched_gold[gold_filename] = ("perceptual_hash", drive_path)
                            unmatched_gold.discard(gold_filename)
                            break
                    except Exception:
                        pass

            print(f"  After perceptual hash: {len(matched_gold)} matched, {len(unmatched_gold)} remaining")

        # ============ TIER 3: VLM for remaining (only if needed) ============
        if unmatched_gold:
            print("  Tier 3: Using VLM for remaining unmatched images...")
            if model is None:
                model = load_model(model_id)

            for gold_filename in list(unmatched_gold):
                gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)
                description = gold_descriptions.get(gold_filename, "")

                for drive_path in drive_images:
                    try:
                        vlm_result = binary_judge_image(
                            model,
                            drive_path,
                            f"Is this the same image or very similar to: {description}"
                        )
                        if vlm_result:
                            print(f"    Matched {gold_filename} via VLM")
                            matched_gold[gold_filename] = ("vlm", drive_path)
                            unmatched_gold.discard(gold_filename)
                            break
                    except Exception as e:
                        print(f"    VLM error for {gold_filename}: {e}")

            print(f"  After VLM: {len(matched_gold)} matched, {len(unmatched_gold)} remaining")

    step_time = time.time() - step_start
    total_gold = len(gold_image_files)
    matched_count = len(matched_gold)
    all_matched = matched_count == total_gold

    if all_matched:
        checkpoint.add_step(
            "All Images in Drive",
            True,
            1,
            f"All {total_gold} gold images found in Drive folder",
            score=10,
            max_score=10,
            execution_time=step_time
        )
    else:
        unmatched_list = list(unmatched_gold)
        checkpoint.add_step(
            "All Images in Drive",
            False,
            1,
            f"Only {matched_count}/{total_gold} images found. Missing: {', '.join(unmatched_list[:5])}{'...' if len(unmatched_list) > 5 else ''}",
            score=0,
            max_score=10,
            execution_time=step_time
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2 (40pt): Images deleted and replaced with red text placeholders.

    Steps (10pt each, percentage-based):
    1. Images removed at original locations
    2. Text box exists at original location (80% overlap)
    3. Text matches gold description (LLM similarity)
    4. Text is red and big (>= 18pt)
    """
    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Placeholder Text Boxes")

    global model

    if not original_locations:
        for step_num in range(1, 5):
            checkpoint.add_step(
                f"Step {step_num}",
                False,
                step_num,
                "Original image locations not available",
                score=0,
                max_score=10,
                execution_time=0
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    total_images = len(original_locations)

    # Track results for each step
    images_removed_count = 0
    textbox_at_location_count = 0
    text_matches_count = 0
    text_style_correct_count = 0

    # Store matched text boxes for use in checkpoint 3
    matched_textboxes = {}

    # Import matching functions for step 1
    from src.browsergym.eval.eval_utils.image_utils import image_exact_match, perceptual_hash_match

    with tempfile.TemporaryDirectory() as temp_dir:
        for gold_filename, loc_info in original_locations.items():
            slide_index = loc_info.get('slide_index', 0)
            original_bbox = loc_info.get('bbox', {})
            expected_description = loc_info.get('description', '')

            if slide_index >= len(slides):
                print(f"Slide index {slide_index} out of range for {gold_filename}")
                continue

            slide = slides[slide_index]

            # Step 1: Check if original image is removed (not present on slide)
            # Download all images on this slide and compare against the gold image
            current_images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
            original_image_found = False
            gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)

            for img_info in current_images:
                content_url = img_info.get('contentUrl')
                if not content_url:
                    continue

                # Download current slide image
                try:
                    pil_img = download_slide_image(content_url)
                    if pil_img:
                        temp_path = os.path.join(temp_dir, f"slide_{slide_index}_{gold_filename}")
                        pil_img.save(temp_path)

                        # Check if this is the original image using exact match or perceptual hash
                        try:
                            if image_exact_match(temp_path, gold_path):
                                original_image_found = True
                                break
                        except Exception:
                            pass

                        try:
                            if perceptual_hash_match(temp_path, gold_path, threshold=15):
                                original_image_found = True
                                break
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Error downloading image for comparison: {e}")

            if not original_image_found:
                images_removed_count += 1

            # Step 2: Check if text box exists at original location
            text_boxes = extract_text_boxes_from_slide(slide)
            matched_textbox = None

            for tb in text_boxes:
                tb_bbox = tb.get('bbox', {})
                # Text box should be 80% inside original image bbox
                if is_bbox_mostly_inside(tb_bbox, original_bbox, threshold=0.8):
                    matched_textbox = tb
                    break

            if matched_textbox:
                textbox_at_location_count += 1
                matched_textboxes[gold_filename] = matched_textbox

                # Step 3: Check if text matches description
                text_content = matched_textbox.get('text', '')

                if model is None:
                    model = load_model(model_id)

                # Use LLM to check similarity (lenient matching)
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are comparing two image descriptions to see if they refer to the same subject. Be lenient - answer 'Yes' if they describe similar content, the same main subject, or could plausibly be describing the same image even with different wording. Minor differences in details or phrasing should still count as a match. Only answer 'No' if they clearly describe completely different subjects."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Could these two descriptions be referring to the same image? Be lenient with wording differences.\n\nExpected description: {expected_description}\n\nActual text found: {text_content}"}]
                        }
                    ]

                    response = model(messages).strip().lower()
                    if 'yes' in response:
                        text_matches_count += 1
                    else:
                        print(f"  Text mismatch for {gold_filename}: LLM said '{response}'")
                except Exception as e:
                    print(f"LLM comparison failed for {gold_filename}: {e}")

                # Step 4: Check if text is red and big
                element = matched_textbox.get('element', {})
                if 'shape' in element:
                    text_style = get_text_style_from_shape(element['shape'])
                    if is_text_red(text_style) and is_text_big(text_style, min_pt=18):
                        text_style_correct_count += 1

    # Calculate scores for each step (percentage-based, 10pt max each)
    step_start = time.time()

    # Step 1: Images removed
    step1_score = calculate_percentage_score(images_removed_count, total_images, 10)
    checkpoint.add_step(
        "Images Removed",
        images_removed_count == total_images,
        1,
        f"{images_removed_count}/{total_images} original image locations have no image",
        score=step1_score,
        max_score=10,
        execution_time=0
    )

    # Step 2: Text boxes at locations
    step2_score = calculate_percentage_score(textbox_at_location_count, total_images, 10)
    checkpoint.add_step(
        "Text Boxes at Locations",
        textbox_at_location_count == total_images,
        2,
        f"{textbox_at_location_count}/{total_images} locations have text boxes (80% overlap required)",
        score=step2_score,
        max_score=10,
        execution_time=0
    )

    # Step 3: Text matches descriptions
    step3_score = calculate_percentage_score(text_matches_count, total_images, 10)
    checkpoint.add_step(
        "Text Matches Description",
        text_matches_count == total_images,
        3,
        f"{text_matches_count}/{total_images} text boxes match expected descriptions",
        score=step3_score,
        max_score=10,
        execution_time=0
    )

    # Step 4: Text style (red and big)
    step4_score = calculate_percentage_score(text_style_correct_count, total_images, 10)
    checkpoint.add_step(
        "Text is Red and Big",
        text_style_correct_count == total_images,
        4,
        f"{text_style_correct_count}/{total_images} text boxes have red text >= 18pt",
        score=step4_score,
        max_score=10,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3(browsing_history=None):
    """
    Checkpoint 3 (50pt): New images from web with URL attribution.

    Steps (10pt each, percentage-based):
    1. Replacement image exists at original location (60% overlap)
    2. URL credit exists on slide
    3. URL matches agent trace
    4. VLM check - new image similar to original
    5. New images fully overlay text boxes (90% coverage)
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=50, result=0, name="Web Images with Attribution")

    global model

    if not original_locations:
        for step_num in range(1, 6):
            checkpoint.add_step(
                f"Step {step_num}",
                False,
                step_num,
                "Original image locations not available",
                score=0,
                max_score=10,
                execution_time=0
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    total_images = len(original_locations)

    # Track results for each step
    new_image_count = 0
    url_on_slide_count = 0
    url_in_trace_count = 0
    image_similar_count = 0
    image_covers_text_count = 0

    # Normalize browsing history
    browsing_set = set(browsing_history) if browsing_history else set()

    with tempfile.TemporaryDirectory() as temp_dir:
        for gold_filename, loc_info in original_locations.items():
            slide_index = loc_info.get('slide_index', 0)
            original_bbox = loc_info.get('bbox', {})
            expected_description = loc_info.get('description', '')

            if slide_index >= len(slides):
                continue

            slide = slides[slide_index]

            # Step 1: Check if new image exists at location
            current_images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
            new_image = None
            new_image_bbox = None

            for img_info in current_images:
                # extract_slide_images returns transform and size at top level
                transform = img_info.get('transform', {})
                size = img_info.get('size', {})

                # Get raw dimensions
                raw_width = size.get('width', {}).get('magnitude', 0)
                raw_height = size.get('height', {}).get('magnitude', 0)

                # Apply scale factors from transform (default to 1 if not present)
                scale_x = transform.get('scaleX', 1)
                scale_y = transform.get('scaleY', 1)

                img_bbox = {
                    'x': transform.get('translateX', 0),
                    'y': transform.get('translateY', 0),
                    'width': raw_width * abs(scale_x),
                    'height': raw_height * abs(scale_y)
                }
                # New image should overlap original location by at least 60%
                if is_bbox_mostly_inside(img_bbox, original_bbox, threshold=0.6):
                    new_image = img_info
                    new_image_bbox = img_bbox
                    new_image_count += 1
                    break

            # Step 2: Check if URL exists on this slide
            slide_links = extract_slide_links(slide)
            if slide_links:
                url_on_slide_count += 1

                # Step 3: Check if any URL matches browsing history
                if browsing_set:
                    for link in slide_links:
                        if link in browsing_set or any(link in h or h in link for h in browsing_set):
                            url_in_trace_count += 1
                            break

            # Step 4: VLM check if new image is similar to original
            if new_image and new_image.get('contentUrl'):
                try:
                    pil_img = download_slide_image(new_image['contentUrl'])
                    if pil_img:
                        temp_path = os.path.join(temp_dir, f"new_{gold_filename}.png")
                        pil_img.save(temp_path)

                        if model is None:
                            model = load_model(model_id)

                        # Check if image reasonably matches the description
                        # Minor differences in details are acceptable
                        vlm_result = binary_judge_image(
                            model,
                            temp_path,
                            f"Could this image be a reasonable replacement for an original image with the following description? Minor differences in details are acceptable. Description: {expected_description}"
                        )

                        if vlm_result:
                            image_similar_count += 1
                except Exception as e:
                    print(f"VLM check failed for {gold_filename}: {e}")

            # Step 5: Check if new image covers the text box (from checkpoint 2)
            if new_image_bbox:
                # Find text boxes at this location
                text_boxes = extract_text_boxes_from_slide(slide)
                for tb in text_boxes:
                    tb_bbox = tb.get('bbox', {})
                    # Text box should be 90% inside new image
                    if is_bbox_mostly_inside(tb_bbox, new_image_bbox, threshold=0.9):
                        image_covers_text_count += 1
                        break

    # Calculate scores for each step (percentage-based, 10pt max each)
    step_start = time.time()

    # Step 1: New images at locations
    step1_score = calculate_percentage_score(new_image_count, total_images, 10)
    checkpoint.add_step(
        "New Images at Locations",
        new_image_count == total_images,
        1,
        f"{new_image_count}/{total_images} locations have new images (60% overlap required)",
        score=step1_score,
        max_score=10,
        execution_time=0
    )

    # Step 2: URL on slide
    step2_score = calculate_percentage_score(url_on_slide_count, total_images, 10)
    checkpoint.add_step(
        "URL Credit on Slide",
        url_on_slide_count == total_images,
        2,
        f"{url_on_slide_count}/{total_images} slides have URL credits",
        score=step2_score,
        max_score=10,
        execution_time=0
    )

    # Step 3: URL in browsing history
    if browsing_history:
        step3_score = calculate_percentage_score(url_in_trace_count, total_images, 10)
        checkpoint.add_step(
            "URL in Agent Trace",
            url_in_trace_count == total_images,
            3,
            f"{url_in_trace_count}/{total_images} URLs found in browsing history",
            score=step3_score,
            max_score=10,
            execution_time=0
        )
    else:
        checkpoint.add_step(
            "URL in Agent Trace",
            False,
            3,
            "No browsing history provided",
            score=0,
            max_score=10,
            execution_time=0
        )

    # Step 4: VLM similarity check
    step4_score = calculate_percentage_score(image_similar_count, total_images, 10)
    checkpoint.add_step(
        "Image Matches Description",
        image_similar_count == total_images,
        4,
        f"{image_similar_count}/{total_images} new images match expected descriptions",
        score=step4_score,
        max_score=10,
        execution_time=0
    )

    # Step 5: Image covers text box
    step5_score = calculate_percentage_score(image_covers_text_count, total_images, 10)
    checkpoint.add_step(
        "Image Covers Text Box",
        image_covers_text_count == total_images,
        5,
        f"{image_covers_text_count}/{total_images} new images fully cover text placeholders (90% required)",
        score=step5_score,
        max_score=10,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
    """
    Grade all checkpoints for the remove images and add placeholders task.

    Args:
        workspace_doc_id (str): Google Slides presentation ID to evaluate.
        cached_models (dict, optional): Dictionary of preloaded models by model_id.
        browsing_history (list, optional): List of URLs visited during task execution.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    try:
        # Setup presentation processing
        setup_presentation(workspace_doc_id)

        # Use cached model if available
        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        checkpoints: List[Checkpoint] = []

        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3(browsing_history))

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return a failed result
        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image replacement presentation task")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                       help="Google Slides presentation ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', default=None,
                       help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history
    )

    print("\n=== EVALUATION RESULTS ===")
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
