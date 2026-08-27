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
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    list_drive_folder_files,
    download_drive_file_as_image,
    download_drive_image_threadsafe
)
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_images,
    download_slide_image,
    extract_slide_links_with_positions,
    extract_text_boxes_from_slide,
    get_text_style_from_shape,
    is_text_color,
    is_text_big,
    find_url_below_image
)
from src.browsergym.knows.eval.eval_utils.image_utils import match_image_tiered, binary_judge_image
from src.browsergym.knows.eval.eval_utils.text_utils import text_fuzzy_match_contained_short
from src.browsergym.knows.eval.eval_utils.utils import is_bbox_mostly_inside
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url, is_unverifiable_url
from src.browsergym.knows.eval.eval_utils.parallel_utils import (
    parallel_download,
    parallel_execute,
    parallel_vlm_calls,
    parallel_image_match,
    fast_parallel_vlm_calls,
    VLM_API_SEMAPHORE
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/slides_17_removeimagesaddplaceholders/instance_4/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
GOLD_IMAGES_DIR = os.path.join(DATA_DIR, "gold_images/")
GOLD_DESCRIPTIONS_CSV = os.path.join(DATA_DIR, "gold_descriptions.csv")
ORIGINAL_LOCATIONS_JSON = os.path.join(DATA_DIR, "original_image_locations.json")
ORIGINAL_TEXTBOX_LOCATIONS_JSON = os.path.join(DATA_DIR,"original_textbox_locations.json")
DRIVE_FOLDER_ID = "1sZUeENx2F8tVnDuHgJP6N9yp7fCD7qPo"

DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Global variables
model = None
model_id = "gemini-3-flash-google-ai"
DRIVE_SERVICE = None
SLIDES_SERVICE = None
presentation_id = None
presentation_data = None
gold_descriptions = None
original_locations = None
original_textbox_locations = None

# Cached slide data (populated by prefetch)
cached_slide_images = {}  # slide_index -> list of (img_info, local_path)
cached_text_boxes = {}    # slide_index -> list of text boxes


def load_gold_data():
    """Load gold descriptions and original image locations."""
    global gold_descriptions, original_locations, original_textbox_locations

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
    
    if os.path.exists(ORIGINAL_TEXTBOX_LOCATIONS_JSON):
        with open(ORIGINAL_TEXTBOX_LOCATIONS_JSON,'r',encoding='utf-8') as f:
            original_textbox_locations = json.load(f)
            print(f"Loaded {len(original_textbox_locations)} original textbox locations")
    else:
        print("Original textbox location file not found")
        original_textbox_locations = {}


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


def prefetch_slide_data(temp_dir: str):
    """
    Prefetch all slide images and text boxes in parallel for reuse across checkpoints.

    This significantly speeds up checkpoint 2 and 3 by avoiding redundant downloads.

    Args:
        temp_dir: Temporary directory to store downloaded images.
    """
    global cached_slide_images, cached_text_boxes

    slides = presentation_data.get('slides', [])

    # Get unique slide indices (union of image and textbox slides)
    slide_indices = set()
    for loc_info in original_locations.values():
        slide_indices.add(loc_info.get('slide_index', 0))
    for loc_info in original_textbox_locations:
        slide_indices.add(loc_info.get('slide_index', 0))

    print(f"  Prefetching data from {len(slide_indices)} slides...")

    # Step 1: Extract all image info and text boxes (fast, no downloads)
    all_download_tasks = []
    for slide_index in slide_indices:
        if slide_index >= len(slides):
            continue

        slide = slides[slide_index]

        # Extract and cache text boxes (no I/O needed)
        cached_text_boxes[slide_index] = extract_text_boxes_from_slide(slide)

        # Extract image info and prepare download tasks
        current_images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
        cached_slide_images[slide_index] = []

        for img_idx, img_info in enumerate(current_images):
            content_url = img_info.get('contentUrl')
            if content_url:
                task_id = f"{slide_index}_{img_idx}"
                all_download_tasks.append({
                    'id': task_id,
                    'func': download_slide_image,
                    'args': (content_url,),
                    'slide_index': slide_index,
                    'img_info': img_info
                })

    # Step 2: Download all slide images in parallel
    if all_download_tasks:
        print(f"  Downloading {len(all_download_tasks)} slide images in parallel...")
        download_results = parallel_download(all_download_tasks, max_workers=5, use_rate_limit=False)

        # Save results and update cache
        for task in all_download_tasks:
            task_id = task['id']
            slide_index = task['slide_index']
            img_info = task['img_info']
            pil_img = download_results.get(task_id)

            if pil_img:
                temp_path = os.path.join(temp_dir, f"prefetch_{task_id}.png")
                pil_img.save(temp_path)
                cached_slide_images[slide_index].append((img_info, temp_path))

    total_images = sum(len(imgs) for imgs in cached_slide_images.values())
    print(f"  Prefetch complete: {total_images} images cached from {len(cached_slide_images)} slides")


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

    try:
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
                execution_time=time.time() - step_start,
                category=StepCategory.DETERMINISTIC
            )
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        print(f"Found {len(drive_files)} files in Drive folder")

        # Create temp directory for downloaded images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download all Drive images in parallel
            drive_images = []
            image_files = [f for f in drive_files if f.get('mimeType', '').startswith('image/')]

            # Get token safely for thread-safe downloads
            token = None
            try:
                if hasattr(DRIVE_SERVICE, '_http') and hasattr(DRIVE_SERVICE._http, 'credentials'):
                     token = DRIVE_SERVICE._http.credentials.token
                elif hasattr(DRIVE_SERVICE, 'credentials'):
                     token = DRIVE_SERVICE.credentials.token
            except Exception as e:
                print(f"ERROR: Failed to extract OAuth token: {e}")

            if not token:
                checkpoint.add_step(
                    "All Images in Drive", False, 1,
                    "Could not extract OAuth token for parallel Drive downloads",
                    score=0, max_score=10, execution_time=time.time() - step_start,
                    category=StepCategory.EXECUTION_ERROR
                )
                checkpoint.execution_time = time.time() - checkpoint_start
                return checkpoint

            print(f"  Downloading {len(image_files)} images in parallel...")
            download_tasks = [
                {
                    'id': file_info['id'],
                    'func': download_drive_image_threadsafe,
                    'args': (file_info['id'], token)
                }
                for file_info in image_files
            ]

            # Use requests-based download, so we can skip the strict API client semaphore
            download_results = parallel_download(download_tasks, max_workers=5, use_rate_limit=False)

            # Save downloaded images to temp files
            for file_id, img in download_results.items():
                if img:
                    temp_path = os.path.join(temp_dir, f"drive_{file_id}.png")
                    img.save(temp_path)
                    drive_images.append(temp_path)

            print(f"Downloaded {len(drive_images)} images from Drive")

            # Get all gold image files
            if not os.path.isdir(GOLD_IMAGES_DIR):
                checkpoint.add_step(
                    "All Images in Drive", False, 1,
                    f"Gold images directory not found: {GOLD_IMAGES_DIR}",
                    score=0, max_score=10, execution_time=time.time() - step_start,
                    category=StepCategory.EXECUTION_ERROR
                )
                checkpoint.execution_time = time.time() - checkpoint_start
                return checkpoint

            gold_image_files = [f for f in os.listdir(GOLD_IMAGES_DIR)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

            if len(gold_image_files) == 0:
                checkpoint.add_step(
                    "All Images in Drive", False, 1,
                    "No gold images found to compare against",
                    score=0, max_score=10, execution_time=time.time() - step_start,
                    category=StepCategory.EXECUTION_ERROR
                )
                checkpoint.execution_time = time.time() - checkpoint_start
                return checkpoint

            # Track matches: gold_filename -> (matched, match_method, drive_path)
            matched_gold = {}
            unmatched_gold = set(gold_image_files)

            # ============ PARALLEL TIERED MATCHING (Exact + Perceptual Hash) ============
            # Build all match tasks for parallel execution
            print("  Building match tasks for parallel execution...")
            match_tasks = []
            for gold_filename in gold_image_files:
                gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)
                for drive_path in drive_images:
                    match_tasks.append({
                        'id': f"{gold_filename}|{os.path.basename(drive_path)}",
                        'gold_filename': gold_filename,
                        'candidate_path': drive_path,
                        'gold_path': gold_path
                    })

            print(f"  Running {len(match_tasks)} match comparisons in parallel...")
            match_results = parallel_image_match(match_tasks, max_workers=8)

            # Process results - find first match for each gold file
            for task_id, (matched, method) in match_results.items():
                if matched:
                    gold_filename = task_id.split('|')[0]
                    if gold_filename in unmatched_gold:
                        # Find the drive path from the task
                        for task in match_tasks:
                            if task['id'] == task_id:
                                print(f"    Matched {gold_filename} via {method}")
                                matched_gold[gold_filename] = (method, task['candidate_path'])
                                unmatched_gold.discard(gold_filename)
                                break

            print(f"  After parallel matching: {len(matched_gold)} matched, {len(unmatched_gold)} remaining")

            # ============ TIER 3: VLM for remaining (PARALLELIZED) ============
            if unmatched_gold:
                print("  Tier 3: Using VLM for remaining unmatched images (parallel search)...")
                if model is None:
                    model = load_model(model_id)

                from concurrent.futures import ThreadPoolExecutor, as_completed

                def find_match_vlm(gold_file):
                    # Search through all drive images for a match with this gold file
                    gold_p = os.path.join(GOLD_IMAGES_DIR, gold_file)
                    desc = gold_descriptions.get(gold_file, "")

                    # Check each drive image
                    for d_path in drive_images:
                        try:
                            is_match = binary_judge_image(
                                model,
                                d_path,
                                f"Is this the same image or very similar to: {desc}"
                            )
                            if is_match:
                                return gold_file, d_path
                        except Exception:
                            continue
                    return gold_file, None

                # Run VLM search in parallel for each unmatched gold file
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(find_match_vlm, gf) for gf in unmatched_gold]

                    for future in as_completed(futures):
                        try:
                            gf, match_path = future.result()
                            if match_path:
                                print(f"    Matched {gf} via VLM")
                                matched_gold[gf] = ("vlm", match_path)
                                unmatched_gold.discard(gf)
                        except Exception as e:
                            print(f"    VLM match future error: {e}")

                print(f"  After VLM: {len(matched_gold)} matched, {len(unmatched_gold)} remaining")

        step_time = time.time() - step_start
        total_gold = len(gold_image_files)
        matched_count = len(matched_gold)
        all_matched = matched_count == total_gold

        if all_matched:
            # Category = deepest tier that had to run to match any gold image
            match_methods = {method for method, _ in matched_gold.values()}
            if "vlm" in match_methods:
                match_category = StepCategory.LLM_VLM_JUDGEMENT
            elif "perceptual_hash" in match_methods:
                match_category = StepCategory.FUZZY_MATCH
            else:
                match_category = StepCategory.DETERMINISTIC
            checkpoint.add_step(
                "All Images in Drive",
                True,
                1,
                f"All {total_gold} gold images found in Drive folder",
                score=10,
                max_score=10,
                execution_time=step_time,
                category=match_category
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
                execution_time=step_time,
                category=StepCategory.LLM_VLM_JUDGEMENT
            )
    except Exception as e:
        checkpoint.add_step("Error", False, 1, f"Checkpoint failed: {e}",
                            score=0, max_score=10, execution_time=time.time() - checkpoint_start,
                            category=StepCategory.EXECUTION_ERROR)
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

    Uses cached slide data from prefetch for performance.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Placeholder Text Boxes")

    global model

    try:
        step_names = [
            "Images Removed",
            "Text Boxes at Locations",
            "Text Matches Description",
            "Text is Red and Big",
        ]

        if not original_locations:
            for step_num, step_name in enumerate(step_names, start=1):
                checkpoint.add_step(
                    step_name,
                    False,
                    step_num,
                    "Original image locations not available",
                    score=0,
                    max_score=10,
                    execution_time=0,
                    category=StepCategory.EXECUTION_ERROR
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

        # Collect data for parallel LLM calls
        llm_comparison_tasks = []

        # ============ STEP 1 & 2: Use cached data, build parallel match tasks ============
        match_tasks = []

        for gold_filename, loc_info in original_locations.items():
            slide_index = loc_info.get('slide_index', 0)
            original_bbox = loc_info.get('bbox', {})
            expected_description = loc_info.get('description', '')

            if slide_index >= len(slides):
                print(f"Slide index {slide_index} out of range for {gold_filename}")
                continue

            gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)

            # Step 1: Build match tasks for checking if original image is removed
            # Use cached slide images instead of downloading again
            cached_images = cached_slide_images.get(slide_index, [])
            for img_info, local_path in cached_images:
                match_tasks.append({
                    'id': f"{gold_filename}|{os.path.basename(local_path)}",
                    'gold_filename': gold_filename,
                    'candidate_path': local_path,
                    'gold_path': gold_path
                })

            # Step 2: Check text box at location (using cached text boxes)
            text_boxes = cached_text_boxes.get(slide_index, [])
            matched_textbox = None

            for tb in text_boxes:
                tb_bbox = tb.get('bbox', {})
                if is_bbox_mostly_inside(tb_bbox, original_bbox, threshold=0.8):
                    matched_textbox = tb
                    break

            if matched_textbox:
                textbox_at_location_count += 1
                matched_textboxes[gold_filename] = matched_textbox

                # Prepare for Step 3: Collect LLM comparison task
                text_content = matched_textbox.get('text', '')
                llm_comparison_tasks.append({
                    'gold_filename': gold_filename,
                    'expected_description': expected_description,
                    'text_content': text_content,
                    'matched_textbox': matched_textbox
                })
            else:
                print(f"    No matching text box for {gold_filename}")

        # ============ STEP 1: Run parallel image matching ============
        print(f"  Running {len(match_tasks)} image removal checks in parallel...")
        match_results = parallel_image_match(match_tasks, max_workers=8)

        # Check which gold images are still present (not removed)
        gold_images_found = set()
        for task_id, (matched, method) in match_results.items():
            if matched:
                gold_filename = task_id.split('|')[0]
                gold_images_found.add(gold_filename)

        # Count images that were successfully removed
        for gold_filename in original_locations.keys():
            if gold_filename not in gold_images_found:
                images_removed_count += 1

        print(f"  {images_removed_count}/{total_images} original images removed")

        # ============ STEP 3: Parallel LLM text comparison ============
        if llm_comparison_tasks:
            if model is None:
                model = load_model(model_id)

            print(f"  Running {len(llm_comparison_tasks)} text comparisons in parallel...")

            # Build VLM tasks for parallel execution
            vlm_tasks = []
            for task in llm_comparison_tasks:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are comparing two image descriptions to see if they refer to the same subject. Be lenient—answer 'Yes' if they describe similar content, the same main subject, or could plausibly be describing the same image, unless they contain a direct factual contradiction like a different color."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Could these two descriptions be referring to the same image? Be lenient with wording differences.\n\nExpected description: {task['expected_description']}\n\nActual text found: {task['text_content']}"}]
                    }
                ]
                vlm_tasks.append({
                    'id': task['gold_filename'],
                    'messages': messages
                })

            # Use faster parallel VLM calls with more workers
            vlm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=10)

            for task in llm_comparison_tasks:
                gold_filename = task['gold_filename']
                if vlm_results.get(gold_filename, False):
                    text_matches_count += 1
                else:
                    print(f"  Text mismatch for {gold_filename}")

        # ============ STEP 4: Check text style (red and big) ============
        for task in llm_comparison_tasks:
            matched_textbox = task['matched_textbox']
            element = matched_textbox.get('element', {})
            if 'shape' in element:
                text_style = get_text_style_from_shape(element['shape'])
                if is_text_color(text_style, r=1.0, g=0.0, b=0.0, tolerance=0.42) and is_text_big(text_style, min_pt=18):
                    text_style_correct_count += 1
                else:
                    print(f"  Text style mismatch for {task['gold_filename']}")

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
            execution_time=0,
            category=StepCategory.FUZZY_MATCH
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
            execution_time=0,
            category=StepCategory.SPATIAL
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
            execution_time=0,
            category=StepCategory.LLM_VLM_JUDGEMENT
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
            execution_time=time.time() - step_start,
            category=StepCategory.FUZZY_MATCH
        )
    except Exception as e:
        checkpoint.add_step("Error", False, 1, f"Checkpoint failed: {e}",
                            score=0, max_score=40, execution_time=time.time() - checkpoint_start,
                            category=StepCategory.EXECUTION_ERROR)
    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (50pt): New images from web with URL attribution.

    Steps (10pt each, percentage-based):
    1. Replacement image exists at original location (60% overlap)
    2. URL credit exists on slide
    3. URL is valid link to image (verified via image matching against replacement)
    4. VLM check - replacement is reasonable substitute for original (compares both images directly, with description fallback)
    5. New images fully overlay text boxes (80% coverage)

    Uses cached slide data from prefetch and parallel processing for performance.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=50, result=0, name="Web Images with Attribution")

    global model

    try:
        step_names = [
            "New Images at Locations",
            "URL Credit on Slide Under Image",
            "URL Points to Correct Image",
            "Replacement Matches Original",
            "Image Covers Text Box",
        ]

        if not original_locations:
            for step_num, step_name in enumerate(step_names, start=1):
                checkpoint.add_step(
                    step_name,
                    False,
                    step_num,
                    "Original image locations not available",
                    score=0,
                    max_score=10,
                    execution_time=0,
                    category=StepCategory.EXECUTION_ERROR
                )
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        slides = presentation_data.get('slides', [])
        total_images = len(original_locations)

        # Track results for each step
        new_image_count = 0
        url_on_slide_count = 0
        url_valid_count = 0
        image_similar_count = 0
        image_covers_text_count = 0

        # Collect data for parallel processing
        image_data = {}  # gold_filename -> {new_image_bbox, replacement_path, matched_url, etc.}
        url_download_tasks = []  # URLs to download in parallel
        vlm_comparison_tasks = []  # VLM tasks to run in parallel

        with tempfile.TemporaryDirectory() as temp_dir:
            # ============ PHASE 1: Collect all data using cached images ============
            for gold_filename, loc_info in original_locations.items():
                slide_index = loc_info.get('slide_index', 0)
                original_bbox = loc_info.get('bbox', {})
                expected_description = loc_info.get('description', '')

                if slide_index >= len(slides):
                    continue

                slide = slides[slide_index]

                # Initialize data for this image
                image_data[gold_filename] = {
                    'slide_index': slide_index,
                    'original_bbox': original_bbox,
                    'expected_description': expected_description,
                    'new_image': None,
                    'new_image_bbox': None,
                    'replacement_path': None,
                    'matched_url': None
                }

                # Step 1: Check if new image exists at location (using cached images)
                cached_images = cached_slide_images.get(slide_index, [])
                for img_info, local_path in cached_images:
                    transform = img_info.get('transform', {})
                    size = img_info.get('size', {})

                    raw_width = size.get('width', {}).get('magnitude', 0)
                    raw_height = size.get('height', {}).get('magnitude', 0)
                    scale_x = transform.get('scaleX', 1)
                    scale_y = transform.get('scaleY', 1)

                    img_bbox = {
                        'x': transform.get('translateX', 0),
                        'y': transform.get('translateY', 0),
                        'width': raw_width * abs(scale_x),
                        'height': raw_height * abs(scale_y)
                    }

                    if is_bbox_mostly_inside(img_bbox, original_bbox, threshold=0.6):
                        image_data[gold_filename]['new_image'] = img_info
                        image_data[gold_filename]['new_image_bbox'] = img_bbox
                        image_data[gold_filename]['replacement_path'] = local_path
                        new_image_count += 1
                        break

                # Step 2: Find URL below the new image
                new_image_bbox = image_data[gold_filename]['new_image_bbox']
                if new_image_bbox:
                    slide_links_with_pos = extract_slide_links_with_positions(slide)
                    matched_url = find_url_below_image(new_image_bbox, slide_links_with_pos)
                    if matched_url:
                        image_data[gold_filename]['matched_url'] = matched_url
                        url_on_slide_count += 1
                        print(f"  URL found below new image for {gold_filename}: {matched_url}")
                        # Prepare URL download task (if not unverifiable)
                        if not is_unverifiable_url(matched_url):
                            url_download_tasks.append({
                                'id': gold_filename,
                                'func': download_image_from_url,
                                'args': (matched_url, temp_dir)
                            })
                    else:
                        print(f"  No URL found below new image for {gold_filename}")

            # ============ PHASE 1.5: Filter out originals still present ============
            # Use exact + perceptual hash matching (no VLM) to check if found
            # images are actually the unchanged originals
            originality_check_tasks = []
            for gold_filename, data in image_data.items():
                replacement_path = data.get('replacement_path')
                if replacement_path:
                    gold_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)
                    originality_check_tasks.append({
                        'id': gold_filename,
                        'candidate_path': replacement_path,
                        'gold_path': gold_path
                    })

            if originality_check_tasks:
                print(f"  Checking {len(originality_check_tasks)} images for originality (exact+hash only)...")
                originality_results = parallel_image_match(originality_check_tasks, max_workers=8)

                for gold_filename, (is_original, method) in originality_results.items():
                    if is_original:
                        print(f"  SKIP: {gold_filename} is the original image (matched via {method})")
                        image_data[gold_filename]['new_image'] = None
                        image_data[gold_filename]['new_image_bbox'] = None
                        image_data[gold_filename]['replacement_path'] = None
                        image_data[gold_filename]['matched_url'] = None

            # Recompute counters after filtering out originals
            new_image_count = 0
            url_on_slide_count = 0
            image_covers_text_count = 0

            for gold_filename, data in image_data.items():
                if data.get('replacement_path'):
                    new_image_count += 1
                    if data.get('matched_url'):
                        url_on_slide_count += 1

            for gold_filename, data in image_data.items():
                new_image_bbox = data.get('new_image_bbox')
                if new_image_bbox:
                    slide_index = data['slide_index']
                    text_boxes = cached_text_boxes.get(slide_index, [])
                    for tb in text_boxes:
                        tb_bbox = tb.get('bbox', {})
                        if is_bbox_mostly_inside(tb_bbox, new_image_bbox, threshold=0.8):
                            image_covers_text_count += 1
                            break

            # Filter url_download_tasks to exclude originals
            url_download_tasks = [
                t for t in url_download_tasks
                if image_data[t['id']].get('replacement_path') is not None
            ]

            print(f"  After originality filter: {new_image_count}/{total_images} genuinely new images")

            # ============ PHASE 2: Parallel URL downloads ============
            url_download_results = {}
            if url_download_tasks:
                print(f"  Downloading {len(url_download_tasks)} URL images in parallel...")
                url_download_results = parallel_download(url_download_tasks, max_workers=5, use_rate_limit=False)

            # ============ PHASE 3: Build VLM comparison tasks ============
            if model is None:
                model = load_model(model_id)

            for gold_filename, data in image_data.items():
                replacement_path = data['replacement_path']
                if replacement_path:
                    original_path = os.path.join(GOLD_IMAGES_DIR, gold_filename)
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are comparing two images to determine if one is a reasonable replacement for the other. Answer 'Yes' if the replacement image retains the key important details and subject matter of the original image, even if the style, quality, or minor details differ. Answer 'No' only if the replacement is clearly showing something completely different or unrelated."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Here is the original image:"},
                                {"type": "image", "image": original_path},
                                {"type": "text", "text": "Here is the replacement image:"},
                                {"type": "image", "image": replacement_path},
                                {"type": "text", "text": "Is the replacement image a reasonable substitute that retains the key important details of the original image?"}
                            ]
                        }
                    ]
                    vlm_comparison_tasks.append({
                        'id': gold_filename,
                        'messages': messages,
                        'expected_description': data['expected_description'],
                        'replacement_path': replacement_path
                    })

            # ============ PHASE 4: Run VLM comparisons in parallel ============
            print(f"  Running {len(vlm_comparison_tasks)} VLM image comparisons in parallel...")
            # Use faster parallel VLM calls with more workers
            vlm_results = fast_parallel_vlm_calls(
                [{'id': t['id'], 'messages': t['messages']} for t in vlm_comparison_tasks],
                model,
                max_workers=10
            )

            # Process VLM results with fallback for failures
            for task in vlm_comparison_tasks:
                gold_filename = task['id']
                if vlm_results.get(gold_filename, False):
                    print(f"  VLM: Replacement for {gold_filename} matched via image comparison")
                    image_similar_count += 1
                else:
                    # Try description fallback
                    expected_description = task['expected_description']
                    replacement_path = task['replacement_path']
                    if expected_description:
                        try:
                            fallback_result = binary_judge_image(
                                model,
                                replacement_path,
                                f"Could this image be a reasonable replacement for an original image with the following description? Minor differences in details are acceptable. Description: {expected_description}"
                            )
                            if fallback_result:
                                print(f"  VLM: Replacement for {gold_filename} matched via description fallback")
                                image_similar_count += 1
                            else:
                                print(f"  VLM: Description fallback also failed for {gold_filename}")
                        except Exception as e:
                            print(f"  VLM fallback error for {gold_filename}: {e}")

            # ============ PHASE 5: Validate URLs against replacement images ============
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def validate_url_image(task_data):
                """Returns (is_valid, category) where category records the deciding mechanism."""
                g_filename = task_data['gold_filename']
                m_url = task_data['matched_url']
                r_path = task_data['replacement_path']
                desc = task_data['expected_description']

                if is_unverifiable_url(m_url):
                    print(f"  URL validation: {g_filename} URL is from unverifiable domain, treating as valid")
                    return True, StepCategory.VACUOUS_PASS

                url_img_path = url_download_results.get(g_filename)
                if url_img_path:
                    try:
                        match_res, match_meth = match_image_tiered(
                            url_img_path,
                            r_path,
                            model=model,
                            description=desc
                        )
                        if match_res:
                            print(f"  URL validation: {g_filename} URL matches replacement via {match_meth}")
                            return True, StepCategory.from_match_method(match_meth)
                        else:
                            print(f"  URL validation: {g_filename} URL does not match replacement image")
                            return False, StepCategory.LLM_VLM_JUDGEMENT
                    except Exception as ex:
                        print(f"  URL validation failed for {g_filename}: {ex}")
                        return False, StepCategory.EXECUTION_ERROR
                else:
                    print(f"  URL validation: Failed to download image from {m_url}")
                    return False, StepCategory.EXECUTION_ERROR

            # Build tasks for parallel execution
            validation_tasks = []
            for gold_filename, data in image_data.items():
                if data['matched_url'] and data['replacement_path']:
                    validation_tasks.append({
                        'gold_filename': gold_filename,
                        'matched_url': data['matched_url'],
                        'replacement_path': data['replacement_path'],
                        'expected_description': data['expected_description']
                    })

            print(f"  Validating {len(validation_tasks)} URLs in parallel...")

            # Per-item (category, success) provenance for the URL validation step
            url_validation_items = [
                (StepCategory.DEPENDENCY_NOT_EVALUATED, False)
                for _ in range(total_images - len(validation_tasks))
            ]

            # Execute validation in parallel
            if validation_tasks:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(validate_url_image, t) for t in validation_tasks]
                    for future in as_completed(futures):
                        try:
                            is_valid, item_category = future.result()
                            url_validation_items.append((item_category, is_valid))
                            if is_valid:
                                url_valid_count += 1
                        except Exception as e:
                            print(f"  URL validation future error: {e}")
                            url_validation_items.append((StepCategory.EXECUTION_ERROR, False))

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
            execution_time=0,
            category=StepCategory.SPATIAL
        )

        # Step 2: URL on slide
        step2_score = calculate_percentage_score(url_on_slide_count, total_images, 10)
        checkpoint.add_step(
            "URL Credit on Slide Under Image",
            url_on_slide_count == total_images,
            2,
            f"{url_on_slide_count}/{total_images} images have URL credits in the right location",
            score=step2_score,
            max_score=10,
            execution_time=0,
            category=StepCategory.SPATIAL
        )

        # Step 3: URL is valid link to image
        step3_score = calculate_percentage_score(url_valid_count, total_images, 10)
        checkpoint.add_step(
            "URL Points to Correct Image",
            url_valid_count == total_images,
            3,
            f"{url_valid_count}/{total_images} URLs point to the correct replacement images as well in the correct locations.",
            score=step3_score,
            max_score=10,
            execution_time=0,
            category=StepCategory.aggregate(url_validation_items)
        )

        # Step 4: VLM similarity check (compares original and replacement images directly)
        step4_score = calculate_percentage_score(image_similar_count, total_images, 10)
        checkpoint.add_step(
            "Replacement Matches Original",
            image_similar_count == total_images,
            4,
            f"{image_similar_count}/{total_images} replacement images are reasonable substitutes for originals as well in the correct locations.",
            score=step4_score,
            max_score=10,
            execution_time=0,
            category=StepCategory.LLM_VLM_JUDGEMENT
        )

        # Step 5: Image covers text box
        step5_score = calculate_percentage_score(image_covers_text_count, total_images, 10)
        checkpoint.add_step(
            "Image Covers Text Box",
            image_covers_text_count == total_images,
            5,
            f"{image_covers_text_count}/{total_images} new images fully cover text placeholders (80% required)",
            score=step5_score,
            max_score=10,
            execution_time=time.time() - step_start,
            category=StepCategory.SPATIAL
        )
    except Exception as e:
        checkpoint.add_step("Error", False, 1, f"Checkpoint failed: {e}",
                            score=0, max_score=50, execution_time=time.time() - checkpoint_start,
                            category=StepCategory.EXECUTION_ERROR)
    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_4():
    """
    Checkpoint 4 (20pt): Formatting

    Steps (10pt each, percentage-based):
    1. Extra Images (Checks for any extra images that were added to the slides)
    2. Extra Textboxes (Checks for any extra textboxes that were added to the slides)

    Higher Scores mean that the agent did not add extra images or textboxes
    """

    print("----------------- CHECKPOINT 4 ----------------")

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total = 20, result = 0, name="Formatting Check")

    try:
        if not original_locations:
            checkpoint.add_step("Images Amount in Slides is Equal", False, 1,
                                "Original image locations not available", score=0, max_score=10, execution_time=0,
                                category=StepCategory.EXECUTION_ERROR)
            checkpoint.add_step("Extra textbox check", False, 2,
                                "Original image locations not available", score=0, max_score=10, execution_time=0,
                                category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        slides = presentation_data.get('slides',[])

        #get the total of textboxes and images in the orignal slides
        total_images_of_original_slides= len(original_locations)
        total_textboxes_of_original_slides = len(original_textbox_locations)

        #get the total of textboxes and images in the new slides
        new_total_images_of_new_slides = sum(len(imgs) for imgs in cached_slide_images.values())
        new_total_textboxes = sum(len(txt) for txt in cached_text_boxes.values())

        step_start = time.time()

        #dictionaries for slide specific totals
        new_tb_count = {}
        img_box_count = {}
        original_textbox_count = {}
        original_image_count = {}

        # Get the amount of images and textboxes in each slide in the new presentation
        for slide_index in range(len(slides)):
            images = cached_slide_images.get(slide_index, [])
            img_box_count[slide_index] = len(images)

            tb = cached_text_boxes.get(slide_index, [])
            new_tb_count[slide_index] = len(tb)

        # Get the amount of images in original slides
        for info in original_locations.values():
            slide = info.get('slide_index',0)

            if(slide in original_image_count):
                original_image_count[slide] +=1
            else:
                original_image_count[slide] = 1

        # Get the amount of original textboxes
        for info in original_textbox_locations:
            slide = info.get('slide_index',0)

            if(slide in original_textbox_count):
                original_textbox_count[slide] += 1
            else:
                original_textbox_count[slide] = 1

        # Only penalize slides that have MORE images than expected (extras)
        extra_img_slides = []
        for slide_number in set(original_image_count.keys()) | set(img_box_count.keys()):
            if img_box_count.get(slide_number, 0) > original_image_count.get(slide_number, 0):
                extra_img_slides.append(slide_number)

        total_slides = len(original_image_count.keys())

        if not extra_img_slides:
            checkpoint.add_step(
                "No Extra Images",
                True,
                1,
                "No extra images were added",
                score=10,
                max_score=10,
                execution_time=time.time() - step_start,
                category=StepCategory.STRUCTURAL
            )
        else:
            step_1_percentage_score = int(max(0, ((total_slides - len(extra_img_slides)) / total_slides)) * 10)
            checkpoint.add_step(
                "No Extra Images",
                False,
                1,
                f"{len(extra_img_slides)}/{total_slides} slides have extra images (slides: {extra_img_slides})",
                score=step_1_percentage_score,
                max_score=10,
                execution_time=time.time() - step_start,
                category=StepCategory.STRUCTURAL
            )

        # Only penalize slides that have MORE textboxes than expected (extras)
        # Expected per slide: original textboxes + 2 * original images (description + URL credit)
        extra_textbox_slides = []
        for slide_number in set(original_textbox_count.keys()) | set(new_tb_count.keys()):
            expected_for_slide = original_textbox_count.get(slide_number, 0) + original_image_count.get(slide_number, 0) * 2
            if new_tb_count.get(slide_number, 0) > expected_for_slide:
                extra_textbox_slides.append(slide_number)

        total_slides_tb = len(original_textbox_count.keys())

        if not extra_textbox_slides:
            checkpoint.add_step(
                "No Extra Textboxes",
                True,
                2,
                "No extra textboxes were added",
                score=10,
                max_score=10,
                execution_time=time.time() - step_start,
                category=StepCategory.STRUCTURAL
            )
        else:
            step_2_percentage_score = int(max(0, ((total_slides_tb - len(extra_textbox_slides)) / total_slides_tb)) * 10)
            checkpoint.add_step(
                "No Extra Textboxes",
                False,
                2,
                f"{len(extra_textbox_slides)}/{total_slides_tb} slides have extra textboxes (slides: {extra_textbox_slides})",
                score=step_2_percentage_score,
                max_score=10,
                execution_time=time.time() - step_start,
                category=StepCategory.STRUCTURAL
            )
    except Exception as e:
        checkpoint.add_step("Error", False, 1, f"Checkpoint failed: {e}",
                            score=0, max_score=20, execution_time=time.time() - checkpoint_start,
                            category=StepCategory.EXECUTION_ERROR)
    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint
        
    

def grade_checkpoints(workspace_doc_id, cached_models=None):
    """
    Grade all checkpoints for the remove images and add placeholders task.

    Args:
        workspace_doc_id (str): Google Slides presentation ID to evaluate.
        cached_models (dict, optional): Dictionary of preloaded models by model_id.

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
        elif model is None:
            model = load_model(model_id)

        checkpoints: List[Checkpoint] = []

        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as prefetch_temp_dir:
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Phase 1: Start CP1 in parallel with prefetch
                # CP1 is independent (only checks Drive folder, no cache needed)
                cp1_future = executor.submit(grade_checkpoint_1)

                # Prefetch slide data (blocks until complete, populates cache)
                print("\n----------------- PREFETCH SLIDE DATA ----------------")
                prefetch_slide_data(prefetch_temp_dir)

                # Phase 2: Run CP2 and CP3 in parallel (both only read from cache)
                cp2_future = executor.submit(grade_checkpoint_2)
                cp3_future = executor.submit(grade_checkpoint_3)
                cp4_future = executor.submit(grade_checkpoint_4)

                # Collect results in order
                cp1 = cp1_future.result()
                cp2 = cp2_future.result()
                cp3 = cp3_future.result()
                cp4 = cp4_future.result()

                checkpoints.append(cp1)
                checkpoints.append(cp2)
                checkpoints.append(cp3)
                checkpoints.append(cp4)

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return a failed result
        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0, category=StepCategory.EXECUTION_ERROR)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image replacement presentation task")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                       help="Google Slides presentation ID to evaluate")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id
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
