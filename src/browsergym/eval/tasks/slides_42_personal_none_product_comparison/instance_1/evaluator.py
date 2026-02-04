import os
import sys
import time
import argparse
import requests
from typing import List, Dict, Any, Optional
import shutil

# base path helper

def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# imports from eval_utils
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.text_utils import (
    keyword_exact_match,
)
from src.browsergym.eval.eval_utils.slides_utils import (
    extract_slide_text,
    extract_slide_images,
    get_slide_background_color,
    colors_are_different,
    extract_slide_images,
    download_slide_image
)
from src.browsergym.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.parallel_utils import parallel_download, fast_parallel_vlm_calls

# Task-specific helpers
from src.browsergym.eval.tasks.slides_42_personal_none_product_comparison.utils import (
    text_matches_style
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/slides_42_personal_none_product_comparison/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SLIDES_SERVICE = initialize_google_services(service_type="slides")

# Global
presentation_id = None
presentation_data = None
gold_devices = None

def fetch_page_text(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            # strip scripts/styles simply
            text = r.text
            return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None


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
    Checkpoint 1 (5pt): Title slide has all required elements.

    Outcome Evaluation:
    - Exact match on "A Gift for Kathy!" found.
    - Title is in bold.
    - Subtitle correctly lists all 3 device options from the gold list.
    - Image representing the University of Utah found.
    - The university's official colors are used.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Title Slide Validation")

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Title Slide Exists", False, 1,
                          "No slides found in presentation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Assume first slide is the title slide
    title_slide = presentation_data['slides'][0]
    slide_text = extract_slide_text(title_slide)

    # Step 1: Exact match on 'A Gift for Kathy!'
    step_start = time.time()
    title_found = keyword_exact_match(slide_text, "A Gift for Kathy!", substring=True)
    title_bold = title_found and text_matches_style(title_slide, "A Gift for Kathy!", "bold")
    
    checkpoint.add_step("Title Match", title_found, 1, "Found exact title 'A Gift for Kathy!'" if title_found else "Title does not exactly match 'A Gift for Kathy!'", execution_time=time.time() - step_start)

    # Step 2: Title is bold
    step_start = time.time()
    checkpoint.add_step("Title Is Bold", title_bold, 2, "Title text is bold" if title_bold else "Title text not bold or could not determine", execution_time=time.time() - step_start)

    # Step 3: Subtitle lists all 3 device options
    step_start = time.time()
    devices_matched = True
    unmatched = ""
    for device in gold_devices:
        if not keyword_exact_match(slide_text, device, substring=True):
            devices_matched = False
            unmatched += device + "; "
        
    checkpoint.add_step("Subtitle Includes All Devices", devices_matched, 3, "Subtitle lists all 3 devices" if devices_matched else f"Subtitle missing devices: {unmatched}", execution_time=time.time() - step_start)

    # Step 4: Image representing University of Utah found
    step_start = time.time()
    images = extract_slide_images(title_slide, presentation_id, SLIDES_SERVICE)
    uni_image_valid = False
    
    global model
    if model is None:
        model = load_model(model_id)
        
    # Create temp directory for downloaded images
    temp_dir = os.path.join(DATA_DIR, "temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        # Download and save each image temporarily
        for idx, img_info in enumerate(images):
            if img_info['contentUrl']:
                img = download_slide_image(img_info['contentUrl'])
                if img:
                    temp_img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                    img.save(temp_img_path)
                    
        # Use binary_judge_image to check if any image is the Red Rising book cover
        if os.listdir(temp_dir):
            matching_image = binary_judge_image(
                model,
                temp_dir,
                "Is this an image of the University of Utah campus?"
            )

            if matching_image:
                uni_image_valid = True
    finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    checkpoint.add_step("University Image", uni_image_valid, 4, "Found an image representing University of Utah" if uni_image_valid else "No valid University of Utah image found", execution_time=time.time() - step_start)

    # Step 5: University colors used (red prominent)
    official_color = {
            'r': 0.75,
            'g': 0.0,
            'b': 0.0
        }
    step_start = time.time()
    bg_color = get_slide_background_color(title_slide, presentation_data)
    # also check title text color via slides element scanning
    color_valid = False
    if bg_color:
        # if bg color has strong red component
        if not colors_are_different(bg_color, official_color, threshold=0.1):
            color_valid = True

    # fallback: presence of red text in title/subtitle via fuzzy check for 'red' in style not available here
    checkpoint.add_step("University Colors", bool(color_valid), 5, "University official color found on slide" if color_valid else "No strong University official color detected", execution_time=time.time() - step_start)

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

        total_execution_time = time.time() - total_start
        return Result(checkpoints, total_execution_time=total_execution_time)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        failed = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed.add_step("Evaluation", False, 1, f"Fatal error: {e}", execution_time=0)
        return Result([failed], total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate personal product comparison presentation")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Slides presentation ID to evaluate")
    # parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    args = parser.parse_args()

    start_time = time.time()

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
            status = "" if step["success"] else ""
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")