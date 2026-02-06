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
    keywords_match_robust,
    keyword_exact_match,
)
from src.browsergym.eval.eval_utils.slides_utils import (
    extract_slide_text,
    extract_title_text,
    extract_slide_images,
    get_slide_background_color,
    colors_are_different,
    extract_slide_images,
    download_slide_image
)
from src.browsergym.eval.eval_utils.parallel_utils import parallel_execute
from src.browsergym.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.eval.eval_utils.models import load_model

# Task-specific helpers
from src.browsergym.eval.tasks.slides_42_personal_none_product_comparison.utils import (
    text_matches_style,
    extract_device_info_with_llm,
    content_is_valid
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


def grade_checkpoint_2():
    """
    Checkpoint 2 (3pt): The Challenge and the goal slides meet requirements.

    Outcome Evaluation:
    - Title is similar to "The challenge and the goal"
    - At least one line in the slide body explains the challenge of the search.
    - At least one line in the slide body explains the goal of the search.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Challenge & Goal")

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Challenge and Goal Slide Exists", False, 1,
                          "No slides found in presentation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Slide 2 is index 1
    slides = presentation_data['slides']
    if len(slides) < 2:
        checkpoint.add_step("Challenge and Goal Slide Exists", False, 1, "Challenge slide not found or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[1]
    # title_text = extract_title_text(slide)

    # # Step 1: Title similar to 'The Challenge & The Goal'
    # title_keywords = ["challenge", "goal"]
    # step_start = time.time()
    # title_matches = keywords_match_robust(title_text, title_keywords, substring=True)
    # checkpoint.add_step("Title Similar", bool(title_matches), 1, "Slide title similar to 'The Challenge & The Goal' found" if title_matches else "No appropriate title found", execution_time=time.time() - step_start)

    # Step 2 & 3: At least one line explaining challenge and one explaining goal
    step_start = time.time()
    slide_text = extract_slide_text(slide)
    if slide_text:
        if model is None:
                model = load_model(model_id)
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant who evaluates whether the text describes at least one challenge in choosing an electronic device for a new college student. Response with ONLY 'yes' or 'no'."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Is there at least one challenge in choosing an electronic device for a new college student in this text?\n\nText: {slide_text}"}]
                }
            ]
            
            response = model(messages).strip().lower()
            checkpoint.add_step("Explains Challenge", 'yes' in response, 2, "Found challenge explanation" if 'yes' in response else "No challenge explanation found", execution_time=time.time() - step_start)
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant who evaluates whether the text describes at least one goal in choosing an electronic device for a new college student. Response with ONLY 'yes' or 'no'."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Is there at least one goal in choosing an electronic device for a new college student in this text?\n\nText: {slide_text}"}]
                }
            ]
            response = model(messages).strip().lower()
            checkpoint.add_step("Explains Goal", 'yes' in response, 3, "Found goal explanation" if 'yes' in response else "No goal explanation found", execution_time=time.time() - step_start)
            
        except Exception as e:
            print(f"LLM failed to evaluate slide text: {e}")
    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (5pt): All evaluation criteria are listed.

    Outcome Evaluation:
    - Battery life found.
    - Weight found.
    - Memory capacity found.
    - Budget considerations found.
    - Processor.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Evaluation Criteria")

    if not presentation_data or 'slides' not in presentation_data:
        checkpoint.add_step("Criteria Slide Exists", False, 1, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Slide 3 index 2
    slides = presentation_data['slides']
    if len(slides) < 3:
        checkpoint.add_step("Criteria Slide Exists", False, 1, "Criteria slide not found or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    
    slide = slides[2]
    slide_text = extract_slide_text(slide)

    category_keyword_list = [
        ["Battery life", "battery", "lasting", "hours"], 
        ["Weight", "lbs", "portability"], 
        ["Memory", 'ram'], 
        ["Budget", "price", "cost", "affordibility"], 
        ["Processor", "cpu", "speed", "performance"]
    ]
    for i, category_keywords in enumerate(category_keyword_list):
        step_start = time.time()
        is_valid = keywords_match_robust(slide_text, category_keywords[0], model=None, substring=True)
        checkpoint.add_step(category_keywords[0], bool(is_valid), i + 1, f"Found {category_keywords[0]}" if bool(is_valid) else f"Missing {category_keywords[0]}", execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_6():
    """
    Checkpoint 6 (3pt): Recommendation slide meets all requirements.

    Outcome Evaluation:
    - All three devices found in the slide.
    - Summaries aligns with the comparison data.
    - Recommendations based on different student styles are provided.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Recommendation Slide")

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Recommendation Slide Exists", False, 1, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    slides = presentation_data.get('slides', [])
    if len(slides) < 7:
        checkpoint.add_step("Recommendation Slide Exists", False, 1, "Recommendation slide missing or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_start = time.time()
    slide = slides[7]
    slide_text = extract_slide_text(slide)
    
    # Step 1: Summary for each device
    if slide_text:
        if model is None:
                model = load_model(model_id)
        device_map = extract_device_info_with_llm(slide_text, model)
        device_names = list(device_map.keys())    
        missing_devices = ""
        print(f"    Phase 1. Validating devices:")
        for device in gold_devices:
            matches = keywords_match_robust(device_names, device.split(), substring=True)
            if not bool(matches):
                missing_devices += device + "; "
                print(f"        Missing device: {device}")
            else:
                print(f"        Found device: {device}")

        checkpoint.add_step("All Devices Mentioned", len(missing_devices) == 0, 1, "All three correct devices discussed in the slide" if len(missing_devices) == 0 else f"Missing information for: {missing_devices}", execution_time=time.time() - step_start)
    
        comparison_slide = slides[6]
        comparison_text = extract_slide_text(comparison_slide)
    
        summary_tasks = []
        recommendation_tasks = []
        missing_sum = 0
        missing_rec = 0
        print(f"    Phase 2. Collecting Summaries and Recommendations for Evaluation Tasks...")
        for device in device_map:
            device_info = device_map[device]
            summary = device_info["summary"]
            if summary:
                sum_task_text = f"Is the following summary for {device} consistent with the source information?\n\nSource: {comparison_text}\n\nSummary: {summary}"
            
                summary_tasks.append({
                    'id': f'{device}',
                    'func': content_is_valid,
                    'args': (sum_task_text,model)
                })
            else:
                missing_sum += 1
                print(f"        Missing summary for {device}")
            
            recommendation = device_info["recommendation"]
            if recommendation:
                rec_task_text = f"Is the following recommendation of {device} based on a student style?\n\nRecommendation: {recommendation}"
                recommendation_tasks.append({
                    'id': f'{device}',
                    'func': content_is_valid,
                    'args': (rec_task_text,model)
                })
            else:
                missing_rec += 1
                print(f"        Missing recommendation for {device}")
        
        step_start = time.time()
        print(f"    Phase 3. Evaluating Summaries:")
        invalid_summaries = 0
        if summary_tasks:
            summary_eval_results = parallel_execute(summary_tasks, max_workers=3)
            for device, isValid in summary_eval_results.items():
                if isValid:
                    print(f"        Summary for {device} is consistent with source information.")
                else:
                    invalid_summaries += 1
                    print(f"        Summary for {device} is NOT consistent with source information.")
        valid_summaries = invalid_summaries == 0 and missing_sum == 0
        checkpoint.add_step("Summaries Align with Comparison Data", valid_summaries, 2, "All summaries are consistent with source information" if valid_summaries else f"Some device summaries are inconsistent with the source information or missing", execution_time=time.time() - step_start)
        
        step_start = time.time()
        print(f"    Phase 4. Evaluating Recommendations:")
        invalid_recommendations = 0
        if recommendation_tasks:
            rec_eval_results = parallel_execute(recommendation_tasks, max_workers=3)
            for device, isValid in rec_eval_results.items():
                if isValid:
                    print(f"        Recommendation for {device} is based on student style.")
                else:
                    invalid_recommendations += 1
                    print(f"        Recommendation for {device} is not based on student style.")
        valid_recommendations = invalid_recommendations == 0 and missing_rec == 0
        checkpoint.add_step("Recommendations Based on Student Styles", valid_recommendations, 3, "All recommendations are valid" if valid_recommendations else f"Some device recommendations are not valid or missing", execution_time=time.time() - step_start)
        

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
        # checkpoints.append(grade_checkpoint_4())
        # checkpoints.append(grade_checkpoint_5())
        checkpoints.append(grade_checkpoint_6())
        # checkpoints.append(grade_checkpoint_7())

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