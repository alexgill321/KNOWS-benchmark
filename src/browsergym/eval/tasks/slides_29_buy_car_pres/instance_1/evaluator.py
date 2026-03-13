import os
import sys
import time
import argparse
import shutil
from typing import List, Dict, Any


# base path resolution
BASE_PATH = None
if os.path.exists("/app/src"):
    BASE_PATH = "/app"
elif os.path.exists("/scratch"):
    BASE_PATH = "/scratch/general/vast/USER/Agent-Benchmark/"
else:
    BASE_PATH = os.getcwd()
sys.path.append(BASE_PATH)

# imports from eval_utils
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.text_utils import (
    keyword_exact_match,
    keywords_match_robust
)
from src.browsergym.eval.eval_utils.slides_utils import (
    extract_slide_links,
    extract_slide_text,
    extract_text_boxes_from_slide,
    extract_title_text,
    extract_slide_images,
    download_slide_image,
    get_text_style_from_shape,
    is_text_big,
)
from src.browsergym.eval.eval_utils.parallel_utils import parallel_download, parallel_execute
from src.browsergym.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.web_utils import download_page_images, fetch_page_text_content

from src.browsergym.eval.tasks.slides_29_buy_car_pres.utils import (
    evaluate_single_car,
    extract_info_with_llm,
    evaluate_with_llm,
    find_kbb_url_for_car,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/slides_29_buy_car_pres/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

DRIVE_SERVICE, SLIDES_SERVICE = initialize_google_services(service_type="slides")

# Global
model = None
model_id = "gemini-2.5-flash-google-ai"
presentation_id = None
presentation_data = None
gold_cars = []


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

    # Fetch presentation data
    presentation_data = SLIDES_SERVICE.presentations().get(presentationId=presentation_id).execute()
    

def grade_checkpoint_1():
    """
    Checkpoint 1 (2pt): Title slide contains the correct text.

    Outcome Evaluation:
    - Exact match on "Comparing Different Cool Cars to Buy" found.
    - The matched text has a font size of at least 30pt.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Title Slide")

    if not presentation_data or 'slides' not in presentation_data or not presentation_data['slides']:
        checkpoint.add_step("Title Match", False, 1, "No slides found in presentation", execution_time=0)
        checkpoint.add_step("Title Font Size at least 30pt", False, 2, "No slides found in presentation", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    title_slide = presentation_data['slides'][0]

    # Step 1: Exact match on title text
    step_start = time.time()
    expected_title = "Comparing Different Cool Cars to Buy"
    title_text = extract_title_text(title_slide)
    title_found = keyword_exact_match(title_text, expected_title)
    font_big = False
    text_boxes = extract_text_boxes_from_slide(title_slide)
    
    for text_box in text_boxes:
        if keyword_exact_match(expected_title, text_box.get('text', '')):
            if not title_found:
                title_text = text_box.get('text', '')
                title_found = True
            element = text_box.get('element', {})
            title_style = get_text_style_from_shape(element.get('shape',{}))
            font_big = is_text_big(title_style, min_pt=30)
            break

    checkpoint.add_step("Title Match", title_found, 1,
                       f"Found exact title '{expected_title}'"  if title_found
                       else f"Title does not match. Found: '{title_text}'",
                       execution_time=time.time() - step_start)


    checkpoint.add_step("Title Font Size at least 30pt", font_big, 2,
                       "The title font size is at least 30pt" if font_big
                       else "The title font size is less than 30pt",
                       execution_time=time.time() - step_start)


    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2(browsing_history=None):
    """
    Checkpoint 2 (6pt): Car content slides exist and have correct cars.

    Outcome Evaluation:
    - The browsing history contains a visit to the article talking about 2023 best minivans.
    - There are at least 5 slides, each contains the name of a car from the article (5pts total).
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model, gold_cars
    if model is None:
        model = load_model(model_id)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Car Content Slides")

    # Step 1: Browsing history contains a visit to minivan article
    step_start = time.time()
    minivan_url = ""
    if browsing_history:
        minivan_url = keywords_match_robust(browsing_history, keywords=["minivans", "cars", "trucks", "vans"], model=model, description="best 2023 minivans article")

    checkpoint.add_step("Article Visit", bool(minivan_url), 1,
                       "Browsing history contains visit to a valid article about best minivans" if bool(minivan_url)
                       else "No visit to minivan article found in browsing history or failed to fetch article content from given url",
                       execution_time=time.time() - step_start)
    
    # Step 2: At least 5 slides, each containing the name of a gold car (5pts)
    step_start = time.time()
    
    if not presentation_data or 'slides' not in presentation_data:
        checkpoint.add_step("At Least 5 Car Slides", False, 2, "No slides found in presentation",
                           score=0, max_score=5, execution_time=time.time() - step_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data['slides']
    if len(slides) < 2:
        checkpoint.add_step("At Least 5 Car Slides", False, 2,
                           f"Not enough slides in presentation (found {len(slides)})",
                           score=0, max_score=5, execution_time=time.time() - step_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Match gold cars to slides using robust keyword matching
#    if not gold_cars and minivan_url:
#        article_content = fetch_page_text_content(minivan_url)
#        if article_content:
#            article_truncated = article_content[0][:15000]

#            task_text = f"""Extract the list of the best 2023 minivans from the following article. 
        
#Respond ONLY with this exact format:
 
#    ["<vehicle 1>", "<vehicle 2>", ..., "<vehicle n>"]

# Return an empty list if no minivans are found.

#Article: 

#{article_truncated}

#"""
        
#            gold_cars = extract_info_with_llm(task_text, model)

    matched_cars = set()
    for i in range(1, min(len(slides), 6)): 
        slide = slides[i]
        title_text = extract_title_text(slide)
        if not title_text.strip():
            continue
        task_text = f"""
            Does the following slide title denotes that the slide is about a specific 2023 minivan model? Returns the name of the minivan if found, otherwise returns an empty string.

            Slide title: {title_text}
        """
        
        vehicle_from_title = evaluate_with_llm(task_text, model, return_type="str")
        if vehicle_from_title:
            matched_cars.add(vehicle_from_title)

    num_matched = len(matched_cars)
    checkpoint.add_step("At Least 5 Car Slides", num_matched >= 5, 2,
                       f"Found {num_matched}/5 cars mentioned: {', '.join(matched_cars)}",
                       score=min(num_matched,5), max_score=5,
                       execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3(browsing_history=None):
     """
     Checkpoint 3 (50pt): Car slides meet the requirements (x5 minivans, 10 pts each).

     Outcome Evaluation (repeats for 5 minivans):
     - The browsing history contains a visit to the corresponding KBB vehicle page.
     - The make and model of the minivan are listed in the slide title.
     - A picture of the correct model is found.
     - The minivan picture takes up at least 50% of the slide.
     - Sticker price matches the price listed on Kelly Blue Book.
     - Fuel efficiency stat matches the listed value on Kelly Blue Book.
     - Horsepower stat matches the listed value on Kelly Blue Book.
     - A URL to a user review platform is provided.
     - The browsing history contains a visit to the user review platform URL.
     - User average rating matches the listed value on the user review platform.
     """
     print("----------------- CHECKPOINT 3 ----------------")
     global model
     checkpoint_start = time.time()

     NUM_CARS = 5
     checkpoint = Checkpoint(total=50, result=0, name="Car Slides Validation")

     step_names = [
         "KBB Visit in History",
         "Make and Model Listed as Title",
         "Correct Model Picture",
         "Picture >= 50% of Slide",
         "Sticker Price Matches KBB",
         "Fuel Efficiency Matches KBB",
         "Horsepower Matches KBB",
         "Review URL Provided",
         "Review URL in History",
         "Rating Matches Review Platform",
     ]

     if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) < 3:
         step_id = 1
         for car_idx in range(NUM_CARS):
             for name in step_names:
                 checkpoint.add_step(f"Car {car_idx+1} - {name}", False, step_id,
                                    "Insufficient slides in presentation", execution_time=0)
                 step_id += 1
         checkpoint.execution_time = time.time() - checkpoint_start
         return checkpoint

     if model is None:
         model = load_model(model_id)

     slides = presentation_data['slides']
     car_slides = slides[1:] if len(slides) >= 2 else []

     # Get slide dimensions for coverage calculation
     page_size = presentation_data.get('pageSize', {})
     slide_width_emu = page_size.get('width', {}).get('magnitude', 9144000)
     slide_height_emu = page_size.get('height', {}).get('magnitude', 5143500)

     # Phase 1: Extract car info from all slides in parallel
     print("Phase 1: Extracting car information from slides...")
     extract_tasks = []
     for idx, slide in enumerate(car_slides[:NUM_CARS]):
         slide_text = extract_slide_text(slide, "\n")
         task_text = f"""Extract the following car information from this slide text.

 Respond ONLY with this exact JSON format:
 {{
     "make_model": "<car make and model>",
     "sticker_price": "<price as listed>",
     "fuel_efficiency": "<MPG or fuel efficiency value as listed>",
     "horsepower": "<HP value as listed>",
     "user_rating": "<user rating as listed>"
 }}

 If a field is not found, use an empty string.

 Slide text:
 {slide_text}"""
         extract_tasks.append({
             'id': idx,
             'func': extract_info_with_llm,
             'args': (task_text, model)
         })

     car_infos = {}
     if extract_tasks:
         car_infos = parallel_execute(extract_tasks)

     # Phase 2: Identify review URLs from slide links and KBB URLs from browsing history
     print("Phase 2: Matching review and KBB URLs...")
     kbb_urls = {}     # car_idx -> kbb_url
     review_urls = {}  # car_idx -> review_url

     for idx, slide in enumerate(car_slides[:NUM_CARS]):
         car_info = car_infos.get(idx) or {}
         make_model = car_info.get('make_model', '')

         # Match KBB URL from browsing history based on car make/model
         if browsing_history and make_model:
             kbb_url = find_kbb_url_for_car(browsing_history, make_model)
             if kbb_url:
                 kbb_urls[idx] = kbb_url

         # Extract review URL from slide links (first link found)
         slide_links = extract_slide_links(slide)
         if slide_links:
             review_urls[idx] = slide_links[0]

     # Phase 3: Fetch unique page contents in parallel (deduplicate URLs)
     print("Phase 3: Fetching page contents...")
     urls_to_fetch = set()
     for idx in range(min(NUM_CARS, len(car_slides))):
         if idx in kbb_urls:
             urls_to_fetch.add(kbb_urls[idx])
         if idx in review_urls:
             urls_to_fetch.add(review_urls[idx])

     web_content_tasks = [
         {'id': url, 'func': fetch_page_text_content, 'args': (url,)}
         for url in urls_to_fetch
     ]

     web_contents = {}
     if web_content_tasks:
         web_contents = parallel_download(web_content_tasks, max_workers=5, use_rate_limit=False)

     # Phase 4: Download example images from KBB pages
     print("Phase 4: Downloading example images from KBB pages...")
     kbb_example_dirs = {}  # slide_idx -> example folder path
     kbb_img_tasks = []
     for idx in range(min(NUM_CARS, len(car_slides))):
         if idx not in kbb_urls:
             continue
         example_dir = os.path.join(DATA_DIR, f"example_{idx}")
         kbb_example_dirs[idx] = example_dir
         kbb_img_tasks.append({
             'id': idx,
             'func': download_page_images,
             'args': (kbb_urls[idx], example_dir)
         })

     if kbb_img_tasks:
         parallel_execute(kbb_img_tasks)

     # Remove empty example dirs (no images downloaded)
     for idx in list(kbb_example_dirs.keys()):
         example_dir = kbb_example_dirs[idx]
         if not os.path.exists(example_dir) or not os.listdir(example_dir):
             kbb_example_dirs.pop(idx)
             if os.path.exists(example_dir):
                 shutil.rmtree(example_dir)

     # Phase 5: Download slide images in parallel
     print("Phase 5: Downloading slide images...")
     os.makedirs(DATA_DIR, exist_ok=True)

     # Collect all image URLs with slide_idx and image_idx
     img_download_tasks = []
     for idx, slide in enumerate(car_slides[:NUM_CARS]):
         images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
         for img_idx, img_info in enumerate(images):
             if img_info['contentUrl']:
                 img_download_tasks.append({
                     'id': f'{idx}_{img_idx}',
                     'func': download_slide_image,
                     'args': (img_info['contentUrl'],)
                 })

     downloaded_images = {}
     if img_download_tasks:
         downloaded_images = parallel_execute(img_download_tasks)

     # Save downloaded images to temp dirs, grouped by slide index
     slide_image_dirs = {}  # car_idx -> temp_dir path
     for task_id, img in downloaded_images.items():
         if img is None:
             continue
         slide_idx = int(task_id.split('_')[0])
         temp_dir = os.path.join(DATA_DIR, f"temp_car_{slide_idx}")
         os.makedirs(temp_dir, exist_ok=True)
         temp_path = os.path.join(temp_dir, f"temp_image_{task_id}.png")
         img.save(temp_path)
         slide_image_dirs[slide_idx] = temp_dir

     # Phase 6: Evaluate each car slide in parallel
     print("Phase 6: Evaluating each car slide in parallel...")

     # Run all car evaluations in parallel
     eval_tasks = [
         {'id': car_idx, 'func': evaluate_single_car, 'args': (car_idx, car_slides, step_names, car_infos, kbb_urls, review_urls, web_contents, browsing_history, slide_image_dirs, kbb_example_dirs, slide_width_emu, slide_height_emu, model)}
         for car_idx in range(NUM_CARS)
     ]
     car_results = parallel_execute(eval_tasks, max_workers=NUM_CARS)

     # Add steps to checkpoint in order (car 0 steps, then car 1 steps, etc.)
     step_id = 1
     for car_idx in range(NUM_CARS):
         car_steps = car_results.get(car_idx, [])
         if not car_steps:
             # Fallback if parallel execution failed for this car
             for name in step_names:
                 checkpoint.add_step(f"Car {car_idx+1} - {name}", False, step_id,
                                    "Evaluation failed", execution_time=0)
                 step_id += 1
         else:
             for step in car_steps:
                 checkpoint.add_step(step["name"], step["success"], step_id,
                                    step["detail"], execution_time=step["execution_time"])
                 step_id += 1

     # Cleanup all temp image and example directories
     for temp_dir in slide_image_dirs.values():
         if temp_dir and os.path.exists(temp_dir):
             shutil.rmtree(temp_dir)
     for example_dir in kbb_example_dirs.values():
         if example_dir and os.path.exists(example_dir):
             shutil.rmtree(example_dir)

     checkpoint.execution_time = time.time() - checkpoint_start
     return checkpoint


def grade_checkpoint_4():
     """
     Checkpoint 4 (5pt): The last slide meets the requirements.

     Outcome Evaluation:
     - The slide title denotes that the slide contains the best car stats.
     - The correct lowest-price car is listed.
     - The correct highest MPG car is listed.
     - The correct highest horsepower car is listed.
     - The most highly-rated car is listed.
     """
     print("----------------- CHECKPOINT 4 ----------------")
     global model
     checkpoint_start = time.time()
     checkpoint = Checkpoint(total=5, result=0, name="Summary Slide")

     summary_step_names = [
         "Title Denotes Best Car Stats",
         "Lowest Price Car",
         "Highest MPG Car",
         "Highest Horsepower Car",
         "Most Highly Rated Car",
     ]

     if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) < 3:
         for i, name in enumerate(summary_step_names, 1):
             checkpoint.add_step(name, False, i, "Insufficient slides in presentation", execution_time=0)
         checkpoint.execution_time = time.time() - checkpoint_start
         return checkpoint

     if model is None:
         model = load_model(model_id)

     slides = presentation_data['slides']
     last_slide = slides[-1]
     last_slide_text = extract_slide_text(last_slide)

     # Step 1: The slide title denotes that the slide contains the best car stats
     step_start = time.time()
     last_slide_title = extract_title_text(last_slide)
     title_denotes_best = False
     if last_slide_title.strip():
         title_denotes_best = evaluate_with_llm(
             f"Does this slide title indicate that the slide contains the best car stats, best categories, or a summary/comparison of which cars performed best?\n\nSlide title: {last_slide_title}",
             model, return_type="bool"
         )
     checkpoint.add_step("Title Denotes Best Car Stats", bool(title_denotes_best), 1,
                        f"Title '{last_slide_title}' denotes best car stats" if title_denotes_best
                        else f"Title '{last_slide_title}' does not denote best car stats",
                        execution_time=time.time() - step_start)

     # Extract car stats from all car slides for comparison
     car_slides = slides[1:-1] if len(slides) > 2 else []
     extract_tasks = []
     for idx, slide in enumerate(car_slides):
         slide_text = extract_slide_text(slide, "\n")
         task_text = f"""Extract numerical car stats from this slide text.

 Respond ONLY with JSON:
 {{
     "make_model": "<car name>",
     "price_numeric": <price as number without currency symbol or commas, 0 if not found>,
     "mpg_numeric": <mpg or fuel efficiency as number, 0 if not found>,
     "hp_numeric": <horsepower as number, 0 if not found>,
     "rating_numeric": <user rating as number, 0 if not found>
 }}

 Slide text:
 {slide_text}"""
         extract_tasks.append({
             'id': idx,
             'func': extract_info_with_llm,
             'args': (task_text, model)
         })

     car_stats = {}
     if extract_tasks:
         car_stats = parallel_execute(extract_tasks)

     # Determine correct winners for each category
     lowest_price = {"name": "", "value": float('inf')}
     highest_mpg = {"name": "", "value": 0}
     highest_hp = {"name": "", "value": 0}
     highest_rating = {"name": "", "value": 0}

     for idx, stats in car_stats.items():
         if not stats:
             continue
         name = stats.get('make_model', '')

         price = stats.get('price_numeric', 0) or 0
         mpg = stats.get('mpg_numeric', 0) or 0
         hp = stats.get('hp_numeric', 0) or 0
         rating = stats.get('rating_numeric', 0) or 0

         try:
             price = float(price)
             if 0 < price < lowest_price["value"]:
                 lowest_price = {"name": name, "value": price}
         except (ValueError, TypeError):
             pass

         try:
             mpg = float(mpg)
             if mpg > highest_mpg["value"]:
                 highest_mpg = {"name": name, "value": mpg}
         except (ValueError, TypeError):
             pass

         try:
             hp = float(hp)
             if hp > highest_hp["value"]:
                 highest_hp = {"name": name, "value": hp}
         except (ValueError, TypeError):
             pass

         try:
             rating = float(rating)
             if rating > highest_rating["value"]:
                 highest_rating = {"name": name, "value": rating}
         except (ValueError, TypeError):
             pass


     step_start = time.time()
     expected_lines = [
         f"Lowest Price: {lowest_price['name']}",
         f"Highest MPG: {highest_mpg['name']}",
         f"Highest Horsepower: {highest_hp['name']}",
         f"Most Highly Rated: {highest_rating['name']}",
     ]

     llm_results = {}
     expected_text = "\n".join(expected_lines)
     llm_results = evaluate_with_llm(
        f"For each category below, does the summary slide list the same car (or a name that refers to the same car)?\n\n"
        f"Expected winners:\n{expected_text}\n\n"
        f"Summary slide text:\n{last_slide_text}\n\n"
        f"Respond ONLY with JSON mapping each category key to true or false:\n"
        f'{{"lowest_price": true/false, "highest_mpg": true/false, "highest_hp": true/false, "highest_rating": true/false}}',
        model, return_type="json"
     ) or {}
     step_elapsed = time.time() - step_start

     categories = [
         ("lowest_price", summary_step_names[1], lowest_price["name"]),
         ("highest_mpg", summary_step_names[2], highest_mpg["name"]),
         ("highest_hp", summary_step_names[3], highest_hp["name"]),
         ("highest_rating", summary_step_names[4], highest_rating["name"]),
     ]

     for i, (category_key, step_name, expected_car) in enumerate(categories, 2):
         if not expected_car:
             checkpoint.add_step(step_name, False, i,
                                f"Could not determine winner for {step_name} from car slides",
                                execution_time=0)
             continue
         correct_winner = bool(llm_results.get(category_key, False))
         checkpoint.add_step(step_name, correct_winner, i,
                            f"The correct car is listed for the category {step_name}" if correct_winner
                            else f"An incorrect or no car is listed for the category {step_name}",
                            execution_time=step_elapsed)

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
        checkpoints.append(grade_checkpoint_2(browsing_history))
        checkpoints.append(grade_checkpoint_3(browsing_history))
        checkpoints.append(grade_checkpoint_4())

        total_execution_time = time.time() - total_start
        return Result(checkpoints, total_execution_time=total_execution_time)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        failed = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed.add_step("Evaluation", False, 1, f"Fatal error: {e}", execution_time=0)
        return Result([failed], total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate car comparison presentation")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Slides presentation ID to evaluate")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    step_start = time.time()

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        cached_models=args.cached_models,
        browsing_history=args.browsing_history
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "✓" if step["success"] else "✗"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - step_start:.2f} seconds")
