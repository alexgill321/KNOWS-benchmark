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
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.text_utils import keyword_exact_match
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_text,
    extract_text_boxes_from_slide,
    extract_title_text,
    extract_slide_images,
    download_slide_image,
    get_text_style_from_shape,
    is_text_big,
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download, parallel_execute
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.web_utils import download_page_images, fetch_page_text_content, fetch_with_fallbacks_extended

from src.browsergym.knows.eval.tasks.slides_29_buy_car_pres.utils import (
    CP3_PER_CAR_STEPS,
    CP_STEP_SHAPES,
    CP4_WINNER_KEY_MAP,
    evaluate_single_car,
    evaluate_slide_for_cars,
    evaluate_with_llm,
    expected_car_in_text,
    find_year_category_article,
    extract_all_slide_urls,
    extract_info_with_llm,
    find_kbb_url_for_car,
    has_stat_data,
    is_truthy_flag,
    make_failure_checkpoint,
    normalize_json_key,
    parse_task_config,
    pick_review_url,
)

# Constants
TASK_DIR = os.path.join(os.path.dirname(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data/")
model_id = "gemini-2.5-flash-google-ai"

# Per-instance config from task.md
_TASK_CONFIG = parse_task_config(os.path.join(os.path.dirname(__file__), "task.md"))
YEAR = _TASK_CONFIG["year"]
CATEGORY = _TASK_CONFIG["category"]
EXPECTED_TITLE = _TASK_CONFIG["title"]

# Lazy-initialized inside setup_presentation.
SLIDES_SERVICE = None

# Global
model = None
presentation_id = None
presentation_data = None

# Cross-CP caches: populated by CP3, read by CP4. Reset per grade_checkpoints().
_cp3_kbb_urls: Dict[int, str] = {}
_cp3_review_urls: Dict[int, str] = {}
_cp3_web_contents: Dict[str, Any] = {}
_cp3_car_infos: Dict[int, Dict[str, Any]] = {}


def setup_presentation(workspace_doc_id):
    """Lazy-init Slides service + fetch presentation data."""
    global presentation_id, presentation_data, SLIDES_SERVICE
    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")
    if SLIDES_SERVICE is None:
        _, SLIDES_SERVICE = initialize_google_services(service_type="slides")
    print(f"Using workspace presentation ID: {workspace_doc_id}")
    presentation_id = workspace_doc_id
    presentation_data = SLIDES_SERVICE.presentations().get(presentationId=presentation_id).execute()


def grade_checkpoint_1():
    """
    Checkpoint 1 (2pt): Title slide contains the correct text.

    Outcome Evaluation:
    - Exact match on expected title found.
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
    title_text = extract_title_text(title_slide)
    title_found = keyword_exact_match(title_text, EXPECTED_TITLE)
    font_big = False
    text_boxes = extract_text_boxes_from_slide(title_slide)

    for text_box in text_boxes:
        # (#3) substring=True so titles wrapped in extra text still match
        if keyword_exact_match(EXPECTED_TITLE, text_box.get('text', ''), substring=True):
            if not title_found:
                title_text = text_box.get('text', '')
                title_found = True
            # (#5) Only check font if title was found
            element = text_box.get('element', {})
            title_style = get_text_style_from_shape(element.get('shape', {}))
            font_big = is_text_big(title_style, min_pt=30, element=element)
            break

    checkpoint.add_step("Title Match", title_found, 1,
                       f"Found exact title '{EXPECTED_TITLE}'" if title_found
                       else f"Title does not match. Found: '{title_text}'",
                       execution_time=time.time() - step_start)

    checkpoint.add_step("Title Font Size at least 30pt", font_big, 2,
                       "The title font size is at least 30pt" if font_big
                       else "The title font size is less than 30pt" if title_found
                       else "Title not found; font size cannot be checked",
                       execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2(browsing_history=None):
    """
    Checkpoint 2 (6pt): Car content slides exist and have correct cars.

    Outcome Evaluation:
    - The browsing history contains a visit to the article talking about best cars.
    - There are at least 5 slides, each contains the name of a car from the article (5pts total).
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model
    if model is None:
        model = load_model(model_id)

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Car Content Slides")

    # Step 1: Browsing history contains a visit to article
    step_start = time.time()
    # (#6) Parameterized keywords from CATEGORY/YEAR
    article_url = ""
    if browsing_history:
        article_url = find_year_category_article(browsing_history, YEAR, CATEGORY, model)

    checkpoint.add_step("Article Visit", bool(article_url), 1,
                       f"Browsing history contains visit to a valid article about best {CATEGORY}s: {article_url}" if bool(article_url)
                       else f"No visit to {CATEGORY} article found in browsing history",
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

    # (#8) Extract gold-car list from the article (was commented out in original)
    gold_cars_list: List[str] = []
    if article_url:
        try:
            article_content = fetch_page_text_content(article_url)
            article_text = article_content[0] if article_content else None
            if article_text:
                article_truncated = article_text[:15000]
                gold_task = f"""Extract the list of {YEAR} {CATEGORY} models discussed in the article.

Respond ONLY with a JSON array of make/model name strings, e.g.:
["<Make Model>", "<Make Model>"]

Return [] if no {CATEGORY}s are mentioned.

Article content:
{article_truncated}"""
                extracted = extract_info_with_llm(gold_task, model)
                if isinstance(extracted, list):
                    gold_cars_list = [str(c).strip() for c in extracted if str(c).strip()]
                elif isinstance(extracted, dict):
                    for v in extracted.values():
                        if isinstance(v, list):
                            gold_cars_list = [str(c).strip() for c in v if str(c).strip()]
                            break
        except Exception as e:
            print(f"Warning: gold-car extraction error: {e}")

    # (#9, #10) Slide range [1:-1] excludes summary; use evaluate_slide_for_cars with gold list
    matched_cars = set()
    car_candidate_slides = slides[1:-1] if len(slides) >= 3 else slides[1:]
    for i, slide in enumerate(car_candidate_slides[:6]):
        car_name, _err = evaluate_slide_for_cars(i, slide, gold_cars_list, model, CATEGORY, YEAR)
        if car_name:
            matched_cars.add(car_name)

    num_matched = len(matched_cars)
    checkpoint.add_step("At Least 5 Car Slides", num_matched >= 5, 2,
                       f"Found {num_matched}/5 cars mentioned: {', '.join(matched_cars)}",
                       score=min(num_matched, 5), max_score=5,
                       execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3(browsing_history=None):
     """
     Checkpoint 3 (55pt): Car slides meet the requirements (x5 cars, 11 pts each).

     Outcome Evaluation (repeats for 5 cars):
     - The browsing history contains a visit to the corresponding KBB vehicle page.
     - The make and model of the car are listed in the slide title.
     - A picture of the correct model is found.
     - The car picture takes up at least 50% of the slide.
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
     checkpoint = Checkpoint(total=55, result=0, name="Car Slides Validation")

     step_names = CP3_PER_CAR_STEPS

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
     # (#13) Exclude summary slide
     car_slides = slides[1:-1] if len(slides) >= 3 else []

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

     # (#17) Fallback make/model extraction for slides where Phase 1 missed it
     for idx, slide in enumerate(car_slides[:NUM_CARS]):
         car_info = car_infos.get(idx) or {}
         make_model = str(car_info.get("make_model", "") or "").strip()
         if not make_model:
             try:
                 title_text = extract_title_text(slide)
             except Exception as e:
                 print(f"Warning: make/model fallback title extraction error on slide {idx}: {e}")
                 title_text = ""
             slide_text = extract_slide_text(slide, "\n") if not title_text else ""
             if title_text or slide_text:
                 fallback_task = f"""Extract the make and model of the vehicle described on this slide.

Respond ONLY with JSON:
{{
    "make_model": "<car make and model, or empty string if not present>"
}}

Slide title:
{title_text}

Slide text:
{slide_text}"""
                 fallback_result = extract_info_with_llm(fallback_task, model)
                 if isinstance(fallback_result, dict):
                     fallback_make = str(fallback_result.get("make_model", "") or "").strip()
                 elif isinstance(fallback_result, str):
                     fallback_make = fallback_result.strip()
                 else:
                     fallback_make = ""
                 if fallback_make:
                     car_infos.setdefault(idx, {})["make_model"] = fallback_make

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

         # (#19) extract_all_slide_urls catches plain-text URLs; pick_review_url prefers known review sites
         slide_links = extract_all_slide_urls(slide)
         review_url = pick_review_url(slide_links)
         if review_url:
             review_urls[idx] = review_url

     # Phase 3: Fetch unique page contents in parallel (deduplicate URLs)
     print("Phase 3: Fetching page contents...")
     urls_to_fetch = set()
     for idx in range(min(NUM_CARS, len(car_slides))):
         if idx in kbb_urls:
             urls_to_fetch.add(kbb_urls[idx])
         if idx in review_urls:
             urls_to_fetch.add(review_urls[idx])

     web_content_tasks = [
         {'id': url, 'func': fetch_with_fallbacks_extended, 'args': (url,)}
         for url in urls_to_fetch
     ]

     web_contents = {}
     if web_content_tasks:
         web_contents = parallel_download(web_content_tasks, max_workers=5, use_rate_limit=False)

     # (#21) Cache CP3 data for CP4 ground-truth validation
     _cp3_kbb_urls.update(kbb_urls)
     _cp3_review_urls.update(review_urls)
     _cp3_web_contents.update(web_contents)
     for idx, info in car_infos.items():
         if info:
             _cp3_car_infos[idx] = dict(info)

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
         {'id': car_idx, 'func': evaluate_single_car, 'args': (car_idx, car_slides, step_names, car_infos, kbb_urls, review_urls, web_contents, browsing_history, slide_image_dirs, kbb_example_dirs, slide_width_emu, slide_height_emu, model, presentation_data)}
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
             f"Does this slide title indicate or imply that the slide contains the best car stats, best categories, top performers, or a summary/comparison of which cars performed best? Answer Yes for titles that reference 'best', 'top', 'winner', 'summary', 'comparison', 'stats by category', or similar.\n\nSlide title: {last_slide_title}",
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

     # (#38, #39) Fetch KBB and review ground-truth stats from CP3-cached content
     kbb_stats = {}
     for idx, kbb_url in _cp3_kbb_urls.items():
         kbb_result = _cp3_web_contents.get(kbb_url)
         kbb_text = kbb_result[0] if kbb_result else None
         if not kbb_text:
             continue
         kbb_text_capped = kbb_text[:15000]
         kbb_task = f"""Extract numerical car stats from this KBB page text.

 Respond ONLY with JSON:
 {{
     "price_numeric": <starting MSRP price as number, 0 if not found>,
     "mpg_numeric": <combined fuel efficiency as number, 0 if not found>,
     "hp_numeric": <horsepower as number, 0 if not found>
 }}

 KBB page content:
 {kbb_text_capped}"""
         result = extract_info_with_llm(kbb_task, model)
         if result:
             kbb_stats[idx] = result

     review_stats = {}
     for idx, review_url in _cp3_review_urls.items():
         review_result = _cp3_web_contents.get(review_url)
         review_text = review_result[0] if review_result else None
         if not review_text:
             continue
         review_text_capped = review_text[:15000]
         review_task = f"""Extract user-rating data from this review page text.

 Distinguish a community/consumer aggregate from an editorial single-reviewer score:
 only return rating_numeric if you find a clear consumer/user average (e.g. "4.6/5
 from 132 reviews"). If the page only shows an editorial score, set rating_numeric
 to 0 and put any individual numeric ratings you see in candidates.

 Respond ONLY with JSON:
 {{
     "rating_numeric": <consumer/user aggregate as number, 0 if no clear aggregate>,
     "rating_scale": <max scale, e.g. 5 or 10, 0 if unknown>,
     "is_aggregate": <true if rating_numeric is a community/user aggregate, else false>,
     "candidates": [<up to 5 distinct numeric ratings found anywhere on the page>]
 }}

 Review page content:
 {review_text_capped}"""
         result = extract_info_with_llm(review_task, model)
         if result:
             review_stats[idx] = result

     # Determine correct winners for each category
     # Prefer KBB ground-truth for price/MPG/HP; review-platform for rating; fall back to slide values
     lowest_price = {"name": "", "value": float('inf')}
     highest_mpg = {"name": "", "value": 0}
     highest_hp = {"name": "", "value": 0}
     highest_rating = {"name": "", "value": 0}

     # (#43) Union iteration so KBB/review-only data contributes
     contributing_indices = sorted(
         set(car_stats.keys()) | set(kbb_stats.keys()) | set(review_stats.keys())
     )
     for idx in contributing_indices:
         stats = car_stats.get(idx) or {}
         kbb_for_car = kbb_stats.get(idx) or {}
         review_for_car = review_stats.get(idx) or {}
         if not stats and not kbb_for_car and not review_for_car:
             continue

         name = stats.get('make_model', '') or ''
         # (#42) Fallback to CP3 car info if CP4 extraction missed the name
         if not name.strip():
             cp3_info = _cp3_car_infos.get(idx) or {}
             name = str(cp3_info.get('make_model', '') or '').strip()
         if not name.strip():
             continue

         # Prefer KBB values over slide values for price/MPG/HP
         slide_price = stats.get('price_numeric', 0) or 0
         slide_mpg = stats.get('mpg_numeric', 0) or 0
         slide_hp = stats.get('hp_numeric', 0) or 0
         slide_rating = stats.get('rating_numeric', 0) or 0
         price = kbb_for_car.get('price_numeric') or slide_price or 0
         mpg = kbb_for_car.get('mpg_numeric') or slide_mpg or 0
         hp = kbb_for_car.get('hp_numeric') or slide_hp or 0

         # Prefer review-platform aggregate rating
         review_rating_is_aggregate = is_truthy_flag(review_for_car.get('is_aggregate')) and review_for_car.get('rating_numeric')
         if review_rating_is_aggregate:
             rating = review_for_car.get('rating_numeric')
         else:
             plausible_candidates: List[float] = []
             for c in (review_for_car.get('candidates') or []):
                 try:
                     v = float(c)
                     if 0 < v <= 10:
                         plausible_candidates.append(v)
                 except (ValueError, TypeError):
                     continue
             if plausible_candidates:
                 plausible_candidates.sort()
                 rating = plausible_candidates[len(plausible_candidates) // 2]
             elif slide_rating:
                 rating = slide_rating
             else:
                 rating = 0

         # (#40) Plausibility bounds to filter hallucinated values
         try:
             price = float(price)
             if 3_000 <= price <= 5_000_000 and price < lowest_price["value"]:
                 lowest_price = {"name": name, "value": price}
         except (ValueError, TypeError):
             pass

         try:
             mpg = float(mpg)
             if 3 <= mpg <= 500 and mpg > highest_mpg["value"]:
                 highest_mpg = {"name": name, "value": mpg}
         except (ValueError, TypeError):
             pass

         try:
             hp = float(hp)
             if 50 <= hp <= 3_000 and hp > highest_hp["value"]:
                 highest_hp = {"name": name, "value": hp}
         except (ValueError, TypeError):
             pass

         try:
             rating = float(rating)
             if 0 < rating <= 10 and rating > highest_rating["value"]:
                 highest_rating = {"name": name, "value": rating}
         except (ValueError, TypeError):
             pass

     # (#35, #41) Extract winner names from summary slide, then compare programmatically
     step_start = time.time()
     last_slide_title_capped = (last_slide_title or "")[:500]
     last_slide_text_capped = (last_slide_text or "")[:8000]
     winner_prompt = f"""Read this summary slide and identify which car it names for each "best stats" category.

Summary slide title:
{last_slide_title_capped}

Summary slide text:
{last_slide_text_capped}

Notes:
- Match cars by make and model only — IGNORE any 4-digit year prefix.
- The slide may use phrasings like "Best MPG", "Highest fuel economy", "Top horsepower",
  "Most powerful", "Best rated", etc. Map these to the canonical category.
- If a category isn't covered on the slide, return an empty string for that field.

Respond ONLY with this JSON:
{{
    "lowest_price": "<make and model>",
    "highest_mpg": "<make and model>",
    "highest_horsepower": "<make and model>",
    "highest_rating": "<make and model>"
}}"""
     summary_winners = extract_info_with_llm(winner_prompt, model) or {}
     _normalized_winners = {normalize_json_key(k): v for k, v in summary_winners.items()}
     step_elapsed = time.time() - step_start

     categories = [
         ("price", summary_step_names[1], lowest_price["name"]),
         ("mpg", summary_step_names[2], highest_mpg["name"]),
         ("hp", summary_step_names[3], highest_hp["name"]),
         ("rating", summary_step_names[4], highest_rating["name"]),
     ]

     for i, (source_key, step_name, expected_car) in enumerate(categories, 2):
         if not expected_car:
             checkpoint.add_step(step_name, False, i,
                                f"Could not determine winner for {step_name} from car slides",
                                execution_time=0)
             continue

         json_key = CP4_WINNER_KEY_MAP[source_key]
         _raw = _normalized_winners.get(normalize_json_key(json_key))
         listed_car = _raw.strip() if isinstance(_raw, str) else ""

         if not listed_car:
             correct_winner = False
             detail = f"Summary slide did not list a car for {step_name} (expected '{expected_car}')"
         else:
             correct_winner = expected_car_in_text(listed_car, expected_car)
             detail = (f"The correct car '{expected_car}' is listed for {step_name}"
                       if correct_winner
                       else f"Slide listed '{listed_car}' for {step_name} (expected '{expected_car}')")

         checkpoint.add_step(step_name, correct_winner, i, detail, execution_time=step_elapsed)

     checkpoint.execution_time = time.time() - checkpoint_start
     return checkpoint


# (#45, #46) Per-checkpoint exception isolation with full-shape failure checkpoints
def grade_checkpoints(workspace_doc_id: str, cached_models: Dict[str, Any] = None, browsing_history: List[str] = None):
    total_start = time.time()
    checkpoints: List[Checkpoint] = []

    # (#44) Reset cross-checkpoint caches
    global model, _cp3_kbb_urls, _cp3_review_urls, _cp3_web_contents, _cp3_car_infos
    _cp3_kbb_urls = {}
    _cp3_review_urls = {}
    _cp3_web_contents = {}
    _cp3_car_infos = {}

    if cached_models and model_id in cached_models:
        model = cached_models[model_id]

    cp_runners = [
        (grade_checkpoint_1, ()),
        (grade_checkpoint_2, (browsing_history,)),
        (grade_checkpoint_3, (browsing_history,)),
        (grade_checkpoint_4, ()),
    ]

    try:
        setup_presentation(workspace_doc_id)
    except Exception as e:
        print(f"Setup failed: {e}")
        for name, total, step_names in CP_STEP_SHAPES:
            checkpoints.append(make_failure_checkpoint(name, total, step_names, f"Could not load presentation: {e}"))
        return Result(checkpoints, total_execution_time=time.time() - total_start)

    for (cp_func, cp_args), (cp_name, cp_total, cp_step_names) in zip(cp_runners, CP_STEP_SHAPES):
        try:
            checkpoints.append(cp_func(*cp_args))
        except Exception as e:
            print(f"{cp_name} failed: {e}")
            checkpoints.append(make_failure_checkpoint(cp_name, cp_total, cp_step_names, f"Checkpoint raised: {e}"))

    return Result(checkpoints, total_execution_time=time.time() - total_start)


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
