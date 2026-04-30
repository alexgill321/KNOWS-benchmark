import os
import shutil
import sys
import time
from typing import List, Dict, Any

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

from src.browsergym.knows.eval.eval_utils.llm_utils import evaluate_with_llm
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services, extract_text_from_doc
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_images,
    extract_slide_links,
    extract_text_boxes_from_slide,
    extract_title_text,
    is_text_in_title_position,
    get_element_bbox,
    get_slide_dimensions,
    get_image_area_percentage_from_api,
    download_slide_image,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keyword_exact_match
from src.browsergym.knows.eval.eval_utils.image_utils import match_image_tiered
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_text_content, fetch_api_with_retry, download_image_from_url
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute

from src.browsergym.knows.eval.tasks.slides_30_Work_Wikipedia_Photos.utils import evaluate_single_client, name_exact_match, download_image_with_retry

# Constants
TASK_DIR = os.path.normpath(os.path.join(BASE_PATH, "src", "browsergym", "eval", "tasks", "slides_30_Work_Wikipedia_Photos", "instance_1"))
DATA_DIR = os.path.join(TASK_DIR, "data")

DRIVE_SERVICE, SLIDES_SERVICE = initialize_google_services(service_type="slides")
_, DOCS_SERVICE = initialize_google_services(service_type="docs")

# Global
presentation_id = None
presentation_data = None
gold_clients = None

# Model for VLM evaluation
model = None
model_id = "gemini-3-flash-google-ai"

# Client to Wiki Title cache
wiki_urls_cache = {}

def setup_presentation(workspace_doc_id, client_doc_id):
    """Setup presentation processing.

    Args:
        workspace_doc_id (str): Google Slides presentation ID to evaluate.
        client_doc_id (str): Google Doc ID containing the client list.
    """
    global presentation_id, presentation_data, gold_clients

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")
    if not client_doc_id:
        raise ValueError("client_doc_id is required")

    print(f"Using workspace presentation ID: {workspace_doc_id}")
    presentation_id = workspace_doc_id

    # Fetch presentation data
    presentation_data = SLIDES_SERVICE.presentations().get(presentationId=presentation_id).execute()

    # Load gold clients list from Google Doc
    doc_result = extract_text_from_doc(client_doc_id, DOCS_SERVICE)
    if doc_result:
        gold_clients = [line.strip().lower() for line in doc_result.strip().splitlines() if line.strip()]
        csv_path = os.path.join(DATA_DIR, "gold_wikis.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        name, url = parts[0].strip().lower(), parts[1].strip()
                        wiki_urls_cache[name] = url
    else:
        print(f"Warning: Could not extract client list from doc {client_doc_id}")
        gold_clients = []
        


def grade_checkpoint_1():
    """Checkpoint 1 (4pt): Title slide has all required elements.

    Steps:
        1. Exact match on "Why we need new Wikipedia headshots" (1pt)
        2. The wiki image of Tom Hanks is present on the slide (1pt)
        3. The Tom Hanks image is positioned in the top right area (1pt)
        4. The Tom Hanks image takes up more than 50% of the slide (1pt)
    """
    global model
    print("----------------- CHECKPOINT 1 ----------------")
    start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Title Slide")

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Title Text Match", False, 1, details="No slides found in presentation")
        checkpoint.add_step("Tom Hanks Image Present", False, 2, details="No slides found in presentation")
        checkpoint.add_step("Image Top Right Position", False, 3, details="No slides found in presentation")
        checkpoint.add_step("Image Coverage", False, 4, details="No slides found in presentation")
        checkpoint.execution_time = time.time() - start
        return checkpoint

    title_slide = presentation_data['slides'][0]
    slide_width_emu, slide_height_emu = get_slide_dimensions(presentation_data)

    # Step 1: Title text exact match
    step_start = time.time()
    TITLE_TEXT = "Why we need new Wikipedia headshots"
    title_text = extract_title_text(title_slide)
    has_title = keyword_exact_match(title_text, TITLE_TEXT, case_sensitive=False)
    checkpoint.add_step(
        "Title Text Match", has_title, 1,
        details=f"Found exact match for '{TITLE_TEXT}'" if has_title else f"Title '{TITLE_TEXT}' not found",
        execution_time=time.time() - step_start
    )

    # Step 2: Tom Hanks image present (VLM)
    step_start = time.time()
    tom_hanks_img = None
    images = extract_slide_images(title_slide, presentation_id, SLIDES_SERVICE)
    if model is None:
        model = load_model(model_id)

    tom_hanks_found = False
    best_tom_hanks_area = 0
    
    temp_example_dir = os.path.join(DATA_DIR, "temp_example_check")
    os.makedirs(temp_example_dir, exist_ok=True)
    wiki_title = wiki_urls_cache["tom hanks"].rstrip("/").rsplit("/", 1)[-1]
    wiki_img_query = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles={wiki_title}"
    wiki_res = fetch_api_with_retry(wiki_img_query, timeout=10, max_retries=2)
    wiki_img_url = None
    wiki_img_path = None
    if wiki_res:
        pages = wiki_res.get('query', {}).get('pages', {})
        for page_data in pages.values():
            wiki_img_url = page_data.get('original', {}).get('source')
            if wiki_img_url:
                break
    if wiki_img_url:
        wiki_img_path = download_image_with_retry(wiki_img_url, temp_example_dir, timeout=15)
    
    temp_dir = os.path.join(DATA_DIR, "temp_person_check")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        for idx, img_info in enumerate(images):
            if not img_info.get('contentUrl'):
                continue
            img = download_slide_image(img_info['contentUrl'])
            if not img:
                continue
            img_area = img.width * img.height
            if tom_hanks_found and img_area <= best_tom_hanks_area:
                continue
            temp_path = os.path.join(temp_dir, f"slide_img.png")
            img.save(temp_path)
            result = match_image_tiered(temp_path, wiki_img_path, model, "Are these the same image?", 5)[0]
            if result:
                tom_hanks_found = True
                tom_hanks_img = img_info
                best_tom_hanks_area = img_area
            os.remove(temp_path)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(temp_example_dir):
            shutil.rmtree(temp_example_dir)

    checkpoint.add_step(
        "Tom Hanks Wiki Image Present", tom_hanks_found, 2,
        details=f"Tom Hanks wiki image found in slide" if tom_hanks_found else "Tom Hanks wiki image not found",
        execution_time=time.time() - step_start
    )

    # Step 3: Image in top right
    step_start = time.time()
    any_top_right = False
    if tom_hanks_img:
        bbox = get_element_bbox(tom_hanks_img)
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        if center_x > slide_width_emu / 2 and center_y < slide_height_emu / 2:
            any_top_right = True
    checkpoint.add_step(
        "Image Top Right Position", any_top_right, 3,
        details="Image positioned in top right" if any_top_right else "No image in top right area",
        execution_time=time.time() - step_start
    )

    # Step 4: Image covers >50% of slide
    step_start = time.time()
    image_percentage = get_image_area_percentage_from_api(
        title_slide, slide_width_emu, slide_height_emu
    )
    covers_majority = image_percentage > 50.0
    checkpoint.add_step(
        "Image Coverage >50%", covers_majority, 4,
        details=f"Image covers {image_percentage:.1f}% of slide",
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2 (x9 clients, 4pts each): Client slides meet the requirements.

    Steps per client:
        1. Client name at the top of the slide (1pt)
        2. Wikipedia image on the left side (1pt)
        3. A different image of the client on the right side (1pt)
        4. Images approximately symmetric in vertical position (1pt)
    """
    global model
    print("----------------- CHECKPOINT 2 ----------------")
    start = time.time()
    num_clients = len(gold_clients)
    checkpoint = Checkpoint(total=4 * num_clients, result=0, name="Client Slides")

    step_names = ["Name at Top", "Wikipedia Image (Left)", "Different Image (Right)", "Symmetric Vertical Position"]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data.get('slides', [])) < 2:
        step_id = 1
        for client in gold_clients:
            for step_name in step_names:
                checkpoint.add_step(f"{client} - {step_name}", False, step_id,
                                   details="No client slides found in presentation", execution_time=0)
                step_id += 1
        checkpoint.execution_time = time.time() - start
        return checkpoint

    if model is None:
        model = load_model(model_id)

    slides = presentation_data['slides']
    _, slide_height_emu = get_slide_dimensions(presentation_data)

    # Build client → slide mapping in one pass over slides
    print("Mapping clients to slides...")
    time_start = time.time()
    client_slide_map = {client: None for client in gold_clients}
    unmatched_clients = set(gold_clients)
    for idx, slide in enumerate(slides):
        if not unmatched_clients:
            break
        text_boxes = extract_text_boxes_from_slide(slide)
        client_match = None
        for tb in text_boxes:
            tb_text = tb['text'].strip().lower()
            for client in list(unmatched_clients):
                client_match = name_exact_match(tb_text, client)
                if client_match:
                    name_on_top = is_text_in_title_position(slide, tb['text'])
                    client_slide_map[client] = {'slide_idx': idx, 'name_on_top': name_on_top}
                    unmatched_clients.discard(client)
                    break
            if client_match:
                break
    print(f"Client to slide mapping completed in {time.time() - time_start:.2f}s")

    # Run all client evaluations in parallel
    eval_tasks = [
        {'id': client, 'func': evaluate_single_client,
         'args': (client, wiki_urls_cache[client].rstrip("/").rsplit("/", 1)[-1], client_slide_map, step_names, slides, slide_height_emu, model, presentation_id, DATA_DIR)}
        for client in gold_clients
    ]
    client_results = parallel_execute(eval_tasks, max_workers=3)

    # Add steps to checkpoint in order
    step_id = 1
    for client in gold_clients:
        client_steps = client_results.get(client, [])
        if not client_steps:
            for name in step_names:
                checkpoint.add_step(f"{client} - {name}", False, step_id,
                                   details="Evaluation failed", execution_time=0)
                step_id += 1
        else:
            for step in client_steps:
                checkpoint.add_step(step["name"], step["success"], step_id,
                                   step["detail"], execution_time=step["execution_time"])
                step_id += 1

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3 (2 + 9 pt): The last slide contains all required URLs.

    Steps:
        1. Wikipedia URL for each client (9 pts total, 1pt each)
        2. A photographer webpage URL is present (1pt)
        3. The photographer is based in San Francisco (1pt)
    """
    global model
    print("----------------- CHECKPOINT 3 ----------------")
    start = time.time()
    num_clients = len(gold_clients)
    checkpoint = Checkpoint(total=2 + num_clients, result=0, name="Last Slide URLs")

    slides = presentation_data.get('slides', [])
    if len(slides) < 3:
        checkpoint.add_step(
            "Wikipedia URLs for clients", False, 1,
            details="Presentation does not have a URL slide",
            score=0, max_score=num_clients, execution_time=0
        )
        checkpoint.add_step(
            "Photographer URL Present", False, 2,
            details="Presentation does not have a URL slide",
            execution_time=0
        )
        checkpoint.add_step(
            "Photographer in San Francisco", False, 3,
            details="Presentation does not have a URL slide",
            execution_time=0
        )
        checkpoint.execution_time = time.time() - start
        return checkpoint

    last_slide = slides[-1]
    all_urls = extract_slide_links(last_slide)

    # Step 1: Wikipedia URL for each client (1pt each, 9 pts total)
    step_start = time.time()
    wiki_url_matches = []
    wiki_url_misses = []
    all_urls_lower = [url.lower().rstrip('/') for url in all_urls]
    for client in gold_clients:
        expected_url = wiki_urls_cache[client].lower()
        has_wiki_url = any(url == expected_url for url in all_urls_lower)
        if has_wiki_url:
            wiki_url_matches.append(client)
        else:
            wiki_url_misses.append(client)
    wiki_score = len(wiki_url_matches)
    checkpoint.add_step(
        "Wikipedia URLs for clients", wiki_score == num_clients, 1,
        details=f"Found {wiki_score}/{num_clients} Wikipedia URLs. Missing: {wiki_url_misses}" if wiki_url_misses else f"All {num_clients} Wikipedia URLs found",
        score=wiki_score, max_score=num_clients,
        execution_time=time.time() - step_start
    )

    # Identify non-Wikipedia URLs as potential photographer sites
    non_wiki_urls = [u for u in all_urls if 'wikipedia.org' not in u.lower()]

    is_in_sf = False
    has_photographer = False
    photographer_url = None
    for url in non_wiki_urls:
        # Fetch page content and check with LLM
        content, status = fetch_page_text_content(url, timeout=10, max_chars=5000)
        if not content:
            continue
        # Step 2: Photographer URL present
        step_start = time.time()

        # Step 3: Photographer is in San Francisco
        step_start = time.time()
        if model is None:
            model = load_model(model_id)
        result = evaluate_with_llm(f"""Based on the content of this webpage, 
is this person
    1. A photographer
    2. Is based in San Francisco? 
Content: 
{content}
Answer with "Yes" or "No" for each question respectively, separated by a comma. 
For example: "Yes, No" means they are a photographer but not based in San Francisco.
""", model, return_type="str")
        photographer_result, sf_result = result.split(",") if result and "," in result else ("no", "no")
        if photographer_result.strip().lower() == "yes":
            has_photographer = True
            photographer_url = url
            if sf_result.strip().lower() == "yes":
                is_in_sf = True

    checkpoint.add_step(
        "Photographer URL Present", has_photographer, 2,
        details=f"Found {len(non_wiki_urls)} non-Wikipedia URL(s)" if has_photographer else "No non-Wikipedia URLs found",
        execution_time=time.time() - step_start
    )
    checkpoint.add_step(
        "Photographer in San Francisco", is_in_sf, 3,
        details=f"Found SF-based photographer at url {photographer_url}" if is_in_sf else "No San Francisco photographer identified",
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoint_4(browsing_history=None):
    """Checkpoint 4 (2 + 9 pt): Browsing history shows required page visits.

    Steps:
        1. Visit to Tom Hanks' Wikipedia page (1pt)
        2. Visit to the Google Doc containing the client list (1pt)
        3. Visit to each client's Wikipedia page (9 pts total, 1pt each)
    """
    print("----------------- CHECKPOINT 4 ----------------")
    start = time.time()
    num_clients = len(gold_clients)
    checkpoint = Checkpoint(total=2 + num_clients, result=0, name="Browsing History")

    # Step 1: Tom Hanks Wikipedia
    step_start = time.time()
    visited_tom_hanks = False
    if not browsing_history:
        checkpoint.add_step("Visited Tom Hanks Wikipedia", False, 1,
                           details="No browsing history provided", execution_time=0)
        checkpoint.add_step("Visited Client List Google Doc", False, 2,
                           details="No browsing history provided", execution_time=0)
        checkpoint.add_step("Visited client Wikipedia pages", False, 3,
                           details="No browsing history provided",
                           score=0, max_score=num_clients, execution_time=0)
        checkpoint.execution_time = time.time() - start
        return checkpoint

    history_lower = [url.lower().rstrip('/') for url in browsing_history]
    expected_tom_url = wiki_urls_cache["tom hanks"].lower()
    visited_tom_hanks = any(url == expected_tom_url for url in history_lower)
    checkpoint.add_step(
        "Visited Tom Hanks Wikipedia", visited_tom_hanks, 1,
        details="Tom Hanks Wikipedia page found in history" if visited_tom_hanks else "No visit to Tom Hanks Wikipedia",
        execution_time=time.time() - step_start
    )

    # Step 2: Google Doc visit
    step_start = time.time()
    visited_google_doc = False
    visited_google_doc = any('docs.google.com/document' in url.lower() for url in browsing_history)
    checkpoint.add_step(
        "Visited Client List Google Doc", visited_google_doc, 2,
        details="Google Doc visit found in history" if visited_google_doc else "No Google Doc visit found",
        execution_time=time.time() - step_start
    )

    # Step 3: Each client's Wikipedia page (1pt each, 9 pts total)
    step_start = time.time()
    visited_matches = []
    visited_misses = []
    for client in gold_clients:
        expected_url = wiki_urls_cache[client].lower()
        visited = any(url == expected_url for url in history_lower)
        if visited:
            visited_matches.append(client)
        else:
            visited_misses.append(client)
    visit_score = len(visited_matches)
    checkpoint.add_step(
        "Visited client Wikipedia pages", visit_score == num_clients, 3,
        details=f"Visited {visit_score}/{num_clients} client Wikipedia pages. Missing: {visited_misses}" if visited_misses else f"All {num_clients} client Wikipedia pages visited",
        score=visit_score, max_score=num_clients,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str, client_doc_id: str = None, cached_models: Dict[str, Any] = None, browsing_history: List[str] = None):
    """Grade all checkpoints for the Work Wikipedia Photos task.

    Args:
        workspace_doc_id (str): Google Slides presentation ID.
        client_doc_id (str): Google Doc ID containing the client list.
        cached_models (dict, optional): Dictionary of preloaded models by model_id.
        browsing_history (list, optional): List of URLs visited during task.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start = time.time()
    try:
        setup_presentation(workspace_doc_id, client_doc_id)

        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]

        checkpoints: List[Checkpoint] = []
        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4(browsing_history))

        total_execution_time = time.time() - total_start
        return Result(checkpoints, total_execution_time=total_execution_time)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        failed = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed.add_step("Evaluation", False, 1, f"Fatal error: {e}", execution_time=0)
        return Result([failed], total_execution_time=time.time() - total_start)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Work Wikipedia Photos Task")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                        help="Google Slides presentation ID")
    parser.add_argument("--client_doc_id", type=str, required=True,
                        help="Google Doc ID containing the client list")
    parser.add_argument("--browsing_history", nargs='+',
                        help="List of URLs visited")

    args = parser.parse_args()

    start_time = time.time()

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        client_doc_id=args.client_doc_id,
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
            status = "✓" if step["success"] else "✗"
            print(f"  [{status}] {step['name']} ({step['score']}/{step['max_score']}): {step['details']}")

    score = report["final_score"]
    print(f"\nFinal Score: {score['result']}/{score['total']}")
    print(f"Total Time: {time.time() - start_time:.2f}s")
