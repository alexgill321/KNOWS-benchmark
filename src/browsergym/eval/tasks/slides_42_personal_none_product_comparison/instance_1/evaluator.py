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
    keywords_exact_match,
    keywords_match_robust,
    keyword_exact_match,
)
from src.browsergym.eval.eval_utils.slides_utils import (
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
from src.browsergym.eval.eval_utils.parallel_utils import parallel_download, parallel_execute
from src.browsergym.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.web_utils import fetch_url_content

from src.browsergym.eval.tasks.slides_42_personal_none_product_comparison.utils import (
    detect_color_name,
    extract_device_info_with_llm,
    evaluate_device_info_with_llm,
    validate_rankings,
    download_images_from_url
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/slides_42_personal_none_product_comparison/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
GOLD_IMAGES_DIR = os.path.join(TASK_DIR, "data/gold_images/")
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
    Checkpoint 1 (6pt): Title slide has all required elements.

    Outcome Evaluation:
    - Exact match on "A Gift for Kathy!" found.
    - Title is in bold.
    - Subtitle correctly lists all 3 device options from the gold list.
    - Image represents the University of Utah found.
    - Image is to the right of the title.
    - The university's official colors are used.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Title Slide Validation")

    checkpoint_1_step_names = [
        "Title Match",
        "Title Is Bold",
        "Subtitle Includes All Devices",
        "University Image Found",
        "Title Left of Image",
        "University Colors",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(checkpoint_1_step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Assume first slide is the title slide
    title_slide = presentation_data['slides'][0]

    # Step 1: Exact match on 'A Gift for Kathy!'
    step_start = time.time()
    title_text = extract_title_text(title_slide)
    title_found = keyword_exact_match(title_text, "A Gift for Kathy!")
    
    checkpoint.add_step("Title Match", title_found, 1, "Found exact title 'A Gift for Kathy!'" if title_found else "Title does not exactly match 'A Gift for Kathy!'", execution_time=time.time() - step_start)
    
    # Step 2: Title is bold
    step_start = time.time()
    text_boxes = extract_text_boxes_from_slide(title_slide)
    
    title_bold = False
    for text_box in text_boxes:
        if keyword_exact_match(title_text, text_box.get('text', '')):
            element = text_box.get('element', {})
            title_style = get_text_style_from_shape(element.get('shape',{}))
            title_bold = title_style.get('bold', False)
            break
    checkpoint.add_step("Title Is Bold", title_bold, 2, "Title text is bold" if title_bold else "Title text not bold or could not determine", execution_time=time.time() - step_start)

    # Step 3: Subtitle lists all 3 device options
    step_start = time.time()
    slide_text = extract_slide_text(title_slide)
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
    matching_image = None

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

        if os.listdir(temp_dir):
            matching_image = binary_judge_image(
                model,
                temp_dir,
                "Is this an image of the University of Utah campus?"
            )

            if matching_image:
                uni_image_valid = True
                
        checkpoint.add_step("University Image Found", uni_image_valid, 4, "Found an image representing University of Utah" if uni_image_valid else "No valid University of Utah image found", execution_time=time.time() - step_start)

        # Step 5: Title is to the left of the university image
        step_start = time.time()
        title_left_of_image = False
        if uni_image_valid and matching_image:
            # Get title element x-position
            title_x = None
            for text_box in text_boxes:
                if keyword_exact_match(title_text, text_box.get('text', '')):
                    title_x = text_box['bbox']['x']
                    break
            # Get matching image element x-position from its index
            match_filename = os.path.basename(matching_image)
            img_idx = int(match_filename.replace("temp_image_", "").replace(".png", ""))
            img_element = title_slide['pageElements'][
                [j for j, el in enumerate(title_slide['pageElements']) if 'image' in el][img_idx]
            ]
            image_x = get_element_bbox(img_element)['x']

            if title_x is not None:
                title_width = text_box['bbox']['width']
                title_center_x = title_x + title_width / 2
                title_left_of_image = image_x > title_center_x
        checkpoint.add_step("Title Left of Image", title_left_of_image, 5, "Title is to the left of the university image" if title_left_of_image else "Title is not to the left of the university image", execution_time=time.time() - step_start)
    finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # Step 6: University colors used (red prominent)
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
    checkpoint.add_step("University Colors", bool(color_valid), 6, "University official color found on slide" if color_valid else "No strong University official color detected", execution_time=time.time() - step_start)

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
        checkpoint.add_step("Explains Challenge", False, 1, "No slides found in presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.add_step("Explains Goal", False, 2, "No slides found in presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Slide 2 is index 1
    slides = presentation_data['slides']
    if len(slides) < 2:
        checkpoint.add_step("Explains Challenge", False, 1, "Challenge slide not found or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.add_step("Explains Goal", False, 2, "Challenge slide not found or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide = slides[1]

    # Step 2 & 3: At least one line explaining challenge and one explaining goal
    step_start = time.time()
    slide_text = extract_slide_text(slide)

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
        checkpoint.add_step("Explains Challenge", 'yes' in response, 1, "Found challenge explanation" if 'yes' in response else "No challenge explanation found", execution_time=time.time() - step_start)
        
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
        checkpoint.add_step("Explains Goal", 'yes' in response, 2, "Found goal explanation" if 'yes' in response else "No goal explanation found", execution_time=time.time() - step_start)
        
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
    global model
    
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Evaluation Criteria")
    categories = ["Battery life", "Weight", "Memory", "Budget", "Processor"]
    if not presentation_data or 'slides' not in presentation_data:
        for i, name in enumerate(categories, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Slide 3 index 2
    slides = presentation_data['slides']
    if len(slides) < 3:
        for i, name in enumerate(categories, 1):
            checkpoint.add_step(name, False, i, "Criteria slide not found or not in the correct order", execution_time=time.time() - checkpoint_start)
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
    
    if model is None:
        model = load_model(model_id)
    for i, category in enumerate(categories):
        step_start = time.time()
        is_valid = keywords_match_robust(slide_text, category_keyword_list[i], model=model, substring=True)
        checkpoint.add_step(category, bool(is_valid), i + 1, f"Found {category}" if bool(is_valid) else f"Missing {category}", execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint
        
def grade_checkpoint_4():
    """
    Checkpoint 4 (24pt): The device slides meet the requirements.

    Outcome Evaluation (x3 devices, 8 pts each):
    - Title of the slide is the device name.
    - Each slide contains at least one source link.
    - Two product images from different angles found.
    - Key features and specificications section found.
    - Pros are listed.
    - Cons are listed.
    - Product images are from the source link(s) in the slide.
    - Product key features is accurate according to the sources."""
    print("----------------- CHECKPOINT 4 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=24, result=0, name="Device Slides")

    step_names_per_device = [
        "Device Name as Title",
        "Source Link(s) in Slide",
        "Product Images",
        "Key Features",
        "Pros",
        "Cons",
        "Product Images From Sources",
        "Content From Sources",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        step_id = 1
        for device in gold_devices:
            for name in step_names_per_device:
                checkpoint.add_step(f"{device} - {name}", False, step_id, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
                step_id += 1
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    if len(slides) < 6:
        step_id = 1
        for device in gold_devices:
            for name in step_names_per_device:
                checkpoint.add_step(f"{device} - {name}", False, step_id, "Device slides missing or not in the correct order", execution_time=time.time() - checkpoint_start)
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
        "15\" M4 MacBook Air": "ma",
        "16\" Lenovo Yoga 7i 2-in-1": "ly",
        "15\" Surface Laptop": "sl",
    }   
    
    print(f"1. Validate titles, source links, and images")
    for i in range(3): # for each device slide
        print(f"    Device slide {i+1}")
        step_start = time.time()
        slide = slides[3+i]
    
        slide_text = extract_slide_text(slide, "\n")
        slide_text_tokens = [part.strip() for part in re.split(split_pattern, slide_text) if part.strip()]
        
        # 1. Validate that title is the device name
        step_start = time.time()
        slide_title = extract_title_text(slide)
        print(f"        Checking that title is device name...")
        # Validate title
        title_match = keywords_match_robust(expected_titles, slide_title)
        if not title_match:
            checkpoint.add_step(f"Device {i+1} - Device Name as Title", False, step_id, "The title is not the device name", execution_time=time.time()-step_start)
        else:
            checkpoint.add_step(f"Device {i+1} - Device Name as Title", True, step_id, "The title is the device name", execution_time=time.time()-step_start)
            expected_titles.remove(title_match)
        step_id += 1    
        
        # 2. Validate that slide contains at least one source link
        print(f"        Checking that the slide provide at least 1 source...")
        step_start = time.time()
        
        slide_links = extract_slide_links(slide)
        
        checkpoint.add_step(f"Device {i+1} - Source Link(s) in Slide", len(slide_links) > 0, step_id, "The slide contains at least one source" if len(slide_links) > 0 else "No source is found in slide", execution_time=time.time() - step_start)
        step_id += 1
        
        # 3. Validate that there are 2 products image from 2 different angle
        print(f"        Checking that images from slide match those in the gold folder...")
        step_start = time.time()
        images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)
        ref_image_folder = ""
        if title_match:
            ref_image_folder = image_file_map[title_match]
        valid_image_count = 0
        temp_dir = ""
        slide_image_paths = []
        binary_judge_tasks = []
        matching_image = None
        image_from_different_angle = None
        if len(images) >= 2:
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
                            slide_image_paths.append(temp_img_path)
                            img.save(temp_img_path)
                            
                # Use binary_judge_image to check if at least 2 slide images are of the appropriate device
                if os.listdir(temp_dir):
                    for idx,img_path in enumerate(slide_image_paths):
                        binary_judge_tasks.append({
                            "id": idx,
                            "func": binary_judge_image,
                            "args": (model, img_path, f"Is this an image of a laptop of the same or similar model as those in the examples?", os.path.join(GOLD_IMAGES_DIR, ref_image_folder))
                        })
                        
                    binary_judge_results = parallel_execute(binary_judge_tasks)
                    
                    for id in binary_judge_results:
                        if valid_image_count == 2:
                            break
                        matching_image = binary_judge_results[id]

                        if matching_image:
                            valid_image_count += 1
                            
                    image_from_different_angle = None
                    temp_example_dir = os.path.join(DATA_DIR, "temp_example")
                    os.makedirs(temp_example_dir, exist_ok=True)
                    if slide_image_paths:
                        shutil.copy2(slide_image_paths[0], temp_example_dir)
                        os.remove(slide_image_paths[0])
                    image_from_different_angle = binary_judge_image(model, temp_dir, f"Is this image showing the laptop from a different perspective or angle compared to the example?", temp_example_dir)
                    # Move image back from temp_example_dir to temp_dir
                    for f in os.listdir(temp_example_dir):
                        shutil.move(os.path.join(temp_example_dir, f), temp_dir)

            except Exception as e:
                print(f"Error evaluating images: {e}")
                    
            valid_images = valid_image_count == 2 and bool(image_from_different_angle)
            checkpoint.add_step(f"Device {i+1} - Product Images", valid_images, step_id, f"Found 2 product images from 2 different angles" if valid_images else "Product images are missing or not from different angles", execution_time=time.time() - step_start)
            step_id += 1
        else:
            checkpoint.add_step(f"Device {i+1} - Product Images", False, step_id, f"Required 2 images, but got {len(images)}", execution_time=time.time()-step_start)
            step_id += 1
            
        # Download all images from url
        print(f"        Checking that at least one image from the slide is from the given source...")
        step_start = time.time()
        found_in_source = False
        try:
            url_temp_dir = os.path.join(DATA_DIR, "temp_url_images")
            os.makedirs(url_temp_dir, exist_ok=True)
            image_download_tasks = []
            for link in slide_links:
                image_download_tasks.append({
                    'id': link,
                    'func': download_images_from_url,
                    'args': (link,url_temp_dir)
                })
                
            image_download_tasks_results = parallel_execute(image_download_tasks)
            
            # Check if any images from source matches the images from the slide:
            if os.listdir(url_temp_dir):
                matching_image = binary_judge_image(
                    model,
                    url_temp_dir,
                    f"Is this an image of a laptop of the same or similar model as the examples?",
                    temp_dir
                )

                if matching_image:
                    found_in_source = True
        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(url_temp_dir):
                shutil.rmtree(url_temp_dir)
            # if os.path.exists(temp_example_dir):
            #     shutil.rmtree(temp_example_dir)
        
        checkpoint.add_step(f"Device {i+1} - Product Images From Sources", found_in_source, step_id, "At least one image in the slide was found in the source" if found_in_source else "No images in the slide was from the source", execution_time=time.time()-step_start)
        step_id += 1
        
        all_slides.append({
            "title": slide_title,
            "links": slide_links,
            "text_tokens": slide_text_tokens
        })
        
    
    # Validate key features, pros, and cons sections
    for i, slide in enumerate(all_slides):
        slide_title = slide["title"]
        slide_text = "\n".join(slide["text_tokens"])
        task_text = f"""Extract the content for key features, pros, and cons of an electronic device from the given slide text.
        
Respond ONLY with this exact JSON format:

{{
    "key_features": "<semicolon-separated point>",
    "pros": "<semicolon-separated points>",
    "cons": "<semicolon-separated points>" 
}}

If no relevant information is found for a certain section, still include it in the response with an empty string as value.

Slide text:

{slide_text}
"""
        section_validation_task.append({
            'id': slide_title,
            'func': extract_device_info_with_llm,
            'args': (task_text, model)
        })
        
    print(f"2. Validating features, pros, and cons")
    print(f"    Extracting all slide section contents...")
    step_start = time.time()
    section_validation_results = parallel_execute(section_validation_task, max_workers=5)    
    print(f"    Finished extracting section content in {time.time()-step_start}")
    for i, slide in enumerate(all_slides):
        step_start = time.time()
        slide_title = slide["title"]
        print(f"    Validating section content for slide device {i+1}")
        section_content = section_validation_results[slide_title]
        
        has_key_features = bool(section_content["key_features"])
        checkpoint.add_step(f"Device {i+1} - Key Features", has_key_features, step_id, 
                           f"Key features found for device {i+1}" if has_key_features else f"Missing key features for device {i+1}",
                           execution_time=time.time() - step_start)
        step_id += 1
        
        step_start = time.time()
        has_pros = bool(section_content["pros"])
        checkpoint.add_step(f"Device {i+1} - Pros", has_pros, step_id,
                           f"Pros found for device {i+1}" if has_pros else f"Missing pros for device {i+1}",
                           execution_time=time.time() - step_start)
        step_id += 1
        
        step_start = time.time()
        has_cons = bool(section_content["cons"])
        checkpoint.add_step(f"Device {i+1} - Cons", has_cons, step_id,
                           f"Cons found for device {i+1}" if has_cons else f"Missing cons for device {i+1}",
                           execution_time=time.time() - step_start)
        step_id += 1
        
        slide["key_features"] = section_content["key_features"]
        
    # Validate that information are pulled from links
    print(f"3. Verifying that information comes from given source")
    match_threshold = 70
    verifying_task = []
    for i, slide in enumerate(all_slides):
        step_start = time.time()
        slide_title = slide["title"]
        
        print(f"    Verifying information from slide Device {i+1}")
        slide_links = slide["links"]
        features = slide["key_features"]

        url_fetch_tasks = []
        for link in slide_links:
            url_fetch_tasks.append({
                'id': link,
                'func': fetch_url_content,
                'args': (link,)
            })
            
        fetched_contents = []
        if url_fetch_tasks:
            print(f"        Downloading web content...")
            start_time = time.time()
            fetch_results = parallel_download(url_fetch_tasks, max_workers=3, use_rate_limit=False)
            print(f"        Finished downloading web content in {time.time()-start_time}")
            for url, content in fetch_results.items():
                fetched_contents.append("\n".join([part.strip() for part in re.split(split_pattern, content) if part.strip()]))
        fetched_text = "\n".join(fetched_contents)
        task_text = f"""Evaluate how much the following information is supported by the source.
        
Respond in a single number between 0 and 100.

Information:
{features}

Content:
{fetched_text}
"""
        verifying_task.append({
            'id': slide_title,
            'func': evaluate_device_info_with_llm,
            'args': (task_text, model, "str")
        })    
        
    verifying_results = parallel_execute(verifying_task, max_workers = 3)
    for i,slide_title in enumerate(verifying_results):
        match_percentage = float(verifying_results[slide_title])
        print(f"        Verifying that info are pulled from websites: {match_percentage:.2f}% of slide text found in sources.")
        checkpoint.add_step(f"Device {i+1} - Content From Sources", match_percentage >= match_threshold, step_id, f"{match_percentage}% of the listed features found in sources" if match_percentage >= match_threshold else f"Only {match_percentage}% of listed features is from sources", execution_time=time.time()-step_start)
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
    - Battery life is covered.
    - An appropriate color applied for each cell under Battery life.
    - Weight is covered.
    - An appropriate color applied for each cell under Weight.
    - Processor is covered.
    - An appropriate color applied for each cell under Processor.
    - Budget consideration is covered.
    - An appropriate color applied for each cell under Budget consideration.
    - Memory capacity is covered.
    - An appropriate color applied for each cell under Memory capacity.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global model
    if model is None:
        model = load_model(model_id)
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=13, result=0, name="Comparison Slide")
    
    comparison_step_names = [
        "Table Has 3 Columns",
        "All Devices as Headers",
        "Green, Yellow, and Red as Color Coding Scheme",
        "Found Battery life Row",
        "Battery life Cell Colors",
        "Found Weight Row",
        "Weight Cell Colors",
        "Found Processor Row",
        "Processor Cell Colors",
        "Found Budget Row",
        "Budget Cell Colors",
        "Found Memory Row",
        "Memory Cell Colors",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(comparison_step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    if len(slides) < 7:
        for i, name in enumerate(comparison_step_names, 1):
            checkpoint.add_step(name, False, i, "Comparison slide missing or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    step_start = time.time()
    slide = slides[6]
    
    # Extract table from slide
    table_data = extract_table_from_slide(slide)
    
    if not table_data:
        for i, name in enumerate(comparison_step_names, 1):
            checkpoint.add_step(name, False, i, "No table found in the slide", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    # Step 1: Verify table has exactly 3 columns
    step_start = time.time()
    has_3_columns = table_data['num_columns'] == 3
    checkpoint.add_step("Table Has 3 Columns", has_3_columns, 1, 
                       "Table has exactly 3 columns" if has_3_columns else f"Table has {table_data['num_columns']} columns, expected 3",
                       execution_time=time.time() - step_start)
    
    # Step 2: Verify all devices are column headers
    step_start = time.time()
    headers = table_data.get('headers', [])
    missing_devices = ""
    print(f"1. Validating table headers.")
    for device in gold_devices:
        header_match = keywords_exact_match(device, headers, substring=True)
        if not bool(header_match):
            missing_devices += f"{device}; " 
    headers_valid = len(missing_devices) == 0
    checkpoint.add_step("All Devices as Headers", headers_valid, 2,
                       "All devices found as column headers" if headers_valid else f"Missing header(s) for the following devices: {missing_devices}",
                       execution_time=time.time() - step_start)
    
    print(f"2. Validating category coverage based on the table content.")
    categories = ["battery life", "weight", "processor", "budget", "memory"]
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
If no line contains relevant information for a category, leave it out of the response.

Categories:
{"; ".join(categories)}

Table Content:
{rows_content}
"""
    print(f"    Sending table content to LLM for category extraction...")
    step_start = time.time()
    category_map = evaluate_device_info_with_llm(task_text, model, return_type="json")
    print(f"    LLM finished category extraction in {time.time() - step_start:.2f} seconds.")
    categories_covered = list(category_map.keys())
    
    llm_ranking_task = []
    ranking_from_table = {}
    colors_used = set()
    task_idx = 3
    
    print(f"    Starting category coverage validation and preparing LLM ranking tasks...")
    for category in categories:
        step_start = time.time()
        if category in categories_covered:
            print(f"        Found category '{category}' in table content. Creating LLM ranking task for this category.")
            checkpoint.add_step(f"Found {category} Row", True, task_idx, f"The {category} category is covered in table", execution_time=time.time() - step_start)
            task_idx += 1
            
            step_start = time.time()
            cells = list(category_map[category].values())
            comparison_content = "\n".join(cells)
            # print(f"    Cell content: {comparison_content}")
            task_text = f"""Given the information for these devices {", ".join(headers)}, respectively, rank them numerically, based on the category {category}, from best to worst, where 1 is best and {len(headers)} is worst. 
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
              'args': (task_text, model, "json")
            })
            
        else:
            print(f"        No clear information about category '{category}' found in table content.")
            checkpoint.add_step(f"Found {category} Row", False, task_idx, f"No clear information about {category} found in table", execution_time=time.time() - step_start)
            task_idx += 1
            checkpoint.add_step(f"{category} Cell Colors", False, task_idx, f"Cannot evaluate cell colors for {category} since category not found", execution_time=time.time() - step_start)
            task_idx += 1
        
    # Extract comparison from table
    print(f"3a. Extracting color coding scheme from table for validation...")
    step_start = time.time()
    for cat_idx,category in enumerate(categories_covered):
        step_start = time.time()
        ranking_from_table[category] = {}
        for dev_idx, device in enumerate(headers):
            color = table_data['cell_colors'][(cat_idx+1,dev_idx)]
            color_str = detect_color_name(color)
            colors_used.add(color_str)
            ranking_from_table[category][device] = color_rank_map.get(color_str, -1)
    print(f"    Validating that the table uses Green, Yellow, and Red as the color coding scheme...")
    if "unknown" in colors_used:
        checkpoint.add_step("Green, Yellow, and Red as Color Coding Scheme", False, 3, f"Unknown colors found in table: {', '.join(colors_used)}", execution_time=time.time() - step_start)
    elif len(colors_used) < 3 or len(colors_used) > 3:
        checkpoint.add_step("Green, Yellow, and Red as Color Coding Scheme", False, 3, f"Too many or too few colors used. Colors found: {', '.join(colors_used)}", execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Green, Yellow, and Red as Color Coding Scheme", True, 3, f"All required colors are used: {', '.join(colors_used)}", execution_time=time.time() - step_start)
    
    start_time = time.time()
    print(f"3b. Validating that the table correctly ranks the devices for each category based on the color coding scheme...")
    if llm_ranking_task:
        llm_ranking_results = parallel_execute(llm_ranking_task, max_workers=len(categories_covered))
        print(f"    LLM finished ranking tasks in {time.time() - start_time:.2f} seconds.")
        print(f"    Validating LLM rankings against table color coding for each category...")
        for category in categories:
            if category in llm_ranking_results:
                start_time = time.time()
                ranking_consistent = validate_rankings(llm_ranking_results[category], ranking_from_table[category])
                checkpoint.add_step(f"{category} - Correct Color Coding", ranking_consistent, task_idx, f"Appropriate colors are used to rank values from best to worst for {category}" if ranking_consistent else f"Colors are not correctly assigned for {category}", execution_time=time.time() - start_time)
            else:
                checkpoint.add_step(f"{category} - Correct Color Coding", False, task_idx, f"LLM failed to rank devices for {category}", execution_time=time.time() - start_time)
            task_idx += 1
    else:
        for category in categories:
            checkpoint.add_step(f"{category} - Correct Color Coding", False, task_idx, f"No LLM ranking performed.", execution_time=time.time() - start_time)
            task_idx += 1

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

    recommendation_step_names = [
        "All Devices Mentioned",
        "Summaries Align with Comparison Data",
        "Recommendations Based on Student Styles",
    ]

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        for i, name in enumerate(recommendation_step_names, 1):
            checkpoint.add_step(name, False, i, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data.get('slides', [])
    if len(slides) < 8:
        for i, name in enumerate(recommendation_step_names, 1):
            checkpoint.add_step(name, False, i, "Recommendation slide missing or not in the correct order", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_start = time.time()
    slide = slides[7]
    slide_text = extract_slide_text(slide)
    
    if model is None:
            model = load_model(model_id)
    
    task_text = f"""Extract the following electronic device summaries and recommendations from this Google slide text.
        
IMPORTANT: This text may contain multiple devices or none at all.
Extract the information for EACH device separately.
                    
Respond ONLY with this exact JSON format (array of devices):
{{
"summary":[["<device_name>","<summary_text>""]],
"recommendation":[["<device_name>","<recommendation_text>""]]
}}


If a summary or recommendation is not found, use an empty string for that field.
If there is NO device, still return an object with the two properties set to empty arrays.

Slide text:
{slide_text}"""
    device_data = extract_device_info_with_llm(task_text, model)
    device_map = {}
    device_names = []    
    
    for summary_item,rec_item in zip(device_data["summary"], device_data["recommendation"]):
        device_name = summary_item[0] if len(summary_item) > 0 else ""
        device_names.append(device_name)
        device_map[device_name] = {}
        device_map[device_name]['summary'] = summary_item[1] if len(summary_item) > 1 else ""
        device_map[device_name]['recommendation'] = rec_item[1] if len(rec_item) > 1 else ""
        
    missing_devices = ""
    print(f"1. Validating devices:")
    for device in gold_devices:
        matches = keywords_match_robust(device_names, device.split(), substring=True)
        if not bool(matches):
            missing_devices += device + "; "
            print(f"    Missing device: {device}")
        else:
            print(f"    Found device: {device}")

    checkpoint.add_step("All Devices Mentioned", len(missing_devices) == 0, 1, "All three correct devices discussed in the slide" if len(missing_devices) == 0 else f"Missing information for: {missing_devices}", execution_time=time.time() - step_start)

    comparison_slide = slides[6]
    comparison_table = extract_table_from_slide(comparison_slide)

    summary_tasks = []
    recommendation_tasks = []
    missing_sum = 0
    missing_rec = 0
    print(f"2. Collecting Summaries and Recommendations for Evaluation Tasks...")
    for device in device_map:
        device_info = device_map[device]
        summary = device_info["summary"]
        comparison_text = ""
        if comparison_table:
            headers = comparison_table.get('headers', [])
            matched_header = keywords_match_robust(headers, device, substring=True)
            if matched_header:
                column_values = [row[matched_header] for row in comparison_table.get('rows', []) if matched_header in row]
                comparison_text = "\n".join(column_values)
        if summary:
            sum_task_text = f"Is the following summary for {device} consistent with the source information?\n\nSource: {comparison_text}\n\nSummary: {summary}"
        
            summary_tasks.append({
                'id': f'{device}',
                'func': evaluate_device_info_with_llm,
                'args': (sum_task_text,model)
            })
        else:
            missing_sum += 1
            print(f"    Missing summary for {device}")
        
        recommendation = device_info["recommendation"]
        if recommendation:
            rec_task_text = f"Is the following recommendation of {device} based on a student style?\n\nRecommendation: {recommendation}"
            recommendation_tasks.append({
                'id': f'{device}',
                'func': evaluate_device_info_with_llm,
                'args': (rec_task_text,model)
            })
        else:
            missing_rec += 1
            print(f"    Missing recommendation for {device}")
    
    step_start = time.time()
    print(f"3. Evaluating Summaries:")
    invalid_summaries = 0
    if summary_tasks:
        summary_eval_results = parallel_execute(summary_tasks, max_workers=3)
        for device, isValid in summary_eval_results.items():
            if isValid:
                print(f"    Summary for {device} is consistent with source information.")
            else:
                invalid_summaries += 1
                print(f"    Summary for {device} is NOT consistent with source information.")
    valid_summaries = invalid_summaries == 0 and missing_sum == 0
    checkpoint.add_step("Summaries Align with Comparison Data", valid_summaries, 2, "All summaries are consistent with source information" if valid_summaries else f"Some device summaries are inconsistent with the source information or missing", execution_time=time.time() - step_start)
    
    step_start = time.time()
    print(f"4. Evaluating Recommendations:")
    invalid_recommendations = 0
    if recommendation_tasks:
        rec_eval_results = parallel_execute(recommendation_tasks, max_workers=3)
        for device, isValid in rec_eval_results.items():
            if isValid:
                print(f"    Recommendation for {device} is based on student style.")
            else:
                invalid_recommendations += 1
                print(f"    Recommendation for {device} is not based on student style.")
    valid_recommendations = invalid_recommendations == 0 and missing_rec == 0
    checkpoint.add_step("Recommendations Based on Student Styles", valid_recommendations, 3, "All recommendations are valid" if valid_recommendations else f"Some device recommendations are not valid or missing", execution_time=time.time() - step_start)
        

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_7():
    """
    Checkppoint 7 (2pt): The overall structure follows the specified task requirements.

    Outcome Evaluation:
    - Exactly 8 slides found.
    - All slides follow the required sequence and titles"""
    print("----------------- CHECKPOINT 7 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Overall Structure")
    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Exactly 8 Slides", False, 1, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.add_step("Correct Titles and Order", False, 2, "No slides found in the presentation", execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    
    step_start = time.time()
    slides = presentation_data.get('slides', [])
    
    if len(slides) < 8:
        checkpoint.add_step("Exactly 8 Slides", False, 1, f"Found only {len(slides)}/8 slides", execution_time=time.time() - step_start)
        checkpoint.add_step(f"Correct Titles and Order", False, 2, "Some slides are missing from the presentation", execution_time=time.time() - step_start)
    
    checkpoint.add_step("Exactly 8 Slides", True, 1, f"Found exactly 8 slides", execution_time=time.time() - step_start)
    
    expected_titles_keywords = [
        ["A Gift for Kathy!"],
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
    for i in chain(range(3), range(6, 8)):
        step_start = time.time()
        title_text = extract_title_text(slides[i])
        if title_text:
            title_match = keywords_match_robust(title_text, expected_titles_keywords[i], model, substring=True)
            if not title_match:
                checkpoint.add_step(f"Correct Titles and Order", False, 2, f"Slide titles and/or orders do not match the task requirements." , execution_time=time.time() - step_start)
                checkpoint.execution_time = time.time() - checkpoint_start
                return checkpoint
        else:
            checkpoint.add_step(f"Correct Titles and Order", False, 2, f"Could not extract title from slide {i+1}.", execution_time=time.time() - step_start)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint
    
    expected_devices = gold_devices.copy()  
    for i in range(3, 6):
        step_start = time.time()
        title_text = extract_title_text(slides[i])
        if title_text:
            title_match = keywords_match_robust(expected_devices, title_text, substring=True)
            if title_match:
                expected_devices.remove(title_match)
                
    checkpoint.add_step(f"Correct Titles and Order", len(expected_devices) == 0, 2, f"Slide titles and order are correct." if len(expected_devices) == 0 else f"No slide titles for the following devices: {expected_devices}", execution_time=time.time() - step_start)
        
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
        failed.add_step("Evaluation", False, 1, f"Fatal error: {e}", execution_time=0)
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
            status = "" if step["success"] else ""
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - step_start:.2f} seconds")