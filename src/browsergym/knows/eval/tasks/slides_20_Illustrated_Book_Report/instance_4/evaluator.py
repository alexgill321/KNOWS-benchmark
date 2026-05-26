import os
import sys
from typing import List
import time
import argparse
import shutil

# Base path setup (same pattern as other evaluators)
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/path/to/KNOWS-benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, EvaluationStep
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.text_utils import text_exact_match_contained, text_fuzzy_match_contained_short
from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download, fast_parallel_vlm_calls
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_text,
    extract_slide_images,
    download_slide_image,
    get_slide_background_color,
    extract_slide_links,
    find_slide_by_title_fuzzy,
    validate_bullet_points,
    extract_bullet_point_texts,
    is_text_in_title_position,
    is_text_at_bottom,
    check_text_vertical_order,
    colors_are_different,
    _extract_text_from_text_element
)
from src.browsergym.knows.eval.tasks.slides_20_Illustrated_Book_Report.utils import *

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/slides_20_Illustrated_Book_Report/instance_4/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SLIDES_SERVICE = initialize_google_services(service_type="slides")

# Global variables
presentation_id = None
presentation_data = None
gold_characters = None
alias_to_canonical = None
slide_height = None


def extract_title_text(slide):
    """
    Extract text from the title placeholder or topmost text element of a slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        str: Title text or empty string if no title found.
    """
    if 'pageElements' not in slide:
        return ""

    title_candidates = []

    for element in slide['pageElements']:
        if 'shape' in element:
            shape = element['shape']

            # Check if it's a title placeholder
            placeholder = shape.get('placeholder', {})
            placeholder_type = placeholder.get('type', '')

            if placeholder_type in ['TITLE', 'CENTERED_TITLE', 'SUBTITLE']:
                if 'text' in shape:
                    return _extract_text_from_text_element(shape['text'])

            # Also check position - collect text from top elements
            transform = element.get('transform', {})
            translate_y = transform.get('translateY', float('inf'))

            if 'text' in shape and translate_y < 1000000:  # Top ~20% of slide
                text = _extract_text_from_text_element(shape['text'])
                if text:
                    title_candidates.append((translate_y, text))

    # Return the topmost text element if no title placeholder found
    if title_candidates:
        title_candidates.sort(key=lambda x: x[0])  # Sort by Y position
        return title_candidates[0][1]

    return ""


def setup_presentation(workspace_doc_id):
    """
    Setup presentation processing.

    Args:
        workspace_doc_id (str): Google Slides presentation ID to evaluate.
    """
    global presentation_id, presentation_data, gold_characters, alias_to_canonical, slide_height

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace presentation ID: {workspace_doc_id}")
    presentation_id = workspace_doc_id

    # Fetch presentation data
    presentation_data = SLIDES_SERVICE.presentations().get(presentationId=presentation_id).execute()

    # Resolve slide height from presentation data
    page_size = presentation_data.get('pageSize', {})
    height_obj = page_size.get('height', {})
    slide_height = height_obj.get('magnitude', 5143500)

    # Load gold characters list (supports tab-separated aliases per line)
    gold_characters_path = os.path.join(DATA_DIR, "gold_characters.txt")
    if os.path.exists(gold_characters_path):
        gold_characters, alias_to_canonical = load_gold_characters(gold_characters_path)
    else:
        print(f"Warning: gold_characters.txt not found at {gold_characters_path}")
        gold_characters = []
        alias_to_canonical = {}


def grade_checkpoint_1():
    """
    Checkpoint 1 (6pt): Title slide has all required elements.

    Outcome Evaluation:
    - Exact match on name "Charlotte Currer Bell"
    - Title of book present
    - Photo of book cover present and valid (The Eyre Affair book cover verified by LLM)
    - Structural Location match for name at bottom of slide (bottom 25%), below title and photo.
    - Structural Location match for title above name and below photo.
    - Photo is above both title and name.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Title Slide Validation")

    if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
        checkpoint.add_step("Title Slide Exists", False, 1,
                          "No slides found in presentation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Assume first slide is the title slide
    title_slide = presentation_data['slides'][0]
    slide_text = extract_slide_text(title_slide)

    # Step 1: Check for "Charlotte Currer Bell" (exact match)
    step_start = time.time()
    name_found = text_exact_match_contained("Charlotte Currer Bell", slide_text)
    step_time = time.time() - step_start

    if name_found:
        checkpoint.add_step("Name Match", True, 1,
                          "Found 'Charlotte Currer Bell' in title slide",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Name Match", False, 1,
                          "'Charlotte Currer Bell' not found in title slide",
                          execution_time=step_time)

    # Step 2: Check for book title "The Eyre Affair"
    step_start = time.time()
    title_found = text_fuzzy_match_contained_short("The Eyre Affair", slide_text)
    step_time = time.time() - step_start

    if title_found:
        checkpoint.add_step("Book Title Match", True, 2,
                          "Found 'The Eyre Affair' in title slide",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Book Title Match", False, 2,
                          "Book title 'The Eyre Affair' not found in title slide",
                          execution_time=step_time)

    # Step 3: Check for book cover photo using LLM-as-judge
    step_start = time.time()
    images = extract_slide_images(title_slide, presentation_id, SLIDES_SERVICE)

    book_cover_valid = False
    verified_cover_idx = None
    if len(images) > 0:
        # Use LLM to verify it's the The Eyre Affair book cover
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

            # Use binary_judge_image to check if any image is the book cover
            if os.listdir(temp_dir):
                matching_image = binary_judge_image(
                    model,
                    temp_dir,
                    "Is this an image of the book cover for 'The Eyre Affair' by Jasper Fforde? The cover should show the title 'The Eyre Affair' and credit Jasper Fforde. Covers typically use the novel's distinctive typography and may feature a surreal or literary motif."
                )

                if matching_image:
                    book_cover_valid = True
                    # Extract the index of the verified cover image from the filename
                    import re
                    idx_match = re.search(r'temp_image_(\d+)', os.path.basename(matching_image))
                    if idx_match:
                        verified_cover_idx = int(idx_match.group(1))

        finally:
            # Cleanup temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    step_time = time.time() - step_start

    if book_cover_valid:
        checkpoint.add_step("Book Cover Photo", True, 3,
                          f"Found valid 'The Eyre Affair' book cover image in title slide",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Book Cover Photo", False, 3,
                          "No valid 'The Eyre Affair' book cover image found in title slide",
                          execution_time=step_time)

    # Steps 4-6: Structural location checks using bounding box positions
    BOTTOM_25_THRESHOLD = slide_height * 0.75  # Y position > this means bottom 25%

    # Step 4: Check that name is at bottom (bottom 25% of slide)
    step_start = time.time()
    name_at_bottom = False
    name_y_position = None

    if name_found and 'pageElements' in title_slide:
        for element in title_slide['pageElements']:
            if 'shape' in element and 'text' in element['shape']:
                element_text = _extract_text_from_text_element(element['shape']['text'])

                if "Charlotte Currer Bell" in element_text:
                    transform = element.get('transform', {})
                    name_y_position = transform.get('translateY', 0)

                    if name_y_position > BOTTOM_25_THRESHOLD:
                        name_at_bottom = True
                    break

    step_time = time.time() - step_start

    if name_at_bottom:
        checkpoint.add_step("Name at Bottom", True, 4,
                          f"Name 'Charlotte Currer Bell' correctly positioned in bottom 25% at y={name_y_position}",
                          execution_time=step_time)
    else:
        if name_y_position is not None:
            checkpoint.add_step("Name at Bottom", False, 4,
                              f"Name not in bottom 25% (y={name_y_position}, threshold={BOTTOM_25_THRESHOLD})",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Name at Bottom", False, 4,
                              "Cannot check location - name position not found",
                              execution_time=step_time)

    # Step 5: Check title is above name
    step_start = time.time()
    title_above_name = False
    title_y_position = None

    if title_found and name_y_position is not None and 'pageElements' in title_slide:
        # Collect all elements containing the book title, prefer placeholders
        title_candidates = []
        for element in title_slide['pageElements']:
            if 'shape' in element and 'text' in element['shape']:
                element_text = _extract_text_from_text_element(element['shape']['text'])
                if "The Eyre Affair" in element_text:
                    transform = element.get('transform', {})
                    y = transform.get('translateY', 0)
                    ph_type = element['shape'].get('placeholder', {}).get('type', '')
                    is_placeholder = ph_type in ('TITLE', 'CENTERED_TITLE', 'SUBTITLE')
                    title_candidates.append((not is_placeholder, y, element))

        if title_candidates:
            title_candidates.sort()  # Placeholders first, then by Y position
            title_y_position = title_candidates[0][1]
            if title_y_position < name_y_position:
                title_above_name = True

    step_time = time.time() - step_start

    if title_above_name:
        checkpoint.add_step("Title Above Name", True, 5,
                          f"Title correctly positioned above name (title_y={title_y_position}, name_y={name_y_position})",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Title Above Name", False, 5,
                          "Title not positioned above name" if name_y_position else "Cannot check - element positions not found",
                          execution_time=step_time)

    # Step 6: Check photo is above both title and name
    step_start = time.time()
    photo_above_both = False
    photo_y_position = None

    if book_cover_valid and verified_cover_idx is not None:
        # Use the verified cover image's transform from the images list
        cover_transform = images[verified_cover_idx].get('transform', {})
        photo_y_position = cover_transform.get('translateY', 0)

        if title_y_position is not None and name_y_position is not None:
            if photo_y_position < title_y_position and photo_y_position < name_y_position:
                photo_above_both = True

    step_time = time.time() - step_start

    if photo_above_both:
        checkpoint.add_step("Photo Above Both", True, 6,
                          f"Photo correctly positioned above title and name (photo_y={photo_y_position})",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Photo Above Both", False, 6,
                          "Photo not positioned above both title and name" if photo_y_position else "Cannot check - photo position not found",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2(browsing_history=None):
    """
    Checkpoint 2 (25 pt) x5: The character slides meet the requirements.

    Outcome Evaluation:
    - The character chosen is among the list of the top 20 characters from the book (based on a predefined list).
    - Each character slide has a different background color.
    - Each character slide has a character's name as the title (in title position).
    - Each character slide has at least 3 bullet points describing their characteristics.
    - Each character slide has a source link at the bottom of the slide.
    - The characteristics are all direct quotes from the source links in the slide (verified by checking browsing history).
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=120, result=0, name="Character Slides Validation")

    if not presentation_data or 'slides' not in presentation_data:
        checkpoint.add_step("Character Slides", False, 1,
                          "No slides found in presentation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data['slides']

    # Skip first slide (title) and last 2 slides (author + references)
    # Character slides should be slides 1-5 (indices 1-5)
    character_slides = []

    # Try to find character slides by looking for slides with gold character names.
    # Track seen canonical names so a character and its alias aren't counted twice.
    seen_canonicals = set()
    for idx, slide in enumerate(slides):
        if idx == 0:  # Skip title slide
            continue

        # Extract title text directly from slide structure
        title_text = extract_title_text(slide)

        # Use match_text_in_list to find the best matching character name
        matched_char_name, match_score = match_character_name(title_text, gold_characters, threshold=80)

        # Reject fuzzy matches that strip diacritics from the gold name
        # (e.g., title "Adele" must not pass as gold "Adèle").
        if matched_char_name and not contains_preserving_diacritics(matched_char_name, title_text):
            print(f"Rejecting '{matched_char_name}' for slide {idx}: diacritics missing in title '{title_text}'")
            matched_char_name = None

        # If matching fails, use LLM-as-judge as fallback
        if not matched_char_name and title_text:
            if model is None:
                model = load_model(model_id)

            # Create a prompt asking if any of the gold characters are mentioned
            char_list = ", ".join(gold_characters)

            # Use LLM to check for character names in the title
            try:
                # Format messages in the same way as binary_judge_image
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant that identifies character names from 'The Eyre Affair' by Jasper Fforde. Respond with ONLY the character name from the provided list, or 'NONE' if no character is mentioned."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Is this slide title any of these 'The Eyre Affair' characters: {char_list}?\n\nSlide title: {title_text}"}]
                    }
                ]

                response = model(messages)

                # Check if response matches any gold character. Require the
                # slide title itself to contain the alias with diacritics
                # preserved, so the LLM can't launder an ASCII-stripped title.
                for char_name in gold_characters:
                    if (char_name.lower() in response.lower()
                            and contains_preserving_diacritics(char_name, title_text)):
                        matched_char_name = char_name
                        print(f"LLM matched character '{char_name}' in slide {idx}")
                        break
            except Exception as e:
                print(f"LLM fallback failed for slide {idx}: {e}")

        # Add to character slides if we found a match (store canonical name).
        # Skip slides whose canonical character has already been seen — the
        # task calls for 5 distinct characters, so duplicates (e.g., a slide
        # titled "Thursday Next" plus another titled "Thursday") don't earn extra credit.
        if matched_char_name:
            canonical = alias_to_canonical.get(matched_char_name, matched_char_name)
            if canonical in seen_canonicals:
                print(f"Skipping slide {idx}: duplicate character '{canonical}' (matched alias '{matched_char_name}')")
                continue
            seen_canonicals.add(canonical)
            character_slides.append((idx, slide, canonical))

    if len(character_slides) < 5:
        print(f"Warning: Only found {len(character_slides)} character slides, expected 5")

    # Collect background colors for comparison
    slide_colors = []
    for idx, slide, char_name in character_slides:
        color = get_slide_background_color(slide)
        slide_colors.append(color)

    # Extract bullet texts for all slides for later validation.
    # Fall back to plain text lines if no bullet formatting is found.
    slide_bullet_texts = {}
    slide_has_real_bullets = {}
    for i, (slide_idx, slide, char_name) in enumerate(character_slides[:5]):
        bullets = extract_bullet_point_texts(slide)
        if bullets:
            slide_bullet_texts[i] = bullets
            slide_has_real_bullets[i] = True
        else:
            title_text = extract_title_text(slide)
            slide_bullet_texts[i] = extract_slide_body_lines(slide, title_text)
            slide_has_real_bullets[i] = False

    # ============ PHASE 1: Parallel bullet characteristic validation ============
    # Batch all bullet validation LLM calls across all slides
    if model is None:
        model = load_model(model_id)

    all_bullet_tasks = []
    for slide_i, (slide_idx, slide, char_name) in enumerate(character_slides[:5]):
        bullet_texts = slide_bullet_texts.get(slide_i, [])
        for bullet_j, bullet_text in enumerate(bullet_texts):
            if bullet_text:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant that evaluates whether text describes a character trait, quality, or characteristic. Respond with ONLY 'Yes' or 'No'."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Does this bullet point describe a characteristic, trait, or quality?\n\nBullet point: {bullet_text}"}]
                    }
                ]
                all_bullet_tasks.append({
                    'id': f'slide_{slide_i}_bullet_{bullet_j}',
                    'messages': messages,
                    'slide_i': slide_i,
                    'bullet_j': bullet_j,
                    'bullet_text': bullet_text
                })

    # Run all bullet characteristic checks in parallel
    bullet_validation_results = {}
    if all_bullet_tasks:
        print(f"  Running {len(all_bullet_tasks)} bullet characteristic checks in parallel...")
        vlm_results = fast_parallel_vlm_calls(all_bullet_tasks, model, max_workers=10)
        for task in all_bullet_tasks:
            task_id = task['id']
            slide_i = task['slide_i']
            bullet_j = task['bullet_j']
            is_valid = vlm_results.get(task_id, False)
            if slide_i not in bullet_validation_results:
                bullet_validation_results[slide_i] = {}
            bullet_validation_results[slide_i][bullet_j] = is_valid

    # ============ PHASE 2: Collect all source URLs from all slides ============
    all_slide_links = {}  # slide_i -> list of links
    all_url_fetch_tasks = []
    seen_urls = set()

    for slide_i, (slide_idx, slide, char_name) in enumerate(character_slides[:5]):
        slide_links = extract_slide_links(slide)
        all_slide_links[slide_i] = slide_links
        for link in slide_links:
            if link not in seen_urls:
                seen_urls.add(link)
                all_url_fetch_tasks.append({
                    'id': link,
                    'func': fetch_url_content,
                    'args': (link,)
                })

    # Fetch all URLs in parallel
    url_contents_cache = {}
    if all_url_fetch_tasks:
        print(f"  Fetching {len(all_url_fetch_tasks)} source URLs in parallel...")
        fetch_results = parallel_download(all_url_fetch_tasks, max_workers=5, use_rate_limit=False)
        for url, content in fetch_results.items():
            if content:
                url_contents_cache[url] = content
                print(f"  Fetched {len(content)} chars from {url[:50]}...")

    # Grade each character slide
    for i, (slide_idx, slide, char_name) in enumerate(character_slides[:5]):  # Limit to 5
        step_num_base = i * 6  # Each character gets 6 steps

        # Step 1: Character in gold list AND name as title
        checkpoint.add_step(f"Character {i+1} - Valid Character and Name as Title", True, step_num_base + 1,
                          f"Character '{char_name}' is in gold list and slide title matches character",
                          execution_time=0)

        # Step 2: Different background color from other character slides
        step_start = time.time()
        current_color = slide_colors[i]
        has_unique_color = True

        for j, other_color in enumerate(slide_colors):
            if i != j and not colors_are_different(current_color, other_color):
                has_unique_color = False
                break

        step_time = time.time() - step_start

        if has_unique_color:
            checkpoint.add_step(f"Character {i+1} - Unique Background", True, step_num_base + 2,
                              f"Slide has different background color from other character slides",
                              execution_time=step_time)
        else:
            checkpoint.add_step(f"Character {i+1} - Unique Background", False, step_num_base + 2,
                              f"Slide background color matches another character slide",
                              execution_time=step_time)

        # Step 3: At least 3 bullet points with text
        has_bullets, bullet_count = validate_bullet_points(slide, min_count=3)
        bullet_texts = slide_bullet_texts.get(i, [])

        checkpoint.add_step(f"Character {i+1} - Bullet Point Count",
                          has_bullets, step_num_base + 3,
                          f"Found {bullet_count} bullet points (>= 3 required)",
                          execution_time=0)

        # Step 4: Bullet points describe characteristics (fractional)
        valid_characteristics_count = 0
        slide_bullet_results = bullet_validation_results.get(i, {})
        for bullet_j, is_valid in slide_bullet_results.items():
            if is_valid:
                valid_characteristics_count += 1

        text_count = len(bullet_texts)
        if text_count > 0:
            char_score = int((valid_characteristics_count / text_count) * 10)
        else:
            char_score = 0

        checkpoint.add_step(f"Character {i+1} - Characteristics Quality",
                          char_score > 0, step_num_base + 4,
                          f"{valid_characteristics_count}/{text_count} text items describe characteristics ({char_score}/10 pts)",
                          score=char_score, max_score=10,
                          execution_time=0)

        # Step 5: Source link at bottom of slide
        slide_links = all_slide_links.get(i, [])
        has_source_links = len(slide_links) > 0

        all_links_at_bottom = False
        links_at_bottom_count = 0

        if has_source_links:
            for link in slide_links:
                if is_text_at_bottom(slide, link, slide_height=slide_height) or is_source_link_at_bottom_of_content(slide, link):
                    links_at_bottom_count += 1

            all_links_at_bottom = links_at_bottom_count == len(slide_links)

        if all_links_at_bottom:
            checkpoint.add_step(f"Character {i+1} - Source Link at Bottom", True, step_num_base + 5,
                              f"All {len(slide_links)} source link(s) correctly positioned at bottom of slide",
                              execution_time=0)
        elif has_source_links:
            checkpoint.add_step(f"Character {i+1} - Source Link at Bottom", False, step_num_base + 5,
                              f"Only {links_at_bottom_count}/{len(slide_links)} source link(s) positioned at bottom of slide",
                              execution_time=0)
        else:
            checkpoint.add_step(f"Character {i+1} - Source Link at Bottom", False, step_num_base + 5,
                              "No source links found on slide",
                              execution_time=0)

        # Step 6: Characteristics are supported by source links
        validation_details = []
        validated_bullets = []

        if has_source_links and bullet_texts:
            url_contents = {}
            for link in slide_links:
                if link in url_contents_cache:
                    url_contents[link] = url_contents_cache[link]

            if not url_contents:
                validation_details.append("Could not fetch any source URLs")
            else:
                for bullet_text in bullet_texts:
                    if not bullet_text or bullet_text.strip() == "":
                        continue

                    found_in_source = False
                    for url, content in url_contents.items():
                        if validate_bullet_in_content(bullet_text, content, model, character_name=char_name, book_title="The Eyre Affair"):
                            found_in_source = True
                            validation_details.append(f"Found in {url[:50]}...")
                            break

                    if found_in_source:
                        validated_bullets.append(bullet_text)
                    else:
                        validation_details.append(f"Not found: {bullet_text[:50]}...")

        # Fractional scoring: floor(validated / total) * 10
        total_bullets_to_check = len(bullet_texts) if bullet_texts else 0
        validated_count = len(validated_bullets)
        if total_bullets_to_check > 0:
            source_score = int((validated_count / total_bullets_to_check) * 10)
        else:
            source_score = 0

        if not has_source_links:
            detail = "No source links to validate"
        elif not bullet_texts:
            detail = "No bullet points to validate"
        elif not url_contents:
            detail = f"Could not fetch source URL content. {'; '.join(validation_details)}"
        else:
            detail = f"{validated_count}/{total_bullets_to_check} characteristics supported by sources ({source_score}/10 pts). {'; '.join(validation_details[:3])}"

        checkpoint.add_step(
            f"Character {i+1} - Characteristics from Sources",
            source_score > 0, step_num_base + 6,
            detail,
            score=source_score, max_score=10,
            execution_time=0
        )

    # Handle missing character slides
    missing_max_scores = {1: 1, 2: 1, 3: 1, 4: 10, 5: 1, 6: 10}
    for i in range(len(character_slides), 5):
        step_num_base = i * 6
        for j in range(1, 7):
            checkpoint.add_step(f"Character {i+1} - Step {j}", False, step_num_base + j,
                              "Character slide not found",
                              score=0, max_score=missing_max_scores[j],
                              execution_time=0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (2 pt): The author slide is correct.

    Outcome Evaluation:
    - Exact match on author name.
    - Photo of author present and valid.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Author Slide Validation")

    if not presentation_data or 'slides' not in presentation_data:
        checkpoint.add_step("Author Slide", False, 1,
                          "No slides found in presentation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data['slides']

    # Author slide should be second to last (before references).
    # Try to find it by looking for "Jasper Fforde".
    author_slide = None
    for idx in range(len(slides) - 2, max(0, len(slides) - 4), -1):  # Check last few slides
        slide = slides[idx]
        slide_text = extract_slide_text(slide)

        if text_fuzzy_match_contained_short("Jasper Fforde", slide_text):
            author_slide = slide
            break

    if author_slide is None and len(slides) >= 2:
        # Fallback: assume second to last slide
        author_slide = slides[-2]

    if author_slide is None:
        checkpoint.add_step("Author Name", False, 1,
                          "Could not find author slide",
                          execution_time=0)
        checkpoint.add_step("Author Photo", False, 2,
                          "Could not find author slide",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slide_text = extract_slide_text(author_slide)

    # Step 1: Check for "Jasper Fforde" (exact match).
    step_start = time.time()
    author_found = text_exact_match_contained("Jasper Fforde", slide_text)
    step_time = time.time() - step_start

    if author_found:
        checkpoint.add_step("Author Name", True, 1,
                          "Found 'Jasper Fforde' in author slide",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Author Name", False, 1,
                          "'Jasper Fforde' not found in author slide",
                          execution_time=step_time)

    # Step 2: Check for author photo using VLM validation
    step_start = time.time()
    images = extract_slide_images(author_slide, presentation_id, SLIDES_SERVICE)
    author_photo_valid = False

    if len(images) > 0:
        if model is None:
            model = load_model(model_id)

        temp_dir = os.path.join(DATA_DIR, "temp_author_images")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            for idx, img_info in enumerate(images):
                if img_info['contentUrl']:
                    img = download_slide_image(img_info['contentUrl'])
                    if img:
                        temp_img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
                        img.save(temp_img_path)

            if os.listdir(temp_dir):
                author_photos_dir = os.path.join(DATA_DIR, "author-photos")
                examples = author_photos_dir if os.path.isdir(author_photos_dir) else None
                matching_image = binary_judge_image(
                    model,
                    temp_dir,
                    "Is this a photo of Jasper Fforde, the author of The Eyre Affair?",
                    examples=examples
                )
                if matching_image:
                    author_photo_valid = True
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    step_time = time.time() - step_start

    if author_photo_valid:
        checkpoint.add_step("Author Photo", True, 2,
                          "Found valid photo of Jasper Fforde in author slide",
                          execution_time=step_time)
    elif len(images) > 0:
        checkpoint.add_step("Author Photo", False, 2,
                          f"Found {len(images)} image(s) but none verified as Jasper Fforde",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Author Photo", False, 2,
                          "No images found in author slide",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4(browsing_history=None):
    """
    Checkpoint 4 (2 pt): References slide is correct.

    Outcome Evaluation:
    - At least 3 different reference links present.
    - All reference links are valid URLs found in the agent trace.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="References Slide Validation")

    if not presentation_data or 'slides' not in presentation_data:
        checkpoint.add_step("References Slide", False, 1,
                          "No slides found in presentation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    slides = presentation_data['slides']

    # References slide should be last
    references_slide = None

    # Try to find it by looking for "Reference" or "Sources" in title
    for idx in range(len(slides) - 1, max(0, len(slides) - 3), -1):  # Check last few slides
        slide = slides[idx]
        slide_text = extract_slide_text(slide)

        if text_fuzzy_match_contained_short("Reference", slide_text) or text_fuzzy_match_contained_short("Sources", slide_text):
            references_slide = slide
            break

    if references_slide is None and len(slides) >= 1:
        # Fallback: assume last slide
        references_slide = slides[-1]

    if references_slide is None:
        checkpoint.add_step("Reference Links Count", False, 1,
                          "Could not find references slide",
                          execution_time=0)
        checkpoint.add_step("Links in Browsing History", False, 2,
                          "Could not find references slide",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Check for at least 3 different links
    step_start = time.time()
    ref_links = extract_slide_links(references_slide)
    unique_links = list(set(ref_links))
    has_enough_links = len(unique_links) >= 3
    step_time = time.time() - step_start

    if has_enough_links:
        checkpoint.add_step("Reference Links Count", True, 1,
                          f"Found {len(unique_links)} unique reference links (>= 3 required)",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Reference Links Count", False, 1,
                          f"Only found {len(unique_links)} unique links, need 3",
                          execution_time=step_time)

    # Step 2: Check if links are in browsing history
    step_start = time.time()

    if browsing_history and unique_links:
        browsing_set = set(browsing_history)

        links_in_history = 0
        for link in unique_links:
            if link in browsing_set or any(link in history_url or history_url in link for history_url in browsing_set):
                links_in_history += 1

        all_links_valid = links_in_history == len(unique_links)
        step_time = time.time() - step_start

        if all_links_valid:
            checkpoint.add_step("Links in Browsing History", True, 2,
                              f"All {len(unique_links)} reference links found in browsing history",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Links in Browsing History", False, 2,
                              f"Only {links_in_history}/{len(unique_links)} links found in browsing history",
                              execution_time=step_time)
    elif not browsing_history:
        step_time = time.time() - step_start
        checkpoint.add_step("Links in Browsing History", False, 2,
                          "No browsing history provided to validate",
                          execution_time=step_time)
    else:
        step_time = time.time() - step_start
        checkpoint.add_step("Links in Browsing History", False, 2,
                          "No links found to validate",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
    """
    Grade all checkpoints for the illustrated book report task.

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
        checkpoints.append(grade_checkpoint_2(browsing_history))
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4(browsing_history))

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
    parser = argparse.ArgumentParser(description="Evaluate illustrated book report presentation")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Slides presentation ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
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
            status = "" if step["success"] else ""
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
