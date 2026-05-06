
import os
import re
import shutil
import time

from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.image_utils import binary_compare_images, binary_judge_image, match_image_tiered
from src.browsergym.knows.eval.eval_utils.slides_utils import download_slide_image, extract_slide_images, get_element_bbox
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url, fetch_api_with_retry

_, SLIDES_SERVICE = initialize_google_services(service_type="slides")


# Per-instance metadata. The featured-client name (row 1 of gold_wikis.csv) is
# discovered automatically; photographer_city varies by theme and is recorded
# here keyed by the contents of each instance's id.txt.
INSTANCE_CONFIG = {
    "30a": {"photographer_city": "San Francisco"},  # Science / Tom Hanks
    "30b": {"photographer_city": "Springfield, MA"},  # Music / Madonna
    "30c": {"photographer_city": "Seoul"},            # K-Pop / Psy
    "30d": {"photographer_city": "Milan"},            # Fashion / Anna Wintour
    "30e": {"photographer_city": "Tel Aviv"},         # Dance / Anne Teresa De Keersmaeker
}


def download_image_with_retry(url, temp_dir, timeout=15, max_retries=3, delay=2):
    """Download an image with retry logic for flaky sources like Wikimedia.

    Args:
        url (str): Image URL to download.
        temp_dir (str): Directory to save the image.
        timeout (int): Request timeout per attempt.
        max_retries (int): Maximum number of attempts.
        delay (int): Seconds to wait between retries.

    Returns:
        str: Path to downloaded image, or None if all attempts failed.
    """
    import time as _time
    for attempt in range(max_retries):
        result = download_image_from_url(url, temp_dir, timeout=timeout)
        if result:
            return result
        if attempt < max_retries - 1:
            print(f"  Retry {attempt + 1}/{max_retries - 1} for image download: {url}")
            _time.sleep(delay)
    return None


def _normalize_name_parts(name):
    """Split a name into normalized parts, stripping punctuation like periods."""
    return [re.sub(r'\.$', '', part) for part in name.lower().split() if part]


def _initial_matches(a, b):
    """Check if one string is an initial of the other (e.g., 'm' matches 'michael')."""
    if len(a) == 1 and len(b) >= 1:
        return b.startswith(a)
    if len(b) == 1 and len(a) >= 1:
        return a.startswith(b)
    return False


def name_exact_match(name, other_name):
    """Check if two names refer to the same person.

    Handles variations like:
        - "John Smith" vs "John Smith" (exact)
        - "John Michael Smith" vs "John Smith" (first + last)
        - "John M. Smith" vs "John Michael Smith" (initial)
        - "John M Smith" vs "John Michael Smith" (initial without period)

    Args:
        name (str): First name string.
        other_name (str): Second name string.

    Returns:
        bool: True if the names match.
    """
    parts_a = _normalize_name_parts(name)
    parts_b = _normalize_name_parts(other_name)

    if not parts_a or not parts_b:
        return False

    # Exact match after normalization
    if parts_a == parts_b:
        return True

    # First and last must match
    if parts_a[0] != parts_b[0] or parts_a[-1] != parts_b[-1]:
        return False

    # If either has only first+last (2 parts), accept as match
    if len(parts_a) <= 2 or len(parts_b) <= 2:
        return True

    # Both have middle names — check middle parts match or are initials
    middles_a = parts_a[1:-1]
    middles_b = parts_b[1:-1]
    if len(middles_a) != len(middles_b):
        return False
    return all(
        a == b or _initial_matches(a, b)
        for a, b in zip(middles_a, middles_b)
    )


def get_wiki_title(client):
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={client}&srlimit=1&format=json"
    search_data = fetch_api_with_retry(search_url, timeout=10, max_retries=2)
    wiki_title = None
    if search_data:
        results = search_data.get('query', {}).get('search', [])
        if results:
            wiki_title = results[0].get('title', '').replace(' ', '_')
    if not wiki_title:
        wiki_title = client.replace(' ', '_')
    return wiki_title

def evaluate_single_client(client, wiki_url, client_slide_map, step_names, slides, slide_height_emu, model, presentation_id, data_dir):
    """Evaluate all 4 steps for a single client. Returns list of step result dicts."""
    steps = []
    step_start = time.time()
    match = client_slide_map[client]

    if match is None:
        for step_name in step_names:
            steps.append({"name": f"{client} - {step_name}", "success": False,
                         "detail": f"No slide found with '{client}' in title", "execution_time": time.time() - step_start})
        return steps

    slide_idx = match['slide_idx']
    name_on_top = match['name_on_top']
    print(f"  Evaluating client: {client} (slide {slide_idx + 1})")

    # Step 1: Client name at top
    steps.append({"name": f"{client} - Name at Top", "success": name_on_top,
                 "detail": "Found the client's name in title position" if name_on_top else "Client name not found in title position",
                 "execution_time": time.time() - step_start})

    slide = slides[slide_idx]
    images = extract_slide_images(slide, presentation_id, SLIDES_SERVICE)

    # Classify images by horizontal position (at most 2 images expected)
    left_image = None
    right_image = None
    if len(images) == 1:
        left_image = images[0]
    elif len(images) >= 2:
        bbox0 = get_element_bbox(images[0])
        bbox1 = get_element_bbox(images[1])
        cx0 = bbox0['x'] + bbox0['width'] / 2
        cx1 = bbox1['x'] + bbox1['width'] / 2
        if cx0 <= cx1:
            left_image = images[0]
            right_image = images[1]
        else:
            left_image = images[1]
            right_image = images[0]

    # Step 2: Wikipedia image on the left — download slide image, fetch wiki image, compare
    step_start = time.time()
    wiki_image_match = False
    step_detail = "No image on left side"
    temp_wiki_dir = os.path.join(data_dir, f"temp_wiki_{client.replace(' ', '_')}")
    temp_example_dir = os.path.join(data_dir, f"temp_wiki_example_{client.replace(' ', '_')}")
    wiki_img_path = None
    try:
        if left_image is not None:
            os.makedirs(temp_wiki_dir, exist_ok=True)
            os.makedirs(temp_example_dir, exist_ok=True)
            # Download slide image
            slide_img = download_slide_image(left_image.get('contentUrl', '')) if left_image.get('contentUrl') else None
            slide_img_path = None
            if slide_img:
                ext = (slide_img.format or "png").lower()
                slide_img_path = os.path.join(temp_wiki_dir, f"slide_img.{ext}")
                slide_img.save(slide_img_path)

            # Fetch Wikipedia main image via search API then summary API
            wiki_title = wiki_url.rstrip("/").rsplit("/", 1)[-1]
            wiki_img_query = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles={wiki_title}"
            wiki_img_url = None
            wiki_res = fetch_api_with_retry(wiki_img_query, timeout=10, max_retries=2)
            if wiki_res:
                pages = wiki_res.get('query', {}).get('pages', {})
                for page_data in pages.values():
                    wiki_img_url = page_data.get('original', {}).get('source')
                    if wiki_img_url:
                        break
            if wiki_img_url:
                wiki_img_path = download_image_with_retry(wiki_img_url, temp_example_dir, timeout=15)

            # Compare the two images
            if slide_img_path and wiki_img_path:
                wiki_image_match = match_image_tiered(slide_img_path, wiki_img_path, model, "Are these the same image?", 5)[0]
                step_detail = "Slide image matches Wikipedia image" if wiki_image_match else "Slide image does not match Wikipedia image"
            elif not slide_img_path:
                step_detail = "Could not download slide image"
            else:
                step_detail = f"Could not fetch Wikipedia image from {wiki_img_url}"
    except Exception as e:
        step_detail = f"Error occurred while fetching Wikipedia image: {str(e)}"
    finally:
        if os.path.exists(temp_example_dir):
            shutil.rmtree(temp_example_dir)

    steps.append({"name": f"{client} - Wikipedia Image (Left)", "success": wiki_image_match,
                 "detail": step_detail, "execution_time": time.time() - step_start})

    # Step 3: Different image of the client on the right (must be a different photo of the same person)
    step_start = time.time()
    is_different_img = False
    right_detail = "No image on right side"
    temp_other_dir = os.path.join(data_dir, f"temp_diff_{client.replace(' ', '_')}")
    try:
        if right_image is not None:
            os.makedirs(temp_other_dir, exist_ok=True)
            other_img = download_slide_image(right_image.get('contentUrl', '')) if right_image.get('contentUrl') else None
            if other_img:
                ext = (other_img.format or "png").lower()
                other_img_path = os.path.join(temp_other_dir, f"right_img.{ext}")
                other_img.save(other_img_path)
                same_person = binary_judge_image(
                    model, temp_other_dir,
                    f"Is this a photo of {client}?", temp_wiki_dir
                )
                same_img = match_image_tiered(
                    other_img_path, slide_img_path, model, f"Is this the exact same photo?", 10
                )[0]
                if same_person and not same_img:
                    is_different_img = True
                    right_detail = f"Right image is a different photo of {client}"
                else:
                    right_detail = f"Right image is not a different photo of {client}"
            else:
                right_detail = "Could not download right image"
    finally:
        if os.path.exists(temp_other_dir):
            shutil.rmtree(temp_other_dir)
        if os.path.exists(temp_wiki_dir):
            shutil.rmtree(temp_wiki_dir)

    steps.append({"name": f"{client} - Different Image (Right)", "success": is_different_img,
                 "detail": right_detail, "execution_time": time.time() - step_start})

    # Step 4: Symmetric vertical position
    step_start = time.time()
    if left_image is None or right_image is None:
        steps.append({"name": f"{client} - Symmetric Vertical Position", "success": False,
                     "detail": "Cannot check symmetry without images on both sides", "execution_time": 0})
        return steps

    left_bbox = get_element_bbox(left_image)
    right_bbox = get_element_bbox(right_image)
    center_y1 = left_bbox['y'] + left_bbox['height'] / 2
    center_y2 = right_bbox['y'] + right_bbox['height'] / 2
    vertical_diff = abs(center_y1 - center_y2) / slide_height_emu if slide_height_emu > 0 else 1.0
    sym_vert = vertical_diff <= 0.15
    steps.append({"name": f"{client} - Symmetric Vertical Position", "success": sym_vert,
                 "detail": f"Left center Y: {center_y1:.0f}, Right center Y: {center_y2:.0f}",
                 "execution_time": time.time() - step_start})

    return steps