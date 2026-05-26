import os
import re
import shutil
import time
import uuid
from urllib.parse import unquote, urlparse

from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_images,
    extract_slide_text,
    extract_image_source_urls,
    extract_text_boxes_from_slide,
    is_text_in_title_position,
    get_element_bbox,
    download_slide_image,
)
from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url


# Two-word room types that should survive adjective stripping. Anything not
# in this set falls back to the last single word, so "a beautiful kitchen"
# becomes "kitchen" but "a beautiful living room" stays "living room".
_COMPOUND_ROOM_TYPES = {
    'living room', 'dining room', 'family room', 'sun room', 'mud room',
    'laundry room', 'powder room', 'game room', 'music room', 'sitting room',
    'guest room', 'breakfast room', 'utility room', 'rec room', 'play room',
    'home office', 'home gym', 'home theater', 'home theatre',
    'guest house', 'pool house', 'guest bedroom', 'master bedroom',
    'master bathroom', 'half bath', 'powder bath',
}

# Single-word room/space types; generic words (room, house, space) excluded.
_KNOWN_ROOM_WORDS = {
    'garage', 'kitchen', 'bedroom', 'bathroom', 'office', 'hallway',
    'basement', 'attic', 'library', 'study', 'foyer', 'nursery',
    'pantry', 'closet', 'lounge', 'den', 'parlor', 'conservatory',
    'mudroom', 'sunroom', 'kitchenette', 'gym', 'theater', 'theatre',
    # Outdoor / utility spaces
    'shed', 'barn', 'carport', 'loft', 'cabana', 'patio', 'deck',
    'porch', 'balcony', 'cellar', 'terrace', 'courtyard', 'workshop',
    'studio', 'entryway',
}


def download_alt_image(url, temp_dir, **kwargs):
    """download_image_from_url with backoff retry; recovers transient 429s."""
    for attempt in range(3):
        result = download_image_from_url(url, temp_dir, **kwargs)
        if result:
            return result
        if attempt < 2:
            time.sleep(0.5 * (2 ** attempt))
    return None


def browser_headers(url):
    """Chrome UA + image Accept + per-URL Referer; defeats most hotlink protection."""
    parsed = urlparse(url)
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{parsed.scheme}://{parsed.netloc}/",
    }


def _download_slide_image_with_retry(image_url, max_retries=2):
    """Wrapper around `download_slide_image` with exponential-backoff retry.

    Slides API content URLs are short-lived signed URLs that occasionally
    return transient errors. `download_slide_image` upstream returns None on
    failure with no retry; this helper retries a few times before giving up.
    Returns a PIL.Image or None.
    """
    for attempt in range(max_retries + 1):
        try:
            img = download_slide_image(image_url)
            if img is not None:
                return img
        except Exception as e:
            if attempt == max_retries:
                print(f"All retries failed for {image_url}: {e}")
        if attempt < max_retries:
            time.sleep(0.5 * (2 ** attempt))
    return None


_GLUED_TO_COMPOUND = {c.replace(' ', ''): c for c in _COMPOUND_ROOM_TYPES}


def _clean_vlm_topic(response):
    """Reduce a VLM 'what room is this?' response to the room noun(s)."""
    text = re.sub(r'\*+', '', response).strip().lower()
    # Strip punctuation per-word so internal commas/periods don't break matching.
    words = [w.strip(',.!?;:') for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return ""
    if len(words) >= 2 and ' '.join(words[-2:]) in _COMPOUND_ROOM_TYPES:
        return ' '.join(words[-2:])
    # Walk back; promote to compound if the preceding word forms one,
    # or split a glued single-word compound (e.g. "livingroom" -> "living room").
    for i in range(len(words) - 1, -1, -1):
        if words[i] in _KNOWN_ROOM_WORDS:
            if i > 0 and f"{words[i-1]} {words[i]}" in _COMPOUND_ROOM_TYPES:
                return f"{words[i-1]} {words[i]}"
            return words[i]
        if words[i] in _GLUED_TO_COMPOUND:
            return _GLUED_TO_COMPOUND[words[i]]
    return words[-1]


def get_image_position(element, slide_width_emu, slide_height_emu):
    """
    Determine if an image is in the bottom left or bottom right of the slide.

    Args:
        element (dict): Page element containing an image.
        slide_width_emu (float): Slide width in EMUs.
        slide_height_emu (float): Slide height in EMUs.

    Returns:
        str: "bottom_left", "bottom_right", or "other".
    """
    bbox = get_element_bbox(element)

    # Calculate center point of image
    center_x = bbox['x'] + bbox['width'] / 2
    center_y = bbox['y'] + bbox['height'] / 2

    # Check if in bottom half
    is_bottom = center_y > slide_height_emu / 2

    # Check if in left or right half
    is_left = center_x < slide_width_emu / 2

    if is_bottom and is_left:
        return "bottom_left"
    elif is_bottom:
        return "bottom_right"
    else:
        return "other"


def check_images_correctly_sourced(slide, workspace_doc_id, slides_service):
    """
    Check if the content URLs of images in the slide match
    the source URLs found in their ALT text (description field).

    Args:
        slide (dict): Slide object from Google Slides API.
        workspace_doc_id (str): Google Slides presentation ID.
        slides_service: Google Slides API service.

    Returns:
        tuple: (num_images, num_verified_sources, source_info)
               - num_images: total number of images
               - num_verified_sources: number of images whose content URL matches an ALT text URL
               - source_info: list of source URL info for debugging
    """
    num_images = 0
    num_verified_sources = 0
    source_info = []

    if 'pageElements' not in slide:
        return (0, 0, [])

    # Get all images with their content URLs
    slide_images = extract_slide_images(slide, workspace_doc_id, slides_service)

    # Create a mapping of objectId to contentUrl
    object_id_to_content_url = {}
    for img_info in slide_images:
        if 'objectId' in img_info and 'contentUrl' in img_info:
            object_id_to_content_url[img_info['objectId']] = img_info['contentUrl']

    # Get image source URLs from ALT text
    image_sources = extract_image_source_urls(slide)

    for img_source in image_sources:
        num_images += 1
        object_id = img_source['objectId']
        alt_urls = img_source['source_urls']
        description = img_source['description']

        if not alt_urls:
            source_info.append(f"No source URL in ALT text (ALT: '{description[:40]}')")
            continue

        content_url = object_id_to_content_url.get(object_id)
        if not content_url:
            source_info.append(f"Cannot get slide image content URL for {object_id}")
            continue

        # Check if the content URL matches any of the ALT text URLs
        matched = any(alt_url in content_url or content_url in alt_url for alt_url in alt_urls)

        if matched:
            num_verified_sources += 1
            source_info.append(f"Verified: content URL matches ALT source ({alt_urls[0][:50]}...)")
        else:
            source_info.append(f"Mismatch: content URL does not match ALT URLs ({alt_urls[0][:50]}...)")

    return (num_images, num_verified_sources, source_info)


def identify_image_subject_vlm(images, model, data_dir):
    """
    Use a VLM to identify the room/project type depicted in the images.

    Args:
        images (list): List of image info dictionaries from extract_slide_images.
        model: The loaded VLM model to use.
        data_dir (str): Directory path for storing temporary image files.

    Returns:
        str: The identified room/project type (e.g., "garage", "kitchen"), or empty string.
    """
    if not images:
        return ""

    temp_dir = os.path.join(data_dir, "temp_images_identify")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Download the first available image
        temp_img_path = None
        for img_info in images:
            if img_info.get('contentUrl'):
                img = _download_slide_image_with_retry(img_info['contentUrl'])
                if img:
                    temp_img_path = os.path.join(temp_dir, "temp_image.png")
                    img.save(temp_img_path)
                    break

        if not temp_img_path or not os.path.exists(temp_img_path):
            return ""

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an image analysis assistant. Respond with only the room or space type as 1-2 words, nothing else. Use a space for compound rooms (e.g., 'living room', not 'livingroom')."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": temp_img_path},
                    {"type": "text", "text": "What type of room or space is shown in this image? Respond with the room type as 1-2 words (e.g., garage, kitchen, bathroom, bedroom, office, living room, dining room, home office)."}
                ]
            }
        ]

        try:
            response = model(messages)
        except Exception as e:
            print(f"VLM call failed in identify_image_subject_vlm: {e}")
            return ""
        if not response:
            return ""
        return _clean_vlm_topic(response)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def evaluate_image_relevance_vlm(images, topic, model, data_dir):
    """
    Use a Vision Language Model to evaluate if ALL images are relevant
    to the specified topic.

    Args:
        images (list): List of image info dictionaries from extract_slide_images.
        topic (str): The topic/subject to check image relevance against.
        model: The loaded VLM model to use.
        data_dir (str): Directory path for storing temporary image files.

    Returns:
        tuple: (all_relevant, num_relevant, total) where:
            - all_relevant (bool): True if every image is relevant.
            - num_relevant (int): Count of images that passed relevance check.
            - total (int): Total number of images checked.
    """
    if not images:
        return False, 0, 0

    # Create unique temp directory for downloaded images (safe for parallel calls
    # even when the topic string repeats — md5(topic) collided when two color
    # slides had identical names).
    temp_dir = os.path.join(data_dir, f"temp_images_vlm_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)

    num_relevant = 0
    total = 0

    try:
        for idx, img_info in enumerate(images):
            if not img_info.get('contentUrl'):
                continue

            img = _download_slide_image_with_retry(img_info['contentUrl'])
            if not img:
                continue

            temp_img_path = os.path.join(temp_dir, f"temp_image_{idx}.png")
            img.save(temp_img_path)
            total += 1

            # Check each image individually
            try:
                result = binary_judge_image(
                    model,
                    temp_img_path,
                    f"Could this real photograph serve as inspiration for '{topic}'? Reject paintings, drawings, or illustrations. Accept any real photograph that fits the theme of '{topic}'."
                )
            except Exception as e:
                print(f"VLM call failed for image {idx} in evaluate_image_relevance_vlm: {e}")
                result = None
            if result:
                num_relevant += 1

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return num_relevant == total and total > 0, num_relevant, total


def get_title_text(slide):
    """
    Extract the title text from a slide by finding the first text box in
    the title position.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        str: Title text (stripped) or "" if no text box is in title position.
    """
    text_boxes = extract_text_boxes_from_slide(slide)
    for tb in text_boxes:
        if is_text_in_title_position(slide, tb['text']):
            return tb['text'].strip()
    return ""


def find_color_slides(slides):
    """
    Find all color slides in the presentation (excluding title and recommendation slides).

    Args:
        slides (list): List of all slides from Google Slides API.

    Returns:
        tuple: (color_slides, recommendation_slide_idx)
            - color_slides: List of dicts with keys: 'index', 'slide', 'color', 'title'
            - recommendation_slide_idx: Index of the recommendation slide or None
    """
    color_slides = []
    recommendation_slide_idx = None

    # First, identify the recommendation slide (contains "is the best choice")
    for idx, slide in enumerate(slides):
        full_slide_text = extract_slide_text(slide).lower()
        if "is the best choice" in full_slide_text:
            recommendation_slide_idx = idx
            break

    # Now find color slides (skip title and recommendation)
    for idx, slide in enumerate(slides):
        if idx == 0:  # Skip title slide
            continue

        if idx == recommendation_slide_idx:  # Skip recommendation slide
            continue

        title_text = get_title_text(slide)

        # Accept any non-empty title as a color name
        if title_text:
            color_slides.append({
                'index': idx,
                'slide': slide,
                'color': title_text,
                'title': title_text
            })

    return color_slides, recommendation_slide_idx


def check_browsing_history(browsing_history, search_terms):
    """
    Check if the browsing history contains evidence of searching for one or more terms
    in the same URL.

    Args:
        browsing_history (list): List of URLs visited.
        search_terms (str or list): A single term or list of terms that must all
            appear in the same URL.

    Returns:
        bool: True if search evidence found.
    """
    if not browsing_history:
        return False

    if isinstance(search_terms, str):
        search_terms = [search_terms]

    terms_lower = [t.lower() for t in search_terms if t]
    if not terms_lower:
        return False

    for url in browsing_history:
        url_lower = url.lower()
        if not any(keyword in url_lower for keyword in ['search', 'google.com', 'images', 'bing.com']):
            continue
        # Decode %xx and convert form-encoded '+' to space so multi-word terms
        # like "living room" match URLs like "?q=living+room" or "?q=living%20room".
        url_decoded = unquote(url_lower).replace('+', ' ')
        if all(term in url_decoded for term in terms_lower):
            return True

    return False
