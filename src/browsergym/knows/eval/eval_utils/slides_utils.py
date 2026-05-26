"""
Utility functions for extracting and validating Google Slides content.

This module provides functions to:
- Extract text, images, links, and layout information from slides
- Validate slide structure and formatting
- Check element positions and ordering
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from io import BytesIO
from PIL import Image
import requests

# Default Google Slides dimensions in EMUs (English Metric Units) - used as fallback
DEFAULT_SLIDE_WIDTH_EMU = 9144000
DEFAULT_SLIDE_HEIGHT_EMU = 5143500

def extract_slide_text(slide: Dict[str, Any], separator: str = " ") -> str:
    """
    Extract all text content from a slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        str: Combined text from all text elements in the slide.
    """
    text_parts = []

    if 'pageElements' not in slide:
        return ""

    for element in slide['pageElements']:
        # Extract from shapes with text
        if 'shape' in element and 'text' in element['shape']:
            shape_text = _extract_text_from_text_element(element['shape']['text'])
            if shape_text:
                text_parts.append(shape_text)

        # Extract from tables
        if 'table' in element:
            for row in element['table'].get('tableRows', []):
                for cell in row.get('tableCells', []):
                    if 'text' in cell:
                        cell_text = _extract_text_from_text_element(cell['text'])
                        if cell_text:
                            text_parts.append(cell_text)

    return separator.join(text_parts)

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
            translate_y = transform.get('translateY', 0)
            content_alignment = shape.get('shapeProperties', {}).get('contentAlignment', {})
            
            if 'text' in shape and (translate_y < 1500000 or 'top' in content_alignment.lower()):  # Top ~20% of slide
                text = _extract_text_from_text_element(shape['text'])
                if text:
                    title_candidates.append((translate_y, text))

    # Return the topmost text element if no title placeholder found
    if title_candidates:
        title_candidates.sort(key=lambda x: x[0])  # Sort by Y position
        return title_candidates[0][1]

    return ""

def _extract_text_from_text_element(text_element: Dict[str, Any]) -> str:
    """
    Extract text from a textual element structure.

    Args:
        text_element (dict): Text element from Slides API.

    Returns:
        str: Extracted text content.
    """
    text_parts = []

    for text_run in text_element.get('textElements', []):
        if 'textRun' in text_run:
            content = text_run['textRun'].get('content', '')
            text_parts.append(content)

    return "".join(text_parts).strip()


def extract_slide_images(slide: Dict[str, Any], presentation_id: str, service: Any) -> List[Dict[str, Any]]:
    """
    Extract image metadata and download URLs from a slide.

    Args:
        slide (dict): Slide object from Google Slides API.
        presentation_id (str): ID of the presentation.
        service (googleapiclient.discovery.Resource): Google Slides service.

    Returns:
        list: List of image metadata dictionaries containing objectId, contentUrl, and properties.
    """
    images = []

    if 'pageElements' not in slide:
        return images

    for element in slide['pageElements']:
        if 'image' in element:
            image_info = {
                'objectId': element.get('objectId'),
                'contentUrl': element['image'].get('contentUrl'),
                'transform': element.get('transform'),
                'size': element.get('size')
            }
            images.append(image_info)
    
    return images


def extract_image_source_urls(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract source URLs from image ALT text (description field).

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        list: List of dictionaries with image info and source URLs.
              Each dict has {'objectId': str, 'description': str, 'source_urls': list}
    """
    image_sources = []

    if 'pageElements' not in slide:
        return image_sources

    for element in slide['pageElements']:
        if 'image' in element:
            object_id = element.get('objectId', '')
            # Get the description (ALT text)
            description = element.get('description', '')

            # Extract URLs from description using regex; strip trailing punctuation that
            # commonly appears in markdown-style links like `[label](https://example.com)`
            # — without this, the regex would capture `https://example.com)` and downstream
            # downloads would 404.
            urls = [u.rstrip('.,!?;:)\'\"]') for u in re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', description)]

            image_sources.append({
                'objectId': object_id,
                'description': description,
                'source_urls': urls
            })

    return image_sources


def download_slide_image(image_url: str) -> Optional[Image.Image]:
    """Download an image with retry and Wayback Machine fallback.

    Tries: direct GET (10s) -> direct GET with longer timeout (20s) -> Wayback snapshot.
    A transient CDN slowness no longer permanently zeros out an evaluator step.

    Args:
        image_url (str): URL of the image to download.

    Returns:
        PIL.Image.Image or None: Downloaded image or None if all strategies failed.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for attempt_timeout in (10, 20):
        try:
            response = requests.get(image_url, timeout=attempt_timeout, headers=headers, allow_redirects=True)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"download_slide_image attempt (timeout={attempt_timeout}s) failed for {image_url}: {e}")

    # Wayback fallback for permanently-dead URLs.
    try:
        wb_api = f"https://archive.org/wayback/available?url={image_url}"
        wb_resp = requests.get(wb_api, timeout=10)
        snapshot = wb_resp.json().get('archived_snapshots', {}).get('closest', {})
        wb_url = snapshot.get('url')
        if wb_url:
            wb_url = re.sub(r"(/web/\d+)/", r"\1im_/", wb_url, count=1)
            wb_img_resp = requests.get(wb_url, timeout=20, headers=headers)
            if wb_img_resp.status_code == 200:
                return Image.open(BytesIO(wb_img_resp.content))
    except Exception as e:
        print(f"download_slide_image wayback fallback failed for {image_url}: {e}")

    return None


def get_slide_background_color(slide: Dict[str, Any], presentation: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """
    Extract background color from a slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        dict or None: Color information (RGB values) or None if not available.
    """
    page_props = slide.get('pageProperties', {}).get('pageBackgroundFill', {})

    # Solid color fill
    if 'solidFill' in page_props:
        color_info = page_props['solidFill'].get('color', {})
        return _parse_color(color_info)
    elif page_props.get('propertyState') == 'INHERIT' and presentation:
        master_id = slide.get('slideProperties', {}).get('masterObjectId')
        if "masters" in presentation:
            for master in presentation['masters']:
                if master['objectId'] == master_id:
                    master_bg = master.get('pageProperties', {}).get('pageBackgroundFill', {})
                    if 'solidFill' in master_bg:
                        color_info = master_bg['solidFill'].get('color', {})
                        return _parse_color(color_info)
        
    # No background or unsupported type
    return None


def resolve_theme_color(theme_color_name: str, presentation: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Resolve a Slides themeColor name (e.g. 'ACCENT1', 'TEXT1') to an RGB dict via the
    master's color scheme. Returns None when the scheme can't be located or the entry isn't
    an rgbColor.

    Returns dict with `red`, `green`, `blue` keys (0-1 range) — same shape as
    `foregroundColor` from `get_text_style_from_shape`. Pass the full `presentation_data`
    object (not a single slide).
    """
    if not theme_color_name or not presentation:
        return None
    for master in presentation.get('masters', []):
        scheme = master.get('pageProperties', {}).get('colorScheme', {})
        for entry in scheme.get('colors', []):
            if entry.get('type') == theme_color_name and 'color' in entry:
                rgb = entry['color'].get('rgbColor', {})
                return {'red': rgb.get('red', 0), 'green': rgb.get('green', 0), 'blue': rgb.get('blue', 0)}
    return None


def _parse_color(color_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse color information from Slides API format.

    Args:
        color_info (dict): Color object from Slides API.

    Returns:
        dict: Dictionary with 'r', 'g', 'b' keys (0-1 range) or None.
    """
    if 'rgbColor' in color_info:
        rgb = color_info['rgbColor']
        return {
            'r': rgb.get('red', 0),
            'g': rgb.get('green', 0),
            'b': rgb.get('blue', 0)
        }

    # Theme colors would need more complex handling
    return None


def colors_are_different(color1: Optional[Dict[str, Any]], color2: Optional[Dict[str, Any]], threshold: float = 0.01) -> bool:
    """
    Check if two colors are sufficiently different.

    Args:
        color1 (dict): First color with 'r', 'g', 'b' keys.
        color2 (dict): Second color with 'r', 'g', 'b' keys.
        threshold (float): Minimum difference threshold (0-1 range).

    Returns:
        bool: True if colors are different enough.
    """
    if color1 is None or color2 is None:
        return True  # If we can't determine, assume different

    # Calculate Euclidean distance in RGB space
    r_diff = color1['r'] - color2['r']
    g_diff = color1['g'] - color2['g']
    b_diff = color1['b'] - color2['b']

    distance = (r_diff**2 + g_diff**2 + b_diff**2) ** 0.5

    return distance > threshold


def extract_slide_links(slide: Dict[str, Any]) -> List[str]:
    """
    Extract all URLs/links from a slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        list: List of URL strings found in the slide.
    """
    links = []

    if 'pageElements' not in slide:
        return links

    for element in slide['pageElements']:
        # Links in shape text
        if 'shape' in element and 'text' in element['shape']:
            shape_links = _extract_links_from_text_element(element['shape']['text'])
            links.extend(shape_links)

        # Links in tables
        if 'table' in element:
            for row in element['table'].get('tableRows', []):
                for cell in row.get('tableCells', []):
                    if 'text' in cell:
                        cell_links = _extract_links_from_text_element(cell['text'])
                        links.extend(cell_links)

    return list(set(links))  # Remove duplicates


def extract_slide_links_with_positions(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all URLs/links from a slide with their position information.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        list: List of dictionaries containing 'url' and 'bbox' keys.
              bbox contains x, y, width, height in EMUs.
    """
    links_with_positions = []

    if 'pageElements' not in slide:
        return links_with_positions

    for element in slide['pageElements']:
        # Get element position
        transform = element.get('transform', {})
        size = element.get('size', {})

        # Calculate bbox
        x = transform.get('translateX', 0)
        y = transform.get('translateY', 0)
        scale_x = transform.get('scaleX', 1)
        scale_y = transform.get('scaleY', 1)
        width = size.get('width', {}).get('magnitude', 0) * abs(scale_x)
        height = size.get('height', {}).get('magnitude', 0) * abs(scale_y)

        bbox = {'x': x, 'y': y, 'width': width, 'height': height}

        # Links in shape text
        if 'shape' in element and 'text' in element['shape']:
            shape_links = _extract_links_from_text_element(element['shape']['text'])
            for link in shape_links:
                links_with_positions.append({'url': link, 'bbox': bbox})

        # Links in tables
        if 'table' in element:
            for row in element['table'].get('tableRows', []):
                for cell in row.get('tableCells', []):
                    if 'text' in cell:
                        cell_links = _extract_links_from_text_element(cell['text'])
                        for link in cell_links:
                            links_with_positions.append({'url': link, 'bbox': bbox})

    return links_with_positions


def _extract_links_from_text_element(text_element: Dict[str, Any]) -> List[str]:
    """
    Extract links from a text element structure.

    Extracts both embedded hyperlinks and plain text URLs.

    Args:
        text_element (dict): Text element from Slides API.

    Returns:
        list: List of URL strings.
    """
    links = []

    # Pattern to match URLs in plain text
    url_pattern = re.compile(
        r'(?:'
        r'https?://[^\s<>"\'\)]+'                                      # scheme-prefixed
        r'|'
        r'www\.[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}(?:/[^\s<>"\'\)]*)?'   # www.example.com
        r'|'
        r'\b[a-zA-Z0-9][a-zA-Z0-9-]*\.(?:com|org|net|edu|gov|io|co|me|info|biz|tech|app|dev|ai|tv|store|shop|blog|news|us|uk|de|fr|jp|cn|au|ca|nz|in|br|mx|ru|kr|za|sg|hk|tw|nl|se|no|dk|fi|es|it|pl|tr|ie|pt|be|at|ch|cz|hu|ro|bg|ua)\b(?:/[^\s<>"\'\)]*)?'  # bare domain
        r')',
        re.IGNORECASE,
    )


    for text_run in text_element.get('textElements', []):
        if 'textRun' in text_run:
            # Extract embedded hyperlinks
            text_style = text_run['textRun'].get('style', {})
            if 'link' in text_style:
                url = text_style['link'].get('url', '')
                if url:
                    links.append(url)

            # Extract plain text URLs
            content = text_run['textRun'].get('content', '')
            if content:
                plain_urls = url_pattern.findall(content)
                for url in plain_urls:
                    # Strip trailing punctuation that's not part of the URL
                    # e.g. "(https://example.com/path)" -> "https://example.com/path"
                    while url and url[-1] in ')],.:;!':
                        # Keep closing parens if they have a matching open paren in the URL
                        if url[-1] == ')' and url.count('(') >= url.count(')'):
                            break
                        url = url[:-1]
                    if url:
                        links.append(url)

    return links


def find_slide_by_title_fuzzy(presentation: Dict[str, Any], title_text: str, threshold: int = 80) -> Optional[int]:
    """
    Find a slide by fuzzy matching its title text.

    Args:
        presentation (dict): Presentation object from Google Slides API.
        title_text (str): Text to search for in slide titles.
        threshold (int): Fuzzy matching threshold (0-100).

    Returns:
        int or None: Index of the slide (0-based) or None if not found.
    """
    from rapidfuzz import fuzz

    slides = presentation.get('slides', [])

    for idx, slide in enumerate(slides):
        slide_text = extract_slide_text(slide)

        # Try fuzzy matching
        if fuzz.partial_ratio(title_text.lower(), slide_text.lower()) >= threshold:
            return idx

    return None


def validate_bullet_points(slide: Dict[str, Any], min_count: int = 3) -> Tuple[bool, int]:
    """
    Check if a slide has at least the minimum number of non-empty bullet points.

    Args:
        slide (dict): Slide object from Google Slides API.
        min_count (int): Minimum number of bullet points required.

    Returns:
        tuple: (bool, int) - (passes validation, actual count of non-empty bullets).
    """
    # Use extract_bullet_point_texts which already filters empties
    bullet_texts = extract_bullet_point_texts(slide)
    bullet_count = len(bullet_texts)
    return bullet_count >= min_count, bullet_count


def extract_bullet_point_texts(slide: Dict[str, Any]) -> List[str]:
    """
    Extract the text content of all bullet points from a slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        list: List of bullet point text strings.
    """
    bullet_texts = []

    if 'pageElements' not in slide:
        return bullet_texts

    for element in slide['pageElements']:
        if 'shape' in element and 'text' in element['shape']:
            text_element = element['shape']['text']
            current_bullet_text = []
            in_bullet = False

            for text_elem in text_element.get('textElements', []):
                # Check if this starts a bullet point
                if 'paragraphMarker' in text_elem:
                    bullet = text_elem['paragraphMarker'].get('bullet', {})
                    if bullet:
                        # Save previous bullet if exists
                        if in_bullet and current_bullet_text:
                            bullet_texts.append(''.join(current_bullet_text).strip())
                            current_bullet_text = []
                        in_bullet = True
                    else:
                        # End of bullet section
                        if in_bullet and current_bullet_text:
                            bullet_texts.append(''.join(current_bullet_text).strip())
                            current_bullet_text = []
                        in_bullet = False

                # Collect text if we're in a bullet point
                if in_bullet and 'textRun' in text_elem:
                    content = text_elem['textRun'].get('content', '')
                    current_bullet_text.append(content)

            # Save last bullet if exists
            if in_bullet and current_bullet_text:
                bullet_texts.append(''.join(current_bullet_text).strip())

    # Filter out empty strings from trailing blank bullets
    return [t for t in bullet_texts if t]


def is_text_in_title_position(slide: Dict[str, Any], text: str) -> bool:
    """
    Check if specified text appears in a title placeholder or at the top of the slide.

    Args:
        slide (dict): Slide object from Google Slides API.
        text (str): Text to search for.

    Returns:
        bool: True if text is in a title position.
    """
    if 'pageElements' not in slide:
        return False

    text_lower = text.lower()

    for element in slide['pageElements']:
        if 'shape' in element:
            shape = element['shape']

            # Check if it's a title placeholder
            shape_type = shape.get('shapeType', '')
            placeholder = shape.get('placeholder', {})
            placeholder_type = placeholder.get('type', '')

            if placeholder_type in ['TITLE', 'CENTERED_TITLE', 'SUBTITLE']:
                if 'text' in shape:
                    element_text = _extract_text_from_text_element(shape['text'])
                    if text_lower in element_text.lower():
                        return True

            # Also check position - if text is in top ~20% of slide, consider it a title
            transform = element.get('transform', {})
            translate_y = transform.get('translateY', 0)

            if 'text' in shape:
                element_text = _extract_text_from_text_element(shape['text'])
                if text_lower in element_text.lower():
                    # Check if Y position is near top (measured in EMUs, typical slide height ~5143500)
                    if translate_y < 1500000:  # Top ~20% of slide
                        return True

    return False


def is_text_at_bottom(slide: Dict[str, Any], text: str, slide_height: Optional[int] = None) -> bool:
    """
    Check if specified text appears at the bottom of the slide.

    Args:
        slide (dict): Slide object from Google Slides API.
        text (str): Text to search for (can be a URL or any text).
        slide_height (int, optional): Slide height in EMUs. Defaults to standard 16:9.

    Returns:
        bool: True if the text is found in the bottom ~25% of the slide.
    """
    if 'pageElements' not in slide:
        return False

    height = slide_height or DEFAULT_SLIDE_HEIGHT_EMU
    BOTTOM_THRESHOLD = height * 0.75  # Bottom 25% of slide

    text_lower = text.lower()

    for element in slide['pageElements']:
        if 'shape' in element and 'text' in element['shape']:
            # Extract text from this element
            element_text = _extract_text_from_text_element(element['shape']['text'])

            # Check if our text is in this element
            if text_lower in element_text.lower():
                # Check position
                transform = element.get('transform', {})
                translate_y = transform.get('translateY', 0)

                if translate_y > BOTTOM_THRESHOLD:
                    return True

    return False


def is_link_at_bottom(slide: Dict[str, Any], slide_height: Optional[int] = None) -> bool:
    """
    Check if there's a link positioned at the bottom of the slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        bool: True if a link is found in the bottom ~20% of the slide.
    """
    if 'pageElements' not in slide:
        return False

    height = slide_height or DEFAULT_SLIDE_HEIGHT_EMU
    BOTTOM_THRESHOLD = height * 0.75  # Bottom 25% of slide

    for element in slide['pageElements']:
        if 'shape' in element and 'text' in element['shape']:
            # Check if this element has links
            links = _extract_links_from_text_element(element['shape']['text'])

            if links:
                # Check position
                transform = element.get('transform', {})
                translate_y = transform.get('translateY', 0)

                if translate_y > BOTTOM_THRESHOLD:
                    return True

    return False


def get_slide_element_positions(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all text elements with their vertical positions for ordering.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        list: List of dicts with 'text', 'y_position', and 'element_type'.
    """
    elements = []

    if 'pageElements' not in slide:
        return elements

    for element in slide['pageElements']:
        transform = element.get('transform', {})
        y_pos = transform.get('translateY', 0)

        if 'shape' in element and 'text' in element['shape']:
            text = _extract_text_from_text_element(element['shape']['text'])

            # Determine element type
            placeholder = element['shape'].get('placeholder', {})
            placeholder_type = placeholder.get('type', '')

            if placeholder_type in ['TITLE', 'CENTERED_TITLE']:
                elem_type = 'title'
            elif placeholder_type == 'SUBTITLE':
                elem_type = 'subtitle'
            elif placeholder_type == 'BODY':
                elem_type = 'body'
            else:
                elem_type = 'text'

            elements.append({
                'text': text,
                'y_position': y_pos,
                'element_type': elem_type
            })

    # Sort by y-position (top to bottom)
    elements.sort(key=lambda x: x['y_position'])

    return elements


def check_text_vertical_order(slide: Dict[str, Any], text_list: List[str]) -> bool:
    """
    Check if texts appear in the specified vertical order (top to bottom).

    Args:
        slide (dict): Slide object from Google Slides API.
        text_list (list): List of texts in expected order (top to bottom).

    Returns:
        bool: True if texts appear in the specified order.
    """
    elements = get_slide_element_positions(slide)

    # Find positions of each text in the list
    positions = []
    for search_text in text_list:
        search_lower = search_text.lower()

        for elem in elements:
            if search_lower in elem['text'].lower():
                positions.append(elem['y_position'])
                break
        else:
            # Text not found
            return False

    # Check if positions are in ascending order (top to bottom)
    for i in range(len(positions) - 1):
        if positions[i] >= positions[i + 1]:
            return False

    return True


def get_element_bbox(element: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a Slides API element (transform + size) to a bounding box dictionary.

    The Google Slides API uses EMUs (English Metric Units) for coordinates.
    1 inch = 914400 EMUs, 1 point = 12700 EMUs.

    The transform matrix includes scaleX and scaleY which must be multiplied
    with the raw size values to get the actual rendered dimensions.

    Args:
        element (dict): Page element from Google Slides API with 'transform' and 'size'.

    Returns:
        dict: Bounding box with 'x', 'y', 'width', 'height' in EMUs.
            Returns zeros if transform/size not available.
    """
    transform = element.get('transform', {})
    size = element.get('size', {})

    # Get raw width and height from size
    raw_width = size.get('width', {}).get('magnitude', 0)
    raw_height = size.get('height', {}).get('magnitude', 0)

    # Apply scale factors from transform (default to 1 if not present)
    scale_x = transform.get('scaleX', 1)
    scale_y = transform.get('scaleY', 1)

    return {
        'x': transform.get('translateX', 0),
        'y': transform.get('translateY', 0),
        'width': raw_width * abs(scale_x),
        'height': raw_height * abs(scale_y)
    }


_EMU_PER_PT = 12700
_AVG_CHAR_WIDTH_FACTOR = 0.55  # avg proportional-font glyph width in pt-units
_DEFAULT_FONT_PT = 14.0


def _find_parent_placeholder(parent_object_id: str, presentation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Look up a placeholder element by objectId in layouts then masters."""
    if not parent_object_id or not presentation:
        return None
    for collection_key in ('layouts', 'masters'):
        for page in presentation.get(collection_key, []) or []:
            for el in page.get('pageElements', []) or []:
                if el.get('objectId') == parent_object_id and 'shape' in el:
                    return el
    return None


def _extract_paragraphs(element: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull per-paragraph (alignment, font_pt, char_count) from an element's text."""
    paragraphs: List[Dict[str, Any]] = []
    text_element = (element.get('shape') or {}).get('text') or {}
    current = None
    for te in text_element.get('textElements', []):
        if 'paragraphMarker' in te:
            if current is not None:
                paragraphs.append(current)
            alignment = te['paragraphMarker'].get('style', {}).get('alignment')
            current = {'alignment': alignment, 'font_pt': None, 'chars': 0}
        elif 'textRun' in te:
            if current is None:
                current = {'alignment': None, 'font_pt': None, 'chars': 0}
            content = te['textRun'].get('content', '') or ''
            if content.endswith('\n'):
                content = content[:-1]
            current['chars'] += len(content)
            mag = te['textRun'].get('style', {}).get('fontSize', {}).get('magnitude')
            if mag and (current['font_pt'] is None or mag > current['font_pt']):
                current['font_pt'] = mag
    if current is not None:
        paragraphs.append(current)
    return paragraphs


def _resolve_paragraph_styles(element: Dict[str, Any], presentation: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve missing alignment / font_pt by walking parent placeholders."""
    paragraphs = _extract_paragraphs(element)
    if not paragraphs:
        return paragraphs
    parent_id = element.get('shape', {}).get('placeholder', {}).get('parentObjectId')
    seen_parents: set = set()
    while presentation and parent_id and parent_id not in seen_parents:
        if all(p.get('alignment') and p.get('font_pt') for p in paragraphs):
            break
        seen_parents.add(parent_id)
        parent_el = _find_parent_placeholder(parent_id, presentation)
        if not parent_el:
            break
        parent_paragraphs = _extract_paragraphs(parent_el)
        default = parent_paragraphs[0] if parent_paragraphs else {}
        for p in paragraphs:
            if not p.get('alignment') and default.get('alignment'):
                p['alignment'] = default['alignment']
            if not p.get('font_pt') and default.get('font_pt'):
                p['font_pt'] = default['font_pt']
        parent_id = parent_el.get('shape', {}).get('placeholder', {}).get('parentObjectId')
    for p in paragraphs:
        if not p.get('alignment'):
            p['alignment'] = 'START'
        if not p.get('font_pt'):
            p['font_pt'] = _DEFAULT_FONT_PT
    return paragraphs


def estimate_text_render_bbox(text_box: Dict[str, Any], presentation: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Approximate the bbox of rendered text within a text-box element.

    Tightens the container bbox down to where the glyphs actually live:
    height = sum of (line-height per paragraph, accounting for word-wrap),
    width = max paragraph text width (chars * font * char-factor), and the
    tight bbox is positioned within the container per (horizontal alignment,
    vertical contentAlignment).

    Inherited fontSize and paragraph alignment are resolved by walking the
    placeholder's parentObjectId chain through layouts/masters when
    `presentation` is provided. Autofit fontScale and lineSpacingReduction
    are applied to shrink the estimate when the text is auto-shrunk.

    Args:
        text_box: Entry from extract_text_boxes_from_slide() (needs 'bbox',
            'element').
        presentation: Optional full presentation dict for inheritance lookup.
            Without it, missing alignment defaults to START and missing font
            size defaults to 14pt.

    Returns:
        Tight bbox dict {x, y, width, height} in EMUs. Returns the raw
        container bbox if input is malformed.
    """
    bbox = text_box.get('bbox') or {}
    element = text_box.get('element') or {}
    shape = element.get('shape') or {}

    box_x = bbox.get('x', 0)
    box_y = bbox.get('y', 0)
    box_w = bbox.get('width', 0)
    box_h = bbox.get('height', 0)
    if box_h <= 0 or box_w <= 0:
        return dict(bbox)

    paragraphs = _resolve_paragraph_styles(element, presentation)
    if not paragraphs:
        return dict(bbox)

    autofit = shape.get('shapeProperties', {}).get('autofit', {}) or {}
    font_scale = autofit.get('fontScale') if autofit.get('fontScale') is not None else 1.0
    line_spacing_reduction = autofit.get('lineSpacingReduction') or 0.0
    line_height_factor = max(0.5, 1.2 * (1.0 - line_spacing_reduction))

    total_height = 0.0
    max_text_width = 0.0
    align_counts: Dict[str, int] = {}
    for p in paragraphs:
        font_pt = (p['font_pt'] or _DEFAULT_FONT_PT) * font_scale
        char_w = font_pt * _EMU_PER_PT * _AVG_CHAR_WIDTH_FACTOR
        line_h = font_pt * _EMU_PER_PT * line_height_factor
        chars = max(1, p['chars']) if p['chars'] else 1
        wrapped_lines = 1
        if char_w > 0:
            chars_per_line = max(1, int(box_w // char_w))
            wrapped_lines = max(1, (chars + chars_per_line - 1) // chars_per_line)
        total_height += wrapped_lines * line_h
        text_w = min(chars * char_w, box_w)
        if text_w > max_text_width:
            max_text_width = text_w
        align = (p['alignment'] or 'START').upper()
        align_counts[align] = align_counts.get(align, 0) + 1

    text_height = min(total_height, box_h)
    text_width = min(max_text_width, box_w)
    if text_width <= 0 or text_height <= 0:
        return dict(bbox)

    v_align = (shape.get('shapeProperties', {}).get('contentAlignment') or 'TOP').upper()
    if v_align == 'MIDDLE':
        tight_y = box_y + (box_h - text_height) / 2
    elif v_align == 'BOTTOM':
        tight_y = box_y + (box_h - text_height)
    else:
        tight_y = box_y

    h_align = max(align_counts, key=align_counts.get) if align_counts else 'START'
    if h_align in ('CENTER', 'JUSTIFIED'):
        tight_x = box_x + (box_w - text_width) / 2
    elif h_align == 'END':
        tight_x = box_x + (box_w - text_width)
    else:
        tight_x = box_x

    return {'x': tight_x, 'y': tight_y, 'width': text_width, 'height': text_height}


def extract_text_boxes_from_slide(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all text box elements from a slide with their positions and content.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        list: List of dictionaries, each containing:
            - 'objectId': Element ID
            - 'text': Text content
            - 'bbox': Bounding box dict with x, y, width, height
            - 'element': The full element for additional processing
    """
    text_boxes = []

    if 'pageElements' not in slide:
        return text_boxes

    for element in slide['pageElements']:
        if 'shape' in element and 'text' in element['shape']:
            text_content = _extract_text_from_text_element(element['shape']['text'])

            if text_content.strip():  # Only include non-empty text boxes
                text_boxes.append({
                    'objectId': element.get('objectId', ''),
                    'text': text_content,
                    'bbox': get_element_bbox(element),
                    'element': element
                })

    return text_boxes


def get_text_style_from_shape(shape: Dict[str, Any], presentation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract text styling information (color, font size) from a shape element.

    Args:
        shape (dict): Shape object from Google Slides API containing 'text'.
        presentation (dict, optional): Full presentation_data. When provided, themeColor
            references (e.g. 'ACCENT1') are auto-resolved to RGB via the master color
            scheme and populated into `foregroundColor` so callers checking color "just
            work" for theme-colored text. When omitted, theme colors are surfaced via
            `foregroundThemeColor` only and `foregroundColor` stays None.

    Returns:
        dict: Text style information containing:
            - 'foregroundColor': RGB color dict (resolved from theme when `presentation`
                provided) or None
            - 'foregroundThemeColor': themeColor name (e.g. 'ACCENT1') if shape uses one,
                else None
            - 'fontSize': Font size dict with 'magnitude' and 'unit', or None
            - 'bold': Boolean or None
            - 'italic': Boolean or None
    """
    result = {
        'foregroundColor': None,
        'foregroundThemeColor': None,
        'fontSize': None,
        'bold': None,
        'italic': None
    }

    if 'text' not in shape:
        return result

    text_element = shape['text']

    # Look through text elements for style information
    for text_run in text_element.get('textElements', []):
        if 'textRun' in text_run:
            style = text_run['textRun'].get('style', {})

            # Extract foreground color (first one wins — match other fields' semantics)
            if 'foregroundColor' in style and result['foregroundColor'] is None and result['foregroundThemeColor'] is None:
                color_info = style['foregroundColor'].get('opaqueColor', {})
                if 'rgbColor' in color_info:
                    rgb = color_info['rgbColor']
                    result['foregroundColor'] = {
                        'red': rgb.get('red', 0),
                        'green': rgb.get('green', 0),
                        'blue': rgb.get('blue', 0)
                    }
                elif 'themeColor' in color_info:
                    theme_name = color_info['themeColor']
                    result['foregroundThemeColor'] = theme_name
                    # Auto-resolve to RGB when caller supplied presentation — populates
                    # foregroundColor so existing color checks (is_text_color, is_text_red)
                    # work without callers needing theme-aware code.
                    if presentation is not None:
                        resolved = resolve_theme_color(theme_name, presentation)
                        if resolved is not None:
                            result['foregroundColor'] = resolved

            # Extract font size
            if 'fontSize' in style and result['fontSize'] is None:
                result['fontSize'] = style['fontSize']

            # Extract bold/italic
            if 'bold' in style and result['bold'] is None:
                result['bold'] = style['bold']
            if 'italic' in style and result['italic'] is None:
                result['italic'] = style['italic']

    return result

def is_text_color(text_style: Dict[str, Any], r: float, g: float, b: float, tolerance: float = 0.25) -> bool:
    """Check if text foreground color is close to the given RGB target.

    Compares the text's foreground color against a target RGB value using
    Euclidean distance in RGB space (0-1 range per channel).

    Args:
        text_style (dict): Text style from get_text_style_from_shape().
        r (float): Target red value (0.0 to 1.0).
        g (float): Target green value (0.0 to 1.0).
        b (float): Target blue value (0.0 to 1.0).
        tolerance (float): Maximum Euclidean distance to consider a match.
            Default 0.25 allows moderate variation.

    Returns:
        bool: True if text color is within tolerance of the target.
    """
    fg = text_style.get('foregroundColor')
    if not fg:
        return False

    dr = fg.get('red', 0) - r
    dg = fg.get('green', 0) - g
    db = fg.get('blue', 0) - b

    distance = (dr ** 2 + dg ** 2 + db ** 2) ** 0.5
    return distance <= tolerance


def is_text_big(text_style: Dict[str, Any], min_pt: float = 18, element: Dict[str, Any] = None) -> bool:
    """
    Check if font size is at least the specified minimum in points.

    Args:
        text_style (dict): Text style from get_text_style_from_shape().
        min_pt (float): Minimum font size in points. Default is 18pt.
        element (dict): Optional page element dict. When provided and fontSize
            is None, title/subtitle placeholders are assumed to inherit a large
            font from the master layout and return True.

    Returns:
        bool: True if font size >= min_pt, False otherwise.
    """
    font_size = text_style.get('fontSize')
    if not font_size:
        # Placeholder titles inherit fontSize from master/layout; the API
        # doesn't include the inherited value. Assume it meets min_pt.
        if element:
            ph_type = element.get('shape', {}).get('placeholder', {}).get('type', '')
            if ph_type in ('TITLE', 'CENTERED_TITLE', 'SUBTITLE'):
                return True
        return False

    magnitude = font_size.get('magnitude', 0)
    unit = font_size.get('unit', 'PT')

    if unit == 'PT':
        return magnitude >= min_pt
    elif unit == 'EMU':
        # 1 point = 12700 EMUs
        return magnitude >= min_pt * 12700

    return False


def find_url_below_image(image_bbox: dict, links_with_positions: list, tolerance: float = 0.3) -> Optional[str]:
    """Find a URL positioned directly below an image.

    Args:
        image_bbox: Bounding box of the image with x, y, width, height (in EMUs).
        links_with_positions: List of dicts with 'url' and 'bbox' keys.
        tolerance: Fraction of image width for horizontal alignment tolerance.

    Returns:
        URL string if found, None otherwise.
    """
    if not links_with_positions:
        return None

    img_bottom = image_bbox['y'] + image_bbox['height']
    img_left = image_bbox['x']
    img_right = image_bbox['x'] + image_bbox['width']

    best_url = None
    best_distance = float('inf')

    for link_info in links_with_positions:
        link_bbox = link_info['bbox']
        link_top = link_bbox['y']
        link_center_x = link_bbox['x'] + link_bbox['width'] / 2

        # Check if link is below the image (link top is at or below image bottom)
        # Allow some tolerance for slight overlaps
        vertical_threshold = image_bbox['height'] * 0.1  # 10% of image height tolerance
        if link_top < img_bottom - vertical_threshold:
            continue  # Link is not below the image

        # Check horizontal alignment - link center should be within image horizontal bounds
        # with some tolerance
        horizontal_tolerance = image_bbox['width'] * tolerance
        if link_center_x < img_left - horizontal_tolerance or link_center_x > img_right + horizontal_tolerance:
            continue  # Link is not horizontally aligned with image

        # Calculate distance from image bottom to link top
        distance = abs(link_top - img_bottom)

        # Prefer the closest link below the image
        if distance < best_distance:
            best_distance = distance
            best_url = link_info['url']

    return best_url


def get_slide_dimensions(presentation_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Extract slide dimensions in EMU. Returns (None, None) when pageSize is missing/malformed."""
    page_size = (presentation_data or {}).get('pageSize', {})
    if not isinstance(page_size, dict):
        return (None, None)

    width_obj = page_size.get('width', {})
    height_obj = page_size.get('height', {})
    width = width_obj.get('magnitude') if isinstance(width_obj, dict) else None
    height = height_obj.get('magnitude') if isinstance(height_obj, dict) else None
    if width is None or height is None:
        return (None, None)
    return (width, height)


def get_image_area_percentage_from_api(slide: Dict[str, Any], slide_width_emu: float = DEFAULT_SLIDE_WIDTH_EMU, slide_height_emu: float = DEFAULT_SLIDE_HEIGHT_EMU) -> float:
    """
    Calculate what percentage of the slide is covered by images using Google Slides API.

    Args:
        slide (dict): Slide object from Google Slides API.
        slide_width_emu (float): Slide width in EMUs. Defaults to standard 16:9 width.
        slide_height_emu (float): Slide height in EMUs. Defaults to standard 16:9 height.

    Returns:
        float: Percentage of slide area covered by images (0-100).
    """
    total_slide_area = slide_width_emu * slide_height_emu
    total_image_area = 0

    if 'pageElements' not in slide:
        return 0.0

    for element in slide['pageElements']:
        if 'image' in element:
            # Get size - check both direct size and nested structure
            size = element.get('size', {})

            # Handle nested magnitude structure
            width_obj = size.get('width', {})
            height_obj = size.get('height', {})

            # Extract magnitude value (could be dict or direct value)
            if isinstance(width_obj, dict):
                width = width_obj.get('magnitude', 0)
            else:
                width = width_obj

            if isinstance(height_obj, dict):
                height = height_obj.get('magnitude', 0)
            else:
                height = height_obj

            # Check for transform scaling (use abs() since negative values indicate flipping)
            transform = element.get('transform', {})
            scale_x = abs(transform.get('scaleX', 1.0))
            scale_y = abs(transform.get('scaleY', 1.0))

            # Apply scaling to get actual rendered size
            actual_width = width * scale_x
            actual_height = height * scale_y

            if actual_width > 0 and actual_height > 0:
                image_area = actual_width * actual_height
                total_image_area += image_area

    percentage = (total_image_area / total_slide_area) * 100

    return percentage


def extract_table_from_slide(slide: Dict[str, Any], normalize_text: bool = True) -> Optional[Dict[str, Any]]:
    """
    Extract structured table data from a slide.

    Args:
        slide (Dict[str, Any]): Google Slides API slide object.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing:
            - 'headers': List of header cell texts
            - 'rows': List of rows, each row is a dict mapping header to cell content
            - 'cell_colors': Dict mapping (row_idx, col_idx) to RGB color dict
            - 'num_columns': Number of columns
            - 'num_rows': Number of rows (excluding header)
            Returns None if no table found.
    """
    if 'pageElements' not in slide:
        return None

    for element in slide['pageElements']:
        if 'table' not in element:
            continue

        table = element['table']
        table_rows = table.get('tableRows', [])

        if not table_rows:
            continue

        # Extract headers from first row
        headers = []
        first_row = table_rows[0]
        for cell in first_row.get('tableCells', []):
            text_element = cell.get('text', {})
            cell_text = _extract_text_from_text_element(text_element)
            headers.append(cell_text)

        num_columns = len(headers)

        # Extract data rows and cell colors
        rows = []
        cell_colors = {}

        for row_idx, row in enumerate(table_rows):
            cells = row.get('tableCells', [])
            if row_idx == 0:
                # Store header colors
                for col_idx, cell in enumerate(cells):
                    color = _get_table_cell_background_color(cell)
                    cell_colors[(0, col_idx)] = color
                continue

            row_data = {}
            for col_idx, cell in enumerate(cells):
                text_element = cell.get('text', {})
                cell_text = _extract_text_from_text_element(text_element)
                if col_idx < len(headers):
                    row_data[headers[col_idx]] = cell_text

                # Store cell background color
                color = _get_table_cell_background_color(cell)
                cell_colors[(row_idx, col_idx)] = color

            rows.append(row_data)

        return {
            'headers': headers,
            'rows': rows,
            'cell_colors': cell_colors,
            'num_columns': num_columns,
            'num_rows': len(rows)
        }

    return None

def _extract_text_from_table_cell(cell: Dict[str, Any], normalize_text: bool = True) -> str:
    """
    Extract text content from a table cell.

    Args:
        cell (Dict[str, Any]): Table cell object from Google Slides API.

    Returns:
        str: Combined text content from all text elements in the cell.
    """
    if 'text' not in cell:
        return ""

    text_parts = []
    text_element = cell['text']

    if 'textElements' in text_element:
        for elem in text_element['textElements']:
            if 'textRun' in elem and 'content' in elem['textRun']:
                text_run = elem['textRun']['content']
                if normalize_text:
                    text_run = text_run.strip().lower()
                text_parts.append(text_run)

    return "".join(text_parts).strip()


def _get_table_cell_background_color(cell: Dict[str, Any]) -> Dict:
    """
    Extract background color name from a table cell.

    Args:
        cell (Dict[str, Any]): Table cell object from Google Slides API.
        threshold (float): Threshold for color detection (0.0-1.0). Default 0.2.

    Returns:
        dict: Dictionary with 'r', 'g', 'b' keys (0-1 range) or None.
    """
    if 'tableCellProperties' in cell:
        props = cell['tableCellProperties']
        if 'tableCellBackgroundFill' in props:
            fill = props['tableCellBackgroundFill']
            if 'solidFill' in fill:
                color = fill['solidFill'].get('color', {})
                return _parse_color(color)

    return None


def extract_speaker_notes_text(slide: Dict[str, Any]) -> str:
    """Extract all text from the speaker notes of a slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        str: Combined text from speaker notes, or empty string if none.
    """
    notes_page = slide.get('slideProperties', {}).get('notesPage', {})
    text_parts = []
    for element in notes_page.get('pageElements', []):
        if 'shape' in element and 'text' in element['shape']:
            for text_elem in element['shape']['text'].get('textElements', []):
                if 'textRun' in text_elem:
                    content = text_elem['textRun'].get('content', '')
                    text_parts.append(content)
    return ''.join(text_parts).strip()


def resolve_theme_color(theme_color_name: str, presentation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve a theme color name to RGB values using the presentation's color scheme.

    Args:
        theme_color_name (str): Theme color name (e.g., 'DARK2', 'ACCENT1').
        presentation (dict): Full presentation object from Google Slides API.

    Returns:
        dict: RGB color dict with 'r', 'g', 'b' keys (0-1 range), or None.
    """
    for master in presentation.get('masters', []):
        color_scheme = master.get('pageProperties', {}).get('colorScheme', {})
        for color_entry in color_scheme.get('colors', []):
            if color_entry.get('type') == theme_color_name:
                color_obj = color_entry.get('color', {})
                if 'rgbColor' in color_obj:
                    rgb = color_obj['rgbColor']
                else:
                    rgb = color_obj
                return {
                    'r': rgb.get('red', 0),
                    'g': rgb.get('green', 0),
                    'b': rgb.get('blue', 0)
                }
    return None


def get_shape_background_fill(element: Dict[str, Any], presentation: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Extract solid fill color from a shape element's background.

    Handles both explicit RGB colors and theme colors (resolved via
    presentation color scheme when provided).

    Args:
        element (dict): Page element from Google Slides API.
        presentation (dict): Full presentation object, needed to resolve theme colors.

    Returns:
        dict: RGB color dict with 'r', 'g', 'b' keys (0-1 range), or None.
    """
    if 'shape' not in element:
        return None
    shape_props = element['shape'].get('shapeProperties', {})
    bg_fill = shape_props.get('shapeBackgroundFill', {})
    if bg_fill.get('propertyState') == 'NOT_RENDERED':
        return None
    if 'solidFill' in bg_fill:
        color = bg_fill['solidFill'].get('color', {})
        if 'rgbColor' in color:
            rgb = color['rgbColor']
            return {
                'r': rgb.get('red', 0),
                'g': rgb.get('green', 0),
                'b': rgb.get('blue', 0)
            }
        if 'themeColor' in color and presentation:
            return resolve_theme_color(color['themeColor'], presentation)
    return None


def is_grey_color(color: Optional[Dict[str, Any]], min_val: float = 0.2, max_val: float = 0.95) -> bool:
    """Check if a color represents grey (R ~= G ~= B, not black, not white).

    Args:
        color (dict): RGB color dict with 'r', 'g', 'b' keys (0-1 range).
        min_val (float): Minimum average brightness to exclude near-black.
        max_val (float): Maximum average brightness to exclude near-white.

    Returns:
        bool: True if color is grey.
    """
    if not color:
        return False
    r, g, b = color.get('r', 0), color.get('g', 0), color.get('b', 0)
    max_diff = max(abs(r - g), abs(r - b), abs(g - b))
    avg = (r + g + b) / 3
    return max_diff < 0.15 and min_val < avg < max_val


def get_paragraph_alignment(shape: Dict[str, Any]) -> Optional[str]:
    """Extract the dominant paragraph alignment from a shape's text.

    Reads the first explicit alignment value found in paragraphMarker style.

    Args:
        shape (dict): Shape object from Google Slides API containing 'text'.

    Returns:
        str: Alignment value ('LEFT', 'CENTER', 'RIGHT', 'JUSTIFIED', 'START',
            'END'), or None if not set.
    """
    if 'text' not in shape:
        return None
    for text_elem in shape['text'].get('textElements', []):
        if 'paragraphMarker' in text_elem:
            alignment = text_elem['paragraphMarker'].get('style', {}).get('alignment')
            if alignment:
                return alignment
    return None


def is_text_centered(shape: Dict[str, Any], bbox: Dict[str, float], slide_width: float,
                     tolerance: float = 0.15) -> bool:
    """Check if text in a shape appears horizontally centered on a slide.

    Passes if EITHER:
    1. The bounding box center is within ``tolerance`` of the slide center
       (positional centering — works regardless of paragraph alignment), OR
    2. The paragraph alignment is explicitly set to CENTER.

    This handles all practical centering patterns: full-width boxes with CENTER
    alignment, narrow boxes positioned at the slide center, and combinations.

    Args:
        shape (dict): Shape object from Google Slides API containing 'text'.
        bbox (dict): Bounding box with 'x' and 'width' keys (EMUs).
        slide_width (float): Slide width in EMUs.
        tolerance (float): Fraction of slide width allowed as center offset (default 0.15).

    Returns:
        bool: True if text appears centered on the slide.
    """
    box_center_x = bbox['x'] + bbox['width'] / 2
    geometric_centered = abs(box_center_x - slide_width / 2) < slide_width * tolerance
    alignment = get_paragraph_alignment(shape)
    return geometric_centered or alignment == 'CENTER'


def is_text_left_aligned(shape: Dict[str, Any], bbox: Dict[str, float], slide_width: float) -> bool:
    """Check if text in a shape is left-aligned on a slide.

    Uses paragraph alignment as the primary signal. Falls back to bounding box
    position only when no explicit alignment is set.

    Args:
        shape (dict): Shape object from Google Slides API containing 'text'.
        bbox (dict): Bounding box with 'x' key (EMUs).
        slide_width (float): Slide width in EMUs.

    Returns:
        bool: True if text is left-aligned.
    """
    alignment = get_paragraph_alignment(shape)
    if alignment in ('LEFT', 'START'):
        return True
    if alignment in ('CENTER', 'RIGHT', 'END', 'JUSTIFIED'):
        return False
    # No explicit alignment set: fall back to bbox position
    return bbox['x'] < slide_width * 0.25
