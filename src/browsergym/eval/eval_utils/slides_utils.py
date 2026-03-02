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
            translate_y = transform.get('translateY', float('inf'))
            content_alignment = shape.get('shapeProperties', {}).get('contentAlignment', {})
            
            if 'text' in shape and (translate_y < 1000000 or 'top' in content_alignment.lower()):  # Top ~20% of slide
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

            # Extract URLs from description using regex
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', description)

            image_sources.append({
                'objectId': object_id,
                'description': description,
                'source_urls': urls
            })

    return image_sources


def download_slide_image(image_url: str) -> Optional[Image.Image]:
    """
    Download an image from a URL and return as PIL Image.

    Args:
        image_url (str): URL of the image to download.

    Returns:
        PIL.Image.Image or None: Downloaded image or None if failed.
    """
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")

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
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
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
                links.extend(plain_urls)

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
    Check if a slide has at least the minimum number of bullet points.

    Args:
        slide (dict): Slide object from Google Slides API.
        min_count (int): Minimum number of bullet points required.

    Returns:
        tuple: (bool, int) - (passes validation, actual count).
    """
    bullet_count = 0

    if 'pageElements' not in slide:
        return False, 0

    for element in slide['pageElements']:
        if 'shape' in element and 'text' in element['shape']:
            text_element = element['shape']['text']

            for paragraph in text_element.get('textElements', []):
                if 'paragraphMarker' in paragraph:
                    bullet = paragraph['paragraphMarker'].get('bullet', {})
                    if bullet:  # Has bullet formatting
                        bullet_count += 1

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

    return bullet_texts


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
            translate_y = transform.get('translateY', float('inf'))

            if 'text' in shape:
                element_text = _extract_text_from_text_element(shape['text'])
                if text_lower in element_text.lower():
                    # Check if Y position is near top (measured in EMUs, typical slide height ~5143500)
                    if translate_y < 1000000:  # Top ~20% of slide
                        return True

    return False


def is_text_at_bottom(slide: Dict[str, Any], text: str) -> bool:
    """
    Check if specified text appears at the bottom of the slide.

    Args:
        slide (dict): Slide object from Google Slides API.
        text (str): Text to search for (can be a URL or any text).

    Returns:
        bool: True if the text is found in the bottom ~25% of the slide.
    """
    if 'pageElements' not in slide:
        return False

    # Typical slide height in EMUs
    SLIDE_HEIGHT = 5143500
    BOTTOM_THRESHOLD = SLIDE_HEIGHT * 0.75  # Bottom 25% of slide

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


def is_link_at_bottom(slide: Dict[str, Any]) -> bool:
    """
    Check if there's a link positioned at the bottom of the slide.

    Args:
        slide (dict): Slide object from Google Slides API.

    Returns:
        bool: True if a link is found in the bottom ~20% of the slide.
    """
    if 'pageElements' not in slide:
        return False

    # Typical slide height in EMUs
    SLIDE_HEIGHT = 5143500
    BOTTOM_THRESHOLD = SLIDE_HEIGHT * 0.75  # Bottom 25% of slide

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


def get_text_style_from_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract text styling information (color, font size) from a shape element.

    Args:
        shape (dict): Shape object from Google Slides API containing 'text'.

    Returns:
        dict: Text style information containing:
            - 'foregroundColor': RGB color dict or None
            - 'fontSize': Font size dict with 'magnitude' and 'unit', or None
            - 'bold': Boolean or None
            - 'italic': Boolean or None
    """
    result = {
        'foregroundColor': None,
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

            # Extract foreground color
            if 'foregroundColor' in style and result['foregroundColor'] is None:
                color_info = style['foregroundColor'].get('opaqueColor', {})
                if 'rgbColor' in color_info:
                    rgb = color_info['rgbColor']
                    result['foregroundColor'] = {
                        'red': rgb.get('red', 0),
                        'green': rgb.get('green', 0),
                        'blue': rgb.get('blue', 0)
                    }

            # Extract font size
            if 'fontSize' in style and result['fontSize'] is None:
                result['fontSize'] = style['fontSize']

            # Extract bold/italic
            if 'bold' in style and result['bold'] is None:
                result['bold'] = style['bold']
            if 'italic' in style and result['italic'] is None:
                result['italic'] = style['italic']

    return result

def is_text_red(text_style: Dict[str, Any], threshold: float = 0.7) -> bool:
    """
    Check if text foreground color is red.

    Args:
        text_style (dict): Text style from get_text_style_from_shape().
        threshold (float): Minimum red value and maximum green/blue values.
            Default 0.7 means red > 0.7 and green < 0.3 and blue < 0.3.

    Returns:
        bool: True if text color is red, False otherwise.
    """
    fg_color = text_style.get('foregroundColor')
    if not fg_color:
        return False

    red = fg_color.get('red', 0)
    green = fg_color.get('green', 0)
    blue = fg_color.get('blue', 0)

    return red > threshold and green < (1 - threshold) and blue < (1 - threshold)


def is_text_big(text_style: Dict[str, Any], min_pt: float = 18) -> bool:
    """
    Check if font size is at least the specified minimum in points.

    Args:
        text_style (dict): Text style from get_text_style_from_shape().
        min_pt (float): Minimum font size in points. Default is 18pt.

    Returns:
        bool: True if font size >= min_pt, False otherwise.
    """
    font_size = text_style.get('fontSize')
    if not font_size:
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
        distance = link_top - img_bottom

        # Prefer the closest link below the image
        if distance < best_distance:
            best_distance = distance
            best_url = link_info['url']

    return best_url


def get_slide_dimensions(presentation_data: Dict[str, Any]) -> Tuple[float, float]:
    """
    Extract slide dimensions from presentation data.

    Args:
        presentation_data (dict): Presentation object from Google Slides API.

    Returns:
        tuple: (width_emu, height_emu) in English Metric Units.
    """
    page_size = presentation_data.get('pageSize', {})

    width_obj = page_size.get('width', {})
    height_obj = page_size.get('height', {})

    width = width_obj.get('magnitude', DEFAULT_SLIDE_WIDTH_EMU) if isinstance(width_obj, dict) else DEFAULT_SLIDE_WIDTH_EMU
    height = height_obj.get('magnitude', DEFAULT_SLIDE_HEIGHT_EMU) if isinstance(height_obj, dict) else DEFAULT_SLIDE_HEIGHT_EMU

    return width, height


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
