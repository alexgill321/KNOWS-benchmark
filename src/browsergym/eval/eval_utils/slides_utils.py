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


def extract_slide_text(slide: Dict[str, Any]) -> str:
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

    return " ".join(text_parts)


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


def get_slide_background_color(slide: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
