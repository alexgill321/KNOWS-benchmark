"""
Helpers for slides_42 task evaluation.

These helpers use existing eval_utils functions where possible and add
small, task-specific utilities (loading gold devices, checking title bold,
parsing simple slide tables and colors, etc.).
"""
import os
from typing import List, Literal, Optional, Tuple, Dict, Any, Union
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from src.browsergym.knows.eval.eval_utils.llm_utils import (
    extract_json_with_llm as _extract_json_with_llm,
    evaluate_with_llm as _evaluate_with_llm,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keyword_exact_match

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "instance_1", "data")
GOLD_DEVICES_PATH = os.path.join(DATA_DIR, "gold_devices.txt")


def extract_device_info_with_llm(task_text: str, model: Any) -> Optional[Dict[str, Any]]:
    """
    Extract device summaries and recommendations from slide text using an LLM.

    Args:
        slide_text (str): Raw text extracted from a slide.
        model (Any): Callable LLM interface.

    Returns:
        Optional[Dict[str, Dict[str, str]]]: A dictionary mapping a list of [device_name, summary] and [device_name, recommendation] to corresponding keys in a dictionary. Example:

            {
                "summary":[["MacBook Air","The MacBook Air is a lightweight..."], ["Dell XPS 13", "The Dell XPS 13 is a powerful ..."]],
                "recommendation":[["MacBook Air","Recommended for ..."], ["Dell XPS 13", "Recommended for ..."]]
            }
    """
    return _extract_json_with_llm(task_text, model)


def evaluate_device_info_with_llm(task_text: str, model: Any, return_type: Literal["bool", "str", "json"] = "bool") -> Optional[Union[bool, str, Any]]:
    """Quick boolean check using an LLM to validate task text.

    Args:
        task_text (str): The text to validate.
        model (Any): Callable LLM interface.
        return_type (str): Format of the returned result: 'bool', 'str', or 'json'.

    Returns:
        Optional[Union[bool, str, Any]]: Result in the specified format, or None on error.
    """
    return _evaluate_with_llm(task_text, model, return_type=return_type)


def detect_color_name(color: Dict, threshold: float = 0.2) -> str:
    """
    Detect color name (red, yellow, or green) from RGB values.
    
    Args:
        color: a dictionary of keys r, g and b (0.0-1.0).
        threshold (float): Threshold for color channel detection (0.0-1.0).
    
    Returns:
        str: Color name ('red', 'green', 'yellow', or 'unknown').
    """
    r, g, b = color['r'], color['g'], color['b']
    high_threshold = 1.0 - threshold
    low_threshold = threshold
    
    # Red: high R, low G, low B
    if r >= high_threshold and g <= low_threshold and b <= low_threshold:
        return 'red'
    
    # Green: low R, high G, low B
    if r <= low_threshold and g >= high_threshold and b <= low_threshold:
        return 'green'
    
    # Yellow: high R, high G, low B
    if r >= high_threshold and g >= high_threshold and b <= low_threshold:
        return 'yellow'
    
    return 'unknown'

def validate_rankings(expected_ranking: Dict[str, int], actual_ranking: Dict[str, int]) -> bool:
    """
    Validate that two rankings express the same ordering of items.

    Compares rankings by group (one group per distinct rank value) rather than
    by index, so ties are considered consistent regardless of the order the
    tied items appear in. Items keys are compared case-insensitively.

    Args:
        expected_ranking (Dict[str, int]): First ranking mapping item to rank (lower is better).
        actual_ranking (Dict[str, int]): Second ranking mapping item to rank.

    Returns:
        bool: True if rankings express the same ordering, False otherwise.
    """
    if not isinstance(expected_ranking, dict) or not isinstance(actual_ranking, dict):
        return False
    # Both must contain the same set of items (case-insensitive) for the
    # comparison to be meaningful.
    exp_keys = {str(k).strip().lower() for k in expected_ranking.keys()}
    act_keys = {str(k).strip().lower() for k in actual_ranking.keys()}
    if exp_keys != act_keys:
        return False

    def _group_by_rank(ranking):
        groups = {}
        for k, v in ranking.items():
            try:
                rank_val = int(v)
            except (TypeError, ValueError):
                return None
            groups.setdefault(rank_val, set()).add(str(k).strip().lower())
        return groups

    exp_groups = _group_by_rank(expected_ranking)
    act_groups = _group_by_rank(actual_ranking)
    if exp_groups is None or act_groups is None:
        return False

    # Sort groups by rank (lower = better) and compare the sequence of item
    # sets. Equal sets at the same position mean the orderings match (with
    # ties handled correctly).
    exp_sequence = [items for _, items in sorted(exp_groups.items())]
    act_sequence = [items for _, items in sorted(act_groups.items())]
    return exp_sequence == act_sequence

def download_images_from_url(url, folder):
    from PIL import Image
    # Only accept these image extensions
    allowed_exts = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # 1. Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # 2. Get the HTML of the website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. Find all <img> tags
    img_tags = soup.find_all('img')
    downloaded_files = []
    download_errors = []
    for i, img in enumerate(img_tags):
        # Get the 'src' attribute
        img_url = img.get('src')
        if not img_url:
            continue

        # Handle relative URLs (e.g., /images/pic.jpg -> https://site.com/images/pic.jpg)
        img_url = urljoin(url, img_url)

        try:
            # Extract extension from URL path
            parsed = urlparse(img_url)
            _, ext = os.path.splitext(parsed.path or "")
            ext = ext.lower().lstrip('.') if ext else ''

            # Download the image data
            response = requests.get(img_url, timeout=10)
            content_type = response.headers.get('Content-Type', '').lower()

            # Prefer URL extension when valid
            if ext and ext in allowed_exts:
                chosen_ext = ext
            else:
                # Map common content-types to extensions
                ct_map = {
                    'image/png': 'png',
                    'image/jpeg': 'jpg',
                    'image/jpg': 'jpg',
                    'image/bmp': 'bmp',
                    'image/tiff': 'tiff',
                    'image/x-tiff': 'tiff'
                }
                ct = content_type.split(';')[0].strip()
                chosen_ext = ct_map.get(ct)

            # Skip if extension is not allowed
            if not chosen_ext or chosen_ext not in allowed_exts:
                continue
            
            # Create a filename
            filename = os.path.basename(urlparse(img_url).path)
            if not filename or '.' not in filename:
                filename = f"image_{i}.{chosen_ext}"
            else:
                # Ensure correct extension
                name_without_ext = os.path.splitext(filename)[0]
                filename = f"{name_without_ext}.{chosen_ext}"
                
            filepath = os.path.join(folder, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            # Validate that the file is a real image
            try:
                img = Image.open(filepath)
                img.verify()
            except Exception as e:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
                download_errors.append(f"{img_url}: invalid image ({e})")
                continue
            downloaded_files.append(filename)
        except Exception as e:
            # Capture the error per-image so callers can distinguish "no images
            # in source" from "every download failed".
            download_errors.append(f"{img_url}: {e}")
            continue
    if download_errors and not downloaded_files:
        # Surface a brief summary so failures are visible in stdout (without
        # printing every failed URL when partial successes occurred).
        print(f"download_images_from_url: {len(download_errors)} image(s) failed for {url}; first error: {download_errors[0]}")
    return downloaded_files

def ensure_scheme(url: str, default: str = "https") -> str:
    """Prepend a scheme if missing. Returns unchanged if url is None/empty."""
    if not url:
        return url
    url = url.strip()
    if url.startswith(("http://", "https://")):
        return url
    if url.startswith("//"):
        return f"{default}:{url}"
    return f"{default}://{url}"