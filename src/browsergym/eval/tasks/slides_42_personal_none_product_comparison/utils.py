"""
Helpers for slides_42 task evaluation.

These helpers use existing eval_utils functions where possible and add
small, task-specific utilities (loading gold devices, checking title bold,
parsing simple slide tables and colors, etc.).
"""
from cmath import exp
import os
import json
from typing import List, Literal, Optional, Tuple, Dict, Any, Union
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from src.browsergym.eval.eval_utils.text_utils import keyword_exact_match

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
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"""You are a data extraction assistant. Extract the information in the provided text and format it as specified. 
            
Always respond with valid JSON only, no other text."""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": task_text}]
        }
    ]
        
    try:
        # Prase JSON from response
        response = model(messages).strip().lower()
        
        # Handle markdown code blocks
        if "```" in response:
            lines = response.split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            if json_lines:
                response = "\n".join(json_lines)
                
        json_data = json.loads(response)
        return json_data

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return None
    
def evaluate_device_info_with_llm(task_text: str, model: Any, return_type: Literal["bool", "str", "json"]="bool") -> Optional[Union[bool, str, Any]]:
    """Quick boolean check using an LLM to validate task text.

    Args:
        task_text (str): The text to validate.
        model (Any): Callable LLM interface.

    Returns:
        bool: True if model indicates the text is valid (contains "yes");
        False on any other response or error.
    """
    return_type_instruction = {
        "bool": "Response with ONLY 'yes' or 'no'.",
        "str": "Format your response strictly as specified in the task instructions.",
        "json": "Respond with the JSON format specified in the task instructions."
    }
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"""You are a helpful assistant who evaluates whether a text satisfies the requirements of the task. 
                         
            {return_type_instruction.get(return_type, "")}"""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{task_text}"}]
        }
    ]
    try:
        response = model(messages).strip().lower()
        
        if return_type == "bool":
            return "yes" in response
        elif return_type == "str":
            return response
        elif return_type == "json":
            # Handle markdown code blocks
            if "```" in response:
                lines = response.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                if json_lines:
                    response = "\n".join(json_lines)
                    
            data_json = json.loads(response)
            return data_json
        else:
            return None
        
    except Exception as e:
            print(f"LLM failed to evaluate slide text: {e}")
            return None


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
    Validate that two rankings are consistent (e.g., higher score means better rank).
    
    Args:
        expected_ranking (Dict[str, int]): First ranking mapping item to rank (lower is better).
        actual_ranking (Dict[str, int]): Second ranking mapping item to rank.
    
    Returns:
        bool: True if rankings are consistent, False otherwise.
    """
    issues = []
    
    sorted_expected = sorted(expected_ranking.items(), key=lambda x: x[1])
    sorted_actual = sorted(actual_ranking.items(), key=lambda x: x[1])
    # Check if ranks are consistent (e.g., if rank1 is better than another item, rank2 should also reflect that)
    for expected_item, actual_item in zip(sorted_expected, sorted_actual):
        if expected_item[0].lower() != actual_item[0].lower():
            return False
    
    return True

def download_images_from_url(url, folder):
    from PIL import Image
    # Only accept these image extensions
    allowed_exts = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"}
    
    # 1. Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # 2. Get the HTML of the website
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 3. Find all <img> tags
    img_tags = soup.find_all('img')
    # print(f"Found {len(img_tags)} images.")
    downloaded_files = []
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
                    'image/gif': 'gif',
                    'image/webp': 'webp',
                    'image/bmp': 'bmp',
                    'image/tiff': 'tiff',
                    'image/x-tiff': 'tiff'
                }
                ct = content_type.split(';')[0].strip()
                chosen_ext = ct_map.get(ct)

            # Skip if extension is not allowed
            if not chosen_ext or chosen_ext not in allowed_exts:
                # print(f"Skipping {img_url} (unsupported type)")
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
            except Exception:
                os.remove(filepath)
                continue
            # print(f"Downloaded: {filename}")
            downloaded_files.append(filename)
        except Exception as e:
            # print(f"Could not download {img_url}: {e}")
            pass
    return downloaded_files
