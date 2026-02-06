"""
Helpers for slides_42 task evaluation.

These helpers use existing eval_utils functions where possible and add
small, task-specific utilities (loading gold devices, checking title bold,
parsing simple slide tables and colors, etc.).
"""
import os
import json
from typing import List, Optional, Tuple, Dict, Any

from src.browsergym.eval.eval_utils.text_utils import keyword_exact_match

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "instance_1", "data")
GOLD_DEVICES_PATH = os.path.join(DATA_DIR, "gold_devices.txt")


def text_matches_style(slide: Dict[str, Any], text: str, style: str) -> bool:
    """
    Check whether a given `text` appears in `slide` with a specific text `style`.

    Args:
        slide (Dict[str, Any]): Google Slides API slide object.
        text (str): The substring to search for inside text runs.
        style (str): The style key to check on the matching text run
            (e.g., 'bold', 'italic').

    Returns:
        bool: True if a text run containing `text` is found and that run's
        style has the requested `style` set; False otherwise.
    """
    if 'pageElements' not in slide:
        return ""

    for element in slide['pageElements']:
        # Extract from shapes with text
        if 'shape' in element and 'text' in element['shape'] and 'textElements' in element['shape']['text']:
            for text_element in element['shape']['text']['textElements']:
                if 'textRun' in text_element and 'content' in text_element['textRun'] and 'style' in text_element['textRun']:
                    shape_text = text_element['textRun']['content']
                    styles = text_element['textRun']['style']
                    if keyword_exact_match(shape_text, text, substring=True):
                        if styles.get(style):
                            return True
                    else:
                            return False

    return False

def extract_device_info_with_llm(slide_text: str, model: Any) -> Optional[Dict[str, Dict[str, str]]]:
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
            "content": [{"type": "text", "text": f"""You are a data extraction assistant. Extract the summaries about some electronic devices and their recommendations from google slides text. 
            
Always respond with valid JSON only, no other text."""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"""Extract the device summaries and recommendations from this Google slide text.
            
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
{slide_text}"""}]
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
                
        data = json.loads(response)
        data_map = {}
        
        for summary_item,rec_item in zip(data["summary"], data["recommendation"]):
            device_name = summary_item[0] if len(summary_item) > 0 else ""
            data_map[device_name] = {}
            data_map[device_name]['summary'] = summary_item[1] if len(summary_item) > 1 else ""
            data_map[device_name]['recommendation'] = rec_item[1] if len(rec_item) > 1 else ""
            
        return data_map

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return None
    
def content_is_valid(task_text: str, model: Any) -> bool:
    """Quick boolean check using an LLM to validate task text.

    Args:
        task_text (str): The text to validate.
        model (Any): Callable LLM interface.

    Returns:
        bool: True if model indicates the text is valid (contains "yes");
        False on any other response or error.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant who evaluates whether a the text satisfies the requirements of the task. Response with ONLY 'yes' or 'no'."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"{task_text}"}]
        }
    ]
    try:
        response = model(messages).strip().lower()
        return "yes" in response
        
    except Exception as e:
            print(f"LLM failed to evaluate slide text: {e}")
            return False