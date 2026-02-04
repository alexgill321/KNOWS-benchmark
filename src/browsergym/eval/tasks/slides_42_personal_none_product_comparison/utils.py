"""
Helpers for slides_42 task evaluation.

These helpers use existing eval_utils functions where possible and add
small, task-specific utilities (loading gold devices, checking title bold,
parsing simple slide tables and colors, etc.).
"""
import os
from typing import Dict, Any

from src.browsergym.eval.eval_utils.text_utils import keyword_exact_match

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "instance_1", "data")
GOLD_DEVICES_PATH = os.path.join(DATA_DIR, "gold_devices.txt")

def text_matches_style(slide: Dict[str, Any], text: str, style: str) -> bool:
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