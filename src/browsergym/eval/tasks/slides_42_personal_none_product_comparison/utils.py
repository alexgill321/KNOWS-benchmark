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
            cell_text = _extract_text_from_table_cell(cell, normalize_text=normalize_text)
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
                    color = _get_cell_background_color(cell)
                    cell_colors[(0, col_idx)] = color
                continue
            
            row_data = {}
            for col_idx, cell in enumerate(cells):
                cell_text = _extract_text_from_table_cell(cell, normalize_text=normalize_text)
                if col_idx < len(headers):
                    row_data[headers[col_idx]] = cell_text
                
                # Store cell background color
                color = _get_cell_background_color(cell)
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


def _get_cell_background_color(cell: Dict[str, Any], threshold: float = 0.2) -> str:
    """
    Extract background color name from a table cell.
    
    Args:
        cell (Dict[str, Any]): Table cell object from Google Slides API.
        threshold (float): Threshold for color detection (0.0-1.0). Default 0.2.
    
    Returns:
        str: Color name ('red', 'green', 'yellow', or 'unknown').
    """
    if 'tableCellProperties' in cell:
        props = cell['tableCellProperties']
        if 'tableCellBackgroundFill' in props:
            fill = props['tableCellBackgroundFill']
            if 'solidFill' in fill:
                color = fill['solidFill'].get('color', {})
                if 'rgbColor' in color:
                    rgb = color['rgbColor']
                    r = rgb.get('red', 0)
                    g = rgb.get('green', 0)
                    b = rgb.get('blue', 0)
                    
                    # Detect color based on RGB values with threshold
                    return _detect_color_name(r, g, b, threshold)
    
    return 'unknown'


def _detect_color_name(r: float, g: float, b: float, threshold: float = 0.2) -> str:
    """
    Detect color name from RGB values.
    
    Args:
        r, g, b (float): RGB values (0.0-1.0).
        threshold (float): Threshold for color channel detection (0.0-1.0).
    
    Returns:
        str: Color name ('red', 'green', 'yellow', or 'unknown').
    """
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
        if expected_item[0] != actual_item[0]:
            return False
    
    return True
