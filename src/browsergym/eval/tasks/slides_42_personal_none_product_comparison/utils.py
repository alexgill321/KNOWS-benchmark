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
            "content": [{"type": "text", "text": f"""You are a data extraction assistant. Extract the information in the provided text and format it as specified. 
            
Always respond with valid JSON only, no other text."""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"""Extract the following electronicdevice summaries and recommendations from this Google slide text.
            
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


def extract_table_from_slide(slide: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            cell_text = _extract_text_from_table_cell(cell)
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
                cell_text = _extract_text_from_table_cell(cell)
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


def _extract_text_from_table_cell(cell: Dict[str, Any]) -> str:
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
                text_parts.append(elem['textRun']['content'])
    
    return "".join(text_parts).strip()


def _get_cell_background_color(cell: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Extract background color from a table cell.
    
    Args:
        cell (Dict[str, Any]): Table cell object from Google Slides API.
    
    Returns:
        Optional[Dict[str, float]]: RGB color dict with 'r', 'g', 'b' keys (0.0-1.0),
            or None if no color found.
    """
    if 'tableCellProperties' in cell:
        props = cell['tableCellProperties']
        if 'tableCellBackgroundFill' in props:
            fill = props['tableCellBackgroundFill']
            if 'solidFill' in fill:
                color = fill['solidFill'].get('color', {})
                if 'rgbColor' in color:
                    rgb = color['rgbColor']
                    return {
                        'r': rgb.get('red', 0),
                        'g': rgb.get('green', 0),
                        'b': rgb.get('blue', 0)
                    }
    
    return None

