import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Any, List, Dict


# =============================================================================
# SheetTable Class
# =============================================================================

@dataclass
class SheetTable:
    """
    Represents a table extracted from a Google Sheet with position metadata.

    Attributes:
        df: The table data as a pandas DataFrame.
        start_col: 0-indexed starting column position in the sheet.
        end_col: 0-indexed ending column position (exclusive).
        start_row: 0-indexed starting row position in the sheet.
        end_row: 0-indexed ending row position (exclusive).
        sheet_name: Name of the sheet tab containing this table.
    """
    df: pd.DataFrame
    start_col: int = 0
    end_col: int = 0
    start_row: int = 0
    end_row: int = 0
    sheet_name: str = ""

    @property
    def col_letter(self) -> str:
        """Return starting column as Excel-style letter (e.g., 'A', 'K', 'AA')."""
        return self._col_index_to_letter(self.start_col)

    @property
    def end_col_letter(self) -> str:
        """Return ending column as Excel-style letter."""
        return self._col_index_to_letter(self.end_col - 1) if self.end_col > 0 else ""

    @property
    def columns(self) -> List[str]:
        """Return list of column names."""
        return list(self.df.columns)

    @property
    def num_rows(self) -> int:
        """Return number of data rows (excluding header)."""
        return len(self.df)

    @property
    def num_cols(self) -> int:
        """Return number of columns."""
        return len(self.df.columns)

    def __len__(self) -> int:
        """Return number of data rows."""
        return len(self.df)

    @staticmethod
    def _col_index_to_letter(idx: int) -> str:
        """Convert 0-indexed column number to Excel-style letter."""
        result = ""
        while idx >= 0:
            result = chr(idx % 26 + ord('A')) + result
            idx = idx // 26 - 1
        return result

    def __repr__(self) -> str:
        return (f"SheetTable(cols={self.col_letter}:{self.end_col_letter}, "
                f"rows={self.num_rows}, columns={self.columns})")

def table_exact_match(df1: pd.DataFrame, df2: pd.DataFrame, ignore_case: bool = False) -> bool:
    if df1.shape != df2.shape:
        return False

    if ignore_case:
        df1 = df1.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)
        df2 = df2.applymap(lambda x: str(x).lower() if isinstance(x, str) else x)

    return df1.equals(df2)

def table_column_check(df: pd.DataFrame, required_columns: list) -> bool:
    return all(col in df.columns for col in required_columns)

def table_number_tolerance(df: pd.DataFrame, gold_dict: dict, nutrient_cols: list, rel_tol: float = 0.05) -> bool:
    for _, row in df.iterrows():
        ingredient = str(row["Ingredient"]).lower().strip()

        if ingredient not in gold_dict:
            print(f"Ingredient '{ingredient}' not in gold data.")
            return False

        for nutrient in nutrient_cols:
            if nutrient not in row or nutrient not in gold_dict[ingredient]:
                print(f"Missing {nutrient} for {ingredient}")
                return False

            try:
                actual = float(row[nutrient])
                expected = float(gold_dict[ingredient][nutrient])
                if not np.isclose(actual, expected, rtol=rel_tol):
                    print(f"{ingredient} → {nutrient} mismatch: got {actual}, expected {expected}")
                    return False
            except Exception as e:
                print(f"Error comparing {ingredient} → {nutrient}: {e}")
                return False

    return True

def table_partial_match(df: pd.DataFrame, gold_df: pd.DataFrame, key_column: str = "Ingredient") -> bool:
    df_keys = set(df[key_column].str.lower().str.strip())
    gold_keys = set(gold_df[key_column].str.lower().str.strip())

    missing = gold_keys - df_keys
    if missing:
        print(f"Missing values: {missing}")
        return False
    return True


def find_matching_column_or_row(
    df: pd.DataFrame,
    criteria: str,
    model: Any,
    search_type: str = "column",
    example_keywords: Optional[list] = None,
) -> Optional[Tuple[str, Union[int, str]]]:
    """
    Find a column or row in a DataFrame that best matches natural language criteria using an LLM.

    Args:
        df (pd.DataFrame): The DataFrame to search.
        criteria (str): Natural language description of what to look for.
        model (Any): LLM model that takes (prompt) and returns a response.
        search_type (str): Either "column" or "row" to specify search direction.
        example_keywords (Optional[list]): Optional list of example keywords/terms that might match
            the criteria. Helps the LLM better understand what to look for.

    Returns:
        Optional[Tuple[str, Union[int, str]]]: Tuple of (title, location) where location is
        column name for columns or row index for rows, or None if no match found.

    Raises:
        ValueError: If search_type is not "column" or "row".
    """
    if search_type not in ["column", "row"]:
        raise ValueError("search_type must be either 'column' or 'row'")

    if search_type == "column":
        headers = df.columns.tolist()
    else:
        # For rows, use the first column as row identifiers if it contains strings,
        # otherwise use row indices
        if len(df) > 0 and df.iloc[:, 0].dtype == 'object':
            headers = df.iloc[:, 0].tolist()
        else:
            headers = [f"Row {i}" for i in range(len(df))]

    if not headers:
        return None

    # Create prompt for LLM
    headers_text = "\n".join([f"{i+1}. {header}" for i, header in enumerate(headers)])

    # Add example keywords to the prompt if provided
    keywords_section = ""
    if example_keywords:
        keywords_section = f"\nExample keywords that might match the criteria: {', '.join(example_keywords)}"

    prompt = f"""You are analyzing a Google Sheets {search_type} headers to find the one that best matches specific criteria.

Criteria: {criteria}{keywords_section}

Available {search_type} headers:
{headers_text}

Please analyze each header and determine which one best matches the criteria. Consider:
- Exact matches
- Synonyms and semantically similar terms
- Common abbreviations
- Spreadsheet naming conventions

Respond with ONLY the number (1, 2, 3, etc.) of the best matching header, or "NONE" if no header adequately matches the criteria.

Your response should be just the number or "NONE", nothing else."""

    try:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        response = model(messages)
        response = response.strip().upper()

        if response == "NONE":
            return None

        # Try to parse the response as a number
        try:
            header_index = int(response) - 1
            if 0 <= header_index < len(headers):
                header_title = headers[header_index]
                if search_type == "column":
                    location = header_title  # Column name
                else:
                    if df.iloc[:, 0].dtype == 'object':
                        location = header_title  # Row identifier from first column
                    else:
                        location = header_index  # Row index
                return (header_title, location)
        except ValueError:
            pass

        return None

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


# =============================================================================
# Google Sheets Text Visibility Utilities
# =============================================================================

def is_text_visible_in_cell(
    content: str,
    col_width: int,
    wrap_strategy: str,
    row_values: List[Dict],
    col_idx: int,
    char_width: int = 7
) -> bool:
    """
    Check if text is fully visible in a Google Sheets cell.

    Text is visible if:
    - Column width is sufficient for the text, OR
    - Wrap strategy is 'WRAP' (text wraps to multiple lines), OR
    - Wrap strategy is 'OVERFLOW_CELL' and adjacent cells are empty (text overflows)

    Text is NOT visible (truncated/hidden) if:
    - Text width exceeds column width AND wrap strategy is 'CLIP', OR
    - Text width exceeds column width AND wrap strategy is 'OVERFLOW_CELL' but next cell has content

    Args:
        content: The text content of the cell.
        col_width: Width of the column in pixels.
        wrap_strategy: One of 'WRAP', 'OVERFLOW_CELL', or 'CLIP'.
        row_values: List of all cell values in the row (to check adjacent cells).
        col_idx: Column index of this cell.
        char_width: Approximate width per character in pixels (default 7).

    Returns:
        True if text is fully visible, False if truncated/hidden.
    """
    if not content:
        return True

    expected_width = len(content) * char_width

    # If text fits in column, it's visible
    if expected_width <= col_width:
        return True

    # If wrapping is enabled, text is visible (wraps to multiple lines)
    if wrap_strategy == 'WRAP':
        return True

    # If CLIP, text is hidden when it exceeds width
    if wrap_strategy == 'CLIP':
        return False

    # For OVERFLOW_CELL (default), check if next cell blocks the overflow
    if wrap_strategy == 'OVERFLOW_CELL':
        # Check subsequent cells to see if overflow is blocked
        overflow_needed = expected_width - col_width
        current_col = col_idx + 1

        while overflow_needed > 0 and current_col < len(row_values):
            next_cell = row_values[current_col] if current_col < len(row_values) else {}
            next_content = next_cell.get('formattedValue', '') if isinstance(next_cell, dict) else ''

            if next_content:
                # Next cell has content, overflow is blocked - text is hidden
                return False

            # Assume default column width for overflow calculation
            overflow_needed -= 100  # Default column width
            current_col += 1

        # Overflow has room, text is visible
        return True

    # Unknown wrap strategy, assume visible
    return True


# =============================================================================
# Google Sheets Image Extraction Utilities
# =============================================================================

def extract_image_url_from_cell(cell_value: str) -> Optional[str]:
    """Extract image URL from a cell value string.

    Handles various formats:
    - Direct URL (http://... or https://...)
    - Google Sheets IMAGE() formula: =IMAGE("url")
    - Just the URL embedded in text

    Args:
        cell_value: The cell value string

    Returns:
        str: The extracted URL, or None if not found
    """
    import re

    if not cell_value:
        return None

    cell_value = str(cell_value).strip()

    # Direct URL
    if cell_value.startswith('http://') or cell_value.startswith('https://'):
        return cell_value

    # IMAGE() formula: =IMAGE("url") or =IMAGE('url')
    image_match = re.search(r'IMAGE\s*\(\s*["\']([^"\']+)["\']', cell_value, re.IGNORECASE)
    if image_match:
        return image_match.group(1)

    # Just extract any URL from the value
    url_match = re.search(r'(https?://[^\s"\'<>]+)', cell_value)
    if url_match:
        return url_match.group(1)

    return None


def get_image_url_from_raw_sheet_cell(
    sheet_raw: Dict[str, Any],
    row_idx: int,
    col_idx: int
) -> Optional[str]:
    """Extract image URL from raw Google Sheets API response at a specific cell position.

    This accesses the raw Google Sheets API response to get:
    1. userEnteredValue.formulaValue - for =IMAGE("url") formulas
    2. userEnteredValue.stringValue - for direct URLs
    3. formattedValue - as fallback

    Args:
        sheet_raw: The raw Google Sheets API response from get_sheet_content()
        row_idx: 0-based row index (including header)
        col_idx: 0-based column index

    Returns:
        str: The extracted image URL, or None if not found
    """
    if not sheet_raw:
        return None

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return None

        # Get the first sheet's data
        sheet_data = sheets[0].get('data', [{}])[0]
        rows = sheet_data.get('rowData', [])

        if row_idx >= len(rows):
            return None

        row = rows[row_idx]
        values = row.get('values', [])

        if col_idx >= len(values):
            return None

        cell = values[col_idx]

        # Try to get the formula (for =IMAGE("url"))
        user_entered = cell.get('userEnteredValue', {})

        # Check formulaValue first (contains =IMAGE("url"))
        formula_value = user_entered.get('formulaValue', '')
        if formula_value:
            url = extract_image_url_from_cell(formula_value)
            if url:
                return url

        # Check stringValue (might be a direct URL)
        string_value = user_entered.get('stringValue', '')
        if string_value:
            url = extract_image_url_from_cell(string_value)
            if url:
                return url

        # Fallback to formattedValue
        formatted_value = cell.get('formattedValue', '')
        if formatted_value:
            url = extract_image_url_from_cell(formatted_value)
            if url:
                return url

        return None

    except Exception as e:
        print(f"Error extracting image URL from raw cell ({row_idx}, {col_idx}): {e}")
        return None


def get_column_index_by_name(
    df: pd.DataFrame,
    col_name: str,
    matched_columns: Dict[str, str]
) -> int:
    """Get the 0-based column index from a logical column name.

    Args:
        df: The pandas DataFrame
        col_name: The logical column name (e.g., "Figure 1")
        matched_columns: Mapping of logical names to actual column names

    Returns:
        int: 0-based column index, or -1 if not found
    """
    if not matched_columns or df is None:
        return -1

    actual_col_name = matched_columns.get(col_name)
    if not actual_col_name:
        return -1

    try:
        col_list = list(df.columns)
        return col_list.index(actual_col_name)
    except ValueError:
        return -1


def get_sheet_row_index_from_dataframe_row(df_row, header_rows: int = 1) -> int:
    """Get the 0-based row index in raw sheet data for a DataFrame row.

    The DataFrame row index corresponds to the data row position.
    Adding header_rows accounts for header row(s) in the raw sheet data.

    Args:
        df_row: A pandas Series representing a matched row (with .name attribute)
        header_rows: Number of header rows in the sheet (default 1)

    Returns:
        int: 0-based row index in raw sheet data, or -1 if invalid
    """
    try:
        return int(df_row.name) + header_rows
    except:
        return -1


# =============================================================================
# Google Sheets Row Color/Formatting Utilities
# =============================================================================

def get_row_background_color(sheet_raw: Dict, row_idx: int) -> Optional[Dict]:
    """Extract background color from a specific row in raw sheet data.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API.
        row_idx: 0-indexed row number.

    Returns:
        Color dict with 'red', 'green', 'blue' keys (0-1 values), or None.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return None

        rows = sheets[0].get('data', [{}])[0].get('rowData', [])
        if row_idx >= len(rows):
            return None

        row = rows[row_idx]
        cells = row.get('values', [])

        if not cells:
            return None

        # Get color from first cell in the row
        cell = cells[0]
        effective_format = cell.get('effectiveFormat', {})
        bg_color = effective_format.get('backgroundColor', {})

        return bg_color if bg_color else None

    except Exception:
        return None


def classify_row_color(color_dict: Optional[Dict]) -> str:
    """Classify a row color as yellow, blue, or none.

    Args:
        color_dict: Color dictionary with 'red', 'green', 'blue' keys.

    Returns:
        'yellow', 'blue', or 'none'.
    """
    if not color_dict:
        return 'none'

    red = color_dict.get('red', 1)
    green = color_dict.get('green', 1)
    blue = color_dict.get('blue', 1)

    # Yellow: high red, high green, low blue
    if red > 0.8 and green > 0.8 and blue < 0.5:
        return 'yellow'

    # Light yellow (Google Sheets default yellow)
    if red > 0.9 and green > 0.9 and blue > 0.6 and blue < 0.9:
        return 'yellow'

    # Blue: low red, low green, high blue
    if red < 0.5 and green < 0.7 and blue > 0.7:
        return 'blue'

    # Light blue
    if red > 0.6 and red < 0.9 and green > 0.8 and blue > 0.9:
        return 'blue'

    # White or near-white
    if red > 0.95 and green > 0.95 and blue > 0.95:
        return 'none'

    return 'none'


def validate_color_grouping(row_colors: List[str]) -> Tuple[bool, str]:
    """Check if same colors are grouped together (not interleaved).

    Args:
        row_colors: List of color classifications for each row.

    Returns:
        Tuple of (is_valid, message).
    """
    if not row_colors:
        return True, "No rows to check"

    # Track which colors we've seen and finished with
    seen_colors = set()
    finished_colors = set()
    current_color = None

    for i, color in enumerate(row_colors):
        if color == 'none':
            continue

        if current_color is None:
            current_color = color
            seen_colors.add(color)
        elif color != current_color:
            # Color changed
            finished_colors.add(current_color)

            if color in finished_colors:
                # We're seeing a color we already finished - interleaving!
                return False, f"Color '{color}' appears in non-contiguous rows (interleaved at row {i+1})"

            current_color = color
            seen_colors.add(color)

    return True, f"Colors are properly grouped: {seen_colors}"


def get_cell_value(sheet_raw: Dict, row_idx: int, col_idx: int) -> str:
    """Get formatted cell value from raw sheet data.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API (get_sheet_content).
        row_idx: 0-indexed row number.
        col_idx: 0-indexed column number.

    Returns:
        Cell value as string, or empty string if not found.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return ""

        sheet_data = sheets[0].get('data', [{}])[0]
        rows = sheet_data.get('rowData', [])

        if row_idx < len(rows):
            values = rows[row_idx].get('values', [])
            if col_idx < len(values):
                return values[col_idx].get('formattedValue', '')
        return ""
    except Exception:
        return ""


def get_cell_background_color(sheet_raw: Dict, row_idx: int, col_idx: int) -> Dict:
    """Get background color of a specific cell.

    Unlike get_row_background_color which gets the first cell's color,
    this function gets the color of a specific cell by column index.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API.
        row_idx: 0-indexed row number.
        col_idx: 0-indexed column number.

    Returns:
        Dict with 'red', 'green', 'blue' keys (0-1 scale), or empty dict.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return {}

        sheet_data = sheets[0].get('data', [{}])[0]
        rows = sheet_data.get('rowData', [])

        if row_idx < len(rows):
            values = rows[row_idx].get('values', [])
            if col_idx < len(values):
                effective_format = values[col_idx].get('effectiveFormat', {})
                return effective_format.get('backgroundColor', {})
        return {}
    except Exception:
        return {}


def check_merged_cells(sheet_raw: Dict, expected_cols: List[int], row_start: int, row_end: int) -> bool:
    """Check if specified columns are merged vertically across rows.

    Useful for validating that certain columns (like shared forecast data)
    are properly merged across multiple data rows.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API.
        expected_cols: List of column indices to check for merges.
        row_start: Start row index (inclusive, 0-indexed).
        row_end: End row index (inclusive, 0-indexed).

    Returns:
        True if all expected columns are merged across the specified rows.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return False

        merges = sheets[0].get('merges', [])

        for col_idx in expected_cols:
            found_merge = False
            for merge in merges:
                # Check if this merge covers the column and row range
                if (merge.get('startColumnIndex') == col_idx and
                    merge.get('endColumnIndex') == col_idx + 1 and
                    merge.get('startRowIndex') <= row_start and
                    merge.get('endRowIndex') >= row_end + 1):  # endRowIndex is exclusive
                    found_merge = True
                    break

            if not found_merge:
                return False

        return True
    except Exception:
        return False


# =============================================================================
# Color Comparison Utilities
# =============================================================================

def colors_are_similar(c1: Dict, c2: Dict, tolerance: float = 0.05) -> bool:
    """Check if two RGB colors are similar within tolerance.

    Compares two color dictionaries with 'red', 'green', 'blue' keys
    on a 0-1 scale (as returned by Google Sheets API).

    Args:
        c1: First color dict with 'red', 'green', 'blue' keys (0-1 scale).
        c2: Second color dict with 'red', 'green', 'blue' keys (0-1 scale).
        tolerance: Maximum allowed difference per channel (default 0.05).

    Returns:
        True if colors are similar within tolerance.
    """
    if not c1 or not c2:
        return False

    for channel in ['red', 'green', 'blue']:
        v1 = c1.get(channel, 1.0)
        v2 = c2.get(channel, 1.0)
        if abs(v1 - v2) > tolerance:
            return False

    return True


def colors_are_distinct(colors: List[Dict], tolerance: float = 0.1) -> bool:
    """Check if a list of colors are all distinct from each other.

    Uses colors_are_similar() to compare each pair of colors.

    Args:
        colors: List of color dicts with 'red', 'green', 'blue' keys (0-1 scale).
        tolerance: Minimum required difference to be considered distinct.

    Returns:
        True if all colors are distinct from each other.
    """
    if len(colors) < 2:
        return True

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if colors_are_similar(colors[i], colors[j], tolerance):
                return False

    return True


# =============================================================================
# Keyword-Based Search Utilities
# =============================================================================

def matches_keywords(
    value: str,
    keywords: List[str],
    case_sensitive: bool = False,
    strict: bool = True
) -> bool:
    """Check if a value matches any of the given keywords.

    This is the core matching function used by find_column_by_keywords,
    find_row_by_keywords, and can be used directly for ingredient matching.

    Args:
        value: The string value to check.
        keywords: List of keyword strings to match against.
        case_sensitive: Whether to perform case-sensitive matching.
        strict: If True, requires exact match (value equals keyword after stripping).
                If False, uses substring matching (keyword in value).

    Returns:
        True if the value matches any keyword, False otherwise.
    """
    if not value or not keywords:
        return False

    value_check = value.strip() if case_sensitive else value.lower().strip()

    for keyword in keywords:
        keyword_check = keyword.strip() if case_sensitive else keyword.lower().strip()
        if strict:
            if value_check == keyword_check:
                return True
        else:
            if keyword_check in value_check:
                return True

    return False


def find_match_in_list(
    values: List[str],
    keywords: List[str],
    case_sensitive: bool = False,
    strict: bool = False
) -> Optional[str]:
    """Find the first value in a list that matches any of the given keywords.

    Args:
        values: List of string values to search through.
        keywords: List of keyword strings to match against.
        case_sensitive: Whether to perform case-sensitive matching.
        strict: If True, requires exact match. If False, uses substring matching.

    Returns:
        The first matching value, or None if not found.
    """
    for value in values:
        if matches_keywords(value, keywords, case_sensitive, strict):
            return value
    return None


def find_column_by_keywords(
    columns: List[str],
    keywords: List[str],
    case_sensitive: bool = False,
    strict: bool = False
) -> Optional[str]:
    """Find a column that matches any of the given keywords.

    This is a fast keyword-based search that can be used as a pre-check
    before falling back to the more expensive LLM-based find_matching_column_or_row().

    Args:
        columns: List of column names.
        keywords: List of keyword strings to match.
        case_sensitive: Whether to perform case-sensitive matching.
        strict: If True, requires exact match or column name equals keyword
                (after stripping whitespace). If False, uses substring matching.

    Returns:
        The matching column name, or None if not found.
    """
    return find_match_in_list(columns, keywords, case_sensitive, strict)


def find_row_by_keywords(
    df: pd.DataFrame,
    search_column: str,
    keywords: List[str],
    case_sensitive: bool = False,
    strict: bool = False
) -> Optional[int]:
    """Find the row index using keyword matching in a specific column.

    This is a fast keyword-based search that can be used as a pre-check
    before falling back to the more expensive LLM-based find_matching_column_or_row().

    Args:
        df: DataFrame containing the data.
        search_column: Name of the column to search in.
        keywords: List of keywords to match.
        case_sensitive: Whether to perform case-sensitive matching.
        strict: If True, requires exact match (cell value equals keyword).
                If False, uses substring matching.

    Returns:
        Row index if found, None otherwise.
    """
    if search_column not in df.columns:
        return None

    for idx, row in df.iterrows():
        cell_value = str(row[search_column])
        if matches_keywords(cell_value, keywords, case_sensitive, strict):
            return idx

    return None


# =============================================================================
# Google Sheets Merged Cell Utilities
# =============================================================================

def find_merged_cell_by_text(
    merges: List[Dict],
    rows: List[Dict],
    text_pattern: str,
    case_sensitive: bool = False
) -> tuple:
    """Find a merged cell that contains the given text pattern.

    Searches through merged cell regions and returns the merge info and cell data
    for the first merge whose cell value contains the text pattern.

    Args:
        merges: List of merge dictionaries from Google Sheets API (sheet.get('merges', [])).
        rows: List of row data from Google Sheets API (gridData.get('rowData', [])).
        text_pattern: Text pattern to search for in merged cells.
        case_sensitive: Whether to perform case-sensitive matching.

    Returns:
        Tuple of (merge_info, cell_data) where:
        - merge_info: The merge dictionary with startRowIndex, endRowIndex, etc.
        - cell_data: The cell dictionary with formattedValue, effectiveFormat, etc.
        Returns (None, None) if no matching merge is found.
    """
    for merge in merges:
        start_row = merge.get('startRowIndex', 0)
        start_col = merge.get('startColumnIndex', 0)

        if start_row < len(rows):
            row = rows[start_row].get('values', [])
            if start_col < len(row):
                cell_value = row[start_col].get('formattedValue', '')
                pattern = text_pattern if case_sensitive else text_pattern.lower()
                value = cell_value if case_sensitive else cell_value.lower()
                if pattern in value:
                    return merge, row[start_col]

    return None, None


def get_merge_column_span(merge: Dict) -> int:
    """Get the number of columns spanned by a merged cell.

    Args:
        merge: Merge dictionary from Google Sheets API.

    Returns:
        Number of columns the merge spans.
    """
    if not merge:
        return 0
    return merge.get('endColumnIndex', 0) - merge.get('startColumnIndex', 0)


# =============================================================================
# Google Sheets Cell Formatting Utilities
# =============================================================================

def is_cell_centered(cell: Dict) -> bool:
    """Check if a cell has centered horizontal alignment.

    Args:
        cell: Cell dictionary from Google Sheets API.

    Returns:
        True if cell is horizontally centered.
    """
    if not cell:
        return False
    h_align = cell.get('effectiveFormat', {}).get('horizontalAlignment', '')
    return h_align == 'CENTER'


def is_cell_italic(cell: Dict) -> bool:
    """Check if a cell has italic text formatting.

    Args:
        cell: Cell dictionary from Google Sheets API.

    Returns:
        True if cell text is italic.
    """
    if not cell:
        return False
    fmt = cell.get('effectiveFormat', {}).get('textFormat', {})
    return fmt.get('italic', False)


def is_cell_bold(cell: Dict) -> bool:
    """Check if a cell has bold text formatting.

    Args:
        cell: Cell dictionary from Google Sheets API.

    Returns:
        True if cell text is bold.
    """
    if not cell:
        return False
    fmt = cell.get('effectiveFormat', {}).get('textFormat', {})
    return fmt.get('bold', False)


def has_bottom_border(cell: Dict) -> bool:
    """Check if a cell has a bottom border.

    Args:
        cell: Cell dictionary from Google Sheets API.

    Returns:
        True if cell has a visible bottom border.
    """
    if not cell:
        return False
    borders = cell.get('effectiveFormat', {}).get('borders', {})
    bottom = borders.get('bottom', {})
    style = bottom.get('style', '')
    return style and style != 'NONE'


def has_top_border(cell: Dict) -> bool:
    """Check if a cell has a top border.

    Args:
        cell: Cell dictionary from Google Sheets API.

    Returns:
        True if cell has a visible top border.
    """
    if not cell:
        return False
    borders = cell.get('effectiveFormat', {}).get('borders', {})
    top = borders.get('top', {})
    style = top.get('style', '')
    return style and style != 'NONE'


def row_has_top_border(row: Dict) -> bool:
    """Check if any cell in a row has a top border.

    Args:
        row: Row dictionary from Google Sheets API (rowData entry).

    Returns:
        True if any cell in the row has a visible top border.
    """
    if not row:
        return False
    values = row.get('values', [])
    return any(has_top_border(cell) for cell in values)


def row_has_bottom_border(row: Dict) -> bool:
    """Check if any cell in a row has a bottom border.

    Args:
        row: Row dictionary from Google Sheets API (rowData entry).

    Returns:
        True if any cell in the row has a visible bottom border.
    """
    if not row:
        return False
    values = row.get('values', [])
    return any(has_bottom_border(cell) for cell in values)


def count_bold_cells_in_row(row: Dict) -> tuple:
    """Count bold and total non-empty cells in a row.

    Args:
        row: Row dictionary from Google Sheets API (rowData entry).

    Returns:
        Tuple of (bold_count, total_count) for non-empty cells.
    """
    if not row:
        return 0, 0

    values = row.get('values', [])
    bold_count = 0
    total_count = 0

    for cell in values:
        value = cell.get('formattedValue', '')
        if value:
            total_count += 1
            if is_cell_bold(cell):
                bold_count += 1

    return bold_count, total_count