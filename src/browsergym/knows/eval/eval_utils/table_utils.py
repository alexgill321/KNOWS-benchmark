import pandas as pd
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

# =============================================================================
# Google Sheets Text Visibility Utilities
# =============================================================================

def is_text_visible_in_cell(
    content: str,
    col_width: int,
    wrap_strategy: str,
    row_values: List[Dict],
    col_idx: int,
    char_width: int = 6
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

    # Use a 1.3x tolerance to account for variable character widths
    # (the char_width estimate assumes wide characters, but most text
    # contains many narrow characters like i, l, /, -, etc.)
    expected_width = len(content) * char_width

    # If text fits in column (with tolerance), it's visible
    if expected_width <= col_width * 1.3:
        return True

    # If wrapping is enabled, text is visible (wraps to multiple lines)
    if wrap_strategy == 'WRAP':
        return True

    # If CLIP, text is hidden (already exceeded tolerance above)
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


def check_all_content_visible(
    sheet_raw_data: Dict[str, Any],
    start_row: int,
    end_row: int,
    num_cols: int
) -> Tuple[bool, str]:
    """
    Check if all table content is fully visible (no truncation/clipping).

    Iterates through cells in the specified range and checks visibility
    using is_text_visible_in_cell().

    Args:
        sheet_raw_data: Raw sheet data from Google Sheets API (get_sheet_content).
        start_row: Starting row index of the table (0-indexed, typically header row).
        end_row: Ending row index (exclusive).
        num_cols: Number of columns to check.

    Returns:
        tuple: (all_visible: bool, details: str)
            - all_visible: True if all content is visible
            - details: Description of result or list of truncated cells
    """
    if not sheet_raw_data:
        return False, "No sheet data available"

    try:
        # Get column metadata for widths
        sheets = sheet_raw_data.get('sheets', [])
        if not sheets:
            return False, "No sheets found in raw data"

        sheet_data = sheets[0]
        data_blocks = sheet_data.get('data', [])
        if not data_blocks:
            return False, "No data blocks found"

        col_metadata = data_blocks[0].get('columnMetadata', [])
        row_data = data_blocks[0].get('rowData', [])

        truncated_cells = []

        # Iterate through table cells and check visibility
        for row_idx in range(start_row, end_row):
            if row_idx >= len(row_data):
                continue
            row = row_data[row_idx]
            row_values = row.get('values', [])

            for col_idx in range(num_cols):
                if col_idx >= len(row_values):
                    continue
                cell = row_values[col_idx]
                content = cell.get('formattedValue', '')

                if not content:
                    continue

                # Get column width (default 100 pixels if not specified)
                col_width = 100
                if col_idx < len(col_metadata):
                    col_width = col_metadata[col_idx].get('pixelSize', 100)

                # Get wrap strategy (default OVERFLOW_CELL)
                wrap_strategy = cell.get('effectiveFormat', {}).get('wrapStrategy', 'OVERFLOW_CELL')

                if not is_text_visible_in_cell(content, col_width, wrap_strategy, row_values, col_idx):
                    # Track which cells are truncated
                    truncated_cells.append(f"Row {row_idx + 1}, Col {col_idx + 1}: '{content[:30]}...'")

        if truncated_cells:
            # Limit to first 5 examples
            examples = truncated_cells[:5]
            more = f" (+{len(truncated_cells) - 5} more)" if len(truncated_cells) > 5 else ""
            return False, f"Truncated cells: {'; '.join(examples)}{more}"

        return True, "All content fully visible"

    except Exception as e:
        return False, f"Error checking visibility: {str(e)}"


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

        # Check hyperlink property (sometimes images are hyperlinked)
        hyperlink = cell.get('hyperlink', '')
        if hyperlink and hyperlink.strip().startswith(('http://', 'https://')):
            return hyperlink.strip()

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

def resolve_sheets_theme_color(sheet_raw: Dict, theme_color_name: str) -> Dict:
    """Resolve a Sheets themeColor name (e.g. 'ACCENT1', 'TEXT') to an RGB dict via the
    spreadsheet's theme. Returns empty dict if the theme or color isn't found.

    Modern Google Sheets writes colors as `themeColor: "ACCENT_1"` rather than `rgbColor`,
    so checks against rgbColor alone miss user-picked theme colors. The actual RGB lives
    in `spreadsheetProperties.spreadsheetTheme.themeColors[]`.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API (with `properties.spreadsheetTheme`).
        theme_color_name: Theme color name like 'ACCENT1' or 'ACCENT_1' (underscore variants accepted).

    Returns:
        Dict with 'red', 'green', 'blue' keys (0-1 scale), or empty dict.
    """
    if not theme_color_name:
        return {}
    # Sheets API uses unsuffixed names ('ACCENT1') in themeColors[].colorType,
    # but cell-level themeColor may appear with or without underscore.
    normalized = theme_color_name.replace("_", "").upper()
    theme = (sheet_raw.get("properties", {})
             .get("spreadsheetTheme", {}))
    for entry in theme.get("themeColors", []):
        if entry.get("colorType", "").replace("_", "").upper() == normalized:
            return entry.get("color", {}).get("rgbColor", {}) or {}
    return {}


def get_text_foreground_color(cell: Dict, sheet_raw: Optional[Dict] = None) -> Dict:
    """Extract a cell's effective text foreground color as an RGB dict.

    Tries (in order): `effectiveFormat.textFormat.foregroundColorStyle.rgbColor`,
    then resolves `foregroundColorStyle.themeColor` via the sheet's theme,
    then falls back to legacy `effectiveFormat.textFormat.foregroundColor`.

    Args:
        cell: Cell dict from `rowData[i].values[j]`.
        sheet_raw: Raw sheet data (needed only for themeColor resolution).

    Returns:
        Dict with 'red', 'green', 'blue' keys (0-1 scale), or empty dict.
    """
    text_format = cell.get("effectiveFormat", {}).get("textFormat", {})
    fg_style = text_format.get("foregroundColorStyle", {})
    rgb = fg_style.get("rgbColor")
    if rgb:
        return rgb
    theme_name = fg_style.get("themeColor")
    if theme_name and sheet_raw:
        return resolve_sheets_theme_color(sheet_raw, theme_name)
    return text_format.get("foregroundColor", {}) or {}


def get_background_color(sheet_raw: Dict, row_idx: int, col_idx: int = 0) -> Dict:
    """Get background color of a cell in raw sheet data.

    This is the unified function for getting cell background colors.
    Use col_idx=0 (default) to get the first cell's color in a row.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API.
        row_idx: 0-indexed row number.
        col_idx: 0-indexed column number (default 0 for row's first cell).

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


# Backwards-compatible alias
def get_row_background_color(sheet_raw: Dict, row_idx: int) -> Optional[Dict]:
    """Get background color of first cell in a row. Alias for get_background_color(sheet_raw, row_idx, 0)."""
    result = get_background_color(sheet_raw, row_idx, 0)
    return result if result else None


def classify_row_color(color_dict: Optional[Dict]) -> str:
    """Classify a row color as yellow, orange, blue, green, red, or none.

    Args:
        color_dict: Color dictionary with 'red', 'green', 'blue' keys (0-1 scale).

    Returns:
        'yellow', 'orange', 'blue', 'green', 'red', or 'none'.
    """
    if not color_dict:
        return 'none'

    red = color_dict.get('red', 0)
    green = color_dict.get('green', 0)
    blue = color_dict.get('blue', 0)

    # White or near-white (check first to avoid false positives)
    if red > 0.95 and green > 0.95 and blue > 0.95:
        return 'none'

    # Yellow: high red, high green, low blue
    if red > 0.8 and green > 0.8 and blue < 0.5:
        return 'yellow'

    # Light yellow (Google Sheets default yellow)
    if red > 0.9 and green > 0.9 and blue > 0.6 and blue < 0.9:
        return 'yellow'

    # Orange: high red, moderate green, low blue (must come after yellow)
    if red > 0.7 and 0.3 < green < 0.85 and blue < 0.5:
        return 'orange'

    # Blue: low red, low green, high blue
    if red < 0.5 and green < 0.7 and blue > 0.7:
        return 'blue'

    # Light blue
    if red > 0.6 and red < 0.9 and green > 0.8 and blue > 0.9:
        return 'blue'

    # Green: green channel dominates
    if green > red and green > blue and green > 0.3:
        return 'green'

    # Red: red channel dominates
    if red > green and red > blue and red > 0.3:
        return 'red'

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


def get_cell(sheet_tab: Dict, row_idx: int, col_idx: int) -> Dict:
    """Return the raw cell dict at (row_idx, col_idx) inside a sheet tab.

    Args:
        sheet_tab: A single sheet tab dict (e.g. ``sheets[0]``).
        row_idx: 0-indexed row number.
        col_idx: 0-indexed column number.

    Returns:
        The cell dict, or an empty dict if the position is out of range.
    """
    data_blocks = sheet_tab.get("data", [{}])
    rows = data_blocks[0].get("rowData", []) if data_blocks else []
    if row_idx >= len(rows):
        return {}
    values = rows[row_idx].get("values", [])
    if col_idx >= len(values):
        return {}
    return values[col_idx]


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
        cell = get_cell(sheets[0], row_idx, col_idx)
        return cell.get('formattedValue', '') if cell else ""
    except Exception:
        return ""


def read_column_values(
    sheet_tab: Dict,
    col_idx: int,
    start_row: int = 0,
    end_row: Optional[int] = None,
) -> List[str]:
    """Read formatted cell values for a single column in a sheet tab.

    Args:
        sheet_tab: A single sheet tab dict (e.g. from ``get_sheet_by_name()`` or
            ``sheet_raw['sheets'][0]``).
        col_idx: 0-based column index.
        start_row: First row to read (inclusive, 0-based).
        end_row: Last row to read (exclusive). None reads to the end.

    Returns:
        List of string values (empty string for blank cells).
    """
    data_blocks = sheet_tab.get("data", [{}])
    rows = data_blocks[0].get("rowData", []) if data_blocks else []
    if end_row is None:
        end_row = len(rows)
    values: List[str] = []
    for r_idx in range(start_row, min(end_row, len(rows))):
        cell = get_cell(sheet_tab, r_idx, col_idx)
        values.append(cell.get("formattedValue", "") or "" if cell else "")
    return values


def cell_bg_hex(sheet_raw: Dict, row_idx: int, col_idx: int) -> Optional[str]:
    """Return the cell background colour as a ``#RRGGBB`` hex string.

    Args:
        sheet_raw: Raw sheet data from Google Sheets API.
        row_idx: 0-indexed row number.
        col_idx: 0-indexed column number.

    Returns:
        Hex colour string (e.g. ``'#3a7ca5'``) or None if the cell has no
        non-white fill.
    """
    bg = get_background_color(sheet_raw, row_idx, col_idx)
    if not bg:
        return None
    r = bg.get("red", 1.0)
    g = bg.get("green", 1.0)
    b = bg.get("blue", 1.0)
    if r > 0.98 and g > 0.98 and b > 0.98:
        return None
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


# Backwards-compatible alias
def get_cell_background_color(sheet_raw: Dict, row_idx: int, col_idx: int) -> Dict:
    """Get background color of a specific cell. Alias for get_background_color()."""
    return get_background_color(sheet_raw, row_idx, col_idx)


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
# Column Matching (uses text_utils)
# =============================================================================

def match_columns(
    df: pd.DataFrame,
    required_columns: List[Tuple[str, List[str]]],
    model: Optional[Any] = None,
    strict: bool = True,
    parallel: bool = False,
    max_workers: int = 5,
    context: str = None,
    return_methods: bool = False,
) -> Dict[str, str]:
    """Match required columns using keyword matching with optional LLM fallback.

    This is the standard column matching function for all sheets evaluators.
    It uses the unified text matching utilities from text_utils.

    Args:
        df: DataFrame to search columns in.
        required_columns: List of (col_name, keywords) tuples.
            Example: [("Stock Symbol", ["symbol", "ticker"]), ("Price", ["price", "cost"])]
        model: Optional LLM model for semantic fallback. If None, keyword-only matching.
        strict: If True, requires exact keyword match. If False, uses substring matching.
            Note: This parameter is kept for backwards compatibility but the new
            text_utils functions always use exact matching. For substring matching,
            use the fuzzy matching utilities in text_utils instead.
        parallel: If True, run LLM fallback calls in parallel using ThreadPoolExecutor.
            Note: Since the new keywords_llm_match makes a single LLM call per column,
            parallel=True now runs multiple column matches concurrently.
        max_workers: Maximum number of parallel LLM calls (only used if parallel=True).
        context: Optional task context passed to the LLM to improve matching accuracy
            (e.g., "an apartment listing spreadsheet with columns for property details").
        return_methods: If True, also return a dict mapping col_name -> "keyword"|"llm"
            recording which matching phase produced each hit.

    Returns:
        Dict mapping col_name -> matched_column_name for all matched columns.
        Columns that couldn't be matched are not included in the dict.
        If return_methods is True, returns (matched, methods) instead.
    """
    from .text_utils import keywords_exact_match, keywords_llm_match

    columns = [str(col) for col in df.columns]
    # Filter out malformed column names (tab/newline indicate merged cells or paste errors)
    valid_columns = [col for col in columns if '\t' not in col and '\n' not in col]
    if not valid_columns:
        valid_columns = columns  # Fallback to all if everything is malformed
    matched = {}
    methods = {}
    unmatched = []

    # Phase 1: Keyword matching (fast)
    for col_name, keywords in required_columns:
        # Try to find a column that matches any keyword
        for col in valid_columns:
            if keywords_exact_match(col, keywords):
                matched[col_name] = col
                methods[col_name] = "keyword"
                break
        else:
            unmatched.append((col_name, keywords))

    # Phase 2: LLM fallback for unmatched (if model provided)
    if unmatched and model is not None:
        if parallel:
            # Parallel LLM matching using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def call_llm_for_column(col_name: str, keywords: List[str]) -> Tuple[str, Optional[str]]:
                result = keywords_llm_match(valid_columns, keywords, model, description=f"column for '{col_name}'", context=context)
                return col_name, result

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(call_llm_for_column, col_name, keywords): col_name
                    for col_name, keywords in unmatched
                }
                for future in as_completed(futures):
                    try:
                        col_name, result = future.result()
                        if result:
                            matched[col_name] = result
                            methods[col_name] = "llm"
                    except Exception as e:
                        print(f"Error in parallel LLM column matching: {e}")
        else:
            # Sequential LLM matching
            for col_name, keywords in unmatched:
                result = keywords_llm_match(valid_columns, keywords, model, description=f"column for '{col_name}'", context=context)
                if result:
                    matched[col_name] = result
                    methods[col_name] = "llm"

    if return_methods:
        return matched, methods
    return matched


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


def has_border(cell: Dict, edge: str = "bottom") -> bool:
    """Check if a cell has a border on the specified edge.

    Args:
        cell: Cell dictionary from Google Sheets API.
        edge: Border edge to check - 'top', 'bottom', 'left', or 'right'.

    Returns:
        True if cell has a visible border on the specified edge.
    """
    if not cell:
        return False
    borders = cell.get('effectiveFormat', {}).get('borders', {})
    edge_data = borders.get(edge, {})
    style = edge_data.get('style', '')
    return style and style != 'NONE'


def row_has_border(row: Dict, edge: str = "bottom") -> bool:
    """Check if any cell in a row has a border on the specified edge.

    Args:
        row: Row dictionary from Google Sheets API (rowData entry).
        edge: Border edge to check - 'top', 'bottom', 'left', or 'right'.

    Returns:
        True if any cell in the row has a visible border on the specified edge.
    """
    if not row:
        return False
    values = row.get('values', [])
    return any(has_border(cell, edge) for cell in values)


# Backwards-compatible aliases
def has_bottom_border(cell: Dict) -> bool:
    """Check if a cell has a bottom border. Alias for has_border(cell, 'bottom')."""
    return has_border(cell, "bottom")


def has_top_border(cell: Dict) -> bool:
    """Check if a cell has a top border. Alias for has_border(cell, 'top')."""
    return has_border(cell, "top")


def row_has_bottom_border(row: Dict) -> bool:
    """Check if any cell in a row has a bottom border. Alias for row_has_border(row, 'bottom')."""
    return row_has_border(row, "bottom")


def row_has_top_border(row: Dict) -> bool:
    """Check if any cell in a row has a top border. Alias for row_has_border(row, 'top')."""
    return row_has_border(row, "top")


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