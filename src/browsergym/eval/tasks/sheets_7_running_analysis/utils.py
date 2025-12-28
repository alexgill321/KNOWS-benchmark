"""
Utility functions for the sheets_7_running_analysis task.

Contains unit conversion functions for distance and speed, date normalization,
chart identification, and other evaluation helpers.
"""

from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import re
import requests


# =============================================================================
# Unit Conversion Functions
# =============================================================================

def meters_to_miles(meters: float) -> float:
    """Convert meters to miles."""
    return meters / 1609.344


def meters_to_km(meters: float) -> float:
    """Convert meters to kilometers."""
    return meters / 1000.0


def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km / 1.60934


def ms_to_min_per_mile(speed_ms: float) -> float:
    """Convert speed from m/s to min/mile pace."""
    if speed_ms <= 0:
        return float('inf')
    return 26.8224 / speed_ms


def ms_to_kmh(speed_ms: float) -> float:
    """Convert speed from m/s to km/h."""
    return speed_ms * 3.6


# =============================================================================
# Date Normalization
# =============================================================================

def normalize_date(date_str) -> str:
    """
    Normalize date string for comparison (full timestamp).

    Handles formats like:
    - "Sep 27, 2021, 2:13:42 AM" (gold format)
    - "2021-09-27 02:13:42"
    - Various other common formats

    Args:
        date_str: Date string to normalize.

    Returns:
        Normalized date string in format "YYYY-MM-DD HH:MM:SS".
    """
    if pd.isna(date_str):
        return ""

    date_str = str(date_str).strip()

    # Try parsing the gold format: "Sep 27, 2021, 2:13:42 AM"
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    # Try other common formats with time
    formats_to_try = [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%b %d, %Y, %H:%M:%S",
        "%Y-%m-%d %I:%M:%S %p",
    ]

    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    # Fallback to lowercase string
    return date_str.lower()


# =============================================================================
# Data Loading
# =============================================================================

def load_gold_run_activities(csv_path: str):
    """Load gold data, filtering only Run activities.

    The CSV has duplicate 'Distance' columns - pandas renames them to 'Distance' and 'Distance.1'.
    - 'Distance' is in km (rounded)
    - 'Distance.1' is in meters (more precise)

    We normalize to use the meters column converted to km for consistency.

    Args:
        csv_path: Path to the gold_activities.csv file.

    Returns:
        DataFrame containing only Run activities with normalized Distance column.
    """
    gold_df = pd.read_csv(csv_path)
    runs = gold_df[gold_df['Activity Type'] == 'Run'].copy()

    # Use Distance.1 (meters) converted to km for more precision
    # This fixes discrepancies like "Anas run" where Distance=0.44 km but Distance.1=448.1 m
    if 'Distance.1' in runs.columns:
        runs['Distance'] = runs['Distance.1'] / 1000.0

    return runs


# =============================================================================
# Table Visibility Helpers
# =============================================================================

def check_all_content_visible(sheet_raw_data, start_row: int, end_row: int, num_cols: int, is_text_visible_fn) -> Tuple[bool, str]:
    """
    Check if all table content is fully visible (no truncation/clipping).

    Args:
        sheet_raw_data: Raw sheet data from get_sheet_content()
        start_row: Starting row index of the table (header row)
        end_row: Ending row index (exclusive)
        num_cols: Number of columns to check
        is_text_visible_fn: Function to check text visibility (is_text_visible_in_cell)

    Returns:
        tuple: (all_visible: bool, details: str)
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

                if not is_text_visible_fn(content, col_width, wrap_strategy, row_values, col_idx):
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
# Chart Analysis Helpers
# =============================================================================

def get_chart_axis_labels(chart: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract X and Y axis labels from chart spec.

    Args:
        chart: Chart object from extract_charts_from_sheet()

    Returns:
        dict: {'x_axis': str, 'y_axis': str} with axis titles
    """
    result = {'x_axis': '', 'y_axis': ''}

    raw_chart = chart.get('raw_chart', {})
    spec = raw_chart.get('spec', {})
    basic_chart = spec.get('basicChart', {})
    axes = basic_chart.get('axis', [])

    for axis in axes:
        position = axis.get('position', '').upper()
        title = axis.get('title', '')

        if position == 'BOTTOM_AXIS':
            result['x_axis'] = title
        elif position in ['LEFT_AXIS', 'RIGHT_AXIS']:
            result['y_axis'] = title

    return result


def check_chart_overlap(chart: Dict[str, Any], table_start_row: int, table_end_row: int, other_charts: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Check if chart overlaps with table or other charts.

    Args:
        chart: Chart object to check
        table_start_row: Starting row of the data table
        table_end_row: Ending row of the data table
        other_charts: List of other chart objects

    Returns:
        tuple: (has_overlap: bool, overlap_details: str)
    """
    position = chart.get('position', {})
    if position.get('type') != 'overlay':
        return False, "Chart not in overlay position"

    anchor_row = position.get('anchor_cell', {}).get('row', 0)
    anchor_col = position.get('anchor_cell', {}).get('col', 0)
    height = position.get('height', 0)
    width = position.get('width', 0)

    # Estimate chart end row (assuming ~20 pixels per row)
    chart_end_row = anchor_row + (height // 20) if height else anchor_row + 15

    # Check overlap with table
    if anchor_row < table_end_row and chart_end_row > table_start_row:
        return True, f"Chart overlaps with data table (chart rows {anchor_row}-{chart_end_row}, table rows {table_start_row}-{table_end_row})"

    # Check overlap with other charts
    chart_id = chart.get('chart_id')
    for other in other_charts:
        if other.get('chart_id') == chart_id:
            continue

        other_pos = other.get('position', {})
        if other_pos.get('type') != 'overlay':
            continue

        other_anchor_row = other_pos.get('anchor_cell', {}).get('row', 0)
        other_anchor_col = other_pos.get('anchor_cell', {}).get('col', 0)
        other_height = other_pos.get('height', 0)
        other_width = other_pos.get('width', 0)
        other_end_row = other_anchor_row + (other_height // 20) if other_height else other_anchor_row + 15
        other_end_col = other_anchor_col + (other_width // 100) if other_width else other_anchor_col + 6

        chart_end_col = anchor_col + (width // 100) if width else anchor_col + 6

        # Check row overlap
        row_overlap = anchor_row < other_end_row and chart_end_row > other_anchor_row
        # Check column overlap
        col_overlap = anchor_col < other_end_col and chart_end_col > other_anchor_col

        if row_overlap and col_overlap:
            return True, f"Chart overlaps with another chart (ID: {other.get('chart_id')})"

    return False, "No overlap detected"


def extract_baseline_series(chart: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract baseline line series from chart (series beyond main data).

    Baselines are typically constant-value horizontal lines.

    Args:
        chart: Chart object from extract_charts_from_sheet()

    Returns:
        list: List of baseline series info dicts
    """
    raw_chart = chart.get('raw_chart', {})
    spec = raw_chart.get('spec', {})
    basic_chart = spec.get('basicChart', {})
    all_series = basic_chart.get('series', [])

    baselines = []

    # Skip the first series (main data) and look for baseline series
    for i, series in enumerate(all_series):
        series_info = {
            'index': i,
            'type': series.get('type', 'UNKNOWN'),
            'line_style': None,
            'color': None,
            'target_axis': series.get('targetAxis', 'LEFT_AXIS'),
        }

        # Check for line style (dotted/dashed)
        line_style = series.get('lineStyle', {})
        if line_style:
            series_info['line_style'] = line_style.get('type', 'SOLID')

        # Check color
        color = series.get('color', {})
        if color:
            series_info['color'] = color

        # Get data source range
        series_data = series.get('series', {})
        source_range = series_data.get('sourceRange', {})
        if source_range:
            sources = source_range.get('sources', [])
            if sources:
                src = sources[0]
                series_info['source_range'] = {
                    'start_row': src.get('startRowIndex'),
                    'end_row': src.get('endRowIndex'),
                    'start_col': src.get('startColumnIndex'),
                    'end_col': src.get('endColumnIndex'),
                }

        baselines.append(series_info)

    return baselines


def find_urls_in_sheet(sheet_rows: List[Dict], start_row: int, num_rows: int = 20) -> List[str]:
    """
    Find URLs in cells starting from a specific row.

    Args:
        sheet_rows: Raw rowData from sheet
        start_row: Row index to start searching from
        num_rows: Number of rows to search

    Returns:
        list: List of URLs found
    """
    urls = []
    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

    for row_idx in range(start_row, min(start_row + num_rows, len(sheet_rows))):
        row = sheet_rows[row_idx] if row_idx < len(sheet_rows) else {}
        values = row.get('values', [])

        for cell in values:
            # Check formatted value
            content = cell.get('formattedValue', '')
            if content:
                found_urls = url_pattern.findall(content)
                urls.extend(found_urls)

            # Check hyperlink
            hyperlink = cell.get('hyperlink', '')
            if hyperlink and hyperlink.startswith('http'):
                urls.append(hyperlink)

    return list(set(urls))  # Remove duplicates


def validate_url_accessible(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Check if URL is accessible via HTTP request.

    Args:
        url: URL to validate
        timeout: Request timeout in seconds

    Returns:
        tuple: (is_accessible: bool, details: str)
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True,
                                  headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code < 400:
            return True, f"URL accessible (status {response.status_code})"
        else:
            return False, f"URL returned status {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "URL request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"URL request failed: {str(e)[:50]}"


def check_circular_points(chart: Dict[str, Any], chart_type: str) -> Tuple[bool, str]:
    """
    Check if chart displays data as circular points.

    Args:
        chart: Chart object
        chart_type: Type of chart (SCATTER, LINE, etc.)

    Returns:
        tuple: (has_circular_points: bool, details: str)
    """
    # SCATTER charts always show points
    if chart_type == 'SCATTER':
        return True, "Scatter chart displays points"

    # For LINE charts, check if points are visible and line is hidden
    raw_chart = chart.get('raw_chart', {})
    spec = raw_chart.get('spec', {})
    basic_chart = spec.get('basicChart', {})
    series_list = basic_chart.get('series', [])

    if not series_list:
        return False, "No series found in chart"

    # Check first series (main data)
    main_series = series_list[0]

    # Check point style
    point_style = main_series.get('pointStyle', {})
    point_size = point_style.get('size', 0)

    # Check line style
    line_style = main_series.get('lineStyle', {})
    line_width = line_style.get('width', 2)  # Default line width is usually 2

    # For points-only: need positive point size and zero/minimal line width
    if point_size > 0 and line_width == 0:
        return True, f"Line chart with points (size={point_size}) and no line"

    # Check if it's a combo chart type showing points
    series_type = main_series.get('type', '')
    if series_type == 'SCATTER':
        return True, "Series type is SCATTER"

    # If point style exists with non-zero size, likely has points
    if point_size > 0:
        return True, f"Chart has point markers (size={point_size})"

    return False, f"Could not confirm circular points (chart_type={chart_type}, point_size={point_size})"


# =============================================================================
# Chart Identification
# =============================================================================

def find_speed_chart_by_metadata(charts: List[Dict[str, Any]], matched_columns: Optional[Dict[str, str]], df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """
    Identify the speed chart using metadata (no VLM/image analysis).

    Matching order:
    1. Chart title contains speed/pace keywords
    2. Y-axis label contains speed/pace/min keywords
    3. Series data matches the speed column from checkpoint 2

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        matched_columns: Column mapping from checkpoint 2 (may contain "Speed (min/mile)")
        df: DataFrame with sheet data

    Returns:
        Chart object or None if not found
    """
    if not charts:
        return None

    speed_keywords = ['speed', 'pace', 'running speed', 'min/mile', 'min per mile']

    # Step 1: Title matching
    for chart in charts:
        title = chart.get('title', '').lower()
        if any(kw in title for kw in speed_keywords):
            return chart

    # Step 2: Axis label matching
    for chart in charts:
        axis_labels = get_chart_axis_labels(chart)
        y_label = axis_labels.get('y_axis', '').lower()
        if any(kw in y_label for kw in ['speed', 'pace', 'min/mile', 'minute']):
            return chart

    # Step 3: Series data matching (if matched_columns available from checkpoint 2)
    if matched_columns and df is not None:
        speed_col_name = matched_columns.get("Speed (min/mile)")
        if speed_col_name and speed_col_name in df.columns:
            speed_col_idx = df.columns.get_loc(speed_col_name)
            for chart in charts:
                for series in chart.get('series', []):
                    src = series.get('source_range', {})
                    if src.get('start_col') == speed_col_idx:
                        return chart

    return None