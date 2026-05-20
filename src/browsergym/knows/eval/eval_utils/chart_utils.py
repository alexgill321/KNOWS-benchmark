"""
Utility functions for validating charts in Google Sheets.

This module provides generalized functions for extracting and validating
chart data against expected values.
"""

import pandas as pd
import re
from typing import List, Tuple, Optional, Dict, Any, Union
from .text_utils import numerical_match_with_error, keywords_match_robust


def _parse_numeric_cell(value: Union[str, int, float, None]) -> Optional[float]:
    """Parse a cell value into a float, stripping common formatting.

    Handles currency symbols ($, €, £, ¥), percent signs, thousands commas,
    and accounting-style parentheses for negatives like "($5.00)".
    Returns None for null/empty/unparseable values.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    negative = s.startswith('(') and s.endswith(')')
    if negative:
        s = s[1:-1]
    s = re.sub(r'[^\d.\-+]', '', s)
    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return None


def debug_chart_structure(chart: Dict[str, Any]) -> None:
    """
    Print debug information about a chart's structure.

    Args:
        chart (dict): Chart object to debug
    """
    print("=== CHART DEBUG INFO ===")
    print(f"Chart keys: {list(chart.keys())}")
    print(f"Chart type: {chart.get('chart_type')}")
    print(f"Chart title: {chart.get('title')}")

    print("\nData range info:")
    data_range = chart.get('data_range', {})
    print(f"  data_range keys: {list(data_range.keys())}")
    if 'domain_range' in data_range:
        print(f"  domain_range: {data_range['domain_range']}")

    print("\nSeries info:")
    series = chart.get('series', [])
    print(f"  Number of series: {len(series)}")
    for i, s in enumerate(series):
        print(f"  Series {i} keys: {list(s.keys())}")
        if 'source_range' in s:
            print(f"  Series {i} source_range: {s['source_range']}")

    print("\nRaw chart info:")
    raw_chart = chart.get('raw_chart', {})
    if raw_chart:
        print(f"  raw_chart keys: {list(raw_chart.keys())}")
        if 'spec' in raw_chart:
            spec = raw_chart['spec']
            print(f"  spec keys: {list(spec.keys())}")
    print("======================\n")


def extract_chart_domain_data(chart: Dict[str, Any], table_data: pd.DataFrame, full_sheet_data: Optional[List[List[str]]] = None) -> List[str]:
    """
    Extract x-axis category labels from a chart's domain range.

    Args:
        chart (dict): Chart object from extract_charts_from_sheet containing:
            - 'data_range': Dict with 'domain_range' containing row/col indices
            - 'raw_chart': Full raw chart specification (fallback)
        table_data (pd.DataFrame): DataFrame containing the sheet data
        full_sheet_data (list, optional): Raw sheet values as list of rows (from Sheets API values().get()).
            Used as fallback when chart references columns outside the table DataFrame.

    Returns:
        list: List of category labels (strings) from the chart's x-axis.
              Returns empty list if domain range not found or data extraction fails.

    Example:
        chart = {'data_range': {'domain_range': {'start_row': 1, 'end_row': 11, 'start_col': 0}}}
        categories = extract_chart_domain_data(chart, df)
        # Returns: ['AAPL', 'MSFT', 'GOOGL', ...]
    """
    try:
        domain_range = chart.get('data_range', {}).get('domain_range')

        # If no domain_range in pre-parsed structure, try parsing from raw_chart
        if not domain_range.get('start_row') or not isinstance(domain_range, dict):
            print("Warning: No domain_range found in parsed chart data, attempting raw_chart fallback")
            raw_chart = chart.get('raw_chart', {})
            chart_spec = raw_chart.get('spec', {})

            # Try basicChart (for COLUMN, BAR, LINE, etc.)
            if 'basicChart' in chart_spec:
                basic_chart = chart_spec['basicChart']
                domains = basic_chart.get('domains', [])
                if domains:
                    domain_source = domains[0].get('domain', {}).get('sourceRange', {})
                    if domain_source:
                        domain_range = {
                            'start_row': domain_source.get('sources')[0].get('startRowIndex'),
                            'end_row': domain_source.get('sources')[0].get('endRowIndex'),
                            'start_col': domain_source.get('sources')[0].get('startColumnIndex'),
                            'end_col': domain_source.get('sources')[0].get('endColumnIndex')
                        }
                        print(f"Extracted domain_range from raw_chart: {domain_range}")

            # Try pieChart
            elif 'pieChart' in chart_spec:
                pie_chart = chart_spec['pieChart']
                if 'domain' in pie_chart:
                    domain_source = pie_chart['domain'].get('sourceRange', {}).get('sources', [{}])[0]
                    if domain_source:
                        domain_range = {
                            'start_row': domain_source.get('startRowIndex'),
                            'end_row': domain_source.get('endRowIndex'),
                            'start_col': domain_source.get('startColumnIndex'),
                            'end_col': domain_source.get('endColumnIndex')
                        }
                        print(f"Extracted domain_range from raw_chart pieChart: {domain_range}")

            if not domain_range:
                print("Warning: Could not extract domain_range from raw_chart either")
                return []

        # Get range values, handling None
        # Note: Chart ranges from Google Sheets API are 0-indexed and include the header row.
        # The DataFrame has the header stripped, so sheet row 1 = DataFrame row 0.
        # Account for headerCount (default 1) to skip header rows.
        raw_start = domain_range.get('start_row')
        raw_end = domain_range.get('end_row')
        start_col = domain_range.get('start_col')
        end_col = domain_range.get('end_col')

        # Check if any required values are None
        if raw_start is None or raw_end is None or start_col is None:
            print(f"Warning: Incomplete domain range data: start_row={raw_start}, end_row={raw_end}, start_col={start_col}, end_col={end_col}")
            return []

        # Determine headerCount from raw chart (default 1)
        header_count = 1
        raw_chart_spec = chart.get('raw_chart', {}).get('spec', {})
        for chart_type_key in ('basicChart', 'pieChart'):
            if chart_type_key in raw_chart_spec:
                header_count = raw_chart_spec[chart_type_key].get('headerCount', 1)
                break

        # Skip header rows and map to DataFrame indices
        # Sheet row 0 = header -> DataFrame columns (already stripped)
        # Sheet row 1 = DataFrame iloc[0], etc.
        start_row = max(raw_start, header_count) - header_count
        end_row = raw_end - header_count

        # Default end_col if not provided (assume single column)
        if end_col is None:
            end_col = start_col + 1

        # Extract data from the DataFrame
        # Chart ranges are 0-indexed and end_row is exclusive

        # Check if chart references columns outside the table DataFrame
        columns_in_range = start_col < len(table_data.columns) and (end_col - 1) < len(table_data.columns)

        if columns_in_range:
            # Handle single column extraction (most common for categories)
            if end_col - start_col == 1:
                col_idx = start_col
                values = table_data.iloc[start_row:end_row, col_idx].astype(str).tolist()
                values = [v.strip() for v in values if v and str(v).strip() and str(v).lower() != 'nan']
                return values
            else:
                values = []
                for row_idx in range(start_row, min(end_row, len(table_data))):
                    row_values = []
                    for col_idx in range(start_col, min(end_col, len(table_data.columns))):
                        val = str(table_data.iloc[row_idx, col_idx])
                        if val and val.strip() and val.lower() != 'nan':
                            row_values.append(val.strip())
                    if row_values:
                        values.append(' '.join(row_values))
                return values
        elif full_sheet_data:
            # Fallback: chart references columns outside the table, use raw sheet data
            print(f"Chart domain columns ({start_col}-{end_col-1}) outside table range (0-{len(table_data.columns)-1}), using full_sheet_data fallback")
            # full_sheet_data is raw rows including header; skip header_count rows
            data_rows = full_sheet_data[header_count:raw_end]
            values = []
            for row in data_rows:
                if start_col < len(row):
                    val = str(row[start_col]).strip()
                    if val and val.lower() != 'nan' and val != '':
                        values.append(val)
            return values
        else:
            print(f"Warning: Chart domain columns ({start_col}-{end_col-1}) outside table range (0-{len(table_data.columns)-1}) and no full_sheet_data provided")
            return []

    except Exception as e:
        print(f"Error extracting chart domain data: {e}")
        import traceback
        traceback.print_exc()
        return []

    return []


def extract_chart_series_data(chart: Dict[str, Any], table_data: pd.DataFrame, full_sheet_data: Optional[List[List[str]]] = None) -> List[float]:
    """
    Extract y-axis numeric values from a chart's series range.

    Args:
        chart (dict): Chart object from extract_charts_from_sheet containing:
            - 'series': List of series dicts with 'source_range' containing row/col indices
            - 'raw_chart': Full raw chart specification (fallback)
        table_data (pd.DataFrame): DataFrame containing the sheet data

    Returns:
        list: List of numeric values (floats) from the chart's y-axis.
              Returns empty list if series range not found or data extraction fails.

    Example:
        chart = {'series': [{'source_range': {'start_row': 1, 'end_row': 11, 'start_col': 6}}]}
        values = extract_chart_series_data(chart, df)
        # Returns: [15.3, 12.8, 10.5, ...]
    """
    try:
        # series_list = chart.get('series', [])
        
        # Use the first series (most common case for simple bar charts)
        # series = series_list[0]
        # source_range = series.get('source_range')

        # if not source_range:
        #     print("Warning: No source_range found in chart series")
        #     print(f"Series structure: {list(series.keys())}")
        #     return []

        # If no series in pre-parsed structure, try parsing from raw_chart
        # if series_list[0].get('type')=='UNKNOWN':
        #     print("Warning: No series found in parsed chart data, attempting raw_chart fallback")
        raw_chart = chart.get('raw_chart', {})
        chart_spec = raw_chart.get('spec', {})

        # Try basicChart (for COLUMN, BAR, LINE, etc.)
        if 'basicChart' in chart_spec:
            basic_chart = chart_spec['basicChart']
            raw_series = basic_chart.get('series', [])
            if raw_series:
                # Extract first series source range
                first_series = raw_series[0].get('series', {})
                if 'sourceRange' in first_series:
                    source_range_raw = first_series['sourceRange'].get('sources', [{}])[0]
                    source_range = {
                        'start_row': source_range_raw.get('startRowIndex'),
                        'end_row': source_range_raw.get('endRowIndex'),
                        'start_col': source_range_raw.get('startColumnIndex'),
                        'end_col': source_range_raw.get('endColumnIndex')
                    }
                    series_list = [{'source_range': source_range}]
                    print(f"Extracted series source_range from raw_chart: {source_range}")

        # Try pieChart
        elif 'pieChart' in chart_spec:
            pie_chart = chart_spec['pieChart']
            if 'series' in pie_chart:
                source_range_raw = pie_chart['series'].get('sourceRange', {}).get('sources', [{}])[0]
                if source_range_raw:
                    source_range = {
                        'start_row': source_range_raw.get('startRowIndex'),
                        'end_row': source_range_raw.get('endRowIndex'),
                        'start_col': source_range_raw.get('startColumnIndex'),
                        'end_col': source_range_raw.get('endColumnIndex')
                    }
                    series_list = [{'source_range': source_range}]
                    print(f"Extracted series source_range from raw_chart pieChart: {source_range}")

        if not series_list:
            print("Warning: Could not extract series from raw_chart either")
            return []

        

        # Get range values, handling None
        raw_start = source_range.get('start_row')
        raw_end = source_range.get('end_row')
        start_col = source_range.get('start_col')
        end_col = source_range.get('end_col')

        # Check if any required values are None
        if raw_start is None or raw_end is None or start_col is None:
            print(f"Warning: Incomplete series range data: start_row={raw_start}, end_row={raw_end}, start_col={start_col}, end_col={end_col}")
            return []

        # Determine headerCount from raw chart (default 1)
        header_count = 1
        raw_chart_spec = chart.get('raw_chart', {}).get('spec', {})
        for chart_type_key in ('basicChart', 'pieChart'):
            if chart_type_key in raw_chart_spec:
                header_count = raw_chart_spec[chart_type_key].get('headerCount', 1)
                break

        # Skip header rows and map to DataFrame indices
        start_row = max(raw_start, header_count) - header_count
        end_row = raw_end - header_count

        # Default end_col if not provided (assume single column)
        if end_col is None:
            end_col = start_col + 1

        # Extract numeric data
        values = []
        columns_in_range = start_col < len(table_data.columns) and (end_col - 1) < len(table_data.columns)

        if columns_in_range:
            # Handle single column extraction (most common for series data)
            if end_col - start_col == 1:
                col_idx = start_col
                raw_values = table_data.iloc[start_row:end_row, col_idx]
                for val in raw_values:
                    numeric_val = _parse_numeric_cell(val)
                    if numeric_val is not None:
                        values.append(numeric_val)
                return values
            else:
                for row_idx in range(start_row, min(end_row, len(table_data))):
                    for col_idx in range(start_col, min(end_col, len(table_data.columns))):
                        val = table_data.iloc[row_idx, col_idx]
                        numeric_val = _parse_numeric_cell(val)
                        if numeric_val is not None:
                            values.append(numeric_val)
                return values
        elif full_sheet_data:
            # Fallback: chart references columns outside the table, use raw sheet data
            print(f"Chart series columns ({start_col}-{end_col-1}) outside table range (0-{len(table_data.columns)-1}), using full_sheet_data fallback")
            data_rows = full_sheet_data[header_count:raw_end]
            for row in data_rows:
                if start_col < len(row):
                    numeric_val = _parse_numeric_cell(row[start_col])
                    if numeric_val is not None:
                        values.append(numeric_val)
            return values
        else:
            print(f"Warning: Chart series columns ({start_col}-{end_col-1}) outside table range (0-{len(table_data.columns)-1}) and no full_sheet_data provided")
            return []

    except Exception as e:
        print(f"Error extracting chart series data: {e}")
        import traceback
        traceback.print_exc()
        return []

    return []


def validate_chart_categories_match(
    chart_categories: List[str],
    expected_categories: List[str],
    tolerance: str = 'fuzzy'
) -> Tuple[int, int, List[str]]:
    """
    Compare chart categories against expected list.

    Args:
        chart_categories (list): List of category labels from the chart
        expected_categories (list): List of expected category labels
        tolerance (str): Matching mode - 'exact' or 'fuzzy' (default: 'fuzzy')
            - 'exact': Categories must match exactly (case-insensitive)
            - 'fuzzy': Categories can match if they contain the expected value

    Returns:
        tuple: (match_count, total_expected, missing_categories)
            - match_count (int): Number of expected categories found in chart
            - total_expected (int): Total number of expected categories
            - missing_categories (list): List of expected categories not found in chart

    Example:
        chart_cats = ['Apple Inc.', 'Microsoft', 'Google LLC']
        expected_cats = ['AAPL', 'MSFT', 'GOOGL']
        matches, total, missing = validate_chart_categories_match(
            chart_cats, expected_cats, tolerance='fuzzy'
        )
        # Returns: (3, 3, []) if all match
    """
    if not expected_categories:
        return 0, 0, []

    # Normalize categories for comparison
    chart_cats_lower = [cat.lower().strip() for cat in chart_categories]
    expected_cats_lower = [cat.lower().strip() for cat in expected_categories]

    missing = []
    match_count = 0

    for expected_cat in expected_categories:
        expected_lower = expected_cat.lower().strip()
        found = False

        if tolerance == 'exact':
            # Exact match (case-insensitive)
            if expected_lower in chart_cats_lower:
                found = True
        else:
            # Fuzzy match - check if expected is contained in any chart category
            for chart_cat_lower in chart_cats_lower:
                if expected_lower in chart_cat_lower or chart_cat_lower in expected_lower:
                    found = True
                    break

        if found:
            match_count += 1
        else:
            missing.append(expected_cat)

    return match_count, len(expected_categories), missing


def validate_chart_values_match(
    chart_values: List[float],
    expected_values: List[float],
    error_percent: float = 5.0
) -> Tuple[int, int, List[str]]:
    """
    Compare chart numeric values against expected values with tolerance.

    Args:
        chart_values (list): List of numeric values from the chart
        expected_values (list): List of expected numeric values
        error_percent (float): Allowed error percentage (default: 5.0)

    Returns:
        tuple: (match_count, total_count, mismatches_list)
            - match_count (int): Number of values that match within tolerance
            - total_count (int): Total number of comparisons made
            - mismatches_list (list): List of mismatch descriptions
                Format: ["Value 0: 15.3 vs expected 14.8 (3.4% diff)", ...]

    Example:
        chart_vals = [15.3, 12.8, 10.5]
        expected_vals = [15.0, 13.0, 10.0]
        matches, total, mismatches = validate_chart_values_match(
            chart_vals, expected_vals, error_percent=5.0
        )
        # Returns: (3, 3, []) if all within 5% tolerance
    """
    if not expected_values or not chart_values:
        return 0, 0, ["No values to compare"]

    # Ensure we compare the minimum length
    comparison_count = min(len(chart_values), len(expected_values))

    if len(chart_values) != len(expected_values):
        print(f"Warning: Chart has {len(chart_values)} values but expected {len(expected_values)}")

    match_count = 0
    mismatches = []

    for i in range(comparison_count):
        chart_val = chart_values[i]
        expected_val = expected_values[i]

        # Use the existing numerical_match_with_error function
        is_match, diff_percent = numerical_match_with_error(
            expected_val, chart_val, error_percent=error_percent
        )

        if is_match:
            match_count += 1
        else:
            mismatch_msg = f"Value {i}: {chart_val} vs expected {expected_val:.2f} ({abs(diff_percent):.1f}% diff)"
            mismatches.append(mismatch_msg)

    return match_count, comparison_count, mismatches


def identify_chart_vlm(
    chart_image_1: str,
    chart_image_2: str,
    description: str,
    model,
) -> str:
    """Use VLM to identify which chart matches a description.

    Presents two chart images to a vision language model and asks which one
    best matches the given description.

    Args:
        chart_image_1: Path to first chart image
        chart_image_2: Path to second chart image
        description: What to look for (e.g., "average running speed over time")
        model: Loaded VLM model (from load_model())

    Returns:
        str: "1" if first chart matches, "2" if second chart matches,
             "none" if neither matches
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are analyzing charts from a spreadsheet. You will see two charts and must identify which one matches the given description. Answer with just '1', '2', or 'none' if neither chart matches."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Chart 1:"},
                {"type": "image", "image": chart_image_1},
                {"type": "text", "text": "Chart 2:"},
                {"type": "image", "image": chart_image_2},
                {"type": "text", "text": f"Which chart shows {description}? Answer with just '1', '2', or 'none'."}
            ]
        }
    ]

    try:
        response = model(messages)
        # Normalize response
        response_lower = response.strip().lower()

        if '1' in response_lower and '2' not in response_lower:
            return "1"
        elif '2' in response_lower and '1' not in response_lower:
            return "2"
        elif 'none' in response_lower or 'neither' in response_lower:
            return "none"
        else:
            # Ambiguous response - try to parse
            print(f"Ambiguous VLM response: '{response}', defaulting to 'none'")
            return "none"

    except Exception as e:
        print(f"Error in identify_chart_vlm: {e}")
        return "none"


def get_series_source_range(chart: Dict[str, Any], series_index: int) -> Optional[Dict[str, int]]:
    """
    Get the source range for a specific series in a chart.

    Args:
        chart: Chart object from extract_charts_from_sheet()
        series_index: 0-based index of the series

    Returns:
        dict: Source range with 'start_row', 'end_row', 'start_col', 'end_col'
              or None if not found
    """
    try:
        raw_chart = chart.get('raw_chart', {})
        spec = raw_chart.get('spec', {})
        basic_chart = spec.get('basicChart', {})
        all_series = basic_chart.get('series', [])

        if series_index >= len(all_series):
            return None

        series = all_series[series_index]
        series_data = series.get('series', {})
        source_range = series_data.get('sourceRange', {})

        if source_range:
            sources = source_range.get('sources', [])
            if sources:
                src = sources[0]
                return {
                    'start_row': src.get('startRowIndex'),
                    'end_row': src.get('endRowIndex'),
                    'start_col': src.get('startColumnIndex'),
                    'end_col': src.get('endColumnIndex'),
                }
        return None

    except Exception as e:
        print(f"Error getting series source range: {e}")
        return None


def get_series_header_label(chart: Dict[str, Any], series_index: int, rows: List[Dict]) -> str:
    """
    Get the column header (legend label) for a specific series in a chart.

    In Google Sheets, legend labels come from the header row of the source column.
    The header is typically the first row of the series source range.

    Args:
        chart: Chart object from extract_charts_from_sheet()
        series_index: 0-based index of the series to get header for
        rows: Raw rowData from sheet (from extract_sheet_data with return_raw=True)

    Returns:
        str: The header text (legend label), or empty string if not found
    """
    try:
        source_range = get_series_source_range(chart, series_index)
        if not source_range:
            return ""

        start_row = source_range.get('start_row')
        start_col = source_range.get('start_col')

        if start_row is None or start_col is None:
            return ""

        # The header is the first row of the source range
        if start_row < len(rows):
            row = rows[start_row]
            values = row.get('values', [])
            if start_col < len(values):
                cell = values[start_col]
                return cell.get('formattedValue', '')

        return ""

    except Exception as e:
        print(f"Error getting series header label: {e}")
        return ""


def get_series_column_values(chart: Dict[str, Any], series_index: int, rows: List[Dict]) -> List[float]:
    """
    Extract all numeric values from a specific series' source column.

    Reads the raw sheet data directly rather than using a DataFrame,
    which allows access to the full data including rows beyond the table.

    Args:
        chart: Chart object from extract_charts_from_sheet()
        series_index: 0-based index of the series
        rows: Raw rowData from sheet (from extract_sheet_data with return_raw=True)

    Returns:
        list: List of float values from the column (excluding header row)
    """
    try:
        source_range = get_series_source_range(chart, series_index)
        if not source_range:
            return []

        start_row = source_range.get('start_row')
        end_row = source_range.get('end_row')
        start_col = source_range.get('start_col')

        if start_row is None or end_row is None or start_col is None:
            return []

        values = []
        # Skip header row (start_row), get data from start_row+1 to end_row
        for row_idx in range(start_row + 1, min(end_row, len(rows))):
            row = rows[row_idx]
            row_values = row.get('values', [])

            if start_col < len(row_values):
                cell = row_values[start_col]
                formatted_value = cell.get('formattedValue', '')

                # Try to extract numeric value
                try:
                    clean_value = formatted_value.strip()

                    # Handle time format (m:ss or mm:ss) -> convert to decimal minutes
                    if ':' in clean_value:
                        parts = clean_value.split(':')
                        if len(parts) == 2:
                            minutes = float(parts[0])
                            seconds = float(parts[1])
                            decimal_minutes = minutes + (seconds / 60.0)
                            values.append(decimal_minutes)
                            continue

                    # Handle percentages, currency, etc.
                    clean_value = clean_value.rstrip('%').replace(',', '')
                    clean_value = clean_value.replace('$', '').replace('€', '').replace('£', '')
                    if clean_value:
                        values.append(float(clean_value))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue

        return values

    except Exception as e:
        print(f"Error getting series column values: {e}")
        return []


def validate_constant_series(
    values: List[float],
    expected_range: Tuple[float, float],
    tolerance: float = 0.01
) -> Tuple[bool, Optional[float], str]:
    """
    Validate that a series represents a constant baseline within expected range.

    Checks that:
    1. All values in the series are identical (within tolerance)
    2. The constant value falls within the expected range

    Args:
        values: List of numeric values from the series
        expected_range: (min, max) tuple for expected value range
        tolerance: Max allowed variation between values (default 0.01)

    Returns:
        tuple: (is_valid, baseline_value, details_message)
            - is_valid: True if constant and in range
            - baseline_value: The constant value, or None if not constant
            - details_message: Description of result or failure reason
    """
    if not values:
        return False, None, "No values found in series"

    # Check if all values are the same (within tolerance)
    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val

    if value_range > tolerance:
        return False, None, f"Values not constant: range {min_val:.2f} to {max_val:.2f} (diff: {value_range:.2f})"

    # Use the mean as the baseline value
    baseline_value = sum(values) / len(values)

    # Check if within expected range
    range_min, range_max = expected_range
    if baseline_value < range_min or baseline_value > range_max:
        return False, baseline_value, f"Value {baseline_value:.2f} outside expected range [{range_min}, {range_max}]"

    return True, baseline_value, f"Constant baseline at {baseline_value:.2f} (within range [{range_min}, {range_max}])"


def get_series_line_style(chart: Dict[str, Any], series_index: int) -> Optional[str]:
    """
    Get the line style for a specific series in a chart.

    Args:
        chart: Chart object from extract_charts_from_sheet()
        series_index: 0-based index of the series

    Returns:
        str: Line style type (e.g., 'SOLID', 'DOTTED', 'DASHED', 'MEDIUM_DASHED')
             or None if not found
    """
    try:
        raw_chart = chart.get('raw_chart', {})
        spec = raw_chart.get('spec', {})
        basic_chart = spec.get('basicChart', {})
        all_series = basic_chart.get('series', [])

        if series_index >= len(all_series):
            return None

        series = all_series[series_index]
        line_style = series.get('lineStyle', {})

        if line_style:
            return line_style.get('type', 'SOLID')

        return 'SOLID'  # Default if no line style specified

    except Exception as e:
        print(f"Error getting series line style: {e}")
        return None


def get_series_color(chart: Dict[str, Any], series_index: int) -> Optional[Dict[str, float]]:
    """
    Get the color for a specific series in a chart.

    Args:
        chart: Chart object from extract_charts_from_sheet()
        series_index: 0-based index of the series

    Returns:
        dict: {'red': float, 'green': float, 'blue': float} with values 0-1,
              or None if not found
    """
    try:
        raw_chart = chart.get('raw_chart', {})
        spec = raw_chart.get('spec', {})
        basic_chart = spec.get('basicChart', {})
        all_series = basic_chart.get('series', [])

        if series_index >= len(all_series):
            return None

        series = all_series[series_index]

        # Try to get color from various possible locations
        # 1. Check 'color' field directly
        color = series.get('color', {})
        if color and ('red' in color or 'green' in color or 'blue' in color):
            return {
                'red': color.get('red', 0),
                'green': color.get('green', 0),
                'blue': color.get('blue', 0)
            }

        # 2. Check 'colorStyle' field
        color_style = series.get('colorStyle', {})
        if color_style:
            rgb_color = color_style.get('rgbColor', {})
            if rgb_color:
                return {
                    'red': rgb_color.get('red', 0),
                    'green': rgb_color.get('green', 0),
                    'blue': rgb_color.get('blue', 0)
                }

        return None

    except Exception as e:
        print(f"Error getting series color: {e}")
        return None


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


def check_chart_overlap(
    chart: Dict[str, Any],
    table_start_row: int,
    table_end_row: int,
    other_charts: List[Dict[str, Any]],
    table_start_col: int = 0,
    table_end_col: int = None
) -> Tuple[bool, str]:
    """
    Check if chart overlaps with table or other charts.

    Args:
        chart: Chart object to check
        table_start_row: Starting row of the data table
        table_end_row: Ending row of the data table
        other_charts: List of other chart objects
        table_start_col: Starting column of the data table (default 0)
        table_end_col: Ending column of the data table (default None, meaning no column check)

    Returns:
        tuple: (has_overlap: bool, overlap_details: str)
    """
    position = chart.get('position', {})
    if position.get('type') != 'overlay':
        return False, "Chart not in overlay position"

    anchor_cell = position.get('anchor_cell', {})
    anchor_row = anchor_cell.get('row') if anchor_cell else None
    anchor_col = anchor_cell.get('col') if anchor_cell else None
    # API omits row/col when they're 0, and stored value is None
    if anchor_row is None:
        anchor_row = 0
    if anchor_col is None:
        anchor_col = 0
    height = position.get('height', 0)
    width = position.get('width', 0)

    # Estimate chart end row (assuming ~20 pixels per row)
    chart_end_row = anchor_row + (height // 20) if height else anchor_row + 15
    # Estimate chart end column (assuming ~100 pixels per column)
    chart_end_col = anchor_col + (width // 100) if width else anchor_col + 6

    # Check overlap with table (requires both row AND column overlap)
    row_overlap = anchor_row < table_end_row and chart_end_row > table_start_row

    # Only check column overlap if table_end_col is specified
    if table_end_col is not None:
        col_overlap = anchor_col < table_end_col and chart_end_col > table_start_col
    else:
        # If no column info provided, only check row overlap (legacy behavior)
        col_overlap = True

    if row_overlap and col_overlap:
        return True, f"Chart overlaps with data table (chart rows {anchor_row}-{chart_end_row}, cols {anchor_col}-{chart_end_col}; table rows {table_start_row}-{table_end_row}, cols {table_start_col}-{table_end_col})"

    # Check overlap with other charts
    chart_id = chart.get('chart_id')
    for other in other_charts:
        if other.get('chart_id') == chart_id:
            continue

        other_pos = other.get('position', {})
        if other_pos.get('type') != 'overlay':
            continue

        other_anchor_cell = other_pos.get('anchor_cell', {})
        other_anchor_row = other_anchor_cell.get('row') if other_anchor_cell else None
        other_anchor_col = other_anchor_cell.get('col') if other_anchor_cell else None
        if other_anchor_row is None:
            other_anchor_row = 0
        if other_anchor_col is None:
            other_anchor_col = 0
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


def check_point_shape(chart: Dict[str, Any], chart_type: str) -> Tuple[bool, str]:
    """
    Check if chart displays data as circular points (not other shapes like square, triangle, etc.).

    According to the Google Sheets API, PointShape enum values are:
    - POINT_SHAPE_UNSPECIFIED (defaults to circle)
    - CIRCLE
    - DIAMOND
    - HEXAGON
    - PENTAGON
    - SQUARE
    - STAR
    - TRIANGLE
    - X_MARK

    Args:
        chart: Chart object from extract_charts_from_sheet()
        chart_type: Type of chart (SCATTER, LINE, etc.)

    Returns:
        tuple: (has_circular_points: bool, details: str)
    """
    # Shapes that are considered circular
    CIRCULAR_SHAPES = {'CIRCLE', 'POINT_SHAPE_UNSPECIFIED', '', None}

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
    point_shape = point_style.get('shape', '')

    # Check line style
    line_style = main_series.get('lineStyle', {})
    line_width = line_style.get('width', 2)  # Default line width is usually 2

    # First check: Does the chart have visible points?
    has_points = False

    # SCATTER charts always show points
    if chart_type == 'SCATTER':
        has_points = True
    # LINE/AREA charts with explicit point size
    elif point_size > 0:
        has_points = True
    # Combo chart with SCATTER series type
    elif main_series.get('type', '') == 'SCATTER':
        has_points = True

    if not has_points:
        return False, f"No visible points (chart_type={chart_type}, point_size={point_size})"

    # Second check: Is the shape circular?
    if point_shape in CIRCULAR_SHAPES:
        shape_desc = "circle" if point_shape in {'', None, 'POINT_SHAPE_UNSPECIFIED'} else point_shape.lower()
        if chart_type == 'SCATTER':
            return True, f"Scatter chart with {shape_desc} points"
        elif point_size > 0 and line_width == 0:
            return True, f"Points-only chart with {shape_desc} markers (size={point_size})"
        else:
            return True, f"Chart has {shape_desc} point markers (size={point_size})"
    else:
        # Shape is not circular
        return False, f"Points are {point_shape.lower()} shape, not circular"


def get_all_series_metadata(chart: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract metadata for all series in a chart.

    Provides comprehensive information about each series including line style,
    color, target axis, and source data range. Useful for identifying baseline
    series or analyzing chart structure.

    Args:
        chart: Chart object from extract_charts_from_sheet()

    Returns:
        list: List of series info dicts, each containing:
            - 'index': Series index (0-based)
            - 'type': Series type (LINE, AREA, etc.)
            - 'line_style': Line style type (SOLID, DOTTED, DASHED, etc.) or None
            - 'color': RGB color dict or None
            - 'target_axis': Target axis (LEFT_AXIS, RIGHT_AXIS)
            - 'source_range': Dict with start_row, end_row, start_col, end_col
    """
    raw_chart = chart.get('raw_chart', {})
    spec = raw_chart.get('spec', {})
    basic_chart = spec.get('basicChart', {})
    all_series = basic_chart.get('series', [])

    series_metadata = []

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

        series_metadata.append(series_info)

    return series_metadata


def get_chart_type(chart: Dict[str, Any]) -> str:
    """
    Get the chart type from a chart object.

    Checks both the pre-parsed 'chart_type' field and the raw basicChart spec.
    Returns values like 'LINE', 'SCATTER', 'COLUMN', 'BAR', 'AREA', 'COMBO', etc.

    Args:
        chart: Chart object from extract_charts_from_sheet()

    Returns:
        str: Chart type string (e.g., 'LINE', 'SCATTER', 'UNKNOWN')
    """
    # First try the pre-parsed chart_type field
    chart_type = chart.get('chart_type', '')
    if chart_type and chart_type != 'UNKNOWN':
        return chart_type.upper()

    # Fallback to raw chart spec
    try:
        raw_chart = chart.get('raw_chart', {})
        spec = raw_chart.get('spec', {})
        basic_chart = spec.get('basicChart', {})

        if basic_chart:
            return basic_chart.get('chartType', 'UNKNOWN').upper()

        # Check for pie chart
        if 'pieChart' in spec:
            return 'PIE'

        # Check for other chart types in spec
        for key in spec.keys():
            if key.endswith('Chart'):
                return key.replace('Chart', '').upper()

    except Exception as e:
        print(f"Error getting chart type: {e}")

    return 'UNKNOWN'


def identify_series_by_content(
    chart: Dict[str, Any],
    rows: List[Dict],
    keywords: List[str],
    expected_value_range: Optional[Tuple[float, float]] = None,
    require_constant: bool = False,
    exclude_indices: Optional[List[int]] = None,
    matched_columns: Optional[Dict[str, str]] = None,
    column_name: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    model: Any = None,
    description: str = ""
) -> Optional[int]:
    """
    Identify a chart series by its content (legend label and/or values).

    Uses keyword matching on legend labels and optional value range analysis
    to identify a specific series in a chart.

    Args:
        chart: Chart object from extract_charts_from_sheet()
        rows: Raw rowData from sheet (from extract_sheet_data with return_raw=True)
        keywords: Keywords to match against series legend labels
        expected_value_range: Optional (min, max) tuple for expected value range
        require_constant: If True, only consider series with constant values (low variance)
        exclude_indices: List of series indices to skip (already identified)
        matched_columns: Column mapping (e.g., from checkpoint matching)
        column_name: Key in matched_columns to match against series source column
        df: DataFrame with sheet data (for column index lookup)
        model: Optional LLM model for fallback keyword matching
        description: Description for LLM fallback on keyword matching

    Returns:
        int: Series index (0-based), or None if not found
    """
    # Get all series from chart
    raw_chart = chart.get('raw_chart', {})
    spec = raw_chart.get('spec', {})
    basic_chart = spec.get('basicChart', {})
    all_series = basic_chart.get('series', [])

    if not all_series:
        return None

    exclude_indices = exclude_indices or []

    # Collect series info
    series_info = []
    for i in range(len(all_series)):
        if i in exclude_indices:
            continue

        label = get_series_header_label(chart, i, rows) if rows else ""
        values = get_series_column_values(chart, i, rows) if rows else []
        source_range = get_series_source_range(chart, i)

        # Calculate variance for variable vs constant detection
        variance = 0
        mean_value = None
        if values and len(values) > 1:
            mean_value = sum(values) / len(values)
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)

        # Check if constant (low variance)
        is_constant = variance < 0.1 if values else False

        series_info.append({
            "index": i,
            "label": label,
            "values": values,
            "variance": variance,
            "mean": mean_value,
            "is_constant": is_constant,
            "source_col": source_range.get('start_col') if source_range else None,
        })

    # Filter by constant requirement if specified
    if require_constant:
        series_info = [s for s in series_info if s["is_constant"]]

    # Step 1: Try to identify by keyword SUBSTRING matching
    for info in series_info:
        label = info["label"]
        if not label:
            continue

        label_lower = label.lower()
        for kw in keywords:
            if kw.lower() in label_lower:
                # If value range specified, verify it matches
                if expected_value_range is not None:
                    if info["mean"] is not None:
                        range_min, range_max = expected_value_range
                        if range_min <= info["mean"] <= range_max:
                            return info["index"]
                    # Value range check failed, continue searching
                else:
                    return info["index"]

    # Step 2: For unidentified series with value range, try value range matching
    if expected_value_range is not None:
        range_min, range_max = expected_value_range
        for info in series_info:
            if info["mean"] is not None and range_min <= info["mean"] <= range_max:
                return info["index"]

    # Step 3: LLM fallback for keyword matching (if model provided)
    if model is not None and keywords:
        for info in series_info:
            label = info["label"]
            if not label:
                continue

            match = keywords_match_robust(
                texts=label,
                keywords=keywords,
                model=model,
                description=description or f"legend label matching keywords {keywords}"
            )
            if match:
                # If value range specified, verify it matches
                if expected_value_range is not None:
                    if info["mean"] is not None:
                        range_min, range_max = expected_value_range
                        if range_min <= info["mean"] <= range_max:
                            return info["index"]
                else:
                    return info["index"]

    # Step 4: Try to match by source column (if matched_columns and column_name provided)
    if matched_columns and column_name and df is not None:
        target_col_name = matched_columns.get(column_name)
        if target_col_name and target_col_name in df.columns:
            expected_col_idx = df.columns.get_loc(target_col_name)
            # Handle duplicate column names (get_loc returns array/slice instead of int)
            if not isinstance(expected_col_idx, int):
                import numpy as np
                if isinstance(expected_col_idx, np.ndarray):
                    expected_col_idx = int(np.where(expected_col_idx)[0][0])
                elif isinstance(expected_col_idx, slice):
                    expected_col_idx = expected_col_idx.start or 0
            for info in series_info:
                if info["source_col"] == expected_col_idx:
                    return info["index"]

    # Step 5: If no keywords/range specified, return highest variance series (main data)
    if not keywords and expected_value_range is None and not require_constant:
        candidates = [info for info in series_info if not info["is_constant"]]
        if candidates:
            best = max(candidates, key=lambda x: x["variance"])
            return best["index"]
        # Fallback: return first unexcluded series
        if series_info:
            return series_info[0]["index"]

    return None


def find_chart_by_metadata(
    charts: List[Dict[str, Any]],
    title_keywords: List[str],
    y_axis_keywords: List[str],
    title_description: str = "",
    axis_description: str = "",
    matched_columns: Optional[Dict[str, str]] = None,
    column_name: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    model: Any = None
) -> Optional[Dict[str, Any]]:
    """
    Find a chart by matching metadata (title, axis labels, series data).

    Matching order:
    1. Chart title matches title_keywords
    2. Y-axis label matches y_axis_keywords
    3. Series data matches column_name from matched_columns

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        title_keywords: Keywords to match in chart title
        y_axis_keywords: Keywords to match in Y-axis label
        title_description: Description for LLM fallback on title matching
        axis_description: Description for LLM fallback on axis matching
        matched_columns: Column mapping (e.g., from checkpoint 2)
        column_name: Key in matched_columns to match against series
        df: DataFrame with sheet data (for column index lookup)
        model: Optional LLM model for fallback matching

    Returns:
        Chart object or None if not found
    """
    if not charts:
        return None

    # Collect all chart titles for robust matching
    chart_titles = [chart.get('title', '') for chart in charts]
    chart_titles = [t for t in chart_titles if t]  # Filter empty titles

    # Step 1: Title matching using keywords_match_robust
    if chart_titles and title_keywords:
        matched_title = keywords_match_robust(
            texts=chart_titles,
            keywords=title_keywords,
            substring=True,
            model=model,
            description=title_description if title_description else "chart title"
        )
        if matched_title:
            # Find and return the chart with matching title
            for chart in charts:
                if chart.get('title', '') == matched_title:
                    return chart

    # Step 2: Axis label matching using keywords_match_robust
    if y_axis_keywords:
        y_axis_labels = []
        for chart in charts:
            axis_labels = get_chart_axis_labels(chart)
            y_label = axis_labels.get('y_axis', '')
            if y_label:
                y_axis_labels.append((chart, y_label))

        if y_axis_labels:
            labels_only = [label for _, label in y_axis_labels]
            matched_label = keywords_match_robust(
                texts=labels_only,
                keywords=y_axis_keywords,
                substring=True,
                model=model,
                description=axis_description if axis_description else "Y-axis label"
            )
            if matched_label:
                # Find and return the chart with matching axis label
                for chart, label in y_axis_labels:
                    if label == matched_label:
                        return chart

    # Step 3: Series data matching (if matched_columns and column_name provided)
    if matched_columns and column_name and df is not None:
        target_col_name = matched_columns.get(column_name)
        if target_col_name and target_col_name in df.columns:
            target_col_idx = df.columns.get_loc(target_col_name)
            for chart in charts:
                series_list = chart.get('series', [])
                for series in series_list:
                    src = series.get('source_range', {})
                    if src.get('start_col') == target_col_idx:
                        return chart

    return None
