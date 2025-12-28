"""
Utility functions for validating charts in Google Sheets.

This module provides generalized functions for extracting and validating
chart data against expected values.
"""

import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from .text_utils import numerical_match_with_error


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


def extract_chart_domain_data(chart: Dict[str, Any], table_data: pd.DataFrame) -> List[str]:
    """
    Extract x-axis category labels from a chart's domain range.

    Args:
        chart (dict): Chart object from extract_charts_from_sheet containing:
            - 'data_range': Dict with 'domain_range' containing row/col indices
            - 'raw_chart': Full raw chart specification (fallback)
        table_data (pd.DataFrame): DataFrame containing the sheet data

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
                    domain_source = pie_chart['domain'].get('sourceRange', {})
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
        # Note: Adjust for 0-indexing in DataFrame
        start_row = domain_range.get('start_row')-1
        end_row = domain_range.get('end_row')-1
        start_col = domain_range.get('start_col')
        end_col = domain_range.get('end_col')

        # Check if any required values are None
        if start_row is None or end_row is None or start_col is None:
            print(f"Warning: Incomplete domain range data: start_row={start_row}, end_row={end_row}, start_col={start_col}, end_col={end_col}")
            return []

        # Default end_col if not provided (assume single column)
        if end_col is None:
            end_col = start_col + 1

        # Extract data from the DataFrame
        # Note: Chart ranges are 0-indexed, DataFrame.iloc uses 0-indexing too
        # end_row is exclusive in the chart API

        # Handle single column extraction (most common for categories)
        if end_col - start_col == 1:
            # Single column
            col_idx = start_col
            if col_idx < len(table_data.columns):
                values = table_data.iloc[start_row:end_row, col_idx].astype(str).tolist()
                # Filter out empty strings and NaN
                values = [v.strip() for v in values if v and str(v).strip() and str(v).lower() != 'nan']
                return values
        else:
            # Multiple columns - concatenate or take first non-empty
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

    except Exception as e:
        print(f"Error extracting chart domain data: {e}")
        import traceback
        traceback.print_exc()
        return []

    return []


def extract_chart_series_data(chart: Dict[str, Any], table_data: pd.DataFrame) -> List[float]:
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
        series_list = chart.get('series', [])

        # If no series in pre-parsed structure, try parsing from raw_chart
        if series_list[0].get('type')=='UNKNOWN':
            print("Warning: No series found in parsed chart data, attempting raw_chart fallback")
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
                    source_range_raw = pie_chart['series'].get('sourceRange', {})
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

        # Use the first series (most common case for simple bar charts)
        series = series_list[0]
        source_range = series.get('source_range')

        if not source_range:
            print("Warning: No source_range found in chart series")
            print(f"Series structure: {list(series.keys())}")
            return []

        # Get range values, handling None
        start_row = source_range.get('start_row')-1
        end_row = source_range.get('end_row')-1
        start_col = source_range.get('start_col')
        end_col = source_range.get('end_col')

        # Check if any required values are None
        if start_row is None or end_row is None or start_col is None:
            print(f"Warning: Incomplete series range data: start_row={start_row}, end_row={end_row}, start_col={start_col}, end_col={end_col}")
            return []

        # Default end_col if not provided (assume single column)
        if end_col is None:
            end_col = start_col + 1

        # Extract numeric data from the DataFrame
        values = []

        # Handle single column extraction (most common for series data)
        if end_col - start_col == 1:
            # Single column
            col_idx = start_col
            if col_idx < len(table_data.columns):
                raw_values = table_data.iloc[start_row:end_row, col_idx]
                for val in raw_values:
                    try:
                        # Convert to float, handling various formats
                        if pd.isna(val):
                            continue

                        # Handle string percentages like "15.3%"
                        if isinstance(val, str):
                            val = val.strip().rstrip('%')

                        numeric_val = float(val)
                        values.append(numeric_val)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
                return values
        else:
            # Multiple columns - extract all numeric values
            for row_idx in range(start_row, min(end_row, len(table_data))):
                for col_idx in range(start_col, min(end_col, len(table_data.columns))):
                    val = table_data.iloc[row_idx, col_idx]
                    try:
                        if pd.isna(val):
                            continue

                        if isinstance(val, str):
                            val = val.strip().rstrip('%')

                        numeric_val = float(val)
                        values.append(numeric_val)
                    except (ValueError, TypeError):
                        continue
            return values

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
