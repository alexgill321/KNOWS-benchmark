"""Google Sheets-specific utility functions.

This module consolidates all Google Sheets operations:
- Sheet content fetching
- Table extraction (formal API and manual)
- Chart extraction
- Structure extraction
- Header detection

For general Google services (Drive, authentication), see google_services_utils.py.
"""

import pandas as pd
from typing import Optional, Union, Tuple, List, Any, Dict
import re
from googleapiclient.discovery import build

# Import SheetTable from table_utils
from src.browsergym.knows.eval.eval_utils.table_utils import SheetTable

# Import authentication helper (needed for extract_structure_from_sheet)
from src.browsergym.knows.eval.eval_utils.google_services_helpers import authenticate


# =============================================================================
# CORE FETCHING FUNCTIONS
# =============================================================================

def get_sheet_content(sheet_id: str, service) -> Optional[dict]:
    """Fetches the content of a Google Sheet.

    Args:
        sheet_id (str): The ID of the Google Sheet.
        service: The Google Sheets service instance.

    Returns:
        dict: The full spreadsheet data. Returns None if an error occurs.
    """
    try:
        print(f"Fetching sheet content for ID: {sheet_id}")

        # `includeGridData=True` includes values, formatting, and layout info
        sheet = service.spreadsheets().get(spreadsheetId=sheet_id, includeGridData=True).execute()

        print("Sheet content fetched successfully.")
        return sheet

    except Exception as e:
        print(f"Error fetching sheet content: {e}")
        return None


def detect_header_row(rows: list, max_rows_to_check: int = 10,
                      required_columns: list = None, model=None) -> int:
    """Detect which row contains the table headers in Google Sheets data.

    Detection strategy (in order):
    1. Keyword matching: When required_columns is provided, scores each row by
       how many expected column keywords match its cell values.
    2. LLM fallback: When keyword matching finds 0 matches and a model is
       provided, asks the LLM to identify the header row.
    3. Heuristic fallback: Legacy path for callers that don't pass
       required_columns (text ratio, density scoring).

    Args:
        rows: List of row data from Google Sheets API (rowData from get_sheet_content).
        max_rows_to_check: Maximum number of rows to scan for headers.
        required_columns: Optional list of (col_name, keywords) tuples, same format
            as match_columns(). Example:
            [("Run Name", ["run name", "name"]), ("Price", ["price", "cost"])]
        model: Optional LLM model for fallback when keyword matching fails.

    Returns:
        0-indexed row number most likely to be the header row.
    """
    if not rows:
        return 0

    # Phase 1: Keyword-based detection
    if required_columns:
        from .text_utils import keywords_exact_match

        best_row = 0
        best_matches = 0

        for row_idx in range(min(len(rows), max_rows_to_check)):
            row = rows[row_idx]
            values = row.get('values', [])
            if not values:
                continue

            cell_texts = [v.get('formattedValue', '') for v in values]
            match_count = 0

            for _col_name, keywords in required_columns:
                for cell_text in cell_texts:
                    if cell_text and keywords_exact_match(cell_text, keywords):
                        match_count += 1
                        break

            if match_count > best_matches:
                best_matches = match_count
                best_row = row_idx

        if best_matches > 0:
            print(f"Header row detected via keyword matching: row {best_row} ({best_matches}/{len(required_columns)} keywords matched)")
            return best_row

        # Phase 2: LLM fallback
        if model is not None:
            row_descriptions = []
            for row_idx in range(min(len(rows), max_rows_to_check)):
                values = rows[row_idx].get('values', [])
                cells = [v.get('formattedValue', '') for v in values if v.get('formattedValue', '')]
                if cells:
                    row_descriptions.append(f"Row {row_idx}: {cells}")

            if row_descriptions:
                prompt = (
                    "Which row number contains the column headers for this spreadsheet?\n\n"
                    + "\n".join(row_descriptions)
                    + "\n\nRespond with ONLY the row number (integer)."
                )
                try:
                    messages = [
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ]
                    response = model(messages)
                    match = re.search(r'\d+', response.strip())
                    if match:
                        detected_row = int(match.group())
                        if 0 <= detected_row < min(len(rows), max_rows_to_check):
                            print(f"Header row detected via LLM: row {detected_row}")
                            return detected_row
                except Exception as e:
                    print(f"WARNING: LLM header detection failed: {e}")

    # Phase 3: Heuristic fallback (legacy path for callers without required_columns)
    best_row = 0
    best_score = -1

    for row_idx in range(min(len(rows), max_rows_to_check)):
        row = rows[row_idx]
        values = row.get('values', [])

        if not values:
            continue

        non_empty_count = 0
        text_count = 0
        total_cells = len(values)

        for cell in values:
            formatted = cell.get('formattedValue', '')
            if formatted:
                non_empty_count += 1
                try:
                    float(formatted.replace(',', '').replace('$', '').replace('%', ''))
                except ValueError:
                    text_count += 1

        if non_empty_count == 0:
            continue

        text_ratio = text_count / non_empty_count if non_empty_count > 0 else 0
        density = non_empty_count / max(total_cells, 1)
        score = (text_ratio * 0.4) + (density * 0.3) + (non_empty_count * 0.02)

        if 1 <= row_idx <= 3:
            score += 0.1

        if score > best_score:
            best_score = score
            best_row = row_idx

    return best_row

# ---------------------------------------------------------------------------
# Sheet tab navigation
# ---------------------------------------------------------------------------

def get_sheet_by_name(sheet_raw: Dict[str, Any], name: str) -> Optional[Dict]:
    """Return the sheet-level dict for a tab whose title matches *name*.

    The Google Sheets API response nests per-tab data under
    ``sheet_raw['sheets'][i]``.  Most ``eval_utils`` helpers hard-code
    ``sheets[0]``; this function lets callers target any tab.

    Args:
        sheet_raw: Full spreadsheet response from ``get_sheet_content()``.
        name: Case-insensitive substring to match against tab titles.

    Returns:
        The matching sheet dict (with 'properties', 'data', etc.) or None.
    """
    if not sheet_raw:
        return None
    for sheet in sheet_raw.get("sheets", []):
        title = sheet.get("properties", {}).get("title", "")
        if name.lower() in title.lower():
            return sheet
    return None


# =============================================================================
# TABLE EXTRACTION FUNCTIONS
# =============================================================================

def _detect_data_bounds(header_row_values: list) -> Tuple[int, int]:
    """Detect start and end column indices from header row.

    Returns (start_col, end_col) where:
    - start_col: Index of first non-empty cell
    - end_col: Index after last non-empty cell (exclusive)
    """
    start_col = None
    end_col = 0

    for i, cell in enumerate(header_row_values):
        value = cell.get('formattedValue', '')
        if value and value.strip():
            if start_col is None:
                start_col = i
            end_col = i + 1

    return (start_col or 0, end_col)


def _manual_extract_to_sheettable(sheet_raw: dict, sheet_index: int = 0) -> Optional[SheetTable]:
    """Manual extraction from raw sheet data, returns SheetTable with position metadata.

    This function extracts table data directly from the raw Google Sheets API response,
    detecting header row, start/end columns, and constructing a proper SheetTable object.

    Args:
        sheet_raw: Raw sheet data from get_sheet_content().
        sheet_index: Which sheet tab to extract from (default 0 = first).

    Returns:
        SheetTable with DataFrame and position metadata, or None if extraction fails.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets or sheet_index >= len(sheets):
            return None

        sheet_data = sheets[sheet_index]
        sheet_name = sheet_data.get('properties', {}).get('title', '')
        grid_data = sheet_data.get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        if not rows:
            return None

        # Detect header row
        header_row_idx = detect_header_row(rows)
        header_row_values = rows[header_row_idx].get('values', [])

        if not header_row_values:
            return None

        # Detect start and end columns from header row
        start_col, end_col = _detect_data_bounds(header_row_values)

        if end_col <= start_col:
            return None

        # Extract headers within the detected bounds
        headers = []
        for i in range(start_col, end_col):
            if i < len(header_row_values):
                cell = header_row_values[i]
                headers.append(cell.get('formattedValue', f'Column{i - start_col}'))
            else:
                headers.append(f'Column{i - start_col}')

        # Extract data rows (after header), stopping at gaps of 5+ consecutive empty rows
        data_rows = []
        consecutive_empty = 0
        for row in rows[header_row_idx + 1:]:
            values = row.get('values', [])
            row_data = []
            for c_idx in range(start_col, end_col):
                if c_idx < len(values):
                    cell = values[c_idx]
                    row_data.append(cell.get('formattedValue', ''))
                else:
                    row_data.append('')

            if any(row_data):
                consecutive_empty = 0
                data_rows.append(row_data)
            else:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    break  # End of table — large gap detected

        if not data_rows:
            return None

        df = pd.DataFrame(data_rows, columns=headers)

        # Calculate position metadata
        end_row = header_row_idx + 1 + len(data_rows)

        return SheetTable(
            df=df,
            start_col=start_col,
            end_col=end_col,
            start_row=header_row_idx,
            end_row=end_row,
            sheet_name=sheet_name
        )

    except Exception as e:
        print(f"Error in manual SheetTable extraction: {e}")
        return None


def extract_tables_from_sheet(sheet_id: str, service) -> List[SheetTable]:
    """Extract all tables from a Google Sheet as SheetTable objects.

    Uses the Google Sheets API 'tables' property to get table positions
    directly from the sheet metadata.

    Args:
        sheet_id (str): The ID of the Google Sheet to extract tables from.
        service: The Google Sheets service instance.

    Returns:
        list[SheetTable]: List of SheetTable objects for each detected table.
    """
    sheet_obj = get_sheet_content(sheet_id, service)
    if not sheet_obj:
        return []

    tables = []
    for tab in sheet_obj.get('sheets', []):
        sheet_name = tab.get('properties', {}).get('title', '')
        tab_tables = tab.get('tables', [])
        grid_data = tab.get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        for table in tab_tables:
            # Extract position from GridRange
            table_range = table.get('range', {})
            start_col = table_range.get('startColumnIndex', 0)
            end_col = table_range.get('endColumnIndex', 0)
            start_row = table_range.get('startRowIndex', 0)
            end_row = table_range.get('endRowIndex', 0)

            # Get column names from columnProperties
            col_props = table.get('columnProperties', [])
            headers = [col.get('columnName', f'Column{i}') for i, col in enumerate(col_props)]

            # If no columnProperties, extract headers from first row
            if not headers and rows and start_row < len(rows):
                header_row = rows[start_row].get('values', [])
                headers = []
                for c_idx in range(start_col, min(end_col, len(header_row))):
                    cell = header_row[c_idx] if c_idx < len(header_row) else {}
                    headers.append(cell.get('formattedValue', f'Column{c_idx - start_col}'))

            if not headers:
                continue

            # Extract data rows (skip header row)
            data_rows = []
            for r_idx in range(start_row + 1, min(end_row, len(rows))):
                row = rows[r_idx].get('values', []) if r_idx < len(rows) else []
                row_data = []
                for c_idx in range(start_col, end_col):
                    cell = row[c_idx] if c_idx < len(row) else {}
                    row_data.append(cell.get('formattedValue', ''))
                # Pad or truncate to match header length
                row_data = (row_data + [''] * len(headers))[:len(headers)]
                data_rows.append(row_data)

            if not data_rows:
                continue

            df = pd.DataFrame(data_rows, columns=headers)

            tables.append(SheetTable(
                df=df,
                start_col=start_col,
                end_col=end_col,
                start_row=start_row,
                end_row=end_row,
                sheet_name=sheet_name
            ))


    return tables

def extract_sheet_data(
    sheet_id: str,
    service,
    prefer_table_api: bool = True,
    sheet_index: int = 0,
    return_raw: bool = False
) -> Union[SheetTable, List[SheetTable], Tuple[Any, dict], None]:
    """Extract data from Google Sheet with automatic fallback.

    Always returns SheetTable objects for consistent evaluator logic.

    First attempts extract_tables_from_sheet() (formal table API).
    Falls back to manual extraction wrapped in SheetTable when that returns empty.

    Args:
        sheet_id: Google Sheets document ID.
        service: Google Sheets API service instance.
        prefer_table_api: If True, try formal table API first (default True).
        sheet_index: Which sheet tab to extract from for manual fallback (default 0).
        return_raw: If True, also return raw sheet data dict.

    Returns:
        SheetTable: Single table (most common case).
        List[SheetTable]: Multiple tables if formal API found >1.
        Tuple[SheetTable/List[SheetTable], dict]: If return_raw=True, includes raw sheet data.
        None: If extraction failed.
    """
    # Fetch raw sheet data first (needed for both methods)
    sheet_raw = get_sheet_content(sheet_id, service)
    if not sheet_raw:
        return (None, None) if return_raw else None

    result = None

    if prefer_table_api:
        # Try formal table API first (uses cached sheet_raw internally via separate call)
        # Note: extract_tables_from_sheet calls get_sheet_content again, but that's okay
        # for now - we could optimize this later
        tables = extract_tables_from_sheet(sheet_id, service)
        if tables:
            result = tables[0] if len(tables) == 1 else tables
            print(f"Extracted {len(tables)} table(s) using formal table API")

    # Fallback to manual extraction
    if result is None:
        print("Formal table API returned no tables, falling back to manual extraction")
        result = _manual_extract_to_sheettable(sheet_raw, sheet_index)
        if result:
            print(f"Manual extraction successful: {len(result.df)} rows, columns {result.start_col}-{result.end_col}")

    return (result, sheet_raw) if return_raw else result


def parse_sheet_to_dataframe(sheet_raw: dict, header_row: int = None,
                             required_columns: list = None, model=None) -> Optional[pd.DataFrame]:
    """Parse raw Google Sheets API response into a pandas DataFrame.

    Takes the raw sheet data from get_sheet_content() and extracts table data
    starting from the specified header row. Useful when extract_tables_from_sheet()
    doesn't detect a formal table structure.

    If header_row is not specified, the function will attempt to automatically
    detect which row contains the column headers. When required_columns is
    provided, uses keyword matching (with LLM fallback) for detection.

    Args:
        sheet_raw: Raw sheet data from get_sheet_content() or similar API call.
        header_row: 0-indexed row number containing column headers.
            If None, will auto-detect the header row.
        required_columns: Optional list of (col_name, keywords) tuples for
            keyword-based header detection. Same format as match_columns().
        model: Optional LLM model for fallback header detection.

    Returns:
        pandas DataFrame with the extracted data, or None if parsing fails.
    """
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            return None

        sheet_data = sheets[0].get('data', [{}])[0]
        rows = sheet_data.get('rowData', [])

        if not rows:
            return None

        # Auto-detect header row if not specified
        if header_row is None:
            header_row = detect_header_row(rows, required_columns=required_columns, model=model)
            print(f"Auto-detected header row: {header_row}")

        if len(rows) <= header_row + 1:
            return None

        # Get headers from header_row
        header_values = rows[header_row].get('values', [])
        headers = [v.get('formattedValue', f'col_{i}') for i, v in enumerate(header_values)]

        # Get data rows (after header row)
        data_rows = []
        for row_idx in range(header_row + 1, len(rows)):
            row = rows[row_idx]
            values = row.get('values', [])

            # Extract formatted values
            row_data = []
            for i in range(len(headers)):
                if i < len(values):
                    row_data.append(values[i].get('formattedValue', ''))
                else:
                    row_data.append('')

            # Skip empty rows
            if any(v for v in row_data):
                data_rows.append(row_data)

        if not data_rows:
            return None

        return pd.DataFrame(data_rows, columns=headers)

    except Exception as e:
        print(f"Error parsing sheet to DataFrame: {e}")
        return None


# =============================================================================
# OTHER SHEET FUNCTIONS
# =============================================================================

def search_sheet(filename: str, service, folder_id: str = None) -> Tuple[int, Optional[str]]:
    """Search for a Google Sheet by its filename.

    Args:
        filename (str): The name of the Google Sheet to search for.
        service: The Google Drive service instance.
        folder_id (str, optional): The ID of the Google Drive folder to search in.
            If None, searches in the entire Drive.

    Returns:
        A tuple (status, sheet_id) containing:
            - status (int): 0 if not found, 1 if found in any location, 2 if found in specified location.
            - sheet_id (str): The ID of the found Google Sheet, or None if not found.
    """
    # Import here to avoid circular imports
    from src.browsergym.knows.eval.eval_utils.google_services_utils import (
        find_doc_specified_location,
        find_file_any
    )

    sheet = None
    if folder_id:
        sheet = find_doc_specified_location(folder_id, filename, service)
    if sheet is None:
        sheet = find_file_any(filename, service, 'spreadsheet')
        if sheet is None:
            return 0, None
        return 1, sheet
    else:
        return 2, sheet


def extract_structure_from_sheet(sheet_id: str, service) -> List[dict]:
    """Returns ordered cell list with metadata for layout check.

    Elements like: {'row': r, 'col': c, 'value': v, 'format': {...}}

    Args:
        sheet_id (str): The ID of the Google Sheet to extract structure from.
        service: The Google Sheets service instance (not used, creates own).

    Returns:
        list: List of dictionaries containing cell data and metadata.
    """
    service = build('sheets', 'v4', credentials=authenticate(['SHEETS']))
    sheet_obj = service.spreadsheets().get(
        spreadsheetId=sheet_id, includeGridData=True).execute()
    structure = []
    for tab in sheet_obj.get('sheets', []):
        title = tab['properties']['title']
        rows = tab['data'][0].get('rowData', [])
        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row.get('values', [])):
                val = cell.get('formattedValue', '')
                fmt = cell.get('effectiveFormat', {})
                structure.append({
                    'sheet': title,
                    'row': r_idx,
                    'col': c_idx,
                    'value': val,
                    'format': fmt
                })
    return structure


def find_urls_in_sheet(
    sheet_rows: List[Dict],
    start_row: int,
    num_rows: int = 20,
    start_col: int = None,
    end_col: int = None
) -> List[str]:
    """
    Find URLs in cells starting from a specific row and within column bounds.

    Thoroughly checks all known URL storage locations in each cell:
    hyperlink, userEnteredValue.stringValue, userEnteredValue.formulaValue
    (=HYPERLINK), textFormatRuns, effectiveFormat.textFormat.link, and
    formattedValue.

    Args:
        sheet_rows: Raw rowData from sheet (from get_sheet_content or extract_sheet_data).
        start_row: Row index to start searching from (0-indexed).
        num_rows: Number of rows to search (default 20).
        start_col: Starting column index (0-indexed, inclusive). If None, starts from column 0.
        end_col: Ending column index (0-indexed, exclusive). If None, searches all columns.

    Returns:
        list: List of unique URLs found in the specified row/column range.
    """
    urls = []

    for row_idx in range(start_row, min(start_row + num_rows, len(sheet_rows))):
        row = sheet_rows[row_idx] if row_idx < len(sheet_rows) else {}
        values = row.get('values', [])

        col_start = start_col if start_col is not None else 0
        col_end = end_col if end_col is not None else len(values)

        for col_idx in range(col_start, min(col_end, len(values))):
            cell = values[col_idx] if col_idx < len(values) else {}
            if not cell:
                continue

            # 1. Explicit hyperlink property
            hyperlink = cell.get("hyperlink", "")
            if hyperlink and hyperlink.startswith(("http://", "https://")):
                urls.append(hyperlink.strip())
                continue

            user_entered = cell.get("userEnteredValue", {})

            # 2. String value (plain URL pasted into cell)
            sv = user_entered.get("stringValue", "")
            if sv and sv.strip().startswith(("http://", "https://")):
                urls.append(sv.strip())
                continue

            # 3. Formula (=HYPERLINK("url", "label"))
            formula = user_entered.get("formulaValue", "")
            if formula:
                m = re.search(r'HYPERLINK\s*\(\s*["\']([^"\']+)["\']', formula, re.IGNORECASE)
                if m:
                    urls.append(m.group(1))
                    continue

            # 4. textFormatRuns – links with display text (rich text)
            found_run = False
            for run in cell.get("textFormatRuns", []):
                uri = run.get("format", {}).get("link", {}).get("uri", "")
                if uri and uri.startswith(("http://", "https://")):
                    urls.append(uri.strip())
                    found_run = True
                    break
            if found_run:
                continue

            # 5. effectiveFormat.textFormat.link
            eff_uri = (
                cell.get("effectiveFormat", {})
                .get("textFormat", {})
                .get("link", {})
                .get("uri", "")
            )
            if eff_uri and eff_uri.startswith(("http://", "https://")):
                urls.append(eff_uri.strip())
                continue

            # 6. Formatted value fallback (regex to find URLs anywhere in text)
            fv = cell.get("formattedValue", "")
            if fv:
                m = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', fv)
                if m:
                    urls.append(m.group(0))

    return list(set(urls))  # Remove duplicates


def extract_charts_from_sheet(sheet_id: str, service) -> List[dict]:
    """Extracts all charts from a Google Sheets document.

    Args:
        sheet_id (str): The ID of the Google Sheets document to extract charts from.
        service: The Google Sheets service instance.

    Returns:
        list: A list of chart objects containing chart metadata and properties.
            Each chart object contains:
            - 'chart_id' (int): The unique ID of the chart
            - 'sheet_id' (int): The ID of the sheet containing the chart
            - 'sheet_name' (str): The name of the sheet containing the chart
            - 'chart_type' (str): The type of chart (e.g., 'COLUMN', 'PIE', 'LINE')
            - 'title' (str): The chart title
            - 'position' (dict): Chart position and size information
            - 'data_range' (dict): Information about the data range used by the chart
            - 'series' (list): List of data series in the chart
            - 'raw_chart' (dict): The complete raw chart specification from the API
            Returns empty list if no charts found or error occurs.
    """
    try:
        # Get sheet content with chart information
        sheet_obj = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        if not sheet_obj:
            print("Failed to fetch sheet content for chart extraction")
            return []

        charts = []
        chart_count = 0

        # Process each sheet/tab to find charts
        for sheet_data in sheet_obj.get('sheets', []):
            sheet_properties = sheet_data.get('properties', {})
            sheet_name = sheet_properties.get('title', 'Unknown')
            sheet_index = sheet_properties.get('sheetId', 0)

            # Get charts from this sheet
            sheet_charts = sheet_data.get('charts', [])

            for chart_data in sheet_charts:
                try:
                    chart_count += 1
                    chart_id = chart_data.get('chartId')
                    print(f"Processing chart {chart_count}: ID {chart_id} in sheet '{sheet_name}'")

                    # Extract chart specification
                    chart_spec = chart_data.get('spec', {})

                    # Get chart type and basic properties
                    chart_type = 'UNKNOWN'
                    chart_title = ''
                    data_range_info = {}
                    series_info = []

                    # Determine chart type and extract relevant information
                    if 'basicChart' in chart_spec:
                        basic_chart = chart_spec['basicChart']
                        chart_type = basic_chart.get('chartType', 'UNKNOWN')
                        chart_title = chart_spec.get('title', '')

                        # Extract data ranges and series
                        series_info = []
                        for series in basic_chart.get('series', []):
                            series_data = {
                                'type': series.get('type', 'UNKNOWN'),
                                'target_axis': series.get('targetAxis', 'LEFT_AXIS')
                            }

                            # Extract source range if available
                            if 'sourceRange' in series:
                                source_range = series['sourceRange']
                                series_data['source_range'] = {
                                    'sheet_id': source_range.get('sheetId'),
                                    'start_row': source_range.get('startRowIndex'),
                                    'end_row': source_range.get('endRowIndex'),
                                    'start_col': source_range.get('startColumnIndex'),
                                    'end_col': source_range.get('endColumnIndex')
                                }

                            series_info.append(series_data)

                        # Extract domain/category axis info
                        domain_axis = basic_chart.get('domains', [])
                        if domain_axis:
                            domain_range = domain_axis[0].get('domain', {}).get('sourceRange', {})
                            data_range_info['domain_range'] = {
                                'sheet_id': domain_range.get('sheetId'),
                                'start_row': domain_range.get('startRowIndex'),
                                'end_row': domain_range.get('endRowIndex'),
                                'start_col': domain_range.get('startColumnIndex'),
                                'end_col': domain_range.get('endColumnIndex')
                            }

                    elif 'pieChart' in chart_spec:
                        pie_chart = chart_spec['pieChart']
                        chart_type = 'PIE'
                        chart_title = chart_spec.get('title', '')

                        # Extract pie chart specific data
                        if 'domain' in pie_chart:
                            domain_range = pie_chart['domain'].get('sourceRange', {})
                            data_range_info['domain_range'] = {
                                'sheet_id': domain_range.get('sheetId'),
                                'start_row': domain_range.get('startRowIndex'),
                                'end_row': domain_range.get('endRowIndex'),
                                'start_col': domain_range.get('startColumnIndex'),
                                'end_col': domain_range.get('endColumnIndex')
                            }

                        if 'series' in pie_chart:
                            series_range = pie_chart['series'].get('sourceRange', {})
                            series_info = [{
                                'type': 'PIE_SERIES',
                                'source_range': {
                                    'sheet_id': series_range.get('sheetId'),
                                    'start_row': series_range.get('startRowIndex'),
                                    'end_row': series_range.get('endRowIndex'),
                                    'start_col': series_range.get('startColumnIndex'),
                                    'end_col': series_range.get('endColumnIndex')
                                }
                            }]

                    elif 'candlestickChart' in chart_spec:
                        chart_type = 'CANDLESTICK'
                        chart_title = chart_spec.get('title', '')

                    elif 'orgChart' in chart_spec:
                        chart_type = 'ORG_CHART'
                        chart_title = chart_spec.get('title', '')

                    elif 'histogramChart' in chart_spec:
                        chart_type = 'HISTOGRAM'
                        chart_title = chart_spec.get('title', '')

                    # Extract position information
                    position_info = {}
                    if 'position' in chart_data:
                        position = chart_data['position']
                        if 'overlayPosition' in position:
                            overlay = position['overlayPosition']
                            position_info = {
                                'type': 'overlay',
                                'anchor_cell': {
                                    'sheet_id': overlay.get('anchorCell', {}).get('sheetId'),
                                    'row': overlay.get('anchorCell', {}).get('rowIndex'),
                                    'col': overlay.get('anchorCell', {}).get('columnIndex')
                                },
                                'offset_x': overlay.get('offsetXPixels', 0),
                                'offset_y': overlay.get('offsetYPixels', 0),
                                'width': overlay.get('widthPixels', 0),
                                'height': overlay.get('heightPixels', 0)
                            }
                        elif 'newSheet' in position:
                            position_info = {'type': 'new_sheet'}

                    # Create chart object
                    chart_obj = {
                        'chart_id': chart_id,
                        'sheet_id': sheet_index,
                        'sheet_name': sheet_name,
                        'chart_type': chart_type,
                        'title': chart_title,
                        'position': position_info,
                        'data_range': data_range_info,
                        'series': series_info,
                        'raw_chart': chart_data  # Include full raw data for advanced analysis
                    }

                    charts.append(chart_obj)
                    print(f"  Extracted {chart_type} chart: '{chart_title}'")

                except Exception as chart_error:
                    print(f"  Error processing chart {chart_id}: {chart_error}")
                    continue

        print(f"Chart extraction complete. Found {len(charts)} charts across {len(sheet_obj.get('sheets', []))} sheets.")
        return charts

    except Exception as e:
        print(f"Error extracting charts from sheet: {e}")
        return []
