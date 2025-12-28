"""Evaluator for the Ski Tour Plan Google Sheets task.

This evaluator validates a spreadsheet containing ski run information:
- 3 ski runs with slope angle <= 26 degrees from Wasatch Backcountry Ski Guide
- Avalanche forecast data from Utah Avalanche Center
- Proper cell coloring based on danger ratings
"""

import os
import sys
import time
import re
import argparse
from typing import List

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.google_sheets_utils import (
    extract_tables_from_sheet,
    extract_sheet_data,
    parse_sheet_to_dataframe,
    get_sheet_content,
)
from src.browsergym.eval.eval_utils.table_utils import (
    get_image_url_from_raw_sheet_cell,
    get_cell_value,
    get_cell_background_color,
    check_merged_cells,
    match_columns,
)
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.image_utils import match_image_tiered
import tempfile
import requests

# Local imports
from src.browsergym.eval.tasks.sheets_25_skitourplan.utils import (
    parse_slope_angle,
    parse_gps_coordinates,
    parse_typical_vertical,
    normalize_aspect,
    is_valid_wbsguide_url,
    is_valid_uac_forecast_url,
    classify_danger_color,
    load_gold_runs,
    find_run_by_name_or_url,
    get_valid_runs,
    gps_coordinates_match,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/sheets_25_skitourplan/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Model configuration
model = None
model_id = "gemini-2.5-flash-google-ai"

# Initialize Google services
DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global state
sheet_id = None
table_data = None
sheet_raw = None
df = None
matched_columns = {}
gold_data = None

# Browsing history (passed from grade_checkpoints)
BROWSING_HISTORY = None


def setup(workspace_doc_id: str):
    """Setup function to initialize the evaluator.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, table_data, sheet_raw, df, gold_data

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    # Load gold data
    gold_data = load_gold_runs()
    if gold_data:
        valid_runs = get_valid_runs(gold_data)
        print(f"Loaded {len(valid_runs)} valid runs from gold data")
    else:
        print("WARNING: Could not load gold data")

    # Get raw sheet content
    sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

    # Try to extract using formal table structure first
    table_data = extract_tables_from_sheet(sheet_id, SHEETS_SERVICE)

    if table_data:
        # Use extracted table
        first_table = table_data[0]
        df = first_table.df if hasattr(first_table, 'df') else first_table
        print(f"Extracted table with {len(df)} rows and {len(df.columns)} columns (using table API)")
    else:
        # Fall back to raw sheet parsing
        # The spreadsheet has a title row (row 0) and headers in row 1
        df = parse_sheet_to_dataframe(sheet_raw, header_row=1)
        if df is not None:
            print(f"Extracted table with {len(df)} rows and {len(df.columns)} columns (using raw parsing)")
        else:
            print("WARNING: Could not extract table data from spreadsheet")

    if df is not None:
        print(f"Columns: {list(df.columns)}")


def grade_checkpoint_1():
    """Checkpoint 1: Spreadsheet Structure (10 pts).

    Validates that the spreadsheet has correct column headers:
    1. Run Name Column
    2. Run Link Column
    3. Starting Location Column
    4. GPS Coordinates Column
    5. Elevation Column (optional based on checkpoints.md)
    6. Typical Vertical Column
    7. Slope Aspect Column
    8. Slope Angle Column
    9. Forecast Date Column
    10. Forecast Link Column

    Uses keyword matching first, then falls back to VLM if needed.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Spreadsheet Structure")

    if df is None or df.empty:
        checkpoint.add_step("Table Data Extraction", False, 1,
                          "No table data found in spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Required columns with keywords for matching
    # Order matters - more specific keywords first to avoid false matches
    required_columns = [
        ("Run Name", ["run name"]),
        ("Run Link", ["run link"]),
        ("Starting Location", ["starting location", "starting", "trailhead"]),
        ("GPS Coordinates", ["gps", "coordinates", "run location"]),
        ("Elevation", ["elevation"]),
        ("Typical Vertical", ["typical vertical", "vertical"]),
        ("Slope Aspect", ["slope aspect", "aspect"]),
        ("Slope Angle", ["slope angle", "angle"]),
        ("Forecast Date", ["forecast date"]),
        ("Forecast Link", ["forecast link"]),
    ]

    original_columns = [str(col) for col in df.columns]

    # Use standardized match_columns() - keyword matching first, then LLM fallback
    vlm_model = load_model(model_id)
    name_matches = match_columns(df, required_columns, model=vlm_model, parallel=True)

    # Convert column names to indices (this evaluator uses indices for .iloc access)
    for col_name, matched_col_name in name_matches.items():
        try:
            matched_columns[col_name] = original_columns.index(matched_col_name)
        except ValueError:
            pass  # Column name not found

    # Record results for all columns
    for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
        step_start = time.time()

        if col_name in matched_columns:
            idx = matched_columns[col_name]
            matched_column = original_columns[idx]
            checkpoint.add_step(f"{col_name} Column", True, step_num,
                              f"Found column: '{matched_column}'",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"{col_name} Column", False, step_num,
                              f"No column found for '{col_name}'. Available: {', '.join(original_columns[:5])}...",
                              execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2: Run Selection Criteria (4 pts).

    Validates:
    1. Run Count - exactly 3 ski runs
    2-4. Slope Angle Compliance for each run (<= 26 degrees)
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Run Selection Criteria")

    if df is None or df.empty:
        checkpoint.add_step("Run Count", False, 1,
                          "No data in user's spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Count data rows (excluding header - df already excludes header)
    run_count = len(df)

    # Step 1: Check run count
    step_start = time.time()
    run_count_valid = run_count == 3
    checkpoint.add_step("Run Count", run_count_valid, 1,
                      f"Found {run_count} runs (expected 3)",
                      execution_time=time.time() - step_start)

    # Get slope angle column index
    angle_col_idx = matched_columns.get('Slope Angle')

    # Steps 2-4: Check slope angle for each run
    for i in range(3):
        step_start = time.time()

        if i >= len(df):
            checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                              f"Run {i+1} not found",
                              execution_time=time.time() - step_start)
            continue

        row = df.iloc[i]

        if angle_col_idx is not None:
            angle_str = str(row.iloc[angle_col_idx])
            angle = parse_slope_angle(angle_str)

            if angle is not None and angle <= 26:
                checkpoint.add_step(f"Slope Angle Run {i+1}", True, i + 2,
                                  f"Angle: {angle}° (max 26°)",
                                  execution_time=time.time() - step_start)
            elif angle is not None:
                checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                                  f"Angle: {angle}° exceeds 26° limit",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                                  f"Could not parse angle: '{angle_str}'",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                              "Slope Angle column not found",
                              execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3: Run Data Accuracy (21 pts).

    For each of 3 runs, validates 7 fields:
    1. Run Name Valid
    2. Run Link Valid
    3. Starting Location Correct
    4. GPS Coordinates Correct
    5. Typical Vertical Correct
    6. Slope Aspect Correct
    7. Slope Angle Correct
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global matched_columns, df, gold_data
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=21, result=0, name="Run Data Accuracy")

    if df is None or df.empty or not gold_data:
        checkpoint.add_step("Data Validation", False, 1,
                          "No data available for validation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get column indices
    name_col = matched_columns.get('Run Name')
    link_col = matched_columns.get('Run Link')
    start_col = matched_columns.get('Starting Location')
    gps_col = matched_columns.get('GPS Coordinates')
    vert_col = matched_columns.get('Typical Vertical')
    aspect_col = matched_columns.get('Slope Aspect')
    angle_col = matched_columns.get('Slope Angle')

    step_num = 1

    for run_idx in range(min(3, len(df))):
        row = df.iloc[run_idx]
        run_num = run_idx + 1

        # Get user values
        user_name = str(row.iloc[name_col]) if name_col is not None else ""
        user_link = str(row.iloc[link_col]) if link_col is not None else ""

        # Find matching gold run
        gold_run = find_run_by_name_or_url(user_name, user_link, gold_data)

        # Step 1: Run Name Valid
        step_start = time.time()
        if gold_run:
            checkpoint.add_step(f"Run {run_num} - Name Valid", True, step_num,
                              f"'{user_name}' found in gold data",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Name Valid", False, step_num,
                              f"'{user_name}' not found in gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 2: Run Link Valid
        step_start = time.time()
        link_valid = is_valid_wbsguide_url(user_link)
        if link_valid:
            checkpoint.add_step(f"Run {run_num} - Link Valid", True, step_num,
                              f"Valid WBSGuide URL: {user_link}",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Link Valid", False, step_num,
                              f"Invalid URL: {user_link}",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 3: Starting Location Correct
        step_start = time.time()
        if start_col is not None and gold_run:
            user_start = str(row.iloc[start_col])
            gold_start = gold_run.get('starting_location', '')

            if gold_start is None:
                # No gold data to compare
                checkpoint.add_step(f"Run {run_num} - Starting Location", True, step_num,
                                  f"No gold data for comparison",
                                  execution_time=time.time() - step_start)
            else:
                # Normalize for comparison (remove apostrophes, lowercase)
                user_norm = user_start.lower().strip().replace("'", "").replace("'", "")
                gold_norm = gold_start.lower().strip().replace("'", "").replace("'", "")

                if user_norm == gold_norm or user_norm in gold_norm or gold_norm in user_norm:
                    checkpoint.add_step(f"Run {run_num} - Starting Location", True, step_num,
                                      f"'{user_start}' matches gold '{gold_start}'",
                                      execution_time=time.time() - step_start)
                else:
                    checkpoint.add_step(f"Run {run_num} - Starting Location", False, step_num,
                                      f"'{user_start}' != '{gold_start}'",
                                      execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Starting Location", False, step_num,
                              "Column not found or no gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 4: GPS Coordinates Correct
        step_start = time.time()
        if gps_col is not None and gold_run:
            user_gps_str = str(row.iloc[gps_col])
            user_coords = parse_gps_coordinates(user_gps_str)
            gold_lat = gold_run.get('gps_lat')
            gold_lon = gold_run.get('gps_lon')

            if user_coords and gold_lat and gold_lon:
                if gps_coordinates_match(user_coords, gold_lat, gold_lon, tolerance=0.01):
                    checkpoint.add_step(f"Run {run_num} - GPS Coordinates", True, step_num,
                                      f"Coordinates match within tolerance",
                                      execution_time=time.time() - step_start)
                else:
                    checkpoint.add_step(f"Run {run_num} - GPS Coordinates", False, step_num,
                                      f"User: {user_coords}, Gold: ({gold_lat}, {gold_lon})",
                                      execution_time=time.time() - step_start)
            elif user_coords:
                checkpoint.add_step(f"Run {run_num} - GPS Coordinates", True, step_num,
                                  f"User coords: {user_coords} (no gold to compare)",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"Run {run_num} - GPS Coordinates", False, step_num,
                                  f"Could not parse: '{user_gps_str}'",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - GPS Coordinates", False, step_num,
                              "Column not found or no gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 5: Typical Vertical Correct
        # Gold data may have a range (typical_vertical_min to typical_vertical)
        step_start = time.time()
        if vert_col is not None and gold_run:
            user_vert_str = str(row.iloc[vert_col])
            user_vert = parse_typical_vertical(user_vert_str)
            gold_vert_max = gold_run.get('typical_vertical')
            gold_vert_min = gold_run.get('typical_vertical_min', gold_vert_max)

            if user_vert and gold_vert_max:
                # Accept values within the range (with some tolerance at edges)
                tolerance = 50  # Allow 50 ft tolerance
                is_valid = (gold_vert_min - tolerance) <= user_vert <= (gold_vert_max + tolerance)

                if is_valid:
                    if gold_vert_min != gold_vert_max:
                        checkpoint.add_step(f"Run {run_num} - Typical Vertical", True, step_num,
                                          f"{user_vert} ft (gold range: {gold_vert_min}-{gold_vert_max} ft)",
                                          execution_time=time.time() - step_start)
                    else:
                        checkpoint.add_step(f"Run {run_num} - Typical Vertical", True, step_num,
                                          f"{user_vert} ft (gold: {gold_vert_max} ft)",
                                          execution_time=time.time() - step_start)
                else:
                    if gold_vert_min != gold_vert_max:
                        checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                          f"{user_vert} ft not in range {gold_vert_min}-{gold_vert_max} ft",
                                          execution_time=time.time() - step_start)
                    else:
                        checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                          f"{user_vert} ft != {gold_vert_max} ft",
                                          execution_time=time.time() - step_start)
            elif user_vert:
                checkpoint.add_step(f"Run {run_num} - Typical Vertical", True, step_num,
                                  f"{user_vert} ft (no gold to compare)",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                  f"Could not parse: '{user_vert_str}'",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                              "Column not found or no gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 6: Slope Aspect Correct
        step_start = time.time()
        if aspect_col is not None and gold_run:
            user_aspect_str = str(row.iloc[aspect_col])
            user_aspect = normalize_aspect(user_aspect_str)
            gold_aspect = gold_run.get('slope_aspect')

            if user_aspect and gold_aspect and user_aspect == gold_aspect:
                checkpoint.add_step(f"Run {run_num} - Slope Aspect", True, step_num,
                                  f"Aspect: {user_aspect}",
                                  execution_time=time.time() - step_start)
            elif user_aspect:
                checkpoint.add_step(f"Run {run_num} - Slope Aspect", False, step_num,
                                  f"User: {user_aspect}, Gold: {gold_aspect}",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"Run {run_num} - Slope Aspect", False, step_num,
                                  f"Could not parse: '{user_aspect_str}'",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Slope Aspect", False, step_num,
                              "Column not found or no gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 7: Slope Angle Correct
        step_start = time.time()
        if angle_col is not None and gold_run:
            user_angle_str = str(row.iloc[angle_col])
            user_angle = parse_slope_angle(user_angle_str)
            gold_angle = gold_run.get('slope_angle')

            if user_angle and gold_angle and user_angle == gold_angle and user_angle <= 26:
                checkpoint.add_step(f"Run {run_num} - Slope Angle", True, step_num,
                                  f"Angle: {user_angle}° (valid <= 26°)",
                                  execution_time=time.time() - step_start)
            elif user_angle and user_angle <= 26:
                checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                  f"User: {user_angle}°, Gold: {gold_angle}°",
                                  execution_time=time.time() - step_start)
            elif user_angle:
                checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                  f"Angle {user_angle}° exceeds 26° limit",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                  f"Could not parse: '{user_angle_str}'",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                              "Column not found or no gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """Checkpoint 4: Website Visit Validation (4 pts).

    Validates browsing history:
    1. Wasatch Guide Visited (wbsguide.com)
    2. Utah Avalanche Center Visited (utahavalanchecenter.org)
    3. Run Links Visited (at least one run link)
    4. Forecast Page Visited
    """
    print("----------------- CHECKPOINT 4 ----------------")
    global BROWSING_HISTORY, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Website Visit Validation")

    if not BROWSING_HISTORY:
        for i, name in enumerate(["Wasatch Guide Visited", "Utah Avalanche Center Visited",
                                   "Run Links Visited", "Forecast Page Visited"], start=1):
            checkpoint.add_step(name, False, i,
                              "No browsing history provided",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    history_lower = [url.lower() for url in BROWSING_HISTORY]

    # Step 1: Check wbsguide.com
    step_start = time.time()
    wbs_visited = any('wbsguide.com' in url for url in history_lower)
    checkpoint.add_step("Wasatch Guide Visited", wbs_visited, 1,
                      f"wbsguide.com {'found' if wbs_visited else 'not found'} in history",
                      execution_time=time.time() - step_start)

    # Step 2: Check utahavalanchecenter.org
    step_start = time.time()
    uac_visited = any('utahavalanchecenter.org' in url for url in history_lower)
    checkpoint.add_step("Utah Avalanche Center Visited", uac_visited, 2,
                      f"utahavalanchecenter.org {'found' if uac_visited else 'not found'} in history",
                      execution_time=time.time() - step_start)

    # Step 3: Check run links
    step_start = time.time()
    link_col = matched_columns.get('Run Link')
    run_link_visited = False

    if link_col is not None and df is not None:
        for i in range(min(3, len(df))):
            user_link = str(df.iloc[i].iloc[link_col]).lower()
            if any(user_link in url or url in user_link for url in history_lower):
                run_link_visited = True
                break

    checkpoint.add_step("Run Links Visited", run_link_visited, 3,
                      f"Run link {'found' if run_link_visited else 'not found'} in history",
                      execution_time=time.time() - step_start)

    # Step 4: Check forecast page
    step_start = time.time()
    forecast_visited = any('utahavalanchecenter.org/forecast' in url for url in history_lower)
    checkpoint.add_step("Forecast Page Visited", forecast_visited, 4,
                      f"Forecast page {'found' if forecast_visited else 'not found'} in history",
                      execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """Checkpoint 5: Avalanche Forecast Data (5 pts).

    Validates:
    1. Forecast Date Correct (02/05/2025)
    2. Forecast Link Valid
    3. Merged Cells (forecast columns merged across run rows)
    4. Danger Rose Screenshot Present
    5. Danger Rose Image Valid (VLM validation)
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global matched_columns, gold_data
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Avalanche Forecast Data")

    forecast_date_col = matched_columns.get('Forecast Date')
    forecast_link_col = matched_columns.get('Forecast Link')

    # Raw sheet has: row 0 = title, row 1 = headers, row 2+ = data
    # So first data row is raw row 2
    HEADER_ROW = 1  # 0-indexed header row in raw sheet
    FIRST_DATA_ROW = 2  # 0-indexed first data row in raw sheet

    # Step 1: Forecast Date Correct
    step_start = time.time()
    if forecast_date_col is not None:
        # Check first data row (row 2 in raw sheet, after title and header)
        forecast_date = get_cell_value(sheet_raw, FIRST_DATA_ROW, forecast_date_col)
        date_valid = "02/05/2025" in forecast_date or "2/5/2025" in forecast_date
        checkpoint.add_step("Forecast Date Correct", date_valid, 1,
                          f"Date: '{forecast_date}' (expected 02/05/2025)",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Forecast Date Correct", False, 1,
                          "Forecast Date column not found",
                          execution_time=time.time() - step_start)

    # Step 2: Forecast Link Valid
    step_start = time.time()
    if forecast_link_col is not None:
        forecast_link = get_cell_value(sheet_raw, FIRST_DATA_ROW, forecast_link_col)
        link_valid = is_valid_uac_forecast_url(forecast_link)
        checkpoint.add_step("Forecast Link Valid", link_valid, 2,
                          f"Link: '{forecast_link[:50]}...'",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Forecast Link Valid", False, 2,
                          "Forecast Link column not found",
                          execution_time=time.time() - step_start)

    # Step 3: Merged Cells
    step_start = time.time()
    # Check if forecast date, link, and danger rose columns are merged across rows 2-4 (data rows)
    cols_to_check = []
    if forecast_date_col is not None:
        cols_to_check.append(forecast_date_col)
    if forecast_link_col is not None:
        cols_to_check.append(forecast_link_col)

    if cols_to_check:
        merged = check_merged_cells(sheet_raw, cols_to_check, FIRST_DATA_ROW, FIRST_DATA_ROW + 2)  # Rows 2-4 (data rows)
        checkpoint.add_step("Merged Cells", merged, 3,
                          f"Columns {cols_to_check} {'are' if merged else 'are not'} merged vertically",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Merged Cells", False, 3,
                          "Required columns not found for merge check",
                          execution_time=time.time() - step_start)

    # Step 4: Danger Rose Screenshot Present
    step_start = time.time()
    # Find danger rose column (could be after forecast link)
    danger_rose_col = None
    if df is not None:
        for i, col in enumerate(df.columns):
            col_lower = str(col).lower()
            if 'rose' in col_lower or ('danger' in col_lower and 'color' not in col_lower):
                danger_rose_col = i
                break

    image_url = None
    if danger_rose_col is not None:
        image_url = get_image_url_from_raw_sheet_cell(sheet_raw, FIRST_DATA_ROW, danger_rose_col)

    image_present = image_url is not None and image_url.startswith('http')
    checkpoint.add_step("Danger Rose Present", image_present, 4,
                      f"Image URL: {image_url[:50] + '...' if image_url else 'not found'}",
                      execution_time=time.time() - step_start)

    # Step 5: Danger Rose Image Valid (tiered image comparison)
    # Compare user's image to the gold danger rose image from the forecast
    step_start = time.time()
    user_image_path = None
    gold_image_path = None

    if image_present and gold_data:
        try:
            # Get gold danger rose image URL from forecast data
            gold_image_url = gold_data.get('forecast', {}).get('danger_rose_image_url')

            if not gold_image_url:
                checkpoint.add_step("Danger Rose Valid", False, 5,
                                  "Gold danger rose image URL not configured",
                                  execution_time=time.time() - step_start)
            else:
                # Download user's image
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                user_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                user_temp.write(response.content)
                user_temp.close()
                user_image_path = user_temp.name

                # Download gold image
                gold_response = requests.get(gold_image_url, timeout=30)
                gold_response.raise_for_status()
                gold_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                gold_temp.write(gold_response.content)
                gold_temp.close()
                gold_image_path = gold_temp.name

                # Use tiered matching: exact -> perceptual hash -> VLM
                vlm_model = load_model(model_id)
                match_result, match_method = match_image_tiered(
                    user_image_path,
                    gold_image_path,
                    model=vlm_model,
                    hash_threshold=15  # Slightly more lenient for web images
                )

                checkpoint.add_step("Danger Rose Valid", match_result, 5,
                                  f"Match: {match_result} (method: {match_method})",
                                  execution_time=time.time() - step_start)
        except Exception as e:
            checkpoint.add_step("Danger Rose Valid", False, 5,
                              f"Image comparison failed: {str(e)[:50]}",
                              execution_time=time.time() - step_start)
        finally:
            # Clean up temp files
            for path in [user_image_path, gold_image_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
    else:
        checkpoint.add_step("Danger Rose Valid", False, 5,
                          "No image to validate or missing gold data",
                          execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """Checkpoint 6: Danger Rating Cell Coloring (6 pts).

    For each of 3 runs:
    1. Danger Rating Determined - correct based on aspect
    2. Cell Color Applied - run name cell has correct background color
    """
    print("----------------- CHECKPOINT 6 ----------------")
    global matched_columns, df, gold_data
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Danger Rating Cell Coloring")

    if df is None or df.empty or not gold_data:
        checkpoint.add_step("Cell Coloring", False, 1,
                          "No data available for validation",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    name_col = matched_columns.get('Run Name')
    link_col = matched_columns.get('Run Link')

    # Raw sheet has: row 0 = title, row 1 = headers, row 2+ = data
    FIRST_DATA_ROW = 2  # 0-indexed first data row in raw sheet

    step_num = 1

    for run_idx in range(min(3, len(df))):
        row = df.iloc[run_idx]
        run_num = run_idx + 1
        sheet_row_idx = FIRST_DATA_ROW + run_idx  # Row index in raw sheet

        # Get user values
        user_name = str(row.iloc[name_col]) if name_col is not None else ""
        user_link = str(row.iloc[link_col]) if link_col is not None else ""

        # Find matching gold run
        gold_run = find_run_by_name_or_url(user_name, user_link, gold_data)

        # Step 1: Danger Rating Determined
        step_start = time.time()
        expected_rating = gold_run.get('expected_danger_rating', 'unknown') if gold_run else 'unknown'

        # Find danger color column if it exists
        danger_col = None
        user_rating = None
        for i, col in enumerate(df.columns):
            col_lower = str(col).lower()
            if 'danger' in col_lower and 'color' in col_lower:
                danger_col = i
                user_rating = str(row.iloc[danger_col]).lower().strip()
                break

        if expected_rating != 'unknown':
            rating_correct = user_rating == expected_rating if user_rating else False
            checkpoint.add_step(f"Run {run_num} - Rating Determined", rating_correct, step_num,
                              f"User: {user_rating}, Expected: {expected_rating}",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Rating Determined", False, step_num,
                              "Could not determine expected rating from gold data",
                              execution_time=time.time() - step_start)
        step_num += 1

        # Step 2: Cell Color Applied
        step_start = time.time()
        if name_col is not None:
            bg_color = get_cell_background_color(sheet_raw, sheet_row_idx, name_col)
            actual_color = classify_danger_color(bg_color)

            # Expected color should match the expected rating
            if expected_rating != 'unknown':
                color_matches = actual_color == expected_rating
                checkpoint.add_step(f"Run {run_num} - Cell Colored", color_matches, step_num,
                                  f"Cell color: {actual_color}, Expected: {expected_rating}",
                                  execution_time=time.time() - step_start)
            else:
                # If we don't know expected, just check if any danger color is applied
                is_colored = actual_color in ['green', 'yellow', 'orange', 'red', 'black']
                checkpoint.add_step(f"Run {run_num} - Cell Colored", is_colored, step_num,
                                  f"Cell color: {actual_color}",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Run {run_num} - Cell Colored", False, step_num,
                              "Run Name column not found",
                              execution_time=time.time() - step_start)
        step_num += 1

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None,
                      browsing_history: List[str] = None) -> Result:
    """Grade all checkpoints for the ski tour plan task.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
        browsing_history: List of URLs visited during task execution.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    global BROWSING_HISTORY

    total_start_time = time.time()

    # Set browsing history for use in checkpoint 4
    BROWSING_HISTORY = browsing_history or []

    try:
        # Setup document processing
        setup(workspace_doc_id)

        checkpoints: List[Checkpoint] = []

        # Grade each checkpoint
        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())
        checkpoints.append(grade_checkpoint_5())
        checkpoints.append(grade_checkpoint_6())

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

        # Return a failed result
        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ski tour plan spreadsheet")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Sheets document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()
    print(f"DEBUG mode: {DEBUG}")
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history
    )

    print("\n=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "✓" if step["success"] else "✗"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
