"""Evaluator for the Ski Tour Plan Google Sheets task (Instance 4).

This evaluator validates a spreadsheet containing ski run information:
- 5 ski runs with slope angle <= 25 degrees from Wasatch Backcountry Ski Guide
- Avalanche forecast data from Utah Avalanche Center for 01/22/2025
- Proper cell coloring based on danger ratings
"""

import os
import sys
import time
import re
import argparse
import traceback
from typing import List

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    extract_tables_from_sheet,
    extract_sheet_data,
    parse_sheet_to_dataframe,
    get_sheet_content,
)
from src.browsergym.knows.eval.eval_utils.table_utils import (
    get_image_url_from_raw_sheet_cell,
    get_cell_value,
    get_cell_background_color,
    check_merged_cells,
    match_columns,
)
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.image_utils import match_image_tiered
import tempfile
import requests

# Local imports
from src.browsergym.knows.eval.tasks.sheets_25_skitourplan.utils import (
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
    gps_coordinates_match_with_fallback,
)

# Constants
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Instance-specific parameters
FORECAST_DATE = "01/22/2025"
FORECAST_DATE_ALT = "1/22/2025"
MAX_SLOPE_ANGLE = 25
EXPECTED_RUN_COUNT = 5

# Required columns with keywords for matching (used for header detection and column matching)
REQUIRED_COLUMNS = [
    ("Run Name", ["run name"]),
    ("Run Link", ["run link"]),
    ("Starting Location", ["starting location", "starting", "trailhead"]),
    ("GPS Coordinates", ["gps", "coordinates", "run location"]),
    ("Typical Vertical", ["typical vertical", "vertical"]),
    ("Slope Aspect", ["slope aspect", "aspect"]),
    ("Slope Angle", ["slope angle", "angle"]),
    ("Forecast Date", ["forecast date"]),
    ("Forecast Link", ["forecast link"]),
]

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
detected_header_row = 0

# Browsing history (passed from grade_checkpoints)
BROWSING_HISTORY = None


def setup(workspace_doc_id: str):
    """Setup function to initialize the evaluator.

    Every external call is isolated so a single failure leaves the relevant
    global as None and lets each checkpoint emit its full step list.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, table_data, sheet_raw, df, gold_data, matched_columns, detected_header_row

    sheet_raw = None
    table_data = None
    df = None
    gold_data = None
    matched_columns = {}

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    try:
        gold_data = load_gold_runs(data_dir=DATA_DIR)
    except Exception as e:
        print(f"WARNING: load_gold_runs failed: {e}")

    if gold_data:
        print(f"Loaded gold data from {DATA_DIR}")
    else:
        print("WARNING: Could not load gold data")

    try:
        sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)
    except Exception as e:
        print(f"WARNING: get_sheet_content failed: {e}")

    try:
        table_data = extract_tables_from_sheet(sheet_id, SHEETS_SERVICE)
    except Exception as e:
        print(f"WARNING: extract_tables_from_sheet failed: {e}")

    if table_data:
        try:
            first_table = table_data[0]
            df = first_table.df if hasattr(first_table, 'df') else first_table
            print(f"Extracted table with {len(df)} rows and {len(df.columns)} columns (using table API)")
        except Exception as e:
            print(f"WARNING: failed to use extracted table: {e}")
            df = None

    if df is None and sheet_raw is not None:
        try:
            from src.browsergym.knows.eval.eval_utils.google_sheets_utils import detect_header_row
            rows = sheet_raw.get('sheets', [{}])[0].get('data', [{}])[0].get('rowData', [])
            detected_header_row = detect_header_row(rows, required_columns=REQUIRED_COLUMNS)
            df = parse_sheet_to_dataframe(sheet_raw, header_row=detected_header_row)
        except Exception as e:
            print(f"WARNING: parse_sheet_to_dataframe failed: {e}")
            df = None
        if df is not None:
            print(f"Extracted table with {len(df)} rows and {len(df.columns)} columns (using raw parsing)")

    if df is None:
        print("WARNING: Could not extract table data from spreadsheet")
    else:
        print(f"Columns: {list(df.columns)}")


def grade_checkpoint_1():
    """Checkpoint 1: Spreadsheet Structure (9 pts).

    Validates that the spreadsheet has correct column headers.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global matched_columns, df, model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=9, result=0, name="Spreadsheet Structure")
    expected_steps = [f"{c} Column" for c in (
        "Run Name", "Run Link", "Starting Location", "GPS Coordinates",
        "Typical Vertical", "Slope Aspect", "Slope Angle",
        "Forecast Date", "Forecast Link",
    )]

    try:
        if df is None or df.empty:
            reason = "No table data found in spreadsheet"
            for i, step_name in enumerate(expected_steps, start=1):
                checkpoint.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        required_columns = REQUIRED_COLUMNS

        original_columns = [str(col) for col in df.columns]

        if model is None:
            model = load_model(model_id)
        name_matches, col_methods = match_columns(df, required_columns, model=model, parallel=True,
                                      context="a ski tour planning spreadsheet with columns for run names, links to Wasatch Backcountry Ski Guide, starting locations, GPS coordinates, elevation, typical vertical, slope aspect, slope angle, avalanche forecast date, and forecast link",
                                      return_methods=True)

        for col_name, matched_col_name in name_matches.items():
            try:
                matched_columns[col_name] = original_columns.index(matched_col_name)
            except ValueError:
                pass

        for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
            step_start = time.time()

            if col_name in matched_columns:
                idx = matched_columns[col_name]
                matched_column = original_columns[idx]
                checkpoint.add_step(f"{col_name} Column", True, step_num,
                                  f"Found column: '{matched_column}'",
                                  execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC if col_methods.get(col_name) == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)
            else:
                checkpoint.add_step(f"{col_name} Column", False, step_num,
                                  f"No column found for '{col_name}'. Available: {', '.join(original_columns[:5])}...",
                                  execution_time=time.time() - step_start, category=StepCategory.LLM_VLM_JUDGEMENT)

        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    except Exception as e:
        traceback.print_exc()
        failed = Checkpoint(total=9, result=0, name="Spreadsheet Structure")
        reason = f"Checkpoint raised: {str(e)[:100]}"
        for i, step_name in enumerate(expected_steps, start=1):
            failed.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
        failed.execution_time = time.time() - checkpoint_start
        return failed


def grade_checkpoint_2():
    """Checkpoint 2: Run Selection Criteria (6 pts).

    Validates:
    1. Run Count - exactly 5 ski runs
    2-6. Slope Angle Compliance for each run (<= 25 degrees)
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Run Selection Criteria")
    expected_steps = ["Run Count"] + [f"Slope Angle Run {i+1}" for i in range(EXPECTED_RUN_COUNT)]

    try:
        if df is None or df.empty:
            reason = "No data in user's spreadsheet"
            for i, step_name in enumerate(expected_steps, start=1):
                checkpoint.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        run_count = len(df)
        step_start = time.time()
        run_count_valid = run_count == EXPECTED_RUN_COUNT
        checkpoint.add_step("Run Count", run_count_valid, 1,
                          f"Found {run_count} runs (expected {EXPECTED_RUN_COUNT})",
                          execution_time=time.time() - step_start, category=StepCategory.STRUCTURAL)

        angle_col_idx = matched_columns.get('Slope Angle')

        for i in range(EXPECTED_RUN_COUNT):
            step_start = time.time()

            if i >= len(df):
                checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                                  f"Run {i+1} not found",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
                continue

            row = df.iloc[i]

            if angle_col_idx is not None:
                angle_str = str(row.iloc[angle_col_idx])
                angle = parse_slope_angle(angle_str)

                if angle is not None and angle <= MAX_SLOPE_ANGLE:
                    checkpoint.add_step(f"Slope Angle Run {i+1}", True, i + 2,
                                      f"Angle: {angle} (max {MAX_SLOPE_ANGLE})",
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                elif angle is not None:
                    checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                                      f"Angle: {angle} exceeds {MAX_SLOPE_ANGLE} limit",
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                else:
                    checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                                      f"Could not parse angle: '{angle_str}'",
                                      execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
            else:
                checkpoint.add_step(f"Slope Angle Run {i+1}", False, i + 2,
                                  "Slope Angle column not found",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)

        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    except Exception as e:
        traceback.print_exc()
        failed = Checkpoint(total=6, result=0, name="Run Selection Criteria")
        reason = f"Checkpoint raised: {str(e)[:100]}"
        for i, step_name in enumerate(expected_steps, start=1):
            failed.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
        failed.execution_time = time.time() - checkpoint_start
        return failed


def grade_checkpoint_3():
    """Checkpoint 3: Run Data Accuracy (35 pts).

    For each of 5 runs, validates 7 fields.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global matched_columns, df, gold_data
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=35, result=0, name="Run Data Accuracy")
    sub_fields = ("Name Valid", "Link Valid", "Starting Location",
                  "GPS Coordinates", "Typical Vertical", "Slope Aspect",
                  "Slope Angle")
    expected_steps = [f"Run {r} - {f}"
                      for r in range(1, EXPECTED_RUN_COUNT + 1)
                      for f in sub_fields]

    try:
        if df is None or df.empty or not gold_data:
            reason = "No data available for validation"
            for i, step_name in enumerate(expected_steps, start=1):
                checkpoint.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        name_col = matched_columns.get('Run Name')
        link_col = matched_columns.get('Run Link')
        start_col = matched_columns.get('Starting Location')
        gps_col = matched_columns.get('GPS Coordinates')
        vert_col = matched_columns.get('Typical Vertical')
        aspect_col = matched_columns.get('Slope Aspect')
        angle_col = matched_columns.get('Slope Angle')

        step_num = 1

        for run_idx in range(EXPECTED_RUN_COUNT):
            run_num = run_idx + 1

            if run_idx >= len(df):
                for f in sub_fields:
                    checkpoint.add_step(f"Run {run_num} - {f}", False, step_num,
                                      f"Run {run_num} not present in spreadsheet",
                                      execution_time=0, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
                    step_num += 1
                continue

            row = df.iloc[run_idx]

            user_name = str(row.iloc[name_col]) if name_col is not None else ""
            user_link = str(row.iloc[link_col]) if link_col is not None else ""

            gold_run = find_run_by_name_or_url(user_name, user_link, gold_data)

            # Step 1: Run Name Valid
            step_start = time.time()
            if gold_run:
                checkpoint.add_step(f"Run {run_num} - Name Valid", True, step_num,
                                  f"'{user_name}' found in gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step(f"Run {run_num} - Name Valid", False, step_num,
                                  f"'{user_name}' not found in gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
            step_num += 1

            # Step 2: Run Link Valid
            step_start = time.time()
            link_valid = is_valid_wbsguide_url(user_link)
            if link_valid:
                checkpoint.add_step(f"Run {run_num} - Link Valid", True, step_num,
                                  f"Valid WBSGuide URL: {user_link}",
                                  execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step(f"Run {run_num} - Link Valid", False, step_num,
                                  f"Invalid URL: {user_link}",
                                  execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
            step_num += 1

            # Step 3: Starting Location Correct
            step_start = time.time()
            if start_col is not None and gold_run:
                user_start = str(row.iloc[start_col])
                gold_start = gold_run.get('starting_location', '')

                if gold_start is None:
                    checkpoint.add_step(f"Run {run_num} - Starting Location", True, step_num,
                                      f"No gold starting_location to validate against (benefit of the doubt)",
                                      execution_time=time.time() - step_start, category=StepCategory.VACUOUS_PASS)
                else:
                    user_norm = user_start.lower().strip().replace("'", "").replace("\u2019", "")
                    gold_norm = gold_start.lower().strip().replace("'", "").replace("\u2019", "")

                    if user_norm == gold_norm or user_norm in gold_norm or gold_norm in user_norm:
                        checkpoint.add_step(f"Run {run_num} - Starting Location", True, step_num,
                                          f"'{user_start}' matches gold '{gold_start}'",
                                          execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                    else:
                        checkpoint.add_step(f"Run {run_num} - Starting Location", False, step_num,
                                          f"'{user_start}' != '{gold_start}'",
                                          execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step(f"Run {run_num} - Starting Location", False, step_num,
                                  "Column not found or no gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
            step_num += 1

            # Step 4: GPS Coordinates Correct
            step_start = time.time()
            if gps_col is not None and gold_run:
                user_gps_str = str(row.iloc[gps_col])
                user_coords = parse_gps_coordinates(user_gps_str)
                gold_lat = gold_run.get('gps_lat')
                gold_lon = gold_run.get('gps_lon')

                if user_coords and gold_lat and gold_lon:
                    run_url = gold_run.get('url', user_link)
                    match_result, detail = gps_coordinates_match_with_fallback(
                        user_coords, gold_lat, gold_lon, run_url=run_url, tolerance=0.01)
                    checkpoint.add_step(f"Run {run_num} - GPS Coordinates", match_result, step_num,
                                      detail, execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
                elif user_coords:
                    checkpoint.add_step(f"Run {run_num} - GPS Coordinates", True, step_num,
                                      f"User coords: {user_coords} (no gold GPS to validate against, benefit of the doubt)",
                                      execution_time=time.time() - step_start, category=StepCategory.VACUOUS_PASS)
                else:
                    checkpoint.add_step(f"Run {run_num} - GPS Coordinates", False, step_num,
                                      f"Could not parse: '{user_gps_str}'",
                                      execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
            else:
                checkpoint.add_step(f"Run {run_num} - GPS Coordinates", False, step_num,
                                  "Column not found or no gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
            step_num += 1

            # Step 5: Typical Vertical Correct
            step_start = time.time()
            if vert_col is not None and gold_run:
                user_vert_str = str(row.iloc[vert_col])
                user_vert = parse_typical_vertical(user_vert_str)
                gold_vert_max = gold_run.get('typical_vertical')
                gold_vert_min = gold_run.get('typical_vertical_min', gold_vert_max)

                if user_vert and gold_vert_max:
                    tolerance = 50
                    is_valid = (gold_vert_min - tolerance) <= user_vert <= (gold_vert_max + tolerance)

                    if is_valid:
                        if gold_vert_min != gold_vert_max:
                            checkpoint.add_step(f"Run {run_num} - Typical Vertical", True, step_num,
                                              f"{user_vert} ft (gold range: {gold_vert_min}-{gold_vert_max} ft)",
                                              execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
                        else:
                            checkpoint.add_step(f"Run {run_num} - Typical Vertical", True, step_num,
                                              f"{user_vert} ft (gold: {gold_vert_max} ft)",
                                              execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
                    else:
                        if gold_vert_min != gold_vert_max:
                            checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                              f"{user_vert} ft not in range {gold_vert_min}-{gold_vert_max} ft",
                                              execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
                        else:
                            checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                              f"{user_vert} ft != {gold_vert_max} ft",
                                              execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
                elif user_vert:
                    checkpoint.add_step(f"Run {run_num} - Typical Vertical", True, step_num,
                                      f"{user_vert} ft (no gold typical_vertical to validate against, benefit of the doubt)",
                                      execution_time=time.time() - step_start, category=StepCategory.VACUOUS_PASS)
                else:
                    checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                      f"Could not parse: '{user_vert_str}'",
                                      execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
            else:
                checkpoint.add_step(f"Run {run_num} - Typical Vertical", False, step_num,
                                  "Column not found or no gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
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
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                elif user_aspect:
                    checkpoint.add_step(f"Run {run_num} - Slope Aspect", False, step_num,
                                      f"User: {user_aspect}, Gold: {gold_aspect}",
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                else:
                    checkpoint.add_step(f"Run {run_num} - Slope Aspect", False, step_num,
                                      f"Could not parse: '{user_aspect_str}'",
                                      execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
            else:
                checkpoint.add_step(f"Run {run_num} - Slope Aspect", False, step_num,
                                  "Column not found or no gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
            step_num += 1

            # Step 7: Slope Angle Correct
            step_start = time.time()
            if angle_col is not None and gold_run:
                user_angle_str = str(row.iloc[angle_col])
                user_angle = parse_slope_angle(user_angle_str)
                gold_angle = gold_run.get('slope_angle')

                if user_angle and gold_angle and user_angle == gold_angle and user_angle <= MAX_SLOPE_ANGLE:
                    checkpoint.add_step(f"Run {run_num} - Slope Angle", True, step_num,
                                      f"Angle: {user_angle} (valid <= {MAX_SLOPE_ANGLE})",
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                elif user_angle and user_angle <= MAX_SLOPE_ANGLE:
                    checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                      f"User: {user_angle}, Gold: {gold_angle}",
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                elif user_angle:
                    checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                      f"Angle {user_angle} exceeds {MAX_SLOPE_ANGLE} limit",
                                      execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
                else:
                    checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                      f"Could not parse: '{user_angle_str}'",
                                      execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
            else:
                checkpoint.add_step(f"Run {run_num} - Slope Angle", False, step_num,
                                  "Column not found or no gold data",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
            step_num += 1

        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    except Exception as e:
        traceback.print_exc()
        failed = Checkpoint(total=35, result=0, name="Run Data Accuracy")
        reason = f"Checkpoint raised: {str(e)[:100]}"
        for i, step_name in enumerate(expected_steps, start=1):
            failed.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
        failed.execution_time = time.time() - checkpoint_start
        return failed


def grade_checkpoint_4():
    """Checkpoint 4: Website Visit Validation (4 pts)."""
    print("----------------- CHECKPOINT 4 ----------------")
    global BROWSING_HISTORY, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Website Visit Validation")
    expected_steps = ["Wasatch Guide Visited", "Utah Avalanche Center Visited",
                      "Run Links Visited", "Forecast Page Visited"]

    try:
        if not BROWSING_HISTORY:
            for i, name in enumerate(expected_steps, start=1):
                checkpoint.add_step(name, False, i,
                                  "No browsing history provided",
                                  execution_time=0, category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        history_lower = [url.lower() for url in BROWSING_HISTORY]

        step_start = time.time()
        wbs_visited = any('wbsguide.com' in url for url in history_lower)
        checkpoint.add_step("Wasatch Guide Visited", wbs_visited, 1,
                          f"wbsguide.com {'found' if wbs_visited else 'not found'} in history",
                          execution_time=time.time() - step_start, category=StepCategory.WEB_VISIT)

        step_start = time.time()
        uac_visited = any('utahavalanchecenter.org' in url for url in history_lower)
        checkpoint.add_step("Utah Avalanche Center Visited", uac_visited, 2,
                          f"utahavalanchecenter.org {'found' if uac_visited else 'not found'} in history",
                          execution_time=time.time() - step_start, category=StepCategory.WEB_VISIT)

        step_start = time.time()
        link_col = matched_columns.get('Run Link')
        run_link_visited = False

        # Match the wbsguide.com run-id portion (e.g., "/2104.php"), not just any
        # substring overlap, so visiting wbsguide.com homepage doesn't credit this.
        if link_col is not None and df is not None:
            for i in range(min(EXPECTED_RUN_COUNT, len(df))):
                user_link = str(df.iloc[i].iloc[link_col]).lower()
                m = re.search(r'/(\d+\.php)', user_link)
                if not m:
                    continue
                run_id_part = m.group(1)
                if any(run_id_part in url for url in history_lower):
                    run_link_visited = True
                    break

        checkpoint.add_step("Run Links Visited", run_link_visited, 3,
                          f"Run link {'found' if run_link_visited else 'not found'} in history",
                          execution_time=time.time() - step_start, category=StepCategory.WEB_VISIT)

        # Require the FORECAST_DATE in the URL path so visiting any other date
        # does not credit this step. UAC URLs use /salt-lake/<m>/<d>/<yyyy>.
        step_start = time.time()
        date_path = "/" + FORECAST_DATE
        date_path_alt = "/" + FORECAST_DATE_ALT
        forecast_visited = any(
            'utahavalanchecenter.org/forecast' in url
            and (date_path in url or date_path_alt in url)
            for url in history_lower
        )
        checkpoint.add_step("Forecast Page Visited", forecast_visited, 4,
                          f"Forecast page for {FORECAST_DATE} {'found' if forecast_visited else 'not found'} in history",
                          execution_time=time.time() - step_start, category=StepCategory.WEB_VISIT)

        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    except Exception as e:
        traceback.print_exc()
        failed = Checkpoint(total=4, result=0, name="Website Visit Validation")
        reason = f"Checkpoint raised: {str(e)[:100]}"
        for i, step_name in enumerate(expected_steps, start=1):
            failed.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
        failed.execution_time = time.time() - checkpoint_start
        return failed


def grade_checkpoint_5():
    """Checkpoint 5: Avalanche Forecast Data (5 pts)."""
    print("----------------- CHECKPOINT 5 ----------------")
    global matched_columns, gold_data, model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Avalanche Forecast Data")
    expected_steps = ["Forecast Date Correct", "Forecast Link Valid", "Merged Cells",
                      "Danger Rose Present", "Danger Rose Valid"]

    try:
        forecast_date_col = matched_columns.get('Forecast Date')
        forecast_link_col = matched_columns.get('Forecast Link')

        FIRST_DATA_ROW = detected_header_row + 1

        # Step 1: Forecast Date Correct
        step_start = time.time()
        if forecast_date_col is not None:
            forecast_date = get_cell_value(sheet_raw, FIRST_DATA_ROW, forecast_date_col) or ""
            date_valid = FORECAST_DATE in forecast_date or FORECAST_DATE_ALT in forecast_date
            checkpoint.add_step("Forecast Date Correct", date_valid, 1,
                              f"Date: '{forecast_date}' (expected {FORECAST_DATE})",
                              execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Forecast Date Correct", False, 1,
                              "Forecast Date column not found",
                              execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)

        # Step 2: Forecast Link Valid
        step_start = time.time()
        if forecast_link_col is not None:
            forecast_link = get_cell_value(sheet_raw, FIRST_DATA_ROW, forecast_link_col) or ""
            link_valid = is_valid_uac_forecast_url(forecast_link)
            checkpoint.add_step("Forecast Link Valid", link_valid, 2,
                              f"Link: '{forecast_link[:50]}...'",
                              execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Forecast Link Valid", False, 2,
                              "Forecast Link column not found",
                              execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)

        # Step 3: Merged Cells
        step_start = time.time()
        cols_to_check = []
        if forecast_date_col is not None:
            cols_to_check.append(forecast_date_col)
        if forecast_link_col is not None:
            cols_to_check.append(forecast_link_col)

        if cols_to_check:
            try:
                merged = check_merged_cells(sheet_raw, cols_to_check, FIRST_DATA_ROW, FIRST_DATA_ROW + EXPECTED_RUN_COUNT - 1)
                checkpoint.add_step("Merged Cells", merged, 3,
                                  f"Columns {cols_to_check} {'are' if merged else 'are not'} merged vertically",
                                  execution_time=time.time() - step_start, category=StepCategory.STRUCTURAL)
            except Exception as e:
                checkpoint.add_step("Merged Cells", False, 3,
                                  f"Merge check failed: {str(e)[:50]}",
                                  execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
        else:
            checkpoint.add_step("Merged Cells", False, 3,
                              "Required columns not found for merge check",
                              execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)

        # Step 4: Danger Rose Screenshot Present
        step_start = time.time()
        danger_rose_col = None
        if df is not None:
            for i, col in enumerate(df.columns):
                col_lower = str(col).lower()
                if 'rose' in col_lower or ('danger' in col_lower and 'color' not in col_lower):
                    danger_rose_col = i
                    break

        image_url = None
        if danger_rose_col is not None:
            try:
                image_url = get_image_url_from_raw_sheet_cell(sheet_raw, FIRST_DATA_ROW, danger_rose_col)
            except Exception as e:
                print(f"WARNING: get_image_url_from_raw_sheet_cell failed: {e}")
                image_url = None

        # Check if URL looks like an actual image (not a webpage link)
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')
        is_image_url = (image_url is not None and image_url.startswith('http')
                       and any(image_url.lower().split('?')[0].endswith(ext) for ext in image_extensions))
        image_present = is_image_url

        if image_url and image_url.startswith('http') and not is_image_url:
            detail = f"URL is not an image (looks like a webpage link): {image_url[:50]}..."
        elif image_url:
            detail = f"Image URL: {image_url[:50] + '...' if len(image_url) > 50 else image_url}"
        else:
            detail = "not found"
        checkpoint.add_step("Danger Rose Present", image_present, 4,
                          detail, execution_time=time.time() - step_start, category=StepCategory.DETERMINISTIC)

        # Step 5: Danger Rose Image Valid (tiered image comparison)
        step_start = time.time()
        user_image_path = None
        gold_image_path = None

        if image_present and gold_data:
            try:
                gold_image_url = (gold_data.get('forecast') or {}).get('danger_rose_image_url')

                if not gold_image_url:
                    checkpoint.add_step("Danger Rose Valid", False, 5,
                                      "Gold danger rose image URL not configured",
                                      execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
                else:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    user_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    user_temp.write(response.content)
                    user_temp.close()
                    user_image_path = user_temp.name

                    gold_response = requests.get(gold_image_url, timeout=30)
                    gold_response.raise_for_status()
                    gold_temp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    gold_temp.write(gold_response.content)
                    gold_temp.close()
                    gold_image_path = gold_temp.name

                    if model is None:
                        model = load_model(model_id)
                    match_result, match_method = match_image_tiered(
                        user_image_path,
                        gold_image_path,
                        model=model,
                        hash_threshold=15
                    )

                    checkpoint.add_step("Danger Rose Valid", match_result, 5,
                                      f"Match: {match_result} (method: {match_method})",
                                      execution_time=time.time() - step_start, category=StepCategory.from_match_method(match_method) if match_result else StepCategory.LLM_VLM_JUDGEMENT)
            except Exception as e:
                checkpoint.add_step("Danger Rose Valid", False, 5,
                                  f"Image comparison failed: {str(e)[:50]}",
                                  execution_time=time.time() - step_start, category=StepCategory.EXECUTION_ERROR)
            finally:
                for path in [user_image_path, gold_image_path]:
                    if path and os.path.exists(path):
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
        else:
            checkpoint.add_step("Danger Rose Valid", False, 5,
                              "No image to validate or missing gold data",
                              execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)

        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    except Exception as e:
        traceback.print_exc()
        failed = Checkpoint(total=5, result=0, name="Avalanche Forecast Data")
        reason = f"Checkpoint raised: {str(e)[:100]}"
        for i, step_name in enumerate(expected_steps, start=1):
            failed.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
        failed.execution_time = time.time() - checkpoint_start
        return failed


def grade_checkpoint_6():
    """Checkpoint 6: Danger Rating Cell Coloring.

    For each run, validates that the run name cell is colored with the
    correct danger rating color based on the avalanche forecast.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    global matched_columns, df, gold_data
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=EXPECTED_RUN_COUNT, result=0, name="Danger Rating Cell Coloring")
    expected_steps = [f"Run {r} - Cell Colored" for r in range(1, EXPECTED_RUN_COUNT + 1)]

    try:
        if df is None or df.empty or not gold_data:
            reason = "No data available for validation"
            for i, step_name in enumerate(expected_steps, start=1):
                checkpoint.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        name_col = matched_columns.get('Run Name')
        link_col = matched_columns.get('Run Link')

        FIRST_DATA_ROW = detected_header_row + 1

        for run_idx in range(EXPECTED_RUN_COUNT):
            run_num = run_idx + 1
            step_num = run_idx + 1

            if run_idx >= len(df):
                checkpoint.add_step(f"Run {run_num} - Cell Colored", False, step_num,
                                  f"Run {run_num} not present in spreadsheet",
                                  execution_time=0, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
                continue

            row = df.iloc[run_idx]
            sheet_row_idx = FIRST_DATA_ROW + run_idx

            user_name = str(row.iloc[name_col]) if name_col is not None else ""
            user_link = str(row.iloc[link_col]) if link_col is not None else ""

            gold_run = find_run_by_name_or_url(user_name, user_link, gold_data)
            expected_rating = gold_run.get('expected_danger_rating', 'unknown') if gold_run else 'unknown'

            step_start = time.time()
            if name_col is not None:
                try:
                    bg_color = get_cell_background_color(sheet_raw, sheet_row_idx, name_col)
                    actual_color = classify_danger_color(bg_color)
                except Exception as e:
                    actual_color = "none"
                    print(f"WARNING: get_cell_background_color failed: {e}")

                if expected_rating != 'unknown':
                    color_matches = actual_color == expected_rating
                    checkpoint.add_step(f"Run {run_num} - Cell Colored", color_matches, step_num,
                                      f"Cell color: {actual_color}, Expected: {expected_rating}",
                                      execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
                else:
                    is_colored = actual_color in ['green', 'yellow', 'orange', 'red', 'black']
                    checkpoint.add_step(f"Run {run_num} - Cell Colored", is_colored, step_num,
                                      f"Cell color: {actual_color}",
                                      execution_time=time.time() - step_start, category=StepCategory.FUZZY_MATCH)
            else:
                checkpoint.add_step(f"Run {run_num} - Cell Colored", False, step_num,
                                  "Run Name column not found",
                                  execution_time=time.time() - step_start, category=StepCategory.DEPENDENCY_NOT_EVALUATED)

        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint
    except Exception as e:
        traceback.print_exc()
        failed = Checkpoint(total=EXPECTED_RUN_COUNT, result=0, name="Danger Rating Cell Coloring")
        reason = f"Checkpoint raised: {str(e)[:100]}"
        for i, step_name in enumerate(expected_steps, start=1):
            failed.add_step(step_name, False, i, reason, execution_time=0, category=StepCategory.EXECUTION_ERROR)
        failed.execution_time = time.time() - checkpoint_start
        return failed


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

    BROWSING_HISTORY = browsing_history or []

    try:
        setup(workspace_doc_id)

        checkpoints: List[Checkpoint] = []

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

        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0, category=StepCategory.EXECUTION_ERROR)
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
            status = "+" if step["success"] else "x"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
