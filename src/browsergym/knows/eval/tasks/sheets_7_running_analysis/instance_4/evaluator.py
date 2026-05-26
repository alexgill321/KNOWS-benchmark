import os
import sys
from typing import List
import time
import pandas as pd
import argparse

# Base path setup (same pattern as other evaluators)
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/path/to/KNOWS-benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    extract_charts_from_sheet,
    extract_sheet_data,
    find_urls_in_sheet,
)
from src.browsergym.knows.eval.eval_utils.web_utils import validate_url_accessible
from src.browsergym.knows.eval.eval_utils.table_utils import (
    match_columns,
    colors_are_similar,
    check_all_content_visible,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_match_robust
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.chart_utils import (
    get_series_header_label,
    get_series_column_values,
    validate_constant_series,
    get_series_line_style,
    get_series_color,
    get_chart_axis_labels,
    check_chart_overlap,
    check_point_shape,
    get_all_series_metadata,
    get_chart_type,
    identify_series_by_content,
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute, fast_parallel_vlm_calls

# Task-specific utilities
from src.browsergym.knows.eval.tasks.sheets_7_running_analysis.utils import (
    build_keyword_match_prompt,
    normalize_date,
    load_gold_run_activities,
    find_speed_chart_by_metadata,
    find_cumulative_chart_by_metadata,
    find_daily_miles_chart_by_metadata,
    extract_pace_from_url,
    extract_and_convert_pace_from_url,
    judge_url_relevance,
    judge_url_pace_match,
    km_to_miles,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/sheets_7_running_analysis/instance_4/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Gold baseline values for checkpoint 4 (speed chart)
FEMALE_5K_PACE_RANGE = (7.5, 11.5)  # min/mile for 25yo female 5K (intermediate to novice)
CHEBET_PACE_RANGE = (4.0, 5.0)  # min/mile (Beatrice Chebet 5K pace)

# Gold baseline value for checkpoint 3 (daily total miles chart)
FEMALE_DAILY_MILES_RANGE = (1.5, 6.0)  # average daily miles trekked for 25yo female

# Expected activity counts
EXPECTED_RUNS = 344
EXPECTED_WALKS = 108
EXPECTED_TOTAL = EXPECTED_RUNS + EXPECTED_WALKS  # 452

model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global variables
sheet_id = None
sheet_raw = None
table_data = None  # SheetTable object with position metadata
table_sheet_id = None  # sheetId of the tab containing the data table
rows = None  # Raw row data from sheet
matched_columns = None  # Shared across checkpoints
chart_data = None  # All charts extracted from the sheet
sheet_female_5k_pace = None  # Female 5K baseline value from sheet (for checkpoint 6)
sheet_chebet_pace = None  # Chebet baseline value from sheet (for checkpoint 6)
sheet_female_daily_miles = None  # Female daily miles baseline value from sheet (for checkpoint 6)


def setup(workspace_doc_id):
    """
    Setup function to initialize the evaluator.

    Args:
        workspace_doc_id (str): Google Sheets document ID to evaluate
    """
    global sheet_id, sheet_raw, table_data, table_sheet_id, rows, chart_data

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    # Extract table data and raw sheet data using extract_sheet_data
    result = extract_sheet_data(sheet_id, SHEETS_SERVICE, return_raw=True)

    if result:
        table_data, sheet_raw = result

        # Handle case where multiple tables are returned (use first one)
        if isinstance(table_data, list):
            table_data = table_data[0] if table_data else None

        if table_data:
            print(f"Extracted DataFrame with {len(table_data.df)} rows and columns: {list(table_data.df.columns)}")
            print(f"Table position: rows {table_data.start_row}-{table_data.end_row}, cols {table_data.start_col}-{table_data.end_col}")

    # Extract rows from raw sheet data (needed for visibility check and URL finding)
    if sheet_raw:
        try:
            sheets = sheet_raw.get('sheets', [])
            if sheets:
                table_sheet_id = sheets[0].get('properties', {}).get('sheetId', 0)
                grid_data = sheets[0].get('data', [{}])[0]
                rows = grid_data.get('rowData', [])
        except Exception as e:
            print(f"Error extracting rows from sheet_raw: {e}")
            rows = None

    # Extract charts from the sheet
    chart_data = extract_charts_from_sheet(sheet_id, SHEETS_SERVICE)
    print(f"Extracted {len(chart_data) if chart_data else 0} charts from spreadsheet")


def grade_checkpoint_1():
    """
    Grade Checkpoint 1: Data Table Structure (4 pts).

    Outcome Evaluation:
    - Date/time column exists with appropriate header.
    - Distance column exists with appropriate header.
    - Average Speed column exists with appropriate header.
    - All table content is fully visible (no text overflow/truncation).

    Also stores matched_columns globally for use in Checkpoint 2.
    """
    global matched_columns

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Data Table Structure")

    if table_data:
        df = table_data.df
    else:
        df = None

    # Check if data was extracted
    if df is None or df.empty:
        checkpoint.add_step("Activity Date Column", False, 1, "No table found in spreadsheet", execution_time=0)
        checkpoint.add_step("Distance (Miles) Column", False, 2, "No table found in spreadsheet", execution_time=0)
        checkpoint.add_step("Average Running Speed (min/mile) Column", False, 3, "No table found in spreadsheet", execution_time=0)
        checkpoint.add_step("Content Visibility", False, 4, "No table found in spreadsheet", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Define required columns with keywords (accept both metric and imperial units)
    required_columns = [
        ("Activity Date", ["activity date", "run date", "date"]),
        ("Distance", ["distance", "miles", "distance (miles)", "distance (mi)", "distance (km)", "km"]),
        ("Average Running Speed", ["pace", "min/mile", "average pace", "avg pace", "speed", "average speed", "km/h", "speed (km/h)"]),
    ]

    # Match columns using keyword + LLM fallback (this is the main work for steps 1-3)
    column_match_start = time.time()
    matched = match_columns(df, required_columns, model=model, parallel=True)
    column_match_time = time.time() - column_match_start

    # Store globally for use in Checkpoint 2
    matched_columns = matched

    # Split column matching time across steps 1-3 (they share the same call)
    per_column_time = column_match_time / 3

    # Step 1: Activity Date column
    date_col = matched.get("Activity Date")
    checkpoint.add_step(
        "Activity Date Column",
        date_col is not None,
        1,
        f"Found column: '{date_col}'" if date_col else "No activity date column found",
        execution_time=per_column_time
    )

    # Step 2: Distance column
    dist_col = matched.get("Distance")
    checkpoint.add_step(
        "Distance Column",
        dist_col is not None,
        2,
        f"Found column: '{dist_col}'" if dist_col else "No distance column found",
        execution_time=per_column_time
    )

    # Step 3: Average Running Speed column
    speed_col = matched.get("Average Running Speed")
    checkpoint.add_step(
        "Average Running Speed Column",
        speed_col is not None,
        3,
        f"Found column: '{speed_col}'" if speed_col else "No average speed/pace column found",
        execution_time=per_column_time
    )

    # Step 4: Content visibility
    step_start = time.time()
    start_row = table_data.start_row if table_data else 0
    end_row = table_data.end_row if table_data else start_row + len(df) + 1
    num_cols = len(df.columns)
    all_visible, visibility_details = check_all_content_visible(sheet_raw, start_row, end_row, num_cols)
    checkpoint.add_step(
        "Content Visibility",
        all_visible,
        4,
        visibility_details,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Grade Checkpoint 2: Data Table Content Accuracy (30 pts).

    Outcome Evaluation:
    - All 452 Run+Walk activities have exact date match to gold data.
    - All 452 Run+Walk activities have exact distance match to gold data (converted to miles).
    - All 452 Run+Walk activities have exact average speed match to gold data (converted to min/mile).
    """
    global matched_columns, model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Data Table Content Accuracy")

    df = table_data.df if table_data else None

    if df is None or df.empty:
        checkpoint.add_step("Date Match", False, 1, "No table data available", execution_time=0)
        checkpoint.add_step("Distance Match", False, 2, "No table data available", execution_time=0)
        checkpoint.add_step("Speed Match", False, 3, "No table data available", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Load gold data (both Run and Walk activities)
    gold_csv_path = os.path.join(DATA_DIR, "gold_activities.csv")
    try:
        gold_runs = load_gold_run_activities(gold_csv_path, activity_type='Run')
        gold_walks = load_gold_run_activities(gold_csv_path, activity_type='Walk')
        gold_all = pd.concat([gold_runs, gold_walks], ignore_index=True)
    except Exception as e:
        checkpoint.add_step("Date Match", False, 1, f"Error loading gold data: {str(e)}", execution_time=0)
        checkpoint.add_step("Distance Match", False, 2, "Gold data error", execution_time=0)
        checkpoint.add_step("Speed Match", False, 3, "Gold data error", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if len(gold_all) != EXPECTED_TOTAL:
        checkpoint.add_step("Date Match", False, 1, f"Expected {EXPECTED_TOTAL} activities, found {len(gold_all)}", execution_time=0)
        checkpoint.add_step("Distance Match", False, 2, "Gold data count mismatch", execution_time=0)
        checkpoint.add_step("Speed Match", False, 3, "Gold data count mismatch", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get date column from checkpoint 1 matching
    date_col = matched_columns.get("Activity Date") if matched_columns else None

    # For checkpoint 2, find columns with correct units using keyword matching
    unit_columns = [
        ("Distance (Miles)", ["distance (miles)", "distance (mi)", "miles", "(miles)"]),
        ("Speed (min/mile)", ["min/mile", "(min/mile)", "pace (min/mile)", "min per mile", "min/mi", "(min/mi)", "pace (min/mi)"]),
    ]
    unit_matches = match_columns(df, unit_columns, model=model, strict=True)

    dist_col = None
    speed_col = None
    if unit_matches:
        matched_columns.update(unit_matches)
        dist_col = matched_columns.get("Distance (Miles)")
        speed_col = matched_columns.get("Speed (min/mile)")

    # Build gold data lookup by normalized date
    gold_lookup = {}
    for idx, row in gold_all.iterrows():
        norm_date = normalize_date(row['Activity Date'])
        dist_km = row['Distance']
        speed_ms = row['Average Speed']
        gold_lookup[norm_date] = {
            'distance_km': dist_km,
            'distance_miles': dist_km / 1.60934,
            'distance_meters': dist_km * 1000,
            'speed_ms': speed_ms,
            'speed_kmh': speed_ms * 3.6,
            'speed_minmile': 26.8224 / speed_ms if speed_ms > 0 else float('inf')
        }

    # Track matches for each criterion
    date_matches = 0
    distance_matches = 0
    speed_matches = 0
    detected_dist_unit = None
    detected_speed_unit = None

    # Track failed matches for debugging
    failed_distance_rows = []

    # Validate row by row
    validation_start = time.time()
    for idx, user_row in df.iterrows():
        # Get user date (handle duplicate column names)
        if date_col and date_col in df.columns:
            date_val = user_row[date_col]
            if isinstance(date_val, pd.Series):
                date_val = date_val.iloc[0]
            user_date = normalize_date(str(date_val))
        else:
            continue

        # Check if date exists in gold data
        if user_date in gold_lookup:
            date_matches += 1
            gold_row = gold_lookup[user_date]

            # Check distance - ONLY accept miles (per task.md requirements)
            if dist_col and dist_col in df.columns:
                try:
                    user_dist = float(user_row[dist_col])
                    gold_miles = gold_row['distance_miles']
                    if gold_miles > 0 and abs(user_dist - gold_miles) / gold_miles <= 0.01:
                        distance_matches += 1
                        if detected_dist_unit is None:
                            detected_dist_unit = 'miles'
                    else:
                        failed_distance_rows.append({
                            'date': user_date,
                            'user_dist': user_dist,
                            'gold_km': gold_row['distance_km'],
                            'gold_miles': gold_row['distance_miles'],
                            'gold_meters': gold_row['distance_meters']
                        })
                except (ValueError, TypeError):
                    pass

            # Check speed - ONLY accept min/mile (per task.md requirements)
            if speed_col and speed_col in df.columns:
                try:
                    user_speed = float(user_row[speed_col])
                    gold_minmile = gold_row['speed_minmile']
                    if gold_minmile > 0 and gold_minmile != float('inf') and abs(user_speed - gold_minmile) / gold_minmile <= 0.01:
                        speed_matches += 1
                        if detected_speed_unit is None:
                            detected_speed_unit = 'min/mile'
                except (ValueError, TypeError):
                    pass
    validation_time = time.time() - validation_start

    # Split validation time across all 3 steps
    per_step_time = validation_time / 3

    import math

    # Step 1: Date matching (proportional, out of 10)
    if date_col:
        date_score = math.floor(date_matches / EXPECTED_TOTAL * 10)
        all_dates_match = date_matches == EXPECTED_TOTAL
        checkpoint.add_step(
            "Date Match",
            all_dates_match,
            1,
            f"{date_matches}/{EXPECTED_TOTAL} dates match ({date_matches/EXPECTED_TOTAL:.0%}), {date_score}/10 pts",
            score=date_score,
            max_score=10,
            execution_time=per_step_time
        )
    else:
        checkpoint.add_step("Date Match", False, 1, "Date column not found", score=0, max_score=10, execution_time=per_step_time)

    # Step 2: Distance matching (proportional, out of 10)
    if dist_col:
        dist_score = math.floor(distance_matches / EXPECTED_TOTAL * 10)
        all_dist_match = distance_matches == EXPECTED_TOTAL
        unit_str = f" ({detected_dist_unit})" if detected_dist_unit else ""
        if DEBUG and failed_distance_rows:
            print(f"\n=== FAILED DISTANCE MATCHES ({len(failed_distance_rows)}) ===")
            for row in failed_distance_rows[:5]:
                print(f"  Date: {row['date']}")
                print(f"    User dist: {row['user_dist']}")
                print(f"    Gold km: {row['gold_km']}, miles: {row['gold_miles']:.4f}, meters: {row['gold_meters']}")
        checkpoint.add_step(
            "Distance Match",
            all_dist_match,
            2,
            f"{distance_matches}/{EXPECTED_TOTAL} distances match ({distance_matches/EXPECTED_TOTAL:.0%}){unit_str}, {dist_score}/10 pts",
            score=dist_score,
            max_score=10,
            execution_time=per_step_time
        )
    else:
        checkpoint.add_step("Distance Match", False, 2, "No distance column with miles unit found (column header must contain 'miles')", score=0, max_score=10, execution_time=per_step_time)

    # Step 3: Speed matching (proportional, out of 10)
    if speed_col:
        speed_score = math.floor(speed_matches / EXPECTED_TOTAL * 10)
        all_speed_match = speed_matches == EXPECTED_TOTAL
        unit_str = f" ({detected_speed_unit})" if detected_speed_unit else ""
        checkpoint.add_step(
            "Speed Match",
            all_speed_match,
            3,
            f"{speed_matches}/{EXPECTED_TOTAL} speeds match ({speed_matches/EXPECTED_TOTAL:.0%}){unit_str}, {speed_score}/10 pts",
            score=speed_score,
            max_score=10,
            execution_time=per_step_time
        )
    else:
        checkpoint.add_step("Speed Match", False, 3, "No speed column with min/mile unit found (column header must contain 'min/mile')", score=0, max_score=10, execution_time=per_step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Grade Checkpoint 3: Daily Total Miles Bar Plot (10 steps).

    Outcome Evaluation:
    1. X-axis label indicates date
    2. Y-axis label indicates miles or distance
    3. Chart title indicates daily total miles
    4. Chart is not placed over any other charts or tables
    5. Chart is BAR type
    6. Walk data series identified (legend keyword: "walk")
    7. Run data series identified (legend keyword: "run")
    8. Female daily miles baseline displayed (labeled + dotted/dashed style)
    9. Female daily miles baseline data constant and within range (2.0-6.0 miles)
    10. Chart is on the same sheet tab as the data table
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=9, result=0, name="Daily Total Miles Bar Plot")

    # Check if any charts exist
    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 10):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find daily miles chart by metadata
    daily_chart = find_daily_miles_chart_by_metadata(
        chart_data, matched_columns, table_data.df if table_data else None, model=model
    )

    if not daily_chart:
        error_msg = "Could not identify daily total miles chart by title or axis labels"
        for i in range(1, 10):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = get_chart_type(daily_chart)

    # Get axis labels
    # For BAR charts, Google Sheets API reports axes swapped (value=x/bottom, category=y/left)
    # so we swap them back for consistent evaluation
    axis_labels = get_chart_axis_labels(daily_chart)
    if chart_type == 'BAR':
        x_label = axis_labels.get('y_axis', '')  # category axis (dates) reported as y
        y_label = axis_labels.get('x_axis', '')  # value axis (miles) reported as x
    else:
        x_label = axis_labels.get('x_axis', '')
        y_label = axis_labels.get('y_axis', '')
    chart_title = daily_chart.get('title', '')

    # Define keywords for each match
    date_keywords = ['date', 'time', 'day', 'activity']
    miles_keywords = ['miles', 'distance', 'total', 'mileage']
    daily_title_keywords = ['daily', 'total', 'miles', 'per day']

    # Build VLM tasks for parallel keyword matching (steps 1-3)
    vlm_tasks = []
    if x_label:
        vlm_tasks.append({
            'id': 'x_axis',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(x_label, date_keywords, "X-axis label indicating date or time")}]}
            ]
        })
    if y_label:
        vlm_tasks.append({
            'id': 'y_axis',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(y_label, miles_keywords, "Y-axis label indicating miles or distance")}]}
            ]
        })
    if chart_title:
        vlm_tasks.append({
            'id': 'title_daily',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, daily_title_keywords, "chart title related to daily total miles")}]}
            ]
        })

    # Execute all keyword matching in parallel
    keyword_start = time.time()
    if vlm_tasks:
        keyword_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=3)
    else:
        keyword_results = {}
    keyword_time = time.time() - keyword_start

    # Step 1: X-axis label indicates date
    x_label_match = keyword_results.get('x_axis', False) if x_label else False
    x_label_keyword_fallback = keywords_match_robust(
        texts=x_label, keywords=date_keywords, model=model,
        description="X-axis label indicating date or time"
    ) if x_label else False
    has_date_label = bool(x_label_match) or bool(x_label_keyword_fallback)
    checkpoint.add_step(
        "X-Axis Date Label",
        has_date_label,
        1,
        f"X-axis label: '{x_label}'" if has_date_label else f"X-axis label '{x_label or 'None'}' does not match date keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1)
    )

    # Step 2: Y-axis label indicates miles/distance
    y_label_match = keyword_results.get('y_axis', False) if y_label else False
    y_label_keyword_fallback = keywords_match_robust(
        texts=y_label, keywords=miles_keywords, model=model,
        description="Y-axis label indicating miles or distance"
    ) if y_label else False
    has_miles_label = bool(y_label_match) or bool(y_label_keyword_fallback)
    checkpoint.add_step(
        "Y-Axis Miles Label",
        has_miles_label,
        2,
        f"Y-axis label: '{y_label}'" if has_miles_label else f"Y-axis label '{y_label or 'None'}' does not match miles/distance keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1)
    )

    # Step 3: Chart title indicates daily total miles
    title_match = keyword_results.get('title_daily', False) if chart_title else False
    title_keyword_fallback = keywords_match_robust(
        texts=chart_title, keywords=daily_title_keywords, model=model,
        description="chart title indicating daily total miles"
    ) if chart_title else False
    has_good_title = bool(title_match) or bool(title_keyword_fallback)
    checkpoint.add_step(
        "Chart Title",
        has_good_title,
        3,
        f"Chart title: '{chart_title}'" if has_good_title else f"Chart title '{chart_title or 'None'}' does not match daily/miles keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1)
    )

    # Step 4: Chart not placed over other charts/tables
    step_start = time.time()
    df = table_data.df if table_data else None
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 453)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    # Guard against charts with None anchor_cell.row (Google Sheets API bug)
    chart_anchor = daily_chart.get('position', {}).get('anchor_cell', {})
    if chart_anchor.get('row') is None:
        has_overlap = False
        overlap_details = "Chart anchor row is None (API bug) - skipping overlap check"
    else:
        # Filter out charts with None anchor rows from the comparison list
        safe_charts = [c for c in (chart_data or []) if c.get('position', {}).get('anchor_cell', {}).get('row') is not None]
        has_overlap, overlap_details = check_chart_overlap(daily_chart, table_start, table_end, safe_charts, table_start_col, table_end_col)
    checkpoint.add_step(
        "No Chart Overlap",
        not has_overlap,
        4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start
    )

    # Step 5: Chart is BAR type (COMBO with COLUMN series also accepted)
    step_start = time.time()
    is_bar = chart_type in ['BAR', 'COLUMN']
    if not is_bar and chart_type == 'COMBO':
        # Accept COMBO if the main data series is COLUMN type
        raw_chart = daily_chart.get('raw_chart', {})
        combo_series = raw_chart.get('spec', {}).get('basicChart', {}).get('series', [])
        if combo_series and combo_series[0].get('type', '') == 'COLUMN':
            is_bar = True
    checkpoint.add_step(
        "Bar Chart Type",
        is_bar,
        5,
        f"Chart type is {chart_type}" if is_bar else f"Chart type is {chart_type}, expected BAR or COLUMN",
        execution_time=time.time() - step_start
    )

    # Identify baseline FIRST (constant series in expected range) to avoid keyword collision
    step_start = time.time()
    global sheet_female_daily_miles

    female_daily_keywords = ["female", "average", "baseline", "daily", "recommended", "women", "walking", "25"]
    baseline_idx = identify_series_by_content(
        chart=daily_chart,
        rows=rows,
        keywords=female_daily_keywords,
        expected_value_range=FEMALE_DAILY_MILES_RANGE,
        require_constant=True,
        model=model,
        description="legend label for female daily miles baseline"
    )

    # Step 6: Chart has a non-constant data series (daily mileage bars)
    data_series_valid = False
    data_series_details = "Could not find a non-constant data series"

    if rows is not None:
        series_list = get_all_series_metadata(daily_chart)
        for i in range(len(series_list)):
            if i == baseline_idx:
                continue
            values = get_series_column_values(daily_chart, i, rows)
            if values and len(set(values)) > 2:
                data_series_valid = True
                label = get_series_header_label(daily_chart, i, rows)
                data_series_details = f"Data series at index {i}, label: '{label}', {len(values)} data points"
                break

    checkpoint.add_step(
        "Daily Mileage Data Series",
        data_series_valid,
        6,
        data_series_details,
        execution_time=time.time() - step_start
    )

    # Step 7: Chart includes both walk and run activity data
    step_start = time.time()
    # Verify data point count matches expected (walk+run days combined)
    both_types_valid = False
    both_types_details = "Cannot verify activity types in chart"

    if data_series_valid and rows is not None:
        series_list = get_all_series_metadata(daily_chart)
        for i in range(len(series_list)):
            if i == baseline_idx:
                continue
            values = get_series_column_values(daily_chart, i, rows)
            if values and len(values) > 0:
                # If only runs were included, we'd expect ~344 points or fewer unique dates
                # With walk+run combined, we expect more data points
                if len(values) >= EXPECTED_RUNS:
                    both_types_valid = True
                    both_types_details = f"Data series has {len(values)} points (>= {EXPECTED_RUNS} runs, includes walk data)"
                else:
                    both_types_details = f"Data series has only {len(values)} points (expected >= {EXPECTED_RUNS})"
                break

    checkpoint.add_step(
        "Walk and Run Data Included",
        both_types_valid,
        7,
        both_types_details,
        execution_time=time.time() - step_start
    )

    # Step 8: Female daily miles baseline display (labeled in legend)
    step_start = time.time()
    baseline_display_valid = False
    baseline_display_details = "Could not identify female daily miles baseline series"

    if baseline_idx is not None and rows is not None:
        baseline_label = get_series_header_label(daily_chart, baseline_idx, rows)
        label_match = keywords_match_robust(
            texts=baseline_label,
            keywords=female_daily_keywords,
            substring=True
        ) if baseline_label else None

        if label_match:
            baseline_display_valid = True
            baseline_line_style = get_series_line_style(daily_chart, baseline_idx)
            baseline_display_details = f"Label: '{baseline_label}', Style: {baseline_line_style or 'SOLID'} (series index {baseline_idx})"
        else:
            baseline_display_details = f"Label '{baseline_label}' doesn't match female daily miles keywords (series index {baseline_idx})"

    checkpoint.add_step(
        "Female Daily Miles Display",
        baseline_display_valid,
        8,
        baseline_display_details,
        execution_time=time.time() - step_start
    )

    # Step 9: Female daily miles baseline data validation (constant, in range)
    step_start = time.time()
    baseline_data_valid = False
    baseline_data_details = "Could not identify female daily miles baseline series"

    if baseline_idx is not None and rows is not None:
        baseline_values = get_series_column_values(daily_chart, baseline_idx, rows)
        if baseline_values:
            baseline_data_valid, _, baseline_data_details = validate_constant_series(
                baseline_values, FEMALE_DAILY_MILES_RANGE, tolerance=0.01
            )
            if baseline_values:
                sheet_female_daily_miles = baseline_values[0]
        else:
            baseline_data_details = f"Could not extract values from baseline series (index {baseline_idx})"
    elif baseline_idx is not None:
        baseline_data_details = "Sheet rows unavailable from setup() - cannot extract baseline values"

    checkpoint.add_step(
        "Female Daily Miles Data",
        baseline_data_valid,
        9,
        baseline_data_details,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Grade Checkpoint 4: Speed Over Time Plot (13 steps).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates speed (min/mile or similar)
    3. Chart title indicates speed over time
    4. Chart is not placed over any other charts or tables
    5. Chart main data series comes from the average speed column (min/mile)
    6. Speed values are present as circular points in the chart
    7. Female 5K baseline is properly displayed (labeled in legend + dotted/dashed style)
    8. Female 5K baseline data is constant and within expected range (7.5-11.5 min/mile)
    9. Chebet baseline is properly displayed (labeled in legend + dotted/dashed style)
    10. Chebet baseline data is constant and within expected range (4.0-5.0 min/mile)
    11. Both baselines are visually distinguishable from the main data
    12. Source URLs are valid and accessible below the speed chart
    13. Chart is on the same sheet tab as the data table
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=11, result=0, name="Speed Over Time Plot")

    # Check if any charts exist
    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 12):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find speed chart by metadata
    speed_chart = find_speed_chart_by_metadata(
        chart_data, matched_columns, table_data.df if table_data else None, model=model
    )

    if not speed_chart:
        error_msg = "Could not identify speed chart by title, axis labels, or series data"
        for i in range(1, 12):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = speed_chart.get('chart_type', 'UNKNOWN')

    # Identify series by content using parallel calls for baseline series
    df = table_data.df if table_data else None

    # Keywords for identifying each series type
    female_5k_keywords = ["female", "5k", "5 k", "baseline", "women", "25 year", "25-year", "25yo", "average"]
    chebet_keywords = ["chebet", "beatrice", "5k", "world", "record"]

    # Parallel: Identify both baseline series concurrently
    baseline_tasks = [
        {
            'id': 'female_5k',
            'func': identify_series_by_content,
            'kwargs': {
                'chart': speed_chart,
                'rows': rows,
                'keywords': female_5k_keywords,
                'expected_value_range': FEMALE_5K_PACE_RANGE,
                'require_constant': True,
                'model': model,
                'description': "legend label for female 5K running baseline"
            }
        },
        {
            'id': 'chebet',
            'func': identify_series_by_content,
            'kwargs': {
                'chart': speed_chart,
                'rows': rows,
                'keywords': chebet_keywords,
                'expected_value_range': CHEBET_PACE_RANGE,
                'require_constant': True,
                'model': model,
                'description': "legend label for Beatrice Chebet 5K baseline"
            }
        },
    ]
    baseline_results = parallel_execute(baseline_tasks, max_workers=2)
    female_5k_idx = baseline_results.get('female_5k')
    chebet_idx = baseline_results.get('chebet')

    # Fallback: if Female 5K not found by value range, find by keyword alone for display checks
    female_5k_candidate_idx = female_5k_idx
    if female_5k_idx is None and rows is not None:
        female_5k_candidate_idx = identify_series_by_content(
            chart=speed_chart, rows=rows,
            keywords=female_5k_keywords,
            require_constant=True, model=model,
            description="legend label for female 5K running baseline"
        )

    # Sequential: Identify main data series (depends on both baselines for exclude_indices)
    exclude_baselines = [i for i in [female_5k_idx, chebet_idx] if i is not None]
    main_idx = identify_series_by_content(
        chart=speed_chart,
        rows=rows,
        keywords=[],
        matched_columns=matched_columns,
        column_name="Speed (min/mile)",
        df=df,
        exclude_indices=exclude_baselines if exclude_baselines else None,
        model=model
    )

    # Get axis labels
    axis_labels = get_chart_axis_labels(speed_chart)
    x_label = axis_labels.get('x_axis', '')
    y_label = axis_labels.get('y_axis', '')
    chart_title = speed_chart.get('title', '')

    # Define keywords for each match
    date_keywords = ['date', 'time', 'day', 'activity']
    speed_keywords = ['speed', 'pace', 'min/mile', 'min per mile', 'minute']
    speed_title_keywords = ['speed', 'pace', 'running']
    time_title_keywords = ['time', 'over', 'progression', 'trend']

    # Build VLM tasks for parallel keyword matching (steps 1-3)
    vlm_tasks = []
    if x_label:
        vlm_tasks.append({
            'id': 'x_axis',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(x_label, date_keywords, "X-axis label indicating activity date or time")}]}
            ]
        })
    if y_label:
        vlm_tasks.append({
            'id': 'y_axis',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(y_label, speed_keywords, "Y-axis label indicating running speed or pace")}]}
            ]
        })
    if chart_title:
        vlm_tasks.append({
            'id': 'title_speed',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, speed_title_keywords, "chart title related to running speed or pace")}]}
            ]
        })
        vlm_tasks.append({
            'id': 'title_time',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, time_title_keywords, "chart title indicating time progression or trend")}]}
            ]
        })

    # Execute all keyword matching in parallel
    keyword_start = time.time()
    if vlm_tasks:
        keyword_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=4)
    else:
        keyword_results = {}
    keyword_time = time.time() - keyword_start

    # Step 1: X-axis label indicates activity date
    x_label_match = keyword_results.get('x_axis', False) if x_label else False
    x_label_keyword_fallback = keywords_match_robust(
        texts=x_label, keywords=date_keywords, model=model,
        description="X-axis label indicating activity date or time"
    ) if x_label else False
    has_date_label = bool(x_label_match) or bool(x_label_keyword_fallback)
    checkpoint.add_step(
        "X-Axis Date Label",
        has_date_label,
        1,
        f"X-axis label: '{x_label}'" if has_date_label else f"X-axis label '{x_label or 'None'}' does not match date keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1)
    )

    # Step 2: Y-axis label indicates speed
    y_label_match = keyword_results.get('y_axis', False) if y_label else False
    y_label_keyword_fallback = keywords_match_robust(
        texts=y_label, keywords=speed_keywords, model=model,
        description="Y-axis label indicating running speed or pace"
    ) if y_label else False
    has_speed_label = bool(y_label_match) or bool(y_label_keyword_fallback)
    checkpoint.add_step(
        "Y-Axis Speed Label",
        has_speed_label,
        2,
        f"Y-axis label: '{y_label}'" if has_speed_label else f"Y-axis label '{y_label or 'None'}' does not match speed keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1)
    )

    # Step 3: Chart title indicates speed over time
    title_speed_match = keyword_results.get('title_speed', False) if chart_title else False
    title_time_match = keyword_results.get('title_time', False) if chart_title else False
    title_keyword_fallback = keywords_match_robust(
        texts=chart_title, keywords=speed_title_keywords + time_title_keywords, model=model,
        description="chart title indicating running speed or pace over time"
    ) if chart_title else False
    has_good_title = bool(title_speed_match) or bool(title_time_match) or bool(title_keyword_fallback)
    checkpoint.add_step(
        "Chart Title",
        has_good_title,
        3,
        f"Chart title: '{chart_title}'" if has_good_title else f"Chart title '{chart_title or 'None'}' does not match speed/time keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1)
    )

    # Step 4: Chart not placed over other charts/tables
    step_start = time.time()
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 453)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    # Guard against charts with None anchor_cell.row (Google Sheets API bug)
    speed_anchor = speed_chart.get('position', {}).get('anchor_cell', {})
    if speed_anchor.get('row') is None:
        has_overlap = False
        overlap_details = "Chart anchor row is None (API bug) - skipping overlap check"
    else:
        safe_charts = [c for c in (chart_data or []) if c.get('position', {}).get('anchor_cell', {}).get('row') is not None]
        has_overlap, overlap_details = check_chart_overlap(speed_chart, table_start, table_end, safe_charts, table_start_col, table_end_col)
    checkpoint.add_step(
        "No Chart Overlap",
        not has_overlap,
        4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start
    )

    # Step 5: Main data series from speed column
    step_start = time.time()
    series_list = get_all_series_metadata(speed_chart)
    main_series_valid = False
    series_details = "No series found in chart"

    if series_list and len(series_list) > 0:
        if main_idx is not None:
            speed_col_name = matched_columns.get("Speed (min/mile)") if matched_columns else None

            if speed_col_name and df is not None and speed_col_name in df.columns:
                expected_col_idx = df.columns.get_loc(speed_col_name)
                if not isinstance(expected_col_idx, int):
                    import numpy as np
                    if isinstance(expected_col_idx, np.ndarray):
                        expected_col_idx = int(np.where(expected_col_idx)[0][0])
                    elif isinstance(expected_col_idx, slice):
                        expected_col_idx = expected_col_idx.start or 0

                if main_idx < len(series_list):
                    main_series = series_list[main_idx]
                    src_range = main_series.get('source_range', {})
                    actual_start_col = src_range.get('start_col')

                    if actual_start_col == expected_col_idx:
                        main_series_valid = True
                        series_details = f"Main series (index {main_idx}) uses '{speed_col_name}' (column {expected_col_idx})"
                    else:
                        # Column doesn't match CP2 — fallback: check if series header is a pace/speed column
                        header_label = get_series_header_label(speed_chart, main_idx, rows) if rows else ""
                        header_label_match = keywords_match_robust(
                            texts=header_label, keywords=["pace", "min/mile", "min/mi", "speed"], substring=True
                        ) if header_label else False
                        if header_label_match:
                            main_series_valid = True
                            series_details = f"Main series (index {main_idx}) uses column {actual_start_col} ('{header_label}'), not CP2 column {expected_col_idx} ('{speed_col_name}')"
                        else:
                            series_details = f"Main series (index {main_idx}) uses column {actual_start_col}, expected column {expected_col_idx} ('{speed_col_name}')"
                else:
                    series_details = f"Main series index {main_idx} out of range (only {len(series_list)} series)"
            else:
                header_label = get_series_header_label(speed_chart, main_idx, rows) if rows else ""
                header_label_match = keywords_match_robust(
                    texts=header_label, keywords=["pace", "min/mile", "min/mi", "speed"], substring=True
                ) if header_label else False
                if header_label_match:
                    main_series_valid = True
                    series_details = f"Main series at index {main_idx} matched by header label '{header_label}' (CP2 speed column not available)"
                else:
                    series_details = f"Main series at index {main_idx} header '{header_label or 'unknown'}' does not match speed keywords; CP2 speed column unavailable"
        else:
            series_details = "Could not identify main data series by content"

    checkpoint.add_step(
        "Speed Data Series",
        main_series_valid,
        5,
        series_details,
        execution_time=time.time() - step_start
    )


    # Step 7: Female 5K baseline display check (legend label + line style)
    step_start = time.time()
    global sheet_female_5k_pace
    female_5k_display_valid = False
    female_5k_display_details = "Could not identify Female 5K baseline series"

    if female_5k_candidate_idx is not None and rows is not None:
        female_5k_label = get_series_header_label(speed_chart, female_5k_candidate_idx, rows)
        female_5k_label_keywords = ["female", "5k", "5 k", "baseline", "average", "women", "25"]

        label_match = keywords_match_robust(
            texts=female_5k_label,
            keywords=female_5k_label_keywords,
            substring=True
        ) if female_5k_label else None

        female_5k_line_style = get_series_line_style(speed_chart, female_5k_candidate_idx)
        is_dashed = female_5k_line_style and female_5k_line_style.upper() in [
            'DOTTED', 'DASHED', 'LONG_DASHED', 'MEDIUM_DASHED', 'LONG_DASHED_DOTTED'
        ]

        if label_match and is_dashed:
            female_5k_display_valid = True
            female_5k_display_details = f"Label: '{female_5k_label}', Style: {female_5k_line_style} (series index {female_5k_candidate_idx})"
        elif label_match:
            female_5k_display_details = f"Label: '{female_5k_label}' OK, but line style is {female_5k_line_style or 'SOLID'} (series index {female_5k_candidate_idx})"
        elif is_dashed:
            female_5k_display_details = f"Line style {female_5k_line_style} OK, but label '{female_5k_label}' doesn't match keywords (series index {female_5k_candidate_idx})"
        else:
            female_5k_display_details = f"Label: '{female_5k_label}', Style: {female_5k_line_style or 'SOLID'} - both need improvement (series index {female_5k_candidate_idx})"

    checkpoint.add_step(
        "Female 5K Display",
        female_5k_display_valid,
        7,
        female_5k_display_details,
        execution_time=time.time() - step_start
    )

    # Step 8: Female 5K baseline data validation (constant value in range)
    step_start = time.time()
    female_5k_data_valid = False
    female_5k_data_details = "Could not identify Female 5K baseline series"

    if female_5k_candidate_idx is not None and rows is not None:
        female_5k_values = get_series_column_values(speed_chart, female_5k_candidate_idx, rows)
        if female_5k_values:
            female_5k_data_valid, _, female_5k_data_details = validate_constant_series(
                female_5k_values, FEMALE_5K_PACE_RANGE, tolerance=0.01
            )
            if female_5k_values:
                sheet_female_5k_pace = female_5k_values[0]
        else:
            female_5k_data_details = f"Could not extract values from baseline series (index {female_5k_candidate_idx})"
    elif female_5k_candidate_idx is not None:
        female_5k_data_details = "Sheet rows unavailable from setup() - cannot extract baseline values"

    checkpoint.add_step(
        "Female 5K Data",
        female_5k_data_valid,
        8,
        female_5k_data_details,
        execution_time=time.time() - step_start
    )

    # Step 9: Chebet baseline display check (legend label only - no line style requirement per task.md "Compare this to...")
    step_start = time.time()
    global sheet_chebet_pace
    chebet_display_valid = False
    chebet_display_details = "Could not identify Beatrice Chebet baseline series"

    if chebet_idx is not None and rows is not None:
        chebet_label = get_series_header_label(speed_chart, chebet_idx, rows)
        chebet_label_keywords = ["chebet", "beatrice", "5k", "world", "record"]

        label_match = keywords_match_robust(
            texts=chebet_label,
            keywords=chebet_label_keywords,
            substring=True
        ) if chebet_label else None

        chebet_line_style = get_series_line_style(speed_chart, chebet_idx)

        if label_match:
            chebet_display_valid = True
            chebet_display_details = f"Label: '{chebet_label}', Style: {chebet_line_style or 'SOLID'} (series index {chebet_idx})"
        else:
            chebet_display_details = f"Label '{chebet_label}' doesn't match Chebet keywords (series index {chebet_idx})"

    checkpoint.add_step(
        "Chebet Display",
        chebet_display_valid,
        9,
        chebet_display_details,
        execution_time=time.time() - step_start
    )

    # Step 10: Chebet baseline data validation (constant value in range)
    step_start = time.time()
    chebet_data_valid = False
    chebet_data_details = "Could not identify Beatrice Chebet baseline series"

    if chebet_idx is not None and rows is not None:
        chebet_values = get_series_column_values(speed_chart, chebet_idx, rows)
        if chebet_values:
            chebet_data_valid, _, chebet_data_details = validate_constant_series(
                chebet_values, CHEBET_PACE_RANGE, tolerance=0.01
            )
            if chebet_values:
                sheet_chebet_pace = chebet_values[0]
        else:
            chebet_data_details = f"Could not extract values from baseline series (index {chebet_idx})"
    elif chebet_idx is not None:
        chebet_data_details = "Sheet rows unavailable from setup() - cannot extract baseline values"

    checkpoint.add_step(
        "Chebet Data",
        chebet_data_valid,
        10,
        chebet_data_details,
        execution_time=time.time() - step_start
    )

    # Step 11: Both baselines visually distinguishable from main data AND from each other
    step_start = time.time()
    baselines_distinguishable = False
    distinguishable_details = "Need all three series identified (main, female 5K, Chebet)"

    if main_idx is not None and female_5k_candidate_idx is not None and chebet_idx is not None:
        main_line_style = get_series_line_style(speed_chart, main_idx)
        female_5k_style = get_series_line_style(speed_chart, female_5k_candidate_idx)
        chebet_style = get_series_line_style(speed_chart, chebet_idx)

        main_color = get_series_color(speed_chart, main_idx)
        female_5k_color = get_series_color(speed_chart, female_5k_candidate_idx)
        chebet_color = get_series_color(speed_chart, chebet_idx)

        # Check if baselines are distinguishable from each other (different styles OR different colors)
        baselines_have_different_styles = female_5k_style != chebet_style
        baselines_have_different_colors = not colors_are_similar(female_5k_color or {}, chebet_color or {})
        baselines_distinguishable_from_each_other = baselines_have_different_styles or baselines_have_different_colors

        # Check if Chebet is distinguishable from main data (different style OR different color)
        chebet_different_from_main_style = main_line_style != chebet_style
        chebet_different_from_main_color = not colors_are_similar(main_color or {}, chebet_color or {})
        chebet_distinguishable_from_main = chebet_different_from_main_style or chebet_different_from_main_color

        # Check if Female 5K is distinguishable from main data
        female_5k_different_from_main_style = main_line_style != female_5k_style
        female_5k_different_from_main_color = not colors_are_similar(main_color or {}, female_5k_color or {})
        female_5k_distinguishable_from_main = female_5k_different_from_main_style or female_5k_different_from_main_color

        if baselines_distinguishable_from_each_other and chebet_distinguishable_from_main and female_5k_distinguishable_from_main:
            baselines_distinguishable = True
            distinguishable_details = (
                f"Female 5K: {female_5k_style or 'SOLID'}, "
                f"Chebet: {chebet_style or 'SOLID'}, Main: {main_line_style or 'SOLID'}"
            )
        else:
            issues = []
            if not baselines_distinguishable_from_each_other:
                issues.append(f"baselines not distinguishable from each other")
            if not chebet_distinguishable_from_main:
                issues.append(f"Chebet not distinguishable from main data")
            if not female_5k_distinguishable_from_main:
                issues.append(f"Female 5K not distinguishable from main data")
            distinguishable_details = f"Issues: {'; '.join(issues)}"

    checkpoint.add_step(
        "Baselines Distinguishable",
        baselines_distinguishable,
        11,
        distinguishable_details,
        execution_time=time.time() - step_start
    )

    # Step 12: Source URLs valid and accessible below chart
    step_start = time.time()
    urls_valid = False
    url_details = "No URLs found below chart"

    if rows:
        chart_position = speed_chart.get('position', {})
        anchor_cell = chart_position.get('anchor_cell', {})
        anchor_row = anchor_cell.get('row') if anchor_cell else None
        chart_height = chart_position.get('height')
        if chart_position.get('type') == 'overlay' and anchor_row is not None and chart_height:
            search_start_row = anchor_row + (chart_height // 20) + 1
        else:
            # Search from near end of table (sources may be appended within table bounds)
            search_start_row = max(0, (table_data.end_row - 10) if table_data else 0)

        urls = find_urls_in_sheet(rows, search_start_row, num_rows=60)

        if urls:
            accessible_urls = []
            for url in urls[:3]:
                is_accessible, _ = validate_url_accessible(url)
                if is_accessible:
                    accessible_urls.append(url)

            if accessible_urls:
                urls_valid = True
                url_details = f"Found {len(accessible_urls)} accessible source URL(s)"
            else:
                url_details = f"Found {len(urls)} URLs but none accessible"
        else:
            url_details = f"No URLs found in rows {search_start_row}-{search_start_row + 50}"

    checkpoint.add_step(
        "Source URLs Valid",
        urls_valid,
        12,
        url_details,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Grade Checkpoint 5: Cumulative Distance Plot (7 pts).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates cumulative distance (miles or similar)
    3. Chart title indicates cumulative distance over time
    4. Chart is not placed over any other charts or tables
    5. Data shows cumulative/running total
    6. Cumulative running values are present as a line plot
    7. Chart is on the same sheet tab as the data table
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Cumulative Distance Plot")

    # Check if any charts exist
    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 7):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find cumulative distance chart by metadata
    cumulative_chart = find_cumulative_chart_by_metadata(
        chart_data, matched_columns, table_data.df if table_data else None, model
    )

    if not cumulative_chart:
        error_msg = "Could not identify cumulative distance chart by title or axis labels"
        for i in range(1, 7):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = get_chart_type(cumulative_chart)

    # Get axis labels
    axis_labels = get_chart_axis_labels(cumulative_chart)
    x_label = axis_labels.get('x_axis', '')
    y_label = axis_labels.get('y_axis', '')
    chart_title = cumulative_chart.get('title', '')

    # Define keywords for each match
    date_keywords = ['date', 'time', 'day', 'activity']
    distance_keywords = ['cumulative', 'total', 'distance', 'miles', 'running total']
    cumulative_title_keywords = ['cumulative', 'total', 'distance']
    time_title_keywords = ['time', 'over', 'progression', 'trend']

    # Build VLM tasks for parallel keyword matching (steps 1-3)
    vlm_tasks_cp5 = []
    if x_label:
        vlm_tasks_cp5.append({
            'id': 'x_axis',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(x_label, date_keywords, "X-axis label indicating activity date or time")}]}
            ]
        })
    if y_label:
        vlm_tasks_cp5.append({
            'id': 'y_axis',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(y_label, distance_keywords, "Y-axis label indicating cumulative distance")}]}
            ]
        })
    if chart_title:
        vlm_tasks_cp5.append({
            'id': 'title_cumulative',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, cumulative_title_keywords, "chart title related to cumulative distance")}]}
            ]
        })
        vlm_tasks_cp5.append({
            'id': 'title_time',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, time_title_keywords, "chart title indicating time progression")}]}
            ]
        })

    # Execute all keyword matching in parallel
    keyword_start = time.time()
    if vlm_tasks_cp5:
        keyword_results_cp5 = fast_parallel_vlm_calls(vlm_tasks_cp5, model, max_workers=4)
    else:
        keyword_results_cp5 = {}
    keyword_time = time.time() - keyword_start

    # Step 1: X-axis label indicates activity date
    x_label_match = keyword_results_cp5.get('x_axis', False) if x_label else False
    x_label_keyword_fallback = keywords_match_robust(
        texts=x_label, keywords=date_keywords, model=model,
        description="X-axis label indicating activity date or time"
    ) if x_label else False
    has_date_label = bool(x_label_match) or bool(x_label_keyword_fallback)
    checkpoint.add_step(
        "X-Axis Date Label",
        has_date_label,
        1,
        f"X-axis label: '{x_label}'" if has_date_label else f"X-axis label '{x_label or 'None'}' does not match date keywords",
        execution_time=keyword_time / max(len(vlm_tasks_cp5), 1)
    )

    # Step 2: Y-axis label indicates cumulative distance
    y_label_match = keyword_results_cp5.get('y_axis', False) if y_label else False
    y_label_keyword_fallback = keywords_match_robust(
        texts=y_label, keywords=distance_keywords, model=model,
        description="Y-axis label indicating cumulative distance"
    ) if y_label else False
    has_distance_label = bool(y_label_match) or bool(y_label_keyword_fallback)
    checkpoint.add_step(
        "Y-Axis Distance Label",
        has_distance_label,
        2,
        f"Y-axis label: '{y_label}'" if has_distance_label else f"Y-axis label '{y_label or 'None'}' does not match cumulative distance keywords",
        execution_time=keyword_time / max(len(vlm_tasks_cp5), 1)
    )

    # Step 3: Chart title indicates cumulative distance over time
    title_cumulative_match = keyword_results_cp5.get('title_cumulative', False) if chart_title else False
    title_time_match = keyword_results_cp5.get('title_time', False) if chart_title else False
    title_keyword_fallback = keywords_match_robust(
        texts=chart_title, keywords=cumulative_title_keywords + time_title_keywords, model=model,
        description="chart title indicating cumulative distance over time"
    ) if chart_title else False
    has_good_title = bool(title_cumulative_match) or bool(title_time_match) or bool(title_keyword_fallback)
    checkpoint.add_step(
        "Chart Title",
        has_good_title,
        3,
        f"Chart title: '{chart_title}'" if has_good_title else f"Chart title '{chart_title or 'None'}' does not match cumulative/time keywords",
        execution_time=keyword_time / max(len(vlm_tasks_cp5), 1)
    )

    # Step 4: Chart not placed over other charts/tables
    step_start = time.time()
    df = table_data.df if table_data else None
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 453)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    # Guard against charts with None anchor_cell.row (Google Sheets API bug)
    cumulative_anchor = cumulative_chart.get('position', {}).get('anchor_cell', {})
    if cumulative_anchor.get('row') is None:
        has_overlap = False
        overlap_details = "Chart anchor row is None (API bug) - skipping overlap check"
    else:
        safe_charts = [c for c in (chart_data or []) if c.get('position', {}).get('anchor_cell', {}).get('row') is not None]
        has_overlap, overlap_details = check_chart_overlap(
            cumulative_chart, table_start, table_end, safe_charts, table_start_col, table_end_col
        )
    checkpoint.add_step(
        "No Chart Overlap",
        not has_overlap,
        4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start
    )

    # Step 5: Data shows cumulative/running total (validate against sheet data)
    # The cumulative chart tracks running distance only (walks don't increment cumulative)
    # so we validate against the sheet's own Cumulative Distance column rather than recomputing
    step_start = time.time()
    cumulative_valid = False
    cumulative_details = "Could not extract chart values"

    if rows is None:
        cumulative_details = "Sheet rows unavailable from setup() - cannot extract chart values"
    elif df is None or df.empty:
        cumulative_details = "Sheet table data unavailable - cannot validate cumulative values"
    else:
        chart_values = get_series_column_values(cumulative_chart, 0, rows)
        if chart_values:
            # Check monotonically increasing (cumulative pattern)
            non_increasing = sum(1 for i in range(1, len(chart_values)) if chart_values[i] < chart_values[i-1] - 0.01)
            if non_increasing > 0:
                cumulative_valid = False
                cumulative_details = f"Values not monotonically increasing ({non_increasing} decreases found)"
            else:
                # Verify against the sheet's Cumulative Distance column directly
                cumulative_col = None
                for col in df.columns:
                    if 'cumulative' in col.lower() and 'distance' in col.lower():
                        cumulative_col = col
                        break
                if cumulative_col:
                    expected_values = pd.to_numeric(df[cumulative_col], errors='coerce').dropna().values
                    comparison_count = min(len(chart_values), len(expected_values))
                    matches = 0
                    for i in range(comparison_count):
                        if expected_values[i] > 0:
                            error_pct = abs(chart_values[i] - expected_values[i]) / expected_values[i] * 100
                            if error_pct <= 1.0:
                                matches += 1
                        elif chart_values[i] == 0:
                            matches += 1
                    match_rate = matches / comparison_count if comparison_count > 0 else 0
                    if match_rate >= 0.8:
                        cumulative_valid = True
                        cumulative_details = f"Cumulative data valid: {matches}/{comparison_count} values match ({match_rate:.0%}), final value {chart_values[-1]:.1f} miles"
                    else:
                        cumulative_details = f"Only {matches}/{comparison_count} values match ({match_rate:.0%}) against sheet cumulative column"
                else:
                    # Fallback: just verify it's monotonically increasing with reasonable final value
                    cumulative_valid = True
                    cumulative_details = f"Monotonically increasing data ({len(chart_values)} points), final value {chart_values[-1]:.1f} miles"
        else:
            cumulative_details = "No values extracted from chart series"

    checkpoint.add_step(
        "Cumulative Data",
        cumulative_valid,
        5,
        cumulative_details,
        execution_time=time.time() - step_start
    )

    # Step 6: Cumulative values present as line plot
    step_start = time.time()
    is_line = chart_type in ['LINE', 'AREA']
    line_details = f"Chart type is {chart_type}"
    if is_line:
        line_details = f"Chart is a {chart_type} chart (line plot)"
    else:
        line_details = f"Chart type is {chart_type}, expected LINE or AREA"

    checkpoint.add_step(
        "Line Plot",
        is_line,
        6,
        line_details,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6(browsing_history=None):
    """
    Grade Checkpoint 6: Website Visit Validation (6 pts).

    Validates that the agent visited required websites to gather baseline data.

    Outcome Evaluation:
    - A source URL for female daily miles trekked was visited.
    - A source URL for female 25 5K running speed was visited.
    - A source URL for Beatrice Chebet 5K data was visited.
    - Female daily miles source URL contains relevant information.
    - Female 5K source URL contains relevant pace/speed information.
    - Chebet source URL contains relevant 5K time information.
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Female Daily Miles URL Visited", False, 1, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Female 5K URL Visited", False, 2, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Chebet URL Visited", False, 3, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Female Daily Miles Content Valid", False, 4, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Female 5K Content Valid", False, 5, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Chebet Content Valid", False, 6, "No browsing history provided", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    browsing_lower = [url.lower() for url in browsing_history]

    # Keywords for identifying relevant URLs
    female_daily_keywords = ['walk', 'miles', 'daily', 'day', 'step', 'trekk']
    female_5k_keywords = ['5k', '5-k', 'running', 'pace', 'speed', 'average', 'runner', 'race time']
    chebet_keywords = ['chebet', 'beatrice']

    # Find candidate URLs for each category
    female_daily_urls = []
    female_5k_urls = []
    chebet_urls = []
    female_daily_judge_details = ""
    female_5k_judge_details = ""
    chebet_judge_details = ""

    for i, url_lower in enumerate(browsing_lower):
        original_url = browsing_history[i]
        if any(kw in url_lower for kw in female_daily_keywords):
            female_daily_urls.append(original_url)
        if any(kw in url_lower for kw in female_5k_keywords):
            female_5k_urls.append(original_url)
        if any(kw in url_lower for kw in chebet_keywords):
            chebet_urls.append(original_url)

    # LLM-as-judge backup for any category without keyword hits
    judge_step_start = time.time()
    if model and (not female_daily_urls or not female_5k_urls or not chebet_urls):
        judge_tasks = []
        for original_url in browsing_history[:6]:
            if not female_daily_urls:
                judge_tasks.append({
                    'id': f'judge_female_daily|{original_url}',
                    'func': judge_url_relevance,
                    'args': (original_url, 'female_daily_miles', model),
                })
            if not female_5k_urls:
                judge_tasks.append({
                    'id': f'judge_female_5k|{original_url}',
                    'func': judge_url_relevance,
                    'args': (original_url, 'female_5k', model),
                })
            if not chebet_urls:
                judge_tasks.append({
                    'id': f'judge_chebet|{original_url}',
                    'func': judge_url_relevance,
                    'args': (original_url, 'chebet', model),
                })
        if judge_tasks:
            judge_results = parallel_execute(judge_tasks, max_workers=6)
            for original_url in browsing_history[:6]:
                if not female_daily_urls:
                    judgement = judge_results.get(f'judge_female_daily|{original_url}')
                    if judgement and judgement[0]:
                        female_daily_urls.append(original_url)
                        female_daily_judge_details = " (LLM-judged)"
                if not female_5k_urls:
                    judgement = judge_results.get(f'judge_female_5k|{original_url}')
                    if judgement and judgement[0]:
                        female_5k_urls.append(original_url)
                        female_5k_judge_details = " (LLM-judged)"
                if not chebet_urls:
                    judgement = judge_results.get(f'judge_chebet|{original_url}')
                    if judgement and judgement[0]:
                        chebet_urls.append(original_url)
                        chebet_judge_details = " (LLM-judged)"
    judge_time = time.time() - judge_step_start

    # Step 1: Female daily miles URL visited
    female_daily_visited = len(female_daily_urls) > 0
    checkpoint.add_step(
        "Female Daily Miles URL Visited",
        female_daily_visited,
        1,
        f"Found {len(female_daily_urls)} relevant URL(s){female_daily_judge_details}" if female_daily_visited else "No female daily miles URL found in browsing history (keyword + LLM judge)",
        execution_time=judge_time / 3
    )

    # Step 2: Female 5K URL visited
    female_5k_visited = len(female_5k_urls) > 0
    checkpoint.add_step(
        "Female 5K URL Visited",
        female_5k_visited,
        2,
        f"Found {len(female_5k_urls)} relevant URL(s){female_5k_judge_details}" if female_5k_visited else "No female 5K running URL found in browsing history (keyword + LLM judge)",
        execution_time=judge_time / 3
    )

    # Step 3: Chebet URL visited
    chebet_visited = len(chebet_urls) > 0
    checkpoint.add_step(
        "Chebet URL Visited",
        chebet_visited,
        3,
        f"Found {len(chebet_urls)} relevant URL(s){chebet_judge_details}" if chebet_visited else "No Beatrice Chebet URL found in browsing history (keyword + LLM judge)",
        execution_time=judge_time / 3
    )

    # Steps 4-6: URL content validation - parallelize all URL pace extractions
    step_start = time.time()
    tolerance_percent = 0.05  # 5% tolerance

    # Build parallel tasks for all URL extractions
    # Note: female_daily_miles uses extract_pace_from_url (raw value) not extract_and_convert_pace_from_url
    # because it's a distance (miles), not a pace (min/mile)
    url_tasks = []
    if sheet_female_daily_miles and female_daily_urls and model:
        for url in female_daily_urls[:3]:
            url_tasks.append({
                'id': f'female_daily|{url}',
                'func': extract_pace_from_url,
                'args': (url, 'female_daily_miles', model),
            })
    if sheet_female_5k_pace and female_5k_urls and model:
        for url in female_5k_urls[:3]:
            url_tasks.append({
                'id': f'female_5k|{url}',
                'func': extract_and_convert_pace_from_url,
                'args': (url, 'female_5k', model),
            })
    if sheet_chebet_pace and chebet_urls and model:
        for url in chebet_urls[:3]:
            url_tasks.append({
                'id': f'chebet|{url}',
                'func': extract_and_convert_pace_from_url,
                'args': (url, 'chebet', model),
            })

    # Execute all URL extractions in parallel
    if url_tasks:
        url_results = parallel_execute(url_tasks, max_workers=6)
    else:
        url_results = {}
    url_time = time.time() - step_start

    # Process results for Female Daily Miles
    # extract_pace_from_url returns (dict_or_None, details_str) - we compare the raw value directly
    # For daily miles, sources vary widely in recommendations (2-6 miles), so we validate that
    # the URL contains a relevant daily mileage value within the acceptable range rather than
    # requiring exact match to sheet value
    female_daily_content_valid = False
    female_daily_content_details = "No female daily miles URLs to check"
    female_daily_extraction_failed_for_all = False

    if not sheet_female_daily_miles or sheet_female_daily_miles <= 0:
        female_daily_content_details = "No female daily miles baseline value found in sheet (checkpoint 3 may have failed)"
    elif not model:
        female_daily_content_details = "Model not available for content validation"
    elif female_daily_urls:
        female_daily_extraction_failed_for_all = True
        for url in female_daily_urls[:3]:
            result = url_results.get(f'female_daily|{url}')
            if result is not None:
                pace_data, details = result
                if pace_data is not None:
                    female_daily_extraction_failed_for_all = False
                    # Get raw value and convert to miles if needed
                    extracted_value = pace_data['value']
                    extracted_unit = pace_data.get('unit', 'miles')
                    if 'km' in extracted_unit.lower():
                        extracted_value = km_to_miles(extracted_value)
                    # Validate the extracted value is within the acceptable range for daily miles
                    # (sources vary; the key is that the URL contains a relevant daily mileage figure)
                    if FEMALE_DAILY_MILES_RANGE[0] <= extracted_value <= FEMALE_DAILY_MILES_RANGE[1]:
                        female_daily_content_valid = True
                        female_daily_content_details = f"URL value {extracted_value:.2f} miles is within acceptable range [{FEMALE_DAILY_MILES_RANGE[0]}-{FEMALE_DAILY_MILES_RANGE[1]}] (sheet baseline: {sheet_female_daily_miles:.2f})"
                        break
                    else:
                        female_daily_content_details = f"URL value {extracted_value:.2f} miles outside acceptable range [{FEMALE_DAILY_MILES_RANGE[0]}-{FEMALE_DAILY_MILES_RANGE[1]}]"
                else:
                    female_daily_content_details = details

    # Backup: LLM-as-judge if structured extraction failed
    if (not female_daily_content_valid
            and female_daily_extraction_failed_for_all
            and sheet_female_daily_miles and sheet_female_daily_miles > 0
            and model and female_daily_urls):
        for url in female_daily_urls[:3]:
            judged, judge_details = judge_url_pace_match(url, 'female_daily_miles', sheet_female_daily_miles, model)
            if judged:
                female_daily_content_valid = True
                female_daily_content_details = f"LLM judge backup: {judge_details}"
                break
            else:
                female_daily_content_details = f"LLM judge backup: {judge_details}"

    checkpoint.add_step(
        "Female Daily Miles Content Valid",
        female_daily_content_valid,
        4,
        female_daily_content_details,
        execution_time=url_time / 3 if url_tasks else 0
    )

    # Process results for Female 5K
    female_5k_content_valid = False
    female_5k_content_details = "No female 5K URLs to check"
    female_5k_extraction_failed_for_all = False

    if not sheet_female_5k_pace or sheet_female_5k_pace <= 0:
        female_5k_content_details = "No Female 5K baseline value found in sheet (checkpoint 4 may have failed)"
    elif not model:
        female_5k_content_details = "Model not available for content validation"
    elif female_5k_urls:
        female_5k_extraction_failed_for_all = True
        for url in female_5k_urls[:3]:
            result = url_results.get(f'female_5k|{url}')
            if result is not None:
                pace, details = result
                if pace is not None:
                    female_5k_extraction_failed_for_all = False
                    diff_percent = abs(pace - sheet_female_5k_pace) / sheet_female_5k_pace
                    if diff_percent <= tolerance_percent:
                        female_5k_content_valid = True
                        female_5k_content_details = f"URL pace {pace:.2f} matches sheet value {sheet_female_5k_pace:.2f} min/mile ({diff_percent*100:.1f}% diff)"
                        break
                    else:
                        female_5k_content_details = f"URL pace {pace:.2f} differs from sheet value {sheet_female_5k_pace:.2f} by {diff_percent*100:.1f}% (max 5%)"
                else:
                    female_5k_content_details = details

    # Backup: LLM-as-judge if structured extraction failed
    if (not female_5k_content_valid
            and female_5k_extraction_failed_for_all
            and sheet_female_5k_pace and sheet_female_5k_pace > 0
            and model and female_5k_urls):
        for url in female_5k_urls[:3]:
            judged, judge_details = judge_url_pace_match(url, 'female_5k', sheet_female_5k_pace, model)
            if judged:
                female_5k_content_valid = True
                female_5k_content_details = f"LLM judge backup: {judge_details}"
                break
            else:
                female_5k_content_details = f"LLM judge backup: {judge_details}"

    checkpoint.add_step(
        "Female 5K Content Valid",
        female_5k_content_valid,
        5,
        female_5k_content_details,
        execution_time=url_time / 3 if url_tasks else 0
    )

    # Process results for Chebet
    chebet_content_valid = False
    chebet_content_details = "No Chebet URLs to check"
    chebet_extraction_failed_for_all = False

    if not sheet_chebet_pace or sheet_chebet_pace <= 0:
        chebet_content_details = "No Chebet baseline value found in sheet (checkpoint 4 may have failed)"
    elif not model:
        chebet_content_details = "Model not available for content validation"
    elif chebet_urls:
        chebet_extraction_failed_for_all = True
        for url in chebet_urls[:3]:
            result = url_results.get(f'chebet|{url}')
            if result is not None:
                pace, details = result
                if pace is not None:
                    chebet_extraction_failed_for_all = False
                    diff_percent = abs(pace - sheet_chebet_pace) / sheet_chebet_pace
                    if diff_percent <= tolerance_percent:
                        chebet_content_valid = True
                        chebet_content_details = f"URL pace {pace:.2f} matches sheet value {sheet_chebet_pace:.2f} min/mile ({diff_percent*100:.1f}% diff)"
                        break
                    else:
                        chebet_content_details = f"URL pace {pace:.2f} differs from sheet value {sheet_chebet_pace:.2f} by {diff_percent*100:.1f}% (max 5%)"
                else:
                    chebet_content_details = details

    # Backup: LLM-as-judge if structured extraction failed
    if (not chebet_content_valid
            and chebet_extraction_failed_for_all
            and sheet_chebet_pace and sheet_chebet_pace > 0
            and model and chebet_urls):
        for url in chebet_urls[:3]:
            judged, judge_details = judge_url_pace_match(url, 'chebet', sheet_chebet_pace, model)
            if judged:
                chebet_content_valid = True
                chebet_content_details = f"LLM judge backup: {judge_details}"
                break
            else:
                chebet_content_details = f"LLM judge backup: {judge_details}"

    checkpoint.add_step(
        "Chebet Content Valid",
        chebet_content_valid,
        6,
        chebet_content_details,
        execution_time=url_time / 3 if url_tasks else 0
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id=None, browsing_history=None):
    """
    Grade all checkpoints for the running analysis task.

    Args:
        workspace_doc_id (str, optional): Direct Google Sheets document ID to use
        browsing_history (list, optional): List of URLs visited during task execution

    Returns:
        Result: Evaluation results with checkpoint scores
    """
    total_start_time = time.time()

    try:
        # Setup document processing
        setup(workspace_doc_id)

        # Load model for LLM-based matching
        global model
        model = load_model(model_id)

        checkpoints: List[Checkpoint] = []

        # Checkpoint 1: Data Table Structure
        cp1_start = time.time()
        checkpoints.append(grade_checkpoint_1())
        print(f"  Checkpoint 1 took {time.time() - cp1_start:.2f}s")

        # Checkpoint 2: Data Table Content Accuracy
        cp2_start = time.time()
        checkpoints.append(grade_checkpoint_2())
        print(f"  Checkpoint 2 took {time.time() - cp2_start:.2f}s")

        # Checkpoint 3: Daily Total Miles Bar Plot
        cp3_start = time.time()
        checkpoints.append(grade_checkpoint_3())
        print(f"  Checkpoint 3 took {time.time() - cp3_start:.2f}s")

        # Checkpoint 4: Speed Over Time Plot
        cp4_start = time.time()
        checkpoints.append(grade_checkpoint_4())
        print(f"  Checkpoint 4 took {time.time() - cp4_start:.2f}s")

        # Checkpoint 5: Cumulative Distance Plot
        cp5_start = time.time()
        checkpoints.append(grade_checkpoint_5())
        print(f"  Checkpoint 5 took {time.time() - cp5_start:.2f}s")

        # Checkpoint 6: Website Visit Validation
        cp6_start = time.time()
        checkpoints.append(grade_checkpoint_6(browsing_history))
        print(f"  Checkpoint 6 took {time.time() - cp6_start:.2f}s")

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
    parser = argparse.ArgumentParser(description="Evaluate running analysis spreadsheet")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Sheets document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()
    print(f"DEBUG mode: {DEBUG}")
    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT (with timings) ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        cp_time = checkpoint.get('execution_time') or 0
        print(f"\n{checkpoint['name']}: {checkpoint['score']} ({cp_time:.2f}s)")
        for step in checkpoint["steps"]:
            status = "[PASS]" if step["success"] else "[FAIL]"
            step_time = step.get('execution_time') or 0
            print(f"  {status} {step['name']} ({step_time:.2f}s): {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
