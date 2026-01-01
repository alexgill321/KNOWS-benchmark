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
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.google_sheets_utils import (
    extract_charts_from_sheet,
    extract_sheet_data,
    find_urls_in_sheet,
)
from src.browsergym.eval.eval_utils.web_utils import validate_url_accessible
from src.browsergym.eval.eval_utils.table_utils import (
    match_columns,
    colors_are_similar,
    check_all_content_visible,
)
from src.browsergym.eval.eval_utils.text_utils import keywords_match_robust
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.chart_utils import (
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

# Task-specific utilities
from src.browsergym.eval.tasks.sheets_7_running_analysis.utils import (
    normalize_date,
    load_gold_run_activities,
    find_speed_chart_by_metadata,
    find_cumulative_chart_by_metadata,
    validate_cumulative_against_sheet,
    extract_and_convert_pace_from_url,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/sheets_7_running_analysis/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Gold baseline values for checkpoint 3
MALE_5K_PACE_RANGE = (8.0, 10.0)  # min/mile for 25yo male
KIPCHOGE_PACE_RANGE = (4.6, 4.65)  # min/mile (4:36-4:39, based on top 3 marathons)

model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global variables
sheet_id = None
sheet_raw = None
table_data = None  # SheetTable object with position metadata
rows = None  # Raw row data from sheet
matched_columns = None  # Shared across checkpoints
chart_data = None  # All charts extracted from the sheet
sheet_male_5k_pace = None  # Male 5K baseline value from sheet (for checkpoint 5)
sheet_kipchoge_pace = None  # Kipchoge baseline value from sheet (for checkpoint 5)


def setup(workspace_doc_id):
    """
    Setup function to initialize the evaluator.

    Args:
        workspace_doc_id (str): Google Sheets document ID to evaluate
    """
    global sheet_id, sheet_raw, df, table_data, rows, chart_data

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
        # No table found - fail all steps
        checkpoint.add_step("Activity Date Column", False, 1, "No table found in spreadsheet")
        checkpoint.add_step("Distance (Miles) Column", False, 2, "No table found in spreadsheet")
        checkpoint.add_step("Average Running Speed (min/mile) Column", False, 3, "No table found in spreadsheet")
        checkpoint.add_step("Content Visibility", False, 4, "No table found in spreadsheet")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Define required columns with keywords (accept both metric and imperial units)
    required_columns = [
        ("Activity Date", ["activity date", "run date", "date"]),
        ("Distance", ["distance", "miles", "distance (miles)", "distance (mi)", "distance (km)", "km"]),
        ("Average Running Speed", ["pace", "min/mile", "average pace", "avg pace", "speed", "average speed", "km/h", "speed (km/h)"]),
    ]

    # Match columns using keyword + LLM fallback
    matched = match_columns(df, required_columns, model=model, parallel=True)

    # Store globally for use in Checkpoint 2
    matched_columns = matched

    # Step 1: Activity Date column
    step_start = time.time()
    date_col = matched.get("Activity Date")
    checkpoint.add_step(
        "Activity Date Column",
        date_col is not None,
        1,
        f"Found column: '{date_col}'" if date_col else "No activity date column found",
        execution_time=time.time() - step_start
    )

    # Step 2: Distance column
    step_start = time.time()
    dist_col = matched.get("Distance")
    checkpoint.add_step(
        "Distance Column",
        dist_col is not None,
        2,
        f"Found column: '{dist_col}'" if dist_col else "No distance column found",
        execution_time=time.time() - step_start
    )

    # Step 3: Average Running Speed column
    step_start = time.time()
    speed_col = matched.get("Average Running Speed")
    checkpoint.add_step(
        "Average Running Speed Column",
        speed_col is not None,
        3,
        f"Found column: '{speed_col}'" if speed_col else "No average speed/pace column found",
        execution_time=time.time() - step_start
    )

    # Step 4: Content visibility
    step_start = time.time()
    # Use table bounds from SheetTable metadata
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
    Grade Checkpoint 2: Data Table Content Accuracy (3 pts).

    Outcome Evaluation:
    - All 109 Run activities have exact date match to gold data.
    - All 109 Run activities have exact distance match to gold data (converted to miles).
    - All 109 Run activities have exact average speed match to gold data (converted to min/mile).

    Also identifies which columns contain the converted values (miles, min/mile)
    for use in Checkpoint 3 (charts).
    """
    global matched_columns, model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Data Table Content Accuracy")
    
    df = table_data.df if table_data else None

    if df is None or df.empty:
        checkpoint.add_step("Date Match", False, 1, "No table data available")
        checkpoint.add_step("Distance Match", False, 2, "No table data available")
        checkpoint.add_step("Speed Match", False, 3, "No table data available")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Load gold data
    gold_csv_path = os.path.join(DATA_DIR, "gold_activities.csv")
    try:
        gold_runs = load_gold_run_activities(gold_csv_path)
    except Exception as e:
        checkpoint.add_step("Date Match", False, 1, f"Error loading gold data: {str(e)}")
        checkpoint.add_step("Distance Match", False, 2, "Gold data error")
        checkpoint.add_step("Speed Match", False, 3, "Gold data error")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if len(gold_runs) != 109:
        checkpoint.add_step("Date Match", False, 1, f"Expected 109 Run activities, found {len(gold_runs)}")
        checkpoint.add_step("Distance Match", False, 2, "Gold data error")
        checkpoint.add_step("Speed Match", False, 3, "Gold data error")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get date column from checkpoint 1 matching
    date_col = matched_columns.get("Activity Date") if matched_columns else None

    # For checkpoint 2, find columns with correct units using keyword matching
    # Per task.md: distance must be in miles, speed must be in min/mile
    unit_columns = [
        ("Distance (Miles)", ["distance (miles)", "distance (mi)", "miles", "(miles)"]),
        ("Speed (min/mile)", ["min/mile", "(min/mile)", "pace (min/mile)", "min per mile"]),
    ]
    unit_matches = match_columns(df, unit_columns, model=model, strict=True)

    if unit_matches not in [None, {}]:
        matched_columns.update(unit_matches)
        dist_col = matched_columns.get("Distance (Miles)")
        speed_col = matched_columns.get("Speed (min/mile)")

    # Build gold data lookup by normalized date
    # Each entry: normalized_date -> {distance_km, distance_miles, distance_m, speed_ms, speed_kmh, speed_minmile}
    gold_lookup = {}
    for idx, row in gold_runs.iterrows():
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
    for idx, user_row in df.iterrows():
        # Get user date
        if date_col and date_col in df.columns:
            user_date = normalize_date(str(user_row[date_col]))
        else:
            continue

        # Check if date exists in gold data
        if user_date in gold_lookup:
            date_matches += 1
            gold_row = gold_lookup[user_date]

            # Check distance for this row - ONLY accept miles (per task.md requirements)
            if dist_col and dist_col in df.columns:
                try:
                    user_dist = float(user_row[dist_col])
                    gold_miles = gold_row['distance_miles']
                    # Only accept miles with 1% tolerance
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

            # Check speed for this row - ONLY accept min/mile (per task.md requirements)
            if speed_col and speed_col in df.columns:
                try:
                    user_speed = float(user_row[speed_col])
                    gold_minmile = gold_row['speed_minmile']
                    # Only accept min/mile with 1% tolerance
                    if gold_minmile > 0 and gold_minmile != float('inf') and abs(user_speed - gold_minmile) / gold_minmile <= 0.01:
                        speed_matches += 1
                        if detected_speed_unit is None:
                            detected_speed_unit = 'min/mile'
                except (ValueError, TypeError):
                    pass

    # Step 1: Date matching
    step_start = time.time()
    if date_col:
        all_dates_match = date_matches == 109
        checkpoint.add_step(
            "Date Match",
            all_dates_match,
            1,
            f"{date_matches}/109 dates match" if not all_dates_match else "All 109 dates match",
            execution_time=time.time() - step_start
        )
    else:
        checkpoint.add_step("Date Match", False, 1, "Date column not found")

    # Step 2: Distance matching (row-level)
    step_start = time.time()
    if dist_col:
        all_dist_match = distance_matches == 109
        unit_str = f" ({detected_dist_unit})" if detected_dist_unit else ""
        # Debug: Print failed distance rows
        if DEBUG and failed_distance_rows:
            print(f"\n=== FAILED DISTANCE MATCHES ({len(failed_distance_rows)}) ===")
            for row in failed_distance_rows:
                print(f"  Date: {row['date']}")
                print(f"    User dist: {row['user_dist']}")
                print(f"    Gold km: {row['gold_km']}, miles: {row['gold_miles']:.4f}, meters: {row['gold_meters']}")
        checkpoint.add_step(
            "Distance Match",
            all_dist_match,
            2,
            f"{distance_matches}/109 distances match{unit_str}" if not all_dist_match else f"All 109 distances match{unit_str}",
            execution_time=time.time() - step_start
        )
    else:
        checkpoint.add_step("Distance Match", False, 2, "No distance column with miles unit found (column header must contain 'miles')")

    # Step 3: Speed matching (row-level)
    step_start = time.time()
    if speed_col:
        all_speed_match = speed_matches == 109
        unit_str = f" ({detected_speed_unit})" if detected_speed_unit else ""
        checkpoint.add_step(
            "Speed Match",
            all_speed_match,
            3,
            f"{speed_matches}/109 speeds match{unit_str}" if not all_speed_match else f"All 109 speeds match{unit_str}",
            execution_time=time.time() - step_start
        )
    else:
        checkpoint.add_step("Speed Match", False, 3, "No speed column with min/mile unit found (column header must contain 'min/mile')")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Grade Checkpoint 3: Speed Over Time Plot (13 steps).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates speed (min/mile or similar)
    3. Chart title indicates speed over time
    4. Chart is not placed over any other charts or tables
    5. Chart main data series comes from the average speed column (min/mile)
    6. Speed values are present as circular points in the chart
    7. Male 5K baseline is properly displayed (labeled in legend + dotted/dashed style)
    8. Male 5K baseline data is constant and within expected range (8-10 min/mile)
    9. Kipchoge baseline is properly displayed (labeled in legend + dotted/dashed style)
    10. Kipchoge baseline data is constant and within expected range (4.5-4.8 min/mile)
    11. Both baselines are visually distinguishable from the main data
    12. Source URLs are valid and accessible below the speed chart
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=12, result=0, name="Speed Over Time Plot")

    # Check if any charts exist
    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 13):
            checkpoint.add_step(f"Step {i}", False, i, error_msg)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find speed chart by metadata (title -> axis labels -> series data)
    speed_chart = find_speed_chart_by_metadata(chart_data, matched_columns, table_data.df, model=model)

    if not speed_chart:
        error_msg = "Could not identify speed chart by title, axis labels, or series data"
        for i in range(1, 13):
            checkpoint.add_step(f"Step {i}", False, i, error_msg)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = speed_chart.get('chart_type', 'UNKNOWN')

    # Identify series by content using three separate calls to the generalized function
    df = table_data.df if table_data else None

    # Keywords for identifying each series type
    male_5k_keywords = ["male", "5k", "5 k", "baseline", "men", "25 year", "25-year", "25yo"]
    kipchoge_keywords = ["kipchoge", "eliud", "marathon", "world", "record"]

    # Identify Male 5K baseline series (constant, 8-10 min/mile range)
    male_5k_idx = identify_series_by_content(
        chart=speed_chart,
        rows=rows,
        keywords=male_5k_keywords,
        expected_value_range=MALE_5K_PACE_RANGE,
        require_constant=True,
        model=model,
        description="legend label for male 5K running baseline"
    )

    # Identify Kipchoge baseline series (constant, 4.6-4.65 min/mile range)
    kipchoge_idx = identify_series_by_content(
        chart=speed_chart,
        rows=rows,
        keywords=kipchoge_keywords,
        expected_value_range=KIPCHOGE_PACE_RANGE,
        require_constant=True,
        exclude_indices=[male_5k_idx] if male_5k_idx is not None else None,
        model=model,
        description="legend label for Kipchoge marathon baseline"
    )

    # Identify main data series (highest variance, from speed column)
    exclude_baselines = [i for i in [male_5k_idx, kipchoge_idx] if i is not None]
    main_idx = identify_series_by_content(
        chart=speed_chart,
        rows=rows,
        keywords=[],  # No keywords - identify by variance/column
        matched_columns=matched_columns,
        column_name="Speed (min/mile)",
        df=df,
        exclude_indices=exclude_baselines if exclude_baselines else None,
        model=model
    )

    # Get axis labels
    axis_labels = get_chart_axis_labels(speed_chart)

    # Step 1: X-axis label indicates activity date
    step_start = time.time()
    x_label = axis_labels.get('x_axis', '')
    date_keywords = ['date', 'time', 'day', 'activity']
    x_label_match = keywords_match_robust(
        texts=x_label,
        keywords=date_keywords,
        model=model,
        description="X-axis label indicating activity date or time"
    ) if x_label else None
    has_date_label = x_label_match is not None or bool(x_label)
    checkpoint.add_step(
        "X-Axis Date Label",
        has_date_label,
        1,
        f"X-axis label: '{x_label}'" if has_date_label else "No date-related X-axis label found",
        execution_time=time.time() - step_start
    )

    # Step 2: Y-axis label indicates speed
    step_start = time.time()
    y_label = axis_labels.get('y_axis', '')
    speed_keywords = ['speed', 'pace', 'min/mile', 'min per mile', 'minute']
    y_label_match = keywords_match_robust(
        texts=y_label,
        keywords=speed_keywords,
        model=model,
        description="Y-axis label indicating running speed or pace"
    ) if y_label else None
    has_speed_label = y_label_match is not None
    checkpoint.add_step(
        "Y-Axis Speed Label",
        has_speed_label,
        2,
        f"Y-axis label: '{y_label}'" if has_speed_label else "No speed-related Y-axis label found",
        execution_time=time.time() - step_start
    )

    # Step 3: Chart title indicates speed over time
    step_start = time.time()
    chart_title = speed_chart.get('title', '')
    speed_title_keywords = ['speed', 'pace', 'running']
    time_title_keywords = ['time', 'over', 'progression', 'trend']
    title_speed_match = keywords_match_robust(
        texts=chart_title,
        keywords=speed_title_keywords,
        model=model,
        description="chart title related to running speed or pace"
    ) if chart_title else None
    title_time_match = keywords_match_robust(
        texts=chart_title,
        keywords=time_title_keywords,
        model=model,
        description="chart title indicating time progression or trend"
    ) if chart_title else None
    has_good_title = title_speed_match is not None or title_time_match is not None or bool(chart_title)
    checkpoint.add_step(
        "Chart Title",
        has_good_title,
        3,
        f"Chart title: '{chart_title or 'None'}'",
        execution_time=time.time() - step_start
    )

    # Step 4: Chart not placed over other charts/tables
    step_start = time.time()
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 110)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    has_overlap, overlap_details = check_chart_overlap(speed_chart, table_start, table_end, chart_data or [], table_start_col, table_end_col)
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
            # Get the expected speed column index from checkpoint 2 matching (uses correct units)
            speed_col_name = matched_columns.get("Speed (min/mile)") if matched_columns else None

            if speed_col_name and df is not None and speed_col_name in df.columns:
                expected_col_idx = df.columns.get_loc(speed_col_name)

                # Check if the identified main series uses the speed column
                if main_idx < len(series_list):
                    main_series = series_list[main_idx]
                    src_range = main_series.get('source_range', {})
                    actual_start_col = src_range.get('start_col')

                    if actual_start_col == expected_col_idx:
                        main_series_valid = True
                        series_details = f"Main series (index {main_idx}) uses '{speed_col_name}' (column {expected_col_idx})"
                    else:
                        series_details = f"Main series (index {main_idx}) uses column {actual_start_col}, expected column {expected_col_idx} ('{speed_col_name}')"
                else:
                    series_details = f"Main series index {main_idx} out of range (only {len(series_list)} series)"
            else:
                # Fallback: main series identified but can't verify column
                main_series_valid = True
                series_details = f"Main series identified at index {main_idx} (could not verify column match)"
        else:
            series_details = "Could not identify main data series by content"

    checkpoint.add_step(
        "Speed Data Series",
        main_series_valid,
        5,
        series_details,
        execution_time=time.time() - step_start
    )

    # Step 6: Circular points in chart
    step_start = time.time()
    has_points, point_details = check_point_shape(speed_chart, chart_type)
    checkpoint.add_step(
        "Circular Points",
        has_points,
        6,
        point_details,
        execution_time=time.time() - step_start
    )

    # Step 7: Male 5K baseline display check (legend label + line style)
    step_start = time.time()
    male_5k_display_valid = False
    male_5k_display_details = "Could not identify Male 5K baseline series"

    if male_5k_idx is not None:
        # Get legend label from identified series
        male_5k_label = get_series_header_label(speed_chart, male_5k_idx, rows)
        male_5k_keywords = ["male", "5k", "5 k", "baseline", "average", "men", "25"]
        label_match = keywords_match_robust(
            texts=male_5k_label,
            keywords=male_5k_keywords,
            model=model,
            description="legend label for male 5K running baseline"
        ) if male_5k_label else None

        # Check line style
        male_5k_line_style = get_series_line_style(speed_chart, male_5k_idx)
        is_dashed = male_5k_line_style and male_5k_line_style.upper() in [
            'DOTTED', 'DASHED', 'LONG_DASHED', 'MEDIUM_DASHED', 'LONG_DASHED_DOTTED'
        ]

        if label_match and is_dashed:
            male_5k_display_valid = True
            male_5k_display_details = f"Label: '{male_5k_label}', Style: {male_5k_line_style} (series index {male_5k_idx})"
        elif label_match:
            male_5k_display_details = f"Label: '{male_5k_label}' OK, but line style is {male_5k_line_style or 'SOLID'} (series index {male_5k_idx})"
        elif is_dashed:
            male_5k_display_details = f"Line style {male_5k_line_style} OK, but label '{male_5k_label}' doesn't match keywords (series index {male_5k_idx})"
        else:
            male_5k_display_details = f"Label: '{male_5k_label}', Style: {male_5k_line_style or 'SOLID'} - both need improvement (series index {male_5k_idx})"

    checkpoint.add_step(
        "Male 5K Display",
        male_5k_display_valid,
        7,
        male_5k_display_details,
        execution_time=time.time() - step_start
    )

    # Step 8: Male 5K baseline data validation (constant value in range)
    step_start = time.time()
    global sheet_male_5k_pace
    male_5k_data_valid = False
    male_5k_data_details = "Could not identify Male 5K baseline series"

    if male_5k_idx is not None:
        male_5k_values = get_series_column_values(speed_chart, male_5k_idx, rows)
        if male_5k_values:
            male_5k_data_valid, _, male_5k_data_details = validate_constant_series(
                male_5k_values, MALE_5K_PACE_RANGE, tolerance=0.01
            )
            # Store the baseline value for use in checkpoint 5
            if male_5k_values:
                sheet_male_5k_pace = male_5k_values[0]  # Constant series, all values same
        else:
            male_5k_data_details = f"Could not extract values from baseline series (index {male_5k_idx})"

    checkpoint.add_step(
        "Male 5K Data",
        male_5k_data_valid,
        8,
        male_5k_data_details,
        execution_time=time.time() - step_start
    )

    # Step 9: Kipchoge baseline display check (legend label only - no line style requirement per task.md)
    step_start = time.time()
    kipchoge_display_valid = False
    kipchoge_display_details = "Could not identify Kipchoge baseline series"

    if kipchoge_idx is not None:
        # Get legend label from identified series
        kipchoge_label = get_series_header_label(speed_chart, kipchoge_idx, rows)
        kipchoge_keywords = ["kipchoge", "eliud", "marathon", "world", "record"]
        label_match = keywords_match_robust(
            texts=kipchoge_label,
            keywords=kipchoge_keywords,
            model=model,
            description="legend label for Kipchoge marathon baseline"
        ) if kipchoge_label else None

        # Get line style for informational purposes only
        kipchoge_line_style = get_series_line_style(speed_chart, kipchoge_idx)

        if label_match:
            kipchoge_display_valid = True
            kipchoge_display_details = f"Label: '{kipchoge_label}', Style: {kipchoge_line_style or 'SOLID'} (series index {kipchoge_idx})"
        else:
            kipchoge_display_details = f"Label '{kipchoge_label}' doesn't match Kipchoge keywords (series index {kipchoge_idx})"

    checkpoint.add_step(
        "Kipchoge Display",
        kipchoge_display_valid,
        9,
        kipchoge_display_details,
        execution_time=time.time() - step_start
    )

    # Step 10: Kipchoge baseline data validation (constant value in range)
    step_start = time.time()
    global sheet_kipchoge_pace
    kipchoge_data_valid = False
    kipchoge_data_details = "Could not identify Kipchoge baseline series"

    if kipchoge_idx is not None:
        kipchoge_values = get_series_column_values(speed_chart, kipchoge_idx, rows)
        if kipchoge_values:
            kipchoge_data_valid, _, kipchoge_data_details = validate_constant_series(
                kipchoge_values, KIPCHOGE_PACE_RANGE, tolerance=0.01
            )
            # Store the baseline value for use in checkpoint 5
            if kipchoge_values:
                sheet_kipchoge_pace = kipchoge_values[0]  # Constant series, all values same
        else:
            kipchoge_data_details = f"Could not extract values from baseline series (index {kipchoge_idx})"

    checkpoint.add_step(
        "Kipchoge Data",
        kipchoge_data_valid,
        10,
        kipchoge_data_details,
        execution_time=time.time() - step_start
    )

    # Step 11: Both baselines visually distinguishable from main data AND from each other
    # Per task.md: only male 5K needs to be dotted, Kipchoge just needs to be distinguishable
    step_start = time.time()
    baselines_distinguishable = False
    distinguishable_details = "Need all three series identified (main, male 5K, Kipchoge)"

    if main_idx is not None and male_5k_idx is not None and kipchoge_idx is not None:
        # Get line styles for all series
        main_line_style = get_series_line_style(speed_chart, main_idx)
        male_5k_style = get_series_line_style(speed_chart, male_5k_idx)
        kipchoge_style = get_series_line_style(speed_chart, kipchoge_idx)

        # Get colors for all series
        main_color = get_series_color(speed_chart, main_idx)
        male_5k_color = get_series_color(speed_chart, male_5k_idx)
        kipchoge_color = get_series_color(speed_chart, kipchoge_idx)

        # Per task.md: only male 5K baseline needs to be dotted/dashed
        male_5k_is_styled = male_5k_style and male_5k_style.upper() != 'SOLID'

        # Check if baselines are distinguishable from each other (different styles OR different colors)
        baselines_have_different_styles = male_5k_style != kipchoge_style
        baselines_have_different_colors = not colors_are_similar(male_5k_color or {}, kipchoge_color or {})
        baselines_distinguishable_from_each_other = baselines_have_different_styles or baselines_have_different_colors

        # Check if Kipchoge is distinguishable from main data (different style OR different color)
        kipchoge_different_from_main_style = main_line_style != kipchoge_style
        kipchoge_different_from_main_color = not colors_are_similar(main_color or {}, kipchoge_color or {})
        kipchoge_distinguishable_from_main = kipchoge_different_from_main_style or kipchoge_different_from_main_color

        # All conditions must be met:
        # 1. Male 5K baseline is dotted/dashed (per task.md requirement)
        # 2. Baselines are distinguishable from each other (different style OR different color)
        # 3. Kipchoge is distinguishable from main data (different style OR different color)
        if male_5k_is_styled and baselines_distinguishable_from_each_other and kipchoge_distinguishable_from_main:
            baselines_distinguishable = True
            distinguishable_details = (
                f"Male 5K: {male_5k_style} (dotted per task.md), "
                f"Kipchoge: {kipchoge_style or 'SOLID'}, Main: {main_line_style or 'SOLID'}"
            )
        else:
            # Build detailed failure message
            issues = []
            if not male_5k_is_styled:
                issues.append(f"Male 5K not dotted/dashed ({male_5k_style or 'SOLID'})")
            if not baselines_distinguishable_from_each_other:
                issues.append(f"baselines not distinguishable from each other")
            if not kipchoge_distinguishable_from_main:
                issues.append(f"Kipchoge not distinguishable from main data")
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
        # Find URLs in rows below the chart, within the chart's column range
        chart_position = speed_chart.get('position', {})
        chart_anchor_row = chart_position.get('anchor_cell', {}).get('row', 0)
        chart_anchor_col = chart_position.get('anchor_cell', {}).get('col', 0)
        chart_height = chart_position.get('height', 300)
        chart_width = chart_position.get('width', 600)
        # Estimate chart end row (assuming ~20 pixels per row)
        search_start_row = chart_anchor_row + (chart_height // 20) + 1
        # Estimate chart end column (assuming ~100 pixels per column)
        search_start_col = chart_anchor_col
        search_end_col = chart_anchor_col + (chart_width // 100) + 1

        urls = find_urls_in_sheet(rows, search_start_row, num_rows=10, start_col=search_start_col, end_col=search_end_col)

        if urls:
            # Validate at least one URL is accessible
            accessible_urls = []
            for url in urls[:3]:  # Check first 3 URLs
                is_accessible, _ = validate_url_accessible(url)
                if is_accessible:
                    accessible_urls.append(url)

            if accessible_urls:
                urls_valid = True
                url_details = f"Found {len(accessible_urls)} accessible source URL(s)"
            else:
                url_details = f"Found {len(urls)} URLs but none accessible"
        else:
            url_details = f"No URLs found in rows {search_start_row}-{search_start_row + 10}"

    checkpoint.add_step(
        "Source URLs Valid",
        urls_valid,
        12,
        url_details,
        execution_time=time.time() - step_start
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Grade Checkpoint 4: Cumulative Distance Plot (6 pts).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates cumulative distance (miles or similar)
    3. Chart title indicates cumulative distance over time
    4. Chart is not placed over any other charts or tables
    5. Data shows cumulative/running total
    6. Cumulative running values are present as a line plot
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Cumulative Distance Plot")

    # Check if any charts exist
    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 7):
            checkpoint.add_step(f"Step {i}", False, i, error_msg)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find cumulative distance chart by metadata
    cumulative_chart = find_cumulative_chart_by_metadata(
        chart_data, matched_columns, table_data.df if table_data else None, model
    )

    if not cumulative_chart:
        error_msg = "Could not identify cumulative distance chart by title or axis labels"
        for i in range(1, 7):
            checkpoint.add_step(f"Step {i}", False, i, error_msg)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = get_chart_type(cumulative_chart)

    # Get axis labels
    axis_labels = get_chart_axis_labels(cumulative_chart)

    # Step 1: X-axis label indicates activity date
    step_start = time.time()
    x_label = axis_labels.get('x_axis', '')
    date_keywords = ['date', 'time', 'day', 'activity']
    x_label_match = keywords_match_robust(
        texts=x_label,
        keywords=date_keywords,
        model=model,
        description="X-axis label indicating activity date or time"
    ) if x_label else None
    has_date_label = x_label_match is not None or bool(x_label)
    checkpoint.add_step(
        "X-Axis Date Label",
        has_date_label,
        1,
        f"X-axis label: '{x_label}'" if has_date_label else "No date-related X-axis label found",
        execution_time=time.time() - step_start
    )

    # Step 2: Y-axis label indicates cumulative distance
    step_start = time.time()
    y_label = axis_labels.get('y_axis', '')
    distance_keywords = ['cumulative', 'total', 'distance', 'miles', 'running total']
    y_label_match = keywords_match_robust(
        texts=y_label,
        keywords=distance_keywords,
        model=model,
        description="Y-axis label indicating cumulative distance"
    ) if y_label else None
    has_distance_label = y_label_match is not None
    checkpoint.add_step(
        "Y-Axis Distance Label",
        has_distance_label,
        2,
        f"Y-axis label: '{y_label}'" if has_distance_label else "No cumulative distance Y-axis label found",
        execution_time=time.time() - step_start
    )

    # Step 3: Chart title indicates cumulative distance over time
    step_start = time.time()
    chart_title = cumulative_chart.get('title', '')
    cumulative_title_keywords = ['cumulative', 'total', 'distance']
    time_title_keywords = ['time', 'over', 'progression', 'trend']
    title_cumulative_match = keywords_match_robust(
        texts=chart_title,
        keywords=cumulative_title_keywords,
        model=model,
        description="chart title related to cumulative distance"
    ) if chart_title else None
    title_time_match = keywords_match_robust(
        texts=chart_title,
        keywords=time_title_keywords,
        model=model,
        description="chart title indicating time progression"
    ) if chart_title else None
    has_good_title = title_cumulative_match is not None or title_time_match is not None or bool(chart_title)
    checkpoint.add_step(
        "Chart Title",
        has_good_title,
        3,
        f"Chart title: '{chart_title or 'None'}'",
        execution_time=time.time() - step_start
    )

    # Step 4: Chart not placed over other charts/tables
    step_start = time.time()
    df = table_data.df if table_data else None
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 110)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    has_overlap, overlap_details = check_chart_overlap(
        cumulative_chart, table_start, table_end, chart_data or [], table_start_col, table_end_col
    )
    checkpoint.add_step(
        "No Chart Overlap",
        not has_overlap,
        4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start
    )

    # Step 5: Data shows cumulative/running total (validate against sheet data)
    step_start = time.time()
    cumulative_valid = False
    cumulative_details = "Could not extract chart values"

    # Get first series values from the chart
    if rows:
        chart_values = get_series_column_values(cumulative_chart, 0, rows)
        if chart_values:
            cumulative_valid, cumulative_details = validate_cumulative_against_sheet(
                chart_values, df, matched_columns, tolerance_percent=1.0
            )
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


def grade_checkpoint_5(browsing_history=None):
    """
    Grade Checkpoint 5: Website Visit Validation (4 pts).

    Validates that the agent visited required websites to gather baseline data.

    Outcome Evaluation:
    - A source URL for male 5K running speed was visited.
    - A source URL for Eliud Kipchoge marathon data was visited.
    - Male 5K source URL contains relevant pace/speed information.
    - Kipchoge source URL contains relevant marathon time information.
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Male 5K URL Visited", False, 1, "No browsing history provided")
        checkpoint.add_step("Kipchoge URL Visited", False, 2, "No browsing history provided")
        checkpoint.add_step("Male 5K Content Valid", False, 3, "No browsing history provided")
        checkpoint.add_step("Kipchoge Content Valid", False, 4, "No browsing history provided")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    browsing_lower = [url.lower() for url in browsing_history]

    # Keywords for identifying relevant URLs
    male_5k_keywords = ['5k', '5-k', 'running', 'pace', 'speed', 'average', 'runner', 'race time']
    kipchoge_keywords = ['kipchoge', 'eliud', 'marathon record', 'world record marathon']

    # Find candidate URLs for each category
    male_5k_urls = []
    kipchoge_urls = []

    for i, url_lower in enumerate(browsing_lower):
        original_url = browsing_history[i]
        if any(kw in url_lower for kw in male_5k_keywords):
            male_5k_urls.append(original_url)
        if any(kw in url_lower for kw in kipchoge_keywords):
            kipchoge_urls.append(original_url)

    # Step 1: Male 5K URL visited
    step_start = time.time()
    male_5k_visited = len(male_5k_urls) > 0
    checkpoint.add_step(
        "Male 5K URL Visited",
        male_5k_visited,
        1,
        f"Found {len(male_5k_urls)} relevant URL(s)" if male_5k_visited else "No male 5K running URL found in browsing history",
        execution_time=time.time() - step_start
    )

    # Step 2: Kipchoge URL visited
    step_start = time.time()
    kipchoge_visited = len(kipchoge_urls) > 0
    checkpoint.add_step(
        "Kipchoge URL Visited",
        kipchoge_visited,
        2,
        f"Found {len(kipchoge_urls)} relevant URL(s)" if kipchoge_visited else "No Kipchoge marathon URL found in browsing history",
        execution_time=time.time() - step_start
    )

    # Step 3: Male 5K source contains valid pace information
    # Compare extracted URL pace against sheet value with 5% tolerance
    step_start = time.time()
    male_5k_content_valid = False
    male_5k_content_details = "No male 5K URLs to check"
    tolerance_percent = 0.05  # 5% tolerance

    if not sheet_male_5k_pace:
        male_5k_content_details = "No Male 5K baseline value found in sheet (checkpoint 3 may have failed)"
    elif male_5k_urls and model:
        # Try each URL until we find one with valid pace data
        for url in male_5k_urls[:3]:  # Check up to 3 URLs
            pace, details = extract_and_convert_pace_from_url(url, "male_5k", model)
            if pace is not None:
                # Compare extracted pace to sheet value with 5% tolerance
                diff_percent = abs(pace - sheet_male_5k_pace) / sheet_male_5k_pace
                if diff_percent <= tolerance_percent:
                    male_5k_content_valid = True
                    male_5k_content_details = f"URL pace {pace:.2f} matches sheet value {sheet_male_5k_pace:.2f} min/mile ({diff_percent*100:.1f}% diff)"
                    break
                else:
                    male_5k_content_details = f"URL pace {pace:.2f} differs from sheet value {sheet_male_5k_pace:.2f} by {diff_percent*100:.1f}% (max 5%)"
            else:
                male_5k_content_details = details
    elif not model:
        male_5k_content_details = "Model not available for content validation"

    checkpoint.add_step(
        "Male 5K Content Valid",
        male_5k_content_valid,
        3,
        male_5k_content_details,
        execution_time=time.time() - step_start
    )

    # Step 4: Kipchoge source contains valid marathon time information
    # Compare extracted URL pace against sheet value with 5% tolerance
    step_start = time.time()
    kipchoge_content_valid = False
    kipchoge_content_details = "No Kipchoge URLs to check"

    if not sheet_kipchoge_pace:
        kipchoge_content_details = "No Kipchoge baseline value found in sheet (checkpoint 3 may have failed)"
    elif kipchoge_urls and model:
        # Try each URL until we find one with valid pace data
        for url in kipchoge_urls[:3]:  # Check up to 3 URLs
            pace, details = extract_and_convert_pace_from_url(url, "kipchoge", model)
            if pace is not None:
                # Compare extracted pace to sheet value with 5% tolerance
                diff_percent = abs(pace - sheet_kipchoge_pace) / sheet_kipchoge_pace
                if diff_percent <= tolerance_percent:
                    kipchoge_content_valid = True
                    kipchoge_content_details = f"URL pace {pace:.2f} matches sheet value {sheet_kipchoge_pace:.2f} min/mile ({diff_percent*100:.1f}% diff)"
                    break
                else:
                    kipchoge_content_details = f"URL pace {pace:.2f} differs from sheet value {sheet_kipchoge_pace:.2f} by {diff_percent*100:.1f}% (max 5%)"
            else:
                kipchoge_content_details = details
    elif not model:
        kipchoge_content_details = "Model not available for content validation"

    checkpoint.add_step(
        "Kipchoge Content Valid",
        kipchoge_content_valid,
        4,
        kipchoge_content_details,
        execution_time=time.time() - step_start
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
        checkpoints.append(grade_checkpoint_1())

        # Checkpoint 2: Data Table Content Accuracy
        checkpoints.append(grade_checkpoint_2())

        # Checkpoint 3: Speed Over Time Plot
        checkpoints.append(grade_checkpoint_3())

        # Checkpoint 4: Cumulative Distance Plot
        checkpoints.append(grade_checkpoint_4())

        # Checkpoint 5: Website Visit Validation
        checkpoints.append(grade_checkpoint_5(browsing_history))

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
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "✓" if step["success"] else "✗"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
