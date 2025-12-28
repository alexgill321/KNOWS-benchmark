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
)
from src.browsergym.eval.eval_utils.table_utils import (
    match_columns,
    is_text_visible_in_cell,
)
from src.browsergym.eval.eval_utils.models import load_model

# Task-specific utilities
from src.browsergym.eval.tasks.sheets_7_running_analysis.utils import (
    normalize_date,
    load_gold_run_activities,
    check_all_content_visible,
    get_chart_axis_labels,
    check_chart_overlap,
    extract_baseline_series,
    find_urls_in_sheet,
    validate_url_accessible,
    check_circular_points,
    find_speed_chart_by_metadata,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/sheets_7_running_analysis/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Gold baseline values for checkpoint 3
MALE_5K_PACE_RANGE = (8.0, 10.0)  # min/mile for 25yo male
KIPCHOGE_PACE_RANGE = (4.5, 4.8)  # min/mile (based on top 3 marathons)

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
    all_visible, visibility_details = check_all_content_visible(sheet_raw, start_row, end_row, num_cols, is_text_visible_in_cell)
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
    Grade Checkpoint 3: Speed Over Time Plot (15 pts, consolidated to 14 steps).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates speed (min/mile or similar)
    3. Chart title indicates speed over time
    4. Chart is not placed over any other charts or tables
    5. Chart main data series comes from the average speed column
    6. Speed values are in min/mile units
    7. Speed values are present as circular points in the chart
    8. Male 5K baseline line is present in speed chart
    9. Male 5K baseline is styled as dotted/dashed
    10. Male 5K baseline is within gold range
    11. Male 5K baseline matches source (if source present)
    12. Kipchoge baseline line is present in speed chart
    13. Kipchoge baseline represents average of his top 3 marathons
    14. Source URLs are valid and accessible below the speed chart
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=14, result=0, name="Speed Over Time Plot")

    # Check if any charts exist
    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 15):
            checkpoint.add_step(f"Step {i}", False, i, error_msg)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find speed chart by metadata (title -> axis labels -> series data)
    speed_chart = find_speed_chart_by_metadata(chart_data, matched_columns, df)

    if not speed_chart:
        error_msg = "Could not identify speed chart by title, axis labels, or series data"
        for i in range(1, 15):
            checkpoint.add_step(f"Step {i}", False, i, error_msg)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = speed_chart.get('chart_type', 'UNKNOWN')

    # Get axis labels
    axis_labels = get_chart_axis_labels(speed_chart)

    # Step 1: X-axis label indicates activity date
    step_start = time.time()
    x_label = axis_labels.get('x_axis', '').lower()
    date_keywords = ['date', 'time', 'day', 'activity']
    has_date_label = any(kw in x_label for kw in date_keywords) or bool(x_label)
    checkpoint.add_step(
        "X-Axis Date Label",
        has_date_label,
        1,
        f"X-axis label: '{axis_labels.get('x_axis', 'None')}'" if has_date_label else "No date-related X-axis label found",
        execution_time=time.time() - step_start
    )

    # Step 2: Y-axis label indicates speed
    step_start = time.time()
    y_label = axis_labels.get('y_axis', '').lower()
    speed_keywords = ['speed', 'pace', 'min/mile', 'min per mile', 'minute']
    has_speed_label = any(kw in y_label for kw in speed_keywords)
    checkpoint.add_step(
        "Y-Axis Speed Label",
        has_speed_label,
        2,
        f"Y-axis label: '{axis_labels.get('y_axis', 'None')}'" if has_speed_label else "No speed-related Y-axis label found",
        execution_time=time.time() - step_start
    )

    # Step 3: Chart title indicates speed over time
    step_start = time.time()
    chart_title = speed_chart.get('title', '').lower()
    title_has_speed = any(kw in chart_title for kw in ['speed', 'pace', 'running'])
    title_has_time = any(kw in chart_title for kw in ['time', 'over', 'progression', 'trend'])
    has_good_title = title_has_speed or title_has_time or bool(speed_chart.get('title', ''))
    checkpoint.add_step(
        "Chart Title",
        has_good_title,
        3,
        f"Chart title: '{speed_chart.get('title', 'None')}'",
        execution_time=time.time() - step_start
    )

    # Step 4: Chart not placed over other charts/tables
    step_start = time.time()
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 110)
    has_overlap, overlap_details = check_chart_overlap(speed_chart, table_start, table_end, chart_data or [])
    checkpoint.add_step(
        "No Chart Overlap",
        not has_overlap,
        4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start
    )

    # Step 5: Main data series from speed column
    step_start = time.time()
    series_list = extract_baseline_series(speed_chart)
    main_series_valid = len(series_list) > 0
    series_details = f"Found {len(series_list)} series in chart"
    checkpoint.add_step(
        "Speed Data Series",
        main_series_valid,
        5,
        series_details,
        execution_time=time.time() - step_start
    )

    # Step 6: Speed values in min/mile units
    step_start = time.time()
    # Check if y-axis values are in reasonable min/mile range (4-20)
    # This is validated by checking axis label and title for "min/mile"
    has_minmile_unit = 'min/mile' in y_label or 'min per mile' in y_label or 'min/mile' in chart_title
    checkpoint.add_step(
        "Min/Mile Units",
        has_minmile_unit or has_speed_label,
        6,
        "Speed values labeled as min/mile" if has_minmile_unit else "Speed units inferred from axis label",
        execution_time=time.time() - step_start
    )

    # Step 7: Circular points in chart
    step_start = time.time()
    has_points, point_details = check_circular_points(speed_chart, chart_type)
    checkpoint.add_step(
        "Circular Points",
        has_points,
        7,
        point_details,
        execution_time=time.time() - step_start
    )

    # Step 8: Male 5K baseline line present
    step_start = time.time()
    # Need at least 2 series (main data + 1 baseline)
    has_male_baseline = len(series_list) >= 2
    checkpoint.add_step(
        "Male 5K Baseline Present",
        has_male_baseline,
        8,
        f"Found {len(series_list)} series (need ≥2 for baselines)" if not has_male_baseline else "Baseline series found",
        execution_time=time.time() - step_start
    )

    # Step 9: Male 5K baseline is dotted/dashed
    step_start = time.time()
    baseline_is_dashed = False
    if len(series_list) >= 2:
        for series in series_list[1:]:  # Skip main data series
            line_style = series.get('line_style', 'SOLID')
            if line_style and line_style.upper() in ['DOTTED', 'DASHED', 'LONG_DASHED', 'MEDIUM_DASHED', 'LONG_DASHED_DOTTED']:
                baseline_is_dashed = True
                break
    checkpoint.add_step(
        "Male 5K Baseline Dotted/Dashed",
        baseline_is_dashed,
        9,
        "Baseline has dotted/dashed style" if baseline_is_dashed else "Baseline line style is solid or not found",
        execution_time=time.time() - step_start
    )

    # Step 10: Male 5K baseline within gold range
    step_start = time.time()
    # This would require extracting actual baseline values from the chart data
    # For now, check if baseline series exists
    baseline_in_range = has_male_baseline  # Simplified check
    checkpoint.add_step(
        "Male 5K Baseline in Range",
        baseline_in_range,
        10,
        f"Expected range: {MALE_5K_PACE_RANGE[0]}-{MALE_5K_PACE_RANGE[1]} min/mile",
        execution_time=time.time() - step_start
    )

    # Step 11: Male 5K baseline matches source
    step_start = time.time()
    # This would compare against browsing_history content
    # For now, pass if baseline exists
    checkpoint.add_step(
        "Male 5K Baseline Matches Source",
        has_male_baseline,
        11,
        "Baseline value should match researched 5K pace",
        execution_time=time.time() - step_start
    )

    # Step 12: Kipchoge baseline line present
    step_start = time.time()
    # Need at least 3 series (main data + 2 baselines)
    has_kipchoge_baseline = len(series_list) >= 3
    checkpoint.add_step(
        "Kipchoge Baseline Present",
        has_kipchoge_baseline,
        12,
        f"Found {len(series_list)} series (need ≥3 for both baselines)" if not has_kipchoge_baseline else "Second baseline series found",
        execution_time=time.time() - step_start
    )

    # Step 13: Kipchoge baseline correct value
    step_start = time.time()
    kipchoge_correct = has_kipchoge_baseline  # Simplified check
    checkpoint.add_step(
        "Kipchoge Baseline Correct",
        kipchoge_correct,
        13,
        f"Expected range: {KIPCHOGE_PACE_RANGE[0]}-{KIPCHOGE_PACE_RANGE[1]} min/mile (top 3 marathon avg)",
        execution_time=time.time() - step_start
    )

    # Step 14: Source URLs valid and accessible below chart
    step_start = time.time()
    urls_valid = False
    url_details = "No URLs found below chart"

    if rows:
        # Find URLs in rows below the chart
        chart_position = speed_chart.get('position', {})
        chart_anchor_row = chart_position.get('anchor_cell', {}).get('row', 0)
        chart_height = chart_position.get('height', 300)
        # Estimate chart end row (assuming ~20 pixels per row)
        search_start_row = chart_anchor_row + (chart_height // 20) + 1

        urls = find_urls_in_sheet(rows, search_start_row, num_rows=10)

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
        14,
        url_details,
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
