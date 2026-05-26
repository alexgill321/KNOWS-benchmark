import os
import sys
import math
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
    find_sets_chart_by_metadata,
    find_cumulative_reps_chart_by_metadata,
    extract_pace_from_url,
    extract_and_convert_pace_from_url,
    judge_url_relevance,
    judge_url_pace_match,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/sheets_7_running_analysis/instance_5/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Gold baseline values for checkpoint 3 (sets chart)
ADULT_SETS_RANGE = (15, 22)  # average adult sets per workout
CUTLER_SETS_RANGE = (40, 80)  # Jay Cutler sets per workout (~20 sets/body part x 2-3 body parts)

# Expected activity count
EXPECTED_STRENGTH = 219

model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global variables
sheet_id = None
sheet_raw = None
table_data = None
table_sheet_id = None  # sheetId of the tab containing the data table
rows = None
matched_columns = None
chart_data = None
sheet_adult_sets = None  # Avg adult sets baseline from sheet (for checkpoint 5)
sheet_cutler_sets = None  # Jay Cutler sets baseline from sheet (for checkpoint 5)


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

    result = extract_sheet_data(sheet_id, SHEETS_SERVICE, return_raw=True)

    if result:
        table_data, sheet_raw = result

        if isinstance(table_data, list):
            table_data = table_data[0] if table_data else None

        if table_data:
            print(f"Extracted DataFrame with {len(table_data.df)} rows and columns: {list(table_data.df.columns)}")
            print(f"Table position: rows {table_data.start_row}-{table_data.end_row}, cols {table_data.start_col}-{table_data.end_col}")

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

    chart_data = extract_charts_from_sheet(sheet_id, SHEETS_SERVICE)
    print(f"Extracted {len(chart_data) if chart_data else 0} charts from spreadsheet")


def grade_checkpoint_1():
    """
    Grade Checkpoint 1: Data Table Structure (4 pts).

    Outcome Evaluation:
    - Date/time column exists with appropriate header.
    - Total Sets column exists with appropriate header.
    - Total Reps column exists with appropriate header.
    - All table content is fully visible.
    """
    global matched_columns

    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Data Table Structure")

    if table_data:
        df = table_data.df
    else:
        df = None

    if df is None or df.empty:
        checkpoint.add_step("Activity Date Column", False, 1, "No table found in spreadsheet", execution_time=0)
        checkpoint.add_step("Total Sets Column", False, 2, "No table found in spreadsheet", execution_time=0)
        checkpoint.add_step("Total Reps Column", False, 3, "No table found in spreadsheet", execution_time=0)
        checkpoint.add_step("Content Visibility", False, 4, "No table found in spreadsheet", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    required_columns = [
        ("Activity Date", ["activity date", "date", "workout date"]),
        ("Total Sets", ["total sets", "sets", "workout sets"]),
        ("Total Reps", ["total reps", "reps", "repetitions"]),
    ]

    column_match_start = time.time()
    matched = match_columns(df, required_columns, model=model, parallel=True)
    column_match_time = time.time() - column_match_start

    matched_columns = matched
    per_column_time = column_match_time / 3

    date_col = matched.get("Activity Date")
    checkpoint.add_step(
        "Activity Date Column",
        date_col is not None,
        1,
        f"Found column: '{date_col}'" if date_col else "No activity date column found",
        execution_time=per_column_time
    )

    sets_col = matched.get("Total Sets")
    checkpoint.add_step(
        "Total Sets Column",
        sets_col is not None,
        2,
        f"Found column: '{sets_col}'" if sets_col else "No total sets column found",
        execution_time=per_column_time
    )

    reps_col = matched.get("Total Reps")
    checkpoint.add_step(
        "Total Reps Column",
        reps_col is not None,
        3,
        f"Found column: '{reps_col}'" if reps_col else "No total reps column found",
        execution_time=per_column_time
    )

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
    - All 219 Strength Training activities have exact date match (10 pts max).
    - All 219 Strength Training activities have exact Total Sets match (10 pts max).
    - All 219 Strength Training activities have exact Total Reps match (10 pts max).
    """
    global matched_columns
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Data Table Content Accuracy")

    df = table_data.df if table_data else None

    if df is None or df.empty:
        checkpoint.add_step("Date Match", False, 1, "No table data available", execution_time=0)
        checkpoint.add_step("Total Sets Match", False, 2, "No table data available", execution_time=0)
        checkpoint.add_step("Total Reps Match", False, 3, "No table data available", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Load gold data - Garmin CSV format (different from Strava instances)
    gold_csv_path = os.path.join(DATA_DIR, "gold_activities.csv")
    try:
        gold_df = pd.read_csv(gold_csv_path)
        gold_strength = gold_df[gold_df['Activity Type'] == 'Strength Training'].copy()
    except Exception as e:
        checkpoint.add_step("Date Match", False, 1, f"Error loading gold data: {str(e)}", execution_time=0)
        checkpoint.add_step("Total Sets Match", False, 2, "Gold data error", execution_time=0)
        checkpoint.add_step("Total Reps Match", False, 3, "Gold data error", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if len(gold_strength) != EXPECTED_STRENGTH:
        checkpoint.add_step("Date Match", False, 1, f"Expected {EXPECTED_STRENGTH} Strength Training activities, found {len(gold_strength)}", execution_time=0)
        checkpoint.add_step("Total Sets Match", False, 2, "Gold data count mismatch", execution_time=0)
        checkpoint.add_step("Total Reps Match", False, 3, "Gold data count mismatch", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    date_col = matched_columns.get("Activity Date") if matched_columns else None
    sets_col = matched_columns.get("Total Sets") if matched_columns else None
    reps_col = matched_columns.get("Total Reps") if matched_columns else None

    # Build gold data lookup by normalized date
    gold_lookup = {}
    for idx, row in gold_strength.iterrows():
        norm_date = normalize_date(str(row['Date']))
        try:
            total_sets = int(row['Total Sets'])
        except (ValueError, TypeError):
            total_sets = None
        try:
            total_reps = int(row['Total Reps'])
        except (ValueError, TypeError):
            total_reps = None
        gold_lookup[norm_date] = {
            'total_sets': total_sets,
            'total_reps': total_reps,
        }

    date_matches = 0
    sets_matches = 0
    reps_matches = 0

    validation_start = time.time()
    for idx, user_row in df.iterrows():
        if date_col and date_col in df.columns:
            user_date = normalize_date(str(user_row[date_col]))
        else:
            continue

        if user_date in gold_lookup:
            date_matches += 1
            gold_row = gold_lookup[user_date]

            if sets_col and sets_col in df.columns:
                try:
                    user_sets = int(float(user_row[sets_col]))
                except (ValueError, TypeError):
                    user_sets = None
                if gold_row['total_sets'] is None and user_sets is None:
                    sets_matches += 1  # Both have no data — match
                elif gold_row['total_sets'] is not None and user_sets == gold_row['total_sets']:
                    sets_matches += 1

            if reps_col and reps_col in df.columns:
                try:
                    user_reps = int(float(user_row[reps_col]))
                except (ValueError, TypeError):
                    user_reps = None
                if gold_row['total_reps'] is None and user_reps is None:
                    reps_matches += 1  # Both have no data — match
                elif gold_row['total_reps'] is not None and user_reps == gold_row['total_reps']:
                    reps_matches += 1
    validation_time = time.time() - validation_start

    per_step_time = validation_time / 3

    if date_col:
        date_score = math.floor(date_matches / EXPECTED_STRENGTH * 10)
        checkpoint.add_step(
            "Date Match",
            date_matches == EXPECTED_STRENGTH,
            1,
            f"{date_matches}/{EXPECTED_STRENGTH} dates match ({date_matches/EXPECTED_STRENGTH:.0%}), {date_score}/10 pts",
            score=date_score,
            max_score=10,
            execution_time=per_step_time
        )
    else:
        checkpoint.add_step("Date Match", False, 1, "Date column not found", score=0, max_score=10, execution_time=per_step_time)

    if sets_col:
        sets_score = math.floor(sets_matches / EXPECTED_STRENGTH * 10)
        checkpoint.add_step(
            "Total Sets Match",
            sets_matches == EXPECTED_STRENGTH,
            2,
            f"{sets_matches}/{EXPECTED_STRENGTH} sets match ({sets_matches/EXPECTED_STRENGTH:.0%}), {sets_score}/10 pts",
            score=sets_score,
            max_score=10,
            execution_time=per_step_time
        )
    else:
        checkpoint.add_step("Total Sets Match", False, 2, "Total Sets column not found", score=0, max_score=10, execution_time=per_step_time)

    if reps_col:
        reps_score = math.floor(reps_matches / EXPECTED_STRENGTH * 10)
        checkpoint.add_step(
            "Total Reps Match",
            reps_matches == EXPECTED_STRENGTH,
            3,
            f"{reps_matches}/{EXPECTED_STRENGTH} reps match ({reps_matches/EXPECTED_STRENGTH:.0%}), {reps_score}/10 pts",
            score=reps_score,
            max_score=10,
            execution_time=per_step_time
        )
    else:
        checkpoint.add_step("Total Reps Match", False, 3, "Total Reps column not found", score=0, max_score=10, execution_time=per_step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Grade Checkpoint 3: Sets Over Time Plot (13 steps).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates sets
    3. Chart title indicates sets over time
    4. Chart is not placed over any other charts or tables
    5. Chart main data series comes from the Total Sets column
    6. Set values are present as circular points in the chart
    7. Avg Adult baseline displayed (labeled + dotted/dashed style)
    8. Avg Adult baseline data constant and within range (15-22)
    9. Jay Cutler baseline displayed (labeled)
    10. Jay Cutler baseline data constant and within range (18-25)
    11. Both baselines visually distinguishable from main data
    12. Source URLs valid and accessible below chart
    13. Chart is on the same sheet tab as the data table
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=12, result=0, name="Sets Over Time Plot")

    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 13):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find sets chart by metadata
    sets_chart = find_sets_chart_by_metadata(
        chart_data, matched_columns, table_data.df if table_data else None, model=model
    )

    if not sets_chart:
        error_msg = "Could not identify sets chart by title, axis labels, or series data"
        for i in range(1, 13):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = sets_chart.get('chart_type', 'UNKNOWN')
    df = table_data.df if table_data else None

    # Keywords for identifying each series type
    adult_sets_keywords = ["average", "adult", "avg", "recommended"]
    cutler_keywords = ["cutler", "jay", "bodybuilder"]

    # Parallel: Identify both baseline series concurrently
    baseline_tasks = [
        {
            'id': 'adult_sets',
            'func': identify_series_by_content,
            'kwargs': {
                'chart': sets_chart,
                'rows': rows,
                'keywords': adult_sets_keywords,
                'expected_value_range': ADULT_SETS_RANGE,
                'require_constant': True,
                'model': model,
                'description': "legend label for average adult sets baseline"
            }
        },
        {
            'id': 'cutler',
            'func': identify_series_by_content,
            'kwargs': {
                'chart': sets_chart,
                'rows': rows,
                'keywords': cutler_keywords,
                'expected_value_range': CUTLER_SETS_RANGE,
                'require_constant': True,
                'model': model,
                'description': "legend label for Jay Cutler sets baseline"
            }
        },
    ]
    baseline_results = parallel_execute(baseline_tasks, max_workers=2)
    adult_sets_idx = baseline_results.get('adult_sets')
    cutler_idx = baseline_results.get('cutler')

    # Fallback: if adult sets not found by value range, find by keyword alone for display checks
    adult_sets_candidate_idx = adult_sets_idx
    if adult_sets_idx is None and rows is not None:
        adult_sets_candidate_idx = identify_series_by_content(
            chart=sets_chart, rows=rows,
            keywords=adult_sets_keywords,
            require_constant=True, model=model,
            description="legend label for average adult sets baseline"
        )

    # Sequential: Identify main data series
    exclude_baselines = [i for i in [adult_sets_idx, cutler_idx] if i is not None]
    main_idx = identify_series_by_content(
        chart=sets_chart,
        rows=rows,
        keywords=[],
        matched_columns=matched_columns,
        column_name="Total Sets",
        df=df,
        exclude_indices=exclude_baselines if exclude_baselines else None,
        model=model
    )

    # Get axis labels
    axis_labels = get_chart_axis_labels(sets_chart)
    x_label = axis_labels.get('x_axis', '')
    y_label = axis_labels.get('y_axis', '')
    chart_title = sets_chart.get('title', '')

    date_keywords = ['date', 'time', 'day', 'activity']
    sets_keywords = ['sets', 'total sets', 'workout', 'set']
    sets_title_keywords = ['sets', 'workout', 'training']
    time_title_keywords = ['time', 'over', 'daily', 'chart']

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
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(y_label, sets_keywords, "Y-axis label indicating workout sets")}]}
            ]
        })
    if chart_title:
        vlm_tasks.append({
            'id': 'title_sets',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, sets_title_keywords, "chart title related to workout sets")}]}
            ]
        })
        vlm_tasks.append({
            'id': 'title_time',
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, time_title_keywords, "chart title indicating time progression")}]}
            ]
        })

    keyword_start = time.time()
    if vlm_tasks:
        keyword_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=4)
    else:
        keyword_results = {}
    keyword_time = time.time() - keyword_start

    # Step 1: X-axis label
    x_match = keyword_results.get('x_axis', False) if x_label else False
    x_fallback = keywords_match_robust(texts=x_label, keywords=date_keywords, model=model, description="X-axis label indicating activity date or time") if x_label else False
    has_date = bool(x_match) or bool(x_fallback)
    checkpoint.add_step("X-Axis Date Label", has_date, 1,
        f"X-axis label: '{x_label}'" if has_date else f"X-axis label '{x_label or 'None'}' does not match date keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1))

    # Step 2: Y-axis label
    y_match = keyword_results.get('y_axis', False) if y_label else False
    y_fallback = keywords_match_robust(texts=y_label, keywords=sets_keywords, model=model, description="Y-axis label indicating workout sets") if y_label else False
    has_sets = bool(y_match) or bool(y_fallback)
    checkpoint.add_step("Y-Axis Sets Label", has_sets, 2,
        f"Y-axis label: '{y_label}'" if has_sets else f"Y-axis label '{y_label or 'None'}' does not match sets keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1))

    # Step 3: Chart title
    title_sets = keyword_results.get('title_sets', False) if chart_title else False
    title_time = keyword_results.get('title_time', False) if chart_title else False
    title_fallback = keywords_match_robust(texts=chart_title, keywords=sets_title_keywords + time_title_keywords, model=model, description="chart title indicating workout sets over time") if chart_title else False
    has_title = bool(title_sets) or bool(title_time) or bool(title_fallback)
    checkpoint.add_step("Chart Title", has_title, 3,
        f"Chart title: '{chart_title}'" if has_title else f"Chart title '{chart_title or 'None'}' does not match sets/time keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1))

    # Step 4: No overlap
    step_start = time.time()
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 220)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    anchor = sets_chart.get('position', {}).get('anchor_cell', {})
    if anchor.get('row') is None:
        has_overlap = False
        overlap_details = "Chart anchor row is None (API bug) - skipping overlap check"
    else:
        safe_charts = [c for c in (chart_data or []) if c.get('position', {}).get('anchor_cell', {}).get('row') is not None]
        has_overlap, overlap_details = check_chart_overlap(sets_chart, table_start, table_end, safe_charts, table_start_col, table_end_col)
    checkpoint.add_step("No Chart Overlap", not has_overlap, 4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start)

    # Step 5: Main data series from Total Sets column
    step_start = time.time()
    series_list = get_all_series_metadata(sets_chart)
    main_valid = False
    series_details = "No series found in chart"
    if series_list and main_idx is not None:
        sets_col_name = matched_columns.get("Total Sets") if matched_columns else None
        if sets_col_name and df is not None and sets_col_name in df.columns:
            expected_col_idx = df.columns.get_loc(sets_col_name)
            if not isinstance(expected_col_idx, int):
                import numpy as np
                if isinstance(expected_col_idx, np.ndarray):
                    expected_col_idx = int(np.where(expected_col_idx)[0][0])
                elif isinstance(expected_col_idx, slice):
                    expected_col_idx = expected_col_idx.start or 0
            if main_idx < len(series_list):
                src_range = series_list[main_idx].get('source_range', {})
                actual_col = src_range.get('start_col')
                if actual_col == expected_col_idx:
                    main_valid = True
                    series_details = f"Main series (index {main_idx}) uses '{sets_col_name}' (column {expected_col_idx})"
                else:
                    # Column doesn't match — fallback: check if series header is a sets column
                    header_label = get_series_header_label(sets_chart, main_idx, rows) if rows else ""
                    header_label_match = keywords_match_robust(
                        texts=header_label, keywords=["sets", "total sets"], substring=True
                    ) if header_label else False
                    if header_label_match:
                        main_valid = True
                        series_details = f"Main series (index {main_idx}) uses column {actual_col} ('{header_label}'), not CP1 column {expected_col_idx} ('{sets_col_name}')"
                    else:
                        series_details = f"Main series (index {main_idx}) uses column {actual_col}, expected {expected_col_idx} ('{sets_col_name}')"
        else:
            # Fallback: verify via series header label since CP1 didn't populate Sets column
            header_label = get_series_header_label(sets_chart, main_idx, rows) if rows else ""
            if keywords_match_robust(texts=header_label, keywords=["sets", "total sets"], substring=True) if header_label else False:
                main_valid = True
                series_details = f"Main series at index {main_idx} matched by header label '{header_label}' (CP1 sets column not available)"
            else:
                series_details = f"Main series at index {main_idx} header '{header_label or 'unknown'}' does not match sets keywords; CP1 sets column unavailable"
    elif main_idx is None:
        series_details = "Could not identify main data series by content"
    checkpoint.add_step("Sets Data Series", main_valid, 5, series_details, execution_time=time.time() - step_start)

    # Step 6: Circular points
    step_start = time.time()
    has_points, point_details = check_point_shape(sets_chart, chart_type)
    checkpoint.add_step("Circular Points", has_points, 6, point_details, execution_time=time.time() - step_start)

    # Step 7: Avg Adult baseline display
    step_start = time.time()
    global sheet_adult_sets
    adult_display_valid = False
    adult_display_details = "Could not identify avg adult sets baseline series"
    if adult_sets_candidate_idx is not None and rows is not None:
        adult_label = get_series_header_label(sets_chart, adult_sets_candidate_idx, rows)
        label_match = keywords_match_robust(texts=adult_label, keywords=adult_sets_keywords, substring=True) if adult_label else None
        adult_line_style = get_series_line_style(sets_chart, adult_sets_candidate_idx)
        is_dashed = adult_line_style and adult_line_style.upper() in ['DOTTED', 'DASHED', 'LONG_DASHED', 'MEDIUM_DASHED', 'LONG_DASHED_DOTTED']
        if label_match and is_dashed:
            adult_display_valid = True
            adult_display_details = f"Label: '{adult_label}', Style: {adult_line_style} (series index {adult_sets_candidate_idx})"
        elif label_match:
            adult_display_details = f"Label: '{adult_label}' OK, but line style is {adult_line_style or 'SOLID'} (series index {adult_sets_candidate_idx})"
        elif is_dashed:
            adult_display_details = f"Line style {adult_line_style} OK, but label '{adult_label}' doesn't match keywords (series index {adult_sets_candidate_idx})"
        else:
            adult_display_details = f"Label: '{adult_label}', Style: {adult_line_style or 'SOLID'} - both need improvement (series index {adult_sets_candidate_idx})"
    checkpoint.add_step("Avg Adult Display", adult_display_valid, 7, adult_display_details, execution_time=time.time() - step_start)

    # Step 8: Avg Adult baseline data
    step_start = time.time()
    adult_data_valid = False
    adult_data_details = "Could not identify avg adult sets baseline series"
    if adult_sets_candidate_idx is not None and rows is not None:
        adult_values = get_series_column_values(sets_chart, adult_sets_candidate_idx, rows)
        if adult_values:
            adult_data_valid, _, adult_data_details = validate_constant_series(adult_values, ADULT_SETS_RANGE, tolerance=0.01)
            if adult_values:
                sheet_adult_sets = adult_values[0]
        else:
            adult_data_details = f"Could not extract values from baseline series (index {adult_sets_candidate_idx})"
    elif adult_sets_candidate_idx is not None:
        adult_data_details = "Sheet rows unavailable from setup() - cannot extract baseline values"
    checkpoint.add_step("Avg Adult Data", adult_data_valid, 8, adult_data_details, execution_time=time.time() - step_start)

    # Step 9: Jay Cutler baseline display (label only — task says "Compare this to")
    step_start = time.time()
    global sheet_cutler_sets
    cutler_display_valid = False
    cutler_display_details = "Could not identify Jay Cutler baseline series"
    if cutler_idx is not None and rows is not None:
        cutler_label = get_series_header_label(sets_chart, cutler_idx, rows)
        label_match = keywords_match_robust(texts=cutler_label, keywords=cutler_keywords, substring=True) if cutler_label else None
        cutler_line_style = get_series_line_style(sets_chart, cutler_idx)
        if label_match:
            cutler_display_valid = True
            cutler_display_details = f"Label: '{cutler_label}', Style: {cutler_line_style or 'SOLID'} (series index {cutler_idx})"
        else:
            cutler_display_details = f"Label '{cutler_label}' doesn't match Cutler keywords (series index {cutler_idx})"
    checkpoint.add_step("Cutler Display", cutler_display_valid, 9, cutler_display_details, execution_time=time.time() - step_start)

    # Step 10: Jay Cutler baseline data
    step_start = time.time()
    cutler_data_valid = False
    cutler_data_details = "Could not identify Jay Cutler baseline series"
    if cutler_idx is not None and rows is not None:
        cutler_values = get_series_column_values(sets_chart, cutler_idx, rows)
        if cutler_values:
            cutler_data_valid, _, cutler_data_details = validate_constant_series(cutler_values, CUTLER_SETS_RANGE, tolerance=0.01)
            if cutler_values:
                sheet_cutler_sets = cutler_values[0]
        else:
            cutler_data_details = f"Could not extract values from baseline series (index {cutler_idx})"
    elif cutler_idx is not None:
        cutler_data_details = "Sheet rows unavailable from setup() - cannot extract baseline values"
    checkpoint.add_step("Cutler Data", cutler_data_valid, 10, cutler_data_details, execution_time=time.time() - step_start)

    # Step 11: Both baselines distinguishable
    step_start = time.time()
    distinguishable = False
    dist_details = "Need all three series identified (main, adult, cutler)"
    if main_idx is not None and adult_sets_candidate_idx is not None and cutler_idx is not None:
        main_style = get_series_line_style(sets_chart, main_idx)
        adult_style = get_series_line_style(sets_chart, adult_sets_candidate_idx)
        cutler_style = get_series_line_style(sets_chart, cutler_idx)
        main_color = get_series_color(sets_chart, main_idx)
        adult_color = get_series_color(sets_chart, adult_sets_candidate_idx)
        cutler_color = get_series_color(sets_chart, cutler_idx)
        baselines_diff = (adult_style != cutler_style) or not colors_are_similar(adult_color or {}, cutler_color or {})
        cutler_diff = (main_style != cutler_style) or not colors_are_similar(main_color or {}, cutler_color or {})
        adult_diff = (main_style != adult_style) or not colors_are_similar(main_color or {}, adult_color or {})
        if baselines_diff and cutler_diff and adult_diff:
            distinguishable = True
            dist_details = f"Adult: {adult_style or 'SOLID'}, Cutler: {cutler_style or 'SOLID'}, Main: {main_style or 'SOLID'}"
        else:
            issues = []
            if not baselines_diff:
                issues.append("baselines not distinguishable from each other")
            if not cutler_diff:
                issues.append("Cutler not distinguishable from main data")
            if not adult_diff:
                issues.append("Adult not distinguishable from main data")
            dist_details = f"Issues: {'; '.join(issues)}"
    checkpoint.add_step("Baselines Distinguishable", distinguishable, 11, dist_details, execution_time=time.time() - step_start)

    # Step 12: Source URLs below chart
    step_start = time.time()
    urls_valid = False
    url_details = "No URLs found below chart"
    if rows:
        chart_position = sets_chart.get('position', {})
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
            accessible = [u for u in urls[:3] if validate_url_accessible(u)[0]]
            if accessible:
                urls_valid = True
                url_details = f"Found {len(accessible)} accessible source URL(s)"
            else:
                url_details = f"Found {len(urls)} URLs but none accessible"
        else:
            url_details = f"No URLs found in rows {search_start_row}-{search_start_row + 50}"
    checkpoint.add_step("Source URLs Valid", urls_valid, 12, url_details, execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Grade Checkpoint 4: Cumulative Reps Plot (7 pts).

    Outcome Evaluation:
    1. X-axis label indicates activity date
    2. Y-axis label indicates cumulative reps
    3. Chart title indicates cumulative reps over time
    4. Chart is not placed over any other charts or tables
    5. Data shows cumulative/running total
    6. Cumulative reps values are present as a line plot
    7. Chart is on the same sheet tab as the data table
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Cumulative Reps Plot")

    if not chart_data:
        error_msg = "No charts found in spreadsheet"
        for i in range(1, 7):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    cumulative_chart = find_cumulative_reps_chart_by_metadata(
        chart_data, matched_columns, table_data.df if table_data else None, model
    )

    if not cumulative_chart:
        error_msg = "Could not identify cumulative reps chart by title or axis labels"
        for i in range(1, 7):
            checkpoint.add_step(f"Step {i}", False, i, error_msg, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart_type = get_chart_type(cumulative_chart)
    axis_labels = get_chart_axis_labels(cumulative_chart)
    x_label = axis_labels.get('x_axis', '')
    y_label = axis_labels.get('y_axis', '')
    chart_title = cumulative_chart.get('title', '')

    date_keywords = ['date', 'time', 'day', 'activity']
    reps_keywords = ['cumulative', 'total', 'reps', 'repetitions', 'running total']
    cumulative_title_keywords = ['cumulative', 'total', 'reps']
    time_title_keywords = ['time', 'over', 'progression', 'trend']

    vlm_tasks = []
    if x_label:
        vlm_tasks.append({'id': 'x_axis', 'messages': [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
            {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(x_label, date_keywords, "X-axis label indicating date")}]}
        ]})
    if y_label:
        vlm_tasks.append({'id': 'y_axis', 'messages': [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
            {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(y_label, reps_keywords, "Y-axis label indicating cumulative reps")}]}
        ]})
    if chart_title:
        vlm_tasks.append({'id': 'title_cumulative', 'messages': [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
            {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, cumulative_title_keywords, "chart title related to cumulative reps")}]}
        ]})
        vlm_tasks.append({'id': 'title_time', 'messages': [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
            {"role": "user", "content": [{"type": "text", "text": build_keyword_match_prompt(chart_title, time_title_keywords, "chart title indicating time progression")}]}
        ]})

    keyword_start = time.time()
    keyword_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=4) if vlm_tasks else {}
    keyword_time = time.time() - keyword_start

    # Step 1
    x_match = keyword_results.get('x_axis', False) if x_label else False
    x_fb = keywords_match_robust(texts=x_label, keywords=date_keywords, model=model, description="X-axis label indicating activity date or time") if x_label else False
    has_date = bool(x_match) or bool(x_fb)
    checkpoint.add_step("X-Axis Date Label", has_date, 1,
        f"X-axis label: '{x_label}'" if has_date else f"X-axis label '{x_label or 'None'}' does not match date keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1))

    # Step 2
    y_match = keyword_results.get('y_axis', False) if y_label else False
    y_fb = keywords_match_robust(texts=y_label, keywords=reps_keywords, model=model, description="Y-axis label indicating cumulative reps") if y_label else False
    has_reps = bool(y_match) or bool(y_fb)
    checkpoint.add_step("Y-Axis Reps Label", has_reps, 2,
        f"Y-axis label: '{y_label}'" if has_reps else f"Y-axis label '{y_label or 'None'}' does not match reps keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1))

    # Step 3
    t_cum = keyword_results.get('title_cumulative', False) if chart_title else False
    t_time = keyword_results.get('title_time', False) if chart_title else False
    t_fb = keywords_match_robust(texts=chart_title, keywords=cumulative_title_keywords + time_title_keywords, model=model, description="chart title indicating cumulative reps over time") if chart_title else False
    has_title = bool(t_cum) or bool(t_time) or bool(t_fb)
    checkpoint.add_step("Chart Title", has_title, 3,
        f"Chart title: '{chart_title}'" if has_title else f"Chart title '{chart_title or 'None'}' does not match cumulative/time keywords",
        execution_time=keyword_time / max(len(vlm_tasks), 1))

    # Step 4: No overlap
    step_start = time.time()
    df = table_data.df if table_data else None
    table_start = table_data.start_row if table_data else 0
    table_end = table_data.end_row if table_data else (table_start + len(df) + 1 if df is not None else table_start + 220)
    table_start_col = table_data.start_col if table_data else 0
    table_end_col = table_data.end_col if table_data else None
    cum_anchor = cumulative_chart.get('position', {}).get('anchor_cell', {})
    if cum_anchor.get('row') is None:
        has_overlap = False
        overlap_details = "Chart anchor row is None (API bug) - skipping overlap check"
    else:
        safe_charts = [c for c in (chart_data or []) if c.get('position', {}).get('anchor_cell', {}).get('row') is not None]
        has_overlap, overlap_details = check_chart_overlap(cumulative_chart, table_start, table_end, safe_charts, table_start_col, table_end_col)
    checkpoint.add_step("No Chart Overlap", not has_overlap, 4,
        overlap_details if has_overlap else "Chart does not overlap with table or other charts",
        execution_time=time.time() - step_start)

    # Step 5: Cumulative data (validate against sheet's Cumulative Reps column)
    step_start = time.time()
    cumulative_valid = False
    cumulative_details = "Could not extract chart values"
    if rows is None:
        cumulative_details = "Sheet rows unavailable from setup() - cannot extract chart values"
    elif df is None or df.empty:
        cumulative_details = "Sheet table data unavailable"
    else:
        chart_values = get_series_column_values(cumulative_chart, 0, rows)
        if chart_values:
            non_increasing = sum(1 for i in range(1, len(chart_values)) if chart_values[i] < chart_values[i-1] - 0.01)
            if non_increasing > 0:
                cumulative_details = f"Values not monotonically increasing ({non_increasing} decreases found)"
            else:
                cum_col = None
                for col in df.columns:
                    if 'cumulative' in col.lower() and 'rep' in col.lower():
                        cum_col = col
                        break
                if cum_col:
                    expected = pd.to_numeric(df[cum_col], errors='coerce').dropna().values
                    comp = min(len(chart_values), len(expected))
                    matches = sum(1 for i in range(comp) if (expected[i] > 0 and abs(chart_values[i] - expected[i]) / expected[i] * 100 <= 1.0) or (expected[i] == 0 and chart_values[i] == 0))
                    rate = matches / comp if comp > 0 else 0
                    if rate >= 0.8:
                        cumulative_valid = True
                        cumulative_details = f"Cumulative data valid: {matches}/{comp} values match ({rate:.0%}), final value {chart_values[-1]:.0f} reps"
                    else:
                        cumulative_details = f"Only {matches}/{comp} values match ({rate:.0%}) against sheet cumulative column"
                else:
                    cumulative_valid = True
                    cumulative_details = f"Monotonically increasing data ({len(chart_values)} points), final value {chart_values[-1]:.0f} reps"
        else:
            cumulative_details = "No values extracted from chart series"
    checkpoint.add_step("Cumulative Data", cumulative_valid, 5, cumulative_details, execution_time=time.time() - step_start)

    # Step 6: Line plot
    step_start = time.time()
    is_line = chart_type in ['LINE', 'AREA']
    checkpoint.add_step("Line Plot", is_line, 6,
        f"Chart is a {chart_type} chart (line plot)" if is_line else f"Chart type is {chart_type}, expected LINE or AREA",
        execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5(browsing_history=None):
    """
    Grade Checkpoint 5: Website Visit Validation (4 pts).

    Outcome Evaluation:
    - A source URL for average adult workout sets was visited.
    - A source URL for Jay Cutler workout sets data was visited.
    - Avg adult sets source URL contains relevant information.
    - Jay Cutler source URL contains relevant information.
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Adult Sets URL Visited", False, 1, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Cutler URL Visited", False, 2, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Adult Sets Content Valid", False, 3, "No browsing history provided", execution_time=0)
        checkpoint.add_step("Cutler Content Valid", False, 4, "No browsing history provided", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    browsing_lower = [url.lower() for url in browsing_history]

    adult_keywords = ['sets', 'workout', 'how many', 'average', 'guide']
    cutler_keywords = ['cutler', 'jay']

    adult_urls = []
    cutler_urls = []
    adult_judge_details = ""
    cutler_judge_details = ""

    for i, url_lower in enumerate(browsing_lower):
        original_url = browsing_history[i]
        if any(kw in url_lower for kw in adult_keywords):
            adult_urls.append(original_url)
        if any(kw in url_lower for kw in cutler_keywords):
            cutler_urls.append(original_url)

    # LLM-as-judge backup
    judge_start = time.time()
    if model and (not adult_urls or not cutler_urls):
        judge_tasks = []
        for original_url in browsing_history[:6]:
            if not adult_urls:
                judge_tasks.append({'id': f'judge_adult|{original_url}', 'func': judge_url_relevance, 'args': (original_url, 'adult_sets', model)})
            if not cutler_urls:
                judge_tasks.append({'id': f'judge_cutler|{original_url}', 'func': judge_url_relevance, 'args': (original_url, 'cutler', model)})
        if judge_tasks:
            judge_results = parallel_execute(judge_tasks, max_workers=6)
            for original_url in browsing_history[:6]:
                if not adult_urls:
                    j = judge_results.get(f'judge_adult|{original_url}')
                    if j and j[0]:
                        adult_urls.append(original_url)
                        adult_judge_details = " (LLM-judged)"
                if not cutler_urls:
                    j = judge_results.get(f'judge_cutler|{original_url}')
                    if j and j[0]:
                        cutler_urls.append(original_url)
                        cutler_judge_details = " (LLM-judged)"
    judge_time = time.time() - judge_start

    # Step 1: Adult sets URL visited
    adult_visited = len(adult_urls) > 0
    checkpoint.add_step("Adult Sets URL Visited", adult_visited, 1,
        f"Found {len(adult_urls)} relevant URL(s){adult_judge_details}" if adult_visited else "No adult sets URL found in browsing history",
        execution_time=judge_time / 2)

    # Step 2: Cutler URL visited
    cutler_visited = len(cutler_urls) > 0
    checkpoint.add_step("Cutler URL Visited", cutler_visited, 2,
        f"Found {len(cutler_urls)} relevant URL(s){cutler_judge_details}" if cutler_visited else "No Jay Cutler URL found in browsing history",
        execution_time=judge_time / 2)

    # Steps 3-4: Content validation using raw extraction (sets, not pace)
    step_start = time.time()
    url_tasks = []
    if sheet_adult_sets and adult_urls and model:
        for url in adult_urls[:3]:
            url_tasks.append({'id': f'adult|{url}', 'func': extract_pace_from_url, 'args': (url, 'adult_sets', model)})
    if sheet_cutler_sets and cutler_urls and model:
        for url in cutler_urls[:3]:
            url_tasks.append({'id': f'cutler|{url}', 'func': extract_pace_from_url, 'args': (url, 'cutler', model)})

    url_results = parallel_execute(url_tasks, max_workers=6) if url_tasks else {}
    url_time = time.time() - step_start

    # Process adult sets results — range-based validation
    adult_content_valid = False
    adult_content_details = "No adult sets URLs to check"
    adult_extraction_failed = False

    if not sheet_adult_sets or sheet_adult_sets <= 0:
        adult_content_details = "No adult sets baseline value found in sheet (checkpoint 3 may have failed)"
    elif not model:
        adult_content_details = "Model not available for content validation"
    elif adult_urls:
        adult_extraction_failed = True
        for url in adult_urls[:3]:
            result = url_results.get(f'adult|{url}')
            if result is not None:
                data, details = result
                if data is not None:
                    adult_extraction_failed = False
                    val = data['value']
                    if ADULT_SETS_RANGE[0] <= val <= ADULT_SETS_RANGE[1]:
                        adult_content_valid = True
                        adult_content_details = f"URL value {val:.0f} sets within range [{ADULT_SETS_RANGE[0]}-{ADULT_SETS_RANGE[1]}] (sheet: {sheet_adult_sets:.0f})"
                        break
                    else:
                        adult_content_details = f"URL value {val:.0f} sets outside range [{ADULT_SETS_RANGE[0]}-{ADULT_SETS_RANGE[1]}]"
                else:
                    adult_content_details = details

    if not adult_content_valid and adult_extraction_failed and sheet_adult_sets and model and adult_urls:
        for url in adult_urls[:3]:
            judged, jd = judge_url_pace_match(url, 'adult_sets', sheet_adult_sets, model)
            if judged:
                adult_content_valid = True
                adult_content_details = f"LLM judge backup: {jd}"
                break
            else:
                adult_content_details = f"LLM judge backup: {jd}"

    checkpoint.add_step("Adult Sets Content Valid", adult_content_valid, 3, adult_content_details,
        execution_time=url_time / 2 if url_tasks else 0)

    # Process Cutler results — range-based validation
    cutler_content_valid = False
    cutler_content_details = "No Cutler URLs to check"
    cutler_extraction_failed = False

    if not sheet_cutler_sets or sheet_cutler_sets <= 0:
        cutler_content_details = "No Cutler baseline value found in sheet (checkpoint 3 may have failed)"
    elif not model:
        cutler_content_details = "Model not available for content validation"
    elif cutler_urls:
        cutler_extraction_failed = True
        for url in cutler_urls[:3]:
            result = url_results.get(f'cutler|{url}')
            if result is not None:
                data, details = result
                if data is not None:
                    cutler_extraction_failed = False
                    val = data['value']
                    if CUTLER_SETS_RANGE[0] <= val <= CUTLER_SETS_RANGE[1]:
                        cutler_content_valid = True
                        cutler_content_details = f"URL value {val:.0f} sets within range [{CUTLER_SETS_RANGE[0]}-{CUTLER_SETS_RANGE[1]}] (sheet: {sheet_cutler_sets:.0f})"
                        break
                    else:
                        cutler_content_details = f"URL value {val:.0f} sets outside range [{CUTLER_SETS_RANGE[0]}-{CUTLER_SETS_RANGE[1]}]"
                else:
                    cutler_content_details = details

    if not cutler_content_valid and cutler_extraction_failed and sheet_cutler_sets and model and cutler_urls:
        for url in cutler_urls[:3]:
            judged, jd = judge_url_pace_match(url, 'cutler', sheet_cutler_sets, model)
            if judged:
                cutler_content_valid = True
                cutler_content_details = f"LLM judge backup: {jd}"
                break
            else:
                cutler_content_details = f"LLM judge backup: {jd}"

    checkpoint.add_step("Cutler Content Valid", cutler_content_valid, 4, cutler_content_details,
        execution_time=url_time / 2 if url_tasks else 0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id=None, browsing_history=None):
    """
    Grade all checkpoints for the strength training analysis task.

    Args:
        workspace_doc_id (str, optional): Direct Google Sheets document ID to use
        browsing_history (list, optional): List of URLs visited during task execution

    Returns:
        Result: Evaluation results with checkpoint scores
    """
    total_start_time = time.time()

    try:
        setup(workspace_doc_id)

        global model
        model = load_model(model_id)

        checkpoints: List[Checkpoint] = []

        cp1_start = time.time()
        checkpoints.append(grade_checkpoint_1())
        print(f"  Checkpoint 1 took {time.time() - cp1_start:.2f}s")

        cp2_start = time.time()
        checkpoints.append(grade_checkpoint_2())
        print(f"  Checkpoint 2 took {time.time() - cp2_start:.2f}s")

        cp3_start = time.time()
        checkpoints.append(grade_checkpoint_3())
        print(f"  Checkpoint 3 took {time.time() - cp3_start:.2f}s")

        cp4_start = time.time()
        checkpoints.append(grade_checkpoint_4())
        print(f"  Checkpoint 4 took {time.time() - cp4_start:.2f}s")

        cp5_start = time.time()
        checkpoints.append(grade_checkpoint_5(browsing_history))
        print(f"  Checkpoint 5 took {time.time() - cp5_start:.2f}s")

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate strength training spreadsheet")
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
