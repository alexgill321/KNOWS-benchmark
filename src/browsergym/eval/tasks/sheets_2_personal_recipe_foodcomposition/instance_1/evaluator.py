"""Evaluator for the Personal Recipe Food Composition Google Sheets task."""

import os
import sys
from typing import List, Dict, Optional, Any
import time
import pandas as pd
import argparse

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

# Imports from eval_utils
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    extract_tables_from_sheet
)
from src.browsergym.eval.eval_utils.google_services_helpers import get_sheet_content
from src.browsergym.eval.eval_utils.text_utils import numerical_match_with_error
from src.browsergym.eval.eval_utils.table_utils import (
    find_matching_column_or_row,
    get_cell_background_color
)
from src.browsergym.eval.eval_utils.models import load_model

# Local utils
from src.browsergym.eval.tasks.sheets_2_personal_recipe_foodcomposition.utils import (
    is_valid_usda_url,
    fetch_usda_page_title,
    ingredient_matches_usda_page,
    normalize_nutrient_name,
    get_nutrient_columns_mapping,
    colors_are_similar,
    colors_are_distinct,
    find_column_by_keywords,
    find_ingredient_row,
    COLUMN_KEYWORDS,
    INGREDIENT_KEYWORDS,
    EXCLUDED_KEYWORDS,
    EXPECTED_INGREDIENTS,
    EXCLUDED_INGREDIENT,
    MACRO_NUTRIENTS,
    MINERAL_NUTRIENTS,
    VITAMIN_NUTRIENTS,
    ALL_NUTRIENTS,
    FDA_DAILY_VALUES,
    VALUE_TOLERANCE,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/sheets_2_personal_recipe_foodcomposition/instance_1/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Model configuration
model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global variables
sheet_id = None
sheet_raw = None
df = None
gold_data = None
matched_columns = {}


def load_gold_data():
    """Load gold label data from CSV file."""
    global gold_data
    gold_path = os.path.join(TASK_DIR, "data", "gold_nutrients.csv")
    try:
        gold_data = pd.read_csv(gold_path)
        print(f"Loaded gold data with {len(gold_data)} ingredients")
        return gold_data
    except Exception as e:
        print(f"Error loading gold data: {e}")
        return None


def setup(workspace_doc_id: str):
    """
    Setup function to initialize the evaluator.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, sheet_raw, df, gold_data

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    # Fetch raw sheet data for formatting checks
    sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

    # Load gold label data
    gold_data = load_gold_data()

    # Extract data as DataFrame
    if sheet_raw:
        try:
            sheets = sheet_raw.get('sheets', [])
            if sheets:
                grid_data = sheets[0].get('data', [{}])[0]
                rows = grid_data.get('rowData', [])

                if rows:
                    # Find the row with actual column headers (contains "Ingredients")
                    header_row_idx = None
                    for idx, row in enumerate(rows):
                        values = row.get('values', [])
                        if values:
                            for cell in values:
                                cell_val = cell.get('formattedValue', '')
                                if cell_val and 'ingredient' in cell_val.lower():
                                    header_row_idx = idx
                                    break
                            if header_row_idx is not None:
                                break

                    if header_row_idx is None:
                        # Default to row 1 (after merged header row)
                        header_row_idx = 1

                    header_row = rows[header_row_idx].get('values', [])
                    headers = [cell.get('formattedValue', f'Column{i}') for i, cell in enumerate(header_row)]

                    # Extract data rows
                    data_rows = []
                    for row in rows[header_row_idx + 1:]:
                        values = row.get('values', [])
                        row_data = [cell.get('formattedValue', '') for cell in values]
                        # Pad to match header length
                        row_data = (row_data + [''] * len(headers))[:len(headers)]
                        if any(row_data):  # Skip empty rows
                            data_rows.append(row_data)

                    df = pd.DataFrame(data_rows, columns=headers)
                    print(f"Extracted DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error extracting DataFrame: {e}")
            df = None


def grade_checkpoint_1():
    """
    Checkpoint 1: Spreadsheet Structure & Column Layout (16 pts)
    Validates column presence and ordering.

    Outcome Evaluation:
    - "Ingredients" column exists and is first.
    - "Link" column exists and is second.
    - Carbohydrates, Fat, Fiber, Protein, Sugar columns present.
    - Calcium, Iron, Potassium, Sodium columns present.
    - Vitamin A, Vitamin C columns present.
    - Macros, Minerals, Vitamins columns are in alphabetical order.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global model, matched_columns
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=16, result=0, name="Spreadsheet Structure & Column Layout")

    if df is None or df.empty:
        checkpoint.add_step("Data Extraction", False, 1,
                          "No data found in spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    columns_lower = [c.lower() for c in columns]

    step_num = 0

    # Step 1: Ingredients column exists and is first
    step_num += 1
    step_start = time.time()
    ingredients_col = find_column_by_keywords(columns, COLUMN_KEYWORDS["Ingredients"])

    if not ingredients_col:
        # LLM fallback
        if model is None:
            model = load_model(model_id)
        result = find_matching_column_or_row(df, "A column for ingredient names", model,
                                             search_type="column", example_keywords=COLUMN_KEYWORDS["Ingredients"])
        if result:
            _, ingredients_col = result

    if ingredients_col and columns.index(ingredients_col) == 0:
        matched_columns["Ingredients"] = ingredients_col
        checkpoint.add_step("Ingredients Column First", True, step_num,
                          f"Found '{ingredients_col}' as first column",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Ingredients Column First", False, step_num,
                          f"Ingredients column not found or not first. First column: '{columns[0] if columns else 'N/A'}'",
                          execution_time=time.time() - step_start)

    # Step 2: Link column exists and is second
    step_num += 1
    step_start = time.time()
    link_col = find_column_by_keywords(columns, COLUMN_KEYWORDS["Link"])

    if not link_col:
        # LLM fallback
        if model is None:
            model = load_model(model_id)
        result = find_matching_column_or_row(df, "A column for links or URLs", model,
                                             search_type="column", example_keywords=COLUMN_KEYWORDS["Link"])
        if result:
            _, link_col = result

    if link_col and columns.index(link_col) == 1:
        matched_columns["Link"] = link_col
        checkpoint.add_step("Link Column Second", True, step_num,
                          f"Found '{link_col}' as second column",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Link Column Second", False, step_num,
                          f"Link column not found or not second. Second column: '{columns[1] if len(columns) > 1 else 'N/A'}'",
                          execution_time=time.time() - step_start)

    # Steps 3-7: Macro nutrient columns present
    nutrient_mapping = get_nutrient_columns_mapping(columns)

    for nutrient in MACRO_NUTRIENTS:
        step_num += 1
        step_start = time.time()

        if nutrient in nutrient_mapping:
            matched_columns[nutrient] = nutrient_mapping[nutrient]
            checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                              f"Found column '{nutrient_mapping[nutrient]}'",
                              execution_time=time.time() - step_start)
        else:
            # LLM fallback
            if model is None:
                model = load_model(model_id)
            result = find_matching_column_or_row(df, f"A column for {nutrient}", model,
                                                 search_type="column", example_keywords=COLUMN_KEYWORDS.get(nutrient, []))
            if result:
                _, col = result
                matched_columns[nutrient] = col
                checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                                  f"Found column '{col}' (LLM match)",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"{nutrient} Column Present", False, step_num,
                                  f"No column found for {nutrient}",
                                  execution_time=time.time() - step_start)

    # Steps 8-11: Mineral nutrient columns present
    for nutrient in MINERAL_NUTRIENTS:
        step_num += 1
        step_start = time.time()

        if nutrient in nutrient_mapping:
            matched_columns[nutrient] = nutrient_mapping[nutrient]
            checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                              f"Found column '{nutrient_mapping[nutrient]}'",
                              execution_time=time.time() - step_start)
        else:
            # LLM fallback
            if model is None:
                model = load_model(model_id)
            result = find_matching_column_or_row(df, f"A column for {nutrient}", model,
                                                 search_type="column", example_keywords=COLUMN_KEYWORDS.get(nutrient, []))
            if result:
                _, col = result
                matched_columns[nutrient] = col
                checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                                  f"Found column '{col}' (LLM match)",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"{nutrient} Column Present", False, step_num,
                                  f"No column found for {nutrient}",
                                  execution_time=time.time() - step_start)

    # Steps 12-13: Vitamin nutrient columns present
    for nutrient in VITAMIN_NUTRIENTS:
        step_num += 1
        step_start = time.time()

        if nutrient in nutrient_mapping:
            matched_columns[nutrient] = nutrient_mapping[nutrient]
            checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                              f"Found column '{nutrient_mapping[nutrient]}'",
                              execution_time=time.time() - step_start)
        else:
            # LLM fallback
            if model is None:
                model = load_model(model_id)
            result = find_matching_column_or_row(df, f"A column for {nutrient}", model,
                                                 search_type="column", example_keywords=COLUMN_KEYWORDS.get(nutrient, []))
            if result:
                _, col = result
                matched_columns[nutrient] = col
                checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                                  f"Found column '{col}' (LLM match)",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"{nutrient} Column Present", False, step_num,
                                  f"No column found for {nutrient}",
                                  execution_time=time.time() - step_start)

    # Step 14: Macros columns are in alphabetical order
    step_num += 1
    step_start = time.time()
    macro_cols = [matched_columns.get(n) for n in MACRO_NUTRIENTS if n in matched_columns]
    macro_indices = [columns.index(c) for c in macro_cols if c in columns]

    if len(macro_indices) == len(MACRO_NUTRIENTS) and macro_indices == sorted(macro_indices):
        checkpoint.add_step("Macros Alphabetical Order", True, step_num,
                          "Macro nutrient columns are in alphabetical order",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Macros Alphabetical Order", False, step_num,
                          f"Macro columns not in alphabetical order or missing ({len(macro_indices)}/{len(MACRO_NUTRIENTS)} found)",
                          execution_time=time.time() - step_start)

    # Step 15: Minerals columns are in alphabetical order
    step_num += 1
    step_start = time.time()
    mineral_cols = [matched_columns.get(n) for n in MINERAL_NUTRIENTS if n in matched_columns]
    mineral_indices = [columns.index(c) for c in mineral_cols if c in columns]

    if len(mineral_indices) == len(MINERAL_NUTRIENTS) and mineral_indices == sorted(mineral_indices):
        checkpoint.add_step("Minerals Alphabetical Order", True, step_num,
                          "Mineral nutrient columns are in alphabetical order",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Minerals Alphabetical Order", False, step_num,
                          f"Mineral columns not in alphabetical order or missing ({len(mineral_indices)}/{len(MINERAL_NUTRIENTS)} found)",
                          execution_time=time.time() - step_start)

    # Step 16: Vitamins columns are in alphabetical order
    step_num += 1
    step_start = time.time()
    vitamin_cols = [matched_columns.get(n) for n in VITAMIN_NUTRIENTS if n in matched_columns]
    vitamin_indices = [columns.index(c) for c in vitamin_cols if c in columns]

    if len(vitamin_indices) == len(VITAMIN_NUTRIENTS) and vitamin_indices == sorted(vitamin_indices):
        checkpoint.add_step("Vitamins Alphabetical Order", True, step_num,
                          "Vitamin nutrient columns are in alphabetical order",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Vitamins Alphabetical Order", False, step_num,
                          f"Vitamin columns not in alphabetical order or missing ({len(vitamin_indices)}/{len(VITAMIN_NUTRIENTS)} found)",
                          execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2: Group Headers & Formatting (6 pts)
    Validates merged headers and formatting.

    Outcome Evaluation:
    - "Macros" merged header exists spanning macro columns.
    - "Minerals" merged header exists spanning mineral columns.
    - "Vitamins" merged header exists spanning vitamin columns.
    - Group headers are centered and italicized.
    - Column titles are bolded (excluding group headers).
    - Horizontal line under column titles exists.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Group Headers & Formatting")

    if not sheet_raw:
        checkpoint.add_step("Sheet Data", False, 1,
                          "Could not access raw sheet data",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i in range(1, 7):
                checkpoint.add_step(f"Check {i}", False, i, "No sheets found")
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Get merges and row data
        merges = sheets[0].get('merges', [])
        grid_data = sheets[0].get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        step_num = 0

        # Helper to check if a merge contains specific text
        def find_merge_with_text(text_pattern: str):
            for merge in merges:
                start_row = merge.get('startRowIndex', 0)
                start_col = merge.get('startColumnIndex', 0)

                if start_row < len(rows):
                    row = rows[start_row].get('values', [])
                    if start_col < len(row):
                        cell_value = row[start_col].get('formattedValue', '')
                        if text_pattern.lower() in cell_value.lower():
                            return merge, row[start_col]
            return None, None

        # Step 1: Macros merged header exists
        step_num += 1
        step_start = time.time()
        macro_merge, macro_cell = find_merge_with_text("macro")
        if macro_merge:
            span = macro_merge.get('endColumnIndex', 0) - macro_merge.get('startColumnIndex', 0)
            checkpoint.add_step("Macros Merged Header", True, step_num,
                              f"Found 'Macros' header spanning {span} columns",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Macros Merged Header", False, step_num,
                              "No merged header found containing 'Macros'",
                              execution_time=time.time() - step_start)

        # Step 2: Minerals merged header exists
        step_num += 1
        step_start = time.time()
        mineral_merge, mineral_cell = find_merge_with_text("mineral")
        if mineral_merge:
            span = mineral_merge.get('endColumnIndex', 0) - mineral_merge.get('startColumnIndex', 0)
            checkpoint.add_step("Minerals Merged Header", True, step_num,
                              f"Found 'Minerals' header spanning {span} columns",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Minerals Merged Header", False, step_num,
                              "No merged header found containing 'Minerals'",
                              execution_time=time.time() - step_start)

        # Step 3: Vitamins merged header exists
        step_num += 1
        step_start = time.time()
        vitamin_merge, vitamin_cell = find_merge_with_text("vitamin")
        if vitamin_merge:
            span = vitamin_merge.get('endColumnIndex', 0) - vitamin_merge.get('startColumnIndex', 0)
            checkpoint.add_step("Vitamins Merged Header", True, step_num,
                              f"Found 'Vitamins' header spanning {span} columns",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Vitamins Merged Header", False, step_num,
                              "No merged header found containing 'Vitamins'",
                              execution_time=time.time() - step_start)

        # Step 4: Group headers are centered and italicized
        step_num += 1
        step_start = time.time()
        headers_formatted = True
        format_details = []

        for name, cell in [("Macros", macro_cell), ("Minerals", mineral_cell), ("Vitamins", vitamin_cell)]:
            if cell:
                fmt = cell.get('effectiveFormat', {}).get('textFormat', {})
                h_align = cell.get('effectiveFormat', {}).get('horizontalAlignment', '')
                is_italic = fmt.get('italic', False)
                is_centered = h_align == 'CENTER'

                if not is_italic or not is_centered:
                    headers_formatted = False
                    format_details.append(f"{name}: italic={is_italic}, centered={is_centered}")
            else:
                headers_formatted = False
                format_details.append(f"{name}: not found")

        if headers_formatted:
            checkpoint.add_step("Headers Centered and Italicized", True, step_num,
                              "All group headers are centered and italicized",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Headers Centered and Italicized", False, step_num,
                              f"Headers not properly formatted: {'; '.join(format_details)}",
                              execution_time=time.time() - step_start)

        # Step 5: Column titles are bolded (excluding group headers)
        step_num += 1
        step_start = time.time()

        # Find the row with column titles (typically row after merged headers)
        title_row_idx = 1  # Default assumption
        for idx, row in enumerate(rows):
            values = row.get('values', [])
            if values:
                for cell in values:
                    cell_val = cell.get('formattedValue', '')
                    if cell_val and 'ingredient' in cell_val.lower():
                        title_row_idx = idx
                        break
                if title_row_idx != 1:
                    break

        if title_row_idx < len(rows):
            title_row = rows[title_row_idx].get('values', [])
            bold_count = 0
            total_titles = 0

            for cell in title_row:
                value = cell.get('formattedValue', '')
                if value:
                    total_titles += 1
                    fmt = cell.get('effectiveFormat', {}).get('textFormat', {})
                    if fmt.get('bold', False):
                        bold_count += 1

            if bold_count == total_titles and total_titles > 0:
                checkpoint.add_step("Column Titles Bolded", True, step_num,
                                  f"All {total_titles} column titles are bolded",
                                  execution_time=time.time() - step_start)
            else:
                checkpoint.add_step("Column Titles Bolded", False, step_num,
                                  f"Only {bold_count}/{total_titles} column titles are bolded",
                                  execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Column Titles Bolded", False, step_num,
                              "Could not find column title row",
                              execution_time=time.time() - step_start)

        # Step 6: Horizontal line under column titles
        step_num += 1
        step_start = time.time()

        has_border = False
        if title_row_idx < len(rows):
            title_row = rows[title_row_idx].get('values', [])
            for cell in title_row:
                borders = cell.get('effectiveFormat', {}).get('borders', {})
                bottom = borders.get('bottom', {})
                if bottom.get('style') and bottom.get('style') != 'NONE':
                    has_border = True
                    break

        if has_border:
            checkpoint.add_step("Horizontal Line Under Titles", True, step_num,
                              "Found horizontal line (border) under column titles",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Horizontal Line Under Titles", False, step_num,
                              "No horizontal line found under column titles",
                              execution_time=time.time() - step_start)

    except Exception as e:
        for i in range(1, 7):
            checkpoint.add_step(f"Format Check {i}", False, i, f"Error: {str(e)[:50]}")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3: Color Formatting (4 pts)
    Validates background colors for nutrient groups.

    Outcome Evaluation:
    - Macro columns share same background color.
    - Mineral columns share same background color.
    - Vitamin columns share same background color.
    - Three groups have distinct colors from each other.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Color Formatting")

    if not sheet_raw or df is None:
        for i in range(1, 5):
            checkpoint.add_step(f"Color Check {i}", False, i, "No sheet data available")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i in range(1, 5):
                checkpoint.add_step(f"Color Check {i}", False, i, "No sheets found")
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        grid_data = sheets[0].get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        columns = list(df.columns)

        def get_col_index(col_name):
            try:
                return columns.index(col_name)
            except ValueError:
                return -1

        def get_bg_color(row_idx, col_idx):
            if row_idx < len(rows):
                row = rows[row_idx].get('values', [])
                if col_idx < len(row):
                    fmt = row[col_idx].get('effectiveFormat', {})
                    bg = fmt.get('backgroundColor', {})
                    return {'red': bg.get('red', 1), 'green': bg.get('green', 1), 'blue': bg.get('blue', 1)}
            return None

        # Get column indices for each group
        macro_indices = [get_col_index(matched_columns.get(n, '')) for n in MACRO_NUTRIENTS if matched_columns.get(n)]
        mineral_indices = [get_col_index(matched_columns.get(n, '')) for n in MINERAL_NUTRIENTS if matched_columns.get(n)]
        vitamin_indices = [get_col_index(matched_columns.get(n, '')) for n in VITAMIN_NUTRIENTS if matched_columns.get(n)]

        # Use a data row for color checking (skip header rows)
        data_row_idx = 2  # Typically row 3 (0-indexed)
        # Find first data row
        for idx, row in enumerate(rows):
            values = row.get('values', [])
            if values and idx > 1:  # Skip header rows
                first_val = values[0].get('formattedValue', '') if values else ''
                if first_val and 'ingredient' not in first_val.lower() and 'macro' not in first_val.lower():
                    data_row_idx = idx
                    break

        step_num = 0

        # Step 1: Macro columns share same background color
        step_num += 1
        step_start = time.time()
        macro_colors = [get_bg_color(data_row_idx, idx) for idx in macro_indices if idx >= 0]
        macro_same = len(macro_colors) > 0 and all(colors_are_similar(macro_colors[0], c) for c in macro_colors)

        if macro_same:
            checkpoint.add_step("Macros Same Color", True, step_num,
                              f"All {len(macro_colors)} macro columns share the same background color",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Macros Same Color", False, step_num,
                              f"Macro columns have inconsistent or no background colors ({len(macro_colors)} found)",
                              execution_time=time.time() - step_start)

        # Step 2: Mineral columns share same background color
        step_num += 1
        step_start = time.time()
        mineral_colors = [get_bg_color(data_row_idx, idx) for idx in mineral_indices if idx >= 0]
        mineral_same = len(mineral_colors) > 0 and all(colors_are_similar(mineral_colors[0], c) for c in mineral_colors)

        if mineral_same:
            checkpoint.add_step("Minerals Same Color", True, step_num,
                              f"All {len(mineral_colors)} mineral columns share the same background color",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Minerals Same Color", False, step_num,
                              f"Mineral columns have inconsistent or no background colors ({len(mineral_colors)} found)",
                              execution_time=time.time() - step_start)

        # Step 3: Vitamin columns share same background color
        step_num += 1
        step_start = time.time()
        vitamin_colors = [get_bg_color(data_row_idx, idx) for idx in vitamin_indices if idx >= 0]
        vitamin_same = len(vitamin_colors) > 0 and all(colors_are_similar(vitamin_colors[0], c) for c in vitamin_colors)

        if vitamin_same:
            checkpoint.add_step("Vitamins Same Color", True, step_num,
                              f"All {len(vitamin_colors)} vitamin columns share the same background color",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Vitamins Same Color", False, step_num,
                              f"Vitamin columns have inconsistent or no background colors ({len(vitamin_colors)} found)",
                              execution_time=time.time() - step_start)

        # Step 4: Three groups have distinct colors
        step_num += 1
        step_start = time.time()

        group_colors = []
        if macro_colors:
            group_colors.append(macro_colors[0])
        if mineral_colors:
            group_colors.append(mineral_colors[0])
        if vitamin_colors:
            group_colors.append(vitamin_colors[0])

        if len(group_colors) == 3 and colors_are_distinct(group_colors):
            checkpoint.add_step("Distinct Group Colors", True, step_num,
                              "All three nutrient groups have distinct background colors",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Distinct Group Colors", False, step_num,
                              f"Nutrient groups do not have distinct colors ({len(group_colors)} groups found)",
                              execution_time=time.time() - step_start)

    except Exception as e:
        for i in range(1, 5):
            checkpoint.add_step(f"Color Check {i}", False, i, f"Error: {str(e)[:50]}")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4: Ingredients Present (8 pts)
    Validates that correct ingredients are present.

    Outcome Evaluation:
    - Raw Cashews row present.
    - Water row present.
    - Garlic row present.
    - Sea Salt row present.
    - Nutritional Yeast row present.
    - Lemon Juice row present.
    - Onion Powder row present.
    - Black Pepper row NOT present (excluded - no specified amount).
    """
    print("----------------- CHECKPOINT 4 ----------------")
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=8, result=0, name="Ingredients Present")

    if df is None or df.empty:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS + [EXCLUDED_INGREDIENT], 1):
            checkpoint.add_step(f"{ingredient}", False, i, "No data found")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    ingredient_col = matched_columns.get("Ingredients") or find_column_by_keywords(columns, COLUMN_KEYWORDS["Ingredients"])

    if not ingredient_col:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS + [EXCLUDED_INGREDIENT], 1):
            checkpoint.add_step(f"{ingredient}", False, i, "No ingredient column found")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    ingredients_in_sheet = df[ingredient_col].astype(str).str.lower().str.strip().tolist()

    step_num = 0

    # Check each expected ingredient is present
    for ingredient in EXPECTED_INGREDIENTS:
        step_num += 1
        step_start = time.time()

        keywords = INGREDIENT_KEYWORDS.get(ingredient, [ingredient.lower()])
        found = False

        # Keyword-based matching first
        for ing in ingredients_in_sheet:
            for keyword in keywords:
                if keyword.lower() in ing:
                    found = True
                    break
            if found:
                break

        # LLM fallback if keyword matching fails
        if not found:
            if model is None:
                model = load_model(model_id)
            # Simple LLM check
            for idx, ing in enumerate(ingredients_in_sheet):
                if ing and len(ing) > 1:
                    # Check semantic similarity
                    prompt_text = f"Does '{ing}' refer to the same ingredient as '{ingredient}'? Answer only Yes or No."
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
                    ]
                    try:
                        response = model(messages)
                        if response and 'yes' in response.lower():
                            found = True
                            break
                    except Exception as e:
                        print(f"LLM error for ingredient matching: {e}")

        if found:
            checkpoint.add_step(f"{ingredient} Present", True, step_num,
                              f"Found '{ingredient}' in spreadsheet",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"{ingredient} Present", False, step_num,
                              f"'{ingredient}' not found in spreadsheet",
                              execution_time=time.time() - step_start)

    # Check that Black Pepper is NOT present
    step_num += 1
    step_start = time.time()

    excluded_found = False
    for ing in ingredients_in_sheet:
        for keyword in EXCLUDED_KEYWORDS:
            if keyword.lower() in ing:
                excluded_found = True
                break
        if excluded_found:
            break

    if not excluded_found:
        checkpoint.add_step(f"{EXCLUDED_INGREDIENT} Not Present", True, step_num,
                          f"'{EXCLUDED_INGREDIENT}' correctly excluded (no specified amount)",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step(f"{EXCLUDED_INGREDIENT} Not Present", False, step_num,
                          f"'{EXCLUDED_INGREDIENT}' should not be present (no specified amount in recipe)",
                          execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Checkpoint 5: USDA Links Validation (7 pts)
    Validates that each ingredient has a valid USDA link.

    Outcome Evaluation:
    - Raw Cashews link valid and matches ingredient (via HTML parsing).
    - Water link valid and matches ingredient (via HTML parsing).
    - Garlic link valid and matches ingredient (via HTML parsing).
    - Sea Salt link valid and matches ingredient (via HTML parsing).
    - Nutritional Yeast link valid and matches ingredient (via HTML parsing).
    - Lemon Juice link valid and matches ingredient (via HTML parsing).
    - Onion Powder link valid and matches ingredient (via HTML parsing).
    """
    print("----------------- CHECKPOINT 5 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=7, result=0, name="USDA Links Validation")

    if df is None or df.empty:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS, 1):
            checkpoint.add_step(f"{ingredient} Link", False, i, "No data found")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    ingredient_col = matched_columns.get("Ingredients") or find_column_by_keywords(columns, COLUMN_KEYWORDS["Ingredients"])
    link_col = matched_columns.get("Link") or find_column_by_keywords(columns, COLUMN_KEYWORDS["Link"])

    if not ingredient_col or not link_col:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS, 1):
            checkpoint.add_step(f"{ingredient} Link", False, i, "Missing ingredient or link column")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_num = 0

    for ingredient in EXPECTED_INGREDIENTS:
        step_num += 1
        step_start = time.time()

        # Find the row for this ingredient
        row_match = None
        keywords = INGREDIENT_KEYWORDS.get(ingredient, [ingredient.lower()])

        for _, row in df.iterrows():
            cell_value = str(row[ingredient_col]).lower().strip()
            for keyword in keywords:
                if keyword.lower() in cell_value:
                    row_match = row
                    break
            if row_match is not None:
                break

        if row_match is None:
            checkpoint.add_step(f"{ingredient} Link Valid", False, step_num,
                              f"Ingredient '{ingredient}' not found in spreadsheet",
                              execution_time=time.time() - step_start)
            continue

        link = str(row_match[link_col]).strip()

        # Check if it's a valid USDA FoodData Central URL
        if not link or not is_valid_usda_url(link):
            checkpoint.add_step(f"{ingredient} Link Valid", False, step_num,
                              f"Invalid or non-USDA link: {link[:50] if link else 'empty'}...",
                              execution_time=time.time() - step_start)
            continue

        # Fetch the page and verify it matches the ingredient
        page_title = fetch_usda_page_title(link)

        if page_title and ingredient_matches_usda_page(ingredient, page_title):
            checkpoint.add_step(f"{ingredient} Link Valid", True, step_num,
                              f"USDA link verified for '{ingredient}'",
                              execution_time=time.time() - step_start)
        elif page_title:
            # Page fetched but may not match - partial credit
            checkpoint.add_step(f"{ingredient} Link Valid", True, step_num,
                              f"Valid USDA link (page: '{page_title[:30]}...')",
                              execution_time=time.time() - step_start)
        else:
            # Could not fetch page, but URL format is correct
            checkpoint.add_step(f"{ingredient} Link Valid", True, step_num,
                              f"Valid USDA URL format (could not verify page content)",
                              execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """
    Checkpoint 6: Nutrient Values Accuracy (77 pts)
    Validates nutrient values match gold labels within tolerance.
    Each nutrient column is worth 7 points (one per ingredient).

    Outcome Evaluation:
    - Carbohydrates values match gold labels within tolerance (7 pts).
    - Fat values match gold labels within tolerance (7 pts).
    - Fiber values match gold labels within tolerance (7 pts).
    - Protein values match gold labels within tolerance (7 pts).
    - Sugar values match gold labels within tolerance (7 pts).
    - Calcium values match gold labels within tolerance (7 pts).
    - Iron values match gold labels within tolerance (7 pts).
    - Potassium values match gold labels within tolerance (7 pts).
    - Sodium values match gold labels within tolerance (7 pts).
    - Vitamin A values match gold labels within tolerance (7 pts).
    - Vitamin C values match gold labels within tolerance (7 pts).
    """
    print("----------------- CHECKPOINT 6 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=77, result=0, name="Nutrient Values Accuracy")

    if df is None or df.empty or gold_data is None:
        for i, nutrient in enumerate(ALL_NUTRIENTS, 1):
            checkpoint.add_step(f"{nutrient} Values", False, i,
                              "No data or gold labels available", max_score=7)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    ingredient_col = matched_columns.get("Ingredients") or find_column_by_keywords(columns, COLUMN_KEYWORDS["Ingredients"])

    if not ingredient_col:
        for i, nutrient in enumerate(ALL_NUTRIENTS, 1):
            checkpoint.add_step(f"{nutrient} Values", False, i,
                              "No ingredient column found", max_score=7)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_num = 0
    tolerance_percent = VALUE_TOLERANCE * 100  # Convert to percentage

    for nutrient in ALL_NUTRIENTS:
        step_num += 1
        step_start = time.time()

        sheet_col = matched_columns.get(nutrient)
        if not sheet_col:
            checkpoint.add_step(f"{nutrient} Values", False, step_num,
                              f"No column found for {nutrient}", max_score=7)
            continue

        # Find matching gold column
        gold_col = None
        for gc in gold_data.columns:
            if nutrient.lower() in gc.lower():
                gold_col = gc
                break

        if not gold_col:
            checkpoint.add_step(f"{nutrient} Values", False, step_num,
                              f"No gold data column for {nutrient}", max_score=7)
            continue

        # Check each ingredient's value
        matches = 0
        mismatches = []

        for ingredient in EXPECTED_INGREDIENTS:
            # Find in sheet
            sheet_value = None
            keywords = INGREDIENT_KEYWORDS.get(ingredient, [ingredient.lower()])

            for _, row in df.iterrows():
                cell_value = str(row[ingredient_col]).lower().strip()
                for keyword in keywords:
                    if keyword.lower() in cell_value:
                        try:
                            sheet_value = float(str(row[sheet_col]).replace(',', ''))
                        except (ValueError, TypeError):
                            sheet_value = None
                        break
                if sheet_value is not None:
                    break

            # Find in gold data
            gold_value = None
            for _, row in gold_data.iterrows():
                if ingredient.lower() in str(row['Ingredient']).lower():
                    try:
                        gold_value = float(row[gold_col])
                    except (ValueError, TypeError):
                        gold_value = None
                    break

            # Compare values
            if sheet_value is not None and gold_value is not None:
                if gold_value == 0:
                    if sheet_value == 0:
                        matches += 1
                    else:
                        mismatches.append(f"{ingredient}: {sheet_value} vs 0")
                else:
                    is_match, _ = numerical_match_with_error(gold_value, sheet_value, error_percent=tolerance_percent)
                    if is_match:
                        matches += 1
                    else:
                        mismatches.append(f"{ingredient}: {sheet_value:.2f} vs {gold_value:.2f}")
            elif sheet_value is None and gold_value is None:
                matches += 1
            elif sheet_value is None:
                mismatches.append(f"{ingredient}: missing in sheet")
            else:
                mismatches.append(f"{ingredient}: missing in gold")

        success = matches == len(EXPECTED_INGREDIENTS)

        if success:
            checkpoint.add_step(f"{nutrient} Values", True, step_num,
                              f"All {matches}/{len(EXPECTED_INGREDIENTS)} values match within {tolerance_percent}% tolerance",
                              score=matches, max_score=7,
                              execution_time=time.time() - step_start)
        else:
            detail = f"{matches}/{len(EXPECTED_INGREDIENTS)} match"
            if mismatches:
                detail += f". Mismatches: {'; '.join(mismatches[:2])}"
            checkpoint.add_step(f"{nutrient} Values", False, step_num,
                              detail, score=matches, max_score=7,
                              execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_7():
    """
    Checkpoint 7: Bold Formatting for >10% DV (1 pt)
    Validates that values exceeding 10% DV are bolded.

    Outcome Evaluation:
    - Values exceeding 10% DV are bolded per FDA guidelines.
    """
    print("----------------- CHECKPOINT 7 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=1, result=0, name="Bold Formatting for >10% DV")

    if not sheet_raw or df is None or gold_data is None:
        checkpoint.add_step("Bold Formatting", False, 1,
                          "No sheet data available",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            checkpoint.add_step("Bold Formatting", False, 1, "No sheets found")
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        grid_data = sheets[0].get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        columns = list(df.columns)

        # Find header row index
        header_row_idx = 1
        for idx, row in enumerate(rows):
            values = row.get('values', [])
            if values:
                for cell in values:
                    cell_val = cell.get('formattedValue', '')
                    if cell_val and 'ingredient' in cell_val.lower():
                        header_row_idx = idx
                        break

        correct_bold = 0
        total_checks = 0

        # Check each nutrient cell
        for row_idx, row in enumerate(rows[header_row_idx + 1:], header_row_idx + 1):
            values = row.get('values', [])

            for nutrient in ALL_NUTRIENTS:
                col_name = matched_columns.get(nutrient)
                if not col_name or col_name not in columns:
                    continue

                col_idx = columns.index(col_name)
                if col_idx >= len(values):
                    continue

                cell = values[col_idx]
                cell_value = cell.get('formattedValue', '')

                try:
                    value = float(str(cell_value).replace(',', ''))
                except (ValueError, TypeError):
                    continue

                total_checks += 1

                # Calculate if >10% DV
                dv = FDA_DAILY_VALUES.get(nutrient, 0)
                if dv > 0:
                    percent_dv = (value / dv) * 100
                    should_be_bold = percent_dv > 10
                else:
                    should_be_bold = False

                # Check if cell is bold
                fmt = cell.get('effectiveFormat', {}).get('textFormat', {})
                is_bold = fmt.get('bold', False)

                if should_be_bold == is_bold:
                    correct_bold += 1

        # Calculate success
        if total_checks > 0:
            accuracy = (correct_bold / total_checks) * 100
            success = accuracy >= 80  # Allow some tolerance

            if success:
                checkpoint.add_step("Bold Formatting", True, 1,
                                  f"{correct_bold}/{total_checks} cells correctly formatted ({accuracy:.1f}%)",
                                  execution_time=time.time() - checkpoint_start)
            else:
                checkpoint.add_step("Bold Formatting", False, 1,
                                  f"Only {correct_bold}/{total_checks} cells correctly formatted ({accuracy:.1f}%)",
                                  execution_time=time.time() - checkpoint_start)
        else:
            checkpoint.add_step("Bold Formatting", False, 1,
                              "No nutrient cells found to check",
                              execution_time=time.time() - checkpoint_start)

    except Exception as e:
        checkpoint.add_step("Bold Formatting", False, 1, f"Error: {str(e)[:50]}")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_8(browsing_history: Optional[List[str]] = None):
    """
    Checkpoint 8: Website Visit Validation (8 pts)
    Validates that required websites were visited.

    Outcome Evaluation:
    - Recipe URL (rainbowplantlife.com) visited (1 pt).
    - USDA database URL visited for Raw Cashews (1 pt).
    - USDA database URL visited for Water (1 pt).
    - USDA database URL visited for Garlic (1 pt).
    - USDA database URL visited for Sea Salt (1 pt).
    - USDA database URL visited for Nutritional Yeast (1 pt).
    - USDA database URL visited for Lemon Juice (1 pt).
    - USDA database URL visited for Onion Powder (1 pt).
    """
    print("----------------- CHECKPOINT 8 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=8, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Recipe URL Visited", False, 1,
                          "No browsing history provided")
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS, 2):
            checkpoint.add_step(f"USDA URL for {ingredient}", False, i,
                              "No browsing history provided")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    browsing_lower = [url.lower() for url in browsing_history]

    step_num = 0

    # Step 1: Recipe URL visited
    step_num += 1
    step_start = time.time()
    recipe_visited = any('rainbowplantlife.com' in url for url in browsing_lower)

    if recipe_visited:
        checkpoint.add_step("Recipe URL Visited", True, step_num,
                          "Found visit to rainbowplantlife.com",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Recipe URL Visited", False, step_num,
                          "No visit to rainbowplantlife.com found",
                          execution_time=time.time() - step_start)

    # Steps 2-8: USDA URL visited for each ingredient
    usda_visited = any('fdc.nal.usda.gov' in url for url in browsing_lower)

    for ingredient in EXPECTED_INGREDIENTS:
        step_num += 1
        step_start = time.time()

        if usda_visited:
            checkpoint.add_step(f"USDA URL for {ingredient}", True, step_num,
                              f"USDA database was visited",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"USDA URL for {ingredient}", False, step_num,
                              f"No USDA database visit found",
                              execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None, browsing_history: List[str] = None):
    """
    Grade all checkpoints for the food composition task.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
        browsing_history: List of URLs visited during task execution.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    try:
        # Setup document processing
        setup(workspace_doc_id)

        checkpoints: List[Checkpoint] = []

        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())
        checkpoints.append(grade_checkpoint_5())
        checkpoints.append(grade_checkpoint_6())
        checkpoints.append(grade_checkpoint_7())
        checkpoints.append(grade_checkpoint_8(browsing_history))

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
    parser = argparse.ArgumentParser(description="Evaluate food composition spreadsheet")
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
