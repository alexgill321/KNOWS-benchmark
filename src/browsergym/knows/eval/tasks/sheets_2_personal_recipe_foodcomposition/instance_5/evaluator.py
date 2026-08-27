"""Evaluator for the Personal Recipe Food Composition Google Sheets task.

Instance 5: Easy Homemade Chili from https://www.budgetbytes.com/easy-homemade-chili/
"""

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
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports from eval_utils
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    extract_tables_from_sheet,
    extract_sheet_data,
    get_sheet_content,
    detect_header_row,
)
from src.browsergym.knows.eval.eval_utils.text_utils import (
    numerical_match_with_error,
    keywords_exact_match,
    keywords_match_robust,
)
from src.browsergym.knows.eval.eval_utils.table_utils import (
    get_cell_background_color,
    colors_are_similar,
    colors_are_distinct,
    match_columns,
    find_merged_cell_by_text,
    get_merge_column_span,
    is_cell_centered,
    is_cell_italic,
    is_cell_bold,
    row_has_bottom_border,
    row_has_top_border,
    count_bold_cells_in_row,
)
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import (
    parallel_download,
    fast_parallel_vlm_calls,
)

# Local utils (template-specific functions and constants)
from src.browsergym.knows.eval.tasks.sheets_2_personal_recipe_foodcomposition.utils import (
    fetch_usda_page_title,
    fetch_usda_nutrients,
    ingredient_matches_usda_page,
    validate_usda_fallback,
    extract_food_id_from_url,
    parse_nutrient_value,
    COLUMN_KEYWORDS,
    MACRO_NUTRIENTS,
    MINERAL_NUTRIENTS,
    VITAMIN_NUTRIENTS,
    ALL_NUTRIENTS,
    FDA_DAILY_VALUES,
    VALUE_TOLERANCE,
)
from src.browsergym.knows.eval.eval_utils.web_utils import is_url_from_domain

# =============================================================================
# Instance-specific constants (specific to easy homemade chili recipe)
# =============================================================================
EXPECTED_INGREDIENTS = [
    "ground beef",
    "onion",
    "tomato sauce",
    "kidney beans",
    "stewed tomatoes",
    "chili powder",
]

EXCLUDED_INGREDIENT = "water"

# Keyword mappings for ingredient detection (instance-specific)
INGREDIENT_KEYWORDS = {
    "ground beef": ["ground beef", "beef"],
    "onion": ["onion"],
    "tomato sauce": ["tomato sauce"],
    "kidney beans": ["kidney bean", "kidney beans"],
    "stewed tomatoes": ["stewed tomato", "stewed tomatoes"],
    "chili powder": ["chili powder"],
}

# Keywords to detect excluded ingredient
EXCLUDED_KEYWORDS = ["water"]

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/sheets_2_personal_recipe_foodcomposition/instance_5/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Recipe domain for browsing history validation
RECIPE_DOMAIN = "allrecipes.com"

# Model configuration
model = None
model_id = "gemini-2.5-flash-google-ai"

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global variables
sheet_id = None
sheet_raw = None
df = None
gold_data = None
gold_by_ingredient = {}  # Pre-built lookup: ingredient_lower -> gold row data
matched_columns = {}
matched_ingredients = {}  # Maps ingredient name -> row index in df (from checkpoint 4)
header_row_idx = None  # Detected header row index


def load_gold_data():
    """Load gold label data from CSV file and build lookup dict."""
    global gold_data, gold_by_ingredient
    gold_path = os.path.join(TASK_DIR, "data", "gold_nutrients.csv")
    try:
        gold_data = pd.read_csv(gold_path, comment='#')
        print(f"Loaded gold data with {len(gold_data)} ingredients")

        # Pre-build lookup dict for O(1) access in checkpoint 6
        gold_by_ingredient = {}
        for _, row in gold_data.iterrows():
            ingredient_key = str(row['Ingredient']).lower().strip()
            gold_by_ingredient[ingredient_key] = row
        print(f"  Built gold lookup dict with {len(gold_by_ingredient)} entries")

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
    global sheet_id, sheet_raw, df, gold_data, header_row_idx

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
                    # Use detect_header_row to find the header row
                    header_row_idx = detect_header_row(rows)
                    print(f"Detected header row at index: {header_row_idx}")

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
                          execution_time=time.time() - checkpoint_start,
                          category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)

    # Use standardized match_columns() - keyword matching first, then parallel LLM fallback
    all_columns_to_find = ["Ingredients", "Link"] + ALL_NUTRIENTS
    required_columns = [(col_name, COLUMN_KEYWORDS.get(col_name, [])) for col_name in all_columns_to_find]

    if model is None:
        model = load_model(model_id)

    print(f"  [PARALLEL] Matching {len(required_columns)} columns with keyword + LLM fallback...")
    match_start = time.time()
    all_matches, col_methods = match_columns(df, required_columns, model=model, strict=True, parallel=True, max_workers=5, return_methods=True)
    print(f"  [PARALLEL] Column matching completed in {time.time() - match_start:.2f}s")

    # Log matches for debugging
    for col_name in all_columns_to_find:
        if col_name in all_matches:
            print(f"  [DEBUG] {col_name} column '{all_matches[col_name]}' matched")

    # Phase 3: Process results and add checkpoint steps
    step_num = 0

    # Step 1: Ingredients column exists and is first
    step_num += 1
    step_start = time.time()
    ingredients_col = all_matches.get("Ingredients")

    if ingredients_col and columns.index(ingredients_col) == 0:
        matched_columns["Ingredients"] = ingredients_col
        checkpoint.add_step("Ingredients Column First", True, step_num,
                          f"Found '{ingredients_col}' as first column",
                          execution_time=time.time() - step_start,
                          category=StepCategory.DETERMINISTIC if col_methods.get("Ingredients") == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)
    else:
        checkpoint.add_step("Ingredients Column First", False, step_num,
                          f"Ingredients column not found or not first. First column: '{columns[0] if columns else 'N/A'}'",
                          execution_time=time.time() - step_start,
                          category=StepCategory.DETERMINISTIC if col_methods.get("Ingredients") == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)

    # Step 2: Link column exists and is second
    step_num += 1
    step_start = time.time()
    link_col = all_matches.get("Link")

    if link_col and columns.index(link_col) == 1:
        matched_columns["Link"] = link_col
        checkpoint.add_step("Link Column Second", True, step_num,
                          f"Found '{link_col}' as second column",
                          execution_time=time.time() - step_start,
                          category=StepCategory.DETERMINISTIC if col_methods.get("Link") == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)
    else:
        checkpoint.add_step("Link Column Second", False, step_num,
                          f"Link column not found or not second. Second column: '{columns[1] if len(columns) > 1 else 'N/A'}'",
                          execution_time=time.time() - step_start,
                          category=StepCategory.DETERMINISTIC if col_methods.get("Link") == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)

    # Steps 3-13: All nutrient columns present (Macros, Minerals, Vitamins)
    for nutrient in ALL_NUTRIENTS:
        step_num += 1
        step_start = time.time()

        nutrient_col = all_matches.get(nutrient)

        if nutrient_col:
            matched_columns[nutrient] = nutrient_col
            checkpoint.add_step(f"{nutrient} Column Present", True, step_num,
                              f"Found column '{nutrient_col}'",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC if col_methods.get(nutrient) == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)
        else:
            print(f"  [DEBUG] {nutrient} column NOT FOUND")
            checkpoint.add_step(f"{nutrient} Column Present", False, step_num,
                              f"No column found for {nutrient}",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC if col_methods.get(nutrient) == "keyword" else StepCategory.LLM_VLM_JUDGEMENT)

    # Step 14: Macros columns are in alphabetical order
    step_num += 1
    step_start = time.time()
    macro_cols = [matched_columns.get(n) for n in MACRO_NUTRIENTS if n in matched_columns]
    macro_indices = [columns.index(c) for c in macro_cols if c in columns]

    if len(macro_indices) == len(MACRO_NUTRIENTS) and macro_indices == sorted(macro_indices):
        checkpoint.add_step("Macros Alphabetical Order", True, step_num,
                          "Macro nutrient columns are in alphabetical order",
                          execution_time=time.time() - step_start,
                          category=StepCategory.STRUCTURAL)
    else:
        checkpoint.add_step("Macros Alphabetical Order", False, step_num,
                          f"Macro columns not in alphabetical order or missing ({len(macro_indices)}/{len(MACRO_NUTRIENTS)} found)",
                          execution_time=time.time() - step_start,
                          category=StepCategory.STRUCTURAL)

    # Step 15: Minerals columns are in alphabetical order
    step_num += 1
    step_start = time.time()
    mineral_cols = [matched_columns.get(n) for n in MINERAL_NUTRIENTS if n in matched_columns]
    mineral_indices = [columns.index(c) for c in mineral_cols if c in columns]

    if len(mineral_indices) == len(MINERAL_NUTRIENTS) and mineral_indices == sorted(mineral_indices):
        checkpoint.add_step("Minerals Alphabetical Order", True, step_num,
                          "Mineral nutrient columns are in alphabetical order",
                          execution_time=time.time() - step_start,
                          category=StepCategory.STRUCTURAL)
    else:
        checkpoint.add_step("Minerals Alphabetical Order", False, step_num,
                          f"Mineral columns not in alphabetical order or missing ({len(mineral_indices)}/{len(MINERAL_NUTRIENTS)} found)",
                          execution_time=time.time() - step_start,
                          category=StepCategory.STRUCTURAL)

    # Step 16: Vitamins columns are in alphabetical order
    step_num += 1
    step_start = time.time()
    vitamin_cols = [matched_columns.get(n) for n in VITAMIN_NUTRIENTS if n in matched_columns]
    vitamin_indices = [columns.index(c) for c in vitamin_cols if c in columns]

    if len(vitamin_indices) == len(VITAMIN_NUTRIENTS) and vitamin_indices == sorted(vitamin_indices):
        checkpoint.add_step("Vitamins Alphabetical Order", True, step_num,
                          "Vitamin nutrient columns are in alphabetical order",
                          execution_time=time.time() - step_start,
                          category=StepCategory.STRUCTURAL)
    else:
        checkpoint.add_step("Vitamins Alphabetical Order", False, step_num,
                          f"Vitamin columns not in alphabetical order or missing ({len(vitamin_indices)}/{len(VITAMIN_NUTRIENTS)} found)",
                          execution_time=time.time() - step_start,
                          category=StepCategory.STRUCTURAL)

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
                          execution_time=time.time() - checkpoint_start,
                          category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i in range(1, 7):
                checkpoint.add_step(f"Check {i}", False, i, "No sheets found",
                                  category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Get merges and row data
        merges = sheets[0].get('merges', [])
        grid_data = sheets[0].get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        step_num = 0

        # Group headers to check: (step_name, text_pattern, display_name)
        group_headers = [
            ("Macros Merged Header", "macro", "Macros"),
            ("Minerals Merged Header", "mineral", "Minerals"),
            ("Vitamins Merged Header", "vitamin", "Vitamins"),
        ]

        # Store found cells for formatting check
        header_cells = {}

        # Steps 1-3: Check each merged header exists
        for step_name, text_pattern, display_name in group_headers:
            step_num += 1
            step_start = time.time()

            merge, cell = find_merged_cell_by_text(merges, rows, text_pattern)
            header_cells[display_name] = cell

            if merge:
                span = get_merge_column_span(merge)
                checkpoint.add_step(step_name, True, step_num,
                                  f"Found '{display_name}' header spanning {span} columns",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step(step_name, False, step_num,
                                  f"No merged header found containing '{display_name}'",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)

        # Step 4: Group headers are centered and italicized
        step_num += 1
        step_start = time.time()
        headers_formatted = True
        format_details = []

        for name, cell in header_cells.items():
            if cell:
                italic = is_cell_italic(cell)
                centered = is_cell_centered(cell)

                if not italic or not centered:
                    headers_formatted = False
                    format_details.append(f"{name}: italic={italic}, centered={centered}")
            else:
                headers_formatted = False
                format_details.append(f"{name}: not found")

        if headers_formatted:
            checkpoint.add_step("Headers Centered and Italicized", True, step_num,
                              "All group headers are centered and italicized",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Headers Centered and Italicized", False, step_num,
                              f"Headers not properly formatted: {'; '.join(format_details)}",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC)

        # Step 5: Column titles are bolded (excluding group headers)
        step_num += 1
        step_start = time.time()

        if header_row_idx is not None and header_row_idx < len(rows):
            title_row = rows[header_row_idx]
            bold_count, total_titles = count_bold_cells_in_row(title_row)

            if bold_count == total_titles and total_titles > 0:
                checkpoint.add_step("Column Titles Bolded", True, step_num,
                                  f"All {total_titles} column titles are bolded",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step("Column Titles Bolded", False, step_num,
                                  f"Only {bold_count}/{total_titles} column titles are bolded",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Column Titles Bolded", False, step_num,
                              "Could not find column title row",
                              execution_time=time.time() - step_start,
                              category=StepCategory.EXECUTION_ERROR)

        # Step 6: Horizontal line under column titles
        step_num += 1
        step_start = time.time()

        if header_row_idx is not None and header_row_idx < len(rows):
            has_bottom = row_has_bottom_border(rows[header_row_idx])
            has_top_below = False
            if header_row_idx + 1 < len(rows):
                has_top_below = row_has_top_border(rows[header_row_idx + 1])

            # Check for frozen rows - frozen boundary creates a visual line
            frozen_rows = sheets[0].get('properties', {}).get('gridProperties', {}).get('frozenRowCount', 0)
            has_frozen_line = frozen_rows > header_row_idx

            has_line = has_bottom or has_top_below or has_frozen_line

            if has_line:
                if has_bottom:
                    line_type = "bottom border on header"
                elif has_top_below:
                    line_type = "top border on data row"
                else:
                    line_type = "frozen row boundary"
                checkpoint.add_step("Horizontal Line Under Titles", True, step_num,
                                  f"Found horizontal line ({line_type}) under column titles",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step("Horizontal Line Under Titles", False, step_num,
                                  "No horizontal line found under column titles",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Horizontal Line Under Titles", False, step_num,
                              "Could not find column title row",
                              execution_time=time.time() - step_start,
                              category=StepCategory.EXECUTION_ERROR)

    except Exception as e:
        for i in range(1, 7):
            checkpoint.add_step(f"Format Check {i}", False, i, f"Error: {str(e)[:50]}",
                              category=StepCategory.EXECUTION_ERROR)

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
            checkpoint.add_step(f"Color Check {i}", False, i, "No sheet data available",
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i in range(1, 5):
                checkpoint.add_step(f"Color Check {i}", False, i, "No sheets found",
                                  category=StepCategory.EXECUTION_ERROR)
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
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)
        else:
            checkpoint.add_step("Macros Same Color", False, step_num,
                              f"Macro columns have inconsistent or no background colors ({len(macro_colors)} found)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)

        # Step 2: Mineral columns share same background color
        step_num += 1
        step_start = time.time()
        mineral_colors = [get_bg_color(data_row_idx, idx) for idx in mineral_indices if idx >= 0]
        mineral_same = len(mineral_colors) > 0 and all(colors_are_similar(mineral_colors[0], c) for c in mineral_colors)

        if mineral_same:
            checkpoint.add_step("Minerals Same Color", True, step_num,
                              f"All {len(mineral_colors)} mineral columns share the same background color",
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)
        else:
            checkpoint.add_step("Minerals Same Color", False, step_num,
                              f"Mineral columns have inconsistent or no background colors ({len(mineral_colors)} found)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)

        # Step 3: Vitamin columns share same background color
        step_num += 1
        step_start = time.time()
        vitamin_colors = [get_bg_color(data_row_idx, idx) for idx in vitamin_indices if idx >= 0]
        vitamin_same = len(vitamin_colors) > 0 and all(colors_are_similar(vitamin_colors[0], c) for c in vitamin_colors)

        if vitamin_same:
            checkpoint.add_step("Vitamins Same Color", True, step_num,
                              f"All {len(vitamin_colors)} vitamin columns share the same background color",
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)
        else:
            checkpoint.add_step("Vitamins Same Color", False, step_num,
                              f"Vitamin columns have inconsistent or no background colors ({len(vitamin_colors)} found)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)

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
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)
        else:
            checkpoint.add_step("Distinct Group Colors", False, step_num,
                              f"Nutrient groups do not have distinct colors ({len(group_colors)} groups found)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)

    except Exception as e:
        for i in range(1, 5):
            checkpoint.add_step(f"Color Check {i}", False, i, f"Error: {str(e)[:50]}",
                              category=StepCategory.EXECUTION_ERROR)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4: Ingredients Present
    Validates that correct ingredients are present.

    TODO: Update point total and outcome evaluation once ingredients are finalized.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    global model, matched_ingredients
    checkpoint_start = time.time()

    # Calculate total: expected ingredients + 1 for excluded ingredient check (if any)
    total_pts = len(EXPECTED_INGREDIENTS) + (1 if EXCLUDED_INGREDIENT else 0)
    if total_pts == 0:
        total_pts = 1  # Minimum 1 point placeholder
    checkpoint = Checkpoint(total=total_pts, result=0, name="Ingredients Present")

    if df is None or df.empty:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS + ([EXCLUDED_INGREDIENT] if EXCLUDED_INGREDIENT else []), 1):
            checkpoint.add_step(f"{ingredient}", False, i, "No data found",
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    ingredient_col = matched_columns.get("Ingredients") or keywords_match_robust(columns, COLUMN_KEYWORDS["Ingredients"])

    if not ingredient_col:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS + ([EXCLUDED_INGREDIENT] if EXCLUDED_INGREDIENT else []), 1):
            checkpoint.add_step(f"{ingredient}", False, i, "No ingredient column found",
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Phase 1: Fast keyword matching for all ingredients
    keyword_matches = {}  # ingredient -> (row_idx, cell_value)
    llm_needed = []  # ingredients that need LLM fallback

    for ingredient in EXPECTED_INGREDIENTS:
        keywords = INGREDIENT_KEYWORDS.get(ingredient, [ingredient.lower()])
        matched_row_idx = None

        for idx, row in df.iterrows():
            cell_value = str(row[ingredient_col]).lower().strip()
            if keywords_exact_match(cell_value, keywords):
                matched_row_idx = idx
                keyword_matches[ingredient] = (idx, cell_value)
                print(f"  [DEBUG] Ingredient '{ingredient}' matched via KEYWORD (row {idx})")
                break

        if matched_row_idx is None:
            llm_needed.append(ingredient)

    # Phase 2: Parallel LLM fallback for unmatched ingredients
    llm_matches = {}  # ingredient -> row_idx
    if llm_needed:
        if model is None:
            model = load_model(model_id)

        # Get all candidate cells from DataFrame (cells that haven't been matched yet)
        matched_rows = set(m[0] for m in keyword_matches.values())
        candidate_cells = []  # (row_idx, cell_value)
        for idx, row in df.iterrows():
            if idx not in matched_rows:
                cell_value = str(row[ingredient_col]).lower().strip()
                if cell_value and len(cell_value) > 1:
                    candidate_cells.append((idx, cell_value))

        # Build VLM tasks: for each unmatched ingredient, check against each candidate cell
        vlm_tasks = []
        for ingredient in llm_needed:
            for row_idx, cell_value in candidate_cells:
                prompt_text = f"Does '{cell_value}' refer to the same ingredient as '{ingredient}'? Answer only Yes or No."
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that answers Yes or No."}]},
                    {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
                ]
                vlm_tasks.append({
                    'id': f"{ingredient}|{row_idx}",
                    'messages': messages,
                    'ingredient': ingredient,
                    'row_idx': row_idx
                })

        if vlm_tasks:
            print(f"  [PARALLEL] Running {len(vlm_tasks)} LLM ingredient matches...")
            llm_start = time.time()
            vlm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=10)
            print(f"  [PARALLEL] LLM ingredient matching completed in {time.time() - llm_start:.2f}s")

            # Process results - find first match for each ingredient
            for task in vlm_tasks:
                task_id = task['id']
                ingredient = task['ingredient']
                row_idx = task['row_idx']

                # Skip if this ingredient already matched
                if ingredient in llm_matches:
                    continue

                if vlm_results.get(task_id, False):
                    llm_matches[ingredient] = row_idx
                    print(f"  [DEBUG] Ingredient '{ingredient}' matched via LLM (row {row_idx})")

    # Phase 3: Process results and add checkpoint steps
    step_num = 0
    for ingredient in EXPECTED_INGREDIENTS:
        step_num += 1
        step_start = time.time()

        if ingredient in keyword_matches:
            row_idx, _ = keyword_matches[ingredient]
            matched_ingredients[ingredient] = row_idx
            checkpoint.add_step(f"{ingredient} Present", True, step_num,
                              f"Found '{ingredient}' in spreadsheet",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC)
        elif ingredient in llm_matches:
            row_idx = llm_matches[ingredient]
            matched_ingredients[ingredient] = row_idx
            checkpoint.add_step(f"{ingredient} Present", True, step_num,
                              f"Found '{ingredient}' in spreadsheet (LLM match)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.LLM_VLM_JUDGEMENT)
        else:
            print(f"  [DEBUG] Ingredient '{ingredient}' NOT FOUND")
            checkpoint.add_step(f"{ingredient} Present", False, step_num,
                              f"'{ingredient}' not found in spreadsheet",
                              execution_time=time.time() - step_start,
                              category=StepCategory.LLM_VLM_JUDGEMENT)

    # Check that excluded ingredient is NOT present (if any)
    if EXCLUDED_INGREDIENT:
        step_num += 1
        step_start = time.time()

        # Use keywords_exact_match for excluded ingredient check
        excluded_found = False
        for _, row in df.iterrows():
            cell_value = str(row[ingredient_col]).lower().strip()
            if keywords_exact_match(cell_value, EXCLUDED_KEYWORDS):
                excluded_found = True
                break

        if not excluded_found:
            print(f"  [DEBUG] Excluded ingredient '{EXCLUDED_INGREDIENT}' correctly NOT found")
            checkpoint.add_step(f"{EXCLUDED_INGREDIENT} Not Present", True, step_num,
                              f"'{EXCLUDED_INGREDIENT}' correctly excluded (no specified amount)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC)
        else:
            print(f"  [DEBUG] Excluded ingredient '{EXCLUDED_INGREDIENT}' FOUND (should not be present)")
            checkpoint.add_step(f"{EXCLUDED_INGREDIENT} Not Present", False, step_num,
                              f"'{EXCLUDED_INGREDIENT}' should not be present (no specified amount in recipe)",
                              execution_time=time.time() - step_start,
                              category=StepCategory.DETERMINISTIC)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Checkpoint 5: USDA Links Validation
    Validates that each ingredient has a valid USDA link.

    TODO: Update point total once ingredients are finalized.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global model
    checkpoint_start = time.time()

    total_pts = len(EXPECTED_INGREDIENTS) if EXPECTED_INGREDIENTS else 1
    checkpoint = Checkpoint(total=total_pts, result=0, name="USDA Links Validation")

    if df is None or df.empty:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS, 1):
            checkpoint.add_step(f"{ingredient} Link", False, i, "No data found",
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    ingredient_col = matched_columns.get("Ingredients") or keywords_match_robust(columns, COLUMN_KEYWORDS["Ingredients"])
    link_col = matched_columns.get("Link") or keywords_match_robust(columns, COLUMN_KEYWORDS["Link"])

    if not ingredient_col or not link_col:
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS, 1):
            checkpoint.add_step(f"{ingredient} Link", False, i, "Missing ingredient or link column",
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Phase 1: Collect all valid links for parallel fetching
    ingredient_links = {}  # ingredient -> link
    invalid_ingredients = {}  # ingredient -> error message

    for ingredient in EXPECTED_INGREDIENTS:
        if ingredient not in matched_ingredients:
            invalid_ingredients[ingredient] = f"Ingredient '{ingredient}' not found in spreadsheet"
            continue

        row_idx = matched_ingredients[ingredient]
        row_match = df.loc[row_idx]
        link = str(row_match[link_col]).strip()

        if not link or not is_url_from_domain(link, 'fdc.nal.usda.gov'):
            invalid_ingredients[ingredient] = f"Invalid or non-USDA link: {link[:50] if link else 'empty'}..."
        else:
            ingredient_links[ingredient] = link
            print(f"  [DEBUG] Ingredient '{ingredient}' link: {link[:60]}...")

    # Phase 2: Parallel fetch all USDA page titles
    fetch_tasks = [
        {'id': ingredient, 'func': fetch_usda_page_title, 'args': (link,)}
        for ingredient, link in ingredient_links.items()
    ]

    print(f"  [PARALLEL] Fetching {len(fetch_tasks)} USDA page titles...")
    fetch_start = time.time()
    fetch_results = parallel_download(fetch_tasks, max_workers=5, use_rate_limit=False)
    print(f"  [PARALLEL] Fetched in {time.time() - fetch_start:.2f}s")

    # Phase 3: Keyword matching first (fast), collect LLM fallback tasks
    keyword_matches = {}  # ingredient -> True/False
    llm_needed = {}  # ingredient -> page_title (needs LLM validation)

    for ingredient, page_title in fetch_results.items():
        if page_title:
            # Try keyword matching first (from INGREDIENT_KEYWORDS)
            keywords = INGREDIENT_KEYWORDS.get(ingredient, [ingredient.lower()])
            if keywords_exact_match(page_title, keywords):
                keyword_matches[ingredient] = True
                print(f"  [DEBUG] USDA page '{page_title}' matched '{ingredient}' via KEYWORD")
            else:
                llm_needed[ingredient] = page_title
        else:
            # Could not fetch - mark as valid URL format
            keyword_matches[ingredient] = True  # Pass with warning

    # Phase 4: Parallel LLM validation for unmatched
    llm_results = {}
    if llm_needed:
        if model is None:
            model = load_model(model_id)

        vlm_tasks = []
        for ingredient, page_title in llm_needed.items():
            prompt_text = f"Is '{page_title}' a valid USDA database entry for the ingredient '{ingredient}'? For example, 'Nuts, almonds, raw' is valid for 'Almonds', and 'Spices, garlic powder' is valid for 'Garlic Powder'. Answer only Yes or No."
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that determines if USDA food database entries match recipe ingredients. Be lenient - USDA entries often have prefixes like 'Nuts,', 'Spices,', 'Beverages,' and suffixes like ', raw', ', dried', etc. Answer Yes or No."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
            ]
            vlm_tasks.append({'id': ingredient, 'messages': messages})

        print(f"  [PARALLEL] Running {len(vlm_tasks)} LLM validations...")
        llm_start = time.time()
        llm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=5)
        print(f"  [PARALLEL] LLM validation completed in {time.time() - llm_start:.2f}s")

    # Phase 5: Add checkpoint steps for all ingredients
    for step_num, ingredient in enumerate(EXPECTED_INGREDIENTS, 1):
        step_start = time.time()

        if ingredient in invalid_ingredients:
            checkpoint.add_step(f"{ingredient} Link Valid", False, step_num,
                              invalid_ingredients[ingredient],
                              execution_time=time.time() - step_start,
                              category=StepCategory.DEPENDENCY_NOT_EVALUATED if ingredient not in matched_ingredients else StepCategory.DETERMINISTIC)
        elif ingredient in keyword_matches:
            page_title = fetch_results.get(ingredient)
            if page_title:
                checkpoint.add_step(f"{ingredient} Link Valid", True, step_num,
                                  f"USDA link verified for '{ingredient}'",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step(f"{ingredient} Link Valid", True, step_num,
                                  f"Valid USDA URL format (could not verify page content)",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.VACUOUS_PASS)
        elif ingredient in llm_results:
            if llm_results[ingredient]:
                print(f"  [DEBUG] USDA page matched '{ingredient}' via LLM")
                checkpoint.add_step(f"{ingredient} Link Valid", True, step_num,
                                  f"USDA link verified for '{ingredient}'",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.LLM_VLM_JUDGEMENT)
            else:
                page_title = llm_needed.get(ingredient, "unknown")
                checkpoint.add_step(f"{ingredient} Link Valid", False, step_num,
                                  f"USDA page '{page_title[:30]}...' does not match '{ingredient}'",
                                  execution_time=time.time() - step_start,
                                  category=StepCategory.LLM_VLM_JUDGEMENT)
        else:
            checkpoint.add_step(f"{ingredient} Link Valid", False, step_num,
                              f"Could not validate link for '{ingredient}'",
                              execution_time=time.time() - step_start,
                              category=StepCategory.EXECUTION_ERROR)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """
    Checkpoint 6: Nutrient Values Accuracy
    Validates nutrient values match gold labels within tolerance.
    Each nutrient column is worth N points (one per ingredient).

    TODO: Update point total once ingredients are finalized.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    checkpoint_start = time.time()

    num_ingredients = len(EXPECTED_INGREDIENTS) if EXPECTED_INGREDIENTS else 1
    total_pts = num_ingredients * len(ALL_NUTRIENTS)
    checkpoint = Checkpoint(total=total_pts, result=0, name="Nutrient Values Accuracy")

    if df is None or df.empty or gold_data is None:
        for i, nutrient in enumerate(ALL_NUTRIENTS, 1):
            checkpoint.add_step(f"{nutrient} Values", False, i,
                              "No data or gold labels available", max_score=num_ingredients,
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    columns = list(df.columns)
    ingredient_col = matched_columns.get("Ingredients") or keywords_match_robust(columns, COLUMN_KEYWORDS["Ingredients"])

    if not ingredient_col:
        for i, nutrient in enumerate(ALL_NUTRIENTS, 1):
            checkpoint.add_step(f"{nutrient} Values", False, i,
                              "No ingredient column found", max_score=num_ingredients,
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    step_num = 0
    tolerance_percent = VALUE_TOLERANCE * 100  # Convert to percentage

    # USDA fallback cache: ingredient -> True/False (whether fallback validated)
    link_col = matched_columns.get("Link")
    usda_fallback_valid = {}

    def _check_usda_fallback(ingredient: str) -> bool:
        """Validate ingredient via USDA API fallback (ratio consistency + gold cross-validation)."""
        global model
        if not link_col or ingredient not in matched_ingredients:
            return False
        try:
            row_data = df.loc[matched_ingredients[ingredient]]
            agent_link = str(row_data[link_col]).strip()
            if not is_url_from_domain(agent_link, 'fdc.nal.usda.gov'):
                return False
            agent_food_id = extract_food_id_from_url(agent_link)
            gold_row = gold_by_ingredient.get(ingredient.lower())
            if gold_row is not None and 'Link' in gold_row.index:
                gold_food_id = extract_food_id_from_url(str(gold_row['Link']))
                if agent_food_id and gold_food_id and agent_food_id == gold_food_id:
                    print(f"    [USDA FALLBACK] Same food code ({agent_food_id}) as gold — no fallback needed")
                    return False
            page_title = fetch_usda_page_title(agent_link)
            if not page_title:
                print(f"    [USDA FALLBACK] Could not fetch page title for {agent_link[:60]}")
                return False
            if model is None:
                model = load_model(model_id)
            keywords = INGREDIENT_KEYWORDS.get(ingredient, [ingredient.lower()])
            if not ingredient_matches_usda_page(ingredient, page_title, keywords=keywords, model=model):
                print(f"    [USDA FALLBACK] LLM rejected '{page_title}' for '{ingredient}'")
                return False
            print(f"    [USDA FALLBACK] LLM confirmed '{page_title}' matches '{ingredient}'")
            api_per_100g = fetch_usda_nutrients(agent_link)
            if not api_per_100g:
                print(f"    [USDA FALLBACK] Could not fetch nutrients from {agent_link[:60]}")
                return False
            sheet_vals = {}
            gold_vals = {}
            for nut in ALL_NUTRIENTS:
                sc = matched_columns.get(nut)
                if not sc:
                    continue
                parsed = parse_nutrient_value(str(row_data[sc]))
                if parsed is not None:
                    sheet_vals[nut] = parsed
                g_row = gold_by_ingredient.get(ingredient.lower())
                if g_row is not None:
                    for gc in gold_data.columns:
                        if nut.lower() in gc.lower():
                            try:
                                gold_vals[nut] = float(g_row[gc])
                            except (ValueError, TypeError):
                                pass
                            break
            return validate_usda_fallback(
                sheet_values=sheet_vals, gold_values=gold_vals,
                api_per_100g=api_per_100g, tolerance=VALUE_TOLERANCE,
            )
        except Exception as e:
            print(f"    [USDA FALLBACK] Error for {ingredient}: {e}")
            return False

    for nutrient in ALL_NUTRIENTS:
        step_num += 1
        step_start = time.time()

        sheet_col = matched_columns.get(nutrient)
        if not sheet_col:
            checkpoint.add_step(f"{nutrient} Values", False, step_num,
                              f"No column found for {nutrient}", max_score=num_ingredients,
                              category=StepCategory.EXECUTION_ERROR)
            continue

        gold_col = None
        for gc in gold_data.columns:
            if nutrient.lower() in gc.lower():
                gold_col = gc
                break

        if not gold_col:
            checkpoint.add_step(f"{nutrient} Values", False, step_num,
                              f"No gold data column for {nutrient}", max_score=num_ingredients,
                              category=StepCategory.EXECUTION_ERROR)
            continue

        matches = 0
        mismatches = []

        for ingredient in EXPECTED_INGREDIENTS:
            sheet_value = None
            if ingredient in matched_ingredients:
                row_idx = matched_ingredients[ingredient]
                row = df.loc[row_idx]
                try:
                    sheet_value = parse_nutrient_value(str(row[sheet_col]))
                    print(f"    [DEBUG] {ingredient} -> {nutrient}: sheet_value={sheet_value}")
                except (ValueError, TypeError):
                    sheet_value = None
                    print(f"    [DEBUG] {ingredient} -> {nutrient}: sheet_value=None (parse error)")
            else:
                print(f"    [DEBUG] {ingredient} -> {nutrient}: not matched in checkpoint 4")

            gold_value = None
            gold_row = gold_by_ingredient.get(ingredient.lower())
            if gold_row is not None:
                try:
                    gold_value = float(gold_row[gold_col])
                    print(f"    [DEBUG] {ingredient} -> {nutrient}: gold_value={gold_value}")
                except (ValueError, TypeError):
                    gold_value = None
                    print(f"    [DEBUG] {ingredient} -> {nutrient}: gold_value=None (parse error)")

            gold_matched = False
            if sheet_value is not None and gold_value is not None:
                if gold_value == 0:
                    gold_matched = (sheet_value == 0)
                else:
                    gold_matched, _ = numerical_match_with_error(gold_value, sheet_value, error_percent=tolerance_percent)
            elif sheet_value is None and gold_value is None:
                gold_matched = True

            if gold_matched:
                matches += 1
                continue

            if ingredient not in usda_fallback_valid:
                print(f"    [USDA FALLBACK] Evaluating fallback for '{ingredient}'...")
                usda_fallback_valid[ingredient] = _check_usda_fallback(ingredient)

            if usda_fallback_valid[ingredient]:
                print(f"    [USDA FALLBACK] {ingredient} -> {nutrient}: PASSED (fallback validated)")
                matches += 1
            else:
                if sheet_value is not None and gold_value is not None:
                    mismatches.append(f"{ingredient}: {sheet_value:.2f} vs {gold_value:.2f}")
                elif sheet_value is None:
                    mismatches.append(f"{ingredient}: missing in sheet")
                else:
                    mismatches.append(f"{ingredient}: missing in gold")

        success = matches == len(EXPECTED_INGREDIENTS)

        if success:
            print(f"  [DEBUG] {nutrient}: {matches}/{len(EXPECTED_INGREDIENTS)} values match")
            checkpoint.add_step(f"{nutrient} Values", True, step_num,
                              f"All {matches}/{len(EXPECTED_INGREDIENTS)} values match within {tolerance_percent}% tolerance",
                              score=matches, max_score=num_ingredients,
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)
        else:
            print(f"  [DEBUG] {nutrient}: {matches}/{len(EXPECTED_INGREDIENTS)} values match. Mismatches: {mismatches}")
            detail = f"{matches}/{len(EXPECTED_INGREDIENTS)} match"
            if mismatches:
                detail += f". Mismatches: {'; '.join(mismatches[:2])}"
            checkpoint.add_step(f"{nutrient} Values", False, step_num,
                              detail, score=matches, max_score=num_ingredients,
                              execution_time=time.time() - step_start,
                              category=StepCategory.FUZZY_MATCH)

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
                          execution_time=time.time() - checkpoint_start,
                          category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            checkpoint.add_step("Bold Formatting", False, 1, "No sheets found",
                              category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        grid_data = sheets[0].get('data', [{}])[0]
        rows = grid_data.get('rowData', [])

        columns = list(df.columns)

        # Use global header_row_idx detected in setup()
        if header_row_idx is None:
            checkpoint.add_step("Bold Formatting", False, 1, "Header row not detected",
                              category=StepCategory.EXECUTION_ERROR)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

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

                value = parse_nutrient_value(str(cell_value))
                if value is None:
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

        # Calculate success - ALL cells must be correctly formatted
        if total_checks > 0:
            success = correct_bold == total_checks

            if success:
                checkpoint.add_step("Bold Formatting", True, 1,
                                  f"All {total_checks} cells correctly formatted",
                                  execution_time=time.time() - checkpoint_start,
                                  category=StepCategory.DETERMINISTIC)
            else:
                checkpoint.add_step("Bold Formatting", False, 1,
                                  f"Only {correct_bold}/{total_checks} cells correctly formatted",
                                  execution_time=time.time() - checkpoint_start,
                                  category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Bold Formatting", False, 1,
                              "No nutrient cells found to check",
                              execution_time=time.time() - checkpoint_start,
                              category=StepCategory.EXECUTION_ERROR)

    except Exception as e:
        checkpoint.add_step("Bold Formatting", False, 1, f"Error: {str(e)[:50]}",
                          category=StepCategory.EXECUTION_ERROR)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_8(browsing_history: Optional[List[str]] = None):
    """
    Checkpoint 8: Website Visit Validation
    Validates that required websites were visited.

    TODO: Update point total once ingredients are finalized.
    """
    print("----------------- CHECKPOINT 8 ----------------")
    checkpoint_start = time.time()

    total_pts = len(EXPECTED_INGREDIENTS) + 1 if EXPECTED_INGREDIENTS else 2
    checkpoint = Checkpoint(total=total_pts, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Recipe URL Visited", False, 1,
                          "No browsing history provided",
                          category=StepCategory.EXECUTION_ERROR)
        for i, ingredient in enumerate(EXPECTED_INGREDIENTS, 2):
            checkpoint.add_step(f"USDA URL for {ingredient}", False, i,
                              "No browsing history provided",
                              category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    browsing_lower = [url.lower() for url in browsing_history]

    step_num = 0

    # Step 1: Recipe URL visited
    step_num += 1
    step_start = time.time()
    recipe_visited = any(RECIPE_DOMAIN in url for url in browsing_lower)

    if recipe_visited:
        checkpoint.add_step("Recipe URL Visited", True, step_num,
                          f"Found visit to {RECIPE_DOMAIN}",
                          execution_time=time.time() - step_start,
                          category=StepCategory.WEB_VISIT)
    else:
        checkpoint.add_step("Recipe URL Visited", False, step_num,
                          f"No visit to {RECIPE_DOMAIN} found",
                          execution_time=time.time() - step_start,
                          category=StepCategory.WEB_VISIT)

    # Steps 2+: USDA URL visited for each ingredient
    usda_visited = any('fdc.nal.usda.gov' in url for url in browsing_lower)

    for ingredient in EXPECTED_INGREDIENTS:
        step_num += 1
        step_start = time.time()

        if usda_visited:
            checkpoint.add_step(f"USDA URL for {ingredient}", True, step_num,
                              f"USDA database was visited",
                              execution_time=time.time() - step_start,
                              category=StepCategory.WEB_VISIT)
        else:
            checkpoint.add_step(f"USDA URL for {ingredient}", False, step_num,
                              f"No USDA database visit found",
                              execution_time=time.time() - step_start,
                              category=StepCategory.WEB_VISIT)

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
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0,
                                 category=StepCategory.EXECUTION_ERROR)
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
            status = "PASS" if step["success"] else "FAIL"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
