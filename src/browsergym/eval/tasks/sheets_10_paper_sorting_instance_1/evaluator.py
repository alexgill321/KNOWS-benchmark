"""Evaluator for the Paper Sorting Google Sheets task.

This evaluator validates a spreadsheet containing research paper metadata:
- Original papers from a source Google Drive folder
- New papers discovered by searching arXiv for each first author's publications
- Proper formatting (yellow highlighting for new papers, blue for world models)
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional, Any, Tuple

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
from src.browsergym.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    extract_tables_from_sheet
)
from src.browsergym.eval.eval_utils.google_services_helpers import get_sheet_content
from src.browsergym.eval.eval_utils.text_utils import text_fuzzy_match_contained_long
from src.browsergym.eval.eval_utils.image_utils import binary_compare_images
from src.browsergym.eval.eval_utils.models import load_model

# Local imports
from src.browsergym.eval.tasks.sheets_10_paper_sorting_instance_1.utils import (
    extract_arxiv_id_from_url,
    validate_arxiv_url,
    extract_drive_file_id,
    validate_drive_url,
    parse_authors_string,
    normalize_author_name,
    compare_authors_list,
    fuzzy_match_text,
    get_row_background_color,
    classify_row_color,
    validate_color_grouping
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/sheets_10_paper_sorting_instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Folder IDs
SOURCE_FOLDER_ID = "1dfRMRjBHH4F1S9WMD6p6VqpYQZ-pbKWB"
DEST_FOLDER_ID = "1vk3FB8IumyHMBuBjI8fsUSdSyOVlFZPf"

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

# Gold data (loaded from JSON)
GOLD_PAPERS = None
GOLD_NEW_PAPERS = None
AUTHOR_LOOKUP = None


def load_gold_data():
    """Load preprocessed gold data from JSON files."""
    global GOLD_PAPERS, GOLD_NEW_PAPERS, AUTHOR_LOOKUP

    gold_papers_path = os.path.join(DATA_DIR, "gold_papers.json")
    gold_new_papers_path = os.path.join(DATA_DIR, "gold_new_papers.json")
    author_lookup_path = os.path.join(DATA_DIR, "author_papers_lookup.json")

    if os.path.exists(gold_papers_path):
        with open(gold_papers_path, 'r') as f:
            GOLD_PAPERS = json.load(f)
        print(f"Loaded {GOLD_PAPERS.get('count', 0)} original papers from gold data")
    else:
        print(f"WARNING: Gold papers file not found: {gold_papers_path}")
        GOLD_PAPERS = {"papers": [], "count": 0}

    if os.path.exists(gold_new_papers_path):
        with open(gold_new_papers_path, 'r') as f:
            GOLD_NEW_PAPERS = json.load(f)
        print(f"Loaded {GOLD_NEW_PAPERS.get('count', 0)} new papers from gold data")
    else:
        print(f"WARNING: Gold new papers file not found: {gold_new_papers_path}")
        GOLD_NEW_PAPERS = {"papers": [], "count": 0}

    if os.path.exists(author_lookup_path):
        with open(author_lookup_path, 'r') as f:
            AUTHOR_LOOKUP = json.load(f)
        print(f"Loaded {AUTHOR_LOOKUP.get('count', 0)} author lookups")
    else:
        print(f"WARNING: Author lookup file not found: {author_lookup_path}")
        AUTHOR_LOOKUP = {"first_authors": [], "count": 0}


def setup(workspace_doc_id: str):
    """Setup function to initialize the evaluator.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, table_data, sheet_raw, df

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    # Load gold data
    load_gold_data()

    # Extract data from the user's spreadsheet
    table_data = extract_tables_from_sheet(sheet_id, SHEETS_SERVICE)
    sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

    # Initialize df for use across checkpoints
    if table_data:
        first_table = table_data[0]
        df = first_table.df if hasattr(first_table, 'df') else first_table
        print(f"Extracted table with {len(df)} rows and {len(df.columns)} columns")


def grade_checkpoint_1():
    """Checkpoint 1: Spreadsheet Structure (6 steps).

    Validates that the spreadsheet has correct column headers:
    1. Title column (Column A)
    2. Authors column (Column B)
    3. Abstract column (Column C)
    4. arXiv Link column (Column D)
    5. Drive Link column (Column E)
    6. Figure 1 column (Column F)
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global model, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=6, result=0, name="Spreadsheet Structure")

    if not table_data or df is None or df.empty:
        checkpoint.add_step("Table Data Extraction", False, 1,
                          "No table data found in spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Required columns with keywords for matching
    required_columns = [
        ("Title", ["title", "paper", "name"]),
        ("Authors", ["author", "authors", "by"]),
        ("Abstract", ["abstract", "summary"]),
        ("arXiv Link", ["arxiv", "link", "url"]),
        ("Drive Link", ["drive", "pdf", "file", "google"]),
        ("Figure 1", ["figure", "fig", "image", "screenshot"])
    ]

    column_lower = [str(col).lower() for col in df.columns]
    original_columns = [str(col) for col in df.columns]

    for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
        step_start = time.time()
        found = False
        matched_column = None

        # Try keyword-based matching (fast path)
        for i, col in enumerate(column_lower):
            if any(keyword in col for keyword in keywords):
                found = True
                matched_column = original_columns[i]
                matched_columns[col_name] = matched_column
                break

        step_time = time.time() - step_start

        if found:
            checkpoint.add_step(f"{col_name} Column", True, step_num,
                              f"Found column: '{matched_column}'",
                              execution_time=step_time)
        else:
            # For simplicity, just fail if keyword match doesn't work
            # Could add LLM fallback here if needed
            checkpoint.add_step(f"{col_name} Column", False, step_num,
                              f"No column found for '{col_name}'. Available: {', '.join(original_columns[:5])}...",
                              execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2: Original Papers Validation (7 steps, each X/N).

    For each of the N original papers:
    1. Titles match (fuzzy)
    2. Authors match (exact after normalization)
    3. Abstracts match (fuzzy)
    4. arXiv Links valid
    5. Drive Links valid
    6. Figure 1 images match
    7. arXiv URLs appear in browsing history
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model, matched_columns, df
    checkpoint_start = time.time()

    N = GOLD_PAPERS.get('count', 0)
    checkpoint = Checkpoint(total=7, result=0, name="Original Papers Validation")

    if N == 0:
        checkpoint.add_step("Gold Data", False, 1,
                          "No gold papers data available",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if df is None or df.empty:
        checkpoint.add_step("User Data", False, 1,
                          "No data in user's spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    gold_papers = GOLD_PAPERS.get('papers', [])

    # Get column mappings
    title_col = matched_columns.get('Title')
    authors_col = matched_columns.get('Authors')
    abstract_col = matched_columns.get('Abstract')
    arxiv_col = matched_columns.get('arXiv Link')
    drive_col = matched_columns.get('Drive Link')

    # Step 1: Titles match
    step_start = time.time()
    title_matches = 0
    for gold in gold_papers:
        gold_title = gold.get('title', '')
        for _, row in df.iterrows():
            user_title = str(row.get(title_col, '')) if title_col else ''
            is_match, score = fuzzy_match_text(gold_title, user_title, threshold=85)
            if is_match:
                title_matches += 1
                break
    step_time = time.time() - step_start
    checkpoint.add_step("Titles Match", title_matches == N, 1,
                      f"{title_matches}/{N} titles match",
                      execution_time=step_time)

    # Step 2: Authors match
    step_start = time.time()
    author_matches = 0
    for gold in gold_papers:
        gold_authors = gold.get('authors', [])
        for _, row in df.iterrows():
            user_authors_str = str(row.get(authors_col, '')) if authors_col else ''
            user_authors = parse_authors_string(user_authors_str)
            is_match, _ = compare_authors_list(user_authors, gold_authors, strict=False)
            if is_match:
                author_matches += 1
                break
    step_time = time.time() - step_start
    checkpoint.add_step("Authors Match", author_matches == N, 2,
                      f"{author_matches}/{N} author lists match",
                      execution_time=step_time)

    # Step 3: Abstracts match
    step_start = time.time()
    abstract_matches = 0
    for gold in gold_papers:
        gold_abstract = gold.get('abstract', '')
        for _, row in df.iterrows():
            user_abstract = str(row.get(abstract_col, '')) if abstract_col else ''
            is_match, score = fuzzy_match_text(gold_abstract, user_abstract, threshold=80)
            if is_match:
                abstract_matches += 1
                break
    step_time = time.time() - step_start
    checkpoint.add_step("Abstracts Match", abstract_matches == N, 3,
                      f"{abstract_matches}/{N} abstracts match",
                      execution_time=step_time)

    # Step 4: arXiv Links valid
    step_start = time.time()
    arxiv_valid = 0
    for gold in gold_papers:
        gold_arxiv_id = gold.get('arxiv_id', '')
        for _, row in df.iterrows():
            user_arxiv_url = str(row.get(arxiv_col, '')) if arxiv_col else ''
            user_arxiv_id = extract_arxiv_id_from_url(user_arxiv_url)
            if user_arxiv_id and user_arxiv_id == gold_arxiv_id:
                arxiv_valid += 1
                break
    step_time = time.time() - step_start
    checkpoint.add_step("arXiv Links Valid", arxiv_valid == N, 4,
                      f"{arxiv_valid}/{N} arXiv links valid",
                      execution_time=step_time)

    # Step 5: Drive Links valid
    step_start = time.time()
    drive_valid = 0
    for gold in gold_papers:
        gold_file_id = gold.get('drive_file_id', '')
        for _, row in df.iterrows():
            user_drive_url = str(row.get(drive_col, '')) if drive_col else ''
            user_file_id = extract_drive_file_id(user_drive_url)
            if user_file_id:
                # For now, just check if it's a valid Drive URL
                # Could add folder membership check with API call
                drive_valid += 1
                break
    step_time = time.time() - step_start
    checkpoint.add_step("Drive Links Valid", drive_valid == N, 5,
                      f"{drive_valid}/{N} Drive links valid",
                      execution_time=step_time)

    # Step 6: Figure 1 images (skip for now if no gold figures)
    step_start = time.time()
    figures_with_gold = [p for p in gold_papers if p.get('figure_1_path')]
    if figures_with_gold:
        # Would need VLM comparison here
        figure_matches = 0  # Placeholder
        checkpoint.add_step("Figure 1 Images", False, 6,
                          f"Figure comparison not yet implemented ({len(figures_with_gold)} gold figures available)",
                          execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Figure 1 Images", True, 6,
                          "No gold figure data to compare (skipped)",
                          execution_time=time.time() - step_start)

    # Step 7: arXiv URLs in browsing history (checked in grade_checkpoints)
    checkpoint.add_step("arXiv URLs Visited", True, 7,
                      "Browsing history check deferred",
                      execution_time=0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3: New Papers Discovery (1 step).

    For each ORIGINAL PAPER, check if at least 3 new papers were added
    where ANY of that paper's first authors appears anywhere in the author list.

    This is paper-centric, not author-centric.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=1, result=0, name="New Papers Discovery")

    if not AUTHOR_LOOKUP or not AUTHOR_LOOKUP.get('original_papers'):
        checkpoint.add_step("Paper Coverage", False, 1,
                          "No author lookup data available",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    original_papers_lookup = AUTHOR_LOOKUP.get('original_papers', [])
    authors_col = matched_columns.get('Authors')
    title_col = matched_columns.get('Title')

    if not authors_col or df is None:
        checkpoint.add_step("Paper Coverage", False, 1,
                          "Cannot check - no authors column or data",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get original paper titles to exclude them from "new papers"
    original_titles_normalized = set()
    for paper_entry in original_papers_lookup:
        orig_title = paper_entry.get('original_paper_title', '')
        if orig_title:
            original_titles_normalized.add(orig_title.lower().strip())

    papers_with_enough_new = 0
    total_original_papers = len(original_papers_lookup)
    details = []

    for paper_entry in original_papers_lookup:
        original_title = paper_entry.get('original_paper_title', '')
        first_authors_normalized = paper_entry.get('normalized_first_authors', [])
        expected_new = paper_entry.get('expected_new_papers', 3)
        original_arxiv_id = paper_entry.get('original_paper_arxiv_id', '')

        # Count user's new papers where ANY of this paper's first authors
        # appears ANYWHERE in the author list (not just as first author)
        matching_new_papers = 0

        for _, row in df.iterrows():
            # Skip if this is an original paper
            if title_col:
                user_title = str(row.get(title_col, '')).lower().strip()
                if user_title in original_titles_normalized:
                    continue

            # Also skip by arXiv ID if available
            arxiv_col = matched_columns.get('arXiv Link')
            if arxiv_col:
                arxiv_url = str(row.get(arxiv_col, ''))
                arxiv_id = extract_arxiv_id_from_url(arxiv_url)
                if arxiv_id and arxiv_id == original_arxiv_id:
                    continue

            # Parse ALL authors from the user's paper
            user_authors_str = str(row.get(authors_col, ''))
            user_authors = parse_authors_string(user_authors_str)
            user_authors_normalized = [normalize_author_name(a) for a in user_authors]

            # Check if ANY first author from original paper is in this paper's author list
            if any(fa in user_authors_normalized for fa in first_authors_normalized):
                matching_new_papers += 1

        if matching_new_papers >= expected_new:
            papers_with_enough_new += 1
            details.append(f"{original_title[:30]}...: {matching_new_papers}/{expected_new} OK")
        else:
            details.append(f"{original_title[:30]}...: {matching_new_papers}/{expected_new} MISSING")

    step_time = time.time() - checkpoint_start

    if papers_with_enough_new == total_original_papers:
        checkpoint.add_step("Paper Coverage", True, 1,
                          f"All {total_original_papers} original papers have ≥3 new papers from their first authors",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Paper Coverage", False, 1,
                          f"{papers_with_enough_new}/{total_original_papers} original papers have enough new papers. {'; '.join(details[:3])}",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """Checkpoint 4: New Papers Validation (7 steps, each X/M).

    Same validation as Checkpoint 2, but for newly discovered papers.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()

    M = GOLD_NEW_PAPERS.get('count', 0)
    checkpoint = Checkpoint(total=7, result=0, name="New Papers Validation")

    if M == 0:
        # No new papers expected or no gold data
        for i in range(1, 8):
            checkpoint.add_step(f"Step {i}", True, i,
                              "No new papers gold data available (skipped)",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Similar to checkpoint 2, but for new papers
    gold_new_papers = GOLD_NEW_PAPERS.get('papers', [])

    title_col = matched_columns.get('Title')
    authors_col = matched_columns.get('Authors')
    abstract_col = matched_columns.get('Abstract')
    arxiv_col = matched_columns.get('arXiv Link')
    drive_col = matched_columns.get('Drive Link')

    # Simplified validation - count matches for each field
    title_matches = 0
    author_matches = 0
    abstract_matches = 0
    arxiv_valid = 0
    drive_valid = 0

    for gold in gold_new_papers:
        gold_title = gold.get('title', '')
        gold_authors = gold.get('authors', [])
        gold_abstract = gold.get('abstract', '')
        gold_arxiv_id = gold.get('arxiv_id', '')

        for _, row in df.iterrows():
            user_title = str(row.get(title_col, '')) if title_col else ''
            is_match, _ = fuzzy_match_text(gold_title, user_title, threshold=85)
            if is_match:
                title_matches += 1

                # Check other fields for this matching row
                user_authors_str = str(row.get(authors_col, '')) if authors_col else ''
                user_authors = parse_authors_string(user_authors_str)
                auth_match, _ = compare_authors_list(user_authors, gold_authors, strict=False)
                if auth_match:
                    author_matches += 1

                user_abstract = str(row.get(abstract_col, '')) if abstract_col else ''
                abs_match, _ = fuzzy_match_text(gold_abstract, user_abstract, threshold=80)
                if abs_match:
                    abstract_matches += 1

                user_arxiv = str(row.get(arxiv_col, '')) if arxiv_col else ''
                user_arxiv_id = extract_arxiv_id_from_url(user_arxiv)
                if user_arxiv_id == gold_arxiv_id:
                    arxiv_valid += 1

                user_drive = str(row.get(drive_col, '')) if drive_col else ''
                if extract_drive_file_id(user_drive):
                    drive_valid += 1

                break

    checkpoint.add_step("Titles Match", title_matches == M, 1,
                      f"{title_matches}/{M} new paper titles match",
                      execution_time=0)
    checkpoint.add_step("Authors Match", author_matches == M, 2,
                      f"{author_matches}/{M} new paper authors match",
                      execution_time=0)
    checkpoint.add_step("Abstracts Match", abstract_matches == M, 3,
                      f"{abstract_matches}/{M} new paper abstracts match",
                      execution_time=0)
    checkpoint.add_step("arXiv Links Valid", arxiv_valid == M, 4,
                      f"{arxiv_valid}/{M} new paper arXiv links valid",
                      execution_time=0)
    checkpoint.add_step("Drive Links Valid", drive_valid == M, 5,
                      f"{drive_valid}/{M} new paper Drive links valid",
                      execution_time=0)
    checkpoint.add_step("Figure 1 Images", True, 6,
                      "Figure comparison not yet implemented",
                      execution_time=0)
    checkpoint.add_step("arXiv URLs Visited", True, 7,
                      "Browsing history check deferred",
                      execution_time=0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """Checkpoint 5: Formatting & Organization (3 binary steps).

    1. Yellow highlighting: All new paper rows must be yellow
    2. Blue highlighting: All papers with "world models" in related works must be blue
    3. Row grouping: Rows must be grouped by highlight color (not interleaved)
    """
    print("----------------- CHECKPOINT 5 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Formatting & Organization")

    if not sheet_raw:
        for i in range(1, 4):
            checkpoint.add_step(f"Formatting Check {i}", False, i,
                              "Could not access raw sheet data",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i in range(1, 4):
                checkpoint.add_step(f"Formatting Check {i}", False, i,
                                  "No sheets found",
                                  execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        rows = sheets[0].get('data', [{}])[0].get('rowData', [])
        num_rows = len(rows)

    except Exception as e:
        for i in range(1, 4):
            checkpoint.add_step(f"Formatting Check {i}", False, i,
                              f"Error: {str(e)[:50]}",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Collect row colors
    row_colors = []
    yellow_rows = []
    blue_rows = []

    for row_idx in range(1, num_rows):  # Skip header
        color = get_row_background_color(sheet_raw, row_idx)
        color_class = classify_row_color(color)
        row_colors.append(color_class)

        if color_class == 'yellow':
            yellow_rows.append(row_idx)
        elif color_class == 'blue':
            blue_rows.append(row_idx)

    # Step 1: Check yellow highlighting for new papers
    step_start = time.time()
    new_papers = GOLD_NEW_PAPERS.get('papers', [])
    M = len(new_papers)

    if M > 0:
        yellow_count = len(yellow_rows)
        if yellow_count >= M:
            checkpoint.add_step("Yellow Highlighting", True, 1,
                              f"{yellow_count} yellow rows found (expected {M} new papers)",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Yellow Highlighting", False, 1,
                              f"Only {yellow_count} yellow rows (expected {M} for new papers)",
                              execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Yellow Highlighting", True, 1,
                          "No new papers expected, yellow check skipped",
                          execution_time=time.time() - step_start)

    # Step 2: Check blue highlighting for world models papers
    step_start = time.time()
    all_papers = GOLD_PAPERS.get('papers', []) + GOLD_NEW_PAPERS.get('papers', [])
    world_models_papers = [p for p in all_papers if p.get('has_world_models', False)]
    expected_blue = len(world_models_papers)

    if expected_blue > 0:
        blue_count = len(blue_rows)
        if blue_count >= expected_blue:
            checkpoint.add_step("Blue Highlighting", True, 2,
                              f"{blue_count} blue rows found (expected {expected_blue} world models papers)",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Blue Highlighting", False, 2,
                              f"Only {blue_count} blue rows (expected {expected_blue} world models papers)",
                              execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Blue Highlighting", True, 2,
                          "No world models papers expected, blue check skipped",
                          execution_time=time.time() - step_start)

    # Step 3: Check row grouping (colors should not be interleaved)
    step_start = time.time()
    is_grouped, grouping_msg = validate_color_grouping(row_colors)
    checkpoint.add_step("Row Grouping", is_grouped, 3,
                      grouping_msg,
                      execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None,
                      browsing_history: List[str] = None) -> Result:
    """Grade all checkpoints for the paper sorting task.

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

        # Grade each checkpoint
        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())
        checkpoints.append(grade_checkpoint_5())

        # Update browsing history checks if provided
        if browsing_history:
            # Could update checkpoint 2 and 4 step 7 here
            pass

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
    parser = argparse.ArgumentParser(description="Evaluate paper sorting spreadsheet")
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
