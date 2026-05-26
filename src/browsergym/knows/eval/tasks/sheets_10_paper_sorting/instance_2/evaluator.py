"""Evaluator for the Paper Sorting Google Sheets task (Instance 2: Astrophysics).

This evaluator validates a spreadsheet containing research paper metadata:
- Original papers from a source Google Drive folder
- New papers discovered by searching arXiv for each first author's publications
- Proper formatting (yellow highlighting for papers mentioning "dark energy" in related works)
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
        return "/path/to/KNOWS-benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.knows.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    extract_drive_file_id
)
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    extract_tables_from_sheet,
    extract_sheet_data,
    get_sheet_content,
    parse_sheet_to_dataframe,
)
from src.browsergym.knows.eval.eval_utils.text_utils import (
    text_fuzzy_match_contained_long,
    fuzzy_match_text,
    split_delimited_text,
    normalize_name
)
from src.browsergym.knows.eval.eval_utils.web_utils import extract_id_from_url
from src.browsergym.knows.eval.eval_utils.image_utils import binary_compare_images
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.table_utils import (
    extract_image_url_from_cell,
    get_image_url_from_raw_sheet_cell,
    get_column_index_by_name,
    get_sheet_row_index_from_dataframe_row,
    get_row_background_color,
    classify_row_color,
    validate_color_grouping,
    match_columns
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import (
    parallel_download,
    fast_parallel_vlm_calls
)
import tempfile
import requests

# Local imports - only task-specific utilities
from src.browsergym.knows.eval.tasks.sheets_10_paper_sorting.utils import compare_authors_list

# ============================================================================
# INSTANCE-SPECIFIC CONFIGURATION
# ============================================================================
INSTANCE_NUM = 2
INSTANCE_NAME = "instance_2"
HIGHLIGHT_KEYWORD = "dark energy"
KEYWORD_FIELD = "has_keyword"  # Field name in gold JSON for keyword detection

# arXiv-specific constants
ARXIV_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# arXiv URL patterns for ID extraction
ARXIV_PATTERNS = [
    r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?',
    r'arxiv\.org/(?:abs|pdf)/([\w\-\.]+/\d+)(?:v\d+)?',
    r'ar5iv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?',
    r'(\d{4}\.\d{4,5})(?:v\d+)?\.pdf',
    r'(\d{4}\.\d{4,5})(?:v\d+)?$',
]


def extract_arxiv_id_from_url(url: str) -> str:
    """Extract arXiv ID from a URL using predefined patterns."""
    arxiv_id = extract_id_from_url(url, ARXIV_PATTERNS)
    if arxiv_id:
        import re
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    return arxiv_id


# Constants
TASK_DIR = os.path.join(BASE_PATH, f"src/browsergym/knows/eval/tasks/sheets_10_paper_sorting/{INSTANCE_NAME}/")
DATA_DIR = os.path.join(TASK_DIR, "data")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Folder IDs (DEST_FOLDER_ID is set by setup_run.py before each benchmark)
SOURCE_FOLDER_ID = "1Qm2gLrC3PhRqhlAI_WXBjYKqECdPOwBE"
DEST_FOLDER_ID = "1lwLNgxT9_S6SKW43Ag-lyed5qPrGLfcP"

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

# Required columns for header detection
REQUIRED_COLUMNS = [
    ("Title", ["title", "paper", "name"]),
    ("Authors", ["author", "authors", "by"]),
    ("Abstract", ["abstract", "summary"]),
    ("arXiv Link", ["arxiv", "link", "url"]),
    ("Drive Link", ["drive", "pdf", "file", "google"]),
    ("Figure 1", ["figure 1", "figure", "fig", "image", "screenshot"]),
    ("New Papers", ["new", "checkbox", "added", "new paper"]),
]

# Gold data (loaded from JSON)
GOLD_PAPERS = None
GOLD_NEW_PAPERS = None
AUTHOR_LOOKUP = None

# Browsing history (passed from grade_checkpoints)
BROWSING_HISTORY = None

# Track matched papers with their keyword status for yellow highlighting validation
MATCHED_PAPERS_KEYWORD_STATUS = []


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


def download_image_from_url(url: str) -> Optional[str]:
    """Download an image from URL and return the temp file path."""
    from src.browsergym.knows.eval.tasks.sheets_10_paper_sorting.utils import _arxiv_request_with_retry

    try:
        if 'arxiv.org' in url and 'export.arxiv.org' not in url:
            url = url.replace('://arxiv.org/', '://export.arxiv.org/')
        if 'arxiv.org' in url:
            response = _arxiv_request_with_retry(url, headers=ARXIV_HEADERS, timeout=30)
        else:
            response = requests.get(url, headers=ARXIV_HEADERS, timeout=30)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"  Failed to download image from {url[:60]}...: {e}")
        return None


def parallel_validate_figures(figure_tasks, vlm_model):
    """Download images and validate figures in parallel."""
    if not figure_tasks:
        return {}

    results = {}
    temp_files = []

    try:
        # Download user images sequentially with early abort on 429
        print(f"  Downloading {len(figure_tasks)} figure images...")
        downloaded = {}
        rate_limited = False
        for task in figure_tasks:
            if rate_limited:
                break
            user_url = task.get('user_url', '')
            if user_url and user_url.startswith('http'):
                image_url = extract_image_url_from_cell(user_url)
                if image_url:
                    path = download_image_from_url(image_url)
                    if path:
                        downloaded[task['id']] = path
                    elif 'arxiv.org' in image_url:
                        rate_limited = True
                        print(f"  Skipping remaining figure downloads (rate limited)")
        if rate_limited:
            print(f"  Downloaded {len(downloaded)}/{len(figure_tasks)} figures before rate limit")

        for path in downloaded.values():
            if path:
                temp_files.append(path)

        vlm_tasks = []
        for task in figure_tasks:
            paper_id = task['id']
            gold_path = task.get('gold_path', '')
            user_url = task.get('user_url', '')

            user_path = downloaded.get(paper_id)
            if not user_path:
                if user_url and not user_url.startswith('http'):
                    user_path = user_url

            if not user_path or not os.path.exists(user_path):
                results[paper_id] = False
                continue

            gold_full_path = os.path.join(TASK_DIR, gold_path)
            if not os.path.exists(gold_full_path):
                print(f"  Gold figure not found: {gold_full_path}")
                results[paper_id] = False
                continue

            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You compare images to verify if a reference figure appears within a screenshot."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "Reference Figure 1 from paper:"},
                    {"type": "image", "image": gold_full_path},
                    {"type": "text", "text": "Screenshot/image from spreadsheet:"},
                    {"type": "image", "image": user_path},
                    {"type": "text", "text": "Does the screenshot contain the reference Figure 1? The screenshot may include additional text, captions, or other content around the figure - that's acceptable. We need to verify that Figure 1 (or its key visual content) appears somewhere in the screenshot. Answer only YES or NO."}
                ]}
            ]
            vlm_tasks.append({'id': paper_id, 'messages': messages})

        if vlm_tasks:
            print(f"  Running {len(vlm_tasks)} VLM figure comparisons in parallel...")
            vlm_results = fast_parallel_vlm_calls(vlm_tasks, vlm_model, max_workers=5)
            results.update(vlm_results)

        return results
    finally:
        for temp_path in temp_files:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass


def setup(workspace_doc_id: str):
    """Setup function to initialize the evaluator."""
    global sheet_id, table_data, sheet_raw, df

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    load_gold_data()

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

    # Fallback to manual extraction if no table object found
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
    """Checkpoint 1: Spreadsheet Structure (7 steps)."""
    print("----------------- CHECKPOINT 1 ----------------")
    global model, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=7, result=0, name="Spreadsheet Structure")

    if df is None or df.empty:
        checkpoint.add_step("Table Data Extraction", False, 1,
                          "No table data found in spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    required_columns = [
        ("Title", ["title", "paper title", "paper name", "name"]),
        ("Authors", ["author", "authors", "by"]),
        ("Abstract", ["abstract", "summary"]),
        ("arXiv Link", ["arxiv", "arxiv link", "arxiv url", "paper link", "link", "url"]),
        ("Drive Link", ["drive link", "drive pdf link", "pdf link", "drive pdf", "drive", "pdf", "google drive"]),
        ("Figure 1", ["figure 1", "figure", "fig", "fig 1", "image", "screenshot"]),
        ("New Papers", ["new paper", "new", "checkbox", "added", "is new"])
    ]

    original_columns = [str(col) for col in df.columns]

    if model is None:
        model = load_model(model_id)
    matched_columns = match_columns(df, required_columns, model=model, parallel=True)

    for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
        step_start = time.time()
        if col_name in matched_columns:
            matched_column = matched_columns[col_name]
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
    """Checkpoint 2: Original Papers Validation (8 steps x N papers)."""
    print("----------------- CHECKPOINT 2 ----------------")
    global model, matched_columns, df, BROWSING_HISTORY, MATCHED_PAPERS_KEYWORD_STATUS
    checkpoint_start = time.time()

    N = GOLD_PAPERS.get('count', 0)
    checkpoint = Checkpoint(total=7*N + 10, result=0, name="Original Papers Validation")

    if N == 0:
        checkpoint.add_step("Gold Data", False, 1, "No gold papers data available",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if df is None or df.empty:
        checkpoint.add_step("User Data", False, 1, "No data in user's spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    gold_papers = GOLD_PAPERS.get('papers', [])

    title_col = matched_columns.get('Title')
    authors_col = matched_columns.get('Authors')
    abstract_col = matched_columns.get('Abstract')
    arxiv_col = matched_columns.get('arXiv Link')
    drive_col = matched_columns.get('Drive Link')
    figure_col = matched_columns.get('Figure 1')
    checkbox_col = matched_columns.get('New Papers')

    vlm_model = None
    figures_with_gold = [p for p in gold_papers if p.get('figure_1_path')]
    if figures_with_gold and figure_col:
        try:
            vlm_model = load_model(model_id)
        except Exception as e:
            print(f"Failed to load VLM model for figure comparison: {e}")

    title_matches = 0
    author_matches = 0
    abstract_matches = 0
    arxiv_valid = 0
    drive_valid = 0
    figure_matches = 0
    unchecked_count = 0
    arxiv_urls_found = 0
    unmatched_papers = []
    figure_tasks = []

    for gold_idx, gold in enumerate(gold_papers):
        gold_title = gold.get('title', '')
        gold_arxiv_id = gold.get('arxiv_id', '')

        matched_row = None
        for _, row in df.iterrows():
            if arxiv_col and gold_arxiv_id:
                user_arxiv_url = str(row.get(arxiv_col, ''))
                user_arxiv_id = extract_arxiv_id_from_url(user_arxiv_url)
                if user_arxiv_id and user_arxiv_id == gold_arxiv_id:
                    matched_row = row
                    break
            if title_col:
                user_title = str(row.get(title_col, ''))
                is_match, _ = fuzzy_match_text(gold_title, user_title, threshold=85)
                if is_match:
                    matched_row = row
                    break

        if matched_row is None:
            unmatched_papers.append(gold_title[:40])
            print(f"  Paper {gold_idx + 1}: '{gold_title[:50]}...' - NO MATCH FOUND")
            continue

        print(f"  Paper {gold_idx + 1}: '{gold_title[:50]}...' - MATCHED")

        MATCHED_PAPERS_KEYWORD_STATUS.append({
            'title': gold_title,
            'arxiv_id': gold_arxiv_id,
            'has_keyword': gold.get(KEYWORD_FIELD, False),
            'keyword_evaluated': gold.get('keyword_evaluated', True),
            'has_figure': bool(gold.get('figure_1_path')),
            'is_new_paper': False,
            'sheet_row': matched_row.name + 1  # +1 for header row
        })

        user_title = str(matched_row.get(title_col, '')) if title_col else ''
        is_match, _ = fuzzy_match_text(gold_title, user_title, threshold=85)
        if is_match:
            title_matches += 1

        gold_authors = gold.get('authors', [])
        user_authors_str = str(matched_row.get(authors_col, '')) if authors_col else ''
        user_authors = split_delimited_text(user_authors_str)
        auth_match, _ = compare_authors_list(user_authors, gold_authors, strict=False)
        if auth_match:
            author_matches += 1

        gold_abstract = gold.get('abstract', '')
        user_abstract = str(matched_row.get(abstract_col, '')) if abstract_col else ''
        abs_match, _ = fuzzy_match_text(gold_abstract, user_abstract, threshold=80)
        if abs_match:
            abstract_matches += 1

        user_arxiv_url = str(matched_row.get(arxiv_col, '')) if arxiv_col else ''
        user_arxiv_id = extract_arxiv_id_from_url(user_arxiv_url)
        if user_arxiv_id and user_arxiv_id == gold_arxiv_id:
            arxiv_valid += 1

        user_drive_url = str(matched_row.get(drive_col, '')) if drive_col else ''
        user_file_id = extract_drive_file_id(user_drive_url)
        if user_file_id:
            drive_valid += 1

        gold_figure_path = gold.get('figure_1_path')
        if gold_figure_path and figure_col and vlm_model:
            row_idx = get_sheet_row_index_from_dataframe_row(matched_row, header_rows=1)
            col_idx = get_column_index_by_name(df, 'Figure 1', matched_columns)
            user_figure_url = None
            if row_idx >= 0 and col_idx >= 0:
                user_figure_url = get_image_url_from_raw_sheet_cell(sheet_raw, row_idx, col_idx)
            if not user_figure_url:
                user_figure_val = str(matched_row.get(figure_col, ''))
                if user_figure_val and user_figure_val.lower() != 'nan':
                    user_figure_url = extract_image_url_from_cell(user_figure_val)
            if user_figure_url:
                figure_tasks.append({
                    'id': f'original_{gold_idx}',
                    'gold_path': gold_figure_path,
                    'user_url': user_figure_url
                })
            else:
                print(f"    No figure URL found for paper: {gold_title[:40]}...")
        # Papers without gold figure are simply not counted (not evaluated)

        if checkbox_col:
            checkbox_val = str(matched_row.get(checkbox_col, '')).upper()
            if checkbox_val in ['FALSE', '', 'NO', 'UNCHECKED', 'N', '0']:
                unchecked_count += 1

        if BROWSING_HISTORY:
            gold_arxiv_url = gold.get('arxiv_url', '')
            for url in BROWSING_HISTORY:
                if gold_arxiv_id and gold_arxiv_id in url:
                    arxiv_urls_found += 1
                    break
                elif gold_arxiv_url and gold_arxiv_url in url:
                    arxiv_urls_found += 1
                    break

    if figure_tasks and vlm_model:
        figure_results = parallel_validate_figures(figure_tasks, vlm_model)
        figure_matches += sum(1 for v in figure_results.values() if v)

    step_time = time.time() - checkpoint_start

    checkpoint.result += title_matches
    checkpoint.add_step("Titles Match", title_matches == N, 1,
                      f"{title_matches}/{N} titles match", execution_time=step_time)
    checkpoint.result += author_matches
    checkpoint.add_step("Authors Match", author_matches == N, 2,
                      f"{author_matches}/{N} author lists match", execution_time=0)
    checkpoint.result += abstract_matches
    checkpoint.add_step("Abstracts Match", abstract_matches == N, 3,
                      f"{abstract_matches}/{N} abstracts match", execution_time=0)
    checkpoint.result += arxiv_valid
    checkpoint.add_step("arXiv Links Valid", arxiv_valid == N, 4,
                      f"{arxiv_valid}/{N} arXiv links valid", execution_time=0)
    checkpoint.result += drive_valid
    checkpoint.add_step("Drive Links Valid", drive_valid == N, 5,
                      f"{drive_valid}/{N} Drive links valid", execution_time=0)

    # Figure 1 step — proportional out of 10, only scored against evaluable papers
    evaluable_figures = len(figures_with_gold)
    if evaluable_figures > 0 and figure_col and vlm_model:
        figure_score = int(figure_matches / evaluable_figures * 10)
        checkpoint.result += figure_score
        checkpoint.add_step("Figure 1 Images", figure_matches == evaluable_figures, 6,
                          f"{figure_matches}/{evaluable_figures} figures correct ({figure_score}/10)", execution_time=0)
    elif evaluable_figures == 0:
        checkpoint.add_step("Figure 1 Images", False, 6, "No gold figure data to evaluate", execution_time=0)
    elif not vlm_model:
        checkpoint.add_step("Figure 1 Images", False, 6, "VLM model not available", execution_time=0)
    else:
        checkpoint.add_step("Figure 1 Images", False, 6, "Figure 1 column not found", execution_time=0)

    if checkbox_col:
        checkpoint.result += unchecked_count
        checkpoint.add_step("Checkbox Unchecked", unchecked_count == N, 7,
                          f"{unchecked_count}/{N} original papers have unchecked checkbox", execution_time=0)
    else:
        checkpoint.add_step("Checkbox Unchecked", False, 7, "Checkbox column not found", execution_time=0)

    if BROWSING_HISTORY:
        checkpoint.result += arxiv_urls_found
        checkpoint.add_step("arXiv URLs Visited", arxiv_urls_found == N, 8,
                          f"{arxiv_urls_found}/{N} arXiv URLs in browsing history", execution_time=0)
    else:
        checkpoint.add_step("arXiv URLs Visited", False, 8, "No browsing history provided", execution_time=0)

    if unmatched_papers:
        print(f"  WARNING: {len(unmatched_papers)} papers could not be matched: {unmatched_papers}")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3: New Papers Discovery (N total points)."""
    print("----------------- CHECKPOINT 3 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()

    if not AUTHOR_LOOKUP or not AUTHOR_LOOKUP.get('original_papers'):
        checkpoint = Checkpoint(total=1, result=0, name="New Papers Discovery")
        checkpoint.add_step("Paper Coverage", False, 1, "No author lookup data available",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    original_papers_lookup = AUTHOR_LOOKUP.get('original_papers', [])
    N = len(original_papers_lookup)
    checkpoint = Checkpoint(total=N, result=0, name="New Papers Discovery")

    authors_col = matched_columns.get('Authors')
    title_col = matched_columns.get('Title')

    if not authors_col or df is None:
        checkpoint.add_step("Paper Coverage", False, 1, "Cannot check - no authors column or data",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    original_titles_normalized = set()
    for paper_entry in original_papers_lookup:
        orig_title = paper_entry.get('original_paper_title', '')
        if orig_title:
            original_titles_normalized.add(orig_title.lower().strip())

    papers_with_enough_new = 0
    missing_authors = []

    for paper_entry in original_papers_lookup:
        original_title = paper_entry.get('original_paper_title', '')
        first_authors_normalized = paper_entry.get('normalized_first_authors', [])
        expected_new = paper_entry.get('expected_new_papers', 3)
        original_arxiv_id = paper_entry.get('original_paper_arxiv_id', '')

        matching_new_papers = 0

        for _, row in df.iterrows():
            if title_col:
                user_title = str(row.get(title_col, '')).lower().strip()
                if user_title in original_titles_normalized:
                    continue
                # Fuzzy fallback for special characters (e.g. Λ vs L)
                if any(fuzzy_match_text(user_title, orig, threshold=90)[0] for orig in original_titles_normalized):
                    continue

            arxiv_col = matched_columns.get('arXiv Link')
            if arxiv_col:
                arxiv_url = str(row.get(arxiv_col, ''))
                arxiv_id = extract_arxiv_id_from_url(arxiv_url)
                if arxiv_id and arxiv_id == original_arxiv_id:
                    continue

            user_authors_str = str(row.get(authors_col, ''))
            user_authors = split_delimited_text(user_authors_str)
            user_authors_normalized = [normalize_name(a, remove_suffixes=True) for a in user_authors]

            if any(fa in user_authors_normalized for fa in first_authors_normalized):
                matching_new_papers += 1

        if matching_new_papers >= expected_new:
            papers_with_enough_new += 1
        else:
            missing_authors.append(f"{original_title[:30]}...: found {matching_new_papers}, need {expected_new}")

    checkpoint.result = papers_with_enough_new
    if papers_with_enough_new == N:
        checkpoint.add_step("Paper Coverage", True, 1,
                          f"All {N} original papers have enough new papers",
                          score=0, execution_time=time.time() - checkpoint_start)
    else:
        checkpoint.add_step("Paper Coverage", False, 1,
                          f"{papers_with_enough_new}/{N} original papers have enough new papers. "
                          f"Missing: {'; '.join(missing_authors)}",
                          score=0, execution_time=time.time() - checkpoint_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """Checkpoint 4: New Papers Validation (8 steps x M papers)."""
    print("----------------- CHECKPOINT 4 ----------------")
    global matched_columns, df, BROWSING_HISTORY, MATCHED_PAPERS_KEYWORD_STATUS
    checkpoint_start = time.time()

    title_col = matched_columns.get('Title')
    authors_col = matched_columns.get('Authors')
    arxiv_col = matched_columns.get('arXiv Link')
    abstract_col = matched_columns.get('Abstract')
    drive_col = matched_columns.get('Drive Link')
    figure_col = matched_columns.get('Figure 1')
    checkbox_col = matched_columns.get('New Papers')

    # Compute MAX_NEW_PAPERS from gold data
    original_papers_lookup = AUTHOR_LOOKUP.get('original_papers', []) if AUTHOR_LOOKUP else []
    MAX_NEW_PAPERS = sum(min(3, p.get('expected_new_papers', 3)) for p in original_papers_lookup)
    if MAX_NEW_PAPERS == 0:
        MAX_NEW_PAPERS = len(original_papers_lookup) * 3

    if not AUTHOR_LOOKUP or not AUTHOR_LOOKUP.get('original_papers') or df is None or df.empty:
        checkpoint = Checkpoint(total=6*MAX_NEW_PAPERS + 20, result=0, name="New Papers Validation")
        checkpoint.add_step("New Papers", False, 1, "Cannot validate - no data",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    original_titles_normalized = set()
    original_arxiv_ids = set()
    all_first_authors = set()
    for paper_entry in original_papers_lookup:
        orig_title = paper_entry.get('original_paper_title', '')
        orig_arxiv_id = paper_entry.get('original_paper_arxiv_id', '')
        if orig_title:
            original_titles_normalized.add(orig_title.lower().strip())
        if orig_arxiv_id:
            original_arxiv_ids.add(orig_arxiv_id)
        all_first_authors.update(paper_entry.get('normalized_first_authors', []))

    user_new_papers_rows = []
    for row_idx, row in df.iterrows():
        if title_col:
            user_title = str(row.get(title_col, '')).lower().strip()
            if user_title in original_titles_normalized:
                continue
            # Fuzzy fallback for special characters (e.g. Λ vs L)
            if any(fuzzy_match_text(user_title, orig, threshold=90)[0] for orig in original_titles_normalized):
                continue
        if arxiv_col:
            arxiv_url = str(row.get(arxiv_col, ''))
            arxiv_id = extract_arxiv_id_from_url(arxiv_url)
            if arxiv_id and arxiv_id in original_arxiv_ids:
                continue
        user_authors_str = str(row.get(authors_col, '')) if authors_col else ''
        user_authors = split_delimited_text(user_authors_str)
        user_authors_normalized = [normalize_name(a, remove_suffixes=True) for a in user_authors]
        if any(fa in user_authors_normalized for fa in all_first_authors):
            user_new_papers_rows.append(row_idx)

    M = len(user_new_papers_rows)
    print(f"Found {M} new papers in user's spreadsheet to validate (max {MAX_NEW_PAPERS})")

    checkpoint = Checkpoint(total=6*MAX_NEW_PAPERS + 20, result=0, name="New Papers Validation")

    if M == 0:
        for i, name in enumerate(["Titles Match", "Authors Match", "Abstracts Match",
                                   "arXiv Links Valid", "Drive Links Valid", "Figure 1 Images",
                                   "Checkbox Checked", "arXiv URLs Visited"], start=1):
            checkpoint.add_step(name, False, i, f"0/{MAX_NEW_PAPERS} - No new papers found", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    all_gold_new_papers = GOLD_NEW_PAPERS.get('papers', [])
    gold_by_arxiv_id = {}
    gold_by_title_lower = {}
    for gold in all_gold_new_papers:
        arxiv_id = gold.get('arxiv_id', '')
        title = gold.get('title', '')
        if arxiv_id:
            gold_by_arxiv_id[arxiv_id] = gold
        if title:
            gold_by_title_lower[title.lower().strip()] = gold

    vlm_model = None
    figures_with_gold = [p for p in all_gold_new_papers if p.get('figure_1_path')]
    if figures_with_gold and figure_col:
        try:
            vlm_model = load_model(model_id)
        except Exception as e:
            print(f"Failed to load VLM model: {e}")

    title_matches = 0
    author_matches = 0
    abstract_matches = 0
    arxiv_valid = 0
    drive_valid = 0
    figure_matches = 0
    checkbox_checked = 0
    arxiv_urls_visited = 0
    unmatched_to_gold = []
    papers_with_gold_figures = 0
    figure_tasks = []

    for paper_idx, row_idx in enumerate(user_new_papers_rows):
        row = df.loc[row_idx]
        user_title = str(row.get(title_col, '')) if title_col else ''
        user_arxiv_url = str(row.get(arxiv_col, '')) if arxiv_col else ''
        user_arxiv_id = extract_arxiv_id_from_url(user_arxiv_url)

        gold = None
        if user_arxiv_id and user_arxiv_id in gold_by_arxiv_id:
            gold = gold_by_arxiv_id[user_arxiv_id]
        else:
            for gold_title_key, gold_paper in gold_by_title_lower.items():
                is_match, _ = fuzzy_match_text(user_title, gold_paper.get('title', ''), threshold=85)
                if is_match:
                    gold = gold_paper
                    break

        # Checkbox validation for ALL non-original rows (regardless of gold match)
        if checkbox_col:
            checkbox_val = str(row.get(checkbox_col, '')).upper()
            if checkbox_val in ['TRUE', 'YES', 'CHECKED', 'Y', '1']:
                checkbox_checked += 1

        if gold is None:
            unmatched_to_gold.append(user_title[:40])
            continue

        MATCHED_PAPERS_KEYWORD_STATUS.append({
            'title': gold.get('title', ''),
            'arxiv_id': gold.get('arxiv_id', ''),
            'has_keyword': gold.get(KEYWORD_FIELD, False),
            'keyword_evaluated': gold.get('keyword_evaluated', True),
            'has_figure': bool(gold.get('figure_1_path')),
            'is_new_paper': True,
            'sheet_row': row_idx + 1  # +1 for header row
        })

        gold_title = gold.get('title', '')
        gold_arxiv_id = gold.get('arxiv_id', '')

        is_match, _ = fuzzy_match_text(gold_title, user_title, threshold=85)
        if is_match:
            title_matches += 1

        gold_authors = gold.get('authors', [])
        user_authors_str = str(row.get(authors_col, '')) if authors_col else ''
        user_authors = split_delimited_text(user_authors_str)
        auth_match, _ = compare_authors_list(user_authors, gold_authors, strict=False)
        if auth_match:
            author_matches += 1

        gold_abstract = gold.get('abstract', '')
        user_abstract = str(row.get(abstract_col, '')) if abstract_col else ''
        abs_match, _ = fuzzy_match_text(gold_abstract, user_abstract, threshold=80)
        if abs_match:
            abstract_matches += 1

        if user_arxiv_id and user_arxiv_id == gold_arxiv_id:
            arxiv_valid += 1

        user_drive_url = str(row.get(drive_col, '')) if drive_col else ''
        user_file_id = extract_drive_file_id(user_drive_url)
        if user_file_id:
            drive_valid += 1

        gold_figure_path = gold.get('figure_1_path')
        if gold_figure_path and figure_col and vlm_model:
            papers_with_gold_figures += 1
            raw_row_idx = row_idx + 1
            col_idx = get_column_index_by_name(df, 'Figure 1', matched_columns)
            user_figure_url = None
            if raw_row_idx >= 0 and col_idx >= 0:
                user_figure_url = get_image_url_from_raw_sheet_cell(sheet_raw, raw_row_idx, col_idx)
            if not user_figure_url:
                user_figure_val = str(row.get(figure_col, ''))
                if user_figure_val and user_figure_val.lower() != 'nan':
                    user_figure_url = extract_image_url_from_cell(user_figure_val)
            if user_figure_url:
                figure_tasks.append({'id': f'new_{paper_idx}', 'gold_path': gold_figure_path, 'user_url': user_figure_url})

        if BROWSING_HISTORY:
            gold_arxiv_url = gold.get('arxiv_url', '')
            for url in BROWSING_HISTORY:
                if gold_arxiv_id and gold_arxiv_id in url:
                    arxiv_urls_visited += 1
                    break
                elif gold_arxiv_url and gold_arxiv_url in url:
                    arxiv_urls_visited += 1
                    break

    if figure_tasks and vlm_model:
        figure_results = parallel_validate_figures(figure_tasks, vlm_model)
        figure_matches += sum(1 for v in figure_results.values() if v)

    step_time = time.time() - checkpoint_start

    checkpoint.result += title_matches
    checkpoint.add_step("Titles Match", title_matches == MAX_NEW_PAPERS, 1,
                      f"{title_matches}/{MAX_NEW_PAPERS} new paper titles match", execution_time=step_time)
    checkpoint.result += author_matches
    checkpoint.add_step("Authors Match", author_matches == MAX_NEW_PAPERS, 2,
                      f"{author_matches}/{MAX_NEW_PAPERS} new paper authors match", execution_time=0)
    checkpoint.result += abstract_matches
    checkpoint.add_step("Abstracts Match", abstract_matches == MAX_NEW_PAPERS, 3,
                      f"{abstract_matches}/{MAX_NEW_PAPERS} new paper abstracts match", execution_time=0)
    checkpoint.result += arxiv_valid
    checkpoint.add_step("arXiv Links Valid", arxiv_valid == MAX_NEW_PAPERS, 4,
                      f"{arxiv_valid}/{MAX_NEW_PAPERS} new paper arXiv links valid", execution_time=0)
    checkpoint.result += drive_valid
    checkpoint.add_step("Drive Links Valid", drive_valid == MAX_NEW_PAPERS, 5,
                      f"{drive_valid}/{MAX_NEW_PAPERS} new papers have valid Drive links", execution_time=0)

    # Figure 1 step — proportional out of 10, only scored against matched papers with gold figures
    if papers_with_gold_figures > 0 and figure_col and vlm_model:
        figure_score = int(figure_matches / papers_with_gold_figures * 10)
        checkpoint.result += figure_score
        checkpoint.add_step("Figure 1 Images", figure_matches == papers_with_gold_figures, 6,
                          f"{figure_matches}/{papers_with_gold_figures} figures correct ({figure_score}/10)", execution_time=0)
    elif papers_with_gold_figures == 0:
        checkpoint.add_step("Figure 1 Images", False, 6, "No gold figure data to evaluate", execution_time=0)
    elif not vlm_model:
        checkpoint.add_step("Figure 1 Images", False, 6, "VLM model not available", execution_time=0)
    else:
        checkpoint.add_step("Figure 1 Images", False, 6, "Figure 1 column not found", execution_time=0)

    # Checkbox step — checked against all non-original rows, proportional out of 10
    total_new_rows = len(user_new_papers_rows)
    if checkbox_col and total_new_rows > 0:
        checkbox_score = int(checkbox_checked / total_new_rows * 10)
        checkpoint.result += checkbox_score
        checkpoint.add_step("Checkbox Checked", checkbox_checked == total_new_rows, 7,
                          f"{checkbox_checked}/{total_new_rows} new papers have checked checkbox ({checkbox_score}/10)", execution_time=0)
    elif checkbox_col:
        checkpoint.add_step("Checkbox Checked", True, 7, "No new papers to check", execution_time=0)
    else:
        checkpoint.add_step("Checkbox Checked", False, 7, "Checkbox column not found", execution_time=0)

    if BROWSING_HISTORY:
        checkpoint.result += arxiv_urls_visited
        checkpoint.add_step("arXiv URLs Visited", arxiv_urls_visited == MAX_NEW_PAPERS, 8,
                          f"{arxiv_urls_visited}/{MAX_NEW_PAPERS} new paper arXiv URLs in browsing history", execution_time=0)
    else:
        checkpoint.add_step("arXiv URLs Visited", False, 8, "No browsing history provided", execution_time=0)

    if unmatched_to_gold:
        print(f"  WARNING: {len(unmatched_to_gold)} new papers could not be matched to gold: {unmatched_to_gold[:5]}")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """Checkpoint 5: Formatting & Organization (3 binary steps)."""
    print("----------------- CHECKPOINT 5 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=12, result=0, name="Formatting & Organization")

    if not sheet_raw:
        for i in range(1, 4):
            checkpoint.add_step(f"Formatting Check {i}", False, i, "Could not access raw sheet data", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i in range(1, 4):
                checkpoint.add_step(f"Formatting Check {i}", False, i, "No sheets found", execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        sheet_data = sheets[0].get('data', [{}])[0]
        rows = sheet_data.get('rowData', [])
        col_metadata = sheet_data.get('columnMetadata', [])
        num_rows = len(rows)
    except Exception as e:
        for i in range(1, 4):
            checkpoint.add_step(f"Formatting Check {i}", False, i, f"Error: {str(e)[:50]}", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    row_colors = []
    yellow_rows = []

    for row_idx in range(1, num_rows):
        color = get_row_background_color(sheet_raw, row_idx)
        color_class = classify_row_color(color)
        row_colors.append(color_class)
        if color_class in ('yellow', 'orange'):
            yellow_rows.append(row_idx)

    # Step 1: Yellow highlighting — only for papers where keyword was evaluated
    step_start = time.time()
    evaluated_papers = [p for p in MATCHED_PAPERS_KEYWORD_STATUS if p.get('keyword_evaluated', True)]
    keyword_positive = [p for p in evaluated_papers if p.get('has_keyword', False)]
    expected_yellow = len(keyword_positive)
    yellow_count = len(yellow_rows)

    if len(evaluated_papers) > 0 and expected_yellow > 0:
        # Check which specific rows should be yellow vs which are yellow
        expected_yellow_rows = set(p.get('sheet_row') for p in keyword_positive if p.get('sheet_row') is not None)
        actual_yellow_rows = set(yellow_rows)
        correctly_yellow = len(expected_yellow_rows & actual_yellow_rows)
        extra_yellow = len(actual_yellow_rows - expected_yellow_rows)
        highlight_score = int(correctly_yellow / expected_yellow * 10)
        if extra_yellow > 0:
            highlight_score = max(0, highlight_score - extra_yellow)
        checkpoint.result += highlight_score
        msg = f"{correctly_yellow}/{expected_yellow} correct yellow rows ({highlight_score}/10)"
        if extra_yellow > 0:
            msg += f" — {extra_yellow} extra yellow rows"
        checkpoint.add_step("Yellow Highlighting", correctly_yellow == expected_yellow and extra_yellow == 0, 1,
                          msg, execution_time=time.time() - step_start)
    elif len(evaluated_papers) > 0 and expected_yellow == 0:
        if yellow_count == 0:
            checkpoint.result += 1
            checkpoint.add_step("Yellow Highlighting", True, 1,
                              f"No '{HIGHLIGHT_KEYWORD}' papers among {len(evaluated_papers)} evaluated papers, correctly no yellow rows",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Yellow Highlighting", False, 1,
                              f"No '{HIGHLIGHT_KEYWORD}' papers expected but found {yellow_count} yellow rows",
                              execution_time=time.time() - step_start)
    else:
        checkpoint.add_step("Yellow Highlighting", False, 1,
                          f"0 papers evaluated for keyword detection",
                          execution_time=time.time() - step_start)

    # Step 2: Row grouping
    step_start = time.time()
    is_grouped, grouping_msg = validate_color_grouping(row_colors)
    if is_grouped and yellow_rows:
        if yellow_rows[0] != 1:
            is_grouped = False
            grouping_msg = f"Yellow rows not at top (first yellow at row {yellow_rows[0] + 1})"
    checkpoint.add_step("Row Grouping", is_grouped, 2, grouping_msg, execution_time=time.time() - step_start)

    # Step 3: Text overflow
    step_start = time.time()
    try:
        from src.browsergym.knows.eval.eval_utils.table_utils import is_text_visible_in_cell

        CHAR_WIDTH = 7
        hidden_cells = 0
        total_cells_checked = 0

        def get_col_width(c_idx):
            if c_idx < len(col_metadata):
                return col_metadata[c_idx].get('pixelSize', 100)
            return 100

        for row_idx, row in enumerate(rows):
            if row_idx == 0:
                continue
            row_values = row.get('values', [])
            for c_idx, cell in enumerate(row_values):
                content = cell.get('formattedValue', '')
                if not content:
                    continue
                total_cells_checked += 1
                col_width = get_col_width(c_idx)
                fmt = cell.get('effectiveFormat', {})
                wrap_strategy = fmt.get('wrapStrategy', 'OVERFLOW_CELL')
                if not is_text_visible_in_cell(content, col_width, wrap_strategy, row_values, c_idx, CHAR_WIDTH):
                    hidden_cells += 1

        if hidden_cells == 0:
            checkpoint.add_step("Text Overflow", True, 3,
                              f"All {total_cells_checked} cells have visible text", execution_time=time.time() - step_start)
        else:
            checkpoint.add_step("Text Overflow", False, 3,
                              f"{hidden_cells}/{total_cells_checked} cells have hidden/clipped text", execution_time=time.time() - step_start)
    except ImportError:
        checkpoint.add_step("Text Overflow", False, 3, "table_utils module not available", execution_time=time.time() - step_start)
    except Exception as e:
        checkpoint.add_step("Text Overflow", False, 3, f"Error: {str(e)[:50]}", execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None, browsing_history: List[str] = None) -> Result:
    """Grade all checkpoints for the paper sorting task."""
    global BROWSING_HISTORY, MATCHED_PAPERS_KEYWORD_STATUS

    total_start_time = time.time()
    BROWSING_HISTORY = browsing_history or []
    MATCHED_PAPERS_KEYWORD_STATUS = []

    try:
        setup(workspace_doc_id)
        checkpoints: List[Checkpoint] = []
        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())
        checkpoints.append(grade_checkpoint_5())

        total_execution_time = time.time() - total_start_time
        return Result(checkpoints, total_execution_time=total_execution_time)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate paper sorting spreadsheet")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Sheets document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Instance: {INSTANCE_NAME} | Keyword: {HIGHLIGHT_KEYWORD}")
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
    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")
