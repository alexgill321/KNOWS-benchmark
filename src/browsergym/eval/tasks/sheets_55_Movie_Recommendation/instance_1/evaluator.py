"""Evaluator for the Movie Recommendation Google Sheets task."""

import os
import re
import sys
import time
import traceback
import argparse
from typing import List, Optional
import pandas as pd


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
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score
from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.google_sheets_utils import (
    get_sheet_content,
    extract_tables_from_sheet,
)
from src.browsergym.eval.eval_utils.table_utils import (
    match_columns,
    get_cell,
    is_cell_bold,
    cell_bg_hex,
    get_background_color,
    classify_row_color,
)
from src.browsergym.eval.eval_utils.models import load_model

# Task-specific utilities
from src.browsergym.eval.tasks.sheets_55_Movie_Recommendation.utils import (
    fetch_imdb_data,
    fetch_imdb_awards_text,
    verify_genre,
    verify_imdb_score,
    verify_mpa_rating,
    verify_oscar_awards,
    close_browser,
)

# Preferred genres from the task description (update per instance)
PREFERRED_GENRES = ["Action", "Drama", "Thriller", "Sci-Fi", "Comedy"]

QUALIFYING_OSCARS = [
    "Best Actor",
    "Best Actress",
    "Best Director",
    "Best Original Screenplay",
    "Best Adapted Screenplay",
    "Best Cinematography",
]

REQUIRED_COLUMNS = [
    ("Movie Title", ["movie", "title", "movie title", "film", "name"]),
    ("Genre", ["genre", "genres", "category"]),
    ("Movie Rating", ["rating", "mpa", "age rating", "movie rating", "mpaa", "certification"]),
    ("IMDb Score", ["imdb", "score", "imdb score", "imdb rating"]),
    ("Release Year", ["year", "release", "release year", "released"]),
    ("Duration", ["duration", "runtime", "length", "time", "run time", "minutes"]),
    ("Oscar Awards Won", ["oscar", "oscars", "awards", "academy", "oscar awards"]),
]


def parse_duration_to_minutes(raw: str) -> Optional[float]:
    """Parse a duration string to a float number of minutes.

    Handles formats:
        - "2h 15m", "2 hours 15 minutes", "2h"
        - "2:15" (H:MM colon format)
        - "135", "135 min", "135 minutes"

    Args:
        raw: Raw duration string.

    Returns:
        Duration in minutes, or None if unparseable.
    """
    s = str(raw).strip().lower()

    # "Xh Ym" format
    m = re.match(r'^(\d+)\s*h(?:ours?)?\s*(?:(\d+)\s*m(?:in(?:utes?)?)?)?$', s)
    if m:
        return int(m.group(1)) * 60 + (int(m.group(2)) if m.group(2) else 0)

    # "H:MM" colon format
    m = re.match(r'^(\d+):(\d+)$', s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))

    # Plain number with optional "min"/"minutes" suffix
    m = re.match(r'^(\d+(?:\.\d+)?)\s*(?:min(?:utes?)?|m)?$', s)
    if m:
        return float(m.group(1))

    return None


# Google services
DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

model = None
model_id = "gemini-2.5-flash-google-ai"

# Global variables
sheet_id = None
table_data = None
sheet_raw = None
matched_columns = None
df = None


def setup(workspace_doc_id):
    """Initialize the evaluator by fetching sheet data.

    Args:
        workspace_doc_id (str): Google Sheets document ID.
    """
    global sheet_id, table_data, sheet_raw, df

    sheet_id = workspace_doc_id
    print(f"Using workspace document ID: {sheet_id}")

    table_data = extract_tables_from_sheet(sheet_id, SHEETS_SERVICE)
    sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

    if table_data:
        first_table = table_data[0]
        df = first_table.df if hasattr(first_table, "df") else first_table
        if isinstance(df, dict):
            df = pd.DataFrame(df)


def grade_checkpoint_1():
    """Checkpoint 1 (4 pts): Spreadsheet Structure.

    Steps:
        1. All required labeled columns present.
        2. At least 5 movies (data rows).
        3. Each row is a unique movie (no duplicate titles).
        4. No blank cells in required columns.
    """
    global model, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Spreadsheet Structure")

    if df is None or df.empty:
        for i in range(1, 5):
            checkpoint.add_step(f"Step {i}", False, i, "No data found in spreadsheet")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Required columns
    step_start = time.time()
    if model is None:
        model = load_model(model_id)
    matched_columns = match_columns(df, REQUIRED_COLUMNS, model=model, parallel=True)

    all_found = len(matched_columns) == len(REQUIRED_COLUMNS)
    missing = [col for col, _ in REQUIRED_COLUMNS if col not in matched_columns]
    details = f"Found {len(matched_columns)}/{len(REQUIRED_COLUMNS)} columns"
    if missing:
        details += f". Missing: {', '.join(missing)}"
    checkpoint.add_step(
        "Required Columns", all_found, 1, details,
        execution_time=time.time() - step_start,
    )

    # Step 2: At least 5 movies
    step_start = time.time()
    num_rows = len(df)
    checkpoint.add_step(
        "At Least 5 Movies", num_rows >= 5, 2,
        f"Found {num_rows} data rows",
        execution_time=time.time() - step_start,
    )

    # Step 3: Unique movies (no duplicates)
    step_start = time.time()
    if "Movie Title" in matched_columns:
        title_col = matched_columns["Movie Title"]
        titles = df[title_col].dropna().astype(str).str.strip().str.lower()
        duplicates = titles[titles.duplicated()].unique().tolist()
        is_unique = len(duplicates) == 0
        details = "No duplicate titles" if is_unique else f"Duplicates: {', '.join(duplicates)}"
    else:
        is_unique = False
        details = "Movie Title column not found"
    checkpoint.add_step(
        "Unique Movies", is_unique, 3, details,
        execution_time=time.time() - step_start,
    )

    # Step 4: No blank cells in required columns
    step_start = time.time()
    blank_details = []
    for col_name, matched_col in matched_columns.items():
        blanks = df[matched_col].isna().sum() + (df[matched_col].astype(str).str.strip() == "").sum()
        if blanks > 0:
            blank_details.append(f"{col_name}: {blanks} blank(s)")
    no_blanks = len(blank_details) == 0
    details = "No blank cells" if no_blanks else f"Blanks found: {'; '.join(blank_details)}"
    checkpoint.add_step(
        "No Blank Cells", no_blanks, 4, details,
        execution_time=time.time() - step_start,
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """Checkpoint 2 (40 pts): Data Accuracy.

    Fetches structured IMDb data (JSON-LD) via Playwright for each movie.
    Uses programmatic comparison for genre, IMDb score, and MPA rating.
    Uses LLM only for Oscar verification (unstructured awards data).
    Each step scored proportionally per movie:
    round((movies_passing / total_movies) * 10).

    Steps:
        1. Each movie belongs to at least one preferred genre (10 pt, proportional).
        2. Every movie won a qualifying Oscar award (10 pt, proportional).
        3. Each IMDb Score matches actual rating (within +/-0.1) and >= 6.5 (10 pt, proportional).
        4. Each MPA/age rating is correct (10 pt, proportional).
    """
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Data Accuracy")

    if df is None or df.empty or not matched_columns:
        for i in range(1, 5):
            checkpoint.add_step(f"Step {i}", False, i, "No data available", score=0, max_score=10)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    title_col = matched_columns.get("Movie Title")
    if not title_col:
        for i in range(1, 5):
            checkpoint.add_step(f"Step {i}", False, i, "Movie Title column not found", score=0, max_score=10)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_movies = len(df)

    # Pre-fetch structured IMDb data for all movies
    year_col = matched_columns.get("Release Year")
    imdb_data_map = {}  # movie -> JSON-LD dict
    imdb_id_map = {}    # movie -> imdb_id
    for idx, row in df.iterrows():
        movie = str(row[title_col]).strip()
        year = None
        if year_col:
            try:
                year = str(row[year_col]).strip()
            except Exception:
                pass
        data, imdb_id = fetch_imdb_data(movie, year)
        imdb_data_map[movie] = data
        imdb_id_map[movie] = imdb_id

    # Step 1: Genre verification (programmatic — no LLM)
    step_start = time.time()
    genre_failures = []
    genre_not_found = []
    for idx, row in df.iterrows():
        movie = str(row[title_col]).strip()
        ok = verify_genre(imdb_data_map.get(movie), PREFERRED_GENRES)
        if ok is None:
            genre_not_found.append(movie)
        elif not ok:
            imdb_genres = (imdb_data_map.get(movie) or {}).get("genre", [])
            genre_failures.append(f"{movie} (IMDb genres: {imdb_genres})")
    genre_pass = num_movies - len(genre_failures) - len(genre_not_found)
    genre_score = calculate_percentage_score(genre_pass, num_movies, max_points=10)
    details = f"{genre_pass}/{num_movies} movies match preferred genres"
    if genre_not_found:
        details += f". Could not retrieve IMDb data: {', '.join(genre_not_found)}"
    if genre_failures:
        details += f". Failed: {', '.join(genre_failures)}"
    checkpoint.add_step(
        "Genre Verification", genre_pass == num_movies, 1, details,
        score=genre_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 2: Oscar verification via IMDb awards page + LLM
    step_start = time.time()
    if model is None:
        model = load_model(model_id)
    oscar_failures = []
    oscar_not_found = []
    for idx, row in df.iterrows():
        movie = str(row[title_col]).strip()
        imdb_id = imdb_id_map.get(movie)
        awards_text = fetch_imdb_awards_text(imdb_id)
        ok = verify_oscar_awards(model, movie, QUALIFYING_OSCARS, awards_text)
        if ok is None:
            oscar_not_found.append(movie)
        elif not ok:
            oscar_failures.append(movie)
    oscar_pass = num_movies - len(oscar_failures) - len(oscar_not_found)
    oscar_score = calculate_percentage_score(oscar_pass, num_movies, max_points=10)
    details = f"{oscar_pass}/{num_movies} movies verified as Oscar winners"
    if oscar_not_found:
        details += f". Could not retrieve awards data: {', '.join(oscar_not_found)}"
    if oscar_failures:
        details += f". Failed: {', '.join(oscar_failures)}"
    checkpoint.add_step(
        "Oscar Verification", oscar_pass == num_movies, 2, details,
        score=oscar_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 3: IMDb Score verification (programmatic — no LLM)
    step_start = time.time()
    score_col = matched_columns.get("IMDb Score")
    if score_col:
        score_failures = []
        score_not_found = []
        for idx, row in df.iterrows():
            movie = str(row[title_col]).strip()
            try:
                sheet_score = float(row[score_col])
            except (ValueError, TypeError):
                score_failures.append(f"{movie} (invalid score)")
                continue

            if sheet_score < 6.5:
                score_failures.append(f"{movie} (score {sheet_score} < 6.5)")
                continue

            ok = verify_imdb_score(imdb_data_map.get(movie), sheet_score, tolerance=0.1)
            if ok is None:
                score_not_found.append(movie)
            elif not ok:
                actual = (imdb_data_map.get(movie) or {}).get("aggregateRating", {}).get("ratingValue", "?")
                score_failures.append(f"{movie} (sheet={sheet_score}, actual={actual})")

        score_pass = num_movies - len(score_failures) - len(score_not_found)
        score_step = calculate_percentage_score(score_pass, num_movies, max_points=10)
        details = f"{score_pass}/{num_movies} IMDb scores verified"
        if score_not_found:
            details += f". Could not retrieve IMDb data: {', '.join(score_not_found)}"
        if score_failures:
            details += f". Failed: {', '.join(score_failures)}"
    else:
        score_pass = 0
        score_evaluated = 0
        score_step = 0
        details = "IMDb Score column not found"
    checkpoint.add_step(
        "IMDb Score Verification", score_pass == num_movies, 3, details,
        score=score_step, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 4: MPA Rating verification (programmatic — no LLM)
    step_start = time.time()
    rating_col = matched_columns.get("Movie Rating")
    if rating_col:
        rating_failures = []
        rating_not_found = []
        for idx, row in df.iterrows():
            movie = str(row[title_col]).strip()
            sheet_rating = str(row[rating_col]).strip()
            ok = verify_mpa_rating(imdb_data_map.get(movie), sheet_rating)
            if ok is None:
                rating_not_found.append(movie)
            elif not ok:
                actual = (imdb_data_map.get(movie) or {}).get("contentRating", "?")
                rating_failures.append(f"{movie} (sheet={sheet_rating}, actual={actual})")
        rating_pass = num_movies - len(rating_failures) - len(rating_not_found)
        rating_score = calculate_percentage_score(rating_pass, num_movies, max_points=10)
        details = f"{rating_pass}/{num_movies} MPA ratings verified"
        if rating_not_found:
            details += f". Could not retrieve IMDb data: {', '.join(rating_not_found)}"
        if rating_failures:
            details += f". Failed: {', '.join(rating_failures)}"
    else:
        rating_pass = 0
        rating_evaluated = 0
        rating_score = 0
        details = "Movie Rating column not found"
    checkpoint.add_step(
        "MPA Rating Verification", rating_pass == num_movies, 4, details,
        score=rating_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Clean up Playwright browser
    close_browser()

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3 (3 pts): Sorting and Conditional Formatting.

    Steps:
        1. Movie list sorted by Duration.
        2. Highest IMDb Score cell(s) have green fill + bold text.
        3. Lowest IMDb Score cell(s) have red fill + bold text.
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Sorting and Conditional Formatting")

    if df is None or df.empty or not matched_columns or not sheet_raw:
        for i in range(1, 4):
            checkpoint.add_step(f"Step {i}", False, i, "No data available")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Sorted by Duration
    step_start = time.time()
    duration_col = matched_columns.get("Duration")
    if duration_col:
        raw_durations = df[duration_col].astype(str).tolist()
        parsed = [parse_duration_to_minutes(d) for d in raw_durations]

        valid = [v for v in parsed if v is not None]
        if len(valid) >= 2:
            is_asc = all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1))
            is_desc = all(valid[i] >= valid[i + 1] for i in range(len(valid) - 1))
            sorted_ok = is_asc or is_desc
            order = "ascending" if is_asc else ("descending" if is_desc else "unsorted")
            details = f"Duration sorted ({order})" if sorted_ok else f"Not sorted: {valid[:5]}..."
        else:
            sorted_ok = False
            details = f"Could not parse enough durations ({len(valid)} valid out of {len(parsed)})"
    else:
        sorted_ok = False
        details = "Duration column not found"
    checkpoint.add_step(
        "Sorted by Duration", sorted_ok, 1, details,
        execution_time=time.time() - step_start,
    )

    # Steps 2 & 3: Conditional formatting on IMDb Score cells
    score_col = matched_columns.get("IMDb Score")
    if score_col and table_data:
        scores = []
        for _, row in df.iterrows():
            try:
                scores.append(float(row[score_col]))
            except (ValueError, TypeError):
                scores.append(None)

        valid_scores = [s for s in scores if s is not None]

        if valid_scores:
            max_score = max(valid_scores)
            min_score = min(valid_scores)

            # Find raw sheet column index for IMDb Score
            table = table_data[0]
            header_row_idx = table.start_row
            score_col_idx = None
            for i, col in enumerate(df.columns):
                if col == matched_columns["IMDb Score"]:
                    score_col_idx = table.start_col + i
                    break

            sheet_tab = sheet_raw["sheets"][0]

            # Step 2: Highest IMDb Score -> green fill + bold
            step_start = time.time()
            if score_col_idx is not None:
                highest_ok = True
                highest_issues = []
                for row_i, score in enumerate(scores):
                    if score == max_score:
                        raw_row = header_row_idx + 1 + row_i
                        cell = get_cell(sheet_tab, raw_row, score_col_idx)
                        bold = is_cell_bold(cell)
                        bg = get_background_color(sheet_raw, raw_row, score_col_idx)
                        color = classify_row_color(bg)
                        if not bold or color != "green":
                            highest_ok = False
                            hex_val = cell_bg_hex(sheet_raw, raw_row, score_col_idx)
                            highest_issues.append(
                                f"Row {row_i + 1}: bold={bold}, color={color} (bg={hex_val})"
                            )
                details = (
                    "Green fill + bold on highest score"
                    if highest_ok
                    else f"Issues: {'; '.join(highest_issues)}"
                )
            else:
                highest_ok = False
                details = "Could not locate IMDb Score column in raw sheet"
            checkpoint.add_step(
                "Highest Score Green + Bold", highest_ok, 2, details,
                execution_time=time.time() - step_start,
            )

            # Step 3: Lowest IMDb Score -> red fill + bold
            step_start = time.time()
            if score_col_idx is not None:
                lowest_ok = True
                lowest_issues = []
                for row_i, score in enumerate(scores):
                    if score == min_score:
                        raw_row = header_row_idx + 1 + row_i
                        cell = get_cell(sheet_tab, raw_row, score_col_idx)
                        bold = is_cell_bold(cell)
                        bg = get_background_color(sheet_raw, raw_row, score_col_idx)
                        color = classify_row_color(bg)
                        if not bold or color != "red":
                            lowest_ok = False
                            hex_val = cell_bg_hex(sheet_raw, raw_row, score_col_idx)
                            lowest_issues.append(
                                f"Row {row_i + 1}: bold={bold}, color={color} (bg={hex_val})"
                            )
                details = (
                    "Red fill + bold on lowest score"
                    if lowest_ok
                    else f"Issues: {'; '.join(lowest_issues)}"
                )
            else:
                lowest_ok = False
                details = "Could not locate IMDb Score column in raw sheet"
            checkpoint.add_step(
                "Lowest Score Red + Bold", lowest_ok, 3, details,
                execution_time=time.time() - step_start,
            )
        else:
            checkpoint.add_step(
                "Highest Score Green + Bold", False, 2, "No valid IMDb scores found"
            )
            checkpoint.add_step(
                "Lowest Score Red + Bold", False, 3, "No valid IMDb scores found"
            )
    else:
        checkpoint.add_step(
            "Highest Score Green + Bold", False, 2, "IMDb Score column not found"
        )
        checkpoint.add_step(
            "Lowest Score Red + Bold", False, 3, "IMDb Score column not found"
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None, cached_models: dict = None, browsing_history: List[str] = None):
    """Grade all checkpoints for the movie recommendation task.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
        cached_models: Dictionary of preloaded models keyed by model_id.
        browsing_history: List of URLs visited during task execution.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    try:
        setup(workspace_doc_id)

        # Use cached model if available
        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        checkpoints: List[Checkpoint] = []

        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()

        # Return a failed result
        failed_checkpoint = Checkpoint(total=1, result=0, name="Evaluation Error")
        failed_checkpoint.add_step("Evaluation", False, 1, f"Fatal error: {str(e)}", execution_time=0)
        return Result([failed_checkpoint], total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sheets_55 Movie Recommendation")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Sheets document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()
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
