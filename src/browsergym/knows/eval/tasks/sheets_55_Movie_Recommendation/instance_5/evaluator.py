"""Evaluator for the Movie Recommendation Google Sheets task."""

import os
import re
import sys
import time
import traceback
import warnings
import argparse
from typing import List
import pandas as pd


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

# Imports from eval_utils
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    get_sheet_content,
    extract_tables_from_sheet,
    parse_sheet_to_dataframe,
    detect_header_row,
)
from src.browsergym.knows.eval.eval_utils.table_utils import (
    match_columns,
    get_cell,
    is_cell_bold,
    cell_bg_hex,
    get_background_color,
    classify_row_color,
)
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute

# Task-specific utilities
from src.browsergym.knows.eval.tasks.sheets_55_Movie_Recommendation.utils import (
    close_browser,
    fetch_imdb_data,
    fetch_imdb_awards_text,
    get_primary_imdb_genre,
    verify_genre,
    verify_imdb_score,
    verify_mpa_rating,
    extract_qualifying_oscars_won,
    parse_duration_to_minutes
)

# Preferred genres from the task description (update per instance)
PREFERRED_GENRES = ["Drama", "Comedy", "Crime"]

# Inclusive year range from the task description
YEAR_RANGE = (1985, 2005)

# Inclusive total-duration range (minutes) from the task description
TOTAL_DURATION_RANGE = (600, 800)

# Minimum number of distinct primary genres required across the sheet
MIN_DISTINCT_GENRES = 3

QUALIFYING_OSCARS = [
    "Best Actor",
    "Best Actress",
    "Best Adapted Screenplay",
    "Best Cinematography",
]

REQUIRED_COLUMNS = [
    ("Movie Title", ["movie", "title", "movie title", "film", "name"]),
    ("Genre", ["genre", "genres", "category"]),
    ("MPA/Age Rating", ["rating", "mpa", "age rating", "movie rating", "mpaa", "certification"]),
    ("IMDb Score", ["imdb", "score", "imdb score", "imdb rating"]),
    ("Release Year", ["year", "release", "release year", "released"]),
    ("Duration", ["duration", "runtime", "length", "time", "run time", "minutes"]),
    ("Oscar Awards Won", ["oscar", "oscars", "awards", "academy", "oscar awards"]),
]


# Google services
DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

model = None
model_id = "gemini-3-flash-google-ai"

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

    # Fallback: parse raw sheet data if no formal tables detected or table detection is empty
    if (df is None or df.empty) and sheet_raw is not None:
        rows = sheet_raw.get('sheets', [{}])[0].get('data', [{}])[0].get('rowData', [])
        detected_header_row = detect_header_row(rows, required_columns=REQUIRED_COLUMNS)
        df = parse_sheet_to_dataframe(sheet_raw, header_row=detected_header_row)


def grade_checkpoint_1():
    """Checkpoint 1 (5 pts): Spreadsheet Structure.

    Steps:
        1. All required labeled columns present.
        2. Columns appear in the required order: Movie Title, Genre,
           MPA/Age Rating, IMDb Score, Release Year, Duration, Oscar Awards Won.
        3. At least 5 movies (data rows).
        4. Each row is a unique movie (no duplicate titles).
        5. No blank cells in required columns.
    """
    global model, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Spreadsheet Structure")

    if df is None or df.empty:
        detail = "No data found in spreadsheet"
        checkpoint.add_step("Required Columns", False, 1, detail, execution_time=0)
        checkpoint.add_step("Columns In Order", False, 2, detail, execution_time=0)
        checkpoint.add_step("At Least 5 Movies", False, 3, detail, execution_time=0)
        checkpoint.add_step("Unique Movies", False, 4, detail, execution_time=0)
        checkpoint.add_step("No Blank Cells", False, 5, detail, execution_time=0)
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

    # Step 2: Columns appear in the required order. 
    step_start = time.time()
    if not matched_columns:
        ordered = False
        order_detail = "No matched columns to verify order"
    else:
        expected_names = [name for name, _ in REQUIRED_COLUMNS]
        actual_cols = list(df.columns)  # NOT matched_columns.keys()
        ordered = True
        order_detail = "Columns in expected order"

        # For each canonical name, find its position in df.columns via the matched name
        positions = []
        for canonical_name in expected_names:
            if canonical_name not in matched_columns:
                ordered = False
                order_detail = f"Cannot verify order: missing column '{canonical_name}'"
                break
            actual_match = matched_columns[canonical_name]
            try:
                positions.append(actual_cols.index(actual_match))
            except ValueError:
                ordered = False
                order_detail = f"Matched column '{actual_match}' not found in sheet"
                break

        if ordered and positions != sorted(positions):
            ordered = False
            expected_str = ", ".join(expected_names)
            matched_set = set(matched_columns.values())
            actual_str = ", ".join(c for c in actual_cols if c in matched_set)
            order_detail = f"Columns out of order. Expected: {expected_str}. Found: {actual_str}"
    checkpoint.add_step(
        "Columns In Order", ordered, 2, order_detail,
        execution_time=time.time() - step_start,
    )

    # Step 3: At least 5 movies
    step_start = time.time()
    num_rows = len(df)
    checkpoint.add_step(
        "At Least 5 Movies", num_rows >= 5, 3,
        f"Found {num_rows} data rows",
        execution_time=time.time() - step_start,
    )

    # Step 4: Unique movies (no duplicates)
    step_start = time.time()
    if "Movie Title" in matched_columns:
        title_col = matched_columns["Movie Title"]
        titles = df[title_col].dropna().astype(str).str.strip().str.lower().str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True)
        duplicates = titles[titles.duplicated()].unique().tolist()
        is_unique = len(duplicates) == 0
        details = "No duplicate titles" if is_unique else f"Duplicates: {', '.join(duplicates)}"
    else:
        is_unique = False
        details = "Movie Title column not found"
    checkpoint.add_step(
        "Unique Movies", is_unique, 4, details,
        execution_time=time.time() - step_start,
    )

    # Step 5: No blank cells in required columns
    step_start = time.time()
    if not matched_columns:
        no_blanks = False
        details = "No required columns matched; cannot check for blanks"
    else:
        blank_details = []
        for expected_name, matched_col in matched_columns.items():
            blanks = df[matched_col].isna().sum() + (df[matched_col].astype(str).str.strip() == "").sum()
            if blanks > 0:
                blank_details.append(f"{expected_name}: {blanks} blank(s)")
        no_blanks = len(blank_details) == 0
        details = "No blank cells" if no_blanks else f"Blanks found: {'; '.join(blank_details)}"
    checkpoint.add_step(
        "No Blank Cells", no_blanks, 5, details,
        execution_time=time.time() - step_start,
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2(browsing_history: List[str] = None):
    """Checkpoint 2 (80 pts): Data Accuracy.

    Fetches structured IMDb data (JSON-LD) via Playwright for each movie.
    Verifies that the agent actually visited each movie's IMDb page (rather
    than relying on parametric memory). Uses programmatic comparison for
    genre, release year, IMDb score, and MPA rating. Uses LLM only for Oscar
    verification (unstructured awards data). Each step scored proportionally
    per movie: round((movies_passing / total_movies) * 10).

    Steps:
        1. Each movie's IMDb page appears in the browsing history (10 pt, proportional).
           Movies whose IMDb page wasn't visited fail every subsequent step too.
        2. Each movie's primary IMDb genre is one of the preferred genres
           (Drama, Comedy, Crime) (10 pt, proportional).
        3. At least 3 distinct primary genres (per IMDb's standard genre list)
            appear among the recommended movies. (10 pt, all or nothing)
        4. Each movie's release year is between 1985 and 2005 (10 pt, proportional).
        5. Oscar Awards Won cell lists exactly the qualifying Oscars won —
           {Best Actor, Best Actress, Best Adapted Screenplay, Best Cinematography}
           (10 pt, proportional).
        6. Each IMDb Score matches actual rating (within +/-0.1) and >= 6.5 (10 pt, proportional).
        7. Each MPA/Age Rating is correct (10 pt, proportional).
        8. Sum of all movie durations is between 600 and 800 minutes, inclusive (10 pt, all or nothing).
    """
    global model
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=80, result=0, name="Data Accuracy")

    if df is None or df.empty or not matched_columns:
        detail = "No data available"
        checkpoint.add_step("IMDb Page Visited", False, 1, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Genre Verification", False, 2, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("At Least 3 Distinct Genres", False, 3, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Release Year In Range", False, 4, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Oscar Verification", False, 5, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("IMDb Score Verification", False, 6, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("MPA Rating Verification", False, 7, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Total Duration In Range", False, 8, detail, score=0, max_score=10, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    title_col = matched_columns.get("Movie Title")
    if not title_col:
        detail = "Movie Title column not found"
        checkpoint.add_step("IMDb Page Visited", False, 1, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Genre Verification", False, 2, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("At Least 3 Distinct Genres", False, 3, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Release Year In Range", False, 4, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Oscar Verification", False, 5, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("IMDb Score Verification", False, 6, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("MPA Rating Verification", False, 7, detail, score=0, max_score=10, execution_time=0)
        checkpoint.add_step("Total Duration In Range", False, 8, detail, score=0, max_score=10, execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_movies = len(df)

    year_col = matched_columns.get("Release Year")
    imdb_data_map = {}
    imdb_id_map = {}
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

    missing = [m for m, d in imdb_data_map.items() if d is None]
    if missing:
        warnings.warn(
            f"IMDb data missing for {len(missing)}/{num_movies} movies: {missing}. "
            "Cause may be agent (fabricated title/year) or evaluator-side (network, "
            "search miss, IMDb HTML change); check logs above. Affected movies are "
            "counted as failures in all Checkpoint 2 steps.",
            stacklevel=2,
        )

    # Step 1: IMDb page visit gate
    step_start = time.time()
    visited_movies = set()
    history = browsing_history or []
    for movie, imdb_id in imdb_id_map.items():
        if imdb_id and any(f"/title/{imdb_id}" in url for url in history):
            visited_movies.add(movie)
    visit_failures = [m for m in imdb_id_map if m not in visited_movies]
    visit_pass = num_movies - len(visit_failures)
    visit_score = calculate_percentage_score(visit_pass, num_movies, max_points=10)
    details = f"{visit_pass}/{num_movies} movies have their IMDb page in browsing history"
    if visit_failures:
        details += f". Not visited: {', '.join(visit_failures)}"
    checkpoint.add_step(
        "IMDb Page Visited", visit_pass == num_movies, 1, details,
        score=visit_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 2: Genre verification (LLM — parallelizable)
    step_start = time.time()
    genre_failures = []
    genre_not_found = []

    if model is None:
        model = load_model(model_id)

    # Pre-flight: queue movies for parallel LLM genre verification
    genre_tasks = []
    for idx, row in df.iterrows():
        movie = str(row[title_col]).strip()
        genre_tasks.append({
            'id': movie,
            'func': verify_genre,
            'args': (imdb_data_map.get(movie), PREFERRED_GENRES),
        })

    genre_results = parallel_execute(genre_tasks, max_workers=4) if genre_tasks else {}

    for movie, ok in genre_results.items():
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
        "Genre Verification", genre_pass == num_movies, 2, details,
        score=genre_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 3: At least 3 distinct primary IMDb genres across the sheet.
    # Uses the first-listed (primary) genre per movie from IMDb's standard
    # genre list; they must be in PREFERRED_GENRES;
    # movies missing IMDb data contribute nothing.
    step_start = time.time()
    primary_genres = set()
    distinct_unknown = []
    for movie, data in imdb_data_map.items():
        if data is None:
            distinct_unknown.append(movie)
            continue
        primary = get_primary_imdb_genre(data)
        if primary and any(re.search(rf"\b{re.escape(pref.lower())}\b", primary) for pref in PREFERRED_GENRES):
            primary_genres.add(primary)
    distinct_ok = len(primary_genres) >= MIN_DISTINCT_GENRES
    distinct_score = 10 if distinct_ok else 0
    distinct_detail = (
        f"Found {len(primary_genres)} distinct primary IMDb genre(s): "
        f"{sorted(primary_genres)} (target: >= {MIN_DISTINCT_GENRES})"
    )
    if distinct_unknown:
        distinct_detail += f". Could not retrieve IMDb data: {', '.join(distinct_unknown)}"
    checkpoint.add_step(
        "At Least 3 Distinct Genres", distinct_ok, 3, distinct_detail,
        score=distinct_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 4: Release year in range.
    step_start = time.time()
    year_low, year_high = YEAR_RANGE
    year_failures = []
    year_not_found = []
    for idx, row in df.iterrows():
        movie = str(row[title_col]).strip()
        data = imdb_data_map.get(movie)
        if data is None:
            year_not_found.append(movie)
            continue
        date_published = (data.get("datePublished", "") or "")[:4]
        try:
            actual_year = int(date_published)
        except ValueError:
            year_not_found.append(f"{movie} (could not parse year from '{date_published}')")
            continue
        if not (year_low <= actual_year <= year_high):
            year_failures.append(f"{movie} ({actual_year} not in {year_low}-{year_high})")
    year_pass = num_movies - len(year_failures) - len(year_not_found)
    year_score = calculate_percentage_score(year_pass, num_movies, max_points=10)
    details = f"{year_pass}/{num_movies} movies released in {year_low}-{year_high}"
    if year_not_found:
        details += f". Could not retrieve IMDb data: {', '.join(year_not_found)}"
    if year_failures:
        details += f". Failed: {', '.join(year_failures)}"
    checkpoint.add_step(
        "Release Year In Range", year_pass == num_movies, 4, details,
        score=year_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 5: Oscar verification — cell must list exactly the qualifying Oscars
    # the movie actually won (no extras, no omissions, at least one).
    step_start = time.time()
    oscar_col = matched_columns.get("Oscar Awards Won")
    canonical_oscars = {o.lower(): o for o in QUALIFYING_OSCARS}
    oscar_failures = []
    oscar_not_found = []
    if oscar_col:
        # Pre-flight: identify which movies need awards-text fetch + LLM call.
        # Skip no-data or blank-cell movies upfront.
        movies_to_check = []  # list of (movie, imdb_id, cell_text)
        for idx, row in df.iterrows():
            movie = str(row[title_col]).strip()
            if imdb_data_map.get(movie) is None:
                oscar_not_found.append(movie)
                continue
            cell = str(row[oscar_col]).strip()
            if not cell or cell.lower() == "nan":
                oscar_failures.append(f"{movie} (Oscar Awards Won cell is blank)")
                continue
            movies_to_check.append((movie, imdb_id_map.get(movie), cell))

        # Extract Oscar wins programmatically (sequential — Playwright isn't thread-safe)
        extracted = {}
        for movie, imdb_id, cell in movies_to_check:
            result = extract_qualifying_oscars_won(
                model, movie, QUALIFYING_OSCARS,
                awards_text=None, imdb_id=imdb_id,
            )
            if result is None:
                oscar_not_found.append(f"{movie} (could not parse awards page)")
            else:
                extracted[movie] = result

        # Compare each movie's extracted Oscars to its sheet cell
        for movie, _, cell in movies_to_check:
            if movie not in extracted:
                continue  # already accounted for in oscar_not_found
            actual = extracted[movie]

            # Parse the cell into recognized qualifying Oscars + unrecognized extras.
            listed = set()
            extras = []
            for piece in cell.split(","):
                p = piece.strip()
                if not p:
                    continue
                key = p.lower().rstrip(".")
                if key in canonical_oscars:
                    listed.add(canonical_oscars[key])
                else:
                    extras.append(p)

            if not actual:
                oscar_failures.append(f"{movie} (won no qualifying Oscars)")
            elif extras:
                oscar_failures.append(f"{movie} (non-qualifying entries: {extras})")
            elif listed != actual:
                oscar_failures.append(
                    f"{movie} (sheet={sorted(listed)}, actual={sorted(actual)})"
                )
        oscar_pass = num_movies - len(oscar_failures) - len(oscar_not_found)
        oscar_score = calculate_percentage_score(oscar_pass, num_movies, max_points=10)
        details = f"{oscar_pass}/{num_movies} movies have correct Oscar listings"
        if oscar_not_found:
            details += f". Could not retrieve awards data: {', '.join(oscar_not_found)}"
        if oscar_failures:
            details += f". Failed: {', '.join(oscar_failures)}"
    else:
        oscar_pass = 0
        oscar_score = 0
        details = "Oscar Awards Won column not found"
    checkpoint.add_step(
        "Oscar Verification", oscar_pass == num_movies, 5, details,
        score=oscar_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 6: IMDb Score verification (programmatic — no LLM)
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
        "IMDb Score Verification", score_pass == num_movies, 6, details,
        score=score_step, max_score=10,
        execution_time=time.time() - step_start,
    )

    # Step 7: MPA Rating verification (programmatic — no LLM)
    step_start = time.time()
    rating_col = matched_columns.get("MPA/Age Rating")
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
        details = "MPA/Age Rating column not found"
    checkpoint.add_step(
        "MPA Rating Verification", rating_pass == num_movies, 7, details,
        score=rating_score, max_score=10,
        execution_time=time.time() - step_start,
    )
    
    # Step 8: Total duration in range. Uses IMDb's actual runtime
    # (JSON-LD `duration` field, ISO 8601 like "PT2H22M") rather than the
    # sheet's Duration column so the agent can't fudge the total.
    step_start = time.time()
    low, high = TOTAL_DURATION_RANGE
    iso_re = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?$")
    minutes_per_movie = {}
    duration_unknown = []
    for movie, data in imdb_data_map.items():
        if data is None:
            duration_unknown.append(movie)
            continue
        raw = (data.get("duration") or "").strip()
        m = iso_re.match(raw)
        if not m or (m.group(1) is None and m.group(2) is None):
            duration_unknown.append(f"{movie} (could not parse IMDb duration '{raw}')")
            continue
        h = int(m.group(1) or 0)
        mi = int(m.group(2) or 0)
        minutes_per_movie[movie] = h * 60 + mi
    if duration_unknown:
        duration_ok = False
        total = sum(minutes_per_movie.values())
        duration_detail = (
            f"Could not retrieve IMDb duration for: {', '.join(duration_unknown)}. "
            f"Partial total (verified movies only) = {total} min"
        )
    else:
        total = sum(minutes_per_movie.values())
        duration_ok = low <= total <= high
        duration_detail = f"Total IMDb duration = {total} min (target: {low}-{high})"
    duration_score = 10 if duration_ok else 0
    checkpoint.add_step(
        "Total Duration In Range", duration_ok, 8, duration_detail,
        score=duration_score, max_score=10,
        execution_time=time.time() - step_start,
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """Checkpoint 3 (3 pts): Sorting and Conditional Formatting.

    Steps:
        1. Movie list sorted by Duration in descending order.
        2. Highest IMDb Score cell(s) have green fill + bold text.
        3. Lowest IMDb Score cell(s) have red fill + bold text.
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Sorting and Conditional Formatting")

    if df is None or df.empty or not matched_columns or not sheet_raw:
        detail = "No data available"
        checkpoint.add_step("Sorted by Duration", False, 1, detail, execution_time=0)
        checkpoint.add_step("Highest Score Green + Bold", False, 2, detail, execution_time=0)
        checkpoint.add_step("Lowest Score Red + Bold", False, 3, detail, execution_time=0)
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
            is_desc = all(valid[i] >= valid[i + 1] for i in range(len(valid) - 1))
            sorted_ok = is_desc
            details = "Duration sorted (descending)" if sorted_ok else f"Not sorted descending: {valid[:5]}..."
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
    if score_col and (table_data or sheet_raw):
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
            try:
                if table_data:
                    table = table_data[0]
                    header_row_idx = table.start_row
                    start_col = table.start_col
                else:
                    rows = sheet_raw.get('sheets', [{}])[0].get('data', [{}])[0].get('rowData', [])
                    header_row_idx = detect_header_row(rows, required_columns=REQUIRED_COLUMNS)
                    start_col = 0
                sheet_tab = sheet_raw["sheets"][0]
                score_col_idx = None
                for i, col in enumerate(df.columns):
                    if col == matched_columns["IMDb Score"]:
                        score_col_idx = start_col + i
                        break
                structure_error = None
            except (AttributeError, KeyError, IndexError, TypeError) as e:
                header_row_idx = sheet_tab = score_col_idx = None
                structure_error = f"Could not read sheet structure: {e}"

            # Step 2: Highest IMDb Score -> green fill + bold
            step_start = time.time()
            if structure_error:
                highest_ok = False
                details = structure_error
            elif score_col_idx is not None:
                highest_ok = True
                highest_issues = []
                for row_i, score in enumerate(scores):
                    if score == max_score:
                        raw_row = header_row_idx + 1 + row_i
                        try:
                            cell = get_cell(sheet_tab, raw_row, score_col_idx)
                            bold = is_cell_bold(cell)
                            bg = get_background_color(sheet_raw, raw_row, score_col_idx)
                            color = classify_row_color(bg)
                        except Exception as e:
                            highest_ok = False
                            highest_issues.append(f"Row {row_i + 1}: cell read error ({e})")
                            continue
                        if not bold or color != "green":
                            highest_ok = False
                            try:
                                hex_val = cell_bg_hex(sheet_raw, raw_row, score_col_idx)
                            except Exception:
                                hex_val = "?"
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
            if structure_error:
                lowest_ok = False
                details = structure_error
            elif score_col_idx is not None:
                lowest_ok = True
                lowest_issues = []
                for row_i, score in enumerate(scores):
                    if score == min_score:
                        raw_row = header_row_idx + 1 + row_i
                        try:
                            cell = get_cell(sheet_tab, raw_row, score_col_idx)
                            bold = is_cell_bold(cell)
                            bg = get_background_color(sheet_raw, raw_row, score_col_idx)
                            color = classify_row_color(bg)
                        except Exception as e:
                            lowest_ok = False
                            lowest_issues.append(f"Row {row_i + 1}: cell read error ({e})")
                            continue
                        if not bold or color != "red":
                            lowest_ok = False
                            try:
                                hex_val = cell_bg_hex(sheet_raw, raw_row, score_col_idx)
                            except Exception:
                                hex_val = "?"
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
        checkpoints.append(grade_checkpoint_2(browsing_history))
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

    finally:
        # Always close the Playwright browser process even on exceptions or early returns
        try:
            close_browser()
        except Exception:
            pass


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
