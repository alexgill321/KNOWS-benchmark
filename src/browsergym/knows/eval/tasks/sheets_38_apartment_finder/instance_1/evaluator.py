"""Evaluator for the Apartment Finder Google Sheets task."""

import os
import sys
from typing import List, Dict, Optional, Any
import time
import pandas as pd
import argparse
import re

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
from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import (
    extract_tables_from_sheet,
    extract_sheet_data,
    get_sheet_content,
)
from src.browsergym.knows.eval.eval_utils.table_utils import match_columns, is_text_visible_in_cell
from src.browsergym.knows.eval.eval_utils.text_utils import numerical_match_with_error
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download, parallel_execute

# Local utils
from src.browsergym.knows.eval.tasks.sheets_38_apartment_finder.utils import (
    fetch_craigslist_page,
    extract_craigslist_data_with_fallback,
    normalize_boolean_value,
    compare_addresses,
    is_valid_craigslist_url
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/sheets_38_apartment_finder/instance_1/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

model = None
model_id = "gemini-2.5-flash-google-ai"

try:
    DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")
    if SHEETS_SERVICE is None:
        raise RuntimeError(
            "Google Sheets service returned None. "
            "Check service account credentials and ensure the Sheets API is enabled."
        )
except Exception as e:
    raise RuntimeError(
        f"FATAL: Failed to initialize Google services. "
        f"The evaluator cannot run without API access. Error: {e}"
    ) from e

# Global variables
sheet_id = None
table_data = None
sheet_raw = None
df = None
matched_columns = {}


def setup(workspace_doc_id: str):
    """
    Setup function to initialize the evaluator.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
    """
    global sheet_id, table_data, sheet_raw, df

    try:
        if workspace_doc_id:
            print(f"Using workspace document ID: {workspace_doc_id}")
            sheet_id = workspace_doc_id

        # Extract data from the spreadsheet
        table_data = extract_tables_from_sheet(sheet_id, SHEETS_SERVICE)
        sheet_raw = get_sheet_content(sheet_id, SHEETS_SERVICE)

        # Validate extracted table data
        if table_data is not None:
            if not isinstance(table_data, list):
                print(f"WARNING: extract_tables_from_sheet returned {type(table_data)}, expected list")
                table_data = None
            elif table_data:
                first = table_data[0]
                test_df = first.df if hasattr(first, 'df') else first
                if isinstance(test_df, dict):
                    test_df = pd.DataFrame(test_df)
                if not isinstance(test_df, pd.DataFrame) or test_df.empty:
                    print("WARNING: First table has invalid/empty DataFrame")
                    table_data = None

        # Pick the table whose columns best match listing data keywords
        if table_data:
            listing_keywords = ["address", "location", "price", "rent", "bed", "bedroom",
                                "bath", "bathroom", "sq ft", "sqft", "url", "link", "listing"]
            best_table = None
            best_hits = -1
            for t in table_data:
                t_df = t.df if hasattr(t, 'df') else t
                if isinstance(t_df, dict):
                    t_df = pd.DataFrame(t_df)
                if not isinstance(t_df, pd.DataFrame) or t_df.empty:
                    continue
                hits = sum(1 for col in t_df.columns
                           for kw in listing_keywords
                           if kw in str(col).lower())
                if hits > best_hits:
                    best_hits = hits
                    best_table = t_df
            df = best_table if best_table is not None else None

    except Exception as e:
        print(f"WARNING: setup() failed: {e}. Globals set to None for graceful degradation.")
        import traceback
        traceback.print_exc()
        table_data = None
        sheet_raw = None
        df = None


def grade_checkpoint_1():
    """
    Checkpoint 1: Spreadsheet Structure (10 pts)
    Validates that the spreadsheet contains columns for all required features
    and at least 5 listings.

    Outcome Evaluation:
    - There is a column for the listing address.
    - There is a column for the monthly rent/price.
    - There is a column for the number of bedrooms.
    - There is a column for the number of bathrooms.
    - There is a column for square footage.
    - There is a column indicating in-unit laundry availability.
    - There is a column indicating pet-friendliness.
    - There is a column containing a link/URL to each listing.
    - There is a column for interesting positive features.
    - There is a column for potential dealbreakers.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    global model, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Spreadsheet Structure")

    required_columns = [
        ("Address", ["address", "location", "property", "street"]),
        ("Price/Rent", ["price", "rent", "monthly", "cost", "$/month", "per month"]),
        ("Bedrooms", ["bed", "bedroom", "br", "beds"]),
        ("Bathrooms", ["bath", "bathroom", "ba", "baths"]),
        ("Square Footage", ["sq ft", "sqft", "square", "size", "sq. ft", "square feet"]),
        ("In-Unit Laundry", ["laundry", "washer", "dryer", "w/d", "in-unit"]),
        ("Pet Friendly", ["pet", "pets", "dog", "cat", "animal"]),
        ("Listing URL", ["url", "link", "craigslist", "listing"]),
        ("Positive Features", ["positive", "pros", "features", "highlights", "amenities", "interesting"]),
        ("Dealbreakers", ["dealbreaker", "cons", "negatives", "issues", "concerns"])
    ]

    if not table_data or df is None or df.empty:
        for step_num, col in enumerate(required_columns, start=1):
            checkpoint.add_step(f"{col[0]} Column", False, step_num,
                              "No table data found in spreadsheet",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    original_columns = [str(col) for col in df.columns]

    # Use standardized match_columns() - keyword matching first, then LLM fallback
    if model is None:
        try:
            model = load_model(model_id)
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Failed to load model '{model_id}'. "
                f"Ensure model ID is correct and API keys are configured. Error: {e}"
            ) from e
    try:
        matched_columns = match_columns(df, required_columns, model=model, parallel=True,
                                        context="an apartment listing spreadsheet with columns for property details like address, price, bedrooms, bathrooms, square footage, amenities, and listing URLs from Craigslist")
    except Exception as e:
        print(f"WARNING: match_columns failed: {e}. Setting matched_columns to None.")
        matched_columns = None

    if matched_columns is None:
        for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
            checkpoint.add_step(f"{col_name} Column", False, step_num,
                              "Column matching failed - unable to identify columns",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Add checkpoint steps for each required column
    for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
        step_start = time.time()
        if col_name in matched_columns:
            matched_column = matched_columns[col_name]
            checkpoint.add_step(f"{col_name} Column", True, step_num,
                              f"Found column matching '{col_name}': '{matched_column}'",
                              execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"{col_name} Column", False, step_num,
                              f"No column found for '{col_name}'. Available: {', '.join(original_columns[:5])}...",
                              execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2: Listing Data Accuracy (35 pts = 5 listings × 7 pts each)
    Each listing's data is verified against the actual Craigslist listing page.

    Outcome Evaluation (repeated for each of 5 listings):
    - The listing URL is valid and accessible.
    - The price matches the extracted value (within 5% tolerance).
    - The bedroom count matches.
    - The bathroom count matches.
    - The address matches or is contained in the extracted address.
    - The in-unit laundry status matches.
    - The pet-friendly status matches.

    PARALLELIZED: Phase 1 fetches all URLs in parallel, Phase 2 extracts data in parallel.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    global model, matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=35, result=0, name="Listing Data Accuracy")

    step_names = [
        "URL Valid",
        "Price Match",
        "Bedrooms Match",
        "Bathrooms Match",
        "Address Match",
        "Laundry Match",
        "Pet-Friendly Match",
    ]

    if df is None or df.empty:
        step_id = 0
        for listing_num in range(1, 6):
            for step_name in step_names:
                step_id += 1
                checkpoint.add_step(f"Listing {listing_num} - {step_name}", False, step_id,
                                  "No listing data found in spreadsheet", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    if model is None:
        try:
            model = load_model(model_id)
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Failed to load model '{model_id}'. "
                f"Ensure model ID is correct and API keys are configured. Error: {e}"
            ) from e

    if matched_columns is None:
        step_id = 0
        for listing_num in range(1, 6):
            for step_name in step_names:
                step_id += 1
                checkpoint.add_step(f"Listing {listing_num} - {step_name}", False, step_id,
                                  "Column matching failed - cannot verify listings", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    url_col = matched_columns.get("Listing URL")
    if not url_col:
        step_id = 0
        for listing_num in range(1, 6):
            for step_name in step_names:
                step_id += 1
                checkpoint.add_step(f"Listing {listing_num} - {step_name}", False, step_id,
                                  "No URL column identified in spreadsheet", execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get other matched columns
    price_col = matched_columns.get("Price/Rent")
    bed_col = matched_columns.get("Bedrooms")
    bath_col = matched_columns.get("Bathrooms")
    addr_col = matched_columns.get("Address")
    laundry_col = matched_columns.get("In-Unit Laundry")
    pet_col = matched_columns.get("Pet Friendly")

    # Process up to 5 listings
    listings_to_check = min(5, len(df))

    # ============ PHASE 1: Parallel URL fetching ============
    print(f"  Phase 1: Fetching {listings_to_check} Craigslist pages in parallel...")
    fetch_start = time.time()

    # Build fetch tasks for valid URLs
    fetch_tasks = []
    listing_urls = {}  # listing_idx -> url
    invalid_urls = {}  # listing_idx -> reason

    for listing_idx in range(listings_to_check):
        row = df.iloc[listing_idx]
        url = str(row.get(url_col, "")) if url_col else ""
        url = url.strip()

        if not url or not is_valid_craigslist_url(url):
            invalid_urls[listing_idx] = f"Invalid or missing URL: {url[:50]}..."
        else:
            listing_urls[listing_idx] = url
            fetch_tasks.append({
                'id': f'listing_{listing_idx}',
                'func': fetch_craigslist_page,
                'args': (url, True)  # raw=True for structured parsing
            })

    # Fetch all URLs in parallel
    html_contents = {}
    if fetch_tasks:
        try:
            fetch_results = parallel_download(fetch_tasks, max_workers=5, use_rate_limit=False)
        except Exception as e:
            print(f"WARNING: parallel_download failed ({e}). Falling back to sequential.")
            fetch_results = {}
            for task in fetch_tasks:
                try:
                    fetch_results[task['id']] = task['func'](*task.get('args', ()))
                except Exception as inner_e:
                    print(f"  Sequential fetch failed for {task['id']}: {inner_e}")
                    fetch_results[task['id']] = None
        for task_id, html in fetch_results.items():
            listing_idx = int(task_id.split('_')[1])
            html_contents[listing_idx] = html

    fetch_time = time.time() - fetch_start
    print(f"  Phase 1 complete: {len(html_contents)} pages fetched in {fetch_time:.2f}s")

    # ============ PHASE 2: Structured extraction + LLM fallback ============
    # Fields this instance needs from extraction
    required_fields = ["price", "bedrooms", "bathrooms", "address", "in_unit_laundry", "pet_friendly"]

    print(f"  Phase 2: Extracting data from {len(html_contents)} pages (structured first, LLM fallback)...")
    extract_start = time.time()

    extraction_tasks = []
    for listing_idx, html in html_contents.items():
        if html:
            extraction_tasks.append({
                'id': f'listing_{listing_idx}',
                'func': extract_craigslist_data_with_fallback,
                'args': (html, model, required_fields)
            })

    extracted_data_map = {}
    if extraction_tasks:
        try:
            extraction_results = parallel_execute(extraction_tasks, max_workers=5)
        except Exception as e:
            print(f"WARNING: parallel_execute failed ({e}). Falling back to sequential.")
            extraction_results = {}
            for task in extraction_tasks:
                try:
                    extraction_results[task['id']] = task['func'](*task.get('args', ()))
                except Exception as inner_e:
                    print(f"  Sequential extraction failed for {task['id']}: {inner_e}")
                    extraction_results[task['id']] = None
        for task_id, data in extraction_results.items():
            listing_idx = int(task_id.split('_')[1])
            extracted_data_map[listing_idx] = data

    extract_time = time.time() - extract_start
    print(f"  Phase 2 complete: {len(extracted_data_map)} extractions in {extract_time:.2f}s")

    # ============ PHASE 3: Sequential validation (fast, no I/O) ============
    step_num = 0

    for listing_idx in range(listings_to_check):
        row = df.iloc[listing_idx]
        listing_num = listing_idx + 1

        # Step 1: URL is valid and accessible
        step_num += 1

        # Check for invalid URL
        if listing_idx in invalid_urls:
            checkpoint.add_step(f"Listing {listing_num} - URL Valid", False, step_num,
                              invalid_urls[listing_idx],
                              execution_time=0)
            for step_name in step_names[1:]:
                step_num += 1
                checkpoint.add_step(f"Listing {listing_num} - {step_name}", False, step_num,
                                  "Skipped due to invalid URL",
                                  execution_time=0)
            continue

        # Check if fetch failed
        html_content = html_contents.get(listing_idx)
        url = listing_urls.get(listing_idx, "")

        if not html_content:
            checkpoint.add_step(f"Listing {listing_num} - URL Valid", False, step_num,
                              f"Could not fetch page: {url[:50]}...",
                              execution_time=0)
            for step_name in step_names[1:]:
                step_num += 1
                checkpoint.add_step(f"Listing {listing_num} - {step_name}", False, step_num,
                                  "Skipped due to page fetch failure",
                                  execution_time=0)
            continue

        checkpoint.add_step(f"Listing {listing_num} - URL Valid", True, step_num,
                          f"Successfully fetched page: {url[:50]}...",
                          execution_time=0)

        # Check if extraction failed
        extracted_data = extracted_data_map.get(listing_idx)
        if not extracted_data:
            for step_name in step_names[1:]:
                step_num += 1
                checkpoint.add_step(f"Listing {listing_num} - {step_name}", False, step_num,
                                  "Could not extract data from Craigslist page",
                                  execution_time=0)
            continue

        # Step 2: Price matches
        step_num += 1
        if not price_col:
            checkpoint.add_step(f"Listing {listing_num} - Price Match", False, step_num,
                              "No 'Price/Rent' column found in spreadsheet",
                              execution_time=0)
        else:
            try:
                user_price = float(re.sub(r'[^\d.]', '', str(row.get(price_col, 0))))
                craigslist_price = extracted_data.get("price")

                if craigslist_price and user_price:
                    is_match, diff = numerical_match_with_error(craigslist_price, user_price, error_percent=5.0)
                    if is_match:
                        checkpoint.add_step(f"Listing {listing_num} - Price Match", True, step_num,
                                          f"Price ${user_price:.0f} matches Craigslist ${craigslist_price:.0f}",
                                          execution_time=0)
                    else:
                        checkpoint.add_step(f"Listing {listing_num} - Price Match", False, step_num,
                                          f"Price mismatch: user ${user_price:.0f} vs Craigslist ${craigslist_price:.0f} ({diff:.1f}% diff)",
                                          execution_time=0)
                else:
                    checkpoint.add_step(f"Listing {listing_num} - Price Match", False, step_num,
                                      f"Missing price data (user: {user_price}, craigslist: {craigslist_price})",
                                      execution_time=0)
            except Exception as e:
                checkpoint.add_step(f"Listing {listing_num} - Price Match", False, step_num,
                                  f"Error comparing prices: {str(e)[:50]}",
                                  execution_time=0)

        # Step 3: Bedroom count matches
        step_num += 1
        if not bed_col:
            checkpoint.add_step(f"Listing {listing_num} - Bedrooms Match", False, step_num,
                              "No 'Bedrooms' column found in spreadsheet",
                              execution_time=0)
        else:
            try:
                user_beds = float(re.sub(r'[^\d.]', '', str(row.get(bed_col, 0))))
                craigslist_beds = extracted_data.get("bedrooms")

                if craigslist_beds is not None:
                    if user_beds == craigslist_beds:
                        checkpoint.add_step(f"Listing {listing_num} - Bedrooms Match", True, step_num,
                                          f"Bedrooms match: {int(user_beds)}",
                                          execution_time=0)
                    else:
                        checkpoint.add_step(f"Listing {listing_num} - Bedrooms Match", False, step_num,
                                          f"Bedroom mismatch: user {user_beds} vs Craigslist {craigslist_beds}",
                                          execution_time=0)
                else:
                    checkpoint.add_step(f"Listing {listing_num} - Bedrooms Match", False, step_num,
                                      "Could not extract bedroom count from Craigslist",
                                      execution_time=0)
            except Exception as e:
                checkpoint.add_step(f"Listing {listing_num} - Bedrooms Match", False, step_num,
                                  f"Error comparing bedrooms: {str(e)[:50]}",
                                  execution_time=0)

        # Step 4: Bathroom count matches
        step_num += 1
        if not bath_col:
            checkpoint.add_step(f"Listing {listing_num} - Bathrooms Match", False, step_num,
                              "No 'Bathrooms' column found in spreadsheet",
                              execution_time=0)
        else:
            try:
                user_baths = float(re.sub(r'[^\d.]', '', str(row.get(bath_col, 0))))
                craigslist_baths = extracted_data.get("bathrooms")

                if craigslist_baths is not None:
                    if user_baths == craigslist_baths:
                        checkpoint.add_step(f"Listing {listing_num} - Bathrooms Match", True, step_num,
                                          f"Bathrooms match: {user_baths}",
                                          execution_time=0)
                    else:
                        checkpoint.add_step(f"Listing {listing_num} - Bathrooms Match", False, step_num,
                                          f"Bathroom mismatch: user {user_baths} vs Craigslist {craigslist_baths}",
                                          execution_time=0)
                else:
                    checkpoint.add_step(f"Listing {listing_num} - Bathrooms Match", False, step_num,
                                      "Could not extract bathroom count from Craigslist",
                                      execution_time=0)
            except Exception as e:
                checkpoint.add_step(f"Listing {listing_num} - Bathrooms Match", False, step_num,
                                  f"Error comparing bathrooms: {str(e)[:50]}",
                                  execution_time=0)

        # Step 5: Address matches
        step_num += 1
        if not addr_col:
            checkpoint.add_step(f"Listing {listing_num} - Address Match", False, step_num,
                              "No 'Address' column found in spreadsheet",
                              execution_time=0)
        else:
            try:
                user_addr = str(row.get(addr_col, ""))
                craigslist_addr = extracted_data.get("address", "")

                if user_addr and craigslist_addr and compare_addresses(user_addr, craigslist_addr, model=model):
                    checkpoint.add_step(f"Listing {listing_num} - Address Match", True, step_num,
                                      f"Address matches: {user_addr[:40]}...",
                                      execution_time=0)
                else:
                    checkpoint.add_step(f"Listing {listing_num} - Address Match", False, step_num,
                                      f"Address mismatch: '{user_addr[:30]}' vs '{str(craigslist_addr)[:30]}'",
                                      execution_time=0)
            except Exception as e:
                checkpoint.add_step(f"Listing {listing_num} - Address Match", False, step_num,
                                  f"Error comparing addresses: {str(e)[:50]}",
                                  execution_time=0)

        # Step 6: In-unit laundry status matches
        step_num += 1
        if not laundry_col:
            checkpoint.add_step(f"Listing {listing_num} - Laundry Match", False, step_num,
                              "No 'In-Unit Laundry' column found in spreadsheet",
                              execution_time=0)
        else:
            try:
                user_laundry = normalize_boolean_value(str(row.get(laundry_col, "")))
                craigslist_laundry_str = extracted_data.get("in_unit_laundry", "Unknown")
                craigslist_laundry = normalize_boolean_value(craigslist_laundry_str)

                # Unknown is acceptable if user also has unknown or if Craigslist doesn't specify
                if user_laundry == craigslist_laundry:
                    status = "Yes" if user_laundry else "No"
                    checkpoint.add_step(f"Listing {listing_num} - Laundry Match", True, step_num,
                                      f"In-unit laundry: {status}",
                                      execution_time=0)
                elif craigslist_laundry is None:
                    checkpoint.add_step(f"Listing {listing_num} - Laundry Match", True, step_num,
                                      f"Craigslist laundry status unclear, skipping check",
                                      execution_time=0)
                else:
                    user_status = "Yes" if user_laundry else "No"
                    cl_status = "Yes" if craigslist_laundry else "No"
                    checkpoint.add_step(f"Listing {listing_num} - Laundry Match", False, step_num,
                                      f"Laundry mismatch: spreadsheet says {user_status}, Craigslist says {cl_status}",
                                      execution_time=0)
            except Exception as e:
                checkpoint.add_step(f"Listing {listing_num} - Laundry Match", False, step_num,
                                  f"Error comparing laundry: {str(e)[:50]}",
                                  execution_time=0)

        # Step 7: Pet-friendly status matches
        step_num += 1
        if not pet_col:
            checkpoint.add_step(f"Listing {listing_num} - Pet-Friendly Match", False, step_num,
                              "No 'Pet Friendly' column found in spreadsheet",
                              execution_time=0)
        else:
            try:
                user_pet = normalize_boolean_value(str(row.get(pet_col, "")))
                craigslist_pet_str = extracted_data.get("pet_friendly", "Unknown")
                craigslist_pet = normalize_boolean_value(craigslist_pet_str)

                # Unknown is acceptable
                if user_pet == craigslist_pet:
                    status = "Yes" if user_pet else "No"
                    checkpoint.add_step(f"Listing {listing_num} - Pet-Friendly Match", True, step_num,
                                      f"Pet-friendly: {status}",
                                      execution_time=0)
                elif craigslist_pet is None:
                    checkpoint.add_step(f"Listing {listing_num} - Pet-Friendly Match", True, step_num,
                                      f"Craigslist pet status unclear, skipping check",
                                      execution_time=0)
                else:
                    user_status = "Yes" if user_pet else "No"
                    cl_status = "Yes" if craigslist_pet else "No"
                    checkpoint.add_step(f"Listing {listing_num} - Pet-Friendly Match", False, step_num,
                                      f"Pet mismatch: spreadsheet says {user_status}, Craigslist says {cl_status}",
                                      execution_time=0)
            except Exception as e:
                checkpoint.add_step(f"Listing {listing_num} - Pet-Friendly Match", False, step_num,
                                  f"Error comparing pet status: {str(e)[:50]}",
                                  execution_time=0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3(browsing_history: Optional[List[str]] = None):
    """
    Checkpoint 3: Website Visit Validation (2 pts)
    Validates that the agent visited Craigslist to find the listings.

    Outcome Evaluation:
    - The browsing history contains a visit to craigslist.org.
    - The listing URLs in the spreadsheet appear in the browsing history.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global matched_columns, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Craigslist Visited", False, 1,
                          "No browsing history provided",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.add_step("Listings in History", False, 2,
                          "No browsing history to check",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Check if craigslist.org was visited
    step_start = time.time()
    craigslist_visits = [url for url in browsing_history if 'craigslist.org' in url.lower()]
    step_time = time.time() - step_start

    if craigslist_visits:
        checkpoint.add_step("Craigslist Visited", True, 1,
                          f"Found {len(craigslist_visits)} Craigslist page visits",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Craigslist Visited", False, 1,
                          "No visits to craigslist.org found in browsing history",
                          execution_time=step_time)

    # Step 2: Check if listing URLs appear in browsing history
    step_start = time.time()

    if df is None or df.empty:
        step_time = time.time() - step_start
        checkpoint.add_step("Listings in History", False, 2,
                          "No listing data to compare",
                          execution_time=step_time)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    url_col = matched_columns.get("Listing URL") if matched_columns else None
    if not url_col:
        step_time = time.time() - step_start
        checkpoint.add_step("Listings in History", False, 2,
                          "No URL column identified",
                          execution_time=step_time)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get listing URLs from spreadsheet
    listing_urls = [str(row.get(url_col, "")).strip() for _, row in df.iterrows() if row.get(url_col)]
    listing_urls = [url for url in listing_urls if url and 'craigslist.org' in url.lower()]

    if not listing_urls:
        step_time = time.time() - step_start
        checkpoint.add_step("Listings in History", False, 2,
                          "No valid Craigslist URLs found in spreadsheet",
                          execution_time=step_time)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Check how many listing URLs appear in browsing history
    browsing_set = set(browsing_history)
    urls_found = 0

    for listing_url in listing_urls:
        # Check for exact match or substring match
        if listing_url in browsing_set:
            urls_found += 1
        else:
            # Check substring match (URL might have tracking params)
            for history_url in browsing_set:
                if listing_url in history_url or history_url in listing_url:
                    urls_found += 1
                    break

    step_time = time.time() - step_start

    if urls_found >= len(listing_urls):
        checkpoint.add_step("Listings in History", True, 2,
                          f"All {len(listing_urls)} listing URLs found in browsing history",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Listings in History", False, 2,
                          f"Only {urls_found}/{len(listing_urls)} listing URLs found in browsing history",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4: Conditional Formatting Applied (2 pts)
    Validates that conditional formatting is correctly applied to numeric columns.

    Outcome Evaluation:
    - Conditional formatting (color scale) is applied to at least one numeric column.
    - The formatting uses a green-to-red scale where green indicates better values and red indicates worse.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    global sheet_raw
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Conditional Formatting Applied")

    if not sheet_raw:
        checkpoint.add_step("Formatting Exists", False, 1,
                          "Could not access raw sheet data",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.add_step("Correct Color Scale", False, 2,
                          "Cannot check - no sheet data",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get conditional formats from sheet
    step_start = time.time()
    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            step_time = time.time() - step_start
            checkpoint.add_step("Formatting Exists", False, 1,
                              "No sheets found in document",
                              execution_time=step_time)
            checkpoint.add_step("Correct Color Scale", False, 2,
                              "Cannot check - no sheets",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        conditional_formats = sheets[0].get('conditionalFormats', [])
        step_time = time.time() - step_start

        if not conditional_formats:
            checkpoint.add_step("Formatting Exists", False, 1,
                              "No conditional formatting rules found",
                              execution_time=step_time)
            checkpoint.add_step("Correct Color Scale", False, 2,
                              "Cannot check - no conditional formatting",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Check for gradient/color scale rules
        gradient_rules = []
        for cf in conditional_formats:
            if 'gradientRule' in cf:
                gradient_rules.append(cf['gradientRule'])

        if gradient_rules:
            checkpoint.add_step("Formatting Exists", True, 1,
                              f"Found {len(gradient_rules)} color scale/gradient formatting rules",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Formatting Exists", False, 1,
                              f"Found {len(conditional_formats)} conditional formats but no color scales",
                              execution_time=step_time)
            checkpoint.add_step("Correct Color Scale", False, 2,
                              "No gradient/color scale rules to validate",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Formatting Exists", False, 1,
                          f"Error checking conditional formats: {str(e)[:50]}",
                          execution_time=step_time)
        checkpoint.add_step("Correct Color Scale", False, 2,
                          "Cannot check - error occurred",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 2: Check color scale direction (green = better, red = worse)
    step_start = time.time()
    try:
        valid_scales = 0

        for gradient in gradient_rules:
            minpoint = gradient.get('minpoint', {})
            maxpoint = gradient.get('maxpoint', {})

            min_color = minpoint.get('color', {})
            max_color = maxpoint.get('color', {})

            # Check for green-red scale
            # Green typically has high green value, low red
            # Red typically has high red value, low green
            min_green = min_color.get('green', 0)
            min_red = min_color.get('red', 0)
            max_green = max_color.get('green', 0)
            max_red = max_color.get('red', 0)

            # For price column: lower is better, so green should be at min
            # For sqft column: higher is better, so green should be at max
            # Accept either direction as valid since different columns have different "better" directions
            is_green_to_red = (min_green > min_red and max_red > max_green)
            is_red_to_green = (min_red > min_green and max_green > max_red)

            if is_green_to_red or is_red_to_green:
                valid_scales += 1

        step_time = time.time() - step_start

        if valid_scales > 0:
            checkpoint.add_step("Correct Color Scale", True, 2,
                              f"Found {valid_scales} valid green-red color scales",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Correct Color Scale", False, 2,
                              "Color scales found but don't use green-red format",
                              execution_time=step_time)

    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Correct Color Scale", False, 2,
                          f"Error validating color scales: {str(e)[:50]}",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Checkpoint 5: Summary Statistics Table (2 pts)
    Validates that a summary statistics table exists with auto-updating formulas.

    Outcome Evaluation:
    - A summary statistics table exists starting at column K (top-right area of sheet).
    - The summary table contains formulas/equations that reference the main listing data.
    """
    print("----------------- CHECKPOINT 5 ----------------")
    global table_data, sheet_raw
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Summary Statistics Table")

    # Column K is index 10 (0-indexed)
    MIN_SUMMARY_COL = 10

    if not table_data:
        checkpoint.add_step("Summary Table Exists", False, 1,
                          "No tables found in spreadsheet",
                          execution_time=time.time() - checkpoint_start)
        checkpoint.add_step("Contains Formulas", False, 2,
                          "Cannot check - no tables found",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Check if a second table exists at column K or later
    step_start = time.time()
    summary_start_col = None
    summary_end_col = None

    try:
        # Find a table that starts at column K (index 10) or later
        summary_sheet_table = None
        for sheet_table in table_data:
            if sheet_table.start_col >= MIN_SUMMARY_COL:
                summary_sheet_table = sheet_table
                summary_start_col = sheet_table.start_col
                summary_end_col = sheet_table.end_col
                break

        step_time = time.time() - step_start

        if summary_sheet_table is not None:
            checkpoint.add_step("Summary Table Exists", True, 1,
                              f"Found summary table at column {summary_sheet_table.col_letter} with {summary_sheet_table.num_cols} columns",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Summary Table Exists", False, 1,
                              f"No table found starting at column K or later (found {len(table_data)} table(s))",
                              execution_time=step_time)
            checkpoint.add_step("Contains Formulas", False, 2,
                              "Cannot check - no summary table at column K+",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Summary Table Exists", False, 1,
                          f"Error checking for summary table: {str(e)[:50]}",
                          execution_time=step_time)
        checkpoint.add_step("Contains Formulas", False, 2,
                          "Cannot check - error occurred",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 2: Check if summary table contains formulas referencing main data
    step_start = time.time()
    try:
        if not sheet_raw:
            step_time = time.time() - step_start
            checkpoint.add_step("Contains Formulas", False, 2,
                              "Cannot check formulas - no raw sheet data",
                              execution_time=step_time)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            step_time = time.time() - step_start
            checkpoint.add_step("Contains Formulas", False, 2,
                              "Cannot check formulas - no sheets found",
                              execution_time=step_time)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        rows = sheets[0].get('data', [{}])[0].get('rowData', [])

        # Look for formulas in the summary table area
        formula_cells = []
        main_data_refs = 0

        for r_idx, row in enumerate(rows):
            for c_idx, cell in enumerate(row.get('values', [])):
                # Check if cell is in the summary table column range
                if summary_start_col <= c_idx < summary_end_col:
                    user_entered = cell.get('userEnteredValue', {})
                    formula = user_entered.get('formulaValue', '')

                    if formula:
                        formula_cells.append(formula)
                        # Check if formula references main data via:
                        # 1. A1-style cell refs in columns A-J (e.g., A2, B1:B10)
                        # 2. Structured table refs (e.g., TABLE1[COLUMN NAME])
                        formula_upper = formula.upper()
                        has_cell_ref = bool(re.search(r'\b[A-J]\d', formula_upper))
                        has_table_ref = bool(re.search(r'TABLE\d*\[', formula_upper))
                        if has_cell_ref or has_table_ref:
                            main_data_refs += 1

        step_time = time.time() - step_start

        if formula_cells and main_data_refs > 0:
            checkpoint.add_step("Contains Formulas", True, 2,
                              f"Found {len(formula_cells)} formulas, {main_data_refs} reference main data",
                              execution_time=step_time)
        elif formula_cells:
            checkpoint.add_step("Contains Formulas", False, 2,
                              f"Found {len(formula_cells)} formulas but none reference main data (columns A-J)",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Contains Formulas", False, 2,
                              "No formulas found in summary table - appears to use hardcoded values",
                              execution_time=step_time)

    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Contains Formulas", False, 2,
                          f"Error checking formulas: {str(e)[:50]}",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """
    Checkpoint 6: Text Visibility and Formatting (4 pts)
    Validates that all text in both tables is fully visible and not cut off.

    Outcome Evaluation:
    - All column headers in the main table are fully visible (not truncated).
    - All data cells in the main table have adequate column width.
    - All text in the summary statistics table is fully visible.

    Key Logic:
    - Text is only considered "out of bounds" if it exceeds cell width AND
      wrapping is not enabled (wrapStrategy != 'WRAP').
    - If wrapStrategy is 'WRAP', text wraps to multiple lines and is visible.
    - If wrapStrategy is 'OVERFLOW_CELL', text overflows into adjacent empty cells.
    - If wrapStrategy is 'CLIP', text is clipped/hidden.
    """
    print("----------------- CHECKPOINT 6 ----------------")
    global sheet_raw, df
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Text Visibility and Formatting")

    step_names = [
        "Headers Visible",
        "Data Cells Adequate",
        "Summary Text Visible",
    ]

    if not sheet_raw:
        for i, step_name in enumerate(step_names, start=1):
            checkpoint.add_step(step_name, False, i,
                              "Could not access raw sheet data",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        sheets = sheet_raw.get('sheets', [])
        if not sheets:
            for i, step_name in enumerate(step_names, start=1):
                checkpoint.add_step(step_name, False, i,
                                  "No sheets found",
                                  execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        sheet_data = sheets[0].get('data', [{}])[0]
        rows = sheet_data.get('rowData', [])
        col_metadata = sheet_data.get('columnMetadata', [])

        # Approximate character width in pixels (default font)
        CHAR_WIDTH = 7

    except Exception as e:
        for i, step_name in enumerate(step_names, start=1):
            checkpoint.add_step(step_name, False, i,
                              f"Error accessing sheet structure: {str(e)[:50]}",
                              execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Helper to get column width
    def get_col_width(c_idx):
        if c_idx < len(col_metadata):
            return col_metadata[c_idx].get('pixelSize', 100)
        return 100  # Default

    # Step 1: Check column headers visibility (first row)
    step_start = time.time()
    try:
        if rows:
            header_row = rows[0].get('values', [])
            truncated_headers = []

            for c_idx, cell in enumerate(header_row):
                content = cell.get('formattedValue', '')
                if not content:
                    continue

                col_width = get_col_width(c_idx)
                fmt = cell.get('effectiveFormat', {})
                wrap_strategy = fmt.get('wrapStrategy', 'OVERFLOW_CELL')

                if not is_text_visible_in_cell(content, col_width, wrap_strategy,
                                               header_row, c_idx, CHAR_WIDTH):
                    truncated_headers.append(content[:20])

            step_time = time.time() - step_start

            if not truncated_headers:
                checkpoint.add_step("Headers Visible", True, 1,
                                  "All column headers are fully visible",
                                  execution_time=step_time)
            else:
                checkpoint.add_step("Headers Visible", False, 1,
                                  f"Truncated headers: {', '.join(truncated_headers[:3])}...",
                                  execution_time=step_time)
        else:
            step_time = time.time() - step_start
            checkpoint.add_step("Headers Visible", False, 1,
                              "No header row found",
                              execution_time=step_time)
    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Headers Visible", False, 1,
                          f"Error checking headers: {str(e)[:50]}",
                          execution_time=step_time)

    # Step 2: Check data cells in main table (columns A-J)
    step_start = time.time()
    try:
        hidden_cells = 0
        total_cells = 0

        for r_idx, row in enumerate(rows[1:], 1):  # Skip header
            row_values = row.get('values', [])
            for c_idx, cell in enumerate(row_values):
                if c_idx >= 10:  # Only main table (columns A-J)
                    continue

                content = cell.get('formattedValue', '')
                if not content:
                    continue

                total_cells += 1
                col_width = get_col_width(c_idx)
                fmt = cell.get('effectiveFormat', {})
                wrap_strategy = fmt.get('wrapStrategy', 'OVERFLOW_CELL')

                if not is_text_visible_in_cell(content, col_width, wrap_strategy,
                                               row_values, c_idx, CHAR_WIDTH):
                    hidden_cells += 1

        step_time = time.time() - step_start

        if total_cells == 0:
            checkpoint.add_step("Data Cells Adequate", False, 2,
                              "No data cells found in main table",
                              execution_time=step_time)
        elif hidden_cells == 0:
            checkpoint.add_step("Data Cells Adequate", True, 2,
                              f"All {total_cells} data cells are fully visible",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Data Cells Adequate", False, 2,
                              f"{hidden_cells}/{total_cells} cells have hidden/truncated text",
                              execution_time=step_time)
    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Data Cells Adequate", False, 2,
                          f"Error checking data cells: {str(e)[:50]}",
                          execution_time=step_time)

    # Step 3: Check summary table text (column K+)
    step_start = time.time()
    try:
        summary_hidden = 0
        summary_total = 0

        for r_idx, row in enumerate(rows):
            row_values = row.get('values', [])
            for c_idx, cell in enumerate(row_values):
                if c_idx < 10:  # Only summary area (column K+)
                    continue

                content = cell.get('formattedValue', '')
                if not content:
                    continue

                summary_total += 1
                col_width = get_col_width(c_idx)
                fmt = cell.get('effectiveFormat', {})
                wrap_strategy = fmt.get('wrapStrategy', 'OVERFLOW_CELL')

                if not is_text_visible_in_cell(content, col_width, wrap_strategy,
                                               row_values, c_idx, CHAR_WIDTH):
                    summary_hidden += 1

        step_time = time.time() - step_start

        if summary_total == 0:
            checkpoint.add_step("Summary Text Visible", True, 3,
                              "No summary text to check (or no summary table)",
                              execution_time=step_time)
        elif summary_hidden == 0:
            checkpoint.add_step("Summary Text Visible", True, 3,
                              f"All {summary_total} summary cells are fully visible",
                              execution_time=step_time)
        else:
            checkpoint.add_step("Summary Text Visible", False, 3,
                              f"{summary_hidden}/{summary_total} summary cells have hidden text",
                              execution_time=step_time)
    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Summary Text Visible", False, 3,
                          f"Error checking summary: {str(e)[:50]}",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str = None, browsing_history: List[str] = None):
    """
    Grade all checkpoints for the apartment finder task.

    Args:
        workspace_doc_id: Google Sheets document ID to evaluate.
        browsing_history: List of URLs visited during task execution.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    # Setup — graceful degradation (sets globals to None on failure)
    setup(workspace_doc_id)

    # Load model — fail fast with clear error if this fails
    global model
    try:
        model = load_model(model_id)
    except Exception as e:
        raise RuntimeError(
            f"FATAL: Failed to load model '{model_id}'. "
            f"Ensure model ID is correct and API keys are configured. Error: {e}"
        ) from e

    checkpoints: List[Checkpoint] = []

    # Run each checkpoint independently so one failure doesn't prevent others
    checkpoint_funcs = [
        ("Checkpoint 1", grade_checkpoint_1),
        ("Checkpoint 2", grade_checkpoint_2),
        ("Checkpoint 3", lambda: grade_checkpoint_3(browsing_history)),
        ("Checkpoint 4", grade_checkpoint_4),
        ("Checkpoint 5", grade_checkpoint_5),
        ("Checkpoint 6", grade_checkpoint_6),
    ]

    for name, func in checkpoint_funcs:
        try:
            checkpoints.append(func())
        except Exception as e:
            print(f"ERROR: {name} failed unexpectedly: {e}")
            import traceback
            traceback.print_exc()
            failed = Checkpoint(total=1, result=0, name=f"{name} Error")
            failed.add_step("Execution", False, 1,
                          f"Unexpected error: {str(e)[:100]}", execution_time=0)
            checkpoints.append(failed)

    return Result(checkpoints, total_execution_time=time.time() - total_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate apartment finder spreadsheet")
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
