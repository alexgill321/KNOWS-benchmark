import os
import sys
from typing import List
import time
import pandas as pd
import argparse
import html2text
import requests
import json


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
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_text_content
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, EvaluationStep
from src.browsergym.knows.eval.eval_utils.google_services_utils import *
from src.browsergym.knows.eval.eval_utils.table_utils import *
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_match_robust, numerical_match_with_error
from src.browsergym.knows.eval.tasks.sheets_6_investmenttracker.utils import calculate_expected_stock_values, verify_past_prices_with_web_content, parse_currency_value
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download, parallel_execute
from src.browsergym.knows.eval.eval_utils.chart_utils import (
    debug_chart_structure,
    extract_chart_domain_data,
    extract_chart_series_data,
    validate_chart_categories_match,
    validate_chart_values_match
)

# Constants
GOLD_LABELS_SHEET_ID = "13APscQfaHe4F1ffabnVPKpjwbnYx5wy2dgaBd6aXJyQ"
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/sheets_6_investmenttracker/instance_4/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

# Instance-specific parameters
NUM_STOCKS = 7
NUM_SHARES = 25
CHART_TYPE = "LINE"  # Expected chart type
SECTOR = "semiconductor"
DATE_LABEL = "December 31, 2025"
PAST_PRICE_KEYWORDS = ["12/31/2025", "12-31-2025", "past", "past price", "q4 2025", "q4 2025 price", "q4 2025 price ($)", "december 2025", "dec 2025", "december 31 2025", "historical", "initial", "closing price"]

model = None
model_id = "gemini-3-flash-google-ai"

DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

# Global variables
sheet_id = None
table_data = None
chart_data = None
gold_data = None
gold_to_user_ticker_map = None
matched_columns = None
df = None
cached_url_contents = {}
cached_price_source_urls = []
full_sheet_data = []

def setup(workspace_doc_id):
    """
    Setup function to initialize the evaluator.

    Args:
        workspace_doc_id (str, optional): Direct Google Sheets document ID to use
    """
    global sheet_id, table_data, chart_data, gold_data, df, full_sheet_data

    if workspace_doc_id:
        print(f"Using workspace document ID: {workspace_doc_id}")
        sheet_id = workspace_doc_id

    # Extract data from the spreadsheet
    # table_data is list of SheetTable objects
    table_data = extract_tables_from_sheet(sheet_id, SHEETS_SERVICE)
    chart_data = extract_charts_from_sheet(sheet_id, SHEETS_SERVICE)
    gold_data = extract_tables_from_sheet(GOLD_LABELS_SHEET_ID, SHEETS_SERVICE)

    # Fetch full sheet data (all columns) for chart validation fallback
    try:
        result = SHEETS_SERVICE.spreadsheets().values().get(
            spreadsheetId=sheet_id, range='Sheet1'
        ).execute()
        full_sheet_data = result.get('values', [])
    except Exception as e:
        print(f"Warning: Could not fetch full sheet data: {e}")
        full_sheet_data = []

    # Initialize df for use across checkpoints (first table's DataFrame)
    if table_data:
        first_table = table_data[0]
        df = first_table.df if hasattr(first_table, 'df') else first_table
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        # Drop completely empty rows (empty strings and NaN)
        df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)

def preprocess_browsing_history(browsing_history):
    """Fetch and cache page content from browsing history URLs for reuse across checkpoints."""
    global cached_url_contents, cached_price_source_urls, model

    cached_url_contents = {}
    cached_price_source_urls = []

    if not browsing_history:
        return

    relevant_keywords = [
        'companiesmarketcap', 'finance', 'market', 'stock', 'investing', 'bloomberg', 'nasdaq', 'wsj', 'cnbc',
        'semiconductor', 'chip', 'semi'
    ]
    sorted_urls = sorted(list(set(browsing_history)),
                        key=lambda u: any(k in u.lower() for k in relevant_keywords),
                        reverse=True)

    print(f"  Preprocessing: fetching {len(sorted_urls)} URLs in parallel...")
    fetch_tasks = [
        {'id': url, 'func': fetch_page_text_content, 'args': (url,)}
        for url in sorted_urls
    ]
    fetch_results = parallel_execute(fetch_tasks)

    for url, html_content in fetch_results.items():
        if html_content[1] == "OK":
            cached_url_contents[url] = html_content[0][:60000]

    print(f"  Preprocessing: cached {len(cached_url_contents)}/{len(sorted_urls)} URLs")


def grade_checkpoint_1_and_2():
    """
    Combined Checkpoint 1 & 2: Spreadsheet structure and data accuracy.
    Returns two separate Checkpoint objects for proper display.

    Checkpoint 1 Outcome Evaluation:
    - There is a column/row with the name of each stock.
    - There is a column/row with the ticker symbol of each stock.
    - There is a column/row with the current price of each stock.
    - There is a column/row with the past price of each stock.
    - There is a column/row with the gain/loss of each stock.
    - There is a column/row with the number of shares owned for each stock.
    - There is a column/row with the total value of each stock.

    Checkpoint 2 Outcome Evaluation:
    - The stocks are the top 7 semiconductor companies by market cap as of December 31, 2025.
    - The past price of each stock is correct.
    - The current price of each stock is correct.
    - The gain/loss of each stock is correct.

    Returns:
        tuple: (checkpoint1, checkpoint2) - Two separate Checkpoint objects
    """
    print("----------------- CHECKPOINT 1 & 2 ----------------")
    global model, matched_columns, gold_to_user_ticker_map, df
    checkpoint_start = time.time()
    checkpoint1 = Checkpoint(total=7, result=0, name="Spreadsheet Structure")
    checkpoint2 = Checkpoint(total=NUM_STOCKS + 3, result=0, name="Data Accuracy")

    cp1_step_names = [
        "Stock Name Column", "Ticker Symbol Column", "Current Price Column",
        "Past Price Column", "Gain/Loss Column", "Number Of Shares Owned Column",
        "Total Value Of Each Stock Column",
    ]
    # Stock selection steps are dynamically generated (one per gold stock)
    cp2_step_names = [
        "Past Price Accuracy",
        "Current Price Accuracy", "Gain/Loss Calculation Accuracy",
    ]

    if not table_data:
        for i, name in enumerate(cp1_step_names, 1):
            checkpoint1.add_step(name, False, i, "No table data found in spreadsheet", execution_time=0)
        for i in range(1, NUM_STOCKS + 1):
            checkpoint2.add_step(f"Stock Match: unknown", False, i, "No table data found in spreadsheet", execution_time=0)
        for i, name in enumerate(cp2_step_names, 1):
            checkpoint2.add_step(name, False, NUM_STOCKS + i, "No table data found in spreadsheet", execution_time=0)
        checkpoint1.execution_time = time.time() - checkpoint_start
        checkpoint2.execution_time = 0
        return (checkpoint1, checkpoint2)

    # Convert table_data to DataFrame for easier analysis
    step_start = time.time()
    try:
        # Assume first table contains the stock data
        first_table = table_data[0] if isinstance(table_data, list) else table_data
        df = first_table.df if hasattr(first_table, 'df') else first_table
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        # Drop completely empty rows (empty strings and NaN)
        df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)

    except Exception as e:
        for i, name in enumerate(cp1_step_names, 1):
            checkpoint1.add_step(name, False, i, f"Failed to parse table data: {str(e)}", execution_time=0)
        for i in range(1, NUM_STOCKS + 1):
            checkpoint2.add_step(f"Stock Match: unknown", False, i, f"Failed to parse table data: {str(e)}", execution_time=0)
        for i, name in enumerate(cp2_step_names, 1):
            checkpoint2.add_step(name, False, NUM_STOCKS + i, f"Failed to parse table data: {str(e)}", execution_time=0)
        checkpoint1.execution_time = time.time() - checkpoint_start
        checkpoint2.execution_time = 0
        return (checkpoint1, checkpoint2)

    # Load reference stocks from gold data
    step_start = time.time()
    gold_df = gold_data[0].df if hasattr(gold_data[0], 'df') else gold_data[0]
    reference_tickers = gold_df['Ticker'].tolist()
    reference_names = gold_df['Stock Name'].tolist()

    # Check for required columns and store matches
    required_columns = [
        ("Stock Name", ["name", "stock", "company"]),
        ("Ticker Symbol", ["ticker", "symbol", "ticker symbol"]),
        ("Current Price", ["current", "current price", "market price", "current price ($)"]),
        ("Past Price", PAST_PRICE_KEYWORDS),
        ("Gain/Loss", ["gain", "loss", "gain/loss", "change", "profit", "profit/loss", "return", "difference"]),
        ("Number of shares owned", ["shares", "quantity", "owned", "holdings"]),
        ("Total value of each stock", ["total", "value", "total value", "position"])
    ]

    original_columns = [str(col) for col in df.columns]

    # Use standardized match_columns() - keyword matching first, then LLM fallback
    if model is None:
        model = load_model(model_id)
    matched_columns = match_columns(df, required_columns, model=model, parallel=True)

    # Relaxed check for "Number of shares owned": if no dedicated column was matched,
    # accept any column header that references the expected share count
    if "Number of shares owned" not in matched_columns:
        shares_str = str(NUM_SHARES)
        for col in original_columns:
            if shares_str in col and "share" in col.lower():
                matched_columns["Number of shares owned"] = col
                print(f"  Relaxed shares match: column '{col}' references {NUM_SHARES} shares")
                break

    # Add checkpoint steps for each required column
    for step_num, (col_name, keywords) in enumerate(required_columns, start=1):
        step_start = time.time()
        if col_name in matched_columns:
            matched_column = matched_columns[col_name]
            checkpoint1.add_step(f"{col_name.title()} Column", True, step_num,
                              f"Found column matching '{col_name}': '{matched_column}'",
                              execution_time=time.time() - step_start)
        else:
            checkpoint1.add_step(f"{col_name.title()} Column", False, step_num,
                              f"No column found for '{col_name}'. Available columns: {', '.join(original_columns)}",
                              execution_time=time.time() - step_start)

    # Now validate the stock names match the reference stocks
    # Create mapping from gold ticker to user ticker for price validation
    gold_to_user_ticker_map = {}

    step_start = time.time()
    if "Ticker Symbol" in matched_columns and "Stock Name" in matched_columns:
        ticker_symbol_col = matched_columns["Ticker Symbol"]
        stock_name_col = matched_columns["Stock Name"]
        user_tickers = df[ticker_symbol_col].dropna().astype(str).tolist()

        # First try exact ticker matching and build mapping
        matching_tickers = set(user_tickers) & set(reference_tickers)
        exact_match_count = len(matching_tickers)

        # Build mapping for exact matches
        for gold_ticker in reference_tickers:
            if gold_ticker in matching_tickers:
                gold_to_user_ticker_map[gold_ticker] = gold_ticker

        # For non-matching reference stocks, try LLM-based fuzzy matching
        llm_matches = []
        if exact_match_count < NUM_STOCKS:
            unmatched_refs = [(name, ticker) for name, ticker in zip(reference_names, reference_tickers)
                             if ticker not in matching_tickers]

            print(f"Trying LLM matching for {len(unmatched_refs)} unmatched reference stocks...")
            try:
                if model is None:
                    model = load_model(model_id)

                for ref_name, ref_ticker in unmatched_refs:
                    # Build user stock candidates string - show ALL user stocks
                    candidates = []
                    for idx, row in df.iterrows():
                        u_name = str(row[stock_name_col]).strip()
                        u_ticker = str(row[ticker_symbol_col]).strip()
                        candidates.append(f"{idx}: {u_name} ({u_ticker})")

                    if not candidates:
                        continue

                    # Ask LLM if reference stock matches any user stock
                    prompt = f"""Reference: {ref_name} ({ref_ticker})
                    User stocks: {', '.join(candidates)}

                    Does the reference match any user stock? Answer with just the row index number if yes, or "no" if no match."""

                    try:
                        messages = [
                            {"role": "user", "content": [{"type": "text", "text": prompt}]}
                        ]
                        response = model(messages)
                        response_text = response.strip().lower()

                        # Try to extract row number
                        if response_text != 'no' and response_text.isdigit():
                            row_idx = int(response_text)
                            if row_idx in df.index:
                                # Get the user's ticker for this row
                                user_ticker = str(df.loc[row_idx, ticker_symbol_col]).strip()

                                # Check if this user ticker was already matched exactly
                                if user_ticker in matching_tickers:
                                    print(f"LLM matched {ref_ticker} to already-matched ticker {user_ticker} at row {row_idx} - counting as failure")
                                else:
                                    llm_matches.append((ref_ticker, row_idx))
                                    gold_to_user_ticker_map[ref_ticker] = user_ticker
                                    print(f"LLM matched {ref_ticker} to row {row_idx} (user ticker: {user_ticker})")
                    except Exception as e:
                        print(f"LLM matching failed for {ref_ticker}: {str(e)}")
            except Exception as e:
                print(f"LLM loading failed for matching stocks: {str(e)}")

        total_match_count = exact_match_count + len(llm_matches)
        step_time = time.time() - step_start

        # Add one step per gold stock
        for step_idx, (ref_name, ref_ticker) in enumerate(zip(reference_names, reference_tickers), start=1):
            if ref_ticker in gold_to_user_ticker_map:
                user_ticker = gold_to_user_ticker_map[ref_ticker]
                match_type = "exact" if ref_ticker == user_ticker else "LLM-matched"
                checkpoint2.add_step(f"Stock Match: {ref_ticker} ({ref_name})", True, step_idx,
                                  f"Matched ({match_type}) to user ticker: {user_ticker}",
                                  execution_time=step_time if step_idx == 1 else 0)
            else:
                checkpoint2.add_step(f"Stock Match: {ref_ticker} ({ref_name})", False, step_idx,
                                  f"Gold stock {ref_ticker} ({ref_name}) not found in user's stocks: {', '.join(user_tickers[:NUM_STOCKS])}",
                                  execution_time=step_time if step_idx == 1 else 0)
    else:
        step_time = time.time() - step_start
        missing = []
        if "Ticker Symbol" not in matched_columns:
            missing.append("ticker symbol")
        if "Stock Name" not in matched_columns:
            missing.append("stock name")
        for step_idx in range(1, NUM_STOCKS + 1):
            checkpoint2.add_step(f"Stock Match: unknown", False, step_idx,
                              f"Cannot validate stocks - {' and '.join(missing)} column(s) not found",
                              execution_time=step_time if step_idx == 1 else 0)

    # Step 2 (checkpoint2): Verify past prices are correct (5% tolerance)
    step_start = time.time()
    if "Past Price" in matched_columns and "Ticker Symbol" in matched_columns and gold_to_user_ticker_map:
        past_price_col = matched_columns["Past Price"]
        ticker_col = matched_columns["Ticker Symbol"]

        # Get gold past prices
        gold_df = gold_data[0].df if hasattr(gold_data[0], 'df') else gold_data[0]

        # Build mapping of gold ticker to gold past price
        gold_past_prices = {}
        for _, row in gold_df.iterrows():
            gold_past_prices[row['Ticker']] = row['Past Price']

        # Compare user's past prices with gold using the ticker mapping
        mismatches = []
        failed_stocks = []
        match_count = 0
        total_comparisons = 0

        for gold_ticker, user_ticker in gold_to_user_ticker_map.items():
            if gold_ticker not in gold_past_prices:
                continue

            # Find the row with this user ticker
            matching_rows = df[df[ticker_col].astype(str).str.strip() == user_ticker]
            if matching_rows.empty:
                continue

            user_price = parse_currency_value(matching_rows.iloc[0][past_price_col])
            gold_price_raw = gold_past_prices[gold_ticker]
            gold_price = parse_currency_value(gold_price_raw)

            if gold_price is None:
                raise ValueError(f"Gold data error: could not parse past price '{gold_price_raw}' for ticker {gold_ticker}. Gold data must not contain #N/A or invalid values.")

            if user_price is None:
                mismatches.append(f"{user_ticker} (gold: {gold_ticker}): could not parse user price")
                total_comparisons += 1
                continue

            # Use numerical matching with 5% error against gold label
            is_match, diff = numerical_match_with_error(gold_price, user_price, error_percent=5.0)
            total_comparisons += 1

            if is_match:
                match_count += 1
            else:
                gold_idx = reference_tickers.index(gold_ticker) if gold_ticker in reference_tickers else -1
                company_name = reference_names[gold_idx] if gold_idx >= 0 else gold_ticker
                failed_stocks.append(
                {
                    'ticker': gold_ticker,
                    'company_name': company_name,
                    'gold_price': gold_price,
                    'user_price': user_price,
                }
                )

        # Batch fallback: verify all failed stocks against web content in one LLM call
        if failed_stocks and cached_url_contents:
            search_url = keywords_match_robust(list(cached_url_contents.keys()), DATE_LABEL, model, f"Stock prices on {DATE_LABEL}")
            web_results = verify_past_prices_with_web_content(
                failed_stocks, cached_url_contents, search_url, model
            )
            for stock in failed_stocks:
                is_match, diff = numerical_match_with_error(web_results.get(stock['ticker'], stock['gold_price']), stock['user_price'], error_percent=5.0)
                if is_match:
                    match_count += 1
                    print(f"  {stock['ticker']}: gold mismatch but web content confirms price ( ({diff:.1f}% diff)")
                else:
                    mismatches.append(f"{stock['ticker']}: {stock['user_price']} vs gold {stock['gold_price']} ({diff:.1f}% diff)")
        else:
            for stock in failed_stocks:
                mismatches.append(f"{stock['ticker']}: {stock['user_price']} vs gold {stock['gold_price']} ({(stock['user_price'] - stock['gold_price'])/stock['gold_price'] * 100:.1f}% diff)")

        step_time = time.time() - step_start
        if total_comparisons == 0:
            checkpoint2.add_step("Past Price Accuracy", False, NUM_STOCKS + 1,
                              "No past prices found to validate",
                              execution_time=step_time)
        elif match_count == total_comparisons:
            checkpoint2.add_step("Past Price Accuracy", True, NUM_STOCKS + 1,
                              f"All {match_count}/{total_comparisons} past prices match (gold or web-verified)",
                              execution_time=step_time)
        else:
            checkpoint2.add_step("Past Price Accuracy", False, NUM_STOCKS + 1,
                              f"Only {match_count}/{total_comparisons} past prices match. Mismatches: {'; '.join(mismatches[:3])}{'...' if len(mismatches) > 3 else ''}",
                              execution_time=step_time)
    else:
        step_time = time.time() - step_start
        checkpoint2.add_step("Past Price Accuracy", False, NUM_STOCKS + 1,
                          "Cannot validate past prices - required columns not found or no ticker mapping available",
                          execution_time=step_time)

    # Step 3 (checkpoint2): Verify current prices are correct (5% tolerance)
    step_start = time.time()
    if "Current Price" in matched_columns and "Ticker Symbol" in matched_columns and gold_to_user_ticker_map:
        current_price_col = matched_columns["Current Price"]
        ticker_col = matched_columns["Ticker Symbol"]

        # Get gold current prices
        gold_df = gold_data[0].df if hasattr(gold_data[0], 'df') else gold_data[0]

        # Build mapping of gold ticker to gold current price
        gold_current_prices = {}
        for _, row in gold_df.iterrows():
            gold_current_prices[row['Ticker']] = row['Current Price']

        # Compare user's current prices with gold using the ticker mapping
        mismatches = []
        failed_stocks = []
        match_count = 0
        total_comparisons = 0

        for gold_ticker, user_ticker in gold_to_user_ticker_map.items():
            if gold_ticker not in gold_current_prices:
                continue

            # Find the row with this user ticker
            matching_rows = df[df[ticker_col].astype(str).str.strip() == user_ticker]
            if matching_rows.empty:
                continue

            user_price = parse_currency_value(matching_rows.iloc[0][current_price_col])
            gold_price_raw = gold_current_prices[gold_ticker]
            gold_price = parse_currency_value(gold_price_raw)

            if gold_price is None:
                raise ValueError(f"Gold data error: could not parse current price '{gold_price_raw}' for ticker {gold_ticker}. Gold data must not contain #N/A or invalid values.")

            if user_price is None:
                mismatches.append(f"{user_ticker} (gold: {gold_ticker}): could not parse user price")
                total_comparisons += 1
                continue

            # Use numerical matching with 5% error
            is_match, diff = numerical_match_with_error(gold_price, user_price, error_percent=5.0)
            total_comparisons += 1

            if is_match:
                match_count += 1
            else:
                gold_idx = reference_tickers.index(gold_ticker) if gold_ticker in reference_tickers else -1
                company_name = reference_names[gold_idx] if gold_idx >= 0 else gold_ticker
                failed_stocks.append({
                    'ticker': gold_ticker,
                    'company_name': company_name,
                    'gold_price': gold_price,
                    'user_price': user_price,
                })

        # Batch fallback: verify all failed stocks against web content in one LLM call
        if failed_stocks and cached_url_contents:
            search_url = keywords_match_robust(list(cached_url_contents.keys()), SECTOR, model, f"Stock prices of the sector {SECTOR}")
            web_results = verify_past_prices_with_web_content(
                failed_stocks, cached_url_contents, search_url, model
            )
            for stock in failed_stocks:
                is_match, diff = numerical_match_with_error(web_results.get(stock['ticker'], stock['gold_price']), stock['user_price'], error_percent=5.0)
                if is_match:
                    match_count += 1
                    print(f"  {stock['ticker']}: gold mismatch but web content confirms price ( ({diff:.1f}% diff)")
                else:
                    mismatches.append(f"{stock['ticker']}: {stock['user_price']} vs gold {stock['gold_price']} ({diff:.1f}% diff)")
        else:
            for stock in failed_stocks:
                mismatches.append(f"{stock['ticker']}: {stock['user_price']} vs gold {stock['gold_price']} ({(stock['user_price'] - stock['gold_price'])/stock['gold_price'] * 100:.1f}% diff)")

        step_time = time.time() - step_start
        if total_comparisons == 0:
            checkpoint2.add_step("Current Price Accuracy", False, NUM_STOCKS + 2,
                              "No current prices found to validate",
                              execution_time=step_time)
        elif match_count == total_comparisons:
            checkpoint2.add_step("Current Price Accuracy", True, NUM_STOCKS + 2,
                              f"All {match_count}/{total_comparisons} current prices match (gold or web-verified)",
                              execution_time=step_time)
        else:
            checkpoint2.add_step("Current Price Accuracy", False, NUM_STOCKS + 2,
                              f"Only {match_count}/{total_comparisons} current prices match. Mismatches: {'; '.join(mismatches[:3])}{'...' if len(mismatches) > 3 else ''}",
                              execution_time=step_time)
    else:
        step_time = time.time() - step_start
        checkpoint2.add_step("Current Price Accuracy", False, NUM_STOCKS + 2,
                          "Cannot validate current prices - required columns not found or no ticker mapping available",
                          execution_time=step_time)

    # Step 4 (checkpoint2): Verify gain/loss calculations are correct
    step_start = time.time()
    if "Gain/Loss" in matched_columns and "Current Price" in matched_columns and "Past Price" in matched_columns:
        gainloss_col = matched_columns["Gain/Loss"]
        current_price_col = matched_columns["Current Price"]
        past_price_col = matched_columns["Past Price"]

        mismatches = []
        match_count = 0
        total_comparisons = 0

        for idx, row in df.iterrows():
            try:
                user_gainloss = parse_currency_value(row[gainloss_col])
                current_price = parse_currency_value(row[current_price_col])
                past_price = parse_currency_value(row[past_price_col])

                if user_gainloss is None or current_price is None or past_price is None:
                    print(f"Could not parse currency values for row {idx}")
                    continue

                # Calculate expected gain/loss - accept dollar, percentage, or total interpretations
                expected_dollar = current_price - past_price
                expected_pct_decimal = (current_price - past_price) / past_price if past_price != 0 else None
                expected_pct_100 = expected_pct_decimal * 100 if expected_pct_decimal is not None else None
                expected_total = expected_dollar * NUM_SHARES

                # Try all interpretations, accept whichever matches
                is_match = False
                best_diff = float('inf')
                for expected in [expected_dollar, expected_pct_decimal, expected_pct_100, expected_total]:
                    if expected is not None:
                        match, diff = numerical_match_with_error(expected, user_gainloss, error_percent=5.0)
                        if match:
                            is_match = True
                            best_diff = diff
                            break
                        if diff < best_diff:
                            best_diff = diff
                diff = best_diff
                total_comparisons += 1

                if is_match:
                    match_count += 1
                else:
                    ticker = str(row[ticker_col]).strip() if "Ticker Symbol" in matched_columns else f"Row {idx}"
                    mismatches.append(f"{ticker}: {user_gainloss} vs expected {expected_dollar:.2f} ({diff:.1f}% diff)")
            except (ValueError, TypeError, KeyError) as e:
                print(f"Error processing gain/loss for row {idx}: {e}")
                continue

        step_time = time.time() - step_start
        if total_comparisons == 0:
            checkpoint2.add_step("Gain/Loss Calculation Accuracy", False, NUM_STOCKS + 3,
                              "No gain/loss values found to validate",
                              execution_time=step_time)
        elif match_count == total_comparisons:
            checkpoint2.add_step("Gain/Loss Calculation Accuracy", True, NUM_STOCKS + 3,
                              f"All {match_count}/{total_comparisons} gain/loss calculations are correct within 5% tolerance",
                              execution_time=step_time)
        else:
            checkpoint2.add_step("Gain/Loss Calculation Accuracy", False, NUM_STOCKS + 3,
                              f"Only {match_count}/{total_comparisons} gain/loss calculations are correct. Mismatches: {'; '.join(mismatches[:3])}{'...' if len(mismatches) > 3 else ''}",
                              execution_time=step_time)
    else:
        step_time = time.time() - step_start
        checkpoint2.add_step("Gain/Loss Calculation Accuracy", False, NUM_STOCKS + 3,
                          "Cannot validate gain/loss - required columns not found",
                          execution_time=step_time)

    checkpoint1.execution_time = time.time() - checkpoint_start
    checkpoint2.execution_time = time.time() - checkpoint_start
    return (checkpoint1, checkpoint2)

def grade_checkpoint_3(browsing_history=None):
    """
    Checkpoint 3: A website containing the information was visited.

    Outcome Evaluation:
    - The website trace contains a URL that has information about the top 7 semiconductor companies by market cap.
    - The website trace contains a URL that has information about the past price of each stock.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    global model, cached_price_source_urls
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Website Visit Validation")

    if not browsing_history:
        checkpoint.add_step("Stock Info Website", False, 1,
                          "No browsing history provided",
                          execution_time=0)
        checkpoint.add_step("Price Data Source", False, 2,
                          "Cannot validate - no browsing history",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Prepare user data for prompt
    user_data_str = ""
    if table_data:
        try:
            first_table = table_data[0] if isinstance(table_data, list) else table_data
            temp_df = first_table.df if hasattr(first_table, 'df') else first_table
            if isinstance(temp_df, dict):
                temp_df = pd.DataFrame(temp_df)
            user_data_str = temp_df.to_markdown()
        except Exception as e:
            print(f"Error converting table data to markdown: {e}")
            user_data_str = str(table_data)

    if model is None:
        model = load_model(model_id)

    # Reuse cached URL contents from preprocess_browsing_history()
    url_contents = cached_url_contents

    urls_checked = len(url_contents)
    found_sector_stocks_info = False
    found_past_prices_info = False
    evidence_urls = {"sector_stocks": [], "past_prices": []}

    step_start = time.time()

    # ============ Parallel LLM analysis of cached content ============
    if url_contents:
        vlm_tasks = []
        for url, markdown_content in url_contents.items():
            prompt = f"""
            You are evaluating if a visited website contains the information used to populate a spreadsheet.

            User's Spreadsheet Data:
            {user_data_str}

            Website Content (Markdown):
            {markdown_content}

            Task Criteria:
            1. Does this website content list the current top highest market cap {SECTOR} companies and their current prices (specifically matching the stocks and current prices in the User's Spreadsheet)?
            2. Does this website content contain the past/historical price information for these stocks from {DATE_LABEL} (matching the values in the User's Spreadsheet)?

            Evaluate strictly based on the provided Website Content. If most of the information is present but not all details match exactly (~75%), still consider it a match.

            Respond with a JSON object strictly in this format:
            {{
                "has_top_sector_stocks": boolean,
                "has_past_prices": boolean,
                "reasoning": "string explaining the decision"
            }}
            """

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            vlm_tasks.append({'id': url, 'messages': messages})

        print(f"  Running {len(vlm_tasks)} LLM content analyses in parallel...")

        # Run LLM tasks in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def analyze_url(task):
            url = task['id']
            messages = task['messages']
            try:
                response_text = model(messages).strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3]
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3]
                result = json.loads(response_text)
                return url, result
            except Exception as e:
                print(f"Error parsing LLM response for {url}: {e}")
                return url, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(analyze_url, task) for task in vlm_tasks]
            for future in as_completed(futures):
                url, result = future.result()
                if result:
                    if result.get("has_top_sector_stocks") and not found_sector_stocks_info:
                        found_sector_stocks_info = True
                        evidence_urls["sector_stocks"].append(url)
                        print(f"  Found {SECTOR} stocks info in {url}")
                    if result.get("has_past_prices") and not found_past_prices_info:
                        found_past_prices_info = True
                        evidence_urls["past_prices"].append(url)
                        print(f"  Found past prices info in {url}")

    step_time = time.time() - step_start

    # Cache price source URLs for fallback use in checkpoint 2
    cached_price_source_urls = evidence_urls["past_prices"]

    if found_sector_stocks_info:
        checkpoint.add_step("Stock Info Website", True, 1,
                          f"Verified website content contains top {SECTOR} stock info. Source(s): {', '.join(evidence_urls['sector_stocks'])}",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Stock Info Website", False, 1,
                          f"Could not verify top {SECTOR} stock info in visited websites (checked {urls_checked} URLs).",
                          execution_time=step_time)

    if found_past_prices_info:
        checkpoint.add_step("Price Data Source", True, 2,
                          f"Verified website content contains past price data. Source(s): {', '.join(evidence_urls['past_prices'])}",
                          execution_time=step_time)
    else:
        checkpoint.add_step("Price Data Source", False, 2,
                          f"Could not verify past price data in visited websites (checked {urls_checked} URLs).",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_4():
    """
    Checkpoint 4: A line chart showing the current total value of each position relative to the others was created.

    Outcome Evaluation:
    - A line chart was created in the spreadsheet.
    - The line chart contains each of the stocks.
    - The line chart correctly shows the current total value of each position.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=3, result=0, name="Line Chart Validation")

    # Step 1: Check if line chart exists
    step_start = time.time()
    if not chart_data or len(chart_data) == 0:
        step_time = time.time() - step_start
        checkpoint.add_step("Line Chart Exists", False, 1,
                          "No charts found in spreadsheet",
                          execution_time=step_time)
        checkpoint.add_step("Chart Contains Stocks", False, 2,
                          "Cannot validate - no chart found",
                          execution_time=0)
        checkpoint.add_step("Chart Shows Total Values", False, 3,
                          "Cannot validate - no chart found",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Find line charts
    line_charts = [c for c in chart_data if c.get('chart_type', '').upper() == 'LINE']

    step_time = time.time() - step_start
    if not line_charts:
        found_types = [c.get('chart_type', 'unknown') for c in chart_data]
        checkpoint.add_step("Line Chart Exists", False, 1,
                          f"No line chart found. Found chart types: {', '.join(found_types)}",
                          execution_time=step_time)
        checkpoint.add_step("Chart Contains Stocks", False, 2,
                          "Cannot validate - no line chart found",
                          execution_time=0)
        checkpoint.add_step("Chart Shows Total Values", False, 3,
                          "Cannot validate - no line chart found",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    chart = line_charts[0]  # Use the first line chart found
    chart_title = chart.get('title', 'Untitled')

    # Debug: Print chart structure to understand what we're working with
    if DEBUG:
        debug_chart_structure(chart)

    checkpoint.add_step("Line Chart Exists", True, 1,
                      f"Found line chart: '{chart_title}'",
                      execution_time=step_time)

    # Step 2: Validate chart contains all stocks
    step_start = time.time()

    # Check if we have the necessary data from checkpoint 1
    if not matched_columns or not gold_to_user_ticker_map or df is None:
        step_time = time.time() - step_start
        checkpoint.add_step("Chart Contains Stocks", False, 2,
                          "Cannot validate - required data from checkpoint 1 not available",
                          execution_time=step_time)
        checkpoint.add_step("Chart Shows Total Values", False, 3,
                          "Cannot validate - required data not available",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Extract chart categories (x-axis - should be stock names or tickers)
    try:
        chart_categories = extract_chart_domain_data(chart, df, full_sheet_data=full_sheet_data)

        if not chart_categories:
            step_time = time.time() - step_start
            checkpoint.add_step("Chart Contains Stocks", False, 2,
                              "Could not extract category data from chart",
                              execution_time=step_time)
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              "Cannot validate - chart categories not found",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Get expected stocks - try both ticker and name columns
        expected_stocks = []
        expected_stocks_alt = []
        if "Ticker Symbol" in matched_columns:
            ticker_col = matched_columns["Ticker Symbol"]
            expected_stocks = df[ticker_col].dropna().astype(str).str.strip().tolist()
        if "Stock Name" in matched_columns:
            name_col = matched_columns["Stock Name"]
            expected_stocks_alt = df[name_col].dropna().astype(str).str.strip().tolist()

        if not expected_stocks:
            step_time = time.time() - step_start
            checkpoint.add_step("Chart Contains Stocks", False, 2,
                              "Could not determine expected stocks from table data",
                              execution_time=step_time)
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              "Cannot validate - expected stocks not found",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Check for duplicate stocks in table data
        stock_unique = set(expected_stocks)
        if len(stock_unique) != len(expected_stocks):
            step_time = time.time() - step_start
            checkpoint.add_step("Chart Contains Stocks", False, 2,
                              f"Duplicate stocks detected in table. Expected {len(expected_stocks)} unique stocks but found {len(stock_unique)} unique.",
                              execution_time=step_time)
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              "Cannot validate - duplicate stocks in table.",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Check for duplicate labels in chart domain
        chart_unique = set(chart_categories)
        if len(chart_unique) != len(chart_categories):
            step_time = time.time() - step_start
            checkpoint.add_step("Chart Contains Stocks", False, 2,
                              "Duplicate labels detected in chart domain. Chart should show each stock exactly once.",
                              execution_time=step_time)
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              "Cannot validate - duplicate stocks in chart.",
                              execution_time=0)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Validate categories match expected stocks (use fuzzy matching)
        match_count, total_expected, missing = validate_chart_categories_match(
            chart_categories, expected_stocks, tolerance='fuzzy'
        )

        if expected_stocks_alt and match_count < NUM_STOCKS:
            alt_count, alt_total, alt_missing = validate_chart_categories_match(
                chart_categories, expected_stocks_alt, tolerance='fuzzy'
            )
            if alt_count > match_count:
                match_count, total_expected, missing = alt_count, alt_total, alt_missing

        step_time = time.time() - step_start

        if match_count >= NUM_STOCKS:
            checkpoint.add_step("Chart Contains Stocks", True, 2,
                              f"Chart contains all {match_count} expected stocks",
                              execution_time=step_time)
        else:
            missing_str = ', '.join(missing[:3]) + ('...' if len(missing) > 3 else '')
            checkpoint.add_step("Chart Contains Stocks", False, 2,
                              f"Chart only contains {match_count}/{total_expected} stocks. Missing: {missing_str}",
                              execution_time=step_time)

    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Chart Contains Stocks", False, 2,
                          f"Error validating chart stocks: {str(e)}",
                          execution_time=step_time)
        checkpoint.add_step("Chart Shows Total Values", False, 3,
                          "Cannot validate - error in stock validation",
                          execution_time=0)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 3: Validate total values are correct
    step_start = time.time()

    try:
        # Extract chart values (y-axis - should be total dollar values)
        chart_values = extract_chart_series_data(chart, df, full_sheet_data=full_sheet_data)

        if not chart_values:
            step_time = time.time() - step_start
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              "Could not extract value data from chart",
                              execution_time=step_time)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Calculate expected total values from table data
        expected_values = calculate_expected_stock_values(df, matched_columns, gold_to_user_ticker_map, NUM_SHARES)

        if not expected_values:
            step_time = time.time() - step_start
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              "Could not calculate expected total values from table data",
                              execution_time=step_time)
            checkpoint.execution_time = time.time() - checkpoint_start
            return checkpoint

        # Validate chart values match expected total values (2% tolerance)
        match_count, total_count, mismatches = validate_chart_values_match(
            chart_values, expected_values, error_percent=2.0
        )

        step_time = time.time() - step_start

        if match_count == total_count and total_count >= NUM_STOCKS:
            checkpoint.add_step("Chart Shows Total Values", True, 3,
                              f"All {match_count}/{total_count} total values match expected values within 2% tolerance",
                              execution_time=step_time)
        else:
            mismatch_str = '; '.join(mismatches[:3]) + ('...' if len(mismatches) > 3 else '')
            checkpoint.add_step("Chart Shows Total Values", False, 3,
                              f"Only {match_count}/{total_count} total values match. Mismatches: {mismatch_str}",
                              execution_time=step_time)

    except Exception as e:
        step_time = time.time() - step_start
        checkpoint.add_step("Chart Shows Total Values", False, 3,
                          f"Error validating chart total values: {str(e)}",
                          execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoints(workspace_doc_id=None, browsing_history=None):
    """
    Grade all checkpoints for the investment tracker task.

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

        # Use cached model if available
        global model
        model = load_model(model_id)

        checkpoints: List[Checkpoint] = []

        # Preprocess browsing history first (caches URL content for CP2 fallback and CP3)
        preprocess_browsing_history(browsing_history)

        checkpoint1, checkpoint2 = grade_checkpoint_1_and_2()
        checkpoints.append(checkpoint1)
        checkpoints.append(checkpoint2)
        checkpoints.append(grade_checkpoint_3(browsing_history))
        checkpoints.append(grade_checkpoint_4())

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
    parser = argparse.ArgumentParser(description="Evaluate investment tracker spreadsheet")
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
