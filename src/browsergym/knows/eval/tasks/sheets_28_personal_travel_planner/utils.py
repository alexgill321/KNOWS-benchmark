"""Template-specific utilities for the Personal Travel Planner task."""

import math
import operator
import os
import re
import time
import unicodedata
from datetime import datetime as _dt, date as _date, time as _time, timedelta
from itertools import permutations
from typing import Optional, List, Dict, Tuple
from urllib.parse import quote

import pandas as pd

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, StepCategory, calculate_percentage_score
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_exact_match
from src.browsergym.knows.eval.eval_utils.llm_utils import extract_json_with_llm
from src.browsergym.knows.eval.eval_utils.table_utils import (
    match_columns, get_column_index_by_name, get_background_color,
    classify_row_color,
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import fast_parallel_vlm_calls, parallel_execute
from src.browsergym.knows.eval.eval_utils.google_sheets_utils import find_urls_in_sheet
from src.browsergym.knows.eval.eval_utils.web_utils import (
    fetch_api_with_retry,
    fetch_with_fallbacks,
    is_unverifiable_url,
)

# --- Constants ---------------------------------------------------------------

COLUMN_KEYWORDS = {
    "Date": ["date", "day"],
    "Time of Day": ["time of day", "time", "period", "time slot"],
    "Destination": ["destination", "food/activity", "food activity", "activity", "place", "location"],
    "Cuisine": ["cuisine", "cuisine type", "type of cuisine", "food type"],
    "Opening Time": ["opening time", "opening hours", "hours", "open", "open time"],
    "Start Time": ["start time", "start", "arrival time", "arrival"],
    "Departure Time": ["departure time", "departure", "end time", "leave time"],
    "Duration": ["duration", "time spent", "length"],
    "Review Link": ["review link", "review", "reviews"],
    "Transportation Mode": ["transportation mode", "transportation", "transport", "travel mode"],
    "Travel Time": ["travel time", "transit time", "commute time"],
    "Cost": ["cost", "price", "fee", "expense"],
    "Alternative Option": ["alternative option", "alternative", "backup", "backup option"],
}
REQUIRED_COLUMNS = list(COLUMN_KEYWORDS.items())
_ALL_COLUMN_KEYWORDS = {kw.lower() for kws in COLUMN_KEYWORDS.values() for kw in kws}

ACTIVITY_TIME_KEYWORDS = ["morning", "afternoon", "evening"]
FOOD_TIME_KEYWORDS = ["lunch", "dinner"]
EXPECTED_ACTIVITIES_PER_DAY = 3
EXPECTED_FOOD_STOPS_PER_DAY = 2

# Feasible start-time windows per Time of Day slot (minutes since midnight).
TIME_SLOT_BLOCKS = {
    "morning":   (8 * 60,  11 * 60),
    "lunch":     (11 * 60, 14 * 60),
    "afternoon": (12 * 60, 17 * 60),
    "dinner":    (17 * 60, 21 * 60),
    "evening":   (17 * 60, 22 * 60),
}
# Hour-of-day windows for the Places-API "is open during slot?" check.
TIME_SLOT_RANGES = {
    "morning":   (8, 11),
    "lunch":     (11, 14),
    "afternoon": (12, 17),
    "dinner":    (17, 22),
    "evening":   (17, 22),
}

_PARSED_ROW_KEYS = (
    "opening_min", "start_min", "depart_min", "dur_min",
    "travel_min", "transport_mode", "cost_amount",
)
_PARSED_INT_KEYS = {"opening_min", "start_min", "depart_min", "dur_min", "travel_min"}
_PARSED_FLOAT_KEYS = {"cost_amount"}
_MAPS_BASE = "https://maps.googleapis.com/maps/api"


def _fold(s: str) -> str:
    """NFKD-normalize, strip diacritics, lowercase. For accent-insensitive comparison."""
    if not s:
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


def _city_tokens(city_name: str) -> List[str]:
    """Split city name into accent-folded tokens, dropping punctuation. So 'Washington D.C.' → ['washington', 'dc']."""
    cleaned = re.sub(r"[^\w\s]", "", city_name or "")
    return [_fold(w) for w in cleaned.split() if len(w) > 1]


def _coerce_int(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except ValueError:
            return None
    return None


def _coerce_float(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_parsed_row(e):
    if not isinstance(e, dict):
        return {k: None for k in _PARSED_ROW_KEYS}
    out = {}
    for k in _PARSED_ROW_KEYS:
        v = e.get(k)
        if k in _PARSED_INT_KEYS:
            out[k] = _coerce_int(v)
        elif k in _PARSED_FLOAT_KEYS:
            out[k] = _coerce_float(v)
        else:  # transport_mode
            out[k] = v.strip().lower() if isinstance(v, str) else None
    return out


# --- Header validation / time-of-day classification -------------------------

def validate_header_row(rows: list, detected_idx: int) -> int:
    """Prefer row 0 if it has more column-keyword hits than the detected row."""
    if detected_idx == 0 or not rows:
        return detected_idx

    def hits(i):
        return sum(
            1 for c in rows[i].get("values", [])
            if any(kw in c.get("formattedValue", "").lower().strip() for kw in _ALL_COLUMN_KEYWORDS)
        )

    return 0 if hits(0) > hits(detected_idx) else detected_idx


def classify_time_of_day(time_value: str) -> Optional[str]:
    """Classify a Time of Day value as 'activity' or 'food'."""
    if not isinstance(time_value, str) or not time_value:
        return None
    if keywords_exact_match(time_value, ACTIVITY_TIME_KEYWORDS, substring=True):
        return "activity"
    if keywords_exact_match(time_value, FOOD_TIME_KEYWORDS, substring=True):
        return "food"
    return None


_DATE_FORMATS = (
    "%d %B %Y", "%d %b %Y",       # 20 May 2026, 20 May 2026
    "%B %d, %Y", "%b %d, %Y",     # May 20, 2026
    "%B %d %Y", "%b %d %Y",       # May 20 2026
    "%m/%d/%Y", "%d/%m/%Y",       # 5/20/2026, 20/5/2026
    "%Y-%m-%d", "%Y/%m/%d",       # 2026-05-20
)


_WEEKDAY_PREFIX_RE = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?[,\s-]+", re.IGNORECASE
)
_DAY_TYPE_SUFFIX_RE = re.compile(
    r"\s*\((Weekday|Weekend|Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\.?\)\s*$", re.IGNORECASE
)
_YEARLESS_FORMATS = ("%B %d", "%b %d", "%m/%d", "%d %B", "%d %b")


def parse_trip_date(date_str) -> Optional[_date]:
    """Parse a Date cell (e.g. '20 May 2026', 'May 20, 2026') to a date object.
    Accepts strings, datetime/date instances, and pandas Timestamp.
    Strips weekday prefixes ("Mon, ", "Monday, ") and assumes the current year
    when one isn't given. Returns None if the value is empty or unparseable.
    """
    if isinstance(date_str, _date) and not isinstance(date_str, _dt):
        return date_str
    if isinstance(date_str, _dt):
        return date_str.date()
    if hasattr(date_str, "to_pydatetime"):  # pandas Timestamp
        try:
            return date_str.to_pydatetime().date()
        except (ValueError, AttributeError):
            pass
    if not isinstance(date_str, str):
        return None
    s = date_str.strip()
    if not s:
        return None
    # If a Timestamp got stringified earlier, drop the trailing time component.
    s = s.split(" 00:00:00")[0]
    # Strip a leading weekday + separator, e.g. "Mon, May 20, 2026" → "May 20, 2026".
    s = _WEEKDAY_PREFIX_RE.sub("", s).strip()
    # Strip trailing day-type suffix, e.g. "May 5, 2026 (Weekday)" → "May 5, 2026".
    s = _DAY_TYPE_SUFFIX_RE.sub("", s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return _dt.strptime(s, fmt).date()
        except ValueError:
            continue
    # Year-less fallback: assume current year for "May 20"-style cells.
    current_year = _date.today().year
    for fmt in _YEARLESS_FORMATS:
        try:
            d = _dt.strptime(s, fmt)
            return d.replace(year=current_year).date()
        except ValueError:
            continue
    return None


# --- Gold file and column matching ------------------------------------------

def parse_gold_file(gold_path: str, model) -> Dict[str, str]:
    """Parse gold.txt into a trip-details dict via a single LLM call."""
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        return {}

    current_year = _date.today().year
    parsed = extract_json_with_llm(
        prompt=(
            "Extract the following details from this trip description. "
            f"The current year is {current_year}; if the description says 'this' "
            "month or otherwise omits a year, assume the trip is in the current year. "
            "Use that to compute day_type accurately. "
            "Return a JSON object with these keys: "
            '"people" (number of travelers as string), '
            '"city" (destination city), '
            '"month" (month of travel), '
            '"days" (number of days as string), '
            '"hotel" (hotel name, or empty string if not mentioned), '
            '"day_type" ("weekday", "weekend", or "mixed" — based on whether '
            "the trip falls on weekdays, weekends, or both).\n\n"
            f"{text}"
        ),
        model=model,
        expect_type="object",
    )
    return {k: str(v).strip() for k, v in parsed.items() if v is not None} if isinstance(parsed, dict) else {}


def classify_destinations_activity_food(names: List[str], model) -> List[str]:
    """Classify each destination name as 'activity' or 'food' via one LLM call."""
    if not names:
        return []
    numbered = "\n".join(f"{i+1}. {n or '(empty)'}" for i, n in enumerate(names))
    parsed = extract_json_with_llm(
        prompt=(
            "For each numbered destination below, classify it as either 'activity' or 'food'. "
            "A 'food' entry is a restaurant, cafe, bar, bakery, brewery, winery, food hall, or any "
            "place whose primary purpose is serving meals or drinks. Everything else "
            "(museums, parks, tours, landmarks, attractions, rentals, shops, experiences) is 'activity'. "
            "Return a JSON array of strings (each either \"activity\" or \"food\") in the same order.\n\n"
            f"{numbered}"
        ),
        model=model,
        expect_type="array",
    )
    if isinstance(parsed, list) and len(parsed) == len(names):
        return [
            "food" if str(p).strip().lower() == "food" else "activity"
            for p in parsed
        ]
    return ["activity"] * len(names)


def match_and_extract(df: pd.DataFrame, model) -> tuple:
    """Match columns and LLM-clean the Destination column in one pass.

    Returns:
        tuple: (matched_columns, dest_names, column_methods) where
            column_methods maps required column name -> "keyword"|"llm"
            recording which match_columns phase produced each hit.
    """
    all_matches, column_methods = match_columns(
        df, REQUIRED_COLUMNS, model=model, strict=True, parallel=True,
        max_workers=10, return_methods=True,
    )
    seen = set()
    matched_columns = {}
    for req, actual in all_matches.items():
        if actual not in seen:
            seen.add(actual)
            matched_columns[req] = actual

    dest_col = matched_columns.get("Destination")
    if not dest_col:
        return matched_columns, {}, column_methods

    total_rows = len(df)
    raw_values = [str(df.iloc[i].get(dest_col, "")).strip() for i in range(total_rows)]
    numbered = "\n".join(f"{i+1}. {v}" for i, v in enumerate(raw_values))

    parsed = extract_json_with_llm(
        prompt=(
            "Below is a numbered list of destination cell values from a travel itinerary spreadsheet. "
            "Clean each entry by:\n"
            "1. Strip the category prefix 'Activity:' or 'Food:' at the start.\n"
            "2. Strip parenthetical suffixes like '(Cuisine: Italian)' or '(Interactive Digital Art)'.\n"
            "3. Strip tour operator prefixes (e.g., 'Adventures Unbound:') to get the actual place name.\n"
            "4. Keep business/brand names that ARE the destination (e.g., 'Nightly Spirits - Ghost Tours & Pub Crawls').\n"
            "Examples:\n"
            "  'Activity: Adventures Unbound: Tidal Basin Pedal Boats' → 'Tidal Basin Pedal Boats'\n"
            "  'Activity: Adventures Unbound: Key Bridge Boathouse (Potomac Kayaking)' → 'Key Bridge Boathouse'\n"
            "  'Food: THE GRILL Washington DC' → 'THE GRILL Washington DC'\n"
            "  'Activity: ARTECHOUSE DC (Interactive Digital Art)' → 'ARTECHOUSE DC'\n"
            "  'Activity: Nightly Spirits - Ghost Tours & Pub Crawls' → 'Nightly Spirits - Ghost Tours & Pub Crawls'\n"
            "Return a JSON array of strings in the same order.\n\n"
            f"{numbered}"
        ),
        model=model,
        expect_type="array",
    )

    if isinstance(parsed, list) and len(parsed) == total_rows:
        return matched_columns, {i: str(n).strip() for i, n in enumerate(parsed)}, column_methods

    # Fallback: strip common category prefixes
    dest_names = {}
    for i, v in enumerate(raw_values):
        for prefix in ("Activity:", "Food:"):
            if v.lower().startswith(prefix.lower()):
                v = v[len(prefix):].strip()
                break
        dest_names[i] = v
    return matched_columns, dest_names, column_methods


# --- Small helpers ----------------------------------------------------------

def _query_name(n: str) -> str:
    """Return the first part of a compound name. Only splits on '&' when both sides look
    like distinct destinations (>=2 words each), so 'Capitol Building & Library of Congress'
    splits but 'Ben & Jerry's' does not."""
    parts = n.split("&", 1)
    if len(parts) == 2 and len(parts[0].split()) >= 2 and len(parts[1].split()) >= 2:
        return parts[0].strip()
    return n.strip()


def yes_no_task(tid: str, system: str, user: str) -> Dict:
    """Build a yes/no VLM task dict for run_vlm_batch()."""
    return {
        "id": tid,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ],
    }


def find_semantic_duplicates(names: List[str], model) -> List[List[str]]:
    """Find groups of names that refer to the same real-world place.

    First pass: exact match (case-insensitive). Second pass: LLM identifies
    semantically equivalent names (e.g. "Smithsonian" vs "Smithsonian Institution").

    Args:
        names: List of place names (destinations + alternatives).
        model: Loaded LLM model for semantic comparison.

    Returns:
        List of duplicate groups, where each group is a list of names that
        refer to the same place. Empty list if no duplicates found.
    """
    # Pass 1: exact case-insensitive dedup
    seen: Dict[str, str] = {}  # lowercase -> original
    exact_dupes: Dict[str, List[str]] = {}
    for name in names:
        if not name.strip():
            continue
        key = name.lower().strip()
        if key in seen:
            exact_dupes.setdefault(key, [seen[key]]).append(name)
        else:
            seen[key] = name

    # Pass 2: LLM semantic dedup on the unique names
    unique_names = list(seen.values())
    llm_dupes: List[List[str]] = []
    if len(unique_names) >= 2 and model:
        numbered = "\n".join(f"{i+1}. {n}" for i, n in enumerate(unique_names))
        result = extract_json_with_llm(
            prompt=(
                "Below is a numbered list of place names from a travel itinerary. "
                "Identify any groups where two or more names refer to the same "
                "real-world place (e.g. 'The Smithsonian' and 'Smithsonian Institution', "
                "or 'Central Park Zoo' and 'Central Park Wildlife Center').\n\n"
                "Return a JSON array of arrays. Each inner array contains the exact "
                "name strings (as written above) that refer to the same place. "
                "If no duplicates exist, return an empty array [].\n\n"
                f"{numbered}"
            ),
            model=model,
            expect_type="array",
        )
        if isinstance(result, list):
            for group in result:
                if isinstance(group, list) and len(group) >= 2:
                    # Validate that all names in the group are actually in our list
                    valid = [n for n in group if isinstance(n, str) and n in unique_names]
                    if len(valid) >= 2:
                        llm_dupes.append(valid)

    # Merge exact and LLM duplicates
    all_groups = list(exact_dupes.values()) + llm_dupes
    return all_groups


def empty_checkpoint(name: str, total: int, category: str = StepCategory.EXECUTION_ERROR) -> Checkpoint:
    """Build an empty checkpoint with a single 'no data' failure step whose max_score equals the total.

    Args:
        name: Checkpoint name.
        total: Checkpoint total points.
        category: StepCategory for the failure step (defaults to
            EXECUTION_ERROR: missing data prevented the checks from running).
    """
    cp = Checkpoint(total=total, result=0, name=name)
    cp.add_step(
        "Data Extraction", False, 1, "No data found in spreadsheet",
        score=0, max_score=total, category=category,
    )
    return cp


def format_failures(failures: list, limit: int = 5, prefix: str = "Failed", sep: str = "; ") -> str:
    """Build a trailing '. {prefix}: ...' string from a list of failure messages."""
    if not failures:
        return ""
    shown = failures[:limit] if limit else failures
    out = f". {prefix}: {sep.join(shown)}"
    if limit and len(failures) > limit:
        out += f" (+{len(failures) - limit} more)"
    return out


def add_fraction_step(checkpoint: Checkpoint, step_name: str, step_id: int,
                     passed: int, total: int, detail: str,
                     step_start: Optional[float] = None,
                     category: Optional[str] = None) -> None:
    """Add a step scored as passed/total out of 10 pts (standard evaluator pattern).

    Args:
        category: Optional StepCategory value naming the mechanism that
            decided the outcome; passed through to Checkpoint.add_step().
    """
    checkpoint.add_step(
        step_name, passed == total and total > 0, step_id, details=detail,
        score=calculate_percentage_score(passed, total, 10) if total else 0,
        max_score=10,
        execution_time=(time.time() - step_start) if step_start is not None else None,
        category=category,
    )


def is_travel_time_close(stated_min, api_min) -> bool:
    """True if stated ≈ api_min: short legs (≤5m) need ≤max(api+5, 10); else within 50%."""
    if stated_min is None or api_min is None:
        return False
    if api_min <= 5:
        return stated_min <= max(api_min + 5, 10)
    return api_min * 0.5 <= stated_min <= api_min * 1.5


# --- Review URL helpers -----------------------------------------------------

def extract_review_urls(df, matched_columns: Dict, raw_rows: list, header_row_idx: int) -> Dict[int, str]:
    """Extract the first review URL for each row from the raw sheet."""
    col_idx = get_column_index_by_name(df, "Review Link", matched_columns)
    if col_idx < 0:
        return {}
    urls = {}
    for idx in range(len(df)):
        found = find_urls_in_sheet(
            raw_rows, header_row_idx + 1 + idx, num_rows=1,
            start_col=col_idx, end_col=col_idx + 1,
        )
        if found:
            urls[idx] = found[0]
    return urls


def fetch_review_url_contents(all_review_urls: Dict[int, str], max_workers: int = 15, max_chars: int = 15000) -> Dict[str, tuple]:
    """Download page content for every verifiable review URL in parallel."""
    fetchable = {i: u for i, u in all_review_urls.items() if not is_unverifiable_url(u)}
    if not fetchable:
        return {}
    tasks = [
        {"id": f"url_{i}", "func": fetch_with_fallbacks, "args": (u,),
         "kwargs": {"max_chars": max_chars}}
        for i, u in fetchable.items()
    ]
    return parallel_execute(tasks, max_workers=max_workers)


# --- VLM batch runner / scoring ---------------------------------------------

def run_vlm_batch(vlm_tasks: list, model, max_workers: int = 20) -> Dict[str, bool]:
    """Execute VLM tasks in parallel. No retry: legitimate "No" answers and transient errors
    both surface as False from fast_parallel_vlm_calls, so retrying flips real Nos to Yes nondeterministically."""
    if not vlm_tasks:
        return {}
    return fast_parallel_vlm_calls(vlm_tasks, model, max_workers=max_workers)


def score_vlm_steps(checkpoint, vlm_results: Dict[str, bool], steps: list,
                    reasons: Optional[Dict[str, str]] = None,
                    seed_categories: Optional[Dict[str, str]] = None):
    """Score multiple VLM-verified steps from a shared results dict. `reasons[tid]` appended in [brackets].

    Args:
        seed_categories: Optional mapping of task id -> StepCategory for
            results that were seeded deterministically (API caches) rather
            than decided by the VLM. Task ids absent from the mapping are
            treated as LLM/VLM-decided. The step category is derived with
            StepCategory.aggregate() over the per-item (category, success)
            pairs.
    """
    reasons = reasons or {}
    seed_categories = seed_categories or {}
    for step_id, step_name, prefix, name_source, msg_template, empty_msg in steps:
        start = time.time()
        matching = [tid for tid in vlm_results if tid.startswith(prefix)]
        if not matching:
            checkpoint.add_step(
                step_name, False, step_id, details=empty_msg,
                score=0, max_score=10, execution_time=time.time() - start,
                category=StepCategory.EXECUTION_ERROR,
            )
            continue
        passed = sum(1 for tid in matching if vlm_results[tid])
        total = len(matching)
        failed = []
        for tid in matching:
            if vlm_results[tid]:
                continue
            idx = int(tid.split("_")[1])
            if isinstance(name_source, dict):
                name = name_source.get(idx, f"row {idx}")
            else:
                name = name_source[idx] if idx < len(name_source) else f"row {idx}"
            reason = reasons.get(tid)
            failed.append(f"{name} [{reason}]" if reason else name)
        detail = msg_template.format(passed=passed, total=total)
        if failed:
            detail += f". Failed: {', '.join(failed[:5])}"
        checkpoint.add_step(
            step_name, passed == total, step_id, details=detail,
            score=calculate_percentage_score(passed, total, 10),
            max_score=10, execution_time=time.time() - start,
            category=StepCategory.aggregate([
                (seed_categories.get(tid, StepCategory.LLM_VLM_JUDGEMENT),
                 vlm_results[tid])
                for tid in matching
            ]),
        )


# --- Cost color classification ----------------------------------------------

def expected_cost_color(amount: float) -> str:
    """Return expected color class for a cost amount."""
    return "green" if amount < 100 else ("yellow" if amount <= 200 else "orange")



# Google Sheets conditional-format comparator → operator map. NUMBER_BETWEEN
# uses two thresholds (low, high) and is handled separately below.
_COND_CMP = {
    "NUMBER_LESS": operator.lt,
    "NUMBER_LESS_THAN_EQ": operator.le,
    "NUMBER_GREATER": operator.gt,
    "NUMBER_GREATER_THAN_EQ": operator.ge,
}


def evaluate_cost_color_coding(sheet_raw: Dict, sheet_tab: Dict, row_data: List[Dict],
                               header_row_idx: int, cost_col_idx: int,
                               total_rows: int) -> Tuple[int, int, List[str]]:
    """Check cost cell colors vs expected; simulates conditional-format rules. Returns (correct, checked, failures)."""
    cost_rules = []
    for rule in sheet_tab.get("conditionalFormats", []):
        for rng in rule.get("ranges", []):
            if rng.get("startColumnIndex", -1) <= cost_col_idx < rng.get("endColumnIndex", 0):
                bool_rule = rule.get("booleanRule", {})
                cond = bool_rule.get("condition", {})
                cond_type = cond.get("type", "")
                cond_values = cond.get("values", [])
                color = classify_row_color(bool_rule.get("format", {}).get("backgroundColor", {}))
                if color == "none":
                    break
                if cond_type == "NUMBER_BETWEEN":
                    try:
                        low = float(cond_values[0].get("userEnteredValue", ""))
                        high = float(cond_values[1].get("userEnteredValue", ""))
                    except (ValueError, AttributeError, IndexError):
                        low = high = None
                    if low is not None and high is not None:
                        cost_rules.append({
                            "type": cond_type, "low": low, "high": high, "color": color,
                        })
                else:
                    try:
                        threshold = float(cond_values[0].get("userEnteredValue", "")) if cond_values else None
                    except (ValueError, AttributeError, IndexError):
                        threshold = None
                    if threshold is not None:
                        cost_rules.append({
                            "type": cond_type, "threshold": threshold, "color": color,
                        })
                break

    correct = checked = 0
    failures: List[str] = []
    for idx in range(total_rows):
        amount = row_data[idx].get("cost_amount") if idx < len(row_data) else None
        if amount is None or not math.isfinite(amount):
            continue
        checked += 1
        expected = expected_cost_color(amount)

        bg = get_background_color(sheet_raw, header_row_idx + 1 + idx, cost_col_idx)
        actual = classify_row_color(bg)

        if actual == "none":
            for rc in cost_rules:
                if rc["type"] == "NUMBER_BETWEEN":
                    if rc["low"] <= amount <= rc["high"]:
                        actual = rc["color"]
                        break
                else:
                    cmp = _COND_CMP.get(rc["type"])
                    if cmp and cmp(amount, rc["threshold"]):
                        actual = rc["color"]
                        break

        if actual == expected:
            correct += 1
        else:
            failures.append(f"Row {idx+1}: ${amount:.0f} expected {expected}, got {actual}")
    return correct, checked, failures


# --- LLM-based structured row extraction ------------------------------------

def extract_row_structured_data(df: pd.DataFrame, matched_columns: Dict, model) -> List[Dict]:
    """Parse times/duration/travel/mode/cost from every row via one LLM call."""
    total_rows = len(df)
    if total_rows == 0:
        return []

    def cell(key, i):
        col = matched_columns.get(key)
        return str(df.iloc[i].get(col, "")).strip() if col else ""

    row_lines = [
        f'Row {i+1}: '
        f'Opening="{cell("Opening Time", i)}", '
        f'Start="{cell("Start Time", i)}", '
        f'Depart="{cell("Departure Time", i)}", '
        f'Duration="{cell("Duration", i)}", '
        f'Travel="{cell("Travel Time", i)}", '
        f'Transport="{cell("Transportation Mode", i)}", '
        f'Cost="{cell("Cost", i)}"'
        for i in range(total_rows)
    ]

    parsed = extract_json_with_llm(
        prompt=(
            "For each row below, extract structured numeric data from the given text values. "
            "Return a JSON array with one object per row (in the same order) containing these keys:\n"
            "- opening_min (int or null): opening time parsed to minutes since midnight\n"
            "- start_min (int or null): start time in minutes since midnight\n"
            "- depart_min (int or null): departure time in minutes since midnight\n"
            "- dur_min (int or null): visit duration in minutes (midpoint for ranges like '1.5 - 2 hours')\n"
            "- travel_min (int or null): travel time in minutes\n"
            "- transport_mode (string or null): one of \"walking\", \"transit\", \"driving\", \"bicycling\", \"train\". "
            "Map metro/bus/subway/tram to \"transit\"; train/rail/amtrak to \"train\"; "
            "taxi/uber/lyft/car/drive to \"driving\"; bike/bicycle/cycling to \"bicycling\"; "
            "walk/on foot to \"walking\". "
            "For compound modes like \"Bus & Walk\", pick the primary motorized mode (\"transit\").\n"
            "- cost_amount (float or null): dollar amount (first number for ranges; 0 for free)\n\n"
            + "\n".join(row_lines)
        ),
        model=model,
        expect_type="array",
    )

    if not isinstance(parsed, list) or len(parsed) != total_rows:
        return [{k: None for k in _PARSED_ROW_KEYS} for _ in range(total_rows)]
    return [_coerce_parsed_row(e) for e in parsed]


# --- Google Maps API helpers ------------------------------------------------

def _find_place_candidate(input_text: str, fields: str, api_key: str,
                          timeout: int = 10) -> Optional[Dict]:
    """Call Places findplacefromtext; return the first candidate dict or None."""
    find = fetch_api_with_retry(
        f"{_MAPS_BASE}/place/findplacefromtext/json"
        f"?input={quote(input_text)}&inputtype=textquery&fields={fields}&key={api_key}",
        timeout=timeout,
    )
    if not find or find.get("status") != "OK" or not find.get("candidates"):
        return None
    return find["candidates"][0]


def get_place_details(name: str, city: str, api_key: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch Place Details via Places API; biased by city. Pair with get_canonical_address() for out-of-city detection."""
    if not name or not api_key:
        return None
    try:
        candidate = _find_place_candidate(f"{name} {city}", "place_id,name", api_key, timeout)
        if not candidate:
            return None
        pid = candidate['place_id']
        details = fetch_api_with_retry(
            f"{_MAPS_BASE}/place/details/json?place_id={pid}"
            f"&fields=name,types,price_level,editorial_summary,formatted_address,"
            f"user_ratings_total,opening_hours&key={api_key}",
            timeout=timeout,
        )
        if not details or details.get("status") != "OK":
            return None
        r = details.get("result", {})
        return {
            "place_id": pid,
            "name": r.get("name", ""),
            "types": r.get("types", []),
            "price_level": r.get("price_level"),
            "summary": r.get("editorial_summary", {}).get("overview", ""),
            "address": r.get("formatted_address", ""),
            "ratings_total": r.get("user_ratings_total", 0),
            "opening_periods": r.get("opening_hours", {}).get("periods", []),
        }
    except (KeyError, IndexError, ValueError):
        return None


def get_canonical_address(name: str, api_key: str, timeout: int = 10) -> Optional[Dict]:
    """Fetch Google's unbiased (name-only) best match as {name, address}, or None."""
    if not name or not api_key:
        return None
    try:
        c = _find_place_candidate(name, "name,formatted_address", api_key, timeout)
        if not c:
            return None
        return {"name": c.get("name", ""), "address": c.get("formatted_address", "")}
    except (KeyError, IndexError, ValueError):
        return None


def reverse_geocode(lat: str, lng: str, api_key: str, timeout: int = 10) -> Optional[str]:
    """Reverse geocode coordinates to a formatted address string."""
    if not api_key:
        return None
    try:
        resp = fetch_api_with_retry(
            f"{_MAPS_BASE}/geocode/json?latlng={lat},{lng}&key={api_key}",
            timeout=timeout,
        )
        if resp and resp.get("status") == "OK" and resp.get("results"):
            return resp["results"][0].get("formatted_address", "")
    except Exception:
        pass
    return None


def extract_url_coords(url: str) -> Optional[tuple]:
    """Extract (lat, lng) from a Google Maps URL.
    Prefers the protobuf !3d/!4d fields (actual place coords) over
    the @lat,lng viewport coords which may differ."""
    # Protobuf coords: !3d<lat>!4d<lng>
    m = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", url)
    if m:
        return (m.group(1), m.group(2))
    # Viewport coords fallback
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    return (m.group(1), m.group(2)) if m else None


def extract_url_place_name(url: str) -> Optional[str]:
    """Extract the place name from a Google Maps URL.
    Supports /place/Name, /search/Name, and ?q=Name formats."""
    from urllib.parse import unquote, urlparse, parse_qs
    # /maps/place/Name or /maps/search/Name
    m = re.search(r"/maps/(?:place|search)/([^/@?]+)", url)
    if m:
        return unquote(m.group(1).replace("+", " "))
    # maps.google.com/?q=Name
    parsed = urlparse(url)
    q = parse_qs(parsed.query).get("q", [None])[0]
    if q:
        return unquote(q.replace("+", " "))
    return None


def _normalize_addr(s: str) -> str:
    """Normalize an address for comparison: lowercase, strip punctuation/whitespace."""
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()


def _addr_city_match(addr_a: str, addr_b: str) -> bool:
    """Check if two Google-formatted addresses are in the same city/state area.
    Strips street address and zip code, compares city + state + country."""
    def _city_part(addr: str) -> str:
        parts = addr.split(",")
        # Drop street (first part), keep city/state/country
        tail = ",".join(parts[1:]) if len(parts) > 1 else addr
        # Strip zip codes (sequences of 5+ digits)
        tail = re.sub(r"\b\d{5,}\b", "", tail)
        return _normalize_addr(tail)
    a, b = _city_part(addr_a), _city_part(addr_b)
    return bool(a and b and (a == b or a in b or b in a))


def _names_match(a: str, b: str) -> bool:
    """Loose name-match: accent/case folded, 'the ' stripped, punctuation collapsed, substring either way."""
    def _norm(s: str) -> str:
        s = _fold(s).removeprefix("the ").strip()
        return re.sub(r"[^a-z0-9]+", " ", s).strip()
    a, b = _norm(a), _norm(b)
    return bool(a and b and (a == b or a in b or b in a))


def get_directions_travel_time(
    origin: str, destination: str, mode: str, api_key: str,
    timeout: int = 10, departure_time: Optional[int] = None,
) -> Optional[int]:
    """Directions API travel time in minutes (None on failure). `departure_time` is a Unix timestamp for day/hour-specific transit."""
    if not origin or not destination or not mode or not api_key:
        return None
    url = (
        f"{_MAPS_BASE}/directions/json"
        f"?origin={quote(origin)}&destination={quote(destination)}"
        f"&mode={mode}&key={api_key}"
    )
    if departure_time is not None:
        url += f"&departure_time={departure_time}"
    try:
        data = fetch_api_with_retry(url, timeout=timeout)
        if not data or data.get("status") != "OK" or not data.get("routes"):
            return None
        return round(data["routes"][0]["legs"][0]["duration"]["value"] / 60)
    except (KeyError, IndexError, ValueError):
        return None


def _llm_metro_check(name: str, addr: str, city: str, model) -> bool:
    """Ask the LLM if a place's address is within the metro area of the target city."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Answer only Yes or No."}]},
        {"role": "user", "content": [{"type": "text", "text":
            f"Place: {name}\nAddress: {addr}\n\n"
            f"Is this place located in or within the greater metropolitan area of {city}? "
            f"Answer Yes or No."}]},
    ]
    try:
        resp = model(messages).strip().lower()
        return resp.startswith("yes")
    except Exception:
        return False


def _future_same_weekday(d: _date) -> _date:
    """If d is today or in the past, shift to the next future occurrence of the same weekday."""
    today = _date.today()
    if d > today:
        return d
    days_ahead = (d.weekday() - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # must be strictly in the future
    return today + timedelta(days=days_ahead)


def _to_api_mode(mode: str) -> str:
    """Map internal mode to Directions API mode (train → transit; API doesn't accept 'train')."""
    return "transit" if mode == "train" else mode


def _hhmm_to_minutes(time_str: str) -> int:
    """Convert a Places API 'HHMM' string to minutes since midnight."""
    try:
        n = int(time_str)
        return (n // 100) * 60 + (n % 100)
    except (ValueError, TypeError):
        return 0


def _period_ranges(opening_periods):
    """Yield (open_min, close_min) for each Places API period."""
    for p in opening_periods or []:
        open_info = p.get("open", {})
        close_info = p.get("close", {})
        om = _hhmm_to_minutes(open_info.get("time", "0000"))
        cm = _hhmm_to_minutes(close_info.get("time", "2359")) if close_info else 1439
        yield om, cm


# --- Mid-level helpers (call tier-0 functions above) ------------------------

def count_rows_per_day(df: pd.DataFrame, date_col: str, row_types: List[str]) -> List[Dict]:
    """Group by date; return dicts {date, activity_count, food_count, parsed_date, is_weekend}."""
    if df is None or df.empty or not row_types or not date_col:
        return []
    d = df[[date_col]].copy()
    d["_rt"] = list(row_types[:len(d)]) + ["activity"] * max(0, len(d) - len(row_types))
    d[date_col] = d[date_col].replace("", pd.NA).ffill()
    d = d.dropna(subset=[date_col])
    results = []
    for date_val, group in d.groupby(date_col, sort=False):
        parsed = parse_trip_date(str(date_val))
        results.append({
            "date": str(date_val),
            "activity_count": int((group["_rt"] == "activity").sum()),
            "food_count": int((group["_rt"] == "food").sum()),
            "parsed_date": parsed,
            "is_weekend": (parsed.weekday() >= 5) if parsed else None,
        })
    return results


def is_open_during_slot(opening_periods: List[Dict], time_slot: str) -> Optional[bool]:
    """True if any period overlaps the time-of-day slot, False if none, None if absent."""
    if not opening_periods:
        return None
    slot_lower = (time_slot or "").lower().strip()
    slot_range = next((h for k, h in TIME_SLOT_RANGES.items() if k in slot_lower), None)
    if not slot_range:
        return None
    ss, se = slot_range[0] * 60, slot_range[1] * 60
    for om, cm in _period_ranges(opening_periods):
        if cm < om:  # overnight wrap
            if ss < cm or se > om:
                return True
        elif om < se and cm > ss:
            return True
    return False


def any_period_contains(opening_periods: List[Dict], minute: int) -> bool:
    """Return True if `minute` falls within any opening period's [open, close)."""
    if not opening_periods or minute is None:
        return False
    for om, cm in _period_ranges(opening_periods):
        if cm < om:  # overnight wrap
            if minute >= om or minute < cm:
                return True
        elif om <= minute < cm:
            return True
    return False


# --- Top-level orchestrators (called from evaluator.py) ---------------------

def build_task_context(
    df: pd.DataFrame,
    sheet_raw: Dict,
    raw_rows: list,
    header_row_idx: int,
    task_dir: str,
    model,
    gold_path: Optional[str] = None,
) -> Dict:
    """Build all pre-computed runtime state for the evaluator (returns a context dict)."""
    total_rows = len(df)

    # Phase 1: column matching + gold file parsing (parallel, both are LLM calls).
    resolved_gold_path = gold_path or os.path.join(task_dir, "data", "gold.txt")
    phase1 = parallel_execute(
        [
            {"id": "match", "func": match_and_extract, "args": (df, model)},
            {"id": "gold", "func": parse_gold_file,
             "args": (resolved_gold_path, model)},
        ],
        max_workers=2,
        timeout=60,
    )
    matched_columns, dest_names, matched_column_methods = phase1.get("match") or ({}, {}, {})
    trip_details = phase1.get("gold") or {}

    all_review_urls = extract_review_urls(df, matched_columns, raw_rows, header_row_idx)

    all_dest_names = [dest_names.get(i, "") for i in range(total_rows)]
    alt_col_name = matched_columns.get("Alternative Option")
    all_alt_names = (
        [str(df.iloc[i].get(alt_col_name, "")).strip() for i in range(total_rows)]
        if alt_col_name else []
    )

    # Phase 2: row parsing, type classification (dest + alt), review URL fetching (parallel).
    phase2 = parallel_execute(
        [
            {"id": "rows", "func": extract_row_structured_data, "args": (df, matched_columns, model)},
            {"id": "types", "func": classify_destinations_activity_food, "args": (all_dest_names, model)},
            {"id": "alt_types", "func": classify_destinations_activity_food, "args": (all_alt_names, model)},
            {"id": "urls", "func": fetch_review_url_contents, "args": (all_review_urls,)},
        ],
        max_workers=4,
        timeout=90,
    )
    parsed_rows = phase2.get("rows") or []
    row_types = phase2.get("types") or []
    all_alt_types = phase2.get("alt_types") or []
    url_content = phase2.get("urls") or {}

    time_col_name = matched_columns.get("Time of Day")
    date_col_name = matched_columns.get("Date")
    dates = (df[date_col_name].replace("", pd.NA).ffill()
             if date_col_name and date_col_name in df.columns else None)

    row_data: List[Dict] = []
    for idx in range(total_rows):
        tod = str(df.iloc[idx].get(time_col_name, "")).strip() if time_col_name else ""
        row_data.append({
            "date": (str(dates.iloc[idx]).strip()
                     if dates is not None and pd.notna(dates.iloc[idx])
                     else ""),
            "time_of_day": tod,
            "classification": classify_time_of_day(tod),
            "row_type": row_types[idx] if idx < len(row_types) else "activity",
            "dest_name": dest_names.get(idx, ""),
            **(parsed_rows[idx] if idx < len(parsed_rows) else {}),
        })

    # --- Maps API caches (guarded on GOOGLE_MAPS_API_KEY) ---
    place_cache: Dict = {}
    places_out_of_city: set = set()
    directions_cache: Dict = {}
    pairwise_transit_cache: Dict = {}
    alt_directions_cache: Dict = {}
    day_specific_transit_cache: Dict = {}
    day_return_transit_cache: Dict = {}
    url_geocode_cache: Dict = {}  # {row_idx: reverse-geocoded address from review URL coords}  # {last_idx_of_day: return-to-hotel minutes at last.depart_min}
    maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    if maps_api_key:
        city_name = trip_details.get("city", "")
        hotel = trip_details.get("hotel", "")
        city_words = _city_tokens(city_name)

        # Day-start indices (first row of each day uses hotel as origin)
        day_starts = set()
        prev_date = None
        day_groups: Dict[str, List[int]] = {}
        for idx, rd in enumerate(row_data):
            if rd["date"] != prev_date:
                day_starts.add(idx)
                prev_date = rd["date"]
            if rd["date"]:
                day_groups.setdefault(rd["date"], []).append(idx)

        # Single batch of all Maps API tasks to keep the worker pool saturated.
        all_tasks = []
        pw_task_keys: Dict[str, tuple] = {}  # tid -> (origin_lower, dest_lower)

        def leg_task(tid, origin, dest, mode, departure_time=None):
            task = {
                "id": tid, "func": get_directions_travel_time,
                "args": (f"{_query_name(origin)}, {city_name}",
                         f"{_query_name(dest)}, {city_name}",
                         _to_api_mode(mode), maps_api_key),
            }
            if departure_time is not None:
                task["kwargs"] = {"departure_time": departure_time}
            all_tasks.append(task)

        # Two Places queries per name: biased (name+city) and canonical (name-only)
        # — canonical detects out-of-city names that biased would mis-route locally.
        unique_names = {n for n in all_dest_names if n} | {n for n in all_alt_names if n}
        for n in unique_names:
            all_tasks.append({
                "id": f"place_{n}", "func": get_place_details,
                "args": (_query_name(n), city_name, maps_api_key),
            })
            all_tasks.append({
                "id": f"canon_{n}", "func": get_canonical_address,
                "args": (_query_name(n), maps_api_key),
            })

        # Per-row legs: consecutive (CP4), day-aware transit (CP6 S6), alt (CP4 S7).
        # dayleg departure_time = prev row's depart_min (when traveler actually left);
        # for day-start legs, fall back to this row's start_min - travel_min.
        day_last_idx = {indices[-1]: day for day, indices in day_groups.items()}
        for idx in range(total_rows):
            rd = row_data[idx]
            dest = rd["dest_name"]
            origin = hotel if idx in day_starts else (row_data[idx - 1]["dest_name"] if idx > 0 else "")
            mode = rd.get("transport_mode")
            if dest and origin and mode:
                leg_task(f"dir_{idx}", origin, dest, mode)
                parsed_date = parse_trip_date(rd["date"]) if rd["date"] else None
                if parsed_date:
                    if idx in day_starts:
                        sm, tm = rd.get("start_min"), rd.get("travel_min")
                        depart_min = (sm - tm) if isinstance(sm, (int, float)) and isinstance(tm, (int, float)) else sm
                    else:
                        depart_min = row_data[idx - 1].get("depart_min")
                        if depart_min is None:
                            sm, tm = rd.get("start_min"), rd.get("travel_min")
                            depart_min = (sm - tm) if isinstance(sm, (int, float)) and isinstance(tm, (int, float)) else sm
                    if isinstance(depart_min, (int, float)) and 0 <= depart_min < 1440:
                        hh, mm = int(depart_min) // 60, int(depart_min) % 60
                    else:
                        hh, mm = 12, 0
                    api_date = _future_same_weekday(parsed_date)
                    ts = int(_dt.combine(api_date, _time(hh, mm)).timestamp())
                    leg_task(f"dayleg_{idx}", origin, dest, mode, departure_time=ts)
                    # Return-to-hotel leg at the last activity of the day (last → hotel at last.depart_min).
                    # Use the first row of the day's mode (hotel→first stop) as the assumed return mode,
                    # since the last row's transport_mode describes how they ARRIVED, not how they leave.
                    if idx in day_last_idx and hotel:
                        last_depart = rd.get("depart_min")
                        if isinstance(last_depart, (int, float)) and 0 <= last_depart < 1440:
                            day_first_idx = day_groups[day_last_idx[idx]][0]
                            return_mode = row_data[day_first_idx].get("transport_mode") or mode
                            rhh, rmm = int(last_depart) // 60, int(last_depart) % 60
                            rts = int(_dt.combine(api_date, _time(rhh, rmm)).timestamp())
                            leg_task(f"dayreturn_{idx}", dest, hotel, return_mode, departure_time=rts)
            alt = all_alt_names[idx] if idx < len(all_alt_names) else ""
            if alt and dest:
                leg_task(f"altd_{idx}", dest, alt, "transit")

        # Reverse-geocode review URL coordinates for deterministic address matching.
        for idx, url in all_review_urls.items():
            coords = extract_url_coords(url)
            if coords:
                all_tasks.append({
                    "id": f"revgeo_{idx}",
                    "func": reverse_geocode,
                    "args": (coords[0], coords[1], maps_api_key),
                })

        # Pairwise transit legs (CP6 S1 TSP check)
        pw_seen = set()
        for day_indices in day_groups.values():
            day_names = [row_data[i]["dest_name"] for i in day_indices if row_data[i]["dest_name"]]
            points = ([hotel] if hotel else []) + day_names
            for a in points:
                for b in points:
                    if not a or not b or a == b:
                        continue
                    key = (a.lower(), b.lower())
                    if key in pw_seen:
                        continue
                    pw_seen.add(key)
                    tid = f"pw_{len(pw_task_keys)}"
                    pw_task_keys[tid] = key
                    leg_task(tid, a, b, "transit")

        all_results = parallel_execute(all_tasks, max_workers=30, timeout=120) if all_tasks else {}

        # Prefer biased's in-city result when its name matches the query (handles multi-city
        # franchises where canonical points at a different city's location). Fall back to
        # canonical when biased is missing or untrustworthy.
        metro_check_tasks = []  # LLM metro-area check for trusted-but-not-in-city names
        metro_check_data = {}   # name -> (best_result, addr)
        for n in unique_names:
            biased = all_results.get(f"place_{n}")
            canon = all_results.get(f"canon_{n}")
            canon_addr = _fold(canon.get("address", "") if canon else "")
            biased_addr = _fold(biased.get("address", "") if biased else "")
            biased_trusted = biased and _names_match(_query_name(n), biased.get("name", ""))
            canon_trusted = canon and _names_match(_query_name(n), canon.get("name", ""))
            biased_in_city = bool(biased and any(w in biased_addr for w in city_words))
            canon_in_city = bool(canon and any(w in canon_addr for w in city_words))
            if biased_trusted and biased_in_city:
                place_cache[n.lower()] = biased
            elif canon_trusted and canon_in_city:
                place_cache[n.lower()] = biased or canon
            elif (biased_trusted and not biased_in_city) or (canon_trusted and not canon_in_city):
                # Name matches but address doesn't contain city token — could be
                # a metro-area location (e.g. Giza for Cairo). LLM check.
                best = biased if biased_trusted else canon
                addr = best.get("address", "") if best else ""
                metro_check_data[n] = (best, addr)
                metro_check_tasks.append({
                    "id": f"metro_{n}",
                    "func": _llm_metro_check,
                    "args": (n, addr, city_name, model),
                })
            # else: untrusted match or no info → fall through to LLM existence check

        # Run metro-area LLM checks in parallel
        if metro_check_tasks:
            metro_results = parallel_execute(metro_check_tasks, max_workers=10, timeout=30)
            for n, (best, addr) in metro_check_data.items():
                if metro_results.get(f"metro_{n}"):
                    place_cache[n.lower()] = best
                else:
                    places_out_of_city.add(n.lower())
        for idx, prefix, cache in (
            (i, p, c) for i in range(total_rows)
            for p, c in (("dir", directions_cache),
                         ("altd", alt_directions_cache),
                         ("dayleg", day_specific_transit_cache),
                         ("dayreturn", day_return_transit_cache))
        ):
            m = all_results.get(f"{prefix}_{idx}")
            if m is not None:
                cache[idx] = m
        for tid, key in pw_task_keys.items():
            m = all_results.get(tid)
            if m is not None:
                pairwise_transit_cache[key] = m
        for idx in all_review_urls:
            m = all_results.get(f"revgeo_{idx}")
            if m is not None:
                url_geocode_cache[idx] = m

    return {
        "matched_columns": matched_columns,
        "matched_column_methods": matched_column_methods,
        "dest_names": dest_names,
        "trip_details": trip_details,
        "all_review_urls": all_review_urls,
        "all_dest_names": all_dest_names,
        "all_alt_names": all_alt_names,
        "all_alt_types": all_alt_types,
        "url_content": url_content,
        "row_data": row_data,
        "maps_api_key": maps_api_key,
        "place_cache": place_cache,
        "places_out_of_city": places_out_of_city,
        "directions_cache": directions_cache,
        "alt_directions_cache": alt_directions_cache,
        "pairwise_transit_cache": pairwise_transit_cache,
        "day_specific_transit_cache": day_specific_transit_cache,
        "day_return_transit_cache": day_return_transit_cache,
        "url_geocode_cache": url_geocode_cache,
    }


def build_cp3_cp4_vlm_results(ctx: Dict, model,
                              seed_categories: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """Seed deterministic results from caches, then run LLM fallback tasks in one batch. Returns (results, reasons).

    Args:
        seed_categories: Optional dict filled in place with task id ->
            StepCategory for results seeded without a VLM verdict (API
            caches, deterministic URL checks). Ids left out were VLM-decided.
    """
    if seed_categories is None:
        seed_categories = {}
    df = ctx["df"]
    matched_columns = ctx["matched_columns"]
    dest_names = ctx["dest_names"]
    all_dest_names = ctx["all_dest_names"]
    all_alt_names = ctx["all_alt_names"]
    all_alt_types = ctx.get("all_alt_types", [])
    row_data = ctx["row_data"]
    place_cache = ctx["place_cache"]
    places_out_of_city = ctx.get("places_out_of_city", set())
    directions_cache = ctx["directions_cache"]
    alt_directions_cache = ctx["alt_directions_cache"]
    url_content = ctx["url_content"]
    all_review_urls = ctx["all_review_urls"]
    url_geocode_cache = ctx.get("url_geocode_cache", {})
    maps_api_key = ctx.get("maps_api_key", "")
    trip_details = ctx["trip_details"]

    total_rows = len(df)
    city_name = trip_details.get("city", "the destination city")
    trip_month = trip_details.get("month", "")
    people_count = trip_details.get("people", "")
    hotel = trip_details.get("hotel", "")
    day_type = trip_details.get("day_type", "")

    col = {
        k: matched_columns.get(v) for k, v in {
            "dest": "Destination", "opening": "Opening Time", "cost": "Cost",
            "transport": "Transportation Mode", "travel": "Travel Time",
            "cuisine": "Cuisine", "tod": "Time of Day", "alt": "Alternative Option",
        }.items()
    }

    vlm_results: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}
    vlm_tasks = []
    _review_check_data: Dict[int, Dict] = {}  # stash for post-VLM review link assembly

    # --- CP3 Steps 1 & 2: destination / alternative exists in city ---
    exists_sys = (
        "Answer Yes or No only. Say Yes if the place exists as a real, visitable "
        "destination (attraction, landmark, park, restaurant, tour, rental service, "
        "activity, or business)."
    )
    for prefix, name_list in (("dest", all_dest_names), ("alt", all_alt_names)):
        for i, name in enumerate(name_list):
            if not name:
                # Empty cell → can't exist in any city. Mark as failure rather than
                # silently dropping it from the denominator.
                vlm_results[f"{prefix}_{i}"] = False
                seed_categories[f"{prefix}_{i}"] = StepCategory.DETERMINISTIC
                continue
            ln = name.lower()
            if ln in place_cache:
                vlm_results[f"{prefix}_{i}"] = True  # Places API confirmed in city
                seed_categories[f"{prefix}_{i}"] = StepCategory.DETERMINISTIC
                continue
            if ln in places_out_of_city:
                vlm_results[f"{prefix}_{i}"] = False  # Places API found it, but NOT in city
                seed_categories[f"{prefix}_{i}"] = StepCategory.DETERMINISTIC
                continue
            # Places API had no result → LLM fallback
            vlm_tasks.append(yes_no_task(
                f"{prefix}_{i}", exists_sys,
                f"Is '{name}' a real place, activity, or business located in or near "
                f"{city_name}? Answer Yes or No.",
            ))

    # --- CP3 Step 4 + CP4 Steps 1-7: per-row checks ---
    dest_col_name = matched_columns.get("Destination", "")
    for idx in range(total_rows):
        row = df.iloc[idx]
        name = dest_names.get(idx, "")
        if not name:
            name = str(row.get(dest_col_name, "")).strip() if dest_col_name else ""
        if not name:
            continue

        def cell(c):
            return str(row.get(c, "")).strip() if c else ""

        tod = cell(col["tod"])
        fetched = url_content.get(f"url_{idx}")
        excerpt = fetched[0][:2000] if isinstance(fetched, (list, tuple)) and fetched and fetched[0] else ""
        rd = row_data[idx]
        is_day_start = idx == 0 or (idx > 0 and rd.get("date") != row_data[idx - 1].get("date"))
        prev_name = hotel if is_day_start else (dest_names.get(idx - 1) or hotel)

        # CP3 Step 4: Open during visit time
        if tod:
            place = place_cache.get(name.lower())
            det = is_open_during_slot(place["opening_periods"], tod) \
                if place and place.get("opening_periods") else None
            if det is not None:
                vlm_results[f"open_{idx}"] = det
                seed_categories[f"open_{idx}"] = StepCategory.DETERMINISTIC
            else:
                opening_hours = cell(col["opening"])
                p = f"Place: {name}\nPlanned visit time: {tod}"
                if opening_hours:
                    p += f"\nStated opening hours: {opening_hours}"
                if trip_month:
                    p += f"\nMonth of visit: {trip_month}"
                p += (
                    f"\n\nWould '{name}' be open during the visit ({tod}"
                    f"{f' in {trip_month}' if trip_month else ''})? "
                    "Consider both daily hours and seasonal closures. Answer Yes or No."
                )
                vlm_tasks.append(yes_no_task(
                    f"open_{idx}",
                    "You are checking if a destination is open during the planned visit time. "
                    "Answer only Yes or No. Morning means roughly 8-11 AM, Afternoon 12-5 PM, "
                    "Evening 5-9 PM, Lunch 11 AM-2 PM, Dinner 6-10 PM.",
                    p,
                ))

        # CP4 Step 1: Opening hours accurate
        val = cell(col["opening"])
        if val:
            stated_min = rd.get("opening_min")
            place = place_cache.get(name.lower())
            if place and place.get("opening_periods") and stated_min is not None:
                vlm_results[f"hours_{idx}"] = any_period_contains(place["opening_periods"], stated_min)
                seed_categories[f"hours_{idx}"] = StepCategory.DETERMINISTIC
            else:
                p = f"Place: {name}\nStated opening time: {val}"
                if trip_month:
                    p += f"\nMonth of visit: {trip_month}"
                if day_type:
                    p += f"\nDay type: {day_type}"
                if rd.get("date"):
                    p += f"\nDay label: {rd['date']}"
                if excerpt:
                    p += f"\n\nWebsite content:\n{excerpt}"
                p += (
                    "\n\nThe stated time may be a single opening time (e.g., '4:00 PM') or a "
                    "range (e.g., '9 AM - 5 PM'). Considering the month, day type, and "
                    "any seasonal/weekday-vs-weekend variations, does this time fall within "
                    "or match the actual opening hours of this place on the visit day? "
                    "Answer Yes or No."
                )
                vlm_tasks.append(yes_no_task(
                    f"hours_{idx}",
                    "You are a travel fact-checker. Answer only Yes or No. "
                    "Take month, weekday/weekend, and seasonal closures into account. "
                    "Say Yes if the stated time falls within the place's real opening hours "
                    "on that day. Say No only if the place is clearly closed at that time.",
                    p,
                ))

        # CP4 Step 2: Cost accurate
        val = cell(col["cost"])
        if val:
            p = (
                f"Place: {name}\nGroup size: {people_count} people\n"
                f"Stated total cost for the group: {val}"
            )
            if excerpt:
                p += f"\n\nWebsite content:\n{excerpt}"
            p += (
                f"\n\nThis is the TOTAL cost for {people_count} people (not per person). "
                "For restaurants, this includes food, drinks, tax, and tip for the whole group. "
                "For activities, this is the total admission/rental for all group members. "
                "Is this total cost plausible? Answer Yes or No."
            )
            vlm_tasks.append(yes_no_task(
                f"cost_{idx}",
                "You are a travel cost estimator. Answer only Yes or No. "
                "Say Yes if the total group cost is within a reasonable range "
                "(within 10% margin). Say No only if clearly unreasonable.",
                p,
            ))

        # CP4 Steps 3 & 4: Transport mode / travel time
        mode_val = cell(col["transport"])
        travel_val = cell(col["travel"])
        route = (
            f"City: {city_name}\n"
            f"Route: from '{_query_name(prev_name)}' to '{_query_name(name)}'"
        )

        if mode_val:
            # Always LLM here — the Directions API route existing doesn't confirm
            # the specific stated mode (e.g. "train") is actually available in the city.
            vlm_tasks.append(yes_no_task(
                f"mode_{idx}",
                "You are a transportation expert. Answer only Yes or No. "
                "Be strict: say No if the exact stated mode (e.g. 'train') isn't "
                "a real public option in this city for this route.",
                f"{route}\nStated transportation mode: {mode_val}\n\n"
                f"Is '{mode_val}' a real, available way to travel between these two "
                f"places in {city_name}? Answer Yes or No.",
            ))

        if travel_val and mode_val:
            api_min = directions_cache.get(idx)
            stated_min = rd.get("travel_min")
            if api_min is not None and stated_min is not None:
                vlm_results[f"ttime_{idx}"] = is_travel_time_close(stated_min, api_min)
                seed_categories[f"ttime_{idx}"] = StepCategory.FUZZY_MATCH
            else:
                vlm_tasks.append(yes_no_task(
                    f"ttime_{idx}",
                    "You are a transportation and routing expert. Answer only Yes or No.",
                    f"{route}\nTransportation: {mode_val}\nStated travel time: {travel_val}\n\n"
                    "Is this travel time realistic (within 50% margin)? Answer Yes or No.",
                ))

        # CP4 Step 5: Cuisine (food rows only)
        if col["cuisine"] and rd.get("row_type") == "food":
            cuisine_val = cell(col["cuisine"])
            if cuisine_val:
                p = f"Restaurant: {name}\nStated cuisine type: {cuisine_val}"
                if excerpt:
                    p += f"\n\nWebsite content:\n{excerpt}"
                p += "\n\nIs the stated cuisine type accurate? Answer Yes or No."
                vlm_tasks.append(yes_no_task(
                    f"cuisine_{idx}",
                    "You are a restaurant fact-checker. Answer only Yes or No.",
                    p,
                ))

        # CP4 Step 6: Review link — 3-part deterministic check:
        #   1. Place name in URL matches destination (LLM compare)
        #   2. Reverse-geocoded URL coords match place_cache address
        #   3. Review tab flag (!9m1!1b1) present
        # /maps/search/ URLs resolve to the place page directly, so only
        # the name match is required (coords and review flag are implicit).
        url = all_review_urls.get(idx, "")
        if url:
            tid = f"review_{idx}"
            url_place_name = extract_url_place_name(url)
            is_search_url = "/maps/search/" in url or "?q=" in url
            place = place_cache.get(name.lower())
            place_addr = place.get("address", "") if place else ""
            url_addr = url_geocode_cache.get(idx, "")
            has_review_flag = "!9m1!1b1" in url
            has_coords = extract_url_coords(url) is not None

            # Check 1: Place name match (LLM)
            if url_place_name:
                vlm_tasks.append(yes_no_task(
                    f"review_name_{idx}",
                    "You are verifying that a review URL is relevant to a destination. "
                    "Answer only Yes or No.",
                    f"Destination: {name}\nName from URL: {url_place_name}\n\n"
                    "Does the URL name refer to the same place, or to one of the places "
                    "listed in the destination (if the destination is a compound name "
                    "like 'A & B')? Answer Yes or No.",
                ))

            # Check 2: Address match — same city/area (deterministic)
            addr_match = None
            if place_addr and url_addr:
                addr_match = _addr_city_match(place_addr, url_addr)

            # Stash partial results for post-VLM assembly
            _review_check_data[idx] = {
                "has_name": url_place_name is not None,
                "addr_match": addr_match,
                "has_review_flag": has_review_flag,
                "has_coords": has_coords,
                "is_search_url": is_search_url,
                "place_addr": place_addr,
                "url_addr": url_addr,
            }
        else:
            vlm_results[f"review_{idx}"] = False
            seed_categories[f"review_{idx}"] = StepCategory.DETERMINISTIC

        # CP4 Step 7: Alternative viable — same category (food↔food, activity↔activity),
        # open during the time slot, and reachable within 30 min transit.
        alt_val = cell(col["alt"])
        if alt_val:
            main_type = rd.get("row_type", "activity")
            alt_type = all_alt_types[idx] if idx < len(all_alt_types) else "activity"
            tid = f"altv_{idx}"
            if main_type != alt_type:
                vlm_results[tid] = False
                seed_categories[tid] = StepCategory.DETERMINISTIC
                reasons[tid] = f"category: main={main_type}, alt={alt_type}"
            else:
                alt_place = place_cache.get(alt_val.lower())
                alt_min = alt_directions_cache.get(idx)
                seeded = False
                if alt_place and alt_place.get("opening_periods") and alt_min is not None and tod:
                    open_ok = is_open_during_slot(alt_place["opening_periods"], tod)
                    if open_ok is not None:
                        viable = bool(open_ok and alt_min <= 30)
                        vlm_results[tid] = viable
                        seed_categories[tid] = StepCategory.DETERMINISTIC
                        seeded = True
                        if not viable:
                            parts = []
                            if not open_ok:
                                parts.append(f"closed at {tod}")
                            if alt_min > 30:
                                parts.append(f"{alt_min}min away (>30)")
                            reasons[tid] = "; ".join(parts)
                if not seeded:
                    reasons[tid] = "LLM judged infeasible (uncached)"
                    vlm_tasks.append(yes_no_task(
                        tid,
                        "You are a travel planning expert. Answer only Yes or No. "
                        "An alternative is viable if it (a) is the same category as the "
                        "main destination (food stop ↔ food stop, activity ↔ activity), "
                        "(b) can be done during the given time slot (place is open then), "
                        "AND (c) is easily reachable from the main destination by public transit.",
                        f"City: {city_name}\nMain destination: {name} ({main_type})\n"
                        f"Alternative option: {alt_val} ({alt_type})\nTime slot: {tod}\n\n"
                        f"Is '{alt_val}' a same-category substitute for '{name}', open during "
                        f"{tod}, and easily travelable by public transit in {city_name}? "
                        "Answer Yes or No.",
                    ))

    vlm_results.update(run_vlm_batch(vlm_tasks, model))

    # --- Post-VLM: assemble review link results from 3-part check ---
    for idx, data in _review_check_data.items():
        tid = f"review_{idx}"
        fail_parts = []

        # Check 1: Place name match (from VLM)
        name_match = vlm_results.pop(f"review_name_{idx}", None)
        if name_match is False:
            fail_parts.append("name mismatch")

        # /maps/search/ URLs resolve to the place page directly —
        # only the name match is required.
        if not data.get("is_search_url"):
            # Check 2: Address match (deterministic)
            if data["addr_match"] is False:
                fail_parts.append(f"address mismatch (place: {data['place_addr']}, url: {data['url_addr']})")
            elif data["addr_match"] is None and not data["has_coords"]:
                fail_parts.append("no coordinates in URL")

            # Check 3: Review tab flag (toggleable)
            if not data["has_review_flag"]:
                fail_parts.append("missing review tab parameter")

        vlm_results[tid] = len(fail_parts) == 0
        # Category: the review verdict combines a VLM name match with
        # deterministic address/flag checks. A failure is attributed to the
        # deterministic parts if any of them failed, else to the VLM name
        # mismatch; a pass is attributed to the VLM when it participated.
        if fail_parts:
            det_fail = any(p != "name mismatch" for p in fail_parts)
            seed_categories[tid] = (StepCategory.DETERMINISTIC if det_fail
                                    else StepCategory.LLM_VLM_JUDGEMENT)
        else:
            seed_categories[tid] = (StepCategory.LLM_VLM_JUDGEMENT
                                    if name_match is not None
                                    else StepCategory.DETERMINISTIC)
        if fail_parts:
            reasons[tid] = "; ".join(fail_parts)

    # Drop stale reasons for LLM-seeded altv tasks that ended up passing.
    for tid in list(reasons):
        if vlm_results.get(tid) is True:
            reasons.pop(tid, None)
    return vlm_results, reasons


# --- CP6 per-day seeding ----------------------------------------------------

def seed_cp6_route_and_daytype(
    day_keys: list,
    day_groups: Dict[str, List[int]],
    row_data: List[Dict],
    hotel_name: str,
    places_out_of_city: set,
    pairwise_cache: Dict,
    day_specific_transit_cache: Dict,
    city_name: str,
    day_return_transit_cache: Optional[Dict] = None,
    seed_categories: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, bool], list, Dict[str, str]]:
    """Seed CP6 route_{di} (TSP-optimal full-loop) and daytype_{di} (transit match + reasonable return). Returns (seeded, pending_tasks, reasons).

    Args:
        seed_categories: Optional dict filled in place with task id ->
            StepCategory for deterministically seeded results (ids handed to
            the VLM fallback are left out).
    """
    results: Dict[str, bool] = {}
    reasons: Dict[str, str] = {}
    vlm_tasks: list = []
    day_return_transit_cache = day_return_transit_cache or {}
    if seed_categories is None:
        seed_categories = {}

    def route_time(seq):
        total = 0
        for a, b in zip(seq, seq[1:]):
            t = pairwise_cache.get((a.lower(), b.lower()))
            if t is None:
                return None
            total += t
        return total

    # Full hotel→...→hotel loop when the hotel is known.
    def full_loop(seq):
        return ([hotel_name] + list(seq) + [hotel_name]) if hotel_name else list(seq)

    for di, day in enumerate(day_keys):
        indices = day_groups[day]
        dests = [row_data[i]["dest_name"] for i in indices if row_data[i]["dest_name"]]

        # --- Route order ---
        day_has_out_of_city = any(
            row_data[i]["dest_name"] and row_data[i]["dest_name"].lower() in places_out_of_city
            for i in indices
        )
        seeded = True  # route_{di} will be deterministically set unless we explicitly defer to LLM
        if day_has_out_of_city:
            results[f"route_{di}"] = False
            seed_categories[f"route_{di}"] = StepCategory.DETERMINISTIC
        elif len(dests) == 0:
            results[f"route_{di}"] = False
            seed_categories[f"route_{di}"] = StepCategory.DETERMINISTIC
            reasons[f"route_{di}"] = "day has no destinations"
        elif len(dests) == 1:
            results[f"route_{di}"] = True
            seed_categories[f"route_{di}"] = StepCategory.DETERMINISTIC
        elif len(dests) <= 7:
            seeded = False
            actual_total = route_time(full_loop(dests))
            if actual_total is not None:
                best = None
                for perm in permutations(dests):
                    t = route_time(full_loop(perm))
                    if t is not None and (best is None or t < best):
                        best = t
                if best is not None and best > 0:
                    results[f"route_{di}"] = actual_total <= best * 1.4
                    seed_categories[f"route_{di}"] = StepCategory.FUZZY_MATCH
                    seeded = True
        else:
            # >7 stops: n! would lock the evaluator. Defer to LLM.
            seeded = False
        if not seeded:
            sequence = "\n".join(f"  {j+1}. {d}" for j, d in enumerate(dests))
            vlm_tasks.append(yes_no_task(
                f"route_{di}",
                f"You are a travel routing expert for {city_name}. Answer only Yes or No. "
                "The itinerary is a round trip from the hotel: hotel → stops → hotel. "
                "Restaurants may not be in the same neighborhood as activities — that is "
                "acceptable as long as the day doesn't involve excessive back-and-forth. "
                "Say Yes if the order is reasonable, even if not perfectly optimal.",
                f"City: {city_name}\nDay: {day}\nHotel: {hotel_name or '(unknown)'}\n"
                f"Itinerary order (round trip from hotel):\n{sequence}\n\n"
                "Is this a reasonable geographic order for a day of sightseeing and dining, "
                "without excessive backtracking? Answer Yes or No.",
            ))

        # --- Day-aware transit accuracy + reasonable return to hotel ---
        tid = f"daytype_{di}"
        if parse_trip_date(day) is None:
            results[tid] = False
            seed_categories[tid] = StepCategory.DETERMINISTIC
            reasons[tid] = "date unparseable"
        else:
            compared = passed = 0
            off_legs: list = []
            for i in indices:
                stated = row_data[i]["travel_min"]
                api_min = day_specific_transit_cache.get(i)
                if stated is None or api_min is None:
                    continue
                compared += 1
                if is_travel_time_close(stated, api_min):
                    passed += 1
                else:
                    dest_label = row_data[i].get("dest_name") or f"Row {i+1}"
                    off_legs.append(f"{dest_label} stated {stated}min vs API {api_min}min")
            # For short days (<=2 comparable legs), require all legs match exactly so
            # a single-leg day cannot trivially pass. For longer days, allow 1 off-leg.
            strict = compared <= 2
            min_compared = max(2, len(indices) // 2)
            ok = (
                compared >= min_compared
                and (passed == compared if strict else passed >= compared - 1)
            )
            results[tid] = ok
            seed_categories[tid] = StepCategory.FUZZY_MATCH
            if not ok:
                parts = []
                if compared == 0:
                    parts.append("no legs have stated+API data")
                elif compared < (len(indices) // 2):
                    parts.append(f"only {compared}/{len(indices)} legs comparable")
                if passed < compared - 1 and off_legs:
                    parts.append(f"{compared - passed} off: {'; '.join(off_legs[:3])}")
                reasons[tid] = "; ".join(parts) or "unknown"

    return results, vlm_tasks, reasons
