"""
Utility functions for the sheets_7_running_analysis task.

Contains unit conversion functions for distance and speed, date normalization,
chart identification, and other evaluation helpers.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.browsergym.knows.eval.eval_utils.chart_utils import (
    find_chart_by_metadata
)
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_text_content


# =============================================================================
# Prompt Building Helpers
# =============================================================================

def build_keyword_match_prompt(text: str, keywords: List[str], description: str) -> str:
    """Build a yes/no VLM prompt that asks whether a text matches a set of keywords.

    Used by grade_checkpoint_N() functions to construct consistent prompts for
    fast_parallel_vlm_calls() when matching chart axis labels and titles.

    Args:
        text: The text to evaluate (e.g. an axis label or chart title).
        keywords: List of concept keywords the text should match.
        description: Human-readable context string appended to the prompt.

    Returns:
        A prompt string suitable for a yes/no LLM call.
    """
    return (
        f"Is the text '{text}' a short, descriptive label whose primary purpose is to indicate "
        f"any of these concepts: {', '.join(keywords)}? "
        f"Context: {description}. "
        f"A source citation, URL, or long explanatory note should be answered No. "
        f"Answer only Yes or No."
    )


# =============================================================================
# Unit Conversion Functions
# =============================================================================

def meters_to_miles(meters: float) -> float:
    """Convert meters to miles."""
    return meters / 1609.344


def meters_to_km(meters: float) -> float:
    """Convert meters to kilometers."""
    return meters / 1000.0


def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km / 1.60934


def ms_to_min_per_mile(speed_ms: float) -> float:
    """Convert speed from m/s to min/mile pace."""
    if speed_ms <= 0:
        return float('inf')
    return 26.8224 / speed_ms


def ms_to_kmh(speed_ms: float) -> float:
    """Convert speed from m/s to km/h."""
    return speed_ms * 3.6


def min_per_km_to_min_per_mile(min_per_km: float) -> float:
    """Convert pace from min/km to min/mile.

    Args:
        min_per_km: Pace in minutes per kilometer.

    Returns:
        Pace in minutes per mile.
    """
    # 1 mile = 1.60934 km, so min/mile = min/km * 1.60934
    return min_per_km * 1.60934


def mph_to_min_per_mile(mph: float) -> float:
    """Convert speed from miles per hour to min/mile pace.

    Args:
        mph: Speed in miles per hour.

    Returns:
        Pace in minutes per mile.
    """
    if mph <= 0:
        return float('inf')
    return 60.0 / mph


def kmh_to_min_per_mile(kmh: float) -> float:
    """Convert speed from km/h to min/mile pace.

    Args:
        kmh: Speed in kilometers per hour.

    Returns:
        Pace in minutes per mile.
    """
    if kmh <= 0:
        return float('inf')
    # Convert km/h to miles/h, then to min/mile
    mph = kmh / 1.60934
    return 60.0 / mph


def marathon_time_to_min_per_mile(hours: float, minutes: float, seconds: float) -> float:
    """Convert marathon finish time to min/mile pace.

    Args:
        hours: Hours component of finish time.
        minutes: Minutes component of finish time.
        seconds: Seconds component of finish time.

    Returns:
        Pace in minutes per mile.
    """
    total_minutes = hours * 60 + minutes + seconds / 60
    marathon_miles = 26.2188  # Official marathon distance in miles
    return total_minutes / marathon_miles


def race_time_to_min_per_mile(total_minutes: float, distance_miles: float) -> float:
    """Convert race time to min/mile pace.

    Args:
        total_minutes: Total race time in minutes.
        distance_miles: Race distance in miles.

    Returns:
        Pace in minutes per mile.
    """
    if distance_miles <= 0:
        return float('inf')
    return total_minutes / distance_miles


def convert_pace_to_min_per_mile(value: float, unit: str) -> Tuple[Optional[float], str]:
    """Convert a pace/speed value from various units to min/mile.

    Args:
        value: The numeric pace or speed value.
        unit: The unit of the value. Supported units:
            - "min/mile" or "min_per_mile": Already in target format
            - "min/km" or "min_per_km": Minutes per kilometer
            - "mph" or "miles_per_hour": Miles per hour
            - "kmh" or "km/h" or "km_per_hour": Kilometers per hour
            - "m/s" or "ms" or "meters_per_second": Meters per second
            - "marathon_time_minutes": Total marathon time in minutes
            - "5k_time_minutes": Total 5K time in minutes

    Returns:
        Tuple of (pace_in_min_per_mile or None, details string)
    """
    unit_lower = unit.lower().strip()

    try:
        if unit_lower in ["min/mile", "min_per_mile", "minutes_per_mile"]:
            return value, f"Already in min/mile: {value:.2f}"

        elif unit_lower in ["min/km", "min_per_km", "minutes_per_km"]:
            result = min_per_km_to_min_per_mile(value)
            return result, f"Converted {value:.2f} min/km to {result:.2f} min/mile"

        elif unit_lower in ["mph", "miles_per_hour"]:
            result = mph_to_min_per_mile(value)
            return result, f"Converted {value:.2f} mph to {result:.2f} min/mile"

        elif unit_lower in ["kmh", "km/h", "km_per_hour", "kph"]:
            result = kmh_to_min_per_mile(value)
            return result, f"Converted {value:.2f} km/h to {result:.2f} min/mile"

        elif unit_lower in ["m/s", "ms", "meters_per_second"]:
            result = ms_to_min_per_mile(value)
            return result, f"Converted {value:.2f} m/s to {result:.2f} min/mile"

        elif unit_lower == "marathon_time_minutes":
            marathon_miles = 26.2188
            result = race_time_to_min_per_mile(value, marathon_miles)
            return result, f"Converted {value:.1f} min marathon to {result:.2f} min/mile"

        elif unit_lower == "5k_time_minutes":
            five_k_miles = 3.10686  # 5km in miles
            result = race_time_to_min_per_mile(value, five_k_miles)
            return result, f"Converted {value:.1f} min 5K to {result:.2f} min/mile"

        elif unit_lower == "half_marathon_time_minutes":
            half_marathon_miles = 13.1094  # Half marathon distance in miles
            result = race_time_to_min_per_mile(value, half_marathon_miles)
            return result, f"Converted {value:.1f} min half-marathon to {result:.2f} min/mile"

        else:
            return None, f"Unknown unit: {unit}"

    except Exception as e:
        return None, f"Conversion error: {str(e)}"


# =============================================================================
# Date Normalization
# =============================================================================

def normalize_date(date_str) -> str:
    """
    Normalize date string for comparison (date-only).

    Handles formats like:
    - "Sep 27, 2021, 2:13:42 AM" (gold format with timestamp)
    - "9/27/2021" (sheet format, date-only)
    - "2021-09-27 02:13:42"
    - Various other common formats

    Args:
        date_str: Date string to normalize.

    Returns:
        Normalized date string in format "YYYY-MM-DD" (date-only for comparison).
    """
    if pd.isna(date_str):
        return ""

    date_str = str(date_str).strip()

    formats_to_try = [
        "%b %d, %Y, %I:%M:%S %p",  # Gold format: "Sep 27, 2021, 2:13:42 AM"
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%b %d, %Y, %H:%M:%S",
        "%Y-%m-%d %I:%M:%S %p",
        "%m/%d/%Y",                 # Sheet format: "9/27/2021"
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%b %d, %Y",
    ]

    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Fallback to lowercase string
    return date_str.lower()


# =============================================================================
# Data Loading
# =============================================================================

def load_gold_run_activities(csv_path: str, activity_type: str = 'Run'):
    """Load gold data, filtering only the requested activity type.

    The CSV has duplicate 'Distance' columns - pandas renames them to 'Distance' and 'Distance.1'.
    - 'Distance' is in km (rounded)
    - 'Distance.1' is in meters (more precise)

    We normalize to use the meters column converted to km for consistency.

    Args:
        csv_path: Path to the gold_activities.csv file.
        activity_type: Strava activity type to filter on (e.g. 'Run', 'Nordic Ski'). Defaults to 'Run' for backward compatibility with instances 1 and 2.

    Returns:
        DataFrame containing only activities of the requested type with normalized Distance column.
    """
    gold_df = pd.read_csv(csv_path)
    runs = gold_df[gold_df['Activity Type'] == activity_type].copy()

    # Use Distance.1 (meters) converted to km for more precision
    # This fixes discrepancies like "Anas run" where Distance=0.44 km but Distance.1=448.1 m
    if 'Distance.1' in runs.columns:
        runs['Distance'] = runs['Distance.1'] / 1000.0

    return runs


# =============================================================================
# Chart Identification
# =============================================================================

def find_speed_chart_by_metadata(
    charts: List[Dict[str, Any]],
    matched_columns: Optional[Dict[str, str]],
    df: Optional[pd.DataFrame],
    model: Any = None
) -> Optional[Dict[str, Any]]:
    """
    Identify the speed chart using metadata (no VLM/image analysis).

    Uses the general find_chart_by_metadata() with speed-specific keywords.

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        matched_columns: Column mapping from checkpoint 2 (may contain "Speed (min/mile)")
        df: DataFrame with sheet data
        model: Optional LLM model for fallback matching

    Returns:
        Chart object or None if not found
    """
    return find_chart_by_metadata(
        charts=charts,
        title_keywords=['speed', 'pace', 'running speed', 'min/mile', 'min per mile'],
        y_axis_keywords=['speed', 'pace', 'min/mile', 'minute'],
        title_description="chart title related to running speed or pace",
        axis_description="Y-axis label related to running speed or pace",
        matched_columns=matched_columns,
        column_name="Speed (min/mile)",
        df=df,
        model=model
    )


def find_cumulative_chart_by_metadata(
    charts: List[Dict[str, Any]],
    matched_columns: Optional[Dict[str, str]],
    df: Optional[pd.DataFrame],
    model: Any = None
) -> Optional[Dict[str, Any]]:
    """
    Identify the cumulative distance chart using metadata.

    Uses the general find_chart_by_metadata() with cumulative/distance keywords.

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        matched_columns: Column mapping (may contain cumulative distance column)
        df: DataFrame with sheet data
        model: Optional LLM model for fallback matching

    Returns:
        Chart object or None if not found
    """
    return find_chart_by_metadata(
        charts=charts,
        title_keywords=['cumulative', 'cumulative distance', 'distance over time'],
        y_axis_keywords=['cumulative', 'cumulative distance', 'distance', 'miles'],
        title_description="chart title related to cumulative distance",
        axis_description="Y-axis label related to cumulative distance",
        matched_columns=matched_columns,
        column_name=None,  # No specific column name for cumulative
        df=df,
        model=model
    )


def find_daily_miles_chart_by_metadata(
    charts: List[Dict[str, Any]],
    matched_columns: Optional[Dict[str, str]],
    df: Optional[pd.DataFrame],
    model: Any = None
) -> Optional[Dict[str, Any]]:
    """
    Identify the daily total miles chart using metadata.

    Uses the general find_chart_by_metadata() with daily miles keywords.

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        matched_columns: Column mapping (may contain distance column)
        df: DataFrame with sheet data
        model: Optional LLM model for fallback matching

    Returns:
        Chart object or None if not found
    """
    return find_chart_by_metadata(
        charts=charts,
        title_keywords=['daily', 'total miles', 'daily miles', 'miles per day', 'daily total'],
        y_axis_keywords=['miles', 'distance', 'total miles', 'total'],
        title_description="chart title related to daily total miles",
        axis_description="Y-axis label related to daily miles or distance",
        matched_columns=matched_columns,
        column_name=None,
        df=df,
        model=model
    )


def find_sets_chart_by_metadata(
    charts: List[Dict[str, Any]],
    matched_columns: Optional[Dict[str, str]],
    df: Optional[pd.DataFrame],
    model: Any = None
) -> Optional[Dict[str, Any]]:
    """
    Identify the workout sets chart using metadata.

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        matched_columns: Column mapping (may contain Total Sets column)
        df: DataFrame with sheet data
        model: Optional LLM model for fallback matching

    Returns:
        Chart object or None if not found
    """
    return find_chart_by_metadata(
        charts=charts,
        title_keywords=['sets', 'workout', 'daily sets', 'sets per workout', 'workout sets'],
        y_axis_keywords=['sets', 'total sets', 'workout sets'],
        title_description="chart title related to workout sets over time",
        axis_description="Y-axis label related to workout sets",
        matched_columns=matched_columns,
        column_name="Total Sets",
        df=df,
        model=model
    )


def find_cumulative_reps_chart_by_metadata(
    charts: List[Dict[str, Any]],
    matched_columns: Optional[Dict[str, str]],
    df: Optional[pd.DataFrame],
    model: Any = None
) -> Optional[Dict[str, Any]]:
    """
    Identify the cumulative reps chart using metadata.

    Args:
        charts: List of chart objects from extract_charts_from_sheet()
        matched_columns: Column mapping
        df: DataFrame with sheet data
        model: Optional LLM model for fallback matching

    Returns:
        Chart object or None if not found
    """
    return find_chart_by_metadata(
        charts=charts,
        title_keywords=['cumulative', 'reps', 'total reps', 'cumulative reps'],
        y_axis_keywords=['cumulative', 'reps', 'total reps'],
        title_description="chart title related to cumulative reps",
        axis_description="Y-axis label related to cumulative reps",
        matched_columns=matched_columns,
        column_name=None,
        df=df,
        model=model
    )


def validate_cumulative_against_sheet(
    chart_values: List[float],
    df: pd.DataFrame,
    matched_columns: Dict[str, str],
    tolerance_percent: float = 5.0,
    match_threshold: float = 0.8
) -> Tuple[bool, str]:
    """
    Validate chart cumulative values against expected cumulative sum from sheet data.

    Uses the already-validated distance column from the spreadsheet (matched in checkpoint 2)
    to compute expected cumulative values. This ensures consistency with checkpoint 2 validation
    and avoids unit conversion issues.

    Checks:
    1. Values are monotonically increasing (cumulative pattern)
    2. Data point count matches expected count
    3. All cumulative values match expected values within tolerance

    Args:
        chart_values: List of values from chart series
        df: DataFrame containing the sheet data
        matched_columns: Column mapping from checkpoint 2 (should contain "Distance (Miles)")
        tolerance_percent: Allowed error for each value match (default 5%)
        match_threshold: Minimum proportion of values that must match (default 0.8 = 80%)

    Returns:
        tuple: (is_valid: bool, details: str)
    """
    if not chart_values:
        return False, "No chart values provided"

    # Check 1: Monotonically increasing (cumulative pattern)
    # Allow small tolerance for floating point comparison
    non_increasing_count = 0
    for i in range(1, len(chart_values)):
        if chart_values[i] < chart_values[i-1] - 0.01:  # Small tolerance
            non_increasing_count += 1

    if non_increasing_count > 0:
        return False, f"Values not monotonically increasing ({non_increasing_count} decreases found)"

    # Get the distance column from matched_columns (already validated in checkpoint 2)
    if df is None or df.empty:
        return False, "No sheet data available"

    if not matched_columns:
        return False, "No matched columns available"

    # Try to find distance column - prefer miles, fall back to general distance
    dist_col = matched_columns.get("Distance (Miles)") or matched_columns.get("Distance")
    if not dist_col or dist_col not in df.columns:
        return False, f"Distance column not found in matched columns: {list(matched_columns.keys())}"

    # Extract distance values and calculate expected cumulative sum
    try:
        distances = pd.to_numeric(df[dist_col], errors='coerce').dropna().values
    except Exception as e:
        return False, f"Error extracting distance values: {str(e)}"

    if len(distances) == 0:
        return False, "No valid distance values found in sheet"

    # Calculate expected cumulative distances
    expected_cumulative = []
    running_total = 0
    for d in distances:
        running_total += d
        expected_cumulative.append(running_total)

    expected_count = len(expected_cumulative)
    actual_count = len(chart_values)

    # Check 2: Data point count matches
    if actual_count != expected_count:
        count_diff_percent = abs(actual_count - expected_count) / expected_count * 100
        if count_diff_percent > 10:  # Allow 10% variance in count
            return False, f"Data point count {actual_count} differs significantly from expected {expected_count}"

    # Check 3: Validate all cumulative values against expected data
    # Compare each value to its expected counterpart
    comparison_count = min(actual_count, expected_count)
    matches = 0
    mismatches = []

    for i in range(comparison_count):
        actual_val = chart_values[i]
        expected_val = expected_cumulative[i]

        if expected_val > 0:
            error_percent = abs(actual_val - expected_val) / expected_val * 100
            if error_percent <= tolerance_percent:
                matches += 1
            else:
                if len(mismatches) < 5:  # Only track first 5 mismatches for reporting
                    mismatches.append(f"Point {i+1}: {actual_val:.1f} vs expected {expected_val:.1f} ({error_percent:.1f}% diff)")
        elif actual_val == 0:
            matches += 1
        else:
            if len(mismatches) < 5:
                mismatches.append(f"Point {i+1}: {actual_val:.1f} vs expected 0")

    match_rate = matches / comparison_count if comparison_count > 0 else 0

    if match_rate < match_threshold:
        mismatch_summary = "; ".join(mismatches[:3])
        return False, f"Only {matches}/{comparison_count} values match ({match_rate:.0%}). Examples: {mismatch_summary}"

    # All checks passed
    actual_final = chart_values[-1]
    expected_total = expected_cumulative[-1] if expected_cumulative else 0
    return True, f"Cumulative data valid: {matches}/{comparison_count} values match ({match_rate:.0%}), final value {actual_final:.1f} miles (expected {expected_total:.1f})"


# =============================================================================
# Website Content Extraction for Checkpoint 5
# =============================================================================

def extract_pace_from_url(
    url: str,
    pace_type: str,
    model: Any,
    timeout: int = 10
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract running pace from a webpage using LLM in structured format.

    Fetches the page content and uses an LLM to extract pace/speed data
    in a structured format with value and unit. Does NOT perform conversion.

    Args:
        url: URL to fetch and analyze
        pace_type: Either "male_5k" or "kipchoge"
        model: LLM model for extraction
        timeout: Request timeout in seconds

    Returns:
        Tuple of (structured_pace_data or None, details: str)
        structured_pace_data is a dict with keys:
            - "value": float - the numeric pace/speed value
            - "unit": str - the unit (e.g., "min/mile", "min/km", "mph", "km/h", "marathon_time_minutes", "5k_time_minutes")
            - "raw_text": str - the original text extracted from the page
    """
    # Fetch page content
    content, fetch_status = fetch_page_text_content(url, timeout)

    if not content:
        return None, f"Failed to fetch URL: {fetch_status}"

    # Build prompt based on pace type
    if pace_type == "male_5k":
        extraction_prompt = """Extract the average running pace for a 5K race for males (especially around age 25-30).
Look for pace in any format: min/mile, min/km, mph, km/h, or total 5K time.
Report the value exactly as found on the page with its original unit."""
    elif pace_type == "kipchoge":
        extraction_prompt = """Extract Eliud Kipchoge's marathon pace or marathon finish time.
Look for his fastest marathon times (around 2:01-2:02 range) or his pace in any format.
Report the value exactly as found on the page with its original unit."""
    elif pace_type == "intermediate_male_30_half_marathon":
        extraction_prompt = """Extract the average running pace for a half-marathon for an intermediate male runner at age 30.

The page may contain MULTIPLE pace tables stacked together (e.g. one labeled "Pace (min/km)" and one labeled "Pace (min/mile)"), each with the same Age x Ability layout. You MUST prefer the min/mile table; do NOT take the min/km value. Also do NOT use the Finish Time table (HH:MM:SS values).

Specifically: locate the Male section, then within it the table whose header indicates "min/mile" (per-mile pace), then read the value at row "30" (or "Age 30") and column "Intermediate". Report it with UNIT=min/mile.

If the value is in MM:SS format (e.g. "07:54"), report VALUE as the decimal equivalent (7 + 54/60 = 7.9), NOT 7.54. The seconds component must be divided by 60.

If only a min/km table is present, take the Intermediate-30 value from it and report with UNIT=min/km so conversion happens downstream."""
    elif pace_type == "kiplimo":
        extraction_prompt = """Extract Jacob Kiplimo's half-marathon pace or half-marathon finish time.
Look for his fastest half-marathon times (around 56-58 minute range) or his pace in any format.
Report the value exactly as found on the page with its original unit."""
    elif pace_type == "klaebo":
        extraction_prompt = """Extract Johannes Høsflot Klæbo's Nordic skiing pace or finish time, ideally averaged across the 20 km skiathlon and the sprint classic at the 2026 Olympics (or an equivalent recent World Cup race).
Look for his pace in any format: min/mile, min/km, mph, km/h, or total race finish time (e.g. 47:00 for a 20 km skiathlon, 3:00 for a sprint).
If the page only reports a single race time, take it; the caller will average elsewhere if needed.
Report the value exactly as found on the page with its original unit."""
    elif pace_type == "female_5k":
        extraction_prompt = """Extract the average running pace for a 5K race for females (especially around age 25).

The page may contain MULTIPLE pace tables stacked together (e.g. one labeled "Pace (min/km)" and one labeled "Pace (min/mile)"), each with the same Age x Ability layout. You MUST prefer the min/mile table; do NOT take the min/km value. Also do NOT use the Finish Time table (HH:MM:SS values).

Specifically: locate the Female section, then within it the table whose header indicates "min/mile" (per-mile pace), then read the value at row "25" (or "Age 25") and column "Intermediate" or "Novice". Report it with UNIT=min/mile.

If the value is in MM:SS format (e.g. "10:30"), report VALUE as the decimal equivalent (10 + 30/60 = 10.5), NOT 10.30. The seconds component must be divided by 60.

If only a min/km table is present, take the value from it and report with UNIT=min/km so conversion happens downstream."""
    elif pace_type == "chebet":
        extraction_prompt = """Extract Beatrice Chebet's 5K pace or 5K finish time.
Look for her fastest 5K times (around sub-14 minutes or ~13:56 range) or her pace in any format.
Report the value exactly as found on the page with its original unit."""
    elif pace_type == "female_daily_miles":
        extraction_prompt = """Extract the recommended or average daily walking/trekking distance in miles for a 25-year-old female (or general adult female).
Look for daily mileage recommendations, average daily walking distance, or similar data.
If the page mentions steps, convert using ~2000 steps per mile.
Report the value exactly as found on the page with its original unit.
Use UNIT=miles if the value is already in miles, or UNIT=km if in kilometers."""
    elif pace_type == "adult_sets":
        extraction_prompt = """Extract the average or recommended number of sets per workout for an average adult.
Look for total sets per workout session, not sets per exercise or per body part.
If the page gives a range (e.g. 15-20 sets), take the midpoint.
Report the value as a number with UNIT=sets."""
    elif pace_type == "cutler":
        extraction_prompt = """Extract Jay Cutler's recommended or typical number of sets per workout.
Look for his total sets per workout session or per body part training session.
If the page gives sets per body part, report that value.
Report the value as a number with UNIT=sets."""
    else:
        return None, f"Unknown pace_type: {pace_type}"

    # Truncate content for LLM (to avoid token limits)
    truncated_content = content[:10000] if len(content) > 10000 else content

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """You are a running data extraction assistant.
Extract running pace/speed data from webpage content and return it in a structured format.
Do NOT convert the value - return it exactly as found on the page.

Return your response in this exact format (3 lines):
VALUE: <number>
UNIT: <unit>
RAW: <original text from page>

For UNIT, use one of these exact values:
- min/mile (for minutes per mile pace, e.g., "8:30 per mile" -> VALUE: 8.5, UNIT: min/mile)
- min/km (for minutes per kilometer pace, e.g., "5:00/km" -> VALUE: 5.0, UNIT: min/km)
- mph (for miles per hour speed)
- km/h (for kilometers per hour speed)
- marathon_time_minutes (for total marathon time, convert H:MM:SS to total minutes, e.g., "2:01:39" -> VALUE: 121.65)
- 5k_time_minutes (for total 5K time, convert MM:SS to total minutes, e.g., "25:30" -> VALUE: 25.5)
- half_marathon_time_minutes (for total half-marathon time, convert H:MM:SS or MM:SS to total minutes, e.g., "56:42" -> VALUE: 56.7)

For pace in MM:SS format, convert to decimal minutes (e.g., "8:30" = 8.5 minutes).

If no relevant pace data is found, return exactly: NOT_FOUND"""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"""{extraction_prompt}

Webpage content:
{truncated_content}

Extract the pace data:"""}]
        }
    ]

    try:
        response = model(messages)
        response_text = response.strip()

        if response_text == "NOT_FOUND" or "not found" in response_text.lower():
            return None, "No pace data found in page content"

        # Parse the structured response
        lines = response_text.strip().split('\n')
        parsed_data = {}

        for line in lines:
            line = line.strip()
            if line.upper().startswith('VALUE:'):
                value_str = line.split(':', 1)[1].strip()
                # Clean up the value string
                value_str = value_str.replace(',', '.').strip()
                value_str = re.sub(r'[^\d.]', '', value_str)
                if value_str:
                    parsed_data['value'] = float(value_str)
            elif line.upper().startswith('UNIT:'):
                parsed_data['unit'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('RAW:'):
                parsed_data['raw_text'] = line.split(':', 1)[1].strip()

        # Validate we got the required fields
        if 'value' not in parsed_data or 'unit' not in parsed_data:
            return None, f"Could not parse structured response: {response_text[:100]}"

        # Set default raw_text if not provided
        if 'raw_text' not in parsed_data:
            parsed_data['raw_text'] = f"{parsed_data['value']} {parsed_data['unit']}"

        return parsed_data, f"Extracted: {parsed_data['value']} {parsed_data['unit']} (raw: {parsed_data['raw_text']})"

    except ValueError as e:
        return None, f"Failed to parse pace from LLM response: {str(e)[:50]}"
    except Exception as e:
        return None, f"LLM extraction error: {str(e)[:50]}"


def extract_and_convert_pace_from_url(
    url: str,
    pace_type: str,
    model: Any,
    timeout: int = 10
) -> Tuple[Optional[float], str]:
    """
    Extract running pace from a webpage and convert to min/mile.

    This is a convenience function that combines extract_pace_from_url()
    with convert_pace_to_min_per_mile() for cases where the final min/mile
    value is needed.

    Args:
        url: URL to fetch and analyze
        pace_type: Either "male_5k" or "kipchoge"
        model: LLM model for extraction
        timeout: Request timeout in seconds

    Returns:
        Tuple of (pace_in_min_per_mile: float or None, details: str)
    """
    # Extract structured pace data
    pace_data, extract_details = extract_pace_from_url(url, pace_type, model, timeout)

    if pace_data is None:
        return None, extract_details

    # Convert to min/mile
    pace_min_mile, convert_details = convert_pace_to_min_per_mile(
        pace_data['value'],
        pace_data['unit']
    )

    if pace_min_mile is None:
        return None, f"Extraction OK but conversion failed: {convert_details}"

    # Sanity check: pace should be between 3 and 20 min/mile
    if not (3.0 <= pace_min_mile <= 20.0):
        return None, f"Converted value {pace_min_mile:.2f} min/mile outside reasonable range (3-20)"

    return pace_min_mile, f"{extract_details} -> {convert_details}"


# =============================================================================
# LLM-as-Judge Backups for Checkpoint 5
# =============================================================================

def judge_url_relevance(
    url: str,
    category: str,
    model: Any,
    timeout: int = 10
) -> Tuple[bool, str]:
    """Use LLM as backup judge to decide if a URL is relevant to a baseline category.

    Used when keyword-string URL filtering returns no candidates.

    Args:
        url: URL to fetch and judge.
        category: Either "male_5k" or "kipchoge".
        model: LLM model with the standard messages interface.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (is_relevant, details).
    """
    content, fetch_status = fetch_page_text_content(url, timeout)
    if not content:
        return False, f"Failed to fetch URL: {fetch_status}"

    if category == "male_5k":
        question = "Does this webpage contain information about average running speeds or pace for a 5K race (especially for males around age 25)?"
    elif category == "kipchoge":
        question = "Does this webpage contain information about Eliud Kipchoge's marathon times or pace?"
    elif category == "intermediate_male_30_half_marathon":
        question = "Does this webpage contain information about average running speeds or pace for a half-marathon (especially for intermediate male runners around age 30)?"
    elif category == "kiplimo":
        question = "Does this webpage contain information about Jacob Kiplimo's half-marathon times or pace?"
    elif category == "klaebo":
        question = "Does this webpage contain information about Johannes Høsflot Klæbo's Nordic skiing race times or pace (e.g. 20 km skiathlon, sprint classic, or related World Cup / Olympic results)?"
    elif category == "female_5k":
        question = "Does this webpage contain information about average running speeds or pace for a 5K race (especially for females around age 25)?"
    elif category == "chebet":
        question = "Does this webpage contain information about Beatrice Chebet's 5K times or pace?"
    elif category == "female_daily_miles":
        question = "Does this webpage contain information about recommended or average daily walking/trekking distance (especially for females or adults around age 25)?"
    elif category == "adult_sets":
        question = "Does this webpage contain information about the average or recommended number of workout sets per session for an average adult?"
    elif category == "cutler":
        question = "Does this webpage contain information about Jay Cutler's workout sets or training volume?"
    else:
        return False, f"Unknown category: {category}"

    truncated_content = content[:5000]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a yes/no classifier. Answer only Yes or No."}]},
        {"role": "user", "content": [{"type": "text", "text": f"URL: {url}\n\n{question}\n\nContent excerpt:\n{truncated_content}\n\nAnswer Yes or No only:"}]}
    ]
    try:
        response = str(model(messages)).strip().lower()
        is_relevant = response.startswith("yes")
        return is_relevant, f"LLM judge: {response[:50]}"
    except Exception as e:
        return False, f"LLM judge error: {str(e)[:50]}"


def judge_url_pace_match(
    url: str,
    category: str,
    sheet_pace: float,
    model: Any,
    timeout: int = 10
) -> Tuple[bool, str]:
    """Use LLM as backup judge to decide if a URL's content supports a baseline pace within ~5% of sheet_pace.

    Used when structured pace extraction (extract_and_convert_pace_from_url) returns None
    for every candidate URL of a category.

    Args:
        url: URL to fetch and judge.
        category: Either "male_5k" or "kipchoge".
        sheet_pace: Baseline pace in min/mile from the user's sheet.
        model: LLM model with the standard messages interface.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (is_match, details).
    """
    content, fetch_status = fetch_page_text_content(url, timeout)
    if not content:
        return False, f"Failed to fetch URL: {fetch_status}"

    if category == "male_5k":
        topic = "an average male 5K running pace (around age 25)"
    elif category == "kipchoge":
        topic = "Eliud Kipchoge's marathon pace"
    elif category == "intermediate_male_30_half_marathon":
        topic = "an average half-marathon pace for an intermediate male runner (around age 30)"
    elif category == "kiplimo":
        topic = "Jacob Kiplimo's half-marathon pace"
    elif category == "klaebo":
        topic = "Johannes Høsflot Klæbo's Nordic skiing race pace (averaged across the 20 km skiathlon and the sprint classic, or equivalent)"
    elif category == "female_5k":
        topic = "an average female 5K running pace (around age 25)"
    elif category == "chebet":
        topic = "Beatrice Chebet's 5K pace"
    elif category == "female_daily_miles":
        topic = "average daily walking/trekking distance for a 25-year-old female"
    elif category == "adult_sets":
        topic = "average or recommended workout sets per session for an average adult"
    elif category == "cutler":
        topic = "Jay Cutler's recommended or typical workout sets per session"
    else:
        return False, f"Unknown category: {category}"

    low = sheet_pace * 0.95
    high = sheet_pace * 1.05
    truncated_content = content[:5000]

    # Use appropriate unit in prompt based on category
    if category == "female_daily_miles":
        unit_str = "miles"
        equiv_str = "or any equivalent unit like km that converts into that range"
    elif category in ("adult_sets", "cutler"):
        unit_str = "sets"
        equiv_str = "or sets per body part that sum to that range"
    else:
        unit_str = "min/mile"
        equiv_str = "or any equivalent unit like km/h, min/km, or marathon/5K time that converts into that range"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a yes/no classifier. Answer only Yes or No."}]},
        {"role": "user", "content": [{"type": "text", "text": (
            f"This webpage should describe {topic}. The user's spreadsheet records this baseline as "
            f"{sheet_pace:.2f} {unit_str}. Does the webpage content support a baseline within roughly 5% of "
            f"{sheet_pace:.2f} {unit_str} (i.e. {low:.2f}–{high:.2f} {unit_str}, {equiv_str})?\n\n"
            f"Content excerpt:\n{truncated_content}\n\nAnswer Yes or No only:"
        )}]}
    ]
    try:
        response = str(model(messages)).strip().lower()
        is_match = response.startswith("yes")
        return is_match, f"LLM judge ({sheet_pace:.2f} {unit_str}): {response[:50]}"
    except Exception as e:
        return False, f"LLM judge error: {str(e)[:50]}"