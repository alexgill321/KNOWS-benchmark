"""
Utility functions for the sheets_7_running_analysis task.

Contains unit conversion functions for distance and speed, and date normalization.
"""

from datetime import datetime
import pandas as pd


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


# =============================================================================
# Date Normalization
# =============================================================================

def normalize_date(date_str) -> str:
    """
    Normalize date string for comparison (full timestamp).

    Handles formats like:
    - "Sep 27, 2021, 2:13:42 AM" (gold format)
    - "2021-09-27 02:13:42"
    - Various other common formats

    Args:
        date_str: Date string to normalize.

    Returns:
        Normalized date string in format "YYYY-MM-DD HH:MM:SS".
    """
    if pd.isna(date_str):
        return ""

    date_str = str(date_str).strip()

    # Try parsing the gold format: "Sep 27, 2021, 2:13:42 AM"
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y, %I:%M:%S %p")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    # Try other common formats with time
    formats_to_try = [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%b %d, %Y, %H:%M:%S",
        "%Y-%m-%d %I:%M:%S %p",
    ]

    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    # Fallback to lowercase string
    return date_str.lower()


# =============================================================================
# Data Loading
# =============================================================================

def load_gold_run_activities(csv_path: str):
    """Load gold data, filtering only Run activities.

    The CSV has duplicate 'Distance' columns - pandas renames them to 'Distance' and 'Distance.1'.
    - 'Distance' is in km (rounded)
    - 'Distance.1' is in meters (more precise)

    We normalize to use the meters column converted to km for consistency.

    Args:
        csv_path: Path to the gold_activities.csv file.

    Returns:
        DataFrame containing only Run activities with normalized Distance column.
    """
    gold_df = pd.read_csv(csv_path)
    runs = gold_df[gold_df['Activity Type'] == 'Run'].copy()

    # Use Distance.1 (meters) converted to km for more precision
    # This fixes discrepancies like "Anas run" where Distance=0.44 km but Distance.1=448.1 m
    if 'Distance.1' in runs.columns:
        runs['Distance'] = runs['Distance.1'] / 1000.0

    return runs