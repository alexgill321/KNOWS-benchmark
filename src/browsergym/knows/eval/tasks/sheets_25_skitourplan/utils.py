"""Task-specific utilities for sheets_25_skitourplan_instance_1 evaluator.

This module contains functions for:
- Parsing slope angle, GPS coordinates, and other run data
- Color classification for avalanche danger ratings
- Gold data loading and run lookup
- URL validation for wbsguide.com and Utah Avalanche Center
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any

import requests

# Task-level constants
# Note: TASK_DIR points to the template level. Instance-specific data is in instance_X/data/
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
# Default DATA_DIR for backwards compatibility (instance_1)
DATA_DIR = os.path.join(TASK_DIR, "instance_1", "data")

# ============================================================================
# Danger Rating Color Mapping
# ============================================================================

# RGB ranges for danger rating colors (0-1 scale from Google Sheets API)
# Each color has (min, max) ranges for R, G, B
DANGER_COLORS = {
    "green": {
        "red": (0, 0.5),
        "green": (0.6, 1.0),
        "blue": (0, 0.5)
    },
    "yellow": {
        "red": (0.85, 1.0),
        "green": (0.7, 1.0),
        "blue": (0, 0.5)
    },
    "orange": {
        "red": (0.85, 1.0),
        "green": (0.4, 0.7),
        "blue": (0, 0.35)
    },
    "red": {
        "red": (0.75, 1.0),
        "green": (0, 0.35),
        "blue": (0, 0.35)
    },
    "black": {
        "red": (0, 0.25),
        "green": (0, 0.25),
        "blue": (0, 0.25)
    },
}

# Map danger level words to colors
DANGER_LEVEL_TO_COLOR = {
    "low": "green",
    "moderate": "yellow",
    "considerable": "orange",
    "high": "red",
    "extreme": "black",
}


# ============================================================================
# Data Parsing Functions
# ============================================================================

def parse_slope_angle(angle_str: str) -> Optional[int]:
    """Parse slope angle from string like '26 degrees' or '26°'.

    Args:
        angle_str: String containing slope angle value.

    Returns:
        Integer slope angle, or None if parsing fails.
    """
    if not angle_str:
        return None

    # Extract first number from string
    match = re.search(r'(\d+)', str(angle_str))
    return int(match.group(1)) if match else None


def parse_gps_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
    """Parse GPS coordinates from string like '40.6645° / -111.6473°'.

    Handles various formats:
    - "40.6645° / -111.6473°"
    - "40.6645 / -111.6473"
    - "40.6645, -111.6473"

    Args:
        coord_str: String containing GPS coordinates.

    Returns:
        Tuple of (latitude, longitude), or None if parsing fails.
    """
    if not coord_str:
        return None

    # Pattern to match lat/lon with optional degree symbol
    pattern = r'([-]?\d+\.?\d*)(?:°|&deg;)?\s*[/,]\s*([-]?\d+\.?\d*)(?:°|&deg;)?'
    match = re.search(pattern, str(coord_str))

    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return (lat, lon)
        except ValueError:
            return None

    return None


def parse_elevation(elev_str: str) -> Optional[int]:
    """Parse elevation from string like '8,986' or '8986 ft'.

    Args:
        elev_str: String containing elevation value.

    Returns:
        Integer elevation in feet, or None if parsing fails.
    """
    if not elev_str:
        return None

    # Remove commas and extract number
    clean_str = str(elev_str).replace(',', '').replace("'", "")
    match = re.search(r'(\d+)', clean_str)
    return int(match.group(1)) if match else None


def parse_typical_vertical(vert_str: str) -> Optional[int]:
    """Parse typical vertical from string like '900' or '900 ft'.

    Args:
        vert_str: String containing vertical value.

    Returns:
        Integer vertical in feet, or None if parsing fails.
    """
    if not vert_str:
        return None

    # Remove commas and extract number
    clean_str = str(vert_str).replace(',', '').replace("'", "")
    match = re.search(r'(\d+)', clean_str)
    return int(match.group(1)) if match else None


def normalize_aspect(aspect_str: str) -> Optional[str]:
    """Normalize slope aspect to standard format (N, NE, E, SE, S, SW, W, NW).

    Args:
        aspect_str: String containing slope aspect.

    Returns:
        Normalized aspect string, or None if invalid.
    """
    if not aspect_str:
        return None

    aspect = str(aspect_str).upper().strip()

    # Valid aspects
    valid_aspects = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    if aspect in valid_aspects:
        return aspect

    # Try to extract from longer string
    for va in valid_aspects:
        if va in aspect:
            return va

    return None


# ============================================================================
# URL Validation Functions
# ============================================================================

def is_valid_wbsguide_url(url: str) -> bool:
    """Check if URL is a valid Wasatch Backcountry Ski Guide run page.

    Args:
        url: URL string to validate.

    Returns:
        True if URL is a valid wbsguide.com run page.
    """
    if not url:
        return False

    # Match pattern: https://wbsguide.com/{id}.php or https://wbsguide.com/guide/{id}.php
    return bool(re.match(r'https?://(?:www\.)?wbsguide\.com/(?:guide/)?\d+\.php', str(url).strip()))


def extract_run_id_from_url(url: str) -> Optional[int]:
    """Extract run ID from wbsguide.com URL.

    Args:
        url: WBSGuide URL like "https://wbsguide.com/2104.php"

    Returns:
        Run ID as integer, or None if extraction fails.
    """
    if not url:
        return None

    match = re.search(r'wbsguide\.com/(?:guide/)?(\d+)\.php', str(url))
    return int(match.group(1)) if match else None


def is_valid_uac_forecast_url(url: str) -> bool:
    """Check if URL is a valid Utah Avalanche Center forecast page.

    Args:
        url: URL string to validate.

    Returns:
        True if URL is a valid utahavalanchecenter.org forecast page.
    """
    if not url:
        return False

    url_lower = str(url).lower().strip()
    return 'utahavalanchecenter.org/forecast' in url_lower


# ============================================================================
# Color Classification Functions
# ============================================================================

def classify_danger_color(bg_color: Dict) -> str:
    """Classify background color as danger rating color name.

    Args:
        bg_color: Dict with 'red', 'green', 'blue' keys (0-1 scale).

    Returns:
        Color name ('green', 'yellow', 'orange', 'red', 'black', or 'none').
    """
    if not bg_color:
        return "none"

    # Default to 0 for missing color channels (Google Sheets omits 0 values)
    r = bg_color.get('red', 0)
    g = bg_color.get('green', 0)
    b = bg_color.get('blue', 0)

    # Check each danger color
    for color_name, ranges in DANGER_COLORS.items():
        r_min, r_max = ranges["red"]
        g_min, g_max = ranges["green"]
        b_min, b_max = ranges["blue"]

        if (r_min <= r <= r_max and
            g_min <= g <= g_max and
            b_min <= b <= b_max):
            return color_name

    return "none"


def danger_level_to_color(level: str) -> str:
    """Convert danger level text to color name.

    Args:
        level: Danger level like 'Low', 'Moderate', 'Considerable', 'High', 'Extreme'.

    Returns:
        Color name ('green', 'yellow', 'orange', 'red', 'black').
    """
    if not level:
        return "none"

    return DANGER_LEVEL_TO_COLOR.get(level.lower().strip(), "none")


# ============================================================================
# Gold Data Functions
# ============================================================================

def load_gold_runs(data_dir: str = None) -> Optional[Dict]:
    """Load gold run data from JSON file.

    Args:
        data_dir: Optional path to data directory. If not provided, uses default DATA_DIR.

    Returns:
        Dict with gold run data, or None if file doesn't exist.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    gold_path = os.path.join(data_dir, "gold_runs.json")

    if not os.path.exists(gold_path):
        return None

    with open(gold_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_run_by_name(name: str, gold_data: Dict) -> Optional[Dict]:
    """Find a run in gold data by name (case-insensitive, fuzzy).

    Args:
        name: Run name to search for.
        gold_data: Gold data dict from load_gold_runs().

    Returns:
        Run dict if found, None otherwise.
    """
    if not name or not gold_data:
        return None

    name_lower = str(name).lower().strip()

    # Remove common punctuation for comparison
    def normalize(s):
        return re.sub(r"['\-\s]+", '', s.lower())

    name_normalized = normalize(name_lower)

    # Search in all_runs
    all_runs = gold_data.get('all_runs', [])

    for run in all_runs:
        run_name = run.get('name', '')

        # Exact match (case-insensitive)
        if run_name.lower() == name_lower:
            return run

        # Normalized match (ignoring punctuation/spaces)
        if normalize(run_name) == name_normalized:
            return run

        # Partial match
        if name_normalized in normalize(run_name) or normalize(run_name) in name_normalized:
            return run

    return None


def find_run_by_url(url: str, gold_data: Dict) -> Optional[Dict]:
    """Find a run in gold data by URL.

    Args:
        url: WBSGuide URL to search for.
        gold_data: Gold data dict from load_gold_runs().

    Returns:
        Run dict if found, None otherwise.
    """
    if not url or not gold_data:
        return None

    run_id = extract_run_id_from_url(url)
    if not run_id:
        return None

    # Search in all_runs
    all_runs = gold_data.get('all_runs', [])

    for run in all_runs:
        if run.get('id') == run_id:
            return run

    return None


def find_run_by_name_or_url(name: str, url: str, gold_data: Dict) -> Optional[Dict]:
    """Find a run in gold data by name or URL.

    Tries URL first (more reliable), then falls back to name.

    Args:
        name: Run name to search for.
        url: WBSGuide URL to search for.
        gold_data: Gold data dict from load_gold_runs().

    Returns:
        Run dict if found, None otherwise.
    """
    # Try URL first
    run = find_run_by_url(url, gold_data)
    if run:
        return run

    # Fall back to name
    return find_run_by_name(name, gold_data)


def get_valid_runs(gold_data: Dict, max_angle: int = None) -> List[Dict]:
    """Get list of valid runs within the max slope angle.

    Looks for the pre-filtered key 'valid_runs_le_<N>' in gold data.
    Falls back to filtering all_runs by max_angle if the key doesn't exist.

    Args:
        gold_data: Gold data dict from load_gold_runs().
        max_angle: Maximum slope angle. If None, reads from metadata.

    Returns:
        List of run dicts with slope angle <= max_angle.
    """
    if not gold_data:
        return []

    if max_angle is None:
        max_angle = gold_data.get('metadata', {}).get('max_slope_angle', 26)

    # Try pre-filtered key first
    key = f'valid_runs_le_{max_angle}'
    if key in gold_data:
        return gold_data[key]

    # Fall back to filtering all_runs
    all_runs = gold_data.get('all_runs', [])
    return [r for r in all_runs if r.get('slope_angle') and r['slope_angle'] <= max_angle]


def get_forecast_data(gold_data: Dict) -> Optional[Dict]:
    """Get forecast data from gold data.

    Args:
        gold_data: Gold data dict from load_gold_runs().

    Returns:
        Forecast dict with date, url, danger_rose.
    """
    if not gold_data:
        return None

    return gold_data.get('forecast')


# ============================================================================
# Validation Helper Functions
# ============================================================================

def validate_run_data(
    run_row: Dict,
    gold_run: Dict,
    fields_to_check: Optional[List[str]] = None
) -> Dict[str, Tuple[bool, str]]:
    """Validate run data against gold data.

    Args:
        run_row: Dict with run data from spreadsheet.
        gold_run: Dict with gold run data.
        fields_to_check: List of fields to validate. If None, validates all.

    Returns:
        Dict mapping field names to (is_valid, message) tuples.
    """
    results = {}

    if fields_to_check is None:
        fields_to_check = [
            'name', 'url', 'starting_location', 'gps',
            'typical_vertical', 'slope_aspect', 'slope_angle'
        ]

    for field in fields_to_check:
        if field == 'name':
            user_val = run_row.get('name', '')
            gold_val = gold_run.get('name', '')
            # Name validation is lenient (handled by lookup)
            is_valid = bool(user_val)
            results['name'] = (is_valid, f"User: {user_val}, Gold: {gold_val}")

        elif field == 'url':
            user_val = run_row.get('url', '')
            gold_val = gold_run.get('url', '')
            is_valid = is_valid_wbsguide_url(user_val)
            # Check if it's the same run ID
            if is_valid:
                user_id = extract_run_id_from_url(user_val)
                gold_id = gold_run.get('id')
                is_valid = user_id == gold_id
            results['url'] = (is_valid, f"User: {user_val}, Gold: {gold_val}")

        elif field == 'starting_location':
            user_val = run_row.get('starting_location', '')
            gold_val = gold_run.get('starting_location', '')
            # Starting location might be missing from gold data
            if gold_val is None:
                is_valid = True  # Can't validate if no gold
            else:
                # Fuzzy match
                user_lower = str(user_val).lower().strip()
                gold_lower = str(gold_val).lower().strip()
                is_valid = user_lower == gold_lower or user_lower in gold_lower or gold_lower in user_lower
            results['starting_location'] = (is_valid, f"User: {user_val}, Gold: {gold_val}")

        elif field == 'gps':
            user_coords = parse_gps_coordinates(run_row.get('gps', ''))
            gold_lat = gold_run.get('gps_lat')
            gold_lon = gold_run.get('gps_lon')

            if user_coords is None or gold_lat is None or gold_lon is None:
                is_valid = user_coords is not None  # Valid if parseable
                results['gps'] = (is_valid, f"User: {user_coords}, Gold: ({gold_lat}, {gold_lon})")
            else:
                # Check within tolerance (0.01 degrees ~ 1km)
                lat_diff = abs(user_coords[0] - gold_lat)
                lon_diff = abs(user_coords[1] - gold_lon)
                is_valid = lat_diff < 0.01 and lon_diff < 0.01
                results['gps'] = (is_valid, f"User: {user_coords}, Gold: ({gold_lat}, {gold_lon}), Diff: ({lat_diff:.4f}, {lon_diff:.4f})")

        elif field == 'typical_vertical':
            user_val = parse_typical_vertical(run_row.get('typical_vertical', ''))
            gold_val = gold_run.get('typical_vertical')

            if user_val is None or gold_val is None:
                is_valid = user_val is not None  # Valid if parseable
            else:
                # Check within 10% tolerance
                tolerance = max(50, gold_val * 0.1)
                is_valid = abs(user_val - gold_val) <= tolerance
            results['typical_vertical'] = (is_valid, f"User: {user_val}, Gold: {gold_val}")

        elif field == 'slope_aspect':
            user_val = normalize_aspect(run_row.get('slope_aspect', ''))
            gold_val = gold_run.get('slope_aspect')

            if user_val is None or gold_val is None:
                is_valid = user_val is not None
            else:
                is_valid = user_val == gold_val
            results['slope_aspect'] = (is_valid, f"User: {user_val}, Gold: {gold_val}")

        elif field == 'slope_angle':
            user_val = parse_slope_angle(run_row.get('slope_angle', ''))
            gold_val = gold_run.get('slope_angle')

            if user_val is None or gold_val is None:
                is_valid = user_val is not None and user_val <= 26
            else:
                is_valid = user_val == gold_val and user_val <= 26
            results['slope_angle'] = (is_valid, f"User: {user_val}, Gold: {gold_val}")

    return results


def gps_coordinates_match(
    user_coords: Tuple[float, float],
    gold_lat: float,
    gold_lon: float,
    tolerance: float = 0.01
) -> bool:
    """Check if GPS coordinates match within tolerance.

    Args:
        user_coords: Tuple of (lat, lon) from user.
        gold_lat: Gold standard latitude.
        gold_lon: Gold standard longitude.
        tolerance: Maximum allowed difference in degrees (default 0.01 ~ 1km).

    Returns:
        True if coordinates match within tolerance.
    """
    if user_coords is None:
        return False

    lat_diff = abs(user_coords[0] - gold_lat)
    lon_diff = abs(user_coords[1] - gold_lon)

    return lat_diff <= tolerance and lon_diff <= tolerance


def scrape_page_gps_coords(url: str, tolerance: float = 0.01) -> List[Tuple[float, float]]:
    """Scrape all GPS coordinate pairs from a wbsguide.com run page.

    Args:
        url: WBSGuide URL to scrape.
        tolerance: Not used here, kept for API consistency.

    Returns:
        List of (lat, lon) tuples found on the page.
    """
    try:
        clean_url = url.split('#')[0]  # Remove anchor
        response = requests.get(
            clean_url,
            headers={"User-Agent": "Agent-Benchmark Research"},
            timeout=10,
        )
        response.raise_for_status()

        # Find all coordinate patterns: lat (40.xxxx) and lon (-111.xxxx)
        coords = re.findall(r'(\d{2}\.\d{3,})\D+(-\d{2,3}\.\d{3,})', response.text)
        return [(float(lat), float(lon)) for lat, lon in coords]
    except Exception as e:
        print(f"WARNING: scrape_page_gps_coords failed for {url}: {e}")
        return []


def gps_coordinates_match_with_fallback(
    user_coords: Tuple[float, float],
    gold_lat: float,
    gold_lon: float,
    run_url: str = None,
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    """Check GPS coordinates against gold, with page-scrape fallback.

    First checks against the gold data coordinates. If that fails and a
    run URL is provided, scrapes the wbsguide page for all GPS coords
    and checks if the user's coordinates match any of them.

    Args:
        user_coords: Tuple of (lat, lon) from user.
        gold_lat: Gold standard latitude.
        gold_lon: Gold standard longitude.
        run_url: Optional WBSGuide URL to scrape as fallback.
        tolerance: Maximum allowed difference in degrees.

    Returns:
        Tuple of (match_result, detail_message).
    """
    # Primary check against gold data
    if gps_coordinates_match(user_coords, gold_lat, gold_lon, tolerance):
        return True, "Coordinates match within tolerance"

    # Fallback: scrape the page for all coordinates
    if run_url and is_valid_wbsguide_url(run_url):
        page_coords = scrape_page_gps_coords(run_url)
        for page_lat, page_lon in page_coords:
            if gps_coordinates_match(user_coords, page_lat, page_lon, tolerance):
                return True, f"Coordinates match page coords ({page_lat}, {page_lon})"

    return False, f"User: {user_coords}, Gold: ({gold_lat}, {gold_lon})"
