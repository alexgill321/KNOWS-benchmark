"""Template-specific utilities for the Personal Recipe Food Composition task.

This module contains utilities that are reusable across all instances of this task template.
Instance-specific constants (like ingredient lists) should be defined in each instance's evaluator.py.
"""

import os
import re
from typing import Any, Optional, List

# Import general utilities from eval_utils
from src.browsergym.knows.eval.eval_utils.web_utils import is_url_from_domain, fetch_api_with_retry
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_exact_match

__all__ = [
    # Task-specific utilities
    'fetch_usda_page_title',
    'ingredient_matches_usda_page',
    'fetch_usda_nutrients',
    'compute_scaled_nutrients',
    'validate_usda_fallback',
    'parse_nutrient_value',
    # Template-specific constants
    'COLUMN_KEYWORDS',
    'MACRO_NUTRIENTS',
    'MINERAL_NUTRIENTS',
    'VITAMIN_NUTRIENTS',
    'ALL_NUTRIENTS',
    'FDA_DAILY_VALUES',
    'VALUE_TOLERANCE',
    'USDA_NUTRIENT_MAP',
]


def parse_nutrient_value(raw: str) -> Optional[float]:
    """Parse a nutrient value from a spreadsheet cell, stripping units.

    Handles values like "51.4 g", "88.6 mg", "12.3 µg", "1,234.5 mg",
    as well as plain numbers like "51.4".

    Args:
        raw: Raw cell value string.

    Returns:
        Parsed float value, or None if parsing fails.
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = raw.strip().replace(',', '')
    cleaned = re.sub(r'\s*(g|mg|µg|mcg|ug|IU|kcal|cal|%)\s*$', '', cleaned, flags=re.IGNORECASE)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def fetch_usda_page_title(url: str, timeout: int = 10, max_retries: int = 3) -> Optional[str]:
    """
    Fetch food name from USDA FoodData Central page.

    Uses the USDA FoodData Central API to get the food description,
    since the website is a JavaScript SPA that doesn't return content
    via simple HTTP requests.

    Args:
        url: USDA FoodData Central URL.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for rate-limited requests.

    Returns:
        Food name/title from the API, or None if fetch failed.
    """
    if not is_url_from_domain(url, 'fdc.nal.usda.gov'):
        return None

    food_id = extract_food_id_from_url(url)
    if not food_id:
        return None

    api_key = os.environ.get('USDA_API_KEY', 'YOUR_USDA_API_KEY')
    api_url = f'https://api.nal.usda.gov/fdc/v1/food/{food_id}?api_key={api_key}'

    data = fetch_api_with_retry(api_url, timeout=timeout, max_retries=max_retries)
    return data.get('description') if data else None


# Maps our nutrient names to USDA API nutrient name patterns
USDA_NUTRIENT_MAP = {
    "Carbohydrates": ["Carbohydrate, by difference"],
    "Fat": ["Total lipid (fat)"],
    "Fiber": ["Fiber, total dietary"],
    "Protein": ["Protein"],
    "Sugar": ["Total Sugars", "Sugars, Total"],
    "Calcium": ["Calcium, Ca"],
    "Iron": ["Iron, Fe"],
    "Potassium": ["Potassium, K"],
    "Sodium": ["Sodium, Na"],
    "Vitamin A": ["Vitamin A, RAE"],
    "Vitamin C": ["Vitamin C, total ascorbic acid"],
}


def extract_food_id_from_url(url: str) -> Optional[str]:
    """Extract USDA food ID from a FoodData Central URL.

    Args:
        url: USDA FoodData Central URL (e.g., https://fdc.nal.usda.gov/food-details/170162/nutrients).

    Returns:
        Food ID string, or None if not found.
    """
    match = re.search(r'/food-details/(\d+)', url)
    return match.group(1) if match else None


def fetch_usda_nutrients(url: str, timeout: int = 10, max_retries: int = 3) -> Optional[dict]:
    """Fetch per-100g nutrient values from a USDA FoodData Central URL.

    Returns a dict mapping our standard nutrient names to per-100g values.
    Only includes nutrients that are found in the API response.

    Args:
        url: USDA FoodData Central URL.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries.

    Returns:
        Dict of {nutrient_name: per_100g_value}, or None if fetch failed.
    """
    if not is_url_from_domain(url, 'fdc.nal.usda.gov'):
        return None

    food_id = extract_food_id_from_url(url)
    if not food_id:
        return None

    api_key = os.environ.get('USDA_API_KEY', 'YOUR_USDA_API_KEY')
    api_url = f'https://api.nal.usda.gov/fdc/v1/food/{food_id}?api_key={api_key}'

    data = fetch_api_with_retry(api_url, timeout=timeout, max_retries=max_retries)
    if not data:
        return None

    # Parse nutrient data from API response
    api_nutrients = data.get('foodNutrients', [])
    result = {}

    for our_name, api_patterns in USDA_NUTRIENT_MAP.items():
        for nutrient_entry in api_nutrients:
            api_name = nutrient_entry.get('nutrient', {}).get('name', '')
            if api_name in api_patterns:
                amount = nutrient_entry.get('amount')
                if amount is not None:
                    result[our_name] = float(amount)
                break

    return result if result else None


def compute_scaled_nutrients(url: str, quantity_g: float, timeout: int = 10, max_retries: int = 3) -> Optional[dict]:
    """Fetch USDA nutrients and scale by recipe quantity.

    Args:
        url: USDA FoodData Central URL.
        quantity_g: Recipe quantity in grams.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries.

    Returns:
        Dict of {nutrient_name: scaled_value}, or None if fetch failed.
    """
    per_100g = fetch_usda_nutrients(url, timeout=timeout, max_retries=max_retries)
    if not per_100g:
        return None

    scale = quantity_g / 100.0
    return {name: value * scale for name, value in per_100g.items()}


def validate_usda_fallback(
    sheet_values: dict,
    gold_values: dict,
    api_per_100g: dict,
    tolerance: float = 0.20,
) -> bool:
    """Validate an ingredient's sheet values against its USDA API data.

    Checks two things:
    1. Ratio consistency: sheet_value / api_per_100g is consistent across
       all nutrients (agent used a valid quantity and scaled correctly).
    2. Gold cross-validation: at least one nutrient matches the gold data
       within tolerance (confirms a reasonable USDA entry was used).

    Args:
        sheet_values: Dict of {nutrient_name: sheet_value} from the agent's spreadsheet.
        gold_values: Dict of {nutrient_name: gold_value} from gold labels.
        api_per_100g: Dict of {nutrient_name: per_100g_value} from USDA API.
        tolerance: Fractional tolerance for comparisons (default 0.20 = 20%).

    Returns:
        True if the agent's values are internally consistent with their USDA
        entry and cross-validated against gold.
    """
    # Compute ratios for all nutrients where both sheet and API are non-zero
    # Use higher tolerance for small decimal values (< 1.0) since rounding
    # at 1-2 decimal places amplifies ratio error on small absolute values
    SMALL_VALUE_THRESHOLD = 1.0
    SMALL_VALUE_TOLERANCE = 0.35  # 35% for small values

    ratios = []
    ratio_details = []  # (ratio, sheet_val, is_small)
    for nutrient in sheet_values:
        sheet_val = sheet_values.get(nutrient)
        api_val = api_per_100g.get(nutrient)
        if sheet_val and api_val and sheet_val > 0 and api_val > 0:
            ratio = sheet_val / api_val
            is_small = sheet_val < SMALL_VALUE_THRESHOLD
            ratios.append(ratio)
            ratio_details.append((ratio, sheet_val, is_small))

    if len(ratios) < 2:
        print(f"    [USDA FALLBACK] Not enough non-zero nutrient pairs for ratio check ({len(ratios)} found)")
        return False

    # Check ratio consistency: all ratios should be within tolerance of the median
    # Small decimal values get a wider tolerance to account for rounding noise
    ratios.sort()
    median_ratio = ratios[len(ratios) // 2]
    if median_ratio == 0:
        print(f"    [USDA FALLBACK] Median ratio is 0 — cannot validate consistency")
        return False
    consistent = True
    for ratio, sheet_val, is_small in ratio_details:
        allowed = SMALL_VALUE_TOLERANCE if is_small else tolerance
        if abs(ratio - median_ratio) / median_ratio > allowed:
            consistent = False
            break

    if not consistent:
        print(f"    [USDA FALLBACK] Ratios inconsistent: min={min(ratios):.4f}, max={max(ratios):.4f}, median={median_ratio:.4f}")
        return False

    print(f"    [USDA FALLBACK] Ratios consistent (median={median_ratio:.4f}, n={len(ratios)})")

    # Gold cross-validation: at least one nutrient must match gold within tolerance
    tolerance_percent = tolerance * 100
    from src.browsergym.knows.eval.eval_utils.text_utils import numerical_match_with_error

    has_gold_match = False
    for nutrient in sheet_values:
        sheet_val = sheet_values.get(nutrient)
        gold_val = gold_values.get(nutrient)
        if sheet_val is not None and gold_val is not None and gold_val != 0:
            is_match, _ = numerical_match_with_error(gold_val, sheet_val, error_percent=tolerance_percent)
            if is_match:
                print(f"    [USDA FALLBACK] Gold cross-validation passed on '{nutrient}' (sheet={sheet_val:.2f}, gold={gold_val:.2f})")
                has_gold_match = True
                break

    if not has_gold_match:
        print(f"    [USDA FALLBACK] No nutrient matched gold data — fallback rejected")

    return has_gold_match


def ingredient_matches_usda_page(
    ingredient: str,
    page_title: str,
    keywords: Optional[List[str]] = None,
    model: Any = None
) -> bool:
    """
    Check if ingredient name matches USDA page title.

    Uses keyword matching with optional LLM fallback.

    Args:
        ingredient: Expected ingredient name.
        page_title: Title/food name from USDA page.
        keywords: List of keywords to match against page_title. If None, uses ingredient.lower().
        model: Optional LLM model for fallback matching.

    Returns:
        True if the ingredient reasonably matches the page title.
    """
    if not page_title or not ingredient:
        return False

    # Use provided keywords or default to ingredient name
    match_keywords = keywords if keywords else [ingredient.lower()]

    # Use keywords_exact_match for consistent matching
    if keywords_exact_match(page_title, match_keywords):
        print(f"  [DEBUG] USDA page '{page_title}' matched ingredient '{ingredient}' via KEYWORD")
        return True

    # LLM fallback if keyword matching fails and model is provided
    if model is not None:
        prompt_text = f"Is '{page_title}' a valid USDA database entry for the ingredient '{ingredient}'? For example, 'Nuts, almonds, raw' is valid for 'Almonds', and 'Spices, garlic powder' is valid for 'Garlic Powder'. Answer only Yes or No."
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that determines if USDA food database entries match recipe ingredients. Be lenient - USDA entries often have prefixes like 'Nuts,', 'Spices,', 'Beverages,' and suffixes like ', raw', ', dried', etc. Answer Yes or No."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
        try:
            response = model(messages)
            if response and 'yes' in response.lower():
                print(f"  [DEBUG] USDA page '{page_title}' matched ingredient '{ingredient}' via LLM")
                return True
        except Exception as e:
            print(f"LLM error for USDA page matching: {e}")

    print(f"  [DEBUG] USDA page '{page_title}' did NOT match ingredient '{ingredient}'")
    return False


# =============================================================================
# Template-Specific Constants
# These apply to any recipe food composition task, regardless of the specific recipe.
# =============================================================================

# Keyword mappings for column detection (common spreadsheet column names)
COLUMN_KEYWORDS = {
    "Ingredients": ["ingredient"],
    "Link": ["link", "url"],
    "Carbohydrates": ["carbohydrate", "carbs"],
    "Fat": ["fat"],
    "Fiber": ["fiber"],
    "Protein": ["protein"],
    "Sugar": ["sugar"],
    "Calcium": ["calcium"],
    "Iron": ["iron"],
    "Potassium": ["potassium"],
    "Sodium": ["sodium"],
    "Vitamin A": ["vitamin a", "vit a"],
    "Vitamin C": ["vitamin c", "vit c"],
}

# Nutrient groupings (standard nutritional categories)
MACRO_NUTRIENTS = ["Carbohydrates", "Fat", "Fiber", "Protein", "Sugar"]
MINERAL_NUTRIENTS = ["Calcium", "Iron", "Potassium", "Sodium"]
VITAMIN_NUTRIENTS = ["Vitamin A", "Vitamin C"]
ALL_NUTRIENTS = MACRO_NUTRIENTS + MINERAL_NUTRIENTS + VITAMIN_NUTRIENTS

# FDA Daily Values for 10% DV calculation (universal standard)
FDA_DAILY_VALUES = {
    "Carbohydrates": 275,  # g
    "Fat": 78,  # g
    "Fiber": 28,  # g
    "Protein": 50,  # g
    "Sugar": 50,  # g (added sugars)
    "Calcium": 1300,  # mg
    "Iron": 18,  # mg
    "Potassium": 4700,  # mg
    "Sodium": 2300,  # mg
    "Vitamin A": 900,  # mcg RAE
    "Vitamin C": 90,  # mg
}

# Tolerance for numerical comparisons (task design choice)
VALUE_TOLERANCE = 0.20  # 20%
