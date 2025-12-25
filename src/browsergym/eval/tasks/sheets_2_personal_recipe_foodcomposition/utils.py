"""Template-specific utilities for the Personal Recipe Food Composition task.

This module contains utilities that are reusable across all instances of this task template.
Instance-specific constants (like ingredient lists) should be defined in each instance's evaluator.py.
"""

import re
from typing import Any, Optional, List

# Import general utilities from eval_utils
from src.browsergym.eval.eval_utils.web_utils import is_url_from_domain, fetch_api_with_retry
from src.browsergym.eval.eval_utils.table_utils import matches_keywords

__all__ = [
    # Task-specific utilities
    'fetch_usda_page_title',
    'ingredient_matches_usda_page',
    # Template-specific constants
    'COLUMN_KEYWORDS',
    'MACRO_NUTRIENTS',
    'MINERAL_NUTRIENTS',
    'VITAMIN_NUTRIENTS',
    'ALL_NUTRIENTS',
    'FDA_DAILY_VALUES',
    'VALUE_TOLERANCE',
]


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

    # Extract food ID from URL (e.g., /food-details/170162/nutrients)
    match = re.search(r'/food-details/(\d+)', url)
    if not match:
        return None

    food_id = match.group(1)

    # Use USDA FoodData Central API
    api_key = '[REDACTED]'
    api_url = f'https://api.nal.usda.gov/fdc/v1/food/{food_id}?api_key={api_key}'

    data = fetch_api_with_retry(api_url, timeout=timeout, max_retries=max_retries)
    return data.get('description') if data else None


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

    # Use matches_keywords for consistent matching
    if matches_keywords(page_title, match_keywords):
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
