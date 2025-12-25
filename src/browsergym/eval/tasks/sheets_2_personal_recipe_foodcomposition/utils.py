"""Task-specific utilities for the Personal Recipe Food Composition task."""

import re
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup


def is_valid_usda_url(url: str) -> bool:
    """
    Validate URL is from USDA FoodData Central domain.

    Args:
        url: URL string to validate.

    Returns:
        True if URL is a valid USDA FoodData Central URL.
    """
    if not url:
        return False
    return 'fdc.nal.usda.gov' in url.lower()


def fetch_usda_page_title(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch food name from USDA FoodData Central page via HTML parsing.

    Args:
        url: USDA FoodData Central URL.
        timeout: Request timeout in seconds.

    Returns:
        Food name/title from the page, or None if fetch failed.
    """
    if not is_valid_usda_url(url):
        return None

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to find the food name from various elements
        # USDA FoodData Central typically has the food name in the title or h1
        title = soup.find('title')
        if title:
            title_text = title.get_text().strip()
            # Clean up the title - often contains site name
            if '|' in title_text:
                title_text = title_text.split('|')[0].strip()
            if title_text:
                return title_text

        # Try h1 as fallback
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()

        return None

    except Exception as e:
        print(f"Error fetching USDA page {url}: {e}")
        return None


def ingredient_matches_usda_page(ingredient: str, page_title: str) -> bool:
    """
    Check if ingredient name matches USDA page title.

    Uses fuzzy matching to handle variations in naming.

    Args:
        ingredient: Expected ingredient name.
        page_title: Title/food name from USDA page.

    Returns:
        True if the ingredient reasonably matches the page title.
    """
    if not page_title or not ingredient:
        return False

    ingredient_lower = ingredient.lower().strip()
    page_lower = page_title.lower().strip()

    # Direct substring match
    if ingredient_lower in page_lower:
        return True

    # Check individual words from ingredient
    ingredient_words = ingredient_lower.split()
    for word in ingredient_words:
        # Skip very short words
        if len(word) <= 2:
            continue
        if word in page_lower:
            return True

    # Special case mappings
    special_mappings = {
        "raw cashews": ["cashew", "nuts, cashew"],
        "water": ["water", "tap water", "drinking water"],
        "garlic": ["garlic"],
        "sea salt": ["salt", "sea salt", "sodium chloride"],
        "nutritional yeast": ["yeast", "nutritional yeast"],
        "lemon juice": ["lemon", "juice, lemon"],
        "onion powder": ["onion", "powder, onion"],
    }

    for key, patterns in special_mappings.items():
        if key in ingredient_lower:
            for pattern in patterns:
                if pattern in page_lower:
                    return True

    return False


def normalize_nutrient_name(col_name: str) -> Optional[str]:
    """
    Map column names to standard nutrient names.

    Args:
        col_name: Column name from spreadsheet.

    Returns:
        Standardized nutrient name, or None if not recognized.
    """
    if not col_name:
        return None

    col_lower = col_name.lower().strip()

    mappings = {
        "carbohydrate": "Carbohydrates",
        "carbs": "Carbohydrates",
        "fat": "Fat",
        "fiber": "Fiber",
        "protein": "Protein",
        "sugar": "Sugar",
        "calcium": "Calcium",
        "iron": "Iron",
        "potassium": "Potassium",
        "sodium": "Sodium",
        "vitamin a": "Vitamin A",
        "vit a": "Vitamin A",
        "vitamin c": "Vitamin C",
        "vit c": "Vitamin C",
    }

    for key, value in mappings.items():
        if key in col_lower:
            return value

    return None


def get_nutrient_columns_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Map standard nutrient names to actual column names in the spreadsheet.

    Args:
        columns: List of column names from the spreadsheet.

    Returns:
        Dictionary mapping standard nutrient names to actual column names.
    """
    mapping = {}
    for col in columns:
        normalized = normalize_nutrient_name(col)
        if normalized and normalized not in mapping:
            mapping[normalized] = col
    return mapping


def colors_are_similar(c1: Dict, c2: Dict, tolerance: float = 0.05) -> bool:
    """
    Check if two RGB colors are similar within tolerance.

    Args:
        c1: First color dict with 'red', 'green', 'blue' keys (0-1 scale).
        c2: Second color dict with 'red', 'green', 'blue' keys (0-1 scale).
        tolerance: Maximum allowed difference per channel (default 0.05).

    Returns:
        True if colors are similar within tolerance.
    """
    if not c1 or not c2:
        return False

    for channel in ['red', 'green', 'blue']:
        v1 = c1.get(channel, 1.0)
        v2 = c2.get(channel, 1.0)
        if abs(v1 - v2) > tolerance:
            return False

    return True


def colors_are_distinct(colors: List[Dict], tolerance: float = 0.1) -> bool:
    """
    Check if a list of colors are all distinct from each other.

    Args:
        colors: List of color dicts with 'red', 'green', 'blue' keys.
        tolerance: Minimum required difference to be considered distinct.

    Returns:
        True if all colors are distinct from each other.
    """
    if len(colors) < 2:
        return True

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if colors_are_similar(colors[i], colors[j], tolerance):
                return False

    return True


def find_column_by_keywords(columns: List[str], keywords: List[str]) -> Optional[str]:
    """
    Find a column that matches any of the given keywords.

    Args:
        columns: List of column names.
        keywords: List of keyword strings to match (case-insensitive).

    Returns:
        The matching column name, or None if not found.
    """
    for col in columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None


def find_ingredient_row(df, ingredient: str, ingredient_col: str, keywords: List[str]) -> Optional[int]:
    """
    Find the row index for an ingredient using keyword matching.

    Args:
        df: DataFrame containing the data.
        ingredient: Standard ingredient name.
        ingredient_col: Name of the ingredient column.
        keywords: List of keywords to match for this ingredient.

    Returns:
        Row index if found, None otherwise.
    """
    if ingredient_col not in df.columns:
        return None

    for idx, row in df.iterrows():
        cell_value = str(row[ingredient_col]).lower().strip()
        for keyword in keywords:
            if keyword.lower() in cell_value:
                return idx

    return None


# Keyword mappings for column detection
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

# Keyword mappings for ingredient detection
INGREDIENT_KEYWORDS = {
    "Raw Cashews": ["cashew", "raw cashew"],
    "Water": ["water"],
    "Garlic": ["garlic"],
    "Sea Salt": ["sea salt", "salt"],
    "Nutritional Yeast": ["nutritional yeast", "yeast"],
    "Lemon Juice": ["lemon juice", "lemon"],
    "Onion Powder": ["onion powder", "onion"],
}

# Keywords to detect excluded ingredient
EXCLUDED_KEYWORDS = ["black pepper", "pepper"]

# Expected ingredients in recipe order
EXPECTED_INGREDIENTS = [
    "Raw Cashews",
    "Water",
    "Garlic",
    "Sea Salt",
    "Nutritional Yeast",
    "Lemon Juice",
    "Onion Powder"
]

EXCLUDED_INGREDIENT = "Black Pepper"

# Nutrient groupings
MACRO_NUTRIENTS = ["Carbohydrates", "Fat", "Fiber", "Protein", "Sugar"]
MINERAL_NUTRIENTS = ["Calcium", "Iron", "Potassium", "Sodium"]
VITAMIN_NUTRIENTS = ["Vitamin A", "Vitamin C"]
ALL_NUTRIENTS = MACRO_NUTRIENTS + MINERAL_NUTRIENTS + VITAMIN_NUTRIENTS

# FDA Daily Values for 10% DV calculation
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

# Tolerance for numerical comparisons
VALUE_TOLERANCE = 0.20  # 20%
