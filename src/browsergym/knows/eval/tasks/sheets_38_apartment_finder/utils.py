"""Utility functions for fetching and parsing Zillow and Craigslist listing data."""

import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Any

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from src.browsergym.knows.eval.eval_utils.llm_utils import (
    strip_markdown_code_blocks,
    extract_json_from_llm_response,
)



def clean_html(html: str) -> str:
    """
    Clean HTML by removing scripts, styles, and other non-content elements.

    Args:
        html: Raw HTML string.

    Returns:
        Cleaned HTML with scripts, styles, and metadata removed.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script and style elements
    for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
        element.decompose()

    return str(soup)


def extract_listing_data_with_llm(html_content: str, model: Any) -> Optional[List[Dict]]:
    """
    Use LLM to extract listing data from Zillow HTML content.

    Handles both single listings and multi-unit apartment buildings that show
    multiple available units on a single page.

    Args:
        html_content: HTML from Zillow listing page.
        model: Loaded LLM model from eval_utils.models.

    Returns:
        List of dictionaries with extracted listing data for each unit,
        or None if extraction fails.
    """
    # Truncate HTML to avoid token limits
    truncated_html = html_content[:50000] if len(html_content) > 50000 else html_content

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """You are a data extraction assistant. Extract rental listing information from Zillow HTML.
Always respond with valid JSON only, no other text."""
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"""Extract rental listing information from this Zillow HTML.

IMPORTANT: This page may show MULTIPLE apartment units available in a building.
Extract information for EACH available unit separately.

For each unit, extract:
1. Monthly rent price in USD (number only, no $ sign)
2. Number of bedrooms (use 0 for studio)
3. Number of bathrooms
4. Full address (include unit number if available)
5. Does it have in-unit laundry/washer/dryer? (Yes/No/Unknown)
6. Is it pet-friendly (allows cats or dogs)? (Yes/No/Unknown)
7. Square footage (number only)

Respond ONLY with this exact JSON format (array of units):
[
    {{
        "price": <number or null>,
        "bedrooms": <number or null>,
        "bathrooms": <number or null>,
        "address": "<string or null>",
        "in_unit_laundry": "<Yes/No/Unknown>",
        "pet_friendly": "<Yes/No/Unknown>",
        "sqft": <number or null>
    }}
]

If there is only ONE unit, still return an array with a single object.

HTML Content:
{truncated_html}"""
            }]
        }
    ]

    try:
        response = model(messages)

        data = extract_json_from_llm_response(response, expect_type="array")

        if data is None:
            return None

        # Ensure we return a list
        if isinstance(data, dict):
            data = [data]

        return data if data else None

    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return None


def normalize_boolean_value(value: str) -> Optional[bool]:
    """
    Normalize various boolean string representations to True/False/None.

    Args:
        value: String value like "Yes", "No", "Unknown", "true", etc.

    Returns:
        True, False, or None for unknown values.
    """
    if value is None:
        return None

    value_lower = str(value).lower().strip()

    if value_lower in ['yes', 'true', '1', 'y', 'allowed', 'included']:
        return True
    elif value_lower in ['no', 'false', '0', 'n', 'not allowed', 'none']:
        return False
    else:
        return None  # Unknown


def compare_addresses(addr1: str, addr2: str, model=None, return_method: bool = False):
    """
    Compare two addresses for approximate match.

    Uses fast string matching first, then LLM fallback for semantic comparison.

    Args:
        addr1: First address string.
        addr2: Second address string.
        model: Optional LLM model for semantic fallback.
        return_method: If True, return (matched, method) where method is
            "exact" when the normalized/containment tier made the final call,
            "llm" when the LLM tier ran last (decided or errored), and None
            when an input address was empty.

    Returns:
        bool: True if addresses are considered a match. If return_method is
            True, returns (bool, str or None) instead.
    """
    def _result(matched, method):
        return (matched, method) if return_method else matched

    if not addr1 or not addr2:
        return _result(False, None)

    # Normalize addresses
    def normalize(addr):
        addr = addr.lower().strip()
        # Remove common abbreviations variations
        addr = addr.replace(',', ' ')
        addr = addr.replace('.', ' ')
        addr = re.sub(r'\s+', ' ', addr)
        # Normalize common words
        replacements = {
            'street': 'st',
            'avenue': 'ave',
            'boulevard': 'blvd',
            'drive': 'dr',
            'road': 'rd',
            'lane': 'ln',
            'court': 'ct',
            'apartment': 'apt',
            'suite': 'ste',
            'unit': '#',
            'north': 'n',
            'south': 's',
            'east': 'e',
            'west': 'w',
        }
        for full, abbr in replacements.items():
            addr = addr.replace(f' {full} ', f' {abbr} ')
            addr = addr.replace(f' {full}', f' {abbr}')
        return addr.strip()

    norm1 = normalize(addr1)
    norm2 = normalize(addr2)

    # Check if one contains the other (for partial matches)
    if norm1 in norm2 or norm2 in norm1:
        return _result(True, "exact")

    # LLM fallback for semantic address comparison
    if model is not None:
        try:
            prompt = f"""You are comparing two address strings from a rental listing to determine if they refer to the same street address.

Address 1: "{addr1}"
Address 2: "{addr2}"

They match if they clearly refer to the same street address, allowing for:
- Abbreviation differences (St vs Street, Ave vs Avenue, N vs North)
- Minor formatting differences
- Slightly different levels of detail (e.g., "North Damen Ave" vs "North Damen Ave near Wolfram")

They do NOT match if:
- One is a neighborhood/area name and the other is a street address (e.g., "Wicker Park" vs "1316 North Artesian Ave")
- They refer to different streets or locations

Respond with ONLY "YES" or "NO"."""

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = model(messages).strip().upper()
            return _result(response == "YES", "llm")
        except Exception as e:
            print(f"Error in LLM address comparison: {e}")
            return _result(False, "llm")

    return _result(False, "exact")


def is_valid_zillow_url(url: str) -> bool:
    """
    Check if a URL is a valid Zillow listing URL.

    Args:
        url: URL string to validate.

    Returns:
        True if URL appears to be a valid Zillow listing URL.
    """
    if not url:
        return False

    url_lower = url.lower().strip()

    # Must be zillow.com
    if 'zillow.com' not in url_lower:
        return False

    # Should be a rental/homedetails page
    valid_patterns = [
        '/homedetails/',
        '/apartments/',
        '/rental/',
        '/homes/',
        '/b/',  # Building pages
    ]

    return any(pattern in url_lower for pattern in valid_patterns)


def capture_zillow_screenshot(url: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Capture a screenshot of a Zillow listing page using Oxylabs with screenshot rendering.

    Uses Oxylabs to bypass Zillow's bot detection and capture a rendered screenshot
    of the listing page.

    Args:
        url: The Zillow listing URL.
        output_path: Optional path to save the image. If None, saves to temp file.

    Returns:
        Path to the saved screenshot image, or None if capture fails.
    """
    import base64

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    try:
        client = RealtimeClient(OXYLABS_USERNAME, OXYLABS_PASSWORD)

        # Use render: png to get a screenshot instead of HTML
        result = client.universal.scrape_url(url, render="png")

        if not result or not result.raw:
            print(f"No content returned from Oxylabs for {url}")
            return None

        # Get the base64-encoded screenshot from the response
        results = result.raw.get('results', [])
        if not results:
            print(f"No results in Oxylabs response for {url}")
            return None

        # The screenshot is base64 encoded in the content field
        screenshot_b64 = results[0].get('content')
        if not screenshot_b64:
            print(f"No screenshot content in Oxylabs response for {url}")
            return None

        # Decode base64 and save to file
        screenshot_bytes = base64.b64decode(screenshot_b64)
        with open(output_path, 'wb') as f:
            f.write(screenshot_bytes)

        print(f"Screenshot captured via Oxylabs: {output_path}")
        return output_path

    except Exception as e:
        print(f"Screenshot capture failed for {url}: {e}")
        return None


def capture_zillow_screenshot_playwright(url: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Capture a screenshot of a Zillow listing page using Playwright (direct access).

    Note: This may be blocked by Zillow's bot detection. Use capture_zillow_screenshot()
    which uses Oxylabs for more reliable results.

    Args:
        url: The Zillow listing URL.
        output_path: Optional path to save the image. If None, saves to temp file.

    Returns:
        Path to the saved screenshot image, or None if capture fails.
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".png")

    try:
        with sync_playwright() as p:
            # Launch headless Chromium browser
            browser = p.chromium.launch(headless=True)

            # Create context with realistic user agent to avoid bot detection
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )

            page = context.new_page()

            # Navigate to the URL with longer timeout
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Wait for network to be idle with longer timeout, but don't fail if it times out
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:
                # If networkidle times out, continue anyway - page should be mostly loaded
                pass

            # Additional wait for dynamic content
            time.sleep(3)

            # Capture full page screenshot
            page.screenshot(path=output_path, full_page=True)

            browser.close()

        print(f"Screenshot captured via Playwright: {output_path}")
        return output_path

    except Exception as e:
        print(f"Playwright screenshot capture failed for {url}: {e}")
        return None


def extract_listing_data_from_screenshot(screenshot_path: str, model: Any) -> Optional[List[Dict]]:
    """
    Use LLM vision capabilities to extract listing data from a Zillow screenshot.

    Args:
        screenshot_path: Path to the screenshot image file.
        model: Loaded LLM model with vision support (e.g., gemma-google-ai).

    Returns:
        List of dictionaries with extracted listing data for each unit,
        or None if extraction fails.
    """
    if not os.path.exists(screenshot_path):
        print(f"Screenshot file not found: {screenshot_path}")
        return None

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """You are a data extraction assistant. Extract rental listing information from Zillow screenshots.
Always respond with valid JSON only, no other text."""
            }]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": screenshot_path},
                {"type": "text", "text": """Extract rental listing information from this Zillow screenshot.

IMPORTANT: This page may show MULTIPLE apartment units available in a building.
Extract information for EACH available unit separately.

For each unit, extract:
1. Monthly rent price in USD (number only, no $ sign)
2. Number of bedrooms (use 0 for studio)
3. Number of bathrooms
4. Full address (include unit number if available)
5. Does it have in-unit laundry/washer/dryer? (Yes/No/Unknown)
6. Is it pet-friendly (allows cats or dogs)? (Yes/No/Unknown)
7. Square footage (number only)

Respond ONLY with this exact JSON format (array of units):
[
    {
        "price": <number or null>,
        "bedrooms": <number or null>,
        "bathrooms": <number or null>,
        "address": "<string or null>",
        "in_unit_laundry": "<Yes/No/Unknown>",
        "pet_friendly": "<Yes/No/Unknown>",
        "sqft": <number or null>
    }
]

If there is only ONE unit, still return an array with a single object."""}
            ]
        }
    ]

    try:
        response = model(messages)

        data = extract_json_from_llm_response(response, expect_type="array")

        if data is None:
            return None

        # Ensure we return a list
        if isinstance(data, dict):
            data = [data]

        return data if data else None

    except Exception as e:
        print(f"Error in LLM extraction from screenshot: {e}")
        return None


# =============================================================================
# Craigslist Functions
# =============================================================================

def fetch_craigslist_page(url: str, raw: bool = False) -> Optional[str]:
    """
    Fetch content from a Craigslist listing URL using requests.

    Craigslist does not have aggressive bot detection, so we can use
    simple HTTP requests with a realistic user agent.

    Args:
        url: The Craigslist listing URL to fetch.
        raw: If True, return raw HTML. If False (default), return cleaned HTML.

    Returns:
        Page content as string (cleaned or raw HTML), or None if fetch fails.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        html_content = response.text

        if raw:
            return html_content

        return clean_html(html_content)

    except requests.RequestException as e:
        print(f"Error fetching Craigslist page {url}: {e}")
        return None


def extract_craigslist_data_structured(html_content: str) -> Optional[Dict]:
    """
    Extract listing data from Craigslist HTML using DOM parsing and regex.

    Extracts all fields needed across all sheets_38 instances:
    - Core: price, bedrooms, bathrooms, address, sqft
    - Instance 1: in_unit_laundry, pet_friendly
    - Instance 2/4: furnished
    - Instance 3: no_app_fee, off_street_parking
    - Instance 4: on_site_laundry
    - Instance 5: air_conditioning

    Args:
        html_content: Raw or cleaned HTML from a Craigslist listing page.

    Returns:
        Dictionary with extracted listing data, or None if parsing fails entirely.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
    except Exception:
        return None

    data = {
        "price": None,
        "bedrooms": None,
        "bathrooms": None,
        "address": None,
        "sqft": None,
        # Boolean fields: "Yes" / "No" / "Unknown"
        "in_unit_laundry": "Unknown",
        "on_site_laundry": "Unknown",
        "pet_friendly": "Unknown",
        "furnished": "Unknown",
        "no_app_fee": "Unknown",
        "off_street_parking": "Unknown",
        "air_conditioning": "Unknown",
    }

    # --- Price: <span class="price">$1,100</span> ---
    price_el = soup.select_one('.price')
    if price_el:
        price_text = price_el.get_text(strip=True)
        price_match = re.search(r'[\$]?([\d,]+)', price_text)
        if price_match:
            try:
                data["price"] = float(price_match.group(1).replace(',', ''))
            except ValueError:
                pass

    # --- Beds/Baths/Sqft from first .attrgroup ---
    # Typical format: "2BR / 1Ba", "900ft2"
    first_attrgroup = soup.select_one('.attrgroup')
    if first_attrgroup:
        for span in first_attrgroup.select('span'):
            text = span.get_text(strip=True)

            # Bedrooms/Bathrooms: "2BR / 1Ba" or "0BR / 1Ba" (studio)
            bed_bath_match = re.search(r'(\d+)\s*BR\s*/\s*(\d+(?:\.\d+)?)\s*Ba', text, re.IGNORECASE)
            if bed_bath_match:
                try:
                    data["bedrooms"] = float(bed_bath_match.group(1))
                    data["bathrooms"] = float(bed_bath_match.group(2))
                except ValueError:
                    pass

            # Square footage: "900ft2" or "1073ft2"
            sqft_match = re.search(r'(\d+)\s*ft2', text, re.IGNORECASE)
            if sqft_match:
                try:
                    data["sqft"] = float(sqft_match.group(1))
                except ValueError:
                    pass

    # --- Address: .mapaddress element ---
    mapaddr = soup.select_one('.mapaddress')
    if mapaddr:
        addr_text = mapaddr.get_text(strip=True)
        if addr_text:
            data["address"] = addr_text

    # --- Scan all .attrgroup spans for amenity/boolean fields ---
    # Craigslist uses consistent span text in later attrgroups for amenities.
    # Known span values (from live inspection across SLC, LA, Houston, Seattle, Chicago):
    #   pets:      "cats are OK - purrr", "dogs are OK - wooof"
    #   laundry:   "w/d in unit", "laundry in bldg", "laundry on site", "no laundry on site"
    #   parking:   "off-street parking", "attached garage", "detached garage", "carport", "street parking"
    #   ac:        "air conditioning"
    #   furnished: "furnished" (rare in attrgroups, more common in body text)
    #   ev:        "EV charging"
    for attrgroup in soup.select('.attrgroup'):
        for span in attrgroup.select('span'):
            text_lower = span.get_text(strip=True).lower()

            # -- Pet-friendly --
            if 'cats are ok' in text_lower or 'dogs are ok' in text_lower:
                data["pet_friendly"] = "Yes"

            # -- In-unit laundry --
            if 'w/d in unit' in text_lower:
                data["in_unit_laundry"] = "Yes"
                # If w/d in unit, on_site_laundry is also effectively yes
                if data["on_site_laundry"] == "Unknown":
                    data["on_site_laundry"] = "Yes"
            elif 'laundry in bldg' in text_lower or 'laundry on site' in text_lower:
                data["on_site_laundry"] = "Yes"
                if data["in_unit_laundry"] == "Unknown":
                    data["in_unit_laundry"] = "No"
            elif 'no laundry' in text_lower:
                data["in_unit_laundry"] = "No"
                data["on_site_laundry"] = "No"

            # -- Off-street parking --
            if text_lower in ('off-street parking', 'attached garage', 'detached garage', 'carport'):
                data["off_street_parking"] = "Yes"
            elif text_lower == 'street parking':
                if data["off_street_parking"] == "Unknown":
                    data["off_street_parking"] = "No"

            # -- Air conditioning --
            if text_lower == 'air conditioning':
                data["air_conditioning"] = "Yes"

            # -- Furnished (rare in attrgroups but possible) --
            if text_lower == 'furnished':
                data["furnished"] = "Yes"

    # --- Posting body fallback for fields not found in attrgroups ---
    body = soup.select_one('#postingbody')
    body_lower = body.get_text().lower() if body else ""

    # Pet-friendly fallback
    if data["pet_friendly"] == "Unknown" and body_lower:
        if 'no pets' in body_lower or 'pets not allowed' in body_lower:
            data["pet_friendly"] = "No"
        elif 'pet friendly' in body_lower or 'pets welcome' in body_lower or 'pets ok' in body_lower:
            data["pet_friendly"] = "Yes"

    # Furnished fallback (commonly mentioned in body text)
    if data["furnished"] == "Unknown" and body_lower:
        # Check for "unfurnished" first to avoid false positive from substring match
        if 'unfurnished' in body_lower:
            data["furnished"] = "No"
        elif 'fully furnished' in body_lower or 'comes furnished' in body_lower:
            data["furnished"] = "Yes"
        elif re.search(r'\bfurnished\b', body_lower):
            data["furnished"] = "Yes"

    # No application fee fallback
    if data["no_app_fee"] == "Unknown" and body_lower:
        if re.search(r'no\s+(application|app)\s+fee', body_lower):
            data["no_app_fee"] = "Yes"
        elif re.search(r'(application|app)\s+fee', body_lower):
            data["no_app_fee"] = "No"

    # Air conditioning fallback
    if data["air_conditioning"] == "Unknown" and body_lower:
        if re.search(r'\b(central\s+air|a/?c\b|air\s+condition)', body_lower):
            data["air_conditioning"] = "Yes"

    # Off-street parking fallback
    if data["off_street_parking"] == "Unknown" and body_lower:
        if re.search(r'(garage|carport|off[- ]street\s+parking|covered\s+parking|parking\s+included)', body_lower):
            data["off_street_parking"] = "Yes"

    # On-site laundry fallback
    if data["on_site_laundry"] == "Unknown" and body_lower:
        if re.search(r'(laundry\s+(room|facility|on[- ]site|in\s+bldg|in\s+building)|washer.*dryer)', body_lower):
            data["on_site_laundry"] = "Yes"

    # Return None only if we got absolutely nothing useful
    if all(v is None or v == "Unknown" for v in data.values()):
        return None

    return data


def is_valid_craigslist_url(url: str) -> bool:
    """
    Check if a URL is a valid Craigslist listing URL.

    Args:
        url: URL string to validate.

    Returns:
        True if URL appears to be a valid Craigslist listing URL.
    """
    if not url:
        return False

    url_lower = url.lower().strip()

    # Must be craigslist.org
    if 'craigslist.org' not in url_lower:
        return False

    # Should be an apartment/housing listing
    valid_patterns = [
        '/apa/',      # Apartments/housing for rent
        '/sub/',      # Sublets/temporary
        '/hsw/',      # Housing swap
        '/hou/',      # Housing
        '/roo/',      # Rooms/shared
    ]

    return any(pattern in url_lower for pattern in valid_patterns)


def extract_craigslist_data_with_llm(html_content: str, model: Any, text: str = "") -> Optional[Dict]:
    """
    Use LLM to extract listing data from Craigslist HTML content.

    Args:
        html_content: HTML from Craigslist listing page.
        model: Loaded LLM model from eval_utils.models.

    Returns:
        Dictionary with extracted listing data, or None if extraction fails.
    """
    # Truncate HTML to avoid token limits
    task_text = text
    truncated_html = html_content[:50000] if len(html_content) > 50000 else html_content
    if task_text:
        task_text += f"""HTML Content:
{truncated_html}"""
    else:
        task_text = f"""Extract rental listing information from this Craigslist HTML.

For this listing, extract:
1. Monthly rent price in USD (number only, no $ sign)
2. Number of bedrooms (use 0 for studio)
3. Number of bathrooms
4. Full address (if available)
5. Does it have in-unit laundry/washer/dryer? (Yes/No/Unknown)
6. Is it pet-friendly (allows cats or dogs)? (Yes/No/Unknown)
7. Square footage (number only)
8. Any other notable amenities or features

Respond ONLY with this exact JSON format:
{{
    "price": <number or null>,
    "bedrooms": <number or null>,
    "bathrooms": <number or null>,
    "address": "<string or null>",
    "in_unit_laundry": "<Yes/No/Unknown>",
    "pet_friendly": "<Yes/No/Unknown>",
    "sqft": <number or null>,
    "amenities": ["list", "of", "amenities"]
}}

HTML Content:
{truncated_html}"""
        
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": """You are a data extraction assistant. Extract rental listing information from Craigslist HTML.
Always respond with valid JSON only, no other text."""
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": task_text
            }]
        }
    ]

    try:
        response = model(messages)

        data = extract_json_from_llm_response(response, expect_type="object")
        return data if data else None

    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return None


def fetch_and_extract_craigslist_listing(url: str, model: Any) -> Optional[Dict]:
    """
    Convenience function to fetch and extract data from a Craigslist listing.

    Combines fetch_craigslist_page and extract_craigslist_data_with_llm into
    a single call.

    Args:
        url: The Craigslist listing URL.
        model: Loaded LLM model from eval_utils.models.

    Returns:
        Dictionary with extracted listing data, or None if fetch/extraction fails.
    """
    if not is_valid_craigslist_url(url):
        print(f"Invalid Craigslist URL: {url}")
        return None

    html_content = fetch_craigslist_page(url)
    if not html_content:
        return None

    return extract_craigslist_data_with_llm(html_content, model)


def extract_craigslist_data_with_fallback(
    html_content: str,
    model: Any,
    required_fields: List[str],
    llm_prompt: str = "",
) -> Optional[Dict]:
    """
    Extract listing data using structured parsing first, LLM as fallback.

    Runs extract_craigslist_data_structured() first (fast, deterministic).
    If any of the required_fields come back as None or "Unknown", runs the
    LLM extraction and merges results — structured values take priority,
    LLM fills in the gaps.

    Args:
        html_content: Raw HTML from a Craigslist listing page.
        model: Loaded LLM model (only called if structured extraction has gaps).
        required_fields: List of field keys that must be resolved (e.g.
            ["price", "bedrooms", "bathrooms", "address", "in_unit_laundry"]).
        llm_prompt: Optional custom prompt text for the LLM extraction.

    Returns:
        Dictionary with extracted listing data, or None if both methods fail.
    """
    # Field aliases: maps evaluator key -> structured extractor key
    # (structured extractor uses canonical names; some evaluators use variants)
    FIELD_ALIASES = {
        "fully_furnished": "furnished",
    }

    # Phase 1: Structured extraction (fast, no LLM)
    structured = extract_craigslist_data_structured(html_content)

    if structured is None:
        structured = {}

    # Copy aliased fields so evaluators can access by their expected key
    for alias, canonical in FIELD_ALIASES.items():
        if canonical in structured and alias not in structured:
            structured[alias] = structured[canonical]

    # Check if all required fields are resolved
    missing_fields = []
    for field in required_fields:
        val = structured.get(field)
        if val is None or val == "Unknown":
            missing_fields.append(field)

    if not missing_fields:
        # All required fields resolved — no need for LLM
        return structured

    # Phase 2: LLM fallback for missing fields
    print(f"    Structured extraction missing {len(missing_fields)} required fields: {missing_fields}")
    print(f"    Running LLM fallback...")

    try:
        llm_result = extract_craigslist_data_with_llm(html_content, model, text=llm_prompt)
    except Exception as e:
        print(f"    LLM fallback failed: {e}")
        llm_result = None

    if not llm_result:
        return structured if structured else None

    # Merge: structured values take priority, LLM fills gaps
    merged = dict(structured)
    for field in missing_fields:
        llm_val = llm_result.get(field)
        if llm_val is not None:
            merged[field] = llm_val

    return merged if merged else None
