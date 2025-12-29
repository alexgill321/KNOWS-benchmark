"""Utility functions for fetching and parsing Zillow and Craigslist listing data."""

import json
import os
import re
import tempfile
import time
from typing import Dict, List, Optional, Any

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright



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

        # Try to parse JSON from response
        response_text = response.strip()

        # Handle markdown code blocks
        if "```" in response_text:
            lines = response_text.split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            if json_lines:
                response_text = '\n'.join(json_lines)

        # Find JSON array in response
        start_idx = response_text.find('[')
        if start_idx != -1:
            bracket_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i
                        break
            response_text = response_text[start_idx:end_idx + 1]

        data = json.loads(response_text)

        # Ensure we return a list
        if isinstance(data, dict):
            data = [data]

        return data if data else None

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
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


def compare_addresses(addr1: str, addr2: str) -> bool:
    """
    Compare two addresses for approximate match.

    Args:
        addr1: First address string.
        addr2: Second address string.

    Returns:
        True if addresses are considered a match.
    """
    if not addr1 or not addr2:
        return False

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
        return True

    # Check if the main street address matches (first part before comma usually)
    parts1 = norm1.split()
    parts2 = norm2.split()

    # Match if first few significant parts match
    if len(parts1) >= 2 and len(parts2) >= 2:
        if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
            return True

    return False


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

        # Try to parse JSON from response
        response_text = response.strip()

        # Handle markdown code blocks
        if "```" in response_text:
            lines = response_text.split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            if json_lines:
                response_text = '\n'.join(json_lines)

        # Find JSON array in response
        start_idx = response_text.find('[')
        if start_idx != -1:
            bracket_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i
                        break
            response_text = response_text[start_idx:end_idx + 1]

        data = json.loads(response_text)

        # Ensure we return a list
        if isinstance(data, dict):
            data = [data]

        return data if data else None

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
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


def extract_craigslist_data_with_llm(html_content: str, model: Any) -> Optional[Dict]:
    """
    Use LLM to extract listing data from Craigslist HTML content.

    Args:
        html_content: HTML from Craigslist listing page.
        model: Loaded LLM model from eval_utils.models.

    Returns:
        Dictionary with extracted listing data, or None if extraction fails.
    """
    # Truncate HTML to avoid token limits
    truncated_html = html_content[:50000] if len(html_content) > 50000 else html_content

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
                "text": f"""Extract rental listing information from this Craigslist HTML.

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
            }]
        }
    ]

    try:
        response = model(messages)

        # Try to parse JSON from response
        response_text = response.strip()

        # Handle markdown code blocks
        if "```" in response_text:
            lines = response_text.split('\n')
            json_lines = []
            in_code_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    json_lines.append(line)
            if json_lines:
                response_text = '\n'.join(json_lines)

        # Find JSON object in response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx + 1]

        data = json.loads(response_text)
        return data if data else None

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response was: {response[:500]}...")
        return None
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
