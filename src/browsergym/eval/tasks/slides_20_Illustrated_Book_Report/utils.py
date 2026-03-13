"""
Utility functions for the Illustrated Book Report task evaluation.

This module provides helper functions for fetching and validating URL content
to verify that bullet point characteristics are direct quotes from sources.
"""

import html2text
from src.browsergym.eval.eval_utils.llm_utils import parse_yes_no
from src.browsergym.eval.eval_utils.text_utils import text_fuzzy_match_contained_long


def fetch_url_content(url):
    """
    Fetch and convert URL to markdown text using Playwright for JavaScript rendering.

    Uses Playwright to render JavaScript-heavy pages (like Fandom wikis) before
    extracting content. Falls back to requests for simpler pages.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: Markdown content (truncated to 60k chars), or None if fetch fails.

    Examples:
        >>> content = fetch_url_content("https://example.com/character-info")
        >>> if content:
        ...     print(f"Fetched {len(content)} characters of content")
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Launch headless browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            page = context.new_page()

            # Navigate: try domcontentloaded, fallback to load
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
            except Exception:
                page.goto(url, wait_until="load", timeout=5000)

            # Wait for selector (only reached if navigation succeeded)
            try:
                page.wait_for_selector("main, article, .mw-parser-output, #content", timeout=3000)
            except Exception:
                page.wait_for_timeout(500)
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    raise Exception(f"Timeout waiting for content from {url}: selector not found and networkidle (3s) exceeded")
           
            # Get the rendered HTML
            html_content = page.content()

            browser.close()

        if "JavaScript is disabled" in html_content:
            print(f"JavaScript appears to be disabled for {url}")
            return None
        
        # Convert HTML to Markdown
        h = html2text.HTML2Text()
        h.ignore_links = True  # Don't convert hyperlinks to markdown format
        h.ignore_images = True  # Skip image references
        h.body_width = 0  # Don't wrap lines
        markdown = h.handle(html_content)

        # Truncate to ~60k chars (~15k tokens) to prevent excessive LLM usage
        if len(markdown) > 60000:
            markdown = markdown[:60000]
        return markdown

    except Exception as e:
        print(f"Error fetching {url} with Playwright: {e}")
        return None


def validate_bullet_in_content(bullet_text, markdown_content, model):
    """
    Check if bullet text is a direct quote from content.

    Uses a two-tier validation approach:
    1. Fast fuzzy matching with 90% threshold for near-exact matches
    2. LLM validation as fallback for edge cases (formatting differences)

    Args:
        bullet_text (str): The bullet point text to validate.
        markdown_content (str): The markdown content to search within.
        model: The LLM model to use for validation if fuzzy match fails.

    Returns:
        bool: True if the bullet text is found as a quote, False otherwise.

    Examples:
        >>> bullet = "He is brave and fearless"
        >>> content = "The character is brave and fearless in battle."
        >>> validate_bullet_in_content(bullet, content, model)
        True
    """
    if not bullet_text or not markdown_content:
        return False

    # Method 1: Fuzzy match with 90% threshold
    # This catches near-exact quotes with minor formatting differences
    fuzzy_result = text_fuzzy_match_contained_long(bullet_text, markdown_content)

    if fuzzy_result[0]:
        print(f"Fuzzy match found for: {bullet_text[:50]}...")
        return True

    # Method 2: LLM validation for edge cases
    # Handles cases where formatting differences prevent fuzzy match
    # but the quote is still verbatim or nearly verbatim
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are validating if a quote appears verbatim or nearly verbatim in source content. "
                        "Respond with ONLY 'Yes' or 'No'."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"""Does this exact quote appear in the source content?

Quote: {bullet_text}

Source Content (Markdown):
{markdown_content}

Answer Yes only if the quote appears verbatim or with minimal formatting differences (like punctuation or whitespace).
Do NOT answer Yes if the content is paraphrased or summarized."""
            }]
        }
    ]

    try:
        response = model(messages)
        result = parse_yes_no(response) or False

        if result:
            print(f"LLM validated quote: {bullet_text[:50]}...")
        else:
            print(f"Quote not found: {bullet_text[:50]}...")

        return result

    except Exception as e:
        print(f"LLM validation error for '{bullet_text[:50]}...': {e}")
        return False
