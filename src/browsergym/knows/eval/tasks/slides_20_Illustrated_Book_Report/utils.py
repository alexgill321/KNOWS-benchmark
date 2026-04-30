"""
Utility functions for the Illustrated Book Report task evaluation.

This module provides helper functions for fetching and validating URL content
to verify that bullet point characteristics are direct quotes from sources.
"""

from src.browsergym.knows.eval.eval_utils.llm_utils import parse_yes_no
from src.browsergym.knows.eval.eval_utils.text_utils import text_fuzzy_match_contained_long
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_with_fallbacks


def contains_preserving_diacritics(needle, haystack):
    """
    Case-insensitive substring check that preserves diacritics.

    Unlike fuzzy matching (e.g., rapidfuzz partial_ratio), this rejects
    ASCII-stripped variants such as "Perisic" as a match for "Perišić",
    so the evaluator penalizes outputs that omit diacritics from gold names.

    Args:
        needle (str): The expected string (may contain diacritics).
        haystack (str): The text to search within.

    Returns:
        bool: True if needle appears in haystack with diacritics preserved.
    """
    if not needle or not haystack:
        return False
    return needle.casefold() in haystack.casefold()


def load_gold_characters(path):
    """
    Load gold characters from a TSV file.

    Each line is an alias group. The first entry on a line is the canonical
    name; any tab-separated entries after it are aliases that refer to the
    same character. Lines with no tabs represent a single-alias character.

    Args:
        path (str): Path to gold_characters.txt.

    Returns:
        tuple[list[str], dict[str, str]]: (all_aliases, alias_to_canonical).
        all_aliases is a flat list of every alias (suitable for fuzzy matching),
        and alias_to_canonical maps each alias back to its canonical name.
    """
    all_aliases = []
    alias_to_canonical = {}
    with open(path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            if not parts:
                continue
            canonical = parts[0]
            for alias in parts:
                all_aliases.append(alias)
                alias_to_canonical[alias] = canonical
    return all_aliases, alias_to_canonical


def fetch_url_content(url):
    """
    Fetch and convert URL to markdown text with multiple fallback strategies.

    Uses fetch_with_fallbacks which tries Playwright, Playwright retry with
    longer timeout, and Wayback Machine as a last resort.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: Markdown content (truncated to 60k chars), or None if fetch fails.
    """
    content, status = fetch_with_fallbacks(url, max_chars=60000, timeout=15)
    if content:
        return content
    print(f"All fetch strategies failed for {url}: {status}")
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
