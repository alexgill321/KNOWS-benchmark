"""
Utility functions for the Illustrated Book Report task evaluation.

This module provides helper functions for fetching and validating URL content
to verify that bullet point characteristics are direct quotes from sources.
"""

import re

from src.browsergym.knows.eval.eval_utils.llm_utils import parse_yes_no
from src.browsergym.knows.eval.eval_utils.text_utils import text_fuzzy_match_contained_long
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_with_fallbacks


def is_source_link_at_bottom_of_content(slide, link):
    """
    Check if a source link appears at the bottom of a text box's content,
    even if the text box itself isn't positioned at the bottom of the slide.

    This handles the common case where the agent puts characteristics and
    the source URL in the same text box, with the source as the last line.

    Args:
        slide (dict): Slide object from Google Slides API.
        link (str): The URL to check.

    Returns:
        bool: True if the link is in the last 2 text runs of any text element.
    """
    if 'pageElements' not in slide:
        return False

    link_lower = link.lower()
    for element in slide.get('pageElements', []):
        if 'shape' not in element or 'text' not in element['shape']:
            continue
        text_runs = []
        for te in element['shape']['text'].get('textElements', []):
            if 'textRun' in te:
                content = te['textRun'].get('content', '').strip()
                if content:
                    text_runs.append(content)
        if not text_runs:
            continue
        # Check if link appears in the last 2 text runs
        last_runs = text_runs[-2:] if len(text_runs) >= 2 else text_runs
        for run in last_runs:
            if link_lower in run.lower():
                return True
    return False


def _line_is_source_reference(line):
    """Check if a text line is a source/URL reference rather than a characteristic."""
    lower = line.lower().strip()
    if 'http://' in lower or 'https://' in lower:
        return True
    if lower.startswith('source:') or lower.startswith('source -'):
        return True
    return False


def extract_slide_body_lines(slide, title_text=""):
    """
    Extract non-title, non-source text lines from a slide as a fallback
    when no bullet points are detected. Filters out empty lines, the title,
    and source/URL reference lines.

    Args:
        slide (dict): Slide object from Google Slides API.
        title_text (str): The title text to exclude.

    Returns:
        list[str]: Non-empty text lines from the slide body.
    """
    lines = []
    if 'pageElements' not in slide:
        return lines

    for element in slide.get('pageElements', []):
        if 'shape' not in element or 'text' not in element['shape']:
            continue
        text_element = element['shape']['text']
        for text_elem in text_element.get('textElements', []):
            if 'textRun' in text_elem:
                content = text_elem['textRun'].get('content', '').strip()
                if content and content != title_text:
                    lines.append(content)

    # Deduplicate while preserving order, filter out title and source lines
    seen = set()
    result = []
    title_lower = title_text.lower().strip() if title_text else ""
    for line in lines:
        if line.lower() == title_lower:
            continue
        if _line_is_source_reference(line):
            continue
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result


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


def match_character_name(title_text, gold_characters, threshold=80, return_method=False):
    """
    Match a slide title against gold character aliases.

    Uses word-boundary substring matching first (longest match wins),
    then falls back to fuzzy ratio matching. This avoids false positives
    from short aliases like "cat" matching inside "Scatterwind" that
    partial_ratio would produce.

    Args:
        title_text (str): The slide title text.
        gold_characters (list[str]): List of gold character aliases.
        threshold (int): Minimum fuzzy match score for fallback.
        return_method (bool): If True, return (matched_alias, score, method)
            where method is "word_boundary", "fuzzy", or None, recording
            which matching tier decided the result.

    Returns:
        tuple: (matched_alias, score) or (None, 0).
        If return_method is True, (matched_alias, score, method) instead.
    """
    from rapidfuzz import fuzz, process

    if not title_text or not gold_characters:
        return (None, 0, None) if return_method else (None, 0)

    # Word-boundary substring: longest alias that appears as a whole word
    best = None
    best_len = 0
    for alias in gold_characters:
        pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
        if re.search(pattern, title_text, re.IGNORECASE) and len(alias) > best_len:
            best = alias
            best_len = len(alias)
    if best:
        print(f"Matched '{best}' in text '{title_text}' with score 100.0")
        return (best, 100.0, "word_boundary") if return_method else (best, 100.0)

    # Fallback: fuzzy ratio (not partial_ratio) to avoid short-alias issues
    result = process.extractOne(
        title_text,
        gold_characters,
        scorer=fuzz.ratio,
        score_cutoff=threshold
    )
    if result:
        matched, score = result[0], result[1]
        print(f"Matched '{matched}' in text '{title_text}' with score {score}")
        return (matched, score, "fuzzy") if return_method else (matched, score)

    return (None, 0, "fuzzy") if return_method else (None, 0)


def load_gold_characters(path):
    """
    Load gold characters from a text file.

    Each line is an alias group. Tab-separated entries are treated as
    explicit aliases. Parenthetical entries like ``Name ("Alias")`` are
    also extracted automatically so that both the base name and the
    parenthetical are registered as aliases for the same canonical character.

    Args:
        path (str): Path to gold_characters.txt.

    Returns:
        tuple[list[str], dict[str, str]]: (all_aliases, alias_to_canonical).
        all_aliases is a flat list of every alias (suitable for fuzzy matching),
        and alias_to_canonical maps each alias back to its canonical name.
    """
    import re
    all_aliases = []
    alias_to_canonical = {}
    with open(path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            if not parts:
                continue
            # Expand parenthetical aliases from each part.
            # e.g. 'Virginia au Augustus ("Mustang")' ->
            #       ['Virginia au Augustus', 'Mustang']
            expanded = []
            for part in parts:
                paren_match = re.match(r'^(.+?)\s*\(\s*"?([^")]+)"?\s*\)\s*$', part)
                if paren_match:
                    expanded.append(paren_match.group(1).strip())
                    expanded.append(paren_match.group(2).strip())
                else:
                    expanded.append(part)
            canonical = expanded[0]
            for alias in expanded:
                if alias not in alias_to_canonical:
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


def validate_bullet_in_content(bullet_text, markdown_content, model, character_name="", book_title="", return_method=False):
    """
    Check if a bullet point characteristic is supported by the source content.

    Uses a two-tier validation approach:
    1. Fast fuzzy matching with 70% threshold for near-exact matches
    2. LLM validation to check if the characteristic is supported by the source

    Args:
        bullet_text (str): The bullet point text to validate.
        markdown_content (str): The markdown content to search within.
        model: The LLM model to use for validation if fuzzy match fails.
        character_name (str): The character the bullet describes.
        book_title (str): The book the character is from.
        return_method (bool): If True, return (bool, method) where method is
            "fuzzy", "llm", or None, recording which tier made the final call.

    Returns:
        bool: True if the characteristic is supported by the source, False otherwise.
        If return_method is True, (bool, method) instead.
    """
    if not bullet_text or not markdown_content:
        return (False, None) if return_method else False

    # Method 1: Fuzzy match with 70% threshold
    # This catches near-exact or closely paraphrased content
    fuzzy_result = text_fuzzy_match_contained_long(bullet_text, markdown_content, threshold=70)

    if fuzzy_result[0]:
        print(f"Fuzzy match found for: {bullet_text[:50]}...")
        return (True, "fuzzy") if return_method else True

    # Method 2: LLM validation — check if the characteristic is supported
    # by the source content for this specific character
    context_parts = []
    if character_name:
        context_parts.append(f"Character: {character_name}")
    if book_title:
        context_parts.append(f"Book: {book_title}")
    context_line = " | ".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are validating whether a character characteristic from a book report "
                        "is supported by the provided source content. "
                        "Respond with ONLY 'Yes' or 'No'."
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"""Is this characteristic of {character_name or 'a character'} supported by the source content?

{context_line}
Characteristic: {bullet_text}

Source Content (Markdown):
{markdown_content}

Answer Yes if the source content contains information that supports or corroborates this characteristic for this specific character.
Answer No if the characteristic is not supported, contradicted, or describes a different character."""
            }]
        }
    ]

    try:
        response = model(messages)
        result = parse_yes_no(response) or False

        if result:
            print(f"LLM validated characteristic: {bullet_text[:50]}...")
        else:
            print(f"Not supported by source: {bullet_text[:50]}...")

        return (result, "llm") if return_method else result

    except Exception as e:
        print(f"LLM validation error for '{bullet_text[:50]}...': {e}")
        return (False, "llm") if return_method else False
