"""
Utility functions for docs_11_personal_recipe_ocr evaluator.

This module provides functions to:
- Extract hyperlinks from Google Docs
- Extract content by section name
- Compare ingredient and preparation lists
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz


def extract_hyperlinks_from_doc(doc_id: str, service) -> List[Dict[str, str]]:
    """
    Extract all hyperlinks from a Google Doc.

    Extracts both embedded hyperlinks (textStyle.link.url) and plain text URLs.

    Args:
        doc_id: The Google Doc ID.
        service: The Google Docs API service instance.

    Returns:
        List of dicts with 'url' and 'text' (the linked text) keys.
    """
    document = service.documents().get(documentId=doc_id).execute()
    body = document.get('body', {})
    content = body.get('content', [])

    links = []
    url_pattern = re.compile(
        r'https?://[^\s<>"\'}\])\u200b\u00a0]+',
        re.IGNORECASE
    )

    def process_paragraph(paragraph):
        """Process a paragraph element for hyperlinks."""
        para_links = []
        para_elements = paragraph.get('elements', [])

        for elem in para_elements:
            if 'textRun' in elem:
                text_run = elem['textRun']
                content_text = text_run.get('content', '')
                text_style = text_run.get('textStyle', {})

                # Check for embedded hyperlink
                if 'link' in text_style:
                    url = text_style['link'].get('url', '')
                    if url:
                        para_links.append({
                            'url': url,
                            'text': content_text.strip()
                        })

                # Also check for plain text URLs
                plain_urls = url_pattern.findall(content_text)
                for plain_url in plain_urls:
                    # Don't duplicate if already found as hyperlink
                    if not any(l['url'] == plain_url for l in para_links):
                        para_links.append({
                            'url': plain_url,
                            'text': plain_url
                        })

        return para_links

    def process_table(table):
        """Process a table element for hyperlinks."""
        table_links = []
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                cell_content = cell.get('content', [])
                for elem in cell_content:
                    if 'paragraph' in elem:
                        table_links.extend(process_paragraph(elem['paragraph']))
        return table_links

    # Process all content elements
    for element in content:
        if 'paragraph' in element:
            links.extend(process_paragraph(element['paragraph']))
        elif 'table' in element:
            links.extend(process_table(element['table']))

    return links


def extract_section_content(doc_structure: List[Dict], section_name: str) -> str:
    """
    Extract text content from a named section in the document.

    Looks for section_name as a header, then collects all text until the next
    section header or end of document.

    Args:
        doc_structure: Document structure from extract_structure_from_doc().
        section_name: The section header to find (e.g., "Ingredients", "Preparation").

    Returns:
        The text content of that section, or empty string if not found.
    """
    section_content = []
    in_section = False
    section_name_lower = section_name.lower().strip()

    # Common section headers to detect section boundaries
    section_headers = ['ingredients', 'preparation', 'tips', 'ready in', 'serves', 'calories']

    for item in doc_structure:
        if item.get('type') != 'text':
            continue

        content = item.get('content', '').strip()
        content_lower = content.lower()

        # Check if this is the start of our target section
        if section_name_lower in content_lower and len(content) < 50:
            in_section = True
            continue

        # Check if we've hit another section header (end of our section)
        if in_section:
            is_new_section = any(
                header in content_lower and len(content) < 50
                for header in section_headers if header != section_name_lower
            )
            if is_new_section:
                break

            # Add content if not empty/whitespace
            if content and content not in section_headers:
                section_content.append(content)

    return '\n'.join(section_content)


def extract_list_items(text: str) -> List[str]:
    """
    Extract individual items from a text that might be a bulleted/numbered list.

    Handles various formats:
    - Newline-separated items
    - Bullet points (•, -, *)
    - Numbered items (1., 2., etc.)

    Args:
        text: The text content to parse.

    Returns:
        List of individual items, cleaned of bullets/numbers.
    """
    items = []

    # Split by newlines first
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common bullet/number prefixes
        line = re.sub(r'^[\•\-\*\>\◦]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = line.strip()

        if line:
            items.append(line)

    return items


def compare_ingredient_lists(
    doc_ingredients: List[str],
    gold_ingredients: List[str],
    fuzzy_threshold: int = 85
) -> Tuple[bool, List[Dict]]:
    """
    Compare document ingredients against gold standard.

    Args:
        doc_ingredients: List of ingredients from the document.
        gold_ingredients: List of expected ingredients.
        fuzzy_threshold: Minimum fuzzy match score (0-100).

    Returns:
        Tuple of (all_matched, details) where details is a list of match info.
    """
    details = []

    for gold in gold_ingredients:
        gold_normalized = gold.lower().strip()
        # Handle & vs "and" normalization
        gold_variants = [
            gold_normalized,
            gold_normalized.replace('&', 'and'),
            gold_normalized.replace('and', '&')
        ]

        best_match = None
        best_score = 0

        for doc_ing in doc_ingredients:
            doc_normalized = doc_ing.lower().strip()

            # Try exact match first
            if doc_normalized in gold_variants or any(gv in doc_normalized for gv in gold_variants):
                best_match = doc_ing
                best_score = 100
                break

            # Try fuzzy match
            score = fuzz.ratio(doc_normalized, gold_normalized)
            if score > best_score:
                best_score = score
                best_match = doc_ing

        matched = best_score >= fuzzy_threshold
        details.append({
            'gold': gold,
            'found': best_match,
            'score': best_score,
            'matched': matched
        })

    all_matched = all(d['matched'] for d in details)
    return all_matched, details


def compare_preparation_steps(
    doc_steps: List[str],
    gold_steps: List[str],
    fuzzy_threshold: int = 85
) -> Tuple[bool, List[Dict]]:
    """
    Compare document preparation steps against gold standard.

    Compares in order - each doc step should match the corresponding gold step.

    Args:
        doc_steps: List of preparation steps from the document.
        gold_steps: List of expected preparation steps.
        fuzzy_threshold: Minimum fuzzy match score (0-100).

    Returns:
        Tuple of (all_matched, details) where details is a list of match info.
    """
    details = []

    # Ensure we have enough steps
    if len(doc_steps) < len(gold_steps):
        for i, gold in enumerate(gold_steps):
            if i < len(doc_steps):
                doc_step = doc_steps[i]
                score = fuzz.token_sort_ratio(doc_step.lower(), gold.lower())
                details.append({
                    'step': i + 1,
                    'gold': gold[:50] + '...' if len(gold) > 50 else gold,
                    'found': doc_step[:50] + '...' if len(doc_step) > 50 else doc_step,
                    'score': score,
                    'matched': score >= fuzzy_threshold
                })
            else:
                details.append({
                    'step': i + 1,
                    'gold': gold[:50] + '...' if len(gold) > 50 else gold,
                    'found': None,
                    'score': 0,
                    'matched': False
                })
    else:
        for i, gold in enumerate(gold_steps):
            doc_step = doc_steps[i]
            score = fuzz.token_sort_ratio(doc_step.lower(), gold.lower())
            details.append({
                'step': i + 1,
                'gold': gold[:50] + '...' if len(gold) > 50 else gold,
                'found': doc_step[:50] + '...' if len(doc_step) > 50 else doc_step,
                'score': score,
                'matched': score >= fuzzy_threshold
            })

    all_matched = all(d['matched'] for d in details)
    return all_matched, details


def extract_recipe_metadata(doc_text: str) -> Dict[str, Optional[str]]:
    """
    Extract Ready In, Serves, and Calories values from document text.

    Args:
        doc_text: Full text content of the document.

    Returns:
        Dict with 'ready_in', 'serves', 'calories' keys (values may be None).
    """
    result = {
        'ready_in': None,
        'serves': None,
        'calories': None
    }

    # Patterns for each field
    ready_pattern = re.compile(r'ready\s*in[:\s]*(\d+(?:\s*-\s*\d+)?)\s*(?:min|minutes?)?', re.IGNORECASE)
    serves_pattern = re.compile(r'serves[:\s]*(\d+)', re.IGNORECASE)
    calories_pattern = re.compile(r'(\d+)\s*(?:cal|calories?)', re.IGNORECASE)

    ready_match = ready_pattern.search(doc_text)
    if ready_match:
        result['ready_in'] = ready_match.group(1)

    serves_match = serves_pattern.search(doc_text)
    if serves_match:
        result['serves'] = serves_match.group(1)

    cal_match = calories_pattern.search(doc_text)
    if cal_match:
        result['calories'] = cal_match.group(1)

    return result


def check_metadata_modified(metadata: Dict[str, Optional[str]], defaults: Dict[str, str]) -> Tuple[bool, str]:
    """
    Check if recipe metadata has been modified from template defaults.

    Args:
        metadata: Extracted metadata from document.
        defaults: Default values from template.

    Returns:
        Tuple of (is_modified, details_string).
    """
    changes = []

    # Check Ready In (expecting 36-50 min for pumpkin soup vs default 20)
    if metadata.get('ready_in'):
        ready_val = metadata['ready_in']
        if '-' in ready_val:
            # Range like "36 - 50"
            parts = ready_val.split('-')
            try:
                low = int(parts[0].strip())
                high = int(parts[1].strip())
                if low != 20 or high != 20:
                    changes.append(f"Ready In: {ready_val} (changed from {defaults.get('ready_in', '20')})")
            except ValueError:
                pass
        else:
            try:
                val = int(ready_val)
                if val != 20:
                    changes.append(f"Ready In: {val} (changed from {defaults.get('ready_in', '20')})")
            except ValueError:
                pass

    # Check Serves
    if metadata.get('serves'):
        try:
            serves_val = int(metadata['serves'])
            if serves_val != 8:
                changes.append(f"Serves: {serves_val} (changed from {defaults.get('serves', '8')})")
        except ValueError:
            pass

    # Check Calories
    if metadata.get('calories'):
        try:
            cal_val = int(metadata['calories'])
            if cal_val != 280:
                changes.append(f"Calories: {cal_val} (changed from {defaults.get('calories', '280')})")
        except ValueError:
            pass

    is_modified = len(changes) > 0
    details = '; '.join(changes) if changes else "No changes detected from defaults"

    return is_modified, details


def verify_tip_is_quote(tip_text: str, webpage_content: str, threshold: int = 90) -> Tuple[bool, int]:
    """
    Verify if a tip appears as a direct quote in webpage content.

    Args:
        tip_text: The tip text to verify.
        webpage_content: The fetched webpage text content.
        threshold: Minimum fuzzy match score for "direct quote" (0-100).

    Returns:
        Tuple of (is_quote, match_score).
    """
    if not tip_text or not webpage_content:
        return False, 0

    tip_normalized = tip_text.lower().strip()
    webpage_lower = webpage_content.lower()

    # First try exact substring match
    if tip_normalized in webpage_lower:
        return True, 100

    # Try fuzzy partial match for longer tips
    # Use sliding window approach for better matching
    tip_words = tip_normalized.split()
    if len(tip_words) > 5:
        # For longer tips, check if most words appear in content
        words_found = sum(1 for word in tip_words if word in webpage_lower)
        word_ratio = (words_found / len(tip_words)) * 100
        if word_ratio >= threshold:
            return True, int(word_ratio)

    # Try partial ratio for shorter tips
    score = fuzz.partial_ratio(tip_normalized, webpage_lower)
    return score >= threshold, score
