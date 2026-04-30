"""
Utility functions for docs_11_personal_recipe_ocr evaluator.

This module provides functions to:
- Discover recipes and their boundaries in a document (Phase 1)
- Extract content from individual recipes (Phase 2)
- Compare ingredient and preparation lists
- Document setup and cleanup utilities
"""

import os
import re
import shutil
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz


# ============================================================================
# RECIPE DISCOVERY CONSTANTS AND DATA STRUCTURES
# ============================================================================

REQUIRED_SECTIONS = ['ingredients', 'preparation', 'tips']

# Section headers used to detect section boundaries within a recipe
SECTION_HEADERS = ['ingredients', 'preparation', 'tips', 'ready in', 'serves', 'calories']

# Default template content (from Coral Recipe template doc)
# Used to detect if sections have been modified from the template
TEMPLATE_DEFAULT_CONTENT = {
    'ingredients': (
        "Lorem ipsum dolor sit amet\n"
        "Consectetuer adipiscing elit\n"
        "Suspendisse scelerisque\n"
        "Libero interdum auctor"
    ),
    'preparation': (
        "Lorem ipsum dolor sit amet consectetuer adipiscing elit sed do tempor incididunt ut labore et dolore magna aliqua.\n"
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.\n"
        "Suspendisse scelerisque mi a mi. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed dolore eiusmod tempor.\n"
        "Vestibulum ante ipsum primis elementum, libero interdum auctor cursus, sapien enim dictum quam.\n"
        "Phasellus vehicula nonummy nunc. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut enim ad minim veniam, quis nostrud exercitation.\n"
        "Ullamco laboris nisi ut aliquip ex ea commodo consequat."
    ),
    'tips': (
        "Lorem ipsum dolor sit amet consectetuer adipiscing elit sed do tempor incididunt ut labore et dolore magna aliqua."
    ),
}


@dataclass
class Recipe:
    """Represents a single recipe extracted from the document."""
    recipe_num: int                                        # 1-based: 1 = original, 2-4 = additional
    text: str                                              # Raw text content for this recipe
    structure: List[Dict] = field(default_factory=list)    # Structure elements scoped to this recipe
    start_index: int = 0                                   # Char offset in full doc_text
    end_index: int = 0                                     # Char offset in full doc_text
    title: str = ""                                        # Populated eagerly during discovery
    sections_found: List[str] = field(default_factory=list)  # Which required sections exist
    structure_indices: Tuple[int, int] = (0, 0)            # (start, end) indices in doc_structure
    pdf_pages: List[int] = field(default_factory=list)     # 0-indexed PDF page numbers for this recipe


def map_recipes_to_pdf_pages(recipes: List['Recipe'], pdf_path: str) -> None:
    """
    Map each recipe to its PDF page(s) by extracting text from each page
    and matching RECIPE headers.

    Uses PyMuPDF to extract text per page, finds pages containing "RECIPE"
    headers, and assigns page ranges to each recipe. Mutates recipes in-place.

    Args:
        recipes: List of Recipe objects with titles set.
        pdf_path: Path to the PDF file.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("Warning: PyMuPDF not available, cannot map recipes to PDF pages")
        return

    if not recipes or not os.path.exists(pdf_path):
        return

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Warning: Failed to open PDF for page mapping: {e}")
        return

    num_pages = doc.page_count

    # Find which pages contain a RECIPE header (case-sensitive)
    recipe_start_pages = []  # List of 0-indexed page numbers
    for page_idx in range(num_pages):
        page_text = doc[page_idx].get_text()
        if 'RECIPE' in page_text:
            recipe_start_pages.append(page_idx)

    doc.close()

    # Assign page ranges: each recipe spans from its start page to
    # the page before the next recipe's start (or end of document)
    for i, recipe in enumerate(recipes):
        if i < len(recipe_start_pages):
            start_page = recipe_start_pages[i]
            if i + 1 < len(recipe_start_pages):
                end_page = recipe_start_pages[i + 1] - 1
            else:
                end_page = num_pages - 1
            recipe.pdf_pages = list(range(start_page, end_page + 1))
        else:
            print(f"Warning: No PDF page found for recipe {recipe.recipe_num}")
            recipe.pdf_pages = []

    print("Recipe to PDF page mapping:")
    for r in recipes:
        print(f"  Recipe {r.recipe_num} ('{r.title}'): pages {[p+1 for p in r.pdf_pages]} (1-indexed)")


def discover_recipes(doc_text: str, doc_structure: List[Dict]) -> List[Recipe]:
    """
    Discover all recipes in the document and validate their structure.

    Phase 1 of the 2-phase recipe processing approach:
    1. Find candidate RECIPE headers in doc_structure
    2. Validate each candidate has all required sections (Ingredients, Preparation, Tips)
    3. Validate no overlapping boundaries between adjacent recipes
    4. Build Recipe objects with text, structure, title, and section info

    Args:
        doc_text: Full document text content.
        doc_structure: Full document structure from extract_structure_from_doc().

    Returns:
        List of Recipe objects, ordered by position in the document.
    """
    if not doc_text or not doc_structure:
        return []

    # Step 1: Find candidate RECIPE headers in doc_structure
    candidates = []  # List of (structure_index, content) tuples
    for idx, item in enumerate(doc_structure):
        if item.get('type') != 'text':
            continue
        content = item.get('content', '').strip()
        # Case-sensitive match per BUG-009: avoid matching lowercase "recipe" in body text
        # Use \b word boundary to match "RECIPE" but not "RECIPES"
        # Also handles concatenated titles like "RECIPEPumpkin Soup"
        if re.match(r'^RECIPE\b', content):
            candidates.append((idx, content))

    if not candidates:
        print("Warning: No RECIPE headers found in document structure")
        return []

    # Step 2: Validate each candidate has required sections
    validated = []  # List of dicts with header_idx, end_idx, sections_found, section_indices, candidate_idx

    for i, (header_idx, header_content) in enumerate(candidates):
        # Determine the range of structure elements for this candidate
        if i + 1 < len(candidates):
            next_header_idx = candidates[i + 1][0]
        else:
            next_header_idx = len(doc_structure)

        # Scan for required section headers within this range
        sections_found = []
        section_indices = {}  # section_name -> structure_index

        for scan_idx in range(header_idx + 1, next_header_idx):
            item = doc_structure[scan_idx]
            if item.get('type') != 'text':
                continue
            content = item.get('content', '').strip()
            content_lower = content.lower()

            # Check if this is a section header (short text matching a known section name)
            if len(content) < 50:
                for section in REQUIRED_SECTIONS:
                    if content_lower == section and section not in sections_found:
                        sections_found.append(section)
                        section_indices[section] = scan_idx
                        break

        # Only accept candidates with all required sections
        if set(REQUIRED_SECTIONS).issubset(set(sections_found)):
            validated.append({
                'header_idx': header_idx,
                'end_idx': next_header_idx,
                'sections_found': sections_found,
                'section_indices': section_indices,
                'candidate_idx': i,  # Track which candidate this was for text marker mapping
            })
        else:
            missing = set(REQUIRED_SECTIONS) - set(sections_found)
            print(f"Warning: RECIPE header at structure index {header_idx} "
                  f"('{header_content[:30]}') missing required sections: {missing}. Skipping.")

    if not validated:
        print("Warning: No valid recipes found (all candidates missing required sections)")
        return []

    # Step 3: Validate no overlapping boundaries
    for i in range(len(validated) - 1):
        current = validated[i]
        next_recipe = validated[i + 1]

        if current['end_idx'] > next_recipe['header_idx']:
            print(f"Warning: Overlapping boundaries between recipe at index "
                  f"{current['header_idx']} and recipe at index {next_recipe['header_idx']}")

        # Check that no section indices in current overlap with next recipe's range
        for section, sec_idx in current['section_indices'].items():
            if sec_idx >= next_recipe['header_idx']:
                print(f"Warning: Section '{section}' at index {sec_idx} in recipe "
                      f"{i + 1} overlaps with recipe {i + 2} starting at index "
                      f"{next_recipe['header_idx']}")

    # Step 4: Build Recipe objects
    # Find text markers for slicing doc_text (same case-sensitive pattern)
    text_markers = list(re.finditer(r'\bRECIPE\b', doc_text))
    recipes = []

    for recipe_idx, v in enumerate(validated):
        recipe_num = recipe_idx + 1
        candidate_idx = v['candidate_idx']

        # Determine text boundaries using candidate_idx to map to text_markers
        if recipe_num == 1:
            # First recipe starts at beginning of document
            text_start = 0
        elif candidate_idx < len(text_markers):
            text_start = text_markers[candidate_idx].start()
        else:
            print(f"Warning: No text marker for candidate {candidate_idx}, using end of previous recipe")
            text_start = recipes[-1].end_index if recipes else 0

        # Find the end: next validated recipe's text marker, or end of document
        if recipe_idx + 1 < len(validated):
            next_candidate_idx = validated[recipe_idx + 1]['candidate_idx']
            if next_candidate_idx < len(text_markers):
                text_end = text_markers[next_candidate_idx].start()
            else:
                text_end = len(doc_text)
        else:
            text_end = len(doc_text)

        recipe_text = doc_text[text_start:text_end].strip()

        # Extract structure elements for this recipe
        structure_start = v['header_idx']
        structure_end = v['end_idx']
        recipe_structure = []
        for s_idx in range(structure_start, structure_end):
            item = doc_structure[s_idx]
            # Skip the RECIPE header element itself from the structure
            # Skip the RECIPE header element itself from the scoped structure
            if s_idx == structure_start:
                content = item.get('content', '').strip()
                if item.get('type') == 'text' and re.match(r'^RECIPE\b', content):
                    continue
            recipe_structure.append(item)

        # Extract title
        title = extract_recipe_title(recipe_text)

        recipe = Recipe(
            recipe_num=recipe_num,
            text=recipe_text,
            structure=recipe_structure,
            start_index=text_start,
            end_index=text_end,
            title=title,
            sections_found=v['sections_found'],
            structure_indices=(structure_start, structure_end),
        )
        recipes.append(recipe)

    print(f"Discovered {len(recipes)} valid recipes in document")
    for r in recipes:
        print(f"  Recipe {r.recipe_num}: title='{r.title}', sections={r.sections_found}, "
              f"structure_indices={r.structure_indices}")

    return recipes


def extract_hyperlinks(
    doc_id: str,
    service,
    recipe_num: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Extract hyperlinks from a Google Doc, optionally filtered to a specific recipe.

    Extracts both embedded hyperlinks (textStyle.link.url) and plain text URLs.
    Uses RECIPE headers as boundaries to scope extraction to a specific recipe.

    Args:
        doc_id: The Google Doc ID.
        service: The Google Docs API service instance.
        recipe_num: If specified (1-4), extract links only from that recipe.
                    If None, extract all links from the entire document.

    Returns:
        List of dicts with 'url' and 'text' keys.
    """
    document = service.documents().get(documentId=doc_id).execute()
    body = document.get('body', {})
    content = body.get('content', [])

    links = []
    url_pattern = re.compile(
        r'https?://[^\s<>"\'}\])\u200b\u00a0]+',
        re.IGNORECASE
    )

    recipe_count = 0
    collecting = recipe_num is None  # If no filter, collect from the start

    def process_paragraph(paragraph) -> List[Dict[str, str]]:
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
                    if not any(l['url'] == plain_url for l in para_links):
                        para_links.append({
                            'url': plain_url,
                            'text': plain_url
                        })

        return para_links

    def is_recipe_header(paragraph) -> bool:
        """Check if a paragraph is a RECIPE header."""
        para_elements = paragraph.get('elements', [])
        for elem in para_elements:
            if 'textRun' in elem:
                content_text = elem['textRun'].get('content', '').strip()
                # Case-sensitive match consistent with discover_recipes
                if re.match(r'^RECIPE\b', content_text):
                    return True
        return False

    def process_table(table) -> Tuple[List[Dict[str, str]], bool]:
        """Process a table element for hyperlinks. Returns (links, recipe_header_found)."""
        table_links = []
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                cell_content = cell.get('content', [])
                for elem in cell_content:
                    if 'paragraph' in elem:
                        if is_recipe_header(elem['paragraph']):
                            return table_links, True
                        if collecting:
                            table_links.extend(process_paragraph(elem['paragraph']))
        return table_links, False

    def handle_recipe_header() -> bool:
        """Handle encountering a RECIPE header. Returns True if we should stop processing."""
        nonlocal recipe_count, collecting

        recipe_count += 1

        if recipe_num is None:
            # No filtering — always collecting, never stop
            return False

        if recipe_count == recipe_num:
            collecting = True
        elif recipe_count > recipe_num:
            collecting = False
            return True  # Past our target recipe, stop

        return False

    # Process content elements
    for element in content:
        if 'paragraph' in element:
            if is_recipe_header(element['paragraph']):
                if handle_recipe_header():
                    break
                continue
            if collecting:
                links.extend(process_paragraph(element['paragraph']))
        elif 'table' in element:
            table_links, found_header = process_table(element['table'])
            if collecting:
                links.extend(table_links)
            if found_header:
                if handle_recipe_header():
                    break

    return links


# ============================================================================
# DOCUMENT SETUP AND CLEANUP UTILITIES
# ============================================================================

def cleanup_generated_files(data_dir: str, cleanup_enabled: bool = True):
    """
    Clean up all generated files and directories created during evaluation.

    Args:
        data_dir: Base data directory path.
        cleanup_enabled: If False, skip cleanup.
    """
    if not cleanup_enabled:
        print("Cleanup disabled by CLEANUP=False environment variable")
        return

    print("Cleaning up generated files...")
    cleanup_dirs = [
        os.path.join(data_dir, "images/"),
        os.path.join(data_dir, "cropped_images/"),
        os.path.join(data_dir, "pdf_images/")
    ]

    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

    # Clean up PDF files
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            os.remove(pdf_file)
            print(f"Removed PDF file: {pdf_file}")
        except Exception as e:
            print(f"Error removing PDF file {pdf_file}: {e}")

    print("Cleanup completed")


def setup_document(
    workspace_doc_id: str,
    data_dir: str,
    drive_service,
    docs_service,
    pdf_dpi: int = 150
) -> Dict[str, Any]:
    """
    Setup document processing using the provided workspace_doc_id.

    Args:
        workspace_doc_id: The Google Docs document ID to evaluate.
        data_dir: Directory for data files.
        drive_service: Google Drive service instance.
        docs_service: Google Docs service instance.
        pdf_dpi: DPI for PDF conversion.

    Returns:
        Dict with keys: doc_id, doc_text, doc_structure
    """
    from src.browsergym.knows.eval.eval_utils.google_services_utils import (
        download_doc_as_pdf,
        extract_text_from_doc,
        extract_structure_from_doc,
        extract_images_from_doc,
        extract_images_from_doc_with_cropping,
    )
    from src.browsergym.knows.eval.eval_utils.image_utils import convert_pdf_to_pngs

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace document ID: {workspace_doc_id}")

    # Ensure data directories exist
    doc_images_dir = os.path.join(data_dir, "images/")
    doc_images_cropped_dir = os.path.join(data_dir, "cropped_images/")
    pdf_images_dir = os.path.join(data_dir, "pdf_images/")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(doc_images_dir, exist_ok=True)
    os.makedirs(doc_images_cropped_dir, exist_ok=True)
    os.makedirs(pdf_images_dir, exist_ok=True)

    # Download and convert PDF
    pdf_path = os.path.join(data_dir, "recipe_doc.pdf")
    download_doc_as_pdf(workspace_doc_id, pdf_path, drive_service)
    convert_pdf_to_pngs(pdf_path, pdf_images_dir, dpi=pdf_dpi)

    # Extract text and structure
    doc_text = extract_text_from_doc(workspace_doc_id, docs_service)
    doc_structure = extract_structure_from_doc(workspace_doc_id, docs_service)

    # Extract images
    extract_images_from_doc(workspace_doc_id, docs_service, doc_images_dir)
    extract_images_from_doc_with_cropping(workspace_doc_id, docs_service, doc_images_cropped_dir)

    return {
        'doc_id': workspace_doc_id,
        'doc_text': doc_text,
        'doc_structure': doc_structure,
    }


def load_gold_data(golds_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load gold standard data from files.

    Args:
        golds_dir: Directory containing gold standard files.

    Returns:
        Tuple of (gold_ingredients, gold_prepsteps).
    """
    gold_ingredients_file = os.path.join(golds_dir, "gold_ingredients.txt")
    gold_prepsteps_file = os.path.join(golds_dir, "gold_prepsteps.txt")

    gold_ingredients = []
    gold_prepsteps = []

    if os.path.exists(gold_ingredients_file):
        with open(gold_ingredients_file, 'r') as f:
            gold_ingredients = [line.strip() for line in f if line.strip()]

    if os.path.exists(gold_prepsteps_file):
        with open(gold_prepsteps_file, 'r') as f:
            gold_prepsteps = [line.strip() for line in f if line.strip()]

    return gold_ingredients, gold_prepsteps


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
    fuzzy_threshold: int = 95,
    max_extra_allowed: int = 1
) -> Tuple[bool, List[Dict]]:
    """
    Compare document ingredients against gold standard with strict 1:1 matching.

    Uses fuzzy_match_text from shared utils for each comparison.
    Each gold ingredient must match exactly one doc ingredient (1:1, no reuse).
    Extra ingredients beyond gold are flagged if they exceed max_extra_allowed.

    Args:
        doc_ingredients: List of ingredients from the document.
        gold_ingredients: List of expected ingredients.
        fuzzy_threshold: Minimum fuzzy match score (0-100). Default 95 for strict matching.
        max_extra_allowed: Maximum extra ingredients allowed beyond gold list. Default 1.

    Returns:
        Tuple of (all_matched, details) where details is a list of match info.
    """
    from src.browsergym.knows.eval.eval_utils.text_utils import fuzzy_match_text

    def _normalize(text: str) -> str:
        """Normalize common variations for ingredient comparison."""
        return text.replace('&', 'and').strip()

    details = []
    remaining_doc = list(doc_ingredients)  # Mutable copy for 1:1 removal

    # Match each gold ingredient to the best remaining doc ingredient
    for gold in gold_ingredients:
        best_match = None
        best_score = 0
        best_idx = -1

        for idx, doc_ing in enumerate(remaining_doc):
            _, score = fuzzy_match_text(_normalize(doc_ing), _normalize(gold), threshold=fuzzy_threshold)
            if score > best_score:
                best_score = score
                best_match = doc_ing
                best_idx = idx

        matched = best_score >= fuzzy_threshold
        if matched and best_idx >= 0:
            remaining_doc.pop(best_idx)  # Remove matched item (1:1 enforcement)

        details.append({
            'gold': gold,
            'found': best_match,
            'score': best_score,
            'matched': matched
        })

    # Check for extra ingredients (bidirectional validation)
    extra_ingredients = [ing for ing in remaining_doc if len(ing.strip()) > 2]

    all_gold_matched = all(d['matched'] for d in details)
    extra_within_limit = len(extra_ingredients) <= max_extra_allowed

    if extra_ingredients:
        details.append({
            'gold': '[Extra ingredients check]',
            'found': extra_ingredients,
            'score': 0 if len(extra_ingredients) > max_extra_allowed else 100,
            'matched': extra_within_limit,
            'extra_count': len(extra_ingredients)
        })

    return all_gold_matched and extra_within_limit, details


def extract_numbers_from_text(text: str) -> List[str]:
    """
    Extract all numbers (including time values) from text.

    Args:
        text: Text to extract numbers from.

    Returns:
        List of number strings found in the text (e.g., ['15', '1', '20']).
    """
    # Match numbers including decimals and those attached to units
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    return numbers


def extract_cooking_verbs(text: str) -> List[str]:
    """
    Extract key cooking verbs from text that are critical to the recipe.

    Args:
        text: Text to extract verbs from.

    Returns:
        List of cooking verbs found (lowercased).
    """
    # Key cooking verbs that affect recipe outcome
    cooking_verbs = [
        'melt', 'cook', 'stir', 'simmer', 'serve', 'add', 'turn',
        'heat', 'boil', 'bake', 'fry', 'sauté', 'saute', 'roast',
        'blend', 'mix', 'pour', 'slice', 'chop', 'dice', 'fold'
    ]

    text_lower = text.lower()
    found_verbs = []

    for verb in cooking_verbs:
        if re.search(r'\b' + verb + r'\b', text_lower):
            found_verbs.append(verb)

    return found_verbs


def compare_preparation_steps(
    doc_steps: List[str],
    gold_steps: List[str],
    fuzzy_threshold: int = 85,
    require_exact_numbers: bool = True,
    require_exact_verbs: bool = True
) -> Tuple[bool, List[Dict]]:
    """
    Compare document preparation steps against gold standard with strict validation.

    Compares in order - each doc step should match the corresponding gold step.
    Additionally validates:
    1. Numbers/times must match exactly (e.g., "15 min" vs "5 min" fails)
    2. Key cooking verbs must match exactly (e.g., "simmer" vs "summer" fails)

    Args:
        doc_steps: List of preparation steps from the document.
        gold_steps: List of expected preparation steps.
        fuzzy_threshold: Minimum fuzzy match score for text (0-100).
        require_exact_numbers: If True, all numbers in gold must appear in doc.
        require_exact_verbs: If True, all cooking verbs in gold must appear in doc.

    Returns:
        Tuple of (all_matched, details) where details is a list of match info.
    """
    details = []

    for i, gold in enumerate(gold_steps):
        if i < len(doc_steps):
            doc_step = doc_steps[i]

            # Phase 1: Fuzzy text match
            text_score = fuzz.token_sort_ratio(doc_step.lower(), gold.lower())

            # Phase 2: Extract and compare numbers (times, quantities)
            gold_numbers = extract_numbers_from_text(gold)
            doc_numbers = extract_numbers_from_text(doc_step)
            numbers_match = True
            missing_numbers = []

            if require_exact_numbers and gold_numbers:
                for num in gold_numbers:
                    if num not in doc_numbers:
                        numbers_match = False
                        missing_numbers.append(num)

            # Phase 3: Extract and compare cooking verbs
            gold_verbs = extract_cooking_verbs(gold)
            doc_verbs = extract_cooking_verbs(doc_step)
            verbs_match = True
            missing_verbs = []

            if require_exact_verbs and gold_verbs:
                for verb in gold_verbs:
                    if verb not in doc_verbs:
                        verbs_match = False
                        missing_verbs.append(verb)

            # Combined match: text must be similar AND numbers/verbs must match
            text_matched = text_score >= fuzzy_threshold
            overall_matched = text_matched and numbers_match and verbs_match

            # Build detail info
            detail = {
                'step': i + 1,
                'gold': gold[:50] + '...' if len(gold) > 50 else gold,
                'found': doc_step[:50] + '...' if len(doc_step) > 50 else doc_step,
                'score': text_score,
                'matched': overall_matched,
                'text_matched': text_matched,
                'numbers_matched': numbers_match,
                'verbs_matched': verbs_match
            }

            # Add failure reasons
            if not overall_matched:
                failure_reasons = []
                if not text_matched:
                    failure_reasons.append(f"text score {text_score} < {fuzzy_threshold}")
                if not numbers_match:
                    failure_reasons.append(f"missing numbers: {missing_numbers}")
                if not verbs_match:
                    failure_reasons.append(f"missing verbs: {missing_verbs}")
                detail['failure_reason'] = '; '.join(failure_reasons)

            details.append(detail)
        else:
            # Missing step
            details.append({
                'step': i + 1,
                'gold': gold[:50] + '...' if len(gold) > 50 else gold,
                'found': None,
                'score': 0,
                'matched': False,
                'failure_reason': 'step missing from document'
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
    # Ready In: handle "X hours Y minutes", "X hour Y minutes", "X minutes", or just "X"
    ready_hour_min = re.compile(
        r'ready\s*in[:\s]*(\d+)\s*hours?\s*(?:and\s*)?(\d+)\s*(?:min|minutes?)?',
        re.IGNORECASE
    )
    ready_min_only = re.compile(
        r'ready\s*in[:\s]*(\d+)\s*(?:min|minutes)',
        re.IGNORECASE
    )
    serves_pattern = re.compile(r'serves[:\s]*(\d+)', re.IGNORECASE)
    calories_pattern = re.compile(r'(\d+)\s*(?:cal|calories?)', re.IGNORECASE)

    # Try hour+min format first, then minutes-only
    hour_min_match = ready_hour_min.search(doc_text)
    if hour_min_match:
        hours = int(hour_min_match.group(1))
        minutes = int(hour_min_match.group(2))
        result['ready_in'] = str(hours * 60 + minutes)
    else:
        min_match = ready_min_only.search(doc_text)
        if min_match:
            result['ready_in'] = min_match.group(1)

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


# ============================================================================
# CHECKPOINT 2: ADDITIONAL RECIPE UTILITIES
# ============================================================================

def extract_recipe_title(recipe_text: str) -> str:
    """
    Extract the title from a recipe's text content.

    The title is typically the first non-empty line after "RECIPE" header,
    usually containing the recipe name like "Pumpkin Soup" or "Butternut Squash Soup".

    BUG-009 fix: Handle cases where title is concatenated with RECIPE header
    (e.g., "RECIPE\x0bPumpkin Soup" or "RECIPESoupe au Potiron").

    Args:
        recipe_text: Text content of the recipe (from discover_recipes).

    Returns:
        The extracted title, or empty string if not found.
    """
    if not recipe_text:
        return ""

    # BUG-009 fix: Check if recipe_text starts with "RECIPE" followed by title
    # The title may be separated by vertical tab (\x0b), space, or directly concatenated
    if recipe_text.upper().startswith('RECIPE'):
        # Extract the portion after "RECIPE"
        after_recipe = recipe_text[6:]  # Skip "RECIPE" (6 chars)
        # Strip leading whitespace and vertical tabs
        after_recipe = after_recipe.lstrip('\x0b \t')
        # Get the first line after RECIPE
        first_line = after_recipe.split('\n')[0].strip()
        # Check if this looks like a title (not a URL or section header)
        if first_line and not first_line.startswith('http'):
            first_line_lower = first_line.lower()
            if first_line_lower not in ['ingredients', 'preparation', 'tips', 'ready in', 'serves', 'calories']:
                if not re.match(r'^\d+\s*(min|cal|servings?)', first_line, re.IGNORECASE):
                    return first_line

    # Fallback: Original line-by-line parsing
    lines = recipe_text.split('\n')

    # Skip the "RECIPE" header and find the title
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip the RECIPE header itself
        if re.match(r'^RECIPE\s*$', line, re.IGNORECASE):
            continue
        # Skip common section headers
        if line.lower() in ['ingredients', 'preparation', 'tips', 'ready in', 'serves', 'calories']:
            continue
        # Skip lines that look like metadata (contains numbers with units)
        if re.match(r'^\d+\s*(min|cal|servings?)', line, re.IGNORECASE):
            continue
        # Skip URLs
        if line.startswith('http'):
            continue
        # This should be the title
        return line

    return ""


def check_title_theme(title: str) -> Tuple[bool, str]:
    """
    Check if a recipe title relates to fall/autumn themes.

    Args:
        title: The recipe title to check.

    Returns:
        Tuple of (is_thematic, details).
    """
    if not title:
        return False, "No title found"

    title_lower = title.lower()

    # Keywords related to fall, soups, pumpkins, thanksgiving
    theme_keywords = [
        'pumpkin', 'squash', 'butternut', 'acorn',
        'soup', 'stew', 'chowder', 'bisque',
        'fall', 'autumn', 'harvest',
        'thanksgiving', 'turkey', 'cranberry',
        'gourd', 'warm', 'comfort', 'cozy',
        'apple', 'cider', 'cinnamon', 'spice',
        'sweet potato', 'yam'
    ]

    found_keywords = [kw for kw in theme_keywords if kw in title_lower]

    if found_keywords:
        return True, f"Found thematic keywords: {', '.join(found_keywords)}"

    return False, f"No fall/soup/pumpkin/thanksgiving keywords found in '{title}'"


def check_content_modified_from_default(
    content: str,
    content_type: str
) -> Tuple[bool, str]:
    """
    Check if recipe content (ingredients, preparation, tips) is modified from template default.

    Compares against the actual Coral Recipe template Lorem ipsum placeholder text
    using fuzzy similarity. Content is considered unmodified if it has high overlap
    with the template default.

    Args:
        content: The extracted content text.
        content_type: One of 'ingredients', 'preparation', 'tips'.

    Returns:
        Tuple of (is_modified, details).
    """
    if not content or len(content.strip()) < 10:
        return False, f"{content_type.capitalize()} section is empty or too short"

    default_content = TEMPLATE_DEFAULT_CONTENT.get(content_type, "")
    if not default_content:
        # No template default to compare against — assume modified if non-empty
        return True, f"{content_type.capitalize()} has content (no template default to compare)"

    # Compare using fuzzy ratio — high similarity means content is unchanged
    similarity = fuzz.ratio(content.strip().lower(), default_content.strip().lower())

    if similarity >= 70:
        return False, f"{content_type.capitalize()} still matches template default ({similarity}% similar)"

    return True, f"{content_type.capitalize()} modified from default ({similarity}% similar to template)"


# ============================================================================
# CHECKPOINT 3: WEBPAGE CONTENT EXTRACTION UTILITIES
# ============================================================================

def extract_recipe_image_url(url: str, timeout: int = 15) -> Optional[str]:
    """
    Extract the main recipe image URL from a webpage.

    Tries multiple strategies in order:
    1. Schema.org Recipe structured data (ld+json) — most precise
    2. Open Graph og:image meta tag — widely supported

    Args:
        url: The recipe webpage URL.
        timeout: Request timeout in seconds.

    Returns:
        The image URL, or None if not found.
    """
    import requests
    import json
    from bs4 import BeautifulSoup

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: Failed to fetch {url} for image extraction: {e}")
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Strategy 1: ld+json Recipe schema
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
            # Handle both single object and @graph array formats
            items = [data] if isinstance(data, dict) and '@type' in data else []
            if isinstance(data, dict) and '@graph' in data:
                items = data['@graph']
            elif isinstance(data, list):
                items = data

            for item in items:
                if item.get('@type') == 'Recipe':
                    img = item.get('image', '')
                    if isinstance(img, list):
                        img = img[0] if img else ''
                    if isinstance(img, dict):
                        img = img.get('url', '')
                    if isinstance(img, str) and img.startswith('http'):
                        print(f"Found recipe image via ld+json: {img[:80]}")
                        return img
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    # Strategy 2: og:image meta tag
    og_tag = soup.find('meta', property='og:image')
    if og_tag:
        img_url = og_tag.get('content', '')
        if img_url and img_url.startswith('http'):
            print(f"Found recipe image via og:image: {img_url[:80]}")
            return img_url

    return None


def extract_ingredients_from_webpage(model, webpage_text: str, recipe_title: str) -> List[str]:
    """
    Extract ingredient list from webpage text using LLM.

    Recipe websites use varied formats (JSON-LD, HTML lists, plain text).
    LLM extraction is more robust than regex for parsing these formats.

    Args:
        model: Pre-loaded LLM model for extraction.
        webpage_text: Raw text content from the recipe webpage.
        recipe_title: The recipe title to help focus extraction.

    Returns:
        List of ingredient strings extracted from the webpage.
    """
    if not webpage_text or len(webpage_text.strip()) < 100:
        return []

    # Truncate webpage text to avoid token limits
    max_chars = 10000
    if len(webpage_text) > max_chars:
        webpage_text = webpage_text[:max_chars]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """You are extracting the ingredient list from a recipe webpage.

Instructions:
1. Find the ingredients section of the recipe
2. Extract each ingredient as a separate line
3. Include quantities and measurements (e.g., "2 cups flour", "1 tsp salt")
4. Do NOT include preparation instructions or notes
5. Return ONLY the ingredient list, one ingredient per line
6. If no ingredients found, return "NO_INGREDIENTS_FOUND"

Example output format:
2 cups all-purpose flour
1 teaspoon baking powder
1/2 cup butter, softened
1 large egg"""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Extract the ingredient list for this recipe: {recipe_title}\n\nWebpage content:\n{webpage_text}"}]
        }
    ]

    try:
        response = model(messages)
        if "NO_INGREDIENTS_FOUND" in response:
            return []

        # Parse the response into a list
        ingredients = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove common bullet/number prefixes
            line = re.sub(r'^[\•\-\*\>\◦]\s*', '', line)
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = line.strip()
            if line and len(line) > 2:
                ingredients.append(line)

        return ingredients
    except Exception as e:
        print(f"Error extracting ingredients from webpage: {e}")
        return []


def extract_preparation_from_webpage(model, webpage_text: str, recipe_title: str) -> List[str]:
    """
    Extract preparation steps from webpage text using LLM.

    Recipe websites use varied formats. LLM extraction is more robust
    than regex for parsing these varied formats.

    Args:
        model: Pre-loaded LLM model for extraction.
        webpage_text: Raw text content from the recipe webpage.
        recipe_title: The recipe title to help focus extraction.

    Returns:
        List of preparation step strings extracted from the webpage.
    """
    if not webpage_text or len(webpage_text.strip()) < 100:
        return []

    # Truncate webpage text to avoid token limits
    max_chars = 10000
    if len(webpage_text) > max_chars:
        webpage_text = webpage_text[:max_chars]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """You are extracting the preparation/cooking steps from a recipe webpage.

Instructions:
1. Find the directions/instructions/method section of the recipe
2. Extract each step as a separate line
3. Include cooking times, temperatures, and specific actions
4. Keep steps in order (they may or may not be numbered)
5. Do NOT include ingredient lists or notes/tips
6. Return ONLY the preparation steps, one step per line
7. If no steps found, return "NO_STEPS_FOUND"

Example output format:
Preheat oven to 350°F (175°C).
Mix flour and baking powder in a bowl.
Cream butter and sugar until fluffy, about 3 minutes.
Add eggs one at a time, beating well after each addition."""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Extract the preparation steps for this recipe: {recipe_title}\n\nWebpage content:\n{webpage_text}"}]
        }
    ]

    try:
        response = model(messages)
        if "NO_STEPS_FOUND" in response:
            return []

        # Parse the response into a list
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove common bullet/number prefixes
            line = re.sub(r'^[\•\-\*\>\◦]\s*', '', line)
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^Step\s*\d+[:\.\)]\s*', '', line, flags=re.IGNORECASE)
            line = line.strip()
            if line and len(line) > 10:
                steps.append(line)

        return steps
    except Exception as e:
        print(f"Error extracting preparation from webpage: {e}")
        return []


def extract_metadata_from_webpage(model, webpage_text: str) -> Dict[str, Optional[str]]:
    """
    Extract recipe metadata (Ready In, Serves, Calories) from webpage text using LLM.

    Websites use varied formats for metadata (JSON-LD, structured data, plain text).
    LLM extraction handles these varied formats robustly.

    Args:
        model: Pre-loaded LLM model for extraction.
        webpage_text: Raw text content from the recipe webpage.

    Returns:
        Dict with 'ready_in', 'serves', 'calories' keys (values may be None).
    """
    result = {
        'ready_in': None,
        'serves': None,
        'calories': None
    }

    if not webpage_text or len(webpage_text.strip()) < 100:
        return result

    # Truncate webpage text to avoid token limits
    max_chars = 8000
    if len(webpage_text) > max_chars:
        webpage_text = webpage_text[:max_chars]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """You are extracting recipe metadata from a webpage.

Extract these three values ONLY:
1. Ready In / Total Time / Cook Time (in minutes)
2. Serves / Servings / Yield (number of servings)
3. Calories (per serving)

Return in this EXACT format (use "null" if not found):
READY_IN: <number or null>
SERVES: <number or null>
CALORIES: <number or null>

Example outputs:
READY_IN: 45
SERVES: 6
CALORIES: 285

READY_IN: null
SERVES: 4
CALORIES: null"""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Extract ready time, servings, and calories from this recipe:\n\n{webpage_text}"}]
        }
    ]

    try:
        response = model(messages)

        # Parse the structured response
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('READY_IN:'):
                val = line.replace('READY_IN:', '').strip()
                if val.lower() != 'null':
                    # Extract just the number
                    match = re.search(r'(\d+)', val)
                    if match:
                        result['ready_in'] = match.group(1)
            elif line.startswith('SERVES:'):
                val = line.replace('SERVES:', '').strip()
                if val.lower() != 'null':
                    match = re.search(r'(\d+)', val)
                    if match:
                        result['serves'] = match.group(1)
            elif line.startswith('CALORIES:'):
                val = line.replace('CALORIES:', '').strip()
                if val.lower() != 'null':
                    match = re.search(r'(\d+)', val)
                    if match:
                        result['calories'] = match.group(1)

        return result
    except Exception as e:
        print(f"Error extracting metadata from webpage: {e}")
        return result


def compare_lists_with_llm(model, doc_list: List[str], source_list: List[str], list_type: str) -> Tuple[bool, str]:
    """
    Compare two lists (ingredients or preparation steps) using LLM for semantic matching.

    This is a fallback when exact/fuzzy matching fails due to formatting differences.

    Args:
        model: Pre-loaded LLM model for comparison.
        doc_list: List from the document.
        source_list: List from the source webpage.
        list_type: Either "ingredients" or "preparation steps".

    Returns:
        Tuple of (is_match, details).
    """
    if not doc_list or not source_list:
        return False, f"Empty list: doc has {len(doc_list)} items, source has {len(source_list)} items"

    # Join as full text blocks rather than bullet lists — this avoids the LLM
    # being misled by different item counts when content is condensed or split differently.
    doc_text = ' '.join(doc_list)
    source_text = ' '.join(source_list)

    # Truncate to avoid token limits
    doc_text = doc_text[:5000]
    source_text = source_text[:5000]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": f"""You are comparing two {list_type} texts from a recipe.

Determine if the document text contains substantially the same content as the source text:
- Minor formatting differences are OK (e.g., "1 cup" vs "1 c", "tbsp" vs "tablespoon")
- The document may condense or merge multiple source steps into fewer combined steps — this is OK as long as the key content is preserved
- Minor quantity variations are OK (e.g., "1-2 cups" vs "2 cups")
- Extra clarifications are OK (e.g., "butter, melted" vs "melted butter")

Answer 'Yes' if the document contains substantially the same content as the source.
Answer 'No' if there are significant differences (missing key content, wrong quantities, different items)."""}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Does the document {list_type} match the source?\n\nDocument {list_type}:\n{doc_text}\n\nSource {list_type}:\n{source_text}"}]
        }
    ]

    try:
        response = model(messages)
        is_match = response.strip().lower().startswith('yes')

        if is_match:
            return True, f"LLM confirmed {list_type} match"
        else:
            return False, f"LLM found significant differences in {list_type}"
    except Exception as e:
        print(f"Error in LLM comparison: {e}")
        return False, f"LLM comparison failed: {str(e)[:50]}"


def validate_image_matches_recipe(model, image_path: str, recipe_title: str) -> Tuple[bool, str]:
    """
    Validate that an image shows the correct recipe using VLM.

    This is used when direct image comparison with source is not possible
    (e.g., source blocks image downloads).

    Args:
        model: Pre-loaded VLM model for image analysis.
        image_path: Path to the document image.
        recipe_title: The recipe title to validate against.

    Returns:
        Tuple of (is_valid, details).
    """
    import os
    if not os.path.exists(image_path):
        return False, f"Image not found: {image_path}"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": """You are validating if a recipe photo matches the expected recipe.

Answer 'Yes' if the image shows food that could reasonably be the recipe mentioned.
Answer 'No' if the image clearly shows something different or is not a food photo.

Be lenient - different presentations of the same dish should pass."""}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Does this image show a dish that could be '{recipe_title}'?"}
            ]
        }
    ]

    try:
        response = model(messages)
        is_valid = response.strip().lower().startswith('yes')

        if is_valid:
            return True, f"Image appears to show {recipe_title}"
        else:
            return False, f"Image does not appear to show {recipe_title}"
    except Exception as e:
        print(f"Error validating image: {e}")
        return False, f"Image validation failed: {str(e)[:50]}"


def compare_metadata_relevance(
    doc_metadata: Dict[str, Optional[str]],
    source_metadata: Dict[str, Optional[str]],
    template_defaults: Dict[str, str]
) -> Tuple[bool, str]:
    """
    Compare document metadata against source metadata for relevance.

    The criterion is "relevant to source" not "exact match", so we check:
    1. Values are modified from template defaults
    2. Values are in reasonable range compared to source

    Args:
        doc_metadata: Metadata extracted from document.
        source_metadata: Metadata extracted from source webpage.
        template_defaults: Default template values.

    Returns:
        Tuple of (is_relevant, details).
    """
    changes = []
    issues = []

    # Check Ready In
    doc_ready = doc_metadata.get('ready_in')
    source_ready = source_metadata.get('ready_in')
    default_ready = template_defaults.get('ready_in', '20')

    if doc_ready:
        if doc_ready != default_ready:
            changes.append(f"Ready In: {doc_ready}")
            # Check if in reasonable range of source (within 50%)
            if source_ready:
                try:
                    doc_val = int(doc_ready)
                    source_val = int(source_ready)
                    if abs(doc_val - source_val) > source_val * 0.5:
                        issues.append(f"Ready In differs significantly from source ({doc_val} vs {source_val})")
                except ValueError:
                    pass

    # Check Serves
    doc_serves = doc_metadata.get('serves')
    source_serves = source_metadata.get('serves')
    default_serves = template_defaults.get('serves', '8')

    if doc_serves:
        if doc_serves != default_serves:
            changes.append(f"Serves: {doc_serves}")
            if source_serves:
                try:
                    doc_val = int(doc_serves)
                    source_val = int(source_serves)
                    if abs(doc_val - source_val) > max(2, source_val * 0.5):
                        issues.append(f"Serves differs significantly from source ({doc_val} vs {source_val})")
                except ValueError:
                    pass

    # Check Calories
    doc_cal = doc_metadata.get('calories')
    source_cal = source_metadata.get('calories')
    default_cal = template_defaults.get('calories', '280')

    if doc_cal:
        if doc_cal != default_cal:
            changes.append(f"Calories: {doc_cal}")
            if source_cal:
                try:
                    doc_val = int(doc_cal)
                    source_val = int(source_cal)
                    if abs(doc_val - source_val) > source_val * 0.5:
                        issues.append(f"Calories differs significantly from source ({doc_val} vs {source_val})")
                except ValueError:
                    pass

    # Determine result
    if not changes:
        return False, "Metadata unchanged from template defaults"

    if issues:
        # Has changes but they differ significantly from source
        return False, f"Modified but differs from source: {'; '.join(issues)}"

    return True, f"Metadata relevant to source: {'; '.join(changes)}"
