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
from src.browsergym.knows.eval.eval_utils.text_utils import text_fuzzy_match_contained_long


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

    # Find which pages contain each recipe's title or RECIPE header
    # For each recipe, search for its title text on each page
    recipe_start_pages = []  # List of 0-indexed page numbers
    for recipe in recipes:
        found_page = None
        for page_idx in range(num_pages):
            page_text = doc[page_idx].get_text()
            # Normalize whitespace for comparison (PDF may break titles across lines)
            page_text_normalized = ' '.join(page_text.split())
            # Check for RECIPE header (original detection) or the recipe title
            if recipe.recipe_num == 1 and 'RECIPE' in page_text:
                found_page = page_idx
                break
            elif recipe.title and recipe.title in page_text_normalized:
                found_page = page_idx
                break
        recipe_start_pages.append(found_page)

    doc.close()

    # Assign page ranges: each recipe spans from its start page to
    # the page before the next recipe's start (or end of document)
    for i, recipe in enumerate(recipes):
        start_page = recipe_start_pages[i]
        if start_page is not None:
            # Find the next recipe's start page
            next_start = None
            for j in range(i + 1, len(recipes)):
                if recipe_start_pages[j] is not None:
                    next_start = recipe_start_pages[j]
                    break
            end_page = (next_start - 1) if next_start is not None else (num_pages - 1)
            end_page = max(end_page, start_page)  # At minimum, include the start page
            recipe.pdf_pages = list(range(start_page, end_page + 1))
        else:
            print(f"Warning: No PDF page found for recipe {recipe.recipe_num} ('{recipe.title}')")
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

    # Step 1: Find recipe boundaries by locating section-header clusters.
    # Find all section headers, then group them: each time we see a section
    # header that already appeared in the current group, start a new group.
    # A group with at least 2 of 3 required sections is a recipe.
    all_section_indices = []  # list of (idx, section_name)
    for idx, item in enumerate(doc_structure):
        if item.get('type') != 'text':
            continue
        content = item.get('content', '').strip()
        if len(content) < 50 and content.lower() in REQUIRED_SECTIONS:
            all_section_indices.append((idx, content.lower()))

    # Group into clusters: a duplicate section header starts a new cluster.
    # Also, "ingredients" always starts a new cluster if the current one
    # already has non-ingredients sections (it's the strongest recipe signal).
    clusters = []
    current_sections_found = []
    current_section_indices = {}

    for idx, section in all_section_indices:
        start_new = False
        if section in current_sections_found:
            start_new = True
        elif section == 'ingredients' and current_sections_found and 'ingredients' not in current_sections_found:
            # "ingredients" appearing after prep/tips means a new recipe
            start_new = True

        if start_new:
            if len(current_sections_found) >= 2:
                clusters.append({
                    'first_section_idx': min(current_section_indices.values()),
                    'section_indices': dict(current_section_indices),
                    'sections_found': list(current_sections_found),
                })
            current_sections_found = []
            current_section_indices = {}

        current_sections_found.append(section)
        current_section_indices[section] = idx

    # Don't forget the last cluster
    if len(current_sections_found) >= 2:
        clusters.append({
            'first_section_idx': min(current_section_indices.values()),
            'section_indices': dict(current_section_indices),
            'sections_found': list(current_sections_found),
        })

    if not clusters:
        print("Warning: No valid section clusters found in document")
        return []

    # Step 1b: For each cluster, find its header element.
    # Look for an explicit RECIPE header first, then fall back to the
    # nearest title-like text element before "Ingredients".
    recipe_headers = []  # (structure_index, content) for RECIPE headers
    for idx, item in enumerate(doc_structure):
        if item.get('type') != 'text':
            continue
        content = item.get('content', '').strip()
        if re.match(r'^RECIPE\b', content):
            recipe_headers.append((idx, content))

    candidates = []  # Final list of (header_idx, content) tuples
    used_recipe_headers = set()

    for cluster in clusters:
        first_sec_idx = cluster['first_section_idx']

        # Check if a RECIPE header precedes this cluster
        matched_header = None
        for rh_idx, rh_content in recipe_headers:
            if rh_idx < first_sec_idx and rh_idx not in used_recipe_headers:
                # Make sure no other cluster sits between this RECIPE header and ours
                intervening = any(c['first_section_idx'] > rh_idx and c['first_section_idx'] < first_sec_idx for c in clusters)
                if not intervening:
                    matched_header = (rh_idx, rh_content)
                    used_recipe_headers.add(rh_idx)
                    break

        if matched_header:
            candidates.append(matched_header)
        else:
            # Fallback: walk backwards from first section to find a title element.
            # Stop at the previous cluster's last section index or beginning of doc.
            # Use the current cluster's position in the list to find
            # the previous cluster's last section index as boundary
            cluster_list_idx = clusters.index(cluster)
            if cluster_list_idx > 0:
                prev_cluster = clusters[cluster_list_idx - 1]
                prev_end = max(prev_cluster['section_indices'].values()) + 1
            else:
                prev_end = 0

            # Walk backwards from first section to find the title element.
            # Use a generous range (up to 15 elements) to handle recipes
            # where unlabeled ingredient items sit between title and sections.
            # Stop at the previous candidate's position.
            # For primary clusters (have ingredients), walk backwards a few
            # elements. For secondary clusters (no ingredients, e.g. missing
            # header), walk forward from prev_end to find the title.
            title_idx = None
            if 'ingredients' in cluster.get('sections_found', []):
                # Backward walk (short range) — title is right before ingredients
                for back_idx in range(first_sec_idx - 1, max(first_sec_idx - 5, prev_end - 1, -1), -1):
                    back_item = doc_structure[back_idx]
                    if back_item.get('type') != 'text':
                        continue
                    back_content = back_item.get('content', '').strip()
                    back_lower = back_content.lower()
                    if back_lower.startswith('source:') or back_content.startswith('http'):
                        continue
                    if back_lower in REQUIRED_SECTIONS or back_lower in SECTION_HEADERS:
                        continue
                    if len(back_content) < 2:
                        continue
                    # Skip metadata lines (Ready in, Serves, calories, Makes)
                    if re.match(r'(ready\s*in|serves\s|makes\s|\d+\s*calories)', back_lower):
                        continue
                    title_idx = back_idx
                    break
            else:
                # Forward walk — title is the first text after previous recipe.
                # Skip quoted text (tip content from previous recipe) and
                # zero-width-space items (unlabeled ingredient lists).
                for fwd_idx in range(prev_end, first_sec_idx):
                    fwd_item = doc_structure[fwd_idx]
                    if fwd_item.get('type') != 'text':
                        continue
                    fwd_content = fwd_item.get('content', '').strip()
                    fwd_lower = fwd_content.lower()
                    if fwd_lower.startswith('source:') or fwd_content.startswith('http'):
                        continue
                    if fwd_lower in REQUIRED_SECTIONS or fwd_lower in SECTION_HEADERS:
                        continue
                    if len(fwd_content) < 2:
                        continue
                    if fwd_content.startswith('\u200b'):
                        continue
                    # Skip quoted text — likely tip content from previous recipe
                    if fwd_content.startswith('"') or fwd_content.startswith('\u201c'):
                        continue
                    title_idx = fwd_idx
                    break

            if title_idx is not None:
                candidates.append((title_idx, doc_structure[title_idx].get('content', '').strip()))
            else:
                # Last resort: use the first section index itself
                candidates.append((first_sec_idx, 'Unknown Recipe'))

    # Step 2: Validate each candidate (assign ranges and confirm sections)
    validated = []
    for i, (header_idx, header_content) in enumerate(candidates):
        if i + 1 < len(candidates):
            next_header_idx = candidates[i + 1][0]
        else:
            next_header_idx = len(doc_structure)

        sections_found = []
        section_indices = {}
        for scan_idx in range(header_idx + 1, next_header_idx):
            item = doc_structure[scan_idx]
            if item.get('type') != 'text':
                continue
            content = item.get('content', '').strip().lower()
            if len(item.get('content', '').strip()) < 50:
                for section in REQUIRED_SECTIONS:
                    if content == section and section not in sections_found:
                        sections_found.append(section)
                        section_indices[section] = scan_idx
                        break

        if len(sections_found) >= 2:
            validated.append({
                'header_idx': header_idx,
                'end_idx': next_header_idx,
                'sections_found': sections_found,
                'section_indices': section_indices,
                'candidate_idx': i,
            })
        else:
            missing = set(REQUIRED_SECTIONS) - set(sections_found)
            print(f"Warning: header at structure index {header_idx} "
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
    # Use the header element's content to locate text boundaries in doc_text
    # rather than relying solely on RECIPE text markers.
    recipes = []

    for recipe_idx, v in enumerate(validated):
        recipe_num = recipe_idx + 1

        # Determine text boundaries using the header element's content
        header_content = doc_structure[v['header_idx']].get('content', '').strip()

        if recipe_num == 1:
            # First recipe starts at beginning of document
            text_start = 0
        else:
            # Find the header content in doc_text after the previous recipe's start
            prev_end = recipes[-1].end_index if recipes else 0
            pos = doc_text.find(header_content, prev_end)
            if pos >= 0:
                text_start = pos
            else:
                print(f"Warning: Could not find '{header_content[:30]}' in doc_text, using end of previous recipe")
                text_start = prev_end

        # Find the end: next validated recipe's header position, or end of document
        if recipe_idx + 1 < len(validated):
            next_header_content = doc_structure[validated[recipe_idx + 1]['header_idx']].get('content', '').strip()
            pos = doc_text.find(next_header_content, text_start + 1)
            if pos >= 0:
                text_end = pos
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
            # Skip the RECIPE header element itself from the scoped structure
            if s_idx == structure_start:
                content = item.get('content', '').strip()
                if item.get('type') == 'text' and re.match(r'^RECIPE\b', content):
                    continue
            recipe_structure.append(item)

        # Extract title: for RECIPE-header recipes use existing parser,
        # for section-pattern recipes the header element IS the title
        if re.match(r'^RECIPE\b', header_content):
            title = extract_recipe_title(recipe_text)
        else:
            title = header_content

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
    recipe_num: Optional[int] = None,
    recipe_titles: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Extract hyperlinks from a Google Doc, optionally filtered to a specific recipe.

    Extracts both embedded hyperlinks (textStyle.link.url) and plain text URLs.
    Uses RECIPE headers and/or recipe title text as boundaries to scope
    extraction to a specific recipe.

    Args:
        doc_id: The Google Doc ID.
        service: The Google Docs API service instance.
        recipe_num: If specified (1-based), extract links only from that recipe.
                    If None, extract all links from the entire document.
        recipe_titles: Ordered list of recipe titles (from discover_recipes).
                       Used as boundaries when RECIPE headers are absent.

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

    # Build a set of recipe title strings for boundary detection
    boundary_titles = set()
    if recipe_titles:
        boundary_titles = {t.strip() for t in recipe_titles if t}

    def get_paragraph_text(paragraph) -> str:
        """Get the plain text content of a paragraph."""
        texts = []
        for elem in paragraph.get('elements', []):
            if 'textRun' in elem:
                texts.append(elem['textRun'].get('content', ''))
        return ''.join(texts).strip()

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

    # Pre-scan: find which RECIPE headers actually have section content after them.
    # Discard any RECIPE header that is followed by another RECIPE header (or title
    # boundary) before any section header appears.
    valid_recipe_headers = set()
    section_names_lower = set(REQUIRED_SECTIONS)
    for i, elem in enumerate(content):
        if 'paragraph' not in elem:
            continue
        text = get_paragraph_text(elem['paragraph'])
        if not re.match(r'^RECIPE\b', text):
            continue
        # Scan forward from this RECIPE header for a section header
        has_section = False
        for j in range(i + 1, len(content)):
            if 'paragraph' not in content[j]:
                continue
            fwd_text = get_paragraph_text(content[j]['paragraph'])
            if fwd_text.lower() in section_names_lower:
                has_section = True
                break
            if re.match(r'^RECIPE\b', fwd_text):
                break  # Hit another RECIPE before any section
        if has_section:
            valid_recipe_headers.add(i)

    last_was_recipe_header = False

    def is_recipe_boundary(paragraph, element_index: int) -> bool:
        """Check if a paragraph is a recipe boundary."""
        nonlocal last_was_recipe_header
        text = get_paragraph_text(paragraph)
        if re.match(r'^RECIPE\b', text):
            if element_index not in valid_recipe_headers:
                return False  # Empty RECIPE header, skip
            last_was_recipe_header = True
            return True
        if text in boundary_titles:
            if last_was_recipe_header:
                last_was_recipe_header = False
                return False
            return True
        if text:
            last_was_recipe_header = False
        return False

    def process_table(table) -> Tuple[List[Dict[str, str]], bool]:
        """Process a table element for hyperlinks. Returns (links, boundary_found)."""
        table_links = []
        for row in table.get('tableRows', []):
            for cell in row.get('tableCells', []):
                cell_content = cell.get('content', [])
                for elem in cell_content:
                    if 'paragraph' in elem:
                        if is_recipe_boundary(elem['paragraph']):
                            return table_links, True
                        if collecting:
                            table_links.extend(process_paragraph(elem['paragraph']))
        return table_links, False

    def handle_recipe_boundary() -> bool:
        """Handle encountering a recipe boundary. Returns True if we should stop."""
        nonlocal recipe_count, collecting

        recipe_count += 1

        if recipe_num is None:
            return False

        if recipe_count == recipe_num:
            collecting = True
        elif recipe_count > recipe_num:
            collecting = False
            return True

        return False

    # Process content elements
    for elem_idx, element in enumerate(content):
        if 'paragraph' in element:
            if is_recipe_boundary(element['paragraph'], elem_idx):
                if handle_recipe_boundary():
                    break
                continue
            if collecting:
                links.extend(process_paragraph(element['paragraph']))
        elif 'table' in element:
            table_links, found_boundary = process_table(element['table'])
            if collecting:
                links.extend(table_links)
            if found_boundary:
                if handle_recipe_boundary():
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


def extract_tips_with_sources(doc_structure: List[Dict]) -> List[Dict[str, Optional[str]]]:
    """
    Extract tips paired with their source URLs from the Tips section.

    Walks the recipe structure starting at the "Tips" header. Each text element
    is classified as either a tip or a source URL line. Each tip is paired with
    the nearest source URL that follows it structurally.

    Args:
        doc_structure: Scoped recipe structure (e.g., first_recipe.structure).

    Returns:
        List of dicts with 'tip' (str) and 'url' (str or None) keys.
    """
    url_pattern = re.compile(r'https?://[^\s<>"\'}\])\u200b\u00a0]+', re.IGNORECASE)
    section_headers = ['ingredients', 'preparation', 'tips', 'ready in', 'serves', 'calories']

    # Find the Tips section and collect elements
    in_tips = False
    elements = []  # list of ('tip', text) or ('url', url_str)

    for item in doc_structure:
        if item.get('type') != 'text':
            continue
        content = item.get('content', '').strip()
        content_lower = content.lower()

        if not in_tips:
            if content_lower == 'tips':
                in_tips = True
            continue

        # Stop at the next section header
        if content_lower in section_headers and len(content) < 50:
            break

        # Classify: standalone source/URL line or tip text?
        urls_found = url_pattern.findall(content)
        if urls_found and (content_lower.startswith('source') or content_lower.startswith('http')):
            elements.append(('url', urls_found[0]))
        elif len(content) >= 10:
            # Check for inline URL at the end of the tip text, e.g.
            # '"Tip text here" (https://example.com/page)'
            inline_url = None
            if urls_found:
                inline_url = urls_found[-1]  # last URL in the text
            elements.append(('tip', content, inline_url))

    # Pair each tip with its source URL:
    # 1. Use inline URL if present in the tip text itself
    # 2. Otherwise use the nearest following standalone URL element
    result = []
    for i, elem in enumerate(elements):
        if elem[0] != 'tip':
            continue
        tip_text = elem[1]
        inline_url = elem[2] if len(elem) > 2 else None

        if inline_url:
            source_url = inline_url
        else:
            source_url = None
            for j in range(i + 1, len(elements)):
                if elements[j][0] == 'url':
                    source_url = elements[j][1]
                    break
                elif elements[j][0] == 'tip':
                    break
        result.append({'tip': tip_text, 'url': source_url})

    return result


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
        text = text.replace('&', 'and')
        # Unicode fraction characters → ASCII equivalents
        fraction_map = {
            '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
            '⅕': '1/5', '⅙': '1/6', '⅛': '1/8',
        }
        for uf, af in fraction_map.items():
            text = text.replace(uf, af)
        return text.strip()

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
    fuzzy_threshold: int = 75,
    model=None,
) -> Tuple[bool, List[Dict]]:
    """
    Compare document preparation steps against gold standard.

    Two checks:
    1. Text coverage: all gold text content is present in the doc steps
       (joined as full text blocks, fuzzy matched).
    2. Reasonableness: each doc step is a reasonable cooking/preparation step
       (single LLM call, skipped if no model provided).

    Args:
        doc_steps: List of preparation steps from the document.
        gold_steps: List of expected preparation steps.
        fuzzy_threshold: Minimum fuzzy match score for joined text (0-100).
        model: Optional LLM model callable for reasonableness check.

    Returns:
        Tuple of (all_matched, details) where details is a list of check results.
    """
    details = []

    if not gold_steps:
        return True, details

    if not doc_steps:
        details.append({
            'step': 1,
            'gold': ' '.join(gold_steps)[:50] + '...',
            'found': None,
            'score': 0,
            'matched': False,
            'failure_reason': 'no preparation steps found in document'
        })
        return False, details

    # --- Check 1: Text coverage ---
    gold_joined = ' '.join(gold_steps).lower()
    doc_joined = ' '.join(doc_steps).lower()

    score = max(
        fuzz.token_sort_ratio(doc_joined, gold_joined),
        fuzz.token_set_ratio(doc_joined, gold_joined),
    )
    text_covered = score >= fuzzy_threshold

    detail_coverage = {
        'step': 1,
        'gold': gold_joined[:80] + '...' if len(gold_joined) > 80 else gold_joined,
        'found': doc_joined[:80] + '...' if len(doc_joined) > 80 else doc_joined,
        'score': score,
        'matched': text_covered,
    }
    if not text_covered:
        detail_coverage['failure_reason'] = f"text coverage score {score} < {fuzzy_threshold}"
    details.append(detail_coverage)

    # --- Check 2: Reasonableness (LLM) ---
    if model is not None:
        numbered_steps = '\n'.join(f"{i+1}. {s}" for i, s in enumerate(doc_steps))
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "You are evaluating whether a list of recipe steps are reasonable. "
                    "Reasonable steps include any cooking actions, preparation actions, "
                    "and serving instructions (e.g., 'serve with bread', 'plate and garnish'). "
                    "Only flag steps that are clearly NOT part of a recipe — for example, "
                    "a single word, a meaningless fragment, or text that has nothing to do "
                    "with food preparation or serving."
                )}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": (
                    f"Here are the preparation steps from a recipe document:\n\n"
                    f"{numbered_steps}\n\n"
                    f"Are all of these reasonable recipe steps? "
                    f"If any are clearly not recipe steps (e.g., random words, fragments, "
                    f"non-food-related text), list them by number and explain why. "
                    f"If all are reasonable, respond with exactly: None"
                )}]
            }
        ]
        try:
            response = model(messages).strip()
            reasonable = response.lower().strip().startswith('none')
        except Exception as e:
            print(f"Warning: LLM reasonableness check failed: {e}")
            reasonable = True  # Don't fail on LLM errors

        detail_reasonable = {
            'step': 2,
            'gold': '[Reasonableness check]',
            'found': response[:80] + '...' if len(response) > 80 else response,
            'score': 100 if reasonable else 0,
            'matched': reasonable,
        }
        if not reasonable:
            detail_reasonable['failure_reason'] = f"Unreasonable steps found: {response[:200]}"
        details.append(detail_reasonable)

    all_matched = all(d['matched'] for d in details)
    return all_matched, details


def extract_recipe_metadata(doc_text: str) -> Dict[str, Optional[str]]:
    """
    Extract Ready In, Serves, and Calories values from the metadata area of
    a recipe — the text before the first section header (Ingredients or
    Preparation) and after the last section header (Tips).

    Args:
        doc_text: Text content of a single recipe.

    Returns:
        Dict with 'ready_in', 'serves', 'calories' keys (values may be None).
    """
    result = {
        'ready_in': None,
        'serves': None,
        'calories': None
    }

    # Extract metadata from standalone metadata lines only.
    # These are short, dedicated lines like "Ready in 30 minutes",
    # "Serves 4 people", "180 calories". This avoids matching numbers
    # embedded in tips or other content.
    metadata_text = ''
    for line in doc_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if re.match(r'ready\s*in\s', lower):
            metadata_text += '\n' + stripped
        elif re.match(r'serves\s', lower):
            metadata_text += '\n' + stripped
        elif re.match(r'\d+\s*calories?\b', lower):
            metadata_text += '\n' + stripped

    # Patterns for each field
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

    hour_min_match = ready_hour_min.search(metadata_text)
    if hour_min_match:
        hours = int(hour_min_match.group(1))
        minutes = int(hour_min_match.group(2))
        result['ready_in'] = str(hours * 60 + minutes)
    else:
        min_match = ready_min_only.search(metadata_text)
        if min_match:
            result['ready_in'] = min_match.group(1)

    serves_match = serves_pattern.search(metadata_text)
    if serves_match:
        result['serves'] = serves_match.group(1)

    cal_match = calories_pattern.search(metadata_text)
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


def verify_tip_is_quote(tip_text: str, webpage_content: str, threshold: int = 85) -> Tuple[bool, int]:
    """
    Verify if a tip appears as a direct quote in webpage content.

    Uses text_fuzzy_match_contained_long from shared utils for robust
    sliding-window fuzzy containment matching.

    Args:
        tip_text: The tip text to verify.
        webpage_content: The fetched webpage text content.
        threshold: Minimum fuzzy match score (0-100).

    Returns:
        Tuple of (is_quote, match_score).
    """
    if not tip_text or not webpage_content:
        return False, 0

    tip_normalized = tip_text.lower().strip()
    webpage_lower = webpage_content.lower()

    # Exact substring match first (fast path)
    if tip_normalized in webpage_lower:
        return True, 100

    # Use shared sliding-window fuzzy containment
    match, score = text_fuzzy_match_contained_long(tip_normalized, webpage_lower, threshold=threshold)
    return match is not None, int(score)


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


def check_title_theme(
    title: str,
    theme_keywords: Optional[List[str]] = None,
    theme_description: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Check if a recipe title relates to expected themes.

    Args:
        title: The recipe title to check.
        theme_keywords: List of keywords to match against. If None, uses default
                        fall/soup/pumpkin keywords for backward compatibility.
        theme_description: Human-readable description of the theme for failure messages.

    Returns:
        Tuple of (is_thematic, details).
    """
    if not title:
        return False, "No title found"

    if theme_keywords is None:
        theme_keywords = [
            'pumpkin', 'squash', 'soup', 'fall', 'autumn',
        ]
    if theme_description is None:
        theme_description = "fall, soups, pumpkins, or thanksgiving"

    title_lower = title.lower()

    found_keywords = [kw for kw in theme_keywords if kw in title_lower]

    if found_keywords:
        return True, f"Found thematic keywords: {', '.join(found_keywords)}"

    return False, f"No {theme_description} keywords found in '{title}'"


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
Answer 'No: <reason>' if there are significant differences (missing key content, wrong quantities, different items). Always include a brief reason after 'No'."""}]
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
            return False, response.strip()[:200]
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
