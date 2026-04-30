"""Utility functions for education lesson plan evaluation."""

import os
from typing import List, Dict
from src.browsergym.knows.eval.eval_utils.parallel_utils import fast_parallel_vlm_calls  # type: ignore


def validate_topics_chemistry_related(topics: List[str], model_instance) -> Dict:
    """
    Validate that all topics are related to Chemistry.

    Uses two-tier validation:
    1. Exact match against known chemistry topics
    2. LLM-based validation for novel topics

    Args:
        topics: List of topic strings to validate.
        model_instance: LLM model for validation.

    Returns:
        {
            'all_valid': bool,
            'topics': List[str],
            'validations': Dict[str, bool],  # topic -> is_valid
            'reasons': Dict[str, str]  # topic -> reason
        }
    """
    # Known chemistry topics (exact match - expanded list for middle school)
    KNOWN_CHEMISTRY_TOPICS = [
        'chemical reactions', 'periodic table', 'acids and bases',
        'atomic structure', 'bonding', 'states of matter',
        'organic chemistry', 'electrochemistry', 'thermochemistry',
        'stoichiometry', 'kinetics', 'equilibrium', 'polymers',
        'crystallography', 'spectroscopy', 'biochemistry',
        'explosions', 'color changes', 'ph', 'elements',
        'molecules', 'compounds', 'mixtures', 'solutions',
        'chemical properties', 'physical properties', 'matter',
        'atoms', 'electrons', 'protons', 'neutrons',
        'chemical change', 'physical change', 'reactions',
        'acids', 'bases', 'salts', 'oxidation', 'reduction',
        'metals', 'nonmetals', 'noble gases', 'halogens',
        'alkali metals', 'transition metals', 'catalysts',
        'enzymes', 'proteins', 'dna', 'rna', 'carbohydrates',
        'lipids', 'amino acids', 'nucleic acids',
        'carbon', 'hydrogen', 'oxygen', 'nitrogen',
        'chemistry', 'chemical'
    ]

    validations = {}
    reasons = {}
    llm_candidates = []

    # Pass 1: Exact match; collect LLM candidates
    for topic in topics:
        topic_lower = topic.lower().strip()

        # Exact match (substring check against known topics)
        if any(known in topic_lower for known in KNOWN_CHEMISTRY_TOPICS):
            validations[topic] = True
            reasons[topic] = "Matched known chemistry topic"
            continue

        # Needs LLM validation
        llm_candidates.append(topic)

    # Pass 2: Batch all LLM calls in parallel
    if llm_candidates:
        vlm_tasks = [
            {
                'id': topic,
                'messages': [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a chemistry education expert. Answer only 'Yes' or 'No'."}]
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": f"Is '{topic}' a topic related to Chemistry that would be appropriate for a middle school lesson plan? Answer only 'Yes' or 'No'."
                        }]
                    }
                ]
            }
            for topic in llm_candidates
        ]
        llm_results = fast_parallel_vlm_calls(vlm_tasks, model_instance, max_workers=5)
        for topic, passed in llm_results.items():
            validations[topic] = passed
            reasons[topic] = "LLM validated as chemistry-related" if passed else "LLM rejected as non-chemistry topic"

    return {
        'all_valid': all(validations.values()),
        'topics': topics,
        'validations': validations,
        'reasons': reasons
    }


def check_images_aligned_horizontally(image_paths: List[str], pdf_images_dir: str,
                                      last_page_only: bool = True,
                                      y_tolerance: float = 0.15,
                                      doc_content: dict = None,
                                      dpi: int = 150) -> Dict:
    """Check if images are aligned horizontally (side by side) using feature-based matching.

    Finds each image's location in the PDF page images using SIFT feature matching
    with API-provided image sizes, then checks if they share approximately the same
    Y position (same row) and are spread across the X axis.

    Args:
        image_paths: List of paths to the extracted document images.
        pdf_images_dir: Path to the folder containing PDF page images.
        last_page_only: If True, only search the last page to avoid false positives.
        y_tolerance: Maximum relative difference in Y position (as fraction of page height)
            to consider images on the same row. Default 0.15 (15% of page height).
        doc_content: Pre-fetched Google Docs document JSON. Used to get image sizes
            for feature-based matching. Falls back to template matching if None.
        dpi: DPI of the PDF page images. Default 150.

    Returns:
        {
            'aligned': bool,
            'locations_found': int,
            'total_images': int,
            'locations': list,
            'details': str
        }
    """
    import shutil
    import tempfile
    import glob as glob_module
    from src.browsergym.knows.eval.eval_utils.image_utils import (
        extract_image_location,
        extract_image_location_size_feature_based
    )
    from src.browsergym.knows.eval.eval_utils.utils import image_id_from_path

    # Build obj_id → size lookup from doc_content
    size_lookup = {}
    if doc_content:
        for obj_id, obj_data in doc_content.get('inlineObjects', {}).items():
            embedded = obj_data.get('inlineObjectProperties', {}).get('embeddedObject', {})
            size = embedded.get('size')
            if size:
                size_lookup[obj_id] = size
        for obj_id, obj_data in doc_content.get('positionedObjects', {}).items():
            embedded = obj_data.get('positionedObjectProperties', {}).get('embeddedObject', {})
            size = embedded.get('size')
            if size:
                size_lookup[obj_id] = size

    # If last_page_only, create a temp dir with just the last page PNG
    search_dir = pdf_images_dir
    temp_dir = None
    if last_page_only:
        page_files = sorted(glob_module.glob(os.path.join(pdf_images_dir, "*.png")))
        if page_files:
            temp_dir = tempfile.mkdtemp(prefix="last_page_")
            shutil.copy2(page_files[-1], os.path.join(temp_dir, os.path.basename(page_files[-1])))
            search_dir = temp_dir

    try:
        locations = []
        for img_path in image_paths:
            try:
                loc = None
                # Try feature-based matching with known size (more robust)
                obj_id = image_id_from_path(img_path)
                image_size = size_lookup.get(obj_id)
                if image_size:
                    loc = extract_image_location_size_feature_based(
                        img_path, image_size, search_dir, dpi=dpi
                    )
                # Fall back to template matching
                if loc is None:
                    loc = extract_image_location(img_path, search_dir)
                if loc is not None:
                    locations.append(loc)
            except Exception as e:
                print(f"  -> Error locating image {os.path.basename(img_path)}: {e}")

        if len(locations) < 2:
            return {
                'aligned': False,
                'locations_found': len(locations),
                'total_images': len(image_paths),
                'locations': locations,
                'details': f"Only found {len(locations)}/{len(image_paths)} image locations on the last page"
            }

        # Check Y positions are similar (same row)
        # Page height scales with DPI: 3300px at 300 DPI, 1650px at 150 DPI
        page_height = int(3300 * dpi / 300)
        y_positions = [loc.y for loc in locations]
        y_range = max(y_positions) - min(y_positions)

        if y_range > page_height * y_tolerance:
            return {
                'aligned': False,
                'locations_found': len(locations),
                'total_images': len(image_paths),
                'locations': locations,
                'details': f"Images not on same row: Y positions vary by {y_range:.0f}px (tolerance: {page_height * y_tolerance:.0f}px)"
            }

        # Check X positions are spread out (not stacked)
        x_positions = sorted(loc.x for loc in locations)
        min_spread = 100
        x_spread = x_positions[-1] - x_positions[0]

        if x_spread < min_spread:
            return {
                'aligned': False,
                'locations_found': len(locations),
                'total_images': len(image_paths),
                'locations': locations,
                'details': f"Images not spread horizontally: X spread is only {x_spread:.0f}px"
            }

        return {
            'aligned': True,
            'locations_found': len(locations),
            'total_images': len(image_paths),
            'locations': locations,
            'details': f"{len(locations)} images aligned horizontally (Y range: {y_range:.0f}px, X spread: {x_spread:.0f}px)"
        }
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def extract_summary_facts_with_colors(doc_content: dict) -> List[Dict]:
    """Extract facts and their colors from the summary section of the document.

    Parses paragraphs after the "Lesson Plan Summary" / "Fact From Lesson" delimiter.
    For each paragraph, concatenates all text runs to get the full fact text and
    extracts the dominant color (checks both foreground and background).

    Args:
        doc_content: Full Google Docs API document JSON.

    Returns:
        List of dicts: [{'text': str, 'color': (r, g, b)}]
        Empty list if no summary section found.
    """
    from collections import Counter

    summary_facts = []
    in_summary = False

    WHITE = (1.0, 1.0, 1.0)
    BLACK = (0.0, 0.0, 0.0)

    for element in doc_content.get('body', {}).get('content', []):
        if 'paragraph' not in element:
            continue

        para = element['paragraph']
        full_text = ''
        colors_found = []

        for elem in para.get('elements', []):
            if 'textRun' not in elem:
                continue
            text_run = elem['textRun']
            run_text = text_run.get('content', '')
            full_text += run_text

            text_style = text_run.get('textStyle', {})

            # Check background color (highlight) first — highlights are intentional color coding
            bg = text_style.get('backgroundColor', {}).get('color', {}).get('rgbColor', {})
            if bg:
                bg_color = (bg.get('red', 0.0), bg.get('green', 0.0), bg.get('blue', 0.0))
                if bg_color != WHITE and bg_color != BLACK:
                    colors_found.append(bg_color)
                    continue

            # Fallback: check foreground color (text color)
            fg = text_style.get('foregroundColor', {}).get('color', {}).get('rgbColor', {})
            if fg:
                fg_color = (fg.get('red', 0.0), fg.get('green', 0.0), fg.get('blue', 0.0))
                if not all(c <= 0.25 for c in fg_color):  # Treat near-black as default
                    colors_found.append(fg_color)

        full_text = full_text.strip()
        if not full_text:
            continue

        # Detect summary section start (flexible keyword matching)
        is_bullet = 'bullet' in para
        text_lower = full_text.lower()
        is_summary_heading = not is_bullet and (
            ('fact' in text_lower and 'lesson' in text_lower) or
            ('summary' in text_lower and 'lesson' in text_lower)
        )
        if is_summary_heading:
            in_summary = True
            continue

        if not in_summary:
            continue

        # Skip very short text (headers, whitespace)
        if len(full_text) < 5:
            continue

        # Use the most common color from text runs (None if no color metadata)
        dominant_color = Counter(colors_found).most_common(1)[0][0] if colors_found else None
        summary_facts.append({'text': full_text, 'color': dominant_color})

    return summary_facts


def extract_bullet_hierarchy_from_doc(document: dict) -> dict:
    """Parse Google Docs JSON using semantic text rules for the lesson plan."""
    from src.browsergym.knows.eval.eval_utils.google_services_utils import extract_hyperlinks_from_doc

    # Build text→URL lookup from all links (hyperlinks + plain text URLs)
    all_links = extract_hyperlinks_from_doc(doc_id=None, service=None, document=document)
    text_to_url = {link['text']: link['url'] for link in all_links}

    items = []
    current_topic = None
    current_website = None
    in_summary_section = False

    try:
        content = document.get('body', {}).get('content', [])

        for element in content:
            if 'paragraph' in element:
                para = element['paragraph']

                # 1. Extract Text and URL
                text_content = ''
                url = None
                for elem in para.get('elements', []):
                    if 'textRun' in elem:
                        text_run = elem['textRun']
                        text_content += text_run.get('content', '')

                        style = text_run.get('textStyle', {})
                        if 'link' in style:
                            url = style['link'].get('url')

                # Fallback: check text→URL lookup for plain text URLs
                if not url:
                    stripped = text_content.strip()
                    if stripped in text_to_url:
                        url = text_to_url[stripped]

                text_content = text_content.strip()
                if not text_content:
                    continue

                # 2. Stop parsing if we hit the summary/facts section
                text_lower = text_content.lower()
                is_bullet = 'bullet' in para
                is_summary_heading = not is_bullet and (
                    ('fact' in text_lower and 'lesson' in text_lower) or
                    ('summary' in text_lower and 'lesson' in text_lower)
                )
                if is_summary_heading:
                    in_summary_section = True
                    continue

                if in_summary_section:
                    continue # Skip everything after the summary header
                
                # LEVEL 0: TOPICS
                if text_lower.startswith('topic'):
                    current_topic = {
                        'text': text_content, 
                        'nesting_level': 0, 
                        'children': []
                    }
                    if url: current_topic['url'] = url
                    items.append(current_topic)
                    current_website = None # Reset website tracker for the new topic
                
                # LEVEL 1: WEBSITES
                elif text_lower.startswith('website'):
                    current_website = {
                        'text': text_content, 
                        'nesting_level': 1, 
                        'children': []
                    }
                    if url: current_website['url'] = url
                    
                    if current_topic:
                        current_topic['children'].append(current_website)
                    else:
                        items.append(current_website) # Fallback if agent missed a topic
                
                # LEVEL 2: FACTS (Must be a bullet point)
                elif 'bullet' in para:
                    fact_item = {
                        'text': text_content, 
                        'nesting_level': 2, 
                        'children': []
                    }
                    if url: fact_item['url'] = url
                    
                    if current_website:
                        current_website['children'].append(fact_item)
                    elif current_topic:
                        current_topic['children'].append(fact_item)

    except Exception as e:
        print(f"Error parsing bullet hierarchy: {e}")

    return {'topics': items}
