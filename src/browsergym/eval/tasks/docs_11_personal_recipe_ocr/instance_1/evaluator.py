"""
Evaluator for docs_11_personal_recipe_ocr task.

This evaluator validates that an agent correctly:
1. OCR'd a pumpkin soup recipe image
2. Populated a Google Docs 2-column template
3. Added proper tips with source citations
4. Found and added similar recipes
"""

import os
import sys
import time
import glob
from typing import List, Optional

# Get the base path that works in both Docker and local environments
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.image_utils import (
    binary_compare_images,
    match_image_tiered,
)
from src.browsergym.eval.eval_utils.text_utils import keyword_exact_match
from src.browsergym.eval.eval_utils.web_utils import (
    validate_url_accessible,
    fetch_page_text_content,
)
from src.browsergym.eval.eval_utils.models import load_model

# Import task-specific utilities
from src.browsergym.eval.tasks.docs_11_personal_recipe_ocr.utils import (
    extract_hyperlinks_from_doc,
    extract_section_content,
    extract_list_items,
    compare_ingredient_lists,
    compare_preparation_steps,
    extract_recipe_metadata,
    check_metadata_modified,
    verify_tip_is_quote,
    # Page 1 isolation functions (BUG-001, BUG-005)
    get_first_recipe_text,
    get_first_recipe_structure,
    extract_section_content_first_recipe,
    extract_hyperlinks_first_recipe,
    # Document setup and cleanup utilities
    cleanup_generated_files,
    setup_document,
    load_gold_data,
    # Checkpoint 2 utilities (additional recipes)
    get_additional_recipes,
    extract_recipe_title,
    extract_hyperlinks_for_recipe,
    extract_section_content_for_recipe,
    check_title_theme,
    check_content_modified_from_default,
    # Checkpoint 3 utilities (content validation)
    extract_ingredients_from_webpage,
    extract_preparation_from_webpage,
    extract_metadata_from_webpage,
    compare_lists_with_llm,
    validate_image_matches_recipe,
    compare_metadata_relevance,
)

# Task directories
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_11_personal_recipe_ocr/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
GOLDS_DIR = os.path.join(DATA_DIR, "golds/")
DOC_IMAGES_DIR = os.path.join(DATA_DIR, "images/")
DOC_IMAGES_CROPPED_DIR = os.path.join(DATA_DIR, "cropped_images/")
PDF_IMAGES_DIR = os.path.join(DATA_DIR, "pdf_images/")

# Configuration
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_DPI = 150

# Model configuration
model = None
model_id = "gemini-2.5-flash"

# Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

# Global variables set by setup_document
doc_id = None
doc_text = None
doc_structure = None
text_ocr = None

# Template defaults
TEMPLATE_DEFAULTS = {
    'title': 'Strawberry Vanilla Pancakes',
    'ready_in': '20',
    'serves': '8',
    'calories': '280',
}

# Gold data
GOLD_TITLE = "Pumpkin Soup"
GOLD_INGREDIENTS_FILE = os.path.join(GOLDS_DIR, "gold_ingredients.txt")
GOLD_PREPSTEPS_FILE = os.path.join(GOLDS_DIR, "gold_prepsteps.txt")
GOLD_IMAGE_ORIGINAL = os.path.join(GOLDS_DIR, "original_image.png")
GOLD_IMAGE_CROPPED = os.path.join(GOLDS_DIR, "original_image_cropped.png")




def grade_checkpoint_1():
    """
    Grade Checkpoint 1: Original Recipe Page (9 pts)

    Criteria:
    1.1 Title changed to "Pumpkin Soup"
    1.2 Ingredients match picture ingredient list
    1.3 Preparation steps match picture preparation steps
    1.4 Tips section has relevant tips
    1.5 Tips section has valid source URLs
    1.6 Tips are direct quotes from source URL
    1.7 Image is from original image
    1.8 Image is properly cropped
    1.9 Info under photo modified from defaults

    Bug fixes applied:
    - BUG-001: Title check now restricted to page 1 only
    - BUG-002: Ingredient check now bidirectional (detects extra ingredients)
    - BUG-003: Ingredient matching uses 1:1 mapping, no substring matching
    - BUG-004: Prep steps validate numbers and cooking verbs exactly
    - BUG-005: Tips URLs check restricted to page 1 only
    - BUG-006: Tips relevance uses stricter LLM prompt for pumpkin-specific tips
    """
    global model

    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=9, result=0, name="Original Recipe Page")

    # Load gold data (using utility from utils.py)
    gold_ingredients, gold_prepsteps = load_gold_data(GOLDS_DIR)

    # Get first recipe only content (BUG-001, BUG-005 fix)
    first_recipe_text = get_first_recipe_text(doc_text)
    first_recipe_structure = get_first_recipe_structure(doc_structure)

    # =========================================================================
    # Step 1.1: Title Check (BUG-001 FIX: now checks ONLY page 1)
    # =========================================================================
    step_start = time.time()
    # Use first recipe text only, not the full document
    title_found = keyword_exact_match(first_recipe_text, GOLD_TITLE, case_sensitive=False, substring=True)
    step_time = time.time() - step_start

    if title_found:
        checkpoint.add_step("Title Match", True, 1, f"Found '{GOLD_TITLE}' in first recipe (page 1)", execution_time=step_time)
    else:
        checkpoint.add_step("Title Match", False, 1, f"Title '{GOLD_TITLE}' not found in first recipe (page 1)", execution_time=step_time)

    # =========================================================================
    # Step 1.2: Ingredients Match (BUG-002/003 FIX: bidirectional, 1:1, no substring)
    # =========================================================================
    step_start = time.time()
    # Use first recipe structure only
    ingredients_text = extract_section_content_first_recipe(doc_structure, "Ingredients")
    doc_ingredients = extract_list_items(ingredients_text)
    # compare_ingredient_lists now uses strict 1:1 matching with bidirectional check
    ingredients_match, ingredients_details = compare_ingredient_lists(doc_ingredients, gold_ingredients)
    step_time = time.time() - step_start

    if ingredients_match:
        checkpoint.add_step("Ingredients Match", True, 2, f"All {len(gold_ingredients)} ingredients matched (1:1 mapping)", execution_time=step_time)
    else:
        # Build detailed failure message
        missing = [d['gold'] for d in ingredients_details if not d['matched'] and d['gold'] != '[Extra ingredients check]']
        extra_check = next((d for d in ingredients_details if d['gold'] == '[Extra ingredients check]'), None)

        failure_parts = []
        if missing:
            failure_parts.append(f"Missing/unmatched: {', '.join(missing)}")
        if extra_check and not extra_check['matched']:
            failure_parts.append(f"Extra ingredients found: {extra_check.get('found', [])}")

        checkpoint.add_step("Ingredients Match", False, 2, '; '.join(failure_parts) if failure_parts else "Ingredient matching failed", execution_time=step_time)

    # =========================================================================
    # Step 1.3: Preparation Steps Match (BUG-004 FIX: validates numbers and verbs exactly)
    # =========================================================================
    step_start = time.time()
    # Use first recipe structure only
    prep_text = extract_section_content_first_recipe(doc_structure, "Preparation")
    doc_steps = extract_list_items(prep_text)
    # compare_preparation_steps now validates numbers and cooking verbs exactly
    steps_match, steps_details = compare_preparation_steps(doc_steps, gold_prepsteps)
    step_time = time.time() - step_start

    if steps_match:
        checkpoint.add_step("Preparation Steps Match", True, 3, f"All {len(gold_prepsteps)} steps matched (text, numbers, verbs)", execution_time=step_time)
    else:
        # Build detailed failure message with reasons
        unmatched = []
        for d in steps_details:
            if not d['matched']:
                reason = d.get('failure_reason', f"score: {d['score']}")
                unmatched.append(f"Step {d['step']} ({reason})")
        checkpoint.add_step("Preparation Steps Match", False, 3, f"Unmatched: {'; '.join(unmatched)}", execution_time=step_time)

    # =========================================================================
    # Step 1.4: Tips Relevance Check (BUG-006 FIX: stricter prompt for pumpkin-specific)
    # =========================================================================
    step_start = time.time()
    # Use first recipe structure only
    tips_text = extract_section_content_first_recipe(doc_structure, "Tips")
    tips_list_raw = extract_list_items(tips_text)

    # Filter out URL lines and "Source:" prefixed lines
    tips_list = []
    for tip in tips_list_raw:
        tip_lower = tip.lower().strip()
        # Skip URL lines, source labels, and empty/short tips
        if tip_lower.startswith('http') or tip_lower.startswith('source:'):
            continue
        if len(tip) < 10:
            continue
        tips_list.append(tip)

    if model is None:
        model = load_model(model_id)

    tips_relevant = True
    tips_relevance_details = []

    for tip in tips_list:
        if not tip:
            continue

        # BUG-006 FIX: Stricter prompt that requires SPECIFIC relevance to pumpkin soup
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": """You are evaluating if a cooking tip is SPECIFICALLY about making pumpkin soup, not just general cooking advice.

A tip is SPECIFIC if it mentions ANY of these:
- Pumpkin, squash, or gourd
- Soup-specific techniques (blending, pureeing soup, soup consistency)
- Specific ingredients in pumpkin soup (cream, butter for this dish, onions for this recipe)
- Soup serving suggestions specific to this type of dish

A tip is GENERIC (not specific) if it applies to ANY cooking:
- "Always taste and adjust seasoning" - generic, applies to everything
- "Use fresh ingredients" - generic
- "Don't overcook" - generic unless specifically about soup
- "Clean as you go" - generic
- "Read the recipe first" - generic

Answer ONLY 'Yes' if the tip is SPECIFICALLY about pumpkin soup.
Answer 'No' if it's generic cooking advice that applies to any recipe."""}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Is this tip SPECIFICALLY about pumpkin soup (not just general cooking advice)?\n\nTip: {tip}"}]
            }
        ]

        response = model(messages)
        is_relevant = response.strip().lower().startswith('yes')
        tips_relevance_details.append({'tip': tip[:50], 'relevant': is_relevant})

        if not is_relevant:
            tips_relevant = False

    step_time = time.time() - step_start

    if tips_relevant and len(tips_relevance_details) > 0:
        checkpoint.add_step("Tips Relevance", True, 4, f"All {len(tips_relevance_details)} tips are specific to pumpkin soup", execution_time=step_time)
    elif len(tips_relevance_details) == 0:
        checkpoint.add_step("Tips Relevance", False, 4, "No tips found in Tips section", execution_time=step_time)
    else:
        irrelevant = [t['tip'][:30] for t in tips_relevance_details if not t['relevant']]
        checkpoint.add_step("Tips Relevance", False, 4, f"Generic tips found (not pumpkin-specific): {irrelevant[:2]}", execution_time=step_time)

    # =========================================================================
    # Step 1.5: Tips URLs Valid (BUG-005 FIX: now checks ONLY page 1 URLs)
    # =========================================================================
    step_start = time.time()
    # Use first recipe hyperlinks only (BUG-005 fix)
    first_recipe_links = extract_hyperlinks_first_recipe(doc_id, DOCS_SERVICE)
    tips_urls = [link['url'] for link in first_recipe_links if link['url'].startswith('http')]

    urls_valid = True
    valid_urls = []
    invalid_urls = []

    for url in tips_urls:
        is_valid, details = validate_url_accessible(url)
        if is_valid:
            valid_urls.append(url)
        else:
            invalid_urls.append(url)
            urls_valid = False

    step_time = time.time() - step_start

    if urls_valid and len(tips_urls) > 0:
        checkpoint.add_step("Tips URLs Valid", True, 5, f"All {len(tips_urls)} URLs in first recipe are accessible", execution_time=step_time)
    elif len(tips_urls) == 0:
        checkpoint.add_step("Tips URLs Valid", False, 5, "No URLs found in first recipe (page 1)", execution_time=step_time)
    else:
        checkpoint.add_step("Tips URLs Valid", False, 5, f"{len(invalid_urls)} URLs not accessible: {invalid_urls[:2]}", execution_time=step_time)

    # =========================================================================
    # Step 1.6: Tips are Direct Quotes
    # =========================================================================
    step_start = time.time()
    tips_are_quotes = True
    quote_details = []

    # For each tip, check if it appears in any of the valid URLs
    for tip in tips_list:
        if not tip or len(tip) < 10:
            continue

        tip_found_in_source = False

        for url in valid_urls:
            webpage_content, status = fetch_page_text_content(url)
            if webpage_content:
                is_quote, score = verify_tip_is_quote(tip, webpage_content)
                if is_quote:
                    tip_found_in_source = True
                    quote_details.append({'tip': tip[:30], 'source': url, 'score': score})
                    break

        if not tip_found_in_source:
            tips_are_quotes = False
            quote_details.append({'tip': tip[:30], 'source': None, 'score': 0})

    step_time = time.time() - step_start

    if tips_are_quotes and len(quote_details) > 0:
        checkpoint.add_step("Tips Are Direct Quotes", True, 6, f"All tips verified as quotes from sources", execution_time=step_time)
    elif len(tips_list) == 0:
        checkpoint.add_step("Tips Are Direct Quotes", False, 6, "No tips found to verify", execution_time=step_time)
    else:
        not_found = [d['tip'] for d in quote_details if d['source'] is None]
        checkpoint.add_step("Tips Are Direct Quotes", False, 6, f"Tips not found in sources: {not_found[:2]}", execution_time=step_time)

    # =========================================================================
    # Step 1.7: Image from Original
    # =========================================================================
    step_start = time.time()

    # Get the first image from the document
    doc_images = glob.glob(os.path.join(DOC_IMAGES_CROPPED_DIR, "*"))
    image_match = False
    image_match_details = "No images found in document"

    if doc_images and os.path.exists(GOLD_IMAGE_ORIGINAL):
        # Try to match any doc image against the gold original
        for doc_img in doc_images:
            match_result, match_method = match_image_tiered(doc_img, GOLD_IMAGE_ORIGINAL, model)
            if match_result:
                image_match = True
                image_match_details = f"Image matched via {match_method}"
                break

        if not image_match:
            image_match_details = "Document image does not match original recipe image"

    step_time = time.time() - step_start
    checkpoint.add_step("Image from Original", image_match, 7, image_match_details, execution_time=step_time)

    # =========================================================================
    # Step 1.8: Image Properly Cropped
    # =========================================================================
    step_start = time.time()
    image_cropped = False
    crop_details = "No images to check"

    if doc_images and os.path.exists(GOLD_IMAGE_CROPPED):
        for doc_img in doc_images:
            # Use VLM to compare if images are similar (both showing cropped recipe)
            is_similar = binary_compare_images(model, doc_img, GOLD_IMAGE_CROPPED, mode="similar")
            if is_similar:
                image_cropped = True
                crop_details = "Document image is properly cropped similar to gold"
                break

        if not image_cropped:
            crop_details = "Document image not properly cropped to show only recipe info"

    step_time = time.time() - step_start
    checkpoint.add_step("Image Properly Cropped", image_cropped, 8, crop_details, execution_time=step_time)

    # =========================================================================
    # Step 1.9: Info Modified from Defaults
    # =========================================================================
    step_start = time.time()
    # Use first recipe text only for metadata extraction
    metadata = extract_recipe_metadata(first_recipe_text)
    is_modified, modification_details = check_metadata_modified(metadata, TEMPLATE_DEFAULTS)
    step_time = time.time() - step_start

    checkpoint.add_step("Info Modified from Defaults", is_modified, 9, modification_details, execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Grade Checkpoint 2: Additional Recipe Pages Formatting (24 pts)

    Evaluates 8 criteria for each of 3 additional recipes (pages 2, 3, 4).

    Criteria per recipe:
    2.1 Title theme: Related to fall, soups, pumpkins, or thanksgiving
    2.2 Source URL: Valid source URL below title
    2.3 Format: Matches first page (2 column layout)
    2.4 Ingredients: Modified from default
    2.5 Preparation: Modified from default
    2.6 Tips: Modified from default
    2.7 Info under photo: Modified from default
    2.8 Visual distinction: Different color or font from other pages
    """
    global model

    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=24, result=0, name="Additional Recipe Pages Formatting")

    # Get additional recipes (pages 2, 3, 4)
    additional_recipes = get_additional_recipes(doc_text, doc_structure)

    if model is None:
        model = load_model(model_id)

    # If fewer than 3 additional recipes found, mark missing ones as failed
    if len(additional_recipes) < 3:
        print(f"Warning: Only found {len(additional_recipes)} additional recipes (expected 3)")
        for i in range(len(additional_recipes), 3):
            recipe_num = i + 2  # Pages 2, 3, 4
            step_names = [
                "Title Theme", "Source URL", "Format Match", "Ingredients Modified",
                "Preparation Modified", "Tips Modified", "Info Modified", "Visual Distinction"
            ]
            for step_idx, step_name in enumerate(step_names):
                step_id = i * 8 + step_idx + 1
                checkpoint.add_step(
                    f"Recipe {recipe_num} - {step_name}",
                    False, step_id, f"Recipe {recipe_num} not found in document",
                    execution_time=0
                )

    # Get PDF page images for visual distinction check
    pdf_images = sorted(glob.glob(os.path.join(PDF_IMAGES_DIR, "*.png")))

    # Evaluate each additional recipe (up to 3)
    for recipe_idx, recipe in enumerate(additional_recipes[:3]):
        recipe_num = recipe['recipe_num']  # 2, 3, or 4
        recipe_text = recipe['text']
        recipe_structure = recipe['structure']

        # Base step ID for this recipe (0-indexed: recipe_idx * 8)
        base_step_id = recipe_idx * 8

        # =====================================================================
        # Step 2.1: Title Theme Check
        # =====================================================================
        step_start = time.time()
        title = extract_recipe_title(recipe_text)
        has_theme, theme_details = check_title_theme(title)

        # If keyword check fails, use LLM as fallback
        if not has_theme and title:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": """You are checking if a recipe title is related to fall/autumn themes.

A title is THEMATIC if it relates to:
- Fall/autumn foods: pumpkin, squash, butternut, acorn, apple, cranberry
- Warm comfort foods: soup, stew, chowder, bisque
- Thanksgiving: turkey, stuffing, pie
- Seasonal themes: harvest, autumn, fall, cozy, warm

Answer 'Yes' if the title is related to any of these themes.
Answer 'No' if the title is completely unrelated (e.g., "Summer Salad", "Grilled Burgers")."""}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Is this recipe title related to fall, soups, pumpkins, or thanksgiving themes?\n\nTitle: {title}"}]
                }
            ]
            response = model(messages)
            has_theme = response.strip().lower().startswith('yes')
            if has_theme:
                theme_details = f"LLM confirmed thematic: '{title}'"
            else:
                theme_details = f"LLM rejected: '{title}' not related to fall/soup/thanksgiving themes"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Title Theme",
            has_theme, base_step_id + 1, theme_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.2: Source URL Valid
        # =====================================================================
        step_start = time.time()
        recipe_links = extract_hyperlinks_for_recipe(doc_id, DOCS_SERVICE, recipe_num)
        source_urls = [link['url'] for link in recipe_links if link['url'].startswith('http')]

        has_valid_url = False
        url_details = "No URLs found in recipe"

        for url in source_urls:
            is_valid, details = validate_url_accessible(url)
            if is_valid:
                has_valid_url = True
                url_details = f"Valid source URL: {url[:50]}..."
                break

        if not has_valid_url and source_urls:
            url_details = f"Found {len(source_urls)} URLs but none accessible"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Source URL",
            has_valid_url, base_step_id + 2, url_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.3: Format Matches First Page
        # =====================================================================
        step_start = time.time()
        has_format = False
        format_details = "Unable to verify format"

        # Check if recipe has expected section headers
        expected_sections = ['ingredients', 'preparation', 'tips']
        found_sections = []

        for item in recipe_structure:
            if item.get('type') == 'text':
                content = item.get('content', '').lower().strip()
                for section in expected_sections:
                    if section in content and len(content) < 50:
                        found_sections.append(section)

        # Check for structure indicating 2-column layout (table elements)
        has_table = any(item.get('type') == 'table' for item in recipe_structure)

        # Use VLM to verify 2-column layout if we have PDF images
        if recipe_num <= len(pdf_images):
            page_image = pdf_images[recipe_num - 1]  # 0-indexed
            vlm_prompt = "Does this recipe page show a 2-column layout with the title and image on the left side, and ingredients/preparation text on the right side? Answer Yes or No."
            has_format = binary_compare_images(model, page_image, pdf_images[0], mode=vlm_prompt) if pdf_images else False

            if has_format:
                format_details = f"VLM confirmed 2-column layout; found sections: {found_sections}"
            else:
                # Fallback: check if we at least have the expected sections
                if len(found_sections) >= 2:
                    has_format = True
                    format_details = f"Found expected sections: {found_sections}"
                else:
                    format_details = f"Format check failed; found sections: {found_sections}"
        else:
            # Fallback without VLM
            if len(found_sections) >= 2:
                has_format = True
                format_details = f"Found expected sections: {found_sections}"
            else:
                format_details = f"Missing sections; found only: {found_sections}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Format Match",
            has_format, base_step_id + 3, format_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.4: Ingredients Modified
        # =====================================================================
        step_start = time.time()
        ingredients_content = extract_section_content_for_recipe(recipe_structure, "Ingredients")
        ing_modified, ing_details = check_content_modified_from_default(ingredients_content, "ingredients")
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Ingredients Modified",
            ing_modified, base_step_id + 4, ing_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.5: Preparation Modified
        # =====================================================================
        step_start = time.time()
        prep_content = extract_section_content_for_recipe(recipe_structure, "Preparation")
        prep_modified, prep_details = check_content_modified_from_default(prep_content, "preparation")
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Preparation Modified",
            prep_modified, base_step_id + 5, prep_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.6: Tips Modified
        # =====================================================================
        step_start = time.time()
        tips_content = extract_section_content_for_recipe(recipe_structure, "Tips")
        tips_modified, tips_details = check_content_modified_from_default(tips_content, "tips")
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Tips Modified",
            tips_modified, base_step_id + 6, tips_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.7: Info Under Photo Modified
        # =====================================================================
        step_start = time.time()
        metadata = extract_recipe_metadata(recipe_text)
        info_modified, info_details = check_metadata_modified(metadata, TEMPLATE_DEFAULTS)
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Info Modified",
            info_modified, base_step_id + 7, info_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 2.8: Visually Distinct
        # =====================================================================
        step_start = time.time()
        is_distinct = False
        visual_details = "Unable to verify visual distinction"

        # Compare this page's appearance with page 1 using VLM
        if recipe_num <= len(pdf_images) and len(pdf_images) > 0:
            page_image = pdf_images[recipe_num - 1]  # Current page (0-indexed)
            reference_image = pdf_images[0]  # Page 1 for comparison

            # VLM prompt for visual distinction
            vlm_prompt = ("Compare these two recipe pages. Do they have different visual styling "
                          "(different colors, fonts, or formatting themes)? "
                          "The second page should have a distinctly different color scheme or font style. "
                          "Answer Yes if they are visually distinct, No if they look the same.")

            try:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are comparing two recipe pages to check if they have distinct visual styles. Answer Yes if they have different colors, fonts, or styling. Answer No if they look the same."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Page 1 (reference):"},
                            {"type": "image", "image": reference_image},
                            {"type": "text", "text": f"Page {recipe_num}:"},
                            {"type": "image", "image": page_image},
                            {"type": "text", "text": vlm_prompt}
                        ]
                    }
                ]

                response = model(messages)
                is_distinct = response.strip().lower().startswith('yes')

                if is_distinct:
                    visual_details = f"Recipe {recipe_num} has distinct visual styling from page 1"
                else:
                    visual_details = f"Recipe {recipe_num} appears similar to page 1 (no distinct colors/fonts)"
            except Exception as e:
                visual_details = f"VLM comparison failed: {str(e)[:50]}"
        else:
            visual_details = f"PDF page {recipe_num} not available for visual comparison"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Visual Distinction",
            is_distinct, base_step_id + 8, visual_details,
            execution_time=step_time
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Grade Checkpoint 3: Additional Recipe Pages Content (15 pts)

    Validates content of 3 additional recipes (pages 2, 3, 4) against their source URLs.
    Requires valid, accessible source URLs from Checkpoint 2.

    Criteria per recipe (5 per recipe × 3 recipes = 15 total):
    3.1 Photo from source URL
    3.2 Ingredients exactly match source
    3.3 Preparation steps exactly match source
    3.4 Tips are direct quotes from source
    3.5 Info under photo relevant to source
    """
    global model

    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=15, result=0, name="Additional Recipe Pages Content")

    # Get additional recipes (pages 2, 3, 4)
    additional_recipes = get_additional_recipes(doc_text, doc_structure)

    if model is None:
        model = load_model(model_id)

    # If fewer than 3 additional recipes found, mark missing ones as failed
    if len(additional_recipes) < 3:
        print(f"Warning: Only found {len(additional_recipes)} additional recipes (expected 3)")
        for i in range(len(additional_recipes), 3):
            recipe_num = i + 2  # Pages 2, 3, 4
            step_names = [
                "Photo from Source", "Ingredients Match", "Preparation Match",
                "Tips Direct Quotes", "Info Relevant"
            ]
            for step_idx, step_name in enumerate(step_names):
                step_id = i * 5 + step_idx + 1
                checkpoint.add_step(
                    f"Recipe {recipe_num} - {step_name}",
                    False, step_id, f"Recipe {recipe_num} not found in document",
                    execution_time=0
                )

    # Cache for webpage content to avoid redundant fetches
    webpage_cache = {}

    # Evaluate each additional recipe (up to 3)
    for recipe_idx, recipe in enumerate(additional_recipes[:3]):
        recipe_num = recipe['recipe_num']  # 2, 3, or 4
        recipe_text = recipe['text']
        recipe_structure = recipe['structure']

        # Base step ID for this recipe (0-indexed: recipe_idx * 5)
        base_step_id = recipe_idx * 5

        # Get source URL for this recipe
        recipe_links = extract_hyperlinks_for_recipe(doc_id, DOCS_SERVICE, recipe_num)
        source_urls = [link['url'] for link in recipe_links if link['url'].startswith('http')]

        # Find first valid, accessible source URL
        source_url = None
        webpage_content = None

        for url in source_urls:
            is_valid, url_details = validate_url_accessible(url)
            if is_valid:
                # Fetch webpage content
                if url in webpage_cache:
                    webpage_content = webpage_cache[url]
                else:
                    content, status = fetch_page_text_content(url)
                    if content:
                        webpage_cache[url] = content
                        webpage_content = content
                        source_url = url
                        break

        # If no accessible source URL, mark all content steps as failed
        if not source_url or not webpage_content:
            url_failure_reason = f"No accessible source URL found (tried {len(source_urls)} URLs)"
            for step_idx, step_name in enumerate(["Photo from Source", "Ingredients Match",
                                                   "Preparation Match", "Tips Direct Quotes", "Info Relevant"]):
                checkpoint.add_step(
                    f"Recipe {recipe_num} - {step_name}",
                    False, base_step_id + step_idx + 1, url_failure_reason,
                    execution_time=0
                )
            continue

        # Get recipe title for validation
        recipe_title = extract_recipe_title(recipe_text)
        print(f"Recipe {recipe_num}: '{recipe_title}' - Source: {source_url[:50]}...")

        # =====================================================================
        # Step 3.1: Photo from Source URL
        # =====================================================================
        step_start = time.time()
        photo_valid = False
        photo_details = "No image found for recipe"

        # Get PDF page image for this recipe
        pdf_images = sorted(glob.glob(os.path.join(PDF_IMAGES_DIR, "*.png")))
        if recipe_num <= len(pdf_images):
            page_image = pdf_images[recipe_num - 1]

            # Since most recipe sites block image downloads, use VLM to validate
            # that the image shows the correct recipe type
            photo_valid, photo_details = validate_image_matches_recipe(model, page_image, recipe_title)

            if photo_valid:
                photo_details = f"Photo appears to show {recipe_title} (validated by VLM)"
        else:
            photo_details = f"PDF page {recipe_num} not available for image validation"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Photo from Source",
            photo_valid, base_step_id + 1, photo_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 3.2: Ingredients Match Source
        # =====================================================================
        step_start = time.time()
        ingredients_match = False
        ingredients_details = "Unable to verify ingredients"

        # Extract ingredients from document
        doc_ingredients_text = extract_section_content_for_recipe(recipe_structure, "Ingredients")
        doc_ingredients = extract_list_items(doc_ingredients_text)

        if not doc_ingredients:
            ingredients_details = "No ingredients found in document recipe"
        else:
            # Extract ingredients from source webpage
            source_ingredients = extract_ingredients_from_webpage(model, webpage_content, recipe_title)

            if not source_ingredients:
                # Source has no extractable ingredients - validate doc has reasonable content
                if len(doc_ingredients) >= 3:
                    ingredients_match = True
                    ingredients_details = f"Source ingredients not extractable; doc has {len(doc_ingredients)} ingredients"
                else:
                    ingredients_details = f"Source ingredients not extractable and doc only has {len(doc_ingredients)} items"
            else:
                # Compare ingredient lists - first try strict matching
                strict_match, strict_details = compare_ingredient_lists(doc_ingredients, source_ingredients, fuzzy_threshold=90)

                if strict_match:
                    ingredients_match = True
                    ingredients_details = f"Ingredients match source ({len(doc_ingredients)} items, strict match)"
                else:
                    # Fallback to LLM comparison
                    llm_match, llm_details = compare_lists_with_llm(model, doc_ingredients, source_ingredients, "ingredients")
                    if llm_match:
                        ingredients_match = True
                        ingredients_details = f"Ingredients match source (LLM verified, {len(doc_ingredients)} items)"
                    else:
                        # Detailed failure message
                        missing = [d['gold'] for d in strict_details if not d['matched'] and d['gold'] != '[Extra ingredients check]']
                        ingredients_details = f"Ingredients differ from source: {', '.join(missing[:3])}..."

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Ingredients Match",
            ingredients_match, base_step_id + 2, ingredients_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 3.3: Preparation Steps Match Source
        # =====================================================================
        step_start = time.time()
        prep_match = False
        prep_details = "Unable to verify preparation steps"

        # Extract preparation steps from document
        doc_prep_text = extract_section_content_for_recipe(recipe_structure, "Preparation")
        doc_steps = extract_list_items(doc_prep_text)

        if not doc_steps:
            prep_details = "No preparation steps found in document recipe"
        else:
            # Extract preparation steps from source webpage
            source_steps = extract_preparation_from_webpage(model, webpage_content, recipe_title)

            if not source_steps:
                # Source has no extractable steps - validate doc has reasonable content
                if len(doc_steps) >= 2:
                    prep_match = True
                    prep_details = f"Source steps not extractable; doc has {len(doc_steps)} steps"
                else:
                    prep_details = f"Source steps not extractable and doc only has {len(doc_steps)} steps"
            else:
                # Compare preparation steps - first try strict matching
                # Use lower threshold since steps can have more variation in phrasing
                strict_match, strict_details = compare_preparation_steps(
                    doc_steps, source_steps, fuzzy_threshold=80,
                    require_exact_numbers=False, require_exact_verbs=False
                )

                if strict_match:
                    prep_match = True
                    prep_details = f"Preparation steps match source ({len(doc_steps)} steps, strict match)"
                else:
                    # Fallback to LLM comparison
                    llm_match, llm_details = compare_lists_with_llm(model, doc_steps, source_steps, "preparation steps")
                    if llm_match:
                        prep_match = True
                        prep_details = f"Preparation steps match source (LLM verified, {len(doc_steps)} steps)"
                    else:
                        # Detailed failure message
                        unmatched = [f"Step {d['step']}" for d in strict_details if not d['matched']][:3]
                        prep_details = f"Preparation differs from source: {', '.join(unmatched)}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Preparation Match",
            prep_match, base_step_id + 3, prep_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 3.4: Tips are Direct Quotes from Source
        # =====================================================================
        step_start = time.time()
        tips_valid = False
        tips_details = "Unable to verify tips"

        # Extract tips from document
        doc_tips_text = extract_section_content_for_recipe(recipe_structure, "Tips")
        doc_tips_raw = extract_list_items(doc_tips_text)

        # Filter out URL lines and "Source:" prefixed lines
        doc_tips = []
        for tip in doc_tips_raw:
            tip_lower = tip.lower().strip()
            if tip_lower.startswith('http') or tip_lower.startswith('source:'):
                continue
            if len(tip) < 10:
                continue
            doc_tips.append(tip)

        if not doc_tips:
            tips_details = "No tips found in document recipe"
        else:
            # Check if each tip appears in the source webpage
            tips_verified = []
            tips_not_found = []

            for tip in doc_tips:
                is_quote, score = verify_tip_is_quote(tip, webpage_content)
                if is_quote:
                    tips_verified.append(tip[:30])
                else:
                    tips_not_found.append(tip[:30])

            if not tips_not_found:
                tips_valid = True
                tips_details = f"All {len(doc_tips)} tips verified as quotes from source"
            elif len(tips_verified) > len(tips_not_found):
                # Majority of tips are valid - use LLM for final verification of unmatched
                unverified_text = '\n'.join(tips_not_found)
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are checking if cooking tips appear in a webpage. Answer 'Yes' if the tips could be paraphrased from the webpage content, 'No' otherwise."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Do these tips appear (possibly paraphrased) in this webpage?\n\nTips:\n{unverified_text}\n\nWebpage (excerpt):\n{webpage_content[:3000]}"}]
                    }
                ]
                response = model(messages)
                if response.strip().lower().startswith('yes'):
                    tips_valid = True
                    tips_details = f"All {len(doc_tips)} tips verified (some by LLM paraphrase check)"
                else:
                    tips_details = f"Tips not found in source: {tips_not_found[:2]}"
            else:
                tips_details = f"Tips not found in source: {tips_not_found[:2]}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Tips Direct Quotes",
            tips_valid, base_step_id + 4, tips_details,
            execution_time=step_time
        )

        # =====================================================================
        # Step 3.5: Info Under Photo Relevant to Source
        # =====================================================================
        step_start = time.time()
        info_relevant = False
        info_details = "Unable to verify metadata relevance"

        # Extract metadata from document
        doc_metadata = extract_recipe_metadata(recipe_text)

        # First check if modified from defaults
        is_modified, mod_details = check_metadata_modified(doc_metadata, TEMPLATE_DEFAULTS)

        if not is_modified:
            info_details = f"Metadata unchanged from template defaults"
        else:
            # Extract metadata from source webpage for comparison
            source_metadata = extract_metadata_from_webpage(model, webpage_content)

            # Compare relevance
            info_relevant, info_details = compare_metadata_relevance(
                doc_metadata, source_metadata, TEMPLATE_DEFAULTS
            )

            # If source has no extractable metadata, just check that doc is modified
            if not any(source_metadata.values()) and is_modified:
                info_relevant = True
                info_details = f"Metadata modified from defaults (source metadata not extractable): {mod_details}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Info Relevant",
            info_relevant, base_step_id + 5, info_details,
            execution_time=step_time
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id: str, cached_models: Optional[dict] = None, browsing_history: Optional[list] = None):
    """
    Grade all checkpoints for the document.

    Args:
        workspace_doc_id: The Google Docs document ID to evaluate.
        cached_models: Dictionary of preloaded models by model_id.
        browsing_history: List of URLs visited by the agent.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    global doc_id, doc_text, doc_structure, text_ocr

    total_start_time = time.time()

    try:
        # Setup document processing (using utility from utils.py)
        doc_data = setup_document(
            workspace_doc_id,
            DATA_DIR,
            DRIVE_SERVICE,
            DOCS_SERVICE,
            PDF_DPI
        )

        # Set global variables from returned data
        doc_id = doc_data['doc_id']
        doc_text = doc_data['doc_text']
        doc_structure = doc_data['doc_structure']
        text_ocr = doc_data['text_ocr']

        # Use cached model if available
        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        checkpoints: List[Checkpoint] = []

        # Grade checkpoint 1
        checkpoints.append(grade_checkpoint_1())

        # Grade checkpoint 2
        checkpoints.append(grade_checkpoint_2())

        # Grade checkpoint 3
        checkpoints.append(grade_checkpoint_3())

        # TODO: Add checkpoint 4

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    finally:
        try:
            cleanup_generated_files(DATA_DIR, CLEANUP_ENABLED)
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed with error: {cleanup_error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate docs_11 recipe OCR task")
    parser.add_argument("--workspace_doc_id", type=str, required=True, help="Google Docs document ID to evaluate")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
    print(f"CLEANUP enabled: {CLEANUP_ENABLED}")

    result = grade_checkpoints(workspace_doc_id=args.workspace_doc_id)

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")

    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "✓" if step["success"] else "✗"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
