"""
Evaluator for docs_11_personal_recipe_ocr task.

This evaluator validates that an agent correctly:
1. OCR'd a Larry's Favorite Cookies recipe image
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
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, StepCategory
from src.browsergym.knows.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    get_image_dimensions_from_doc,
)
from src.browsergym.knows.eval.eval_utils.image_utils import (
    match_image_tiered,
    binary_compare_images,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keyword_exact_match
from src.browsergym.knows.eval.eval_utils.web_utils import (
    validate_url_accessible,
    fetch_page_text_content,
    download_image_from_url,
    normalize_url_for_comparison,
)
from src.browsergym.knows.eval.eval_utils.models import load_model

# Import task-specific utilities
from src.browsergym.knows.eval.tasks.docs_11_personal_recipe_ocr.utils import (
    # Phase 1: Recipe discovery
    Recipe,
    discover_recipes,
    map_recipes_to_pdf_pages,
    # Phase 2: Generic extraction functions
    extract_section_content,
    extract_hyperlinks,
    extract_list_items,
    extract_tips_with_sources,
    # Comparison and validation
    compare_ingredient_lists,
    compare_preparation_steps,
    extract_recipe_metadata,
    check_metadata_modified,
    verify_tip_is_quote,
    check_title_theme,
    check_content_modified_from_default,
    # Document setup and cleanup utilities
    cleanup_generated_files,
    setup_document,
    load_gold_data,
    # Checkpoint 3 utilities (content validation)
    extract_recipe_image_url,
    extract_ingredients_from_webpage,
    extract_preparation_from_webpage,
    extract_metadata_from_webpage,
    compare_lists_with_llm,
    validate_image_matches_recipe,
    compare_metadata_relevance,
)

# Task directories
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/knows/eval/tasks/docs_11_personal_recipe_ocr/instance_3/")
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
model_id = "gemini-2.5-flash-google-ai"

# Google services — initialized lazily in grade_checkpoints to avoid import-time crashes
DRIVE_SERVICE = None
DOCS_SERVICE = None

# Global variables set by setup_document
doc_id = None
doc_text = None
doc_structure = None
recipes = []  # List[Recipe] populated by discover_recipes()

# Template defaults
TEMPLATE_DEFAULTS = {
    'title': 'Strawberry Vanilla Pancakes',
    'ready_in': '20',
    'serves': '8',
    'calories': '280',
}

# Gold data
GOLD_TITLE = "Larry's Favorite Cookies"
GOLD_INGREDIENTS_FILE = os.path.join(GOLDS_DIR, "gold_ingredients.txt")
GOLD_PREPSTEPS_FILE = os.path.join(GOLDS_DIR, "gold_prepsteps.txt")
GOLD_IMAGE_ORIGINAL = os.path.join(GOLDS_DIR, "original_image.jpeg")
GOLD_IMAGE_CROPPED = os.path.join(GOLDS_DIR, "original_image_cropped.jpeg")
GOLD_TEMPLATE_IMAGE = os.path.join(GOLDS_DIR, "template_image.jpg")
GOLD_TEMPLATE_PAGE1 = os.path.join(GOLDS_DIR, "template_page1.png")

# Template image slot dimensions in PT (from the blank Coral Recipe template)
TEMPLATE_IMAGE_WIDTH_PT = 214.0
TEMPLATE_IMAGE_HEIGHT_PT = 286.9


def grade_checkpoint_1():
    """
    Grade Checkpoint 1: Original Recipe Page (9 pts)

    Criteria:
    1.1 Title changed to "Larry's Favorite Cookies"
    1.2 Ingredients match picture ingredient list
    1.3 Preparation steps match picture preparation steps
    1.4 Tips section has relevant tips
    1.5 Tips section has valid source URLs
    1.6 Tips are direct quotes from source URL
    1.7 Image is from original image
    1.8 Image is properly cropped to fit template
    1.9 Image shows recipe text clearly
    1.10 Info under photo modified from defaults
    """
    global model

    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Original Recipe Page")

    # Load gold data — ingredients are required; prep steps may be empty for this instance
    gold_ingredients, gold_prepsteps = load_gold_data(GOLDS_DIR)
    if not gold_ingredients:
        raise FileNotFoundError(f"Gold ingredients file missing or empty in {GOLDS_DIR}")

    # Get first recipe from discovered recipes
    first_recipe = recipes[0] if recipes else None
    if not first_recipe:
        step_names = ["Title Match", "Ingredients Match", "Preparation Steps Match",
                      "Tips Relevance", "Tips URLs Valid", "Tips Are Direct Quotes",
                      "Image from Original", "Image Properly Cropped",
                      "Image Shows Recipe Text", "Info Modified from Defaults"]
        for step_id, name in enumerate(step_names, 1):
            checkpoint.add_step(name, False, step_id, "No recipes found in document", execution_time=0, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # =========================================================================
    # Step 1.1: Title Check — extract title position, not full text search
    # =========================================================================
    step_start = time.time()
    title = first_recipe.title
    title_found = bool(title) and keyword_exact_match(title, GOLD_TITLE, case_sensitive=False, substring=False)
    step_time = time.time() - step_start

    if title_found:
        checkpoint.add_step("Title Match", True, 1, f"Title '{title}' matches '{GOLD_TITLE}'", execution_time=step_time, category=StepCategory.DETERMINISTIC)
    else:
        checkpoint.add_step("Title Match", False, 1, f"Title '{title}' does not match '{GOLD_TITLE}'", execution_time=step_time, category=StepCategory.DETERMINISTIC)

    # =========================================================================
    # Step 1.2: Ingredients Match (BUG-002/003 FIX: bidirectional, 1:1, no substring)
    # =========================================================================
    step_start = time.time()
    # Use first recipe structure only
    ingredients_text = extract_section_content(first_recipe.structure, "Ingredients")
    doc_ingredients = extract_list_items(ingredients_text)

    if not doc_ingredients:
        step_time = time.time() - step_start
        checkpoint.add_step("Ingredients Match", False, 2, "No ingredients found in first recipe Ingredients section", execution_time=step_time, category=StepCategory.EXECUTION_ERROR)
    else:
        # Strict 1:1 fuzzy matching first
        ingredients_match, ingredients_details = compare_ingredient_lists(doc_ingredients, gold_ingredients)
        ingredients_category = StepCategory.FUZZY_MATCH

        # LLM fallback if fuzzy matching fails
        if not ingredients_match:
            if model is None:
                model = load_model(model_id)
            ingredients_match, llm_explanation = compare_lists_with_llm(model, doc_ingredients, gold_ingredients, "ingredients")
            ingredients_category = StepCategory.LLM_VLM_JUDGEMENT

        step_time = time.time() - step_start

        if ingredients_match:
            checkpoint.add_step("Ingredients Match", True, 2, f"All {len(gold_ingredients)} ingredients matched", execution_time=step_time, category=ingredients_category)
        else:
            missing = [d['gold'] for d in ingredients_details if not d['matched'] and d['gold'] != '[Extra ingredients check]']
            extra_check = next((d for d in ingredients_details if d['gold'] == '[Extra ingredients check]'), None)

            failure_parts = []
            if missing:
                failure_parts.append(f"Missing/unmatched: {', '.join(missing)}")
            if extra_check and not extra_check['matched']:
                failure_parts.append(f"Extra ingredients found: {extra_check.get('found', [])}")

            checkpoint.add_step("Ingredients Match", False, 2, '; '.join(failure_parts) if failure_parts else "Ingredient matching failed", execution_time=step_time, category=ingredients_category)

    # =========================================================================
    # Step 1.3: Preparation Steps Match
    # This instance has NO prep steps intentionally — pass if both are empty,
    # or if gold is empty but agent found steps on their own.
    # =========================================================================
    step_start = time.time()
    # Use first recipe structure only
    prep_text = extract_section_content(first_recipe.structure, "Preparation")
    doc_steps = extract_list_items(prep_text)

    if not gold_prepsteps and not doc_steps:
        step_time = time.time() - step_start
        checkpoint.add_step("Preparation Steps Match", True, 3, "No preparation steps expected for this recipe", execution_time=step_time, category=StepCategory.VACUOUS_PASS)
    elif not gold_prepsteps and doc_steps:
        step_time = time.time() - step_start
        checkpoint.add_step("Preparation Steps Match", True, 3, f"No gold preparation steps for this recipe; agent found {len(doc_steps)} steps on their own", execution_time=step_time, category=StepCategory.VACUOUS_PASS)
    elif not doc_steps:
        step_time = time.time() - step_start
        checkpoint.add_step("Preparation Steps Match", False, 3, "No preparation steps found in first recipe Preparation section", execution_time=step_time, category=StepCategory.EXECUTION_ERROR)
    else:
        # compare_preparation_steps now validates numbers and cooking verbs exactly
        if model is None:
            model = load_model(model_id)
        steps_match, steps_details = compare_preparation_steps(doc_steps, gold_prepsteps, model=model)
        step_time = time.time() - step_start

        if steps_match:
            checkpoint.add_step("Preparation Steps Match", True, 3, f"All {len(gold_prepsteps)} steps matched (text, numbers, verbs)", execution_time=step_time, category=StepCategory.LLM_VLM_JUDGEMENT)
        else:
            # Build detailed failure message with reasons
            unmatched = []
            for d in steps_details:
                if not d['matched']:
                    reason = d.get('failure_reason', f"score: {d['score']}")
                    unmatched.append(f"Step {d['step']} ({reason})")
            checkpoint.add_step("Preparation Steps Match", False, 3, f"Unmatched: {'; '.join(unmatched)}", execution_time=step_time, category=StepCategory.LLM_VLM_JUDGEMENT)

    # =========================================================================
    # Step 1.4: Tips Relevance Check
    # =========================================================================
    step_start = time.time()

    # Extract tips paired with their source URLs from the structure
    tip_pairs = extract_tips_with_sources(first_recipe.structure)
    tips_list = [p['tip'] for p in tip_pairs]

    if model is None:
        model = load_model(model_id)

    tips_relevant = True
    tips_relevance_details = []

    full_tips_context = "\n".join(tips_list)
    source_urls = [p['url'] for p in tip_pairs if p['url']]
    source_url_context = ""
    if source_urls:
        source_url_context = f"\n\nThese tips come from: {', '.join(source_urls)}"

    for tip in tips_list:
        if not tip:
            continue

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": """You are evaluating if a cooking tip is relevant to making Larry's Favorite Cookies. Consider the tip in the context of the full Tips section.

A tip is RELEVANT if ANY of these apply:
- It mentions cookies, baking, dough, or cookie-specific techniques
- It discusses cookie-specific techniques (chilling dough, cookie sheets, baking times, cookie texture)
- It talks about complementing cookie flavors or sweetness
- It mentions adding ingredients to enhance cookies or desserts
- It comes from a source URL about Larry's Favorite Cookies (context provided)
- Even if the tip title seems generic, if the explanation mentions cookie texture, baking characteristics, or making cookies better, it is relevant

A tip is NOT RELEVANT only if it's truly generic advice with no connection to cookies or baking:
- "Always taste and adjust seasoning" - generic, applies to everything
- "Clean as you go" - generic kitchen advice
- "Read the recipe first" - generic

Answer 'Yes' if the tip is relevant to Larry's Favorite Cookies (directly or contextually).
Answer 'No' only if it's completely generic advice unrelated to cookies or baking."""}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Is this tip relevant to Larry's Favorite Cookies?\n\nTip being evaluated: {tip}\n\nFull Tips section context:\n{full_tips_context}{source_url_context}"}]
            }
        ]

        is_relevant = False
        for attempt in range(2):
            try:
                response = model(messages)
                is_relevant = response.strip().lower().startswith('yes')
                break
            except Exception as e:
                if attempt == 0:
                    print(f"Warning: LLM tip relevance check failed, retrying: {e}")
                else:
                    print(f"Warning: LLM tip relevance check failed after retry: {e}")
        tips_relevance_details.append({'tip': tip[:50], 'relevant': is_relevant})

        if not is_relevant:
            tips_relevant = False

    step_time = time.time() - step_start

    if tips_relevant and len(tips_relevance_details) > 0:
        checkpoint.add_step("Tips Relevance", True, 4, f"All {len(tips_relevance_details)} tips are relevant to Larry's Favorite Cookies", execution_time=step_time, category=StepCategory.LLM_VLM_JUDGEMENT)
    elif len(tips_relevance_details) == 0:
        checkpoint.add_step("Tips Relevance", False, 4, "No tips found in Tips section", execution_time=step_time, category=StepCategory.EXECUTION_ERROR)
    else:
        irrelevant = [t['tip'][:30] for t in tips_relevance_details if not t['relevant']]
        checkpoint.add_step("Tips Relevance", False, 4, f"Generic tips found (not cookie-relevant): {irrelevant[:2]}", execution_time=step_time, category=StepCategory.LLM_VLM_JUDGEMENT)

    # =========================================================================
    # Step 1.5: Tips URLs Valid — each tip should have a source URL
    # =========================================================================
    step_start = time.time()

    tips_urls = [p['url'] for p in tip_pairs if p['url']]
    valid_urls = []
    invalid_urls = []

    if not tips_list:
        step_time = time.time() - step_start
        checkpoint.add_step("Tips URLs Valid", False, 5,
                            "No tips found in Tips section",
                            execution_time=step_time, category=StepCategory.EXECUTION_ERROR)
    elif not tips_urls:
        step_time = time.time() - step_start
        checkpoint.add_step("Tips URLs Valid", False, 5,
                            "No source URLs found paired with tips",
                            execution_time=step_time, category=StepCategory.DETERMINISTIC)
    else:
        urls_valid = True
        for url in tips_urls:
            is_valid, _ = validate_url_accessible(url)
            if is_valid:
                valid_urls.append(url)
            else:
                invalid_urls.append(url)
                urls_valid = False

        step_time = time.time() - step_start

        if urls_valid:
            checkpoint.add_step("Tips URLs Valid", True, 5,
                                f"All {len(tips_urls)} tip source URLs are accessible",
                                execution_time=step_time, category=StepCategory.DETERMINISTIC)
        else:
            checkpoint.add_step("Tips URLs Valid", False, 5,
                                f"{len(invalid_urls)} tip URLs not accessible: {invalid_urls[:2]}",
                                execution_time=step_time, category=StepCategory.DETERMINISTIC)

    # =========================================================================
    # Step 1.6: Tips are Direct Quotes
    # =========================================================================
    step_start = time.time()

    if not tips_list:
        step_time = time.time() - step_start
        checkpoint.add_step("Tips Are Direct Quotes", False, 6, "No tips found to verify", execution_time=step_time, category=StepCategory.EXECUTION_ERROR)
    elif not valid_urls:
        step_time = time.time() - step_start
        checkpoint.add_step("Tips Are Direct Quotes", False, 6,
                            "No valid source URLs available to verify quotes against (step 1.5 prerequisite)",
                            execution_time=step_time, category=StepCategory.DEPENDENCY_NOT_EVALUATED)
    else:
        tips_are_quotes = True
        quote_details = []
        quote_items = []  # (category, success) per tip for StepCategory.aggregate

        # Pre-fetch webpage content for valid URLs
        url_content_cache = {}
        for url in valid_urls:
            if url not in url_content_cache:
                try:
                    webpage_content, _ = fetch_page_text_content(url)
                    if webpage_content:
                        url_content_cache[url] = webpage_content
                except Exception as e:
                    print(f"Warning: Failed to fetch {url}: {e}")

        if not url_content_cache:
            # No URLs had scrapable content — check if that's because
            # they all failed to scrape (use LLM fallback) or truly empty
            url_content_cache = {}

        for pair in tip_pairs:
            tip = pair['tip']
            tip_url = pair['url']

            tip_found_in_source = False
            tip_category = StepCategory.FUZZY_MATCH  # last mechanism that ran for this tip

            # First check the tip's own paired source URL via fuzzy text match
            if tip_url and tip_url in url_content_cache:
                is_quote, quote_score = verify_tip_is_quote(tip, url_content_cache[tip_url])
                if is_quote:
                    tip_found_in_source = True
                    tip_category = StepCategory.DETERMINISTIC if quote_score == 100 else StepCategory.FUZZY_MATCH
                    quote_details.append({'tip': tip[:30], 'source': tip_url})

            # Fallback: check all scrapable URLs
            if not tip_found_in_source:
                for url, webpage_content in url_content_cache.items():
                    is_quote, quote_score = verify_tip_is_quote(tip, webpage_content)
                    if is_quote:
                        tip_found_in_source = True
                        tip_category = StepCategory.DETERMINISTIC if quote_score == 100 else StepCategory.FUZZY_MATCH
                        quote_details.append({'tip': tip[:30], 'source': url})
                        break

            # LLM fallback: if text matching failed and the source URL
            # wasn't successfully scraped (content too short/missing),
            # ask the LLM whether the scraped text looks like a real
            # page or a failed scrape, and if the tip is plausible.
            if not tip_found_in_source and tip_url:
                scraped = url_content_cache.get(tip_url, '')
                messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": (
                                "You are verifying whether a cooking tip could plausibly "
                                "be a direct quote from a given source URL. The webpage "
                                "could not be fully scraped, so you cannot check the text "
                                "directly. Instead, judge based on the URL and the tip content."
                            )}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": (
                                f"Tip: {tip}\n\n"
                                f"Source URL: {tip_url}\n\n"
                                f"Scraped content (may be incomplete): {scraped[:300]}\n\n"
                                f"Could this tip plausibly be a direct quote from this URL? "
                                f"Consider whether the URL is a cooking/recipe page and whether "
                                f"the tip content is the kind of advice that page would contain. "
                                f"Answer 'Yes' or 'No'."
                            )}]
                        }
                ]
                try:
                    response = model(messages).strip()
                    tip_category = StepCategory.LLM_VLM_JUDGEMENT
                    if response.lower().startswith('yes'):
                        tip_found_in_source = True
                        quote_details.append({'tip': tip[:30], 'source': tip_url, 'method': 'llm_fallback'})
                except Exception as e:
                    print(f"Warning: LLM quote fallback failed: {e}")

            quote_items.append((tip_category, tip_found_in_source))
            if not tip_found_in_source:
                tips_are_quotes = False
                quote_details.append({'tip': tip[:30], 'source': None, 'score': 0})

        step_time = time.time() - step_start

        quotes_category = StepCategory.aggregate(quote_items)
        if tips_are_quotes and len(quote_details) > 0:
            checkpoint.add_step("Tips Are Direct Quotes", True, 6, "All tips verified as quotes from sources", execution_time=step_time, category=quotes_category)
        else:
            not_found = [d['tip'] for d in quote_details if d['source'] is None]
            checkpoint.add_step("Tips Are Direct Quotes", False, 6, f"Tips not found in sources: {not_found[:2]}", execution_time=step_time, category=quotes_category)

    # =========================================================================
    # Step 1.7: Image from Original
    # =========================================================================
    step_start = time.time()

    # Only check images that belong to the first recipe's structure
    first_recipe_image_ids = [
        item['content'] for item in first_recipe.structure
        if item.get('type') == 'image'
    ]

    # Filter images to only those whose filename contains a first-recipe image ID.
    # Check both uncropped and cropped directories — the original match should work
    # regardless of whether the agent cropped the image.
    all_images = (
        glob.glob(os.path.join(DOC_IMAGES_DIR, "*")) +
        glob.glob(os.path.join(DOC_IMAGES_CROPPED_DIR, "*"))
    )
    doc_images = [
        img for img in all_images
        if any(img_id in os.path.basename(img) for img_id in first_recipe_image_ids)
    ]

    image_match = False
    image_match_details = "No images found in first recipe"
    image_category = StepCategory.EXECUTION_ERROR  # default: images/gold unavailable

    if doc_images and os.path.exists(GOLD_IMAGE_ORIGINAL):
        for doc_img in doc_images:
            try:
                match_result, match_method = match_image_tiered(doc_img, GOLD_IMAGE_ORIGINAL, model)
                if match_result:
                    image_match = True
                    image_match_details = f"Image matched via {match_method}"
                    image_category = StepCategory.from_match_method(match_method)
                    break
            except Exception as e:
                print(f"Warning: Image matching failed for {os.path.basename(doc_img)}: {e}")

        if not image_match:
            image_match_details = "First recipe image does not match original recipe image"
            image_category = StepCategory.LLM_VLM_JUDGEMENT  # VLM tier made the final rejection
    elif not first_recipe_image_ids:
        image_match_details = "No image elements found in first recipe structure"

    step_time = time.time() - step_start
    checkpoint.add_step("Image from Original", image_match, 7, image_match_details, execution_time=step_time, category=image_category)

    # =========================================================================
    # Step 1.8: Image Properly Cropped — dimensions match template slot
    # =========================================================================
    step_start = time.time()
    image_cropped = False
    crop_details = "No image elements found in first recipe"
    crop_category = StepCategory.EXECUTION_ERROR  # default: no image/dimensions available

    if first_recipe_image_ids:
        image_id = first_recipe_image_ids[0]
        try:
            doc_image_dims = get_image_dimensions_from_doc(doc_id, image_id, DOCS_SERVICE)
        except Exception as e:
            print(f"Warning: Failed to get image dimensions: {e}")
            doc_image_dims = None

        if doc_image_dims:
            crop_category = StepCategory.FUZZY_MATCH  # tolerance-based dimension comparison
            doc_w = doc_image_dims.get('width', {}).get('magnitude', 0)
            doc_h = doc_image_dims.get('height', {}).get('magnitude', 0)

            tolerance = 0.25
            w_ratio = abs(doc_w - TEMPLATE_IMAGE_WIDTH_PT) / TEMPLATE_IMAGE_WIDTH_PT if TEMPLATE_IMAGE_WIDTH_PT else 1
            h_ratio = abs(doc_h - TEMPLATE_IMAGE_HEIGHT_PT) / TEMPLATE_IMAGE_HEIGHT_PT if TEMPLATE_IMAGE_HEIGHT_PT else 1

            if w_ratio <= tolerance and h_ratio <= tolerance:
                image_cropped = True
                crop_details = (f"Image dimensions ({doc_w:.0f}x{doc_h:.0f} PT) fit template slot "
                                f"({TEMPLATE_IMAGE_WIDTH_PT:.0f}x{TEMPLATE_IMAGE_HEIGHT_PT:.0f} PT)")
            else:
                crop_details = (f"Image dimensions ({doc_w:.0f}x{doc_h:.0f} PT) don't fit template "
                                f"({TEMPLATE_IMAGE_WIDTH_PT:.0f}x{TEMPLATE_IMAGE_HEIGHT_PT:.0f} PT) — "
                                f"width off {w_ratio:.0%}, height off {h_ratio:.0%}")
        else:
            crop_details = f"Could not retrieve dimensions for image {image_id}"

    step_time = time.time() - step_start
    checkpoint.add_step("Image Properly Cropped", image_cropped, 8, crop_details, execution_time=step_time, category=crop_category)

    # =========================================================================
    # Step 1.9: Image Shows Recipe Text — VLM confirms recipe content is visible
    # =========================================================================
    step_start = time.time()
    shows_text = False
    text_details = "No image elements found in first recipe"
    text_category = StepCategory.EXECUTION_ERROR  # default: guards / VLM failure

    if first_recipe_image_ids:
        doc_images_cropped = [
            img for img in glob.glob(os.path.join(DOC_IMAGES_CROPPED_DIR, "*"))
            if any(img_id in os.path.basename(img) for img_id in first_recipe_image_ids)
        ]

        if doc_images_cropped and os.path.exists(GOLD_IMAGE_ORIGINAL):
            try:
                shows_text = binary_compare_images(
                    model, doc_images_cropped[0], GOLD_IMAGE_ORIGINAL,
                    "Does the first image show the same recipe text/content as the second image? "
                    "The recipe name, ingredients, and preparation text should be clearly visible and readable. "
                    "Answer Yes if the recipe text content is clearly shown, No if it's missing, cut off, or unreadable."
                )
                text_category = StepCategory.LLM_VLM_JUDGEMENT
                if shows_text:
                    text_details = "Recipe text clearly visible in image (VLM confirmed)"
                else:
                    text_details = "Recipe text not clearly visible in image (VLM rejected)"
            except Exception as e:
                print(f"Warning: VLM content check failed: {e}")
                text_details = f"VLM content check failed: {str(e)[:50]}"
        elif not doc_images_cropped:
            text_details = "No cropped image files found for first recipe"
        else:
            text_details = "Gold reference image not available"

    step_time = time.time() - step_start
    checkpoint.add_step("Image Shows Recipe Text", shows_text, 9, text_details, execution_time=step_time, category=text_category)

    # =========================================================================
    # Step 1.10: Info Modified from Defaults
    # Ready In is fully optional for this instance — only fail if it equals
    # the template default "20". Serves and Calories just need to differ from
    # the template defaults.
    # =========================================================================
    step_start = time.time()
    metadata = extract_recipe_metadata(first_recipe.text)

    info_pass = True
    info_parts = []

    # Check Ready In: fully optional — pass if not present, fail only if template default
    ready_in = metadata.get('ready_in')
    if ready_in:
        if ready_in == TEMPLATE_DEFAULTS.get('ready_in'):
            info_pass = False
            info_parts.append(f"Ready In: {ready_in} (unchanged from template default)")
        else:
            info_parts.append(f"Ready In: {ready_in} (modified from default)")
    else:
        info_parts.append("Ready In: not specified (acceptable for this recipe)")

    # Check Serves: must differ from default (8) if present, but not required
    # (the recipe image may not contain serving info)
    serves = metadata.get('serves')
    if serves and serves != TEMPLATE_DEFAULTS.get('serves'):
        info_parts.append(f"Serves: {serves} (changed from default {TEMPLATE_DEFAULTS.get('serves')})")
    elif serves and serves == TEMPLATE_DEFAULTS.get('serves'):
        info_pass = False
        info_parts.append(f"Serves: {serves} (unchanged from default)")
    else:
        info_parts.append("Serves: not specified")

    # Check Calories: must differ from default (280) if present, but not required
    calories = metadata.get('calories')
    if calories and calories != TEMPLATE_DEFAULTS.get('calories'):
        info_parts.append(f"Calories: {calories} (changed from default {TEMPLATE_DEFAULTS.get('calories')})")
    elif calories and calories == TEMPLATE_DEFAULTS.get('calories'):
        info_pass = False
        info_parts.append(f"Calories: {calories} (unchanged from default)")
    else:
        info_parts.append("Calories: not specified")

    step_time = time.time() - step_start
    checkpoint.add_step("Info Modified from Defaults", info_pass, 10, '; '.join(info_parts), execution_time=step_time, category=StepCategory.DETERMINISTIC)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Grade Checkpoint 2: Additional Recipe Pages Formatting (32 pts)

    Evaluates 8 criteria for each of 4 additional recipes (pages 2, 3, 4, 5).

    Criteria per recipe:
    2.1 Title theme: Related to cookies or someone's favorite dessert
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
    checkpoint = Checkpoint(total=32, result=0, name="Additional Recipe Pages Formatting")

    # Validate gold template page exists — required for format comparison
    if not os.path.exists(GOLD_TEMPLATE_PAGE1):
        raise FileNotFoundError(f"Gold template page 1 image missing: {GOLD_TEMPLATE_PAGE1}")

    # Get additional recipes (pages 2, 3, 4, 5) from discovered recipes
    additional_recipes = recipes[1:]  # Skip first recipe

    if model is None:
        model = load_model(model_id)

    # If fewer than 4 additional recipes found, mark missing ones as failed
    if len(additional_recipes) < 4:
        print(f"Warning: Only found {len(additional_recipes)} additional recipes (expected 4)")
        for i in range(len(additional_recipes), 4):
            recipe_num = i + 2  # Pages 2, 3, 4, 5
            step_names = [
                "Title Theme", "Source URL", "Format Match", "Ingredients Modified",
                "Preparation Modified", "Tips Modified", "Info Modified", "Visual Distinction"
            ]
            for step_idx, step_name in enumerate(step_names):
                step_id = i * 8 + step_idx + 1
                checkpoint.add_step(
                    f"Recipe {recipe_num} - {step_name}",
                    False, step_id, f"Recipe {recipe_num} not found in document",
                    execution_time=0, category=StepCategory.EXECUTION_ERROR
                )

    # Get PDF page images for visual distinction check
    pdf_images = sorted(glob.glob(os.path.join(PDF_IMAGES_DIR, "*.png")))

    # Evaluate each additional recipe (up to 4)
    for recipe_idx, recipe in enumerate(additional_recipes[:4]):
        recipe_num = recipe.recipe_num
        recipe_text = recipe.text
        recipe_structure = recipe.structure

        # Base step ID for this recipe (0-indexed: recipe_idx * 8)
        base_step_id = recipe_idx * 8

        # =====================================================================
        # Step 2.1: Title Theme Check
        # =====================================================================
        step_start = time.time()
        title = recipe.title
        has_theme, theme_details = check_title_theme(title, theme_keywords=['cookie', 'cookies', 'dessert', 'favorite'], theme_description="cookies or someone's favorite dessert")
        theme_category = StepCategory.DETERMINISTIC  # keyword check decided

        # If keyword check fails, use LLM as fallback
        if not has_theme and title:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": """You are checking if a recipe title is related to cookies or dessert themes.

A title is THEMATIC if it relates to:
- Cookies: cookie, cookies, biscuit, snickerdoodle, macaroon, shortbread
- Desserts: cake, brownie, bar, treat, sweet, dessert
- Favorites: favorite, best, classic, homemade
- Baking: baked goods, pastry, confection

Answer 'Yes' if the title is related to any of these themes.
Answer 'No' if the title is completely unrelated (e.g., "Grilled Chicken", "Tomato Soup")."""}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Is this recipe title related to cookies, desserts, or someone's favorite baked treat?\n\nTitle: {title}"}]
                    }
                ]
                response = model(messages)
                theme_category = StepCategory.LLM_VLM_JUDGEMENT
                has_theme = response.strip().lower().startswith('yes')
                if has_theme:
                    theme_details = f"LLM confirmed thematic: '{title}'"
                else:
                    theme_details = f"LLM rejected: '{title}' not related to cookies/dessert themes"
            except Exception as e:
                theme_details = f"LLM theme check failed: {str(e)[:50]}"
                theme_category = StepCategory.EXECUTION_ERROR

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Title Theme",
            has_theme, base_step_id + 1, theme_details,
            execution_time=step_time, category=theme_category
        )

        # =====================================================================
        # Step 2.2: Source URL Valid — must be positioned below the title
        # =====================================================================
        step_start = time.time()
        try:
            recipe_links = extract_hyperlinks(doc_id, DOCS_SERVICE, recipe_num=recipe_num, recipe_titles=[r.title for r in recipes])
        except Exception as e:
            print(f"Warning: Failed to extract hyperlinks for recipe {recipe_num}: {e}")
            recipe_links = []

        # Find title position and first section header position in recipe text
        title_pos = recipe_text.find(title) if title and len(title) > 2 else -1

        first_section_pos = len(recipe_text)
        for section in ['ingredients', 'preparation', 'tips', 'ready in', 'serves', 'calories']:
            pos = recipe_text.lower().find(section)
            if 0 <= pos < first_section_pos:
                first_section_pos = pos

        # Filter to URLs that appear after the title but before the first section
        source_urls = []
        for link in recipe_links:
            url = link['url']
            if not url.startswith('http'):
                continue
            link_pos = recipe_text.find(url[:30])
            if link_pos < 0:
                link_pos = recipe_text.find(link['text'][:30])
            if title_pos < link_pos < first_section_pos:
                source_urls.append(url)

        has_valid_url = False
        url_details = "No URLs found between title and recipe sections"

        for url in source_urls:
            is_valid, details = validate_url_accessible(url)
            if is_valid:
                has_valid_url = True
                url_details = f"Valid source URL below title: {url[:50]}..."
                break

        if not has_valid_url and source_urls:
            url_details = f"Found {len(source_urls)} URLs below title but none accessible"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Source URL",
            has_valid_url, base_step_id + 2, url_details,
            execution_time=step_time, category=StepCategory.DETERMINISTIC
        )

        # =====================================================================
        # Step 2.3: Format Matches First Page
        # =====================================================================
        step_start = time.time()
        has_format = False
        format_details = "Unable to verify format"
        format_category = StepCategory.STRUCTURAL  # section/image fallback decides unless VLM confirms

        # The recipe structure was already validated by discover_recipes to have
        # all required sections. Check that recipe.sections_found covers the format.
        found_sections = recipe.sections_found  # From discover_recipes validation

        # Check if recipe has an image element (template has image on left)
        has_image = any(item.get('type') == 'image' for item in recipe_structure)

        # Compare recipe page against gold template page 1 for format match
        if recipe.pdf_pages and recipe.pdf_pages[0] < len(pdf_images) and os.path.exists(GOLD_TEMPLATE_PAGE1):
            page_image = pdf_images[recipe.pdf_pages[0]]
            try:
                vlm_confirmed = binary_compare_images(
                    model, page_image, GOLD_TEMPLATE_PAGE1,
                    "Do these two pages have the same layout format? Both should have a 2-column layout "
                    "with title and image on the left, and ingredients/preparation/tips on the right. "
                    "Ignore content differences — only compare the structural layout."
                )
            except Exception as e:
                print(f"Warning: VLM format check failed for recipe {recipe_num}: {e}")
                vlm_confirmed = False

            if vlm_confirmed:
                has_format = True
                format_details = f"Format matches template layout; sections: {found_sections}, has_image: {has_image}"
                format_category = StepCategory.LLM_VLM_JUDGEMENT
            else:
                # Fallback: structural check — has all sections + an image
                if len(found_sections) >= 3 and has_image:
                    has_format = True
                    format_details = f"Structural match: sections={found_sections}, has_image={has_image} (VLM did not confirm layout match)"
                else:
                    format_details = f"Format check failed; sections={found_sections}, has_image={has_image}"
        else:
            # No PDF page or template reference available — structural fallback only
            if len(found_sections) >= 3 and has_image:
                has_format = True
                format_details = f"Structural match: sections={found_sections}, has_image={has_image} (no PDF page or template reference)"
            else:
                format_details = f"Missing format elements; sections={found_sections}, has_image={has_image}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Format Match",
            has_format, base_step_id + 3, format_details,
            execution_time=step_time, category=format_category
        )

        # =====================================================================
        # Step 2.4: Ingredients Modified
        # =====================================================================
        step_start = time.time()
        ingredients_content = extract_section_content(recipe_structure, "Ingredients")
        ing_modified, ing_details = check_content_modified_from_default(ingredients_content, "ingredients")
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Ingredients Modified",
            ing_modified, base_step_id + 4, ing_details,
            execution_time=step_time, category=StepCategory.FUZZY_MATCH
        )

        # =====================================================================
        # Step 2.5: Preparation Modified
        # =====================================================================
        step_start = time.time()
        prep_content = extract_section_content(recipe_structure, "Preparation")
        prep_modified, prep_details = check_content_modified_from_default(prep_content, "preparation")
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Preparation Modified",
            prep_modified, base_step_id + 5, prep_details,
            execution_time=step_time, category=StepCategory.FUZZY_MATCH
        )

        # =====================================================================
        # Step 2.6: Tips Modified
        # =====================================================================
        step_start = time.time()
        tips_content = extract_section_content(recipe_structure, "Tips")
        tips_modified, tips_details = check_content_modified_from_default(tips_content, "tips")
        step_time = time.time() - step_start

        checkpoint.add_step(
            f"Recipe {recipe_num} - Tips Modified",
            tips_modified, base_step_id + 6, tips_details,
            execution_time=step_time, category=StepCategory.FUZZY_MATCH
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
            execution_time=step_time, category=StepCategory.DETERMINISTIC
        )

        # =====================================================================
        # Step 2.8: Visually Distinct — must differ from ALL other recipes
        # =====================================================================
        step_start = time.time()
        is_distinct = False
        visual_details = "Unable to verify visual distinction"
        visual_category = StepCategory.EXECUTION_ERROR  # default: comparison data unavailable

        if recipe.pdf_pages and recipe.pdf_pages[0] < len(pdf_images):
            page_image = pdf_images[recipe.pdf_pages[0]]

            # Build list of all other recipe page images to compare against
            all_recipes_with_pages = recipes  # Includes recipe 1 and all additional
            other_pages = []
            for other in all_recipes_with_pages:
                if other.recipe_num == recipe_num:
                    continue
                if other.pdf_pages and other.pdf_pages[0] < len(pdf_images):
                    other_pages.append((other.recipe_num, other, pdf_images[other.pdf_pages[0]]))

            if not other_pages:
                visual_details = "No other recipe pages available for comparison"
            else:
                similar_to = []
                distinct_from = []
                visual_items = []  # (category, distinct) per comparison for StepCategory.aggregate
                for other_num, other_recipe_obj, other_image in other_pages:
                    same_page = (recipe.pdf_pages and other_recipe_obj.pdf_pages
                                 and recipe.pdf_pages[0] == other_recipe_obj.pdf_pages[0])
                    same_page_note = (
                        f"IMPORTANT: Both recipes appear on the SAME page image. "
                        f"Look for the recipe titled '{title}' and compare its header/text styling "
                        f"to the recipe titled '{other_recipe_obj.title}' on that same page. "
                    ) if same_page else ""
                    styling_prompt = (
                        f"{same_page_note}"
                        f"Compare the visual styling of two recipes: '{title}' vs '{other_recipe_obj.title}'. "
                        f"Focus ONLY on the title and header colors, font styles, and text formatting of each recipe. "
                        f"First, check if '{title}' has any custom styling "
                        f"(colored headers, colored text, non-default fonts, bold/italic formatting, "
                        f"background colors, or decorative elements). If it uses only plain "
                        f"default black text with no color or font changes, answer 'Yes' (unstyled). "
                        f"If '{title}' DOES have custom styling, answer 'Yes' only if '{other_recipe_obj.title}' "
                        f"uses the exact same color scheme and font styles. "
                        f"Answer 'No' if '{title}' has distinct custom styling different from '{other_recipe_obj.title}'."
                    )
                    try:
                        looks_same = binary_compare_images(model, page_image, other_image, mode=styling_prompt)
                        visual_items.append((StepCategory.LLM_VLM_JUDGEMENT, not looks_same))
                        if looks_same:
                            similar_to.append(other_num)
                        else:
                            distinct_from.append(other_num)
                    except Exception as e:
                        print(f"Warning: VLM comparison failed for recipe {recipe_num} vs {other_num}: {e}")
                        distinct_from.append(other_num)  # Assume distinct on error
                        visual_items.append((StepCategory.VACUOUS_PASS, True))  # silent pass: no check ran

                visual_category = StepCategory.aggregate(visual_items)
                if not similar_to:
                    is_distinct = True
                    visual_details = f"Visually distinct from all other recipes ({[f'R{n}' for n in distinct_from]})"
                else:
                    visual_details = f"Visually similar to recipe(s) {similar_to}; distinct from {distinct_from}"
        else:
            visual_details = "PDF page not available for visual comparison"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Visual Distinction",
            is_distinct, base_step_id + 8, visual_details,
            execution_time=step_time, category=visual_category
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Grade Checkpoint 3: Additional Recipe Pages Content (15 pts)

    Validates content of 3 additional recipes (pages 2, 3, 4) against their source URLs.
    Requires valid, accessible source URLs from Checkpoint 2.

    Criteria per recipe (5 per recipe x 3 recipes = 15 total):
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

    # Get additional recipes (pages 2, 3, 4) from discovered recipes
    additional_recipes = recipes[1:]  # Skip first recipe

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
                    execution_time=0, category=StepCategory.EXECUTION_ERROR
                )

    # Cache for webpage content to avoid redundant fetches
    webpage_cache = {}

    # Evaluate each additional recipe (up to 3)
    for recipe_idx, recipe in enumerate(additional_recipes[:3]):
        recipe_num = recipe.recipe_num
        recipe_text = recipe.text
        recipe_structure = recipe.structure

        # Base step ID for this recipe (0-indexed: recipe_idx * 5)
        base_step_id = recipe_idx * 5

        # Get source URL for this recipe
        try:
            recipe_links = extract_hyperlinks(doc_id, DOCS_SERVICE, recipe_num=recipe_num, recipe_titles=[r.title for r in recipes])
        except Exception as e:
            print(f"Warning: Failed to extract hyperlinks for recipe {recipe_num}: {e}")
            recipe_links = []
        source_urls = [link['url'] for link in recipe_links if link['url'].startswith('http')]

        # Find first valid, accessible source URL and try to fetch its content
        source_url = None
        webpage_content = None

        for url in source_urls:
            is_valid, url_details = validate_url_accessible(url)
            if is_valid:
                source_url = url
                # Fetch webpage content
                if url in webpage_cache:
                    webpage_content = webpage_cache[url]
                else:
                    try:
                        content, _ = fetch_page_text_content(url)
                    except Exception as e:
                        print(f"Warning: Failed to fetch {url}: {e}")
                        content = None
                    if content:
                        webpage_cache[url] = content
                        webpage_content = content
                break  # Use the first accessible URL even if content fetch fails

        # If no accessible source URL at all, fail all steps
        if not source_url:
            if not source_urls:
                url_failure_reason = "No hyperlinks found in recipe"
            else:
                url_failure_reason = f"None of the {len(source_urls)} URLs were accessible"
            for step_idx, step_name in enumerate(["Photo from Source", "Ingredients Match",
                                                   "Preparation Match", "Tips Direct Quotes", "Info Relevant"]):
                checkpoint.add_step(
                    f"Recipe {recipe_num} - {step_name}",
                    False, base_step_id + step_idx + 1, url_failure_reason,
                    execution_time=0, category=StepCategory.EXECUTION_ERROR
                )
            continue

        content_unavailable = f"Source URL accessible ({source_url[:50]}) but content could not be fetched (site may block scraping)"

        # Get recipe title for validation
        recipe_title = recipe.title
        print(f"Recipe {recipe_num}: '{recipe_title}' - Source: {source_url[:50]}...")

        # =====================================================================
        # Step 3.1: Photo from Source URL
        # =====================================================================
        step_start = time.time()
        photo_valid = False
        photo_details = "No image found for recipe"
        photo_category = StepCategory.EXECUTION_ERROR  # default: image data unavailable

        # Get this recipe's image IDs from its structure
        recipe_image_ids = [
            item['content'] for item in recipe_structure
            if item.get('type') == 'image'
        ]

        # Get doc images scoped to this recipe
        all_doc_images = (
            glob.glob(os.path.join(DOC_IMAGES_DIR, "*")) +
            glob.glob(os.path.join(DOC_IMAGES_CROPPED_DIR, "*"))
        )
        recipe_doc_images = [
            img for img in all_doc_images
            if any(img_id in os.path.basename(img) for img_id in recipe_image_ids)
        ]

        if recipe_doc_images:
            # Try to extract the recipe image URL from the webpage, then download it
            source_image_path = None
            try:
                image_url = extract_recipe_image_url(source_url)
                if image_url:
                    source_image_path = download_image_from_url(
                        image_url, DATA_DIR, wayback_fallback=True
                    )
            except Exception as e:
                print(f"Warning: Failed to extract/download source image: {e}")

            if source_image_path:
                # Direct image comparison against source
                for doc_img in recipe_doc_images:
                    try:
                        match_result, match_method = match_image_tiered(doc_img, source_image_path, model)
                        if match_result:
                            photo_valid = True
                            photo_details = f"Recipe image matches source via {match_method}"
                            photo_category = StepCategory.from_match_method(match_method)
                            break
                    except Exception as e:
                        print(f"Warning: Image match failed for {os.path.basename(doc_img)}: {e}")

            # Fallback: VLM check if direct comparison didn't work
            if not photo_valid:
                try:
                    photo_valid, photo_details = validate_image_matches_recipe(
                        model, recipe_doc_images[0], recipe_title
                    )
                    photo_category = StepCategory.LLM_VLM_JUDGEMENT
                    if photo_valid:
                        photo_details = f"Photo appears to show {recipe_title} (VLM validated)"
                except Exception as e:
                    print(f"Warning: VLM photo validation failed for recipe {recipe_num}: {e}")
                    photo_details = f"VLM photo validation failed: {str(e)[:50]}"
                    photo_category = StepCategory.EXECUTION_ERROR

            # Clean up downloaded source image
            if source_image_path and os.path.exists(source_image_path):
                try:
                    os.remove(source_image_path)
                except OSError:
                    pass
        elif not recipe_image_ids:
            photo_details = "No image elements found in recipe structure"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Photo from Source",
            photo_valid, base_step_id + 1, photo_details,
            execution_time=step_time, category=photo_category
        )

        # =====================================================================
        # Step 3.2: Ingredients Match Source
        # =====================================================================
        step_start = time.time()
        ingredients_match = False
        ingredients_details = "Unable to verify ingredients"
        ing_category = StepCategory.EXECUTION_ERROR  # default: source data unavailable

        doc_ingredients_text = extract_section_content(recipe_structure, "Ingredients")
        doc_ingredients = extract_list_items(doc_ingredients_text)

        if not webpage_content and source_url:
            # URL accessible but content blocked — LLM reasonableness check against title
            if not doc_ingredients:
                ingredients_details = "No ingredients found in document recipe"
            else:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are checking if a list of ingredients is reasonable for a given recipe. Answer 'Yes' if the ingredients are plausible for this recipe. Answer 'No' if they are clearly wrong or unrelated."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Are these ingredients reasonable for a recipe called '{recipe_title}'?\n\nIngredients:\n" + '\n'.join(f"- {i}" for i in doc_ingredients)}]
                        }
                    ]
                    response = model(messages)
                    ing_category = StepCategory.LLM_VLM_JUDGEMENT
                    if response.strip().lower().startswith('yes'):
                        ingredients_match = True
                        ingredients_details = f"Ingredients reasonable for '{recipe_title}' (LLM verified, source content not fetchable)"
                    else:
                        ingredients_details = f"Ingredients not reasonable for '{recipe_title}' (LLM rejected, source content not fetchable)"
                except Exception as e:
                    print(f"Warning: LLM ingredient reasonableness check failed: {e}")
                    ingredients_details = content_unavailable
        elif not webpage_content:
            ingredients_details = content_unavailable
        else:

            if not doc_ingredients:
                ingredients_details = "No ingredients found in document recipe"
            else:
                try:
                    source_ingredients = extract_ingredients_from_webpage(model, webpage_content, recipe_title)
                except Exception as e:
                    print(f"Warning: Failed to extract ingredients from webpage: {e}")
                    source_ingredients = []

                if not source_ingredients:
                    ingredients_details = "Could not extract ingredients from source webpage for comparison"
                else:
                    strict_match, strict_details = compare_ingredient_lists(doc_ingredients, source_ingredients, fuzzy_threshold=90)
                    ing_category = StepCategory.FUZZY_MATCH

                    if strict_match:
                        ingredients_match = True
                        ingredients_details = f"Ingredients match source ({len(doc_ingredients)} items, strict match)"
                    else:
                        llm_match = False
                        for attempt in range(2):
                            try:
                                llm_match, _ = compare_lists_with_llm(model, doc_ingredients, source_ingredients, "ingredients")
                                ing_category = StepCategory.LLM_VLM_JUDGEMENT
                                break
                            except Exception as e:
                                if attempt == 0:
                                    print(f"Warning: LLM ingredient comparison failed, retrying: {e}")
                                else:
                                    print(f"Warning: LLM ingredient comparison failed after retry: {e}")
                        if llm_match:
                            ingredients_match = True
                            ingredients_details = f"Ingredients match source (LLM verified, {len(doc_ingredients)} items)"
                        else:
                            missing = [d['gold'] for d in strict_details if not d['matched'] and d['gold'] != '[Extra ingredients check]']
                            ingredients_details = f"Ingredients differ from source: {', '.join(missing[:3])}..."

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Ingredients Match",
            ingredients_match, base_step_id + 2, ingredients_details,
            execution_time=step_time, category=ing_category
        )

        # =====================================================================
        # Step 3.3: Preparation Steps Match Source
        # =====================================================================
        step_start = time.time()
        prep_match = False
        prep_details = "Unable to verify preparation steps"
        prep_category = StepCategory.EXECUTION_ERROR  # default: source data unavailable

        doc_prep_text = extract_section_content(recipe_structure, "Preparation")
        doc_steps = extract_list_items(doc_prep_text)

        if not webpage_content and source_url:
            # URL accessible but content blocked — LLM reasonableness check against title
            if not doc_steps:
                prep_details = "No preparation steps found in document recipe"
            else:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are checking if preparation steps are reasonable for a given recipe. Answer 'Yes' if the steps are plausible for this recipe. Answer 'No' if they are clearly wrong or unrelated."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Are these preparation steps reasonable for a recipe called '{recipe_title}'?\n\nSteps:\n" + '\n'.join(f"{i+1}. {s}" for i, s in enumerate(doc_steps))}]
                        }
                    ]
                    response = model(messages)
                    prep_category = StepCategory.LLM_VLM_JUDGEMENT
                    if response.strip().lower().startswith('yes'):
                        prep_match = True
                        prep_details = f"Preparation reasonable for '{recipe_title}' (LLM verified, source content not fetchable)"
                    else:
                        prep_details = f"Preparation not reasonable for '{recipe_title}' (LLM rejected, source content not fetchable)"
                except Exception as e:
                    print(f"Warning: LLM preparation reasonableness check failed: {e}")
                    prep_details = content_unavailable
        elif not webpage_content:
            prep_details = content_unavailable
        else:

            if not doc_steps:
                prep_details = "No preparation steps found in document recipe"
            else:
                try:
                    source_steps = extract_preparation_from_webpage(model, webpage_content, recipe_title)
                except Exception as e:
                    print(f"Warning: Failed to extract preparation from webpage: {e}")
                    source_steps = []

                if not source_steps:
                    prep_details = "Could not extract preparation steps from source webpage for comparison"
                else:
                    try:
                        prep_match, llm_explanation = compare_lists_with_llm(model, doc_steps, source_steps, "preparation steps")
                        prep_category = StepCategory.LLM_VLM_JUDGEMENT
                    except Exception as e:
                        print(f"Warning: LLM preparation comparison failed: {e}")
                        prep_match = False
                        llm_explanation = str(e)

                    if prep_match:
                        prep_details = f"Preparation matches source (LLM verified, {len(doc_steps)} doc steps vs {len(source_steps)} source steps)"
                    else:
                        prep_details = f"Preparation content differs from source (LLM judged mismatch, {len(doc_steps)} doc steps vs {len(source_steps)} source steps): {llm_explanation[:150]}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Preparation Match",
            prep_match, base_step_id + 3, prep_details,
            execution_time=step_time, category=prep_category
        )

        # =====================================================================
        # Step 3.4: Tips are Direct Quotes from Source
        # =====================================================================
        step_start = time.time()
        tips_valid = False
        tips_details = "Unable to verify tips"
        tips_category = StepCategory.EXECUTION_ERROR  # default: source data unavailable

        tip_pairs_additional = extract_tips_with_sources(recipe_structure)
        doc_tips = [p['tip'] for p in tip_pairs_additional]

        if not webpage_content and source_url:
            # URL accessible but content blocked — LLM reasonableness check
            if not doc_tips:
                tips_details = "No tips found in document recipe"
            else:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are checking if cooking tips are reasonable and relevant for a given recipe. Answer 'Yes' if the tips are plausible and useful for this recipe. Answer 'No' if they are clearly wrong or unrelated."}]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": f"Are these tips reasonable for a recipe called '{recipe_title}'?\n\nTips:\n" + '\n'.join(f"- {t}" for t in doc_tips)}]
                        }
                    ]
                    response = model(messages)
                    tips_category = StepCategory.LLM_VLM_JUDGEMENT
                    if response.strip().lower().startswith('yes'):
                        tips_valid = True
                        tips_details = f"Tips reasonable for '{recipe_title}' (LLM verified, source content not fetchable)"
                    else:
                        tips_details = f"Tips not reasonable for '{recipe_title}' (LLM rejected, source content not fetchable)"
                except Exception as e:
                    print(f"Warning: LLM tips reasonableness check failed: {e}")
                    tips_details = content_unavailable
        elif not webpage_content:
            tips_details = content_unavailable
        else:
            if not doc_tips:
                tips_details = "No tips found in document recipe"
            else:
                # Check each tip against its own paired source URL first,
                # then fall back to the recipe's main source page
                tips_verified = []
                tips_not_found = []
                tip_items = []  # (category, success) per tip for StepCategory.aggregate

                for pair in tip_pairs_additional:
                    tip = pair['tip']
                    tip_url = pair.get('url')
                    tip_found = False
                    tip_category = StepCategory.FUZZY_MATCH  # last mechanism that ran for this tip

                    # Try the tip's own source URL if different from main source
                    if tip_url and tip_url != source_url:
                        tip_content = webpage_cache.get(tip_url)
                        if tip_content is None:
                            try:
                                fetched, _ = fetch_page_text_content(tip_url)
                                if fetched:
                                    webpage_cache[tip_url] = fetched
                                    tip_content = fetched
                            except Exception:
                                pass
                        if tip_content:
                            is_quote, quote_score = verify_tip_is_quote(tip, tip_content)
                            if is_quote:
                                tip_found = True
                                tip_category = StepCategory.DETERMINISTIC if quote_score == 100 else StepCategory.FUZZY_MATCH

                    # Try the recipe's main source page
                    if not tip_found:
                        is_quote, quote_score = verify_tip_is_quote(tip, webpage_content)
                        if is_quote:
                            tip_found = True
                            tip_category = StepCategory.DETERMINISTIC if quote_score == 100 else StepCategory.FUZZY_MATCH

                    # LLM fallback for unscrapable tip source URLs
                    if not tip_found and tip_url:
                        scraped = webpage_cache.get(tip_url, '')
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": (
                                    "You are verifying whether a cooking tip could plausibly "
                                    "be a direct quote from a given source URL. The webpage "
                                    "could not be fully scraped, so you cannot check the text "
                                    "directly. Instead, judge based on the URL and the tip content."
                                )}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": (
                                    f"Tip: {tip}\n\n"
                                    f"Source URL: {tip_url}\n\n"
                                    f"Scraped content (may be incomplete): {scraped[:300] if scraped else '(empty)'}\n\n"
                                    f"Could this tip plausibly be a direct quote from this URL? "
                                    f"Consider whether the URL is a cooking/recipe page and whether "
                                    f"the tip content is the kind of advice that page would contain. "
                                    f"Answer 'Yes' or 'No'."
                                )}]
                            }
                        ]
                        try:
                            response = model(messages).strip()
                            tip_category = StepCategory.LLM_VLM_JUDGEMENT
                            if response.lower().startswith('yes'):
                                tip_found = True
                        except Exception as e:
                            print(f"Warning: LLM tip quote fallback failed: {e}")

                    tip_items.append((tip_category, tip_found))
                    if tip_found:
                        tips_verified.append(tip[:30])
                    else:
                        tips_not_found.append(tip[:30])

                tips_category = StepCategory.aggregate(tip_items)
                if not tips_not_found:
                    tips_valid = True
                    tips_details = f"All {len(doc_tips)} tips verified as quotes from sources"
                else:
                    tips_details = f"Tips not found in source: {tips_not_found[:2]}"

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Tips Direct Quotes",
            tips_valid, base_step_id + 4, tips_details,
            execution_time=step_time, category=tips_category
        )

        # =====================================================================
        # Step 3.5: Info Under Photo Relevant to Source
        # =====================================================================
        step_start = time.time()
        info_relevant = False
        info_details = "Unable to verify metadata relevance"
        info_category = StepCategory.EXECUTION_ERROR  # default: source data unavailable

        doc_metadata = extract_recipe_metadata(recipe_text)
        is_modified, _ = check_metadata_modified(doc_metadata, TEMPLATE_DEFAULTS)

        if not is_modified:
            info_details = "Metadata unchanged from template defaults"
            info_category = StepCategory.DETERMINISTIC
        elif not webpage_content and source_url:
            # URL accessible but content blocked — LLM reasonableness check
            try:
                meta_str = ', '.join(f"{k}: {v}" for k, v in doc_metadata.items() if v)
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are checking if recipe metadata (Ready In time, Servings, Calories) is reasonable for a given recipe. Answer 'Yes' if the values are plausible. Answer 'No' if they are clearly wrong."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Is this metadata reasonable for a recipe called '{recipe_title}'?\n\n{meta_str}"}]
                    }
                ]
                response = model(messages)
                info_category = StepCategory.LLM_VLM_JUDGEMENT
                if response.strip().lower().startswith('yes'):
                    info_relevant = True
                    info_details = f"Metadata reasonable for '{recipe_title}' (LLM verified, source content not fetchable): {meta_str}"
                else:
                    info_details = f"Metadata not reasonable for '{recipe_title}' (LLM rejected, source content not fetchable): {meta_str}"
            except Exception as e:
                print(f"Warning: LLM metadata reasonableness check failed: {e}")
                info_details = content_unavailable
        elif not webpage_content:
            info_details = content_unavailable
        else:
            try:
                source_metadata = extract_metadata_from_webpage(model, webpage_content)
            except Exception as e:
                print(f"Warning: Failed to extract metadata from webpage: {e}")
                source_metadata = {}

            if not any(source_metadata.values()):
                info_details = "Could not extract metadata from source webpage for comparison"
            else:
                info_relevant, info_details = compare_metadata_relevance(
                    doc_metadata, source_metadata, TEMPLATE_DEFAULTS
                )
                info_category = StepCategory.FUZZY_MATCH  # +/-50% range comparison

        step_time = time.time() - step_start
        checkpoint.add_step(
            f"Recipe {recipe_num} - Info Relevant",
            info_relevant, base_step_id + 5, info_details,
            execution_time=step_time, category=info_category
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4(browsing_history: Optional[list] = None):
    """
    Grade Checkpoint 4: Websites Visited Check (4 pts)

    Verifies that the agent visited the relevant source URLs during task execution.

    Criteria:
    4.1 First page tips source URLs are in the browsing history (1 pt)
    4.2 Additional recipe source URLs are in the browsing history (3 pts — 1 per recipe)
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Websites Visited Check")

    if not browsing_history:
        checkpoint.add_step("Tips Source Visited", False, 1, "No browsing history provided", execution_time=0, category=StepCategory.EXECUTION_ERROR)
        checkpoint.add_step("Additional Recipe Sources Visited", False, 2, "No browsing history provided", execution_time=0, category=StepCategory.EXECUTION_ERROR)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Normalize browsing history for comparison
    normalized_history = set()
    for url in browsing_history:
        try:
            normalized_history.add(normalize_url_for_comparison(url))
        except Exception:
            pass  # Skip malformed URLs

    # =========================================================================
    # Step 4.1: First page tips source URLs visited (1 pt)
    # =========================================================================
    step_start = time.time()
    tips_visited = False
    tips_visit_details = "No tip source URLs found in first recipe"
    tips_visit_category = StepCategory.EXECUTION_ERROR  # default: no URLs available to check

    if recipes:
        first_recipe = recipes[0]
        try:
            recipe1_links = extract_hyperlinks(doc_id, DOCS_SERVICE, recipe_num=1, recipe_titles=[r.title for r in recipes])
        except Exception as e:
            print(f"Warning: Failed to extract hyperlinks for recipe 1: {e}")
            recipe1_links = []

        # Get tips section text to identify which links are in the tips section
        tips_text = extract_section_content(first_recipe.structure, "Tips")
        tip_source_urls = [
            link['url'] for link in recipe1_links
            if link['url'] in tips_text or link['text'] in tips_text
        ]

        if tip_source_urls:
            tips_visit_category = StepCategory.WEB_VISIT
            not_visited = [url for url in tip_source_urls
                           if normalize_url_for_comparison(url) not in normalized_history]

            if not not_visited:
                tips_visited = True
                tips_visit_details = f"All {len(tip_source_urls)} tip source URLs found in browsing history"
            else:
                tips_visit_details = f"Tip source URLs not in history: {[u[:50] for u in not_visited]}"

    step_time = time.time() - step_start
    checkpoint.add_step("Tips Source Visited", tips_visited, 1, tips_visit_details, execution_time=step_time, category=tips_visit_category)

    # =========================================================================
    # Step 4.2: Additional recipe source URLs visited (3 pts — 1 per recipe)
    # =========================================================================
    step_start = time.time()
    additional_recipes = recipes[1:] if recipes else []
    recipes_visited = 0
    visit_details_parts = []

    for recipe in additional_recipes[:3]:
        recipe_num = recipe.recipe_num
        try:
            recipe_links = extract_hyperlinks(doc_id, DOCS_SERVICE, recipe_num=recipe_num, recipe_titles=[r.title for r in recipes])
        except Exception as e:
            print(f"Warning: Failed to extract hyperlinks for recipe {recipe_num}: {e}")
            recipe_links = []

        source_urls = [link['url'] for link in recipe_links if link['url'].startswith('http')]

        found = False
        for url in source_urls:
            if normalize_url_for_comparison(url) in normalized_history:
                found = True
                visit_details_parts.append(f"Recipe {recipe_num}: visited ({url[:40]})")
                break

        if found:
            recipes_visited += 1
        else:
            if source_urls:
                visit_details_parts.append(f"Recipe {recipe_num}: not visited ({source_urls[0][:40]})")
            else:
                visit_details_parts.append(f"Recipe {recipe_num}: no source URL found")

    # Add entries for missing recipes
    for i in range(len(additional_recipes), 3):
        visit_details_parts.append(f"Recipe {i + 2}: not found in document")

    step_time = time.time() - step_start
    checkpoint.add_step(
        "Additional Recipe Sources Visited",
        recipes_visited >= 3, 2,
        f"{recipes_visited}/3 recipe sources visited. {'; '.join(visit_details_parts)}",
        score=recipes_visited, max_score=3,
        execution_time=step_time, category=StepCategory.WEB_VISIT
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def _all_zero_result(reason: str, category: str = StepCategory.EXECUTION_ERROR) -> Result:
    """Create a Result with all checkpoints scored at 0, with a failure reason.

    Args:
        reason (str): Failure reason recorded on every step.
        category (str): StepCategory for every step (evaluation could not run).
    """
    checkpoints = []

    cp1 = Checkpoint(total=10, result=0, name="Original Recipe Page")
    for step_id, name in enumerate(["Title Match", "Ingredients Match", "Preparation Steps Match",
                                     "Tips Relevance", "Tips URLs Valid", "Tips Are Direct Quotes",
                                     "Image from Original", "Image Properly Cropped",
                                     "Image Shows Recipe Text", "Info Modified from Defaults"], 1):
        cp1.add_step(name, False, step_id, reason, execution_time=0, category=category)
    checkpoints.append(cp1)

    cp2 = Checkpoint(total=32, result=0, name="Additional Recipe Pages Formatting")
    step_names_2 = ["Title Theme", "Source URL", "Format Match", "Ingredients Modified",
                    "Preparation Modified", "Tips Modified", "Info Modified", "Visual Distinction"]
    for i in range(4):
        for step_idx, name in enumerate(step_names_2):
            cp2.add_step(f"Recipe {i+2} - {name}", False, i * 8 + step_idx + 1, reason, execution_time=0, category=category)
    checkpoints.append(cp2)

    cp3 = Checkpoint(total=15, result=0, name="Additional Recipe Pages Content")
    step_names_3 = ["Photo from Source", "Ingredients Match", "Preparation Match",
                    "Tips Direct Quotes", "Info Relevant"]
    for i in range(3):
        for step_idx, name in enumerate(step_names_3):
            cp3.add_step(f"Recipe {i+2} - {name}", False, i * 5 + step_idx + 1, reason, execution_time=0, category=category)
    checkpoints.append(cp3)

    cp4 = Checkpoint(total=4, result=0, name="Websites Visited Check")
    cp4.add_step("Tips Source Visited", False, 1, reason, execution_time=0, category=category)
    cp4.add_step("Additional Recipe Sources Visited", False, 2, reason, score=0, max_score=3, execution_time=0, category=category)
    checkpoints.append(cp4)

    return Result(checkpoints)


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
    global doc_id, doc_text, doc_structure, recipes, DRIVE_SERVICE, DOCS_SERVICE

    total_start_time = time.time()

    try:
        # Initialize Google services (moved from import time to avoid crashes on missing credentials)
        DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

        # Setup document processing — fatal if this fails
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

        if not doc_text or not doc_structure:
            print("FATAL: Document text or structure could not be extracted")
            return _all_zero_result("Document text or structure could not be extracted")

        # Phase 1: Discover all recipes and their boundaries
        recipes = discover_recipes(doc_text, doc_structure)

        if not recipes:
            print("FATAL: No valid recipes found — template may not have been used")
            return _all_zero_result("No valid recipes found in document — template may not have been used")

        # Map recipes to PDF pages using actual page text
        pdf_path = os.path.join(DATA_DIR, "recipe_doc.pdf")
        map_recipes_to_pdf_pages(recipes, pdf_path)

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

        # Grade checkpoint 4
        checkpoints.append(grade_checkpoint_4(browsing_history))

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    except Exception as e:
        print(f"FATAL: Evaluator failed: {e}")
        return _all_zero_result(f"Evaluator failed: {str(e)[:100]}")

    finally:
        try:
            cleanup_generated_files(DATA_DIR, CLEANUP_ENABLED)
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed with error: {cleanup_error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate docs_11 recipe OCR task")
    parser.add_argument("--workspace_doc_id", type=str, required=True, help="Google Docs document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='*', help="List of URLs visited during task")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
    print(f"CLEANUP enabled: {CLEANUP_ENABLED}")

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history=args.browsing_history
    )

    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")

    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "+" if step["success"] else "x"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
