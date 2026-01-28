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
import shutil
import glob
from typing import List, Optional
from pathlib import Path

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
from src.browsergym.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    download_doc_as_pdf,
    extract_text_from_doc,
    extract_structure_from_doc,
    extract_images_from_doc,
    extract_images_from_doc_with_cropping,
)
from src.browsergym.eval.eval_utils.image_utils import (
    convert_pdf_to_pngs,
    binary_compare_images,
    match_image_tiered,
)
from src.browsergym.eval.eval_utils.text_utils import (
    extract_text_from_pdf,
    keyword_exact_match,
)
from src.browsergym.eval.eval_utils.web_utils import (
    validate_url_accessible,
    fetch_page_text_content,
)
from src.browsergym.eval.eval_utils.models import load_model

# Import task-specific utilities
from ..utils import (
    extract_hyperlinks_from_doc,
    extract_section_content,
    extract_list_items,
    compare_ingredient_lists,
    compare_preparation_steps,
    extract_recipe_metadata,
    check_metadata_modified,
    verify_tip_is_quote,
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
model_id = "gemma-google-ai"

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
GOLD_IMAGE_ORIGINAL = os.path.join(GOLDS_DIR, "original_image.avif")
GOLD_IMAGE_CROPPED = os.path.join(GOLDS_DIR, "original_image_cropped.png")


def cleanup_generated_files():
    """Clean up all generated files and directories created during evaluation."""
    if not CLEANUP_ENABLED:
        print("Cleanup disabled by CLEANUP=False environment variable")
        return

    print("Cleaning up generated files...")
    cleanup_dirs = [DOC_IMAGES_DIR, DOC_IMAGES_CROPPED_DIR, PDF_IMAGES_DIR]

    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

    # Clean up PDF files
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            os.remove(pdf_file)
            print(f"Removed PDF file: {pdf_file}")
        except Exception as e:
            print(f"Error removing PDF file {pdf_file}: {e}")

    print("Cleanup completed")


def setup_document(workspace_doc_id: str):
    """
    Setup document processing using the provided workspace_doc_id.

    Args:
        workspace_doc_id: The Google Docs document ID to evaluate.
    """
    global doc_id, doc_text, doc_structure, text_ocr

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace document ID: {workspace_doc_id}")
    doc_id = workspace_doc_id

    # Ensure data directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DOC_IMAGES_DIR, exist_ok=True)
    os.makedirs(DOC_IMAGES_CROPPED_DIR, exist_ok=True)
    os.makedirs(PDF_IMAGES_DIR, exist_ok=True)

    # Download and convert PDF
    pdf_path = os.path.join(DATA_DIR, "recipe_doc.pdf")
    download_doc_as_pdf(doc_id, pdf_path, DRIVE_SERVICE)
    convert_pdf_to_pngs(pdf_path, PDF_IMAGES_DIR, dpi=PDF_DPI)

    # Extract text and structure
    doc_text = extract_text_from_doc(doc_id, DOCS_SERVICE)
    doc_structure = extract_structure_from_doc(doc_id, DOCS_SERVICE)

    # Extract images
    extract_images_from_doc(doc_id, DOCS_SERVICE, DOC_IMAGES_DIR)
    extract_images_from_doc_with_cropping(doc_id, DOCS_SERVICE, DOC_IMAGES_CROPPED_DIR)

    # Run OCR on PDF images
    text_ocr = extract_text_from_pdf(PDF_IMAGES_DIR)


def load_gold_data():
    """Load gold standard data from files."""
    gold_ingredients = []
    gold_prepsteps = []

    if os.path.exists(GOLD_INGREDIENTS_FILE):
        with open(GOLD_INGREDIENTS_FILE, 'r') as f:
            gold_ingredients = [line.strip() for line in f if line.strip()]

    if os.path.exists(GOLD_PREPSTEPS_FILE):
        with open(GOLD_PREPSTEPS_FILE, 'r') as f:
            gold_prepsteps = [line.strip() for line in f if line.strip()]

    return gold_ingredients, gold_prepsteps


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
    """
    global model

    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=9, result=0, name="Original Recipe Page")

    # Load gold data
    gold_ingredients, gold_prepsteps = load_gold_data()

    # Step 1.1: Title Check
    step_start = time.time()
    title_found = keyword_exact_match(doc_text, GOLD_TITLE, case_sensitive=False)
    step_time = time.time() - step_start

    if title_found:
        checkpoint.add_step("Title Match", True, 1, f"Found '{GOLD_TITLE}' in document", execution_time=step_time)
    else:
        checkpoint.add_step("Title Match", False, 1, f"Title '{GOLD_TITLE}' not found in document", execution_time=step_time)

    # Step 1.2: Ingredients Match
    step_start = time.time()
    ingredients_text = extract_section_content(doc_structure, "Ingredients")
    doc_ingredients = extract_list_items(ingredients_text)
    ingredients_match, ingredients_details = compare_ingredient_lists(doc_ingredients, gold_ingredients)
    step_time = time.time() - step_start

    if ingredients_match:
        checkpoint.add_step("Ingredients Match", True, 2, f"All {len(gold_ingredients)} ingredients found", execution_time=step_time)
    else:
        missing = [d['gold'] for d in ingredients_details if not d['matched']]
        checkpoint.add_step("Ingredients Match", False, 2, f"Missing ingredients: {', '.join(missing)}", execution_time=step_time)

    # Step 1.3: Preparation Steps Match
    step_start = time.time()
    prep_text = extract_section_content(doc_structure, "Preparation")
    doc_steps = extract_list_items(prep_text)
    steps_match, steps_details = compare_preparation_steps(doc_steps, gold_prepsteps)
    step_time = time.time() - step_start

    if steps_match:
        checkpoint.add_step("Preparation Steps Match", True, 3, f"All {len(gold_prepsteps)} steps matched", execution_time=step_time)
    else:
        unmatched = [f"Step {d['step']} (score: {d['score']})" for d in steps_details if not d['matched']]
        checkpoint.add_step("Preparation Steps Match", False, 3, f"Unmatched steps: {', '.join(unmatched)}", execution_time=step_time)

    # Step 1.4: Tips Relevance Check
    step_start = time.time()
    tips_text = extract_section_content(doc_structure, "Tips")
    tips_list = extract_list_items(tips_text)

    if model is None:
        model = load_model(model_id)

    tips_relevant = True
    tips_relevance_details = []

    for tip in tips_list:
        if not tip or len(tip) < 10:  # Skip very short/empty tips
            continue

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are evaluating whether cooking tips are relevant to making pumpkin soup. Answer only 'Yes' or 'No'."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Is this tip relevant to making pumpkin soup?\n\nTip: {tip}"}]
            }
        ]

        response = model(messages)
        is_relevant = response.strip().lower().startswith('yes')
        tips_relevance_details.append({'tip': tip[:50], 'relevant': is_relevant})

        if not is_relevant:
            tips_relevant = False

    step_time = time.time() - step_start

    if tips_relevant and len(tips_relevance_details) > 0:
        checkpoint.add_step("Tips Relevance", True, 4, f"All {len(tips_relevance_details)} tips are relevant to pumpkin soup", execution_time=step_time)
    elif len(tips_relevance_details) == 0:
        checkpoint.add_step("Tips Relevance", False, 4, "No tips found in Tips section", execution_time=step_time)
    else:
        irrelevant_count = sum(1 for t in tips_relevance_details if not t['relevant'])
        checkpoint.add_step("Tips Relevance", False, 4, f"{irrelevant_count} tips not relevant to pumpkin soup", execution_time=step_time)

    # Step 1.5: Tips URLs Valid
    step_start = time.time()
    all_links = extract_hyperlinks_from_doc(doc_id, DOCS_SERVICE)

    # Filter to links that appear after "Tips" section (simplified: just check all links for now)
    tips_urls = [link['url'] for link in all_links if link['url'].startswith('http')]

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
        checkpoint.add_step("Tips URLs Valid", True, 5, f"All {len(tips_urls)} URLs are accessible", execution_time=step_time)
    elif len(tips_urls) == 0:
        checkpoint.add_step("Tips URLs Valid", False, 5, "No URLs found in document", execution_time=step_time)
    else:
        checkpoint.add_step("Tips URLs Valid", False, 5, f"{len(invalid_urls)} URLs not accessible: {invalid_urls[:2]}", execution_time=step_time)

    # Step 1.6: Tips are Direct Quotes
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

    # Step 1.7: Image from Original
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

    # Step 1.8: Image Properly Cropped
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

    # Step 1.9: Info Modified from Defaults
    step_start = time.time()
    metadata = extract_recipe_metadata(doc_text)
    is_modified, modification_details = check_metadata_modified(metadata, TEMPLATE_DEFAULTS)
    step_time = time.time() - step_start

    checkpoint.add_step("Info Modified from Defaults", is_modified, 9, modification_details, execution_time=step_time)

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
    total_start_time = time.time()

    try:
        # Setup document processing
        setup_document(workspace_doc_id)

        # Use cached model if available
        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        checkpoints: List[Checkpoint] = []

        # Grade checkpoint 1
        checkpoints.append(grade_checkpoint_1())

        # TODO: Add checkpoints 2, 3, 4

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    finally:
        try:
            cleanup_generated_files()
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
