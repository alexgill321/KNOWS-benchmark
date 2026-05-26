import os
from typing import List
import sys
import time
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor

def get_base_path():
  # First check if we're in a Docker container at /app
  if os.path.exists("/app/src"):
    return "/app"
  # Otherwise use current working directory
  elif os.path.exists("/scratch"):
    return "/path/to/KNOWS-benchmark/"
  else:
    return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result # type: ignore
from src.browsergym.knows.eval.eval_utils.google_services_utils import * # type: ignore
from src.browsergym.knows.eval.eval_utils.text_utils import extract_text_from_pdf, keyword_exact_match, extract_text_location, get_smallest_x_position, find_gold_text_location, strip_label_prefix # type: ignore
from src.browsergym.knows.eval.eval_utils.image_utils import * # type: ignore
from src.browsergym.knows.eval.eval_utils.utils import layout, image_id_from_path, location as Location # type: ignore
from src.browsergym.knows.eval.eval_utils.models import load_model # type: ignore

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_IMAGES_DIR = os.path.join(TASK_DIR, "data/images/")
DOC_IMAGES_CROPPED_DIR = os.path.join(TASK_DIR, "data/cropped_images/")
PDF_IMAGES_DIR =  os.path.join(TASK_DIR, "data/pdf_images/")
GOLD_IMAGES_DIR = os.path.join(TASK_DIR, "data/gold_images/")
GOLD_SIGNATURE_PATH = os.path.join(TASK_DIR, "data/gold_signature.png")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_DPI = 300

model = None
model_id = "gemini-2.5-flash-google-ai" # Using AI API instead of cloud service

DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services(service_type="docs")

# Global variables that will be set by setup_document
doc_id = None
gold_text = None
text_ocr = None
doc_structure = None

def cleanup_generated_files():
  """
  Clean up all generated files and directories created during evaluation.
  This includes images, PDFs, and any temporary files.
  """
  if not CLEANUP_ENABLED:
    print("Cleanup disabled by CLEANUP=False environment variable")
    return

  print("Cleaning up generated files...")
  cleanup_dirs = [
    DOC_IMAGES_DIR,
    DOC_IMAGES_CROPPED_DIR,
    PDF_IMAGES_DIR
  ]

  # Clean up directories
  for dir_path in cleanup_dirs:
    if os.path.exists(dir_path):
      try:
        shutil.rmtree(dir_path)
        print(f"Removed directory: {dir_path}")
      except Exception as e:
        print(f"Error removing directory {dir_path}: {e}")

  # Clean up individual PDF files in the data directory
  pdf_files = glob.glob(os.path.join(TASK_DIR, "data", "*.pdf"))
  for pdf_file in pdf_files:
    try:
      os.remove(pdf_file)
      print(f"Removed PDF file: {pdf_file}")
    except Exception as e:
      print(f"Error removing PDF file {pdf_file}: {e}")

  # Clean up any other temporary files that might be created
  temp_patterns = [
    os.path.join(TASK_DIR, "data", "*.tmp"),
    os.path.join(TASK_DIR, "data", "*.temp"),
    os.path.join(TASK_DIR, "data", "*.log")
  ]

  for pattern in temp_patterns:
    temp_files = glob.glob(pattern)
    for temp_file in temp_files:
      try:
        os.remove(temp_file)
        print(f"Removed temporary file: {temp_file}")
      except Exception as e:
        print(f"Error removing temporary file {temp_file}: {e}")

  print("Cleanup completed")

def setup_document(workspace_doc_id):
  """
  Setup document processing using the provided workspace_doc_id.

  Args:
    workspace_doc_id (str): The Google Docs document ID (gold instance ID) to use
  """
  global doc_id, gold_text, text_ocr, doc_structure, smallest_x

  if not workspace_doc_id:
    raise ValueError("workspace_doc_id is required")

  print(f"Using workspace document ID: {workspace_doc_id}")
  doc_id = workspace_doc_id

  # Phase 1: Download and convert PDF (sequential, required order)
  pdf_path = os.path.join(TASK_DIR, "data/martin_tutek_formal_letter.pdf")
  download_doc_as_pdf(doc_id, pdf_path, DRIVE_SERVICE)
  convert_pdf_to_pngs(pdf_path, PDF_IMAGES_DIR, dpi=PDF_DPI)

  # Phase 2: Run OCR and Google API calls in parallel
  # OCR doesn't depend on Google APIs, so we can run them concurrently
  def run_ocr():
    try:
      return extract_text_from_pdf(PDF_IMAGES_DIR)
    except Exception as e:
      print(f"OCR extraction failed: {e}")
      return {}

  def run_google_api_calls():
    # Keep Google API calls sequential to avoid rate limits
    extract_images_from_doc_with_cropping(doc_id, DOCS_SERVICE, DOC_IMAGES_CROPPED_DIR)
    extract_images_from_doc(doc_id, DOCS_SERVICE, DOC_IMAGES_DIR)
    text = extract_text_from_doc(doc_id, DOCS_SERVICE)
    structure = extract_structure_from_doc(doc_id, DOCS_SERVICE)
    return text, structure

  with ThreadPoolExecutor(max_workers=2) as executor:
    ocr_future = executor.submit(run_ocr)
    api_future = executor.submit(run_google_api_calls)

    text_ocr = ocr_future.result()
    gold_text, doc_structure = api_future.result()
    smallest_x = get_smallest_x_position(text_ocr) if text_ocr else float('inf')
    if smallest_x == float('inf'):
      smallest_x = 0

def is_first_page_text(loc) -> bool:
    return loc is not None and getattr(loc, "page_number", None) == 0

def is_first_page_img(loc) -> bool:
    return loc is not None and getattr(loc, "page_number", None) == 0


### Checkpoint 1 ###
def _check_field(field_name, search_text, gold_text, text_ocr, doc_structure, smallest_x,
                  text_step, loc_step, checkpoint, standalone_line=True, substring=False,
                  strip_prefix=False):
    """Standardized check for a contact info field: text match + location.

    Uses gold_text for text matching and OCR (with fuzzy fallback) for location.
    Text match and location are reported as independent steps.
    """
    stripped_gold = strip_label_prefix(gold_text) if strip_prefix else gold_text

    # Text match against gold_text
    step_start = time.time()
    if isinstance(search_text, list):
        found_text = None
        for t in search_text:
            if keyword_exact_match(stripped_gold, t, standalone_line=standalone_line, substring=substring):
                found_text = t
                break
    else:
        found_text = search_text if keyword_exact_match(stripped_gold, search_text, standalone_line=standalone_line, substring=substring) else None
    step_time = time.time() - step_start

    if found_text:
        checkpoint.add_step(f"{field_name} Text Match", True, text_step, f"Found '{found_text}' in document", execution_time=step_time)
    else:
        checkpoint.add_step(f"{field_name} Text Match", False, text_step, f"{field_name} not found in document", execution_time=step_time)
        checkpoint.add_step(f"{field_name} Location", False, loc_step, f"Cannot check location - {field_name.lower()} not found")
        return

    # Location via OCR with fuzzy fallback
    step_start = time.time()
    location = extract_text_location(text_ocr, found_text)
    if location is None:
        location = find_gold_text_location(found_text, doc_structure, text_ocr)
    step_time = time.time() - step_start

    location_desc = location.describe(dpi=PDF_DPI) if location else 'not found in OCR'

    # Use the containing line's x position for alignment (handles labels/substrings)
    line_x = location.x if location else None
    if location:
        for item in (text_ocr or {}).get(location.page_number, []):
            item_loc = item.get("location", None)
            if item_loc and abs(item_loc.y - location.y) < 5:
                line_x = item_loc.x
                break

    if location and location.is_upper_left() and line_x is not None and int(line_x) < int(smallest_x) + 20:
        checkpoint.add_step(f"{field_name} Location", True, loc_step, f"{field_name} correctly positioned in upper left ({location_desc})", execution_time=step_time)
    else:
        checkpoint.add_step(f"{field_name} Location", False, loc_step, f"{field_name} not in upper left ({location_desc})", execution_time=step_time)

def grade_checkpoint_1(gold_text, text_ocr, doc_structure):
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=8, result=0, name="Contact Information")

    name = "Martin Tutek"
    email = "martin.tutek@gmail.com"
    github_urls = ["https://github.com/mttk"]
    linkedin_urls = ["https://www.linkedin.com/in/mtutek/"]

    _check_field("Name", name, gold_text, text_ocr, doc_structure, smallest_x,
                 text_step=1, loc_step=2, checkpoint=checkpoint, standalone_line=True)
    _check_field("Email", email, gold_text, text_ocr, doc_structure, smallest_x,
                 text_step=3, loc_step=4, checkpoint=checkpoint, standalone_line=True, strip_prefix=True)
    _check_field("Github", github_urls, gold_text, text_ocr, doc_structure, smallest_x,
                 text_step=5, loc_step=6, checkpoint=checkpoint, standalone_line=True, strip_prefix=True)
    _check_field("LinkedIn", linkedin_urls, gold_text, text_ocr, doc_structure, smallest_x,
                 text_step=7, loc_step=8, checkpoint=checkpoint, standalone_line=True, strip_prefix=True)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_2():
    print("---------------- CHECKPOINT 2 ----------------")
    global model
    if model is None:
        model = load_model(model_id)
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Logo Image")

    # Find logo using tiered matching: exact -> perceptual hash -> VLM
    step_start = time.time()
    logo_match_path = None
    logo_match_method = None
    gold_images = [os.path.join(GOLD_IMAGES_DIR, f) for f in os.listdir(GOLD_IMAGES_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))] if os.path.isdir(GOLD_IMAGES_DIR) else []
    doc_images = [os.path.join(DOC_IMAGES_CROPPED_DIR, f) for f in os.listdir(DOC_IMAGES_CROPPED_DIR)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))] if os.path.isdir(DOC_IMAGES_CROPPED_DIR) else []

    for doc_img in doc_images:
        for gold_img in gold_images:
            try:
                match_result, match_method = match_image_tiered(doc_img, gold_img, model)
                if match_result:
                    logo_match_path = doc_img
                    logo_match_method = match_method
                    break
            except Exception as e:
                print(f"Error matching {doc_img} vs {gold_img}: {e}")
        if logo_match_path:
            break
    step_time = time.time() - step_start

    if logo_match_path:
        print(f"Logo match found via {logo_match_method}")
        checkpoint.add_step("Logo Image Match", True, 9, f"Logo matched via {logo_match_method} at {os.path.basename(logo_match_path)}", execution_time=step_time)

        print("Checking Logo Location via VLM")
        step_start = time.time()
        logo_in_region = verify_image_in_region(model, PDF_IMAGES_DIR, logo_match_path, region="upper_left", dpi=PDF_DPI)
        step_time = time.time() - step_start
        if logo_in_region:
            checkpoint.add_step("Logo Location", True, 10, "Logo verified in upper left region by VLM", execution_time=step_time)
        else:
            checkpoint.add_step("Logo Location", False, 10, "Logo not found in upper left region by VLM", execution_time=step_time)
    else:
        print("No logo match found")
        checkpoint.add_step("Logo Image Match", False, 9, "No logo found via exact, perceptual hash, or VLM matching", execution_time=step_time)
        checkpoint.add_step("Logo Location", False, 10, "Cannot check location - logo not found")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_3(doc_structure):
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Signature Image")

    step_start = time.time()
    try:
        signature_path = image_exact_match(DOC_IMAGES_DIR, GOLD_SIGNATURE_PATH)
    except FileNotFoundError:
        print("No images found in document - scoring signature as 0")
        signature_path = None
    step_time = time.time() - step_start

    if signature_path:
        cropped_signature_path = signature_path.replace("images", "cropped_images")
        signature_uri = signature_path.split("_")[-1].replace(".png", "")
        signature_size = get_image_dimensions_from_doc(doc_id, signature_uri, DOCS_SERVICE)
        print("Image match successful")
        checkpoint.add_step("Signature Image Match", True, 11, f"Signature found at {signature_path}", execution_time=step_time)

        print("Locating Signature Image")
        step_start = time.time()
        location = extract_image_location_size_feature_based(cropped_signature_path, signature_size, PDF_IMAGES_DIR, DEBUG, dpi=PDF_DPI)
        print(f"Signature Location: {location}")

        location_success = False
        location_details = ""

        # Scale is_lower() threshold for actual PDF DPI (default assumes 300 DPI)
        scale = PDF_DPI / 300
        lower_region = Location(location.page_number, 0, int(2200 * scale), int(2550 * scale), int(1100 * scale)) if location else None
        if location and location.is_mostly_inside(lower_region):
            location_success = True
            location_details = f"Signature correctly positioned in lower section ({location.describe(dpi=PDF_DPI)})"
        else:
            print("Signature location exact match failed")
            # Try structural location as fallback
            image_id = image_id_from_path(cropped_signature_path)
            image_layout = layout(image_id, "image", doc_structure)
            if image_layout.at_end():
                print("Signature structured location successful")
                location_success = True
                location_details = f"Signature found at end of document structure (fallback from pixel location at {location.describe(dpi=PDF_DPI) if location else 'not found'})"
            else:
                print("Signature location failed")
                location_details = f"Signature not in lower section ({location.describe(dpi=PDF_DPI) if location else 'not found'}) and not at end of document structure"

        step_time = time.time() - step_start
        checkpoint.add_step("Signature Location", location_success, 12, location_details, execution_time=step_time)
    else:
        print("Image match failed")
        checkpoint.add_step("Signature Image Match", False, 11, "No signature image found", execution_time=step_time)
        checkpoint.add_step("Signature Location", False, 12, "Cannot check location - signature not found")

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoints(workspace_doc_id, cached_models=None):
    """
    Grade all checkpoints for the document.

    Args:
        workspace_doc_id (str): The Google Docs document ID (gold instance ID) to use
        cached_models (dict, optional): Dictionary of preloaded models by model_id

    Returns:
        Result: Evaluation results with checkpoint scores
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

        checkpoints.append(grade_checkpoint_1(gold_text, text_ocr, doc_structure))
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3(doc_structure))

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    finally:
        # Always attempt cleanup, even if evaluation failed
        try:
            cleanup_generated_files()
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed with error: {cleanup_error}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate formal letter document")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Docs document ID to evaluate")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    args = parser.parse_args()

    start_time = time.time()

    # Add environment variable info for debugging
    print(f"DEBUG mode: {DEBUG}")
    print(f"CLEANUP enabled: {CLEANUP_ENABLED}")

    result = grade_checkpoints(workspace_doc_id=args.workspace_doc_id, cached_models=args.cached_models)
    print("=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()
    for checkpoint in detailed_report["checkpoints"]:
        print(f"\n{checkpoint['name']}: {checkpoint['score']}")
        for step in checkpoint["steps"]:
            status = "PASS" if step["success"] else "FAIL"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")
    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
