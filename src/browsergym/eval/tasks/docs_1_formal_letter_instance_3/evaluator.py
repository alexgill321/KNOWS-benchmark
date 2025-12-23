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
    return "/scratch/general/vast/USER/Agent-Benchmark/"
  else:
    return os.getcwd()
  
BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result # type: ignore
from src.browsergym.eval.eval_utils.google_services_utils import * # type: ignore
from src.browsergym.eval.eval_utils.text_utils import extract_text_from_pdf, text_exact_match_contained, extract_text_location # type: ignore
from src.browsergym.eval.eval_utils.image_utils import * # type: ignore
from src.browsergym.eval.eval_utils.utils import layout, image_id_from_path # type: ignore
from src.browsergym.eval.eval_utils.models import load_model # type: ignore

TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_1_formal_letter_instance_3/")
DOC_IMAGES_DIR = os.path.join(TASK_DIR, "data/images/")
DOC_IMAGES_CROPPED_DIR = os.path.join(TASK_DIR, "data/cropped_images/")
PDF_IMAGES_DIR =  os.path.join(TASK_DIR, "data/pdf_images/")
GOLD_IMAGES_DIR = os.path.join(TASK_DIR, "data/gold_images/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_DPI = 150  # Lower DPI for faster OCR while maintaining text recognition quality

model = None
model_id = "gemma-google-ai" # Using AI API instead of cloud service

DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

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

  # Clean up individual PDF files inthe data directory
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
  global doc_id, gold_text, text_ocr, doc_structure

  if not workspace_doc_id:
    raise ValueError("workspace_doc_id is required")
  
  print(f"Using workspace document ID: {workspace_doc_id}")
  doc_id = workspace_doc_id

  # Phase 1: Download and convert PDF (sequential, required order)
  pdf_path = os.path.join(TASK_DIR, "data/elisa_kreiss_formal_letter.pdf")
  download_doc_as_pdf(doc_id, pdf_path, DRIVE_SERVICE)
  convert_pdf_to_pngs(pdf_path, PDF_IMAGES_DIR, dpi=PDF_DPI)

  # Phase 2: Run OCR and Google API calls in parallel
  # OCR doesn't depend on Google APIs, so we can run them concurrently
  def run_ocr():
    return extract_text_from_pdf(PDF_IMAGES_DIR)
  
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

### Checkpoint 1 ###
def grade_checkpoint_1(gold_text, text_ocr):
  print("----------------- CHECKPOINT 1 ----------------")
  checkpoint_start = time.time()
  checkpoint = Checkpoint(total=8, result=0, name="Contact Information")

  name = "Elisa Kreiss"
  titles = ["Lab Director", "Assistant Professor"]
  institutions = ["Coalas Lab", "Communication at UCLA"]
  email = "ekreiss@ucla.edu"

  # Name evaluation
  step_start = time.time()
  name_found = text_exact_match_contained(name, gold_text)
  step_time = time.time() - step_start

  if name_found:
    print("Name match successful")
    checkpoint.add_step("Name Text Match", True, 1, f"Found '{name}' in document", execution_time=step_time)

    step_start = time.time()
    location = extract_text_location(text_ocr, name)
    step_time = time.time() - step_start

    if location.is_upper_left():
      checkpoint.add_step("Name Location", True, 2, f"Name correctly positioned in upper left at {location}", execution_time=step_time)
    else:
      print("Name location failed")
      checkpoint.add_step("Name Location", False, 2, f"Name not in upper left, found at {location}", execution_time=step_time)
  else:
    print("Name match failed")
    checkpoint.add_step("Name Text Match", False, 1, f"'{name}' not found in document", execution_time=step_time)
    checkpoint.add_step("Name Location", False, 2, "Cannot check location - name not found")

  # Title evaluation
  step_start = time.time()
  institution_found = any(text_exact_match_contained(t, gold_text) for t in titles)
  institution_match = [t for t in titles if text_exact_match_contained(t, gold_text)]
  step_time = time.time() - step_start
  if institution_found:
      print("Title match successful")
      checkpoint.add_step("Title Text Match", True, 3, f"Found title in document", execution_time=step_time)

      step_start = time.time()
      location = extract_text_location(text_ocr, institution_match[0])
      step_time = time.time() - step_start

      if location.is_upper_left():
          checkpoint.add_step("Title Location", True, 4, f"Title correctly positioned in upper left at {location}", execution_time=step_time)
      else:
          print("Title location failed")
          checkpoint.add_step("Title Location", False, 4, f"Title not in upper left, found at {location}", execution_time=step_time)
  else:
      print("Title match failed")
      checkpoint.add_step("Title Text Match", False, 3, f"Title not found in document", execution_time=step_time)
      checkpoint.add_step("Title Location", False, 4, "Cannot check location - title not found")  

  # Institution evaluation
  step_start = time.time()
  institution_found = any(text_exact_match_contained(t, gold_text) for t in institutions)
  institution_match = [t for t in institutions if text_exact_match_contained(t, gold_text)]
  step_time = time.time() - step_start
  if institution_found:
      print("Institution match successful")
      checkpoint.add_step("Institution Text Match", True, 5, f"Found institution in document", execution_time=step_time)

      step_start = time.time()
      location = extract_text_location(text_ocr, institution_match[0])
      step_time = time.time() - step_start

      if location.is_upper_left():
          checkpoint.add_step("Institution Location", True, 6, f"Institution correctly positioned in upper left at {location}", execution_time=step_time)
      else:
          print("Institution location failed")
          checkpoint.add_step("Institution Location", False, 6, f"Institution not in upper left, found at {location}", execution_time=step_time)
  else:
      print("Institution match failed")
      checkpoint.add_step("Institution Text Match", False, 5, f"Institution not found in document", execution_time=step_time)
      checkpoint.add_step("Institution Location", False, 6, "Cannot check location - Institution not found")  

  # Email evaluation
  step_start = time.time()
  email_found = text_exact_match_contained(email, gold_text)
  step_time = time.time() - step_start
  
  if email_found:
      print("Email match successful")
      checkpoint.add_step("Email Text Match", True, 7, f"Found '{email}' in document", execution_time=step_time)

      step_start = time.time()
      location = extract_text_location(text_ocr, email)
      step_time = time.time() - step_start

      if location is not None:
          if location.is_upper_left():
              checkpoint.add_step("Email Location", True, 8, f"Email correctly positioned in upper left at {location}", execution_time=step_time)
          else:
              print("Email location failed")
              checkpoint.add_step("Email Location", False, 8, f"Email not in upper left, found at {location}", execution_time=step_time)
      else:
          print("Email location extraction failed")
          checkpoint.add_step("Email Location", False, 7, "Could not extract email location from OCR text", execution_time=step_time)
  else:
      print("Email match failed")
      checkpoint.add_step("Email Text Match", False, 7, f"'{email}' not found in document", execution_time=step_time)
      checkpoint.add_step("Email Location", False, 8, "Cannot check location - email not found")

  checkpoint.execution_time = time.time() - checkpoint_start
  return checkpoint

def grade_checkpoint_2():
  print("---------------- CHECKPOINT 2 ----------------")
  checkpoint_start = time.time()
  checkpoint = Checkpoint(total=2, result=0, name="Logo Image")

  step_start = time.time()
  logo_path = image_exact_match(DOC_IMAGES_DIR, GOLD_IMAGES_DIR + "logos/")
  step_time = time.time() - step_start

  if logo_path:
    cropped_logo_path = logo_path.replace("images", "cropped_images")
    logo_uri = logo_path.split("_")[-1].replace(".png", "")
    logo_size = get_image_dimensions_from_doc(doc_id, logo_uri, DOCS_SERVICE)
    print("Image match successful\n")
    checkpoint.add_step("Logo Image Match", True, 9, f"Exact logo match found at {logo_path}", execution_time=step_time)

    print("Locating Logo Image")
    step_start = time.time()
    exact_size_location = extract_image_location_size_feature_based(cropped_logo_path, logo_size, PDF_IMAGES_DIR, True, dpi=PDF_DPI)
    step_time = time.time() - step_start
    print(f"Location is {exact_size_location}")

    if exact_size_location.is_upper_left():
       print("Image location match successful")
       checkpoint.add_step("Logo Location", True, 10, f"Logo correctly positioned in upper left at {exact_size_location}", execution_time=step_time)
    else:
       print("Image location exact match failed")
       checkpoint.add_step("Logo Location", False, 10, f"Logo not in upper left, found at {exact_size_location}", execution_time=step_time)
  else:
    # Try AI-based detection as fallbac
    global model
    if model is None:
        model = load_model(model_id)

    step_start = time.time()
    logo_path = binary_judge_image(model, DOC_IMAGES_DIR, "Is this an image of ONLY one of these institutions' logos: Coalas Lab, Communication at UCLA", GOLD_IMAGES_DIR + "logos/")
    step_time = time.time() - step_start
    
    if logo_path:
      cropped_logo_path = logo_path.replace("images", "cropped_images")
      cropped_logo_uri = cropped_logo_path.split("_")[-1].replace(".png", "")
      logo_size = get_image_dimensions_from_doc(doc_id, cropped_logo_uri, DOCS_SERVICE)
      print("Image match successful\n")
      checkpoint.add_step("Logo Image Match", True, 9, f"Exact logo match found at {logo_path}", execution_time=step_time)

      print("Locating Logo Image")
      step_start = time.time()
      exact_size_location = extract_image_location_size_feature_based(cropped_logo_path, logo_size, PDF_IMAGES_DIR, True, dpi=PDF_DPI)
      step_time = time.time() - step_start
      print(f"Location is {exact_size_location}")
      if exact_size_location:
          if exact_size_location.is_upper_left():
              print("Image location match successful")
              checkpoint.add_step("Logo Location", True, 10, f"Logo correctly positioned in upper left at {exact_size_location}", execution_time=step_time)
          else:
              print("Image location exact match failed")
              checkpoint.add_step("Logo Location", False, 10, f"Logo not in upper left, found at {exact_size_location}", execution_time=step_time)
      else:
          print("Image location extraction failed")
          checkpoint.add_step("Logo Location", False, 10, "Could not extract logo location from PDF images", execution_time=step_time)
    else:
        print("Image match failed")
        checkpoint.add_step("Logo Image Match", False, 9, "No logo found via exact match or AI detection", execution_time=step_time)
        checkpoint.add_step("Logo Location", False, 10, "Cannot check location - logo not found")

  checkpoint.execution_time = time.time() - checkpoint_start
  return checkpoint

