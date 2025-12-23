import os
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

TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_1_formal_letter_instance_3/")
DOC_IMAGES_DIR = os.path.join(TASK_DIR, "data/images/")
DOC_IMAGES_CROPPED_DIR = os.path.join(TASK_DIR, "data/cropped_images/")
PDF_IMAGES_DIR =  os.path.join(TASK_DIR, "data/pdf_images/")

CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_DPI = 150  # Lower DPI for faster OCR while maintaining text recognition quality

DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

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
  checkpoint = Checkpoint(total=8)
  