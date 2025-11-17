import os
from typing import List
import sys
import time
import shutil
import glob
from pathlib import Path
import re
from urllib.parse import urlparse
from paperscraper.citations import get_citations_by_doi
import arxiv


# Base path setup (same pattern as other evaluators)
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Imports
from src.browsergym.eval.eval_scripts.test.test_doc_to_images import convert_pdf_to_pngs
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result, EvaluationStep
from src.browsergym.eval.eval_utils.google_services_utils import *
from src.browsergym.eval.eval_utils.text_utils import extract_text_from_pdf, text_fuzzy_match_contained, extract_text_location
from src.browsergym.eval.eval_utils.models import load_model
from .utils import extract_arxiv_links_from_text, normalize_arxiv_url, match_document_links_with_browsing_history

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_5_influential_papers_instance_1/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_IMAGES_DIR = os.path.join(TASK_DIR, "data/pdf_images/")

# Model setup
model = None
model_id = "gemma-3-27b-it"

# Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

# Global variables for document processing
doc_id = None
gold_text = None
doc_structure = None
browsing_history = None  # Track visited websites

def cleanup_generated_files():
    """Clean up generated files and directories created during evaluation."""
    # Similar to other evaluators - clean PDFs, images, temp files
    pass

def setup_document(browsing_history, workspace_doc_id):
    """
    Setup document processing and browsing history analysis.
    
    Args:
        workspace_doc_id (str): Direct Google Docs document ID to use
        browsing_history (list): List of URLs/pages visited during task
    """
    global doc_id, gold_text, doc_structure, text_ocr

    doc_id = workspace_doc_id

    # Download and process the document
    pdf_path = os.path.join(TASK_DIR, "data/ethan_ashby_formal_letter.pdf")
    download_doc_as_pdf(doc_id, pdf_path, DRIVE_SERVICE)
    convert_pdf_to_pngs(pdf_path, PDF_IMAGES_DIR)
    # Extract document content
    gold_text = extract_text_from_doc(doc_id, DOCS_SERVICE)
    doc_structure = extract_structure_from_doc(doc_id, DOCS_SERVICE)
    text_ocr = extract_text_from_pdf(PDF_IMAGES_DIR)
    # Store browsing history for website access validation
    browsing_history = browsing_history or []

def grade_checkpoint_1(gold_text, browsing_history):
    """
    Checkpoint 1 (2pt): 5 different papers were looked at and placed into the google docs.
    
    Outcome Evaluation:
    - Paper website was accessed at least 5 times for different papers.
    - Links in the google docs lines up with the pages that were accessed.
    """
    print("----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Paper Research and Documentation")

    # Step 1: Check if paper websites were accessed (analyze browsing history)
    step_start = time.time()
    arxiv_visits = [url for url in browsing_history if 'arxiv.org' in url]
    unique_papers_visited = len(set(arxiv_visits))  # Count unique arxiv papers visited
    step_time = time.time() - step_start

    if unique_papers_visited >= 5:
        checkpoint.result += 1
        checkpoint.add_step("Paper Website Access", True, 1,
                        f"Accessed {unique_papers_visited} different paper websites",
                        execution_time=step_time)
    else:
        checkpoint.add_step("Paper Website Access", False, 1,
                        f"Only accessed {unique_papers_visited} paper websites, need 5",
                        execution_time=step_time)

    # Step 2: Check if links in document match visited pages
    step_start = time.time()

    # Use helper function to match document links with browsing history
    links_match, doc_paper_ids, visited_paper_ids, matched_count = match_document_links_with_browsing_history(gold_text, browsing_history)

    step_time = time.time() - step_start

    if links_match:
        checkpoint.result += 1
        checkpoint.add_step("Document Links Match", True, 2,
                        f"Document links match browsing history: {matched_count} papers matched out of {len(doc_paper_ids)} in document",
                        execution_time=step_time)
    else:
        detail_msg = f"Links don't match: {matched_count} papers matched out of {len(doc_paper_ids)} in document. "
        detail_msg += f"Document has {len(doc_paper_ids)} arxiv links, visited {len(visited_paper_ids)} unique papers"
        checkpoint.add_step("Document Links Match", False, 2,
                        detail_msg,
                        execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_2(gold_text):
    """
    Checkpoint 2 (2pt): The papers meet the requirements for citation counts and recency.
    
    Outcome Evaluation:
    - Each paper has at least 50 citations.
    - All papers are from the last 3 years.
    """
    print("----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Paper Requirements Validation")

    # Step 1: Check citation counts (would need to parse paper metadata or use AI)
    step_start = time.time()
    arxiv_ids = extract_arxiv_links_from_text(gold_text)
    arxiv_ids = [normalize_arxiv_url(url) for url in arxiv_ids if normalize_arxiv_url(url)]
    arxiv_ids = list(set(arxiv_ids))  # Unique IDs
    if len(arxiv_ids) < 5:
        print(f"Warning: Only found {len(arxiv_ids)} unique arxiv paper links in document, expected at least 5.")

    if not arxiv_ids:
        print("Error: No valid arxiv paper links found.")
        return checkpoint

    papers_info = arxiv.Search(id_list=arxiv_ids).results()


    has_count = 0
    for paper in papers_info:
        try:
            cite_history = get_citations_by_doi(paper.doi)
        except Exception as e:
            print(f"Error fetching citation history for {paper.title}: {e}")
            continue

        if not cite_history:
            print(f"No citation history found for {paper.title}")
            continue
        total_citations = sum(cite_history.values())
        if total_citations < 50:
            checkpoint.add_step("Citation Count Check", False, 1,
                            f"Paper {paper.title} has only {total_citations} citations, needs at least 50",
                            execution_time=time.time() - step_start)
        else:
            has_count += 1
    if has_count == 5:
        checkpoint.add_step("Citation Count Check", True, 1,
                        "All papers have at least 50 citations",
                        execution_time=time.time() - step_start)

    # Step 2: Check recency (last 3 years)
    step_start = time.time()
    for paper in papers_info:
        date = paper.published
        if (time.time() - date.timestamp()) > (3 * 365 * 24)* 3600:
            checkpoint.add_step("Recency Check", False, 2,
                            f"Paper {paper.title} published on {date.date()} is older than 3 years",
                            execution_time=time.time() - step_start)
            break
    else:
        recency_valid = True

    step_time = time.time() - step_start

    if recency_valid:
        checkpoint.result += 1
        checkpoint.add_step("Recency Check", True, 2,
                        "All papers are from the last 3 years",
                        execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_3(gold_text, doc_structure):
    """
    Checkpoint 3 (4pt): The doc structure for each paper is correct.
    
    Outcome Evaluation:
    - Each paper abstract is included in the google docs.
    - Abstracts are placed below the paper titles and links.
    - Each paper title is included in the google docs.
    - Each paper link is placed below the paper title.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=4, result=0, name="Document Structure Validation")

    # Step 1: Check if abstracts are included
    step_start = time.time()
    
    paper_links = extract_arxiv_links_from_text(gold_text)
    paper_links = [normalize_arxiv_url(url) for url in paper_links if normalize_arxiv_url(url)]
    paper_links = list(set(paper_links))  # Unique IDs
    if len(paper_links) < 5:
        print(f"Warning: Only found {len(paper_links)} unique arxiv paper links in document, expected at least 5.")

    papers_info = arxiv.Search(id_list=paper_links).results()

    for paper in papers_info:
        abstract = paper.summary
        abstract_match = text_fuzzy_match_contained(abstract, gold_text)
        title_match = text_fuzzy_match_contained(paper.title, gold_text)
        if abstract_match:
            abstracts_found += 1
            abstract_location = extract_text_location(abstract_match, doc_structure)
            title_location = extract_text_location(title_match, doc_structure)
            
        else:
            checkpoint.add_step("Abstract Inclusion", False, 1,
                            f"Abstract for paper {paper.title} not found in document",
                            execution_time=time.time() - step_start)

    step_time = time.time() - step_start

    if abstracts_found >= 5:
        checkpoint.result += 1
        checkpoint.add_step("Abstract Inclusion", True, 1,
                        f"Found {abstracts_found} abstracts in document",
                        execution_time=step_time)

    # Step 2: Check abstract placement (below titles and links)
    step_start = time.time()
    
    structure_correct = False  # Placeholder
    step_time = time.time() - step_start

    if structure_correct:
        checkpoint.result += 1
        checkpoint.add_step("Abstract Placement", True, 2,
                        "Abstracts correctly placed below titles and links",
                        execution_time=step_time)
    else:
        checkpoint.add_step("Abstract Placement", False, 2,
                        "Abstract placement structure incorrect",
                        execution_time=step_time)

    # Step 3: Check link placement (below titles)
    step_start = time.time()
    # TODO: Verify links are positioned correctly below paper titles
    links_positioned_correctly = False  # Placeholder
    step_time = time.time() - step_start

    if links_positioned_correctly:
        checkpoint.result += 1
        checkpoint.add_step("Link Placement", True, 3,
                        "Links correctly placed below paper titles",
                        execution_time=step_time)
    else:
        checkpoint.add_step("Link Placement", False, 3,
                        "Link placement structure incorrect",
                        execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_4(gold_text):
    """
    Checkpoint 4 (1pt): The papers are from the correct relevant domain.
    
    Outcome Evaluation:
    - LLM as Judge for the relevance of each paper abstract to high quality dataset creation for Large 
Language Models.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=1, result=0, name="Domain Relevance Validation")

    # Use AI model to judge relevance
    global model
    if model is None:
        model = load_model(model_id)

    step_start = time.time()
    # TODO: Extract abstracts from document and use AI to judge relevance
    # Query: "Is this paper abstract relevant to high quality dataset creation for Large Language Models?"
    relevance_score = 0  # Count of relevant papers
    step_time = time.time() - step_start

    if relevance_score >= 5:  # All 5 papers should be relevant
        checkpoint.result += 1
        checkpoint.add_step("Domain Relevance", True, 1,
                        f"All {relevance_score} papers are relevant to LLM dataset creation",
                        execution_time=step_time)
    else:
        checkpoint.add_step("Domain Relevance", False, 1,
                        f"Only {relevance_score} papers are relevant, need all 5",
                        execution_time=step_time)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoints(workspace_doc_id=None, cached_models=None, browsing_history=None):
    """
    Grade all checkpoints for the influential papers task.
    
    Args:
        workspace_doc_id (str, optional): Direct Google Docs document ID to use
        cached_models (dict, optional): Dictionary of preloaded models by model_id
        browsing_history (list, optional): List of URLs visited during task execution
        
    Returns:
        Result: Evaluation results with checkpoint scores
    """
    total_start_time = time.time()

    try:
        # Setup document processing
        setup_document(workspace_doc_id, browsing_history)

        # Use cached model if available
        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        checkpoints: List[Checkpoint] = []

        checkpoints.append(grade_checkpoint_1(gold_text, browsing_history))
        checkpoints.append(grade_checkpoint_2(gold_text))
        checkpoints.append(grade_checkpoint_3(gold_text, doc_structure))
        checkpoints.append(grade_checkpoint_4(gold_text))

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

    parser = argparse.ArgumentParser(description="Evaluate influential papers document")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Docs document ID to evaluate")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
    parser.add_argument("--browsing_history", type=list, default=None, help="List of URLs visited during 
task")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
    print(f"CLEANUP enabled: {CLEANUP_ENABLED}")

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        cached_models=args.cached_models,
        browsing_history=args.browsing_history
    )
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