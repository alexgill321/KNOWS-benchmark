import os
from typing import List
import sys
import time
import arxiv
import argparse

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
from src.browsergym.eval.eval_utils.text_utils import text_fuzzy_match_contained_short, text_fuzzy_match_contained_long
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.parallel_utils import fast_parallel_vlm_calls
from src.browsergym.eval.tasks.docs_5_influential_papers.utils import *

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_5_influential_papers/instance_2/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_IMAGES_DIR = os.path.join(TASK_DIR, "data/pdf_images/")

# Instance-specific parameters
NUM_PAPERS = 5
MIN_CITATIONS = 100
RECENCY_YEARS = 3
TOPIC = "parameter-efficient fine-tuning (PEFT)"
RELEVANCE_QUESTION = "Is this paper highly relevant to parameter-efficient fine-tuning (PEFT) methods for language models or other deep learning models?"

# Model setup
model = None
model_id = "gemini-2.5-flash-google-ai"

# Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

# Global variables for document processing
doc_id = None
gold_text = None
doc_structure = None
cached_arxiv_papers = None  # Cache arXiv paper info to avoid redundant API calls

def cleanup_generated_files():
    """Clean up generated files and directories created during evaluation."""
    pass


def prefetch_arxiv_papers():
    """
    Prefetch arXiv paper info once for reuse across checkpoints 3 and 4.
    This avoids redundant arXiv API calls.
    """
    global cached_arxiv_papers

    paper_links = extract_arxiv_links_from_text(gold_text)
    paper_links = [normalize_arxiv_url(url) for url in paper_links if normalize_arxiv_url(url)]
    paper_links = list(set(paper_links))  # Unique IDs

    if len(paper_links) < NUM_PAPERS:
        print(f"Warning: Only found {len(paper_links)} unique arxiv paper links in document, expected at least {NUM_PAPERS}.")

    if not paper_links:
        print("Warning: No arXiv paper links found in document.")
        cached_arxiv_papers = []
        return

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=paper_links)
        cached_arxiv_papers = list(client.results(search))
        print(f"  Prefetched {len(cached_arxiv_papers)} arXiv papers")
    except Exception as e:
        print(f"Error prefetching arXiv papers: {e}")
        cached_arxiv_papers = []

def setup_document(workspace_doc_id):
    """
    Setup document processing and browsing history analysis.

    Args:
        workspace_doc_id (str): Direct Google Docs document ID to use
    """
    global gold_text, doc_structure
    gold_text = extract_text_from_doc(workspace_doc_id, DOCS_SERVICE)
    doc_structure = extract_structure_from_doc(workspace_doc_id, DOCS_SERVICE)

def grade_checkpoint_1(browsing_history):
    """
    Checkpoint 1 (2pt): 5 different papers were looked at and placed into the google docs.

    Outcome Evaluation:
    - Paper website was accessed at least 5 times for different papers.
    - Links in the google docs lines up with the pages that were accessed.
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=2, result=0, name="Paper Research and Documentation")

    if browsing_history is None:
        browsing_history = []

    # Step 1: Check if paper websites were accessed
    step_start = time.time()
    arxiv_visits = [url for url in browsing_history if 'arxiv.org' in url]
    unique_papers_visited = len(set(arxiv_visits))
    step_time = time.time() - step_start

    if unique_papers_visited >= NUM_PAPERS:
        checkpoint.add_step("Paper Website Access", True, 1,
                        f"Accessed {unique_papers_visited} different paper websites",
                        execution_time=step_time)
    else:
        checkpoint.add_step("Paper Website Access", False, 1,
                        f"Only accessed {unique_papers_visited} paper websites, need {NUM_PAPERS}",
                        execution_time=step_time)

    # Step 2: Check if links in document match visited pages
    step_start = time.time()
    links_match, doc_paper_ids, visited_paper_ids, matched_count = match_document_links_with_browsing_history(gold_text, browsing_history)
    step_time = time.time() - step_start

    if links_match:
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

def grade_checkpoint_2():
    """
    Checkpoint 2 (10pt): The papers meet the requirements for citation counts and recency.

    Outcome Evaluation:
    - Each paper has at least MIN_CITATIONS citations (1pt each).
    - Each paper is from the last RECENCY_YEARS years (1pt each).
    """
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Paper Requirements Validation")

    step_start = time.time()
    arxiv_ids = extract_arxiv_links_from_text(gold_text)
    arxiv_ids = [normalize_arxiv_url(url) for url in arxiv_ids if normalize_arxiv_url(url)]
    arxiv_ids = list(set(arxiv_ids))
    if len(arxiv_ids) < NUM_PAPERS:
        print(f"Warning: Only found {len(arxiv_ids)} unique arxiv paper links in document, expected at least {NUM_PAPERS}.")

    if not arxiv_ids:
        print("Error: No valid arxiv paper links found.")
        checkpoint.add_step("Paper Requirements", False, 0, "No valid arxiv paper links found to check requirements against.", score=0, max_score=10)
        return checkpoint

    import requests
    try:
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={'fields': 'citationCount,title,publicationDate,externalIds'},
            json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]}
        )
        papers_info = response.json()

        if not isinstance(papers_info, list):
            error_msg = f"Unexpected API response format: {type(papers_info).__name__}"
            if isinstance(papers_info, dict):
                error_msg += f" - {papers_info.get('message', 'Unknown error')}"
            elif isinstance(papers_info, str):
                error_msg += f" - {papers_info}"
            print(f"Error: {error_msg}")
            checkpoint.add_step("Semantic Scholar API", False, 0, f"API Error: {error_msg}", score=0, max_score=10)
            return checkpoint
    except Exception as e:
        print(f"Error fetching data from Semantic Scholar: {e}")
        checkpoint.add_step("Semantic Scholar API", False, 0, f"API Error: {e}", score=0, max_score=10)
        return checkpoint

    for i, paper in enumerate(papers_info):
        paper_step_start = time.time()
        if paper is not None and isinstance(paper, dict):
            title = paper.get('title', 'Unknown')
            total_citations = paper.get("citationCount", 0)
            publication_date = paper.get("publicationDate", "1900-01-01")

            # Citation Check (1pt)
            if total_citations >= MIN_CITATIONS:
                checkpoint.add_step(f"Citation Check {i+1}", True, 1,
                                f"Paper '{title}' has {total_citations} citations (>= {MIN_CITATIONS})",
                                execution_time=time.time() - paper_step_start)
            else:
                checkpoint.add_step(f"Citation Check {i+1}", False, 1,
                                f"Paper '{title}' has only {total_citations} citations, need {MIN_CITATIONS}",
                                execution_time=time.time() - paper_step_start)

            # Recency Check (1pt)
            if is_within_x_years(publication_date, RECENCY_YEARS):
                checkpoint.add_step(f"Recency Check {i+1}", True, 1,
                                f"Paper '{title}' published on {publication_date} is within {RECENCY_YEARS} years",
                                execution_time=time.time() - paper_step_start)
            else:
                checkpoint.add_step(f"Recency Check {i+1}", False, 1,
                                f"Paper '{title}' published on {publication_date} is older than {RECENCY_YEARS} years",
                                execution_time=time.time() - paper_step_start)
        else:
            arxiv_id = arxiv_ids[i] if i < len(arxiv_ids) else "Unknown"
            if paper is not None and not isinstance(paper, dict):
                error_type = type(paper).__name__
                error_msg = f"Paper with arXiv ID {arxiv_id} returned invalid format ({error_type})"
            else:
                error_msg = f"Paper with arXiv ID {arxiv_id} not found in Semantic Scholar"
            checkpoint.add_step(f"Citation Check {i+1}", False, 1,
                            error_msg,
                            execution_time=time.time() - paper_step_start)
            checkpoint.add_step(f"Recency Check {i+1}", False, 1,
                            error_msg,
                            execution_time=time.time() - paper_step_start)

    if len(papers_info) < NUM_PAPERS:
        for j in range(len(papers_info), NUM_PAPERS):
            checkpoint.add_step(f"Citation Check {j+1}", False, 1, f"Missing paper (fewer than {NUM_PAPERS} found)", execution_time=0)
            checkpoint.add_step(f"Recency Check {j+1}", False, 1, f"Missing paper (fewer than {NUM_PAPERS} found)", execution_time=0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_3():
    """
    Checkpoint 3 (20pt): The doc structure for each paper is correct.

    Outcome Evaluation:
    - Each paper abstract is included in the google docs.
    - Each paper title is included in the google docs.
    - Each paper link is included in the google docs.
    - Each paper structure is correct: Title -> Link -> Abstract.

    OPTIMIZED: Uses cached arXiv papers from prefetch to avoid redundant API calls.
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Document Structure Validation")

    step_start = time.time()

    if cached_arxiv_papers is None or len(cached_arxiv_papers) == 0:
        checkpoint.add_step("Paper Data", False, 1, "No arXiv papers found or prefetch failed.")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    papers_info = cached_arxiv_papers

    for i, paper in enumerate(papers_info):
        abstract = paper.summary
        abstract_match, abstract_score = text_fuzzy_match_contained_long(abstract, gold_text)
        title_match = text_fuzzy_match_contained_short(paper.title, gold_text)
        links_match = text_fuzzy_match_contained_short(paper.entry_id, gold_text)
        found_elements_for_ordering = []

        if abstract_match:
            checkpoint.add_step(f"Abstract Inclusion {i+1}", True, (i*5)+1,
                            f"Abstract for paper {paper.title} found in document",
                            execution_time=time.time() - step_start)
            found_elements_for_ordering.append(abstract_match)
        else:
            checkpoint.add_step(f"Abstract Inclusion {i+1}", False, (i*5)+1,
                            f"Abstract for paper {paper.title} not found in document, best match score: {abstract_score}",
                            execution_time=time.time() - step_start)

        if title_match:
            checkpoint.add_step(f"Title Inclusion {i+1}", True, (i*5)+2,
                            f"Title for paper {paper.title} found in document",
                            execution_time=time.time() - step_start)
            found_elements_for_ordering.append(title_match)
        else:
            checkpoint.add_step(f"Title Inclusion {i+1}", False, (i*5)+2,
                            f"Title for paper {paper.title} not found in document",
                            execution_time=time.time() - step_start)

        if links_match:
            checkpoint.add_step(f"Link Inclusion {i+1}", True, (i*5)+3,
                            f"Link for paper {paper.title} found in document",
                            execution_time=time.time() - step_start)
            found_elements_for_ordering.append(links_match)
        else:
            checkpoint.add_step(f"Link Inclusion {i+1}", False, (i*5)+3,
                            f"Link for paper {paper.title} not found in document",
                            execution_time=time.time() - step_start)

        expected_components = [
            ("Title", title_match),
            ("Link", links_match),
            ("Abstract", abstract_match)
        ]

        missing_component_names = [name for name, match in expected_components if match is None]

        if not missing_component_names:
            expected_text_order = [match for _, match in expected_components]
            ordered_elements = get_structural_element_order(doc_structure, expected_text_order)

            if ordered_elements == expected_text_order:
                checkpoint.add_step(f"Structure Check {i+1}", True, (i*5)+4,
                                    f"Correct structure for paper {paper.title}: Title -> Link -> Abstract",
                                    execution_time=time.time() - step_start)
            else:
                actual_order_titles = [text.split(':')[0] for text in ordered_elements]
                expected_order_titles = [text.split(':')[0] for text in expected_text_order]
                checkpoint.add_step(f"Structure Check {i+1}", False, (i*5)+4,
                                    f"Incorrect structure for paper {paper.title}. Expected order: {expected_order_titles}, Actual order: {actual_order_titles}",
                                    execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Structure Check {i+1}", False, (i*5)+4,
                                f"Cannot verify structure for paper {paper.title} due to missing elements: {', '.join(missing_component_names)}",
                                execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoint_4():
    """
    Checkpoint 4 (5pt): The papers are from the correct relevant domain.

    Outcome Evaluation:
    - LLM as Judge for the relevance of each paper abstract to the topic.
    - 1 point for each relevant paper.

    PARALLELIZED: Uses cached arXiv papers and parallel LLM calls for relevance checking.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Domain Relevance Validation")

    global model
    if model is None:
        model = load_model(model_id)

    step_start = time.time()

    if cached_arxiv_papers is None or len(cached_arxiv_papers) == 0:
        checkpoint.add_step("Paper Data", False, 5,
                          "No arXiv papers found or prefetch failed.",
                          execution_time=time.time() - step_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    papers_info = cached_arxiv_papers

    vlm_tasks = []
    for i, paper in enumerate(papers_info):
        abstract = paper.summary
        title = paper.title

        prompt = f"""
        Paper Title: {title}
        Abstract: {abstract}

        {RELEVANCE_QUESTION}

        i.e. Does this paper discuss {TOPIC}?

        Answer with exactly "YES" or "NO".
        """

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        vlm_tasks.append({
            'id': f'paper_{i}',
            'messages': messages,
            'title': title
        })

    print(f"  Running {len(vlm_tasks)} relevance checks in parallel...")
    vlm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=5)
    vlm_time = time.time() - step_start
    print(f"  Parallel relevance checks completed in {vlm_time:.2f}s")

    for i, paper in enumerate(papers_info):
        task_id = f'paper_{i}'
        title = paper.title
        is_relevant = vlm_results.get(task_id, False)

        if is_relevant:
            checkpoint.add_step(f"Relevance Check {i+1}", True, i+1,
                            f"Paper '{title}' is relevant.",
                            execution_time=0)
        else:
            checkpoint.add_step(f"Relevance Check {i+1}", False, i+1,
                            f"Paper '{title}' judged NOT relevant.",
                            execution_time=0)

    if len(papers_info) < NUM_PAPERS:
        for j in range(len(papers_info), NUM_PAPERS):
            checkpoint.add_step(f"Relevance Check {j+1}", False, j+1,
                            "Missing paper (fewer than 5 found).",
                            execution_time=0)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint

def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
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
        setup_document(workspace_doc_id)

        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        print("Prefetching arXiv papers...")
        prefetch_arxiv_papers()

        checkpoints: List[Checkpoint] = []

        checkpoints.append(grade_checkpoint_1(browsing_history))
        checkpoints.append(grade_checkpoint_2())
        checkpoints.append(grade_checkpoint_3())
        checkpoints.append(grade_checkpoint_4())

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    finally:
        try:
            cleanup_generated_files()
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed with error: {cleanup_error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate influential papers document")
    parser.add_argument("--workspace_doc_id", type=str, help="Google Docs document ID to evaluate")
    parser.add_argument("--browsing_history", nargs='+', help="List of URLs visited during task")
    parser.add_argument("--cached_models", type=dict, default=None, help="Dictionary of preloaded models")
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
