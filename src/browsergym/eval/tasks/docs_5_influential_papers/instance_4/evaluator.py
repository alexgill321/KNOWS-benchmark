import os
from typing import List
import sys
import time
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
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result, EvaluationStep
from src.browsergym.eval.eval_utils.google_services_utils import *
from src.browsergym.eval.eval_utils.text_utils import text_fuzzy_match_contained_short, text_fuzzy_match_contained_long
from src.browsergym.eval.eval_utils.models import load_model
from src.browsergym.eval.eval_utils.parallel_utils import fast_parallel_vlm_calls
from src.browsergym.eval.tasks.docs_5_influential_papers.utils import (
    is_within_x_years,
    extract_paper_links_from_text,
    extract_paper_id,
    fetch_papers_from_semantic_scholar,
    match_paper_links_with_browsing_history,
)

# Constants
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_5_influential_papers/instance_4/")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"

# Instance-specific parameters
NUM_PAPERS = 5
MIN_CITATIONS = 50
RECENCY_YEARS = 5
TOPIC = "De Novo Protein Design"
RELEVANCE_QUESTION = "Is this paper highly relevant to De Novo Protein Design, i.e. computationally designing novel protein structures or sequences?"
# Accepted paper platforms for this instance
PAPER_DOMAINS = ['biorxiv.org', 'nature.com']

# Model setup
model = None
model_id = "gemini-2.5-flash-google-ai"

# Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services()

# Global variables for document processing
doc_id = None
gold_text = None
doc_structure = None
cached_papers_info = None  # Cache paper info from Semantic Scholar


def cleanup_generated_files():
    """Clean up generated files and directories created during evaluation."""
    pass


def setup_document(workspace_doc_id):
    """
    Setup document processing.

    Args:
        workspace_doc_id (str): Direct Google Docs document ID to use
    """
    global gold_text, doc_structure
    gold_text = extract_text_from_doc(workspace_doc_id, DOCS_SERVICE)
    doc_structure = extract_structure_from_doc(workspace_doc_id, DOCS_SERVICE)


def prefetch_papers():
    """Prefetch paper metadata from Semantic Scholar for reuse across checkpoints."""
    global cached_papers_info

    doc_links = extract_paper_links_from_text(gold_text, PAPER_DOMAINS)
    paper_ids = []
    for link in doc_links:
        pid = extract_paper_id(link)
        if pid:
            paper_ids.append(pid)
    paper_ids = list(set(paper_ids))

    if len(paper_ids) < NUM_PAPERS:
        print(f"Warning: Only found {len(paper_ids)} unique paper links in document, expected at least {NUM_PAPERS}.")

    if not paper_ids:
        print("Warning: No paper links found in document.")
        cached_papers_info = []
        return

    cached_papers_info = fetch_papers_from_semantic_scholar(
        paper_ids, fields='citationCount,title,publicationDate,abstract,externalIds,url'
    )
    found = len([p for p in cached_papers_info if p and isinstance(p, dict)])
    print(f"  Prefetched {found} papers from Semantic Scholar")


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
    paper_visits = [url for url in browsing_history
                    if any(domain in url for domain in PAPER_DOMAINS)]
    unique_papers_visited = len(set(paper_visits))
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
    links_match, doc_ids, visited_ids, matched_count = match_paper_links_with_browsing_history(
        gold_text, browsing_history, domains=PAPER_DOMAINS, min_papers=NUM_PAPERS
    )
    step_time = time.time() - step_start

    if links_match:
        checkpoint.add_step("Document Links Match", True, 2,
                        f"Document links match browsing history: {matched_count} papers matched out of {len(doc_ids)} in document",
                        execution_time=step_time)
    else:
        detail_msg = f"Links don't match: {matched_count} papers matched out of {len(doc_ids)} in document. "
        detail_msg += f"Document has {len(doc_ids)} paper links, visited {len(visited_ids)} unique papers"
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

    if cached_papers_info is None or not cached_papers_info:
        checkpoint.add_step("Paper Requirements", False, 0, "No paper data available.", score=0, max_score=10)
        return checkpoint

    for i, paper in enumerate(cached_papers_info):
        paper_step_start = time.time()
        if paper is not None and isinstance(paper, dict):
            title = paper.get('title', 'Unknown')
            total_citations = paper.get("citationCount", 0)
            publication_date = paper.get("publicationDate", "1900-01-01")

            if total_citations >= MIN_CITATIONS:
                checkpoint.add_step(f"Citation Check {i+1}", True, 1,
                                f"Paper '{title}' has {total_citations} citations (>= {MIN_CITATIONS})",
                                execution_time=time.time() - paper_step_start)
            else:
                checkpoint.add_step(f"Citation Check {i+1}", False, 1,
                                f"Paper '{title}' has only {total_citations} citations, need {MIN_CITATIONS}",
                                execution_time=time.time() - paper_step_start)

            if is_within_x_years(publication_date, RECENCY_YEARS):
                checkpoint.add_step(f"Recency Check {i+1}", True, 1,
                                f"Paper '{title}' published on {publication_date} is within {RECENCY_YEARS} years",
                                execution_time=time.time() - paper_step_start)
            else:
                checkpoint.add_step(f"Recency Check {i+1}", False, 1,
                                f"Paper '{title}' published on {publication_date} is older than {RECENCY_YEARS} years",
                                execution_time=time.time() - paper_step_start)
        else:
            checkpoint.add_step(f"Citation Check {i+1}", False, 1,
                            f"Paper {i+1} not found in Semantic Scholar",
                            execution_time=time.time() - paper_step_start)
            checkpoint.add_step(f"Recency Check {i+1}", False, 1,
                            f"Paper {i+1} not found in Semantic Scholar",
                            execution_time=time.time() - paper_step_start)

    papers_found = len([p for p in cached_papers_info if p and isinstance(p, dict)])
    if papers_found < NUM_PAPERS:
        for j in range(papers_found, NUM_PAPERS):
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
    """
    print("----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Document Structure Validation")

    step_start = time.time()

    if cached_papers_info is None or not cached_papers_info:
        checkpoint.add_step("Paper Data", False, 1, "No paper data available.")
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Get document links for link-presence verification
    doc_links = extract_paper_links_from_text(gold_text, PAPER_DOMAINS)
    doc_paper_ids = set()
    for link in doc_links:
        pid = extract_paper_id(link)
        if pid:
            doc_paper_ids.add(pid)

    for i, paper in enumerate(cached_papers_info):
        if paper is None or not isinstance(paper, dict):
            checkpoint.add_step(f"Paper {i+1} Data", False, 1, f"Paper {i+1} not found")
            continue

        title = paper.get('title', '')
        abstract = paper.get('abstract', '')

        # Abstract check
        abstract_match = None
        if abstract:
            abstract_match, abstract_score = text_fuzzy_match_contained_long(abstract, gold_text)
            if abstract_match:
                checkpoint.add_step(f"Abstract Inclusion {i+1}", True, (i*5)+1,
                                f"Abstract for paper '{title}' found in document",
                                execution_time=time.time() - step_start)
            else:
                checkpoint.add_step(f"Abstract Inclusion {i+1}", False, (i*5)+1,
                                f"Abstract for paper '{title}' not found in document, best match score: {abstract_score}",
                                execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Abstract Inclusion {i+1}", False, (i*5)+1,
                            f"No abstract available for paper '{title}' from Semantic Scholar",
                            execution_time=time.time() - step_start)

        # Title check
        title_match = text_fuzzy_match_contained_short(title, gold_text) if title else None
        if title_match:
            checkpoint.add_step(f"Title Inclusion {i+1}", True, (i*5)+2,
                            f"Title for paper '{title}' found in document",
                            execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Title Inclusion {i+1}", False, (i*5)+2,
                            f"Title for paper '{title}' not found in document",
                            execution_time=time.time() - step_start)

        # Link check - verify the paper's link actually appears in the document
        paper_external_ids = paper.get('externalIds', {}) or {}
        paper_doi = paper_external_ids.get('DOI')
        link_found = False
        if paper_doi:
            for pid in doc_paper_ids:
                if pid[0] == "DOI" and pid[1] == paper_doi:
                    link_found = True
                    break

        if link_found:
            checkpoint.add_step(f"Link Inclusion {i+1}", True, (i*5)+3,
                            f"Link for paper '{title}' found in document",
                            execution_time=time.time() - step_start)
        else:
            checkpoint.add_step(f"Link Inclusion {i+1}", False, (i*5)+3,
                            f"Link for paper '{title}' not found in document",
                            execution_time=time.time() - step_start)

        # Structure check: Title -> Link -> Abstract
        if title_match and link_found and abstract and abstract_match:
            checkpoint.add_step(f"Structure Check {i+1}", True, (i*5)+4,
                                f"Structure validated for paper '{title}'",
                                execution_time=time.time() - step_start)
        else:
            missing = []
            if not title_match:
                missing.append("Title")
            if not link_found:
                missing.append("Link")
            if not abstract or not abstract_match:
                missing.append("Abstract")
            checkpoint.add_step(f"Structure Check {i+1}", False, (i*5)+4,
                                f"Cannot verify structure for paper '{title}' due to missing: {', '.join(missing)}",
                                execution_time=time.time() - step_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4 (5pt): The papers are from the correct relevant domain.

    Outcome Evaluation:
    - LLM as Judge for the relevance of each paper abstract to the topic.
    - 1 point for each relevant paper.
    """
    print("----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Domain Relevance Validation")

    global model
    if model is None:
        model = load_model(model_id)

    step_start = time.time()

    if cached_papers_info is None or not cached_papers_info:
        checkpoint.add_step("Paper Data", False, 5, "No paper data available.",
                          execution_time=time.time() - step_start)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    vlm_tasks = []
    valid_papers = []
    for i, paper in enumerate(cached_papers_info):
        if paper is None or not isinstance(paper, dict):
            continue
        title = paper.get('title', 'Unknown')
        abstract = paper.get('abstract', '')
        if not abstract:
            continue
        valid_papers.append((i, title))

        prompt = f"""
        Paper Title: {title}
        Abstract: {abstract}

        {RELEVANCE_QUESTION}

        Answer with exactly "YES" or "NO".
        """

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        vlm_tasks.append({
            'id': f'paper_{i}',
            'messages': messages,
            'title': title
        })

    if vlm_tasks:
        print(f"  Running {len(vlm_tasks)} relevance checks in parallel...")
        vlm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=5)
        vlm_time = time.time() - step_start
        print(f"  Parallel relevance checks completed in {vlm_time:.2f}s")

        for idx, (i, title) in enumerate(valid_papers):
            task_id = f'paper_{i}'
            is_relevant = vlm_results.get(task_id, False)

            if is_relevant:
                checkpoint.add_step(f"Relevance Check {idx+1}", True, idx+1,
                                f"Paper '{title}' is relevant.",
                                execution_time=0)
            else:
                checkpoint.add_step(f"Relevance Check {idx+1}", False, idx+1,
                                f"Paper '{title}' judged NOT relevant.",
                                execution_time=0)

    papers_checked = len(valid_papers)
    if papers_checked < NUM_PAPERS:
        for j in range(papers_checked, NUM_PAPERS):
            checkpoint.add_step(f"Relevance Check {j+1}", False, j+1,
                            "Missing paper or abstract not available.",
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

        print("Prefetching paper metadata...")
        prefetch_papers()

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
