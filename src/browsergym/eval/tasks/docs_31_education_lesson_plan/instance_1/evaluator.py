import os
import sys
import time
import shutil
import glob
from typing import List
from concurrent.futures import ThreadPoolExecutor

# Get the base path that works in both Docker and local environments
def get_base_path():
    # First check if we're in a Docker container at /app
    if os.path.exists("/app/src"):
        return "/app"
    # Check for CHPC scratch directory
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    # Otherwise use current working directory
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

# Core evaluation imports
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result 
from src.browsergym.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    extract_text_from_doc,
    extract_text_colors_from_doc,
    get_doc_content,
    download_doc_as_pdf,
    extract_images_from_doc_extended
)
from src.browsergym.eval.eval_utils.text_utils import (
    extract_text_from_pdf,
    fuzzy_match_text
)
from src.browsergym.eval.eval_utils.image_utils import convert_pdf_to_pngs 
from src.browsergym.eval.eval_utils.models import load_model 
from src.browsergym.eval.eval_utils.utils import rgb_to_hex  

# Task-specific utilities
from src.browsergym.eval.tasks.docs_31_education_lesson_plan.utils import (
    validate_topics_chemistry_related,
    check_images_aligned_horizontally,
    extract_summary_facts_with_colors,
    extract_bullet_hierarchy_from_doc
)
from src.browsergym.eval.eval_utils.google_services_utils import extract_hyperlinks_from_doc

# Checkpoint 2+ imports
from src.browsergym.eval.eval_utils.web_utils import (
    validate_url_accessible,
    normalize_url_for_comparison,
    fetch_with_fallbacks
)
from src.browsergym.eval.eval_utils.parallel_utils import (
    parallel_execute,
    fast_parallel_vlm_calls,
)

# Directory paths
TASK_DIR = os.path.join(BASE_PATH, "src/browsergym/eval/tasks/docs_31_education_lesson_plan/instance_1/")
DATA_DIR = os.path.join(TASK_DIR, "data/")
PDF_IMAGES_DIR = os.path.join(DATA_DIR, "pdf_images/")
IMAGES_DIR = os.path.join(DATA_DIR, "doc_images/")

# Configuration
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_DPI = 150

# Model configuration
model = None
model_id = "gemini-2.5-flash-google-ai"  # Fast and capable for topic/fact validation

# Initialize Google services
DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services(service_type="docs")

# Global variables set by setup_document()
doc_id = None
doc_text = None           # Raw text from extract_text_from_doc
doc_content = None        # Full document JSON from get_doc_content
bullet_hierarchy = None   # Hierarchical bullet structure
text_colors = None        # Text colors for summary validation (Checkpoint 4)
text_ocr = None           # OCR from PDF for image positioning (Checkpoint 5)
browsing_history = None   # Will be passed from grade_checkpoints()
website_content_cache = {}  # url -> (content_str, status_str); populated by CP2, reused by CP3


def cleanup_generated_files():
    """Clean up all generated files and directories created during evaluation."""
    if not CLEANUP_ENABLED:
        print("Cleanup disabled by CLEANUP=False environment variable")
        return

    print("Cleaning up generated files...")
    cleanup_dirs = [PDF_IMAGES_DIR, IMAGES_DIR]

    # Clean up directories
    for dir_path in cleanup_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

    # Clean up PDF files in data directory
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            os.remove(pdf_file)
            print(f"Removed PDF file: {pdf_file}")
        except Exception as e:
            print(f"Error removing PDF {pdf_file}: {e}")

    print("Cleanup completed")


def setup_document(workspace_doc_id):
    """
    Setup document processing using the provided workspace_doc_id.

    Extracts document structure, bullet hierarchy, colors, and OCR data
    for comprehensive lesson plan evaluation.

    Args:
        workspace_doc_id (str): The Google Docs document ID to evaluate.
    """
    global doc_id, doc_text, doc_content, bullet_hierarchy, text_colors, text_ocr

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    print(f"Using workspace document ID: {workspace_doc_id}")
    doc_id = workspace_doc_id

    # Phase 1: Download PDF and get document content (sequential, required order)
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(DATA_DIR, "lesson_plan.pdf")
    download_doc_as_pdf(doc_id, pdf_path, DRIVE_SERVICE)

    # Get full document JSON (needed for bullet hierarchy and colors)
    doc_content = get_doc_content(doc_id, DOCS_SERVICE)

    # Phase 2: Run parallel extraction operations
    def run_ocr():
        # Convert PDF to images first, then OCR
        convert_pdf_to_pngs(pdf_path, PDF_IMAGES_DIR, dpi=PDF_DPI)
        return extract_text_from_pdf(PDF_IMAGES_DIR)

    with ThreadPoolExecutor(max_workers=4) as executor:
        text_future = executor.submit(extract_text_from_doc, doc_id, DOCS_SERVICE)
        bullets_future = executor.submit(extract_bullet_hierarchy_from_doc, doc_content)
        colors_future = executor.submit(extract_text_colors_from_doc, doc_content)
        ocr_future = executor.submit(run_ocr)

        doc_text = text_future.result()
        bullet_hierarchy = bullets_future.result()
        text_colors = colors_future.result()
        text_ocr = ocr_future.result()

    # Log setup completion
    num_topics = len(bullet_hierarchy.get('topics', [])) if bullet_hierarchy else 0
    print(f"Setup complete: extracted {num_topics} top-level items")

    # Debug output (only if DEBUG is enabled)
    if DEBUG:
        print(f"\nDEBUG: Bullet hierarchy structure:")
        if bullet_hierarchy:
            print(f"  Keys in bullet_hierarchy: {list(bullet_hierarchy.keys())}")
            print(f"  Number of topics: {len(bullet_hierarchy.get('topics', []))}")
            if bullet_hierarchy.get('topics'):
                for i, topic in enumerate(bullet_hierarchy['topics'][:3]):  # Show first 3
                    print(f"  Topic {i}: nesting_level={topic.get('nesting_level')}, text={topic.get('text', '')[:50]}...")
        else:
            print(f"  bullet_hierarchy is None or empty")

        print(f"\nDEBUG: Document text (first 500 chars):")
        if doc_text:
            print(f"  {doc_text[:500]}")
        else:
            print(f"  doc_text is None or empty")

        print(f"\nDEBUG: Document structure check:")
        if doc_content:
            # Check if document has any bullet points or list items
            bullet_count = 0
            list_count = 0
            element_types = {}
            for element in doc_content.get('body', {}).get('content', []):
                # Track all element types
                for key in element.keys():
                    element_types[key] = element_types.get(key, 0) + 1

                if 'paragraph' in element:
                    para_style = element['paragraph'].get('paragraphStyle', {})
                    if 'bullet' in para_style:
                        bullet_count += 1
                elif 'listItem' in element:
                    list_count += 1
            print(f"  Found {bullet_count} paragraph bullets and {list_count} list items in document")
            print(f"  Element types in document: {element_types}")

            # Show first few elements for inspection
            print(f"\nDEBUG: First 3 document elements:")
            for i, element in enumerate(doc_content.get('body', {}).get('content', [])[:3]):
                print(f"  Element {i}: keys={list(element.keys())}")
                if 'paragraph' in element:
                    para = element['paragraph']
                    text_content = ''
                    for elem in para.get('elements', []):
                        if 'textRun' in elem:
                            text_content += elem['textRun'].get('content', '')
                    print(f"    Paragraph text: {text_content[:80]}")
                    print(f"    Has bullet: {'bullet' in para.get('paragraphStyle', {})}")
        else:
            print(f"  doc_content is None")


def grade_checkpoint_1():
    """
    Checkpoint 1 (15pt): Related Topics

    Validates that document has 2-3 main-level topics related to Chemistry.

    Checkpoint criteria mapping (from checkpoints.md):
    - Criterion 1: "At least 2 and no more than 3 main level topics" → Step 1 (10pt)
    - Criterion 2: "All top level bullet points are chemistry-related" → Step 2 (5pt)
    """
    print("\n----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=15, result=0, name="Related Topics")

    global model

    # Extract main-level topics (nesting_level == 0)
    main_topics = []
    if bullet_hierarchy and 'topics' in bullet_hierarchy:
        main_topics = [
            item for item in bullet_hierarchy['topics']
            if item.get('nesting_level', 0) == 0
        ]
    else:
        # Error case: no bullet hierarchy
        checkpoint.add_step(
            "Extract Main Topics",
            False,
            15,
            "Failed to extract bullet hierarchy from document. Document may not contain properly formatted bullet points.",
            execution_time=time.time() - checkpoint_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Check if we have any topics at all (even if not at nesting level 0)
    total_items = len(bullet_hierarchy.get('topics', []))
    if total_items == 0:
        step_time = time.time() - checkpoint_start
        checkpoint.add_step(
            "Document Format",
            False,
            15,
            f"Document has no bullet points. Expected bulleted list structure with 2-3 main topics. Found plain text instead.",
            execution_time=step_time
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Step 1: Count validation (10 points)
    step_start = time.time()
    num_topics = len(main_topics)
    count_valid = 2 <= num_topics <= 3
    step_time = time.time() - step_start

    if count_valid:
        checkpoint.add_step(
            "Topic Count (2-3)",
            True,
            1,
            f"Found {num_topics} main-level topics (valid range: 2-3)",
            score=10,
            max_score=10,
            execution_time=step_time
        )
    else:
        checkpoint.add_step(
            "Topic Count (2-3)",
            False,
            1,
            f"Found {num_topics} main-level topics (expected: 2-3)",
            score=0,
            max_score=10,
            execution_time=step_time
        )

    # Step 2: Topic relevance validation (5 points)
    step_start = time.time()

    # Load model if not already loaded
    if model is None:
        model = load_model(model_id)
        print(f"Loaded model: {model_id}")

    topics_valid = validate_topics_chemistry_related(
        [t['text'].strip() for t in main_topics],
        model
    )
    step_time = time.time() - step_start

    if topics_valid['all_valid']:
        topics_str = ', '.join(topics_valid['topics'])
        checkpoint.add_step(
            "Topics Related to Chemistry",
            True,
            2,
            f"All {num_topics} topics validated as chemistry-related: {topics_str}",
            score = 5,
            max_score=5,
            execution_time=step_time
        )
    else:
        invalid_topics = [
            t for t in topics_valid['topics']
            if not topics_valid['validations'][t]
        ]
        invalid_str = ', '.join(invalid_topics)
        reasons_str = '; '.join([
            f"{t}: {topics_valid['reasons'][t]}"
            for t in invalid_topics
        ])
        invalid_topic_score = ((len(topics_valid['validations']) - len(invalid_topics)) / len(topics_valid['validations'])) * 5
        checkpoint.add_step(
            "Topics Related to Chemistry",
            False,
            2,
            f"Invalid topics found (not chemistry-related): {invalid_str}. Details: {reasons_str}",
            score = invalid_topic_score,
            max_score = 5,
            execution_time=step_time
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2 (40pt): Website URLs match
    Validates that each topic has valid, visited, and relevant website URLs.

    Checkpoint criteria mapping (from checkpoints.md):
    - Each topic has at least 4 websites → Step per topic (URL Count)
    - Each website is valid → Step per topic (URL Validity)
    - Agent visited each website → Step per topic (URLs Visited)
    - Each website is related to topic → Step per topic (URL Relevance)
    """
    print("\n----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=40, result=0, name="Website URLs Match")

    global model, browsing_history

    # --- Data Accumulators for Global Grading ---
    url_count_per_topic = {}
    
    # Step 2 Accumulators
    failed_website_per_topic = {} # Tracks topics that had inaccessible URLs
    
    # Step 3 Accumulators (Global)
    total_urls_found = 0
    total_urls_visited = 0
    
    # Step 4 Accumulators (Global)
    total_relevance_checked = 0
    total_relevance_passed = 0
    relevance_failure_examples = []

    # Handle None browsing_history
    if browsing_history is None:
        browsing_history = []
        print("Warning: No browsing history provided")

    # Normalize browsing history once for efficient comparison
    normalized_browsing = set([
        normalize_url_for_comparison(url)
        for url in browsing_history
        if url  # Skip empty URLs
    ])

    # Get main topics from Checkpoint 1
    main_topics = []
    if bullet_hierarchy and 'topics' in bullet_hierarchy:
        main_topics = [
            item for item in bullet_hierarchy['topics']
            if item.get('nesting_level', 0) == 0
        ]

    if not main_topics:
        checkpoint.add_step(
            "Extract Topics",
            False,
            1,
            "No main topics found in document - cannot validate URLs",
            score = 0,
            max_score = 10,
            execution_time=time.time() - checkpoint_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_topics = len(main_topics)

    # Collects accessible URLs per topic for batched relevance check after the loop
    urls_to_check_per_topic = {}  # topic_idx -> (topic_dict, [urls])
    all_urls_for_cache = set()    # all unique URLs across all topics, for content fetching

    # ------------------- MAIN VALIDATION LOOP -------------------
    for topic_idx, topic in enumerate(main_topics):
        topic_name = topic['text'][:40]
        print(f"Validating Topic {topic_idx+1}: {topic_name}")

        # Extract URLs from this topic's children
        topic_urls = []
        for child in topic.get('children', []):
            if child.get('nesting_level') == 1 and 'url' in child:
                topic_urls.append(child['url'])

        # Collect all valid URLs for the shared content cache
        all_urls_for_cache.update(url for url in topic_urls if url and url != 'No URL')

        # --- Step 1 Prep: Count ---
        url_count = len(topic_urls)
        url_count_per_topic[topic_idx] = url_count
        total_urls_found += url_count

        # --- Step 2 Prep: Validity ---
        accessible_urls = []
        failed_urls = []
        for url in topic_urls:
            is_accessible, details = validate_url_accessible(url, timeout=10)
            if is_accessible:
                accessible_urls.append(url)
            else:
                failed_urls.append((url, details))
        
        # Only mark topic as failed if it had inaccessible URLs
        if failed_urls:
             failed_website_per_topic[topic_idx] = failed_urls

        # --- Step 3 Prep: Visitation (Accumulate Global Counts) ---
        for url in topic_urls:
            normalized_url = normalize_url_for_comparison(url)
            if normalized_url in normalized_browsing:
                total_urls_visited += 1
            # We don't need else/unvisited list here anymore, just the raw count

        # --- Step 4 Prep: Relevance (collect URLs; LLM calls batched after loop) ---
        urls_to_check_per_topic[topic_idx] = (topic, accessible_urls[:4])

    # ------------------- FETCH ALL SITE CONTENT (POPULATE CACHE) -------------------
    global model, website_content_cache
    fetch_tasks = [
        {'id': url, 'func': fetch_with_fallbacks, 'args': (url,), 'kwargs': {'max_chars': 15000}}
        for url in all_urls_for_cache
        if url not in website_content_cache
    ]
    if fetch_tasks:
        print(f"Fetching content for {len(fetch_tasks)} URLs (max 3 workers)...")
        fetched = parallel_execute(fetch_tasks, max_workers=3)
        website_content_cache.update(fetched)

    # ------------------- BATCH RELEVANCE VLM CALLS -------------------
    if model is None:
        model = load_model(model_id)
        print(f"Loaded model for relevance checking: {model_id}")

    vlm_tasks = []
    relevance_task_map = {}  # task_id -> url string for failure reporting

    for topic_idx, (topic, urls) in urls_to_check_per_topic.items():
        for url in urls:
            total_relevance_checked += 1
            task_id = f"topic_{topic_idx}|{url}"
            relevance_task_map[task_id] = url

            content_tuple = website_content_cache.get(url)
            content_excerpt = content_tuple[0][:1500] if content_tuple and content_tuple[0] else ""

            prompt_parts = [f"Topic: {topic['text']}\nURL: {url}"]
            if content_excerpt:
                prompt_parts.append(f"\nWebsite content excerpt:\n{content_excerpt}")
            prompt_parts.append("\n\nIs this website relevant to the topic? Answer only Yes or No.")

            vlm_tasks.append({
                'id': task_id,
                'messages': [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are evaluating website relevance. Answer only 'Yes' or 'No'."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "".join(prompt_parts)}]
                    }
                ]
            })

    print(f"Checking relevance of {len(vlm_tasks)} URLs in parallel...")
    relevance_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=5)

    for task_id, passed in relevance_results.items():
        if passed:
            total_relevance_passed += 1
        else:
            url = relevance_task_map[task_id]
            relevance_failure_examples.append(f"{url[:30]}... (Not Relevant)")

    # ------------------- FINAL SCORING -------------------

    # Part 1 Check: Topic URL Counts
    valid_topics = 0
    for topic_index in url_count_per_topic:
        if url_count_per_topic[topic_index] >= 4:
            valid_topics += 1
    
    topic_amount = len(url_count_per_topic)
    
    if valid_topics == topic_amount:
        checkpoint.add_step(
            "All topics have 4+ links",
            True,
            1,
            f"All topics have the correct amount of URLs",
            score = 10,
            max_score = 10
        )
    else:
        improperLinkScore = (valid_topics / topic_amount) * 10
        checkpoint.add_step(
            "All topics have 4+ links",
            False,
            1,
            f"Only {valid_topics} out of {topic_amount} topics have 4+ links",
            score = improperLinkScore,
            max_score = 10
        )

    # Part 2 Check: Website Validity
    # Logic corrected: if length is 0, it means NO topics failed (Success)
    failed_website_topic_length = len(failed_website_per_topic)
    
    if failed_website_topic_length == 0:
        checkpoint.add_step(
            "Website Validity",
            True,
            2,
            "All Websites are valid and accessible",
            score = 10,
            max_score = 10
        )
    else:
        # Calculate score based on topics that were fully valid
        failed_website_score = round(((topic_amount - failed_website_topic_length) / topic_amount) * 10)
        checkpoint.add_step(
            "Website Validity",
            False,
            2,
            f"Only {topic_amount - failed_website_topic_length} out of {topic_amount} topics contain fully valid websites",
            score=failed_website_score,
            max_score=10
        )

    # Part 3 Check: Agent Visited Websites (Global Score)
    # Score = (Total Visited / Total Found) * 10
    if total_urls_found == 0:
        # Edge case: No URLs found at all
        visited_score = 0
        visited_msg = "No URLs found to check visitation."
    else:
        visited_score = round((total_urls_visited / total_urls_found) * 10)
        visited_msg = f"Agent visited {total_urls_visited} out of {total_urls_found} total URLs found."

    checkpoint.add_step(
        "Agent Visited Websites",
        total_urls_visited == total_urls_found, # Pass only if 100% matched
        3,
        visited_msg,
        score = visited_score,
        max_score = 10
    )

    # Part 4 Check: URL Relevance (Global Score)
    # Score = (Total Relevant / Total Checked) * 10
    if total_relevance_checked == 0:
        relevance_score = 0
        relevance_msg = "No accessible URLs to check for relevance."
        passed = False
    else:
        relevance_score = round((total_relevance_passed / total_relevance_checked) * 10)
        if total_relevance_passed == total_relevance_checked:
            relevance_msg = f"All {total_relevance_checked} of valid websites are relevant."
            passed = True
        else:
            example_failures = "; ".join(relevance_failure_examples[:2])
            relevance_msg = f"{total_relevance_passed}/{total_relevance_checked} websites relevant. Failures: {example_failures}"
            passed = False

    checkpoint.add_step(
        "Website Relevance",
        passed,
        4,
        relevance_msg,
        score = relevance_score,
        max_score = 10
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (30pt): Website Facts are valid

    Validates that each website has facts and those facts exist on the website.

    Checkpoint criteria mapping (from checkpoints.md):
    - Each website has at least 5 facts → Step 1 (15pt - Global score)
    - Each fact exists on the website → Step 2 (15pt - Global score)
    """
    print("\n----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=30, result=0, name="Website Facts are Valid")

    # --- Data Accumulators ---
    total_websites = 0
    websites_with_5plus_facts = 0

    total_facts_checked = 0
    facts_verified_on_website = 0
    verification_failure_examples = []

    # Get main topics
    main_topics = []
    if bullet_hierarchy and 'topics' in bullet_hierarchy:
        main_topics = [
            item for item in bullet_hierarchy['topics']
            if item.get('nesting_level', 0) == 0
        ]

    if not main_topics:
        checkpoint.add_step(
            "Extract Topics",
            False,
            1,
            "No main topics found - cannot validate facts",
            score=0,
            max_score=15,
            execution_time=time.time() - checkpoint_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # ------------------- PHASE A: COLLECT URL ENTRIES -------------------
    # Gather all URLs and their facts across topics, tracking Step 1 counts
    url_entries = []  # list of (url, facts)
    for topic_idx, topic in enumerate(main_topics):
        topic_name = topic['text'][:40]
        print(f"Collecting facts for Topic {topic_idx+1}: {topic_name}")

        for url_item in topic.get('children', []):
            if url_item.get('nesting_level') == 1:
                total_websites += 1
                url = url_item.get('url', 'No URL')

                facts = [
                    fact_item['text'].strip()
                    for fact_item in url_item.get('children', [])
                    if fact_item.get('nesting_level') == 2
                ]

                # --- Step 1 Prep: Fact Count ---
                if len(facts) >= 5:
                    websites_with_5plus_facts += 1

                url_entries.append((url, facts))

    # ------------------- PHASE B: USE SHARED CONTENT CACHE -------------------
    # Content was already fetched in grade_checkpoint_2(); read from the global cache.

    # ------------------- PHASE C: BUILD & RUN PARALLEL LLM CALLS -------------------
    # Load model once before parallel calls
    global model
    if model is None:
        model = load_model(model_id)
        print(f"Loaded model for fact verification: {model_id}")

    vlm_tasks = []
    vlm_results = {}
    fact_task_map = {}  # task_id -> fact text for failure reporting
    unavailable_page_facts = []  # (task_id, fact) for pages we couldn't fetch at all
    url_content_map = {}  # task_id -> content length, to detect 404s in failed results

    for url, facts in url_entries:
        if not url or url == 'No URL':
            for i, fact in enumerate(facts):
                total_facts_checked += 1
                task_id = f"no_url|fact_{i}"
                fact_task_map[task_id] = fact
                unavailable_page_facts.append((task_id, fact))
            continue
        result = website_content_cache.get(url)
        content = result[0] if result else None
        if not content:
            print(f"  Cache miss for {url[:60]}, fetching with fallbacks...")
            try:
                content, status = fetch_with_fallbacks(url, max_chars=15000)
                if content:
                    website_content_cache[url] = (content, status)
            except Exception as e:
                print(f"  All fetch strategies failed: {e}")
        if not content:
            print(f"  No content for {url[:60]} — will verify as chemistry facts")
            for i, fact in enumerate(facts):
                total_facts_checked += 1
                task_id = f"{url}|fact_{i}"
                fact_task_map[task_id] = fact
                unavailable_page_facts.append((task_id, fact))
            continue
        for i, fact in enumerate(facts):
            total_facts_checked += 1
            task_id = f"{url}|fact_{i}"
            fact_task_map[task_id] = fact
            url_content_map[task_id] = len(content.strip())

            # LLM verification — can answer Yes, No, or Page Not Available
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are checking if a fact from a lesson plan could reasonably come from this website. Be very generous — the fact does not need to be a direct quote. Answer 'Yes' if the website discusses the same topic, concept, or any related information, even if worded very differently, only partially covered, or loosely related. Answer 'Page Not Available' if the website content appears to be an error page, 404 page, access denied page, or otherwise does not contain real content. Answer only 'Yes', 'No', or 'Page Not Available'."}]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Fact: {fact}\n\nWebsite content:\n{content}\n\nCould this fact reasonably be derived from or inspired by this website content? Even loose topical relevance counts. Answer only 'Yes', 'No', or 'Page Not Available'."
                    }]
                }
            ]
            vlm_tasks.append({'id': task_id, 'messages': messages})

    print(f"Sending {len(vlm_tasks)} facts to LLM for verification...")
    if vlm_tasks:
        llm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=5)
        vlm_results.update(llm_results)

        # Retry failed facts once (LLM non-determinism causes false negatives)
        failed_tasks = [task for task in vlm_tasks if not vlm_results.get(task['id'], False)]
        if failed_tasks:
            print(f"Retrying {len(failed_tasks)} failed fact verifications...")
            retry_results = fast_parallel_vlm_calls(failed_tasks, model, max_workers=5)
            for task_id, verified in retry_results.items():
                if verified:
                    vlm_results[task_id] = True

        # For facts that still failed on short content (likely 404/error pages),
        # fall back to chemistry fact validation
        still_failed = [task_id for task_id, verified in vlm_results.items()
                        if not verified and url_content_map.get(task_id, 0) < 500]
        for task_id in still_failed:
            fact_text = fact_task_map.get(task_id, "")
            unavailable_page_facts.append((task_id, fact_text))
            del vlm_results[task_id]
            print(f"  Page likely unavailable (short content) for: {fact_text[:60]}...")

    # For unavailable pages, verify facts are valid chemistry facts instead
    if unavailable_page_facts:
        print(f"  {len(unavailable_page_facts)} facts from unavailable pages — verifying as chemistry facts...")
        chem_tasks = []
        for task_id, fact in unavailable_page_facts:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are verifying whether a statement is a valid fact about chemistry. Answer 'Yes' if it is a factually accurate statement related to chemistry, chemical reactions, chemical properties, or science experiments involving chemistry. Answer 'No' only if the statement is clearly false or completely unrelated to chemistry. Answer only 'Yes' or 'No'."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Is this a valid chemistry fact?\n\n\"{fact}\"\n\nAnswer only Yes or No."}]
                }
            ]
            chem_tasks.append({'id': task_id, 'messages': messages})
        chem_results = fast_parallel_vlm_calls(chem_tasks, model, max_workers=5)
        for task_id, verified in chem_results.items():
            vlm_results[task_id] = verified

    for task_id, verified in vlm_results.items():
        if verified:
            facts_verified_on_website += 1
        else:
            fact_text = fact_task_map.get(task_id, task_id)
            url_part = task_id.split("|")[0] if "|" in task_id else "unknown"
            print(f"  FAILED: [{url_part[:60]}] {fact_text}")
            verification_failure_examples.append(f"{fact_text[:40]}... (Not Found)")

    # ------------------- FINAL SCORING -------------------

    # Step 1: Fact Count
    if total_websites == 0:
        fact_count_score = 0
        fact_count_msg = "No websites found to check fact counts."
        fact_count_passed = False
    else:
        fact_count_score = round((websites_with_5plus_facts / total_websites) * 15)
        fact_count_passed = (websites_with_5plus_facts == total_websites)
        fact_count_msg = f"{websites_with_5plus_facts}/{total_websites} websites have 5+ facts."

    checkpoint.add_step(
        "Websites have 5+ facts",
        fact_count_passed,
        1,
        fact_count_msg,
        score=fact_count_score,
        max_score=15
    )

    # Step 2: Fact Verification
    if total_facts_checked == 0:
        fact_verification_score = 0
        fact_msg = "No facts found to verify on websites."
        fact_passed = False
    else:
        fact_verification_score = round((facts_verified_on_website / total_facts_checked) * 15)
        if facts_verified_on_website == total_facts_checked:
            fact_msg = f"All {total_facts_checked} facts verified on their websites."
            fact_passed = True
        else:
            example_failures = "; ".join(verification_failure_examples[:2])
            fact_msg = f"{facts_verified_on_website}/{total_facts_checked} facts verified. Failures: {example_failures}"
            fact_passed = False

    checkpoint.add_step(
        "Facts exist on websites",
        fact_passed,
        2,
        fact_msg,
        score=fact_verification_score,
        max_score=15
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4 (20pt): Color-Coded Summary Facts

    Validates that facts are copied correctly into the summary and
    color-coded consistently by topic.

    Checkpoint criteria mapping (from checkpoints.md):
    - All facts copied correctly → Step 1 (10pt - Global score)
    - Facts color-coded by topic → Step 2 (10pt - Global score)
    """
    print("\n----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Color-Coded Summary Facts")

    # Constants
    FUZZY_THRESHOLD = 85

    # Get main topics
    main_topics = []
    if bullet_hierarchy and 'topics' in bullet_hierarchy:
        main_topics = [
            item for item in bullet_hierarchy['topics']
            if item.get('nesting_level', 0) == 0
        ]

    if not main_topics:
        checkpoint.add_step(
            "Extract Topics",
            False,
            1,
            "No main topics found - cannot validate summary facts",
            score=0,
            max_score=10,
            execution_time=time.time() - checkpoint_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # ------------------- PHASE 1: DATA EXTRACTION -------------------

    # Extract original facts by topic
    original_facts_by_topic = {}
    for topic in main_topics:
        topic_name = topic['text']
        topic_facts = []

        # Level 1: websites
        for website in topic.get('children', []):
            if website.get('nesting_level') == 1:
                # Level 2: facts (cap at 5 per website to avoid summary pollution)
                # extract_bullet_hierarchy_from_doc may not stop at the summary
                # heading, causing summary bullets to be appended to the last website
                facts_added = 0
                for fact in website.get('children', []):
                    if fact.get('nesting_level') == 2:
                        fact_text = fact['text'].strip()
                        if fact_text:
                            topic_facts.append(fact_text)
                            facts_added += 1
                            if facts_added >= 5:
                                break

        original_facts_by_topic[topic_name] = topic_facts

    # Extract summary facts directly from doc_content (paragraph-level, both color types)
    summary_facts_list = extract_summary_facts_with_colors(doc_content)
    # ALL summary facts (for Step 1 - fact accuracy check)
    all_summary_facts = [item['text'] for item in summary_facts_list]
    # Only colored summary facts (for Step 2 - color consistency check)
    summary_facts_with_colors = {item['text']: item['color'] for item in summary_facts_list if item['color'] is not None}
    print(f"Found {len(all_summary_facts)} facts in summary section ({len(summary_facts_with_colors)} with color)")

    # Check if any summary facts exist at all
    if not all_summary_facts:
        checkpoint.add_step(
            "Facts Copied Correctly",
            False,
            1,
            "No facts found in summary section. Summary section may be missing.",
            score=0,
            max_score=10
        )
        checkpoint.add_step(
            "Color Consistency by Topic",
            False,
            2,
            "No summary facts found to validate color consistency.",
            score=0,
            max_score=10
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # ------------------- STEP 1: FACT ACCURACY -------------------

    step1_start = time.time()

    total_original_facts = 0
    facts_found_in_summary = 0
    missing_fact_examples = []

    for topic_name, original_facts in original_facts_by_topic.items():
        for original_fact in original_facts:
            total_original_facts += 1

            # Fuzzy match against ALL summary facts (regardless of color)
            found = False
            for summary_fact in all_summary_facts:
                is_match, _ = fuzzy_match_text(
                    original_fact,
                    summary_fact,
                    threshold=FUZZY_THRESHOLD
                )

                if is_match:
                    found = True
                    break

            if found:
                facts_found_in_summary += 1
            else:
                fact_preview = original_fact[:40] + "..." if len(original_fact) > 40 else original_fact
                missing_fact_examples.append(fact_preview)

    # Calculate Step 1 score
    if total_original_facts == 0:
        accuracy_score = 0
        accuracy_msg = "No facts found in bullet hierarchy to validate"
        accuracy_passed = False
    else:
        accuracy_score = (facts_found_in_summary / total_original_facts) * 10
        accuracy_passed = (facts_found_in_summary == total_original_facts)

        if accuracy_passed:
            accuracy_msg = f"All {total_original_facts} facts found in color-coded summary"
        else:
            missing_count = total_original_facts - facts_found_in_summary
            examples = "; ".join(missing_fact_examples[:2])
            accuracy_msg = f"{facts_found_in_summary}/{total_original_facts} facts found. {missing_count} missing (e.g., {examples})"

    step1_time = time.time() - step1_start

    checkpoint.add_step(
        "Facts Copied Correctly",
        accuracy_passed,
        1,
        accuracy_msg,
        score=accuracy_score,
        max_score=10,
        execution_time=step1_time
    )

    # ------------------- STEP 2: COLOR CONSISTENCY -------------------

    step2_start = time.time()

    # If no colored facts at all, Step 2 fails early
    if not summary_facts_with_colors:
        checkpoint.add_step(
            "Color Consistency by Topic",
            False,
            2,
            f"No color-coded facts found in summary section ({len(all_summary_facts)} facts exist but none have color metadata).",
            score=0,
            max_score=10,
            execution_time=time.time() - step2_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Build mapping: original_fact -> (topic_name, summary_fact, color)
    fact_color_mapping = {}

    for topic_name, original_facts in original_facts_by_topic.items():
        for original_fact in original_facts:
            # Find matching summary fact
            for summary_fact, color in summary_facts_with_colors.items():
                is_match, _ = fuzzy_match_text(
                    original_fact,
                    summary_fact,
                    threshold=FUZZY_THRESHOLD
                )

                if is_match:
                    fact_color_mapping[original_fact] = {
                        'topic': topic_name,
                        'summary_fact': summary_fact,
                        'color': color
                    }
                    break

    # Check consistency: each topic uses exactly one color
    topic_colors = {}
    for fact_data in fact_color_mapping.values():
        topic = fact_data['topic']
        color = fact_data['color']

        if topic not in topic_colors:
            topic_colors[topic] = set()
        topic_colors[topic].add(color)

    # Validate one color per topic
    total_topics_checked = len(original_facts_by_topic)
    topics_with_consistent_coloring = 0
    color_violation_examples = []

    for topic_name in original_facts_by_topic.keys():
        colors_used = topic_colors.get(topic_name, set())

        if len(colors_used) == 1:
            topics_with_consistent_coloring += 1
        elif len(colors_used) > 1:
            color_hex_list = [rgb_to_hex(*c) for c in colors_used]
            topic_preview = topic_name[:25] + "..." if len(topic_name) > 25 else topic_name
            color_violation_examples.append(
                f"'{topic_preview}' uses {len(colors_used)} colors: {', '.join(color_hex_list)}"
            )

    # Check for color collisions (same color for multiple topics)
    color_to_topics = {}
    for topic_name, colors in topic_colors.items():
        if len(colors) == 1:
            color = list(colors)[0]
            if color not in color_to_topics:
                color_to_topics[color] = []
            color_to_topics[color].append(topic_name)

    for color, topics in color_to_topics.items():
        if len(topics) > 1:
            color_hex = rgb_to_hex(*color)
            topic_names = ", ".join([t[:20] + "..." if len(t) > 20 else t for t in topics])
            color_violation_examples.append(
                f"Color {color_hex} shared by {len(topics)} topics: {topic_names}"
            )
            # Penalize: only one topic gets credit
            topics_with_consistent_coloring -= (len(topics) - 1)

    # Ensure non-negative
    topics_with_consistent_coloring = max(0, topics_with_consistent_coloring)

    # Calculate Step 2 score
    if total_topics_checked == 0:
        color_score = 0
        color_msg = "No topics found to check color consistency"
        color_passed = False
    else:
        color_score = (topics_with_consistent_coloring / total_topics_checked) * 10
        color_passed = (topics_with_consistent_coloring == total_topics_checked)

        if color_passed:
            color_msg = f"All {total_topics_checked} topics properly color-coded with unique colors"
        else:
            proper_count = topics_with_consistent_coloring
            examples = "; ".join(color_violation_examples[:3])
            color_msg = f"{proper_count}/{total_topics_checked} topics properly color-coded. Issues: {examples}"

    step2_time = time.time() - step2_start

    checkpoint.add_step(
        "Color Consistency by Topic",
        color_passed,
        2,
        color_msg,
        score=color_score,
        max_score=10,
        execution_time=step2_time
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Checkpoint 5 (20pt): Check Topic Images

    Validates that each topic has a matching image, the images are at the bottom
    of the last page, arranged side by side, and each image is related to its topic.

    Checkpoint criteria mapping (from checkpoints.md):
    - Every topic has an image → Step 1 (5pt)
    - Images at bottom of last page → Step 2 (5pt)
    - Images aligned side by side → Step 3 (5pt)
    - Each image is related to its topic → Step 4 (5pt)
    """
    print("\n----------------- CHECKPOINT 5 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Topic Images")

    global model

    # Get main topics
    main_topics = []
    if bullet_hierarchy and 'topics' in bullet_hierarchy:
        main_topics = [
            item for item in bullet_hierarchy['topics']
            if item.get('nesting_level', 0) == 0
        ]

    if not main_topics:
        checkpoint.add_step(
            "Extract Topics",
            False,
            1,
            "No main topics found - cannot validate images",
            score=0,
            max_score=5,
            execution_time=time.time() - checkpoint_start
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_topics = len(main_topics)
    topic_names = [t['text'] for t in main_topics]

    # ------------------- STEP 1: IMAGE COUNT -------------------
    step1_start = time.time()
    inline_objects = (doc_content.get('inlineObjects', {}) or {}) if doc_content else {}
    positioned_objects = (doc_content.get('positionedObjects', {}) or {}) if doc_content else {}
    total_images = len(inline_objects) + len(positioned_objects)

    image_count_passed = (total_images == num_topics)
    if image_count_passed:
        image_count_msg = f"Found {total_images} images matching {num_topics} topics"
        image_count_score = 5
    else:
        matched = min(total_images, num_topics)
        image_count_score = (matched / num_topics) * 5 if num_topics > 0 else 0
        image_count_msg = f"Found {total_images} images, expected {num_topics} (one per topic)"

    checkpoint.add_step(
        "Image Count Matches Topics",
        image_count_passed,
        1,
        image_count_msg,
        score=image_count_score,
        max_score=5,
        execution_time=time.time() - step1_start
    )

    # ------------------- SETUP FOR VISUAL CHECKS -------------------
    if model is None:
        model = load_model(model_id)
        print(f"Loaded model: {model_id}")

    page_image_files = sorted(glob.glob(os.path.join(PDF_IMAGES_DIR, "page_*.png")))

    downloaded_images = []
    if total_images > 0:
        try:
            extract_images_from_doc_extended(
                DOCS_SERVICE, output_dir=IMAGES_DIR,
                document=doc_content, include_positioned=True
            )
            downloaded_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "image_*.*")))
            print(f"Downloaded {len(downloaded_images)} images from document")
        except Exception as e:
            print(f"Error downloading images: {e}")

    # ------------------- STEP 3: IMAGES SIDE BY SIDE (extract_image_location) ---
    alignment_result = None
    if downloaded_images and page_image_files:
        alignment_result = check_images_aligned_horizontally(
            downloaded_images, PDF_IMAGES_DIR, doc_content=doc_content, dpi=PDF_DPI
        )

    # ------------------- BUILD ALL VLM TASKS (PARALLEL) -------------------
    vlm_tasks = []

    # Step 4: topic relevance — one task per (topic × downloaded image) pair
    for topic_name in topic_names:
        for img_path in downloaded_images:
            task_id = f"topic_relevance|{topic_name}|{img_path}"
            vlm_tasks.append({
                'id': task_id,
                'messages': [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are checking if an image is related to a given chemistry topic. Answer only 'Yes' or 'No'."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": f"Is this image related to the chemistry topic '{topic_name}'? Answer only Yes or No."}
                        ]
                    }
                ]
            })

    print(f"Running {len(vlm_tasks)} image validation checks in parallel...")
    vlm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=5)

    # ------------------- STEP 2: IMAGES AT BOTTOM (programmatic) -------------------
    # is_lower() assumes 300 DPI (page 2550x3300). Scale for PDF_DPI.
    from src.browsergym.eval.eval_utils.utils import location as Location
    scale = PDF_DPI / 300

    image_locations = alignment_result.get('locations', []) if alignment_result else []
    if image_locations:
        image_page = image_locations[0].page_number
        lower_region = Location(image_page, 0, int(2200 * scale), int(2550 * scale), int(1100 * scale))
        images_at_bottom = [loc for loc in image_locations if loc.is_mostly_inside(lower_region)]
        bottom_passed = len(images_at_bottom) == len(image_locations)
        bottom_msg = (f"{len(images_at_bottom)}/{len(image_locations)} images in lower region of last page"
                      if not bottom_passed
                      else f"All {len(image_locations)} images found in lower region of last page")
    else:
        bottom_passed = False
        bottom_msg = "No image locations found for position validation"

    checkpoint.add_step(
        "Images at Bottom of Last Page",
        bottom_passed,
        2,
        bottom_msg,
        score=5 if bottom_passed else 0,
        max_score=5
    )

    # ------------------- STEP 3: IMAGES SIDE BY SIDE -------------------
    if alignment_result is not None:
        side_by_side_passed = alignment_result['aligned']
        side_by_side_msg = alignment_result['details']
    else:
        side_by_side_passed = False
        side_by_side_msg = "No images or PDF pages available for alignment check"

    checkpoint.add_step(
        "Images Arranged Side by Side",
        side_by_side_passed,
        3,
        side_by_side_msg,
        score=5 if side_by_side_passed else 0,
        max_score=5
    )

    # ------------------- STEP 4: TOPIC RELEVANCE -------------------
    step4_start = time.time()
    if not downloaded_images:
        checkpoint.add_step(
            "Images Related to Topics",
            False,
            4,
            "No images downloaded from document for topic relevance check",
            score=0,
            max_score=5
        )
    else:
        topics_with_match = 0
        unmatched_topics = []
        for topic_name in topic_names:
            topic_has_match = any(
                vlm_results.get(f"topic_relevance|{topic_name}|{img_path}", False)
                for img_path in downloaded_images
            )
            if topic_has_match:
                topics_with_match += 1
            else:
                unmatched_topics.append(topic_name[:30])

        relevance_score = (topics_with_match / num_topics) * 5 if num_topics > 0 else 0
        relevance_passed = (topics_with_match == num_topics)

        if relevance_passed:
            relevance_msg = f"All {num_topics} topics have a matching image"
        else:
            unmatched = "; ".join(unmatched_topics[:2])
            relevance_msg = (f"{topics_with_match}/{num_topics} topics have a matching image. "
                             f"No match for: {unmatched}")

        checkpoint.add_step(
            "Images Related to Topics",
            relevance_passed,
            4,
            relevance_msg,
            score=relevance_score,
            max_score=5,
            execution_time=time.time() - step4_start
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history_list=None):
    """
    Grade all checkpoints for the education lesson plan document.

    Args:
        workspace_doc_id (str): The Google Docs document ID to evaluate.
        cached_models (dict, optional): Preloaded models by model_id.
        browsing_history_list (list, optional): List of URLs visited by agent.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    try:
        # Setup document processing
        setup_document(workspace_doc_id)

        # Store browsing history globally for URL validation (Checkpoint 2)
        global browsing_history
        browsing_history = browsing_history_list or []

        # Use cached model if available
        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        # Grade checkpoints
        checkpoints: List[Checkpoint] = []
        checkpoints.append(grade_checkpoint_1())
        checkpoints.append(grade_checkpoint_2())  # Website URLs validation
        checkpoints.append(grade_checkpoint_3())  # Website facts validation
        checkpoints.append(grade_checkpoint_4())  # Color-coded summary validation
        checkpoints.append(grade_checkpoint_5())  # Topic images

        total_execution_time = time.time() - total_start_time
        result = Result(checkpoints, total_execution_time=total_execution_time)

        return result

    finally:
        # Cleanup generated files
        if CLEANUP_ENABLED:
            cleanup_generated_files()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate education lesson plan document")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                       help="Google Docs document ID to evaluate")
    parser.add_argument("--browsing_history_doc_id", type=str, default=None,
                       help="Google Doc ID containing browsing history URLs (one per line)")
    parser.add_argument("--cached_models", type=dict, default=None,
                       help="Dictionary of preloaded models")
    args = parser.parse_args()

    start_time = time.time()

    print(f"DEBUG mode: {DEBUG}")
    print(f"CLEANUP enabled: {CLEANUP_ENABLED}")

    # Extract browsing history from Google Doc if provided
    browsing_history_list = None
    if args.browsing_history_doc_id:
        all_links = extract_hyperlinks_from_doc(args.browsing_history_doc_id, DOCS_SERVICE)
        browsing_history_list = [link['url'] for link in all_links if link['url'].startswith('http')]
        print(f"Loaded {len(browsing_history_list)} URLs from browsing history doc")

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        cached_models=args.cached_models,
        browsing_history_list=browsing_history_list
    )

    print("\n=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()

    for checkpoint_data in detailed_report["checkpoints"]:
        print(f"\n{checkpoint_data['name']}: {checkpoint_data['score']}")
        for step in checkpoint_data["steps"]:
            status = "[PASS]" if step["success"] else "[FAIL]"
            print(f"  {status} {step['name']}: {step['details'] or 'No details'}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
