import os
import sys
import time
import shutil
import traceback
import glob
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

# Base path used for sys.path injection so Docker and CHPC layouts both resolve.
if os.path.exists("/app/src"):
    BASE_PATH = "/app"
elif os.path.exists("/scratch"):
    BASE_PATH = os.path.expanduser("~/Agent-Benchmark")
else:
    BASE_PATH = os.getcwd()
sys.path.append(BASE_PATH)

# Core evaluation imports
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint, Result, calculate_percentage_score
from src.browsergym.knows.eval.eval_utils.google_services_utils import (
    initialize_google_services,
    get_doc_content,
    download_doc_as_pdf,
    extract_images_from_doc_extended,
    extract_hyperlinks_from_doc,
)
from src.browsergym.knows.eval.eval_utils.text_utils import (
    fuzzy_match_text,
    text_fuzzy_match_contained_long,
)
from src.browsergym.knows.eval.eval_utils.image_utils import convert_pdf_to_pngs
from src.browsergym.knows.eval.eval_utils.models import load_model
from src.browsergym.knows.eval.eval_utils.utils import location as Location
from src.browsergym.knows.eval.eval_utils.web_utils import (
    validate_url_accessible,
    normalize_url_for_comparison,
    fetch_with_fallbacks,
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import (
    parallel_execute,
    fast_parallel_vlm_calls,
)

# Task-specific utilities
from src.browsergym.knows.eval.tasks.docs_31_education_lesson_plan.utils import (
    parse_subject_and_audience,
    validate_topics_subject_related,
    validate_topics_engaging,
    count_unique_normalized,
    check_images_aligned_horizontally,
    images_in_single_table_row,
    extract_summary_facts_with_colors,
    classify_summary_paragraphs_as_facts,
    extract_bullet_hierarchy_from_doc,
    format_color,
    get_main_topics,
    get_doc_page_dimensions_px,
    dedup_files_by_sha256,
    vlm_uniform_failure_warning,
    LOWER_REGION_PAGE_FRACTION,
)

# ---------------------------------------------------------------------------
# Directory paths — derived from this file's location so the evaluator works
# regardless of where BASE_PATH points.
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
PDF_IMAGES_DIR = os.path.join(DATA_DIR, "pdf_images")
IMAGES_DIR = os.path.join(DATA_DIR, "doc_images")
TASK_MD_PATH = os.path.join(TASK_DIR, "task.md")

# ---------------------------------------------------------------------------
# Runtime configuration
CLEANUP_ENABLED = os.environ.get("CLEANUP", "True").lower() == "true"
PDF_DPI = 150

# ---------------------------------------------------------------------------
# Spec thresholds (from task.md / checkpoints.md)
MIN_TOPICS = 2
MAX_TOPICS = 3
MIN_WEBSITES_PER_TOPIC = 4
MIN_FACTS_PER_WEBSITE = 5

# Fuzzy-match thresholds
FACT_DEDUP_FUZZY_THRESHOLD = 90   # CP3 within-website fact uniqueness
FACT_PRESENCE_FUZZY_THRESHOLD = 75   # CP4 Step 1: did agent copy this fact?
COLOR_MATCH_FUZZY_THRESHOLD = 85     # CP4 Steps 3/4: which paragraph contains this fact?

# Concurrency / timeouts
URL_VALIDATE_WORKERS = 4
URL_FETCH_WORKERS = 3
VLM_WORKERS = 5
URL_VALIDATE_TIMEOUT_S = 10

# LLM prompt/content sizing
WEBSITE_CONTENT_FETCH_CHARS = 15000   # how much each URL body we cache
WEBSITE_CONTENT_EXCERPT_CHARS = 1500  # how much we paste into the relevance prompt

# Display caps for failure-message tails
MAX_FAILURE_EXAMPLES = 2
MAX_VIOLATION_EXAMPLES = 3

# Model configuration
model = None
model_id = "gemini-3-flash-google-ai"  # Fast and capable for topic/fact validation

# Lazy-initialized in setup_document so missing task.md or Google credentials
# don't crash module import. "(unknown)" surfaces clearly in step details.
SUBJECT = "(unknown)"
AUDIENCE_LEVEL = "(unknown)"
DRIVE_SERVICE = None
DOCS_SERVICE = None

# Global variables set by setup_document()
doc_id = None
doc_content = None        # Full document JSON from get_doc_content
bullet_hierarchy = None   # Hierarchical bullet structure
browsing_history = None   # Will be passed from grade_checkpoints()
website_content_cache = {}  # url -> (content_str, status_str); populated by CP2, reused by CP3


def setup_document(workspace_doc_id):
    """Set up document processing for the provided workspace_doc_id.

    Lazy-initializes subject/audience and Google services on first call.
    Populates the module globals doc_id, doc_content, and bullet_hierarchy
    that the grade_checkpoint_N functions read.
    """
    global doc_id, doc_content, bullet_hierarchy
    global SUBJECT, AUDIENCE_LEVEL, DRIVE_SERVICE, DOCS_SERVICE

    if not workspace_doc_id:
        raise ValueError("workspace_doc_id is required")

    if SUBJECT == "(unknown)":
        try:
            SUBJECT, AUDIENCE_LEVEL = parse_subject_and_audience(TASK_MD_PATH)
        except Exception as e:
            print(f"Warning: could not parse subject/audience from task.md: {e}")
    if DRIVE_SERVICE is None or DOCS_SERVICE is None:
        DRIVE_SERVICE, DOCS_SERVICE = initialize_google_services(service_type="docs")

    print(f"Using workspace document ID: {workspace_doc_id}")
    doc_id = workspace_doc_id

    # Phase 1: download PDF and fetch document JSON (sequential, both required).
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(DATA_DIR, "lesson_plan.pdf")
    download_doc_as_pdf(doc_id, pdf_path, DRIVE_SERVICE)
    doc_content = get_doc_content(doc_id, DOCS_SERVICE)

    # Phase 2: parse bullets + rasterize PDF in parallel; each future is
    # independently wrapped so one failure doesn't take down the other.
    with ThreadPoolExecutor(max_workers=2) as executor:
        bullets_future = executor.submit(extract_bullet_hierarchy_from_doc, doc_content)
        pngs_future = executor.submit(convert_pdf_to_pngs, pdf_path, PDF_IMAGES_DIR, PDF_DPI)

        try:
            bullet_hierarchy = bullets_future.result()
        except Exception as e:
            print(f"setup_document: bullet hierarchy parse failed: {e}")
            bullet_hierarchy = {'topics': []}
        try:
            pngs_future.result()
        except Exception as e:
            print(f"setup_document: PDF rasterization failed (CP5 will degrade): {e}")

    num_topics = len(bullet_hierarchy.get('topics', [])) if bullet_hierarchy else 0
    print(f"Setup complete: extracted {num_topics} top-level items")


def grade_checkpoint_1():
    """
    Checkpoint 1 (15pt): Related Topics — step order matches checkpoints.md
    bullet order:
      1. Topics related to {SUBJECT}        (4pt)
      2. Topic count 2-3                    (3pt)
      3. Topics unique                      (2pt)
      4. Topics appropriate for audience    (3pt)
      5. Topics engaging/fun                (3pt)
    """
    print("\n----------------- CHECKPOINT 1 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=15, result=0, name="Related Topics")

    global model

    main_topics = get_main_topics(bullet_hierarchy)

    if not main_topics:
        empty_reason = (
            "Document has no main-level bulleted topics. Expected 2-3 topics as top-level bullets."
        )
        for name, step_id, max_score in [
            (f"Topics Related to {SUBJECT}", 1, 4),
            (f"Topic Count ({MIN_TOPICS}-{MAX_TOPICS})", 2, 3),
            ("Topics Unique", 3, 2),
            (f"Topics Appropriate for {AUDIENCE_LEVEL}", 4, 3),
            ("Topics Engaging", 5, 3),
        ]:
            checkpoint.add_step(
                name, False, step_id, empty_reason, score=0, max_score=max_score,
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_topics = len(main_topics)
    topic_texts = [t['text'].strip() for t in main_topics]

    # Lazy-load model. On failure leave model=None so per-step try/except
    # surfaces "could not run" without taking down Steps 2/3 (pure logic).
    if model is None:
        try:
            model = load_model(model_id)
            print(f"Loaded model: {model_id}")
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to load model {model_id}: {e}")
            model = None

    # Step 1 (4pt). Subject-only — audience appropriateness is Step 4.
    try:
        if model is None:
            raise RuntimeError(f"LLM model {model_id} unavailable")
        subject_result = validate_topics_subject_related(
            topic_texts, SUBJECT, model, max_workers=VLM_WORKERS,
        )
        if subject_result['all_valid']:
            checkpoint.add_step(
                f"Topics Related to {SUBJECT}", True, 1,
                f"All {num_topics} topics validated as {SUBJECT}-related: {', '.join(topic_texts)}",
                score=4, max_score=4,
            )
        else:
            # Count per-topic-instance (not per-unique-name) so duplicate-named
            # topics each contribute their own credit. CP1 Step 3 handles the
            # duplication penalty separately.
            valid_count = sum(1 for t in topic_texts
                              if subject_result['validations'].get(t, False))
            invalid = [t for t in topic_texts if not subject_result['validations'].get(t)]
            score = calculate_percentage_score(valid_count, num_topics, 4)
            warning = vlm_uniform_failure_warning(subject_result['validations'])
            details = (f"{valid_count}/{num_topics} topics validated. "
                       f"Not {SUBJECT}-related: {', '.join(invalid)}")
            if warning:
                details += f" {warning}"
            checkpoint.add_step(
                f"Topics Related to {SUBJECT}", False, 1,
                details, score=score, max_score=4,
            )
    except Exception as e:
        traceback.print_exc()
        checkpoint.add_step(
            f"Topics Related to {SUBJECT}", False, 1,
            f"Subject-relevance check could not run: {e}",
            score=0, max_score=4,
        )

    # Step 2: count (3pt)
    count_valid = MIN_TOPICS <= num_topics <= MAX_TOPICS
    checkpoint.add_step(
        f"Topic Count ({MIN_TOPICS}-{MAX_TOPICS})", count_valid, 2,
        f"Found {num_topics} main-level topics"
        + ("" if count_valid else f" (expected {MIN_TOPICS}-{MAX_TOPICS})"),
        score=3 if count_valid else 0, max_score=3,
    )

    # Step 3: uniqueness (2pt)
    unique_count, total_count, dupes = count_unique_normalized(topic_texts)
    if not dupes:
        checkpoint.add_step(
            "Topics Unique", True, 3,
            f"All {total_count} topic names are unique",
            score=2, max_score=2,
        )
    else:
        score = calculate_percentage_score(unique_count, total_count, 2)
        checkpoint.add_step(
            "Topics Unique", False, 3,
            f"{unique_count}/{total_count} unique topics; duplicates: {', '.join(dupes[:3])}",
            score=score, max_score=2,
        )

    # Step 4: audience appropriateness (3pt) — explicit standalone step.
    try:
        if model is None:
            raise RuntimeError(f"LLM model {model_id} unavailable")
        audience_tasks = [
            {
                'id': topic,
                'messages': [
                    {"role": "system", "content": [{"type": "text", "text": f"You are evaluating whether a {SUBJECT} topic is appropriate for {AUDIENCE_LEVEL} students. Answer only 'Yes' or 'No'."}]},
                    {"role": "user", "content": [{"type": "text", "text": (
                        f"Is the topic '{topic}' appropriate in depth and complexity for "
                        f"{AUDIENCE_LEVEL} students learning {SUBJECT}? Answer only 'Yes' or 'No'."
                    )}]},
                ],
            }
            for topic in topic_texts
        ]
        audience_results = fast_parallel_vlm_calls(audience_tasks, model, max_workers=VLM_WORKERS)
        audience_passed = sum(1 for t in topic_texts if audience_results.get(t, False))
        audience_invalid = [t for t in topic_texts if not audience_results.get(t, False)]
        if audience_passed == num_topics:
            checkpoint.add_step(
                f"Topics Appropriate for {AUDIENCE_LEVEL}", True, 4,
                f"All {num_topics} topics judged appropriate for {AUDIENCE_LEVEL}",
                score=3, max_score=3,
            )
        else:
            score = calculate_percentage_score(audience_passed, num_topics, 3)
            warning = vlm_uniform_failure_warning(audience_results)
            details = (f"{audience_passed}/{num_topics} topics appropriate. "
                       f"Inappropriate: {', '.join(audience_invalid)}")
            if warning:
                details += f" {warning}"
            checkpoint.add_step(
                f"Topics Appropriate for {AUDIENCE_LEVEL}", False, 4,
                details, score=score, max_score=3,
            )
    except Exception as e:
        traceback.print_exc()
        checkpoint.add_step(
            f"Topics Appropriate for {AUDIENCE_LEVEL}", False, 4,
            f"Audience-appropriateness check could not run: {e}",
            score=0, max_score=3,
        )

    # Step 5: engagement / "fun" (3pt)
    try:
        if model is None:
            raise RuntimeError(f"LLM model {model_id} unavailable")
        engagement_result = validate_topics_engaging(
            topic_texts, AUDIENCE_LEVEL, model, max_workers=VLM_WORKERS,
        )
        # Count per-topic-instance (mirrors Step 1/Step 4 — duplicate-named
        # topics each contribute credit; CP1 Step 3 penalizes duplication).
        eng_passed = sum(1 for t in topic_texts
                         if engagement_result['validations'].get(t, False))
        eng_invalid = [t for t in topic_texts if not engagement_result['validations'].get(t)]
        if engagement_result['all_valid']:
            checkpoint.add_step(
                "Topics Engaging", True, 5,
                f"All {num_topics} topics judged engaging for {AUDIENCE_LEVEL} students",
                score=3, max_score=3,
            )
        else:
            score = calculate_percentage_score(eng_passed, num_topics, 3)
            warning = vlm_uniform_failure_warning(engagement_result['validations'])
            details = (f"{eng_passed}/{num_topics} topics engaging. "
                       f"Flagged as unengaging: {', '.join(eng_invalid)}")
            if warning:
                details += f" {warning}"
            checkpoint.add_step(
                "Topics Engaging", False, 5,
                details, score=score, max_score=3,
            )
    except Exception as e:
        traceback.print_exc()
        checkpoint.add_step(
            "Topics Engaging", False, 5,
            f"Engagement check could not run: {e}",
            score=0, max_score=3,
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_2():
    """
    Checkpoint 2 (45pt): Website URLs match — step order matches checkpoints.md:
      1. ≥4 websites per topic               (10pt)
      2. URLs unique within topic            (5pt)
      3. Each website valid                  (10pt)
      4. Agent visited each website          (10pt)
      5. Each website related to topic       (10pt)
    """
    print("\n----------------- CHECKPOINT 2 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=45, result=0, name="Website URLs Match")

    global browsing_history

    # Step accumulators
    url_count_per_topic = {}
    failed_website_per_topic = {}
    total_urls_found = 0
    total_urls_visited = 0
    total_relevance_checked = 0
    total_relevance_passed = 0
    relevance_failure_examples = []

    if browsing_history is None:
        browsing_history = []
        print("Warning: No browsing history provided")

    normalized_browsing = {
        normalize_url_for_comparison(url) for url in browsing_history if url
    }

    # Get main topics from Checkpoint 1
    main_topics = get_main_topics(bullet_hierarchy)

    if not main_topics:
        for name, step_id, max_score in [
            (f"All topics have {MIN_WEBSITES_PER_TOPIC}+ links", 1, 10),
            ("URLs Unique Within Topic", 2, 5),
            ("Website Validity", 3, 10),
            ("Agent Visited Websites", 4, 10),
            ("Website Relevance", 5, 10),
        ]:
            checkpoint.add_step(
                name, False, step_id,
                "No main topics found in document - cannot validate URLs",
                score=0, max_score=max_score,
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    urls_to_check_per_topic = {}     # topic_idx -> (topic_dict, [urls])
    all_urls_for_cache = set()        # unique URLs across all topics, for content fetch
    topics_with_unique_urls = 0
    url_dup_examples = []

    # Track website bullets and URLs separately so an un-linked bullet still
    # counts toward MIN_WEBSITES_PER_TOPIC and fails validity.
    topic_urls_map = {}                  # topic_idx -> list of URLs
    websites_without_url_per_topic = {}  # topic_idx -> count
    for topic_idx, topic in enumerate(main_topics):
        topic_name = topic['text'][:40]
        print(f"Collecting URLs for Topic {topic_idx+1}: {topic_name}")

        level1_children = [c for c in topic.get('children', []) if c.get('nesting_level') == 1]
        topic_urls = [c['url'] for c in level1_children if 'url' in c]
        websites_without_url_per_topic[topic_idx] = len(level1_children) - len(topic_urls)
        topic_urls_map[topic_idx] = topic_urls
        all_urls_for_cache.update(url for url in topic_urls if url and url != 'No URL')

        url_count_per_topic[topic_idx] = len(level1_children)
        total_urls_found += len(level1_children)

        # New step: URL uniqueness within topic
        unique_count, total_count, dupes = count_unique_normalized(
            topic_urls, normalizer=normalize_url_for_comparison
        )
        if not dupes:
            topics_with_unique_urls += 1
        else:
            url_dup_examples.append(f"Topic '{topic_name}': {len(dupes)} duplicate URL(s)")

        # Step 3: visitation
        for url in topic_urls:
            if normalize_url_for_comparison(url) in normalized_browsing:
                total_urls_visited += 1

    # Validate URLs in parallel so one hung host doesn't block the evaluator.
    all_urls = sorted({url for urls in topic_urls_map.values() for url in urls if url})
    validity_tasks = [
        {'id': url, 'func': validate_url_accessible, 'args': (url,),
         'kwargs': {'timeout': URL_VALIDATE_TIMEOUT_S}}
        for url in all_urls
    ]
    print(f"Validating {len(validity_tasks)} URLs in parallel...")
    validity_results = (parallel_execute(validity_tasks, max_workers=URL_VALIDATE_WORKERS)
                        if validity_tasks else {})

    for topic_idx, topic_urls in topic_urls_map.items():
        accessible_urls = []
        failed_urls = []
        for url in topic_urls:
            result = validity_results.get(url)
            if result and result[0]:
                accessible_urls.append(url)
            else:
                details = result[1] if result else "Validation timed out or failed"
                failed_urls.append((url, details))
        # Website bullets without any URL count as validity failures.
        for _ in range(websites_without_url_per_topic.get(topic_idx, 0)):
            failed_urls.append(("(no URL)", "Website bullet has no URL"))
        if failed_urls:
            failed_website_per_topic[topic_idx] = failed_urls
        urls_to_check_per_topic[topic_idx] = (main_topics[topic_idx],
                                              accessible_urls[:MIN_WEBSITES_PER_TOPIC])

    # Fetch all site content into the shared cache (CP3 will reuse).
    fetch_tasks = [
        {'id': url, 'func': fetch_with_fallbacks, 'args': (url,),
         'kwargs': {'max_chars': WEBSITE_CONTENT_FETCH_CHARS}}
        for url in all_urls_for_cache
        if url not in website_content_cache
    ]
    if fetch_tasks:
        print(f"Fetching content for {len(fetch_tasks)} URLs ({URL_FETCH_WORKERS} workers)...")
        fetched = parallel_execute(fetch_tasks, max_workers=URL_FETCH_WORKERS)
        website_content_cache.update(fetched)

    vlm_tasks = []
    relevance_task_map = {}  # task_id -> url for failure reporting

    # task_id includes web_idx so duplicate URLs within a topic don't collide.
    for topic_idx, (topic, urls) in urls_to_check_per_topic.items():
        for web_idx, url in enumerate(urls):
            total_relevance_checked += 1
            task_id = f"topic_{topic_idx}|web{web_idx}|{url}"
            relevance_task_map[task_id] = url

            content_tuple = website_content_cache.get(url)
            content_excerpt = (content_tuple[0][:WEBSITE_CONTENT_EXCERPT_CHARS]
                               if content_tuple and content_tuple[0] else "")

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
    relevance_call_failed = False
    relevance_warning = None
    try:
        if model is None:
            raise RuntimeError(f"LLM model {model_id} unavailable")
        relevance_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=VLM_WORKERS)
        relevance_warning = vlm_uniform_failure_warning(relevance_results)
        for task_id, passed in relevance_results.items():
            if passed:
                total_relevance_passed += 1
            else:
                url = relevance_task_map[task_id]
                relevance_failure_examples.append(f"{url[:30]}... (Not Relevant)")
    except Exception as e:
        traceback.print_exc()
        print(f"URL relevance LLM call failed: {e}")
        relevance_call_failed = True

    # ------------------- FINAL SCORING -------------------

    # Part 1 Check: Topic URL Counts
    valid_topics = sum(1 for n in url_count_per_topic.values() if n >= MIN_WEBSITES_PER_TOPIC)
    topic_amount = len(url_count_per_topic)

    step_name_count = f"All topics have {MIN_WEBSITES_PER_TOPIC}+ links"
    if valid_topics == topic_amount:
        checkpoint.add_step(
            step_name_count, True, 1,
            "All topics have the correct amount of URLs",
            score=10, max_score=10,
        )
    else:
        improper_link_score = calculate_percentage_score(valid_topics, topic_amount, 10)
        checkpoint.add_step(
            step_name_count, False, 1,
            f"Only {valid_topics} out of {topic_amount} topics have "
            f"{MIN_WEBSITES_PER_TOPIC}+ links",
            score=improper_link_score, max_score=10,
        )

    # New Step: URL Uniqueness Within Topic (5pt)
    if topics_with_unique_urls == topic_amount:
        checkpoint.add_step(
            "URLs Unique Within Topic", True, 2,
            "All topics have unique URLs within their 4 websites",
            score=5, max_score=5,
        )
    else:
        unique_score = calculate_percentage_score(topics_with_unique_urls, topic_amount, 5)
        checkpoint.add_step(
            "URLs Unique Within Topic", False, 2,
            f"{topics_with_unique_urls}/{topic_amount} topics have unique URLs. "
            f"{'; '.join(url_dup_examples[:MAX_FAILURE_EXAMPLES])}",
            score=unique_score, max_score=5,
        )

    # Part 2 Check: Website Validity (per-website, not per-topic).
    total_websites = (sum(len(urls) for urls in topic_urls_map.values())
                      + sum(websites_without_url_per_topic.values()))
    failed_websites_flat = [
        (main_topics[idx]['text'], url, details)
        for idx, failures in failed_website_per_topic.items()
        for (url, details) in failures
    ]
    valid_websites = total_websites - len(failed_websites_flat)

    if total_websites == 0:
        validity_score = 0
        validity_passed = False
        validity_msg = "No websites found to validate"
    elif valid_websites == total_websites:
        validity_score = 10
        validity_passed = True
        validity_msg = f"All {total_websites} websites are valid and accessible"
    else:
        validity_score = calculate_percentage_score(valid_websites, total_websites, 10)
        validity_score = min(validity_score, 9)  # Cap imperfect at max-1.
        validity_passed = False
        examples = "; ".join(
            f"Topic '{topic_name[:30]}': {url} ({details})"
            for topic_name, url, details in failed_websites_flat[:MAX_FAILURE_EXAMPLES]
        )
        validity_msg = (f"{valid_websites}/{total_websites} websites are valid. "
                        f"Failures: {examples}")

    checkpoint.add_step(
        "Website Validity", validity_passed, 3,
        validity_msg, score=validity_score, max_score=10,
    )

    # Part 3 Check: Agent Visited Websites (Global Score)
    # Score = (Total Visited / Total Found) * 10
    if total_urls_found == 0:
        # Edge case: No URLs found at all
        visited_score = 0
        visited_msg = "No URLs found to check visitation."
    else:
        visited_score = calculate_percentage_score(total_urls_visited, total_urls_found, 10)
        visited_msg = f"Agent visited {total_urls_visited} out of {total_urls_found} total URLs found."

    checkpoint.add_step(
        "Agent Visited Websites",
        total_urls_visited == total_urls_found, # Pass only if 100% matched
        4,
        visited_msg,
        score = visited_score,
        max_score = 10
    )

    # Part 4 Check: URL Relevance (Global Score)
    # Score = (Total Relevant / Total Checked) * 10
    if relevance_call_failed:
        relevance_score = 0
        relevance_msg = "URL relevance check could not run: LLM call failed."
        passed = False
    elif total_relevance_checked == 0:
        relevance_score = 0
        relevance_msg = "No accessible URLs to check for relevance."
        passed = False
    else:
        relevance_score = calculate_percentage_score(total_relevance_passed, total_relevance_checked, 10)
        if total_relevance_passed == total_relevance_checked:
            relevance_msg = f"All {total_relevance_checked} valid websites are relevant."
            passed = True
        else:
            example_failures = "; ".join(relevance_failure_examples[:MAX_FAILURE_EXAMPLES])
            relevance_msg = (f"{total_relevance_passed}/{total_relevance_checked} "
                             f"websites relevant. Failures: {example_failures}")
            if relevance_warning:
                relevance_msg += f" {relevance_warning}"
            passed = False

    checkpoint.add_step(
        "Website Relevance",
        passed,
        5,
        relevance_msg,
        score = relevance_score,
        max_score = 10
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_3():
    """
    Checkpoint 3 (35pt): Website Facts are valid — step order matches checkpoints.md:
      1. ≥5 facts per website                (15pt)
      2. Facts unique within website         (5pt)
      3. Each fact exists on the website     (15pt)
    """
    print("\n----------------- CHECKPOINT 3 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=35, result=0, name="Website Facts are Valid")

    total_websites = 0
    websites_with_5plus_facts = 0
    websites_with_unique_facts = 0
    fact_dup_examples = []

    total_facts_checked = 0
    facts_verified_on_website = 0
    verification_failure_examples = []

    main_topics = get_main_topics(bullet_hierarchy)

    if not main_topics:
        for name, step_id, max_score in [
            (f"Websites have {MIN_FACTS_PER_WEBSITE}+ facts", 1, 15),
            ("Facts Unique Within Website", 2, 5),
            ("Facts exist on websites", 3, 15),
        ]:
            checkpoint.add_step(
                name, False, step_id,
                "No main topics found - cannot validate facts",
                score=0, max_score=max_score,
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Phase A: walk the bullet hierarchy, collect (url, facts) per website
    # and tally Step 1/Step 2 inputs.
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
                    and fact_item.get('text', '').strip()
                ]

                if len(facts) >= MIN_FACTS_PER_WEBSITE:
                    websites_with_5plus_facts += 1

                # Step 2 prep: fact uniqueness within website. Empty fact lists
                # don't earn vacuous credit; fuzzy_match_text failure leaves the
                # website out of the unique count (counted as a failure).
                if not facts:
                    fact_dup_examples.append(f"{url[:40]}: no facts to deduplicate")
                else:
                    try:
                        fact_has_dup = False
                        for i, fact_a in enumerate(facts):
                            for fact_b in facts[i + 1:]:
                                is_match, _ = fuzzy_match_text(
                                    fact_a, fact_b, threshold=FACT_DEDUP_FUZZY_THRESHOLD,
                                )
                                if is_match:
                                    fact_has_dup = True
                                    break
                            if fact_has_dup:
                                break
                        if fact_has_dup:
                            fact_dup_examples.append(f"{url[:40]}: duplicate fact text")
                        else:
                            websites_with_unique_facts += 1
                    except Exception as e:
                        traceback.print_exc()
                        fact_dup_examples.append(f"{url[:40]}: dedup check failed ({e})")

                url_entries.append((url, facts))

    # Phase B+C: build LLM tasks against the cache populated by CP2 and run in parallel.
    # task_id includes the website's index in url_entries so two website
    # bullets sharing a URL don't collide on `{url}|fact_{i}`.
    vlm_tasks = []
    vlm_results = {}
    fact_task_map = {}  # task_id -> (url_label, fact_text) for failure reporting
    unavailable_task_ids = set()

    for web_idx, (url, facts) in enumerate(url_entries):
        # Missing-URL and unfetchable-URL get the same hard-fail treatment.
        result = website_content_cache.get(url) if url and url != 'No URL' else None
        content = result[0] if result else None
        url_label = url if url and url != 'No URL' else "no_url"

        if not content:
            for i, fact in enumerate(facts):
                total_facts_checked += 1
                task_id = f"web{web_idx}|fact_{i}"
                fact_task_map[task_id] = (url_label, fact)
                unavailable_task_ids.add(task_id)
            continue

        for i, fact in enumerate(facts):
            total_facts_checked += 1
            task_id = f"web{web_idx}|fact_{i}"
            fact_task_map[task_id] = (url_label, fact)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are checking if a fact from a lesson plan could reasonably come from this website. Be very generous — the fact does not need to be a direct quote. Answer 'Yes' if the website discusses the same topic, concept, or any related information, even if worded very differently, only partially covered, or loosely related. Answer only 'Yes' or 'No'."}]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"Fact: {fact}\n\nWebsite content:\n{content}\n\nCould this fact reasonably be derived from or inspired by this website content? Even loose topical relevance counts. Answer only 'Yes' or 'No'."
                    }]
                }
            ]
            vlm_tasks.append({'id': task_id, 'messages': messages})

    print(f"Sending {len(vlm_tasks)} facts to LLM for verification...")
    llm_call_failed = False
    verification_warning = None
    if vlm_tasks:
        try:
            if model is None:
                raise RuntimeError(f"LLM model {model_id} unavailable")
            llm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=VLM_WORKERS)
            vlm_results.update(llm_results)

            # Retry once for un-resolved (False) facts — LLM non-determinism
            # causes occasional false negatives.
            failed_tasks = [task for task in vlm_tasks if not vlm_results.get(task['id'], False)]
            if failed_tasks:
                print(f"Retrying {len(failed_tasks)} failed fact verifications...")
                retry_results = fast_parallel_vlm_calls(failed_tasks, model, max_workers=VLM_WORKERS)
                for task_id, verified in retry_results.items():
                    if verified:
                        vlm_results[task_id] = True

            # Warn after retry so the signal reflects the final merged state.
            verification_warning = vlm_uniform_failure_warning(
                {tid: vlm_results.get(tid, False) for tid in (t['id'] for t in vlm_tasks)}
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Fact verification LLM call failed: {e}")
            llm_call_failed = True

    # Facts under unavailable pages hard-fail the "fact exists" criterion —
    # we won't re-grade them under a weaker rubric.
    facts_unverifiable_unavailable = len(unavailable_task_ids)
    for task_id in unavailable_task_ids:
        vlm_results[task_id] = False
        _, fact_text = fact_task_map[task_id]
        verification_failure_examples.append(f"{fact_text[:40]}... (Page Unavailable)")

    for task_id, verified in vlm_results.items():
        if verified:
            facts_verified_on_website += 1
        elif task_id not in unavailable_task_ids:
            url_part, fact_text = fact_task_map.get(task_id, ("unknown", task_id))
            print(f"  FAILED: [{url_part[:60]}] {fact_text}")
            verification_failure_examples.append(f"{fact_text[:40]}... (Not Found)")

    # ------------------- FINAL SCORING -------------------

    # Step 1: Fact Count
    fact_count_step_name = f"Websites have {MIN_FACTS_PER_WEBSITE}+ facts"
    if total_websites == 0:
        fact_count_score = 0
        fact_count_msg = "No websites found to check fact counts."
        fact_count_passed = False
    else:
        fact_count_score = calculate_percentage_score(websites_with_5plus_facts, total_websites, 15)
        fact_count_passed = (websites_with_5plus_facts == total_websites)
        fact_count_msg = (f"{websites_with_5plus_facts}/{total_websites} websites have "
                          f"{MIN_FACTS_PER_WEBSITE}+ facts.")

    checkpoint.add_step(
        fact_count_step_name, fact_count_passed, 1,
        fact_count_msg, score=fact_count_score, max_score=15,
    )

    # Step 2: Fact uniqueness within each website (5pt)
    if total_websites == 0:
        checkpoint.add_step(
            "Facts Unique Within Website", False, 2,
            "No websites found to check fact uniqueness.",
            score=0, max_score=5,
        )
    elif websites_with_unique_facts == total_websites:
        checkpoint.add_step(
            "Facts Unique Within Website", True, 2,
            f"All {total_websites} websites have unique facts",
            score=5, max_score=5,
        )
    else:
        unique_score = calculate_percentage_score(websites_with_unique_facts, total_websites, 5)
        checkpoint.add_step(
            "Facts Unique Within Website", False, 2,
            f"{websites_with_unique_facts}/{total_websites} websites have unique facts. "
            f"{'; '.join(fact_dup_examples[:MAX_FAILURE_EXAMPLES])}",
            score=unique_score, max_score=5,
        )

    # Step 3: Fact Verification
    if llm_call_failed:
        fact_verification_score = 0
        fact_msg = "Fact verification could not run: LLM call failed."
        fact_passed = False
    elif total_facts_checked == 0:
        fact_verification_score = 0
        fact_msg = "No facts found to verify on websites."
        fact_passed = False
    else:
        fact_verification_score = calculate_percentage_score(facts_verified_on_website, total_facts_checked, 15)
        if facts_verified_on_website == total_facts_checked:
            fact_msg = f"All {total_facts_checked} facts verified on their websites."
            fact_passed = True
        else:
            unavailable_note = (
                f" {facts_unverifiable_unavailable} fact(s) could not be verified because "
                f"the source page was unavailable."
                if facts_unverifiable_unavailable else ""
            )
            example_failures = "; ".join(verification_failure_examples[:MAX_FAILURE_EXAMPLES])
            warning_suffix = f" {verification_warning}" if verification_warning else ""
            fact_msg = (
                f"{facts_verified_on_website}/{total_facts_checked} facts verified."
                f"{unavailable_note} Failures: {example_failures}{warning_suffix}"
            )
            fact_passed = False
            # Cap imperfect score at max-1 so a near-miss can't earn 15/15.
            fact_verification_score = min(fact_verification_score, 14)

    checkpoint.add_step(
        "Facts exist on websites",
        fact_passed,
        3,
        fact_msg,
        score=fact_verification_score,
        max_score=15
    )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_4():
    """
    Checkpoint 4 (20pt): Color-Coded Summary Facts — step order matches checkpoints.md:
      1. Facts copied correctly             (8pt)
      2. Facts formatted as bullet points   (4pt)
      3. Color consistent within topic      (4pt)
      4. Topic colors distinct              (4pt)
    """
    print("\n----------------- CHECKPOINT 4 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Color-Coded Summary Facts")

    main_topics = get_main_topics(bullet_hierarchy)

    if not main_topics:
        for name, step_id, max_score in [
            ("Facts Copied Correctly", 1, 8),
            ("Facts Formatted as Bullet Points", 2, 4),
            ("Color Consistency by Topic", 3, 4),
            ("Topic Colors Distinct", 4, 4),
        ]:
            checkpoint.add_step(
                name, False, step_id,
                "No main topics found - cannot validate summary facts",
                score=0, max_score=max_score,
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Index-keyed list of {'name', 'facts'} so duplicate-named topics stay
    # distinct (a name-keyed dict would silently collapse them, hiding both
    # color collisions in Step 3 and dropped facts in Step 1).
    topic_data: List[Dict] = []
    for topic in main_topics:
        facts = []
        for website in topic.get('children', []):
            if website.get('nesting_level') == 1:
                for fact in website.get('children', []):
                    if fact.get('nesting_level') == 2:
                        fact_text = fact['text'].strip()
                        if fact_text:
                            facts.append(fact_text)
        topic_data.append({'name': topic['text'], 'facts': facts})

    # Extract summary facts from doc_content; failure → zero-score placeholders.
    try:
        summary_facts_list = extract_summary_facts_with_colors(doc_content)
    except Exception as e:
        traceback.print_exc()
        print(f"Summary fact extraction failed: {e}")
        summary_facts_list = []
    all_summary_facts = [item['text'] for item in summary_facts_list]
    summary_facts_with_colors = {item['text']: item['color'] for item in summary_facts_list if item['color'] is not None}
    print(f"Found {len(all_summary_facts)} facts in summary section ({len(summary_facts_with_colors)} with color)")

    # Check if any summary facts exist at all
    if not all_summary_facts:
        checkpoint.add_step(
            "Facts Copied Correctly", False, 1,
            "No facts found in summary section. Summary section may be missing.",
            score=0, max_score=8,
        )
        checkpoint.add_step(
            "Facts Formatted as Bullet Points", False, 2,
            "No fact-like paragraphs found in summary",
            score=0, max_score=4,
        )
        checkpoint.add_step(
            "Color Consistency by Topic", False, 3,
            "No summary facts found to validate color consistency.",
            score=0, max_score=4,
        )
        checkpoint.add_step(
            "Topic Colors Distinct", False, 4,
            "No summary facts found to validate color distinctness.",
            score=0, max_score=4,
        )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Identify topic-wise fact paragraphs once; shared by Steps 1 and 2.
    fact_paragraphs = classify_summary_paragraphs_as_facts(
        summary_facts_list, model, max_workers=VLM_WORKERS,
    )
    fact_paragraph_texts = [f['text'] for f in fact_paragraphs]

    if not fact_paragraphs:
        # No topic-wise fact summaries means Steps 1 and 2 both fail at 0.
        # Steps 3/4 are handled by the no-colored-facts early-exit below.
        msg = ("No topic-wise fact summary paragraphs found "
               "(summary section contains only headings or a narrative lesson summary).")
        checkpoint.add_step("Facts Copied Correctly", False, 1, msg, score=0, max_score=8)
        checkpoint.add_step("Facts Formatted as Bullet Points", False, 2, msg, score=0, max_score=4)
    else:
        step1_start = time.time()
        # Step 1 wrapped so a match failure can't take Steps 2/3/4 with it.
        try:
            total_original_facts = 0
            facts_found_in_summary = 0
            missing_fact_examples = []

            for entry in topic_data:
                for original_fact in entry['facts']:
                    total_original_facts += 1

                    # Contained-match so paragraph-format summaries (multiple
                    # facts per paragraph) work; first-match-wins for accuracy.
                    found = False
                    for summary_fact in fact_paragraph_texts:
                        match, _ = text_fuzzy_match_contained_long(
                            original_fact, summary_fact, threshold=FACT_PRESENCE_FUZZY_THRESHOLD,
                        )
                        if match is not None:
                            found = True
                            break

                    if found:
                        facts_found_in_summary += 1
                    else:
                        fact_preview = original_fact[:40] + ("..." if len(original_fact) > 40 else "")
                        missing_fact_examples.append(fact_preview)

            if total_original_facts == 0:
                accuracy_score = 0
                accuracy_msg = "No facts found in bullet hierarchy to validate"
                accuracy_passed = False
            else:
                accuracy_score = calculate_percentage_score(facts_found_in_summary, total_original_facts, 8)
                accuracy_passed = (facts_found_in_summary == total_original_facts)
                # Cap imperfect score at max-1 so a near-miss can't earn 8/8.
                if not accuracy_passed:
                    accuracy_score = min(accuracy_score, 7)
                if accuracy_passed:
                    accuracy_msg = f"All {total_original_facts} facts found in color-coded summary"
                else:
                    unmatched_count = total_original_facts - facts_found_in_summary
                    examples = "; ".join(missing_fact_examples[:MAX_FAILURE_EXAMPLES])
                    accuracy_msg = (
                        f"{facts_found_in_summary}/{total_original_facts} facts matched in summary. "
                        f"{unmatched_count} did not match — may be absent, or present "
                        f"but paraphrased below threshold (e.g., {examples})"
                    )

            checkpoint.add_step(
                "Facts Copied Correctly", accuracy_passed, 1,
                accuracy_msg, score=accuracy_score, max_score=8,
                execution_time=time.time() - step1_start,
            )
        except Exception as e:
            traceback.print_exc()
            checkpoint.add_step(
                "Facts Copied Correctly", False, 1,
                f"Fact-accuracy check could not run: {e}",
                score=0, max_score=8,
                execution_time=time.time() - step1_start,
            )

        # Step 2: among the LLM-identified fact paragraphs, what fraction are
        # bulleted? Independent of Step 1's fuzzy matching.
        step2_start = time.time()
        try:
            bullet_count = sum(1 for f in fact_paragraphs if f['is_bullet'])
            total_facts = len(fact_paragraphs)

            bullet_score = calculate_percentage_score(bullet_count, total_facts, 4)
            bullet_passed = (bullet_count == total_facts)
            if bullet_passed:
                bullet_msg = f"All {total_facts} summary fact paragraphs are bullets"
            else:
                non_bullet = total_facts - bullet_count
                bullet_msg = (f"{bullet_count}/{total_facts} summary fact paragraphs "
                              f"are bullets ({non_bullet} in paragraph format)")

            checkpoint.add_step(
                "Facts Formatted as Bullet Points", bullet_passed, 2,
                bullet_msg, score=bullet_score, max_score=4,
                execution_time=time.time() - step2_start,
            )
        except Exception as e:
            traceback.print_exc()
            checkpoint.add_step(
                "Facts Formatted as Bullet Points", False, 2,
                f"Bullet-format check could not run: {e}",
                score=0, max_score=4,
                execution_time=time.time() - step2_start,
            )

    step34_start = time.time()
    total_topics_checked = len(topic_data)

    # If no colored facts at all, Steps 3 and 4 fail early.
    if not summary_facts_with_colors:
        msg = (f"No color-coded facts found in summary section "
               f"({len(all_summary_facts)} facts exist but none have color metadata).")
        checkpoint.add_step("Color Consistency by Topic", False, 3, msg, score=0, max_score=4)
        checkpoint.add_step("Topic Colors Distinct", False, 4, msg, score=0, max_score=4)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    # Steps 3+4: index-keyed (duplicate-named topics stay distinct). Use
    # contained-match so paragraph-format summaries (multiple facts per
    # paragraph) work; no consumption since one paragraph may bundle many facts.
    try:
        topic_colors_by_idx: Dict[int, set] = {idx: set() for idx in range(total_topics_checked)}
        for idx, entry in enumerate(topic_data):
            for original_fact in entry['facts']:
                for summary_fact, color in summary_facts_with_colors.items():
                    match, _ = text_fuzzy_match_contained_long(
                        original_fact, summary_fact, threshold=COLOR_MATCH_FUZZY_THRESHOLD,
                    )
                    if match is not None:
                        topic_colors_by_idx[idx].add(color)
                        break

        # Step 3: each topic uses exactly one color
        consistent_topics = 0
        consistency_violations = []
        for idx, entry in enumerate(topic_data):
            colors_used = topic_colors_by_idx[idx]
            if len(colors_used) == 1:
                consistent_topics += 1
            elif len(colors_used) > 1:
                color_hex_list = [format_color(c) for c in colors_used]
                topic_preview = entry['name'][:25] + ("..." if len(entry['name']) > 25 else "")
                consistency_violations.append(
                    f"'{topic_preview}' uses {len(colors_used)} colors: {', '.join(color_hex_list)}"
                )

        if total_topics_checked == 0:
            consistency_score = 0
            consistency_msg = "No topics found to check color consistency"
            consistency_passed = False
        else:
            consistency_score = calculate_percentage_score(consistent_topics, total_topics_checked, 4)
            consistency_passed = (consistent_topics == total_topics_checked)
            if consistency_passed:
                consistency_msg = f"All {total_topics_checked} topics use exactly one color"
            else:
                examples = "; ".join(consistency_violations[:MAX_VIOLATION_EXAMPLES])
                consistency_msg = (
                    f"{consistent_topics}/{total_topics_checked} topics use exactly one color. "
                    f"Issues: {examples}"
                )

        checkpoint.add_step(
            "Color Consistency by Topic", consistency_passed, 3,
            consistency_msg, score=consistency_score, max_score=4,
            execution_time=time.time() - step34_start,
        )

        # Step 4: no color shared between topics (collisions across topic indices).
        color_to_topic_idxs: Dict = {}
        for idx, colors in topic_colors_by_idx.items():
            if len(colors) == 1:
                color = next(iter(colors))
                color_to_topic_idxs.setdefault(color, []).append(idx)

        distinct_topics = total_topics_checked
        distinctness_violations = []
        for color, sharing_idxs in color_to_topic_idxs.items():
            if len(sharing_idxs) > 1:
                color_label = format_color(color)
                preview_names = ", ".join(
                    topic_data[i]['name'][:20]
                    + ("..." if len(topic_data[i]['name']) > 20 else "")
                    for i in sharing_idxs
                )
                distinctness_violations.append(
                    f"Color {color_label} shared by {len(sharing_idxs)} topics: {preview_names}"
                )
                distinct_topics -= len(sharing_idxs)
        # Topics without a single consistent color also can't be distinct.
        distinct_topics -= sum(
            1 for colors in topic_colors_by_idx.values() if len(colors) != 1
        )
        distinct_topics = max(0, distinct_topics)

        if total_topics_checked == 0:
            distinctness_score = 0
            distinctness_msg = "No topics found to check color distinctness"
            distinctness_passed = False
        else:
            distinctness_score = calculate_percentage_score(distinct_topics, total_topics_checked, 4)
            distinctness_passed = (distinct_topics == total_topics_checked)
            if distinctness_passed:
                distinctness_msg = f"All {total_topics_checked} topics use a distinct color"
            else:
                examples = ("; ".join(distinctness_violations[:MAX_VIOLATION_EXAMPLES])
                            or "topics share or lack a single color")
                distinctness_msg = (
                    f"{distinct_topics}/{total_topics_checked} topics have a distinct color. "
                    f"Issues: {examples}"
                )

        checkpoint.add_step(
            "Topic Colors Distinct", distinctness_passed, 4,
            distinctness_msg, score=distinctness_score, max_score=4,
        )
    except Exception as e:
        traceback.print_exc()
        checkpoint.add_step(
            "Color Consistency by Topic", False, 3,
            f"Color consistency check could not run: {e}",
            score=0, max_score=4,
        )
        checkpoint.add_step(
            "Topic Colors Distinct", False, 4,
            f"Color distinctness check could not run: {e}",
            score=0, max_score=4,
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_5():
    """
    Checkpoint 5 (20pt): Check Topic Images — step order matches checkpoints.md
    bullet order:
      1. Each topic has a uniquely-matched image (one-to-one)   (5pt)
      2. Images at bottom of last page                          (5pt)
      3. Images aligned side by side                            (5pt)
      4. Each image depicts a topic concept (concept quality)   (5pt)
    """
    print("\n----------------- CHECKPOINT 5 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=20, result=0, name="Topic Images")

    main_topics = get_main_topics(bullet_hierarchy)

    if not main_topics:
        for name, step_id in [
            ("Each Topic has a Uniquely Matched Image", 1),
            ("Images at Bottom of Last Page", 2),
            ("Images Arranged Side by Side", 3),
            ("Images Depict Topic Concept", 4),
        ]:
            checkpoint.add_step(
                name, False, step_id,
                "No main topics found - cannot validate images",
                score=0, max_score=5,
            )
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    num_topics = len(main_topics)
    topic_names = [t['text'] for t in main_topics]

    # Download embedded images (deduped by file hash).
    page_image_files = sorted(glob.glob(os.path.join(PDF_IMAGES_DIR, "page_*.png")))

    inline_objects = (doc_content.get('inlineObjects', {}) or {}) if doc_content else {}
    positioned_objects = (doc_content.get('positionedObjects', {}) or {}) if doc_content else {}
    has_any_objects = bool(inline_objects or positioned_objects)

    downloaded_images = []
    extraction_failed = False
    if has_any_objects:
        try:
            extract_images_from_doc_extended(
                DOCS_SERVICE, output_dir=IMAGES_DIR,
                document=doc_content, include_positioned=True
            )
            raw_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "image_*.*")))
            downloaded_images = dedup_files_by_sha256(raw_paths)
            print(f"Downloaded {len(raw_paths)} image files ({len(downloaded_images)} unique by hash)")
        except Exception as e:
            extraction_failed = True
            traceback.print_exc()
            print(f"Error downloading images: {e}")

    # Locate images on the last page via SIFT (used by Steps 1, 2, 3).
    alignment_result = None
    if downloaded_images and page_image_files:
        try:
            alignment_result = check_images_aligned_horizontally(
                downloaded_images, PDF_IMAGES_DIR, doc_content=doc_content, dpi=PDF_DPI
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Image alignment check failed: {e}")

    image_locations = alignment_result.get('locations', []) if alignment_result else []

    # Concept-level VLM matrix — one Yes/No per topic×image, shared by Steps 1+4.
    vlm_results = {}
    vlm_failed = False
    concept_warning = None
    if downloaded_images:
        try:
            if model is None:
                raise RuntimeError(f"LLM model {model_id} unavailable")
            vlm_tasks = []
            for topic_name in topic_names:
                for img_path in downloaded_images:
                    task_id = f"topic_relevance|{topic_name}|{img_path}"
                    vlm_tasks.append({
                        'id': task_id,
                        'messages': [
                            {"role": "system", "content": [{"type": "text", "text": (
                                f"You are checking if an image visually depicts a {SUBJECT} concept "
                                f"or directly related people/objects, not merely a surface keyword "
                                f"association. Answer only 'Yes' or 'No'."
                            )}]},
                            {"role": "user", "content": [
                                {"type": "image", "image": img_path},
                                {"type": "text", "text": (
                                    f"Does this image depict the concept of '{topic_name}' "
                                    f"(or directly related people/objects), beyond a surface keyword "
                                    f"association? Answer only Yes or No."
                                )},
                            ]},
                        ],
                    })
            print(f"Running {len(vlm_tasks)} image-topic concept checks in parallel...")
            vlm_results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=VLM_WORKERS)
            concept_warning = vlm_uniform_failure_warning(vlm_results)
        except Exception as e:
            traceback.print_exc()
            print(f"Image-topic concept VLM call failed: {e}")
            vlm_failed = True

    # Step 1: one-to-one topic ↔ image matching (count + assignment combined).
    step1_start = time.time()
    if extraction_failed:
        checkpoint.add_step(
            "Each Topic has a Uniquely Matched Image", False, 1,
            "Image extraction from document failed; cannot run matching.",
            score=0, max_score=5, execution_time=time.time() - step1_start,
        )
    elif not downloaded_images:
        checkpoint.add_step(
            "Each Topic has a Uniquely Matched Image", False, 1,
            "No images embedded in the document.",
            score=0, max_score=5, execution_time=time.time() - step1_start,
        )
    elif vlm_failed:
        checkpoint.add_step(
            "Each Topic has a Uniquely Matched Image", False, 1,
            "Concept-level VLM call failed; cannot run one-to-one matching.",
            score=0, max_score=5, execution_time=time.time() - step1_start,
        )
    else:
        # Greedy DFS bipartite matching over Yes-edges. Indexed by topic
        # position so duplicate-named topics each get their own slot (a name-
        # keyed dict would silently collapse them and undercount matches).
        edges_by_idx = {
            idx: [img for img in downloaded_images
                  if vlm_results.get(f"topic_relevance|{name}|{img}", False)]
            for idx, name in enumerate(topic_names)
        }

        assignment = {}   # topic_idx -> img_path
        used_images = set()

        def _try_assign(pos, ordered_idxs):
            if pos == len(ordered_idxs):
                return True
            idx = ordered_idxs[pos]
            for img in edges_by_idx[idx]:
                if img in used_images:
                    continue
                used_images.add(img)
                assignment[idx] = img
                if _try_assign(pos + 1, ordered_idxs):
                    return True
                used_images.discard(img)
                assignment.pop(idx, None)
            return _try_assign(pos + 1, ordered_idxs)

        # Process topics with fewest matching images first to maximize matches.
        ordered_idxs = sorted(range(len(topic_names)),
                              key=lambda i: len(edges_by_idx[i]))
        _try_assign(0, ordered_idxs)

        topics_with_match = len(assignment)
        unmatched_topics = [topic_names[i][:30] for i in range(len(topic_names))
                            if i not in assignment]

        match_score = calculate_percentage_score(topics_with_match, num_topics, 5)
        match_passed = (topics_with_match == num_topics
                        and len(downloaded_images) == num_topics)

        if match_passed:
            match_msg = (f"All {num_topics} topics have a uniquely matched image "
                         f"(count and one-to-one match)")
        else:
            unmatched = "; ".join(unmatched_topics[:2]) if unmatched_topics else "—"
            match_msg = (
                f"{topics_with_match}/{num_topics} topics uniquely matched "
                f"({len(downloaded_images)} images downloaded). Unmatched: {unmatched}"
            )
            if concept_warning:
                match_msg += f" {concept_warning}"

        checkpoint.add_step(
            "Each Topic has a Uniquely Matched Image", match_passed, 1,
            match_msg, score=match_score, max_score=5,
            execution_time=time.time() - step1_start,
        )

    # Step 2: page-format-agnostic lower-region check, proportional credit.
    page_width_px, page_height_px, _, _ = get_doc_page_dimensions_px(doc_content, PDF_DPI)
    lower_y = int(page_height_px * LOWER_REGION_PAGE_FRACTION)
    lower_height = page_height_px - lower_y

    if not image_locations:
        bottom_passed = False
        bottom_score = 0
        bottom_msg = "No image locations found for position validation"
    else:
        pages = sorted({loc.page_number for loc in image_locations})
        if len(pages) > 1:
            # Multi-page = hard fail: criterion explicitly requires all on last page.
            bottom_passed = False
            bottom_score = 0
            bottom_msg = f"Images scattered across pages {pages}; expected all on the last page"
        else:
            image_page = pages[0]
            lower_region = Location(image_page, 0, lower_y, page_width_px, lower_height)
            # cutoff=0.5: inline tables placed at end-of-content rarely have
            # >60% inside the bottom region due to whitespace below the table.
            images_at_bottom = [loc for loc in image_locations
                                if loc.is_mostly_inside(lower_region, cutoff=0.5)]
            bottom_passed = len(images_at_bottom) == len(image_locations)
            bottom_score = calculate_percentage_score(len(images_at_bottom), len(image_locations), 5)
            bottom_msg = (f"All {len(image_locations)} images found in lower region of last page"
                          if bottom_passed else
                          f"{len(images_at_bottom)}/{len(image_locations)} images in lower region of last page")

    checkpoint.add_step(
        "Images at Bottom of Last Page", bottom_passed, 2,
        bottom_msg, score=bottom_score, max_score=5,
    )

    # Step 3: side-by-side. Independent of Step 2 — first check doc structure
    # for a 1xN table containing all images, then fall back to SIFT alignment.
    if images_in_single_table_row(doc_content):
        side_by_side_passed = True
        side_by_side_msg = "All images are arranged side by side in a single table row"
    elif alignment_result is not None and alignment_result.get('aligned'):
        side_by_side_passed = True
        side_by_side_msg = alignment_result['details']
    elif extraction_failed:
        side_by_side_passed = False
        side_by_side_msg = "Image extraction failed; cannot run alignment check"
    elif not downloaded_images:
        side_by_side_passed = False
        side_by_side_msg = "No images embedded in the document"
    elif alignment_result is not None:
        side_by_side_passed = False
        side_by_side_msg = alignment_result['details']
    elif not page_image_files:
        side_by_side_passed = False
        side_by_side_msg = "PDF rasterization produced no page images; cannot run alignment check"
    else:
        side_by_side_passed = False
        side_by_side_msg = "Alignment check did not run (unknown error)"

    checkpoint.add_step(
        "Images Arranged Side by Side", side_by_side_passed, 3,
        side_by_side_msg, score=5 if side_by_side_passed else 0, max_score=5,
    )

    # Step 4: per-image concept quality. Thematic = VLM said Yes for any topic.
    step4_start = time.time()
    if extraction_failed:
        checkpoint.add_step(
            "Images Depict Topic Concept", False, 4,
            "Image extraction failed; cannot run concept check.",
            score=0, max_score=5, execution_time=time.time() - step4_start,
        )
    elif not downloaded_images:
        checkpoint.add_step(
            "Images Depict Topic Concept", False, 4,
            "No images embedded in the document.",
            score=0, max_score=5, execution_time=time.time() - step4_start,
        )
    elif vlm_failed:
        checkpoint.add_step(
            "Images Depict Topic Concept", False, 4,
            "Concept-level VLM call failed; cannot evaluate image concept depiction.",
            score=0, max_score=5, execution_time=time.time() - step4_start,
        )
    else:
        thematic_images = sum(
            1 for img in downloaded_images
            if any(vlm_results.get(f"topic_relevance|{t}|{img}", False) for t in topic_names)
        )
        total_images = len(downloaded_images)
        concept_score = calculate_percentage_score(thematic_images, total_images, 5)
        concept_passed = (thematic_images == total_images)
        if concept_passed:
            concept_msg = (f"All {total_images} images depict a topic concept "
                           f"(no surface-keyword-only images)")
        else:
            non_thematic = total_images - thematic_images
            concept_msg = (
                f"{thematic_images}/{total_images} images depict a topic concept; "
                f"{non_thematic} appear to be surface-keyword associations"
            )
            if concept_warning:
                concept_msg += f" {concept_warning}"

        checkpoint.add_step(
            "Images Depict Topic Concept", concept_passed, 4,
            concept_msg, score=concept_score, max_score=5,
            execution_time=time.time() - step4_start,
        )

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoint_6():
    """
    Checkpoint 6 (10pt): Lesson Summary — step order matches checkpoints.md:
      1. Lesson summary present                        (3pt)
      2. Lesson summary uses the facts collected       (7pt)
    """
    print("\n----------------- CHECKPOINT 6 ----------------")
    checkpoint_start = time.time()
    checkpoint = Checkpoint(total=10, result=0, name="Lesson Summary")

    main_topics = get_main_topics(bullet_hierarchy)
    if not main_topics:
        for name, step_id, max_score in [
            ("Lesson Summary Present", 1, 3),
            ("Lesson Summary Uses Facts", 2, 7),
        ]:
            checkpoint.add_step(name, False, step_id,
                "No main topics found - cannot validate lesson summary",
                score=0, max_score=max_score)
        checkpoint.execution_time = time.time() - checkpoint_start
        return checkpoint

    try:
        summary_facts_list = extract_summary_facts_with_colors(doc_content or {})
    except Exception as e:
        traceback.print_exc()
        print(f"Summary fact extraction failed: {e}")
        summary_facts_list = []

    # Shortlist non-fact candidates once (also reused by Step 2's source choice).
    fact_paragraphs = []
    try:
        fact_paragraphs = classify_summary_paragraphs_as_facts(
            summary_facts_list, model, max_workers=VLM_WORKERS,
        )
    except Exception as e:
        traceback.print_exc()

    fact_ids = {id(f) for f in fact_paragraphs}
    candidates = [f for f in summary_facts_list
                  if id(f) not in fact_ids and len(f['text']) > 100]

    # Step 1: LLM Yes/No per candidate — pick the first that's classified as a
    # lesson summary covering all topics.
    step1_start = time.time()
    lesson_summary_text = None
    try:
        if candidates and model is not None:
            # Genre check independent of hierarchy-parser output: is the
            # paragraph a narrative lesson recap (vs heading / fact list / etc.)?
            vlm_tasks = [
                {
                    'id': str(i),
                    'messages': [
                        {"role": "system", "content": [{"type": "text", "text": (
                            "Answer 'Yes' if the paragraph is a lesson summary — "
                            "a narrative paragraph that introduces or recaps a "
                            "lesson, synthesizing multiple topics or themes. "
                            "Answer 'No' if it is a heading, a list of facts, a "
                            "single-topic deep dive, or unrelated text."
                        )}]},
                        {"role": "user", "content": [{"type": "text", "text": (
                            f"Paragraph:\n---\n{c['text'][:2000]}\n---\n\n"
                            f"Is this paragraph a lesson summary? Answer 'Yes' or 'No'."
                        )}]},
                    ],
                }
                for i, c in enumerate(candidates)
            ]
            results = fast_parallel_vlm_calls(vlm_tasks, model, max_workers=VLM_WORKERS)
            for i, c in enumerate(candidates):
                if results.get(str(i), False):
                    lesson_summary_text = c['text']
                    break

        if lesson_summary_text:
            checkpoint.add_step("Lesson Summary Present", True, 1,
                f"Lesson summary detected ({len(lesson_summary_text)} chars)",
                score=3, max_score=3,
                execution_time=time.time() - step1_start)
        else:
            checkpoint.add_step("Lesson Summary Present", False, 1,
                "No lesson summary paragraph found in the summary section",
                score=0, max_score=3,
                execution_time=time.time() - step1_start)
    except Exception as e:
        traceback.print_exc()
        checkpoint.add_step("Lesson Summary Present", False, 1,
            f"Lesson-summary detection could not run: {e}",
            score=0, max_score=3,
            execution_time=time.time() - step1_start)

    # Step 2: single LLM Yes/No — is the lesson summary derived from the
    # source facts? Source = topic-wise fact paragraphs if present, else the
    # bullet-hierarchy facts under each topic.
    step2_start = time.time()
    try:
        if fact_paragraphs:
            source_label = "topic-wise fact paragraphs from the summary section"
            source_text = "\n\n".join(f['text'] for f in fact_paragraphs)
        else:
            source_label = "bullet-hierarchy facts under each topic"
            lines = []
            for topic in main_topics:
                lines.append(f"Topic: {topic['text']}")
                for website in topic.get('children', []):
                    if website.get('nesting_level') == 1:
                        for fact in website.get('children', []):
                            if fact.get('nesting_level') == 2:
                                txt = fact['text'].strip()
                                # Skip URL-only bullets (noise; not real facts).
                                if txt and not (
                                    txt.startswith(('http://', 'https://'))
                                    and ' ' not in txt
                                ):
                                    lines.append(f"- {txt}")
            source_text = "\n".join(lines)

        if not lesson_summary_text:
            checkpoint.add_step("Lesson Summary Uses Facts", False, 2,
                "Lesson summary not detected; cannot verify it uses the facts",
                score=0, max_score=7,
                execution_time=time.time() - step2_start)
        elif not source_text.strip():
            checkpoint.add_step("Lesson Summary Uses Facts", False, 2,
                "No source facts found (no fact paragraphs and no bullet-hierarchy facts)",
                score=0, max_score=7,
                execution_time=time.time() - step2_start)
        elif model is None:
            raise RuntimeError("Model not loaded for uses-facts check")
        else:
            tasks = [{
                'id': 'uses_facts',
                'messages': [
                    {"role": "system", "content": [{"type": "text", "text": (
                        "You evaluate whether a lesson summary is grounded in a "
                        "provided list of source facts. Paraphrasing, rewording, "
                        "and synthesizing multiple facts into one sentence are "
                        "expected and count as Yes. The summary does NOT need to "
                        "cover every fact. Answer 'Yes' if most of the summary's "
                        "key claims correspond to (or are paraphrases of) the "
                        "source facts. Answer 'No' only if the summary's content "
                        "is largely unrelated to the source facts or contradicts "
                        "them."
                    )}]},
                    {"role": "user", "content": [{"type": "text", "text": (
                        f"Source facts ({source_label}):\n"
                        f"---\n{source_text[:6000]}\n---\n\n"
                        f"Lesson summary:\n---\n{lesson_summary_text[:3000]}\n---\n\n"
                        f"Is the lesson summary grounded in the source facts above "
                        f"(paraphrase / synthesis is fine)? Answer 'Yes' or 'No'."
                    )}]},
                ],
            }]
            results = fast_parallel_vlm_calls(tasks, model, max_workers=1)
            uses_facts = results.get('uses_facts', False)
            score = 7 if uses_facts else 0
            msg = (f"Lesson summary IS written using the {source_label}"
                   if uses_facts else
                   f"Lesson summary does NOT appear to be derived from the {source_label}")
            checkpoint.add_step("Lesson Summary Uses Facts", uses_facts, 2,
                msg, score=score, max_score=7,
                execution_time=time.time() - step2_start)
    except Exception as e:
        traceback.print_exc()
        checkpoint.add_step("Lesson Summary Uses Facts", False, 2,
            f"Uses-facts check could not run: {e}",
            score=0, max_score=7,
            execution_time=time.time() - step2_start)

    checkpoint.execution_time = time.time() - checkpoint_start
    return checkpoint


def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history_list=None,
                      browsing_history_doc_id=None):
    """
    Grade all checkpoints for the education lesson plan document.

    Args:
        workspace_doc_id (str): The Google Docs document ID to evaluate.
        cached_models (dict, optional): Preloaded models by model_id.
        browsing_history_list (list, optional): List of URLs visited by agent.
        browsing_history_doc_id (str, optional): Google Doc ID containing
            browsing history URLs. If provided, links are extracted after
            services are initialized in setup_document.

    Returns:
        Result: Evaluation results with checkpoint scores.
    """
    total_start_time = time.time()

    # Reset per-run cache so prior call's content can't bleed in.
    global website_content_cache
    website_content_cache = {}

    # Step list per checkpoint, used to build a full-zero result if setup fails.
    checkpoint_specs = [
        ("Related Topics", 15),
        ("Website URLs Match", 45),
        ("Website Facts are Valid", 35),
        ("Color-Coded Summary Facts", 20),
        ("Topic Images", 20),
        ("Lesson Summary", 10),
    ]

    try:
        # setup_document also initializes Google services and SUBJECT/AUDIENCE.
        try:
            setup_document(workspace_doc_id)
        except Exception as setup_error:
            traceback.print_exc()
            failed_checkpoints = []
            for name, total in checkpoint_specs:
                cp = Checkpoint(total=total, result=0, name=name)
                cp.add_step(
                    "evaluator setup failed", False, 1,
                    f"Could not set up document evaluation: {setup_error}",
                    score=0, max_score=total,
                )
                failed_checkpoints.append(cp)
            return Result(failed_checkpoints,
                          total_execution_time=time.time() - total_start_time)

        if browsing_history_list is None and browsing_history_doc_id:
            try:
                all_links = extract_hyperlinks_from_doc(browsing_history_doc_id, DOCS_SERVICE)
                browsing_history_list = [link['url'] for link in all_links if link['url'].startswith('http')]
                print(f"Loaded {len(browsing_history_list)} URLs from browsing history doc")
            except Exception as e:
                print(f"Warning: could not extract browsing history: {e}")
                browsing_history_list = []

        global browsing_history
        browsing_history = browsing_history_list or []

        global model
        if cached_models and model_id in cached_models:
            model = cached_models[model_id]
            print(f"Using preloaded model {model_id}")

        # Per-checkpoint isolation — one crash doesn't lose the rest.
        def _safe(fn, name, total):
            try:
                return fn()
            except Exception as e:
                traceback.print_exc()
                cp = Checkpoint(total=total, result=0, name=name)
                cp.add_step("evaluator crashed", False, 1,
                            f"Uncaught exception: {e}", score=0, max_score=total)
                return cp

        checkpoints: List[Checkpoint] = [
            _safe(grade_checkpoint_1, *checkpoint_specs[0]),
            _safe(grade_checkpoint_2, *checkpoint_specs[1]),
            _safe(grade_checkpoint_3, *checkpoint_specs[2]),
            _safe(grade_checkpoint_4, *checkpoint_specs[3]),
            _safe(grade_checkpoint_5, *checkpoint_specs[4]),
            _safe(grade_checkpoint_6, *checkpoint_specs[5]),
        ]

        return Result(checkpoints, total_execution_time=time.time() - total_start_time)

    finally:
        if CLEANUP_ENABLED:
            print("Cleaning up generated files...")
            for dir_path in [PDF_IMAGES_DIR, IMAGES_DIR]:
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        print(f"Removed directory: {dir_path}")
                    except Exception as e:
                        print(f"Error removing directory {dir_path}: {e}")
            for pdf_file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
                try:
                    os.remove(pdf_file)
                    print(f"Removed PDF file: {pdf_file}")
                except Exception as e:
                    print(f"Error removing PDF {pdf_file}: {e}")
            print("Cleanup completed")
        else:
            print("Cleanup disabled by CLEANUP=False environment variable")


if __name__ == "__main__":
    import argparse

    # Avoid UnicodeEncodeError on Windows consoles when non-ASCII characters
    # (formulas, symbols, accented names) appear in step details.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Evaluate education lesson plan document")
    parser.add_argument("--workspace_doc_id", type=str, required=True,
                       help="Google Docs document ID to evaluate")
    parser.add_argument("--browsing_history_doc_id", type=str, default=None,
                       help="Google Doc ID containing browsing history URLs (one per line)")
    args = parser.parse_args()

    start_time = time.time()

    print(f"CLEANUP enabled: {CLEANUP_ENABLED}")

    result = grade_checkpoints(
        workspace_doc_id=args.workspace_doc_id,
        browsing_history_doc_id=args.browsing_history_doc_id,
    )

    print("\n=== EVALUATION RESULTS ===")
    print(f"Final Score: {result.final_score}")
    print("\n=== DETAILED REPORT ===")
    detailed_report = result.get_detailed_report()

    for checkpoint_data in detailed_report["checkpoints"]:
        print(f"\n{checkpoint_data['name']}: {checkpoint_data['score']}")
        for step in checkpoint_data["steps"]:
            status = "[PASS]" if step["success"] else "[FAIL]"
            print(f"  {status} {step['name']} ({step['score']}/{step['max_score']}): {step['details'] or 'No details'}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
