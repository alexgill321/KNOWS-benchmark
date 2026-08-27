#!/usr/bin/env python3
"""Standalone script for detecting keyword mentions in Related Work sections.

Generalized from detect_chain_of_thought.py to support any keyword per instance.

Uses multi-stage approach:
1. arXiv HTML - Parse section headers, extract Related Work content
2. LaTeX source - Fall back to parsing .tex files
3. LLM detection - Use LLM to determine if mentions are meaningful

Usage:
    python detect_keyword.py --instance 1 --keyword "chain-of-thought"
    python detect_keyword.py --instance 2 --keyword "dark energy"
    python detect_keyword.py --instance 3 --keyword "Monte Carlo"
    python detect_keyword.py --instance 4 --keyword "deep learning"
    python detect_keyword.py --instance 5 --keyword "Bayesian"

This script reads gold_papers.json and gold_new_papers.json from the specified
instance's data directory, detects keyword mentions in Related Work sections,
and updates the JSON files with has_keyword and keyword_note fields.
"""

import os
import sys
import time
import argparse
import tempfile
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Local imports
from utils import (
    BASE_PATH,
    load_json,
    save_json,
    fetch_arxiv_html,
    extract_section_headers_from_html,
    extract_section_content_from_html,
    find_related_work_section,
    extract_section_from_latex,
    detect_keyword_in_text,
    download_arxiv_source,
    find_tex_files,
)

sys.path.append(BASE_PATH)

# Instance-specific keyword defaults
INSTANCE_KEYWORDS = {
    1: "chain-of-thought",
    2: "dark energy",
    3: "Monte Carlo",
    4: "deep learning",
    5: "Bayesian",
}

# Related work section patterns for LaTeX
RELATED_WORK_PATTERNS = [
    'related work',
    'related literature',
    'prior work',
    'previous work',
    'background',
    'related works'
]


def detect_keyword_html(arxiv_id: str, keyword: str, model) -> Tuple[bool, str]:
    """Detect keyword mentions using arXiv HTML."""
    success, html_content, msg = fetch_arxiv_html(arxiv_id)
    if not success:
        return False, f"[HTML] Could not fetch: {msg}"

    headers = extract_section_headers_from_html(html_content)
    if not headers:
        return False, "[HTML] No section headers found"

    related_section = find_related_work_section(headers, model)
    if not related_section:
        return False, "[HTML] No Related Work section found"

    content = extract_section_content_from_html(html_content, related_section)
    if not content:
        return False, f"[HTML] Could not extract content from '{related_section}'"

    has_keyword, explanation = detect_keyword_in_text(content, keyword, model)
    return has_keyword, f"[HTML] {explanation}"


def detect_keyword_latex(arxiv_id: str, keyword: str, model) -> Tuple[bool, str]:
    """Detect keyword mentions using LaTeX source."""
    with tempfile.TemporaryDirectory() as temp_dir:
        success, msg, files = download_arxiv_source(arxiv_id, temp_dir)
        if not success:
            return False, f"[LaTeX] Could not download source: {msg}"

        source_dir = os.path.join(temp_dir, arxiv_id.replace('/', '_'))
        tex_files = find_tex_files(source_dir)

        if not tex_files:
            return False, "[LaTeX] No .tex files found"

        for tex_path in tex_files:
            try:
                with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                    tex_content = f.read()

                content = extract_section_from_latex(tex_content, RELATED_WORK_PATTERNS)

                if content and len(content) > 100:
                    has_keyword, explanation = detect_keyword_in_text(content, keyword, model)
                    return has_keyword, f"[LaTeX] {explanation}"

            except Exception:
                continue

        return False, "[LaTeX] No Related Work section found in any .tex file"


def detect_keyword_in_paper(arxiv_id: str, keyword: str, model) -> Tuple[bool, str]:
    """Detect keyword mentions using multi-stage approach.

    Stage 1: Try arXiv HTML
    Stage 2: Try LaTeX source
    """
    # Stage 1: Try HTML
    has_keyword, msg = detect_keyword_html(arxiv_id, keyword, model)
    if not msg.startswith("[HTML] Could not fetch"):
        # HTML was fetched — use whatever result we got (even if no Related Work section)
        return has_keyword, msg

    print(f"      HTML stage: {msg}")

    # Stage 2: Only try LaTeX if HTML couldn't be fetched (429, no HTML version, etc.)
    has_keyword, msg = detect_keyword_latex(arxiv_id, keyword, model)
    return has_keyword, msg


def process_papers(papers: List[Dict], keyword: str, model, instance: int, skip_existing: bool = False) -> Tuple[List[Dict], int, int]:
    """Process a list of papers and detect keyword mentions."""
    has_keyword_count = 0
    no_keyword_count = 0
    skipped_count = 0

    for i, paper in enumerate(papers):
        arxiv_id = paper.get('arxiv_id')
        title = paper.get('title', 'Unknown')[:50]

        if not arxiv_id:
            print(f"  [{i+1}/{len(papers)}] {title}... - No arXiv ID")
            paper['has_keyword'] = False
            paper['keyword_note'] = "No arXiv ID"
            paper['keyword_evaluated'] = False
            no_keyword_count += 1
            continue

        if skip_existing and 'has_keyword' in paper:
            skipped_count += 1
            if paper.get('has_keyword'):
                has_keyword_count += 1
            else:
                no_keyword_count += 1
            continue

        print(f"  [{i+1}/{len(papers)}] {title}...")

        has_keyword, msg = detect_keyword_in_paper(arxiv_id, keyword, model)

        paper['has_keyword'] = has_keyword
        paper['keyword_note'] = msg
        # Evaluated = content was retrieved (even if no Related Work section).
        # Not evaluated = couldn't fetch HTML or download LaTeX source at all.
        paper['keyword_evaluated'] = not any(
            p in msg for p in ["Could not fetch", "Could not download"]
        )

        if has_keyword:
            print(f"    FOUND \"{keyword}\": {msg[:80]}...")
            has_keyword_count += 1
        else:
            print(f"    No \"{keyword}\": {msg[:80]}...")
            no_keyword_count += 1

        # Rate limiting between papers (3 seconds for export.arxiv.org)
        time.sleep(3)

    if skipped_count:
        print(f"  Skipped {skipped_count} papers with existing keyword data")

    return papers, has_keyword_count, no_keyword_count


def main():
    parser = argparse.ArgumentParser(
        description="Detect keyword mentions in Related Work sections"
    )
    parser.add_argument('--instance', type=int, default=1,
                        help="Instance number (default: 1)")
    parser.add_argument('--keyword', type=str, default=None,
                        help="Keyword to detect (overrides instance default)")
    parser.add_argument('--papers-only', action='store_true',
                        help="Only process original papers (gold_papers.json)")
    parser.add_argument('--new-papers-only', action='store_true',
                        help="Only process new papers (gold_new_papers.json)")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip papers that already have has_keyword set")
    args = parser.parse_args()

    instance = args.instance
    keyword = args.keyword or INSTANCE_KEYWORDS.get(instance)

    if not keyword:
        print(f"ERROR: No keyword configured for instance {instance}. Use --keyword.")
        sys.exit(1)

    print("=" * 60)
    print(f"Keyword Detection Script (instance {instance})")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Keyword: \"{keyword}\"")
    print(f"Multi-stage approach: HTML -> LaTeX")

    # Load LLM model
    model = None
    try:
        from src.browsergym.knows.eval.eval_utils.models import load_model
        model = load_model("gemini-2.5-flash-google-ai")
        print("LLM model loaded for detection")
    except Exception as e:
        print(f"ERROR: Could not load LLM model: {e}")
        print("LLM is required for keyword detection.")
        sys.exit(1)

    total_has_keyword = 0
    total_no_keyword = 0

    # Process original papers
    if not args.new_papers_only:
        gold_papers = load_json("gold_papers.json", instance)
        if gold_papers and 'papers' in gold_papers:
            print(f"\n=== Processing {len(gold_papers['papers'])} Original Papers ===")
            papers, has_kw, no_kw = process_papers(gold_papers['papers'], keyword, model, instance,
                                                       skip_existing=args.skip_existing)
            gold_papers['papers'] = papers
            gold_papers['keyword_detection_date'] = datetime.now().isoformat()
            gold_papers['keyword'] = keyword
            save_json(gold_papers, "gold_papers.json", instance)

            total_has_keyword += has_kw
            total_no_keyword += no_kw
            print(f"\nOriginal papers: {has_kw} with \"{keyword}\", {no_kw} without")
        else:
            print("\nWARNING: gold_papers.json not found or empty")

    # Process new papers
    if not args.papers_only:
        gold_new_papers = load_json("gold_new_papers.json", instance)
        if gold_new_papers and 'papers' in gold_new_papers:
            print(f"\n=== Processing {len(gold_new_papers['papers'])} New Papers ===")
            papers, has_kw, no_kw = process_papers(gold_new_papers['papers'], keyword, model, instance,
                                                       skip_existing=args.skip_existing)
            gold_new_papers['papers'] = papers
            gold_new_papers['keyword_detection_date'] = datetime.now().isoformat()
            gold_new_papers['keyword'] = keyword
            save_json(gold_new_papers, "gold_new_papers.json", instance)

            total_has_keyword += has_kw
            total_no_keyword += no_kw
            print(f"\nNew papers: {has_kw} with \"{keyword}\", {no_kw} without")
        else:
            print("\nWARNING: gold_new_papers.json not found or empty")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Instance: {instance}")
    print(f"Keyword: \"{keyword}\"")
    print(f"Total with keyword: {total_has_keyword}")
    print(f"Total without keyword: {total_no_keyword}")
    total = total_has_keyword + total_no_keyword
    if total > 0:
        print(f"Percentage with keyword: {total_has_keyword / total * 100:.1f}%")
    print(f"Completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
