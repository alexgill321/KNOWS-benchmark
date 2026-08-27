#!/usr/bin/env python3
"""Standalone script for extracting Figure 1 from arXiv papers.

Uses 3-stage approach:
1. arXiv HTML - Parse from experimental HTML rendering (with LLM fallback)
2. LaTeX - Parse from source .tex files (with LLM fallback)
3. VLM - Use vision-language model on images as final fallback

Usage:
    python extract_figures.py --instance 1
    python extract_figures.py --instance 2 [--skip-llm] [--papers-only] [--new-papers-only]

This script reads gold_papers.json and gold_new_papers.json from the specified
instance's data directory, extracts Figure 1 for each paper, saves the images
to data/gold_figures/, and updates the JSON files with figure_1_path.
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
    get_figures_dir,
    ensure_data_directories,
    load_json,
    save_json,
    extract_figure_1_from_html,
    download_arxiv_source,
    extract_figure_1_with_latex_parsing,
    extract_figure_1_with_llm,
)

sys.path.append(BASE_PATH)


def extract_figure_1(arxiv_id: str, model=None, use_latex: bool = True) -> Tuple[bool, Optional[bytes], str]:
    """Extract Figure 1 from an arXiv paper.

    Stage 1: arXiv HTML (with LLM fallback)
    Stage 2: LaTeX parsing (enabled by default, disable with use_latex=False)
    """
    # Stage 1: Try arXiv HTML
    print(f"    Stage 1: Trying arXiv HTML...")
    success, img_bytes, msg = extract_figure_1_from_html(arxiv_id, model=model)
    if success and img_bytes:
        return True, img_bytes, msg

    print(f"    Stage 1 failed: {msg}")

    # Stage 2: Try LaTeX parsing (disabled by default to avoid API pressure)
    if use_latex:
        print(f"    Stage 2: Trying LaTeX source...")
        with tempfile.TemporaryDirectory() as temp_dir:
            dl_success, dl_msg, files = download_arxiv_source(arxiv_id, temp_dir)

            if dl_success:
                source_dir = os.path.join(temp_dir, arxiv_id.replace('/', '_'))

                # Stage 2a: Automatic LaTeX parsing
                success, fig_path, msg = extract_figure_1_with_latex_parsing(source_dir)
                if success and fig_path:
                    with open(fig_path, 'rb') as f:
                        return True, f.read(), f"[LaTeX] {msg}"

                print(f"    Stage 2a failed: {msg}")

                # Stage 2b: LLM LaTeX parsing
                if model:
                    success, fig_path, msg = extract_figure_1_with_llm(source_dir, model)
                    if success and fig_path:
                        with open(fig_path, 'rb') as f:
                            return True, f.read(), f"[LaTeX+LLM] {msg}"
                    print(f"    Stage 2b failed: {msg}")
            else:
                print(f"    Could not download source: {dl_msg}")

    # All stages failed
    return False, None, "All extraction stages failed"


def process_papers(papers: List[Dict], prefix: str, instance: int, model=None, skip_existing: bool = False, use_latex: bool = False) -> Tuple[List[Dict], int, int]:
    """Process a list of papers and extract Figure 1 for each."""
    figures_dir = get_figures_dir(instance)
    found_count = 0
    not_found_count = 0
    skipped_count = 0

    for i, paper in enumerate(papers):
        arxiv_id = paper.get('arxiv_id')
        title = paper.get('title', 'Unknown')[:50]

        if not arxiv_id:
            print(f"  [{i+1}/{len(papers)}] {title}... - No arXiv ID")
            paper['figure_1_path'] = None
            not_found_count += 1
            continue

        if skip_existing:
            # Check if figure already exists on disk (by JSON path or expected filename)
            expected_filename = f"{prefix}_{i+1}_fig1.png"
            fig_path = None
            if paper.get('figure_1_path'):
                fig_path = os.path.join(figures_dir, os.path.basename(paper['figure_1_path']))
            if not fig_path or not os.path.exists(fig_path):
                fig_path = os.path.join(figures_dir, expected_filename)
            if os.path.exists(fig_path):
                # Restore the path in the JSON if it was cleared
                paper['figure_1_path'] = f"data/gold_figures/{os.path.basename(fig_path)}"
                skipped_count += 1
                found_count += 1
                continue

        print(f"  [{i+1}/{len(papers)}] {title}...")

        success, img_bytes, msg = extract_figure_1(arxiv_id, model=model, use_latex=use_latex)

        if success and img_bytes:
            ext = '.png'
            dest_filename = f"{prefix}_{i+1}_fig1{ext}"
            dest_path = os.path.join(figures_dir, dest_filename)

            with open(dest_path, 'wb') as f:
                f.write(img_bytes)

            paper['figure_1_path'] = f"data/gold_figures/{dest_filename}"
            print(f"    SUCCESS: {msg}")
            print(f"    Saved: {dest_filename}")
            found_count += 1
        else:
            print(f"    FAILED: {msg}")
            paper['figure_1_path'] = None
            not_found_count += 1

        # Additional cooldown between papers (on top of per-request rate limiting in utils.py)
        time.sleep(5)

    if skipped_count:
        print(f"  Skipped {skipped_count} papers with existing figures")

    return papers, found_count, not_found_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract Figure 1 from arXiv papers using 3-stage approach"
    )
    parser.add_argument('--instance', type=int, default=1,
                        help="Instance number (default: 1)")
    parser.add_argument('--skip-llm', action='store_true',
                        help="Skip LLM-based extraction stages")
    parser.add_argument('--papers-only', action='store_true',
                        help="Only process original papers (gold_papers.json)")
    parser.add_argument('--new-papers-only', action='store_true',
                        help="Only process new papers (gold_new_papers.json)")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip papers that already have figure_1_path set")
    parser.add_argument('--skip-latex', action='store_true',
                        help="Disable LaTeX source fallback (enabled by default)")
    args = parser.parse_args()

    instance = args.instance

    print("=" * 60)
    print(f"Figure 1 Extraction Script (instance {instance})")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"3-Stage approach: HTML -> LaTeX -> VLM")
    print(f"LLM fallback: {'Disabled' if args.skip_llm else 'Enabled'}")

    # Ensure directories exist
    ensure_data_directories(instance)

    # Load LLM model if needed
    model = None
    if not args.skip_llm:
        try:
            from src.browsergym.knows.eval.eval_utils.models import load_model
            model = load_model("gemini-2.5-flash-google-ai")
            print("LLM model loaded for fallback stages")
        except Exception as e:
            print(f"WARNING: Could not load LLM model: {e}")
            print("Will use automatic parsing only")

    total_found = 0
    total_not_found = 0

    # Process original papers
    if not args.new_papers_only:
        gold_papers = load_json("gold_papers.json", instance)
        if gold_papers and 'papers' in gold_papers:
            print(f"\n=== Processing {len(gold_papers['papers'])} Original Papers ===")
            papers, found, not_found = process_papers(
                gold_papers['papers'], 'original', instance, model=model,
                skip_existing=args.skip_existing, use_latex=not args.skip_latex
            )
            gold_papers['papers'] = papers
            gold_papers['figure_extraction_date'] = datetime.now().isoformat()
            save_json(gold_papers, "gold_papers.json", instance)

            total_found += found
            total_not_found += not_found
            print(f"\nOriginal papers: {found} found, {not_found} not found")
        else:
            print("\nWARNING: gold_papers.json not found or empty")

    # Process new papers
    if not args.papers_only:
        gold_new_papers = load_json("gold_new_papers.json", instance)
        if gold_new_papers and 'papers' in gold_new_papers:
            print(f"\n=== Processing {len(gold_new_papers['papers'])} New Papers ===")
            papers, found, not_found = process_papers(
                gold_new_papers['papers'], 'new', instance, model=model,
                skip_existing=args.skip_existing, use_latex=not args.skip_latex
            )
            gold_new_papers['papers'] = papers
            gold_new_papers['figure_extraction_date'] = datetime.now().isoformat()
            save_json(gold_new_papers, "gold_new_papers.json", instance)

            total_found += found
            total_not_found += not_found
            print(f"\nNew papers: {found} found, {not_found} not found")
        else:
            print("\nWARNING: gold_new_papers.json not found or empty")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Instance: {instance}")
    print(f"Total Figure 1 found: {total_found}")
    print(f"Total not found: {total_not_found}")
    print(f"Success rate: {total_found / (total_found + total_not_found) * 100:.1f}%"
          if (total_found + total_not_found) > 0 else "N/A")
    print(f"Completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
