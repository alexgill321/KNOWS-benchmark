#!/usr/bin/env python3
"""Preprocessing script for sheets_10_paper_sorting evaluator.

This script extracts gold data from the Gold Labels sheet ONLY.
The Gold Implementation sheet is used for testing, not preprocessing.

Gold Labels Sheet Structure:
- Column A (Og Paper): Original paper titles (7 papers in source folder)
- Column B (First Author(s)): First author name(s)
- Column C (gscholar Link): Google Scholar profile URL(s) for scraping
- Column D (Paper Links): Direct arXiv URLs (fallback when no gscholar)

Preprocessing Logic:
1. For each original paper, search arXiv by title to get full metadata
2. For new papers discovery:
   - If gscholar link exists: Scrape Google Scholar → cross-reference arXiv
   - If no gscholar but Paper Links exist: Use provided arXiv URLs directly

Output Files:
- data/gold_papers.json - Original papers with metadata from arXiv
- data/gold_new_papers.json - Expected new papers (from gscholar or direct links)
- data/author_papers_lookup.json - Mapping of authors to their papers

Usage:
    python preprocess.py [--skip-figures] [--skip-scholar]
"""

import os
import sys
import json
import time
import argparse
import re
from typing import List, Dict, Optional, Any
from datetime import datetime

# Base path setup
def get_base_path():
    if os.path.exists("/app/src"):
        return "/app"
    elif os.path.exists("/scratch"):
        return "/scratch/general/vast/USER/Agent-Benchmark/"
    else:
        return os.getcwd()

BASE_PATH = get_base_path()
sys.path.append(BASE_PATH)

from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.google_services_helpers import get_sheet_content

# Local imports
from utils import (
    extract_arxiv_id_from_url,
    normalize_author_name,
    download_arxiv_source,
    extract_figure_1_from_source,
    search_arxiv_by_title,
    detect_world_models_in_pdf,
)

# Constants
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
FIGURES_DIR = os.path.join(DATA_DIR, "gold_figures")

# Gold Labels Sheet ID (the ONLY source for preprocessing)
GOLD_LABELS_SHEET_ID = "1xQNSQBE7uw4-bPuCf1uDW4XJ-v5F_vPsXO2ooF1BdFM"

# Drive folder IDs (for reference)
SOURCE_FOLDER_ID = "1dfRMRjBHH4F1S9WMD6p6VqpYQZ-pbKWB"
DEST_FOLDER_ID = "1vk3FB8IumyHMBuBjI8fsUSdSyOVlFZPf"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")


def extract_gold_labels_data(sheets_service) -> List[Dict]:
    """Extract data from the Gold Labels sheet.

    Returns:
        List of dicts, one per original paper, with:
        - original_paper_title
        - first_authors (list)
        - gscholar_urls (list)
        - direct_paper_links (list of arXiv URLs, if no gscholar)
    """
    print("\n=== Extracting Gold Labels Data ===")

    sheet_raw = get_sheet_content(GOLD_LABELS_SHEET_ID, sheets_service)
    sheets = sheet_raw.get('sheets', [])

    if not sheets:
        print("ERROR: No sheets found in Gold Labels spreadsheet")
        return []

    rows = sheets[0].get('data', [{}])[0].get('rowData', [])
    print(f"Found {len(rows)} rows in Gold Labels sheet")

    # Skip header row
    entries = []
    for i, row in enumerate(rows[1:], start=1):
        values = row.get('values', [])

        def get_cell_value(idx):
            if idx < len(values):
                ev = values[idx].get('effectiveValue', {})
                return ev.get('stringValue', ev.get('numberValue', ''))
            return ''

        original_title = str(get_cell_value(0)).strip()
        first_authors_raw = str(get_cell_value(1)).strip()
        gscholar_raw = str(get_cell_value(2)).strip()
        paper_links_raw = str(get_cell_value(3)).strip()

        if not original_title:
            continue

        # Parse first authors (comma-separated)
        first_authors = [a.strip() for a in first_authors_raw.split(',') if a.strip()]

        # Parse gscholar URLs (newline-separated)
        gscholar_urls = [url.strip() for url in gscholar_raw.split('\n') if url.strip() and 'scholar.google' in url]

        # Parse direct paper links (newline-separated arXiv URLs)
        direct_links = [url.strip() for url in paper_links_raw.split('\n') if url.strip() and 'arxiv' in url.lower()]

        entry = {
            'row_index': i,
            'original_paper_title': original_title,
            'first_authors': first_authors,
            'first_authors_normalized': [normalize_author_name(a) for a in first_authors],
            'gscholar_urls': gscholar_urls,
            'direct_paper_links': direct_links,
        }

        entries.append(entry)
        print(f"  [{i}] {original_title[:50]}...")
        print(f"      Authors: {first_authors}")
        print(f"      GScholar: {len(gscholar_urls)} URLs, Direct Links: {len(direct_links)} URLs")

    print(f"\nExtracted {len(entries)} original paper entries")
    return entries


def fetch_arxiv_metadata(arxiv_id: str) -> Optional[Dict]:
    """Fetch paper metadata from arXiv API.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.12345")

    Returns:
        Dict with title, authors, abstract, arxiv_url, etc.
    """
    try:
        import arxiv

        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))

        if not results:
            return None

        result = results[0]
        return {
            'arxiv_id': arxiv_id,
            'title': result.title,
            'authors': [str(a) for a in result.authors],
            'first_author': str(result.authors[0]) if result.authors else '',
            'abstract': result.summary,
            'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
            'pdf_url': result.pdf_url,
            'published': str(result.published),
        }

    except Exception as e:
        print(f"    Error fetching arXiv metadata for {arxiv_id}: {e}")
        return None


def fetch_original_papers_metadata(entries: List[Dict]) -> List[Dict]:
    """Fetch full metadata for original papers from arXiv.

    Args:
        entries: List of entries from Gold Labels sheet

    Returns:
        List of original paper dicts with full metadata
    """
    print("\n=== Fetching Original Papers Metadata from arXiv ===")

    original_papers = []

    for entry in entries:
        title = entry['original_paper_title']
        print(f"\n  Searching arXiv for: {title[:60]}...")

        result = search_arxiv_by_title(title)

        if result:
            paper = {
                'title': result['title'],
                'authors': result['authors'],
                'first_author': result['authors'][0] if result['authors'] else '',
                'first_author_normalized': normalize_author_name(result['authors'][0]) if result['authors'] else '',
                'abstract': result['abstract'],
                'arxiv_id': result['arxiv_id'],
                'arxiv_url': f"https://arxiv.org/abs/{result['arxiv_id']}",
                'gold_labels_first_authors': entry['first_authors'],
                'gold_labels_first_authors_normalized': entry['first_authors_normalized'],
                'figure_1_path': None,
                'has_world_models': False,
            }
            original_papers.append(paper)
            print(f"    Found: {result['arxiv_id']} (match score: {result.get('match_score', 'N/A')})")
        else:
            # Create entry with just the title if arXiv search fails
            paper = {
                'title': title,
                'authors': entry['first_authors'],
                'first_author': entry['first_authors'][0] if entry['first_authors'] else '',
                'first_author_normalized': entry['first_authors_normalized'][0] if entry['first_authors_normalized'] else '',
                'abstract': '',
                'arxiv_id': None,
                'arxiv_url': None,
                'gold_labels_first_authors': entry['first_authors'],
                'gold_labels_first_authors_normalized': entry['first_authors_normalized'],
                'figure_1_path': None,
                'has_world_models': False,
            }
            original_papers.append(paper)
            print(f"    NOT FOUND on arXiv")

        time.sleep(0.5)  # Rate limiting

    print(f"\nFetched metadata for {len(original_papers)} original papers")
    return original_papers


def scrape_google_scholar(gscholar_url: str) -> List[str]:
    """Scrape paper titles from a Google Scholar profile.

    Args:
        gscholar_url: Google Scholar profile URL

    Returns:
        List of paper titles from the profile
    """
    try:
        from scholarly import scholarly

        # Extract author ID from URL
        match = re.search(r'user=([a-zA-Z0-9_-]+)', gscholar_url)
        if not match:
            print(f"      Could not extract author ID from: {gscholar_url}")
            return []

        author_id = match.group(1)

        # Get author publications
        author = scholarly.search_author_id(author_id)
        author = scholarly.fill(author, sections=['publications'])

        titles = []
        for pub in author.get('publications', []):
            title = pub.get('bib', {}).get('title', '')
            if title:
                titles.append(title)

        return titles

    except ImportError:
        print("      Warning: scholarly package not installed")
        return []
    except Exception as e:
        print(f"      Error scraping Google Scholar: {e}")
        return []


def discover_new_papers_for_entry(entry: Dict, skip_scholar: bool = False) -> List[Dict]:
    """Discover new papers for a single Gold Labels entry.

    Uses either:
    1. Google Scholar scraping → arXiv cross-reference (if gscholar URLs present)
    2. Direct arXiv URLs (if provided as fallback)

    Args:
        entry: Entry from Gold Labels sheet
        skip_scholar: If True, skip Google Scholar scraping

    Returns:
        List of new paper dicts with arXiv metadata
    """
    new_papers = []
    original_title = entry['original_paper_title']

    # Path A: Google Scholar scraping
    if entry['gscholar_urls'] and not skip_scholar:
        print(f"    Using Google Scholar for: {original_title[:40]}...")

        all_scholar_titles = []
        for gscholar_url in entry['gscholar_urls']:
            print(f"      Scraping: {gscholar_url[:60]}...")
            titles = scrape_google_scholar(gscholar_url)
            all_scholar_titles.extend(titles)
            time.sleep(1)  # Rate limiting

        print(f"      Found {len(all_scholar_titles)} papers on Google Scholar")

        # Cross-reference with arXiv
        for title in all_scholar_titles[:20]:  # Limit to first 20
            # Skip if it's the original paper
            if title.lower().strip() == original_title.lower().strip():
                continue

            result = search_arxiv_by_title(title)
            if result:
                paper = {
                    'title': result['title'],
                    'authors': result['authors'],
                    'first_author': result['authors'][0] if result['authors'] else '',
                    'first_author_normalized': normalize_author_name(result['authors'][0]) if result['authors'] else '',
                    'abstract': result['abstract'],
                    'arxiv_id': result['arxiv_id'],
                    'arxiv_url': f"https://arxiv.org/abs/{result['arxiv_id']}",
                    'source': 'gscholar',
                    'associated_original_paper': original_title,
                    'associated_first_authors': entry['first_authors'],
                    'figure_1_path': None,
                    'has_world_models': False,
                }
                new_papers.append(paper)
                print(f"        Found on arXiv: {result['arxiv_id']}")

            time.sleep(0.3)  # Rate limiting

    # Path B: Direct arXiv URLs (fallback or when no gscholar)
    elif entry['direct_paper_links']:
        print(f"    Using direct arXiv links for: {original_title[:40]}...")

        for arxiv_url in entry['direct_paper_links']:
            arxiv_id = extract_arxiv_id_from_url(arxiv_url)
            if not arxiv_id:
                print(f"      Could not extract ID from: {arxiv_url}")
                continue

            metadata = fetch_arxiv_metadata(arxiv_id)
            if metadata:
                paper = {
                    **metadata,
                    'first_author_normalized': normalize_author_name(metadata['first_author']),
                    'source': 'direct_link',
                    'associated_original_paper': original_title,
                    'associated_first_authors': entry['first_authors'],
                    'figure_1_path': None,
                    'has_world_models': False,
                }
                new_papers.append(paper)
                print(f"      Fetched: {arxiv_id} - {metadata['title'][:40]}...")

            time.sleep(0.3)  # Rate limiting

    return new_papers


def discover_all_new_papers(entries: List[Dict], skip_scholar: bool = False) -> List[Dict]:
    """Discover new papers for all Gold Labels entries.

    Args:
        entries: List of entries from Gold Labels sheet
        skip_scholar: If True, skip Google Scholar scraping

    Returns:
        List of all new paper dicts
    """
    print("\n=== Discovering New Papers ===")

    all_new_papers = []

    for entry in entries:
        new_papers = discover_new_papers_for_entry(entry, skip_scholar)
        all_new_papers.extend(new_papers)
        print(f"    Found {len(new_papers)} new papers for {entry['original_paper_title'][:40]}...")

    print(f"\nTotal new papers discovered: {len(all_new_papers)}")
    return all_new_papers


def build_author_lookup(entries: List[Dict], original_papers: List[Dict], new_papers: List[Dict]) -> Dict:
    """Build a lookup table organized by ORIGINAL PAPER, not by author.

    This is paper-centric: for each original paper, we list all its first authors
    and all new papers where ANY of those first authors appears as an author.

    Args:
        entries: Original Gold Labels entries
        original_papers: List of original paper metadata
        new_papers: List of new paper metadata

    Returns:
        Dict with 'original_papers' key containing list of paper lookup entries
    """
    print("\n=== Building Paper-Centric Author Lookup ===")

    original_papers_lookup = []

    for entry in entries:
        paper_title = entry['original_paper_title']
        first_authors = entry['first_authors']
        first_authors_normalized = entry['first_authors_normalized']

        # Find the original paper's arXiv ID
        original_paper = None
        for op in original_papers:
            # Match by checking if any first author overlaps
            op_first_authors_norm = op.get('gold_labels_first_authors_normalized', [])
            if any(fa in op_first_authors_norm for fa in first_authors_normalized):
                original_paper = op
                break

        # Find new papers where ANY first author from this original paper
        # appears ANYWHERE in the author list (not just as first author)
        matching_new_papers = []
        for np in new_papers:
            # Get all authors from the new paper
            np_authors = np.get('authors', [])
            np_authors_normalized = [normalize_author_name(a) for a in np_authors]

            # Check if ANY first author from original paper is in the author list
            if any(fa in np_authors_normalized for fa in first_authors_normalized):
                matching_new_papers.append(np)

        lookup_entry = {
            'original_paper_title': paper_title,
            'original_paper_arxiv_id': original_paper['arxiv_id'] if original_paper else None,
            'first_authors': first_authors,
            'normalized_first_authors': first_authors_normalized,
            'gscholar_urls': entry['gscholar_urls'],
            'new_papers_count': len(matching_new_papers),
            'new_paper_arxiv_ids': [np['arxiv_id'] for np in matching_new_papers],
            'new_paper_titles': [np['title'] for np in matching_new_papers],
            'expected_new_papers': min(3, len(matching_new_papers)),
        }
        original_papers_lookup.append(lookup_entry)

        print(f"  {paper_title[:50]}...")
        print(f"    First authors: {first_authors}")
        print(f"    New papers found: {len(matching_new_papers)}")

    return {'original_papers': original_papers_lookup}


def download_figures_for_papers(papers: List[Dict], prefix: str, skip_figures: bool = False, use_llm: bool = True) -> List[Dict]:
    """Download Figure 1 PNG from arXiv source for each paper.

    Uses 2-stage approach:
    1. Automatic LaTeX parsing to find Figure 1 and its image file
    2. If Stage 1 fails and use_llm=True, use LLM to read .tex files

    Only extracts PNG images. Sets figure_1_path to None if no PNG exists.

    Args:
        papers: List of paper dicts with arxiv_id
        prefix: Prefix for output filenames ('original' or 'new')
        skip_figures: If True, skip downloading
        use_llm: If True, use LLM as fallback for figure identification

    Returns:
        Updated papers list with figure_1_path set
    """
    if skip_figures:
        print(f"\n=== Skipping Figure 1 downloads for {prefix} papers ===")
        return papers

    print(f"\n=== Downloading Figure 1 PNG for {len(papers)} {prefix} papers ===")
    print("Using 2-stage approach: (1) LaTeX parsing, (2) LLM fallback")

    # Load model for Stage 2 if requested
    model = None
    if use_llm:
        try:
            from src.browsergym.eval.eval_utils.models import load_model
            model = load_model("gemma-google-ai")
            print("LLM model loaded for Stage 2 fallback")
        except Exception as e:
            print(f"WARNING: Could not load LLM model: {e}")
            print("Will use Stage 1 (LaTeX parsing) only")

    found_count = 0
    not_found_count = 0

    for i, paper in enumerate(papers):
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            paper['figure_1_path'] = None
            continue

        print(f"  [{i+1}/{len(papers)}] {paper['title'][:40]}...")

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            success, msg, files = download_arxiv_source(arxiv_id, temp_dir)

            if not success:
                print(f"    ERROR downloading source: {msg}")
                paper['figure_1_path'] = None
                not_found_count += 1
                continue

            source_dir = os.path.join(temp_dir, arxiv_id.replace('/', '_'))
            success, fig_path, msg = extract_figure_1_from_source(source_dir, arxiv_id, model=model)

            if success and fig_path:
                # Verify it's a PNG
                if not fig_path.lower().endswith('.png'):
                    print(f"    WARNING: Found non-PNG file, skipping: {fig_path}")
                    paper['figure_1_path'] = None
                    not_found_count += 1
                    continue

                import shutil
                dest_filename = f"{prefix}_{i+1}_fig1.png"
                dest_path = os.path.join(FIGURES_DIR, dest_filename)

                shutil.copy2(fig_path, dest_path)
                paper['figure_1_path'] = f"data/gold_figures/{dest_filename}"
                print(f"    {msg}")
                print(f"    Saved: {dest_filename}")
                found_count += 1
            else:
                print(f"    No PNG Figure 1 found: {msg}")
                paper['figure_1_path'] = None
                not_found_count += 1

        time.sleep(1)  # Rate limiting

    print(f"\nFigure extraction summary for {prefix} papers:")
    print(f"  Found PNG: {found_count}/{len(papers)}")
    print(f"  Not found: {not_found_count}/{len(papers)}")

    return papers


def detect_world_models_for_papers(papers: List[Dict], skip: bool = False) -> List[Dict]:
    """Detect which papers mention 'world models' in related works.

    Args:
        papers: List of paper dicts
        skip: If True, skip detection

    Returns:
        Updated papers list with has_world_models set
    """
    if skip:
        print("\n=== Skipping world models detection ===")
        return papers

    print(f"\n=== Detecting 'world models' mentions ===")

    for i, paper in enumerate(papers):
        arxiv_id = paper.get('arxiv_id')
        if not arxiv_id:
            paper['has_world_models'] = False
            continue

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        try:
            import tempfile
            import requests

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                tmp.write(response.content)
                tmp_path = tmp.name

            has_world_models, _ = detect_world_models_in_pdf(tmp_path)
            paper['has_world_models'] = has_world_models

            os.unlink(tmp_path)

            status = "YES" if has_world_models else "no"
            print(f"  [{i+1}/{len(papers)}] {paper['title'][:40]}... - {status}")

            time.sleep(0.5)

        except Exception as e:
            print(f"  [{i+1}/{len(papers)}] {paper['title'][:40]}... - ERROR: {e}")
            paper['has_world_models'] = False

    return papers


def save_json(data: Any, filename: str):
    """Save data to JSON file."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess gold data for sheets_10 evaluator")
    parser.add_argument('--skip-figures', action='store_true',
                        help="Skip downloading Figure 1 images")
    parser.add_argument('--skip-scholar', action='store_true',
                        help="Skip Google Scholar scraping")
    parser.add_argument('--skip-world-models', action='store_true',
                        help="Skip world models detection in PDFs")
    args = parser.parse_args()

    print("=" * 60)
    print("Preprocessing Gold Data for sheets_10_paper_sorting")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"\nUsing ONLY Gold Labels Sheet: {GOLD_LABELS_SHEET_ID}")
    print("(Gold Implementation sheet is for TESTING only)")

    # Ensure directories exist
    ensure_directories()

    # Initialize Google services
    print("\n=== Initializing Google Services ===")
    DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

    # Step 1: Extract data from Gold Labels sheet
    entries = extract_gold_labels_data(SHEETS_SERVICE)

    if not entries:
        print("ERROR: No entries extracted from Gold Labels sheet")
        return

    # Step 2: Fetch original papers metadata from arXiv
    original_papers = fetch_original_papers_metadata(entries)

    # Step 3: Discover new papers (via gscholar or direct links)
    new_papers = discover_all_new_papers(entries, skip_scholar=args.skip_scholar)

    # Step 4: Build author lookup table
    author_lookup = build_author_lookup(entries, original_papers, new_papers)

    # Step 5: Download Figure 1 images (optional)
    original_papers = download_figures_for_papers(original_papers, 'original', skip_figures=args.skip_figures)
    new_papers = download_figures_for_papers(new_papers, 'new', skip_figures=args.skip_figures)

    # Step 6: Detect world models mentions (optional)
    if not args.skip_world_models:
        all_papers = original_papers + new_papers
        all_papers = detect_world_models_for_papers(all_papers)
        original_papers = all_papers[:len(original_papers)]
        new_papers = all_papers[len(original_papers):]

    # Save all data
    print("\n=== Saving Preprocessed Data ===")

    save_json({
        "papers": original_papers,
        "count": len(original_papers),
        "source_folder_id": SOURCE_FOLDER_ID,
        "generated_at": datetime.now().isoformat()
    }, "gold_papers.json")

    save_json({
        "papers": new_papers,
        "count": len(new_papers),
        "dest_folder_id": DEST_FOLDER_ID,
        "generated_at": datetime.now().isoformat()
    }, "gold_new_papers.json")

    save_json({
        "original_papers": author_lookup.get('original_papers', []),
        "count": len(author_lookup.get('original_papers', [])),
        "generated_at": datetime.now().isoformat()
    }, "author_papers_lookup.json")

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Original papers: {len(original_papers)}")
    print(f"New papers: {len(new_papers)}")
    print(f"Paper-centric lookups: {len(author_lookup.get('original_papers', []))} (should be 7, one per original paper)")
    print(f"Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
