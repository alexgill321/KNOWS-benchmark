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

Note: Figure 1 extraction is now handled by a separate script: extract_figures.py
Run that script after preprocessing to populate figure_1_path in the JSON files.

Usage:
    python preprocess.py [--skip-scholar] [--skip-world-models]
"""

import os
import sys
import time
import argparse
import re
from typing import List, Dict, Optional, Any
from datetime import datetime

# Local imports first (for BASE_PATH)
from utils import (
    get_base_path,
    BASE_PATH,
    DATA_DIR,
    ensure_data_directories,
    save_json,
    extract_arxiv_id_from_url,
    normalize_author_name,
    search_arxiv_by_title,
    search_arxiv_by_author,
    match_gscholar_to_arxiv_papers
)

sys.path.append(BASE_PATH)

from src.browsergym.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.eval.eval_utils.google_services_helpers import get_sheet_content

# Gold Labels Sheet ID (the ONLY source for preprocessing)
GOLD_LABELS_SHEET_ID = "1xQNSQBE7uw4-bPuCf1uDW4XJ-v5F_vPsXO2ooF1BdFM"

# Drive folder IDs (for reference)
SOURCE_FOLDER_ID = "1dfRMRjBHH4F1S9WMD6p6VqpYQZ-pbKWB"
DEST_FOLDER_ID = "1vk3FB8IumyHMBuBjI8fsUSdSyOVlFZPf"


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


def scrape_google_scholar(gscholar_url: str) -> List[Dict]:
    """Scrape paper titles from a Google Scholar profile.

    This function uses the scholarly library to get publication data.
    It returns basic publication info without doing per-publication fill calls
    (which are slow and rate-limited). The cross-referencing with arXiv
    is done separately via match_gscholar_to_arxiv_papers().

    Args:
        gscholar_url: Google Scholar profile URL

    Returns:
        List of dicts with 'title', 'eprint_url', and 'pub_url' fields
    """
    try:
        from scholarly import scholarly

        # Extract author ID from URL
        match = re.search(r'user=([a-zA-Z0-9_-]+)', gscholar_url)
        if not match:
            print(f"      Could not extract author ID from: {gscholar_url}")
            return []

        author_id = match.group(1)

        # Get author publications (without filling each one - too slow)
        author = scholarly.search_author_id(author_id)
        author = scholarly.fill(author, sections=['publications'])

        papers = []
        for pub in author.get('publications', []):
            title = pub.get('bib', {}).get('title', '')
            if not title:
                continue

            paper_data = {
                'title': title,
                'eprint_url': pub.get('eprint_url', ''),
                'pub_url': pub.get('pub_url', ''),
                'num_citations': pub.get('num_citations', 0),
            }
            papers.append(paper_data)

        return papers

    except ImportError:
        print("      Warning: scholarly package not installed")
        return []
    except Exception as e:
        print(f"      Error scraping Google Scholar: {e}")
        return []


def discover_new_papers_for_entry(entry: Dict, skip_scholar: bool = False, model=None) -> List[Dict]:
    """Discover new papers for a single Gold Labels entry.

    Uses cross-referencing approach:
    1. Scrape Google Scholar to get paper titles (authoritative list)
    2. Search arXiv by author name(s) to get papers with arXiv IDs
    3. Match GScholar titles to arXiv papers using multi-stage matching:
       - Direct URL extraction from eprint_url/pub_url
       - Exact title match (normalized)
       - Fuzzy title match (80% threshold)
       - LLM semantic match (if model provided)

    Falls back to direct arXiv URLs if no Google Scholar profile available.

    Args:
        entry: Entry from Gold Labels sheet
        skip_scholar: If True, skip Google Scholar scraping
        model: Optional LLM model for semantic title matching

    Returns:
        List of new paper dicts with arXiv metadata
    """
    new_papers = []
    seen_arxiv_ids = set()  # Deduplicate
    original_title = entry['original_paper_title']

    # Path A: Cross-reference Google Scholar with arXiv author search
    if entry['gscholar_urls'] and not skip_scholar:
        print(f"    Cross-referencing for: {original_title[:40]}...")

        # Step 1: Scrape Google Scholar to get paper titles
        all_scholar_papers = []
        for gscholar_url in entry['gscholar_urls']:
            print(f"      Scraping GScholar: {gscholar_url[:60]}...")
            papers = scrape_google_scholar(gscholar_url)
            all_scholar_papers.extend(papers)
            time.sleep(1)  # Rate limiting

        # Filter out the original paper
        all_scholar_papers = [
            p for p in all_scholar_papers
            if p.get('title', '').lower().strip() != original_title.lower().strip()
        ]
        print(f"      Found {len(all_scholar_papers)} papers on Google Scholar (excluding original)")

        if not all_scholar_papers:
            print(f"      No papers found on Google Scholar")
            return []

        # Step 2: Search arXiv by author name(s)
        all_arxiv_papers = []
        arxiv_ids_seen = set()
        for author_name in entry['first_authors']:
            print(f"      Searching arXiv for author: {author_name}...")
            arxiv_papers = search_arxiv_by_author(author_name, max_results=50)
            # Deduplicate across authors
            for p in arxiv_papers:
                if p['arxiv_id'] not in arxiv_ids_seen:
                    arxiv_ids_seen.add(p['arxiv_id'])
                    all_arxiv_papers.append(p)
            time.sleep(0.5)  # Rate limiting

        print(f"      Found {len(all_arxiv_papers)} papers on arXiv for author(s)")

        if not all_arxiv_papers:
            print(f"      No arXiv papers found for author(s), falling back to title search...")
            # Fallback: try title search for each GScholar paper
            for gs_paper in all_scholar_papers:
                gs_title = gs_paper.get('title', '')

                # First try direct URL extraction
                arxiv_id = None
                for url in [gs_paper.get('eprint_url', ''), gs_paper.get('pub_url', '')]:
                    arxiv_id = extract_arxiv_id_from_url(url)
                    if arxiv_id:
                        break

                # Then try title search
                if not arxiv_id:
                    result = search_arxiv_by_title(gs_title)
                    if result:
                        arxiv_id = result['arxiv_id']
                    time.sleep(0.3)

                if arxiv_id and arxiv_id not in seen_arxiv_ids:
                    seen_arxiv_ids.add(arxiv_id)
                    metadata = fetch_arxiv_metadata(arxiv_id)
                    if metadata:
                        paper = {
                            **metadata,
                            'first_author_normalized': normalize_author_name(metadata['first_author']),
                            'source': 'gscholar_title_search',
                            'gscholar_title': gs_title,
                            'associated_original_paper': original_title,
                            'associated_first_authors': entry['first_authors'],
                            'figure_1_path': None,
                            'has_world_models': False,
                        }
                        new_papers.append(paper)
                        print(f"        Added via title search: {arxiv_id}")
                    time.sleep(0.3)

            return new_papers

        # Step 3: Cross-reference GScholar papers with arXiv papers
        matched_papers = match_gscholar_to_arxiv_papers(
            all_scholar_papers,
            all_arxiv_papers,
            model=model
        )

        # Step 4: Build final paper list with full metadata
        for matched in matched_papers:
            arxiv_id = matched['arxiv_id']
            if arxiv_id in seen_arxiv_ids:
                continue
            seen_arxiv_ids.add(arxiv_id)

            # Use the matched data directly (already has full metadata from arXiv)
            paper = {
                'arxiv_id': matched['arxiv_id'],
                'title': matched['title'],
                'authors': matched['authors'],
                'first_author': matched['authors'][0] if matched['authors'] else '',
                'first_author_normalized': normalize_author_name(matched['authors'][0]) if matched['authors'] else '',
                'abstract': matched.get('abstract', ''),
                'arxiv_url': f"https://arxiv.org/abs/{matched['arxiv_id']}",
                'pdf_url': matched.get('pdf_url', ''),
                'source': 'gscholar_crossref',
                'gscholar_title': matched.get('gscholar_title', ''),
                'match_method': matched.get('match_method', ''),
                'associated_original_paper': original_title,
                'associated_first_authors': entry['first_authors'],
                'figure_1_path': None,
                'has_world_models': False,
            }
            new_papers.append(paper)

    # Path B: Direct arXiv URLs (fallback or when no gscholar)
    elif entry['direct_paper_links']:
        print(f"    Using direct arXiv links for: {original_title[:40]}...")

        for arxiv_url in entry['direct_paper_links']:
            arxiv_id = extract_arxiv_id_from_url(arxiv_url)
            if not arxiv_id:
                print(f"      Could not extract ID from: {arxiv_url}")
                continue

            if arxiv_id in seen_arxiv_ids:
                continue
            seen_arxiv_ids.add(arxiv_id)

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

    model = None
    try:
        from src.browsergym.eval.eval_utils.models import load_model
        model = load_model("gemini-2.5-flash-google-ai")
        print("LLM model loaded for fallback stages")
    except Exception as e:
        print(f"WARNING: Could not load LLM model: {e}")
        print("Will use automatic parsing only")

    for entry in entries:
        new_papers = discover_new_papers_for_entry(entry, skip_scholar, model=model)
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess gold data for sheets_10 evaluator")
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
    ensure_data_directories()

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

    # Note: Figure 1 extraction is now done separately via extract_figures.py

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
    print(f"\nNext step: Run extract_figures.py to populate figure_1_path")
    print(f"Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
