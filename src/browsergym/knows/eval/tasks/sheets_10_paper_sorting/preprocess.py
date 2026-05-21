#!/usr/bin/env python3
"""Preprocessing script for sheets_10_paper_sorting evaluator.

This script extracts gold data from the Gold Labels sheet ONLY.
The Gold Implementation sheet is used for testing, not preprocessing.

Gold Labels Sheet Structure:
- Column A (Og Paper): Original paper titles (papers in source folder)
- Column B (First Author(s)): First author name(s)
- Column C (gscholar Link): Google Scholar profile URL(s) for scraping
- Column D (Paper Links): Direct arXiv URLs (fallback when no gscholar)

Preprocessing Logic:
1. For each original paper, search arXiv by title to get full metadata
2. For new papers discovery:
   - If gscholar link exists: Scrape Google Scholar -> cross-reference arXiv
   - If no gscholar but Paper Links exist: Use provided arXiv URLs directly

Output Files (in instance_X/data/):
- gold_papers.json - Original papers with metadata from arXiv
- gold_new_papers.json - Expected new papers (from gscholar or direct links)
- author_papers_lookup.json - Mapping of authors to their papers

Note: Figure 1 extraction is handled by a separate script: extract_figures.py
Run that script after preprocessing to populate figure_1_path in the JSON files.

Usage:
    python preprocess.py --instance 1 --gold-sheet-id "SHEET_ID" --source-folder "FOLDER_ID" --dest-folder "FOLDER_ID"
    python preprocess.py --instance 2 --gold-sheet-id "SHEET_ID" --source-folder "FOLDER_ID" --dest-folder "FOLDER_ID" [--skip-scholar]
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
    get_data_dir,
    ensure_data_directories,
    save_json,
    load_json,
    extract_arxiv_id_from_url,
    normalize_author_name,
    search_arxiv_by_title,
    search_arxiv_by_author,
    fetch_arxiv_batch,
    match_gscholar_to_arxiv_papers
)

sys.path.append(BASE_PATH)

from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.google_services_helpers import get_sheet_content

# Instance-specific config defaults (instance 1 for backward compat)
INSTANCE_CONFIGS = {
    1: {
        'gold_sheet_id': '1xQNSQBE7uw4-bPuCf1uDW4XJ-v5F_vPsXO2ooF1BdFM',
        'source_folder_id': '1dfRMRjBHH4F1S9WMD6p6VqpYQZ-pbKWB',
        'dest_folder_id': '1vk3FB8IumyHMBuBjI8fsUSdSyOVlFZPf',
    },
    2: {
        'gold_sheet_id': '1Ii69xyjTzsqpYhrmNJWjznyoRoxiWDVJaGdDEI47_7s',
        'source_folder_id': '1Qm2gLrC3PhRqhlAI_WXBjYKqECdPOwBE',
        'dest_folder_id': None,  # Set by setup_run.py before each benchmark
    },
    3: {
        'gold_sheet_id': '1uYA5xL9Enij6kOWQHp84vCxc4XlqaKBZcgdtBLf3DLk',
        'source_folder_id': '1Fc1GthzO8dAuekt-L3dL4FfUbW7wjeZM',
        'dest_folder_id': None,  # Set by setup_run.py before each benchmark
    },
    4: {
        'gold_sheet_id': '1OYSkAGF6rTD2FzxdHu09FcqSemdJx8PnEL6PMujm_hg',
        'source_folder_id': '1NIx27u2aOywiZRzBNaeucR4x7yWNf8YB',
        'dest_folder_id': None,
    },
    5: {
        'gold_sheet_id': '1FYDf9HuzLCOppaGJsKp08hDzhyD1OfITv43CY3-O2Bo',
        'source_folder_id': '1xDDOPz_AH55ONpjQRgC__IWjgWtbw08O',
        'dest_folder_id': None,
    },
    # Add configs for new instances here as they are set up:
}


def extract_gold_labels_data(sheets_service, gold_sheet_id: str) -> List[Dict]:
    """Extract data from the Gold Labels sheet.

    Returns:
        List of dicts, one per original paper, with:
        - original_paper_title
        - first_authors (list)
        - gscholar_urls (list)
        - direct_paper_links (list of arXiv URLs, if no gscholar)
    """
    print("\n=== Extracting Gold Labels Data ===")

    sheet_raw = get_sheet_content(gold_sheet_id, sheets_service)
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

        # Parse first authors (comma or newline separated)
        first_authors_raw = first_authors_raw.replace('\n', ',')
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
    """Fetch paper metadata from arXiv API."""
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
    """Fetch full metadata for original papers from arXiv."""
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
            }
            original_papers.append(paper)
            print(f"    Found: {result['arxiv_id']} (match score: {result.get('match_score', 'N/A')})")
        else:
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
            }
            original_papers.append(paper)
            print(f"    NOT FOUND on arXiv")

        time.sleep(3)  # arXiv rate limit: max 1 request per 3 seconds

    print(f"\nFetched metadata for {len(original_papers)} original papers")
    return original_papers


def scrape_google_scholar(gscholar_url: str, top_n: int = 0) -> List[Dict]:
    """Scrape paper titles from a Google Scholar profile.

    Args:
        gscholar_url: Google Scholar profile URL.
        top_n: If > 0, only return the top N most-cited papers.
               If 0, return all papers (default).
    """
    try:
        from scholarly import scholarly

        match = re.search(r'user=([a-zA-Z0-9_-]+)', gscholar_url)
        if not match:
            print(f"      Could not extract author ID from: {gscholar_url}")
            return []

        author_id = match.group(1)

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

        # If top_n specified, sort by citations and take top N
        if top_n > 0 and papers:
            papers.sort(key=lambda p: p.get('num_citations', 0), reverse=True)
            papers = papers[:top_n]
            print(f"      Filtered to top {top_n} most-cited papers")

        return papers

    except ImportError:
        print("      Warning: scholarly package not installed")
        return []
    except Exception as e:
        print(f"      Error scraping Google Scholar: {e}")
        return []


def discover_new_papers_for_entry(entry: Dict, skip_scholar: bool = False, model=None, top_cited: int = 0, first_author_only: bool = False) -> List[Dict]:
    """Discover new papers for a single Gold Labels entry.

    Args:
        entry: Entry from Gold Labels sheet.
        skip_scholar: If True, skip Google Scholar scraping.
        model: Optional LLM model for semantic title matching.
        top_cited: If > 0, only consider the top N most-cited papers from gscholar.
    """
    new_papers = []
    seen_arxiv_ids = set()
    original_title = entry['original_paper_title']

    # Path A: Cross-reference Google Scholar with arXiv author search
    if entry['gscholar_urls'] and not skip_scholar:
        print(f"    Cross-referencing for: {original_title[:40]}...")

        all_scholar_papers = []
        for gscholar_url in entry['gscholar_urls']:
            print(f"      Scraping GScholar: {gscholar_url[:60]}...")
            papers = scrape_google_scholar(gscholar_url, top_n=top_cited)
            all_scholar_papers.extend(papers)
            time.sleep(1)

        all_scholar_papers = [
            p for p in all_scholar_papers
            if p.get('title', '').lower().strip() != original_title.lower().strip()
        ]
        print(f"      Found {len(all_scholar_papers)} papers on Google Scholar (excluding original)")

        if not all_scholar_papers:
            print(f"      No papers found on Google Scholar")
            return []

        all_arxiv_papers = []
        arxiv_ids_seen = set()
        for author_name in entry['first_authors']:
            print(f"      Searching arXiv for author: {author_name}...")
            arxiv_papers = search_arxiv_by_author(author_name, max_results=500, first_author_only=first_author_only)
            for p in arxiv_papers:
                if p['arxiv_id'] not in arxiv_ids_seen:
                    arxiv_ids_seen.add(p['arxiv_id'])
                    all_arxiv_papers.append(p)
            time.sleep(0.5)

        print(f"      Found {len(all_arxiv_papers)} papers on arXiv for author(s)")

        if not all_arxiv_papers:
            print(f"      No arXiv papers found for author(s), falling back...")

            # Step 1: Extract arxiv IDs directly from gscholar URLs (no API calls)
            ids_from_urls = {}
            needs_title_search = []
            for gs_paper in all_scholar_papers:
                gs_title = gs_paper.get('title', '')
                arxiv_id = None
                for url in [gs_paper.get('eprint_url', ''), gs_paper.get('pub_url', '')]:
                    arxiv_id = extract_arxiv_id_from_url(url)
                    if arxiv_id:
                        break
                if arxiv_id:
                    ids_from_urls[arxiv_id] = gs_title
                else:
                    needs_title_search.append(gs_paper)

            # Step 2: Batch-fetch all URL-extracted IDs in one API call
            if ids_from_urls:
                print(f"      Batch-fetching {len(ids_from_urls)} papers from gscholar URLs...")
                batch_results = fetch_arxiv_batch(list(ids_from_urls.keys()))
                for metadata in batch_results:
                    arxiv_id = metadata['arxiv_id']
                    if arxiv_id not in seen_arxiv_ids:
                        seen_arxiv_ids.add(arxiv_id)
                        paper = {
                            **metadata,
                            'first_author_normalized': normalize_author_name(metadata.get('authors', [''])[0]),
                            'source': 'gscholar_url_batch',
                            'gscholar_title': ids_from_urls.get(arxiv_id, ''),
                            'associated_original_paper': original_title,
                            'associated_first_authors': entry['first_authors'],
                            'figure_1_path': None,
                        }
                        new_papers.append(paper)
                        print(f"        Added via URL batch: {arxiv_id}")

            # Step 3: Title search only for papers without direct arxiv URLs
            if needs_title_search:
                print(f"      Title-searching {len(needs_title_search)} remaining papers...")
                for gs_paper in needs_title_search:
                    gs_title = gs_paper.get('title', '')
                    result = search_arxiv_by_title(gs_title)
                    if result:
                        arxiv_id = result['arxiv_id']
                        if arxiv_id not in seen_arxiv_ids:
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
                                }
                                new_papers.append(paper)
                                print(f"        Added via title search: {arxiv_id}")
                            time.sleep(3)
                    time.sleep(3)

            return new_papers

        matched_papers = match_gscholar_to_arxiv_papers(
            all_scholar_papers,
            all_arxiv_papers,
            model=model
        )

        for matched in matched_papers:
            arxiv_id = matched['arxiv_id']
            if arxiv_id in seen_arxiv_ids:
                continue
            seen_arxiv_ids.add(arxiv_id)

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
            }
            new_papers.append(paper)

    # Path B: No gscholar — search arXiv by author name and include all results
    else:
        print(f"    No gscholar for: {original_title[:40]}...")
        print(f"    Searching arXiv by author name directly")
        for author_name in entry['first_authors']:
            print(f"      Searching arXiv for author: {author_name}...")
            arxiv_papers = search_arxiv_by_author(author_name, max_results=500, first_author_only=first_author_only)
            for p in arxiv_papers:
                arxiv_id = p['arxiv_id']
                if arxiv_id in seen_arxiv_ids:
                    continue
                # Skip the original paper itself
                if arxiv_id == entry.get('original_paper_arxiv_id'):
                    continue
                seen_arxiv_ids.add(arxiv_id)
                paper = {
                    **p,
                    'first_author_normalized': normalize_author_name(p.get('authors', [''])[0]),
                    'source': 'arxiv_author_search',
                    'associated_original_paper': original_title,
                    'associated_first_authors': entry['first_authors'],
                    'figure_1_path': None,
                }
                new_papers.append(paper)
            print(f"      Added {len(new_papers)} papers from arXiv author search")
            time.sleep(0.5)

    return new_papers


def discover_all_new_papers(entries: List[Dict], skip_scholar: bool = False, top_cited: int = 0, first_author_only: bool = False) -> List[Dict]:
    """Discover new papers for all Gold Labels entries.

    Args:
        entries: List of entries from Gold Labels sheet.
        skip_scholar: If True, skip Google Scholar scraping.
        top_cited: If > 0, only consider the top N most-cited papers from gscholar.
        first_author_only: If True, only match papers where the author is first author.
    """
    print("\n=== Discovering New Papers ===")
    if top_cited > 0:
        print(f"Filtering to top {top_cited} most-cited papers per author")
    if first_author_only:
        print("Filtering to first-author papers only")

    all_new_papers = []

    model = None
    try:
        from src.browsergym.knows.eval.eval_utils.models import load_model
        model = load_model("gemini-2.5-flash-google-ai")
        print("LLM model loaded for fallback stages")
    except Exception as e:
        print(f"WARNING: Could not load LLM model: {e}")
        print("Will use automatic parsing only")

    for entry in entries:
        new_papers = discover_new_papers_for_entry(entry, skip_scholar, model=model, top_cited=top_cited, first_author_only=first_author_only)
        all_new_papers.extend(new_papers)
        print(f"    Found {len(new_papers)} new papers for {entry['original_paper_title'][:40]}...")

    print(f"\nTotal new papers discovered: {len(all_new_papers)}")
    return all_new_papers


def build_author_lookup(entries: List[Dict], original_papers: List[Dict], new_papers: List[Dict]) -> Dict:
    """Build a lookup table organized by ORIGINAL PAPER, not by author."""
    print("\n=== Building Paper-Centric Author Lookup ===")

    original_papers_lookup = []

    for entry in entries:
        paper_title = entry['original_paper_title']
        first_authors = entry['first_authors']
        first_authors_normalized = entry['first_authors_normalized']

        original_paper = None
        for op in original_papers:
            op_first_authors_norm = op.get('gold_labels_first_authors_normalized', [])
            if any(fa in op_first_authors_norm for fa in first_authors_normalized):
                original_paper = op
                break

        matching_new_papers = []
        for np in new_papers:
            np_authors = np.get('authors', [])
            np_authors_normalized = [normalize_author_name(a) for a in np_authors]

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


def rematch_missing_papers(entries: List[Dict], instance: int, top_cited: int = 0, first_author_only: bool = False) -> List[Dict]:
    """Re-attempt matching for GScholar papers not already in gold_new_papers.json.

    Loads existing gold data, re-scrapes GScholar and arXiv, filters out
    already-matched papers, and runs all matching strategies (including LLM)
    on the remaining unmatched papers.

    Args:
        entries: List of entries from Gold Labels sheet.
        instance: Instance number for loading existing data.
        top_cited: If > 0, only consider top N most-cited papers from gscholar.

    Returns:
        List of newly matched paper dicts to merge with existing gold data.
    """
    print("\n=== Rematch Mode: Finding Missing Papers ===")

    existing_data = load_json("gold_new_papers.json", instance)
    if not existing_data or 'papers' not in existing_data:
        print("ERROR: No existing gold_new_papers.json found. Run full preprocessing first.")
        return []

    existing_arxiv_ids = {p['arxiv_id'] for p in existing_data['papers'] if 'arxiv_id' in p}
    print(f"Loaded {len(existing_arxiv_ids)} existing papers from gold_new_papers.json")

    model = None
    try:
        from src.browsergym.knows.eval.eval_utils.models import load_model
        model = load_model("gemini-2.5-flash-google-ai")
        print("LLM model loaded for semantic matching")
    except Exception as e:
        print(f"WARNING: Could not load LLM model: {e}")

    newly_matched = []

    for entry in entries:
        original_title = entry['original_paper_title']
        if not entry['gscholar_urls']:
            continue

        print(f"\n  Processing: {original_title[:50]}...")

        # Re-scrape GScholar
        all_scholar_papers = []
        for gscholar_url in entry['gscholar_urls']:
            print(f"    Scraping GScholar: {gscholar_url[:60]}...")
            papers = scrape_google_scholar(gscholar_url, top_n=top_cited)
            all_scholar_papers.extend(papers)
            time.sleep(1)

        all_scholar_papers = [
            p for p in all_scholar_papers
            if p.get('title', '').lower().strip() != original_title.lower().strip()
        ]
        print(f"    Found {len(all_scholar_papers)} GScholar papers (excluding original)")

        if not all_scholar_papers:
            continue

        # Re-search arXiv by author
        all_arxiv_papers = []
        arxiv_ids_seen = set()
        for author_name in entry['first_authors']:
            print(f"    Searching arXiv for author: {author_name}...")
            arxiv_papers = search_arxiv_by_author(author_name, max_results=500, first_author_only=first_author_only)
            for p in arxiv_papers:
                if p['arxiv_id'] not in arxiv_ids_seen:
                    arxiv_ids_seen.add(p['arxiv_id'])
                    all_arxiv_papers.append(p)
            time.sleep(0.5)

        print(f"    Found {len(all_arxiv_papers)} arXiv papers for author(s)")

        if not all_arxiv_papers:
            continue

        # Filter out arXiv papers already in gold
        remaining_arxiv = [p for p in all_arxiv_papers if p['arxiv_id'] not in existing_arxiv_ids]
        print(f"    {len(remaining_arxiv)} arXiv papers not yet in gold (skipping {len(all_arxiv_papers) - len(remaining_arxiv)} already matched)")

        if not remaining_arxiv:
            continue

        # Run full matching on remaining papers
        matched = match_gscholar_to_arxiv_papers(
            all_scholar_papers,
            remaining_arxiv,
            model=model
        )

        for m in matched:
            arxiv_id = m['arxiv_id']
            if arxiv_id in existing_arxiv_ids:
                continue

            # Fetch full metadata if not already present
            if 'abstract' not in m or not m.get('abstract'):
                metadata = fetch_arxiv_metadata(arxiv_id)
                if metadata:
                    m.update(metadata)

            paper = {
                'arxiv_id': m['arxiv_id'],
                'title': m.get('title', ''),
                'authors': m.get('authors', []),
                'first_author': m.get('authors', [''])[0] if m.get('authors') else '',
                'first_author_normalized': normalize_author_name(m.get('authors', [''])[0]) if m.get('authors') else '',
                'abstract': m.get('abstract', ''),
                'arxiv_url': f"https://arxiv.org/abs/{m['arxiv_id']}",
                'pdf_url': m.get('pdf_url', ''),
                'source': 'rematch',
                'gscholar_title': m.get('gscholar_title', ''),
                'match_method': m.get('match_method', ''),
                'associated_original_paper': original_title,
                'associated_first_authors': entry['first_authors'],
                'figure_1_path': None,
            }
            newly_matched.append(paper)
            existing_arxiv_ids.add(arxiv_id)
            print(f"    NEW MATCH: {arxiv_id} - {m.get('title', '')[:60]}... ({m.get('match_method', '')})")

    print(f"\n=== Rematch Complete: {len(newly_matched)} new papers found ===")
    return newly_matched


def main():
    parser = argparse.ArgumentParser(description="Preprocess gold data for sheets_10 evaluator")
    parser.add_argument('--instance', type=int, default=1,
                        help="Instance number (default: 1)")
    parser.add_argument('--gold-sheet-id', type=str, default=None,
                        help="Google Sheet ID for Gold Labels (overrides instance config)")
    parser.add_argument('--source-folder', type=str, default=None,
                        help="Source Google Drive folder ID (overrides instance config)")
    parser.add_argument('--dest-folder', type=str, default=None,
                        help="Destination Google Drive folder ID (overrides instance config)")
    parser.add_argument('--skip-scholar', action='store_true',
                        help="Skip Google Scholar scraping")
    parser.add_argument('--top-cited', type=int, default=0,
                        help="Only consider top N most-cited papers per author from gscholar (0 = all)")
    parser.add_argument('--skip-authors', type=str, default=None,
                        help="Comma-separated row numbers (1-indexed) to skip during new paper discovery. "
                             "Existing data for skipped authors is preserved from previous runs.")
    parser.add_argument('--skip-originals', action='store_true',
                        help="Skip fetching original papers metadata (use existing gold_papers.json)")
    parser.add_argument('--rematch-only', action='store_true',
                        help="Only re-attempt matching for GScholar papers not already in gold_new_papers.json. "
                             "Loads existing gold data, re-scrapes GScholar/arXiv, and uses LLM matching "
                             "for papers that weren't matched in previous runs.")
    parser.add_argument('--first-author-only', action='store_true',
                        help="Only include arXiv papers where the searched author is first author. "
                             "Useful for prolific authors with many co-authored papers.")
    args = parser.parse_args()

    instance = args.instance

    # Resolve config: CLI args override instance defaults
    config = INSTANCE_CONFIGS.get(instance, {})
    gold_sheet_id = args.gold_sheet_id or config.get('gold_sheet_id')
    source_folder_id = args.source_folder or config.get('source_folder_id')
    dest_folder_id = args.dest_folder or config.get('dest_folder_id')

    if not gold_sheet_id:
        print(f"ERROR: No gold_sheet_id configured for instance {instance}. "
              f"Use --gold-sheet-id or add to INSTANCE_CONFIGS.")
        sys.exit(1)

    print("=" * 60)
    print(f"Preprocessing Gold Data for sheets_10_paper_sorting (instance {instance})")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Gold Labels Sheet: {gold_sheet_id}")
    print(f"Source folder: {source_folder_id}")
    print(f"Dest folder: {dest_folder_id}")

    # Ensure directories exist
    ensure_data_directories(instance)

    # Initialize Google services
    print("\n=== Initializing Google Services ===")
    DRIVE_SERVICE, SHEETS_SERVICE = initialize_google_services(service_type="sheets")

    # Step 1: Extract data from Gold Labels sheet
    entries = extract_gold_labels_data(SHEETS_SERVICE, gold_sheet_id)

    if not entries:
        print("ERROR: No entries extracted from Gold Labels sheet")
        return

    # Handle --rematch-only: re-attempt matching for unmatched papers and exit
    if args.rematch_only:
        newly_matched = rematch_missing_papers(entries, instance, top_cited=args.top_cited, first_author_only=args.first_author_only)

        existing_data = load_json("gold_new_papers.json", instance)
        existing_papers = existing_data.get('papers', []) if existing_data else []

        if newly_matched:
            merged_papers = existing_papers + newly_matched
            save_json({
                "papers": merged_papers,
                "count": len(merged_papers),
                "dest_folder_id": dest_folder_id,
                "generated_at": datetime.now().isoformat()
            }, "gold_new_papers.json", instance)
            print(f"\nMerged {len(newly_matched)} new papers into gold_new_papers.json (total: {len(merged_papers)})")
        else:
            merged_papers = existing_papers
            print("\nNo new papers found.")

        # Always rebuild author lookup to reflect current gold data
        existing_orig_data = load_json("gold_papers.json", instance)
        original_papers = existing_orig_data.get('papers', []) if existing_orig_data else []
        author_lookup = build_author_lookup(entries, original_papers, merged_papers)

        save_json({
            "original_papers": author_lookup.get('original_papers', []),
            "count": len(author_lookup.get('original_papers', [])),
            "generated_at": datetime.now().isoformat()
        }, "author_papers_lookup.json", instance)
        print(f"Updated author_papers_lookup.json")

        print(f"Finished at: {datetime.now().isoformat()}")
        return

    # Handle --skip-authors
    skip_rows = set()
    if args.skip_authors:
        skip_rows = {int(r.strip()) for r in args.skip_authors.split(',')}
        print(f"\nSkipping rows: {skip_rows}")

    entries_to_process = [e for e in entries if e['row_index'] not in skip_rows]
    skipped_entries = [e for e in entries if e['row_index'] in skip_rows]

    if skipped_entries:
        print(f"Processing {len(entries_to_process)} entries, skipping {len(skipped_entries)}:")
        for e in skipped_entries:
            print(f"  Skipped: [{e['row_index']}] {e['original_paper_title'][:50]}...")

    # Step 2: Fetch original papers metadata from arXiv
    if args.skip_originals:
        existing_orig_data = load_json("gold_papers.json", instance)
        if existing_orig_data and 'papers' in existing_orig_data:
            original_papers = existing_orig_data['papers']
            print(f"\nSkipping original papers fetch — loaded {len(original_papers)} from existing gold_papers.json")
        else:
            print("ERROR: --skip-originals but no existing gold_papers.json found")
            return
    else:
        original_papers = fetch_original_papers_metadata(entries_to_process)
        # Merge preserved originals from skipped rows
        if skip_rows:
            existing_orig_data = load_json("gold_papers.json", instance)
            if existing_orig_data and 'papers' in existing_orig_data:
                skipped_titles = {e['original_paper_title'] for e in skipped_entries}
                preserved = [
                    p for p in existing_orig_data['papers']
                    if p.get('title', '') in skipped_titles or
                       p.get('gold_labels_first_authors', [''])[0] in
                       {a for e in skipped_entries for a in e['first_authors']}
                ]
                print(f"Preserved {len(preserved)} original papers from skipped rows")
                original_papers = preserved + original_papers

    # Load existing new papers to preserve skipped authors' results
    existing_new_papers = []
    if skip_rows:
        existing_new_data = load_json("gold_new_papers.json", instance)
        if existing_new_data and 'papers' in existing_new_data:
            existing_new_papers = existing_new_data['papers']
            print(f"Loaded {len(existing_new_papers)} existing new papers from previous run")

    new_papers = discover_all_new_papers(entries_to_process, skip_scholar=args.skip_scholar, top_cited=args.top_cited, first_author_only=args.first_author_only)

    # Merge: keep existing new papers for skipped authors, add freshly discovered ones
    if existing_new_papers:
        # Get titles of skipped authors' original papers to identify their new papers
        skipped_original_titles = {e['original_paper_title'] for e in skipped_entries}
        preserved_papers = [
            p for p in existing_new_papers
            if p.get('associated_original_paper', '') in skipped_original_titles
        ]
        print(f"Preserved {len(preserved_papers)} new papers from skipped authors")
        new_papers = preserved_papers + new_papers

    # Step 4: Build author lookup table (uses ALL entries, including skipped)
    author_lookup = build_author_lookup(entries, original_papers, new_papers)

    # Save all data
    print("\n=== Saving Preprocessed Data ===")

    save_json({
        "papers": original_papers,
        "count": len(original_papers),
        "source_folder_id": source_folder_id,
        "generated_at": datetime.now().isoformat()
    }, "gold_papers.json", instance)

    save_json({
        "papers": new_papers,
        "count": len(new_papers),
        "dest_folder_id": dest_folder_id,
        "generated_at": datetime.now().isoformat()
    }, "gold_new_papers.json", instance)

    save_json({
        "original_papers": author_lookup.get('original_papers', []),
        "count": len(author_lookup.get('original_papers', [])),
        "generated_at": datetime.now().isoformat()
    }, "author_papers_lookup.json", instance)

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Instance: {instance}")
    print(f"Original papers: {len(original_papers)}")
    print(f"New papers: {len(new_papers)}")
    print(f"Paper-centric lookups: {len(author_lookup.get('original_papers', []))}")
    print(f"\nNext step: Run extract_figures.py --instance {instance}")
    print(f"Finished at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
