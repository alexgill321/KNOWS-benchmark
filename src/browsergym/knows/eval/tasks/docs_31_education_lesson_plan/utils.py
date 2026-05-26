"""Utility functions for education lesson plan evaluation."""

import glob
import hashlib
import os
import re
import shutil
import tempfile
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from src.browsergym.knows.eval.eval_utils.google_services_utils import extract_hyperlinks_from_doc
from src.browsergym.knows.eval.eval_utils.image_utils import (
    extract_image_location,
    extract_image_location_size_feature_based,
)
from src.browsergym.knows.eval.eval_utils.parallel_utils import fast_parallel_vlm_calls  # type: ignore
from src.browsergym.knows.eval.eval_utils.utils import image_id_from_path, rgb_to_hex


# Phrases that the parser anchors on. Instance task.md files MUST keep this wording:
#   "I am a <Audience Level> educator" and "My class is on <Subject>"
_AUDIENCE_RE = re.compile(r"I am an?\s+([A-Z][A-Za-z ]+?)\s+educator", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"My class is on\s+([A-Z][A-Za-z ]+?)[.\n]", re.IGNORECASE)

# US Letter page size in PT (points), used as the fallback when a doc lacks
# documentStyle.pageSize metadata. 1pt = 1/72 inch.
LETTER_WIDTH_PT = 612
LETTER_HEIGHT_PT = 792
# CP5 Step 2 considers an image "at the bottom" if it lies within the bottom
# fraction of the page. 3/5 → bottom two-fifths (loose enough that inline
# tables placed at end-of-content can still qualify).
LOWER_REGION_PAGE_FRACTION = 3 / 5


def get_doc_page_dimensions_px(doc_content: Optional[Dict], dpi: int) -> Tuple[int, int, int, int]:
    """Page dimensions from doc_content.documentStyle.pageSize at the given DPI.

    Returns (width_px, height_px, width_pt, height_pt). Falls back to US Letter
    if the doc lacks pageSize metadata or doc_content is None.
    """
    page_size = (doc_content.get('documentStyle', {}).get('pageSize', {})
                 if doc_content else {})
    width_pt = page_size.get('width', {}).get('magnitude', LETTER_WIDTH_PT)
    height_pt = page_size.get('height', {}).get('magnitude', LETTER_HEIGHT_PT)
    return int(width_pt * dpi / 72), int(height_pt * dpi / 72), width_pt, height_pt


def dedup_files_by_sha256(paths: List[str]) -> List[str]:
    """Return paths whose file contents have unique SHA-256 hashes (first wins).

    Files that can't be opened are silently skipped (treated as missing).
    """
    seen = set()
    unique: List[str] = []
    for path in paths:
        try:
            with open(path, 'rb') as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            continue
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique


def parse_subject_and_audience(task_md_path: str) -> Tuple[str, str]:
    """Extract subject and audience level from an instance's task.md.

    Returns:
        (subject, audience_level), e.g. ("Chemistry", "Middle School").

    Raises:
        ValueError: if either field cannot be extracted (silent defaults would
            misgrade the document).
    """
    with open(task_md_path, "r", encoding="utf-8") as f:
        text = f.read()

    audience_match = _AUDIENCE_RE.search(text)
    subject_match = _SUBJECT_RE.search(text)
    if not audience_match or not subject_match:
        raise ValueError(
            f"Could not parse subject/audience from {task_md_path}. "
            f"Expected phrases 'I am a <X> educator' and 'My class is on <Y>'."
        )
    return subject_match.group(1).strip(), audience_match.group(1).strip()


def validate_topics_subject_related(
    topics: List[str], subject: str, model_instance, max_workers: int = 5,
) -> Dict:
    """Validate that all topics are related to the given subject.

    Audience-appropriateness is intentionally NOT checked here — the
    evaluator's audience-appropriateness step handles that separately so the
    two scores stay independent.
    """
    if not topics:
        return {
            'all_valid': False,
            'topics': [],
            'validations': {},
            'reasons': {'_empty': 'No topics provided'},
        }

    vlm_tasks = [
        {
            'id': topic,
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": f"You are a {subject} education expert. Answer only 'Yes' or 'No'."}]},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"Is '{topic}' a topic related to {subject}? Answer only 'Yes' or 'No'."
                )}]},
            ],
        }
        for topic in topics
    ]
    llm_results = fast_parallel_vlm_calls(vlm_tasks, model_instance, max_workers=max_workers)

    validations, reasons = {}, {}
    for topic in topics:
        passed = llm_results.get(topic, False)
        validations[topic] = passed
        reasons[topic] = (
            f"LLM validated as {subject}-related" if passed else
            f"LLM rejected as not {subject}-related"
        )

    return {
        'all_valid': all(validations.values()),
        'topics': topics,
        'validations': validations,
        'reasons': reasons,
    }


def validate_topics_engaging(
    topics: List[str], audience_level: str, model_instance, max_workers: int = 5,
) -> Dict:
    """Validate that topics are engaging/fun for the given audience level."""
    if not topics:
        return {
            'all_valid': False,
            'topics': [],
            'validations': {},
            'reasons': {'_empty': 'No topics provided'},
        }

    vlm_tasks = [
        {
            'id': topic,
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": "You are evaluating topic engagement for students. Answer only 'Yes' or 'No'."}]},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"Would '{topic}' be an engaging, interesting topic likely to be "
                    f"enjoyed by {audience_level} students? Answer only 'Yes' or 'No'."
                )}]},
            ],
        }
        for topic in topics
    ]
    llm_results = fast_parallel_vlm_calls(vlm_tasks, model_instance, max_workers=max_workers)

    validations, reasons = {}, {}
    for topic in topics:
        passed = llm_results.get(topic, False)
        validations[topic] = passed
        reasons[topic] = (
            f"LLM judged engaging for {audience_level} students" if passed else
            f"LLM judged not engaging for {audience_level} students"
        )

    return {
        'all_valid': all(validations.values()),
        'topics': topics,
        'validations': validations,
        'reasons': reasons,
    }


def classify_summary_paragraphs_as_facts(
    summary_facts_list: List[Dict], model_instance,
    max_workers: int = 5, min_text_length: int = 20,
) -> List[Dict]:
    """LLM-classify summary paragraphs; return only those judged as fact lists.

    Falls back to a color-or-bullet static filter on LLM failure or all-False
    result (heuristic for API outages).
    """
    candidates = [f for f in summary_facts_list if len(f['text']) > min_text_length]
    if not candidates:
        return []
    fallback = [f for f in candidates if f['is_bullet'] or f['color'] is not None]
    if model_instance is None:
        return fallback

    vlm_tasks = [
        {
            'id': str(i),
            'messages': [
                {"role": "system", "content": [{"type": "text", "text": (
                    "Answer 'Yes' if the paragraph is primarily a list of factual "
                    "claims about a topic, 'No' if it is a heading, introduction, "
                    "narrative summary, or other non-fact text."
                )}]},
                {"role": "user", "content": [{"type": "text", "text": (
                    f"Is this paragraph primarily a list of facts? Answer 'Yes' or 'No'.\n\n"
                    f"---\n{f['text'][:1500]}\n---"
                )}]},
            ],
        }
        for i, f in enumerate(candidates)
    ]
    try:
        results = fast_parallel_vlm_calls(vlm_tasks, model_instance, max_workers=max_workers)
        result_map = {str(i): results.get(str(i), False) for i in range(len(candidates))}
        if vlm_uniform_failure_warning(result_map):
            return fallback
        return [candidates[i] for i in range(len(candidates)) if result_map[str(i)]]
    except Exception:
        return fallback


def images_in_single_table_row(doc_content: Optional[Dict]) -> bool:
    """True iff every inline image in the doc lives in one single tableRow.

    Detects the "side-by-side via 1xN table" layout that agents typically use.
    Independent of PDF rasterization and SIFT location finding.
    """
    if not doc_content:
        return False
    inline_ids = set(doc_content.get('inlineObjects', {}).keys())
    if len(inline_ids) < 2:
        return False
    body = doc_content.get('body', {}).get('content', [])
    for el in body:
        if 'table' not in el:
            continue
        rows = el['table'].get('tableRows', [])
        if len(rows) != 1:
            continue
        row_ids = set()
        for cell in rows[0].get('tableCells', []):
            for inner in cell.get('content', []):
                if 'paragraph' not in inner:
                    continue
                for sub in inner['paragraph'].get('elements', []):
                    if 'inlineObjectElement' in sub:
                        row_ids.add(sub['inlineObjectElement']['inlineObjectId'])
        if row_ids == inline_ids:
            return True
    return False


def vlm_uniform_failure_warning(
    results: Dict[str, bool], min_count: int = 2
) -> Optional[str]:
    """Warn when every fast_parallel_vlm_calls result is False.

    `fast_parallel_vlm_calls` swallows per-task exceptions and returns False,
    so a uniformly-False dict can mean either "LLM said no to everything" OR
    "every API call failed." Returns a short warning string when the result
    is suspicious (>= min_count tasks, all False), else None. Callers append
    the warning to step `details` so annotators see the ambiguity.
    """
    if len(results) >= min_count and not any(results.values()):
        return (f"({len(results)} LLM calls all returned False — may indicate "
                f"API errors rather than genuine 'No' answers)")
    return None


def count_unique_normalized(
    items: List[str], normalizer: Optional[Callable[[str], str]] = None
) -> Tuple[int, int, List[str]]:
    """Count unique items after normalization. Returns (unique_count, total_count, duplicate_examples)."""
    if normalizer is None:
        normalizer = lambda s: s.strip().lower()
    seen = set()
    duplicates = []
    for item in items:
        key = normalizer(item)
        if key in seen:
            duplicates.append(item)
        else:
            seen.add(key)
    return len(seen), len(items), duplicates


def get_main_topics(bullet_hierarchy: Optional[Dict]) -> List[Dict]:
    """Return the top-level (nesting_level == 0) topics from a bullet hierarchy.

    Centralized so all five grade_checkpoint_N functions extract topics
    identically and a None / malformed hierarchy is handled in one place.
    """
    if not bullet_hierarchy or 'topics' not in bullet_hierarchy:
        return []
    return [
        item for item in bullet_hierarchy['topics']
        if item.get('nesting_level', 0) == 0
    ]


def check_images_aligned_horizontally(image_paths: List[str], pdf_images_dir: str,
                                      last_page_only: bool = True,
                                      y_tolerance: float = 0.15,
                                      doc_content: dict = None,
                                      dpi: int = 150) -> Dict:
    """Check if images are aligned horizontally (side by side) using feature-based matching.

    Returns:
        {'aligned', 'locations_found', 'total_images', 'locations', 'details'}
    """
    size_lookup = {}
    if doc_content:
        for obj_id, obj_data in doc_content.get('inlineObjects', {}).items():
            embedded = obj_data.get('inlineObjectProperties', {}).get('embeddedObject', {})
            size = embedded.get('size')
            if size:
                size_lookup[obj_id] = size
        for obj_id, obj_data in doc_content.get('positionedObjects', {}).items():
            embedded = obj_data.get('positionedObjectProperties', {}).get('embeddedObject', {})
            size = embedded.get('size')
            if size:
                size_lookup[obj_id] = size

    search_dir = pdf_images_dir
    temp_dir = None
    if last_page_only:
        page_files = sorted(glob.glob(os.path.join(pdf_images_dir, "*.png")))
        if page_files:
            temp_dir = tempfile.mkdtemp(prefix="last_page_")
            shutil.copy2(page_files[-1], os.path.join(temp_dir, os.path.basename(page_files[-1])))
            search_dir = temp_dir

    try:
        locations = []
        for img_path in image_paths:
            try:
                loc = None
                obj_id = image_id_from_path(img_path)
                image_size = size_lookup.get(obj_id)
                if image_size:
                    loc = extract_image_location_size_feature_based(
                        img_path, image_size, search_dir, dpi=dpi
                    )
                if loc is None:
                    loc = extract_image_location(img_path, search_dir)
                if loc is not None:
                    locations.append(loc)
            except Exception as e:
                print(f"  -> Error locating image {os.path.basename(img_path)}: {e}")

        if len(locations) < 2:
            return {
                'aligned': False,
                'locations_found': len(locations),
                'total_images': len(image_paths),
                'locations': locations,
                'details': f"Only found {len(locations)}/{len(image_paths)} image locations on the last page"
            }

        # Page-aware tolerances (Letter, A4, custom — all handled by the helper).
        page_width, page_height, _, _ = get_doc_page_dimensions_px(doc_content, dpi)
        y_positions = [loc.y for loc in locations]
        y_range = max(y_positions) - min(y_positions)

        if y_range > page_height * y_tolerance:
            return {
                'aligned': False,
                'locations_found': len(locations),
                'total_images': len(image_paths),
                'locations': locations,
                'details': f"Images not on same row: Y positions vary by {y_range:.0f}px (tolerance: {page_height * y_tolerance:.0f}px)"
            }

        x_positions = sorted(loc.x for loc in locations)
        min_spread = int(page_width * 0.05)  # ≥5% of page width
        x_spread = x_positions[-1] - x_positions[0]

        if x_spread < min_spread:
            return {
                'aligned': False,
                'locations_found': len(locations),
                'total_images': len(image_paths),
                'locations': locations,
                'details': f"Images not spread horizontally: X spread is only {x_spread:.0f}px"
            }

        return {
            'aligned': True,
            'locations_found': len(locations),
            'total_images': len(image_paths),
            'locations': locations,
            'details': f"{len(locations)} images aligned horizontally (Y range: {y_range:.0f}px, X spread: {x_spread:.0f}px)"
        }
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def find_summary_start_index(doc_content: dict) -> Optional[int]:
    """Index of the first non-bulleted paragraph that follows the bullet hierarchy.

    Marks the boundary used by both the hierarchy parser (stops here) and the
    summary extractor (starts here). Returns None if no bullets exist or no
    non-bullet paragraph follows them.
    """
    body = doc_content.get('body', {}).get('content', []) if doc_content else []
    seen_bullet = False
    for i, element in enumerate(body):
        if 'paragraph' not in element:
            continue
        para = element['paragraph']
        is_bullet = 'bullet' in para
        if is_bullet:
            seen_bullet = True
            continue
        if seen_bullet:
            text = ''.join(
                e.get('textRun', {}).get('content', '')
                for e in para.get('elements', [])
            ).strip()
            if text:
                return i
    return None


_WHITE_RGB = (1.0, 1.0, 1.0)
_BLACK_RGB = (0.0, 0.0, 0.0)


def format_color(color) -> str:
    """Format a color value (RGB tuple or theme-color string) for display."""
    if isinstance(color, tuple):
        return rgb_to_hex(*color)
    return str(color)


def _read_run_color(text_style: dict):
    """Hashable color key for a textRun's style, or None.

    Returns an RGB tuple, or a "theme:NAME" string for Google theme colors.
    `rgbColor` is authoritative when present — white/black rgb does NOT fall
    through to themeColor for the same color slot.
    """
    bg = text_style.get('backgroundColor', {}).get('color', {})
    if bg:
        if 'rgbColor' in bg:
            rgb = bg['rgbColor']
            tup = (rgb.get('red', 0.0), rgb.get('green', 0.0), rgb.get('blue', 0.0))
            return tup if tup not in (_WHITE_RGB, _BLACK_RGB) else None
        if bg.get('themeColor'):
            return f"theme:{bg['themeColor']}"

    fg = text_style.get('foregroundColor', {}).get('color', {})
    if fg:
        if 'rgbColor' in fg:
            rgb = fg['rgbColor']
            tup = (rgb.get('red', 0.0), rgb.get('green', 0.0), rgb.get('blue', 0.0))
            # Near-black is the default text color, not a deliberate highlight.
            return tup if not all(c <= 0.25 for c in tup) else None
        if fg.get('themeColor'):
            return f"theme:{fg['themeColor']}"

    return None


def extract_summary_facts_with_colors(doc_content: dict) -> List[Dict]:
    """Extract facts and their colors from the summary section of the document.

    Summary section starts at the first non-bulleted paragraph after the last
    bulleted paragraph (structural detection).

    Returns:
        List of dicts: [{'text': str, 'color': (r, g, b) | str | None, 'is_bullet': bool}]
        Color is an RGB tuple for explicit colors, a "theme:<NAME>" string for
        Google theme colors, or None if no color metadata is present.
        is_bullet is True iff the paragraph has the 'bullet' field.
    """
    summary_facts = []
    start_idx = find_summary_start_index(doc_content)
    if start_idx is None:
        return summary_facts

    body = doc_content.get('body', {}).get('content', [])
    for element in body[start_idx:]:
        if 'paragraph' not in element:
            continue
        para = element['paragraph']
        is_bullet = 'bullet' in para

        full_text = ''
        colors_found = []
        for elem in para.get('elements', []):
            if 'textRun' not in elem:
                continue
            text_run = elem['textRun']
            full_text += text_run.get('content', '')

            color = _read_run_color(text_run.get('textStyle', {}))
            if color is not None:
                colors_found.append(color)

        full_text = full_text.strip()
        if len(full_text) < 5:
            continue

        dominant_color = Counter(colors_found).most_common(1)[0][0] if colors_found else None
        summary_facts.append({'text': full_text, 'color': dominant_color, 'is_bullet': is_bullet})

    return summary_facts


_PLAIN_URL_RE = re.compile(r"https?://\S+")


def extract_bullet_hierarchy_from_doc(document: dict) -> dict:
    """Parse Google Docs JSON using bullet nesting metadata.

    Topics are bullets at nestingLevel 0, websites at level 1, facts at level >= 2.
    Anything past the last bulleted paragraph is treated as the summary section
    and ignored here.

    Exceptions are not caught here — the caller is responsible for fault
    tolerance (so the underlying error reaches setup_document's logging path
    instead of being silently swallowed).
    """
    all_links = extract_hyperlinks_from_doc(doc_id=None, service=None, document=document)
    # First-wins on duplicate anchor text.
    text_to_url: Dict[str, str] = {}
    for link in all_links:
        text_to_url.setdefault(link['text'], link['url'])

    topics: List[Dict] = []
    stack: List[Tuple[int, Dict]] = []

    summary_start = find_summary_start_index(document) or float('inf')
    body = document.get('body', {}).get('content', [])

    for idx, element in enumerate(body):
        if idx >= summary_start:
            break
        if 'paragraph' not in element:
            continue
        para = element['paragraph']
        if 'bullet' not in para:
            continue

        text_content = ''
        url = None
        for elem in para.get('elements', []):
            if 'textRun' not in elem:
                continue
            text_run = elem['textRun']
            text_content += text_run.get('content', '')
            style = text_run.get('textStyle', {})
            if 'link' in style:
                url = style['link'].get('url')

        text_content = text_content.strip()
        if not text_content:
            continue
        level = para['bullet'].get('nestingLevel', 0)
        if not url:
            # Fallback 1: anchor-text lookup.
            url = text_to_url.get(text_content)
        if not url and level >= 1:
            # Fallback 2: raw URL in bullet text. Restricted to level≥1 so a
            # topic title that mentions a URL isn't polluted with a stray URL.
            match = _PLAIN_URL_RE.search(text_content)
            if match:
                candidate = match.group(0)
                # Strip trailing punctuation. Only strip ')' when unbalanced
                # so Wikipedia paths like "Gunk_(mereology)" stay intact.
                while candidate:
                    last = candidate[-1]
                    if last in '.,;':
                        candidate = candidate[:-1]
                    elif last == ')' and candidate.count('(') < candidate.count(')'):
                        candidate = candidate[:-1]
                    else:
                        break
                url = candidate

        item: Dict = {'text': text_content, 'nesting_level': level, 'children': []}
        if url:
            item['url'] = url

        while stack and stack[-1][0] >= level:
            stack.pop()

        if level == 0 or not stack:
            topics.append(item)
        else:
            stack[-1][1]['children'].append(item)

        stack.append((level, item))

    return {'topics': topics}
