"""
Task-level utilities for slides_51_event_announcement_poster.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from rapidfuzz import fuzz
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_text_boxes_from_slide,
    get_text_style_from_shape,
    extract_speaker_notes_text,
    get_shape_background_fill,
    is_grey_color,
)
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_text_content
from src.browsergym.knows.eval.eval_utils.utils import is_bbox_mostly_inside, bbox_overlap_ratio
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_execute

PANCHEKHA_REFERENCE_URL = "https://browser.engineering/onepage.html"
_REFERENCE_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "instance_1", "data", "pavel_panchekha_reference.txt"
)


# Canonical RGB targets (0-1 range) for the descriptive color names used across
# instances 2-5. Values approximate common digital interpretations; pair with
# `is_color_close` and a generous tolerance to allow for designer variation.
COLORS = {
    "mint_green":      (0.60, 0.95, 0.70),
    "deep_ocean_blue": (0.00, 0.30, 0.50),
    "navy_blue":       (0.00, 0.00, 0.50),
    "deep_teal":       (0.00, 0.30, 0.30),
    "light_orange":    (1.00, 0.75, 0.45),
    "seafoam_green":   (0.50, 0.92, 0.75),
    "charcoal_grey":   (0.20, 0.25, 0.30),
    "warm_amber":      (1.00, 0.65, 0.10),
}


def is_color_close(color: Optional[Dict[str, Any]], target_rgb: Tuple[float, float, float],
                   tolerance: float = 0.30) -> bool:
    """Check whether a color dict is within Euclidean RGB distance of a target.

    Accepts colors in either ``{'r','g','b'}`` form (e.g. from
    ``get_shape_background_fill``) or ``{'red','green','blue'}`` form (e.g. from
    ``get_text_style_from_shape``).

    Args:
        color (dict): RGB color dict, or None.
        target_rgb (tuple): Target ``(r, g, b)`` floats in 0-1 range.
        tolerance (float): Maximum Euclidean distance to consider a match.

    Returns:
        bool: True if color is within tolerance of target_rgb.
    """
    if not color:
        return False
    r = color.get('r', color.get('red', 0))
    g = color.get('g', color.get('green', 0))
    b = color.get('b', color.get('blue', 0))
    tr, tg, tb = target_rgb
    distance = ((r - tr) ** 2 + (g - tg) ** 2 + (b - tb) ** 2) ** 0.5
    return distance <= tolerance


def classify_citation_group(
    group_urls: List[str],
    fetch_results: Dict[str, Any],
    verify_results: Dict[str, Any],
    group_prefix: str
) -> Tuple[bool, str]:
    """Aggregate citation verification results for a URL group (topic or bio).

    For each URL in the group, determines a state:
        - 'supported'     — fetched successfully and LLM verified claims are supported
        - 'not_supported' — fetched but LLM said claims not supported
        - 'unverifiable'  — fetch failed, no content to check

    Success rule (per the citation step):
        - Empty group                              → FAIL
        - All URLs unverifiable (fetch failed)     → PASS (full credit, neutral default)
        - At least one URL 'supported'             → PASS
        - All 'not_supported'                      → FAIL

    Args:
        group_urls: URLs classified into this group.
        fetch_results: Output of parallel_execute over fetch_page_text_content.
            Maps url → (content_or_None, status).
        verify_results: Output of parallel_execute over per-URL LLM verifier calls.
            Maps f'{group_prefix}::{url}' → LLM response string.
        group_prefix: Identifier used when building verify task ids ('topic' or 'bio').

    Returns:
        (success, detail_string)
    """
    if not group_urls:
        return False, "No URLs classified for this group"

    supported = 0
    not_supported = 0
    unverifiable = 0

    for url in group_urls:
        result = fetch_results.get(url)
        content = result[0] if result and result[0] else None
        if not content:
            unverifiable += 1
            continue
        resp = verify_results.get(f'{group_prefix}::{url}')
        if resp and resp.strip().lower().startswith("yes"):
            supported += 1
        else:
            not_supported += 1

    total = len(group_urls)
    if supported > 0:
        success = True
    elif unverifiable == total:
        # All failed to fetch → full credit (neutral default)
        success = True
    else:
        success = False

    detail = (f"{total} URL(s): {supported} supported, "
              f"{not_supported} not supported, {unverifiable} unverifiable")
    return success, detail


def load_panchekha_reference(max_chars: int = 20000) -> str:
    """Load the cached text of Pavel Panchekha's "Web Browser Engineering" book.

    On first call (or if the cache file is missing), fetches the HTML from
    browser.engineering/onepage.html using `fetch_page_text_content`, strips
    it to plain text, and caches it to disk. Subsequent calls load from disk.

    This text is used to ground the CP3 LLM check that verifies the poster
    summary is tied to Panchekha's work.

    Args:
        max_chars (int): Maximum characters to fetch/cache.

    Returns:
        str: Plain-text reference content, or empty string on failure.
    """
    if os.path.exists(_REFERENCE_CACHE_PATH):
        with open(_REFERENCE_CACHE_PATH, 'r', encoding='utf-8') as f:
            return f.read()

    content, status = fetch_page_text_content(PANCHEKHA_REFERENCE_URL, max_chars=max_chars, timeout=30)
    if not content:
        print(f"Failed to fetch Panchekha reference: {status}")
        return ""

    os.makedirs(os.path.dirname(_REFERENCE_CACHE_PATH), exist_ok=True)
    with open(_REFERENCE_CACHE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    return content


def _get_font_size(shape_dict: Dict[str, Any]) -> float:
    """Extract font size in pt from a shape's text style.

    Args:
        shape_dict (dict): Shape object containing 'text'.

    Returns:
        float: Font size in points, or 0 if not found.
    """
    style = get_text_style_from_shape(shape_dict)
    if style and style.get('fontSize'):
        return style['fontSize'].get('magnitude', 0)
    return 0


def find_header_box(text_boxes: List[Dict], slide_height: float, expected_text: str = None) -> Optional[Dict]:
    """Find the header text box using a tiered strategy.

    Strategy:
        1. Fuzzy match (>90) on expected_text. If exactly one match, use it.
        2. If no match or multiple matches, fall back to the text box with the
           largest font size across the entire slide.

    This guarantees a candidate is returned even if the header has wrong text
    or is in the wrong position.

    Args:
        text_boxes (list): Text boxes from extract_text_boxes_from_slide().
        slide_height (float): Slide height in EMUs.
        expected_text (str): Expected header text for fuzzy matching.

    Returns:
        dict: Header text box dict, or None.
    """
    if not text_boxes:
        return None

    # Tier 1: Fuzzy match on expected text
    if expected_text:
        fuzzy_matches = []
        for tb in text_boxes:
            score = fuzz.ratio(expected_text.lower(), tb['text'].strip().lower())
            if score > 90:
                fuzzy_matches.append((score, tb))

        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0][1]

    # Tier 2: Largest font size across entire slide
    best_font = 0
    header = None
    for tb in text_boxes:
        fs = _get_font_size(tb['element']['shape'])
        if fs > best_font:
            best_font = fs
            header = tb
    return header


def find_subheader_box(text_boxes: List[Dict], header_box: Optional[Dict]) -> Optional[Dict]:
    """Find the subheader text box: the first text box immediately below the header.

    Does not rely on font size — simply takes the closest text box below the
    header's y-position. If it isn't the subheader, the eval steps will fail
    on their own checks.

    Args:
        text_boxes (list): Text boxes from extract_text_boxes_from_slide().
        header_box (dict): Header text box from find_header_box().

    Returns:
        dict: Subheader text box dict, or None.
    """
    if not header_box:
        return None
    header_y = header_box['bbox']['y']
    candidates = []
    for tb in text_boxes:
        if tb is header_box:
            continue
        if tb['bbox']['y'] > header_y:
            candidates.append((tb['bbox']['y'], tb))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return None


def find_body_box(text_boxes: List[Dict], slide_width: float, slide_height: float) -> Optional[Dict]:
    """Find the central body text box: the text box most contained within the
    central zone that also has substantial text content.

    Uses `is_bbox_mostly_inside` to filter to central-zone candidates, then
    picks the one with the highest overlap ratio (ties broken by text length).

    Args:
        text_boxes (list): Text boxes from extract_text_boxes_from_slide().
        slide_width (float): Slide width in EMUs.
        slide_height (float): Slide height in EMUs.

    Returns:
        dict: Body text box dict, or None.
    """
    candidates = find_body_box_candidates(text_boxes, slide_width, slide_height)
    return candidates[0] if candidates else None


def find_body_box_candidates(text_boxes: List[Dict], slide_width: float,
                             slide_height: float) -> List[Dict]:
    """Rank the central-zone text boxes that could serve as the poster body.

    Applies the same central-zone filter as `find_body_box` but returns every
    qualifying candidate ordered best-first (highest central-zone overlap, with
    text length as a tiebreaker) instead of only the top one.

    Args:
        text_boxes (list): Text boxes from extract_text_boxes_from_slide().
        slide_width (float): Slide width in EMUs.
        slide_height (float): Slide height in EMUs.

    Returns:
        list: Candidate text box dicts, best-first. Empty if none qualify.
    """
    # Central zone: excludes top header strip, bottom footer strip, right sidebar
    central_zone = {
        'x': slide_width * 0.05,
        'y': slide_height * 0.20,
        'width': slide_width * 0.60,
        'height': slide_height * 0.65,
    }
    scored = []
    for tb in text_boxes:
        text_len = len(tb['text'].strip())
        if text_len < 50:
            continue
        if not is_bbox_mostly_inside(tb['bbox'], central_zone, threshold=0.6):
            continue
        # Score: overlap ratio, text length as tiebreaker
        overlap = bbox_overlap_ratio(tb['bbox'], central_zone)
        scored.append((overlap + (text_len / 100000.0), tb))
    # Stable sort keeps slide order for exact score ties, matching find_body_box
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [tb for _, tb in scored]


def is_topic_summary(model, text: str) -> Tuple[Optional[bool], str]:
    """Ask an LLM whether text is a background summary rather than bare logistics.

    Deliberately topic-agnostic: it only decides whether the text describes the
    event's subject matter at all. Whether that description covers the *correct*
    topic is judged by the downstream content-quality steps.

    Args:
        model (callable): Loaded model from load_model().
        text (str): Candidate body text.

    Returns:
        tuple: (verdict, detail) where verdict is True/False, or None when the
            model gave no usable response, and detail is a short reason string.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text":
            "You evaluate event poster content. Answer only Yes or No, then give "
            "a one-sentence reason."}]},
        {"role": "user", "content": [{"type": "text", "text":
            "Below is the text of one text box from the central body of an event "
            "poster. Is this text a background summary that describes or explains "
            "the event's subject matter?\n\n"
            "Answer No if it contains only logistical or boilerplate information — "
            "such as the date, time, location, host, contact details, registration "
            "instructions, the event title, or the speaker's name and title — "
            "without any substantive description of the topic itself.\n\n"
            f"Text:\n{text[:2000]}"}]}
    ]
    response = model(messages)
    if not response or not response.strip():
        return None, "no response"
    cleaned = response.strip()
    # models.py returns an "Error: ..." sentinel string instead of raising when
    # the backend is unreachable; that is a failed check, not a "No" verdict.
    if cleaned.startswith("Error:"):
        return None, cleaned[:150]
    return cleaned.lower().startswith("yes"), cleaned[:150]


def select_summary_body_box(text_boxes: List[Dict], slide_width: float, slide_height: float,
                            model, max_candidates: int = 4) -> Tuple[Optional[Dict], Optional[bool], str]:
    """Pick the central body text box that actually holds a background summary.

    Ranks central-zone candidates structurally, then LLM-gates them in parallel
    and returns the best-ranked candidate the gate accepts as a summary. When no
    candidate is accepted, the top-ranked candidate is still returned (so the
    downstream content-quality steps have text to judge) alongside a False
    verdict.

    Args:
        text_boxes (list): Text boxes from extract_text_boxes_from_slide().
        slide_width (float): Slide width in EMUs.
        slide_height (float): Slide height in EMUs.
        model (callable): Loaded model from load_model().
        max_candidates (int): Cap on candidates sent to the LLM gate.

    Returns:
        tuple: (body_box, is_summary, detail). `body_box` is None when no
            central-zone candidate exists. `is_summary` is True/False, or None
            when the gate produced no usable response for any candidate.
    """
    candidates = find_body_box_candidates(text_boxes, slide_width, slide_height)
    if not candidates:
        return None, False, "No substantial text box found in central area"

    candidates = candidates[:max_candidates]
    tasks = [
        {'id': f"cand_{i}", 'func': is_topic_summary, 'args': (model, tb['text'].strip())}
        for i, tb in enumerate(candidates)
    ]
    results = parallel_execute(tasks, max_workers=min(len(tasks), 4))

    first_detail = None
    for i, tb in enumerate(candidates):
        outcome = results.get(f"cand_{i}")
        if not outcome:
            continue
        verdict, detail = outcome
        if verdict is None:
            continue
        if first_detail is None:
            first_detail = detail
        if verdict:
            text_len = len(tb['text'].strip())
            return tb, True, (f"Body box found ({text_len} chars, candidate {i + 1} of "
                              f"{len(candidates)}), LLM: {detail}")

    if first_detail is None:
        return candidates[0], None, (f"Summary check unavailable for all "
                                     f"{len(candidates)} central candidate(s): no LLM response")

    return candidates[0], False, (f"No central text box contains a topic summary "
                                  f"({len(candidates)} candidate(s) checked), LLM: {first_detail}")
