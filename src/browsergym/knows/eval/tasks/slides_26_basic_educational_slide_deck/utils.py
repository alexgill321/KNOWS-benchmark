"""
Shared utilities for slides_26_basic_educational_slide_deck evaluator.
"""

import colorsys
import concurrent.futures
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from src.browsergym.knows.eval.eval_utils.scoring import StepCategory
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    get_element_bbox,
    get_text_style_from_shape,
    is_text_big,
    resolve_theme_color,
)
from src.browsergym.knows.eval.eval_utils.llm_utils import evaluate_with_llm
from src.browsergym.knows.eval.eval_utils.image_utils import (
    match_image_tiered,
    binary_compare_images,
    perceptual_hash_match,
)
from src.browsergym.knows.eval.eval_utils.web_utils import download_image_from_url, download_page_images
from src.browsergym.knows.eval.eval_utils.models import load_model


def _browser_headers(url: str) -> Dict[str, str]:
    """Chrome UA + same-origin Referer to defeat hotlink protection on image hosts."""
    parsed = urlparse(url)
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{parsed.scheme}://{parsed.netloc}/",
    }


# ---- Constants ----

WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp', '.ico')

CREDIT_KEYWORDS = (
    "source", "credit", "image", "photo", "courtesy",
    # Bare "http" excluded — body bullets like "Visit https://..." would false-match.
    # URL-shaped credits are still caught via the TLD/www hints below.
    "www", ".com", ".org", ".net", ".edu",
    "unsplash", "pexels", "pixabay", "wikimedia", "commons", "flickr", "getty",
)


# ---- Functions ----

def parse_task_details(task_text: str) -> Dict[str, object]:
    """Parse topic, presenter, and section count from task.md text.

    Accepts straight or curly single quotes around the topic.
    """
    topic_match = re.search(r"on\s+['‘’]([^'‘’]+)['‘’]", task_text)
    topic = topic_match.group(1) if topic_match else None

    name_match = re.search(r"my name\s*\(([^)]+)\)", task_text)
    presenter = name_match.group(1) if name_match else None

    sections_match = re.search(r"(\w+)\s+student-friendly sections", task_text)
    num_slides = None
    if sections_match:
        word = sections_match.group(1).lower()
        num_slides = WORD_TO_NUM.get(word) or (int(word) if word.isdigit() else None)

    return {
        "topic": topic,
        "presenter": presenter,
        "num_content_slides": num_slides,
    }


def parse_task_md(task_dir: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse task.md. Returns `(details, None)` on success or `(None, error)`
    on any failure — never raises, so a malformed file surfaces as per-CP
    failures instead of bricking module load.
    """
    try:
        with open(os.path.join(task_dir, "task.md"), encoding="utf-8") as f:
            details = parse_task_details(f.read())
        if any(details.get(k) is None for k in ("topic", "presenter", "num_content_slides")):
            return None, (
                f"task.md unparseable in {task_dir} — got {details}. "
                "Expected: topic in single quotes after 'on ', presenter in parens after "
                "'my name', section count as a word/digit before 'student-friendly sections'."
            )
        return details, None
    except Exception as e:
        return None, f"Error reading task.md: {e}"


def llm_extract_source_url(slide_text: str, model) -> Optional[str]:
    """Ask the LLM to extract a source URL from a slide's text.

    Handles citations the regex extractor misses — bare-domain references
    like `kids.kiddle.co/page` (no protocol), URLs split across lines, or
    natural-language citations ("Source: BBC News (bbc.com/article)").

    Returns:
        URL string with `https://` prefix, or None when no URL is identifiable
        or the LLM response can't be parsed as a URL.
    """
    if not slide_text or not slide_text.strip():
        return None

    system_prompt = (
        "You extract a single source URL from slide text. The URL may appear "
        "without a protocol (e.g. 'kids.kiddle.co/page'). Output rules:\n"
        "- Return ONLY the URL, with no surrounding text, quotes, or punctuation.\n"
        "- If the URL has no protocol, prepend 'https://'.\n"
        "- PRESERVE THE ORIGINAL CASING EXACTLY. Do NOT lowercase any part of "
        "the URL. Path segments are case-sensitive on many sites (e.g. kiddle.co), "
        "so `Great_Pyramid_of_Giza` and `great_pyramid_of_giza` are different pages.\n"
        "- If multiple URLs are present, return the one cited as the source/reference.\n"
        "- If no URL or domain reference is present, return exactly 'NONE'."
    )
    prompt = f"Slide text:\n{slide_text[:4000]}"
    try:
        raw = evaluate_with_llm(prompt, model, return_type="str", system_prompt=system_prompt)
    except Exception:
        return None
    if not raw:
        return None
    candidate = raw.strip().strip('"').strip("'").strip('.')
    if not candidate or candidate.lower() == 'none':
        return None
    if not candidate.lower().startswith(('http://', 'https://')):
        candidate = 'https://' + candidate.lstrip('/')
    # Sanity check: must look like a domain (has a dot in the host).
    try:
        scheme, host_path = candidate.split('://', 1)
        host = host_path.split('/', 1)[0]
    except (ValueError, IndexError):
        return None
    if '.' not in host or ' ' in host:
        return None

    # Restore original casing from the slide text — models often lowercase paths
    # despite the prompt, and case-sensitive hosts (kiddle.co, GitHub) then 404.
    idx = slide_text.lower().find(host_path.lower())
    if idx >= 0:
        candidate = f"{scheme}://{slide_text[idx:idx + len(host_path)]}"
    return candidate


def filter_non_image_links(links: List[str]) -> List[str]:
    """Return URLs whose path doesn't end in a common image extension."""
    return [
        link for link in links
        if not any(link.lower().split('?')[0].endswith(ext) for ext in IMAGE_EXTENSIONS)
    ]


def find_small_font_credit(
    text_boxes: List[Dict],
    slide_h: float,
    keywords: Optional[List[str]] = None,
    y_min: Optional[float] = None,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Tuple[bool, Optional[Dict]]:
    """Find a small-font (<18pt) credit text box (positions in EMU).
    Returns (found, text_box). Defaults: y_min=slide_h*0.65, no x-bound.
    """
    if keywords is None:
        keywords = CREDIT_KEYWORDS
    if y_min is None:
        y_min = slide_h * 0.65

    for tb in text_boxes:
        text = tb['text'].lower().strip()
        if not text:
            continue
        style = get_text_style_from_shape(tb['element'].get('shape', {}))
        font_size = style.get('fontSize')
        if not font_size or is_text_big(style, min_pt=18):
            continue
        bbox = tb['bbox']
        if bbox.get('y', 0) < y_min:
            continue
        if x_min is not None and bbox.get('x', 0) < x_min:
            continue
        if x_max is not None and bbox.get('x', 0) > x_max:
            continue
        if any(kw in text for kw in keywords):
            return True, tb

    return False, None


def match_source_image(
    links: List[str],
    slide_img_path: str,
    temp_dir: str,
    slide_idx: int,
    model,
) -> Tuple[bool, str]:
    """Check if any link points to (or contains) an image matching the slide image.

    For each candidate URL: try direct GET → Wayback → page-image extraction.
    Each candidate is tested via match_image_tiered (exact + pHash) plus a VLM
    "replacement" comparison. Returns `(matched, details)`.
    """
    if not links or not slide_img_path or not os.path.exists(slide_img_path):
        return False, "No image or links available for comparison"

    for link in links:
        headers = _browser_headers(link)
        source_path = download_image_from_url(link, temp_dir, headers=headers, wayback_fallback=True)
        if not source_path:
            # Some hosts (e.g. upload.wikimedia.org) 403 same-origin Referers; retry without.
            no_ref_headers = {k: v for k, v in headers.items() if k != 'Referer'}
            source_path = download_image_from_url(link, temp_dir, headers=no_ref_headers, wayback_fallback=False)
        if source_path:
            matched, method = match_image_tiered(source_path, slide_img_path, model=None)
            if matched:
                return True, f"Source image matches ({method}): {link}"
            if binary_compare_images(model, source_path, slide_img_path, mode="replacement"):
                return True, f"VLM accepts as plausible source: {link}"
            continue

        # Not a direct image — fetch the page and test embedded images.
        page_dir = os.path.join(temp_dir, f"page_{slide_idx}_{abs(hash(link))}")
        try:
            page_images = download_page_images(link, page_dir, headers=headers)
        except Exception:
            continue
        for fname in page_images:
            img_path = os.path.join(page_dir, fname)
            matched, method = match_image_tiered(img_path, slide_img_path, model=None)
            if matched:
                return True, f"Page: Source image matches ({method}): {link}"
            if binary_compare_images(model, img_path, slide_img_path, mode="replacement"):
                return True, f"Page: VLM accepts as plausible source: {link}"

    return False, f"Found {len(links)} URL(s) but none point to the slide image"


EMU_PER_INCH = 914400  # Google Slides API unit; used to report overflow in inches


def to_deck_positions(content_slide_indices):
    """Convert 0-based content-slide indices to 1-based deck positions (+2: title is slide 1).
    Non-int entries (e.g. unparsed task ids) pass through unchanged.
    """
    return [s + 2 if isinstance(s, int) else s for s in content_slide_indices]


def cluster_images_by_phash(image_paths: List[str], threshold: int = 10) -> List[int]:
    """Cluster by pHash similarity; return one cluster id per input. Pairs within
    `threshold` are unioned; hashing errors leave the image as its own singleton.
    """
    n = len(image_paths)
    parents = list(range(n))

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for a in range(n):
        for b in range(a + 1, n):
            if find(a) == find(b):
                continue
            try:
                if perceptual_hash_match(image_paths[a], image_paths[b], threshold=threshold):
                    parents[find(a)] = find(b)
            except Exception:
                continue
    return [find(i) for i in range(n)]


def count_unique_images(image_paths: List[str], threshold: int = 10) -> int:
    """Number of distinct perceptual-hash clusters across `image_paths`."""
    return len(set(cluster_images_by_phash(image_paths, threshold=threshold)))


# ---- Model state + lifecycle ----

_model_cache = None
_model_load_failed = False


def ensure_model(model_id: str):
    """Lazily load the LLM. Returns the model on success, None on failure."""
    global _model_cache, _model_load_failed
    if _model_cache is not None:
        return _model_cache
    if _model_load_failed:
        return None
    try:
        t0 = time.time()
        _model_cache = load_model(model_id)
        print(f"Loaded model '{model_id}' in {time.time() - t0:.2f}s")
        return _model_cache
    except Exception as e:
        print(f"Failed to load model '{model_id}': {e}")
        _model_load_failed = True
        return None


def set_cached_model(loaded_model):
    """Pre-populate the cached model (used by grade_checkpoints' cached_models param)."""
    global _model_cache
    _model_cache = loaded_model


def reset_model_state():
    """Clear cached model + failure flag (call between docs in batched runs)."""
    global _model_cache, _model_load_failed
    _model_cache = None
    _model_load_failed = False


# ---- Pure text/image/color helpers ----

def normalize_for_match(text: str) -> str:
    """Lower-case, collapse whitespace, fold curly quotes to straight (D1)."""
    if not text:
        return ""
    text = (text.replace('‘', "'").replace('’', "'")
                .replace('“', '"').replace('”', '"'))
    return " ".join(text.split()).lower()


_FLEX_SEARCH_HAYSTACK_CAP = 200_000  # chars; regex slow path scales O(n*m) so cap it


def find_with_flexible_whitespace(haystack: str, needle: str) -> int:
    """Case-insensitive `find` that tolerates whitespace differences.

    Returns offset of match (0 if needle empty, -1 if no match). Used to center an LLM
    excerpt on a fuzzy-match position when source HTML re-flowed whitespace.
    """
    if not needle:
        return 0
    h_lower = haystack.lower()
    n_lower = needle.lower()
    idx = h_lower.find(n_lower)
    if idx >= 0:
        return idx
    # Slow-path regex is O(n*m) worst-case — cap haystack to bound it. Callers
    # default the excerpt offset to 0 when this returns -1.
    if len(h_lower) > _FLEX_SEARCH_HAYSTACK_CAP:
        return -1
    # Allow any whitespace run between needle's tokens.
    tokens = n_lower.split()
    if not tokens:
        return -1
    pattern = r'\s+'.join(re.escape(t) for t in tokens)
    m = re.search(pattern, h_lower)
    return m.start() if m else -1


def contains_normalized(haystack: str, needle: str) -> bool:
    """Substring containment using normalize_for_match on both sides."""
    return normalize_for_match(needle) in normalize_for_match(haystack)


def img_area(meta: Dict) -> float:
    """Bbox-area for a Slides image element; used to pick the largest image (D4)."""
    bb = get_element_bbox({'transform': meta.get('transform', {}),
                           'size': meta.get('size', {})})
    return bb.get('width', 0) * bb.get('height', 0)


def font_pt(tb: Dict) -> float:
    """Font size in PT for ranking text boxes; 0 when fontSize is unset."""
    style = get_text_style_from_shape(tb['element'].get('shape', {}))
    fs = style.get('fontSize') or {}
    mag = fs.get('magnitude', 0)
    return mag / 12700 if fs.get('unit') == 'EMU' else mag


def placeholder_type(tb: Dict) -> str:
    """Placeholder type (TITLE/CENTERED_TITLE/SUBTITLE/BODY/...) for a text box; '' if absent."""
    return tb.get('element', {}).get('shape', {}).get('placeholder', {}).get('type', '')


def bbox_center_x(bbox: Dict) -> Optional[float]:
    """Horizontal center of a bbox in EMU; None for empty/None (forces explicit handling)."""
    if not bbox:
        return None
    return bbox.get('x', 0) + bbox.get('width', 0) / 2


def get_master_placeholder_font_pt(presentation_data: Dict, ph_type: str) -> Optional[float]:
    """First explicit fontSize (PT) for a `ph_type` placeholder in layouts then masters.
    None when no explicit size is found. Used by CP1 step 1 to validate inheritance.
    """
    if not presentation_data:
        return None
    for source in (presentation_data.get('layouts', []), presentation_data.get('masters', [])):
        for page in source:
            for elem in page.get('pageElements', []):
                shape = elem.get('shape', {})
                if shape.get('placeholder', {}).get('type') != ph_type:
                    continue
                for te in shape.get('text', {}).get('textElements', []):
                    fs = te.get('textRun', {}).get('style', {}).get('fontSize') if 'textRun' in te else None
                    if fs and 'magnitude' in fs:
                        mag = fs['magnitude']
                        return mag / 12700 if fs.get('unit') == 'EMU' else mag
    return None


def fill_failure_steps(checkpoint, step_specs: List[Tuple[str, int]], reason: str, only_missing: bool = False,
                       category: str = StepCategory.EXECUTION_ERROR):
    """Add zero-score steps to checkpoint for each (name, max_score) in step_specs.

    If only_missing=True, skip step_ids already present in checkpoint.steps.
    All current call sites are guards/crash fillers, so the category defaults
    to EXECUTION_ERROR; pass `category` to override.
    """
    existing = {s.step_id for s in checkpoint.steps} if only_missing else set()
    for step_id, (name, pts) in enumerate(step_specs, start=1):
        if step_id not in existing:
            checkpoint.add_step(name, False, step_id, reason, max_score=pts, execution_time=0, category=category)


def call_with_timeout(func: Callable, timeout_s: float, label: str, *args, **kwargs) -> Tuple[Any, bool]:
    """Run func with a wall-clock timeout. Returns (result, ok). On timeout/error, returns (None, False)."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(func, *args, **kwargs).result(timeout=timeout_s), True
    except concurrent.futures.TimeoutError:
        print(f"Timed out {label} after {timeout_s}s")
    except Exception as e:
        print(f"Failed {label}: {e}")
    return None, False


def join_extras(*items) -> str:
    """Join non-empty items into ' (a, b, c)' suffix, or '' when none are truthy."""
    parts = [x for x in items if x]
    return f" ({', '.join(parts)})" if parts else ""


def has_min_slides(presentation_data, min_count: int) -> bool:
    """True if presentation_data is loaded and has at least min_count slides."""
    if not presentation_data or 'slides' not in presentation_data:
        return False
    return len(presentation_data['slides']) >= min_count


def score_credit_match(tb: Dict, slide_w: float, slide_h: float, keywords) -> int:
    """Count how many of (small font, lower 40%, left half, credit keyword) a text box matches.

    Returns 0-4. Used by CP2 step 3 for partial-credit citation scoring.
    """
    text = (tb.get('text') or '').lower().strip()
    if not text:
        return 0
    style = get_text_style_from_shape(tb['element'].get('shape', {}))
    font_size = style.get('fontSize')
    is_small = font_size is not None and not is_text_big(style, min_pt=18)
    bbox = tb.get('bbox', {}) or {}
    y_low = bbox.get('y', 0) >= slide_h * 0.60
    # Center-x check (vs. left edge) — a wide centered box wouldn't count as "left half".
    x_center = bbox_center_x(bbox)
    x_left = x_center is not None and x_center <= slide_w * 0.5
    has_keyword = any(kw in text for kw in keywords)
    return sum([is_small, y_low, x_left, has_keyword])


def is_dark_orange(text_style, hue_lo=0.04, hue_hi=0.13, min_sat=0.6, min_val=0.4, rgb_tolerance=0.20,
                   presentation_data=None):
    """Dark-orange detection: HSV-primary + relaxed RGB-box fallback. Resolves
    `foregroundThemeColor` via `presentation_data` when not pre-resolved upstream.
    """
    fg = text_style.get('foregroundColor') if text_style else None
    if not fg and text_style:
        theme_name = text_style.get('foregroundThemeColor')
        fg = resolve_theme_color(theme_name, presentation_data) if theme_name else None
    if not fg:
        return False
    r = fg.get('red', 0)
    g = fg.get('green', 0)
    b = fg.get('blue', 0)
    if max(r, g, b) == 0:
        return False
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if hue_lo <= h <= hue_hi and s >= min_sat and v >= min_val:
        return True
    return (abs(r - 1.0) < rgb_tolerance and abs(g - 0.549) < rgb_tolerance
            and abs(b - 0.0) < rgb_tolerance)
