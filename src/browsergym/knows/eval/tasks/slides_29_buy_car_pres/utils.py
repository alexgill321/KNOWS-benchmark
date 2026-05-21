"""Helpers for slides_29 task evaluation: LLM extraction, image coverage,
URL parsing, and per-car CP3 step evaluation."""
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.llm_utils import (
    extract_json_with_llm as extract_info_with_llm,
    evaluate_with_llm,
)
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    estimate_text_render_bbox,
    extract_slide_links,
    extract_slide_text,
    extract_text_boxes_from_slide,
    extract_title_text,
    get_element_bbox,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_match_robust
from src.browsergym.knows.eval.eval_utils.utils import bbox_overlap_ratio
from src.browsergym.knows.eval.eval_utils.parallel_utils import parallel_download
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_text_content


def make_failure_checkpoint(name: str, total: int, step_names: List[str], reason: str) -> Checkpoint:
    """Structurally-complete failed Checkpoint — preserves report shape on upstream error."""
    cp = Checkpoint(total=total, result=0, name=name)
    for i, step_name in enumerate(step_names, 1):
        cp.add_step(step_name, False, i, reason, execution_time=0)
    return cp


_TASK_CATEGORY_RE = re.compile(r'categorized as an?\s+(.+?)[.,]', re.IGNORECASE)
_TASK_YEAR_RE = re.compile(r'\b(\d{4})\s+models\b', re.IGNORECASE)
_TASK_TITLE_RE = re.compile(r'title slide called\s+"([^"]+)"', re.IGNORECASE)


def parse_task_config(task_md_path: str) -> Dict[str, Any]:
    """Extract per-instance {category, year, title} from task.md.

    Source-of-truth is the agent's prompt — keeps evaluators task-agnostic.
    Raises ValueError on missing fields so malformed task.md fails loudly.
    """
    if not os.path.exists(task_md_path):
        raise FileNotFoundError(f"task.md not found at {task_md_path}")
    with open(task_md_path, "r", encoding="utf-8") as f:
        text = f.read()
    cat_m = _TASK_CATEGORY_RE.search(text)
    year_m = _TASK_YEAR_RE.search(text)
    title_m = _TASK_TITLE_RE.search(text)
    missing = [name for name, m in (("category", cat_m), ("year", year_m), ("title", title_m)) if not m]
    if missing:
        raise ValueError(f"task.md at {task_md_path} missing fields: {missing}")
    return {
        "category": cat_m.group(1).strip(),
        "year": int(year_m.group(1)),
        "title": title_m.group(1).strip(),
    }


# Per-CP step shapes used by make_failure_checkpoint() and evaluate_single_car().
CP3_PER_CAR_STEPS = [
    "KBB Visit in History",
    "Make and Model Listed as Title",
    "Correct Model Picture",
    "Picture >= 50% of Slide",
    "Picture Does Not Overlap Text",
    "Sticker Price Matches KBB",
    "Fuel Efficiency Matches KBB",
    "Horsepower Matches KBB",
    "Review URL Provided",
    "Review URL in History",
    "Rating Matches Review Platform",
]
CP4_STEP_NAMES = [
    "Title Denotes Best Car Stats",
    "Lowest Price Car",
    "Highest MPG Car",
    "Highest Horsepower Car",
    "Most Highly Rated Car",
]
CP_STEP_SHAPES = [
    ("Title Slide", 2, ["Title Match", "Title Font Size at least 30pt"]),
    ("Car Content Slides", 6, ["Article Visit", "At Least 5 Car Slides"]),
    ("Car Slides Validation", 55,
     [f"Car {c+1} - {n}" for c in range(5) for n in CP3_PER_CAR_STEPS]),
    ("Summary Slide", 5, CP4_STEP_NAMES),
]

# Maps CP4 internal source_key to JSON key used in the winner-extraction prompt.
CP4_WINNER_KEY_MAP = {
    'price': 'lowest_price',
    'mpg': 'highest_mpg',
    'hp': 'highest_horsepower',
    'rating': 'highest_rating',
}


def evaluate_slide_for_cars(
    slide_idx: int, slide: Any, gold_cars_list: List[str], model: Any,
    category: str = "vehicle", year: Optional[int] = None,
) -> "tuple[Optional[str], Optional[str]]":
    """Classify a slide against a {year} {category} target.

    Returns (car_name_or_None, error_str_or_None). When gold_cars_list is
    given, constrain to that list; otherwise free-form LLM classification.
    """
    try:
        title_text = extract_title_text(slide)
        if not title_text.strip():
            return (None, None)

        year_str = f"{year} " if year else ""
        target = f"{year_str}{category}".strip()

        if gold_cars_list:
            gold_csv = ", ".join(gold_cars_list)
            task_text = f"""Given a slide title and a gold list of {target}s from a source article, decide whether the slide is about one of those {category}s.
Gold list: {gold_csv}
Slide title: {title_text}

Respond ONLY with JSON:
{{
    "is_match": <true if slide is about a {target} from the gold list, false otherwise>,
    "name": "<exact name from the gold list when is_match is true, otherwise empty string>"
}}"""
        else:
            task_text = f"""Decide whether the following slide title denotes a specific {target} model.
Slide title: {title_text}

Respond ONLY with JSON:
{{
    "is_match": <true if slide is about a {target}, false otherwise>,
    "name": "<make and model of the {category} when is_match is true, otherwise empty string>"
}}"""

        result = extract_info_with_llm(task_text, model)
        if result is None:
            return (None, "LLM extraction returned no result")
        if isinstance(result, dict) and result.get("is_match"):
            name = str(result.get("name") or "").strip()
            if name:
                return (name, None)
    except Exception as e:
        print(f"Warning: slide {slide_idx} evaluation error: {e}")
        return (None, f"slide evaluation errored: {e}")
    return (None, None)


REVIEW_DOMAINS = (
    "edmunds.com", "cars.com", "cargurus.com", "jdpower.com",
    "consumerreports.org", "caranddriver.com", "motortrend.com",
    "truecar.com", "autotrader.com", "carcomplaints.com",
    "usnews.com", "cars.usnews.com", "carbuzz.com", "carfax.com", "carmax.com",
)

_PLAIN_URL_RE = re.compile(r'https?://[^\s<>"\']+')
_PROTOCOL_LESS_URL_RE = re.compile(
    r'\b(?:www\.[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s<>"\']*)?)',
    re.IGNORECASE,
)
_URL_TRAILING_PUNCT = '.,;:)]}>'


def extract_all_slide_urls(slide: Any) -> List[str]:
    """Hyperlinks + plain-text URLs (incl. protocol-less www.* forms), deduped."""
    urls: List[str] = []
    seen: set = set()

    def _add(u: str) -> None:
        cleaned = u.rstrip(_URL_TRAILING_PUNCT) if u else ""
        if not cleaned:
            return
        canonical = cleaned[:-1] if cleaned.endswith("/") and cleaned.count("/") > 2 else cleaned.rstrip("/")
        if canonical in seen:
            return
        urls.append(cleaned)
        seen.add(canonical)

    try:
        for u in (extract_slide_links(slide) or []):
            if u:
                _add(u)
    except Exception as e:
        print(f"Warning: hyperlink extraction error: {e}")
    try:
        text = extract_slide_text(slide, " ") or ""
        for m in _PLAIN_URL_RE.findall(text):
            _add(m)
        for m in _PROTOCOL_LESS_URL_RE.findall(text):
            _add(f"https://{m}")
    except Exception as e:
        print(f"Warning: text URL scan error: {e}")
    return urls


def pick_review_url(slide_links: List[str]) -> Optional[str]:
    """Pick the most likely review URL: known review domain > non-KBB > first."""
    if not slide_links:
        return None
    for url in slide_links:
        url_lower = url.lower()
        if any(d in url_lower for d in REVIEW_DOMAINS):
            return url
    for url in slide_links:
        if "kbb.com" not in url.lower():
            return url
    return slide_links[0]


def normalize_json_key(s: Any) -> str:
    """Normalize a JSON key to lowercase alphanumeric for case-insensitive lookups."""
    return ''.join(c.lower() for c in str(s) if c.isalnum())


def is_truthy_flag(value: Any) -> bool:
    """Robust JSON-ish bool — guards against `bool("false") is True`."""
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return value != 0 and value is not False
    return False


_EXPECTED_CAR_STOPWORDS = frozenset({"a", "an", "the", "and", "or", "of", "i"})


def expected_car_in_text(text: str, expected_car: str, category: str = "") -> bool:
    """All required tokens of expected_car present in text (case-insensitive).

    Years (1900-2099), stopwords, and task-category words are optional — the
    category descriptor (e.g. "coupe", "sports car") is shared by every car in
    the deck and doesn't distinguish one car from another.
    """
    if not text or not expected_car:
        return False
    text_lower = str(text).lower()
    optional = set(_EXPECTED_CAR_STOPWORDS)
    optional.update(re.findall(r'\w+', str(category).lower()))
    raw_tokens = re.findall(r'\w+', str(expected_car).lower())
    required_tokens = []
    for t in raw_tokens:
        if len(t) == 4 and t.isdigit() and 1900 <= int(t) <= 2099:
            continue
        if t in optional:
            continue
        required_tokens.append(t)
    if not required_tokens:
        return False
    return all(
        re.search(r'\b' + re.escape(t) + r'\b', text_lower) for t in required_tokens
    )


def find_kbb_url_for_car(browsing_history: List[str], make_model: str) -> Optional[str]:
    """Pick kbb.com URL with strongest token overlap; require at least one non-year match; prefer overview over consumer-reviews."""
    if not browsing_history or not make_model:
        return None

    tokens = [t for t in re.findall(r'\w+', make_model.lower()) if len(t) > 1]
    if not tokens:
        return None

    best_url = None
    best_score = 0.0
    for url in browsing_history:
        url_lower = url.lower()
        if 'kbb.com' not in url_lower:
            continue
        matched = [t for t in tokens if t in url_lower]
        non_year_matches = [t for t in matched if not (len(t) == 4 and t.isdigit() and 1900 <= int(t) <= 2099)]
        if not non_year_matches:
            continue
        score = float(sum(len(t) for t in matched))
        if 'consumer-reviews' in url_lower:
            score -= 0.5
        if score > best_score:
            best_score = score
            best_url = url
    return best_url


_STATS_KEYWORDS = ('$', 'mpg', 'hp', 'horsepower', 'rating', '/5', '/10', 'review', 'price')


def largest_image_slide_coverage(slide: Any, slide_width_emu: float, slide_height_emu: float) -> float:
    """Largest single image's on-slide coverage as a percentage (0-100).

    Each image bbox is clipped to the slide rectangle; overflow off the slide
    edges is not counted. Returns the max clipped coverage across all images.
    """
    total_slide_area = slide_width_emu * slide_height_emu
    if total_slide_area <= 0:
        return 0.0
    best = 0.0
    for element in slide.get('pageElements', []) or []:
        if 'image' not in element:
            continue
        bbox = get_element_bbox(element)
        x1 = max(0.0, bbox['x'])
        y1 = max(0.0, bbox['y'])
        x2 = min(slide_width_emu, bbox['x'] + bbox['width'])
        y2 = min(slide_height_emu, bbox['y'] + bbox['height'])
        if x2 > x1 and y2 > y1:
            coverage = (x2 - x1) * (y2 - y1) / total_slide_area * 100
            best = max(best, coverage)
    return min(best, 100.0)


def find_title_and_stats_text_boxes(slide: Any) -> "tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]":
    """Identify the title and stats/URL combined text boxes on a car slide.

    Title: first text box backed by a TITLE/CENTERED_TITLE/SUBTITLE placeholder,
    falling back to the box whose text matches extract_title_text(slide).
    Stats: largest remaining text box that contains stats keywords ($, mpg, hp,
    rating, /5, review, etc.). Falls back to largest non-title box if no
    keyword match. Either may be None if not found.
    """
    text_boxes = extract_text_boxes_from_slide(slide)
    if not text_boxes:
        return (None, None)

    title_box = None
    for tb in text_boxes:
        ph_type = tb.get('element', {}).get('shape', {}).get('placeholder', {}).get('type', '')
        if ph_type in ('TITLE', 'CENTERED_TITLE', 'SUBTITLE'):
            title_box = tb
            break
    if title_box is None:
        title_text = (extract_title_text(slide) or '').strip()
        if title_text:
            for tb in text_boxes:
                if tb.get('text', '').strip() == title_text:
                    title_box = tb
                    break

    candidates = [tb for tb in text_boxes if tb is not title_box]
    if not candidates:
        return (title_box, None)

    def _area(tb):
        b = tb.get('bbox') or {}
        return b.get('width', 0) * b.get('height', 0)

    keyword_matches = [
        tb for tb in candidates
        if any(kw in tb.get('text', '').lower() for kw in _STATS_KEYWORDS)
    ]
    pool = keyword_matches or candidates
    stats_box = max(pool, key=_area, default=None)
    return (title_box, stats_box)


def find_year_category_article(
    browsing_history: List[str],
    year: int,
    category: str,
    model: Any,
) -> Optional[str]:
    """Return a browsing-history URL the LLM accepts as a relevant {year} {category} article, else None."""
    if not browsing_history or model is None or not category:
        return None

    candidates = [u for u in browsing_history if u and 'kbb.com' not in u.lower()]
    if not candidates:
        return None

    fetch_tasks = [
        {'id': url, 'func': fetch_page_text_content, 'args': (url, 1_000_000)}
        for url in candidates
    ]
    fetched = parallel_download(fetch_tasks, max_workers=5, use_rate_limit=False)

    for url in candidates:
        page = fetched.get(url)
        content = page[0] if page else None
        if not content:
            continue

        prompt = f"""Decide whether this web page is an article that could serve as a source for a presentation about {year} {category} cars.

Accept if:
- The article discusses or lists multiple {category} models. The article may also cover other car types (SUVs, trucks, etc.) — that's fine as long as several {category}s are included.
- The models discussed are reasonably relevant to model year {year}. Lenient on year — guides from a few years earlier or later are fine when the models remained on sale.
- The page is a real article, not a single-vehicle product listing or homepage.

Reject single-car product/listing pages, homepages, unrelated topics, or articles that don't actually cover any {category}s.

URL: {url}
Page content:
{content}

Answer YES or NO."""
        try:
            verdict = evaluate_with_llm(prompt, model, return_type="bool")
        except Exception as e:
            print(f"Warning: LLM relevance check failed for {url}: {e}")
            continue
        if verdict:
            return url
    return None


def find_review_url_in_history(browsing_history: List[str], review_url: str) -> bool:
    """Check if a review URL (or its domain) appears in the browsing history."""
    if not browsing_history or not review_url:
        return False

    if review_url in browsing_history:
        return True

    try:
        parsed = urlparse(review_url)
        review_domain = parsed.netloc.lower()
        if review_domain:
            return any(review_domain in url.lower() for url in browsing_history)
    except Exception:
        pass

    return False


_NUMBER_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')


def _extract_first_number(s) -> Optional[float]:
    """First numeric value in s as float; commas stripped. None if none found."""
    if s is None:
        return None
    m = _NUMBER_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group().replace(',', ''))
    except ValueError:
        return None


def _extract_all_numbers(s) -> List[float]:
    """All numeric values in s as floats; commas stripped."""
    if s is None:
        return []
    out: List[float] = []
    for m in _NUMBER_RE.findall(str(s)):
        try:
            out.append(float(m.replace(',', '')))
        except ValueError:
            continue
    return out


def compare_price(slide_value, kbb_value) -> bool:
    """True if slide price is within $1000 of KBB MSRP."""
    try:
        kbb = float(kbb_value or 0)
    except (ValueError, TypeError):
        kbb = 0.0
    if kbb <= 0:
        return False
    slide_num = _extract_first_number(slide_value)
    if slide_num is None:
        return False
    return abs(slide_num - kbb) <= 1000


def compare_hp(slide_value, kbb_value) -> bool:
    """True if any HP number on the slide is within ±5 of the KBB value."""
    try:
        kbb = float(kbb_value or 0)
    except (ValueError, TypeError):
        kbb = 0.0
    if kbb <= 0:
        return False
    slide_numbers = _extract_all_numbers(slide_value)
    return any(abs(sn - kbb) <= 5 for sn in slide_numbers)


def compare_mpg(slide_value, mpg_combined, mpg_city, mpg_hwy) -> bool:
    """True if any number on the slide is within ±1 of any KBB MPG value (combined/city/hwy)."""
    kbb_values: List[float] = []
    for v in (mpg_combined, mpg_city, mpg_hwy):
        try:
            fv = float(v or 0)
        except (ValueError, TypeError):
            continue
        if fv > 0:
            kbb_values.append(fv)
    if not kbb_values:
        return False
    slide_numbers = _extract_all_numbers(slide_value)
    if not slide_numbers:
        return False
    return any(abs(sn - kv) <= 1 for sn in slide_numbers for kv in kbb_values)


def extract_kbb_stats(kbb_text: str, make_model: str, model: Any) -> Dict[str, Any]:
    """Extract numeric KBB stats for one car; retry per-field when the bulk pass returns 0."""
    bulk_task = f"""Extract numerical car stats from this KBB page text.

 The page may cover multiple trims/variants of the model. If "{make_model}" names a specific trim or variant, extract that trim's figures. If "{make_model}" names only a make and model with no trim qualifier, use the model's base/standard configuration — the starting (lowest) MSRP and base-trim specs — not a higher or optioned-up trim. If a figure is not broken out for the relevant trim, fall back to the model's generally-listed figure rather than returning 0; only return 0 when the figure is genuinely absent from the page.

 For fuel efficiency, use whichever metric the page reports for this trim: MPG for gas/diesel vehicles, or MPGe for electric/plug-in-hybrid vehicles. Return it as a plain number.

 Respond ONLY with JSON:
 {{
     "price_numeric": <original MSRP when new for this trim (NOT Fair Purchase Price or used market value) as number, 0 if not found>,
     "mpg_combined": <combined fuel efficiency (MPG or MPGe) as number, 0 if not found>,
     "mpg_city": <city fuel efficiency (MPG or MPGe) as number, 0 if not found>,
     "mpg_hwy": <highway fuel efficiency (MPG or MPGe) as number, 0 if not found>,
     "hp_numeric": <horsepower as number, 0 if not found>
 }}

 KBB page content:
 {kbb_text}"""
    stats = extract_info_with_llm(bulk_task, model)
    if not isinstance(stats, dict):
        stats = {}

    retry_prompts = {
        "price_numeric": f"""Find the original MSRP (price when new) for the {make_model} on this KBB page. It may be labeled "MSRP", "Original MSRP", "Sticker Price", or "Price When New" — do not return the Fair Purchase Price or used market value.

Respond ONLY with JSON: {{"price_numeric": <number, 0 only if truly absent>}}

KBB page content:
{kbb_text}""",
        "hp_numeric": f"""Find the horsepower for the {make_model} on this KBB page. It may be labeled "horsepower", "hp", or "bhp", possibly with an RPM figure, or listed in an engine/specs table.

Respond ONLY with JSON: {{"hp_numeric": <number, 0 only if truly absent>}}

KBB page content:
{kbb_text}""",
        "mpg_combined": f"""Find the combined fuel efficiency for the {make_model} on this KBB page. It may be labeled "combined MPG", "MPG", or "MPGe" for electric/hybrid vehicles, in a fuel-economy section.

Respond ONLY with JSON: {{"mpg_combined": <number, 0 only if truly absent>}}

KBB page content:
{kbb_text}""",
    }

    for field, prompt in retry_prompts.items():
        try:
            if float(stats.get(field, 0) or 0) > 0:
                continue
        except (ValueError, TypeError):
            pass
        retry = extract_info_with_llm(prompt, model)
        if isinstance(retry, dict) and retry.get(field):
            stats[field] = retry[field]
    return stats


def _bounded_value(v, lo, hi):
    """Return float(v) if it falls in [lo, hi], else 0."""
    try:
        fv = float(v or 0)
        return fv if lo <= fv <= hi else 0
    except (ValueError, TypeError):
        return 0


def resolve_stat_value(cp3_match, slide_v, source_v, lo, hi):
    """CP3-validated slide first, then source, then slide last-resort."""
    if cp3_match:
        v = _bounded_value(slide_v, lo, hi)
        if v:
            return v
    v = _bounded_value(source_v, lo, hi)
    if v:
        return v
    return _bounded_value(slide_v, lo, hi)


def resolve_price_value(cp3_match, slide_v, kbb_v, lo, hi):
    """Pick price using slide-vs-KBB ratio to detect KBB-used-value vs slide-typo."""
    sv = _bounded_value(slide_v, lo, hi)
    kv = _bounded_value(kbb_v, lo, hi)
    if not sv:
        return kv
    if not kv:
        return sv
    if cp3_match:
        return sv
    ratio = sv / kv
    if ratio >= 1.3:
        return sv
    if ratio <= 0.7:
        return kv
    return sv


def compute_winners(contributing_indices, car_stats, kbb_stats, review_stats, cp3_car_infos, cp3_stat_matches) -> Dict[str, Dict[str, Any]]:
    """Pick best-per-category car. Returns {lowest_price, highest_mpg, highest_hp, highest_rating} each as {name, value}."""
    lowest_price = {"name": "", "value": float('inf')}
    highest_mpg = {"name": "", "value": 0.0}
    highest_hp = {"name": "", "value": 0.0}
    highest_rating = {"name": "", "value": 0.0}

    for idx in contributing_indices:
        stats = car_stats.get(idx) or {}
        kbb_for_car = kbb_stats.get(idx) or {}
        review_for_car = review_stats.get(idx) or {}
        if not stats and not kbb_for_car and not review_for_car:
            continue

        name = (stats.get('make_model') or '').strip()
        if not name:
            cp3_info = cp3_car_infos.get(idx) or {}
            name = str(cp3_info.get('make_model', '') or '').strip()
        if not name:
            continue

        matches = cp3_stat_matches.get(idx, {})
        slide_price = stats.get('price_numeric', 0) or 0
        slide_mpg = stats.get('mpg_numeric', 0) or 0
        slide_hp = stats.get('hp_numeric', 0) or 0
        slide_rating = stats.get('rating_numeric', 0) or 0
        price = resolve_price_value(matches.get('price'), slide_price, kbb_for_car.get('price_numeric'), 3_000, 5_000_000)
        mpg = resolve_stat_value(matches.get('mpg'), slide_mpg, kbb_for_car.get('mpg_combined'), 3, 500)
        hp = resolve_stat_value(matches.get('hp'), slide_hp, kbb_for_car.get('hp_numeric'), 50, 3_000)

        # Rating: CP3-validated slide first, then review aggregate, then candidates median, then slide fallback.
        if matches.get('rating') and isinstance(slide_rating, (int, float)) and 0 < float(slide_rating) <= 10:
            rating_raw = float(slide_rating)
        elif is_truthy_flag(review_for_car.get('is_aggregate')) and review_for_car.get('rating_numeric'):
            rating_raw = review_for_car.get('rating_numeric')
        else:
            plausible_candidates: List[float] = []
            for c in (review_for_car.get('candidates') or []):
                try:
                    v = float(c)
                    if 0 < v <= 10:
                        plausible_candidates.append(v)
                except (ValueError, TypeError):
                    continue
            if plausible_candidates:
                plausible_candidates.sort()
                rating_raw = plausible_candidates[len(plausible_candidates) // 2]
            elif slide_rating:
                rating_raw = slide_rating
            else:
                rating_raw = 0
        rating = _bounded_value(rating_raw, 0, 10)

        if price and price < lowest_price["value"]:
            lowest_price = {"name": name, "value": price}
        if mpg and mpg > highest_mpg["value"]:
            highest_mpg = {"name": name, "value": mpg}
        if hp and hp > highest_hp["value"]:
            highest_hp = {"name": name, "value": hp}
        if rating and rating > highest_rating["value"]:
            highest_rating = {"name": name, "value": rating}

    return {
        "lowest_price": lowest_price,
        "highest_mpg": highest_mpg,
        "highest_hp": highest_hp,
        "highest_rating": highest_rating,
    }


def evaluate_single_car(car_idx, car_slides, step_names, car_infos, kbb_urls, review_urls, web_contents, browsing_history, slide_image_dirs, kbb_example_dirs, slide_width_emu, slide_height_emu, model, presentation=None, stat_matches_out=None, kbb_stats_in=None):
    """Evaluate all 10 steps for a single car. Returns list of step result dicts.

    If stat_matches_out is a dict, records per-stat boolean verdicts at
    stat_matches_out[car_idx] with keys 'price', 'mpg', 'hp', 'rating'.
    kbb_stats_in (optional) is the {car_idx: {price_numeric, mpg_combined, mpg_city, mpg_hwy, hp_numeric}}
    cache produced by CP3 phase 3b; used for deterministic stat comparison.
    """
    steps = []

    if car_idx >= len(car_slides):
        for name in step_names:
            steps.append({"name": f"Car {car_idx+1} - {name}", "success": False,
                        "detail": "Car slide not found", "execution_time": 0})
        return steps

    slide = car_slides[car_idx]
    car_info = car_infos.get(car_idx) or {}
    make_model = car_info.get('make_model', '')

    print(f"  Evaluating car {car_idx+1}: {make_model}")

    # Step 1: KBB visit in browsing history
    step_start = time.time()
    has_kbb_visit = car_idx in kbb_urls
    steps.append({"name": f"Car {car_idx+1} - KBB Visit in History", "success": has_kbb_visit,
                "detail": f"Found KBB visit: {kbb_urls.get(car_idx, 'N/A')}" if has_kbb_visit
                else "No KBB visit found for this car in browsing history",
                "execution_time": time.time() - step_start})

    # Step 2: Make and Model Listed as Title
    step_start = time.time()
    try:
        slide_title = extract_title_text(slide)
    except Exception as e:
        print(f"Warning: title extraction failed for car {car_idx}: {e}")
        slide_title = ""
    try:
        title_match = keywords_match_robust(slide_title, make_model, model=model)
    except Exception as e:
        print(f"Warning: title-match LLM failed for car {car_idx}: {e}")
        title_match = False

    make_model_in_title = bool(make_model.strip()) and bool(title_match)
    steps.append({"name": f"Car {car_idx+1} - Make and Model Listed as Title", "success": make_model_in_title,
                "detail": f"Make/model found: {make_model}" if make_model_in_title
                else "No make/model found on slide",
                "execution_time": time.time() - step_start})

    # Step 3: Picture of correct model
    step_start = time.time()
    correct_picture = False
    picture_detail = None
    temp_dir = slide_image_dirs.get(car_idx)
    example_dir = kbb_example_dirs.get(car_idx)
    has_picture = bool(temp_dir and os.path.exists(temp_dir) and os.listdir(temp_dir))

    if has_picture:
        try:
            if example_dir:
                matching = binary_judge_image(
                    model, temp_dir,
                    "Is this an image of the same car model as shown in the example images?",
                    examples=example_dir,
                )
            else:
                matching = binary_judge_image(
                    model, temp_dir,
                    f"Is this an image of a {make_model} vehicle?",
                )
            correct_picture = bool(matching)
        except Exception as e:
            print(f"Error checking car image for car {car_idx}: {e}")
            picture_detail = f"Could not evaluate picture (error: {e})"
    else:
        picture_detail = "No picture found on slide"

    if correct_picture:
        picture_detail = "Found picture of correct car model"
    elif picture_detail is None:
        picture_detail = f"Picture on slide does not match the expected model ({make_model or 'unknown'})"

    steps.append({"name": f"Car {car_idx+1} - Correct Model Picture", "success": correct_picture,
                "detail": picture_detail,
                "execution_time": time.time() - step_start})

    # Step 4: Picture takes up at least 50% of slide
    step_start = time.time()
    try:
        max_coverage = largest_image_slide_coverage(slide, slide_width_emu, slide_height_emu)
    except Exception as e:
        print(f"Warning: image-coverage extraction failed for car {car_idx}: {e}")
        max_coverage = 0.0
    picture_large = max_coverage >= 50
    steps.append({"name": f"Car {car_idx+1} - Picture >= 50% of Slide", "success": picture_large,
                "detail": f"Largest image covers {max_coverage:.2f}% of slide" if picture_large
                else f"Largest image covers {max_coverage:.2f}% (need >= 50%)",
                "execution_time": time.time() - step_start})

    # Step 5: Picture does not overlap title or stats text boxes (>20% of tight
    # text region inside any image bbox => fail). Only the title and the stats
    # /URL combined box are in scope.
    step_start = time.time()
    picture_clear = True
    no_overlap_detail = "Picture does not overlap title or stats text"
    try:
        title_box, stats_box = find_title_and_stats_text_boxes(slide)
        in_scope_text = [tb for tb in (title_box, stats_box) if tb]

        image_bboxes = []
        for element in slide.get('pageElements', []) or []:
            if 'image' in element:
                image_bboxes.append(get_element_bbox(element))

        overlap_threshold = 0.2
        overlapping_text = None
        for img_bbox in image_bboxes:
            if not img_bbox.get('width') or not img_bbox.get('height'):
                continue
            for tb in in_scope_text:
                tight = estimate_text_render_bbox(tb, presentation=presentation)
                if bbox_overlap_ratio(tight, img_bbox) > overlap_threshold:
                    overlapping_text = (tb.get('text', '') or '')[:60]
                    break
            if overlapping_text:
                break

        picture_clear = overlapping_text is None
        if not in_scope_text:
            no_overlap_detail = "No title or stats text boxes found; overlap check skipped"
        elif not picture_clear:
            no_overlap_detail = f"Picture overlaps text region: '{overlapping_text}...'"
    except Exception as e:
        print(f"Warning: overlap check failed for car {car_idx}: {e}")
        picture_clear = False
        no_overlap_detail = f"Could not evaluate overlap (error: {e})"

    steps.append({"name": f"Car {car_idx+1} - Picture Does Not Overlap Text", "success": picture_clear,
                "detail": no_overlap_detail,
                "execution_time": time.time() - step_start})

    # Steps 6-8: Sticker price, fuel efficiency, horsepower (deterministic compare against CP3-cached KBB stats)
    kbb_stats = (kbb_stats_in or {}).get(car_idx, {})
    has_kbb_data = bool(kbb_stats)

    for stat_name, stat_key, stat_desc, match_key in (
        ("Sticker Price Matches KBB", "sticker_price", "sticker price", "price"),
        ("Fuel Efficiency Matches KBB", "fuel_efficiency", "fuel efficiency or MPG", "mpg"),
        ("Horsepower Matches KBB", "horsepower", "horsepower", "hp"),
    ):
        step_start = time.time()
        slide_value = car_info.get(stat_key, '')
        stat_matches = False
        detail = f"No {stat_desc} data to compare"

        if not slide_value:
            detail = f"No {stat_desc} found on slide"
        elif not has_kbb_data:
            detail = "No KBB stats extracted for this car"
        elif match_key == "price":
            kbb_val = kbb_stats.get('price_numeric', 0) or 0
            stat_matches = compare_price(slide_value, kbb_val)
            if stat_matches:
                diff = abs((_extract_first_number(slide_value) or 0) - kbb_val)
                suffix = "matched" if diff < 1 else f"matched with diff ${diff:.0f}"
                detail = f"Slide: {slide_value} | KBB MSRP: ${kbb_val:.0f} ({suffix})"
            else:
                detail = f"Slide '{slide_value}' does not match KBB MSRP ${kbb_val:.0f}"
        elif match_key == "hp":
            kbb_val = kbb_stats.get('hp_numeric', 0) or 0
            stat_matches = compare_hp(slide_value, kbb_val)
            if stat_matches:
                diff = min(abs(sn - kbb_val) for sn in _extract_all_numbers(slide_value))
                suffix = "matched" if diff < 1 else f"matched with diff {diff:.0f} hp"
                detail = f"Slide: {slide_value} | KBB HP: {kbb_val:.0f} ({suffix})"
            else:
                detail = f"Slide '{slide_value}' does not match KBB HP {kbb_val:.0f}"
        else:  # mpg
            mc = kbb_stats.get('mpg_combined', 0) or 0
            mcity = kbb_stats.get('mpg_city', 0) or 0
            mhwy = kbb_stats.get('mpg_hwy', 0) or 0
            stat_matches = compare_mpg(slide_value, mc, mcity, mhwy)
            if stat_matches:
                kbb_vals = [v for v in (mc, mcity, mhwy) if v]
                diff = min(abs(sn - kv) for sn in _extract_all_numbers(slide_value) for kv in kbb_vals)
                suffix = "matched" if diff < 0.05 else f"matched with diff {diff:.1f} mpg"
                detail = f"Slide: {slide_value} | KBB MPG combined={mc}/city={mcity}/hwy={mhwy} ({suffix})"
            else:
                detail = f"Slide '{slide_value}' does not match KBB MPG combined={mc}/city={mcity}/hwy={mhwy}"

        steps.append({"name": f"Car {car_idx+1} - {stat_name}", "success": bool(stat_matches),
                    "detail": detail, "execution_time": time.time() - step_start})
        if stat_matches_out is not None:
            stat_matches_out.setdefault(car_idx, {})[match_key] = bool(stat_matches)

    # Step 8: Review URL provided
    step_start = time.time()
    has_review_url = car_idx in review_urls
    steps.append({"name": f"Car {car_idx+1} - Review URL Provided", "success": has_review_url,
                "detail": f"Review URL found: {review_urls.get(car_idx, 'N/A')}" if has_review_url
                else "No review URL found in slide",
                "execution_time": time.time() - step_start})

    # Step 9: Browsing history contains the review platform URL
    step_start = time.time()
    review_in_history = False
    if has_review_url and browsing_history:
        review_in_history = find_review_url_in_history(browsing_history, review_urls[car_idx])

    steps.append({"name": f"Car {car_idx+1} - Review URL in History", "success": review_in_history,
                "detail": "Review URL found in browsing history" if review_in_history
                else "Review URL not found in browsing history",
                "execution_time": time.time() - step_start})

    # Step 10: User average rating matches review platform
    step_start = time.time()
    rating_matches = False
    detail = "No rating data to compare"
    review_result = web_contents.get(review_urls.get(car_idx))
    if isinstance(review_result, tuple) and review_result:
        review_content = review_result[0]
    elif isinstance(review_result, str):
        review_content = review_result
    else:
        review_content = None
    slide_rating = car_info.get('user_rating', '')

    if slide_rating and review_content:
        try:
            rating_matches = evaluate_with_llm(
                f"Does the following user average rating from the slide match or closely match the average user rating found on the review source page?\n\nSlide rating: {slide_rating}\n\nReview source content:\n{review_content}",
                model, return_type="bool"
            )
            detail = (f"Slide rating: {slide_rating}, verified against review platform" if rating_matches
                    else f"Slide rating '{slide_rating}' does not match review platform")
        except Exception as e:
            detail = f"Error comparing rating: {e}"
    elif not slide_rating:
        detail = "No user rating found on slide"
    elif not review_content:
        detail = "Could not fetch review platform content for comparison"

    steps.append({"name": f"Car {car_idx+1} - Rating Matches Review Platform", "success": bool(rating_matches),
                "detail": detail, "execution_time": time.time() - step_start})
    if stat_matches_out is not None:
        stat_matches_out.setdefault(car_idx, {})["rating"] = bool(rating_matches)

    return steps