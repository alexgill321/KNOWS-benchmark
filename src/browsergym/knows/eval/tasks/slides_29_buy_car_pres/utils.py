"""Helpers for slides_29 task evaluation: LLM extraction, image coverage,
URL parsing, and per-car CP3 step evaluation."""
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.llm_utils import (
    extract_json_with_llm as _extract_json_with_llm,
    evaluate_with_llm as _evaluate_with_llm,
)
from src.browsergym.knows.eval.eval_utils.scoring import Checkpoint
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_links,
    extract_slide_text,
    extract_title_text,
    get_image_area_percentage_from_api,
)
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_match_robust


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
    "Sticker Price Matches KBB",
    "Fuel Efficiency Matches KBB",
    "Horsepower Matches KBB",
    "Review URL Provided",
    "Review URL in History",
    "Rating Matches Review Platform",
]
CP_STEP_SHAPES = [
    ("Title Slide", 2, ["Title Match", "Title Font Size at least 30pt"]),
    ("Car Content Slides", 6, ["Article Visit", "At Least 5 Car Slides"]),
    ("Car Slides Validation", 50,
     [f"Car {c+1} - {n}" for c in range(5) for n in CP3_PER_CAR_STEPS]),
    ("Summary Slide", 5, [
        "Title Denotes Best Car Stats",
        "Lowest Price Car",
        "Highest MPG Car",
        "Highest Horsepower Car",
        "Most Highly Rated Car",
    ]),
]

# Maps CP4 internal source_key to JSON key used in the winner-extraction prompt.
CP4_WINNER_KEY_MAP = {
    'price': 'lowest_price',
    'mpg': 'highest_mpg',
    'hp': 'highest_horsepower',
    'rating': 'highest_rating',
}


def extract_info_with_llm(task_text: str, model: Any) -> Optional[Dict[str, Any]]:
    """Extract structured information from text using an LLM.

    Args:
        task_text (str): Prompt text with extraction instructions.
        model (Any): Callable LLM interface.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dict, or None on failure.
    """
    return _extract_json_with_llm(task_text, model)


def evaluate_with_llm(task_text: str, model: Any, return_type: Literal["bool", "str", "json"] = "bool") -> Optional[Union[bool, str, Any]]:
    """Evaluate text using an LLM and return result in specified format.

    Args:
        task_text (str): The text/question to evaluate.
        model (Any): Callable LLM interface.
        return_type (str): Format of the returned result: 'bool', 'str', or 'json'.

    Returns:
        Optional[Union[bool, str, Any]]: Result in the specified format, or None on error.
    """
    return _evaluate_with_llm(task_text, model, return_type=return_type)


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


def has_stat_data(stats: Any) -> bool:
    """True when stats is a dict with at least one non-falsy stat field."""
    if not isinstance(stats, dict):
        return False
    return any(stats.get(k) for k in ("price_numeric", "mpg_numeric", "hp_numeric", "rating_numeric"))


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


def expected_car_in_text(text: str, expected_car: str) -> bool:
    """All required tokens of expected_car present in text (case-insensitive).

    Years (1900-2099) are optional; stopwords dropped.
    """
    if not text or not expected_car:
        return False
    text_lower = str(text).lower()
    raw_tokens = re.findall(r'\w+', str(expected_car).lower())
    required_tokens = []
    for t in raw_tokens:
        if len(t) == 4 and t.isdigit() and 1900 <= int(t) <= 2099:
            continue
        if t in _EXPECTED_CAR_STOPWORDS:
            continue
        required_tokens.append(t)
    if not required_tokens:
        return False
    return all(
        re.search(r'\b' + re.escape(t) + r'\b', text_lower) for t in required_tokens
    )


def find_kbb_url_for_car(browsing_history: List[str], make_model: str) -> Optional[str]:
    """Find a kbb.com URL matching make_model — try all tokens, then >3-char ones."""
    if not browsing_history or not make_model:
        return None

    name_tokens = [t for t in re.findall(r'\w+', make_model.lower()) if len(t) > 2]

    for url in browsing_history:
        url_lower = url.lower()
        if 'kbb.com' in url_lower:
            if all(token in url_lower for token in name_tokens):
                return url

    long_tokens = [t for t in name_tokens if len(t) > 3]
    if long_tokens:
        for url in browsing_history:
            url_lower = url.lower()
            if 'kbb.com' in url_lower:
                if any(token in url_lower for token in long_tokens):
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


def evaluate_single_car(car_idx, car_slides, step_names, car_infos, kbb_urls, review_urls, web_contents, browsing_history, slide_image_dirs, kbb_example_dirs, slide_width_emu, slide_height_emu, model):
    """Evaluate all 10 steps for a single car. Returns list of step result dicts."""
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
    slide_title = extract_title_text(slide)
    title_match = keywords_match_robust(slide_title, make_model, model=model)

    make_model_in_title = bool(make_model.strip()) and bool(title_match)
    steps.append({"name": f"Car {car_idx+1} - Make and Model Listed as Title", "success": make_model_in_title,
                "detail": f"Make/model found: {make_model}" if make_model_in_title
                else "No make/model found on slide",
                "execution_time": time.time() - step_start})

    # Step 3: Picture of correct model
    step_start = time.time()
    correct_picture = False
    temp_dir = slide_image_dirs.get(car_idx)
    example_dir = kbb_example_dirs.get(car_idx)
    try:
        if temp_dir and os.path.exists(temp_dir) and os.listdir(temp_dir):
            if example_dir:
                matching = binary_judge_image(
                    model,
                    temp_dir,
                    "Is this an image of the same car model as shown in the example images?",
                    examples=example_dir
                )
            else:
                matching = binary_judge_image(
                    model,
                    temp_dir,
                    f"Is this an image of a {make_model} car or minivan?"
                )
            correct_picture = bool(matching)
    except Exception as e:
        print(f"Error checking car image: {e}")

    steps.append({"name": f"Car {car_idx+1} - Correct Model Picture", "success": correct_picture,
                "detail": "Found picture of correct car model" if correct_picture
                else "No matching car picture found",
                "execution_time": time.time() - step_start})

    # Step 4: Picture takes up at least 50% of slide
    step_start = time.time()
    max_coverage = get_image_area_percentage_from_api(slide, slide_width_emu, slide_height_emu)
    picture_large = max_coverage >= 50
    steps.append({"name": f"Car {car_idx+1} - Picture >= 50% of Slide", "success": picture_large,
                "detail": f"Largest image covers {max_coverage:.2f}% of slide" if picture_large
                else f"Largest image covers {max_coverage:.2f}% (need >= 50%)",
                "execution_time": time.time() - step_start})

    # Steps 5-7: Sticker price, fuel efficiency, horsepower from KBB
    kbb_result = web_contents.get(kbb_urls.get(car_idx))
    kbb_content = kbb_result[0] if kbb_result else None
    stat_fields = [
        ("Sticker Price Matches KBB", "sticker_price", "sticker price"),
        ("Fuel Efficiency Matches KBB", "fuel_efficiency", "fuel efficiency or MPG"),
        ("Horsepower Matches KBB", "horsepower", "horsepower"),
    ]

    for stat_name, stat_key, stat_desc in stat_fields:
        step_start = time.time()
        slide_value = car_info.get(stat_key, '')
        stat_matches = False
        detail = f"No {stat_desc} data to compare"

        if slide_value and kbb_content:
            try:
                kbb_truncated = kbb_content[:15000]
                stat_matches = evaluate_with_llm(
                    f"Does the following {stat_desc} value from the slide match or closely match the {stat_desc} listed on the Kelly Blue Book source page for this vehicle?\n\nSlide value: {slide_value}\n\nKBB Source content:\n{kbb_truncated}",
                    model, return_type="bool"
                )
                detail = (f"Slide: {slide_value}, verified against KBB" if stat_matches
                        else f"Slide value '{slide_value}' does not match KBB data")
            except Exception as e:
                detail = f"Error comparing {stat_desc}: {e}"
        elif not slide_value:
            detail = f"No {stat_desc} found on slide"
        elif not kbb_content:
            detail = f"Could not fetch KBB content for comparison"

        steps.append({"name": f"Car {car_idx+1} - {stat_name}", "success": bool(stat_matches),
                    "detail": detail, "execution_time": time.time() - step_start})

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
    review_content = review_result[0] if review_result else None
    slide_rating = car_info.get('user_rating', '')

    if slide_rating and review_content:
        try:
            review_truncated = review_content[:15000] if review_content else ""
            rating_matches = evaluate_with_llm(
                f"Does the following user average rating from the slide match or closely match the average user rating found on the review source page?\n\nSlide rating: {slide_rating}\n\nReview source content:\n{review_truncated}",
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

    return steps