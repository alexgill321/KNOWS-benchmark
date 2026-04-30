"""
Helpers for slides_29 task evaluation.

These helpers provide task-specific utilities for extracting and validating
car comparison presentation data, including LLM-based extraction, image
coverage calculation, and browsing history URL matching.
"""
import os
import re
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse
import time

from src.browsergym.knows.eval.eval_utils.image_utils import binary_judge_image
from src.browsergym.knows.eval.eval_utils.llm_utils import (
    extract_json_with_llm as _extract_json_with_llm,
    evaluate_with_llm as _evaluate_with_llm,
)
from src.browsergym.knows.eval.eval_utils.slides_utils import extract_title_text, get_image_area_percentage_from_api
from src.browsergym.knows.eval.eval_utils.text_utils import keywords_match_robust


def extract_info_with_llm(task_text: str, model: Any) -> Optional[Dict[str, Any]]:
    """
    Extract structured information from text using an LLM.

    Args:
        task_text (str): Prompt text with extraction instructions.
        model (Any): Callable LLM interface.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dict, or None on failure.
    """
    return _extract_json_with_llm(task_text, model)


def evaluate_with_llm(task_text: str, model: Any, return_type: Literal["bool", "str", "json"] = "bool") -> Optional[Union[bool, str, Any]]:
    """
    Evaluate text using an LLM and return result in specified format.

    Args:
        task_text (str): The text/question to evaluate.
        model (Any): Callable LLM interface.
        return_type (str): Format of the returned result: 'bool', 'str', or 'json'.

    Returns:
        Optional[Union[bool, str, Any]]: Result in the specified format, or None on error.
    """
    return _evaluate_with_llm(task_text, model, return_type=return_type)

def find_kbb_url_for_car(browsing_history: List[str], make_model: str) -> Optional[str]:
    """
    Find a KBB URL in browsing history that matches a specific car make/model.

    Splits the make/model into tokens and searches for kbb.com URLs
    containing those tokens. Uses progressively relaxed matching:
    first all tokens, then just tokens longer than 3 chars.

    Args:
        browsing_history (List[str]): List of visited URLs.
        make_model (str): Car make and model (e.g., "Toyota Sienna").

    Returns:
        Optional[str]: Matching KBB URL, or None if not found.
    """
    if not browsing_history or not make_model:
        return None

    name_tokens = [t for t in re.findall(r'\w+', make_model.lower()) if len(t) > 2]

    # Try matching all tokens
    for url in browsing_history:
        url_lower = url.lower()
        if 'kbb.com' in url_lower:
            if all(token in url_lower for token in name_tokens):
                return url

    # Fallback: match tokens longer than 3 chars
    long_tokens = [t for t in name_tokens if len(t) > 3]
    if long_tokens:
        for url in browsing_history:
            url_lower = url.lower()
            if 'kbb.com' in url_lower:
                if any(token in url_lower for token in long_tokens):
                    return url

    return None

def find_review_url_in_history(browsing_history: List[str], review_url: str) -> bool:
    """
    Check if a review URL (or its domain) appears in the browsing history.

    Args:
        browsing_history (List[str]): List of visited URLs.
        review_url (str): The review platform URL to search for.

    Returns:
        bool: True if the review URL or its domain is found in history.
    """
    if not browsing_history or not review_url:
        return False

    # Exact match
    if review_url in browsing_history:
        return True

    # Domain match
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