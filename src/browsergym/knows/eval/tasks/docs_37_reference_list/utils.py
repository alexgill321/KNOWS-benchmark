"""Shared utilities for docs_37_reference_list evaluator.

Provides functions to parse Google Doc structure for the reference list task,
including heading extraction, bookmark detection, bullet list parsing, and
hyperlink metadata extraction.
"""

import io
import json
import re
import contextlib

import requests
from rapidfuzz import fuzz

from src.browsergym.knows.eval.eval_utils.llm_utils import extract_json_with_llm
from src.browsergym.knows.eval.eval_utils.table_utils import colors_are_similar
from src.browsergym.knows.eval.eval_utils.text_utils import (
    fuzzy_match_text,
    keywords_match_robust,
    match_text_in_list,
)
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_title

# Dark green 2 RGB values as seen in Google Docs API 
DARK_GREEN_2_RGB = {"red": 0.219, "green": 0.463, "blue": 0.113}
# Dark cyan 1 RGB values as seen in Google Docs API 
DARK_CYAN_1_RGB = {"red": 0.270, "green": 0.506, "blue": 0.557}
# Light magenta 1 RGB values as seen in Google Docs API 
LIGHT_MAGENTA_1_RGB = {"red": 0.760, "green": 0.4824, "blue": 0.6275}
# Dark yellow 1 RGB values as seen in Google Docs API 
DARK_YELLOW_1_RGB = {"red": 0.945, "green": 0.761, "blue": 0.196}
# Dark purple 1 RGB values as seen in Google Docs API 
DARK_PURPLE_1_RGB = {"red": 0.4039, "green": 0.3059, "blue": 0.6549}
COLOR_TOLERANCE = 0.03


def iter_paragraphs(document):
    """Iterate over paragraphs in a Google Doc, yielding parsed metadata.

    Args:
        document (dict): Full Google Docs API document response.

    Yields:
        dict: Parsed paragraph info with keys:
            - paragraph (dict): Raw paragraph object.
            - style_type (str): The namedStyleType (e.g. 'HEADING_3').
            - bullet (dict): Bullet info (empty dict if not a bullet).
            - para_style (dict): The paragraphStyle dict.
    """
    body_content = document.get("body", {}).get("content", [])
    for element in body_content:
        if "paragraph" not in element:
            continue
        paragraph = element["paragraph"]
        para_style = paragraph.get("paragraphStyle", {})
        yield {
            "paragraph": paragraph,
            "style_type": para_style.get("namedStyleType", ""),
            "bullet": paragraph.get("bullet", {}),
            "para_style": para_style,
        }


def get_paragraph_text(paragraph):
    """Get the full plain text of a paragraph.

    Args:
        paragraph (dict): Paragraph object from Google Docs API.

    Returns:
        str: Concatenated text content, stripped.
    """
    text = ""
    for elem in paragraph.get("elements", []):
        if "textRun" in elem:
            text += elem["textRun"].get("content", "")
    return text.strip()


def is_dark_green_2(text_style):
    """Check if a textStyle's foregroundColor matches dark green 2.

    Args:
        text_style (dict): The textStyle dict from a textRun.

    Returns:
        bool: True if the color matches dark green 2 within tolerance.
    """
    fg = text_style.get("foregroundColor", {}).get("color", {}).get("rgbColor", {})
    if not fg:
        return False

    return colors_are_similar(fg, DARK_GREEN_2_RGB, tolerance=COLOR_TOLERANCE)


def is_dark_cyan_1(text_style):
    """Check if a textStyle's foregroundColor matches dark cyan 1.

    Args:
        text_style (dict): The textStyle dict from a textRun.

    Returns:
        bool: True if the color matches dark cyan 1 within tolerance.
    """
    fg = text_style.get("foregroundColor", {}).get("color", {}).get("rgbColor", {})
    if not fg:
        return False

    return colors_are_similar(fg, DARK_CYAN_1_RGB, tolerance=COLOR_TOLERANCE)


def is_light_magenta_1(text_style):
    """Check if a textStyle's foregroundColor matches light magenta 1.

    Args:
        text_style (dict): The textStyle dict from a textRun.

    Returns:
        bool: True if the color matches light magenta 1 within tolerance.
    """
    fg = text_style.get("foregroundColor", {}).get("color", {}).get("rgbColor", {})
    if not fg:
        return False

    return colors_are_similar(fg, LIGHT_MAGENTA_1_RGB, tolerance=COLOR_TOLERANCE)


def is_dark_yellow_1(text_style):
    """Check if a textStyle's foregroundColor matches dark yellow 1.

    Args:
        text_style (dict): The textStyle dict from a textRun.

    Returns:
        bool: True if the color matches dark yellow 1 within tolerance.
    """
    fg = text_style.get("foregroundColor", {}).get("color", {}).get("rgbColor", {})
    if not fg:
        return False

    return colors_are_similar(fg, DARK_YELLOW_1_RGB, tolerance=COLOR_TOLERANCE)


def is_dark_purple_1(text_style):
    """Check if a textStyle's foregroundColor matches dark purple 1.

    Args:
        text_style (dict): The textStyle dict from a textRun.

    Returns:
        bool: True if the color matches dark purple 1 within tolerance.
    """
    fg = text_style.get("foregroundColor", {}).get("color", {}).get("rgbColor", {})
    if not fg:
        return False

    return colors_are_similar(fg, DARK_PURPLE_1_RGB, tolerance=COLOR_TOLERANCE)


def parse_slide_numbers(text):
    """Extract slide numbers from text like ' Slide: 6, 7'.

    Args:
        text (str): Text potentially containing slide references.

    Returns:
        list: Sorted list of integer slide numbers, or empty list.
    """
    match = re.search(r"Slides?:\s*([\d,\s]+)", text)
    if not match:
        return []
    return sorted(
        int(n.strip()) for n in match.group(1).split(",") if n.strip().isdigit()
    )


def extract_headings_with_bookmarks(document):
    """Extract paragraphs that could be lecture titles and check style/bookmark.

    Searches all paragraph styles (not just HEADING_3) so that each evaluation
    step (format, style, bookmark) can assess independently.

    Args:
        document (dict): Full Google Docs API document response.

    Returns:
        list[dict]: List of dicts with keys:
            - text (str): The paragraph text content.
            - has_bookmark (bool): Whether the paragraph has a headingId or bookmarkId.
            - style (str): The named style type (e.g. 'HEADING_3', 'HEADING_4', 'NORMAL_TEXT').
            - heading_id (str|None): The headingId value if present.
    """
    headings = []

    # Build set of character indices covered by named ranges (explicit bookmarks)
    bookmarked_indices = set()
    for name, named_range in document.get("namedRanges", {}).items():
        for nr in named_range.get("namedRanges", []):
            for rng in nr.get("ranges", []):
                start = rng.get("startIndex", 0)
                end = rng.get("endIndex", 0)
                bookmarked_indices.update(range(start, end + 1))

    for info in iter_paragraphs(document):
        text = get_paragraph_text(info["paragraph"])
        if not text:
            continue

        # Skip bullet items (they're reference entries, not lecture headings)
        if info["bullet"]:
            continue

        # For heading styles, headingId serves as the bookmark anchor
        heading_id = info["para_style"].get("headingId")
        has_bookmark = bool(heading_id)

        # For non-heading styles, check if paragraph overlaps a namedRange
        if not has_bookmark:
            para_start = info["paragraph"].get("elements", [{}])[0].get("startIndex", -1)
            if para_start in bookmarked_indices:
                has_bookmark = True

        headings.append({
            "text": text,
            "has_bookmark": has_bookmark,
            "style": info["style_type"],
            "heading_id": heading_id,
        })

    return headings


def pair_headings_to_gold(gold_lectures, doc_headings, model=None):
    """Pair gold lecture strings to their corresponding doc heading dicts.

    Two-stage matching:
      1. Prefix match with coverage tie-break: the part of a heading's text
         before the first ':' (the lecture identifier — "Module 1",
         "Session 13", "10/15", "Guest Lecture", etc.) must equal the gold
         lecture's prefix, case-insensitive and whitespace-stripped. When
         multiple doc headings share the same prefix as a gold lecture, the
         one whose full text best COVERS gold is chosen (rapidfuzz
         token_set_ratio — rewards candidates containing all gold tokens, so
         "guest lecture: fine tuning is challenging" beats "guest lecture:
         tuning" when matching "guest lecture: tuning is challenging";
         fuzz.ratio breaks ties toward candidates of similar length to gold).
      2. LLM fallback (only when ``model`` is provided): any gold lecture left
         unmatched after stage 1 is paired against the remaining unmatched doc
         headings by the LLM. The LLM's JSON pairing is parsed and applied
         one-to-one — no doc heading is bound to two gold lectures.

    Args:
        gold_lectures (list[str]): Expected lecture title strings.
        doc_headings (list[dict]): Heading dicts from
            :func:`extract_headings_with_bookmarks`. Each dict must have a
            'text' key.
        model: Optional callable LLM interface. When omitted, stage 2 is
            skipped and any unmatched gold lectures stay unmatched.

    Returns:
        dict[str, dict|None]: Map from each gold lecture string to its matched
        heading dict (or ``None`` if no match was found after both stages).
    """
    # --- Stage 1: prefix match with coverage tie-break for duplicates ---
    matched = {}
    used_indices = set()
    headings_by_prefix = {}
    for idx, h in enumerate(doc_headings):
        prefix = h["text"].partition(":")[0].strip().lower()
        if prefix:
            headings_by_prefix.setdefault(prefix, []).append((idx, h))

    for gold_lecture in gold_lectures:
        gold_prefix = gold_lecture.partition(":")[0].strip().lower()
        candidates = [
            (idx, h) for idx, h in headings_by_prefix.get(gold_prefix, [])
            if idx not in used_indices
        ]
        if not candidates:
            matched[gold_lecture] = None
            continue
        if len(candidates) == 1:
            chosen_idx, chosen_h = candidates[0]
        else:
            chosen_idx, chosen_h = max(
                candidates,
                key=lambda ic: (
                    fuzz.token_set_ratio(gold_lecture, ic[1]["text"]),
                    fuzz.ratio(gold_lecture, ic[1]["text"]),
                ),
            )
        matched[gold_lecture] = chosen_h
        used_indices.add(chosen_idx)

    # --- Stage 2: LLM fallback for remaining unmatched pairs ---
    unmatched_gold = [g for g, h in matched.items() if h is None]
    remaining_headings = [h for idx, h in enumerate(doc_headings)
                          if idx not in used_indices]

    if model is None or not unmatched_gold or not remaining_headings:
        return matched

    available_texts = [h["text"] for h in remaining_headings]
    prompt = (
        "Pair each expected lecture title with the document heading that most "
        "likely refers to the same lecture. Return ONLY a JSON object whose "
        "keys are the expected titles (exact strings from the first list) and "
        "whose values are the matched document heading string (exact strings "
        "from the second list), or null if no good match exists. Do not "
        "invent headings — only use strings from the provided list.\n\n"
        f"Expected titles:\n{json.dumps(unmatched_gold, ensure_ascii=False)}\n\n"
        f"Document headings:\n{json.dumps(available_texts, ensure_ascii=False)}"
    )
    result = extract_json_with_llm(prompt, model, expect_type="object")
    if not isinstance(result, dict):
        return matched

    heading_lookup = {h["text"]: h for h in remaining_headings}
    used_doc_texts = set()
    for gold_lecture in unmatched_gold:
        doc_text = result.get(gold_lecture)
        if (isinstance(doc_text, str)
                and doc_text in heading_lookup
                and doc_text not in used_doc_texts):
            matched[gold_lecture] = heading_lookup[doc_text]
            used_doc_texts.add(doc_text)

    return matched


def title_matches(doc_title, gold_title, threshold=70):
    """Soft-check that a doc title fuzzy-covers the gold title.

    Uses rapidfuzz ``partial_ratio`` (sliding-window substring match) so a doc
    title that wraps gold's title with extra annotation (e.g. an additional
    leading clause) still passes. An empty gold title passes vacuously.

    Args:
        doc_title (str): The title text from the document heading (the part
            after the prefix colon).
        gold_title (str): The expected title text from the gold lecture (the
            part after the prefix colon).
        threshold (int): Minimum partial_ratio score (0-100). Default 70.

    Returns:
        bool: True if ``gold_title`` is empty, or if the partial_ratio of
        ``doc_title`` against ``gold_title`` meets or exceeds ``threshold``.
    """
    if not gold_title:
        return True
    return fuzz.partial_ratio(doc_title or "", gold_title) >= threshold


def extract_bullet_sections(document):
    """Extract reference category sections grouped by lecture.

    Walks the document section by section: a heading-styled (or M/D-formatted)
    non-bullet paragraph starts a new lecture. Within a lecture, every non-empty
    non-link line starts a new category block, and the hyperlink lines that
    follow it become that block's items.

    A category block is created from ANY non-link line — bulleted or not — so a
    category header that is missing its bullet (or otherwise malformed) still
    yields a section whose links can be evaluated. The `category_is_bold` and
    `category_is_bullet` flags capture the header's actual formatting so the
    format-checking steps can report what is wrong.

    Args:
        document (dict): Full Google Docs API document response.

    Returns:
        list[dict]: One dict per category block, with keys:
            - lecture (str): The parent lecture heading text.
            - category (str): The category header text (e.g. "Academic Articles").
            - category_is_bold (bool): Whether the category header text is bold.
            - category_is_bullet (bool): Whether the category header is a bullet.
            - items (list[dict]): Link items in this block, each with:
                - text (str): Full text of the link line.
                - has_link (bool): Always True (only hyperlink lines are items).
    """
    sections = []
    current_lecture = None
    current_section = None

    for info in iter_paragraphs(document):
        paragraph = info["paragraph"]
        bullet = info["bullet"]

        # Section boundary: a heading-styled (or M/D-formatted) non-bullet
        # paragraph starts a new lecture.
        if not bullet and (info["style_type"].startswith("HEADING")
                           or matches_lecture_title_format(get_paragraph_text(paragraph))):
            if current_section:
                sections.append(current_section)
                current_section = None
            current_lecture = get_paragraph_text(paragraph)
            continue

        if not current_lecture:
            continue

        has_link = any(
            "link" in e.get("textRun", {}).get("textStyle", {})
            for e in paragraph.get("elements", [])
            if "textRun" in e
        )

        if has_link:
            # A hyperlink line is an item of the current category block.
            if current_section:
                current_section["items"].append({
                    "text": get_paragraph_text(paragraph),
                    "has_link": True,
                })
            continue

        text = get_paragraph_text(paragraph)
        if not text:
            continue

        # A non-empty, non-link line is a category header — it starts a new
        # category block, regardless of whether it is bulleted.
        if current_section:
            sections.append(current_section)

        category_is_bold = False
        for elem in paragraph.get("elements", []):
            if "textRun" in elem:
                tr = elem["textRun"]
                if tr.get("content", "").strip():
                    category_is_bold = tr.get("textStyle", {}).get("bold", False)
                    break

        current_section = {
            "lecture": current_lecture,
            "category": text,
            "category_is_bold": category_is_bold,
            "category_is_bullet": bool(bullet),
            "items": [],
        }

    if current_section:
        sections.append(current_section)

    return sections


def matches_lecture_title_format(text):
    """Check if text matches 'Month/Day: Lecture title' format.

    Validates that the first number is a valid month (1-12).

    Args:
        text (str): The heading text to validate.

    Returns:
        bool: True if the text matches the expected format with a valid month.
    """
    match = re.match(r"^(\d{1,2})/(\d{1,2}):\s+.+", text)
    if not match:
        return False
    month = int(match.group(1))
    return 1 <= month <= 12


def get_gold_lectures(gold_data):
    """Extract unique lecture titles from gold outputs, preserving order.

    Args:
        gold_data (list[dict]): Loaded gold_outputs.json data.

    Returns:
        list[str]: Ordered list of unique lecture titles.
    """
    seen = set()
    lectures = []
    for item in gold_data:
        lecture = item["lecture"]
        if lecture not in seen:
            seen.add(lecture)
            lectures.append(lecture)
    return lectures


def match_valid_category(category_text, valid_categories, model=None, return_method=False):
    """Check if a category title matches one of the valid reference categories.

    Uses keywords_match_robust for exact match first, then LLM semantic fallback.
    The valid category set is instance-specific and must be supplied by the
    caller (each instance defines its own categories based on its task.md).

    Args:
        category_text (str): The category title from the document.
        valid_categories (set|list): The instance's valid category names.
        model: Optional LLM model callable for fallback matching.
        return_method (bool): If True, return (match, method) where method is
            "exact"|"llm"|None, recording which phase decided the match.

    Returns:
        str|None: The matched canonical category name, or None if no match.
        If return_method is True, returns (match, method) instead.
    """
    return keywords_match_robust(
        category_text,
        list(valid_categories),
        model=model,
        description="reference list category type",
        return_method=return_method,
    )



def extract_reference_links(document):
    """Extract detailed reference link items from the document.

    Walks the document section by section: a heading-styled (or M/D-formatted)
    non-bullet paragraph starts a new lecture section, and everything until the
    next such paragraph belongs to that section. Within a section, every line
    carrying a hyperlink is collected as a reference; any other non-empty line
    is treated as the current category label.

    Reference links are collected regardless of whether the category line above
    them is correctly formatted (bulleted/bold or not) — a malformed category
    header therefore does NOT cause its links to be dropped. Category-format
    checks are handled separately (see extract_bullet_sections).

    Args:
        document (dict): Full Google Docs API document response.

    Returns:
        list[dict]: List of reference dicts with keys:
            - lecture (str): The parent lecture/section heading text.
            - category (str): The most recent category label above the link
              ("" if no category line preceded it within the section).
            - anchor_text (str): The visible hyperlink text.
            - url (str): The hyperlink URL.
            - slide_numbers (list[int]): Parsed slide numbers from the item text.
            - full_text (str): The full text of the line.
            - link_is_* (bool): Formatting flags for the hyperlink span.
            - non_link_is_* (bool): Whether any non-whitespace, non-link text in
              the same line is also styled — used to enforce the "(and only the
              hyperlink)" rule from task.md.
    """
    references = []
    current_lecture = None
    current_category = ""

    for info in iter_paragraphs(document):
        paragraph = info["paragraph"]
        bullet = info["bullet"]

        # Section boundary: a heading-styled (or M/D-formatted) non-bullet
        # paragraph starts a new lecture section.
        if not bullet and (info["style_type"].startswith("HEADING")
                           or matches_lecture_title_format(get_paragraph_text(paragraph))):
            current_lecture = get_paragraph_text(paragraph)
            current_category = ""
            continue

        if not current_lecture:
            continue

        # Within a section: a line carrying a hyperlink is a reference entry;
        # any other non-empty line is treated as a category label. Links are
        # collected regardless of the category line's formatting.
        has_link = any(
            "link" in e.get("textRun", {}).get("textStyle", {})
            for e in paragraph.get("elements", [])
            if "textRun" in e
        )

        if not has_link:
            category_text = get_paragraph_text(paragraph)
            if category_text:
                current_category = category_text
            continue

        # Reference (link) entry — extract URL, anchor text, and formatting.
        anchor_text = ""
        url = ""
        full_text = get_paragraph_text(paragraph)
        link_is_bold = False
        link_is_italic = False
        link_is_underline = False
        link_is_dark_green = False
        link_is_dark_cyan = False
        link_is_light_magenta = False
        link_is_dark_yellow = False
        link_is_dark_purple = False
        link_is_strikethrough = False
        link_font_size = None
        non_link_is_bold = False
        non_link_is_italic = False
        non_link_is_underline = False
        non_link_is_dark_green_2 = False
        non_link_is_dark_cyan_1 = False
        non_link_is_light_magenta_1 = False
        non_link_is_dark_yellow_1 = False
        non_link_is_dark_purple_1 = False
        non_link_is_12pt = False
        non_link_font_size = None

        for elem in paragraph.get("elements", []):
            if "textRun" not in elem:
                continue
            tr = elem["textRun"]
            ts = tr.get("textStyle", {})
            content = tr.get("content", "")

            if "link" in ts:
                link_url = ts["link"].get("url", "")
                if link_url and not url:
                    url = link_url
                    anchor_text = content.strip()
                    link_is_bold = ts.get("bold", False)
                    link_is_italic = ts.get("italic", False)
                    link_is_underline = ts.get("underline", False)
                    link_is_dark_green = is_dark_green_2(ts)
                    link_is_dark_cyan = is_dark_cyan_1(ts)
                    link_is_light_magenta = is_light_magenta_1(ts)
                    link_is_dark_yellow = is_dark_yellow_1(ts)
                    link_is_dark_purple = is_dark_purple_1(ts)
                    link_is_strikethrough = ts.get("strikethrough", False)
                    link_font_size = ts.get("fontSize", {}).get("magnitude")
            else:
                if content.strip():
                    # Track formatting leaks outside the hyperlink span
                    if ts.get("bold", False):
                        non_link_is_bold = True
                    if ts.get("italic", False):
                        non_link_is_italic = True
                    if ts.get("underline", False):
                        non_link_is_underline = True
                    if is_dark_green_2(ts):
                        non_link_is_dark_green_2 = True
                    if is_dark_cyan_1(ts):
                        non_link_is_dark_cyan_1 = True
                    if is_light_magenta_1(ts):
                        non_link_is_light_magenta_1 = True
                    if is_dark_yellow_1(ts):
                        non_link_is_dark_yellow_1 = True
                    if is_dark_purple_1(ts):
                        non_link_is_dark_purple_1 = True
                    nl_size = ts.get("fontSize", {}).get("magnitude")
                    if nl_size == 12:
                        non_link_is_12pt = True
                    if nl_size is not None and (non_link_font_size is None or nl_size > non_link_font_size):
                        non_link_font_size = nl_size

        slide_numbers = parse_slide_numbers(full_text)

        if url:
            references.append({
                "lecture": current_lecture,
                "category": current_category,
                "anchor_text": anchor_text,
                "url": url,
                "slide_numbers": slide_numbers,
                "full_text": full_text,
                "link_is_bold": link_is_bold,
                "link_is_italic": link_is_italic,
                "link_is_underline": link_is_underline,
                "link_is_dark_green_2": link_is_dark_green,
                "link_is_dark_cyan_1": link_is_dark_cyan,
                "link_is_light_magenta_1": link_is_light_magenta,
                "link_is_dark_yellow_1": link_is_dark_yellow,
                "link_is_dark_purple_1": link_is_dark_purple,
                "link_is_strikethrough": link_is_strikethrough,
                "link_font_size": link_font_size,
                "non_link_is_bold": non_link_is_bold,
                "non_link_is_italic": non_link_is_italic,
                "non_link_is_underline": non_link_is_underline,
                "non_link_is_dark_green_2": non_link_is_dark_green_2,
                "non_link_is_dark_cyan_1": non_link_is_dark_cyan_1,
                "non_link_is_light_magenta_1": non_link_is_light_magenta_1,
                "non_link_is_dark_yellow_1": non_link_is_dark_yellow_1,
                "non_link_is_dark_purple_1": non_link_is_dark_purple_1,
                "non_link_is_12pt": non_link_is_12pt,
                "non_link_font_size": non_link_font_size,
            })

    return references


def extract_inactive_links_section(document):
    """Locate the 'Inactive Links' section and return its hyperlink URLs.

    Scans the document for a paragraph whose text is 'Inactive Links'
    (case-insensitive, whitespace-normalized). Once found, collects every
    hyperlink URL appearing in subsequent paragraphs until the next
    heading-styled non-bullet paragraph (or the end of the document).

    Args:
        document (dict): Full Google Docs API document response.

    Returns:
        tuple: (section_found (bool), urls (list[str])). urls may contain
        duplicates if the same link appears on more than one bullet.
    """
    found = False
    in_section = False
    urls = []
    for info in iter_paragraphs(document):
        paragraph = info["paragraph"]
        text = re.sub(r"\s+", " ", get_paragraph_text(paragraph)).strip()
        if not in_section:
            if text.lower() == "inactive links":
                found = True
                in_section = True
            continue
        # Inside the section: stop at the next heading-styled, non-bullet paragraph.
        if not info["bullet"] and info["style_type"].startswith("HEADING"):
            break
        for elem in paragraph.get("elements", []):
            tr = elem.get("textRun")
            if not tr:
                continue
            url = tr.get("textStyle", {}).get("link", {}).get("url", "")
            if url:
                urls.append(url)
    return found, urls


def check_link_name_relevance(anchor_text, url, model, page_title=None, return_method=False):
    """Check if a link's anchor text is relevant to the webpage it opens.

    First tries fuzzy matching between anchor text and a pre-fetched page title.
    If the page title is not available or fuzzy match fails, falls back to
    LLM-based semantic judgment using the anchor text and URL.

    Args:
        anchor_text (str): The hyperlink text from the document.
        url (str): The URL the link points to.
        model: LLM model callable for semantic fallback.
        page_title (str|None): Pre-fetched page title, or None.
        return_method (bool): If True, return (relevant, method) where method
            is "fuzzy"|"llm"|"error", recording which phase decided the verdict.

    Returns:
        bool: True if the anchor text is relevant to the source.
        If return_method is True, returns (relevant, method) instead.
    """
    # Phase 1: Fuzzy match anchor text against page title
    if page_title:
        is_match, _ = fuzzy_match_text(anchor_text, page_title, threshold=50)
        if is_match:
            return (True, "fuzzy") if return_method else True

    # Phase 2: LLM semantic judgment
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": (
                "You are a link relevance checker. Given a hyperlink's anchor text "
                "and its URL (and optionally the page title), determine if the anchor "
                "text is a reasonable, relevant name for the linked resource. "
                "Answer only 'Yes' or 'No'."
            )}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": (
                f"Anchor text: \"{anchor_text}\"\n"
                f"URL: {url}\n"
                f"Page title: {page_title or 'Could not fetch'}\n\n"
                "Is the anchor text a relevant name for this link? Answer Yes or No."
            )}],
        },
    ]
    try:
        response = model(messages)
        verdict = "yes" in response.strip().lower()
        return (verdict, "llm") if return_method else verdict
    except Exception:
        return (False, "error") if return_method else False


def is_raw_url(text):
    """Check if text looks like a raw URL rather than a descriptive title.

    Args:
        text (str): The anchor text to check.

    Returns:
        bool: True if the text appears to be a raw URL.
    """
    return bool(re.match(r"^https?://", text.strip()))


def check_slide_format(text):
    """Check if text contains slide numbers in the format 'Slide: {n1, n2, ...}'.

    Accepts both 'Slide: 6' and 'Slides: 6, 7, 8' variations.

    Args:
        text (str): The full text of a bullet item.

    Returns:
        bool: True if the text contains a properly formatted slide reference.
    """
    return bool(re.search(r"Slides?:\s*\d+(\s*,\s*\d+)*\s*$", text))


def match_text_quiet(text, text_list, threshold=75):
    """Fuzzy match text against a list, suppressing debug print output.

    Wraps match_text_in_list to silence its internal print statements.

    Args:
        text (str): The text to match.
        text_list (list[str]): Candidate strings to match against.
        threshold (int): Minimum fuzzy match score (0-100).

    Returns:
        tuple: (matched_text, score) or (None, 0) if no match.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return match_text_in_list(text, text_list, threshold=threshold)


# Non-HTML file extensions to skip immediately
_NON_HTML_EXTENSIONS = re.compile(
    r"\.(pdf|png|jpg|jpeg|gif|svg|mp4|mp3|zip|tar|gz|bz2|xz|doc|docx|ppt|pptx|xls|xlsx|csv)(\?|#|$)",
    re.IGNORECASE,
)


def is_dead_link(url, timeout=10):
    """Check if a URL is dead by detecting HTTP 404 or 410 responses.

    Only HTTP 404 (Not Found) and 410 (Gone) are treated as dead.
    All other outcomes are treated as alive.

    Args:
        url (str): The URL to check.
        timeout (int): Seconds to wait per request.

    Returns:
        tuple[bool, str]: (is_dead, reason) — reason always includes the HTTP status code
            or error type.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)
        status = resp.status_code

        if status == 404:
            return True, "HTTP 404 (Not Found)"
        if status == 410:
            return True, "HTTP 410 (Gone)"

        return False, f"HTTP {status}"

    except requests.exceptions.Timeout:
        return False, "No HTTP status (Timeout)"
    except requests.exceptions.ConnectionError:
        return False, "No HTTP status (Connection Error)"
    except Exception as e:
        return False, f"No HTTP status (Error: {e})"


def fetch_page_title_safe(url):
    """Fetch page title, skipping non-HTML URLs via pattern + HEAD check.

    Phase 1: Skip URLs with known non-HTML extensions (zero network cost).
    Phase 2: HEAD request to check Content-Type for ambiguous URLs.
    Phase 3: Delegate to fetch_page_title for HTML pages.

    Args:
        url (str): The URL to fetch.

    Returns:
        str|None: Page title if HTML page, None otherwise.
    """
    # Phase 1: URL pattern skip
    if _NON_HTML_EXTENSIONS.search(url):
        return None

    # Phase 2: HEAD Content-Type check
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and "text/html" not in content_type:
            return None
    except Exception:
        pass  # Fall through — fetch_page_title has its own error handling

    # Phase 3: Delegate to fetch_page_title
    return fetch_page_title(url)


