"""Shared utilities for docs_37_reference_list evaluator.

Provides functions to parse Google Doc structure for the reference list task,
including heading extraction, bookmark detection, bullet list parsing, and
hyperlink metadata extraction.
"""

import io
import re
import contextlib

import requests

from src.browsergym.knows.eval.eval_utils.table_utils import colors_are_similar
from src.browsergym.knows.eval.eval_utils.text_utils import (
    fuzzy_match_text,
    keywords_match_robust,
    match_text_in_list,
)
from src.browsergym.knows.eval.eval_utils.web_utils import fetch_page_title

# Dark green 2 RGB values as seen in Google Docs API (approx #38761D)
DARK_GREEN_2_RGB = {"red": 0.2196, "green": 0.4627, "blue": 0.1137}
COLOR_TOLERANCE = 0.03

# Valid reference categories
VALID_CATEGORIES = {
    "Academic Articles",
    "Textbooks",
    "Blogs",
    "Code Implementations",
    "Tutorials",
    "Wikipedia",
    "Presentations",
}


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


def extract_bullet_sections(document):
    """Extract all bullet list sections grouped by lecture and category.

    Walks the document paragraphs, tracking the current lecture (any heading
    style or M/D-formatted paragraph) and category (nesting-level-0 bullet).
    Returns a list of section dicts, each representing one category under
    one lecture.

    Args:
        document (dict): Full Google Docs API document response.

    Returns:
        list[dict]: List of dicts with keys:
            - lecture (str): The lecture heading text.
            - category (str): The category title text (e.g. "Academic Articles").
            - category_is_bold (bool): Whether the category title is bold.
            - items (list[dict]): List of link items in this section, each with:
                - text (str): Full text of the bullet item.
                - has_link (bool): Whether the item contains a hyperlink.
    """
    sections = []
    current_lecture = None
    current_section = None

    for info in iter_paragraphs(document):
        paragraph = info["paragraph"]
        bullet = info["bullet"]

        # Detect lecture boundary: any heading style or M/D-formatted non-bullet paragraph
        if not bullet and (info["style_type"].startswith("HEADING") or matches_lecture_title_format(get_paragraph_text(paragraph))):
            # Save previous section if exists
            if current_section:
                sections.append(current_section)
                current_section = None
            current_lecture = get_paragraph_text(paragraph)
            continue

        if bullet and current_lecture:
            nesting_level = bullet.get("nestingLevel", 0)

            if nesting_level == 0:
                # Category header bullet
                has_link = any(
                    "link" in e.get("textRun", {}).get("textStyle", {})
                    for e in paragraph.get("elements", [])
                    if "textRun" in e
                )
                if not has_link:
                    # Save previous section
                    if current_section:
                        sections.append(current_section)

                    category_text = get_paragraph_text(paragraph)
                    # Check if the category title text itself is bold
                    category_is_bold = False
                    for elem in paragraph.get("elements", []):
                        if "textRun" in elem:
                            ts = elem["textRun"].get("textStyle", {})
                            content = elem["textRun"].get("content", "").strip()
                            if content:
                                category_is_bold = ts.get("bold", False)
                                break

                    current_section = {
                        "lecture": current_lecture,
                        "category": category_text,
                        "category_is_bold": category_is_bold,
                        "items": [],
                    }
                    continue

            # Link entry (nesting level >= 1, or level 0 with a link)
            if current_section:
                item_text = get_paragraph_text(paragraph)
                has_link = any(
                    "link" in e.get("textRun", {}).get("textStyle", {})
                    for e in paragraph.get("elements", [])
                    if "textRun" in e
                )
                current_section["items"].append({
                    "text": item_text,
                    "has_link": has_link,
                })
        else:
            # Non-bullet, non-heading: finalize current section
            if current_section:
                sections.append(current_section)
                current_section = None

    # Don't forget last section
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


def match_valid_category(category_text, model=None):
    """Check if a category title matches one of the valid reference categories.

    Uses keywords_match_robust for exact match first, then LLM semantic fallback.

    Args:
        category_text (str): The category title from the document.
        model: Optional LLM model callable for fallback matching.

    Returns:
        str|None: The category_text if it matches a valid category, or None.
    """
    return keywords_match_robust(
        category_text,
        list(VALID_CATEGORIES),
        model=model,
        description="reference list category type",
    )



def extract_reference_links(document):
    """Extract detailed reference link items from the document.

    Walks the document structure and extracts every hyperlink bullet item
    with its URL, anchor text, slide numbers, parent lecture, and category.

    Args:
        document (dict): Full Google Docs API document response.

    Returns:
        list[dict]: List of reference dicts with keys:
            - lecture (str): The parent lecture heading text.
            - category (str): The category this link is under.
            - anchor_text (str): The visible hyperlink text.
            - url (str): The hyperlink URL.
            - slide_numbers (list[int]): Parsed slide numbers from the item text.
            - full_text (str): The full text of the bullet item.
            - link_is_bold (bool): Whether the hyperlink text is bold.
            - link_is_dark_green_2 (bool): Whether the hyperlink text is dark green 2.
    """
    references = []
    current_lecture = None
    current_category = None

    for info in iter_paragraphs(document):
        paragraph = info["paragraph"]
        bullet = info["bullet"]

        # Detect lecture boundary: any heading style or M/D-formatted non-bullet paragraph
        if not bullet and (info["style_type"].startswith("HEADING") or matches_lecture_title_format(get_paragraph_text(paragraph))):
            current_lecture = get_paragraph_text(paragraph)
            current_category = None
            continue

        if bullet and current_lecture:
            nesting_level = bullet.get("nestingLevel", 0)

            if nesting_level == 0:
                # Check if this is a category header (no link) or a link item
                has_link = any(
                    "link" in e.get("textRun", {}).get("textStyle", {})
                    for e in paragraph.get("elements", [])
                    if "textRun" in e
                )
                if not has_link:
                    current_category = get_paragraph_text(paragraph)
                    continue

            # This is a link entry — extract URL and anchor text
            if current_category:
                anchor_text = ""
                url = ""
                full_text = get_paragraph_text(paragraph)
                link_is_bold = False
                link_is_dark_green = False

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
                            link_is_dark_green = is_dark_green_2(ts)

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
                        "link_is_dark_green_2": link_is_dark_green,
                    })
        else:
            if not bullet:
                current_category = None

    return references


def check_link_name_relevance(anchor_text, url, model, page_title=None):
    """Check if a link's anchor text is relevant to the webpage it opens.

    First tries fuzzy matching between anchor text and a pre-fetched page title.
    If the page title is not available or fuzzy match fails, falls back to
    LLM-based semantic judgment using the anchor text and URL.

    Args:
        anchor_text (str): The hyperlink text from the document.
        url (str): The URL the link points to.
        model: LLM model callable for semantic fallback.
        page_title (str|None): Pre-fetched page title, or None.

    Returns:
        bool: True if the anchor text is relevant to the source.
    """
    # Phase 1: Fuzzy match anchor text against page title
    if page_title:
        is_match, _ = fuzzy_match_text(anchor_text, page_title, threshold=50)
        if is_match:
            return True

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
        return "yes" in response.strip().lower()
    except Exception:
        return False


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


