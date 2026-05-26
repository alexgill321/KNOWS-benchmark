import unicodedata
import warnings
from typing import List, Optional, Union, Any
from rapidfuzz import fuzz, process

# Global cache for DocTR OCR model to avoid reloading
_ocr_model_cache = None
import sys
import os
import re
sys.path.append(os.getcwd())
from src.browsergym.knows.eval.eval_utils.utils import retrieve_validate_doc_path, bbox_ratio_to_location, location
from src.browsergym.knows.eval.eval_utils.text_helpers import *


# =============================================================================
# Text Normalization
# =============================================================================

def _normalize_text(text: str, *,
                    lowercase: bool = True,
                    strip: bool = True,
                    collapse_whitespace: bool = True,
                    replace_nbsp: bool = True,
                    replace_smart_quotes: bool = True) -> str:
    """Unified text normalization for matching.

    Consolidates normalization logic from multiple sources into a single utility.

    Args:
        text: The text to normalize.
        lowercase: Convert to lowercase. Default True.
        strip: Strip leading/trailing whitespace. Default True.
        collapse_whitespace: Collapse multiple spaces to single space. Default True.
        replace_nbsp: Replace non-breaking spaces with regular spaces. Default True.
        replace_smart_quotes: Replace various unicode double-quote characters (“, ”, «, », etc.)
                              with standard ASCII double-quote '"'. Default True.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    if replace_nbsp:
        text = text.replace("\u00a0", " ")

    if replace_smart_quotes:
        smart_double_quotes = "\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033\u2036"
        text = re.sub(f"[{smart_double_quotes}]", '"', text)

    if strip:
        text = text.strip()

    if lowercase:
        text = text.lower()

    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)

    return text


# =============================================================================
# Keyword Exact Matching
# =============================================================================

def keyword_exact_match(text: str, keyword: str, *,
                        case_sensitive: bool = False,
                        normalize: bool = True,
                        standalone_line: bool = False,
                        substring: bool = False) -> bool:
    """Strict exact match of a keyword to text.

    Performs exact comparison after optional normalization. This is the core
    matching function for keyword-based text matching.

    Normalization (if enabled):
    - Lowercase (unless case_sensitive)
    - Strip whitespace
    - Replace non-breaking spaces
    - Collapse multiple whitespace

    Args:
        text: The text to check.
        keyword: The keyword to match against.
        case_sensitive: If True, perform case-sensitive matching. Default False.
        normalize: If True, normalize both text and keyword before comparison. Default True.
        standalone_line: If True, check line-by-line with trailing punctuation
                        tolerance (.,;:). Useful for header matching where
                        "Name." should match "Name". Default False.
        substring: If True, check if keyword is contained within text (substring match).
                   Default False (exact match).

    Returns:
        True if text exactly matches keyword after normalization (or contains it if substring=True).

    Examples:
        >>> keyword_exact_match("Hello World", "hello world")
        True
        >>> keyword_exact_match("Hello World", "hello", case_sensitive=True)
        False
        >>> keyword_exact_match("Name.", "Name", standalone_line=True)
        True
        >>> keyword_exact_match("Avg Male 25 Speed", "male", substring=True)
        True
    """
    if not text or not keyword:
        return False

    if normalize:
        text_norm = _normalize_text(text, lowercase=not case_sensitive)
        keyword_norm = _normalize_text(keyword, lowercase=not case_sensitive)
    else:
        text_norm = text.strip() if not case_sensitive else text.strip()
        keyword_norm = keyword.strip() if not case_sensitive else keyword.strip()
        if not case_sensitive:
            text_norm = text_norm.lower()
            keyword_norm = keyword_norm.lower()

    if not standalone_line:
        if substring:
            return keyword_norm in text_norm
        return text_norm == keyword_norm

    # Standalone line mode: check line-by-line with trailing punctuation tolerance
    allowed_trailing = set(".,;:")

    for line in text.splitlines():
        line_norm = _normalize_text(line, lowercase=not case_sensitive)

        # Exact match
        if line_norm == keyword_norm:
            return True

        # Match with trailing punctuation
        if line_norm.startswith(keyword_norm):
            remainder = line_norm[len(keyword_norm):]
            if all(c in allowed_trailing for c in remainder):
                return True

    return False


def keywords_exact_match(text: str, keywords: List[str], *,
                         case_sensitive: bool = False,
                         normalize: bool = True,
                         standalone_line: bool = False,
                         substring: bool = False) -> Optional[str]:
    """Match text against a list of keywords using exact matching.

    Iterates through keywords and returns the first one that matches the text.
    This is useful when you have multiple acceptable keywords for the same concept.

    Args:
        text: The text to check.
        keywords: List of keyword strings to match against.
        case_sensitive: If True, perform case-sensitive matching. Default False.
        normalize: If True, normalize text and keywords before comparison. Default True.
        standalone_line: If True, check line-by-line with trailing punctuation tolerance.
        substring: If True, check if any keyword is contained within text (substring match).
                   Default False (exact match).

    Returns:
        The first matching keyword, or None if no match.

    Examples:
        >>> keywords_exact_match("price", ["cost", "price", "amount"])
        'price'
        >>> keywords_exact_match("TICKER", ["symbol", "ticker"])
        'ticker'
        >>> keywords_exact_match("Avg Male 25 Speed", ["male", "5k"], substring=True)
        'male'
    """
    if not text or not keywords:
        return None

    for keyword in keywords:
        if keyword_exact_match(text, keyword,
                               case_sensitive=case_sensitive,
                               normalize=normalize,
                               standalone_line=standalone_line,
                               substring=substring):
            return keyword

    return None


# =============================================================================
# LLM-Based Matching
# =============================================================================

def keywords_llm_match(texts: Union[str, List[str]],
                       keywords: List[str],
                       model: Any,
                       description: str = None,
                       context: str = None) -> Optional[str]:
    """Use LLM to find semantic match between texts and keywords.

    Makes a single LLM call that evaluates all texts at once against the keywords.
    The prompt presents all candidate texts as a numbered list and asks the LLM
    to identify which one (if any) matches the keywords/description.

    Args:
        texts: Single text or list of candidate texts (e.g., column headers).
        keywords: List of keywords describing what we're looking for.
        model: LLM model callable that accepts messages list.
        description: Optional description of what we're matching (e.g., "stock symbol column").
        context: Optional task context to help the LLM make better matching decisions
            (e.g., "an apartment listing spreadsheet with columns for property details").

    Returns:
        The matching text from the texts list, or None if no match.

    Examples:
        >>> keywords_llm_match(
        ...     ["Stock Symbol", "Share Price", "Volume"],
        ...     ["ticker", "symbol"],
        ...     model,
        ...     description="column for stock ticker"
        ... )
        'Stock Symbol'
    """
    if not texts or not keywords or model is None:
        return None

    # Normalize to list
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return None

    # Build numbered list of texts
    texts_numbered = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])

    # Build prompt
    keywords_hint = f"Example keywords that might match: {', '.join(keywords)}"
    desc_text = f"'{description}'" if description else "the specified criteria"
    context_text = f"\nContext: {context}" if context else ""

    prompt = f"""You are analyzing text values to find one that matches specific criteria.

Criteria: Find the text that represents {desc_text}.
{keywords_hint}{context_text}

Available texts:
{texts_numbered}

Please analyze each text and determine which one matches the criteria. Consider:
- Exact matches
- Synonyms and semantically similar terms
- Common abbreviations
- Naming conventions

If none of the texts are a reasonable semantic match for the criteria, you MUST respond "NONE". Do not force a match — only return a number if the text is clearly related to the criteria.

Respond with ONLY the number (1, 2, 3, etc.) of the matching text, or "NONE" if no text adequately matches."""

    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        response = model(messages).strip().upper()

        if response == "NONE":
            return None

        idx = int(response) - 1
        if 0 <= idx < len(texts):
            return texts[idx]
    except Exception as e:
        print(f"Error in LLM text matching: {e}")

    return None


def keywords_match_robust(texts: Union[str, List[str]],
                          keywords: Union[str, List[str]],
                          model: Any = None,
                          description: str = None,
                          substring: bool = False,
                          unlimited_texts: bool = False) -> Optional[str]:
    """Robust matching: exact match first, then LLM fallback.

    This is the primary entry point for keyword-based text matching. It uses
    a two-phase approach:

    Phase 1: Try exact keyword matching against all texts (fast, deterministic)
    Phase 2: If no exact match and model provided, use single LLM call for semantic matching

    Args:
        texts: Single text or list of candidate texts to search.
        keywords: Single keyword or list of keywords to match against.
        model: Optional LLM model for fallback. If None, only exact matching is used.
        description: Optional description for LLM context (e.g., "stock symbol column").
        substring: If True, check if any keyword is contained within text (substring match).
                   Default False (exact match). Useful for legend labels where "Avg Male 25"
                   should match keyword "male".

    Returns:
        The matching text, or None if no match found.

    Examples:
        >>> keywords_match_robust(
        ...     ["Stock Symbol", "Share Price", "Volume"],
        ...     ["symbol", "ticker"],
        ...     model=my_model,
        ...     description="stock ticker column"
        ... )
        'Stock Symbol'
        >>> keywords_match_robust(
        ...     "Avg Male 25 Speed (min/mile)",
        ...     ["male", "5k", "25"],
        ...     substring=True
        ... )
        'Avg Male 25 Speed (min/mile)'
    """
    if not texts or not keywords:
        return None

    # Normalize inputs to lists
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(keywords, str):
        keywords = [keywords]

    # Phase 1: Exact keyword matching (or substring matching if enabled)
    for text in texts:
        matched_keyword = keywords_exact_match(text, keywords, substring=substring)
        if matched_keyword:
            return text

    # Phase 2: LLM fallback (if model provided)
    if model is not None and (unlimited_texts or (len(texts) < 30 and len(keywords) < 30)):
        return keywords_llm_match(texts, keywords, model, description)

    return None


# =============================================================================
# Legacy Functions (Deprecated)
# =============================================================================

def text_exact_match_contained(src_text: str, ref_text: str, *, standalone_line: bool = False) -> bool:
    """Return True if *target* is found in *text* with an 'exact' notion.

    .. deprecated::
        Use `keyword_exact_match()` for exact matching or `keywords_exact_match()`
        for matching against multiple keywords. This function is kept for backwards
        compatibility but will be removed in a future version.

    - If standalone_line=False: simple containment check (target in text).
    - If standalone_line=True: require an exact full-line match after normalization
      (case-insensitive, whitespace-collapsed). Only trailing punctuation that typically
      ends a standalone item (period, comma, colon, semicolon) is tolerated.
      Opening brackets/parens attached to the text indicate concatenation and are rejected.
      This is useful for header fields like a name.
    """
    warnings.warn(
        "text_exact_match_contained() is deprecated. Use keyword_exact_match() or "
        "keywords_exact_match() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if not src_text or not ref_text:
        return False

    if not standalone_line:
        return src_text in ref_text

    # Use the new keyword_exact_match for standalone_line mode
    return keyword_exact_match(ref_text, src_text, standalone_line=True)
    
def text_fuzzy_match_contained_long(target, full_text, threshold=85):
    """
    Uses rapidfuzz for faster fuzzy matching.
    """
    target_words = target.split()
    full_words = full_text.split()
    
    # Use a window slightly larger than the target to accommodate inserted text/noise
    # partial_ratio will find the best match WITHIN this window
    window_size = int(len(target_words) * 1.2) + 10
    
    best_score = 0
    best_match = None
    
    # Optimization: Step size can be larger because we use a larger window and partial_ratio
    # If window is 120% of target, stepping by 10% ensures we cover all potential full matches
    step_size = max(1, int(len(target_words) * 0.1))
    
    for i in range(0, max(1, len(full_words) - len(target_words)), step_size):
        # Define window bounds
        end_idx = min(i + window_size, len(full_words))
        window_words = full_words[i:end_idx]
        if not window_words:
            break
            
        window = ' '.join(window_words)
        
        # Use partial_ratio as originally intended for substring matching
        score = fuzz.partial_ratio(target, window)
        
        if score > best_score:
            best_score = score
            best_match = window
            
            # Early exit for perfect match
            if score == 100:
                break
    
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score

def text_fuzzy_match_contained_short(query, larger_text):
    """Check if text1 is contained in text2 with fuzzy matching.

    USE THIS ONE FOR SHORT TEXTS (e.g., a sentence or less).

    Uses a sliding window approach with fuzzy string matching to find text1 within text2,
    even when there are slight variations in spelling or formatting.

    Args:
        text1 (str): The text to search for.
        text2 (str): The text to search within.

    Returns:
        str: The best matching substring found in text2, or None if no match is found.
    """
    query_len = len(query)
    best_match_tuple = (None, 0) # (matching_substring, score)

    # --- Simple Sliding Window (Character-based) ---
    # Define a window size slightly larger than the query to allow for variations
    # You might need to tune this window_size_factor
    window_size_factor = 1.1
    window_size = max(query_len, int(query_len * window_size_factor))
    step_size = 1 # Move window one character at a time for max granularity

    chunks = []
    for i in range(0, len(larger_text) - window_size + 1, step_size):
        chunk = larger_text[i : i + window_size]
        chunks.append(chunk)

    # --- Using process.extractOne ---
    # Find the best match from the generated chunks
    # You can choose different scorers: fuzz.ratio, fuzz.WRatio (often good), etc.
    if chunks: # Ensure there are chunks to process
        # extractOne returns (choice, score, index) if choices is a dict,
        # or (choice, score) if choices is a list/iterable
        best_match_tuple = process.extractOne(
            query,
            chunks,
            scorer=fuzz.WRatio, # WRatio often handles variations well
            score_cutoff=85
            # You could add score_cutoff=80 (or other value) to ignore poor matches
        )
    else:
        print("No chunks generated to compare.")


    #print(f"Query: '{query}'")
    if best_match_tuple and best_match_tuple[0] is not None:
        print(f"Best match found in larger text: '{best_match_tuple[0]}'")
        print(f"Score: {best_match_tuple[1]}%")
        return best_match_tuple[0]
    else:
        #print("No suitable match found.")
        return None

def match_text_in_list(text, text_list, threshold=80):
    """
    Find the best matching item from a list in the given text.

    Uses fuzzy matching to find which item from a predefined list best matches
    the given text, handling minor variations in spelling, formatting, or extra text.

    Args:
        text (str): The text to search within.
        text_list (list[str]): List of valid text items to match against.
        threshold (int): Minimum similarity score (0-100) to consider a match. Default is 80.

    Returns:
        tuple: (matched_item, score) where matched_item is the item from the list that best
               matches (or None if no match above threshold), and score is the match quality (0-100).

    Examples:
        >>> match_text_in_list("Darrow O'Lycos", ["Darrow", "Sevro", "Mustang"])
        ("Darrow", 90)

        >>> match_text_in_list("Character: Sevro au Barca", ["Darrow", "Sevro", "Mustang"])
        ("Sevro", 85)
    """
    if not text or not text_list:
        return None, 0

    # Use rapidfuzz's process.extractOne to find the best match
    result = process.extractOne(
        text,
        text_list,
        scorer=fuzz.partial_ratio,  # partial_ratio works well for finding list items within larger text
        score_cutoff=threshold
    )

    if result:
        matched_item, score = result[0], result[1]
        print(f"Matched '{matched_item}' in text '{text}' with score {score}")
        return matched_item, score

    return None, 0

def extract_text_from_pdf(pdf_images_path):
    """
    Extract text from a PDF file OCR via doctr.

    Args:
        pdf_images_path (str): Path to the PDF file images.

    Outputs:
        str: Extracted text from the PDF.
    """
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    global _ocr_model_cache

    image_paths = retrieve_validate_doc_path(pdf_images_path)
    doc = DocumentFile.from_images(image_paths)

    # Use cached OCR model or create new one
    if _ocr_model_cache is None:
        print("Loading DocTR OCR model for the first time...")
        _ocr_model_cache = ocr_predictor(pretrained=True)
        print("DocTR OCR model loaded and cached!")

    result = _ocr_model_cache(doc)
    result_json = result.export()
    formatted_results = {}
    for page in result_json["pages"]:
        formatted_results[page["page_idx"]] = []
        image_width = doc[page["page_idx"]].shape[1]
        image_height = doc[page["page_idx"]].shape[0]
        for block in page["blocks"]:
            for line in block["lines"]:
                words = []
                for word in line["words"]:
                    word = {
                        "value": word["value"],
                        "location": bbox_ratio_to_location(
                            [word["geometry"][0][0], word["geometry"][0][1], word["geometry"][1][0], word["geometry"][1][1]], 
                            page["page_idx"], 
                            image_width, 
                            image_height
                        ),
                    }
                    words.append(word)
                line = {
                    "text": "".join(word["value"] + " " for word in line["words"]),
                    "location": bbox_ratio_to_location(
                        [line["geometry"][0][0], line["geometry"][0][1], line["geometry"][1][0], line["geometry"][1][1]], 
                        page["page_idx"], 
                        image_width, 
                        image_height
                    ),
                    "words": words                
                }
                formatted_results[page["page_idx"]].append(line)
    return formatted_results


def extract_text_location(ocr_result, text_to_find):
    """
    Extract the location of a specific text in the OCR result from extract_text_from_pdf.
    Will return the bounding box coordinates around the located text, which could span
    across a single line or multiple lines.

    Args:
        ocr_result (dict): The OCR result from extract_text_from_pdf.
        text_to_find (str): The text to find in the OCR results.

    Returns:
        location: A bounding box object with the coordinates of the located text.
                 Returns None if the text is not found.
    """
    
    if not ocr_result or not text_to_find:
        print("Empty OCR results or search text.")
        return None
    
    text_to_find = text_to_find.strip()
    text_lower = text_to_find.lower()
    
    # Case 1: Check for exact matches in single lines
    for page_num, page_lines in ocr_result.items():
        for line in page_lines:
            line_text = line.get('text', '').strip()
            
            # Check for exact line match
            if preprocess_text(line_text.lower()) == preprocess_text(text_lower):
                print(f"Found exact line match on page {page_num}: '{line_text}'")
                return line['location']
            
            # Check if text is within a line
            if preprocess_text(text_lower) in preprocess_text(line_text.lower()):
                # Calculate approximate position within the line
                # Based on character position in the line
                char_width = line['location'].width / len(line_text)
                start_idx = line_text.lower().find(text_lower)
                end_idx = start_idx + len(text_lower)
                
                # Estimate x position and width based on character indices
                x_start = line['location'].x + (start_idx * char_width)
                text_width = (end_idx - start_idx) * char_width
                
                print(f"Found text within line on page {page_num}: '{text_to_find}'")
                return location(
                    page_number=line['location'].page_number,
                    x=x_start,
                    y=line['location'].y,
                    width=text_width,
                    height=line['location'].height
                )
    
    # Case 2: Look for text spread across words in a single line
    for page_num, page_lines in ocr_result.items():
        for line in page_lines:
            # Check if line contains words that might form our search text
            words = line.get('words', [])
            
            for i in range(len(words)):
                # Try to build up text from consecutive words
                current_text = ""
                word_locations = []
                
                for j in range(i, len(words)):
                    word = words[j]
                    current_text += word['value']
                    word_locations.append(word['location'])
                    
                    # Check if we've found our text
                    if text_lower in current_text.lower():
                        # Found a match across multiple words within a line
                        print(f"Found text across words in line on page {page_num}: '{text_to_find}'")
                        
                        # Merge all word locations
                        merged_loc = location.merge_locations(word_locations)
                        return merged_loc
    
    # Case 3: Look for text spread across multiple lines
    # Group lines by page for multi-line matching
    for page_num, page_lines in ocr_result.items():
        # Sort lines by vertical position (y-coordinate)
        sorted_lines = sorted(page_lines, key=lambda line: line['location'].y)
        
        # Try combinations of consecutive lines
        for i in range(len(sorted_lines)):
            combined_text = ""
            line_locations = []
            
            for j in range(i, min(i + 10, len(sorted_lines))):  # Limit to 10 consecutive lines max
                line = sorted_lines[j]
                combined_text += " " + line['text']
                line_locations.append(line['location'])
                
                # Check if combined text contains our search text
                if text_lower in combined_text.lower():
                    print(f"Found text across multiple lines on page {page_num}: '{text_to_find}'")
                    
                    # Create a merged location from all relevant line locations
                    merged_loc = location.merge_locations(line_locations)
                    return merged_loc
    
    # Text not found
    print(f"Text not found in OCR results: '{text_to_find}'")
    return None

def numerical_match_with_error(value1, value2, error_percent=5.0):
    """
    Match numerical values with a given error bound in percentage.

    Can compare either:
    - Two single numerical values
    - Two lists of numerical values (element-wise comparison)

    Args:
        value1: A single number or list of numbers (reference/gold values)
        value2: A single number or list of numbers (actual values to check)
        error_percent (float): Allowed percentage error. Default is 5.0%

    Returns:
        If comparing single values: tuple (bool, float)
            - bool: True if values match within error bound
            - float: Actual percentage difference
        If comparing lists: tuple (list[bool], list[float], float)
            - list[bool]: Match result for each pair
            - list[float]: Percentage difference for each pair
            - float: Overall match rate (percentage of pairs that matched)

    Examples:
        >>> numerical_match_with_error(100, 103, error_percent=5.0)
        (True, 3.0)

        >>> numerical_match_with_error([100, 200], [103, 210], error_percent=5.0)
        ([True, False], [3.0, 5.0], 50.0)
    """
    def compare_single(ref, actual, threshold):
        """Compare two single values."""
        try:
            ref = float(ref)
            actual = float(actual)

            if ref == 0:
                # Special case: if reference is 0, check if actual is also 0
                return (actual == 0, 0.0 if actual == 0 else float('inf'))

            # Calculate percentage difference relative to reference
            diff = abs(actual - ref)
            percent_diff = (diff / abs(ref)) * 100.0

            is_match = percent_diff <= threshold
            return (is_match, percent_diff)

        except (ValueError, TypeError) as e:
            print(f"Error comparing values {ref} and {actual}: {e}")
            return (False, float('inf'))

    # Handle single value comparison
    if not isinstance(value1, (list, tuple)) and not isinstance(value2, (list, tuple)):
        return compare_single(value1, value2, error_percent)

    # Handle list comparison
    if isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
        if len(value1) != len(value2):
            print(f"Warning: List lengths don't match ({len(value1)} vs {len(value2)})")
            # Pad shorter list with None or truncate
            max_len = max(len(value1), len(value2))
            value1 = list(value1) + [None] * (max_len - len(value1))
            value2 = list(value2) + [None] * (max_len - len(value2))

        results = []
        diffs = []

        for v1, v2 in zip(value1, value2):
            if v1 is None or v2 is None:
                results.append(False)
                diffs.append(float('inf'))
            else:
                match, diff = compare_single(v1, v2, error_percent)
                results.append(match)
                diffs.append(diff)

        # Calculate overall match rate
        match_count = sum(results)
        match_rate = (match_count / len(results)) * 100.0 if results else 0.0

        return (results, diffs, match_rate)

    # Mismatched types (one is list, one is not)
    raise TypeError("Both values must be either single numbers or lists of numbers")
def get_smallest_x_position(text_ocr):
    """
    Loop through all the lines in the text and return the smallest x postion for checking alignment. 
    """
    smallest_x = float('inf')
    for page_num, page_line in text_ocr.items(): 
        for line in page_line:
            if line['location'].x < smallest_x:
                smallest_x = line['location'].x
    return smallest_x

def strip_label_prefix(text):
    """Strip common label prefixes like 'Email:', 'GitHub:', 'LinkedIn:' from text lines.

    Args:
        text (str): Text potentially containing label prefixes.

    Returns:
        str: Text with prefixes removed from each line.
    """
    prefixes = ["email:", "github:", "linkedin:", "phone:", "website:", "address:"]
    lines = text.split("\n")
    stripped = []
    for line in lines:
        trimmed = line.strip()
        lower = trimmed.lower()
        for p in prefixes:
            if lower.startswith(p):
                trimmed = trimmed[len(p):].strip()
                break
        stripped.append(trimmed)
    return "\n".join(stripped)


def find_gold_text_location(target_text, doc_structure, text_ocr, threshold=60):
    """Find OCR position of text by matching structural doc lines to OCR lines.

    Uses the Google Docs API structure to find which line contains the target
    text, then fuzzy-matches that line against OCR output to get pixel position.
    Useful when OCR mangles text (e.g. URLs) but the line is still recognizable.

    Args:
        target_text (str): The text to locate (e.g. an email address or URL).
        doc_structure (list): Document structure from extract_structure_from_doc().
        text_ocr (dict): OCR result from extract_text_from_pdf().
        threshold (int): Minimum fuzzy match score (0-100).

    Returns:
        location: Location object from the best-matching OCR line, or None.
    """
    if not target_text or not doc_structure or not text_ocr:
        return None

    # 1. Find the structural line containing target_text
    gold_line = None
    for elem in doc_structure:
        if elem.get("type") == "text" and target_text.lower() in elem.get("content", "").lower():
            gold_line = elem["content"].strip()
            break
    if gold_line is None:
        return None

    # 2. Fuzzy match gold_line against all OCR lines
    best_match = None
    best_score = 0
    for page_num, lines in text_ocr.items():
        for line in lines:
            is_match, score = fuzzy_match_text(gold_line, line["text"], threshold=threshold)
            if score > best_score:
                best_score = score
                best_match = line["location"]

    if best_match and best_score >= threshold:
        print(f"Fuzzy matched gold line '{gold_line[:50]}' to OCR (score={best_score})")
        return best_match
    return None


def fuzzy_match_text(text1: str, text2: str, threshold: int = 80) -> tuple:
    """Perform fuzzy matching between two texts.

    Uses token_sort_ratio for better matching of reordered text.
    Normalizes texts by lowercasing and collapsing whitespace.

    Args:
        text1: First text.
        text2: Second text.
        threshold: Minimum similarity score (0-100).

    Returns:
        Tuple of (is_match, similarity_score).
    """
    if not text1 or not text2:
        return False, 0

    # Normalize texts
    text1 = ' '.join(text1.lower().split())
    text2 = ' '.join(text2.lower().split())

    # Use token_sort_ratio for better matching of reordered text
    score = fuzz.token_sort_ratio(text1, text2)

    return score >= threshold, score

def split_delimited_text(text: str, delimiters: List[str] = None) -> List[str]:
    """Split text by multiple delimiters.

    Useful for parsing comma-separated lists, author lists, or any text
    with multiple possible delimiters.

    Args:
        text: Text to split.
        delimiters: List of delimiter strings to split by.
            Default: [',', '\\n', ' and ']
            Delimiters are applied in order; the text is first split by
            the first delimiter, then each part by the second, etc.

    Returns:
        List of non-empty stripped strings.

    Examples:
        >>> split_delimited_text("Alice, Bob and Charlie")
        ['Alice', 'Bob', 'Charlie']
        >>> split_delimited_text("One\\nTwo\\nThree", delimiters=['\\n'])
        ['One', 'Two', 'Three']
    """
    if not text:
        return []

    if delimiters is None:
        delimiters = [',', '\n', ' and ']

    # Start with the full text as a single item
    parts = [text]

    # Apply each delimiter in sequence
    for delimiter in delimiters:
        new_parts = []
        for part in parts:
            if delimiter == ' and ':
                # Case-insensitive replacement for ' and '
                import re
                split_parts = re.split(r'\s+and\s+', part, flags=re.IGNORECASE)
            else:
                split_parts = part.split(delimiter)
            new_parts.extend(split_parts)
        parts = new_parts

    # Strip whitespace and remove empty strings
    return [p.strip() for p in parts if p.strip()]


# Common name suffixes to remove during normalization
_NAME_SUFFIXES = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' phd', ' md', ' esq']


def normalize_name(name: str, remove_suffixes: bool = True) -> str:
    """Normalize a name for comparison.

    Performs the following normalizations:
    - Converts to lowercase
    - Removes accents/diacritics (e.g., 'é' -> 'e')
    - Normalizes whitespace (collapses multiple spaces)
    - Optionally removes common suffixes (Jr., Sr., III, etc.)

    Args:
        name: The name to normalize.
        remove_suffixes: If True, removes common name suffixes like
            Jr., Sr., III, II, IV, PhD, MD, Esq. Default: True

    Returns:
        Normalized name string.

    Examples:
        >>> normalize_name("José García Jr.")
        'jose garcia'
        >>> normalize_name("André François-Xavier")
        'andre francois-xavier'
        >>> normalize_name("Dr. John Smith III", remove_suffixes=True)
        'dr. john smith'
    """
    if not name:
        return ""

    # Convert to lowercase
    name = name.lower().strip()

    # Remove accents/diacritics using Unicode normalization
    # NFKD decomposes characters into base + combining characters
    # Then we filter out combining characters
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remove common suffixes if requested
    if remove_suffixes:
        for suffix in _NAME_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

    # Normalize whitespace (collapse multiple spaces into one)
    name = ' '.join(name.split())

    return name



