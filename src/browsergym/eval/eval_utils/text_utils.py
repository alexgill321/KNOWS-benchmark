from rapidfuzz import fuzz, process

# Global cache for DocTR OCR model to avoid reloading
_ocr_model_cache = None
import sys
import os
sys.path.append(os.getcwd())
from src.browsergym.eval.eval_utils.utils import retrieve_validate_doc_path, bbox_ratio_to_location, location
from src.browsergym.eval.eval_utils.text_helpers import *

def text_exact_match_contained(src_text, ref_text):
    """
    Check if any text in text_options is contained as an exact match in text2.

    Args:
        src_text (Union[str, List[str]]): A string or a list of strings to be checked.
        ref_text (str): The reference text.

    Returns:
        string: The matched text if found, otherwise None.
    """
    if isinstance(src_text, str):
        if preprocess_text(src_text) in preprocess_text(ref_text):
            return src_text
    elif isinstance(src_text, list):
        for text in src_text:
            if preprocess_text(text) in preprocess_text(ref_text):
                return text
        return None
    
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


    print(f"Query: '{query}'")
    if best_match_tuple and best_match_tuple[0] is not None:
        print(f"Best match found in larger text: '{best_match_tuple[0]}'")
        print(f"Score: {best_match_tuple[1]}%")
        return best_match_tuple[0]
    else:
        print("No suitable match found.")
        return None

def binary_judge_text(model, src_text, gld_text):
    """Classifies a text based on its presence in another text using a pre-trained LLM.

    Args:
        model (str): The model to use for classification. Model should be loaded beforehand.
        src_text (str): The text to be classified.
        gld_text (str): The reference text.

    Returns:
        bool: True if text1 is deemed to be present in text2, False otherwise.
    """

    messages = [
        {
            "role": "system",
            "content": 
            [{
                "type": "text", 
                "text": "Given text 1 and a reference text, determine whether there is a semantically similar match for text 1 in the "
                "reference text. If there is, return True, otherwise return False."
            }]
        },
        {
            "role": "user",
            "content": 
            [{
                "type": "text", 
                "text": f"Text 1: {src_text}\nReference Text: {gld_text}"
            }]
        }
    ]

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
            if line_text.lower() == text_lower:
                print(f"Found exact line match on page {page_num}: '{line_text}'")
                return line['location']
            
            # Check if text is within a line
            if text_lower in line_text.lower():
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