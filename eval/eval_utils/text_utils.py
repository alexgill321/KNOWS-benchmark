from fuzzywuzzy import fuzz
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import sys
sys.path.append("C:/Users/alexg/Documents/GitHub/Agent-Benchmark")
from eval.eval_utils.utils import retrieve_validate_doc_path, bbox_ratio_to_location, location

def text_exact_match_contained(text1, text2):
    """
    Check if text1 is contained in text2.
    """
    return text1 in text2

def text_fuzzy_match_contained(text1, text2):
    """
    Check if text1 is contained in text2 with fuzzy matching.
    """
    return fuzz.partial_ratio(text1, text2) >= 85

def binary_judge_text(model, text1, text2):
    """
    Classifies a text based on its presence in another text using a pre-trained LLM.

    Args:
        model (str): The model to use for classification. Model should be loaded beforehand.
        text1 (str): The text to be classified.
        text2 (str): The reference text.

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
                "text": f"Text 1: {text1}\nReference Text: {text2}"
            }]
        }
    ]

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file OCR via doctr.

    Args:
        pdf_path (str): Path to the PDF file.

    Outputs:
        str: Extracted text from the PDF.
    """
    image_paths = retrieve_validate_doc_path(pdf_path)
    doc = DocumentFile.from_images(image_paths)
    model = ocr_predictor(pretrained=True)
    result = model(doc)

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