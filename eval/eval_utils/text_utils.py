from fuzzywuzzy import fuzz
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import sys
sys.path.append("C:/Users/alexg/Documents/GitHub/Agent-Benchmark")
from eval.eval_utils.utils import retrieve_validate_doc_path

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
    result.show()

    return result.export()

def extract_text_location(page_number, ocr_result, )