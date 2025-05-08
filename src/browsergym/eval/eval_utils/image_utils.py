import sys
import os
sys.path.append(os.getcwd())
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
from src.browsergym.eval.eval_utils.image_helpers import *
from src.browsergym.eval.eval_utils.utils import location, retrieve_validate_doc_path
import torch

def convert_pdf_to_pngs(pdf_path, output_dir, dpi=300):
    """Converts each page of a PDF file to a PNG image.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir (str): The directory where PNG images will be saved.
        dpi (int): The DPI (dots per inch) for the output images. Default is 300.

    Returns:
        int: The number of pages converted to PNG images. If an error occurs, returns None.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        
        num_pages = len(doc)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # number of page
            # Increase DPI for higher resolution PNGs
            # Default DPI is 72. 300 is good for general quality.
            zoom = dpi / 72  # Calculate zoom factor based on desired DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            output_filename = os.path.join(output_dir, f"page_{page_num + 1:03d}.png")
            pix.save(output_filename)

        doc.close()
        return num_pages

    except Exception as e:
        print(f"An error occurred during PDF to PNG conversion: {e}")
        return None
    
def binary_judge_image(model, image_path, text):
    """Classifies an image (or a folder of images) based on the provided text using a pre-trained model.
    
    If a folder is provided, it will check each image until one passes (returns True) or all fail.

    Args:
        model: The pre-trained model to use. Model should be loaded beforehand.
        image_path (str): The path to the image file or a folder containing images.
        text (str): The text to classify the image(s) against.

    Returns:
        string: The path of the image that passed the classification, or None if no image passed.

    Raises:
        FileNotFoundError: If the provided image path does not exist or contains no valid images.
    """
    # Get a list of image paths to process
    image_paths = []
    
    # Process image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Path does not exist: {image_path}")
    
    if os.path.isdir(image_path):
        # Get all image files in the directory
        for filename in os.listdir(image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(image_path, filename))
        print(f"Found {len(image_paths)} images in {image_path}")
    else:
        # Single image file
        image_paths.append(image_path)
    
    # Check if we have images to process
    if not image_paths:
        raise FileNotFoundError(f"No valid images found in {image_path}")
    
    # Process each image until one passes or all fail
    for img_path in image_paths:
        try:
            print(f"Processing image: {os.path.basename(img_path)}")
            image = Image.open(img_path)
            image = image.convert("RGB")
            
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Given an image and a text which will ask a question about the image, answer the question with either \"Yes\" or \"No\". If the question is not answerable with \"Yes\" or \"No\", respond with \"I don't know\"."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                },
            ]
            
            response = model(messages)
            result = parse_response(response)
            
            if result is True:
                print(f"Image {os.path.basename(img_path)} passed the classification")
                return img_path  # Return the path of the image that passed
            else:
                print(f"Image {os.path.basename(img_path)} did not pass the classification")
                
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    
    # If we get here, no image passed
    print("No images passed the classification")
    return None

def extract_text_from_pdf(doc_path):
    """Extracts text items from screenshots of a PDF document using Omniparser.

    Args:
        doc_path (str): The path to the folder where images of the pdf are stored.

    Returns:
        dict: A dictionary containing dictionaries with the text items and their locations in the document. 
            The dictionary has entries for each page of the document. Only items with type "text" are included.
    """
    image_paths = retrieve_validate_doc_path(doc_path)
    
    # Try to import OmniParser modules
    try:
        # Add path for OmniParser if needed
        sys_path_added = False
        if "OmniParser" not in ' '.join(sys.path):
            omniparser_possible_paths = [
                os.path.expanduser("~/Documents/GitHub/OmniParser"),
                "C:/Users/alexg/Documents/GitHub/OmniParser"  # From process_screenshot_omniparser.py
            ]
            
            for path in omniparser_possible_paths:
                if os.path.exists(path):
                    sys.path.append(path)
                    sys_path_added = True
                    print(f"Added OmniParser path: {path}")
                    break
        
        if not sys_path_added:
            print("Warning: Could not find OmniParser path. Attempting to import anyway.")
        
        # Import OmniParser utilities
        from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
    except ImportError as e:
        print(f"Error importing OmniParser utilities: {e}")
        print("Please ensure OmniParser is installed and accessible.")
        return None
    
    # Check for device availability (CPU/GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set model paths based on process_screenshot_omniparser.py
    model_path = "C:/Users/alexg/Documents/GitHub/OmniParser/weights/icon_detect/model.pt"
    caption_path = "C:/Users/alexg/Documents/GitHub/OmniParser/weights/icon_caption_florence"
    
    # Load models
    try:
        # Load SOM model for object detection
        som_model = get_yolo_model(model_path)
        som_model.to(device)
        
        # Get caption model processor
        caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path=caption_path, 
            device=device
        )
    except Exception as e:
        print(f"Error loading OmniParser models: {e}")
        return None
    
    # Process each page and extract text
    result = {}
    
    for page_index, image_path in enumerate(image_paths):
        page_number = page_index + 1  # 1-indexed page numbers
        print(f"Processing page {page_number}: {os.path.basename(image_path)}")
        
        try:
            # Get image size for converting bbox ratios to pixel coordinates
            image = Image.open(image_path)
            image_width, image_height = image.size
            print(f"Image size: {image_width}x{image_height}")
            
            # Configure bbox visualization (for debugging)
            box_overlay_ratio = max(image_width, image_height) / 3200
            draw_bbox_config = {
                'text_scale': 0.8 * box_overlay_ratio,
                'text_thickness': max(int(2 * box_overlay_ratio), 1),
                'text_padding': max(int(3 * box_overlay_ratio), 1),
                'thickness': max(int(3 * box_overlay_ratio), 1),
            }
            
            # Detection threshold
            BOX_THRESHOLD = 0.05
            
            # Run OCR to get text and bounding boxes
            print("Running OCR...")
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image_path, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
                use_paddleocr=True
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            # Run semantic analysis
            print("Running semantic analysis...")
            _, label_coordinates, parsed_content_list = get_som_labeled_img(
                image_path, 
                som_model, 
                BOX_TRESHOLD=BOX_THRESHOLD, 
                output_coord_in_ratio=True,  # Coordinates as ratio of image size
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config, 
                caption_model_processor=caption_model_processor, 
                ocr_text=text,
                use_local_semantics=True, 
                iou_threshold=0.7, 
                scale_img=False, 
                batch_size=128
            )
            
            # Filter for text items only and convert to location objects
            text_items = {}
            for item in parsed_content_list:
                if item.get('type') == 'text':
                    bbox = item.get('bbox', [0, 0, 0, 0])  # [x1, y1, x2, y2] as ratios
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        
                        # Convert ratios to pixel coordinates
                        x_px = x1 * image_width
                        y_px = y1 * image_height
                        width_px = (x2 - x1) * image_width
                        height_px = (y2 - y1) * image_height
                        
                        # Create location object
                        loc = location(
                            page_number=page_number,
                            x=x_px,
                            y=y_px,
                            width=width_px,
                            height=height_px
                        )
                        
                        # Get item ID (use original ID if available, otherwise generate one)
                        item_id = item.get('ID', len(text_items))
                        
                        # Store the text item with its location
                        text_items[item_id] = {
                            'text': item.get('content', ''),
                            'location': loc,
                            'bbox_ratio': bbox  # Keep original ratio coordinates
                        }
            
            # Add text items to result dictionary
            result[page_number] = text_items
            print(f"Found {len(text_items)} text items on page {page_number}")
            
        except Exception as e:
            print(f"Error processing page {page_number}: {e}")
            import traceback
            traceback.print_exc()
    
    if not result:
        print("No text items were extracted from any pages.")
        return None
    
    print(f"Text extraction complete. Extracted text from {len(result)} pages.")
    return result

def extract_image_location(image_path, doc_path):
    """Extracts the location of an image in a document from jpg images of a PDF doc/slide/sheet.

    Args:
        image_path (str): The path to the image file.
        doc_path (str): The path to the folder where images of the pdf are stored. This should be done before running this method.

    Returns:
        location: A location object of the bounding boc of the image in the document. 
        Returns None if not found.

    Raises:
        FileNotFoundError: If the provided image path does not exist or the document path is invalid.
    """
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    doc_images = retrieve_validate_doc_path(doc_path)
    
    # print(f"Looking for {image_path} in document with {len(doc_images)} pages")
    
    # Parameters for the image search
    match_threshold = 0.8
    min_source_fraction = 1/8
    max_source_fraction = 0.8
    scale_steps = 100
    
    # Try to find the image in each page of the document
    for page_index, doc_image_path in enumerate(doc_images):
        # print(f"Searching page {page_index + 1} of {len(doc_images)}: {os.path.basename(doc_image_path)}")
        
        location_info = find_template_scale_invariant(
            doc_image_path,          # The source image (document page)
            image_path,              # The template image to find
            min_source_fraction=min_source_fraction,
            max_source_fraction=max_source_fraction,
            num_scale_steps=scale_steps,
            threshold=match_threshold,
            visualize_final=False,
            visualize_steps=False,
            verbose=False            # Change to True for more debugging information
        )
        
        if location_info:
            # Found a match! Extract the coordinates
            top_left_x, top_left_y, bottom_right_x, bottom_right_y, found_scale = location_info

            width = bottom_right_x - top_left_x
            height = bottom_right_y - top_left_y
            
            # print(f"Found match on page {page_index + 1}!")
            # print(f"Coordinates: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
            # print(f"Size: {width}x{height} at scale {found_scale:.3f}")
            
            # Create and return a location object
            return location(
                page_number=page_index + 1,  # 1-indexed page number
                x=top_left_x,
                y=top_left_y,
                width=width,
                height=height
            )
    
    # Image not found in any page
    # print("Image not found in any page of the document.")
    return None
    
def image_exact_match(src_image_path, gld_image_path):
    """Performs an exact pixel-by-pixel comparison between images.
    
    Both parameters can be single image paths or folder paths containing multiple images. 
    
    Args:
        src_image_path (str): Path to the first image file or folder containing images.
        gld_image_path (str): Path to the second image file or folder containing images.
        
    Returns:
        string: The path of the image that matched, or None if no match was found.

    Raises:
        FileNotFoundError: If the provided image paths do not exist or contain no valid images.
    """
    # Get lists of image paths
    src_paths = []
    gld_paths = []
    
    # Process image1_path
    if not os.path.exists(src_image_path):
        raise FileNotFoundError(f"Path does not exist: {src_image_path}")
    
    if os.path.isdir(src_image_path):
        # Get all image files in the directory
        for filename in os.listdir(src_image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                src_paths.append(os.path.join(src_image_path, filename))
    else:
        # Single image file
        src_paths.append(src_image_path)
    
    # Process image2_path
    if not os.path.exists(gld_image_path):
        raise FileNotFoundError(f"Path does not exist: {gld_image_path}")
    
    if os.path.isdir(gld_image_path):
        for filename in os.listdir(gld_image_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                gld_paths.append(os.path.join(gld_image_path, filename))
    else:
        # Single image file
        gld_paths.append(gld_image_path)
    
    # Check if we have images to compare
    if not src_paths:
        raise FileNotFoundError(f"No valid images found in {src_image_path}")
    
    if not gld_paths:
        raise FileNotFoundError(f"No valid images found in {gld_image_path}")
    
    # Compare each image from set 1 with each image from set 2
    for src_path in src_paths:
        try:
            img1 = cv2.imread(src_path)
            if img1 is None:
                print(f"Warning: Could not read image: {src_path}")
                continue
                
            for gld_path in gld_paths:
                try:
                    img2 = cv2.imread(gld_path)
                    if img2 is None:
                        print(f"Warning: Could not read image: {gld_path}")
                        continue
                    
                    # Check dimensions match
                    if img1.shape != img2.shape:
                        continue  # Skip to next comparison if dimensions don't match
                    
                    # Calculate absolute difference between images
                    difference = cv2.absdiff(img1, img2)
                    
                    # If all pixel differences are zero, we have a match
                    if np.count_nonzero(difference) == 0:
                        print(f"Match found between {src_path.split('/')[-1]} and {gld_path.split('/')[-1]}")
                        return src_path
                        
                except Exception as e:
                    print(f"Error comparing with {gld_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error loading {src_path}: {e}")
            continue
    
    # No matches found
    print("No exact matches found between the images")
    return None