import sys
import os
sys.path.append(os.getcwd())
import fitz  # PyMuPDF
from PIL import Image
import cv2
import numpy as np
import io

# Check if GUI functions are available (not available in headless mode)
# Set to False by default for headless environments
GUI_AVAILABLE = False
from src.browsergym.knows.eval.eval_utils.image_helpers import load_process_images, find_template_scale_invariant, parse_response
from src.browsergym.knows.eval.eval_utils.utils import location, retrieve_validate_doc_path

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
    
def binary_judge_image(model, image_path, text, examples=None):
    """Classifies an image (or a folder of images) based on the provided text using a pre-trained model.

    If a folder is provided, it will check each image until one passes (returns True) or all fail.

    Args:
        model: The pre-trained model to use. Model should be loaded beforehand.
        image_path (str): The path to the image file or a folder containing images.
        text (str): The text to classify the image(s) against.
        examples (str, optional): The path to a folder containing example images to provide as reference.

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
    
    # Prepare example images content if examples folder is provided
    example_content = []
    if examples and os.path.exists(examples) and os.path.isdir(examples):
        print(f"Loading example images from: {examples}")
        example_files = []
        for filename in os.listdir(examples):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                example_files.append(os.path.join(examples, filename))

        print(f"Found {len(example_files)} example images")

        # Add example images to content
        if example_files:
            example_content.append({"type": "text", "text": "Here are example images for reference:"})
            for example_path in example_files:
                try:
                    example_content.append({"type": "image", "image": example_path})
                    example_content.append({"type": "text", "text": f"Example: {os.path.basename(example_path)}"})
                except Exception as e:
                    print(f"Error loading example image {example_path}: {e}")

    # Process each image until one passes or all fail
    for img_path in image_paths:
        try:
            # print(f"Processing image: {os.path.basename(img_path)}")
            # image = Image.open(img_path)
            # image = image.convert("RGB")
            
            # Build user content starting with examples (if any)
            user_content = []
            user_content.extend(example_content)

            # Add the current image and question
            user_content.append({"type": "image", "image": img_path})
            user_content.append({"type": "text", "text": text})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Given an image and a text which will ask a question about the image, answer the question with either \"Yes\" or \"No\". If the question is not answerable with \"Yes\" or \"No\", respond with \"I don't know\". If example images are provided, use them as reference for what you should be looking for."}]
                },
                {
                    "role": "user",
                    "content": user_content
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


def verify_image_in_region(model, pdf_images_dir, gold_image_path, region="upper_left", dpi=300):
    """Verify a gold image is fully visible in a region of the document page using a VLM.

    Crops the specified region from the first page of the rendered PDF, then asks
    a VLM whether the gold image appears fully (not cropped) in that region.

    Args:
        model: The pre-trained VLM model to use.
        pdf_images_dir (str): Directory containing rendered PDF page images (page_001.png, etc.).
        gold_image_path (str): Path to the gold image (e.g. logo) to look for.
        region (str): Region to crop. One of "upper_left", "upper_right", "lower_left", "lower_right".
        dpi (int): DPI used to render the PDF pages. Default 300.

    Returns:
        bool: True if the VLM confirms the image is fully visible in the region.
    """
    import glob

    # Find the first page image
    page_images = sorted(glob.glob(os.path.join(pdf_images_dir, "page_*.png")))
    if not page_images:
        print(f"No page images found in {pdf_images_dir}")
        return False

    page_img = Image.open(page_images[0])
    w, h = page_img.size

    # Define region crop box (left, upper, right, lower)
    regions = {
        "upper_left":  (0, 0, w // 2, h // 3),
        "upper_right": (w // 2, 0, w, h // 3),
        "lower_left":  (0, 2 * h // 3, w // 2, h),
        "lower_right": (w // 2, 2 * h // 3, w, h),
    }

    if region not in regions:
        print(f"Unknown region: {region}")
        return False

    crop_box = regions[region]
    cropped = page_img.crop(crop_box)

    # Save cropped region to a temp file
    crop_path = os.path.join(pdf_images_dir, f"_region_{region}.png")
    cropped.save(crop_path)

    try:
        # Prepare gold image examples
        gold_content = []
        if os.path.isdir(gold_image_path):
            for filename in os.listdir(gold_image_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    gold_content.append({"type": "image", "image": os.path.join(gold_image_path, filename)})
            if gold_content:
                gold_content.insert(0, {"type": "text", "text": "Here is the logo to look for:"})
        else:
            gold_content = [
                {"type": "text", "text": "Here is the logo to look for:"},
                {"type": "image", "image": gold_image_path},
            ]

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are checking whether a specific logo appears in a document screenshot. Answer only 'Yes' or 'No'."}]
            },
            {
                "role": "user",
                "content": gold_content + [
                    {"type": "image", "image": crop_path},
                    {"type": "text", "text": "Is this image fully visible (not cropped or cut off) in this document screenshot? Answer Yes or No."},
                ]
            },
        ]

        response = model(messages)
        result = parse_response(response)
        print(f"VLM logo region check ({region}): {response}")
        return result is True

    except Exception as e:
        print(f"Error in verify_image_in_region: {e}")
        return False
    finally:
        if os.path.exists(crop_path):
            os.remove(crop_path)


def binary_compare_images(model, image1_path, image2_path, mode="same"):
    """Compare two images using a VLM with different comparison modes.

    Args:
        model: The pre-trained VLM model to use. Model should be loaded beforehand.
        image1_path (str): Path to the first image file.
        image2_path (str): Path to the second image file.
        mode (str): Comparison mode. Options:
            - "same": Check if images are the same (default)
            - "similar": Check if images are similar/related
            - "replacement": Check if image2 is a reasonable replacement for image1
            - Custom string: Use as the question text directly

    Returns:
        bool: True if comparison passes, False otherwise.

    Raises:
        FileNotFoundError: If either image path does not exist.
    """
    if not os.path.exists(image1_path):
        raise FileNotFoundError(f"Image path does not exist: {image1_path}")
    if not os.path.exists(image2_path):
        raise FileNotFoundError(f"Image path does not exist: {image2_path}")

    # Define system prompts and questions for different modes
    mode_configs = {
        "same": {
            "system": "You are comparing two images to determine if they are the same image or very similar. Answer 'Yes' if they appear to be the same image (even with minor differences like compression, resizing, or slight color variations). Answer 'No' if they are clearly different images.",
            "question": "Are these the same image or very similar versions of the same image?"
        },
        "similar": {
            "system": "You are comparing two images to determine if they show similar content or subjects. Answer 'Yes' if the images depict the same general subject, scene, or concept, even if they are different photos. Answer 'No' if they show completely different subjects.",
            "question": "Do these images show similar content or the same subject?"
        },
        "replacement": {
            "system": "You are comparing two images to determine if the second image is a reasonable replacement for the first. Answer 'Yes' if the second image retains the key important details and subject matter of the first image, even if the style, quality, or minor details differ. Answer 'No' only if the second image is clearly showing something completely different or unrelated.",
            "question": "Is the second image a reasonable replacement that retains the key details of the first image?"
        }
    }

    # Get config for the mode, or use custom text
    if mode in mode_configs:
        config = mode_configs[mode]
        system_text = config["system"]
        question_text = config["question"]
    else:
        # Custom mode - use the mode string as the question
        system_text = "You are comparing two images. Answer 'Yes' or 'No' based on the question asked."
        question_text = mode

    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1:"},
                    {"type": "image", "image": image1_path},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image", "image": image2_path},
                    {"type": "text", "text": question_text}
                ]
            }
        ]

        response = model(messages)
        result = parse_response(response)
        return result is True

    except Exception as e:
        print(f"Error comparing images: {e}")
        return False


# def extract_text_from_pdf(doc_path):
#     """Extracts text items from screenshots of a PDF document using Omniparser.

#     Args:
#         doc_path (str): The path to the folder where images of the pdf are stored.

#     Returns:
#         dict: A dictionary containing dictionaries with the text items and their locations in the document. 
#             The dictionary has entries for each page of the document. Only items with type "text" are included.
#     """
#     image_paths = retrieve_validate_doc_path(doc_path)
    
#     # Try to import OmniParser modules
#     try:
#         # Add path for OmniParser if needed
#         sys_path_added = False
#         if "OmniParser" not in ' '.join(sys.path):
#             omniparser_possible_paths = [
#                 os.path.expanduser("~/Documents/GitHub/OmniParser"),
#             ]
            
#             for path in omniparser_possible_paths:
#                 if os.path.exists(path):
#                     sys.path.append(path)
#                     sys_path_added = True
#                     print(f"Added OmniParser path: {path}")
#                     break
        
#         if not sys_path_added:
#             print("Warning: Could not find OmniParser path. Attempting to import anyway.")
        
#         # Import OmniParser utilities
#         from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
#     except ImportError as e:
#         print(f"Error importing OmniParser utilities: {e}")
#         print("Please ensure OmniParser is installed and accessible.")
#         return None
    
#     # Check for device availability (CPU/GPU)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     # Set model paths based on process_screenshot_omniparser.py
    
#     # Load models
#     try:
#         # Load SOM model for object detection
#         som_model = get_yolo_model(model_path)
#         som_model.to(device)
        
#         # Get caption model processor
#         caption_model_processor = get_caption_model_processor(
#             model_name="florence2", 
#             model_name_or_path=caption_path, 
#             device=device
#         )
#     except Exception as e:
#         print(f"Error loading OmniParser models: {e}")
#         return None
    
#     # Process each page and extract text
#     result = {}
    
#     for page_index, image_path in enumerate(image_paths):
#         page_number = page_index + 1  # 1-indexed page numbers
#         print(f"Processing page {page_number}: {os.path.basename(image_path)}")
        
#         try:
#             # Get image size for converting bbox ratios to pixel coordinates
#             image = Image.open(image_path)
#             image_width, image_height = image.size
#             print(f"Image size: {image_width}x{image_height}")
            
#             # Configure bbox visualization (for debugging)
#             box_overlay_ratio = max(image_width, image_height) / 3200
#             draw_bbox_config = {
#                 'text_scale': 0.8 * box_overlay_ratio,
#                 'text_thickness': max(int(2 * box_overlay_ratio), 1),
#                 'text_padding': max(int(3 * box_overlay_ratio), 1),
#                 'thickness': max(int(3 * box_overlay_ratio), 1),
#             }
            
#             # Detection threshold
#             BOX_THRESHOLD = 0.05
            
#             # Run OCR to get text and bounding boxes
#             print("Running OCR...")
#             ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
#                 image_path, 
#                 display_img=False, 
#                 output_bb_format='xyxy', 
#                 goal_filtering=None, 
#                 easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
#                 use_paddleocr=True
#             )
#             text, ocr_bbox = ocr_bbox_rslt
            
#             # Run semantic analysis
#             print("Running semantic analysis...")
#             _, label_coordinates, parsed_content_list = get_som_labeled_img(
#                 image_path, 
#                 som_model, 
#                 BOX_TRESHOLD=BOX_THRESHOLD, 
#                 output_coord_in_ratio=True,  # Coordinates as ratio of image size
#                 ocr_bbox=ocr_bbox,
#                 draw_bbox_config=draw_bbox_config, 
#                 caption_model_processor=caption_model_processor, 
#                 ocr_text=text,
#                 use_local_semantics=True, 
#                 iou_threshold=0.7, 
#                 scale_img=False, 
#                 batch_size=128
#             )
            
#             # Filter for text items only and convert to location objects
#             text_items = {}
#             for item in parsed_content_list:
#                 if item.get('type') == 'text':
#                     bbox = item.get('bbox', [0, 0, 0, 0])  # [x1, y1, x2, y2] as ratios
#                     if len(bbox) == 4:
#                         x1, y1, x2, y2 = bbox
                        
#                         # Convert ratios to pixel coordinates
#                         x_px = x1 * image_width
#                         y_px = y1 * image_height
#                         width_px = (x2 - x1) * image_width
#                         height_px = (y2 - y1) * image_height
                        
#                         # Create location object
#                         loc = location(
#                             page_number=page_number,
#                             x=x_px,
#                             y=y_px,
#                             width=width_px,
#                             height=height_px
#                         )
                        
#                         # Get item ID (use original ID if available, otherwise generate one)
#                         item_id = item.get('ID', len(text_items))
                        
#                         # Store the text item with its location
#                         text_items[item_id] = {
#                             'text': item.get('content', ''),
#                             'location': loc,
#                             'bbox_ratio': bbox  # Keep original ratio coordinates
#                         }
            
#             # Add text items to result dictionary
#             result[page_number] = text_items
#             print(f"Found {len(text_items)} text items on page {page_number}")
            
#         except Exception as e:
#             print(f"Error processing page {page_number}: {e}")
#             import traceback
#             traceback.print_exc()
    
#     if not result:
#         print("No text items were extracted from any pages.")
#         return None
    
#     print(f"Text extraction complete. Extracted text from {len(result)} pages.")
#     return result

def extract_image_location_size_feature_based(image_path, image_size, doc_path, debug=False, dpi=300):
    """Extracts the location of an image in a document with known size using feature-based matching.

    Args:
        image_path (str): The path to the image file.
        image_size (dict): Known size of the image as it appears in the document.
            Format: {'width': {'magnitude': float, 'unit': str}, 'height': {'magnitude': float, 'unit': str}}
        doc_path (str): The path to the folder where images of the pdf are stored.
        debug (bool): Whether to enable debug visualization. Default is False.
        dpi (int): DPI of the PDF images. Default is 300.

    Returns:
        location: A location object of the bounding box of the image in the document.
        Returns None if not found.

    Raises:
        FileNotFoundError: If the provided image path does not exist or the document path is invalid.
    """
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    doc_images = retrieve_validate_doc_path(doc_path)

    if debug:
        print(f"Looking for {image_path} in document with {len(doc_images)} pages")
        print(f"Known size: {image_size}")

    # Convert known size to pixels based on DPI
    target_width_px = None
    target_height_px = None

    if image_size and 'width' in image_size and 'height' in image_size:
        width_info = image_size['width']
        height_info = image_size['height']

        if width_info.get('unit') == 'PT':  # Points
            target_width_px = width_info.get('magnitude', 0) * dpi / 72  # Convert points to pixels at specified DPI
            target_height_px = height_info.get('magnitude', 0) * dpi / 72
        elif width_info.get('unit') == 'PX':  # Pixels
            target_width_px = width_info.get('magnitude', 0)
            target_height_px = height_info.get('magnitude', 0)

        if debug:
            print(f"Target dimensions in pixels: {target_width_px:.1f}x{target_height_px:.1f}")
    
    # Initialize SIFT detector for feature matching
    sift = cv2.SIFT_create()
    
    # FLANN parameters for fast matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Try to find the image in each page of the document
    for page_index, doc_image_path in enumerate(doc_images):
        if debug:
            print(f"Searching page {page_index + 1} of {len(doc_images)}: {os.path.basename(doc_image_path)}")
        
        _, doc_image, template, _, _ = load_process_images(doc_image_path, image_path, False)
        
        # If we have target dimensions, resize template to match expected size
        if target_width_px and target_height_px:
            template_resized = cv2.resize(template, (int(target_width_px), int(target_height_px)))
        else:
            template_resized = template
        
        # Detect features in both images
        kp1, des1 = sift.detectAndCompute(template_resized, None)
        kp2, des2 = sift.detectAndCompute(doc_image, None)
        
        # Check if features were found
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            if debug:
                print(f"Page {page_index + 1} - Insufficient features detected")
            continue
        
        # Match features using FLANN
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if debug:
            print(f"Page {page_index + 1} - Found {len(good_matches)} good feature matches")
            
            # Display template and document with detected features
            template_with_features = cv2.drawKeypoints(template_resized, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            doc_with_features = cv2.drawKeypoints(doc_image, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imwrite(f"template_features_page_{page_index + 1}.png", template_with_features)
            cv2.imwrite(f"doc_features_page_{page_index + 1}.png", doc_with_features)
            
            # Display feature matches if there are good matches
            if len(good_matches) > 0:
                match_img = cv2.drawMatches(template_resized, kp1, doc_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(f"feature_matches_page_{page_index + 1}.png", match_img)
        
        # Need at least 4 good matches for homography
        if len(good_matches) >= 4:
            # Extract matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Calculate match confidence based on inliers
                inliers = np.sum(mask)
                match_confidence = inliers / len(good_matches)
                
                if debug:
                    print(f"Page {page_index + 1} - Homography found with {inliers}/{len(good_matches)} inliers")
                    print(f"Match confidence: {match_confidence:.3f}")
                
                # Check if confidence is sufficient
                if match_confidence >= 0.3:  # Adjust threshold as needed
                    # Get template dimensions
                    h, w = template_resized.shape[:2]
                    
                    # Define template corners
                    template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    
                    # Transform corners to document coordinates
                    doc_corners = cv2.perspectiveTransform(template_corners, M)
                    
                    # Calculate bounding box
                    x_coords = doc_corners[:, 0, 0]
                    y_coords = doc_corners[:, 0, 1]
                    
                    top_left_x = int(np.min(x_coords))
                    top_left_y = int(np.min(y_coords))
                    bottom_right_x = int(np.max(x_coords))
                    bottom_right_y = int(np.max(y_coords))
                    
                    width = bottom_right_x - top_left_x
                    height = bottom_right_y - top_left_y
                    
                    if debug:
                        print(f"Match found on page {page_index + 1}!")
                        print(f"Coordinates: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
                        print(f"Size: {width}x{height}")
                        
                        # Create visualization
                        vis_image = doc_image.copy()
                        
                        # Draw the transformed corners
                        cv2.polylines(vis_image, [np.int32(doc_corners)], True, (0, 255, 0), 3)
                        
                        # Draw bounding box
                        cv2.rectangle(vis_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)
                        
                        # Add confidence text
                        label = f"Confidence: {match_confidence:.3f}"
                        cv2.putText(vis_image, label, (top_left_x, top_left_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # cv2.imshow("Feature Match Visualization", vis_image)
                        cv2.imwrite(f"feature_match_page_{page_index + 1}.png", vis_image)
                    
                    # Create and return a location object
                    return location(
                        page_number=page_index + 1,  # 1-indexed page number
                        x=top_left_x,
                        y=top_left_y,
                        width=width,
                        height=height
                    )
    
    # Image not found in any page
    if debug:
        print("Image not found in any page of the document.")
    return None

def extract_image_location_size(image_path, image_size, doc_path, debug=False):
    """Extracts the location of an image in a document with known size from jpg images of a PDF doc/slide/sheet.

    Args:
        image_path (str): The path to the image file.
        image_size (dict): Known size of the image as it appears in the document.
            Format: {'width': {'magnitude': float, 'unit': str}, 'height': {'magnitude': float, 'unit': str}}
        doc_path (str): The path to the folder where images of the pdf are stored.
        debug (bool): Whether to enable debug visualization. Default is False.

    Returns:
        location: A location object of the bounding box of the image in the document.
        Returns None if not found.

    Raises:
        FileNotFoundError: If the provided image path does not exist or the document path is invalid.
    """
    # Validate inputs
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    doc_images = retrieve_validate_doc_path(doc_path)
    
    if debug:
        print(f"Looking for {image_path} in document with {len(doc_images)} pages")
        print(f"Known size: {image_size}")
    
    # Convert known size to pixels (screenshots are at 300 DPI)
    target_width_px = None
    target_height_px = None
    
    if image_size and 'width' in image_size and 'height' in image_size:
        width_info = image_size['width']
        height_info = image_size['height']
        
        if width_info.get('unit') == 'PT':  # Points
            target_width_px = width_info.get('magnitude', 0) * 300 / 72  # Convert points to pixels at 300 DPI
            target_height_px = height_info.get('magnitude', 0) * 300 / 72
        elif width_info.get('unit') == 'PX':  # Pixels
            target_width_px = width_info.get('magnitude', 0)
            target_height_px = height_info.get('magnitude', 0)
        
        if debug:
            print(f"Target dimensions in pixels: {target_width_px:.1f}x{target_height_px:.1f}")
    
    # Use direct template matching since we know the exact size
    match_threshold = 0.99
    
    # Try to find the image in each page of the document
    for page_index, doc_image_path in enumerate(doc_images):
        if debug:
            print(f"Searching page {page_index + 1} of {len(doc_images)}: {os.path.basename(doc_image_path)}")
        
        
        _, doc_image, template, _, _ = load_process_images( doc_image_path, image_path, False )
        
        # If we have target dimensions, resize template to match expected size
        if target_width_px and target_height_px:
            template_resized = cv2.resize(template, (int(target_width_px), int(target_height_px)))
        else:
            template_resized = template
        if debug and GUI_AVAILABLE:
            cv2.imshow("Template Image", template_resized)
        # Perform template matching
        result = cv2.matchTemplate(doc_image, template_resized, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get match info regardless of threshold
        top_left_x, top_left_y = max_loc
        height, width = template_resized.shape[:2]
        bottom_right_x = top_left_x + width
        bottom_right_y = top_left_y + height
        
        if debug:
            print(f"Page {page_index + 1} - Best match confidence: {max_val:.3f}")
            print(f"Coordinates: ({top_left_x}, {top_left_y}) to ({bottom_right_x}, {bottom_right_y})")
            print(f"Size: {width}x{height}")
            
            # Create visualization
            vis_image = doc_image.copy()
            color = (0, 255, 0) if max_val >= match_threshold else (0, 0, 255)  # Green if match, red if not
            cv2.rectangle(vis_image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
            
            # Add confidence text
            label = f"Confidence: {max_val:.3f}"
            cv2.putText(vis_image, label, (top_left_x, top_left_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Save visualization
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            page_name = os.path.splitext(os.path.basename(doc_image_path))[0]
            if GUI_AVAILABLE:
                cv2.imshow("Template Match Visualization", vis_image)
        
        if max_val >= match_threshold:
            if debug:
                print(f"Match found on page {page_index + 1}!")
            
            # Create and return a location object
            return location(
                page_number=page_index + 1,  # 1-indexed page number
                x=top_left_x,
                y=top_left_y,
                width=width,
                height=height
            )
    
    # Image not found in any page
    if debug:
        print("Image not found in any page of the document.")
    return None

def extract_image_location(image_path, doc_path, debug=False):
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
    match_threshold = 0.98
    min_source_fraction = 1/8
    max_source_fraction = 0.8
    scale_steps = 200
    
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
            visualize_final=debug,
            visualize_steps=debug,
            verbose=debug            # Change to True for more debugging information
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
    return None


def perceptual_hash_match(img1_path, img2_path, threshold=10):
    """Compare two images using perceptual hashing.

    Uses the pHash (perceptual hash) algorithm which is robust to minor
    differences like compression artifacts, slight color changes, and resizing.

    Args:
        img1_path (str): Path to the first image file.
        img2_path (str): Path to the second image file.
        threshold (int): Maximum hash difference to consider a match.
            Lower values are more strict. Default is 10.
            - 0: Identical images
            - 1-10: Very similar images (compression differences, minor edits)
            - 10-20: Similar images (might have some modifications)
            - >20: Likely different images

    Returns:
        bool: True if images match within the threshold, False otherwise.

    Raises:
        ImportError: If imagehash library is not installed.
        FileNotFoundError: If image paths don't exist.
    """
    try:
        import imagehash
    except ImportError:
        raise ImportError("imagehash library is required. Install with: pip install imagehash")

    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"Image path does not exist: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Image path does not exist: {img2_path}")

    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        hash_diff = hash1 - hash2

        return hash_diff <= threshold
    except Exception as e:
        print(f"Error computing perceptual hash: {e}")
        return False


def match_image_tiered(candidate_path, gold_path, model=None, description="", hash_threshold=10):
    """Match images using a tiered approach: exact match -> perceptual hash -> VLM.

    This function attempts to match two images using increasingly flexible methods:
    1. Exact pixel match (fastest, most strict)
    2. Perceptual hash comparison (fast, tolerant to minor differences)
    3. VLM-based comparison (slowest, most flexible) - compares both images directly

    Args:
        candidate_path (str): Path to the candidate image to check.
        gold_path (str): Path to the gold/reference image.
        model: Pre-loaded VLM model for fallback comparison. If None, VLM step is skipped.
        description (str): Optional text description (unused, kept for backwards compatibility).
        hash_threshold (int): Threshold for perceptual hash matching. Default is 10.

    Returns:
        tuple: (bool, str) - (match_result, match_method)
            - match_result: True if images match, False otherwise
            - match_method: One of "exact", "perceptual_hash", "vlm", or "no_match"

    Raises:
        FileNotFoundError: If image paths don't exist.
    """
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate image path does not exist: {candidate_path}")
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Gold image path does not exist: {gold_path}")

    # Step 1: Try exact pixel match
    try:
        exact_result = image_exact_match(candidate_path, gold_path)
        if exact_result:
            return True, "exact"
    except Exception as e:
        print(f"Exact match failed: {e}")

    # Step 2: Try perceptual hash
    try:
        if perceptual_hash_match(candidate_path, gold_path, threshold=hash_threshold):
            return True, "perceptual_hash"
    except ImportError:
        print("Perceptual hash skipped - imagehash library not available")
    except Exception as e:
        print(f"Perceptual hash failed: {e}")

    # Step 3: Try VLM comparison as fallback - compare both images directly
    if model is not None:
        try:
            if binary_compare_images(model, candidate_path, gold_path):
                return True, "vlm"
        except Exception as e:
            print(f"VLM comparison failed: {e}")

    return False, "no_match"