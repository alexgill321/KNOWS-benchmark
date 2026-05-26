import cv2
import numpy as np
import os
import sys
sys.path.append(os.getcwd())

# Check if GUI functions are available (not available in headless mode)
# Set to False by default for headless environments
GUI_AVAILABLE = False

def parse_response(response):
    """Parses yes or no response from the model and returns a boolean value.
    
    Accounts for variations in phrasing by looking for "yes" or "no" in the response.

    Args:
        response (str): The response from the model.

    Returns:
        bool or None: True if the response contains "yes", False if it contains "no",
            None if neither is found.
    """
    if response is None:
        return None
    response = response.lower()
    if "yes" in response:
        return True
    elif "no" in response:
        return False
    else:
        return None  # or raise an exception, or handle as needed


def resize_for_display(image, max_width=None, max_height=None):
    """
    Resizes an image for display, maintaining aspect ratio, if it exceeds
    the specified max_width or max_height.

    Args:
        image (np.array): The input image.
        max_width (int, optional): Maximum desired width. Defaults to None (no width limit).
        max_height (int, optional): Maximum desired height. Defaults to None (no height limit).

    Returns:
        np.array: The resized image, or the original image if no resizing was needed.
    """
    if image is None:
        return None
    if max_width is None and max_height is None:
        return image # No limits set

    h, w = image.shape[:2]
    scale = 1.0

    if max_width is not None and w > max_width:
        scale = min(scale, max_width / w)
    if max_height is not None and h > max_height:
        scale = min(scale, max_height / h)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Use INTER_AREA for shrinking, it's generally better
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        return image # No resizing needed


def read_transparent_png(filename):
    """
    Reads a PNG potentially with transparency and renders it on a white background.
    
    Args:
        filename (str): The path to the image file.
        
    Returns:
        np.array: The image with transparency rendered on white background.
    """
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_4channel is None:
        print(f"Error: Could not read image {filename}")
        return None
    # Check if image has 4 channels (BGRA)
    if len(image_4channel.shape) < 3 or image_4channel.shape[2] < 4:
        # print(f"Warning: Image {filename} does not appear to have an alpha channel. Reading as BGR.")
        # Ensure it still has 3 channels if it was grayscale
        if len(image_4channel.shape) == 2:
            return cv2.cvtColor(image_4channel, cv2.COLOR_GRAY2BGR)
        # If it had 3 channels already, return as is
        elif image_4channel.shape[2] == 3:
             return image_4channel
        else: # Should not happen with imread unless weird format
            print(f"Error: Unexpected image shape {image_4channel.shape} for {filename}")
            return None

    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    # Ensure alpha factor is broadcastable to 3 channels
    if alpha_factor.shape[2] == 1:
        alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def crop_whitespace(image, threshold=245):
    """
    Crops whitespace (pixels close to white) from the borders of an image.

    Args:
        image (np.array): Input image (BGR or Grayscale).
        threshold (int): Pixel intensity threshold to consider as whitespace (0-255).
                         Pixels >= threshold are considered whitespace.

    Returns:
        np.array: The cropped image, or the original image if no cropping occurred
                  or if the image is entirely whitespace. Returns None on error.
    """
    if image is None:
        print("Error: Input image to crop_whitespace is None.")
        return None

    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else: # Handle unexpected shapes
         print(f"Warning: Unexpected image shape {image.shape} in crop_whitespace. Trying BGRA->Gray conversion.")
         try:
             gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) # Attempt common conversion
         except cv2.error as e:
             print(f"Error: Could not convert image to grayscale for cropping: {e}. Returning original.")
             return image

    # Find all non-whitespace pixels
    coords = np.argwhere(gray < threshold) # Coordinates where pixel value is less than threshold

    if coords.size == 0:
        print("Warning: Template image appears to be entirely whitespace after thresholding. Cannot crop.")
        return image # Return original image if all pixels are >= threshold

    # Get the bounding box of non-whitespace pixels
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Ensure bounds are valid before slicing
    if y_min >= y_max or x_min >= x_max:
        print(f"Warning: Invalid crop bounds calculated (min >= max). Returning original image.")
        return image

    # Crop the original image using the bounding box
    # Add +1 to max values because slicing is exclusive of the end index
    try:
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    except IndexError as e:
         print(f"Error during image slicing for crop: {e}. Bounds were {y_min}:{y_max+1}, {x_min}:{x_max+1} on image shape {image.shape}")
         return image # Return original on slicing error

    return cropped_image


def load_process_images(source_image_path, template_image_path, crop_template_whitespace=True, whitespace_crop_threshold=245):
    """Load and process source and template images for template matching.
    
    Args:
        source_image_path (str): Path to the source image.
        template_image_path (str): Path to the template image.
        crop_template_whitespace (bool): If True, crop whitespace from template.
        whitespace_crop_threshold (int): Threshold for whitespace cropping.
        
    Returns:
        tuple: A tuple containing processed images and dimensions: 
            (source_img_bgr, source_img_gray, template_gray, source_dims, template_dims) 
            or (None, None, None, None, None) on error.
    """
    # Load source image
    # print(f"Loading source image: {source_image_path}")
    if not os.path.exists(source_image_path): 
        print(f"Error: Source image not found")
        return None, None, None, None, None
    
    source_img_bgr = cv2.imread(source_image_path)
    if source_img_bgr is None: 
        print(f"Error: Could not read source image")
        return None, None, None, None, None
    
    source_img_gray = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2GRAY)
    source_h, source_w = source_img_gray.shape[:2]
    # print(f"Source Image    (HxW): {source_h} x {source_w}")

    # Load and process template image
    # print(f"Loading template image: {template_image_path}")
    if not os.path.exists(template_image_path): 
        print(f"Error: Template image not found")
        return None, None, None, None, None
    
    template_bgr = read_transparent_png(template_image_path)
    if template_bgr is None: 
        print(f"Error: Failed to load/process template")
        return None, None, None, None, None
    
    # Crop whitespace if requested
    if crop_template_whitespace:
        # print(f"Cropping template whitespace (threshold={whitespace_crop_threshold})...")
        original_shape = template_bgr.shape[:2]
        template_bgr_cropped = crop_whitespace(template_bgr, threshold=whitespace_crop_threshold)
        if template_bgr_cropped is not None and template_bgr_cropped.shape[0]>0 and template_bgr_cropped.shape[1]>0:
            template_bgr = template_bgr_cropped
        else: 
            print("Warning: Cropping failed or resulted in empty image. Using uncropped template.")
    
    if template_bgr.shape[0] == 0 or template_bgr.shape[1] == 0: 
        print("Error: Template image is empty")
        return None, None, None, None, None
    
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    orig_template_h, orig_template_w = template_gray.shape[:2]
    
    if orig_template_h == 0 or orig_template_w == 0: 
        print(f"Error: Template dimensions are zero")
        return None, None, None, None, None
    
    return (source_img_bgr, source_img_gray, template_gray, 
            (source_h, source_w), (orig_template_h, orig_template_w))


def _determine_scale_range(template_dims, source_dims, min_scale=None, max_scale=None, 
                          min_source_fraction=None, max_source_fraction=None):
    """Determine the scale range for template matching.
    
    Args:
        template_dims (tuple): (height, width) of template image.
        source_dims (tuple): (height, width) of source image.
        min_scale (float, optional): Min explicit scaling factor.
        max_scale (float, optional): Max explicit scaling factor.
        min_source_fraction (float, optional): Min template width as source width fraction.
        max_source_fraction (float, optional): Max template width as source width fraction.
        
    Returns:
        tuple: A tuple containing scale range information: 
            (min_scale, max_scale, method_name) or (None, None, None) on error.
    """
    orig_template_h, orig_template_w = template_dims
    source_h, source_w = source_dims
    method = "Default"
    
    # Initialize scales
    actual_min_scale = 0.0
    actual_max_scale = 0.0

    if min_source_fraction is not None and max_source_fraction is not None:
        method = "Fraction of Source Width"
        # print(f"\nCalculating scale range based on source width fractions:")
        # print(f"  Min Fraction: {min_source_fraction:.4f}")
        # print(f"  Max Fraction: {max_source_fraction:.4f}")

        if orig_template_w == 0:
            print("Error: Template width is zero, cannot calculate scales based on fractions.")
            return None, None, None

        target_min_w = source_w * min_source_fraction
        target_max_w = source_w * max_source_fraction

        # Calculate the scale multipliers needed for the template
        calculated_min_scale = target_min_w / orig_template_w
        calculated_max_scale = target_max_w / orig_template_w

        # Ensure min_scale is less than or equal to max_scale
        actual_min_scale = min(calculated_min_scale, calculated_max_scale)
        actual_max_scale = max(calculated_min_scale, calculated_max_scale)
        if calculated_min_scale > calculated_max_scale:
             print("Warning: min_source_fraction resulted in a larger scale than max_source_fraction. Swapped.")

    elif min_scale is not None and max_scale is not None:
        method = "Explicit Scales"
        # Use explicitly provided scales
        actual_min_scale = min(min_scale, max_scale)
        actual_max_scale = max(min_scale, max_scale)
        if min_scale > max_scale:
             print("Warning: Explicit min_scale > max_scale. Swapped.")
    else:
        # Default fallback if no scale info provided
        method = "Default Fallback"
        print("\nWarning: No scale range specified (min/max_scale or min/max_source_fraction).")
        print("         Using default scale range [0.5, 1.5]. Provide scale parameters for control.")
        actual_min_scale = 0.5
        actual_max_scale = 1.5

    # Final check for validity
    if actual_min_scale <= 0 or actual_max_scale <= 0 or actual_min_scale > actual_max_scale:
         print(f"Error: Invalid scale range calculated/determined: Min={actual_min_scale:.4f}, Max={actual_max_scale:.4f}")
         return None, None, None
    
    return actual_min_scale, actual_max_scale, method


def _visualize_match_step(source_img_bgr, current_best_loc, current_score, scale, 
                         resized_w, resized_h, best_match_info, resize_display,
                         max_display_width, max_display_height, progress_window_name):
    """Visualize a single matching step.
    
    Args:
        source_img_bgr (np.array): Source image in BGR format.
        current_best_loc (tuple): (x, y) coordinates of current match.
        current_score (float): Current match score.
        scale (float): Current scale being tested.
        resized_w (int): Width of resized template.
        resized_h (int): Height of resized template.
        best_match_info (tuple): Best match information so far.
        resize_display (bool): Whether to resize the visualization.
        max_display_width (int): Maximum display width.
        max_display_height (int): Maximum display height.
        progress_window_name (str): Name of visualization window.
        
    Returns:
        int: Key pressed (or 0 if no key pressed/not applicable).
    """
    display_img = source_img_bgr.copy()
    curr_tl = current_best_loc 
    curr_br = (curr_tl[0] + resized_w, curr_tl[1] + resized_h)
    cv2.rectangle(display_img, curr_tl, curr_br, (255, 150, 0), 2)
    cv2.putText(display_img, f"Curr Sc:{current_score:.3f} @ {scale:.2f}", 
                (curr_tl[0], curr_tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,150,0), 1)
    
    # Draw best match so far
    if best_match_info:
        overall_score, overall_loc, overall_scale, overall_w, overall_h = best_match_info
        overall_tl = overall_loc
        overall_br = (overall_tl[0] + overall_w, overall_tl[1] + overall_h)
        is_new_best = (current_score == overall_score)
        best_color = (0,255,0) if not is_new_best else (50,255,255)
        thick = 2 if not is_new_best else 3
        cv2.rectangle(display_img, overall_tl, overall_br, best_color, thick)
        cv2.putText(display_img, f"Best:{overall_score:.3f} @ {overall_scale:.3f}", 
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, best_color, 2)

    # Display the visualization
    if resize_display:
        display_img_resized = resize_for_display(display_img, max_display_width, max_display_height)
        if display_img_resized is not None: 
            if GUI_AVAILABLE: cv2.imshow(progress_window_name, display_img_resized)
    else:
        if GUI_AVAILABLE: cv2.imshow(progress_window_name, display_img)

    key = cv2.waitKey(50) & 0xFF if GUI_AVAILABLE else 0
    return key


def _visualize_final_result(source_img_bgr, best_loc, best_w, best_h, best_score, best_scale,
                           threshold, source_w, source_h, match_found, resize_display,
                           max_display_width, max_display_height):
    """Visualize the final match result.
    
    Args:
        source_img_bgr (np.array): Source image in BGR format.
        best_loc (tuple): (x, y) coordinates of best match.
        best_w (int): Width of best match.
        best_h (int): Height of best match.
        best_score (float): Score of best match.
        best_scale (float): Scale of best match.
        threshold (float): Matching threshold.
        source_w (int): Source image width.
        source_h (int): Source image height.
        match_found (bool): Whether a match was found.
        resize_display (bool): Whether to resize the visualization.
        max_display_width (int): Maximum display width.
        max_display_height (int): Maximum display height.
    """
    final_display_img = source_img_bgr.copy()
    top_left_x, top_left_y = best_loc[0], best_loc[1]
    bottom_right_x = min(top_left_x + best_w, source_w)
    bottom_right_y = min(top_left_y + best_h, source_h)
    
    # Set color based on match found
    rect_color = (0,255,0) if match_found else (0,0,255)
    text = f'Match:{best_score:.3f} @{best_scale:.3f}' if match_found else f'Best<{threshold:.2f}:{best_score:.3f} @{best_scale:.3f}'
    title = f'Detected (>{threshold:.2f})' if match_found else f'Best Found (<{threshold:.2f})'
    
    # Draw rectangle and text
    cv2.rectangle(final_display_img, (top_left_x,top_left_y), (bottom_right_x, bottom_right_y), rect_color, 2)
    text_y = top_left_y-10 if top_left_y>20 else top_left_y+best_h+15
    cv2.putText(final_display_img, text, (top_left_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rect_color, 2)
    
    print(f"\nDisplaying final result (Resize:{resize_display}, Max WxH:{max_display_width}x{max_display_height}). Press any key.")
    
    # Display the visualization
    if resize_display:
        final_display_img_resized = resize_for_display(final_display_img, max_display_width, max_display_height)
        if final_display_img_resized is not None: 
            if GUI_AVAILABLE: cv2.imshow(title, final_display_img_resized)
    else: 
        if GUI_AVAILABLE: cv2.imshow(title, final_display_img)
    
    if GUI_AVAILABLE: cv2.waitKey(0)


def find_template_scale_invariant(source_image_path, template_image_path,
                              # --- Scale Control ---
                              min_scale=None,        # Explicit scale multiplier (lower bound)
                              max_scale=None,        # Explicit scale multiplier (upper bound)
                              min_source_fraction=None, # Min size relative to source width
                              max_source_fraction=None, # Max size relative to source width
                              num_scale_steps=20,
                              # --- Other Parameters ---
                              threshold=0.8,
                              crop_template_whitespace=True,
                              whitespace_crop_threshold=245,
                              visualize_final=False,
                              visualize_steps=False,
                              step_delay_ms=50,
                              verbose=False,
                              resize_display=True,
                              max_display_width=1200,
                              max_display_height=800):
    """
    Finds a template image within a source image across scales, with options
    for transparency, cropping, display resizing, and automatic scale range calculation.

    Scale Range Determination:
    - If `min_source_fraction` and `max_source_fraction` are provided (not None),
      they define the desired template size range relative to the source image *width*.
      These calculated scales will override any `min_scale` and `max_scale` values.
    - Otherwise, the explicitly provided `min_scale` and `max_scale` are used.
      Default values are applied if neither method is specified.

    Args:
        source_image_path (str): Path to the source image.
        template_image_path (str): Path to the template image.
        min_scale (float, optional): Min explicit scaling factor for the template's original size.
        max_scale (float, optional): Max explicit scaling factor for the template's original size.
        min_source_fraction (float, optional): Minimum desired template width as a fraction of source width.
        max_source_fraction (float, optional): Maximum desired template width as a fraction of source width.
        num_scale_steps (int): Number of scales to try within the determined range.
        threshold (float): Matching threshold (0.0 to 1.0) for TM_CCORR_NORMED.
        crop_template_whitespace (bool): If True, crop whitespace from template.
        whitespace_crop_threshold (int): Threshold for whitespace cropping.
        visualize_final (bool): If True, display the final best match.
        visualize_steps (bool): If True, display matching process at each scale.
        step_delay_ms (int): Milliseconds to wait between steps (0 for key press).
        verbose (bool): If True, print details for each scale tested.
        resize_display (bool): If True, resize visualization windows to fit max dimensions.
        max_display_width (int): Maximum width for visualization windows.
        max_display_height (int): Maximum height for visualization windows.

    Returns:
        tuple: (top_left_x, top_left_y, bottom_right_x, bottom_right_y, best_scale)
               or None if no match above threshold found or on error.
    """
    # --- 1. Load Images ---
    images = load_process_images(
        source_image_path, 
        template_image_path, 
        crop_template_whitespace, 
        whitespace_crop_threshold
    )
    
    if images[0] is None:
        return None
    
    source_img_bgr, source_img_gray, template_gray, source_dims, template_dims = images
    source_h, source_w = source_dims
    orig_template_h, orig_template_w = template_dims

    # --- 2. Determine Scale Range ---
    scale_range = _determine_scale_range(
        template_dims, 
        source_dims, 
        min_scale, 
        max_scale, 
        min_source_fraction, 
        max_source_fraction
    )
    
    if scale_range[0] is None:
        return None
    
    actual_min_scale, actual_max_scale, scale_calculation_method = scale_range
    # print(f"\nUsing Scale Calculation Method: {scale_calculation_method}")
    # print(f"Effective Scale Range: Min={actual_min_scale:.4f}, Max={actual_max_scale:.4f}")
    # print(f"Searching across {num_scale_steps} steps...")

    # --- 3. Iterate Through Scales ---
    best_match_info = None
    found_score = -np.inf
    method = cv2.TM_CCORR_NORMED
    # print(f"Using matching method: TM_CCORR_NORMED")
    
    if visualize_steps:
        print(f"-> Step visualization enabled (Resize: {resize_display}, Max WxH: {max_display_width}x{max_display_height}). Press 'q'/'Esc' to stop.")

    quit_early = False
    progress_window_name = "Matching Progress (Press 'q'/'Esc' to Stop)"

    # Generate scales within the determined range
    scales_to_try = np.linspace(actual_min_scale, actual_max_scale, num_scale_steps)
    # Sort descending - start with larger scales first (often faster)
    scales_to_try = scales_to_try[np.argsort(scales_to_try)[::-1]]

    # Main matching loop
    for i, scale in enumerate(scales_to_try):
        # Skip invalid scales
        if scale <= 0: 
            continue

        # Calculate resized dimensions
        resized_w = int(orig_template_w * scale)
        resized_h = int(orig_template_h * scale)

        # Skip invalid dimensions
        if resized_w <= 0 or resized_h <= 0:
            if verbose: 
                print(f"  ({i+1}/{num_scale_steps}) Skip scale {scale:.3f} (invalid size {resized_w}x{resized_h})")
            continue
        if resized_w > source_w or resized_h > source_h:
            if verbose: 
                print(f"  ({i+1}/{num_scale_steps}) Skip scale {scale:.3f} (resized {resized_w}x{resized_h} > source {source_w}x{source_h})")
            continue

        # Resize template
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        try:
            resized_template_gray = cv2.resize(template_gray, (resized_w, resized_h), interpolation=interpolation)
        except (cv2.error, ValueError) as e:
            print(f"  Error resizing template at scale {scale:.3f} to {resized_w}x{resized_h}: {e}")
            continue
        
        if resized_template_gray.shape[0]==0 or resized_template_gray.shape[1]==0: 
            continue  # Safety check

        # Perform matching
        try:
            if resized_template_gray.shape[0] > source_img_gray.shape[0] or \
               resized_template_gray.shape[1] > source_img_gray.shape[1]:
                raise cv2.error("Template > Source") # Should be caught earlier
            result = cv2.matchTemplate(source_img_gray, resized_template_gray, method)
        except cv2.error as e:
            print(f"  Error matchTemplate scale {scale:.3f}: {e} (Src:{source_img_gray.shape} Tmpl:{resized_template_gray.shape})")
            continue

        # Get best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        current_best_loc = max_loc
        current_score = max_val

        if verbose: 
            print(f"  ({i+1}/{num_scale_steps}) Scale: {scale:.3f}, Size: {resized_w}x{resized_h}, Score: {current_score:.4f}")

        # Update best match if better score found
        if current_score > found_score:
            found_score = current_score
            best_match_info = (current_score, current_best_loc, scale, resized_w, resized_h)
            if verbose: 
                print(f"    ---> New best score!")

        # Visualize matching step if requested
        if visualize_steps:
            key = _visualize_match_step(
                source_img_bgr, current_best_loc, current_score, scale,
                resized_w, resized_h, best_match_info, resize_display,
                max_display_width, max_display_height, progress_window_name
            )
            if key == ord('q') or key == 27: 
                quit_early = True
                break

    # Clean up step visualization
    if visualize_steps:
        try: 
            if GUI_AVAILABLE: cv2.destroyWindow(progress_window_name)
        except cv2.error: 
            pass

    # --- 4. Process Best Match Found ---
    if best_match_info is None: 
        print("\nError: No matching could be performed.")
        return None
    
    best_score, best_loc, best_scale, best_w, best_h = best_match_info
    print(f"\nOverall Best Score: {best_score:.4f} found at scale {best_scale:.3f} (Size: {best_w}x{best_h})")
    
    match_found = best_score >= threshold
    if match_found and verbose: 
        print(f"Match found ABOVE threshold ({threshold:.2f})!")
    elif not quit_early and verbose: 
        print(f"-> No match found exceeding the threshold ({threshold:.2f}).")
    elif quit_early and verbose: 
        print(f"-> Search stopped early. Best score {best_score:.4f} may be below threshold {threshold:.2f}.")

    # --- 5. Final Visualization ---
    if visualize_final:
        _visualize_final_result(
            source_img_bgr, best_loc, best_w, best_h, best_score, best_scale,
            threshold, source_w, source_h, match_found, resize_display,
            max_display_width, max_display_height
        )

    # Clean up all windows
    try: 
        if GUI_AVAILABLE: cv2.destroyAllWindows()
    except cv2.error: 
        pass

    # --- 6. Return Result ---
    if match_found:
        # Return original image coordinates
        return (best_loc[0], best_loc[1], best_loc[0] + best_w, best_loc[1] + best_h, best_scale)
    else:
        return None