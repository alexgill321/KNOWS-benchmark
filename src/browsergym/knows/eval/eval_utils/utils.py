import os
import glob
import cv2
import numpy as np

# Check if GUI functions are available (not available in headless mode)
# Set to False by default for headless environments
GUI_AVAILABLE = False


class location(object):
    def __init__(self, page_number, x, y, width, height):
        self.page_number = page_number
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Location(page_number={self.page_number}, x={self.x}, y={self.y}, width={self.width}, height={self.height})"
    
    def describe(self, dpi=300):
        """Return a human-readable description of this location.

        Args:
            dpi (int): The DPI used to render the page. Default 300.

        Returns:
            str: e.g. "page 1, 12% from left, 16% from top"
        """
        scale = dpi / 300
        page_w = 2550 * scale
        page_h = 3300 * scale
        x_pct = self.x / page_w * 100
        y_pct = self.y / page_h * 100
        return f"page {self.page_number + 1}, {x_pct:.0f}% from left, {y_pct:.0f}% from top"

    def is_upper_left(self, mostly=False):
        """
        Assumes a standard coordinate system where (0,0) is the top-left corner, and y increases downwards.

        Upper left is defined assuming a screenshot of a google doc page saved with 300 dpi. The dimensions should be 2550x3300 pixels.

        Args:
            mostly (bool): If True, only checks if the location is mostly in the upper left corner.
                           If False, checks if the location is entirely in the upper left corner.

        
        Returns:
            bool: True if the location is in the upper left corner, False otherwise.
        """
        upper_left = location(self.page_number, 0, 0, 1250, 1100)
        if mostly:
            return self.is_mostly_inside(upper_left)
        else:
            return self.is_inside(upper_left)
    
    def is_upper_right(self, mostly=False):
        """
        Assumes a standard coordinate system where (0,0) is the top-left corner, and y increases downwards.

        Upper right is defined assuming a screenshot of a google doc page saved with 300 dpi. The dimensions should be 2550x3300 pixels.

        Args:
            mostly (bool): If True, only checks if the location is mostly in the upper right corner.
                          If False, checks if the location is entirely in the upper right corner.

        Returns:
            bool: True if the location is in the upper right corner, False otherwise.
        """
        upper_right = location(self.page_number,1250,0,1250,1100)
        if mostly:
            return self.is_mostly_inside(upper_right)
        else:
            return self.is_inside(upper_right)
    
    def is_lower_left(self, mostly=False):
        """
        Assumes a standard coordinate system where (0,0) is the top-left corner, and y increases downwards.

        Lower left is defined assuming a screenshot of a google doc page saved with 300 dpi. The dimensions should be 2550x3300 pixels.

        Args:
            mostly (bool): If True, only checks if the location is mostly in the lower left corner.
                          If False, checks if the location is entirely in the lower left corner.

        Returns:
            bool: True if the location is in the lower left corner, False otherwise.
        """
        lower_left = location(self.page_number, 0, 0, 1250, 1100)
        if mostly:
            return self.is_mostly_inside(lower_left)
        else:
            return self.is_inside(lower_left)
    
    def is_lower_right(self, mostly=False):
        """
        Assumes a standard coordinate system where (0,0) is the top-left corner, and y increases downwards.

        Lower right is defined assuming a screenshot of a google doc page saved with 300 dpi. The dimensions should be 2550x3300 pixels.

        Args:
            mostly (bool): If True, only checks if the location is mostly in the lower right corner.
                          If False, checks if the location is entirely in the lower right corner.

        Returns:
            bool: True if the location is in the lower right corner, False otherwise.
        """
        lower_right = location(self.page_number,1250,2200,1250,1100)
        if mostly:
            return self.is_mostly_inside(lower_right)
        else:
            return self.is_inside(lower_right)
    
    def is_upper(self, mostly=False):
        """
        Assumes a standard coordinate system where (0,0) is the top-left corner, and y increases downwards.

        Upper region is defined assuming a screenshot of a google doc page saved with 300 dpi. 
        The dimensions should be 2550x3300 pixels.

        Args:
            mostly (bool): If True, only checks if the location is mostly in the upper region.
                          If False, checks if the location is entirely in the upper region.

        Returns:
            bool: True if the location is in the upper region, False otherwise.
        """
        upper = location(self.page_number,0,0,2550,1100)
        if mostly:
            return self.is_mostly_inside(upper)
        else:
            return self.is_inside(upper)
    
    def is_lower(self, mostly=False):
        """
        Assumes a standard coordinate system where (0,0) is the top-left corner, and y increases downwards.

        Lower region is defined assuming a screenshot of a google doc page saved with 300 dpi. 
        The dimensions should be 2550x3300 pixels.

        Args:
            mostly (bool): If True, only checks if the location is mostly in the lower region.
                          If False, checks if the location is entirely in the lower region.

        Returns:
            bool: True if the location is in the lower region, False otherwise.
        """
        lower = location(self.page_number,0,2200,2550,1100)
        if mostly:
            return self.is_mostly_inside(lower)
        else:
            return self.is_inside(lower)
    
    def is_inside(self, other):
        """
        Checks if this location is completely inside another location.
        
        Args:
            other (location): The other location to check against
            
        Returns:
            bool: True if this location is completely inside the other location
        """
        return (self.x >= other.x and
                self.y >= other.y and
                self.x + self.width <= other.x + other.width and
                self.y + self.height <= other.y + other.height)
    
    def is_mostly_inside(self, other, cutoff=0.6):
        """
        Checks if this location is mostly inside another location based on area overlap.
        
        Args:
            other (location): The other location to check against
            cutoff (float): The percentage threshold of overlap required (0.0 to 1.0)
                             Default is 0.6, meaning at least 60% of this location must
                             be inside the other location
        
        Returns:
            bool: True if the overlap percentage is greater than or equal to the specified percent
        """
        # First check that the locations are on the same page
        if self.page_number != other.page_number:
            return False
        
        # Calculate the intersection coordinates
        x_intersect_start = max(self.x, other.x)
        y_intersect_start = max(self.y, other.y)
        x_intersect_end = min(self.x + self.width, other.x + other.width)
        y_intersect_end = min(self.y + self.height, other.y + other.height)
        
        # Check if there's any overlap at all
        if x_intersect_start >= x_intersect_end or y_intersect_start >= y_intersect_end:
            return False  # No overlap
        
        # Calculate the area of overlap
        overlap_area = (x_intersect_end - x_intersect_start) * (y_intersect_end - y_intersect_start)
        
        # Calculate the area of this location
        self_area = self.width * self.height
        
        # Calculate the percentage of this location that overlaps with the other location
        if self_area == 0:
            return False  # Avoid division by zero
        
        overlap_ratio = (overlap_area / self_area)
        
        # Return True if the overlap percentage exceeds the threshold
        return overlap_ratio >= cutoff
                
    @staticmethod
    def merge_locations(locations):
        """
        Merges multiple location objects into a single location object that encompasses all of them.
        All locations must be from the same page.
        
        Args:
            locations (list): List of location objects to merge
            
        Returns:
            location: A new location object that encompasses all input locations
            
        Raises:
            ValueError: If locations are from different pages or list is empty
        """
        if not locations:
            raise ValueError("Cannot merge an empty list of locations")
            
        if len(locations) == 1:
            return locations[0]
            
        # Check that all locations are from the same page
        page_number = locations[0].page_number
        for loc in locations[1:]:
            if loc.page_number != page_number:
                raise ValueError("Cannot merge locations from different pages")
        
        # Find the minimum bounding box that contains all locations
        min_x = min(loc.x for loc in locations)
        min_y = min(loc.y for loc in locations) 
        max_x = max(loc.x + loc.width for loc in locations)
        max_y = max(loc.y + loc.height for loc in locations)
        
        # Create a new location object with the merged coordinates
        return location(
            page_number=page_number,
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y
        )

class layout(object):
    def __init__(self, element, element_type, doc_structure):
        """Initialize a layout object with an element and document structure.
        
        Args:
            element: The element to find in the document structure. Can be a text string or image ID.
            doc_structure: The document structure (as returned by extract_structure_from_doc).
        """
        self.doc_structure = doc_structure
        (self.start_position, self.end_position, self.elements_before, self.elements_after) = self.find_in_structure(element, element_type, doc_structure)

    def comes_before(self, other_element, other_element_type):
        if self._find_in_structure(other_element, other_element_type, self.elements_after) is not None:
            return True
        return False
    
    def comes_after(self, other_element, other_element_type):
        if self._find_in_structure(other_element, other_element_type, self.elements_before) is not None:
            return True
        return False
    
    def at_end(self):
        """Check if the element is at the end of the document structure."""
        return self.end_position == len(self.doc_structure) - 1
    
    def at_start(self):
        """Check if the element is at the start of the document structure."""
        return self.start_position == 0
        
    @staticmethod
    def find_in_structure(element, element_type, doc_structure):
        # TODO: Test this method thoroughly with various document structures 
        """Recursively searches for an element in the document structure content.

        For images, the content would be the image ID, and for text, it would be the text content.
        This function traverses the document structure and returns the start index of the found element.
        Matches must be exact, but can span across multiple lines of content in the document structure.
        """
        # TODO: In the future might need to handle this. For now positioned images are removed from the structure. so only inline images are considered.
        for idx, item in enumerate(doc_structure):
            # Check for match in this element
            content_match = False
            if item.get('type') == 'image':
                if item.get("source") == "positioned":
                    doc_structure.remove(item)
                    continue
        # Iterate through structure to find the element
        for idx, item in enumerate(doc_structure):
            if element_type == 'image':
                # For image IDs, check exact match of content field
                if item.get('type') == 'image' and item.get('content') == element:
                    content_match = True
            else:
                # For text, check if the text contains our element or vice versa
                if item.get('type') == 'text':
                    item_content = item.get('content', '')
                    if element in item_content:
                        content_match = True
            
            if content_match:
                # We found a match - build the before/after lists
                elements_before = doc_structure[:idx]
                elements_after = doc_structure[idx+1:]
                start_position = idx
                end_position = idx
                return (start_position, end_position, elements_before, elements_after)
                
        # Handle multi-element text matches (text that spans across multiple elements)
        if element_type != 'image':
            content_match = False
            best_start_idx = None
            best_end_idx = None
            # Try concatenating text elements to find longer matches
            for start_idx in range(len(doc_structure)):
                combined_text = ""
                for end_idx in range(start_idx, len(doc_structure)):
                    if doc_structure[end_idx].get('type') == 'text':
                        combined_text += doc_structure[end_idx].get('content', '')
                        
                    # Check if our combined text contains the element or vice versa
                    if element in combined_text:
                        content_match = True
                        best_start_idx = start_idx
                        best_end_idx = end_idx
            
            if content_match:
                for start_idx in range(best_start_idx, best_end_idx + 1):
                    combined_text = ""
                    # Collect all text elements in the range
                    for end_idx in range(start_idx, best_end_idx + 1):
                        if doc_structure[end_idx].get('type') == 'text':
                            combined_text += doc_structure[end_idx].get('content', '')
                    
                    if element in combined_text:
                        best_start_idx = start_idx
                    else:
                        return (best_start_idx, best_end_idx, doc_structure[:best_start_idx], doc_structure[best_end_idx+1:])
        # Element not found
        return None

def bbox_ratio_to_location(bbox_ratio, page_number, image_width, image_height):
    """
    Converts bounding box coordinates from ratio format to absolute pixel values 
    and creates a location object.
    
    Args:
        bbox_ratio (list): The bounding box in ratio format [x1, y1, x2, y2] where
                           each value is a ratio of the image dimensions (0.0 to 1.0)
        page_number (int): The page number where this bounding box is located
        image_width (int): The width of the source image in pixels
        image_height (int): The height of the source image in pixels
        
    Returns:
        location: A location object with absolute pixel coordinates
    """
    if len(bbox_ratio) != 4:
        raise ValueError("bbox_ratio must contain exactly 4 values [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox_ratio
    
    # Convert from ratio to absolute pixels
    x_px = x1 * image_width
    y_px = y1 * image_height
    width_px = (x2 - x1) * image_width
    height_px = (y2 - y1) * image_height
    
    # Create and return a location object
    return location(
        page_number=page_number,
        x=x_px,
        y=y_px,
        width=width_px,
        height=height_px
    )
    
def retrieve_validate_doc_path(doc_path):
    """
    Validates and retrieves the document path.
    
    Args:
        doc_path (str): Path to the document.
        
    Returns:
        str: Validated image paths.
    """
    # Validate the doc_path
    if not os.path.exists(doc_path):
        print(f"Error: Document path does not exist: {doc_path}")
        return None
    
    # Find the page images in the directory
    image_paths = glob.glob(os.path.join(doc_path, "*.png"))

    if not image_paths:
        print(f"Error: No PNG images found in document path: {doc_path}")
        return None
    
    # Sort the images to ensure correct page order
    image_paths.sort()

    return image_paths
    
def display_location_overlay(doc_path, loc, color=(0, 255, 0), max_width=1200, max_height=800, save_path=None):
    """
    Displays an image with a bounding box overlay for a location object, 
    resized to fit the display if needed.
    
    Args:
        doc_path (str): Path to the directory containing document page images
        loc (location): Location object with the bounding box to display
        color (tuple, optional): Color of the bounding box in BGR format (default: green)
        max_width (int, optional): Maximum display width. Default is 1200.
        max_height (int, optional): Maximum display height. Default is 800.
        save_path (str, optional): Path to save the overlaid image
        
    Returns:
        numpy.ndarray: The image with the bounding box overlay
    """
    import cv2
    import sys
    
    # Import resize_for_display from image_helpers
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from image_helpers import resize_for_display
    
    # Get image paths and select the right page
    image_paths = retrieve_validate_doc_path(doc_path)
    if not image_paths or loc.page_number > len(image_paths):
        print(f"Error: Invalid page number or path")
        return None
    
    # Load the image
    image_path = image_paths[loc.page_number]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Create a copy for drawing
    overlay_image = image.copy()
    
    # Draw the bounding box
    x1, y1 = int(loc.x), int(loc.y)
    x2, y2 = int(loc.x + loc.width), int(loc.y + loc.height)
    cv2.rectangle(overlay_image, (x1, y1), (x2, y2), color, 2)
    
    # Add a small label with coordinates
    label = f"({x1},{y1})"
    cv2.putText(
        overlay_image, 
        label, 
        (x1, y1-5 if y1 > 20 else y1+20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        color, 
        1
    )
    
    # Resize for display if needed
    display_image = resize_for_display(overlay_image, max_width, max_height)
    
    # Display the image (only if GUI is available)
    if GUI_AVAILABLE:
        window_name = f"Location Overlay (Page {loc.page_number})"
        cv2.imshow(window_name, display_image)
        
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"GUI not available in headless mode. Image overlay created for page {loc.page_number}.")
    
    # Save the original (non-resized) overlay image if requested
    if save_path:
        cv2.imwrite(save_path, overlay_image)
        print(f"Saved overlay image to {save_path}")
        
    return overlay_image

def image_id_from_path(image_path):
    """
    Extracts the image ID from the file name of an image path.

    The expected format is: image_{image_count}_{obj_id}{extension}
    where obj_id is the actual image ID we want to extract.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Image ID (obj_id) extracted from the file name.
    """
    # Extract the file name without the directory path
    file_name = os.path.basename(image_path)

    # Remove the file extension
    base_name = os.path.splitext(file_name)[0]

    # Check if the file follows the expected format
    if base_name.startswith('image_'):
        # Split by underscore and get the obj_id (the part after the second underscore)
        parts = base_name.split('_', 2)
        if len(parts) >= 3:
            return parts[2]  # Return the obj_id part

    # If the format doesn't match, return the base name as fallback
    return base_name


def bbox_overlap_ratio(bbox1, bbox2):
    """
    Calculate the overlap ratio of bbox1 with respect to bbox2.

    This calculates what percentage of bbox1's area overlaps with bbox2.

    Args:
        bbox1 (dict): First bounding box with 'x', 'y', 'width', 'height'.
        bbox2 (dict): Second bounding box with 'x', 'y', 'width', 'height'.

    Returns:
        float: Overlap ratio (0.0 to 1.0) representing the percentage of bbox1
               that overlaps with bbox2. Returns 0.0 if no overlap or invalid input.
    """
    # Extract coordinates
    x1_1 = bbox1.get('x', 0)
    y1_1 = bbox1.get('y', 0)
    x2_1 = x1_1 + bbox1.get('width', 0)
    y2_1 = y1_1 + bbox1.get('height', 0)

    x1_2 = bbox2.get('x', 0)
    y1_2 = bbox2.get('y', 0)
    x2_2 = x1_2 + bbox2.get('width', 0)
    y2_2 = y1_2 + bbox2.get('height', 0)

    # Calculate intersection
    x_intersect_start = max(x1_1, x1_2)
    y_intersect_start = max(y1_1, y1_2)
    x_intersect_end = min(x2_1, x2_2)
    y_intersect_end = min(y2_1, y2_2)

    # Check if there's any overlap
    if x_intersect_start >= x_intersect_end or y_intersect_start >= y_intersect_end:
        return 0.0

    # Calculate areas
    overlap_area = (x_intersect_end - x_intersect_start) * (y_intersect_end - y_intersect_start)
    bbox1_area = bbox1.get('width', 0) * bbox1.get('height', 0)

    if bbox1_area == 0:
        return 0.0

    return overlap_area / bbox1_area


def is_bbox_mostly_inside(inner_bbox, outer_bbox, threshold=0.6):
    """
    Check if inner_bbox is mostly inside outer_bbox based on overlap ratio.

    Args:
        inner_bbox (dict): The bbox to check, with 'x', 'y', 'width', 'height'.
        outer_bbox (dict): The containing bbox, with 'x', 'y', 'width', 'height'.
        threshold (float): Minimum overlap ratio required (0.0 to 1.0).
            Default is 0.6, meaning at least 60% of inner_bbox must be inside outer_bbox.

    Returns:
        bool: True if the overlap ratio >= threshold, False otherwise.
    """
    ratio = bbox_overlap_ratio(inner_bbox, outer_bbox)
    return ratio >= threshold


def bboxes_overlap(bbox1, bbox2):
    """
    Check if two bounding boxes overlap at all.

    Args:
        bbox1 (dict): First bounding box with 'x', 'y', 'width', 'height'.
        bbox2 (dict): Second bounding box with 'x', 'y', 'width', 'height'.

    Returns:
        bool: True if bboxes overlap, False otherwise.
    """
    return bbox_overlap_ratio(bbox1, bbox2) > 0


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert RGB (0-1 floats) to hex color string.

    Args:
        r: Red component (0-1).
        g: Green component (0-1).
        b: Blue component (0-1).

    Returns:
        Hex color string like '#FF0000'.
    """
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)
    return f'#{r_int:02X}{g_int:02X}{b_int:02X}'


def rgb_colors_match(color1: tuple, color2: tuple, tolerance: float = 0.05) -> bool:
    """Check if two RGB colors match within tolerance.

    Args:
        color1: First RGB tuple (r, g, b) with values 0-1.
        color2: Second RGB tuple (r, g, b) with values 0-1.
        tolerance: Maximum difference allowed per channel.

    Returns:
        True if colors match within tolerance.
    """
    if not color1 or not color2:
        return False

    if len(color1) != 3 or len(color2) != 3:
        return False

    return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))
