import os
import glob
import cv2

class location(object):
    def __init__(self, page_number, x, y, width, height):
        self.page_number = page_number
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Location(page_number={self.page_number}, x={self.x}, y={self.y}, width={self.width}, height={self.height})"
    
    def is_upper_left(self):
        return self.x + self.width < 0 and self.y < 0
    
    def is_upper_right(self):
        return self.x > 0 and self.y < 0
    
    def is_lower_left(self):
        return self.x + self.width < 0 and self.y + self.height> 0
    
    def is_lower_right(self):
        return self.x > 0 and self.y > 0
    
    def is_upper(self):
        return self.y < 0
    
    def is_lower(self):
        return self.y + self.height > 0
    
    def is_inside(self, other):
        return (self.x >= other.x and
                self.y >= other.y and
                self.x + self.width <= other.x + other.width and
                self.y + self.height <= other.y + other.height)
                
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
    print(f"Found {len(image_paths)} page images in {doc_path}")

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
    
    # Display the image
    window_name = f"Location Overlay (Page {loc.page_number})"
    cv2.imshow(window_name, display_image)
    
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the original (non-resized) overlay image if requested
    if save_path:
        cv2.imwrite(save_path, overlay_image)
        print(f"Saved overlay image to {save_path}")
        
    return overlay_image