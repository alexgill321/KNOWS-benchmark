import os
import glob

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