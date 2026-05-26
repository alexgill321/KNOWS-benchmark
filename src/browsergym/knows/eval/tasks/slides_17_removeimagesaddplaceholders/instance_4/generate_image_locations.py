#!/usr/bin/env python3
"""
One-time script to extract image locations from the original presentation.

This script:
1. Connects to Google Slides API
2. Extracts all images from the original presentation
3. Downloads each image and matches it against gold images
4. Saves the image locations and metadata to original_image_locations.json

Run this script once to generate the static data file before using the evaluator.

Usage:
    python generate_image_locations.py
"""

import os
import sys
import json
import csv
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_slide_images,
    download_slide_image
)
from src.browsergym.knows.eval.eval_utils.image_utils import match_image_tiered

# Constants
ORIGINAL_PRESENTATION_ID = "1E49YKdl9qM1UwIx0bcAV1TX3RXE7rK09kefy3WhbttE"
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
GOLD_IMAGES_DIR = os.path.join(DATA_DIR, "gold_images")
GOLD_DESCRIPTIONS_CSV = os.path.join(DATA_DIR, "gold_descriptions.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "original_image_locations.json")


def load_gold_descriptions():
    """Load gold image filenames and descriptions from CSV."""
    descriptions = {}
    with open(GOLD_DESCRIPTIONS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            descriptions[filename] = row['description']
    return descriptions


def get_gold_image_paths():
    """Get list of all gold image paths."""
    gold_paths = []
    for filename in os.listdir(GOLD_IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            gold_paths.append(os.path.join(GOLD_IMAGES_DIR, filename))
    return gold_paths


def extract_image_locations():
    """Extract image locations from the original presentation."""
    print("Initializing Google Services...")
    drive_service, slides_service = initialize_google_services('slides')

    if not slides_service:
        print("ERROR: Failed to initialize Google Slides service")
        return None

    print(f"Fetching presentation: {ORIGINAL_PRESENTATION_ID}")
    presentation = slides_service.presentations().get(
        presentationId=ORIGINAL_PRESENTATION_ID
    ).execute()

    slides = presentation.get('slides', [])
    print(f"Found {len(slides)} slides")

    # Load gold data
    gold_descriptions = load_gold_descriptions()
    gold_image_paths = get_gold_image_paths()
    print(f"Loaded {len(gold_descriptions)} gold descriptions")
    print(f"Found {len(gold_image_paths)} gold images")

    # Track which gold images have been matched
    matched_gold_images = set()
    image_locations = {}

    # Create temp directory for downloaded images
    with tempfile.TemporaryDirectory() as temp_dir:
        for slide_index, slide in enumerate(slides):
            print(f"\nProcessing slide {slide_index + 1}...")

            # Extract images from this slide
            images = extract_slide_images(slide, ORIGINAL_PRESENTATION_ID, slides_service)

            for img_info in images:
                content_url = img_info.get('contentUrl')
                if not content_url:
                    continue

                # Download the image
                pil_image = download_slide_image(content_url)
                if pil_image is None:
                    print(f"  Failed to download image")
                    continue

                # Save to temp file for matching
                img_count = len([f for f in os.listdir(temp_dir) if f.startswith(f"slide_{slide_index}")])
                temp_path = os.path.join(temp_dir, f"slide_{slide_index}_img_{img_count}.png")
                pil_image.save(temp_path)
                print(f"  Downloaded image: {pil_image.size}")

                # Try to match against each gold image
                for gold_path in gold_image_paths:
                    gold_filename = os.path.basename(gold_path)

                    # Skip if already matched
                    if gold_filename in matched_gold_images:
                        continue

                    # Get description for VLM fallback
                    description = gold_descriptions.get(gold_filename, "")

                    # Try tiered matching with a lenient threshold
                    # Hash threshold of 20 is more forgiving for resized/recompressed images
                    is_match, match_method = match_image_tiered(
                        temp_path,
                        gold_path,
                        model=None,  # Skip VLM for speed during generation
                        description=description,
                        hash_threshold=20  # More lenient for extraction
                    )

                    if is_match:
                        print(f"  Matched: {gold_filename} (method: {match_method})")
                        matched_gold_images.add(gold_filename)

                        # Get bounding box from img_info directly
                        # extract_slide_images returns transform and size at top level
                        transform = img_info.get('transform', {})
                        size = img_info.get('size', {})

                        # Get raw dimensions
                        raw_width = size.get('width', {}).get('magnitude', 0)
                        raw_height = size.get('height', {}).get('magnitude', 0)

                        # Apply scale factors from transform (default to 1 if not present)
                        # The transform matrix includes scaleX/scaleY which must be multiplied
                        # with the raw size to get the actual rendered dimensions in EMUs
                        scale_x = transform.get('scaleX', 1)
                        scale_y = transform.get('scaleY', 1)

                        bbox = {
                            'x': transform.get('translateX', 0),
                            'y': transform.get('translateY', 0),
                            'width': raw_width * abs(scale_x),
                            'height': raw_height * abs(scale_y)
                        }

                        # Store the location data
                        image_locations[gold_filename] = {
                            'slide_index': slide_index,
                            'bbox': bbox,
                            'description': description,
                            'object_id': img_info.get('objectId', ''),
                            'content_url': content_url
                        }
                        break

    # Report unmatched gold images
    unmatched = set(gold_descriptions.keys()) - matched_gold_images
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} gold images were not matched:")
        for filename in sorted(unmatched):
            print(f"  - {filename}")

    print(f"\nMatched {len(matched_gold_images)} of {len(gold_descriptions)} gold images")
    return image_locations


def main():
    print("=" * 60)
    print("Generating original image locations JSON")
    print("=" * 60)

    image_locations = extract_image_locations()

    if image_locations:
        # Save to JSON
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(image_locations, f, indent=2)
        print(f"\nSaved image locations to: {OUTPUT_JSON}")
        print(f"Total images mapped: {len(image_locations)}")
    else:
        print("\nERROR: No image locations were extracted")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
