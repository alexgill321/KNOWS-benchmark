#!/usr/bin/env python3
"""
One time script to extract text box locations from the original presentation

This script:
1. Connects to Google Slides API
2. Extracts all textboxes from original presentation
"""

import os
import sys
import json
import csv
import tempfile

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))

from src.browsergym.knows.eval.eval_utils.google_services_utils import initialize_google_services
from src.browsergym.knows.eval.eval_utils.slides_utils import (
    extract_text_boxes_from_slide
)

# Constants
ORIGINAL_PRESENTATION_ID = "1V6r5Hg17LBtBYW5DBwmY4hW6eK3MYtnId9JgWw0cDvA"
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TASK_DIR, "data")
GOLD_IMAGES_DIR = os.path.join(DATA_DIR, "gold_images")
GOLD_DESCRIPTIONS_CSV = os.path.join(DATA_DIR, "gold_descriptions.csv")
OUTPUT_JSON = os.path.join(DATA_DIR, "original_textbox_locations.json")

def extract_textbox_location():
    """Extract textbox locations from original presentation"""
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
     
    textbox_locations = []

    #extract the textboxes and locations from each slide and store in dictionary
    with tempfile.TemporaryDirectory() as timp_dir:
        for slide_index, slide in enumerate(slides):
            print(f"\nProcessing slide {slide_index + 1}...")

            textboxes = extract_text_boxes_from_slide(slide)

            for textbox in textboxes:
                bbox = textbox.get('bbox',{})

                textbox_locations.append(
                    {
                        'slide_index' : slide_index,
                        'bbox' : bbox
                    }
                )
    
    return textbox_locations

def main():
    print("=" * 60)
    print("Generating original textbox locations JSON")
    print("=" * 60)

    textbox_locations = extract_textbox_location()

    if textbox_locations:
        with open(OUTPUT_JSON,'w',encoding = 'utf-8') as f:
            json.dump(textbox_locations,f,indent=2)
        print(f"\nSaved textbox locations to: {OUTPUT_JSON}")
        print(f"Total textbox mapped: {len(textbox_locations)}")
    else:
        print(f"\nERROR: No textbox locations extracted")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())





