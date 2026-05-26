# Checkpoints
This task has 120 points in total.

## Checkpoint 1 (10 pt):
All original images from the presentation are saved to the specified Drive folder.

### Outcome Evaluation:
- All ground truth images from the original presentation are present in the Drive folder (verified via tiered image matching: exact -> perceptual hash -> VLM).

## Checkpoint 2 (40 pt):
All images in the presentation are deleted and replaced with red text description placeholders. Points assigned based on percentage of images correctly processed, rounded to nearest 10%.

### Outcome Evaluation:
- At each original image location, the image has been removed (10pt).
- A text box exists at the same location where each original image was - 80% overlap required (10pt).
- The text box contains a description of the original image - LLM similarity check against ground truth descriptions (10pt).
- The text in the placeholder is styled in big red text (>= 18pt) (10pt).

## Checkpoint 3 (50 pt):
Original images are replaced with similar images found online, with proper URL attribution. Points assigned based on percentage of images correctly processed, rounded to nearest 10%.

### Outcome Evaluation:
- Each image placeholder location now contains a replacement image from the web - 60% overlap required (10pt).
- Each replacement image has a URL credit beneath it (10pt).
- The URL below the image is a valid link to the image address online. The link must be below the image and in a valid location.(10pt)
- VLM check confirms the replacement image is a reasonable substitute for the original image by comparing both images directly, with description-based fallback if image comparison fails. The image must be in the original location otherwise this test will fail as well. (10pt).
- The replacement image fully covers the text placeholder so no red text is visible - 90% coverage required (10pt).

## Checkpoint 4 (20 pt):
No extra images were added to the presentation beyond the replacements. No extra textboxes were added to the new presentation besides the the ones that go under the image and the textboxes that hold the links.

### Outcome Evaluation:
- The total number of images in the presentation matches the original count - no extra images were added (10pt). Points are assigned proportionally based on the percentage of slides that have the correct image count.
- The total number of textboxes matches the orginal presentations count plus two times the original image count. No extra or less textboxes were added (10pt). Points are assigned proportionally based on the percentage of slides that have the correct textbox count.
