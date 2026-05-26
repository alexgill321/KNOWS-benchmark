# Checkpoints
This task has 122 points in total.

## Checkpoint 1 (12 pt, 5 steps):
Check that the title slide contains an appropriate high-quality image of a room/project that fills most of the slide space.

### Outcome Evaluation:
- Verify that the room/project is put as the title (2 pts)
- Confirm that exactly one image was placed on the title slide (2 pt)
- Check that the image covers at least 70% of the slide area (4 pt)
- Verify that the agent searched for the room/project type in browsing history (2 pt)
- Verify image relevance to the room/project type (VLM judge) (2 pt)

## Checkpoint 2 (30 pt, 3 steps × 10 pt each):
Check that 5-10 color selection slides were created with appropriate color names as titles.

### Outcome Evaluation:
- Count the number of color slides to ensure it falls within the 5-10 range (10 pt)
- Verify each content slide has a color name as its title (10 pt, proportional)
- Confirm that the color names are distinct and appropriate for interior design of the room/project (LLM judge) (10 pt, proportional)

## Checkpoint 3 (70 pt, 7 steps × 10 pt each):
Verify that each color slide contains two relevant images positioned correctly on the slide, with proper source attribution.

### Outcome Evaluation:
- Confirm that the agent searched for images combining the specific color with the room/project type in browsing history (10 pt, proportional)
- Check that exactly two images appear on each color slide (10 pt, proportional)
- Verify image positioning: one image in bottom left, one image in bottom right (10 pt, proportional)
- Ensure images are relevant to both the color theme and the room/project type (VLM judge) (10 pt, proportional)
- Verify that each image has a source URL in its ALT text (10 pt, proportional)
- Confirm that the ALT text source URL leads to the same image as displayed on the slide (10 pt, proportional)
- Check that the two images for a color are unique (10 pt, proportional)

## Checkpoint 4 (10 pt, 4 steps):
Check that a final recommendation slide was created as the last slide with the agent's color choice clearly stated.

### Outcome Evaluation:
- Verify that a recommendation slide exists and is the last slide in the presentation (2 pt)
- Check that the text "{COLOR} is the best choice" appears on the slide (case-insensitive match) (3 pt)
- Confirm that the chosen color matches one of the previously presented color slide options (exact match) (3 pt)
- Verify that the recommendation text is displayed in large font (at least 18pt) (2 pt)