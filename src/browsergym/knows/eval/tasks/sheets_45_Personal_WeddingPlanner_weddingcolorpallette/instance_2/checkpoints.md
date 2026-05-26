# Checkpoints

This task has 63 points in total.

## Checkpoint 1 (5 pts): Color Extraction
The agent found and extracted wedding color information from at least three articles and color names are listed vertically in the top-left area of the main sheet.

### Outcome Evaluation:
- All extracted color names appear in a vertical list in a single column.
- The color list starts in the top-left area of the sheet (rows 1-15 approximately).
- At least 10 unique color names/shades are present, extracted from articles.
- The agent searched for and found at least 3 articles about wedding colors (agent trace, LLM judge that the articles are about weddings).
- All extracted colors belong to the specified purple, yellow, or orange categories (LLM judge that each color falls into the category).

## Checkpoint 2 (13 pts): Article Source Links
Article links are placed in the column immediately to the right of each color name.

### Outcome Evaluation:
- The article column is a URL/link column (every populated cell contains a URL, not free text). (1 pt)
- Each color name has a corresponding article link in the same row. (1 pt)
- The article links are functional and reachable. (1 pt)
- The links lead to relevant wedding-color content for that color. (10 pts, proportional)

## Checkpoint 3 (12 pts): Color Cell Formatting
Cells are filled with colors matching the color names in the third column.

### Outcome Evaluation:
- The fill column is a colour-fill column (every populated cell has a background fill, not just typed text). (1 pt)
- The fill colors match the named colors or are extremely close to them (VLM judge). (5 pts, proportional)
- The hex value for the cell color matches the color shade hex found in https://www.colorhexa.com/. (5 pts, proportional)
- Each color name has a corresponding colored cell. (1 pt)

## Checkpoint 4 (13 pts): Paint Store References
Paint store links are provided in the fourth column next to each color.

### Outcome Evaluation:
- The store column is a URL/link column (every populated cell contains a URL, not free text). (1 pt)
- Each color has an associated paint store link. (1 pt)
- The paint store links are functional and reachable. (1 pt)
- The links lead to relevant paint store / color content for that color. (10 pts, proportional)

## Checkpoint 5 (15 pts): Wedding Decoration Matrix
A wedding decoration matrix is created below the color list with images.

### Outcome Evaluation:
- At least 3 types of wedding decorations are listed in the leftmost column (API for location, LLM judge for content). (1 pt)
- Column headers contain the same color names from the original list (exact text match). (1 pt)
- The matrix column headers appear in the same order as the original color list. (1 pt)
- At least half of the matrix cells contain images. (2 pts)
- Images show the specified decoration type in the corresponding color (VLM judge). (10 pts, proportional)

## Checkpoint 6 (5 pts): Color Palette Tab
A separate tab contains at least 10 color palette combinations.

### Outcome Evaluation:
- A new sheet/tab was created for color palettes.
- At least 10 rows of color combinations exist.
- Each row contains exactly 3 colored cells representing a palette.
- Colors are filled as background colors (not just text).
- Color combinations use colors from the original extracted list.
