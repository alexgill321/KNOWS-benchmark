# Checkpoints

This task has 31 points in total.

## Checkpoint 1 (5 pts): Color Extraction
The agent found and extracted fall/autumn wedding color information from at least three articles and color names are listed vertically in the top-left area of the main sheet.

### Outcome Evaluation:
- All extracted color names appear in a vertical list in a single column.
- The color list starts in the top-left area of the sheet (rows 1-20 approximately).
- At least 15 unique color names/shades are present, extracted from articles.
- The agent searched for and found at least 3 articles about fall or autumn wedding colors (agent trace, LLM judge that the articles are about fall/autumn weddings).
- All extracted colors belong to the specified burgundy, rust, or mustard categories (LLM judge that each color falls into the category).

## Checkpoint 2 (4 pts): Article Source Links
Article links are placed in the column immediately to the right of each color name.

### Outcome Evaluation:
- The column next to color names contains URLs/links.
- Each color name has a corresponding article link in the same row.
- The article links are functional and reachable.
- The links lead to relevant wedding-color content.

## Checkpoint 3 (4 pts): Color Cell Formatting
Cells are filled with colors matching the color names in the third column.

### Outcome Evaluation:
- The column to the right of the links contains cells filled with background colors.
- The fill colors visually match or closely approximate the named colors (VLM judge).
- The hex value for the cell color matches the color shade hex found in https://www.colorhexa.com/.
- Each color name has a corresponding colored cell.

## Checkpoint 4 (3 pts): Paint Store References
Paint store links are provided in the fourth column next to each color.

### Outcome Evaluation:
- The column to the right contains links to paint stores or color pages.
- Each color has an associated paint store link.
- Links are functional and lead to relevant paint/color information.

## Checkpoint 5 (4 pts): Mood Descriptions
A fifth column contains a one-sentence description of the mood or feeling each color evokes.

### Outcome Evaluation:
- The fifth column contains a textual description for each color.
- Each description is a short, single sentence (LLM judge).
- Descriptions describe the mood or feeling evoked by the color (LLM judge that content is about mood/feeling).
- Each color has a corresponding mood description (no missing entries).

## Checkpoint 6 (5 pts): Wedding Decoration Matrix
A wedding decoration matrix is created below the color list with images.

### Outcome Evaluation:
- At least 4 types of wedding decorations are listed in the leftmost column (API for location, LLM judge for content).
- Column headers contain the same color names from the original list (exact text match).
- The matrix column headers appear in the same order as the original color list.
- At least half of the matrix cells contain images.
- Images show the specified decoration type in the corresponding color (VLM judge).

## Checkpoint 7 (6 pts): Named Color Palette Tab
A separate tab contains at least 15 named color palette combinations.

### Outcome Evaluation:
- A new sheet/tab was created for color palettes.
- At least 15 rows of color combinations exist.
- The first column of each row contains a creative palette name (LLM judge that the label is a creative name).
- Each row contains exactly 3 colored cells representing the palette.
- Colors are filled as background colors (not just text).
- Color combinations use colors from the original extracted list.
