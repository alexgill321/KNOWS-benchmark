# Checkpoints

This task has 28 points in total.

## Checkpoint 1 (5 pts): Color Extraction
The agent found and extracted beach/coastal wedding color information from at least two articles and color names are listed vertically in the top-left area of the main sheet.

### Outcome Evaluation:
- All extracted color names appear in a vertical list in a single column.
- The color list starts in the top-left area of the sheet (rows 1-15 approximately).
- At least 10 unique color names/shades are present, extracted from articles.
- The agent searched for and found at least 2 articles about beach or coastal wedding colors (agent trace, LLM judge that the articles are about beach/coastal weddings).
- All extracted colors belong to the specified coral, turquoise, or sand categories (LLM judge that each color falls into the category).

## Checkpoint 2 (4 pts): Article Source Links
Article links are placed in the column immediately to the right of each color name.

### Outcome Evaluation:
- The column next to color names contains URLs/links.
- Each color name has a corresponding article link.
- Links are functional and lead to the source articles.
- Proper alignment exists between color names and their source links.

## Checkpoint 3 (4 pts): Color Cell Formatting
Cells are filled with colors matching the color names in the third column.

### Outcome Evaluation:
- The column to the right of the links contains cells filled with background colors.
- The fill colors visually match or closely approximate the named colors (VLM judge).
- The hex value for the cell color matches the color shade hex found in https://www.colorhexa.com/.
- Each color name has a corresponding colored cell.

## Checkpoint 4 (3 pts): Fabric/Textile Store References
Fabric or textile store links are provided in the fourth column next to each color.

### Outcome Evaluation:
- The column to the right of the colored cells contains links to fabric or textile stores or color pages.
- Links are functional and lead to relevant fabric/textile color information (LLM judge that the linked page is from a fabric or textile store, not a paint store).
- Each color has an associated fabric/textile store link.

## Checkpoint 5 (4 pts): Beach Wedding Decoration Matrix
A beach wedding decoration matrix is created below the color list with images.

### Outcome Evaluation:
- At least 5 types of beach wedding decorations are listed in the leftmost column (API for location, LLM judge that items are beach-wedding-specific such as centerpieces, bridesmaid dresses, table runners, shell decorations, or invitations).
- Column headers contain the same color names from the original list (exact text match).
- At least half of the matrix cells contain images.
- Images show the specified decoration type in the corresponding color (VLM judge).

## Checkpoint 6 (4 pts): Day/Evening Color Palette Tab
A new tab contains at least 10 labeled color palette combinations.

### Outcome Evaluation:
- A new sheet/tab was created for color palettes.
- At least 10 rows of three-color combinations exist.
- Each row contains exactly 3 colored cells representing a palette.
- Colors are filled as background colors (not just text).
- Each palette row is labeled either "Day Ceremony" or "Evening Reception" (exact text match in a label column).
- Color combinations use colors from the original extracted list.

## Checkpoint 7 (4 pts): Decoration-Palette Checklist Tab
A second new tab contains a checklist mapping decoration items to top three palette choices.

### Outcome Evaluation:
- A second new sheet/tab was created.
- Column A contains a list of wedding decoration items (LLM judge for content).
- Columns B, C, and D represent three palette choices (e.g., labeled or color-coded as the top three palettes from the palette tab).
- Cells in columns B-D indicate which color from each palette goes with each decoration item (filled with colors or color references).
