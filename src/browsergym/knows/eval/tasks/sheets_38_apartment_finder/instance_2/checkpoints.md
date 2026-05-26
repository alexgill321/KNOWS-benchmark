# Checkpoints

This task has 37 points in total.

## Checkpoint 1 (9 pts): Spreadsheet Structure

The spreadsheet contains columns for all desired features and at least 3 listings.

### Outcome Evaluation:

- There is a column for the listing address.
- There is a column for the monthly rent/price.
- There is a column for the number of bedrooms.
- There is a column for the number of bathrooms.
- There is a column for square footage.
- There is a column indicating if the unit is fully furnished.
- There is a column containing a link/URL to each listing.
- There is a column for positive features.
- There is a column for dealbreakers.

## Checkpoint 2 (18 pts): Listing Data Accuracy (x3 listings, 6 pts each)

Each listing's data in the spreadsheet is verified against the actual Craigslist listing page via HTML parsing and LLM extraction.

### Outcome Evaluation (repeated for each of 3 listings):

- The listing URL is valid and accessible.
- The price in the spreadsheet matches the price extracted from the listing HTML (within 5% tolerance).
- The bedroom count in the spreadsheet matches the value extracted from the listing HTML.
- The bathroom count in the spreadsheet matches the value extracted from the listing HTML.
- The address in the spreadsheet matches or is contained in the address extracted from the listing HTML.
- The fully furnished status in the spreadsheet matches the information extracted from the listing HTML.

## Checkpoint 3 (2 pts): Website Visit Validation

The agent visited Craigslist to find the listings.

### Outcome Evaluation:

- The browsing history contains a visit to craigslist.org.
- The listing URLs in the spreadsheet appear in the browsing history.

## Checkpoint 4 (2 pts): Conditional Formatting Applied

Conditional formatting is correctly applied to numeric columns.

### Outcome Evaluation:

- Conditional formatting (color scale) is applied to at least one numeric column (e.g., price, sq ft).
- The formatting uses a green-to-red scale where green indicates better values and red indicates worse values.

## Checkpoint 5 (2 pts): Summary Statistics Table

A summary statistics table exists with auto-updating formulas.

### Outcome Evaluation:

- A summary statistics table exists starting at column L (top-right area of sheet).
- The summary table contains formulas/equations that reference the main listing data (not hardcoded values).

## Checkpoint 6 (4 pts): Text Visibility and Formatting

All text in both tables is fully visible and not cut off.

### Outcome Evaluation:

- All column headers in the main table are fully visible (not truncated).
- All data cells in the main table have adequate column width to display content without truncation.
- All text in the summary statistics table is fully visible.
- No text is hidden due to cell overflow issues.
