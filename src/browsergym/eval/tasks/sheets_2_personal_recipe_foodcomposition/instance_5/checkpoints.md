# Checkpoints

This task has 113 points in total.

## Checkpoint 1 (16 pts): Spreadsheet Structure & Column Layout
The spreadsheet contains the required columns in the correct order with proper nutrient groupings.

### Outcome Evaluation:
- "Ingredients" column exists and is first.
- "Link" column exists and is second.
- Carbohydrates column present.
- Fat column present.
- Fiber column present.
- Protein column present.
- Sugar column present.
- Calcium column present.
- Iron column present.
- Potassium column present.
- Sodium column present.
- Vitamin A column present.
- Vitamin C column present.
- Macros columns are in alphabetical order.
- Minerals columns are in alphabetical order.
- Vitamins columns are in alphabetical order.

## Checkpoint 2 (6 pts): Group Headers & Formatting
The spreadsheet has properly formatted group headers for nutrient categories.

### Outcome Evaluation:
- "Macros" merged header exists spanning macro columns.
- "Minerals" merged header exists spanning mineral columns.
- "Vitamins" merged header exists spanning vitamin columns.
- Group headers are centered and italicized.
- Column titles are bolded (excluding group headers).
- Horizontal line under column titles exists.

## Checkpoint 3 (4 pts): Color Formatting
Each nutrient group has a distinct background color applied to all cells in that group.

### Outcome Evaluation:
- Macro columns share same background color.
- Mineral columns share same background color.
- Vitamin columns share same background color.
- Three groups have distinct colors from each other.

## Checkpoint 4 (7 pts): Ingredients Present
The correct ingredients from the recipe are present in the spreadsheet, excluding those without specified amounts.

### Outcome Evaluation:
- ground beef row present.
- onion row present.
- tomato sauce row present.
- kidney beans row present.
- stewed tomatoes row present.
- chili powder row present.
- water row NOT present (excluded - no specified amount).

## Checkpoint 5 (6 pts): USDA Links Validation
Each ingredient has a valid USDA FoodData Central link that corresponds to the correct food item.

### Outcome Evaluation:
- ground beef link valid and matches ingredient (via HTML parsing).
- onion link valid and matches ingredient (via HTML parsing).
- tomato sauce link valid and matches ingredient (via HTML parsing).
- kidney beans link valid and matches ingredient (via HTML parsing).
- stewed tomatoes link valid and matches ingredient (via HTML parsing).
- chili powder link valid and matches ingredient (via HTML parsing).

## Checkpoint 6 (66 pts): Nutrient Values Accuracy
Nutrient values for each ingredient match the expected gold label values within tolerance. Each nutrient column is worth 6 points (one per ingredient).

### Outcome Evaluation:
- Carbohydrates values match gold labels within tolerance (6 pts).
- Fat values match gold labels within tolerance (6 pts).
- Fiber values match gold labels within tolerance (6 pts).
- Protein values match gold labels within tolerance (6 pts).
- Sugar values match gold labels within tolerance (6 pts).
- Calcium values match gold labels within tolerance (6 pts).
- Iron values match gold labels within tolerance (6 pts).
- Potassium values match gold labels within tolerance (6 pts).
- Sodium values match gold labels within tolerance (6 pts).
- Vitamin A values match gold labels within tolerance (6 pts).
- Vitamin C values match gold labels within tolerance (6 pts).

## Checkpoint 7 (1 pt): Bold Formatting for >10% DV
Nutrient values exceeding 10% of the FDA daily value are bolded.

### Outcome Evaluation:
- Values exceeding 10% DV are bolded per FDA guidelines.

## Checkpoint 8 (7 pts): Website Visit Validation
The agent visited the required websites to gather recipe and nutritional information.

### Outcome Evaluation:
- Recipe URL (allrecipes.com) visited (1 pt).
- USDA database URL visited for ground beef (1 pt).
- USDA database URL visited for onion (1 pt).
- USDA database URL visited for tomato sauce (1 pt).
- USDA database URL visited for kidney beans (1 pt).
- USDA database URL visited for stewed tomatoes (1 pt).
- USDA database URL visited for chili powder (1 pt).
