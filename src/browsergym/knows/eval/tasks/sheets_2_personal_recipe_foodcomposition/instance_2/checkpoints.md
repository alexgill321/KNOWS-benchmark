# Checkpoints

This task has 99 points in total.

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

## Checkpoint 4 (6 pts): Ingredients Present
The correct ingredients from the recipe are present in the spreadsheet, excluding those without specified amounts.

### Outcome Evaluation:
- granulated sugar row present.
- unsweetened cocoa powder row present.
- hot coffee row present.
- milk row present.
- Water row present.
- whipped cream row NOT present (excluded - optional).

## Checkpoint 5 (5 pts): USDA Links Validation
Each ingredient has a valid USDA FoodData Central link that corresponds to the correct food item.

### Outcome Evaluation:
- granulated sugar link valid and matches ingredient (via HTML parsing).
- unsweetened cocoa powder link valid and matches ingredient (via HTML parsing).
- hot coffee link valid and matches ingredient (via HTML parsing).
- milk link valid and matches ingredient (via HTML parsing).
- Water link valid and matches ingredient (via HTML parsing).

## Checkpoint 6 (55 pts): Nutrient Values Accuracy
Nutrient values for each ingredient match the expected gold label values within tolerance. Each nutrient column is worth 5 points (one per ingredient).

### Outcome Evaluation:
- Carbohydrates values match gold labels within tolerance (5 pts).
- Fat values match gold labels within tolerance (5 pts).
- Fiber values match gold labels within tolerance (5 pts).
- Protein values match gold labels within tolerance (5 pts).
- Sugar values match gold labels within tolerance (5 pts).
- Calcium values match gold labels within tolerance (5 pts).
- Iron values match gold labels within tolerance (5 pts).
- Potassium values match gold labels within tolerance (5 pts).
- Sodium values match gold labels within tolerance (5 pts).
- Vitamin A values match gold labels within tolerance (5 pts).
- Vitamin C values match gold labels within tolerance (5 pts).

## Checkpoint 7 (1 pt): Bold Formatting for >10% DV
Nutrient values exceeding 10% of the FDA daily value are bolded.

### Outcome Evaluation:
- Values exceeding 10% DV are bolded per FDA guidelines.

## Checkpoint 8 (6 pts): Website Visit Validation
The agent visited the required websites to gather recipe and nutritional information.

### Outcome Evaluation:
- Recipe URL (bakingmischief.com) visited (1 pt).
- USDA database URL visited for granulated sugar (1 pt).
- USDA database URL visited for unsweetened cocoa powder (1 pt).
- USDA database URL visited for hot coffee (1 pt).
- USDA database URL visited for milk (1 pt).
- USDA database URL visited for Water (1 pt).
