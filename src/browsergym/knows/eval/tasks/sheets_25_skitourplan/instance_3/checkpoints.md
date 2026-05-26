# Checkpoints

This task has 50 points in total.

---

## Checkpoint 1 (10 pts): Spreadsheet Structure
The spreadsheet contains all required columns for ski run information and avalanche forecast data.

### Eval Steps:
1. **Run Name Column:** There is a column for the run name.
2. **Run Link Column:** There is a column for the run link (URL to Wasatch Backcountry Ski Guide).
3. **Starting Location Column:** There is a column for the starting location.
4. **GPS Coordinates Column:** There is a column for GPS coordinates of the run location.
5. **Typical Vertical Column:** There is a column for typical vertical of the run.
6. **Slope Aspect Column:** There is a column for slope aspect.
7. **Slope Angle Column:** There is a column for slope angle.
8. **Forecast Date Column:** There is a column for forecast date (merged vertically).
9. **Forecast Link Column:** There is a column for forecast link (merged vertically).
10. **Danger Rose Screenshot Column:** There is a column for the danger rose screenshot (merged vertically).

---

## Checkpoint 2 (4 pts): Run Selection Criteria
The spreadsheet contains exactly 3 ski runs that meet the slope angle requirement.

### Eval Steps:
1. **Run Count:** The spreadsheet contains exactly 3 ski runs.
2. **Slope Angle Compliance (Run 1):** Run 1 has a slope angle of 28 degrees or less.
3. **Slope Angle Compliance (Run 2):** Run 2 has a slope angle of 28 degrees or less.
4. **Slope Angle Compliance (Run 3):** Run 3 has a slope angle of 28 degrees or less.

---

## Checkpoint 3 (21 pts): Run Data Accuracy
Each run's data matches information from the Wasatch Backcountry Ski Guide. (7 pts per run x 3 runs)

### Eval Steps (repeated for each of 3 runs):
1. **Run Name Valid:** The run name exists in the Wasatch Backcountry Ski Guide.
2. **Run Link Valid:** The run link is a valid URL to the Wasatch Backcountry Ski Guide page for this run.
3. **Starting Location Correct:** The starting location matches the guide information.
4. **GPS Coordinates Correct:** The GPS coordinates match the guide information (within reasonable tolerance).
5. **Typical Vertical Correct:** The typical vertical matches the guide information.
6. **Slope Aspect Correct:** The slope aspect matches the guide information.
7. **Slope Angle Correct:** The slope angle matches the guide information and is <= 28 degrees.

---

## Checkpoint 4 (4 pts): Website Visit Validation
The agent visited the required websites to gather information.

### Eval Steps:
1. **Wasatch Guide Visited:** The browsing history contains a visit to the Wasatch Backcountry Ski Guide website.
2. **Utah Avalanche Center Visited:** The browsing history contains a visit to the Utah Avalanche Center website.
3. **Run Links Visited:** At least one of the run links from the spreadsheet appears in the browsing history.
4. **Forecast Page Visited:** The browsing history contains a visit to the avalanche forecast page for the relevant date.

---

## Checkpoint 5 (5 pts): Avalanche Forecast Data
The avalanche forecast information is correctly included and formatted.

### Eval Steps:
1. **Forecast Date Correct:** The forecast date is 02/12/2025.
2. **Forecast Link Valid:** The forecast link is a valid URL to the Utah Avalanche Center forecast.
3. **Merged Cells:** The forecast date, forecast link, and danger rose screenshot columns are vertically merged across all run rows.
4. **Danger Rose Screenshot Present:** An image of the danger rose is present in the spreadsheet.
5. **Danger Rose Image Valid:** The danger rose screenshot shows the avalanche danger rose from the Utah Avalanche Center.

---

## Checkpoint 6 (6 pts): Danger Rating Cell Coloring
Each run name cell is colored according to its avalanche danger rating. (2 pts per run x 3 runs)

### Eval Steps (repeated for each of 3 runs):
1. **Danger Rating Determined:** The danger color rating is correctly determined based on the run's slope aspect and angle relative to the danger rose.
2. **Cell Color Applied:** The run name cell is colored with the appropriate danger rating color (green=Low, yellow=Moderate, orange=Considerable, red=High, black=Extreme).
