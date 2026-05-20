# Checkpoints

This task has 58 points in total.

## Checkpoint 1 (4 pts): Data Table Structure
The spreadsheet contains a properly structured table with required columns for running data.

### Outcome Evaluation:
- Date/time column exists with a header containing "date" or "activity date".
- Distance column exists with a header containing "distance", "miles", or an equivalent miles-unit token (e.g., "(miles)", "(mi)").
- Average Speed column exists with a header containing "speed", "pace", or "min/mile".
- All table content is fully visible (no text overflow/truncation).

## Checkpoint 2 (30 pts): Data Table Content Accuracy
The data in the table matches the source running data for all 109 Run activities. Each step is scored out of 10 points using `floor(matches/109 * 10)`. The boolean PASS state requires every row to match exactly.

### Outcome Evaluation:
- Run activity rows have an exact date match to gold data (10 pts, proportional credit).
- Run activity rows have an exact distance match to gold data, converted to miles (10 pts, proportional credit).
- Run activity rows have an exact average speed match to gold data, converted to min/mile (10 pts, proportional credit).

## Checkpoint 3 (13 pts): Speed Over Time Plot
A scatter plot showing average running speed (min/mile) over time.

### Outcome Evaluation:
- X-axis label indicates activity date.
- Y-axis label indicates speed (min/mile or similar).
- Chart title indicates speed over time.
- Chart is not placed over any other charts or tables.
- Chart main data series comes from the average speed column.
- Speed values are present as circular points in the chart.
- Male 5K baseline is properly displayed (labeled in legend + dotted/dashed style).
- Male 5K baseline data is constant and within expected range (7.1-10 min/mile).
- Kipchoge baseline is properly displayed (labeled in legend).
- Kipchoge baseline data is constant and within expected range (4.5-4.8 min/mile).
- Both baselines are visually distinguishable from the main data.
- Source URLs are valid and accessible below the speed chart.
- Chart is on the same sheet tab as the data table.

## Checkpoint 4 (7 pts): Cumulative Distance Plot
A chart showing cumulative distance ran in miles over time.

### Outcome Evaluation:
- X-axis label indicates activity date.
- Y-axis label indicates cumulative distance (miles or similar).
- Chart title indicates cumulative distance over time.
- Chart is not placed over any other charts or tables.
- Data shows cumulative/running total.
- Cumulative running values are present as a line plot.
- Chart is on the same sheet tab as the data table.

## Checkpoint 5 (4 pts): Website Visit Validation
The agent visited the required websites to gather data. URL relevance is determined first by keyword matching against the URL string, with an LLM-as-judge backup that fetches and inspects page content when the keyword filter returns no candidates. Content validity is determined first by structured LLM pace extraction + numeric comparison against the sheet baseline (≤5% tolerance), with an LLM-as-judge backup that asks the model whether the page supports a baseline within ~5% of the sheet value when extraction fails for every URL.

### Outcome Evaluation:
- A source URL for male 5K running speed was visited (keyword match, then LLM judge backup).
- A source URL for Eliud Kipchoge marathon data was visited (keyword match, then LLM judge backup).
- Male 5K source URL content matches the sheet baseline within 5% (pace extraction, then LLM judge backup).
- Kipchoge source URL content matches the sheet baseline within 5% (pace extraction, then LLM judge backup).