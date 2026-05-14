# Checkpoints

This task has 66 points in total.

## Checkpoint 1 (4 pts): Data Table Structure
The spreadsheet contains a properly structured table with required columns for running and walking data.

### Outcome Evaluation:
- Date/time column exists with a header containing "date" or "activity date".
- Distance column exists with a header containing "distance", "miles", or an equivalent miles-unit token (e.g., "(miles)", "(mi)").
- Average Speed column exists with a header containing "speed", "pace", or "min/mile".
- All table content is fully visible (no text overflow/truncation).

## Checkpoint 2 (30 pts): Data Table Content Accuracy
The data in the table matches the source data for all 452 Run and Walk activities. Each step is scored proportionally to the number of rows that match: a step's awarded points equal `floor(matches/452 * 10)` out of 10 (e.g. 440/452 -> 9/10 pts). The boolean PASS state still requires every row to match exactly.

### Outcome Evaluation:
- Activity rows have an exact date match to gold data (proportional credit, 10 pts max).
- Activity rows have an exact distance match to gold data, converted to miles (proportional credit, 10 pts max).
- Activity rows have an exact average speed match to gold data, converted to min/mile (proportional credit, 10 pts max).

## Checkpoint 3 (9 pts): Daily Total Miles Bar Plot
A bar chart showing daily total miles with both walk and run data.

### Outcome Evaluation:
- X-axis label indicates date.
- Y-axis label indicates miles or distance.
- Chart title indicates daily total miles.
- Chart is not placed over any other charts or tables.
- Chart is a BAR type.
- Chart has a non-constant data series representing daily mileage.
- Chart data includes both walk and run activity data (data point count >= run count).
- Female 25 daily miles baseline is properly displayed (labeled in legend).
- Female 25 daily miles baseline data is constant and within expected range (1.5-6.0 miles).

## Checkpoint 4 (11 pts): Speed Over Time Plot
A plot showing average running speed (min/mile) over time.

### Outcome Evaluation:
- X-axis label indicates activity date.
- Y-axis label indicates speed (min/mile or similar).
- Chart title indicates speed over time.
- Chart is not placed over any other charts or tables.
- Chart main data series comes from the average speed column.
- Female 25 5K baseline is properly displayed (labeled in legend + dotted/dashed style).
- Female 25 5K baseline data is constant and within expected range (7.5-11.5 min/mile).
- Beatrice Chebet baseline is properly displayed (labeled in legend).
- Beatrice Chebet baseline data is constant and within expected range (4.0-5.0 min/mile).
- Both baselines are visually distinguishable from the main data.
- Source URLs are valid and accessible below the speed chart.

## Checkpoint 5 (6 pts): Cumulative Distance Plot
A chart showing cumulative distance ran in miles over time.

### Outcome Evaluation:
- X-axis label indicates activity date.
- Y-axis label indicates cumulative distance (miles or similar).
- Chart title indicates cumulative distance over time.
- Chart is not placed over any other charts or tables.
- Data shows cumulative/running total.
- Cumulative running values are present as a line plot.

## Checkpoint 6 (6 pts): Website Visit Validation
The agent visited the required websites to gather data. URL relevance is determined first by keyword matching against the URL string, with an LLM-as-judge backup that fetches and inspects page content when the keyword filter returns no candidates. Content validity is determined first by structured LLM pace extraction + numeric comparison against the sheet baseline (<=5% tolerance), with an LLM-as-judge backup that asks the model whether the page supports a baseline within ~5% of the sheet value when extraction fails for every URL.

### Outcome Evaluation:
- A source URL for average female 25 daily miles trekked was visited (keyword match, then LLM judge backup).
- A source URL for average female 25 5K running speed was visited (keyword match, then LLM judge backup).
- A source URL for Beatrice Chebet 5K data was visited (keyword match, then LLM judge backup).
- Female daily miles source URL content matches the sheet baseline within 5% (pace extraction, then LLM judge backup).
- Female 5K source URL content matches the sheet baseline within 5% (pace extraction, then LLM judge backup).
- Beatrice Chebet source URL content matches the sheet baseline within 5% (pace extraction, then LLM judge backup).
