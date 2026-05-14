# Checkpoints

This task has 56 points in total.

## Checkpoint 1 (4 pts): Data Table Structure
The spreadsheet contains a properly structured table with required columns for strength training data.

### Outcome Evaluation:
- Date/time column exists with a header containing "date" or "activity date".
- Total Sets column exists with a header containing "sets" or "total sets".
- Total Reps column exists with a header containing "reps" or "total reps".
- All table content is fully visible (no text overflow/truncation).

## Checkpoint 2 (30 pts): Data Table Content Accuracy
The data in the table matches the source strength training data for all 219 Strength Training activities. Each step is scored proportionally: `floor(matches/219 * 10)` points out of 10 (e.g. 210/219 -> 9/10 pts). The boolean PASS state still requires every row to match exactly.

### Outcome Evaluation:
- Activity rows have an exact date match to gold data (proportional credit, 10 pts max).
- Activity rows have an exact Total Sets match to gold data (proportional credit, 10 pts max).
- Activity rows have an exact Total Reps match to gold data (proportional credit, 10 pts max).

## Checkpoint 3 (12 pts): Sets Over Time Plot
A chart showing individual workout sets over time.

### Outcome Evaluation:
- X-axis label indicates activity date.
- Y-axis label indicates sets.
- Chart title indicates sets over time.
- Chart is not placed over any other charts or tables.
- Chart main data series comes from the Total Sets column.
- Set values are present as circular points in the chart.
- Average adult sets baseline is properly displayed (labeled in legend + dotted/dashed style).
- Average adult sets baseline data is constant and within expected range (15-22 sets).
- Jay Cutler baseline is properly displayed (labeled in legend).
- Jay Cutler baseline data is constant and within expected range (40-80 sets).
- Both baselines are visually distinguishable from the main data.
- Source URLs are valid and accessible below the sets chart.

## Checkpoint 4 (6 pts): Cumulative Reps Plot
A chart showing cumulative reps lifted over time.

### Outcome Evaluation:
- X-axis label indicates activity date.
- Y-axis label indicates cumulative reps.
- Chart title indicates cumulative reps over time.
- Chart is not placed over any other charts or tables.
- Data shows cumulative/running total.
- Cumulative reps values are present as a line plot.

## Checkpoint 5 (4 pts): Website Visit Validation
The agent visited the required websites to gather data. URL relevance is determined first by keyword matching against the URL string, with an LLM-as-judge backup that fetches and inspects page content when the keyword filter returns no candidates. Content validity is determined first by structured LLM extraction + numeric comparison against the sheet baseline, with an LLM-as-judge backup that asks the model whether the page supports a baseline within the acceptable range when extraction fails for every URL.

### Outcome Evaluation:
- A source URL for average adult workout sets was visited (keyword match, then LLM judge backup).
- A source URL for Jay Cutler workout sets data was visited (keyword match, then LLM judge backup).
- Average adult sets source URL content matches the sheet baseline within acceptable range (pace extraction, then LLM judge backup).
- Jay Cutler source URL content matches the sheet baseline within acceptable range (pace extraction, then LLM judge backup).
