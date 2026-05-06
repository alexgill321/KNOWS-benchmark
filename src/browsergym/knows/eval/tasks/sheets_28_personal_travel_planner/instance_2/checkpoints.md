# Checkpoints

This task has 370 points in total.

## Checkpoint 1 (40 pts, 10 pts each step): Table Structure and Layout
The spreadsheet follows the required column structure, frozen header, and correct row-per-day breakdown.

### Outcome Evaluation:
- The Google Sheet has all headers matching the requested columns: Date, Time of Day, Destination (Food/Activity), Cuisine, Opening time, Start Time, Departure Time, Duration, Review link, Transportation Mode, Travel Time, Cost, and Alternative Option (in blue).
- The header row is frozen, so it stays visible when scrolling.
- The sheet contains exactly 3 rows per day for activities/sightseeing (morning, afternoon, evening).
- The sheet contains exactly 2 rows per day for food stops (lunch and dinner).

## Checkpoint 2 (130 pts, 10 pts each step): Content Completeness
Every activity and restaurant entry has all required fields filled in.

### Outcome Evaluation:
- Each activity/restaurant row has a non-empty destination name.
- Each activity/restaurant row has a non-empty time of day value (e.g., Morning, Afternoon, Evening, Lunch, Dinner).
- Each activity/restaurant row has a non-empty opening hours value.
- Each activity/restaurant row has a non-empty start time.
- Each activity/restaurant row has a non-empty departure time.
- Each activity/restaurant row has a non-empty duration value.
- Each activity/restaurant row has a valid Google Maps review link URL.
- Each activity/restaurant row has a non-empty transportation mode.
- Each activity/restaurant row has a non-empty travel time.
- Each activity/restaurant row has a non-empty cost value.
- Each restaurant row has a non-empty cuisine type.
- Each activity/restaurant row has a non-empty alternative option.
- Each activity/restaurant row has a non-empty date column.

## Checkpoint 3 (40 pts, 10 pts each step): Destination Validity and Uniqueness
Every destination and alternative option refers to a real place in the target city, all are unique, and all attractions are open during their planned visit hours.

### Outcome Evaluation:
- Every activity and restaurant destination actually exists in the target city.
- Every alternative option actually exists in the target city.
- All destinations and alternative options are unique — no destination or alternative option is repeated across the entire itinerary.
- All attractions are open during the planned visit hours.

## Checkpoint 4 (70 pts, 10 pts each step): Content Accuracy
All details are verified for factual accuracy against real-world data.

### Outcome Evaluation:
- Each destination's opening hours are verified as accurate.
- Each destination's entrance fee/cost is verified as accurate.
- Every transportation mode is verified as available in the city for the given route.
- Every estimated travel time is verified as realistic for the given route and transportation mode.
- Each restaurant entry's type of cuisine is verified as accurate.
- The review link URL contains actual reviews or information for the specific place mentioned.
- Each alternative option is a viable substitute and is within a reachable distance.

## Checkpoint 5 (20 pts, 10 pts each step): Conditional Formatting and Styling
Cost cells are color-coded by price range and alternative options are formatted in blue.

### Outcome Evaluation:
- The Cost column is correctly conditionally formatted:
  - under $100 → green cell
  - $100–$200 → yellow cell
  - above $200 → orange cell
- The Alternative Option column's text is formatted in blue.

## Checkpoint 6 (70 pts, 10 pts each step): Logical Scheduling and Route Planning
The itinerary follows a sensible daily order with realistic timing, minimal backtracking, well-placed meal stops, feasible start/departure times, correct weekday/weekend transit adjustments, and transit time limits.

### Outcome Evaluation:
- Activities are arranged in a logical order, minimizing travel time and avoiding backtracking across the city.
- Lunch is placed between the Morning and Afternoon activity, and Dinner is placed between the Afternoon and Evening activity each day.
- Each row's duration is consistent with typical visit or dining times (e.g., 1–2 hours for lunch, 2–3 hours for museums).
- Each row's start time is feasible: it falls within the correct time block for its Time of Day slot (e.g., Morning starts between 8–11 AM, Lunch between 11 AM–2 PM, Afternoon between 12–5 PM, Dinner between 5–9 PM, Evening between 5–10 PM), and does not start before the previous event's departure time plus its travel time.
- Each row's departure time is feasible: it comes after the start time, the gap between start and departure is consistent with the stated duration, and it does not overlap with the next event's start time.
- Transit time estimates are adjusted accordingly for each day depending on whether it is a weekday or weekend (e.g., rush hour delays on weekdays, reduced service or increased congestion on weekends).
- Individual transit times are ≤ 30 minutes and total transit time per day is under 120 minutes.