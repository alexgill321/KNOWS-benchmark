# Checkpoints

This task has 47 points in total.

## Checkpoint 1 (4 pts, 4 steps): Spreadsheet Structure

The spreadsheet contains all required columns, at least 5 unique movies, and no blank cells.

### Outcome Evaluation:
- The sheet has all required labeled columns: Movie Title, Genre, Movie Rating (MPA/age rating), IMDb Score, Release Year, Duration, and Oscar Awards Won. (1 pt)
- The sheet includes at least 5 movies (data rows, excluding the header). (1 pt)
- Each row corresponds to one unique movie (no duplicate movie titles). (1 pt)
- No cell in any of the required columns is left blank. (1 pt)

## Checkpoint 2 (40 pts, 4 steps): Data Accuracy

Each movie's genre, Oscar wins, IMDb score, and MPA rating are verified against authoritative sources. Each step is scored proportionally per movie: round((movies passing / total movies) * 10).

### Outcome Evaluation:
- Each recommended movie belongs to at least one of the preferred genres [GENRE-1...GENRE-N], verified against the movie's IMDB genre listing. (10 pt, proportional)
- Every movie has won at least one Oscar award among: Best Actor, Best Actress, Best Director, Best Original Screenplay, Best Adapted Screenplay, or Best Cinematography. Verified against www.oscars.org. (10 pt, proportional)
- Each IMDb Score in the sheet matches the movie's actual IMDb rating (within +/- 0.1 tolerance) and is >= 6.5. (10 pt, proportional)
- Each MPA/age rating (e.g., PG, PG-13, R) is correct per the movie's IMDb listing or official distributor listing. (10 pt, proportional)

## Checkpoint 3 (3 pts, 3 steps): Sorting and Conditional Formatting

The list is sorted by duration and the highest/lowest IMDb scores are visually highlighted.

### Outcome Evaluation:
- The movie list is sorted by Duration (ascending or descending order). (1 pt)
- The cell(s) with the highest IMDb Score have green background fill and bold text. (1 pt)
- The cell(s) with the lowest IMDb Score have red background fill and bold text. (1 pt)
