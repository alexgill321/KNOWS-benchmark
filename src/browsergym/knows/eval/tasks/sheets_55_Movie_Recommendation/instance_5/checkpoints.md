# Checkpoints

This task has 88 points in total.

## Checkpoint 1 (5 pts, 5 steps): Spreadsheet Structure

The spreadsheet contains all required columns, at least 5 unique movies with no blank cells, includes at least 3 distinct primary genres, and the total duration falls within the required window.

### Outcome Evaluation:
- The sheet has all required labeled columns: Movie Title, Genre, MPA/Age Rating, IMDb Score, Release Year, Duration, and Oscar Awards Won. (1 pt)
- The columns are in this exact order: Movie Title, Genre, MPA/Age Rating, IMDb Score, Release Year, Duration, Oscar Awards Won (1 pt)
- The sheet includes at least 5 movies (data rows, excluding the header). (1 pt)
- Each row corresponds to one unique movie (no duplicate movie titles). (1 pt)
- No cell in any of the required columns is left blank. (1 pt)

## Checkpoint 2 (80 pts, 8 steps): Data Accuracy

Each movie's genre, Oscar wins, IMDb score, and MPA rating are verified against authoritative sources. Each step is scored proportionally per movie: round((movies passing / total movies) * 10).

### Outcome Evaluation:
- Each recommended movie's IMDb page (e.g., imdb.com/title/tt#####) appears in the browsing history. (10 pt, proportional)
- Each recommended movie's primary genre on IMDb (the first standard genre IMDb lists for the movie, not UI tags) is one of the preferred genres (Drama, Comedy, Crime). Word-boundary match: compound forms like "Crime Drama" pass for "Crime"; unrelated tags do not. (10 pt, proportional)
- At least 3 distinct primary genres from the specified categories (per IMDb's standard genre list, first listed) appear among the recommended movies. (10 pt, all or nothing)
- Each movie's release year must be between 1985 and 2005. (10 pt, proportional)
- The Oscar Awards Won cell lists exactly the qualifying Oscars (from {Best Actor, Best Actress, Best Adapted Screenplay, Best Cinematography}) that the movie actually won, comma-separated, with no extras and no omissions. The movie must have won at least one. Verified against the IMDb awards page. (10 pt, proportional)
- Each IMDb Score in the sheet matches the movie's actual IMDb rating (within +/- 0.1 tolerance) and is >= 6.5. (10 pt, proportional)
- Each MPA/Age Rating (e.g., PG, PG-13, R) is correct per the movie's IMDb listing or official distributor listing. (10 pt, proportional)
- The sum of all movie durations is between 600 and 800 minutes, inclusive. (10 pt, all or nothing)

## Checkpoint 3 (3 pts, 3 steps): Sorting and Conditional Formatting

The list is sorted by duration and the highest/lowest IMDb scores are visually highlighted.

### Outcome Evaluation:
- The movie list is sorted by Duration in descending order. (1 pt)
- The cell(s) with the highest IMDb Score have green background fill and bold text. (1 pt)
- The cell(s) with the lowest IMDb Score have red background fill and bold text. (1 pt)
