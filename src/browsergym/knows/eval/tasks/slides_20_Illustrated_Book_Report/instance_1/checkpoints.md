# Checkpoints
This task has 130 points in total.

## Checkpoint 1 (6pt):
Title slide has all required elements.

### Outcome Evaluation:
- (1 pt) Exact match on name "Stacey Johnson"
- (1 pt) Title of book present
- (1 pt) Photo of book cover present and valid
- (1 pt) Structural Location match for name at bottom of slide, below title and photo.
- (1 pt) Structural Location match for title above name and below photo.
- (1 pt) Photo is above both title and name.

## Checkpoint 2 (120 pt):
The character slides meet the requirements. Per character (5 characters, 24 pts each):

### Outcome Evaluation:
- (1 pt) The character chosen is among the top 20 characters from the book (based on a predefined list) and the character's name is the title.
- (1 pt) Each character slide has a different background color.
- (1 pt) Each character slide has at least 3 bullet points with text.
- (10 pt) Bullet points describe actual characteristics. Scored as floor(valid_count / total_count) * 10 where valid_count is the number of bullet points that describe characteristics.
- (1 pt) Each character slide has a source link at the bottom of the slide.
- (10 pt) The characteristics are supported by the source links on the slide. Scored as floor(validated_count / total_count) * 10 where validated_count is the number of characteristics corroborated by the cited source content.

## Checkpoint 3 (2 pt):
The author slide is correct.

### Outcome Evaluation:
- (1 pt) Exact match on author name.
- (1 pt) Photo of author present and valid. Em or VLM match with examples.

## Checkpoint 4 (2 pt):
References slide is correct.

### Outcome Evaluation:
- (1 pt) At least 3 different reference links present.
- (1 pt) All reference links are valid URLs found in the agent trace.
