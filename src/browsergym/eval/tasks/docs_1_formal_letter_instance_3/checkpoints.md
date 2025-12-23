# Checkpoints

This task has 12 points in total.

## Checkpoint 1 (8pt):

Check that common information about the letter writer is placed at a reasonable location in the document letter.

### Outcome Evaluation:

- Exact match of the name to the ground-truth name.
- Location match of the name to the upper-left or upper-right of the document.
- Exact match of the job/position title to the ground-truth job/position title.
- Location match of the job/position title to the upper-left or upper-right of the document.
- Exact match of the institution/company to the ground-truth institution/company.
- Location match of the institution/company to the upper-left or upper-right of the document
- Exact match of the email to the ground-truth email.
- Location match of the email to the upper-left or upper-right of the document.

### Eval Template(s)

- Text Exact Match, Text Location Match

## Checkpoint 2 (2 pt):

The logo was added to the requested place in the document.

### Outcome Evaluation:

- Image in the doc is an image of UCLA or Coalas Lab.
- Logo is placed at the top left of the document

### Eval Template(s)

- Image Similarity Match, Image Location Match

## Checkpoint 3 (2 pt):

The signature was added to the requested place in the document.

### Outcome Evaluation:

- Image in the doc is an exact match to the ground-truth signature.
- Signature is placed at the bottom of the document, below the letter body and after any other content.

### Eval Template(s)

- Image Similarity Match, Image location match
