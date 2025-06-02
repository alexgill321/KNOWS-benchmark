# Checkpoints

This task has 3 points in total. 

## Checkpoint 1 (1pt): 
Check that common information about the letter writer is placed at a reasonable location in the document letter. 

### Outcome Evaluation:
- Exact match of the name to the ground-truth name.
- Location match of the name to the upper-left corner of the document.
- Exact match of the address to the ground-truth address.
- Location match of the address to the upper-left corner of the document.
- Exact match of the email to the ground-truth email.
- Location match of the email to the upper-left corner of the document.

### Eval Template(s)
- Text Exact Match, Text Location Match

## Checkpoint 2:
The logo was added to the requested place in the document.

### Outcome Evaluation:
- Image in the doc is an image of the ground-truth logo.
- Logo is placed at the top left of the document, above the writer’s name and address.

### Eval Template(s)
- Image Similarity Match, Image Location Match
 
## Checkpoint 3: 
The signature was added to the requested place in the document.

### Outcome Evaluation:
- Image in the doc is an exact match to the ground-truth signature.
- Signature is placed at the bottom of the document, below the letter body and after any other content.

### Eval Template(s)
- Image Similarity Match, Image location match

