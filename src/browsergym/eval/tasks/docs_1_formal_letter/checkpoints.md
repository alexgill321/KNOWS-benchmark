# Checkpoints

This task has 3 points in total. 

## Checkpoint 1 (1pt): 
Check that common information about the letter writer is placed at a reasonable location in the document letter. 

### Outcome Evaluation:
- Exact match to check that the document includes the writer’s right name, title, and address.
    - Use OCR to locate the text and ensure it is on the left hand side of the document. beneath the logo.

### Eval Template(s)
- Text Exact Match, Text Location Match

## Checkpoint 2:
The logo was added to the requested place in the document.

### Outcome Evaluation:
- Ensure that the logo is contained in the document using LLM-as-judge.
- Check that the logo is placed at the upper left corner of the document, above the writer’s name and address.
- Check that the logo is not too big or too small.

### Eval Template(s)
- Image Similarity Match, Image Location Match
 
## Checkpoint 3: 
The signature was added to the requested place in the document.

### Outcome Evaluation:
- Using some image similarity measure between the placed signature and the ground-truth signature.
- Check that the signature is placed at the bottom of the document, below the writer’s name and address.
- Check that the signature is not too big or too small.

### Eval Template(s)
- Image Similarity Match, Image location match

