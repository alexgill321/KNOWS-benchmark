# Checkpoints

This task has 5 checkpoints.

Let N = number of original papers in the source Google Drive folder.
Let M = number of new papers added by the agent.

---

## Checkpoint 1: Spreadsheet Structure
The spreadsheet has the correct column headers.

### Eval Steps:
1. **Column A Header:** Column A is labeled "Title" (or similar)
2. **Column B Header:** Column B is labeled "Authors" (or similar)
3. **Column C Header:** Column C is labeled "Abstract" (or similar)
4. **Column D Header:** Column D is labeled "arXiv Link" (or similar)
5. **Column E Header:** Column E is labeled "Drive Link" (or similar)
6. **Column F Header:** Column F is labeled "Figure 1" (or similar)

---

## Checkpoint 2: Original Papers Validation
The spreadsheet contains correct information for all original papers from the source Drive folder.

### Eval Steps (each scored X/N):
1. **Titles (Column A):** X/N papers have correct titles (fuzzy match to original PDF titles)
2. **Authors (Column B):** X/N papers have correct author lists (exact match)
3. **Abstracts (Column C):** X/N papers have correct abstracts (fuzzy match)
4. **arXiv Links (Column D):** X/N papers have valid arXiv URLs pointing to correct papers
5. **Drive Links (Column E):** X/N papers have valid Drive URLs pointing to correct PDFs
6. **Figure 1 Images (Column F):** X/N papers have images that match Figure 1 from the paper
7. **arXiv URLs Visited:** X/N original paper arXiv URLs appear in the agent browsing history

---

## Checkpoint 3: New Papers Discovery
For each first author of the original papers, at least 3 additional papers by that author have been added to the spreadsheet (or fewer if the author has fewer than 3 other papers on arXiv).

### Eval Steps:
1. **Author Coverage:** X/N original first authors have 3+ new papers added

---

## Checkpoint 4: New Papers Validation
The spreadsheet contains correct information for all new papers added by the agent.

### Eval Steps (each scored X/M):
1. **Titles (Column A):** X/M new papers have correct titles (fuzzy match)
2. **Authors (Column B):** X/M new papers have correct author lists (exact match)
3. **Abstracts (Column C):** X/M new papers have correct abstracts (fuzzy match)
4. **arXiv Links (Column D):** X/M new papers have valid arXiv URLs pointing to correct papers
5. **Drive Links (Column E):** X/M new papers have valid Drive URLs in the correct folder
6. **Figure 1 Images (Column F):** X/M new papers have images that match Figure 1 from the paper
7. **arXiv URLs Visited:** X/M new paper arXiv URLs appear in the agent browsing history

---

## Checkpoint 5: Formatting & Organization
Rows are correctly highlighted and organized by color grouping.

### Eval Steps:
1. **Yellow Highlighting:** Binary (pass/fail) - all new paper rows must be yellow highlighted
2. **Blue Highlighting:** Binary (pass/fail) - all papers with "world models" in related works must be blue highlighted
3. **Row Grouping:** Binary (pass/fail) - rows must be grouped by highlight color (not interleaved)
