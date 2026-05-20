# Checkpoints

This task has 5 checkpoints. Points scale with N (original papers) and M (new papers added).

---

## Checkpoint 1 (7 pts): Spreadsheet Structure
The spreadsheet has the correct column headers.

### Eval Steps:
1. **Column A Header:** Column A is labeled "Title" (or similar).
2. **Column B Header:** Column B is labeled "Authors" (or similar).
3. **Column C Header:** Column C is labeled "Abstract" (or similar).
4. **Column D Header:** Column D is labeled "arXiv Link" (or similar).
5. **Column E Header:** Column E is labeled "Drive Link" (or similar).
6. **Column F Header:** Column F is labeled "Figure 1" (or similar).
7. **Column G Header:** Column G is labeled "New Papers" (or similar) and is a column of checkboxes.

---

## Checkpoint 2 (8×N pts): Original Papers Validation
The spreadsheet contains correct information for all original papers from the source Drive folder.

### Eval Steps (each scored X/N):
1. **Titles (Column A):** X/N papers have correct titles (fuzzy match to original PDF titles).
2. **Authors (Column B):** X/N papers have correct author lists (exact match).
3. **Abstracts (Column C):** X/N papers have correct abstracts (fuzzy match).
4. **arXiv Links (Column D):** X/N papers have valid arXiv URLs pointing to correct papers.
5. **Drive Links (Column E):** X/N papers have valid Drive URLs pointing to correct PDFs (new papers in pdfs/ subfolder).
6. **Figure 1 Images (Column F):** Proportional (0-10 pts) — scored as floor(correct/evaluated × 10). Figures uploaded to figures/ subfolder and displayed via =IMAGE().
7. **New Papers Checkbox (Column G):** X/N papers have unchecked checkboxes (not new papers).
8. **arXiv URLs Visited:** X/N original paper arXiv URLs appear in the agent browsing history.

---

## Checkpoint 3 (N pts): New Papers Discovery
For each first author of the original papers, at least 3 additional papers by that author have been added to the spreadsheet (or fewer if the author has fewer than 3 other papers on arXiv).

### Eval Steps:
1. **Author Coverage:** X/N original first authors have 3+ new papers added.

---

## Checkpoint 4 (8×M pts): New Papers Validation
The spreadsheet contains correct information for all new papers added by the agent.

### Eval Steps (each scored X/M):
1. **Titles (Column A):** X/M new papers have correct titles (fuzzy match).
2. **Authors (Column B):** X/M new papers have correct author lists (exact match).
3. **Abstracts (Column C):** X/M new papers have correct abstracts (fuzzy match).
4. **arXiv Links (Column D):** X/M new papers have valid arXiv URLs pointing to correct papers.
5. **Drive Links (Column E):** X/M new papers have valid Drive URLs in the correct folder.
6. **Figure 1 Images (Column F):** Proportional (0-10 pts) — scored as floor(correct/evaluated × 10) where evaluated = papers with extractable gold figures.
7. **New Papers Checkbox (Column G):** Proportional (0-10 pts) — scored as floor(checked/total_new_rows × 10) across all non-original rows.
8. **arXiv URLs Visited:** X/M new paper arXiv URLs appear in the agent browsing history.

---

## Checkpoint 5 (3 pts): Formatting & Organization
Rows are correctly highlighted and organized by color grouping.

### Eval Steps:
1. **Yellow Highlighting:** Proportional (0-10 pts) — scored as floor(correct/evaluated × 10) where evaluated = papers whose keyword status was successfully determined. Papers where content couldn't be fetched are excluded from evaluation.
2. **Row Grouping:** Binary (pass/fail) - rows must be grouped by highlight color (not interleaved).
3. **Text Overflow:** No text in any cell is hidden due to cell overflow issues.
