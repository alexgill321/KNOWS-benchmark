# Instance Creation Onboarding

This document gives an overview of the instance creation effort on the `add-52-new-instances` branch: what's been done, what still needs work, and how to do it.

---

## Background

Each task template needs 5 instances (`instance_1` through `instance_5`). Instance 1 is the original, hand-built reference. Instances 2-5 were **auto-generated** (commit `9a66f44`) across 14 templates. These auto-generated instances need manual review and fixes before they're ready — they often have placeholder values, wrong paths, and missing gold data.

---

## Current Status

### Partially Merged to Main
Some instances from these templates have already been reviewed and merged to main. The remaining instances are on this branch.

| Template | On main | Still on branch |
|----------|---------|-----------------|
| `docs_1_formal_letter` | 1, 2, 3 | 4, 5 (reviewed, commit `eb7a927`) |
| `sheets_38_apartment_finder` | 1, 2, 3 | 4, 5 (not yet reviewed) |

### Reviewed on this Branch (not yet merged)
These were reviewed on `add-52-new-instances` but haven't been merged to main yet.

| Template | Instances reviewed | Evidence |
|----------|--------------------|----------|
| `docs_11_personal_recipe_ocr` | 2, 3, 4, 5 | commit `9f61b8c` |

### Still Need Review (instances 2-5 on this branch)
These have auto-generated instances 2-5 but have **not been reviewed yet**. This is the remaining work.

| Template |
|----------|
| `docs_5_influential_papers` |
| `sheets_10_paper_sorting` |
| `sheets_2_personal_recipe_foodcomposition` |
| `sheets_25_skitourplan` |
| `sheets_45_Personal_WeddingPlanner_weddingcolorpallette` |
| `sheets_6_investmenttracker` |
| `sheets_7_running_analysis` |
| `sheets_38_apartment_finder` (instances 4, 5 only) |
| `slides_17_removeimagesaddplaceholders` |
| `slides_20_Illustrated_Book_Report` |
| `slides_39_Personal_Lookbook_PaintColors` |
| `slides_42_personal_none_product_comparison` |

### No Instances 2-5 Generated
These templates only have `instance_1` and were **not included** in the auto-generation. They may need instances created from scratch.

| Template |
|----------|
| `docs_31_education_lesson_plan` |
| `docs_37_reference_list` |
| `slides_29_buy_car_pres` |

---

## How to Review an Instance

### Quick Start
1. Make sure you're on the `add-52-new-instances` branch
2. Read `INSTANCE_REVIEW_GUIDE.md` (in this same directory) — it has a detailed 7-phase checklist and report template
3. Pick a template from the "Still Need Review" list above
4. Read `instance_1` first as the reference baseline
5. Review each of instances 2-5 against instance_1
6. Fix issues you find, then commit with a clear message (e.g., "finished review of sheets_10 instances")

### What You'll Typically Need to Fix

The auto-generated instances commonly have these problems:

1. **Placeholder/TODO values** — `evaluator.py` constants set to `"TODO"` or empty strings instead of real values. These need to be filled in with content that matches the instance's `task.md`.

2. **Wrong path references** — Hardcoded references to `instance_1` instead of the correct instance directory. Search for `instance_1` in each evaluator and replace with the correct instance number.

3. **Missing or placeholder gold data** — Files in `data/` copied from instance_1 without being updated. Each instance needs its own unique gold data (reference images, CSVs, etc.) that matches what its `task.md` describes.

4. **`gold_instances.csv` without real document IDs** — Each instance needs a real Google Doc/Sheet/Slides document ID. You'll need to create the actual Google documents and populate these.

5. **Identical `task.md` across instances** — Each instance should describe a different scenario (different person, different data source, different topic, etc.) so they're not just duplicates.

6. **Checkpoint/evaluator misalignment** — The number of checkpoints in `checkpoints.md` should match the number of `grade_checkpoint_N()` functions, and point values should agree.

### Key Rules

- **No helper functions in `evaluator.py`** — only `grade_checkpoints()` and `grade_checkpoint_N()` functions. Shared logic goes in the template's `utils.py` or in `eval_utils/`.
- **One criterion = one `add_step()` call** (except loops over repeated items).
- **Reuse utilities from `eval_utils/`** — read `eval_utils/EVAL_UTILS_DIRECTORY.md` before writing anything new.
- **Use specific exception types** — no bare `except:` statements.
- **Generalize the template-level `utils.py` when instances vary** — Instance 1's `utils.py` may only support its specific scenario (e.g., arxiv-only paper lookups). If new instances introduce different data sources, platforms, or formats, you'll need to update `utils.py` to handle all variants rather than duplicating logic locally in each evaluator. Write tests for the generalized utilities. See `docs_5_influential_papers/utils.py` and its `test/test_utils.py` as an example of generalizing arxiv-only utilities to support biorxiv, nature, chemrxiv, and doi.org.

### Evaluator Quality Standards

- **Early-exit error handling** — When a checkpoint depends on fetched data (e.g., slides, document content), check for failure up front and mark all dependent steps as failed with a clear reason before returning. Don't let missing data cause crashes or misleading results. Example:
  ```python
  if not presentation_data or 'slides' not in presentation_data or len(presentation_data['slides']) == 0:
      checkpoint.add_step("Title Text Match", False, 1, details="No slides found in presentation")
      checkpoint.add_step("Tom Hanks Image Present", False, 2, details="No slides found in presentation")
      checkpoint.add_step("Image Top Right Position", False, 3, details="No slides found in presentation")
      checkpoint.add_step("Image Coverage", False, 4, details="No slides found in presentation")
      checkpoint.execution_time = time.time() - start
      return checkpoint
  ```

- **Every step that `task.md` will be evaluated on must be explicitly requested in `task.md`** — If the evaluator checks for something, the task description must ask for it. Don't evaluate things the agent was never told to do.

- **No compound evaluation steps** — Avoid ANDs in step logic. Each `add_step()` should test one discrete, independently evaluable thing. If you find yourself writing `check_X and check_Y`, split it into two steps. Only combine when the two checks are truly inseparable (e.g., "value exists and is in the correct cell" where the cell location is the only meaningful check).

- **Every step must handle its data dependency failing** — If a step relies on data gathered earlier (an image download, an API response, a parsed value), it must have a failure path for when that data is missing or malformed. Never assume upstream data gathering succeeded.

### Reference Documents
- [INSTANCE_REVIEW_GUIDE.md](INSTANCE_REVIEW_GUIDE.md) — Full 7-phase review checklist with report template
- [TASK_CREATION_GUIDELINES.md](TASK_CREATION_GUIDELINES.md) — Canonical rules for task/evaluator structure
- [TEMPLATE_CREATION_GUIDELINES.md](TEMPLATE_CREATION_GUIDELINES.md) — Guidelines for creating new templates from scratch
- `eval_utils/EVAL_UTILS_DIRECTORY.md` — Catalog of shared utility functions

### Using Claude Code for Reviews
You can point Claude Code at the review guide and have it do the mechanical checking:
```
Read src/browsergym/eval/tasks/INSTANCE_REVIEW_GUIDE.md and review <template_name>
```
This will produce a structured report of PASS/WARN/FAIL for each check. You still need to manually verify gold data content and create Google documents.
