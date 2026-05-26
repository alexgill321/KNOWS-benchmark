# Instance Review Guide

> **Usage**: Point any Claude Code instance at this file and say:
> `Read src/browsergym/eval/tasks/INSTANCE_REVIEW_GUIDE.md and review <template_name>` (e.g., `review docs_1_formal_letter`)

This guide defines a systematic review process for validating new evaluator instances. Each task template has 5 instances (`instance_1` through `instance_5`). **Instance_1 is the reference baseline** — instances 2-5 are reviewed against it.

---

## How to Run a Review

1. You are given a **template name** (e.g., `docs_1_formal_letter`)
2. The template lives at `src/browsergym/eval/tasks/<template_name>/`
3. **Read instance_1 first** as the reference — note its structure, functions, constants, data files, checkpoint structure
4. **For each of instances 2-5**, read: `evaluator.py`, `checkpoints.md`, `task.md`, `id.txt`, `gold_instances.csv`, and list `data/`
5. **Run all checks below** (Phases 1-7) against each instance
6. **Produce a report** in the format at the bottom of this document

### Files Per Instance
Each instance directory should contain:
- `evaluator.py` — evaluation logic implementing `grade_checkpoints()`
- `checkpoints.md` — evaluation criteria with point values
- `task.md` — human-readable task description for the agent
- `id.txt` — unique task identifier
- `gold_instances.csv` — maps instance names to Google document IDs
- `data/` — (optional) reference datasets, gold images, expected outputs

### Key Reference Files
- `src/browsergym/eval/tasks/TASK_CREATION_GUIDELINES.md` — canonical quality rules
- `src/browsergym/eval/eval_utils/scoring.py` — Checkpoint/Result/EvaluationStep API
- `src/browsergym/eval/eval_utils/EVAL_UTILS_DIRECTORY.md` — shared utility catalog (to detect duplication)

---

## Review Checklist

### Phase 1: BLOCKER — Placeholder Detection

These are the most critical checks. If TODOs or placeholder values exist, the instance is fundamentally non-functional.

**Check 1.1: TODO markers in `evaluator.py`**
- Search for: `= "TODO"`, `= ["TODO"]`, `= "" # TODO`, `= [] # TODO`, empty string assignments where instance_1 has real values
- **PASS**: No placeholder assignments
- **WARN**: `# TODO` comments that are dev notes (not unfilled values)
- **FAIL**: Any variable assigned a TODO placeholder or empty value that instance_1 has filled

**Check 1.2: TODO markers in `data/` files**
- Search all gold data files (`.txt`, `.csv`, `.json`) in the `data/` directory for `TODO`
- **PASS**: No TODO markers in any data file
- **FAIL**: Any data file contains TODO

**Check 1.3: Empty points in `checkpoints.md`**
- Find the line like "This task has X points in total"
- **PASS**: Contains an actual number
- **FAIL**: Line is blank, missing, or contains TODO

---

### Phase 2: BLOCKER — File Completeness

**Check 2.1: Required files exist**
- Verify these 5 files are present: `evaluator.py`, `checkpoints.md`, `task.md`, `id.txt`, `gold_instances.csv`
- **PASS**: All 5 present
- **FAIL**: Any missing

**Check 2.2: `gold_instances.csv` has data rows**
- Count lines beyond the header
- **PASS**: At least 1 data row with a real Google doc/sheet/slides ID (alphanumeric string)
- **FAIL**: Header only, or no real document IDs

**Check 2.3: `id.txt` is unique**
- Read the ID value, compare against all other instances in the template
- **PASS**: Non-empty and not duplicated
- **FAIL**: Empty or duplicate

**Check 2.4: `data/` directory completeness**
- List files in `instance_1/data/` and compare against the new instance's `data/`
- The same *categories* of files should exist (gold images, reference CSVs/JSONs, etc.)
- **PASS**: Same types of gold files present as instance_1
- **FAIL**: Missing entire categories (e.g., instance_1 has `gold_images/` but new instance doesn't)
- **WARN**: Files exist but may be empty/placeholder (note for Phase 5 to verify content)

---

### Phase 3: HIGH — Evaluator Code Structure

**Check 3.1: Path references correct instance**
- Search `evaluator.py` for instance directory references (e.g., `instance_1`, `instance_2`)
- **PASS**: References match the instance's own directory
- **FAIL**: References another instance (e.g., instance_3's evaluator references `instance_1`)

**Check 3.2: No NEW local helper functions**
- List all `def` statements in the new instance's `evaluator.py`
- Compare against instance_1's `def` statements
- Allowed functions: `grade_checkpoints()`, `grade_checkpoint_N()`, `get_base_path()`, `setup*()`, `cleanup*()`, and any helpers that also exist in instance_1
- **PASS**: Same functions as instance_1
- **WARN**: Has helper functions, but they're the same ones as instance_1 (pre-existing issue)
- **FAIL**: New helper functions not present in instance_1

**Check 3.3: No bare `except:` statements**
- Search for `except:` without an exception type
- **PASS**: All exception handlers specify a type (e.g., `except Exception as e:`)
- **WARN**: Bare `except:` exists but is also present in instance_1 (pre-existing)
- **FAIL**: New bare `except:` not in instance_1

**Check 3.4: Import consistency**
- Compare the import block (typically first ~60 lines) against instance_1
- **PASS**: Same imports
- **WARN**: Minor additions that make sense for instance-specific needs
- **FAIL**: Missing imports that instance_1 has, or wildly different import structure

---

### Phase 4: HIGH — Checkpoint-to-Evaluator Alignment

**Check 4.1: Checkpoint count matches**
- Count `Checkpoint(total=` constructor calls in `evaluator.py`
- Count `## Checkpoint` headings in `checkpoints.md`
- **PASS**: Counts are equal
- **FAIL**: Mismatch

**Check 4.2: Total points match**
- Sum all `Checkpoint(total=N)` values from evaluator
- Extract the declared total from checkpoints.md ("This task has X points")
- **PASS**: Values match
- **FAIL**: Mismatch (or checkpoints.md total is empty — already caught in 1.3)

**Check 4.3: Per-checkpoint point values match**
- For each checkpoint, compare the `total=X` in evaluator against the `(Xpt)` in checkpoints.md
- **PASS**: Each pair matches
- **FAIL**: Any mismatch

**Check 4.4: Criterion-to-step mapping**
- For each checkpoint section in checkpoints.md, count the outcome evaluation bullet points
- For each `grade_checkpoint_N()` function, count distinct `add_step()` call sites
- **PASS**: Counts match (with allowance for loops that iterate over multiple items)
- **WARN**: Minor discrepancy that may be explained by loop iterations
- **FAIL**: Clear mismatch (e.g., 3 bullets but 5 steps, or 5 bullets but 3 steps)

---

### Phase 5: MEDIUM-HIGH — Content Differentiation

**Check 5.1: `task.md` varies across instances**
- Compare task.md content across all instances of the template
- **PASS**: Substantively different (different person, data, scenario, etc.)
- **FAIL**: Identical or near-identical to another instance

**Check 5.2: Evaluator constants match `task.md`**
- Extract key details from task.md (names, emails, URLs, products, topics)
- Cross-reference with hardcoded constants in evaluator.py
- **PASS**: Constants align with the task description
- **FAIL**: Mismatch (e.g., task says "Sarah Johnson" but evaluator checks for "Ethan Ashby")
- **BLOCKED**: If constants are TODO (already caught in Phase 1)

**Check 5.3: Gold data is instance-specific**
- Compare gold data files (images, CSVs, JSONs) across instances
- **PASS**: Files contain different content per instance
- **FAIL**: Identical gold data copied from another instance
- **BLOCKED**: If gold data files are missing/placeholder

**Check 5.4: No cross-instance path references**
- Search evaluator for patterns like `instance_1/`, `instance_2/`, etc. that don't match the current instance
- **PASS**: No references to other instances' directories
- **FAIL**: References another instance's data or paths

---

### Phase 6: MEDIUM — Cross-Instance Consistency

**Check 6.1: Structural parity with instance_1**
- Compare function signatures (`def grade_checkpoint_N(...)`) with instance_1
- **PASS**: Same set of `grade_checkpoint_N` functions with compatible signatures
- **FAIL**: Missing or extra checkpoint functions without corresponding checkpoints.md changes

**Check 6.2: Variable naming consistency**
- Compare global constants and module-level variables against instance_1
- **PASS**: Same naming conventions (e.g., both use `EXPECTED_NAME` not one using `GOLD_NAME`)
- **WARN**: Minor naming differences that don't affect functionality

**Check 6.3: Checkpoint names match**
- Compare the `name=` parameter in `Checkpoint()` constructors across instances
- **PASS**: Same checkpoint names (or minor wording changes reflecting instance content)
- **FAIL**: Completely different checkpoint naming scheme

---

### Phase 7: LOW — Ancillary Quality

**Check 7.1: Python syntax valid**
- Can the file be parsed without syntax errors?
- **PASS**: Valid Python
- **FAIL**: Syntax errors

**Check 7.2: `checkpoints.md` well-formatted**
- Has `## Checkpoint N (Xpt):` headings
- Has outcome evaluation sections with bullet points
- **PASS**: Follows expected format
- **WARN**: Minor formatting issues

**Check 7.3: `task.md` non-trivial**
- Check character count and line count
- **PASS**: >100 characters and >3 lines
- **FAIL**: Too short to be a real task description

**Check 7.4: No debug artifacts**
- Search for `FIXME`, `HACK`, `XXX`, commented-out `print()` statements
- **PASS**: Clean
- **WARN**: Minor artifacts that don't affect functionality

---

## Report Format

Produce the report in this exact structure:

```
========================================
TEMPLATE REVIEW: <template_name>
Reference: instance_1
Reviewing: instance_2, instance_3, instance_4, instance_5
========================================

--- instance_2 ---
[PASS] 1.1 No TODO placeholders in evaluator.py
[PASS] 1.2 No TODO in data files
[PASS] 1.3 checkpoints.md has valid point total (X)
[PASS] 2.1 All required files present
[FAIL] 2.2 gold_instances.csv has 0 data rows (header only)
[PASS] 2.3 id.txt unique ("Xn")
[PASS] 2.4 data/ directory matches instance_1
[PASS] 3.1 Path references correct instance
[PASS] 3.2 No new local helper functions
[PASS] 3.3 No bare except statements
[PASS] 3.4 Imports match instance_1
[PASS] 4.1 Checkpoint count: N == N
[PASS] 4.2 Total points: X == X
[PASS] 4.3 Per-checkpoint points all match
[PASS] 4.4 Criterion-step mapping correct
[PASS] 5.1 task.md is unique
[PASS] 5.2 Evaluator constants match task.md
[PASS] 5.3 Gold data differs from other instances
[PASS] 5.4 No cross-instance path references
[PASS] 6.1 Same functions as instance_1
[PASS] 6.2 Consistent variable naming
[PASS] 6.3 Checkpoint names match
[PASS] 7.1 Valid Python syntax
[PASS] 7.2 checkpoints.md well-formatted
[PASS] 7.3 task.md substantive (X chars, Y lines)
[PASS] 7.4 No debug artifacts

VERDICT: N FAIL, N WARN, N PASS
Issues to address:
  - [CRITICAL] 2.2: gold_instances.csv needs real document IDs

--- instance_3 ---
... (same format)

--- instance_4 ---
... (same format)

--- instance_5 ---
... (same format)

========================================
TEMPLATE SUMMARY
========================================
instance_2: N FAIL — brief description of issues
instance_3: N FAIL — brief description of issues
instance_4: N FAIL — brief description of issues
instance_5: N FAIL — brief description of issues

ACTION ITEMS (prioritized):
1. [CRITICAL] ...
2. [HIGH] ...
3. [MEDIUM] ...
```

For FAIL and WARN results, always include:
- The specific line number(s) in the file
- The actual value found vs. what was expected
- A brief description of what needs to change

---

## Notes

- **FAIL** = must fix before the instance is usable
- **WARN** = pre-existing issue (also in instance_1) or minor concern — note but don't block on it
- **PASS** = meets criteria
- **BLOCKED** = cannot check because a prerequisite check failed (e.g., can't verify constants match task.md if constants are TODO)
- Phase 1 and 2 failures are **CRITICAL** — the instance is non-functional
- Phase 3 and 4 failures are **HIGH** — the evaluator has structural problems
- Phase 5 and 6 failures are **MEDIUM** — content or consistency issues
- Phase 7 failures are **LOW** — quality polish
