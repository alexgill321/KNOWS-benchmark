# Task Creation Guidelines

> **IMPORTANT: Before writing or modifying any evaluator, read [`eval_utils/EVAL_UTILS_DIRECTORY.md`](../eval_utils/EVAL_UTILS_DIRECTORY.md) in full.** It is a comprehensive directory of every shared utility function with descriptions and real usage examples. Always reuse existing utilities from `eval_utils/` rather than writing new helper functions. This is essential for maintaining consistency across evaluators and avoiding duplicate code.

## Directory Structure

```
tasks/
├── <template_name>/
│   ├── utils.py              # Shared utilities (optional)
│   ├── DEV.md                # Development state tracking (required when developing)
│   ├── test/                 # Test suite (optional)
│   └── instance_1/
│       ├── task.md           # Task description
│       ├── checkpoints.md    # Evaluation rubric
│       ├── evaluator.py      # Evaluation logic
│       ├── id.txt            # Unique identifier
│       ├── gold_instances.csv
│       └── data/             # Task-specific assets (optional)
```

### Required Files (Instance Level)
- **`task.md`**: Human-readable task description
- **`checkpoints.md`**: Evaluation criteria with point values
- **`evaluator.py`**: Implements `grade_checkpoints()` returning a `Result` object
- **`id.txt`**: Unique task identifier
- **`gold_instances.csv`**: Maps instance names to Google document IDs

### Optional Files
- **`utils.py`** (template level): Shared utilities - check `eval_utils/` first before creating new ones
- **`data/`**: Reference datasets, gold images, expected outputs
- **`test/`**: Unit and integration tests

---

## DEV.md - Development State Tracking

**CRITICAL**: When developing or modifying an evaluator, maintain a `DEV.md` file at the template level. This file tracks development state, issues, and decisions.

### Requirements
- **Maximum 300 lines** - keep it concise and relevant
- **Read DEV.md** at the start of each session working on the evaluator
- **Update DEV.md** after significant changes or when issues are discussed
- **Delete irrelevant sections** as issues are resolved

### DEV.md Structure
```markdown
# <Task Name> - Development Notes

## Current Status
Brief summary of evaluator state (working/in-progress/blocked)

## Known Issues
- Issue 1: description and status
- Issue 2: description and status

## Recent Changes
- YYYY-MM-DD: What changed and why

## Implementation Decisions
Key decisions made and their rationale

## TODO
- [ ] Remaining work items

## Notes from Discussions
Relevant context from user conversations
```

### When to Update DEV.md
- After implementing a checkpoint
- When encountering a bug or edge case
- After discussing issues with the user
- When making non-obvious implementation decisions
- Before ending a development session

---

## Checkpoint Mapping

**Each criterion in `checkpoints.md` maps to exactly ONE `add_step()` call.**

Exception: Repeated validations (e.g., 5 slides) use loops with clear naming like `f"Slide {i+1} - Title"`.

### Correct Example
```markdown
# checkpoints.md
## Checkpoint 1 (3pt): Title slide elements
- Student name present
- Book title present
- Cover image present
```

```python
def grade_checkpoint_1():
    checkpoint = Checkpoint(total=3, result=0, name="Title Slide")
    checkpoint.add_step("Student Name", check_name(), 1, "...")
    checkpoint.add_step("Book Title", check_title(), 2, "...")
    checkpoint.add_step("Book Cover", check_cover(), 3, "...")
    return checkpoint
```

### Incorrect - Don't Split Criteria
```python
# DON'T: One criterion becoming multiple steps
checkpoint.add_step("Bullet Count", has_bullets, 1, "...")
checkpoint.add_step("Bullet Content", valid_content, 2, "...")  # Combine these!
```

---

## Planning Evaluation Steps

For each step, document:

1. **Summary**: 1-2 sentence implementation description
2. **Utilities**: List from `eval_utils/`, `utils.py`, or other evaluators
3. **Hierarchy**: Fallback chain (exact → fuzzy → LLM-based)
4. **New Utilities**: If needed, specify name, purpose, and location

Example:
```markdown
### Step 1.2: Book title (fuzzy match)
**Summary**: Validate title appears on slide with minor variation tolerance.
**Utilities**: `text_fuzzy_match_contained()`, `get_slide_text_content()`
**Hierarchy**: exact match → fuzzy (85%) → LLM validation
**New Utilities**: None
```

---

## Code Patterns

### Imports
```python
import os, time
from typing import List

from rapidfuzz import fuzz
from src.browsergym.eval.eval_utils.scoring import Checkpoint, Result
from src.browsergym.eval.eval_utils.text_utils import (
    keyword_exact_match,
    keywords_exact_match,
    keywords_match_robust,
)
```

### LLM Queries
```python
from src.browsergym.eval.eval_utils.models import load_model

model = load_model("gemma-google-ai")
messages = [
    {"role": "system", "content": [{"type": "text", "text": "Instructions"}]},
    {"role": "user", "content": [
        {"type": "image", "image": "/path/to/image.png"},  # For vision
        {"type": "text", "text": "Query"}
    ]}
]
response = model(messages)
```

### Evaluator Structure
```python
def grade_checkpoints(workspace_doc_id, cached_models=None, browsing_history=None):
    total_start = time.time()
    checkpoints = [grade_checkpoint_1(), grade_checkpoint_2()]
    return Result(checkpoints, total_execution_time=time.time() - total_start)

def grade_checkpoint_1():
    start = time.time()
    checkpoint = Checkpoint(total=5, result=0, name="Checkpoint Name")

    step_start = time.time()
    success = perform_validation()
    checkpoint.add_step("Step Name", success, 1, "Details", time.time() - step_start)

    checkpoint.execution_time = time.time() - start
    return checkpoint
```

---

## Best Practices

1. **One criterion = one step** (except loops)
2. **Reuse before create** - check `eval_utils/` first
3. **Track timing** for performance monitoring
4. **Specific exceptions** - no bare `except`
5. **Detailed failure messages** for debugging
6. **Clean up temp files** after evaluation
7. **Support cached models** to avoid reloading

## Common Pitfalls

- ❌ Multiple steps for one criterion
- ❌ Bare `except` statements
- ❌ Missing failure details
- ❌ Hardcoded paths
- ❌ Duplicating existing utilities
- ❌ Forgetting to update DEV.md

---

## Rules

**These rules are mandatory and must always be followed.**

1. **No utility methods in `evaluator.py`**: Never define helper functions or utility methods directly in evaluator files. All utilities must go in:
   - `eval_utils/` - for functions reusable across multiple tasks
   - `utils.py` (template level) - for task-specific shared utilities

   Evaluators should only contain `grade_checkpoints()` and `grade_checkpoint_N()` functions.

2. **Run eval-utils-reviewer agent after major evaluator changes**: After making significant updates to an evaluator (adding checkpoints, modifying evaluation logic, refactoring), run the `eval-utils-reviewer` agent to verify compliance with utility placement rules and identify any helper functions that should be moved.

---

## Running Evaluators

### Prerequisites
```bash
source /path/to/your/.venv/bin/activate
```

### Via VSCode (Recommended)
Check `.vscode/launch.json` for debug profiles with required args.

### Command Line
```bash
python src/browsergym/eval/tasks/<template>/instance_1/evaluator.py --workspace_doc_id="<id>"
```

---

## Checklist

- [ ] `task.md` describes the task clearly
- [ ] `checkpoints.md` lists criteria with points
- [ ] Each criterion maps to one step in `evaluator.py`
- [ ] `id.txt` contains unique identifier
- [ ] `DEV.md` tracks development state (when developing)
- [ ] Utilities tested and documented
- [ ] Error handling is robust
- [ ] Temp files cleaned up
