# Template Creation Guidelines

This guide walks you through the complete process of creating a new task template for the Agent-Benchmark project. Follow these steps in order.

## Overview

Creating a new task template involves 5 steps:

1. **Create the Gold Instance** - Complete the task yourself in Google Drive
2. **Create Task Folder & checkpoints.md** - Set up the folder structure and evaluation criteria
3. **Run the Perturbations Pipeline** - Generate test cases for evaluator robustness
4. **Create the Evaluator** - Implement the evaluation logic
5. **Finalize & Submit PR** - Commit and submit for review

---

## Step 1: Create the Gold Instance

Before writing any code, you need to create a completed example of the task.

### Instructions

1. Navigate to the [Golds Drive Folder](https://drive.google.com/drive/folders/1fP_A9024A9LTZiR9bIFLrnplRgPgs2ud?usp=sharing)
2. Create a new Google Doc/Sheet/Slides document
3. Complete the task exactly as an agent would be expected to complete it
4. Ensure your completed document represents a "gold standard" solution

### Guidelines for Gold Instances

- The document should be **fully complete** with all required elements
- Use realistic data and formatting
- Document any external sources used (URLs, images, etc.)
- Follow the structure of the existing task gold folders.

---

## Step 2: Create Task Folder and checkpoints.md

### 2.1 Create a New Branch

```bash
git checkout -b <your-name>/<task-name>
```

### 2.2 Create the Task Folder Structure

Create a new folder following the naming convention:

```
{type}_{number}_{taskname}/
```

Where `{type}` is one of: `docs`, `sheets`, `slides`

**Examples:**
- `docs_15_quarterly_report`
- `sheets_8_budget_tracker`
- `slides_25_product_pitch`

Model your folder structure after existing tasks:

```
src/browsergym/eval/tasks/
└── {type}_{number}_{taskname}/
    └── instance_1/
        ├── task.md              # Task description
        ├── checkpoints.md       # Evaluation criteria (CRITICAL)
        ├── id.txt               # Unique identifier
        ├── gold_instances.csv   # Reference to gold document
        └── data/                # Task-specific assets (optional)
```

### 2.3 Create Required Files

#### task.md
Write a clear, human-readable task description. This is what an agent will receive as instructions. You can modify from the document task description as needed during development of the evaluator.

```markdown
Create a quarterly report in Google Docs that includes:
- Executive summary section
- Sales data from Q3 2024
- Charts visualizing the key metrics
...
```

#### id.txt
A unique identifier for the task. Format: `{number}{letter}`

```
15a
```

#### gold_instances.csv
Don't worry about this one for now

### 2.4 Write checkpoints.md (CRITICAL)

This file defines how the evaluator will assess task completion. **This is the most important file** - take time to get it right.

#### Structure

```markdown
# Checkpoints

This task has X points in total.

## Checkpoint 1 (Npt): [Checkpoint Name]

[Brief description of what this checkpoint evaluates]

### Outcome Evaluation:
- Criterion 1: [Specific, measurable requirement]
- Criterion 2: [Specific, measurable requirement]
- ...

## Checkpoint 2 (Npt): [Checkpoint Name]

...
```

#### Guidelines for Writing Checkpoints

1. **One criterion = one evaluation step**
   - Each bullet point in "Outcome Evaluation" maps to exactly ONE `add_step()` call in the evaluator
   - Exception: Repeated validations (e.g., "all 5 slides have titles") use loops

2. **Be specific and measurable**
   - Bad: "The document looks professional"
   - Good: "The header contains the company name 'Acme Corp' in bold"

3. **Include validation methods**
   - you don't need to do this, but can be helpful if you already have an idea.

4. **Point allocation**
   - Assign points based on importance
   - Total points should reflect task complexity
   - Typically I will start out assigning 1 point to each step. But you can assign based on complexity if it seems valid for that task.

#### Example

```markdown
# Checkpoints

This task has 10 points in total.

## Checkpoint 1 (4pt): Header Section

The document header contains all required elements.

### Outcome Evaluation:
- Company logo present in top-left corner (image match)
- Document title "Q3 2024 Report" present (exact match)
- Date formatted as "Month DD, YYYY" (format validation)
- Author name matches "John Smith" (exact match)

## Checkpoint 2 (6pt): Data Accuracy

The sales data is correctly transcribed.

### Outcome Evaluation:
- Total revenue matches $1,234,567 (exact match)
- Growth percentage shows 15.3% (numeric tolerance ±0.1%)
- All 4 regional breakdowns are present (count validation)
- Chart data matches table values (cross-reference validation)
```

#### What NOT to Do

```python
# DON'T: Split one criterion into multiple steps
checkpoint.add_step("Bullet Count", has_bullets, 1, "...")
checkpoint.add_step("Bullet Content", valid_content, 2, "...")  # Combine these!

# DO: One criterion = one step
checkpoint.add_step("Bullet Points", has_valid_bullets(), 1, "Must have 3+ bullets with content")
```

### 2.5 STOP - Review Checkpoint

**Before proceeding to Step 3, send your `checkpoints.md` file to the project lead for review.**

The checkpoints.md file is foundational to the entire evaluation system. Iterating on this file early saves significant rework later.

---

## Step 3: Run the Perturbations Pipeline

The perturbation pipeline generates test cases that verify your evaluator correctly handles edge cases and failure modes.

### 3a: Generate Perturbations

Run the generation script:

```bash
./src/browsergym/eval/eval_scripts/perturbation_testing/generate_perturbations.sh \
    src/browsergym/eval/tasks/{your_task}/instance_1
```

This runs a 4-session Claude pipeline:
1. **Session 1**: Generate edge cases
2. **Session 2**: Identify agent failure modes
3. **Session 3**: Consolidate and deduplicate
4. **Session 4**: Create executable prompts

**Options:**
- `--start-session N`: Resume from session N (1-4)
- `--append`: Add to existing perturbations
- `--skip-checkpoints "1,2"`: Skip specific checkpoints

**Output:** `perturbation_testing/perturbations.json`

### 3b: Setup Test Instances

Before running this script, ensure your `perturbations.json` has the required metadata fields:

```json
{
  "metadata": {
    "task_id": "your_task_id",
    "task_name": "your_task_name",
    "gold_doc_id": "YOUR_GOLD_DOC_ID",
    "gold_doc_url": "https://docs.google.com/document/d/YOUR_GOLD_DOC_ID/edit",
    "parent_folder_id": "YOUR_DRIVE_FOLDER_ID"
  },
  "perturbations": [...]
}
```

**Required fields:**
- `gold_doc_id`: The Google Drive ID of your gold document (from the URL)
- `parent_folder_id`: The Google Drive folder ID where test instances will be created

Create Google Drive copies for each perturbation:

```bash
python src/browsergym/eval/eval_scripts/perturbation_testing/setup_test_instances.py \
    --config src/browsergym/eval/tasks/{your_task}/instance_1/perturbation_testing/perturbations.json
```

**Options:**
| Option | Description |
|--------|-------------|
| `--config` | Path to perturbations.json (required) |
| `--reset` | Recreate all instances even if they already exist |

This will:
- Authenticate with Google (may open browser window)
- Create a folder named `perturbation_test_instances_{task_name}` in your Drive folder
- Copy the gold document for each perturbation
- Set permissions to "anyone with link can edit"
- Update `perturbations.json` with `doc_id` and `doc_url` fields

### 3c: Run Perturbations

Perturbations run inside a Docker container with Claude's computer use API.

#### Prerequisites

**1. Install Docker**

If you don't have Docker installed, download and install it from [Docker's website](https://www.docker.com/products/docker-desktop/):
- **macOS**: Download Docker Desktop for Mac
- **Windows**: Download Docker Desktop for Windows
- **Linux**: Follow the [Linux installation guide](https://docs.docker.com/engine/install/)

Verify Docker is installed:
```bash
docker --version
```

**2. Get an Anthropic API Key**

I'll provide you with one

**3. Pull the Docker Image**

Pull the Anthropic computer use demo image:
```bash
docker pull ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
```

#### Running the Container

**Set your API key:**
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Or, if you store it in a file:
```bash
export ANTHROPIC_API_KEY=$(cat auth-data/claude_api_key.txt)
```

**Start the Docker container:**

```bash
docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v $(pwd):/workspace \
    -v $HOME/.anthropic:/home/computeruse/.anthropic \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
```

**Access points once running:**
| URL | Description |
|-----|-------------|
| http://localhost:8080 | Combined UI (recommended) |
| http://localhost:8501 | Streamlit interface only |
| http://localhost:6080/vnc.html | Desktop VNC viewer |
| vnc://localhost:5900 | Direct VNC connection |

**Recommended: Custom screen resolution:**
```bash
docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -e WIDTH=1920 \
    -e HEIGHT=1080 \
    -v $(pwd):/workspace \
    -p 5900:5900 -p 8501:8501 -p 6080:6080 -p 8080:8080 \
    -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
```

#### One-time Google Authentication Setup

Inside the container, authenticate with Google so Claude can access your test documents:

1. Open Firefox in the container (via VNC at http://localhost:6080/vnc.html)
2. Navigate to Google and sign in with your account
3. Close Firefox

#### Run the Perturbations

Inside a terminal in the container:

```bash
cd /workspace
python src/browsergym/eval/eval_scripts/perturbation_testing/run_perturbations.py \
    --config src/browsergym/eval/tasks/{your_task}/instance_1/perturbation_testing/perturbations.json
```

**Options:**
| Option | Description |
|--------|-------------|
| `--filter pending` | Run only pending perturbations (default) |
| `--filter failed` | Retry failed perturbations |
| `--ids 001,002,003` | Run specific IDs only |
| `--timeout 3000` | Timeout per perturbation in seconds (default: 2000) |
| `--run-evaluator` | Run evaluator after each perturbation |
| `--dry-run` | Preview without executing |
| `--max-concurrent 1` | Max concurrent perturbations (keep at 1, recommended by Anthropic) |

**Logs** are saved to `/workspace/logs/{task_name}/`

### 3d: Fix Failed Perturbations

If perturbations fail, use the fix script:

```bash
./src/browsergym/eval/eval_scripts/perturbation_testing/fix_failed_perturbations.sh \
    src/browsergym/eval/tasks/{your_task}/instance_1/perturbation_testing/perturbations.json
```

**Options:**
- `--in-place` or `-i`: Update the original file (preserves doc_id/doc_url, reuses existing test documents)

The script analyzes failures, generates revised prompts, and creates `perturbations_fixed.json`.

**Re-run fixed perturbations:**

```bash
python run_perturbations.py \
    --config src/browsergym/eval/tasks/{your_task}/instance_1/perturbation_testing/perturbations_fixed.json
```

### 3e: Manual Review

Before proceeding, manually verify your `perturbations.json`:

- [ ] All perturbations have status `completed`
- [ ] Each checkpoint has adequate test coverage
- [ ] Edge cases cover boundary conditions
- [ ] Both expected `true` (should pass) and `false` (should fail) cases exist

**If you have questions about the perturbations, ask the project lead before continuing.**

---

## Step 4: Create the Evaluator

### Evaluator Writing Guidelines

Before creating the evaluator, understand these key principles from [TASK_CREATION_GUIDELINES.md](TASK_CREATION_GUIDELINES.md):

#### Code Patterns

**Imports:**
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

**LLM Queries:**
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

**Evaluator Structure:**
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

#### Best Practices

1. **One criterion = one step** (except loops for repeated validations)
2. **Reuse before create** - check `eval_utils/` for existing utilities
3. **Track timing** for performance monitoring
4. **Use specific exceptions** - no bare `except` clauses
5. **Include detailed failure messages** for debugging
6. **Clean up temp files** after evaluation
7. **Support cached models** to avoid reloading

#### Common Pitfalls

- Multiple steps for one criterion
- Bare `except` statements
- Missing failure details
- Hardcoded paths
- Duplicating existing utilities

#### Critical Rule

**No utility methods in `evaluator.py`**. All helpers must go in:
- `eval_utils/` - for functions reusable across multiple tasks
- `utils.py` (template level) - for task-specific shared utilities

Evaluators should only contain `grade_checkpoints()` and `grade_checkpoint_N()` functions.

---

### Option A: Manual Writing

Write the evaluator yourself by following patterns in existing evaluators.

**Steps:**
1. Study existing evaluators (e.g., `docs_1_formal_letter`, `slides_20_Illustrated_Book_Report`)
2. Use utilities from `eval_utils/`
3. Test manually against your gold instance
4. Test against perturbation documents

**Best for:** Learning the codebase, simple tasks, or when you want full control.

---

### Option B: Claude Code Generation

Prompt Claude Code to generate the evaluator, then validate manually.

**Steps:**
1. Provide Claude with `task.md` and `checkpoints.md`
2. Ask Claude to generate `evaluator.py`
3. **Manually verify** all checkpoints work correctly
4. Test against perturbations

**Caution:** Claude-generated evaluators always need manual review and testing.

---

### Option C: create_evaluator.sh Script (RECOMMENDED)

Use the automated pipeline to iteratively generate and test the evaluator.

```bash
./src/browsergym/eval/eval_scripts/create_evaluator.sh \
    src/browsergym/eval/tasks/{your_task}/instance_1
```

**Options:**
| Option | Description |
|--------|-------------|
| `--max-retries N` | Max iterations per checkpoint (default: 3) |
| `--verbose, -v` | Enable verbose logging |
| `--interactive, -i` | Claude can ask clarifying questions |
| `--resume` | Resume from existing PIPELINE_STATUS.json |
| `--skip-utils-review` | Skip utility placement review |

**What it does:**
1. Plans implementation for each checkpoint
2. Implements evaluation code
3. Runs perturbation tests
4. Analyzes failures and categorizes as bugs or bad perturbations
5. Fixes bugs iteratively
6. Reviews utility placement

**Important:** This script is experimental and may need debugging. Claude Code can help troubleshoot issues.

**Generated files:**
- `evaluator.py` - The evaluation logic
- `PIPELINE_STATUS.json` - Progress tracking
- `BUGS.md` - Identified bugs
- `PERT_EXP.md` - Perturbations flagged for review
- `logs/` - Detailed execution logs

---

### Testing with Perturbations

Regardless of which option you choose, test your evaluator against perturbations using the test script:

```bash
python src/browsergym/eval/eval_scripts/perturbation_testing/test_perturbation_evaluators.py \
    --config src/browsergym/eval/tasks/{your_task}/instance_1/perturbation_testing/perturbations.json
```

**Options:**
| Option | Description |
|--------|-------------|
| `--ids 001,002,003` | Test specific perturbation IDs only |
| `--skip-baseline` | Skip baseline evaluation (use cached results) |
| `--filter completed` | Filter by status: `pending`, `completed`, `failed`, `all` |
| `--output path.json` | Custom output path (default: `eval_results.json`) |

**What it does:**
1. Runs baseline evaluation on the gold document
2. Runs evaluation on each perturbed document
3. Compares actual vs expected results for targeted steps
4. Detects unexpected side effects on non-targeted steps
5. Outputs results to `eval_results.json`

**Interpreting results:**

Check `eval_results.json` for:
- `primary_check_passed`: Did the targeted step match the expected result?
- `side_effects`: Did any non-targeted steps change unexpectedly?

For each perturbation:
- `expected_result: true` - evaluator should PASS for that step
- `expected_result: false` - evaluator should FAIL for that step
- Investigate any mismatches between expected and actual results

**Testing a single document manually:**

```bash
python src/browsergym/eval/tasks/{your_task}/instance_1/evaluator.py \
    --workspace_doc_id="<perturbation_doc_id>"
```

---

## Step 5: Finalize and Submit PR

### Required Files Checklist

Ensure your task folder contains all required files:

```
{type}_{number}_{taskname}/
├── utils.py                    # If you created shared utilities
└── instance_1/
    ├── task.md                 # Task description
    ├── checkpoints.md          # Evaluation criteria
    ├── evaluator.py            # Evaluation implementation
    ├── id.txt                  # Unique identifier
    ├── gold_instances.csv      # Reference to gold document
    ├── data/                   # Task-specific assets (if needed)
    └── perturbation_testing/
        └── perturbations.json  # Test cases
```

### Commit Your Changes

```bash
git add .
git commit -m "Add {task_name} task template

- Created task.md with task description
- Defined checkpoints in checkpoints.md
- Implemented evaluator.py with N checkpoints
- Generated perturbation test cases

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### Create Pull Request

```bash
git push -u origin <your-branch>
gh pr create --title "Add {task_name} task template" --body "## Summary
- New task template: {type}_{number}_{taskname}
- N checkpoints covering [brief description]
- Perturbation testing complete

## Test plan
- [ ] Evaluator passes on gold instance
- [ ] Perturbation tests pass
- [ ] Manual review of edge cases"
```

### Review Process

The project lead will review your PR and either:
- **Request changes** - Address feedback and update the PR
- **Approve** - Your task template is merged!

---

## Appendix

### Reference Tasks

Study these existing tasks for patterns:

- **Docs:** `docs_1_formal_letter` - Simple document structure validation
- **Slides:** `slides_20_Illustrated_Book_Report` - Multi-slide validation with images
- **Sheets:** `sheets_6_investmenttracker` - Data validation with external API calls

### Folder Structure Diagram

```
src/browsergym/eval/tasks/
├── TASK_CREATION_GUIDELINES.md       # Evaluator writing guidelines
├── TEMPLATE_CREATION_GUIDELINES.md   # This file
├── docs_1_formal_letter/
│   └── instance_1/
│       ├── task.md
│       ├── checkpoints.md
│       ├── evaluator.py
│       ├── id.txt
│       ├── gold_instances.csv
│       └── perturbation_testing/
├── sheets_6_investmenttracker/
│   ├── utils.py                      # Shared utilities (stock price API)
│   ├── test/                         # Template-level tests
│   └── instance_1/
│       └── ...
└── slides_20_Illustrated_Book_Report/
    ├── utils.py
    └── instance_1/
        └── ...
```

### Troubleshooting

**Perturbation generation fails:**
- Check that `task.md` and `checkpoints.md` exist
- Verify you have Claude API access
- Try `--start-session N` to resume from a specific session

**Docker container issues:**
- Ensure Docker is running
- Check API key file exists at `auth-data/claude_api_key.txt`
- Verify port availability (5900, 8501, 6080, 8080)

**Evaluator tests fail unexpectedly:**
- Review the perturbation's `expected_result`
- Check if the perturbation prompt is ambiguous
- Verify the test document was modified correctly

**create_evaluator.sh errors:**
- Check `PIPELINE_STATUS.json` for current state
- Use `--resume` to continue from where it stopped
- Ask Claude Code for help debugging specific errors

---

## Questions?

If you have questions at any point in this process, reach out to the project lead. It's better to ask early than to redo work later.
