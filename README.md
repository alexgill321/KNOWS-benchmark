# KNOWS Benchmark

KNOWS (KNowledge Of Workspace Software) is a benchmark for evaluating browser agents
on Google Workspace tasks (Docs, Sheets, Slides). It provides structured task definitions
with automated evaluation through fine-grained checkpoint scoring.

## Overview

- **22 task templates** across Google Docs, Sheets, and Slides
- **110 task instances** (5 instances per template)
- **Automated evaluators** with step-level scoring
- **Shared evaluation utilities** for text, image, and document analysis

## Repository Structure

```
src/browsergym/knows/
├── __init__.py              # Package registration with BrowserGym
├── task.py                  # Base task classes
├── doc_setup.py             # Document setup utilities
└── eval/
    ├── eval_utils/          # Shared evaluation utilities (16 modules)
    │   ├── scoring.py       # EvaluationStep, Checkpoint, Result classes
    │   ├── text_utils.py    # Text matching and OCR
    │   ├── image_utils.py   # Image comparison and VLM queries
    │   ├── models.py        # LLM/VLM model loading interface
    │   ├── google_services_utils.py  # Google API integration
    │   └── ...
    └── tasks/               # Task definitions
        ├── docs_1_formal_letter/
        │   ├── utils.py     # Shared utilities for this template
        │   ├── instance_1/
        │   │   ├── task.md          # Task instructions
        │   │   ├── checkpoints.md   # Evaluation rubric
        │   │   ├── evaluator.py     # Automated evaluation logic
        │   │   ├── id.txt           # Unique instance ID
        │   │   └── data/            # Supporting data files
        │   ├── instance_2/
        │   └── ...
        ├── sheets_7_running_analysis/
        ├── slides_20_Illustrated_Book_Report/
        └── ...
```

## Task Categories

| Category | Templates | Description |
|----------|-----------|-------------|
| Docs | 5 | Document creation and formatting (letters, papers, lesson plans, OCR, references) |
| Sheets | 9 | Data analysis and organization (recipes, investments, travel, running, sorting, etc.) |
| Slides | 8 | Presentation creation (book reports, car comparisons, event posters, lookbooks, etc.) |

## Evaluation System

Each task instance includes:
- **`task.md`**: Human-readable instructions given to the agent
- **`checkpoints.md`**: Detailed rubric with specific pass/fail criteria
- **`evaluator.py`**: Automated grading logic using the `scoring.py` framework

The evaluation produces step-level results via `EvaluationStep`, `Checkpoint`, and `Result` classes,
enabling fine-grained analysis of agent performance.

## Setup

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## Evaluation Utilities

See `src/browsergym/knows/eval/eval_utils/EVAL_UTILS_DIRECTORY.md` for detailed documentation
of all shared utility modules.

## License

Apache-2.0
