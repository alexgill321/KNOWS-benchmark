# KNOWS Benchmark

KNOWS is a benchmark for evaluating web agents on realistic, open-ended **Google Workspace** tasks: writing documents, building spreadsheets, and composing slide decks that require web research, multi-step tool use, and faithful grounding in retrieved sources.

- **22 task templates × 5 instances = 110 tasks** across Google Docs (5 templates), Sheets (9), and Slides (8).
- **Hybrid, white-box evaluation**: each task ships a programmatic evaluator (`evaluator.py`) that scores the produced artifact step by step, combining deterministic checks, fuzzy/tolerance matching, geometric layout tests, document-structure tests, browsing-trace checks, and LLM/VLM judgements.
- **Step-level failure categories**: every evaluation step records the mechanism that decided its outcome (`StepCategory` in `src/browsergym/knows/eval/eval_utils/scoring.py`), enabling quantitative failure-mode analysis of any run via `Result.get_category_summary()`.

## Repository layout

```
src/browsergym/knows/
  __init__.py            # BrowserGym task registration (knows.<family>.<n>)
  task.py                # BrowserGym task classes (setup, prompting, grading glue)
  doc_setup.py           # Workspace provisioning & Drive-sharing helpers (CLI included)
  eval/eval_utils/       # Shared evaluation utilities (scoring, text/image/table/chart, LLM judge)
  eval/tasks/<family>/   # One directory per task template
    utils.py             #   template-shared evaluation helpers (where present)
    instance_N/          #   5 instances per template
      task.md            #     the agent prompt (verbatim)
      checkpoints.md     #     human-readable evaluation criteria
      evaluator.py       #     the automated evaluator (standalone CLI)
      data/              #     gold reference assets used by the evaluator
analysis/                # Scripts + data to reproduce the paper's dataset statistics
RUNNING_STANDALONE.md    # Run tasks on ANY agent harness (Comet, proprietary, ...)
ASSETS.md                # External Drive assets some tasks depend on, and how to host your own copies
```

## Quickstart

### Option A — Run with the reference harnesses

The maintained agent harnesses live in companion repos:

- **[BrowserGym-Knows](https://github.com/farhanishmam/BrowserGym-Knows)** — a BrowserGym fork with the `knows` backend, benchmark splits (`knows_docs_1`, `knows_sheets_7`, ...), runner scripts, and environment setup. **Start here**; this repository is consumed as its `browsergym/knows` submodule.
- **[AgentLab-Knows](https://github.com/farhanishmam/AgentLab-Knows)** — an AgentLab fork with agent configurations used in the paper. It requires the BrowserGym-Knows setup; see its README.

```bash
git clone https://github.com/farhanishmam/BrowserGym-Knows
cd BrowserGym-Knows
git submodule update --init --recursive   # pulls this repo into browsergym/knows/
# then follow BrowserGym-Knows' README for installation and runs
```

### Option B — Run on your own harness

Every evaluator is a standalone script: give your agent the `task.md` prompt plus a Google file it can edit, record the URLs it visits, then run the evaluator against the resulting file ID. **See [RUNNING_STANDALONE.md](RUNNING_STANDALONE.md)** for the complete protocol (provisioning, sharing, prompting, history capture, grading, and output parsing).

## Setup (required for both options)

Evaluators read the agent's artifact through the Google Workspace APIs and use a Gemini model as the LLM/VLM judge.

1. **Google Cloud project** with the **Drive, Docs, Sheets, and Slides APIs** enabled.
2. **Evaluator credentials** — one of:
   - *Service account (recommended)*: create a service account, download its JSON key to `auth-data/service-account.json` (or set `SERVICE_ACCOUNT_PATH`). Every graded file must be **shared with the service account's email** (see RUNNING_STANDALONE.md — the harnesses do this automatically).
   - *OAuth*: place an OAuth desktop-client `credentials.json` in `auth-data/`; the first run opens a browser consent flow and caches `auth-data/token.json`.
3. **Judge model key**: create a [Google AI Studio](https://aistudio.google.com/) API key and export `GOOGLE_AI_API_KEY` (or configure Vertex AI application-default credentials with `GOOGLE_CLOUD_PROJECT`).
4. `cp .env.example .env` and fill in the values (some task families use additional optional keys).
5. Install dependencies:

```bash
pip install -r requirements.txt
python -m playwright install chromium   # only needed for harness runs / doc_setup provisioning
```

`auth-data/` is git-ignored — never commit credentials.

## External task assets

Six task families reference source documents/folders on Google Drive from their prompts. To run them, download the released assets bundle and upload the files to **your own** Drive, then substitute the URLs — full instructions in [ASSETS.md](ASSETS.md).

## Reproducing paper analyses

`analysis/` contains the dataset-statistics script (`analyze_task_instances.py`) and the step-type taxonomy labels (`analysis/taxonomy/`). Step-level failure categories are produced natively by the evaluators (`Result.get_category_summary()`; see `eval/eval_utils/scoring.py`).

## Maintenance, Versioning, and Issue Reporting

KNOWS tasks are curated to be time-agnostic and independent of any single website, but some tasks reference live web pages and Drive-hosted assets that can change over time. Our maintenance policy:

- **Periodic freshness checks.** We maintain an internal list of the external URLs and website-extracted gold data our evaluators rely on, and manually verify them at periodic intervals after release (automated checks will replace manual ones once validated against them).
- **Updates and deprecation.** If a website change or outage significantly alters a task's difficulty or solvability, we will update the affected task or replace it with a new task of similar complexity and scope.
- **Versioning.** Every change to tasks or evaluators is released as a new tagged version on GitHub. Prior versions remain permanently available via their release tags, so results reported against any version stay interpretable and reproducible. Always report the benchmark version (release tag) alongside your results.

### Reporting outdated tasks or evaluator issues

Found a task whose referenced website changed, a dead link, or an evaluator that scores incorrectly? Please [open a GitHub issue](../../issues/new/choose) using the provided templates. Include:

1. the task and instance (e.g. `sheets_7_running_analysis/instance_3`) and the benchmark version (release tag);
2. what you observed vs. what you expected (for evaluator issues: the step name and the evaluator's printed output — scores, step details, and category summary);
3. for outdated-content reports: the affected URL and what changed;
4. if relevant and shareable: a link to the graded document (shared as view-only).

We triage reports against the policy above; fixes ship as new tagged releases.

## Citation

If you use KNOWS, please cite:

```bibtex
@article{knows2026,
  title  = {KNOWS: A Benchmark for Knowledge-Grounded Web Agents in Google Workspace},
  author = {},
  year   = {2026},
  note   = {Under review}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
