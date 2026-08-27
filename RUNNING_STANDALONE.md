# Running KNOWS on Your Own Harness

The KNOWS evaluators are harness-agnostic. Each task needs exactly three things from your agent framework — whether that's Comet, a proprietary stack, or a plain script driving a browser:

1. a **Google file the agent edits** (its file ID is the evaluator's main input),
2. the **task prompt** from `task.md`,
3. the agent's **browsing history** as a list of URL strings (optional but scored on most tasks).

Everything else — provisioning, prompting, grading — is described below. Complete the credential setup in the [README](README.md#setup-required-for-both-options) first.

## 1. Provision the workspace file

For most families the agent starts from a **blank** Google Doc / Sheet / Slides file that *you* create before the episode (any method works — the Drive API's `files().create`, the web UI, or the bundled Playwright helper):

```bash
python src/browsergym/knows/doc_setup.py \
  --storage-state <playwright_auth_state.json> \
  --agent-name myagent --task-name docs_1_formal_letter --instance-id 1 --kind docs
# prints: doc_id then doc_url
```

**Critical:** the file must be **shared (writer) with your evaluator service-account's email**, or grading will fail with a 404. The helper above shares automatically; to share a file you created yourself:

```bash
python src/browsergym/knows/doc_setup.py --share-doc <FILE_ID>
```

The file kind per family is implied by the prefix: `docs_*` → Document, `sheets_*` → Spreadsheet, `slides_*` → Presentation.

## 2. Build the prompt

The prompt is `task.md` **verbatim**, plus a suffix telling the agent where to work (mirroring what the reference harness appends):

> A Google Doc has already been created for you for this task. Continue this task in that document; do NOT create a new file and do NOT rename it. File URL: `https://docs.google.com/document/d/<FILE_ID>/edit`
> When you are done, reply with 'DONE' and include this URL.

URL patterns: `document/d/<id>/edit` (Docs), `spreadsheets/d/<id>/edit` (Sheets), `presentation/d/<id>/edit` (Slides).

Six families additionally reference source documents/folders on Drive inside `task.md` — host your own copies first (see [ASSETS.md](ASSETS.md)).

## 3. Record browsing history

Evaluators consume a **flat list of raw URL strings** — no timestamps or titles needed. Record every top-level (main-frame) navigation, in order, with the **full URL including query string** (checks use bidirectional substring matching against links cited in the artifact). An empty or missing history is scored as failure on the affected steps — roughly 19 of 22 families check it.

## 4. Per-family setup hooks

- `sheets_10_paper_sorting`: run `python src/browsergym/knows/eval/tasks/sheets_10_paper_sorting/setup_run.py --instance <N>` before the episode (creates the fresh Drive folders the task needs).
- Families listed in [ASSETS.md](ASSETS.md) need their Drive assets uploaded and URLs substituted once, before any run.

## 5. Grade

Run the evaluator **from the repository root** (evaluators resolve imports relative to the current working directory):

```bash
export GOOGLE_AI_API_KEY=...   # judge model
python src/browsergym/knows/eval/tasks/<family>/instance_<N>/evaluator.py \
  --workspace_doc_id "<FILE_ID>" \
  --browsing_history "https://example.com/a" "https://example.com/b"
```

Signature notes (the CLI mirrors `grade_checkpoints(...)` in each evaluator):

- Most evaluators accept `--workspace_doc_id` and `--browsing_history` (space-separated URLs). A few take no history at all (e.g. `docs_1_formal_letter`).
- `slides_30_Work_Wikipedia_Photos` additionally **requires** `--client_doc_id <id>` (the client-list Doc — your hosted copy, see ASSETS.md).
- `docs_31_education_lesson_plan` also accepts a `--browsing_history_doc_id` alternative (a Doc whose hyperlinks are treated as the history).
- `cached_models` exists for in-process reuse of loaded judge models only — do not pass it on the CLI.

For in-process integration, import the module and introspect `grade_checkpoints`'s signature before calling (the reference implementation of this pattern is `_accepted_kwargs` in `src/browsergym/knows/task.py`).

## 6. Read the results

`grade_checkpoints` returns a `Result` (`src/browsergym/knows/eval/eval_utils/scoring.py`):

```python
result.final_score              # {"total": int, "result": int} -> reward = result/total
result.get_detailed_report()    # per-checkpoint, per-step breakdown (name, success, details, category)
result.to_dict()                # JSON-serializable form for persistence
result.get_category_summary()   # {category: {total, passed, failed}} failure-mode breakdown
```

Every step carries a `category` naming the mechanism that decided its outcome (`deterministic`, `fuzzy_match`, `llm_vlm_judgement`, `spatial`, `structural`, `web_visit`, `dependency_not_evaluated`, `execution_error`, `vacuous_pass`) — see `StepCategory` in `scoring.py`. Use the summary to separate genuine agent errors from judge-mediated failures, cascaded failures, and harness/environment faults.

The CLI prints the final score and the detailed report; for programmatic runs persist `result.to_dict()`.

## Gotchas checklist

- [ ] Run evaluators from the repo root (imports depend on CWD).
- [ ] Share every graded file with the service-account email.
- [ ] Record full URLs, main-frame navigations only.
- [ ] `GOOGLE_AI_API_KEY` exported (judge failures degrade to failed steps with the judge's error in `details`).
- [ ] Evaluators may fetch live web pages referenced in the artifact (Craigslist, IMDb, arXiv, USDA, ...) — scores on those steps can vary with site availability; the `execution_error` category isolates such cases.
