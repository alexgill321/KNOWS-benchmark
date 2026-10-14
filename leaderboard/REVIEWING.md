# Reviewing a leaderboard submission (maintainers)

Triggered by an issue with the `leaderboard` + `needs-review` labels. Branch on the submission type.

## Both options — intake (5 min)

Check the issue has: submission type, release tag, modality, runs/aggregation, contact. Confirm the run is complete (110 instances). If anything is missing, comment naming the template field and add `waiting-on-submitter`.

## Option A — we grade

1. **Sharing and names.** The Drive folder lists 110 files named `<family>__instance_<N>`, each shared with the grading service account. Histories exist at `histories/<family>/instance_<N>.json` for every instance in the 15 history-scored families (`leaderboard/history_requirements.md`); each is a JSON list of strings. Missing histories for a scored family: ask once; if the submitter never logged navigations, offer to post under Option B with the caveat, or grade with empty histories and note the affected steps in the row.
2. **Freeze.** Copy every file into `KNOWS Leaderboard/<issue>-<system>/` on the lab Drive (Drive API `files.copy`), recording `original_id -> frozen_id`. Grade the frozen copies only; note the copy date on the issue.
3. **Grade.** Check out the submitted tag. For each task run the evaluator against the frozen ID with the submitted history (`--workspace_doc_id`, `--browsing_history`; `--client_doc_id` for `slides_30`; `--browsing_history_doc_id` is the alternative for `docs_31`), lab judge model and key. Persist `result.to_dict()`. If any task has `llm_vlm_judgement` steps that flip between two runs, grade a third time and take the majority per step.
4. **External-site errors.** `execution_error` steps from a dead external site are re-run once; if still failing, scored as failure and noted.
5. **Post** the row with the lab-graded SR / ASC / ACF / SF per modality and overall, badge **verified**, issue link, release tag in the row's title attribute. Swap `needs-review` for `verified`, comment the final numbers and category summary, close.

## Option B — score dicts

1. **Completeness.** `results/<family>/instance_<N>.json` exists for all 110 instances and each parses as a `Result.to_dict()` (checkpoints → steps with `success`, `details`, `category`). Reject files that look hand-edited (missing `details`/`category`, inconsistent totals).
2. **Compute.** Compute SR / ASC / ACF / SF per modality and overall from the dicts; these are the posted numbers.
3. **History check.** For each of the 15 history-scored families, every history-dependent step must carry a category other than `execution_error`-for-missing-history, and `details` must not say "No browsing history provided". If they do, the run was graded without histories: post with a note on the row ("graded without browsing history; N steps failed by default") or ask the submitter to re-grade.
4. **Judge.** Record the judge model. If it is not the lab's current judge, note it on the row.
5. **Post** with badge **self-reported**, issue link, release tag. Swap `needs-review` for `self-reported`, comment the computed numbers, close.

## Both options — archive

Commit `leaderboard/<issue>/` with the computed `summary.json`, the per-task result JSONs (lab-graded for A, submitted for B), the frozen-ID map where applicable, and the histories as submitted, so every row is reproducible from the repo alone.

Decline, with reasons on the issue, if: coverage is partial; the agent had access to evaluators or gold data; artifacts or dicts were edited; or the tag is older than the current release and the affected families changed in a way that makes scores incomparable.
